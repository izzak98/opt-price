import argparse
import json
import os
import random
from dataclasses import dataclass
import traceback
from typing import List, Tuple, Optional

import numpy as np
import optuna
import torch
from accelerate import Accelerator

from q_storm.StormSampler import BufferedStormSampler, StormSampler
from q_storm.StormTrainer import CallStormTrainer
from q_storm.qStorm import QStorm
from q_storm.qstorm_train import train_storm
# Import VarPi to ensure it's available when loading the model
from varpi.tain_varpi import VarPi  # noqa: F401


@dataclass(frozen=True)
class OptunaTrainConfig:
    taus: List[float]
    storage: str
    study_name: str
    seed: int


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism is helpful for study stability (at a cost of some perf).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _ensure_sqlite_dir(storage: str) -> None:
    # storage looks like "sqlite:///optuna_dbs/optuna.db"
    if not storage.startswith("sqlite:///"):
        return
    db_path = storage.replace("sqlite:///", "", 1)
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _suggest_arch(trial: optuna.Trial) -> Tuple[List[int], int, int, str, bool, bool, float, Optional[int]]:
    activation = trial.suggest_categorical("activation", ["relu", "tanh", "swish", "gelu"])

    n_hidden = trial.suggest_int("n_hidden_layers", 2, 6)
    # Keep sizes reasonably wide; the baseline uses very large layers (1054, ...).
    size_low = trial.suggest_int("hidden_min", 64, 512, log=True)
    size_high = trial.suggest_int("hidden_max", 512, 2048, log=True)
    if size_high < size_low:
        size_low, size_high = size_high, size_low

    hidden_dims: List[int] = []
    for i in range(n_hidden):
        # Encourage decreasing widths.
        upper = max(size_low, int(size_high * (0.85 ** i)))
        dim = trial.suggest_int(f"hidden_dim_{i}", size_low, upper, log=True)
        hidden_dims.append(int(dim))

    use_dgm = trial.suggest_categorical("use_dgm", ["False", "True"])
    use_dgm = use_dgm == "True"
    if use_dgm:
        dgm_dims = trial.suggest_int("dgm_dims", 8, 128, log=True)
        n_dgm_layers = trial.suggest_int("n_dgm_layers", 1, 4)
    else:
        dgm_dims = 0
        n_dgm_layers = 0

    # New architecture hyperparameters
    use_attention = trial.suggest_categorical("use_attention", ["False", "True"])
    use_residual = trial.suggest_categorical("use_residual", ["False", "True"])
    use_attention = use_attention == "True"
    use_residual = use_residual == "True"
    dropout = trial.suggest_float("dropout", 0.0, 0.5)

    quantile_embed_dim = None
    if use_attention:
        # embed_dim must be divisible by num_heads (4) for MultiheadAttention
        # Suggest values that are multiples of 4: 16, 32, 64, 128, 256
        quantile_embed_dim = trial.suggest_categorical(
            "quantile_embed_dim", [16, 32, 48, 64, 96, 128, 192, 256]
        )

    return hidden_dims, dgm_dims, n_dgm_layers, activation, use_attention, use_residual, dropout, quantile_embed_dim


def _build_model_from_trial(trial: optuna.Trial) -> QStorm:
    arch_params = _suggest_arch(trial)
    hidden_dims, dgm_dims, n_dgm_layers, activation, use_attention, use_residual, dropout, quantile_embed_dim = arch_params

    # Input dims are fixed by sampler: [S', t', rf] + varpi_q (len(taus)) = 3 + 37 = 40
    model = QStorm(
        input_dims=40,
        hidden_dims=hidden_dims,
        dgm_dims=dgm_dims,
        n_dgm_layers=n_dgm_layers,
        hidden_activation=activation,
        use_attention=use_attention,
        use_residual=use_residual,
        dropout=dropout,
        quantile_embed_dim=quantile_embed_dim,
    )
    return model


def _objective(
    trial: optuna.Trial,
    *,
    accelerator: Accelerator,
    sampler: StormSampler,
    taus: List[float],
    base_seed: int,
    n_epochs: int,
    val_mc_samples: int,
) -> float:
    _seed_everything(base_seed + trial.number)

    # Trainer weights (>= 1 per assertion in StormTrainer)
    lambda_p = trial.suggest_float("lambda_p", 1.0, 10.0, log=True)
    lambda_u = trial.suggest_float("lambda_u", 1.0, 10.0, log=True)
    lambda_o = trial.suggest_float("lambda_o", 1.0, 10.0, log=True)
    lambda_i = trial.suggest_float("lambda_i", 1.0, 10.0, log=True)
    trainer = CallStormTrainer(
        lambda_p=lambda_p, lambda_u=lambda_u, lambda_o=lambda_o, lambda_i=lambda_i
    )

    # Training params
    lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = 10_000
    n_points = 10_000
    val_split = 0.2
    grad_clip = trial.suggest_float("grad_clip", 0.0, 5.0)
    resample_freq = trial.suggest_int("resample_frequency", 2, 20)
    # MC samples as hyperparameter
    train_mc_samples = trial.suggest_int("train_mc_samples", 1, 20)

    optim_name = trial.suggest_categorical("optimizer", ["adam", "adamw", "lbfgs"])
    use_lbfgs = optim_name == "lbfgs"

    model = _build_model_from_trial(trial).to(accelerator.device)

    if use_lbfgs:
        # LBFGS often likes larger steps; still keep it bounded.
        lbfgs_lr = trial.suggest_float("lbfgs_lr", 0.1, 2.0, log=True)
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=lbfgs_lr,
            max_iter=trial.suggest_int("lbfgs_max_iter", 10, 40),
            history_size=trial.suggest_int("lbfgs_history_size", 10, 100),
            line_search_fn="strong_wolfe",
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
        )
        use_lfbgs = True
    else:
        if optim_name == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
        )
        use_lfbgs = False

    # Initialize buffered sampler for this trial
    # Stop any existing buffer worker and clear buffer (frees memory)
    if hasattr(sampler, 'stop'):
        sampler.stop()

    # Clean up sampler's CPU VarPi model to free memory
    if hasattr(sampler, 'varPi_cpu') and sampler.varPi_cpu is not None:
        del sampler.varPi_cpu
        sampler.varPi_cpu = None

    # Clear GPU cache before starting new trial
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Ensure all operations complete

    # Start buffer with trial-specific parameters
    # Use default min_resample_frequency=5 (matches training script default)
    if hasattr(sampler, 'start_buffer'):
        sampler.start_buffer(
            batch_size=int(n_points),
            n_epochs=n_epochs,
            resample_frequency=int(resample_freq),
            min_resample_frequency=5
        )

    # Train with TensorBoard logging; runs will be named trial_{trial.number}
    try:
        best_val_loss, _ = train_storm(
            model=model,
            optimizer=optimizer,
            trainer=trainer,
            scheduler=scheduler,
            n_epochs=n_epochs,
            batch_size=int(batch_size),
            n_points=int(n_points),
            taus=taus,
            sampler=sampler,
            verbose=True,
            use_lfbgs=use_lfbgs,
            val_split=val_split,
            grad_clip_norm=grad_clip,
            resample_frequency=int(resample_freq),
            trial=trial,
            log_dir='runs/optuna',
            train_mc_samples=train_mc_samples,
            val_mc_samples=val_mc_samples,
        )
    except Exception as e:
        import traceback
        # Print and log a clean error message and traceback for debugging
        print(f"Exception in trial {trial.number if hasattr(trial, 'number') else ''}: {e}")
        traceback_str = traceback.format_exc()
        print(traceback_str)

        # Attach error (message + traceback) to user attributes for the trial
        error_info = {"message": str(e), "traceback": traceback_str}
        trial.set_user_attr("error", error_info)

        # Ensure any sampler resources are freed
        if hasattr(sampler, 'stop'):
            try:
                sampler.stop()
            except Exception as samperr:
                print(f"Exception while stopping sampler: {samperr}")

        # Explicitly delete model to free GPU memory
        if 'model' in locals():
            del model

        # Proactively clear GPU memory to avoid OOM on future trials
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Ensure all operations complete

        # Optionally, log or raise further for outside error tracking systems here...

        # Proper Optuna trial pruning (marks as failed/pruned so study continues)
        raise optuna.TrialPruned(f"Exception in trial: {e}")

    # Stop buffer worker after trial completes and free memory
    if hasattr(sampler, 'stop'):
        sampler.stop()

    # Explicitly delete model to free GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Ensure all operations complete

    return float(best_val_loss)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna HPO for qStorm (q_storm train)")
    parser.add_argument("--study-name", default="qStorm", help="Optuna study name")
    parser.add_argument("--storage", default="sqlite:///optuna_dbs/qstorm_optuna.db",
                        help="Optuna storage URL (e.g., sqlite:///...)")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout (seconds)")
    parser.add_argument(
        "--n-epochs", type=int, default=40, help="Epochs per trial (tuning budget)"
    )
    args = parser.parse_args()

    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    general = config["general"]
    taus = general["quantiles"]
    storage = args.storage or general["db_path"]
    seed = int(general.get("seed", 271198))
    opt_cfg = OptunaTrainConfig(
        taus=taus, storage=storage, study_name=args.study_name, seed=seed
    )

    _ensure_sqlite_dir(opt_cfg.storage)
    os.makedirs("models", exist_ok=True)

    accelerator = Accelerator()

    sampler_obj = optuna.samplers.TPESampler(seed=opt_cfg.seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
    study = optuna.create_study(
        study_name=opt_cfg.study_name,
        storage=opt_cfg.storage,
        direction="minimize",
        load_if_exists=True,
        sampler=sampler_obj,
        pruner=pruner,
    )
    sampler = BufferedStormSampler(accelerator.device, taus)

    def wrapped_objective(trial: optuna.Trial) -> float:
        # Initialize buffer for this trial (will be done inside _objective with trial-specific params)
        # The buffer will be initialized in _objective after we know n_points and resample_frequency
        try:
            return _objective(
                trial,
                accelerator=accelerator,
                sampler=sampler,
                taus=opt_cfg.taus,
                base_seed=opt_cfg.seed,
                n_epochs=int(args.n_epochs),
                val_mc_samples=10,  # Fixed validation MC samples
            )
        except Exception as e:
            print(f"Exception in trial {trial.number if hasattr(trial, 'number') else ''}: {e}")
            traceback_str = traceback.format_exc()
            print(traceback_str)
            trial.set_user_attr("error", {"message": str(e), "traceback": traceback_str})
            raise optuna.TrialPruned(f"Exception in trial: {e}")

    while True:
        completed_trials = study.get_trials(
            deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
        if len(completed_trials) >= int(args.n_trials):
            break
        n_trials = int(args.n_trials) - len(completed_trials)
        print(f"Completed {len(completed_trials)} trials, running {n_trials} more trials")
        study.optimize(wrapped_objective, n_trials=n_trials, timeout=args.timeout, n_jobs=1)


if __name__ == "__main__":
    main()
