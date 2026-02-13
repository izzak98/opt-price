import os
import json
import argparse
import logging
import time
import random
from datetime import datetime
from tqdm import tqdm
import torch
import numpy as np
from accelerate import Accelerator
from typing import Optional, Any
from torch.utils.tensorboard import SummaryWriter
from q_storm.qStorm import QStorm
from q_storm.StormSampler import BufferedStormSampler
from q_storm.StormTrainer import CallStormTrainer
# Import VarPi to ensure it's available when loading the model
from varpi.tain_varpi import VarPi  # noqa: F401
import optuna

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

accelerator = Accelerator()
with open("config.json", "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

# Make anomaly detection configurable (disable for production)
DETECT_ANOMALY = os.getenv("DETECT_ANOMALY", "False").lower() == "true"
torch.autograd.set_detect_anomaly(DETECT_ANOMALY)
if DETECT_ANOMALY:
    logger.warning("Anomaly detection is enabled - this will slow down training significantly")


def compute_gradient_norms(model):
    """Compute gradient statistics for all model parameters."""
    total_norm = 0.0
    max_norm = 0.0
    param_norms = []

    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_norms.append(param_norm.item())
            max_norm = max(max_norm, param_norm.item())

    total_norm = total_norm ** (1. / 2)
    mean_norm = np.mean(param_norms) if param_norms else 0.0

    return {
        'global_norm': total_norm,
        'max_norm': max_norm,
        'mean_norm': mean_norm
    }


def train_storm(model,
                optimizer,
                trainer,
                scheduler,
                n_epochs,
                batch_size,
                n_points,
                taus,
                sampler,
                verbose=True,
                use_lfbgs=True,
                val_split=0.2,
                grad_clip_norm=1.0,
                resample_frequency=10,
                min_resample_frequency=5,
                trial: Optional[Any] = None,
                train_mc_samples: int = 5,
                val_mc_samples: int = 1,
                log_dir: Optional[str] = None):

    best_val_loss = float("inf")
    best_train_loss = float("inf")
    best_model_weights = model.state_dict()
    model.train()

    # Initialize TensorBoard writer if log_dir is provided
    writer = None
    if log_dir is not None:
        # Determine run name based on whether this is an Optuna trial
        if trial is not None:
            # Optuna trial: use trial_{trial.number} format
            run_name = f"trial_{trial.number}"
            log_path = os.path.join(log_dir, run_name)
        else:
            # Regular training: use timestamp-based name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"run_{timestamp}"
            log_path = os.path.join(log_dir, run_name)

        os.makedirs(log_path, exist_ok=True)
        writer = SummaryWriter(log_dir=log_path)
        logger.info(f"TensorBoard logging enabled: {log_path}")

    # Timing tracking
    epoch_times = []
    training_start_time = time.time()

    if verbose:
        pbar = tqdm(range(n_epochs), desc="Training",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    else:
        pbar = range(n_epochs)

    # Initial data sampling (use get_batch if buffered sampler, else sample)
    tqdm.write("[Training] Fetching initial training data...")
    sample_start = time.time()
    if hasattr(sampler, 'get_batch'):
        sampled_data = sampler.get_batch(n_points)
    else:
        sampled_data = sampler.sample(n_points)
    sample_time = time.time() - sample_start
    tqdm.write(f"[Training] Initial data loaded in {sample_time:.2f}s")

    # Split into train/validation sets
    n_samples = len(sampled_data["S_prime"])
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val

    def split_data(data_dict, start_idx, end_idx):
        """Helper to split dictionary of tensors"""
        return {k: v[start_idx:end_idx] for k, v in data_dict.items()}

    train_data = split_data(sampled_data, 0, n_train)
    val_data = split_data(sampled_data, n_train, n_samples)

    tqdm.write(f"[Training] Train samples: {n_train:,}, Val samples: {n_val:,}")
    tqdm.write(f"[Training] Starting training for {n_epochs} epochs...")

    for e in pbar:
        epoch_start_time = time.time()
        # Adaptive resampling: more frequent early, then stabilize
        if e > 0:
            if e < 20:
                # First 20 epochs: resample more frequently
                resample_freq = min_resample_frequency
            else:
                # After 20 epochs: use configured frequency
                resample_freq = resample_frequency

            if e % resample_freq == 0:
                # Use get_batch if buffered sampler, else sample
                if hasattr(sampler, 'get_batch'):
                    sampled_data = sampler.get_batch(n_points)
                else:
                    sampled_data = sampler.sample(n_points)
                # Re-split train/val
                n_samples = len(sampled_data["S_prime"])
                n_val = int(n_samples * val_split)
                n_train = n_samples - n_val
                train_data = split_data(sampled_data, 0, n_train)
                val_data = split_data(sampled_data, n_train, n_samples)

        # Training phase
        n_batches = n_train // batch_size
        n_batches = max(n_batches, 1)
        total_train_loss = 0
        train_losses_dict = None

        if use_lfbgs:
            epoch_num = e
            gradient_norms = None

            def closure():
                nonlocal gradient_norms
                optimizer.zero_grad()
                losses = trainer.forward(model, train_data, taus, train_mc_samples)
                loss = losses["total_loss"]

                # Check for NaN/Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning("NaN/Inf loss detected in closure at epoch %d", epoch_num)
                    return torch.tensor(0.0, device=loss.device, requires_grad=True)

                accelerator.backward(loss)

                # Compute gradient norms before clipping
                gradient_norms = compute_gradient_norms(model)

                # Gradient clipping
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

                # Log L-BFGS iteration losses
                return loss

        try:
            gradient_norms = None
            if use_lfbgs:
                optimizer.step(closure)
                # gradient_norms is set inside closure
            else:
                batch_losses = []
                batch_gradient_norms = []
                for b in range(n_batches):
                    optimizer.zero_grad()
                    batch_data = {k: v[b * batch_size:(b + 1) * batch_size]
                                  for k, v in train_data.items()}
                    losses = trainer.forward(model, batch_data, taus, train_mc_samples)
                    loss = losses["total_loss"]

                    # Check for NaN/Inf
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.warning("NaN/Inf loss detected at epoch %d, batch %d", e, b)
                        continue

                    accelerator.backward(loss)

                    # Compute gradient norms before clipping
                    batch_grad_norms = compute_gradient_norms(model)
                    batch_gradient_norms.append(batch_grad_norms)

                    # Gradient clipping
                    if grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

                    optimizer.step()
                    total_train_loss += loss.item()
                    batch_losses.append(loss.item())

                # Log batch statistics
                if batch_losses:
                    total_train_loss = np.mean(batch_losses)
                    # Average gradient norms across batches
                    if batch_gradient_norms:
                        gradient_norms = {
                            'global_norm': np.mean([g['global_norm'] for g in batch_gradient_norms]),
                            'max_norm': np.max([g['max_norm'] for g in batch_gradient_norms]),
                            'mean_norm': np.mean([g['mean_norm'] for g in batch_gradient_norms])
                        }

        except (RuntimeError, ValueError, TypeError) as err:
            epoch_num = e
            logger.error("Error during training at epoch %d: %s", epoch_num, err, exc_info=True)
            raise err

        # Compute full training loss for logging
        # Note: We need gradients enabled for residual_loss computation, so we can't use no_grad()
        if use_lfbgs or train_losses_dict is None:
            # For L-BFGS, we already computed the loss in the closure
            # For regular training, use the average batch loss or compute on a sample
            if not use_lfbgs and batch_losses:
                # Use average of batch losses as approximation
                total_train_loss = np.mean(batch_losses)
                # Create a dummy losses dict for logging
                train_losses_dict = {"total_loss": torch.tensor(total_train_loss)}
            else:
                # For L-BFGS, compute on full dataset (needs gradients)
                model.train()  # Ensure train mode for gradient computation
                train_losses_dict = trainer.forward(model, train_data, taus, val_mc_samples)
                total_train_loss = train_losses_dict["total_loss"].item()
                model.train()  # Keep in train mode

        # Check for NaN/Inf in training loss and prune Optuna trial if needed
        if np.isnan(total_train_loss) or np.isinf(total_train_loss):
            logger.warning("NaN/Inf training loss detected at epoch %d", e)
            if trial is not None:
                try:
                    logger.info("Optuna pruning trial at epoch %d due to NaN/Inf training loss", e)
                    raise optuna.TrialPruned()
                except ImportError:
                    pass
                except (AttributeError, TypeError):
                    pass
            continue

        # Validation phase
        # Note: trainer.forward() needs gradients for PDE residual computation,
        # so we can't use no_grad(). Instead, we ensure inputs require grad but don't update model.
        model.eval()
        # Ensure validation data requires gradients (needed for autograd in residual_loss)
        val_data_grad = {k: v.requires_grad_(True) if v.requires_grad is not None else v
                         for k, v in val_data.items()}
        val_losses_dict = trainer.forward(model, val_data_grad, taus, val_mc_samples)
        total_val_loss = val_losses_dict["total_loss"].item()
        model.train()

        # Check for NaN/Inf in validation loss and prune Optuna trial if needed
        if np.isnan(total_val_loss) or np.isinf(total_val_loss):
            logger.warning("NaN/Inf validation loss detected at epoch %d", e)
            if trial is not None:
                try:
                    logger.info("Optuna pruning trial at epoch %d due to NaN/Inf validation loss", e)
                    raise optuna.TrialPruned()
                except ImportError:
                    pass
                except (AttributeError, TypeError):
                    pass
            continue

        # Log learning rate
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            # Handle Accelerator-wrapped scheduler
            actual_scheduler = scheduler.scheduler if hasattr(scheduler, 'scheduler') else scheduler
            if not isinstance(actual_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if hasattr(scheduler, 'get_last_lr'):
                    current_lr = scheduler.get_last_lr()[0]

        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        avg_epoch_time = np.mean(epoch_times[-10:])  # Rolling average of last 10
        total_training_time = time.time() - training_start_time

        # TensorBoard logging
        if writer is not None:
            # Log training loss components
            if train_losses_dict is not None:
                for loss_name, loss_value in train_losses_dict.items():
                    if isinstance(loss_value, torch.Tensor):
                        scalar_value = loss_value.item() if loss_value.numel() == 1 else loss_value.mean().item()
                    else:
                        scalar_value = float(loss_value)
                    writer.add_scalar(f'loss/train_{loss_name}', scalar_value, e)

            # Log validation loss components
            if val_losses_dict is not None:
                for loss_name, loss_value in val_losses_dict.items():
                    if isinstance(loss_value, torch.Tensor):
                        scalar_value = loss_value.item() if loss_value.numel() == 1 else loss_value.mean().item()
                    else:
                        scalar_value = float(loss_value)
                    writer.add_scalar(f'loss/val_{loss_name}', scalar_value, e)

            # Log training metrics
            writer.add_scalar('metrics/learning_rate', current_lr, e)
            writer.add_scalar('metrics/epoch_time', epoch_time, e)
            writer.add_scalar('metrics/total_time', total_training_time / 60.0, e)  # in minutes
            writer.add_scalar('metrics/avg_epoch_time', avg_epoch_time, e)

            # Log system metrics
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated() / 1024**3  # GB
                writer.add_scalar('system/gpu_memory_gb', gpu_mem, e)
                if e == 0:  # Log peak memory once
                    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
                    writer.add_scalar('system/peak_gpu_memory_gb', peak_mem, e)

            # Log buffer statistics
            if hasattr(sampler, 'get_buffer_stats'):
                buf_stats = sampler.get_buffer_stats()
                writer.add_scalar('system/buffer_size', buf_stats.get('buffer_size', 0), e)
                writer.add_scalar('system/cache_hits', buf_stats.get('cache_hits', 0), e)
                writer.add_scalar('system/cache_misses', buf_stats.get('cache_misses', 0), e)
                writer.add_scalar('system/total_sampled', buf_stats.get('total_sampled', 0), e)
                writer.add_scalar('system/total_consumed', buf_stats.get('total_consumed', 0), e)

            # Log gradient metrics
            if gradient_norms is not None:
                writer.add_scalar('gradients/global_norm', gradient_norms['global_norm'], e)
                writer.add_scalar('gradients/max_norm', gradient_norms['max_norm'], e)
                writer.add_scalar('gradients/mean_norm', gradient_norms['mean_norm'], e)

        if verbose and isinstance(pbar, tqdm):
            # Build compact, informative postfix
            postfix = {
                'loss': f'{total_train_loss:.4f}',
                'val': f'{total_val_loss:.4f}',
                'lr': f'{current_lr:.2e}',
                'time': f'{epoch_time:.1f}s',
            }

            # Add buffer stats if available
            if hasattr(sampler, 'get_buffer_stats'):
                buf_stats = sampler.get_buffer_stats()
                buf_size = buf_stats.get('buffer_size', 0)
                postfix['buf'] = f'{buf_size}'

            # Add GPU memory if available
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated() / 1024**3  # GB
                postfix['gpu'] = f'{gpu_mem:.1f}G'

            pbar.set_postfix(postfix)

            # Detailed logging every 10 epochs
            if e % 10 == 0 or e == n_epochs - 1:
                train_loss_str = ', '.join(
                    f'{k}={v.item():.4f}' if isinstance(v, torch.Tensor) else f'{k}={v:.4f}'
                    for k, v in train_losses_dict.items() if k != 'total_loss'
                )
                tqdm.write(
                    f"[Epoch {e:4d}] train={total_train_loss:.4f}, val={total_val_loss:.4f}, "
                    f"lr={current_lr:.2e}, time={epoch_time:.1f}s | {train_loss_str}"
                )

        # Update learning rate scheduler
        if scheduler is not None:
            # Handle Accelerator-wrapped scheduler - check actual scheduler type
            actual_scheduler = scheduler.scheduler if hasattr(scheduler, 'scheduler') else scheduler
            if isinstance(actual_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(total_val_loss)
            else:
                scheduler.step()

        # Early stopping and model checkpointing based on validation loss
        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            best_train_loss = total_train_loss
            best_model_weights = model.state_dict()
        # Optuna pruning hook (report AFTER we've computed validation loss for this epoch)
        if trial is not None:
            try:
                trial.report(total_val_loss, step=e)
                if trial.should_prune():
                    logger.info("Optuna pruning trial at epoch %d (val_loss=%.6f)",
                                e, total_val_loss)
                    raise optuna.TrialPruned()
            except ImportError:
                pass
            except (AttributeError, TypeError):
                # Defensive: allow passing a non-Optuna trial-like object.
                pass

    model.load_state_dict(best_model_weights)

    # Final training summary
    total_training_time = time.time() - training_start_time
    avg_epoch_time = np.mean(epoch_times) if epoch_times else 0

    tqdm.write("\n" + "="*60)
    tqdm.write("TRAINING COMPLETE")
    tqdm.write("="*60)
    tqdm.write(f"  Best validation loss: {best_val_loss:.6f}")
    tqdm.write(f"  Best training loss:   {best_train_loss:.6f}")
    tqdm.write(f"  Total epochs run:     {len(epoch_times)}")
    tqdm.write(f"  Total training time:  {total_training_time/60:.1f} minutes")
    tqdm.write(f"  Avg epoch time:       {avg_epoch_time:.2f} seconds")

    if hasattr(sampler, 'get_buffer_stats'):
        buf_stats = sampler.get_buffer_stats()
        tqdm.write(f"  Buffer cache hits:    {buf_stats.get('cache_hits', 0)}")
        tqdm.write(f"  Buffer cache misses:  {buf_stats.get('cache_misses', 0)}")
        tqdm.write(f"  Total sampled:        {buf_stats.get('total_sampled', 0):,}")
        tqdm.write(f"  Total consumed:       {buf_stats.get('total_consumed', 0):,}")

    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        tqdm.write(f"  Peak GPU memory:      {peak_mem:.2f} GB")

    tqdm.write("="*60 + "\n")

    logger.info("Training completed. Best validation loss: %.6f, Best train loss: %.6f",
                best_val_loss, best_train_loss)

    # Close TensorBoard writer
    if writer is not None:
        writer.close()
        logger.info("TensorBoard logs saved")

    return best_val_loss, model


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train qStorm model')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--n-points', type=int, default=None, help='Number of sampling points')
    parser.add_argument('--n-epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--val-split', type=float, default=None, help='Validation split ratio')
    parser.add_argument('--grad-clip', type=float, default=None, help='Gradient clipping norm')
    parser.add_argument('--resample-freq', type=int, default=None, help='Data resampling frequency')
    parser.add_argument('--lambda-p', type=float, default=None, help='Payoff loss weight')
    parser.add_argument('--lambda-u', type=float, default=None, help='Under boundary loss weight')
    parser.add_argument('--lambda-o', type=float, default=None, help='Over boundary loss weight')
    parser.add_argument('--lambda-i', type=float, default=None, help='Inequality loss weight')
    parser.add_argument('--use-lbfgs', action='store_true', help='Use L-BFGS optimizer')
    parser.add_argument('--use-best-optuna', action='store_true',
                        help='Use best parameters from Optuna study')
    parser.add_argument('--optuna-study-name', default='qStorm',
                        help='Optuna study name (when using --use-best-optuna)')
    parser.add_argument('--optuna-storage', default="sqlite:///optuna_dbs/qstorm_optuna.db",
                        help='Optuna storage URL (when using --use-best-optuna)')
    parser.add_argument('--log-dir', type=str, default='runs/storm_training',
                        help='Directory for TensorBoard logs')
    args = parser.parse_args()

    # Create runs directory if it doesn't exist
    os.makedirs('runs', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Load configuration with defaults
    taus = CONFIG["general"]["quantiles"]
    training_config = CONFIG.get("training", {})

    # Load best Optuna params if requested
    if args.use_best_optuna:
        storage = args.optuna_storage or CONFIG["general"]["db_path"]
        study = optuna.load_study(study_name=args.optuna_study_name, storage=storage)
        best = study.best_params
        best_trial = study.best_trial
        logger.info("Loading best parameters from Optuna study: %s", args.optuna_study_name)
        logger.info("Best validation loss: %.6f", study.best_value)
        logger.info("Best trial number: %d", best_trial.number)

        # CRITICAL: Set random seed to match Optuna trial behavior for reproducibility
        base_seed = int(CONFIG["general"].get("seed", 271198))
        trial_seed = base_seed + best_trial.number
        random.seed(trial_seed)
        np.random.seed(trial_seed)
        torch.manual_seed(trial_seed)
        torch.cuda.manual_seed_all(trial_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info("Set random seed to %d (base=%d + trial=%d) for reproducibility",
                    trial_seed, base_seed, best_trial.number)

        # Extract parameters from best trial
        n_hidden = int(best["n_hidden_layers"])
        hidden_dims = [int(best[f"hidden_dim_{i}"]) for i in range(n_hidden)]

        # Fix use_dgm handling (Optuna stores as string "True"/"False")
        use_dgm_str = best.get("use_dgm", "False")
        use_dgm = use_dgm_str == "True" if isinstance(use_dgm_str, str) else bool(use_dgm_str)
        dgm_dims = int(best.get("dgm_dims", 0)) if use_dgm else 0
        n_dgm_layers = int(best.get("n_dgm_layers", 0)) if use_dgm else 0
        activation = str(best["activation"])

        # New architecture hyperparameters (with backward compatibility)
        # Fix string handling for boolean parameters
        use_attention_str = best.get("use_attention", "False")
        use_attention = use_attention_str == "True" if isinstance(
            use_attention_str, str) else bool(use_attention_str)
        use_residual_str = best.get("use_residual", "False")
        use_residual = use_residual_str == "True" if isinstance(
            use_residual_str, str) else bool(use_residual_str)
        dropout = float(best.get("dropout", 0.0))
        quantile_embed_dim = None
        if use_attention and "quantile_embed_dim" in best:
            quantile_embed_dim = int(best["quantile_embed_dim"])

        # Model architecture from best params
        q_storm = QStorm(
            40, hidden_dims, dgm_dims, n_dgm_layers, activation,
            use_attention=use_attention,
            use_residual=use_residual,
            dropout=dropout,
            quantile_embed_dim=quantile_embed_dim
        )

        # Training hyperparameters from best params
        lr = float(best["lr"]) if not args.lr else args.lr
        batch_size = 10000
        n_points = 10000
        # Use same epoch count as Optuna trials (default 40) if not specified
        # This ensures reproducibility with Optuna trial results
        if args.n_epochs is None:
            n_epochs = 40  # Match Optuna default
            logger.info("Using Optuna trial epoch count: %d (for reproducibility)", n_epochs)
        else:
            n_epochs = args.n_epochs
        val_split = 0.2
        grad_clip_norm = float(best["grad_clip"]) if args.grad_clip is None else args.grad_clip
        resample_freq = int(best["resample_frequency"]
                            ) if not args.resample_freq else args.resample_freq
        min_resample_freq = training_config.get("min_resample_frequency", 5)

        # Loss weights from best params
        lambda_p = float(best["lambda_p"]) if args.lambda_p is None else args.lambda_p
        lambda_u = float(best["lambda_u"]) if args.lambda_u is None else args.lambda_u
        lambda_o = float(best["lambda_o"]) if args.lambda_o is None else args.lambda_o
        lambda_i = float(best["lambda_i"]) if args.lambda_i is None else args.lambda_i

        # MC samples from best params
        train_mc_samples = int(best["train_mc_samples"])
        val_mc_samples = 10  # Fixed as requested

        # Optimizer setup from best params
        # Fix optimizer handling (Optuna stores as "adam"/"adamw"/"lbfgs", not use_lbfgs boolean)
        optim_name = str(best.get("optimizer", "adam"))
        use_lbfgs = optim_name == "lbfgs"
        # Override with command line arg if provided
        if args.use_lbfgs:
            use_lbfgs = True
            optim_name = "lbfgs"
        if use_lbfgs:
            lbfgs_lr = float(best["lbfgs_lr"])
            optimizer = torch.optim.LBFGS(
                q_storm.parameters(),
                lr=lbfgs_lr,
                max_iter=int(best["lbfgs_max_iter"]),
                history_size=int(best["lbfgs_history_size"]),
                line_search_fn="strong_wolfe",
            )
            use_lfbgs = True
        else:
            optim_name = str(best["optimizer"])
            weight_decay = float(best["weight_decay"])
            if optim_name == "adamw":
                optimizer = torch.optim.AdamW(
                    q_storm.parameters(), lr=lr, weight_decay=weight_decay)
            else:
                optimizer = torch.optim.Adam(q_storm.parameters(), lr=lr, weight_decay=weight_decay)
            use_lfbgs = False

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )

        trainer = CallStormTrainer(
            lambda_p=lambda_p,
            lambda_u=lambda_u,
            lambda_o=lambda_o,
            lambda_i=lambda_i
        )

        logger.info("Training Configuration (from Optuna best trial):")
        logger.info("  Model architecture: hidden_dims=%s, dgm_dims=%d, n_dgm_layers=%d, activation=%s",
                    hidden_dims, dgm_dims, n_dgm_layers, activation)
        logger.info("  Enhanced features: use_attention=%s, use_residual=%s, dropout=%.3f",
                    use_attention, use_residual, dropout)
        logger.info("  Learning rate: %f", lr)
        logger.info("  Batch size: %d", batch_size)
        logger.info("  Sampling points: %d", n_points)
        logger.info("  Epochs: %d", n_epochs)
        logger.info("  Validation split: %f", val_split)
        logger.info("  Gradient clip norm: %f", grad_clip_norm)
        logger.info("  Resample frequency: %d", resample_freq)
        logger.info("  Loss weights: λ_p=%f, λ_u=%f, λ_o=%f, λ_i=%f",
                    lambda_p, lambda_u, lambda_o, lambda_i)
        logger.info("  Train MC samples: %d, Val MC samples: %d", train_mc_samples, val_mc_samples)
        logger.info("  Optimizer: %s", "LBFGS" if use_lbfgs else optim_name)
    else:
        # Training hyperparameters (can be overridden by args or config)
        training_config = CONFIG.get("training", {})
        lr = args.lr or training_config.get("learning_rate", 0.001)
        batch_size = args.batch_size or training_config.get("batch_size", 10000)
        n_points = args.n_points or training_config.get("n_points", 10000)
        n_epochs = args.n_epochs or training_config.get("n_epochs", 100)
        val_split = args.val_split or training_config.get("val_split", 0.2)
        grad_clip_norm = args.grad_clip if args.grad_clip is not None else training_config.get(
            "grad_clip_norm", 1.0)
        resample_freq = args.resample_freq or training_config.get("resample_frequency", 10)
        min_resample_freq = training_config.get("min_resample_frequency", 5)

        # Loss weights
        lambda_p = args.lambda_p if args.lambda_p is not None else training_config.get(
            "lambda_p", 1.0)
        lambda_u = args.lambda_u if args.lambda_u is not None else training_config.get(
            "lambda_u", 1.0)
        lambda_o = args.lambda_o if args.lambda_o is not None else training_config.get(
            "lambda_o", 1.0)
        lambda_i = args.lambda_i if args.lambda_i is not None else training_config.get(
            "lambda_i", 1.0)

        # Default MC samples
        train_mc_samples = 5
        val_mc_samples = 1

        logger.info("Training Configuration:")
        logger.info("  Learning rate: %f", lr)
        logger.info("  Batch size: %d", batch_size)
        logger.info("  Sampling points: %d", n_points)
        logger.info("  Epochs: %d", n_epochs)
        logger.info("  Validation split: %f", val_split)
        logger.info("  Gradient clip norm: %f", grad_clip_norm)
        logger.info("  Resample frequency: %d", resample_freq)
        logger.info("  Loss weights: λ_p=%f, λ_u=%f, λ_o=%f, λ_i=%f",
                    lambda_p, lambda_u, lambda_o, lambda_i)

        # Initialize model
        q_storm = QStorm(40, [1054, 1054, 256, 64, 32], 32, 4, 'relu')

        # Initialize optimizer
        optimizer = torch.optim.Adam(q_storm.parameters(), lr=lr)
        use_lfbgs = args.use_lbfgs

        # Initialize scheduler (ReduceLROnPlateau)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )

        # Initialize trainer with configurable loss weights
        trainer = CallStormTrainer(
            lambda_p=lambda_p,
            lambda_u=lambda_u,
            lambda_o=lambda_o,
            lambda_i=lambda_i
        )

    sampler = BufferedStormSampler(accelerator.device, taus)

    # Start asynchronous buffer (conditionally starts background worker)
    # Pass training params so sampler can calculate if cache is sufficient
    logger.info("Starting buffered sampler with batch size %d...", n_points)
    sampler.start_buffer(
        n_points,
        n_epochs=n_epochs,
        resample_frequency=resample_freq,
        min_resample_frequency=min_resample_freq
    )

    q_storm, optimizer, scheduler = accelerator.prepare(q_storm, optimizer, scheduler)

    best_loss, q_storm = train_storm(
        model=q_storm,
        optimizer=optimizer,
        trainer=trainer,
        scheduler=scheduler,
        n_epochs=n_epochs,
        batch_size=batch_size,
        n_points=n_points,
        taus=taus,
        sampler=sampler,
        use_lfbgs=use_lfbgs,
        val_split=val_split,
        grad_clip_norm=grad_clip_norm,
        resample_frequency=resample_freq,
        trial=None,
        log_dir=args.log_dir,
        train_mc_samples=train_mc_samples,
        val_mc_samples=val_mc_samples)

    logger.info("Training completed with best validation loss: %f", best_loss)

    # Stop buffer worker gracefully
    if hasattr(sampler, 'stop'):
        sampler.stop()
        stats = sampler.get_buffer_stats()
        logger.info(f"Buffer statistics: {stats}")

    save_path = "models/q_storm.pth"
    torch.save(q_storm, save_path)
    logger.info("Model saved to %s", save_path)


if __name__ == "__main__":
    main()
