"""
Generate varphi quantile predictions for VarPi training.

This module generates quantile predictions from the trained varphi (qLSTM) model
on the validation set for different time horizons (T = 15 to 30 days). These
predictions are saved as pickle files that gen_walks.py uses to generate random
walk training data for VarPi (Stage 2 of qStorm pipeline).

Usage:
    python varpi/generate_varphi_quants.py

The generated files are saved to stored_quants/quantiles_{T}.pkl for T in [15, 30].
"""

import json
import os
import pickle
import optuna
import torch
from tqdm import tqdm
from accelerate import Accelerator

from varpi.QLSTM import LSTM_Model
from varpi.varpi_utils.varphi_utils import get_all_quantiles
from utils.data_utils import get_test_dataset

accelerator = Accelerator()

with open("config.json", "r", encoding="utf-8") as file:
    CONFIG = json.load(file)


def load_varphi_model():
    """
    Load trained varphi (qLSTM) model and hyperparameters.

    Loads:
    1. Model architecture and hyperparameters from Optuna study
    2. Trained model weights from models/varphi.pth

    Returns:
        model: Trained varphi model (moved to appropriate device)
        best_params: Dictionary of optimal hyperparameters
    """
    # Load best hyperparameters from Optuna study
    study = optuna.load_study(
        study_name="LSTM",
        storage="sqlite:///optuna_dbs/optuna.db"
    )
    best_params = study.best_params

    # Create model with optimal architecture
    model = LSTM_Model(
        lstm_layers=best_params['raw_lstm_layers'],
        lstm_h=best_params['raw_lstm_h'],
        hidden_layers=[best_params[f'raw_hidden_layer_{i}']
                       for i in range(best_params['raw_hidden_layers'])],
        hidden_activation=best_params['hidden_activation'],
        market_lstm_layers=best_params['market_lstm_layers'],
        market_lstm_h=best_params['market_lstm_h'],
        market_hidden_layers=[best_params[f'market_hidden_layer_{i}'] for i in range(
            best_params['market_hidden_layers'])],
        market_hidden_activation=best_params['market_activation'],
        dropout=best_params['dropout'],
        layer_norm=best_params['use_layer_norm']
    )

    # Load trained model (saved as full model object, not just state_dict)
    model_path = "models/varphi.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained varphi model not found at {model_path}. "
            "Please train the model first using train_varphi.py"
        )

    # Load full model object (weights_only=False for PyTorch 2.6+ compatibility)
    # The model was saved as a full object in train_varphi.py, not just state_dict
    model = torch.load(model_path, map_location=accelerator.device, weights_only=False)
    model = model.to(accelerator.device)
    model.eval()  # Set to evaluation mode

    return model, best_params


def generate_varphi_quantiles(overwrite=False, start_t=15, end_t=31):
    """
    Generate varphi quantile predictions for different time horizons.

    For each time horizon T in [start_t, end_t), this function:
    1. Creates a validation dataset with lookforward=T
    2. Runs varphi model inference to predict quantiles
    3. Saves results to stored_quants/quantiles_{T}.pkl

    The output format matches what gen_walks.py expects:
    {
        "cryptocurrencies": {
            "all_pred_quantiles": np.array,  # (N, 37) quantile predictions
            "observed_returns": np.array,
            "future_returns": np.array
        },
        "currency pairs": {...},
        ...
    }

    Args:
        overwrite: If True, overwrite existing files. If False, skip existing files.
        start_t: Starting time horizon (default 15)
        end_t: Ending time horizon (exclusive, default 31, so T in [15, 30])

    Returns:
        List of generated file paths
    """
    # Load trained model
    print("Loading trained varphi model...")
    varphi_model, best_params = load_varphi_model()
    print(f"Model loaded successfully. Normalization window: {best_params['normalazation_window']}")

    # Get validation period dates from config
    validation_start_date = CONFIG["general"]["dates"]["validation_period"]["start_date"]
    validation_end_date = CONFIG["general"]["dates"]["validation_period"]["end_date"]
    normalization_window = best_params['normalazation_window']

    print(f"Validation period: {validation_start_date} to {validation_end_date}")
    print(f"Generating quantiles for time horizons T in [{start_t}, {end_t})...")

    # Create output directory
    output_dir = "stored_quants"
    os.makedirs(output_dir, exist_ok=True)

    generated_files = []

    # Generate quantiles for each time horizon T
    for T in tqdm(range(start_t, end_t), desc="Time horizons"):
        output_file = os.path.join(output_dir, f"quantiles_{T}.pkl")

        # Skip if file exists and not overwriting
        if os.path.exists(output_file) and not overwrite:
            print(f"Skipping T={T}: File already exists at {output_file}")
            generated_files.append(output_file)
            continue

        print(f"\nProcessing T={T} days...")

        # Create validation dataset with lookforward=T
        # This dataset contains sequences where we predict T days ahead
        val_dataset = get_test_dataset(
            normalization_lookback=normalization_window,
            start_date=validation_start_date,
            end_date=validation_end_date,
            lookforward=T,  # Prediction horizon
            test=True
        )

        print(f"Dataset size: {len(val_dataset)} samples")

        # Run inference to get quantile predictions
        # get_all_quantiles returns a dict with asset classes as keys
        # Each asset class has "all_pred_quantiles", "observed_returns", "future_returns"
        quantile_predictions = get_all_quantiles(val_dataset, varphi_model)

        # Save predictions to pickle file
        with open(output_file, "wb") as pickle_file:
            pickle.dump(quantile_predictions, pickle_file)

        print(f"Saved quantile predictions to {output_file}")
        generated_files.append(output_file)

    print(f"\n✓ Generated {len(generated_files)} quantile prediction files")
    return generated_files


if __name__ == "__main__":
    # Main entry point for generating varphi quantiles.
    # Generates quantile predictions for time horizons T ∈ [15, 30] days
    # on the validation set. These are used by gen_walks.py to create
    # random walk training data for VarPi.
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate varphi quantile predictions for VarPi training"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing quantile files"
    )
    parser.add_argument(
        "--start-t",
        type=int,
        default=15,
        help="Starting time horizon (default: 15)"
    )
    parser.add_argument(
        "--end-t",
        type=int,
        default=31,
        help="Ending time horizon (exclusive, default: 31, so T in [15, 30])"
    )

    args = parser.parse_args()

    try:
        generated_files = generate_varphi_quantiles(
            overwrite=args.overwrite,
            start_t=args.start_t,
            end_t=args.end_t
        )
        print(f"\nSuccessfully generated {len(generated_files)} files:")
        for f in generated_files:
            print(f"  - {f}")
    except Exception as e:
        print(f"\nError generating quantiles: {e}")
        raise
