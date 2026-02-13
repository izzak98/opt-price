"""
Training script for qLSTM (Quantile LSTM) model.

Based on: "Generalized Distribution Estimation for Asset Returns" 
by Pétursson & Óskarsdóttir (qLSTM paper).

The qLSTM model predicts quantiles of log return distributions over n days
given n days of historical data. It uses a two-stage approach:
1. Normalized/standardized quantiles (normalize_module)
2. Raw quantiles (scaled by market volatility estimate)

Key features:
- Uses only asset-neutral features (generalizable across asset classes)
- Two-stage quantile loss (both raw and normalized outputs)
- LSTM architecture for sequential pattern learning
- Quantile regression for full distribution estimation
"""

import json
import os
import optuna
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator

from varpi.QLSTM import LSTM_Model
from varpi.varpi_utils.varphi_utils import TwoStageQuantileLoss
from utils.data_utils import collate_fn, DynamicBatchSampler, get_dataset

accelerator = Accelerator()

with open("config.json", "r") as file:
    CONFIG = json.load(file)


def create_model():
    """
    Create qLSTM model using best hyperparameters from Optuna study.

    The model architecture consists of:
    1. Normalize Module: LSTM + MLP that predicts standardized return quantiles
       - Processes raw asset features (x)
       - Outputs normalized quantiles (mean=0, std=1)

    2. Market Module: LSTM + MLP that estimates volatility scaling factor
       - Processes market-wide features (z) 
       - Outputs estimated sigma for scaling normalized quantiles to raw quantiles

    The final raw quantiles = normalized_quantiles * estimated_sigma

    Returns:
        model: Initialized LSTM_Model with best hyperparameters
        best_params: Dictionary of optimal hyperparameters from Optuna study
    """
    study = optuna.load_study(
        study_name="LSTM",
        storage="sqlite:///optuna_dbs/optuna.db"
    )
    best_params = study.best_params

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
    return model, best_params


def validade_model(model, criterion, val_loader):
    """
    Validate the model on the validation set.

    Computes the two-stage quantile loss:
    L_total = L_raw + L_normalized

    Where:
    - L_raw: Quantile loss on raw returns (y) vs predicted raw quantiles (raw_output)
    - L_normalized: Quantile loss on standardized returns (sy) vs normalized quantiles (normalized_output)

    The quantile loss for each quantile τ is:
    L_τ = mean(max(τ * (y_true - y_pred), (τ - 1) * (y_true - y_pred)))

    Args:
        model: qLSTM model
        criterion: TwoStageQuantileLoss instance
        val_loader: DataLoader for validation set

    Returns:
        Average validation loss (scalar)
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        running_len = 0
        for x, s, z, y, sy in val_loader:
            # x: (batch_size, seq_len, input_size) - raw asset features
            # s: (batch_size, seq_len, 1) - standardized returns (for normalization)
            # z: (batch_size, seq_len, market_data_size) - market-wide features
            # y: (batch_size, seq_len, 1) - true raw returns (target)
            # sy: (batch_size, seq_len, 1) - true standardized returns (target)

            # Forward pass: get predicted quantiles
            normalized_output, raw_output = model(x, s, z)
            # normalized_output: (batch_size, num_quantiles) - predicted standardized quantiles
            # raw_output: (batch_size, num_quantiles) - predicted raw quantiles

            # Compute two-stage quantile loss
            loss = criterion(raw_output, y, normalized_output, sy)
            total_loss += loss.item()
            running_len += 1
    return total_loss / running_len


def train_model(model, l1_reg, optimizer, criterion, train_loader, val_loader, verbose=True):
    """
    Train the qLSTM model with early stopping based on validation loss.

    Training procedure:
    1. For each batch:
       - Forward pass: predict quantiles (normalized + raw)
       - Compute two-stage quantile loss: L = L_raw + L_normalized
       - Add L1 regularization: L_total = L + λ_L1 * ||θ||_1
       - Backpropagate and update weights

    2. After each epoch:
       - Evaluate on validation set
       - Save best model weights if validation loss improves
       - Early stop if no improvement for 'patience' epochs

    The quantile loss encourages the model to predict accurate quantiles
    across the entire return distribution, not just the mean. This is crucial
    for capturing tail risks and distributional characteristics.

    Args:
        model: qLSTM model to train
        l1_reg: L1 regularization coefficient (sparsity penalty)
        optimizer: Optimizer (Adam) with learning rate and L2 weight decay
        criterion: TwoStageQuantileLoss instance
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        verbose: Whether to print training progress

    Returns:
        best_loss: Best validation loss achieved
        model: Model loaded with best weights
    """
    patience = 10  # Early stopping patience
    n_epochs = 100
    best_loss = float("inf")
    best_weights = {}

    for e in range(n_epochs):
        model.train()
        total_loss = 0
        count = 0

        if verbose:
            p_bar = tqdm(train_loader, desc="Training", leave=False)
        else:
            p_bar = train_loader

        for X, s, Z, y, sy in p_bar:
            # X: (batch_size, seq_len, input_size) - raw asset features
            # s: (batch_size, seq_len, 1) - standardized returns
            # Z: (batch_size, seq_len, market_data_size) - market features
            # y: (batch_size, seq_len, 1) - true raw returns
            # sy: (batch_size, seq_len, 1) - true standardized returns

            optimizer.zero_grad()

            # Forward pass: predict quantiles for both raw and normalized returns
            normalized_output, raw_output = model(X, s, Z)

            # Compute two-stage quantile loss
            # This loss function evaluates quantile predictions at multiple quantile levels
            # (e.g., 0.01, 0.05, 0.1, ..., 0.95, 0.99) to capture the full distribution
            loss = criterion(raw_output, y, normalized_output, sy)

            # Add L1 regularization for sparsity (encourages feature selection)
            l1_loss = 0
            for param in model.parameters():
                l1_loss += torch.sum(torch.abs(param))
            loss += l1_reg * l1_loss

            # Backpropagation and optimization
            accelerator.backward(loss)
            optimizer.step()

            total_loss += loss.item()
            count += 1

            if verbose and isinstance(p_bar, tqdm):
                p_bar.set_postfix({'loss': total_loss / count})

        # Validation step
        val_loss = validade_model(model, criterion, val_loader)

        # Early stopping: save best model if validation loss improves
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = model.state_dict()
            patience = 10  # Reset patience counter
        else:
            patience -= 1
            if patience == 0:
                # Early stop if no improvement for 'patience' epochs
                if verbose:
                    tqdm.write(f"Early stopping at epoch {e+1}")
                break

        if verbose:
            out = (
                f"Epoch {e+1}/{n_epochs} - "
                f"Train Loss: {total_loss / count:.4f} - "
                f"Val Loss: {val_loss:.4f}"
            )
            tqdm.write(out)

    # Load best model weights
    model.load_state_dict(best_weights)
    return best_loss, model


def train(model, best_params):
    """
    Main training function for qLSTM model.

    Sets up data loaders, optimizer, and loss function, then trains the model.

    Data preparation:
    - Uses normalization_window to compute rolling statistics for standardization
    - Splits data into train/validation periods
    - Uses DynamicBatchSampler to handle variable-length sequences

    The model predicts quantiles of log returns over n days given n days of data.
    This extends previous work (Barunik et al., 2024) which only considered 22-day
    fixed look-ahead periods.

    Args:
        model: qLSTM model instance
        best_params: Dictionary of optimal hyperparameters from Optuna study
    """
    batch_size = best_params["batch_size"]
    normalization_window = best_params["normalazation_window"]  # Rolling window for standardization
    taus = CONFIG["general"]["quantiles"]  # Quantile levels, e.g., [0.01, 0.02, ..., 0.99]

    # Date ranges for train/validation split
    validation_start_date = CONFIG["general"]["dates"]["validation_period"]["start_date"]
    validation_end_date = CONFIG["general"]["dates"]["validation_period"]["end_date"]

    # Create datasets
    # Training: from 1998-01-01 to validation_start_date
    train_dataset = get_dataset(
        normalization_window, "1998-01-01", validation_start_date)
    # Validation: from validation_start_date to validation_end_date
    val_dataset = get_dataset(normalization_window, validation_start_date, validation_end_date)

    # Dynamic batch samplers handle variable-length sequences efficiently
    train_batch_sampler = DynamicBatchSampler(
        train_dataset, batch_size=batch_size)
    val_batch_sampler = DynamicBatchSampler(val_dataset, batch_size=batch_size)

    # Data loaders with custom collate function for variable-length sequences
    train_loader = DataLoader(
        train_dataset, batch_sampler=train_batch_sampler, collate_fn=collate_fn)
    val_loader = DataLoader(
        val_dataset, batch_sampler=val_batch_sampler, collate_fn=collate_fn)

    # Regularization and optimization hyperparameters
    l1_reg = best_params["l1_reg"]  # L1 regularization (sparsity)
    l2_reg = best_params["l2_reg"]  # L2 regularization (weight decay)
    lr = best_params["learning_rate"]

    # Adam optimizer with L2 weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)

    # Two-stage quantile loss: evaluates both raw and normalized quantile predictions
    # This ensures the model learns both the standardized distribution shape and
    # the volatility scaling needed to recover raw returns
    criterion = TwoStageQuantileLoss(taus)

    # Print training configuration summary
    prep_out = (
        f"Model has been prepared for training with the following parameters:\n"
        f"Batch Size: {batch_size}\n"
        f"Normalization Window: {normalization_window} days\n"
        f"Validation Start Date: {validation_start_date}\n"
        f"Validation End Date: {validation_end_date}\n"
        f"L1 Regularization: {l1_reg:.6f}\n"
        f"L2 Regularization: {l2_reg:.6f}\n"
        f"Optimizer: Adam\n"
        f"Learning Rate: {lr:.6f}\n"
        f"Train Dataset Size: {len(train_dataset)}\n"
        f"Validation Dataset Size: {len(val_dataset)}\n"
        f"Model has {sum(p.numel() for p in model.parameters())} parameters"
    )
    print(prep_out)

    # Prepare model and data loaders for distributed training (if using multiple GPUs)
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader)

    # Compile model for optimized execution (PyTorch 2.0+)
    model.compile()

    # Train the model with early stopping
    best_loss, best_model = train_model(
        model, l1_reg, optimizer, criterion, train_loader, val_loader)

    print(f"Best Validation Loss: {best_loss}")

    # Save the trained model (this is the "varphi" model - Stage 1 of qStorm pipeline)
    # This model predicts quantiles of return distributions, which are then used
    # by the varpi model (Stage 2) to predict quantiles of random walks
    os.makedirs("models", exist_ok=True)
    torch.save(best_model, "models/varphi.pth")


if __name__ == "__main__":
    model, best_params = create_model()
    train(model, best_params)
