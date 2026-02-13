"""
Training script for VarPi model (Stage 2 of qStorm pipeline).

VarPi predicts quantiles of random walk distributions given:
- q: Quantiles from varphi model (Stage 1) - quantiles of return distribution
- t: Normalized time position within the random walk (t/T)
- T: Total length of the random walk (normalized)

This is Stage 2 of the qStorm pipeline:
- Stage 1 (varphi/qLSTM): Predicts quantiles of return distributions from historical data
- Stage 2 (varpi): Predicts quantiles of random walks constructed from varphi quantiles

The model learns to map from return distribution quantiles to random walk quantiles,
which are then used in the SPDE-based option pricing model (qStorm).
"""

import os
import torch
from torch import nn
from tqdm import tqdm
from accelerate import Accelerator
from torch.utils.data import random_split, DataLoader

accelerator = Accelerator()


class VarPi(nn.Module):
    """
    VarPi model: Predicts quantiles of random walk distributions.

    Architecture:
    - Input: [q (37 quantiles), t (normalized time), T (normalized total time)]
    - Deep MLP with ReLU activations
    - Output: 37 quantiles of random walk distribution (sorted)

    The model learns the mapping:
    f: (varphi_quantiles, t/T, T) → random_walk_quantiles

    This enables predicting the distribution of cumulative returns (random walks)
    given the distribution of individual returns (from varphi).

    Args:
        hidden_layers: List of hidden layer dimensions, e.g., [4096, 4096, 2048, ...]
    """

    def __init__(self, hidden_layers: list[int]):
        super().__init__()
        # Input: 37 quantiles + 1 normalized time + 1 normalized total time = 39 features
        self.input_layer = nn.Linear(37+1+1, hidden_layers[0])

        # Build hidden layers with ReLU activations
        self.layers = nn.ModuleList()
        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            self.layers.append(nn.ReLU())

        # Output: 37 quantiles (same number as input quantiles)
        self.output_layer = nn.Linear(hidden_layers[-1], 37)

    def forward(self, q, t, T):
        """
        Forward pass: Predict quantiles of random walk distribution.

        Args:
            q: (batch_size, 37) - Quantiles from varphi model (return distribution quantiles)
            t: (batch_size, 1) - Normalized time position in walk (t/T, where t is current step)
            T: (batch_size, 1) - Normalized total walk length (T/30, where 30 is max days)

        Returns:
            u: (batch_size, 37) - Predicted quantiles of random walk distribution (sorted)
        """
        # Concatenate inputs: [quantiles, normalized_time, normalized_total_time]
        X = torch.cat([q, t, T], dim=1)

        # Forward through network
        f = self.input_layer(X)
        for layer in self.layers:
            f = layer(f)
        u = self.output_layer(f)

        # Sort quantiles to ensure monotonicity (quantile function must be non-decreasing)
        u, _ = torch.sort(u)
        return u


def mae_loss(q_pred, q_true):
    """
    Mean Absolute Error (MAE) loss for quantile prediction.

    Computes: L = mean(|q_pred - q_true|)

    This loss measures the average absolute difference between predicted and true quantiles.
    Unlike quantile loss, MAE treats all quantiles equally (no asymmetric weighting).

    Args:
        q_pred: (batch_size, num_quantiles) - Predicted quantiles
        q_true: (batch_size, num_quantiles) - True quantiles

    Returns:
        Scalar loss value
    """
    return torch.mean(torch.abs(q_pred - q_true))


def train_varpi(model: VarPi,
                optimzer,
                scheduler,
                criterion,
                train_data_loader,
                val_data_loader,
                epochs,
                patience,
                verbose=True):
    """
    Train VarPi model with early stopping.

    Training procedure:
    1. For each batch:
       - Input: (q, t, T) - varphi quantiles, normalized time, normalized total time
       - Target: tq - true quantiles of random walk distribution
       - Forward pass: predict random walk quantiles
       - Compute MAE loss between predicted and true quantiles
       - Backpropagate and update weights

    2. After each epoch:
       - Evaluate on validation set
       - Save best model if validation loss improves
       - Apply learning rate scheduling
       - Early stop if no improvement for 'patience' epochs

    The model learns to predict how return distribution quantiles evolve into
    random walk quantiles over time, which is essential for option pricing.

    Args:
        model: VarPi model instance
        optimzer: Optimizer (Adam)
        scheduler: Learning rate scheduler (StepLR)
        criterion: Loss function (MAE)
        train_data_loader: DataLoader for training set
        val_data_loader: DataLoader for validation set
        epochs: Maximum number of training epochs
        patience: Early stopping patience (epochs without improvement)
        verbose: Whether to print training progress

    Returns:
        best_val_loss: Best validation loss achieved
        model: Model loaded with best weights
    """
    model.train()
    best_val_loss = float("inf")
    patience_counter = 0
    best_weights = model.state_dict()

    for e in range(epochs):
        total_loss = 0
        counter = 0

        if verbose:
            p_bar = tqdm(train_data_loader, desc=f"Epoch {e+1}/{epochs}", leave=False)
        else:
            p_bar = train_data_loader

        for q, t, T, tq in p_bar:
            # q: (batch_size, 37) - varphi quantiles (return distribution quantiles)
            # t: (batch_size, 1) - normalized time position in walk (t/T)
            # T: (batch_size, 1) - normalized total walk length (T/30)
            # tq: (batch_size, 37) - true quantiles of random walk distribution

            optimzer.zero_grad()

            # Forward pass: predict random walk quantiles
            u = model(q, t, T)
            # u: (batch_size, 37) - predicted random walk quantiles

            # Compute MAE loss between predicted and true quantiles
            loss = criterion(u, tq)

            # Backpropagation and optimization
            accelerator.backward(loss)
            optimzer.step()

            total_loss += loss.item()

            if verbose:
                p_bar.set_postfix({"loss": total_loss / (counter + 1)})
            counter += 1

        # Update learning rate scheduler
        if scheduler is not None:
            scheduler.step()

        # Validation step
        val_loss = validate_varpi(model, criterion, val_data_loader, verbose=True)

        # Early stopping: save best model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                # Early stop if no improvement for 'patience' epochs
                if verbose:
                    tqdm.write(f"Early stopping at epoch {e+1}")
                break

        # Print epoch summary
        out = (
            f"Epoch {e+1}/{epochs} "
            f"Train Loss: {total_loss / len(train_data_loader):.4f} "
            f"Val Loss: {val_loss:.4f}"
        )
        tqdm.write(out)

    # Load best model weights
    model.load_state_dict(best_weights)
    return best_val_loss, model


def validate_varpi(model, criterion, val_data_loader, verbose):
    """
    Validate VarPi model on validation set.

    Computes average MAE loss between predicted and true random walk quantiles.

    Args:
        model: VarPi model instance
        criterion: Loss function (MAE)
        val_data_loader: DataLoader for validation set
        verbose: Whether to show progress bar

    Returns:
        Average validation loss (scalar)
    """
    total_loss = 0
    model.eval()

    if verbose:
        p_bar = tqdm(val_data_loader, desc="Validating Model", leave=False)
    else:
        p_bar = val_data_loader

    with torch.no_grad():
        for q, t, T, tq in p_bar:
            # Forward pass without gradients
            u = model(q, t, T)
            loss = criterion(u, tq)
            total_loss += loss.item()

    return total_loss / len(val_data_loader)


def main(varpi_dataset, batch_size=1024):
    """
    Main training function for VarPi model.

    Sets up model, data loaders, optimizer, and trains the model.

    Dataset:
    - Contains synthetic random walks generated from varphi quantiles
    - Each sample: (varphi_quantiles, normalized_time, normalized_total_time, random_walk_quantiles)
    - Random walks are constructed by sampling from varphi return distributions

    Architecture:
    - Deep MLP with decreasing hidden dimensions: [4096, 4096, 4096, 2048, 2048, 1024, 256, 256, 64, 32]
    - This large architecture is needed to learn the complex mapping from return quantiles
      to random walk quantiles

    Training:
    - Adam optimizer with initial learning rate 5e-4
    - StepLR scheduler: reduces LR by 0.5x every 10 epochs
    - MAE loss for quantile prediction
    - 70/20/10 train/val/test split
    - Early stopping with patience=10

    Args:
        varpi_dataset: Dataset containing (q, t, T, tq) tuples
        batch_size: Batch size for training (default 1024)

    Returns:
        Trained VarPi model
    """
    # Model architecture: deep MLP with decreasing hidden dimensions
    # This architecture learns the complex mapping from return distribution quantiles
    # to random walk quantiles
    hidden_dims = [4096, 4096, 4096, 2048, 2048, 1024, 256, 256, 64, 32]
    model = VarPi(hidden_dims)

    # Optimizer and learning rate scheduling
    optim = torch.optim.Adam(model.parameters(), lr=5e-4)
    criterion = mae_loss
    # StepLR: reduce learning rate by 0.5x every 10 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optim, 10, 0.5)

    # Train/validation/test split: 70/20/10
    dataset_size = len(varpi_dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.2 * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_data_set = random_split(
        varpi_dataset, [train_size, val_size, test_size])

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data_set, batch_size=batch_size, shuffle=False)

    # Prepare for distributed training (if using multiple GPUs)
    model, optim, train_loader, val_loader, scheduler, test_loader = accelerator.prepare(
        model, optim, train_loader, val_loader, scheduler, test_loader)

    # Compile model for optimized execution (PyTorch 2.0+)
    model.compile()

    # Load existing model or train new one
    if os.path.exists("models/varpi.pth"):
        # Load full model object (weights_only=False for PyTorch 2.6+ compatibility)
        model = torch.load("models/varpi.pth", weights_only=False)
    else:
        # Train the model
        best_loss, model = train_varpi(
            model=model,
            optimzer=optim,
            scheduler=scheduler,
            criterion=criterion,
            train_data_loader=train_loader,
            val_data_loader=val_loader,
            epochs=100,
            patience=10,
            verbose=True
        )
        # Save trained model (Stage 2 of qStorm pipeline)
        os.makedirs("models", exist_ok=True)
        torch.save(model, "models/varpi.pth")

    # Evaluate on test set
    test_loss = validate_varpi(model, criterion, test_loader, verbose=True)
    print(f"Test Loss: {test_loss:.4f}, Len of Test Dataset: {len(test_data_set)}")
    return model


if __name__ == "__main__":
    from varpi.gen_walks import get_walk_dataset, WalkDataSet
    # Load dataset: contains random walks generated from varphi quantiles
    varpi_dataset = get_walk_dataset()
    model = main(varpi_dataset)
