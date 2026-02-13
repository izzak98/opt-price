# VarPhi (qLSTM) Documentation

VarPhi is the first stage of the qStorm pipeline. It uses a quantile LSTM (qLSTM) model to predict quantiles of return distributions from historical data.

## Overview

**Purpose**: Predict quantiles of log return distributions for a given time horizon T

**Model Type**: Two-stage quantile regression with LSTM architecture

**Input**: Historical asset features and market data

**Output**: 37 quantiles of the return distribution

## Architecture

### Two-Stage Design

The qLSTM model consists of two modules:

1. **Normalize Module**: 
   - LSTM processes raw asset features
   - MLP predicts standardized return quantiles (mean=0, std=1)
   - Output: Normalized quantiles

2. **Market Module**:
   - LSTM processes market-wide features
   - MLP estimates volatility scaling factor
   - Output: Volatility scaling

3. **Raw Quantiles**:
   - Computed from normalized quantiles × volatility scaling
   - Output: Raw return quantiles

### Model Components

```python
LSTM_Model(
    lstm_layers,           # Number of LSTM layers for asset features
    lstm_h,                # Hidden dimension for asset LSTM
    hidden_layers,         # MLP layers for normalized quantiles
    market_lstm_layers,    # Number of LSTM layers for market features
    market_lstm_h,         # Hidden dimension for market LSTM
    market_hidden_layers,  # MLP layers for volatility scaling
    dropout,               # Dropout rate
    layer_norm             # Whether to use layer normalization
)
```

## Training

### Data Preparation

- **Training Period**: 1998-01-01 to validation_start_date (default: 2018-01-01)
- **Validation Period**: validation_start_date to validation_end_date (default: 2018-01-01 to 2019-01-01)
- **Normalization**: Rolling window statistics (default: 219 days)
- **Sequence Length**: Variable (handled by DynamicBatchSampler)

### Loss Function

**Two-Stage Quantile Loss**:
```
L_total = L_raw + L_normalized
```

Where:
- **L_raw**: Quantile loss on raw returns vs predicted raw quantiles
- **L_normalized**: Quantile loss on standardized returns vs normalized quantiles

**Quantile Loss** (for quantile level τ):
```
L_τ = mean(max(τ * (y_true - y_pred), (τ - 1) * (y_true - y_pred)))
```

### Training Procedure

1. Load best hyperparameters from Optuna study
2. Create model with optimal architecture
3. Prepare datasets with normalization window
4. Train with:
   - Adam optimizer
   - L1 regularization for sparsity
   - Early stopping based on validation loss
   - Learning rate scheduling

### Hyperparameter Optimization

Hyperparameters are optimized using Optuna:
- LSTM layers and hidden dimensions
- MLP architecture
- Dropout rates
- Learning rate
- Batch size
- Normalization window

## Inference

### Generating Quantiles

After training, use `generate_varphi_quants.py` to generate quantile predictions:

```bash
python varpi/generate_varphi_quants.py
```

This creates quantile predictions for T ∈ [15, 30] days on the validation set.

### Output Format

Each quantile prediction file (`stored_quants/quantiles_{T}.pkl`) contains:
```python
{
    "cryptocurrencies": {
        "all_pred_quantiles": np.array,  # (N, 37) quantile predictions
        "observed_returns": np.array,     # (N,) observed returns
        "future_returns": np.array        # (N,) future returns
    },
    "currency pairs": {...},
    "commodities": {...},
    "euro stoxx 50": {...},
    "s&p 500": {...},
    "nikkei 225": {...}
}
```

## Key Features

### Asset-Neutral Learning

The model uses asset-neutral features (normalized returns) combined with asset-specific categorical encoding, enabling generalization across asset classes.

### Quantile Regression

Unlike point predictions, quantile regression captures the full distribution, enabling:
- Tail risk assessment
- Nonparametric density estimation
- Robust prediction intervals

### Time-Varying Distributions

The model predicts different distributions for different time horizons, capturing:
- Volatility term structure
- Time-dependent return distributions
- Horizon-specific risk characteristics

## Model Files

- **Training**: `varpi/train_varphi.py`
- **Model Definition**: `varpi/QLSTM.py`
- **Loss Function**: `varpi/varpi_utils/varphi_utils.py`
- **Trained Model**: `models/varphi.pth`

## References

See `qLSTM-1.pdf` for the complete methodology and theoretical background.

