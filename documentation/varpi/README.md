# VarPi Module Documentation

The VarPi module implements the two-stage quantile prediction pipeline:
1. **VarPhi (qLSTM)**: Predicts quantiles of return distributions
2. **VarPi**: Predicts quantiles of random walks from return quantiles

## Module Structure

```
varpi/
├── QLSTM.py                    # qLSTM model architecture
├── train_varphi.py             # Stage 1 training script
├── generate_varphi_quants.py   # Generate quantile predictions
├── gen_walks.py                # Generate random walk dataset
├── tain_varpi.py               # Stage 2 training script
├── wasserstein_min.py          # PDF parameter optimization
└── varpi_utils/
    └── varphi_utils.py         # Quantile loss and utilities
```

## Documentation Files

- [VarPhi (qLSTM)](varphi.md) - Return distribution quantile prediction
- [VarPi](varpi.md) - Random walk quantile prediction
- [Walk Generation](walk_generation.md) - Synthetic random walk creation

## Quick Reference

### Training VarPhi
```bash
python varpi/train_varphi.py
```

### Generating Quantiles
```bash
python varpi/generate_varphi_quants.py
```

### Generating Random Walks
```bash
python varpi/gen_walks.py
```

### Training VarPi
```bash
python varpi/tain_varpi.py
```

## Key Concepts

### Two-Stage Quantile Loss

VarPhi uses a two-stage quantile loss that evaluates both:
- **Raw quantiles**: Predictions on actual returns
- **Normalized quantiles**: Predictions on standardized returns

This ensures the model learns both the distribution shape and the volatility scaling.

### Time Horizon Constraints

- **Minimum T = 15 days**: While qLSTM can handle shorter horizons (even T=1), it performs better with more samples. The minimum of 15 days ensures:
  - Sufficient data quality for reliable quantile estimation
  - Better statistical properties for random walk generation
  - More stable VarPi training
- **Maximum T = 30 days**: Covers typical short-term option expiration ranges

### Quantile Levels

The model predicts 37 quantiles spanning the full distribution:
- Lower tail: 0.00005, 0.00025, 0.00075, ...
- Body: 0.1, 0.2, ..., 0.9
- Upper tail: 0.99, 0.995, 0.99975, 0.99995

This captures both normal market behavior and extreme tail events.

