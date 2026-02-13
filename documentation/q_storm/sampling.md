# qStorm Sampling Documentation

This document describes how training samples are generated for qStorm training.

## Overview

**Purpose**: Generate training data points for qStorm SPDE training

**Components**: Stock prices, time-to-expiration, risk-free rates, and varphi quantiles

## StormSampler Class

### Initialization

```python
StormSampler(
    device,           # GPU/CPU device
    taus,             # Quantile levels
    min_time=15,      # Minimum time horizon (days)
    max_time=31,      # Maximum time horizon (days)
    rf_min=0.0,       # Minimum risk-free rate
    rf_max=0.05348,   # Maximum risk-free rate
    s_max=5.0,        # Maximum normalized stock price
    noise_std=0.001   # Noise for regularization
)
```

### Sampling Process

#### 1. Sample Stock Prices

```python
S_prime = torch.rand(n_points, 1) * s_max
```

Uniform sampling in [0, S_max] where S_max = 5.0 (5x strike price).

#### 2. Sample Time-to-Expiration

```python
t_prime = torch.randint(min_time, max_time, (n_points, 1))
```

Uniform integer sampling in [15, 30] days.

#### 3. Sample Risk-Free Rate

```python
rf = torch.rand(n_points, 1) * (rf_max - rf_min) + rf_min
```

Uniform sampling in [rf_min, rf_max].

#### 4. Sample Varphi Quantiles

```python
# Load cached quantiles
quantiles = self.quantiles  # Shape: (num_unique_quantiles, 37)

# Sample random quantile vectors
indices = torch.randint(0, len(quantiles), (n_points,))
sampled_quantiles = quantiles[indices]
```

Samples from cached varphi quantile predictions.

### Caching Quantiles

Varphi quantiles are cached for efficiency:

1. **Load from file**: If `varphi_quantiles.pkl` exists, load it
2. **Extract from walk dataset**: Otherwise, extract unique quantiles from `walk_dataset.pkl`
3. **Cache**: Store in memory for fast sampling

### Sample Format

Each sample contains:
```python
{
    'S_prime': torch.Tensor,      # (n_points, 1) normalized stock prices
    't_prime': torch.Tensor,      # (n_points, 1) time-to-expiration (days)
    'rf': torch.Tensor,           # (n_points, 1) risk-free rates
    'varphi_q': torch.Tensor,     # (n_points, 37) varphi quantiles
    'varpi_grid': torch.Tensor,   # (n_points, grid_size) PDF grid
    'varpi_pdf': torch.Tensor,    # (n_points, grid_size) PDF values
}
```

## Boundary Point Sampling

Different types of boundary points are sampled:

### Terminal Points (t' = 0)
```python
t_prime_terminal = torch.zeros(n_terminal, 1)
```

### Lower Boundary (S' = 0)
```python
S_prime_lower = torch.zeros(n_lower, 1)
```

### Upper Boundary (S' = S_max)
```python
S_prime_upper = torch.full((n_upper, 1), s_max)
```

### Inequality Points
Random points for American option constraint enforcement.

## Integration with VarPhi/VarPi

The sampler uses:
- **VarPhi quantiles**: Cached quantile predictions
- **VarPi**: Not directly used (quantiles are pre-computed)
- **PDF parameters**: Optimized via Wasserstein distance

## Performance Optimizations

1. **Quantile Caching**: Loads quantiles once, samples from cache
2. **Tensor Operations**: All sampling done on GPU when available
3. **Batch Sampling**: Generates all samples in one pass

## Model Files

- **Sampler**: `q_storm/StormSampler.py`
- **Usage**: Called by `q_storm/qstorm_train.py`

## References

See [Training](training.md) for how samples are used in training.

