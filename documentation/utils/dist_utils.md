# Distribution Utils Documentation

Utility functions for converting quantiles to probability distributions and computing distribution metrics.

## Overview

The `dist_utils` module provides:
- PDF/CDF generation from quantiles
- Smoothing and normalization
- Wasserstein distance computation

## Key Functions

### generate_smooth_pdf()

Converts quantiles to smooth probability density function (PDF).

**Purpose**: Transform quantile predictions into continuous distributions for sampling

**Input**:
- `quantiles`: Array of quantile values (37 values)
- `taus`: Quantile levels (0.00005 to 0.99995)
- `min_density`: Minimum PDF value (default: 1e-3)
- `eps`: Numerical stability threshold (default: 1e-6)
- `window`: Smoothing window size (default: 61)
- `grid_points`: Number of grid points (default: 1000)

**Output**:
- `grid_x`: Support points for PDF
- `smoothed_density`: PDF values
- `cdf`: Cumulative distribution function

**Process**:
1. **Remove duplicates**: Filter quantiles with differences < eps
2. **Create grid**: Linspace from min to max quantile
3. **Interpolate CDF**: Use PCHIP (monotonic) or linear interpolation
4. **Clamp CDF**: Ensure values in [0, 1] and monotonicity
5. **Compute PDF**: Numerical differentiation of CDF
6. **Smooth PDF**: Convolution with uniform window
7. **Normalize**: Ensure PDF integrates to 1
8. **Regenerate CDF**: From smoothed PDF

**Usage**:
```python
grid, pdf, cdf = generate_smooth_pdf(
    quantiles, 
    taus, 
    min_density=1e-3,
    eps=1e-6,
    window=61,
    grid_points=1000
)
```

### calculate_wasserstein()

Computes Wasserstein distance between predicted and realized distributions.

**Purpose**: Evaluate PDF quality and optimize PDF parameters

**Input**:
- `cdf`: Predicted CDF
- `predicted_grid`: Support points for predicted CDF
- `realized_returns`: Actual return values

**Output**: Wasserstein distance (scalar)

**Usage**:
```python
distance = calculate_wasserstein(cdf, predicted_grid, realized_returns)
```

## PDF Parameter Optimization

PDF parameters are optimized via Wasserstein distance minimization:
- `min_density`: Minimum PDF value
- `eps`: Numerical stability threshold
- `window`: Smoothing window size

See `varpi/wasserstein_min.py` for optimization procedure.

## Key Features

### Monotonicity

CDF is ensured to be non-decreasing:
- Uses PCHIP interpolation (monotonic spline)
- Post-processes with `np.maximum.accumulate()`

### Smoothing

PDF is smoothed to:
- Reduce noise from numerical differentiation
- Ensure positive density everywhere
- Improve sampling quality

### Normalization

PDF is normalized to integrate to 1:
```python
area = np.trapz(smoothed_density, grid_x)
smoothed_density = smoothed_density / area
```

## Use Cases

### Random Walk Generation

Used in `gen_walks.py` to convert varphi quantiles to PDFs for sampling:
```python
grid, _, cdf = generate_smooth_pdf(varphi_quant, taus, **best_pdf_params)
# Sample from CDF
samples = np.interp(uniform_samples, cdf, grid)
```

### Distribution Evaluation

Used to evaluate how well quantile predictions match realized returns.

## Model Files

- **Module**: `utils/dist_utils.py`
- **Usage**: Used by `varpi/gen_walks.py` and `varpi/wasserstein_min.py`

## References

- See `varpi/wasserstein_min.py` for parameter optimization
- See `varpi/gen_walks.py` for usage in walk generation

