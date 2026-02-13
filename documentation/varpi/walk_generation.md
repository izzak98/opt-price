# Random Walk Generation Documentation

This document explains how synthetic random walks are generated from varphi quantile predictions for VarPi training.

## Overview

**Purpose**: Generate training data for VarPi by creating synthetic random walks from varphi quantile predictions

**Input**: Varphi quantile predictions for different time horizons T ∈ [15, 30]

**Output**: Training dataset with (varphi_quants, t, T, walk_quantiles) tuples

## Process Flow

```
Varphi Quantiles → PDF Generation → Random Walk Sampling → Quantile Computation → Training Dataset
```

## Step-by-Step Process

### 1. Load Quantile Predictions

For each time horizon T ∈ [15, 30]:
- Load `stored_quants/quantiles_{T}.pkl`
- Extract `all_pred_quantiles` for each asset class
- Each quantile vector contains 37 quantile levels

### 2. Convert Quantiles to PDF

For each quantile prediction:
```python
grid, pdf, cdf = generate_smooth_pdf(varphi_quant, taus, **best_pdf_params)
```

**Process**:
- Interpolate quantiles to create smooth CDF
- Compute PDF via numerical differentiation
- Smooth PDF using convolution (window=61 by default)
- Ensure non-negative density and normalize

**Output**:
- `grid`: Support points for PDF
- `pdf`: Probability density values
- `cdf`: Cumulative distribution function

### 3. Generate Random Walks

For each quantile prediction, generate 10,000 random walks:

```python
def generate_random_walk(grid, cdf, T):
    # Sample T uniform random values
    random_uniform_samples = np.random.uniform(0, 1, T)
    # Map through CDF to get returns
    predicted_samples = np.interp(random_uniform_samples, cdf, grid)
    # Prepend 0 and cumulatively sum
    predicted_samples = np.concatenate([[0], predicted_samples])
    discrete_random_walk = np.cumsum(predicted_samples)
    return discrete_random_walk
```

**Result**: Random walk of length T+1: `[0, r1, r1+r2, r1+r2+r3, ...]`

### 4. Compute Walk Quantiles

For each time step t ∈ [0, T-1]:
- Extract values at time t from all 10,000 walks
- Compute 37 quantiles: `np.quantile(walks[:, t], taus)`

**Output**: `varpi_quants` with shape `(T, 37)` - quantiles at each time step

### 5. Create Training Samples

For each time step t in the walk:
```python
dataset.ingest(varphi_quant, varpi_quants, T)
```

This creates training samples:
- **X**: varphi quantiles (37 values) - input
- **t**: normalized time position (t/T) - input  
- **T**: normalized total walk length (T/30) - input
- **Y**: random walk quantiles at time t (37 values) - target

## Why This Works

### Intuition

1. **Varphi predicts return distribution**: Given historical data, varphi predicts quantiles of the return distribution for T days ahead

2. **Random walks simulate cumulative returns**: By sampling from the predicted distribution and cumulatively summing, we simulate possible paths of cumulative returns

3. **VarPi learns the mapping**: VarPi learns to predict quantiles of cumulative returns (random walks) given quantiles of individual returns (varphi)

### Mathematical Justification

If returns are i.i.d. with distribution F, then:
- Individual return quantiles: Q_τ(F)
- Cumulative return quantiles: Q_τ(F*T) where F*T is T-fold convolution

VarPi learns this convolution mapping without assuming a parametric form for F.

## Dataset Structure

### WalkDataSet Class

```python
class WalkDataSet:
    X: List[varphi_quantiles]      # Input: return distribution quantiles
    t: List[int]                    # Time position in walk
    T: List[int]                    # Total walk length
    Y: List[walk_quantiles]         # Target: cumulative return quantiles
    max_T: int                      # Maximum T seen so far
```

### Sample Format

Each `__getitem__` returns:
- **X**: `torch.Tensor` shape `(37,)` - varphi quantiles
- **t**: `torch.Tensor` shape `(1,)` - normalized time (t/T)
- **T**: `torch.Tensor` shape `(1,)` - normalized total length (T/30)
- **Y**: `numpy.ndarray` shape `(37,)` - walk quantiles at time t

## Time Horizon Constraints

### Minimum T = 15 Days

**Why?** While qLSTM can handle shorter horizons (even T=1), it performs better with more samples. The minimum of 15 days ensures:

1. **Sufficient data quality**: More samples lead to more stable quantile estimates
2. **Better statistical properties**: Random walks of length ≥15 have better convergence properties
3. **More stable VarPi training**: Longer walks provide more training samples per varphi prediction

### Maximum T = 30 Days

Covers typical short-term option expiration ranges (2-4 weeks).

## Implementation Details

### PDF Generation Parameters

Optimized via Wasserstein distance minimization:
- `min_density`: Minimum PDF value (default: 1e-3)
- `eps`: Numerical stability threshold (default: 1e-6)
- `window`: Smoothing window size (default: 61)
- `grid_points`: Number of grid points (default: 1000)

### Random Walk Sampling

- **Number of samples**: 10,000 walks per quantile prediction
- **Sampling method**: Inverse CDF transform (uniform → CDF → returns)
- **Initial value**: Always starts at 0 (no initial return)

### Quantile Computation

- **Quantile levels**: Same 37 levels as varphi (0.00005 to 0.99995)
- **Method**: `np.quantile()` with linear interpolation
- **Time steps**: Computed for t ∈ [1, T-1] (t=0 always has quantile 0)

## File Structure

### Input Files
- `stored_quants/quantiles_{T}.pkl` - Varphi quantile predictions

### Output Files
- `walk_dataset.pkl` - Complete training dataset for VarPi

### Intermediate Files
- None (all processing done in memory)

## Usage

### Generate Full Dataset
```bash
python varpi/gen_walks.py
```

### Resume Generation
If `walk_dataset.pkl` exists but `max_T < 30`, the script automatically resumes from `max_T + 1`.

### Check Dataset Size
```python
from varpi.gen_walks import get_walk_dataset
dataset = get_walk_dataset()
print(f"Dataset size: {len(dataset)}")
print(f"Max T: {dataset.max_T}")
```

## Performance Considerations

- **Memory**: Each quantile prediction generates 10,000 walks, so memory usage scales with number of predictions
- **Time**: PDF generation and quantile computation are the bottlenecks
- **Parallelization**: Currently sequential, but could be parallelized across asset classes

## References

- See `varpi/gen_walks.py` for implementation
- See `utils/dist_utils.py` for PDF generation
- See `varpi/wasserstein_min.py` for PDF parameter optimization

