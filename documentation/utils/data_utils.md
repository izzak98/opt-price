# Data Utils Documentation

Utility functions and dataset classes for loading and processing financial data.

## Overview

The `data_utils` module provides:
- Dataset classes for training and testing
- Data loading and preprocessing functions
- Dynamic batching for variable-length sequences

## Dataset Classes

### DistDataset

Training dataset for VarPhi (qLSTM) model.

**Purpose**: Load historical data for return distribution prediction

**Features**:
- Variable-length sequences
- Rolling normalization
- Asset-neutral features with categorical encoding

**Data Structure**:
- `x`: Input features (normalized returns)
- `s`: Cross-sectional volatility
- `z`: Market data indices
- `y`: Target returns
- `sy`: Standardized returns (for two-stage loss)

**Usage**:
```python
dataset = get_dataset(
    normalization_lookback=219,
    start_date="1998-01-01",
    end_date="2018-01-01"
)
```

### TestDataset

Test/validation dataset with lookforward periods.

**Purpose**: Generate data for specific prediction horizons

**Features**:
- Configurable lookforward period (T days)
- Test mode with observed returns
- Date-based querying

**Data Structure**:
- Same as DistDataset, but with `lookforward` parameter
- In test mode: returns `(x, s, z, y, observed_returns)`
- In training mode: returns `(x, s, z, y, sy)`

**Usage**:
```python
dataset = get_test_dataset(
    normalization_lookback=219,
    start_date="2018-01-01",
    end_date="2019-01-01",
    lookforward=30,  # Predict 30 days ahead
    test=True
)
```

## Key Functions

### get_dataset()

Creates DistDataset for training/validation.

**Parameters**:
- `normalization_lookback`: Rolling window size for normalization
- `start_date`: Start date (YYYY-MM-DD)
- `end_date`: End date (YYYY-MM-DD)
- `lookahead`: Optional lookahead period
- `flow`: Whether to include observed returns

### get_test_dataset()

Creates TestDataset with lookforward period.

**Parameters**:
- `normalization_lookback`: Rolling window size
- `start_date`: Start date
- `end_date`: End date
- `lookforward`: Prediction horizon in days
- `test`: Whether to return observed returns

### DynamicBatchSampler

Handles variable-length sequences efficiently.

**Features**:
- Groups sequences by length
- Creates batches of similar-length sequences
- Pads sequences to batch maximum length

**Usage**:
```python
sampler = DynamicBatchSampler(dataset, batch_size=32)
loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn)
```

## Data Processing

### Normalization

Rolling window normalization:
```python
rolling_mean = df.rolling(window=normalization_lookback).mean()
rolling_std = df.rolling(window=normalization_lookback).std()
normalized_df = (df - rolling_mean) / rolling_std
```

### Cross-Sectional Volatility

Computes volatility across asset classes:
```python
cross_vol = cross_sectional_volatility(groupings, decay_factor=0.94)
```

Uses exponentially weighted moving average (EWMA).

### Categorical Encoding

One-hot encoding for asset classes:
- cryptocurrencies
- currency pairs
- commodities
- euro stoxx 50
- s&p 500
- nikkei 225

## File Structure

### Input Files
- `data/{asset_class}/{asset}.csv` - Asset price data
- `data/market_data/market_data.csv` - Market-wide data

### Data Format
- Index: DatetimeIndex
- Columns: `return_2d` (2-day returns), price data, etc.

## Performance Considerations

### Memory
- Datasets load data into memory
- Large datasets may require significant RAM

### Speed
- DynamicBatchSampler reduces padding overhead
- Caching normalization statistics improves speed

## Model Files

- **Module**: `utils/data_utils.py`
- **Usage**: Used by `varpi/train_varphi.py` and `varpi/generate_varphi_quants.py`

