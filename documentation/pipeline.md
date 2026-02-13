# qStorm Pipeline Overview

This document describes the end-to-end workflow of the qStorm option pricing framework.

## High-Level Pipeline

```
Historical Data → VarPhi (qLSTM) → Quantile Predictions → Random Walks → VarPi → qStorm → Option Prices
```

## Stage-by-Stage Breakdown

### Stage 1: VarPhi (qLSTM) - Return Distribution Prediction

**Purpose**: Predict quantiles of return distributions from historical data

**Input**: 
- Historical asset features (returns, volatility, etc.)
- Market-wide features
- Sequence length: variable (typically 2-22 days)

**Output**: 
- 37 quantiles of the return distribution for a given time horizon T

**Key Files**:
- `varpi/train_varphi.py` - Training script
- `varpi/QLSTM.py` - Model architecture
- `varpi/varpi_utils/varphi_utils.py` - Quantile loss and utilities

**Training Process**:
1. Load historical data (1998-2018 for training, 2018-2019 for validation)
2. Normalize features using rolling statistics
3. Train qLSTM with two-stage quantile loss (raw + normalized)
4. Save model to `models/varphi.pth`

### Stage 1.5: Quantile Prediction Generation

**Purpose**: Generate quantile predictions for different time horizons

**Input**: Trained varphi model, validation dataset

**Output**: Pickle files with quantile predictions for T ∈ [15, 30] days

**Key Files**:
- `varpi/generate_varphi_quants.py` - Generation script

**Process**:
1. Load trained varphi model
2. For each T in [15, 30]:
   - Create validation dataset with `lookforward=T`
   - Run inference to get quantile predictions
   - Save to `stored_quants/quantiles_{T}.pkl`

**Why T ≥ 15?**: While qLSTM can handle shorter horizons (even T=1), it performs better with more samples. The minimum of 15 days ensures:
- Sufficient data quality for reliable quantile estimation
- Better statistical properties for random walk generation
- More stable VarPi training

### Stage 2: Random Walk Generation

**Purpose**: Generate synthetic random walks from varphi quantiles

**Input**: Quantile predictions from Stage 1.5

**Output**: Training dataset for VarPi

**Key Files**:
- `varpi/gen_walks.py` - Walk generation script

**Process**:
1. Load quantile predictions for each T
2. For each quantile prediction:
   - Convert quantiles → smooth PDF using `generate_smooth_pdf()`
   - Generate 10,000 random walks by sampling from PDF
   - Compute quantiles of walks at each time step
   - Store: `(varphi_quants, t, T, walk_quantiles[t])`
3. Save to `walk_dataset.pkl`

**Data Structure**:
- Each sample: `(X, t, T, Y)` where:
  - `X`: varphi quantiles (37 values)
  - `t`: normalized time position (t/T)
  - `T`: normalized total walk length (T/30)
  - `Y`: random walk quantiles at time t (37 values)

### Stage 3: VarPi - Random Walk Quantile Prediction

**Purpose**: Learn mapping from return quantiles to random walk quantiles

**Input**: Random walk dataset from Stage 2

**Output**: Trained VarPi model

**Key Files**:
- `varpi/tain_varpi.py` - Training script

**Process**:
1. Load random walk dataset
2. Split into train/val/test (70/20/10)
3. Train deep MLP to predict walk quantiles given:
   - varphi quantiles
   - normalized time position
   - normalized total walk length
4. Save model to `models/varpi.pth`

**Architecture**: Deep MLP with decreasing hidden dimensions: [4096, 4096, 4096, 2048, 2048, 1024, 256, 256, 64, 32]

### Stage 4: qStorm - Option Pricing

**Purpose**: Price options using SPDE-based approach

**Input**: VarPhi and VarPi models, option data

**Output**: Option prices

**Key Files**:
- `q_storm/qStorm.py` - Model architecture
- `q_storm/qstorm_train.py` - Training script
- `q_storm/StormTrainer.py` - SPDE loss implementation
- `q_storm/StormSampler.py` - Training data sampling

**Process**:
1. Sample training points:
   - Stock prices (S'), time-to-expiration (t'), risk-free rates (rf)
   - VarPhi quantiles for return distribution
2. Use VarPi to predict random walk quantiles
3. Train neural network to solve SPDE:
   ```
   ∂V'/∂t' + (1/2)(S')²ρ²∂²V'/∂(S')² + rf·S'·∂V'/∂S' - rf·V' = 0
   ```
4. Enforce boundary conditions:
   - Terminal: V'(S', 0) = max(S' - 1, 0) for calls
   - Lower: V'(0, t') = 0
   - Upper: V'(S_max, t') = S_max - 1
   - American: V'(S', t') ≥ max(S' - 1, 0)

## Data Flow Diagram

```
Historical Data
    ↓
[VarPhi Training]
    ↓
Trained VarPhi Model
    ↓
[Quantile Generation] → stored_quants/quantiles_{T}.pkl
    ↓
[Walk Generation] → walk_dataset.pkl
    ↓
[VarPi Training]
    ↓
Trained VarPi Model
    ↓
[qStorm Training] → Option Prices
```

## File Dependencies

```
train_varphi.py
    → QLSTM.py
    → varphi_utils.py
    → data_utils.py

generate_varphi_quants.py
    → varphi_utils.py (get_all_quantiles)
    → data_utils.py (get_test_dataset)
    → models/varphi.pth

gen_walks.py
    → stored_quants/quantiles_{T}.pkl
    → dist_utils.py (generate_smooth_pdf)
    → walk_dataset.pkl

tain_varpi.py
    → gen_walks.py (get_walk_dataset)
    → models/varpi.pth

qstorm_train.py
    → qStorm.py
    → StormTrainer.py
    → StormSampler.py
    → models/varphi.pth
    → models/varpi.pth
```

## Configuration

All configuration is in `config.json`:
- **Dates**: Training, validation, and test periods
- **Quantiles**: 37 quantile levels (0.00005 to 0.99995)
- **Time Horizons**: T ∈ [15, 30] days for walk generation

## Output Files

- `models/varphi.pth` - Trained VarPhi model
- `models/varpi.pth` - Trained VarPi model
- `models/q_storm.pth` - Trained qStorm model
- `stored_quants/quantiles_{T}.pkl` - Quantile predictions for each T
- `walk_dataset.pkl` - Random walk training dataset
- `optuna_dbs/optuna.db` - Hyperparameter optimization results

