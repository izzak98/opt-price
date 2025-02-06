# qStorm: A Quantile Stochastic Regression Model for Option Pricing

## Overview

qStorm is a deep learning framework designed for **nonparametric option pricing**. Unlike traditional models such as Black-Scholes, which assume normal log returns and constant volatility, qStorm **learns arbitrary return distributions** directly from data. It integrates **stochastic partial differential equations (SPDEs)** with **physics-informed neural networks (PINNs)** to achieve robust and accurate derivative pricing.

The methodology behind qStorm is detailed in our paper:

> *From Arbitrary Returns to SPDEs: qStorm, a Quantile STOchastic Regression Model for Option Pricing*  
> Anonymous Authors, 2025.  

## Features

- **Nonparametric Return Modeling:** Uses a learned distribution instead of assuming normality.
- **SPDE-Based Option Pricing:** Derives option values from an SPDE that naturally emerges from arbitrary return distributions.
- **Physics-Informed Neural Networks (PINNs):** Enforces financial constraints and mathematical consistency.
- **Quantile-Based Approach:** Utilizes quantile regression for density estimation.

## Repository Structure

```
├── data_operations       # Data fetching and preprocessing
│   ├── data_fetch.py     # Loads and processes market data
│   ├── data_utils        # Utility functions for data handling
│   └── ta_utils.py       # Technical analysis utilities
├── infrence.py           # Inference script for qStorm models
├── optuna_dbs            # Databases for hyperparameter optimization
│   ├── optuna.db
│   ├── sharpe_study.db
│   └── wasserstein_distance.db
├── q_storm               # Core qStorm implementation
│   ├── qStorm.py         # The qStorm model architecture
│   ├── qstorm_train.py   # Training script for qStorm
│   ├── StormSampler.py   # Sampling module for training
│   ├── StormTrainer.py   # Trainer implementing the SPDE residual loss
│   └── storm_utils
│       └── varrho.py     # Computes the stochastic drift term
├── utils                 # General utility functions
│   ├── data_utils.py     # Data loading and processing functions
│   ├── dist_utils.py     # Distribution utilities for sampling
│   └── optuna_utils.py   # Hyperparameter optimization helpers
├── varpi                 # VarPi: The second-stage quantile model
│   ├── QLSTM.py          # qLSTM model for log return estimation
│   ├── tain_varpi.py     # Training script for VarPi
│   ├── train_varphi.py   # Training script for first-stage quantiles
│   ├── gen_walks.py      # Generates quantile-based random walks
│   ├── wasserstein_min.py # Wasserstein distance minimization
│   └── varpi_utils
│       └── varphi_utils.py # Quantile transformation utilities
```

## Installation

To use qStorm, ensure you have the following dependencies installed:

```bash
pip install torch numpy pandas tqdm optuna accelerate pyarrow
```

For GPU acceleration, install the appropriate version of **CUDA** for PyTorch.

## Training the Model

1. **Train `VarPi`** (quantile model for option prices):

   ```bash
   python varpi/tain_varpi.py
   ```

2. **Train `qStorm`** (SPDE-based option pricing model):

   ```bash
   python q_storm/qstorm_train.py
   ```

3. **Run inference on option pricing data**:

   ```bash
   python infrence.py
   ```
