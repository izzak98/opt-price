# qStorm Documentation

This directory contains comprehensive documentation for the qStorm option pricing framework. The documentation is organized by module, following the same structure as the codebase.

## Overview

qStorm is a **Quantile STOchastic Regression Model** for nonparametric option pricing. It learns arbitrary return distributions from data and uses stochastic partial differential equations (SPDEs) with physics-informed neural networks (PINNs) for derivative pricing.

## Documentation Structure

### Core Modules

- **[Pipeline Overview](pipeline.md)** - End-to-end workflow and data flow
- **[VarPi Module](varpi/README.md)** - Stage 1 (varphi) and Stage 2 (varpi) quantile models
  - [VarPhi Training](varpi/varphi.md) - qLSTM model for return distribution prediction
  - [VarPi Training](varpi/varpi.md) - Random walk quantile prediction model
  - [Walk Generation](varpi/walk_generation.md) - Synthetic random walk dataset creation
- **[qStorm Module](q_storm/README.md)** - SPDE-based option pricing model
  - [Model Architecture](q_storm/model.md) - Neural network architecture
  - [Training](q_storm/training.md) - SPDE residual loss and training procedure
  - [Sampling](q_storm/sampling.md) - Training data generation
- **[Utils Module](utils/README.md)** - Utility functions and data processing
  - [Data Utils](utils/data_utils.md) - Dataset classes and data loading
  - [Distribution Utils](utils/dist_utils.md) - PDF generation and distribution utilities

## Quick Start

1. **Train VarPhi (Stage 1)**: Predict return distribution quantiles
   ```bash
   python varpi/train_varphi.py
   ```

2. **Generate Quantile Predictions**: Create quantiles for different time horizons
   ```bash
   python varpi/generate_varphi_quants.py
   ```

3. **Generate Random Walks**: Create training data for VarPi
   ```bash
   python varpi/gen_walks.py
   ```

4. **Train VarPi (Stage 2)**: Learn random walk quantile mapping
   ```bash
   python varpi/tain_varpi.py
   ```

5. **Train qStorm**: Learn option pricing from SPDE
   ```bash
   python q_storm/qstorm_train.py
   ```

## Key Concepts

### Two-Stage Pipeline

1. **Stage 1 (VarPhi)**: Predicts quantiles of return distributions from historical data
2. **Stage 2 (VarPi)**: Predicts quantiles of random walks (cumulative returns) from return quantiles
3. **qStorm**: Uses these quantiles to solve SPDEs for option pricing

### Time Horizons

- **Minimum T = 15 days**: While qLSTM can handle shorter horizons (even T=1), it performs better with more samples. The minimum of 15 days ensures sufficient data quality for reliable quantile estimation and random walk generation.
- **Maximum T = 30 days**: Covers typical short-term option expiration ranges

### Quantile-Based Approach

The entire pipeline uses quantile regression to capture full return distributions without parametric assumptions, enabling nonparametric option pricing.

## Mathematical Background

See [MATHEMATICAL_FORMULATION.md](../MATHEMATICAL_FORMULATION.md) for the complete mathematical formulation of the qStorm loss function.

## Papers

- **qStorm**: *From Arbitrary Returns to SPDEs: qStorm, a Quantile STOchastic Regression Model for Option Pricing*
- **qLSTM**: See `qLSTM-1.pdf` for the quantile LSTM methodology

