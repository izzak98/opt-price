# qStorm Model Architecture

The qStorm model is a physics-informed neural network that solves stochastic partial differential equations (SPDEs) for option pricing.

## Overview

**Purpose**: Predict normalized option prices V'(S', t') given:
- Normalized stock price S' = S/K
- Time-to-expiration t' (in days)
- Risk-free rate rf (daily)
- Varphi quantiles (return distribution)

**Model Type**: Feedforward neural network with physics-informed loss

## Architecture

### Input Features

The model takes:
- **S'**: Normalized stock price (S/K), shape: `(batch_size, 1)`
- **t'**: Time-to-expiration in days, shape: `(batch_size, 1)`
- **rf**: Risk-free rate (daily, already /365), shape: `(batch_size, 1)`
- **varphi_q**: Quantiles from varphi model, shape: `(batch_size, num_quantiles)`

### Network Structure

Multi-layer perceptron (MLP) with:
- Input layer: Concatenates all inputs
- Hidden layers: Multiple fully connected layers with activation
- Output layer: Single value (normalized option price)

### Output

- **V'**: Normalized option price (V/K), shape: `(batch_size, 1)`

## Physics-Informed Learning

Unlike standard neural networks, qStorm incorporates physical constraints:

1. **SPDE Residual**: Enforces the stochastic PDE
2. **Boundary Conditions**: Enforces terminal, lower, and upper boundaries
3. **Inequality Constraints**: Enforces American option constraints

These constraints are incorporated into the loss function rather than the architecture itself.

## Stochastic Coefficient (varrho)

The SPDE uses a stochastic volatility coefficient ρ computed from quantiles:

```python
ρ = calc_varrho(taus, varpi_q, varpi_grid, varpi_pdf, t')
```

**Process**:
1. Sample z ~ N(0, 1)
2. Compute ρ_G = φ(z) (standard normal PDF)
3. Compute u = Φ(z) (standard normal CDF)
4. Map u to quantile q(u) via varphi quantiles
5. Get density f(q) from varphi PDF
6. Compute ρ = ρ_G / (sqrt(t') * f(q))

This coefficient adapts to the learned return distribution.

## Normalization

All inputs/outputs are normalized:
- **S' = S/K**: Stock price normalized by strike
- **V' = V/K**: Option price normalized by strike
- **t'**: Time in days (not normalized)
- **rf**: Daily risk-free rate

This normalization makes the model scale-invariant and improves training stability.

## Model Files

- **Architecture**: `q_storm/qStorm.py`
- **Training**: `q_storm/qstorm_train.py`
- **Loss Implementation**: `q_storm/StormTrainer.py`

## References

See [Training](training.md) for loss function details and [MATHEMATICAL_FORMULATION.md](../../MATHEMATICAL_FORMULATION.md) for complete mathematical formulation.

