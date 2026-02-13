# qStorm Module Documentation

The qStorm module implements the SPDE-based option pricing model that uses quantile predictions from VarPhi and VarPi to price options.

## Module Structure

```
q_storm/
├── qStorm.py              # Main model architecture
├── qstorm_train.py        # Training script
├── StormTrainer.py        # SPDE loss implementation
├── StormSampler.py        # Training data sampling
└── storm_utils/
    └── varrho.py         # Stochastic volatility coefficient
```

## Documentation Files

- [Model Architecture](model.md) - Neural network architecture
- [Training](training.md) - SPDE residual loss and training procedure
- [Sampling](sampling.md) - Training data generation

## Quick Reference

### Training qStorm
```bash
python q_storm/qstorm_train.py
```

### Model Components

- **qStorm Model**: Neural network that predicts option prices V'(S', t')
- **StormTrainer**: Implements SPDE residual loss and boundary conditions
- **StormSampler**: Generates training samples with varphi quantiles
- **varrho**: Computes stochastic volatility coefficient from quantiles

## Key Concepts

### SPDE Formulation

The model solves the stochastic partial differential equation:
```
∂V'/∂t' + (1/2)(S')²ρ²∂²V'/∂(S')² + rf·S'·∂V'/∂S' - rf·V' = 0
```

Where:
- V'(S', t'): Normalized option price
- S' = S/K: Normalized stock price
- t': Time-to-expiration in days
- ρ: Stochastic volatility coefficient (from varrho)
- rf: Risk-free rate (daily)

### Boundary Conditions

1. **Terminal**: V'(S', 0) = max(S' - 1, 0) for calls
2. **Lower**: V'(0, t') = 0
3. **Upper**: V'(S_max, t') = S_max - 1
4. **American**: V'(S', t') ≥ max(S' - 1, 0)

### Loss Function

Total loss combines:
- SPDE residual loss (Monte Carlo averaged)
- Terminal payoff loss
- Lower boundary loss
- Upper boundary loss
- American option inequality loss

## Integration with VarPhi/VarPi

qStorm uses:
- **VarPhi**: Provides quantiles of return distributions
- **VarPi**: Predicts quantiles of random walks from return quantiles
- **varrho**: Computes stochastic coefficient from quantiles

See [Pipeline Overview](../pipeline.md) for the complete workflow.

