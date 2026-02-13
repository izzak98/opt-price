# Utils Module Documentation

Utility functions and data processing modules used throughout the qStorm pipeline.

## Module Structure

```
utils/
├── data_utils.py    # Dataset classes and data loading
├── dist_utils.py   # PDF generation and distribution utilities
└── optuna_utils.py # Hyperparameter optimization helpers
```

## Documentation Files

- [Data Utils](data_utils.md) - Dataset classes and data loading
- [Distribution Utils](dist_utils.md) - PDF generation and distribution utilities

## Key Components

### Data Utils

- **DistDataset**: Training dataset for VarPhi
- **TestDataset**: Test/validation dataset with lookforward periods
- **DynamicBatchSampler**: Handles variable-length sequences

### Distribution Utils

- **generate_smooth_pdf**: Converts quantiles to smooth PDF/CDF
- **calculate_wasserstein**: Computes Wasserstein distance for PDF optimization

### Optuna Utils

- Hyperparameter optimization utilities
- Study management and result storage

## Usage

These utilities are used throughout the pipeline:
- Data loading in VarPhi training
- PDF generation in walk generation
- Hyperparameter optimization in model training

