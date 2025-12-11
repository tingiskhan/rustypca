# Probabilistic PCA Library

A high-performance Python library implementing Probabilistic Principal Component Analysis (PPCA) with support for missing values using the EM algorithm.

## Features

- **Rust Backend**: High-performance PPCA implementation using Rust and nalgebra
- **Missing Value Support**: Handle missing values during the EM algorithm
- **Scikit-learn Compatible**: Drop-in replacement for scikit-learn's PCA with extended functionality
- **Python Bindings**: Clean Python interface via PyO3

## Installation

### Build from source

```bash
cd ppca-py
pip install -e .
```

This will build the Rust extension and install the Python package.

### Requirements

- Python >= 3.9
- Rust toolchain (for building from source)
- numpy >= 1.21.0
- scikit-learn >= 1.0.0

## Usage

### Basic Usage

```python
import numpy as np
from ppca import PPCA

# Create data
X = np.random.randn(100, 10)

# Fit and transform
ppca = PPCA(n_components=2)
X_transformed = ppca.fit_transform(X)

# Reconstruct
X_reconstructed = ppca.inverse_transform(X_transformed)

# Get reconstruction error
error = ppca.reconstruction_error(X)
```

### With Missing Values

```python
import numpy as np
from ppca import PPCA

# Create data with missing values
X = np.random.randn(100, 10)
missing_mask = np.random.rand(100, 10) < 0.1  # 10% missing

# Fit model handling missing values
ppca = PPCA(n_components=2)
ppca.fit(X, missing_mask=missing_mask)

# Transform (no missing mask needed for new data)
X_transformed = ppca.transform(X)

# Compute error on data with missing values
error = ppca.reconstruction_error(X, missing_mask=missing_mask)
```

## API Reference

### PPCA

#### Parameters

- **n_components** (int, default=2): Number of components to extract
- **max_iterations** (int, default=100): Maximum number of EM iterations
- **tol** (float, default=1e-4): Convergence tolerance

#### Methods

- **fit(X, y=None, missing_mask=None)**: Fit the model
- **transform(X)**: Transform data to latent space
- **fit_transform(X, y=None, missing_mask=None)**: Fit and transform
- **inverse_transform(Y)**: Reconstruct data from latent space
- **reconstruction_error(X, missing_mask=None)**: Compute MSE reconstruction error
- **get_params(deep=True)**: Get parameters (sklearn compatible)
- **set_params(**params)**: Set parameters (sklearn compatible)

#### Attributes (after fitting)

- **components_**: Principal axes
- **explained_variance_ratio_**: Explained variance by component
- **mean_**: Feature means
- **noise_variance_**: Estimated Gaussian noise variance
- **n_features_in_**: Number of input features

## Testing

Run the test suite:

```bash
cd ppca-py
pytest tests/
```

Tests compare the PPCA implementation against scikit-learn's PCA for:
- Transformed data shape and consistency
- Reconstruction error
- Explained variance
- Edge cases and numerical stability

## Algorithm

PPCA uses the EM algorithm as described in:

**Tipping & Bishop (1999)**: "Probabilistic Principal Component Analysis" 
*Journal of the Royal Statistical Society*, 61(3), 611-622

The implementation supports missing values by treating them as latent variables in the E-step.

## Project Structure

```
ppca/
├── ppca-core/          # Rust core implementation
│   └── src/
│       ├── lib.rs      # Library entry point
│       ├── ppca.rs     # Main PPCA algorithm
│       └── errors.rs   # Error types
├── ppca-py/            # Python package and bindings
│   ├── src/
│   │   └── lib.rs      # PyO3 bindings
│   ├── ppca/
│   │   ├── __init__.py
│   │   └── _ppca.py    # Scikit-learn interface
│   ├── tests/
│   │   └── test_ppca.py
│   ├── Cargo.toml
│   └── pyproject.toml
└── README.md
```

## Development

### Building the Rust extension

```bash
cd ppca-py
maturin develop
```

### Running tests

```bash
cd ppca-py
pytest tests/ -v
```

### Building documentation

Documentation is provided inline in docstrings following NumPy format.

## License

MIT License

## Contributing

Contributions are welcome! Please ensure:
1. Code passes all tests
2. New features include comprehensive tests
3. Documentation is updated accordingly
