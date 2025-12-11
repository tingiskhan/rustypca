# PPCA Development Guide

## Project Overview

PPCA is a Probabilistic Principal Component Analysis library with:
- **Rust backend** (`ppca-core`): High-performance core implementation using nalgebra
- **Python bindings** (`ppca-py`): PyO3 bindings providing scikit-learn compatible interface

## Architecture

```
ppca/
├── ppca-core/                    # Rust core
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs               # Entry point
│       ├── ppca.rs              # Main algorithm
│       └── errors.rs            # Error types
│
└── ppca-py/                      # Python package
    ├── Cargo.toml               # PyO3 config
    ├── pyproject.toml           # Package config
    ├── src/
    │   └── lib.rs               # PyO3 bindings
    ├── ppca/
    │   ├── __init__.py
    │   └── _ppca.py             # Scikit-learn interface
    └── tests/
        └── test_ppca.py         # Comprehensive tests
```

## Algorithm Implementation

The PPCA implementation uses the EM algorithm with the following key features:

### E-Step
Computes expectations of latent variables given observed data:
```
E[z_i | x_i^obs] = M W_obs^T x_obs
where M = (W_obs^T W_obs + σ² I)^-1
```

### M-Step
Updates loadings and noise variance:
```
W_new = X^T Z (Z^T Z + n·C^-1)^-1
σ²_new = Tr(X^T X - 2 X^T Z W + Z^T Z W^T W) / (n·p)
```

### Missing Value Handling
Missing values are treated as additional latent variables, computed in the E-step using only observed dimensions.

## Build Instructions

### Prerequisites

1. **Rust toolchain** (if building from source):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **Python 3.9+** with development headers

3. **maturin** for building PyO3 extensions:
   ```bash
   pip install maturin
   ```

### Building

```bash
cd ppca-py

# Development build (creates editable install)
maturin develop

# Or production build
pip install .
```

### Running Tests

```bash
cd ppca-py

# Run all tests
pytest tests/ -v

# Run specific test class
pytest tests/test_ppca.py::TestPPCABasics -v

# Run with coverage
pytest tests/ --cov=ppca --cov-report=html
```

## Testing Strategy

### Test Coverage

1. **Basic Functionality** (`TestPPCABasics`)
   - Initialization
   - Fit/transform/fit_transform
   - Reconstruction and error computation

2. **Missing Values** (`TestPPCAMissingValues`)
   - Fitting with missing value mask
   - Transform/inverse_transform with missing data
   - Error computation excluding missing values

3. **Comparison with sklearn** (`TestPPCAComparison`)
   - Shape consistency
   - Reconstruction error reduction with components
   - Explained variance tracking

4. **Edge Cases** (`TestPPCAEdgeCases`)
   - n_components > n_features
   - Transform before fit
   - Wrong feature dimensions
   - Missing value mask shape mismatch

5. **Numerical Stability** (`TestPPCANumericalStability`)
   - Very small values
   - Very large values
   - Constant features

### Comparison Methodology

Tests compare PPCA against scikit-learn's PCA on:
- **Transformed data shape**: Ensures output dimensions match
- **Reconstruction error**: Verifies that error decreases with more components
- **Edge case handling**: Confirms graceful degradation

## Performance Optimization

### Rust Implementation
- Uses nalgebra for efficient matrix operations
- SVD computed via nalgebra's built-in functions
- Memory-efficient row-wise operations for large datasets

### Python Binding Optimization
- NumPy array conversion minimizes copying
- Direct nalgebra-NumPy interop via PyO3's numpy feature
- Lazy evaluation where possible

## Extension Points

### Adding new features

1. **Rust side** (`ppca-core/src/ppca.rs`):
   - Add new methods to the `PPCA` struct
   - Update result types if needed

2. **Python binding** (`ppca-py/src/lib.rs`):
   - Add PyO3 methods to `PPCARust`
   - Handle numpy array conversions

3. **Python wrapper** (`ppca-py/ppca/_ppca.py`):
   - Add methods to the `PPCA` class
   - Maintain sklearn compatibility

## Troubleshooting

### Build Issues

**maturin not found**:
```bash
pip install maturin
```

**Rust compilation errors**:
- Ensure Rust is up-to-date: `rustup update`
- Check Cargo.toml dependencies match your Rust version

**PyO3 issues**:
- Rebuild with `maturin develop --release` for debug info
- Check Python version compatibility (requires 3.9+)

### Runtime Issues

**Module not found**:
```bash
cd ppca-py
maturin develop
```

**Segmentation fault**:
- Check for mismatched array dimensions
- Verify numpy arrays are contiguous

## Contributing Guidelines

1. **Test coverage**: All new features must have corresponding tests
2. **Documentation**: Update docstrings using NumPy format
3. **Performance**: Benchmark changes on large datasets
4. **Compatibility**: Maintain sklearn BaseEstimator/TransformerMixin interface

## References

- Tipping & Bishop (1999): "Probabilistic Principal Component Analysis"
  *Journal of the Royal Statistical Society*, Series B
- nalgebra documentation: https://nalgebra.org/
- PyO3 guide: https://pyo3.rs/
- scikit-learn API: https://scikit-learn.org/

## License

MIT
