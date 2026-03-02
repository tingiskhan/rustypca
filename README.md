# rustypca

A Python library for Probabilistic PCA that handles missing data without asking you to impute first. It uses the EM algorithm under the hood, with the number-crunching done in Rust.

## Why?

Regular PCA doesn't cope well when your data has holes in it. Probabilistic PCA treats missing values as latent variables in a generative model — a more principled approach than patching NaNs and hoping for the best.

For the full story, see [Tipping & Bishop (1999)](https://www.robots.ox.ac.uk/~cbishop/papers/PPCA.pdf), *"Probabilistic Principal Component Analysis"*, Journal of the Royal Statistical Society, 61(3), 611–622.

## Features

- **Rust backend** — keeps things snappy on larger datasets
- **Missing value support** — the main reason this exists
- **Scikit-learn compatible** — fits in wherever you'd use `sklearn.decomposition.PCA`

## Installation

```bash
pip install -e .
```

You'll need a Rust toolchain to build from source. Python >= 3.10.

## Quick start

```python
import numpy as np
from rustypca import PPCA

X = np.random.randn(100, 10)

model = PPCA(n_components=2)
X_transformed = model.fit_transform(X)
X_reconstructed = model.inverse_transform(X_transformed)
```

### With missing values

```python
import numpy as np
from rustypca import PPCA

X = np.random.randn(100, 10)
missing_mask = np.random.rand(100, 10) < 0.1  # 10% missing

model = PPCA(n_components=2)
model.fit(X, missing_mask=missing_mask)
```

No preprocessing or imputation needed.

## Testing

```bash
make test
```

## Disclaimer

This project was built with the help of [Claude](https://claude.ai).

## License

MIT
