# rustypca

If you've ever had the misfortune of staring at a dataset full of NaNs and trying to perform PCA, wondering where your afternoon went — this library is for you.

**rustypca** is a Python library for Probabilistic PCA that actually handles missing data gracefully, without requiring you to impute first and pray later. It uses the EM algorithm under the hood, with the heavy lifting done in Rust so you're not waiting around.

## Why?

Regular PCA falls apart the moment your data has holes in it. Probabilistic PCA treats the problem properly — missing values become latent variables in a generative model, which is a fancy way of saying "we do the math right instead of duct-taping NaNs."

For the full story, see [Tipping & Bishop (1999)](https://www.robots.ox.ac.uk/~cbishop/papers/PPCA.pdf), *"Probabilistic Principal Component Analysis"*, Journal of the Royal Statistical Society, 61(3), 611–622.

## Features

- **Rust backend** — fast enough that you can actually use it on real data
- **Missing value support** — the whole point, really
- **Scikit-learn compatible** — drop it in wherever you'd use `sklearn.decomposition.PCA`

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

No preprocessing. No imputation. Just hand it the data and the mask.

## Testing

```bash
make test
```

## Disclaimer

This project was built with the help of [Claude](https://claude.ai).

## License

MIT
