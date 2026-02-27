"""Scikit-learn compatible Probabilistic PCA."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

try:
    from .ppca_rs import PPCARust
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "ppca_rs module not found. Build with 'pip install -e .' or 'maturin develop'"
    ) from e


class PPCA(BaseEstimator, TransformerMixin):
    """Probabilistic PCA with missing-value support (Tipping & Bishop, 1999).

    Uses EM to fit the model ``x = W z + mu + eps`` where ``z ~ N(0, I)``
    and ``eps ~ N(0, Psi)``.  Missing values are marginalised during the
    E-step so no imputation is required.

    Parameters
    ----------
    n_components : int, default=2
        Number of latent dimensions.
    max_iterations : int, default=100
        Maximum EM iterations.
    tol : float, default=1e-4
        Convergence tolerance on relative log-likelihood change.
    random_state : int, optional
        Seed for the loading-matrix initialisation.  ``None`` (default) uses
        a deterministic PCA warm-start; an integer uses a seeded random draw.
    noise_type : {"isotropic", "diagonal"}, default="isotropic"
        ``"isotropic"`` — classic PPCA: eps ~ N(0, sigma^2 I).
        ``"diagonal"`` — Factor Analysis: eps ~ N(0, diag(psi)).
    l2_penalty : float, default=0.0
        Ridge regularisation on W (adds lambda I in M-step).

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Principal axes in feature space (rows of W^T).
    explained_variance_ratio_ : ndarray of shape (n_components,)
        Fraction of total model variance per component.
    mean_ : ndarray of shape (n_features,)
        Per-feature mean from observed training values.
    noise_variances_ : ndarray of shape (n_features,)
        Per-feature noise variances (uniform when isotropic).
    noise_variance_ : float
        Mean noise variance across features.
    n_iter_ : int
        Number of EM iterations run.
    log_likelihoods_ : ndarray of shape (n_iter_,)
        Observed-data log-likelihood after each iteration.
    n_features_in_ : int
        Number of features seen during fit.
    n_samples_seen_ : int
        Number of samples seen during fit.

    """

    def __init__(
        self,
        n_components: int = 2,
        max_iterations: int = 100,
        tol: float = 1e-4,
        random_state: int | None = None,
        noise_type: str = "isotropic",
        l2_penalty: float = 0.0,
    ):
        self.n_components = n_components
        self.max_iterations = max_iterations
        self.tol = tol
        self.random_state = random_state
        self.noise_type = noise_type
        self.l2_penalty = l2_penalty

    def fit(self, X, y=None, missing_mask=None):
        """Fit the model to *X*.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : None
            Ignored.
        missing_mask : array-like of shape (n_samples, n_features), optional
            Boolean mask where ``True`` marks a missing value.

        Returns
        -------
        self : PPCA
            The fitted estimator.

        """
        X = check_array(X, accept_sparse=False, ensure_2d=True, dtype=np.float64, ensure_all_finite="allow-nan")
        n_samples, n_features = X.shape

        if self.n_components > n_features:
            raise ValueError(f"n_components ({self.n_components}) must be <= n_features ({n_features})")

        if self.n_components > n_samples:
            warnings.warn(
                f"n_components ({self.n_components}) > n_samples ({n_samples}). Results may be unstable.",
                UserWarning,
                stacklevel=2,
            )

        if self.noise_type not in ("isotropic", "diagonal"):
            raise ValueError(f"noise_type must be 'isotropic' or 'diagonal', got {self.noise_type!r}")

        if self.l2_penalty < 0:
            raise ValueError(f"l2_penalty must be >= 0, got {self.l2_penalty}")

        if missing_mask is None:
            missing_mask = np.zeros_like(X, dtype=bool)
        else:
            missing_mask = check_array(missing_mask, accept_sparse=False, ensure_2d=True, dtype=bool)
            if missing_mask.shape != X.shape:
                raise ValueError(f"missing_mask shape {missing_mask.shape} does not match X shape {X.shape}")

        self._rust_model = PPCARust(
            n_components=self.n_components,
            max_iterations=self.max_iterations,
            tol=self.tol,
            random_state=self.random_state,
            noise_type=self.noise_type,
            l2_penalty=self.l2_penalty,
        )
        self._rust_model.fit(X, missing_mask)

        self.mean_ = np.nanmean(np.where(missing_mask, np.nan, X), axis=0)
        self.explained_variance_ratio_ = np.array(self._rust_model.explained_variance_ratio())
        self.noise_variances_ = np.array(self._rust_model.noise_variances())
        self.noise_variance_ = float(self.noise_variances_.mean())
        self.components_ = self._rust_model.loadings().T
        self.n_iter_ = self._rust_model.n_iter()
        self.log_likelihoods_ = np.array(self._rust_model.log_likelihoods())
        self.n_features_in_ = n_features
        self.n_samples_seen_ = n_samples

        return self

    def transform(self, X):
        """Project *X* into latent space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Projected data.

        """
        check_is_fitted(self, ["_rust_model", "n_features_in_"])
        X = check_array(X, accept_sparse=False, ensure_2d=True, dtype=np.float64, ensure_all_finite="allow-nan")

        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features but model expects {self.n_features_in_}")

        X = np.where(np.isnan(X), self.mean_, X)
        return self._rust_model.transform(X)

    def fit_transform(self, X, y=None, missing_mask=None):
        """Fit and transform in one call.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : None
            Ignored.
        missing_mask : array-like of shape (n_samples, n_features), optional
            Boolean mask where ``True`` marks a missing value.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Projected data.

        """
        return self.fit(X, y, missing_mask).transform(X)

    def inverse_transform(self, X):
        """Reconstruct feature-space data from latent representation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            Data in latent space.

        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            Reconstructed data.

        """
        check_is_fitted(self, ["_rust_model"])
        X = check_array(X, accept_sparse=False, ensure_2d=True, dtype=np.float64, ensure_all_finite="allow-nan")

        if X.shape[1] != self.n_components:
            raise ValueError(f"X has {X.shape[1]} components but model has {self.n_components}")

        return self._rust_model.inverse_transform(X)

    def reconstruction_error(self, X, missing_mask=None):
        """Mean squared reconstruction error (observed entries only).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        missing_mask : array-like of shape (n_samples, n_features), optional
            Boolean mask where ``True`` marks a missing value.

        Returns
        -------
        error : float
            Mean squared error over observed entries.

        """
        check_is_fitted(self, ["_rust_model"])
        X = check_array(X, accept_sparse=False, ensure_2d=True, dtype=np.float64, ensure_all_finite="allow-nan")

        if missing_mask is None:
            missing_mask = np.zeros_like(X, dtype=bool)
        else:
            missing_mask = check_array(missing_mask, accept_sparse=False, ensure_2d=True, dtype=bool)

        return self._rust_model.reconstruction_error(X, missing_mask)
