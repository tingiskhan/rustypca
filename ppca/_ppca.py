"""Scikit-learn compatible PPCA implementation."""

import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

try:
    from .ppca_rs import PPCARust
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "ppca_rs module not found. Please build the Rust extension with 'pip install -e .' or 'maturin develop'"
    ) from e


class PPCA(BaseEstimator, TransformerMixin):
    """Probabilistic PCA with missing value support.

    Implements PPCA via the EM algorithm. Missing values can be supplied
    as a boolean mask and are handled during fitting. Compatible with
    scikit-learn's transformer interface.

    Parameters
    ----------
    n_components : int, default=2
        Number of components to extract.
    max_iterations : int, default=100
        Maximum number of EM iterations.
    tol : float, default=1e-4
        Convergence tolerance for the EM algorithm.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Principal axes in feature space.
    explained_variance_ratio_ : ndarray of shape (n_components,)
        Fraction of variance explained by each component.
    mean_ : ndarray of shape (n_features,)
        Per-feature mean estimated from observed training values.
    noise_variance_ : float
        Estimated variance of the Gaussian noise.

    Examples
    --------
    >>> import numpy as np
    >>> from ppca import PPCA
    >>> X = np.random.randn(100, 10)
    >>> X_transformed = PPCA(n_components=2).fit_transform(X)

    """

    def __init__(self, n_components=2, max_iterations=100, tol=1e-4):
        self.n_components = n_components
        self.max_iterations = max_iterations
        self.tol = tol

    def fit(self, X, y=None, missing_mask=None):
        """Fit the model to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : None
            Ignored.
        missing_mask : array-like of shape (n_samples, n_features), optional
            Boolean mask where True indicates a missing value. Defaults to
            no missing values.

        Returns
        -------
        self : PPCA

        """
        X = check_array(X, accept_sparse=False, ensure_2d=True, dtype=np.float64, ensure_all_finite="allow-nan")
        n_samples, n_features = X.shape

        if self.n_components > n_features:
            raise ValueError(f"n_components ({self.n_components}) must be <= n_features ({n_features})")

        if self.n_components > n_samples:
            warnings.warn(
                f"n_components ({self.n_components}) > n_samples ({n_samples}). This may lead to unstable results.",
                UserWarning,
                stacklevel=2,
            )

        if missing_mask is None:
            missing_mask = np.zeros_like(X, dtype=bool)
        else:
            missing_mask = check_array(missing_mask, accept_sparse=False, ensure_2d=True, dtype=bool)
            if missing_mask.shape != X.shape:
                raise ValueError(f"missing_mask shape {missing_mask.shape} does not match X shape {X.shape}")

        self._rust_model = PPCARust(n_components=self.n_components, max_iterations=self.max_iterations, tol=self.tol)
        self._rust_model.fit(X, missing_mask)

        self.mean_ = np.nanmean(np.where(missing_mask, np.nan, X), axis=0)
        self.explained_variance_ratio_ = np.array(self._rust_model.explained_variance_ratio())
        self.noise_variance_ = self._rust_model.noise_variance()
        self.n_features_in_ = n_features
        self.n_samples_seen_ = n_samples

        return self

    def transform(self, X):
        """Reduce X to latent components.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to transform.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)

        """
        check_is_fitted(self, ["_rust_model", "n_features_in_"])
        X = check_array(X, accept_sparse=False, ensure_2d=True, dtype=np.float64, ensure_all_finite="allow-nan")

        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features but model was fitted with {self.n_features_in_} features")

        X = np.where(np.isnan(X), self.mean_, X)
        return self._rust_model.transform(X)

    def fit_transform(self, X, y=None, missing_mask=None):
        """Fit and return the transformed data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : None
            Ignored.
        missing_mask : array-like of shape (n_samples, n_features), optional
            Boolean mask where True indicates a missing value.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)

        """
        return self.fit(X, y, missing_mask).transform(X)

    def inverse_transform(self, X):
        """Map latent components back to feature space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            Data in latent space.

        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)

        """
        check_is_fitted(self, ["_rust_model"])
        X = check_array(X, accept_sparse=False, ensure_2d=True, dtype=np.float64, ensure_all_finite="allow-nan")

        if X.shape[1] != self.n_components:
            raise ValueError(f"X has {X.shape[1]} components but model has {self.n_components} components")

        return self._rust_model.inverse_transform(X)

    def reconstruction_error(self, X, missing_mask=None):
        """Mean squared reconstruction error, excluding missing values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        missing_mask : array-like of shape (n_samples, n_features), optional
            Boolean mask where True indicates a missing value.

        Returns
        -------
        error : float

        """
        check_is_fitted(self, ["_rust_model"])
        X = check_array(X, accept_sparse=False, ensure_2d=True, dtype=np.float64, ensure_all_finite="allow-nan")

        if missing_mask is None:
            missing_mask = np.zeros_like(X, dtype=bool)
        else:
            missing_mask = check_array(missing_mask, accept_sparse=False, ensure_2d=True, dtype=bool)

        return self._rust_model.reconstruction_error(X, missing_mask)
