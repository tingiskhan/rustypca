"""Scikit-learn compatible PPCA implementation."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array
import warnings

try:
    from .ppca_rs import PPCARust
except ImportError:
    raise ImportError(
        "ppca_rs module not found. Please build the Rust extension with "
        "'pip install -e .' or 'maturin develop'"
    )


class PPCA(BaseEstimator, TransformerMixin):
    """Probabilistic Principal Component Analysis with missing value support.
    
    Implements PPCA using the EM algorithm, supporting missing values during
    the fitting process. Compatible with scikit-learn's transformer interface.
    
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
        Percentage of variance explained by each component.
    
    mean_ : ndarray of shape (n_features,)
        Per-feature empirical mean, computed from the training set.
    
    noise_variance_ : float
        The estimated variance of the Gaussian noise.
    
    Examples
    --------
    >>> import numpy as np
    >>> from ppca import PPCA
    >>> X = np.random.randn(100, 10)
    >>> ppca = PPCA(n_components=2)
    >>> X_transformed = ppca.fit_transform(X)
    """

    def __init__(self, n_components=2, max_iterations=100, tol=1e-4):
        self.n_components = n_components
        self.max_iterations = max_iterations
        self.tol = tol
        self._rust_model = None

    def fit(self, X, y=None, missing_mask=None):
        """Fit the PPCA model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        
        y : None
            Ignored. Present for API consistency by convention.
        
        missing_mask : array-like of shape (n_samples, n_features), optional
            Boolean mask indicating missing values (True = missing).
            If None, assumes no missing values.
        
        Returns
        -------
        self : PPCA
            Returns self.
        """
        X = check_array(X, accept_sparse=False, ensure_2d=True, dtype=np.float64)
        
        n_samples, n_features = X.shape
        
        if self.n_components > n_features:
            raise ValueError(
                f"n_components ({self.n_components}) must be <= "
                f"n_features ({n_features})"
            )
        
        if self.n_components > n_samples:
            warnings.warn(
                f"n_components ({self.n_components}) > n_samples ({n_samples}). "
                "This may lead to unstable results.",
                UserWarning
            )
        
        if missing_mask is None:
            missing_mask = np.zeros_like(X, dtype=bool)
        else:
            missing_mask = check_array(
                missing_mask, accept_sparse=False, ensure_2d=True, dtype=bool
            )
            if missing_mask.shape != X.shape:
                raise ValueError(
                    f"missing_mask shape {missing_mask.shape} does not match "
                    f"X shape {X.shape}"
                )
        
        # Create and fit the Rust model
        self._rust_model = PPCARust(
            n_components=self.n_components,
            max_iterations=self.max_iterations,
            tol=self.tol
        )
        self._rust_model.fit(X, missing_mask)
        
        # Store attributes for sklearn compatibility
        self.mean_ = np.mean(X[~missing_mask.reshape(-1)].reshape(-1, X.shape[1]), axis=0)
        
        # Get components from the model (loadings)
        X_transformed = self._rust_model.transform(X)
        
        # Approximate explained variance
        explained_var = np.var(X_transformed, axis=0)
        total_var = np.var(X, axis=0).sum()
        self.explained_variance_ratio_ = explained_var / (total_var + 1e-10)
        
        self.n_features_in_ = n_features
        self.n_samples_seen_ = n_samples
        
        return self

    def transform(self, X):
        """Apply dimensionality reduction to X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data.
        
        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        check_is_fitted(self, ["_rust_model", "n_features_in_"])
        
        X = check_array(X, accept_sparse=False, ensure_2d=True, dtype=np.float64)
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features but model was fitted with "
                f"{self.n_features_in_} features"
            )
        
        return self._rust_model.transform(X)

    def fit_transform(self, X, y=None, missing_mask=None):
        """Fit to data, then transform it.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        
        y : None
            Ignored. Present for API consistency by convention.
        
        missing_mask : array-like of shape (n_samples, n_features), optional
            Boolean mask indicating missing values (True = missing).
        
        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        return self.fit(X, y, missing_mask).transform(X)

    def inverse_transform(self, X):
        """Transform data back to the original space.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            Data in the latent space.
        
        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            Reconstructed data.
        """
        check_is_fitted(self, ["_rust_model"])
        
        X = check_array(X, accept_sparse=False, ensure_2d=True, dtype=np.float64)
        
        if X.shape[1] != self.n_components:
            raise ValueError(
                f"X has {X.shape[1]} components but model has "
                f"{self.n_components} components"
            )
        
        return self._rust_model.inverse_transform(X)

    def reconstruction_error(self, X, missing_mask=None):
        """Compute the reconstruction error.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        
        missing_mask : array-like of shape (n_samples, n_features), optional
            Boolean mask indicating missing values (True = missing).
        
        Returns
        -------
        error : float
            Mean squared reconstruction error (excluding missing values).
        """
        check_is_fitted(self, ["_rust_model"])
        
        X = check_array(X, accept_sparse=False, ensure_2d=True, dtype=np.float64)
        
        if missing_mask is None:
            missing_mask = np.zeros_like(X, dtype=bool)
        else:
            missing_mask = check_array(
                missing_mask, accept_sparse=False, ensure_2d=True, dtype=bool
            )
        
        return self._rust_model.reconstruction_error(X, missing_mask)

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            "n_components": self.n_components,
            "max_iterations": self.max_iterations,
            "tol": self.tol,
        }

    def set_params(self, **params):
        """Set the parameters of this estimator."""
        for key, value in params.items():
            if key not in self.get_params():
                raise ValueError(f"Invalid parameter: {key}")
            setattr(self, key, value)
        return self
