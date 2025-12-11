"""Comprehensive tests for PPCA implementation."""

import os
import sys

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ppca import PPCA


class TestPPCABasics:
    """Test basic PPCA functionality."""

    @pytest.fixture
    def simple_data(self):
        """Generate simple test data."""
        np.random.seed(42)
        return np.random.randn(50, 10)

    @pytest.fixture
    def iris_data(self):
        """Load iris dataset."""
        iris = load_iris()
        return iris.data

    def test_initialization(self):
        """Test PPCA initialization."""
        ppca = PPCA(n_components=2)
        assert ppca.n_components == 2
        assert ppca.max_iterations == 100
        assert ppca.tol == 1e-4

    def test_fit_simple(self, simple_data):
        """Test fitting on simple data."""
        ppca = PPCA(n_components=2)
        ppca.fit(simple_data)
        assert hasattr(ppca, "n_features_in_")
        assert ppca.n_features_in_ == 10

    def test_transform_after_fit(self, simple_data):
        """Test that transform works after fit."""
        ppca = PPCA(n_components=2)
        ppca.fit(simple_data)
        X_transformed = ppca.transform(simple_data)
        assert X_transformed.shape == (50, 2)

    def test_fit_transform(self, simple_data):
        """Test fit_transform method."""
        ppca = PPCA(n_components=2)
        X_transformed = ppca.fit_transform(simple_data)
        assert X_transformed.shape == (50, 2)

    def test_inverse_transform(self, simple_data):
        """Test reconstruction from latent space."""
        ppca = PPCA(n_components=2)
        X_transformed = ppca.fit_transform(simple_data)
        X_reconstructed = ppca.inverse_transform(X_transformed)
        assert X_reconstructed.shape == simple_data.shape

    def test_reconstruction_error(self, simple_data):
        """Test reconstruction error computation."""
        ppca = PPCA(n_components=2)
        ppca.fit(simple_data)
        error = ppca.reconstruction_error(simple_data)
        assert isinstance(error, float | np.floating)
        assert error >= 0


class TestPPCAMissingValues:
    """Test PPCA with missing values."""

    @pytest.fixture
    def data_with_missing(self):
        """Generate data and missing value mask."""
        np.random.seed(42)
        X = np.random.randn(30, 8)
        mask = np.random.rand(30, 8) < 0.2  # 20% missing
        return X, mask

    def test_fit_with_missing_mask(self, data_with_missing):
        """Test fitting with explicit missing value mask."""
        X, mask = data_with_missing
        ppca = PPCA(n_components=2)
        ppca.fit(X, missing_mask=mask)
        assert hasattr(ppca, "n_features_in_")

    def test_transform_with_fitted_model(self, data_with_missing):
        """Test transform on data with missing values."""
        X, mask = data_with_missing
        ppca = PPCA(n_components=2)
        ppca.fit(X, missing_mask=mask)
        X_transformed = ppca.transform(X)
        assert X_transformed.shape == (30, 2)

    def test_fit_transform_with_missing(self, data_with_missing):
        """Test fit_transform with missing values."""
        X, mask = data_with_missing
        ppca = PPCA(n_components=2)
        X_transformed = ppca.fit_transform(X, missing_mask=mask)
        assert X_transformed.shape == (30, 2)

    def test_reconstruction_error_with_missing(self, data_with_missing):
        """Test reconstruction error with missing values."""
        X, mask = data_with_missing
        ppca = PPCA(n_components=2)
        ppca.fit(X, missing_mask=mask)
        error = ppca.reconstruction_error(X, missing_mask=mask)
        assert isinstance(error, float | np.floating)
        assert error >= 0


class TestPPCAComparison:
    """Compare PPCA to sklearn PCA."""

    @pytest.fixture
    def comparison_data(self):
        """Generate data for comparison with sklearn."""
        np.random.seed(42)
        # Use iris for realistic data
        iris = load_iris()
        return iris.data

    def test_transformed_data_shape_matches(self, comparison_data):
        """Test that transformed data has expected shape."""
        n_components = 2
        ppca = PPCA(n_components=n_components)
        X_ppca = ppca.fit_transform(comparison_data)

        sklearn_pca = PCA(n_components=n_components)
        X_sklearn = sklearn_pca.fit_transform(comparison_data)

        assert X_ppca.shape == X_sklearn.shape

    def test_reconstruction_error_reasonable(self, comparison_data):
        """Test that reconstruction error is reasonable."""
        ppca = PPCA(n_components=2)
        ppca.fit(comparison_data)
        error = ppca.reconstruction_error(comparison_data)

        # Error should be less than variance of original data
        original_var = np.var(comparison_data)
        assert error < original_var

    def test_reconstruction_decreases_with_components(self, comparison_data):
        """Test that error decreases with more components."""
        errors = []
        for n_comp in [1, 2, 3, 4]:
            ppca = PPCA(n_components=n_comp)
            ppca.fit(comparison_data)
            error = ppca.reconstruction_error(comparison_data)
            errors.append(error)

        # Errors should generally decrease (within tolerance for numerical precision)
        assert errors[-1] <= errors[0] + 1e-6

    def test_explained_variance_ratio_shape(self, comparison_data):
        """Test explained variance ratio attribute."""
        ppca = PPCA(n_components=2)
        ppca.fit(comparison_data)
        assert hasattr(ppca, "explained_variance_ratio_")
        assert ppca.explained_variance_ratio_.shape == (2,)


class TestPPCAEdgeCases:
    """Test edge cases and error handling."""

    def test_n_components_greater_than_features_raises(self):
        """Test that n_components > n_features raises error."""
        X = np.random.randn(50, 5)
        ppca = PPCA(n_components=10)
        with pytest.raises(ValueError):
            ppca.fit(X)

    def test_transform_before_fit_raises(self):
        """Test that transform before fit raises error."""
        X = np.random.randn(50, 5)
        ppca = PPCA(n_components=2)
        with pytest.raises(NotFittedError):
            ppca.transform(X)

    def test_wrong_feature_dimension_raises(self):
        """Test that wrong feature dimension raises error."""
        X_train = np.random.randn(50, 5)
        X_test = np.random.randn(30, 3)
        ppca = PPCA(n_components=2)
        ppca.fit(X_train)
        with pytest.raises(ValueError):
            ppca.transform(X_test)

    def test_missing_mask_shape_mismatch_raises(self):
        """Test that mismatched mask shape raises error."""
        X = np.random.randn(50, 5)
        mask = np.zeros((30, 5), dtype=bool)
        ppca = PPCA(n_components=2)
        with pytest.raises(ValueError):
            ppca.fit(X, missing_mask=mask)

    def test_scikit_learn_params_interface(self):
        """Test get_params and set_params."""
        ppca = PPCA(n_components=2, max_iterations=50)
        params = ppca.get_params()
        assert params["n_components"] == 2
        assert params["max_iterations"] == 50

        ppca.set_params(n_components=3, max_iterations=200)
        assert ppca.n_components == 3
        assert ppca.max_iterations == 200


class TestPPCANumericalStability:
    """Test numerical stability of PPCA."""

    def test_handles_very_small_values(self):
        """Test that PPCA handles very small values."""
        np.random.seed(42)
        X = np.random.randn(50, 5) * 1e-10
        ppca = PPCA(n_components=2)
        ppca.fit(X)
        X_transformed = ppca.transform(X)
        assert not np.any(np.isnan(X_transformed))
        assert not np.any(np.isinf(X_transformed))

    def test_handles_very_large_values(self):
        """Test that PPCA handles very large values."""
        np.random.seed(42)
        X = np.random.randn(50, 5) * 1e10
        ppca = PPCA(n_components=2)
        ppca.fit(X)
        X_transformed = ppca.transform(X)
        assert not np.any(np.isnan(X_transformed))
        assert not np.any(np.isinf(X_transformed))

    def test_handles_constant_features(self):
        """Test that PPCA handles constant features gracefully."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        X[:, 0] = 1.0  # Make first column constant
        ppca = PPCA(n_components=2)
        # Should not raise, though it may warn
        ppca.fit(X)
        X_transformed = ppca.transform(X)
        assert X_transformed.shape == (50, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
