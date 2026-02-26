import warnings

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError

from ppca import PPCA


class TestPPCABasics:
    @pytest.fixture
    def simple_data(self):
        np.random.seed(42)
        return np.random.randn(50, 10)

    def test_initialization(self):
        ppca = PPCA(n_components=2)
        assert ppca.n_components == 2
        assert ppca.max_iterations == 100
        assert ppca.tol == 1e-4

    def test_fit_simple(self, simple_data):
        ppca = PPCA(n_components=2)
        ppca.fit(simple_data)
        assert ppca.n_features_in_ == 10

    def test_transform_after_fit(self, simple_data):
        ppca = PPCA(n_components=2)
        X_new = ppca.fit(simple_data).transform(simple_data)
        assert X_new.shape == (50, 2)

    def test_fit_transform(self, simple_data):
        assert PPCA(n_components=2).fit_transform(simple_data).shape == (50, 2)

    def test_inverse_transform(self, simple_data):
        ppca = PPCA(n_components=2)
        X_back = ppca.fit_transform(simple_data)
        assert ppca.inverse_transform(X_back).shape == simple_data.shape

    def test_reconstruction_error(self, simple_data):
        ppca = PPCA(n_components=2)
        ppca.fit(simple_data)
        error = ppca.reconstruction_error(simple_data)
        assert isinstance(error, float | np.floating)
        assert error >= 0


class TestPPCAMissingValues:
    @pytest.fixture
    def data_with_missing(self):
        np.random.seed(42)
        X = np.random.randn(30, 8)
        mask = np.random.rand(30, 8) < 0.2
        return X, mask

    def test_fit_with_missing_mask(self, data_with_missing):
        X, mask = data_with_missing
        ppca = PPCA(n_components=2)
        ppca.fit(X, missing_mask=mask)
        assert hasattr(ppca, "n_features_in_")

    def test_transform_with_fitted_model(self, data_with_missing):
        X, mask = data_with_missing
        ppca = PPCA(n_components=2)
        ppca.fit(X, missing_mask=mask)
        assert ppca.transform(X).shape == (30, 2)

    def test_fit_transform_with_missing(self, data_with_missing):
        X, mask = data_with_missing
        assert PPCA(n_components=2).fit_transform(X, missing_mask=mask).shape == (30, 2)

    def test_reconstruction_error_with_missing(self, data_with_missing):
        X, mask = data_with_missing
        ppca = PPCA(n_components=2)
        ppca.fit(X, missing_mask=mask)
        error = ppca.reconstruction_error(X, missing_mask=mask)
        assert isinstance(error, float | np.floating)
        assert error >= 0


class TestPPCAComparison:
    @pytest.fixture
    def iris(self):
        return load_iris().data

    def test_transformed_shape_matches_sklearn(self, iris):
        ppca = PPCA(n_components=2)
        sklearn_pca = PCA(n_components=2)
        assert ppca.fit_transform(iris).shape == sklearn_pca.fit_transform(iris).shape

    def test_reconstruction_error_reasonable(self, iris):
        ppca = PPCA(n_components=2)
        ppca.fit(iris)
        assert ppca.reconstruction_error(iris) < np.var(iris)

    def test_reconstruction_decreases_with_components(self, iris):
        errors = []
        for n in [1, 4]:
            ppca = PPCA(n_components=n)
            ppca.fit(iris)
            errors.append(ppca.reconstruction_error(iris))

        for earlier, later in zip(errors[:-1], errors[1:], strict=False):
            assert earlier <= later + 1e-4

    def test_explained_variance_ratio_shape(self, iris):
        ppca = PPCA(n_components=2)
        ppca.fit(iris)
        assert ppca.explained_variance_ratio_.shape == (2,)


class TestPPCAEdgeCases:
    def test_n_components_greater_than_features_raises(self):
        ppca = PPCA(n_components=10)
        with pytest.raises(ValueError):
            ppca.fit(np.random.randn(50, 5))

    def test_n_components_greater_than_samples_warns(self):
        X = np.random.randn(3, 10)
        ppca = PPCA(n_components=5)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ppca.fit(X)
        assert any(issubclass(warning.category, UserWarning) for warning in w)

    def test_transform_before_fit_raises(self):
        ppca = PPCA(n_components=2)
        with pytest.raises(NotFittedError):
            ppca.transform(np.random.randn(50, 5))

    def test_wrong_feature_dimension_raises(self):
        ppca = PPCA(n_components=2)
        ppca.fit(np.random.randn(50, 5))
        with pytest.raises(ValueError):
            ppca.transform(np.random.randn(30, 3))

    def test_inverse_transform_wrong_components_raises(self):
        ppca = PPCA(n_components=2)
        ppca.fit(np.random.randn(50, 5))
        with pytest.raises(ValueError):
            ppca.inverse_transform(np.random.randn(10, 4))

    def test_missing_mask_shape_mismatch_raises(self):
        ppca = PPCA(n_components=2)
        with pytest.raises(ValueError):
            ppca.fit(np.random.randn(50, 5), missing_mask=np.zeros((30, 5), dtype=bool))

    def test_sklearn_params_interface(self):
        ppca = PPCA(n_components=2, max_iterations=50)
        params = ppca.get_params()
        assert params["n_components"] == 2
        assert params["max_iterations"] == 50

        ppca.set_params(n_components=3, max_iterations=200)
        assert ppca.n_components == 3
        assert ppca.max_iterations == 200


class TestPPCANumericalStability:
    def test_handles_very_small_values(self):
        np.random.seed(42)
        X = np.random.randn(50, 5) * 1e-10
        ppca = PPCA(n_components=2)
        ppca.fit(X)
        result = ppca.transform(X)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_handles_very_large_values(self):
        np.random.seed(42)
        X = np.random.randn(50, 5) * 1e10
        ppca = PPCA(n_components=2)
        ppca.fit(X)
        result = ppca.transform(X)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_handles_constant_features(self):
        np.random.seed(42)
        X = np.random.randn(50, 5)
        X[:, 0] = 1.0
        ppca = PPCA(n_components=2)
        ppca.fit(X)
        assert ppca.transform(X).shape == (50, 2)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
