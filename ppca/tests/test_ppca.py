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
            assert earlier >= later - 1e-4

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


class TestLoadingSigns:
    """Tests for the optional loading_signs identification constraint.

    Most sign-sensitive tests use structured data (strong first PC) so that
    EM reliably converges to the same subspace regardless of random init.
    """

    @pytest.fixture
    def structured_data(self):
        """Data with a dominant first PC: X ≈ z @ ones^T + noise."""
        np.random.seed(0)
        z = np.random.randn(80, 1)
        return z @ np.ones((1, 6)) + np.random.randn(80, 6) * 0.1

    def test_none_by_default(self):
        assert PPCA(n_components=2).loading_signs is None

    def test_positive_constraint_consistent_across_fits(self, structured_data):
        scores_a = PPCA(n_components=2, loading_signs=[1, 0]).fit_transform(structured_data)
        scores_b = PPCA(n_components=2, loading_signs=[1, 0]).fit_transform(structured_data)
        corr = np.corrcoef(scores_a[:, 0], scores_b[:, 0])[0, 1]
        assert corr > 0.5

    def test_negative_constraint_flips_sign(self, structured_data):
        pos = PPCA(n_components=2, loading_signs=[1, 0]).fit_transform(structured_data)
        neg = PPCA(n_components=2, loading_signs=[-1, 0]).fit_transform(structured_data)
        corr = np.corrcoef(pos[:, 0], neg[:, 0])[0, 1]
        assert corr < -0.5

    def test_reconstruction_unchanged_by_sign_constraint(self, structured_data):
        ppca_free = PPCA(n_components=2)
        ppca_free.fit(structured_data)
        err_free = ppca_free.reconstruction_error(structured_data)

        ppca_sign = PPCA(n_components=2, loading_signs=[1, 0])
        ppca_sign.fit(structured_data)
        err_sign = ppca_sign.reconstruction_error(structured_data)

        np.testing.assert_allclose(err_free, err_sign, rtol=0.05)

    def test_wrong_length_raises(self):
        ppca = PPCA(n_components=3, loading_signs=[1, 0])
        with pytest.raises(ValueError, match="loading_signs length"):
            ppca.fit(np.random.randn(20, 6))

    def test_invalid_value_raises(self):
        ppca = PPCA(n_components=2, loading_signs=[1, 2])
        with pytest.raises(ValueError, match="must be -1, 0, or \\+1"):
            ppca.fit(np.random.randn(20, 6))

    def test_all_zeros_produces_valid_output(self):
        np.random.seed(42)
        X = np.random.randn(50, 6)
        ppca = PPCA(n_components=2, loading_signs=[0, 0])
        scores = ppca.fit_transform(X)
        assert scores.shape == (50, 2)
        assert not np.any(np.isnan(scores))

    def test_with_missing_values(self):
        np.random.seed(42)
        X = np.random.randn(30, 8)
        mask = np.random.rand(30, 8) < 0.2
        ppca = PPCA(n_components=2, loading_signs=[1, 0])
        ppca.fit(X, missing_mask=mask)
        scores = ppca.transform(X)
        assert scores.shape == (30, 2)

    def test_sklearn_get_params_includes_loading_signs(self):
        ppca = PPCA(n_components=2, loading_signs=[1, -1])
        params = ppca.get_params()
        assert params["loading_signs"] == [1, -1]


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
