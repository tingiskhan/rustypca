import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError

from ppca import PPCA


def _make_low_rank(n=100, p=10, d=3, noise=0.1, seed=0):
    rng = np.random.RandomState(seed)
    Z = rng.randn(n, d)
    W = rng.randn(p, d)
    return Z @ W.T + noise * rng.randn(n, p)


def _add_missing(X, frac=0.1, seed=1):
    rng = np.random.RandomState(seed)
    mask = rng.rand(*X.shape) < frac
    X_masked = X.copy()
    X_masked[mask] = 0.0  # value doesn't matter, mask says missing
    return X_masked, mask


class TestBasics:
    def test_fit_returns_self(self):
        X = _make_low_rank()
        model = PPCA(n_components=2)
        assert model.fit(X) is model

    def test_output_shapes(self):
        X = _make_low_rank(n=50, p=8)
        model = PPCA(n_components=3).fit(X)
        assert model.components_.shape == (3, 8)
        assert model.explained_variance_ratio_.shape == (3,)
        assert model.mean_.shape == (8,)
        assert model.noise_variances_.shape == (8,)
        assert isinstance(model.noise_variance_, float)
        assert isinstance(model.n_iter_, int)
        assert model.log_likelihoods_.shape == (model.n_iter_,)
        assert model.n_features_in_ == 8
        assert model.n_samples_seen_ == 50

    def test_transform_shape(self):
        X = _make_low_rank(n=50, p=8)
        model = PPCA(n_components=3).fit(X)
        Y = model.transform(X)
        assert Y.shape == (50, 3)

    def test_fit_transform(self):
        X = _make_low_rank()
        model = PPCA(n_components=2)
        Y = model.fit_transform(X)
        assert Y.shape == (100, 2)

    def test_inverse_transform(self):
        X = _make_low_rank(n=50, p=8)
        model = PPCA(n_components=3).fit(X)
        Y = model.transform(X)
        X_hat = model.inverse_transform(Y)
        assert X_hat.shape == X.shape

    def test_reconstruction_error(self):
        X = _make_low_rank()
        model = PPCA(n_components=3).fit(X)
        err = model.reconstruction_error(X)
        assert err >= 0.0
        assert np.isfinite(err)


class TestMissingValues:
    def test_fit_with_mask(self):
        X = _make_low_rank()
        X_m, mask = _add_missing(X)
        model = PPCA(n_components=2).fit(X_m, missing_mask=mask)
        assert model.components_.shape == (2, 10)

    def test_transform_with_nan(self):
        X = _make_low_rank()
        model = PPCA(n_components=2).fit(X)
        X_nan = X.copy()
        X_nan[0, 0] = np.nan
        Y = model.transform(X_nan)
        assert Y.shape == (100, 2)
        assert np.all(np.isfinite(Y))

    def test_reconstruction_error_with_mask(self):
        X = _make_low_rank()
        X_m, mask = _add_missing(X)
        model = PPCA(n_components=2).fit(X_m, missing_mask=mask)
        err = model.reconstruction_error(X_m, missing_mask=mask)
        assert err >= 0.0
        assert np.isfinite(err)


class TestVsSklearn:
    def test_similar_reconstruction(self):
        X = _make_low_rank(n=200, p=10, d=3, noise=0.05)
        d = 3
        ppca = PPCA(n_components=d).fit(X)
        pca = PCA(n_components=d).fit(X)

        ppca_err = ppca.reconstruction_error(X)
        X_pca = pca.inverse_transform(pca.transform(X))
        pca_err = np.mean((X - X_pca) ** 2)

        # PPCA should be comparable to PCA on complete data
        assert ppca_err < pca_err * 3.0

    def test_evr_sums_below_one(self):
        X = _make_low_rank()
        model = PPCA(n_components=3).fit(X)
        assert model.explained_variance_ratio_.sum() <= 1.0 + 1e-6
        assert np.all(model.explained_variance_ratio_ >= 0.0)


class TestNoiseType:
    def test_isotropic_uniform_variances(self):
        X = _make_low_rank()
        model = PPCA(n_components=2, noise_type="isotropic").fit(X)
        assert np.allclose(model.noise_variances_, model.noise_variances_[0])

    def test_diagonal_nonuniform(self):
        rng = np.random.RandomState(42)
        # Heteroskedastic noise: different variance per feature
        X = rng.randn(200, 5)
        X[:, 0] *= 10.0
        model = PPCA(n_components=2, noise_type="diagonal").fit(X)
        # At least not all equal
        assert not np.allclose(model.noise_variances_, model.noise_variances_[0], atol=0.01)

    def test_diagonal_with_missing(self):
        X = _make_low_rank()
        X_m, mask = _add_missing(X)
        model = PPCA(n_components=2, noise_type="diagonal").fit(X_m, missing_mask=mask)
        assert model.components_.shape == (2, 10)

    def test_invalid_noise_type(self):
        with pytest.raises(ValueError, match="noise_type"):
            PPCA(n_components=2, noise_type="invalid").fit(_make_low_rank())

    def test_transform_both_types(self):
        X = _make_low_rank()
        for nt in ("isotropic", "diagonal"):
            model = PPCA(n_components=2, noise_type=nt).fit(X)
            Y = model.transform(X)
            assert Y.shape == (100, 2)


class TestL2Penalty:
    def test_shrinks_loadings(self):
        X = _make_low_rank()
        model0 = PPCA(n_components=3, l2_penalty=0.0).fit(X)
        model1 = PPCA(n_components=3, l2_penalty=1.0).fit(X)
        norm0 = np.linalg.norm(model0.components_)
        norm1 = np.linalg.norm(model1.components_)
        assert norm1 < norm0

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="l2_penalty"):
            PPCA(n_components=2, l2_penalty=-0.1).fit(_make_low_rank())

    def test_with_diagonal_noise(self):
        X = _make_low_rank()
        model = PPCA(n_components=2, noise_type="diagonal", l2_penalty=0.5).fit(X)
        assert model.components_.shape == (2, 10)


class TestConvergence:
    def test_ll_non_decreasing(self):
        X = _make_low_rank(n=100, p=10, d=3)
        model = PPCA(n_components=3, max_iterations=50, tol=1e-10, random_state=42).fit(X)
        ll = model.log_likelihoods_
        # Allow tiny numerical noise
        diffs = np.diff(ll)
        assert np.all(diffs >= -1e-4), f"LL decreased: {diffs[diffs < -1e-4]}"

    def test_converges_before_max_iter(self):
        X = _make_low_rank()
        model = PPCA(n_components=2, max_iterations=500, tol=1e-4).fit(X)
        assert model.n_iter_ < 500

    def test_n_iter_matches_ll_length(self):
        X = _make_low_rank()
        model = PPCA(n_components=2).fit(X)
        assert model.n_iter_ == len(model.log_likelihoods_)


class TestRandomState:
    def test_reproducibility(self):
        X = _make_low_rank()
        m1 = PPCA(n_components=2, random_state=123).fit(X)
        m2 = PPCA(n_components=2, random_state=123).fit(X)
        np.testing.assert_allclose(m1.components_, m2.components_)

    def test_different_seeds_differ(self):
        X = _make_low_rank()
        m1 = PPCA(n_components=2, random_state=1).fit(X)
        m2 = PPCA(n_components=2, random_state=2).fit(X)
        # Could converge to same solution, but very unlikely with different seeds
        assert not np.allclose(m1.components_, m2.components_, atol=1e-3)


class TestEdgeCases:
    def test_n_components_gt_features_raises(self):
        X = np.random.randn(20, 3)
        with pytest.raises(ValueError, match="n_components"):
            PPCA(n_components=5).fit(X)

    def test_n_components_gt_samples_warns(self):
        X = np.random.randn(3, 20)
        with pytest.warns(UserWarning, match="n_components"):
            PPCA(n_components=5).fit(X)

    def test_not_fitted_transform(self):
        model = PPCA(n_components=2)
        with pytest.raises(NotFittedError):
            model.transform(np.random.randn(10, 5))

    def test_wrong_features_transform(self):
        X = _make_low_rank(p=10)
        model = PPCA(n_components=2).fit(X)
        with pytest.raises(ValueError, match="features"):
            model.transform(np.random.randn(5, 7))

    def test_wrong_components_inverse_transform(self):
        X = _make_low_rank()
        model = PPCA(n_components=2).fit(X)
        with pytest.raises(ValueError, match="components"):
            model.inverse_transform(np.random.randn(5, 4))

    def test_mask_shape_mismatch(self):
        X = _make_low_rank()
        mask = np.zeros((5, 3), dtype=bool)
        with pytest.raises(ValueError, match="shape"):
            PPCA(n_components=2).fit(X, missing_mask=mask)

    def test_sklearn_get_set_params(self):
        model = PPCA(n_components=3, noise_type="diagonal", l2_penalty=0.5)
        params = model.get_params()
        assert params["n_components"] == 3
        assert params["noise_type"] == "diagonal"
        assert params["l2_penalty"] == 0.5

        model.set_params(n_components=5)
        assert model.n_components == 5


# ── Numerical stability ─────────────────────────────────────────────────────


class TestNumericalStability:
    def test_small_values(self):
        X = _make_low_rank() * 1e-10
        model = PPCA(n_components=2).fit(X)
        Y = model.transform(X)
        assert np.all(np.isfinite(Y))

    def test_large_values(self):
        X = _make_low_rank() * 1e10
        model = PPCA(n_components=2).fit(X)
        Y = model.transform(X)
        assert np.all(np.isfinite(Y))

    def test_constant_feature(self):
        X = _make_low_rank()
        X[:, 0] = 5.0
        model = PPCA(n_components=2).fit(X)
        assert np.all(np.isfinite(model.components_))
