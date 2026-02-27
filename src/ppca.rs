//! Probabilistic PCA implementation using the EM algorithm
//!
//! Reference: Tipping & Bishop, "Probabilistic Principal Component Analysis", JMLR, 1999

use nalgebra::{DMatrix, DVector, SVD};
use rand::Rng;
use rand::SeedableRng;
use crate::errors::{PPCAError, Result};
use std::f64;

/// Configuration for PPCA training
#[derive(Debug, Clone)]
pub struct PPCAConfig {
    pub n_components: usize,
    pub max_iterations: usize,
    pub tol: f64,
    pub random_state: Option<u64>,
    /// Per-component sign constraints for the loadings.
    /// +1 = column mean must be positive, -1 = negative, 0 = unconstrained.
    pub loading_signs: Option<Vec<i8>>,
}

impl Default for PPCAConfig {
    fn default() -> Self {
        PPCAConfig {
            n_components: 2,
            max_iterations: 100,
            tol: 1e-4,
            random_state: None,
            loading_signs: None,
        }
    }
}

/// Result of PPCA fitting
#[derive(Debug, Clone)]
pub struct PPCAResult {
    /// Loading matrix (n_features x n_components)
    pub loadings: DMatrix<f64>,
    /// Noise variance
    pub sigma2: f64,
    /// Mean of the training data
    pub mean: DVector<f64>,
    /// Explained variance ratio
    pub explained_variance_ratio: DVector<f64>,
}

/// Probabilistic Principal Component Analysis model
pub struct PPCA {
    config: PPCAConfig,
    result: Option<PPCAResult>,
}

impl PPCA {
    /// Create a new PPCA model with default configuration
    pub fn new(n_components: usize) -> Self {
        let mut config = PPCAConfig::default();
        config.n_components = n_components;
        PPCA {
            config,
            result: None,
        }
    }

    /// Create a new PPCA model with custom configuration
    pub fn with_config(config: PPCAConfig) -> Self {
        PPCA {
            config,
            result: None,
        }
    }

    /// Fit the PPCA model to data with missing values
    ///
    /// # Arguments
    /// * `X` - Data matrix (n_samples x n_features)
    /// * `mask` - Boolean mask where true indicates missing values
    pub fn fit(&mut self, X: &DMatrix<f64>, mask: &DMatrix<bool>) -> Result<()> {
        let (n_samples, n_features) = X.shape();

        if n_samples == 0 || n_features == 0 {
            return Err(PPCAError::NoDimensionality);
        }

        if self.config.n_components > n_features {
            return Err(PPCAError::InvalidComponents {
                n_components: self.config.n_components,
                n_features,
            });
        }

        // Center the data, setting missing entries to 0.0 so they don't
        // contribute to matrix products in the M-step.
        let mean = self.compute_mean(X, mask)?;
        let mut X_centered = X.clone();
        for i in 0..n_samples {
            for j in 0..n_features {
                if mask[(i, j)] {
                    X_centered[(i, j)] = 0.0;
                } else {
                    X_centered[(i, j)] = X[(i, j)] - mean[j];
                }
            }
        }

        // Precompute per-sample observed indices — the mask is constant across iterations.
        let obs_by_sample: Vec<Vec<usize>> = (0..n_samples)
            .map(|i| (0..n_features).filter(|&j| !mask[(i, j)]).collect())
            .collect();

        // Initialise W and sigma2.
        // Default: PCA on mean-imputed data — places W near the optimum so EM
        // converges in tens of iterations rather than thousands.
        // random_state=Some(seed): random init (for explicit reproducibility testing).
        let (mut loadings, mut sigma2) = match self.config.random_state {
            Some(seed) => {
                let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
                let w = DMatrix::from_fn(n_features, self.config.n_components, |_, _| {
                    rng.gen::<f64>()
                });
                (w, 1.0_f64)
            }
            None => self.pca_init(&X_centered)?,
        };

        // Run EM algorithm
        for _iteration in 0..self.config.max_iterations {
            let old_loadings = loadings.clone();

            // E-step: per-sample posterior mean (mu_i) and covariance factor (M_i^{-1})
            let (expectations, posterior_covs) =
                self.e_step(&X_centered, &loadings, sigma2, &obs_by_sample)?;

            // M-step: update W and sigma2 using per-sample posterior quantities
            let (new_loadings, new_sigma2) =
                self.m_step(&X_centered, &expectations, &posterior_covs, sigma2, &obs_by_sample)?;

            loadings = new_loadings;
            sigma2 = new_sigma2;

            // Convergence: relative change in loadings (Frobenius norm)
            let delta = (&loadings - &old_loadings).norm();
            let scale = old_loadings.norm().max(1e-10);
            if delta / scale < self.config.tol {
                break;
            }
        }

        // Apply sign identification constraints to resolved rotational ambiguity
        if let Some(ref signs) = self.config.loading_signs {
            for (k, &sign) in signs.iter().enumerate() {
                if sign != 0 && k < self.config.n_components {
                    let col_sum: f64 = (0..n_features).map(|j| loadings[(j, k)]).sum();
                    let needs_flip = (sign > 0 && col_sum < 0.0) || (sign < 0 && col_sum > 0.0);
                    if needs_flip {
                        for j in 0..n_features {
                            loadings[(j, k)] *= -1.0;
                        }
                    }
                }
            }
        }

        // Compute explained variance
        let explained_variance = self.compute_explained_variance(&loadings, sigma2, n_features)?;

        self.result = Some(PPCAResult {
            loadings,
            sigma2,
            mean,
            explained_variance_ratio: explained_variance,
        });

        Ok(())
    }

    /// Transform data using the fitted model
    pub fn transform(&self, X: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        let result = self.result.as_ref().ok_or(PPCAError::NoDimensionality)?;

        let (n_samples, n_features) = X.shape();
        if n_features != result.mean.len() {
            return Err(PPCAError::InvalidDimensions {
                expected: result.mean.len(),
                got: n_features,
            });
        }

        let X_centered = {
            let mut centered = X.clone();
            for i in 0..n_samples {
                for j in 0..n_features {
                    let v = X[(i, j)];
                    centered[(i, j)] = if v.is_nan() { 0.0 } else { v - result.mean[j] };
                }
            }
            centered
        };

        // Project onto principal components: Y = X @ W @ (W^T @ W + sigma2 * I)^-1
        let WtW = result.loadings.transpose() * &result.loadings;
        let sigma2_I = DMatrix::from_diagonal(&DVector::from_element(
            self.config.n_components,
            result.sigma2,
        ));
        let M = WtW + sigma2_I;

        // Compute inverse (Cholesky decomposition)
        let M_inv = self.matrix_inverse(&M)?;

        let Y = &X_centered * &result.loadings * &M_inv;
        Ok(Y)
    }

    /// Reconstruct data from latent representation
    pub fn inverse_transform(&self, Y: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        let result = self.result.as_ref().ok_or(PPCAError::NoDimensionality)?;

        let (n_samples, n_components) = Y.shape();
        if n_components != self.config.n_components {
            return Err(PPCAError::InvalidDimensions {
                expected: self.config.n_components,
                got: n_components,
            });
        }

        // Reconstruct: X_reconstructed = Y @ W^T + mean
        let X_reconstructed = Y * result.loadings.transpose();
        let X_with_mean = &X_reconstructed + DMatrix::from_element(n_samples, result.mean.len(), 1.0)
            .component_mul(&DMatrix::from_fn(n_samples, result.mean.len(), |_, j| result.mean[j]));

        Ok(X_with_mean)
    }

    /// Compute reconstruction error for data with potential missing values
    pub fn reconstruction_error(
        &self,
        X: &DMatrix<f64>,
        mask: &DMatrix<bool>,
    ) -> Result<f64> {
        let Y = self.transform(X)?;
        let X_reconstructed = self.inverse_transform(&Y)?;

        let mut error = 0.0;
        let mut count = 0.0;

        for i in 0..X.nrows() {
            for j in 0..X.ncols() {
                if !mask[(i, j)] {
                    let diff = X[(i, j)] - X_reconstructed[(i, j)];
                    error += diff * diff;
                    count += 1.0;
                }
            }
        }

        Ok(if count > 0.0 { error / count } else { 0.0 })
    }

    // Private helper methods

    fn compute_mean(&self, X: &DMatrix<f64>, mask: &DMatrix<bool>) -> Result<DVector<f64>> {
        let n_features = X.ncols();
        let mut mean: DVector<f64> = DVector::zeros(n_features);
        let mut counts: DVector<f64> = DVector::zeros(n_features);

        for i in 0..X.nrows() {
            for j in 0..n_features {
                if !mask[(i, j)] {
                    mean[j] += X[(i, j)];
                    counts[j] += 1.0;
                }
            }
        }

        for j in 0..n_features {
            if counts[j] > 0.0 {
                mean[j] /= counts[j];
            }
        }

        Ok(mean)
    }

    /// PCA initialisation: compute W and sigma2 from the top-d SVD of the
    /// (mean-imputed) centred data.  Missing entries are already 0 in X_centered.
    fn pca_init(&self, X_centered: &DMatrix<f64>) -> Result<(DMatrix<f64>, f64)> {
        let (n_samples, n_features) = X_centered.shape();
        let d = self.config.n_components;

        // Full SVD of X_centered (n x p).  V_t rows are the principal directions,
        // singular values^2 / n are the sample covariance eigenvalues.
        let svd = SVD::new(X_centered.clone(), false, true);
        let sv = &svd.singular_values;
        let v_t = svd.v_t.as_ref().ok_or(PPCAError::MatrixError("SVD V^T unavailable".to_string()))?;

        // Initial sigma2: average eigenvalue of non-principal directions.
        let sigma2_init = if n_features > d {
            let tail: f64 = sv.iter().skip(d).map(|s| s * s / n_samples as f64).sum();
            (tail / (n_features - d) as f64).max(1e-6)
        } else {
            1e-3
        };

        // W_init[:, k] = v_k * sqrt(max(lambda_k - sigma2, eps))
        let mut w_init = DMatrix::zeros(n_features, d);
        for k in 0..d {
            let lambda_k = sv[k] * sv[k] / n_samples as f64;
            let scale = (lambda_k - sigma2_init).max(1e-6).sqrt();
            for j in 0..n_features {
                w_init[(j, k)] = v_t[(k, j)] * scale;
            }
        }

        Ok((w_init, sigma2_init))
    }

    fn e_step(
        &self,
        X_centered: &DMatrix<f64>,
        W: &DMatrix<f64>,
        sigma2: f64,
        obs_by_sample: &[Vec<usize>],
    ) -> Result<(Vec<DVector<f64>>, Vec<DMatrix<f64>>)> {
        let n_samples = X_centered.nrows();
        let d = self.config.n_components;
        let sigma2_I = DMatrix::from_diagonal(&DVector::from_element(d, sigma2));

        let mut expectations = Vec::with_capacity(n_samples);
        let mut posterior_covs = Vec::with_capacity(n_samples); // M_i^{-1} per sample

        for i in 0..n_samples {
            let obs_idx = &obs_by_sample[i];

            if obs_idx.is_empty() {
                // No observations: posterior equals prior  E[z]=0, Cov[z|x]=I
                expectations.push(DVector::zeros(d));
                posterior_covs.push(DMatrix::identity(d, d));
                continue;
            }

            let n_obs = obs_idx.len();
            let mut W_obs = DMatrix::zeros(n_obs, d);
            let mut x_obs = DVector::zeros(n_obs);
            for (k, &j) in obs_idx.iter().enumerate() {
                x_obs[k] = X_centered[(i, j)];
                for l in 0..d {
                    W_obs[(k, l)] = W[(j, l)];
                }
            }

            // M_i = W_obs^T W_obs + sigma2 * I  (d x d)
            let WtW = W_obs.transpose() * &W_obs;
            let M_i = WtW + &sigma2_I;
            let M_i_inv = M_i
                .try_inverse()
                .ok_or(PPCAError::MatrixError("Cannot invert M_i in E-step".to_string()))?;

            // E[z_i | x_i^obs] = M_i^{-1} W_obs^T x_obs
            let mu_i = &M_i_inv * W_obs.transpose() * &x_obs;

            expectations.push(mu_i);
            posterior_covs.push(M_i_inv);
        }

        Ok((expectations, posterior_covs))
    }

    fn m_step(
        &self,
        X_centered: &DMatrix<f64>,
        expectations: &[DVector<f64>],
        posterior_covs: &[DMatrix<f64>], // M_i^{-1} from E-step
        sigma2: f64,
        obs_by_sample: &[Vec<usize>],
    ) -> Result<(DMatrix<f64>, f64)> {
        let (n_samples, n_features) = X_centered.shape();
        let d = self.config.n_components;

        // Precompute E[z_i z_i^T | x_i^obs] = mu_i mu_i^T + sigma2 * M_i^{-1}
        // This is reused for every feature in the W update.
        let second_moments: Vec<DMatrix<f64>> = (0..n_samples)
            .map(|i| {
                let mu = &expectations[i];
                let m_inv = &posterior_covs[i];
                mu * mu.transpose() + sigma2 * m_inv
            })
            .collect();

        // Accumulate xz[j] = Σ_{i: j observed} x_ij mu_i
        //            zz[j] = Σ_{i: j observed} E[z_i z_i^T]
        // Outer loop over samples so each second_moments[i] is loaded once.
        let mut xz = vec![DVector::<f64>::zeros(d); n_features];
        let mut zz = vec![DMatrix::<f64>::zeros(d, d); n_features];

        for i in 0..n_samples {
            let mu_i = &expectations[i];
            let A_i = &second_moments[i];
            for &j in &obs_by_sample[i] {
                xz[j] += X_centered[(i, j)] * mu_i;
                zz[j] += A_i;
            }
        }

        // W update: W_j = zz[j]^{-1} xz[j]
        let mut W = DMatrix::zeros(n_features, d);
        for j in 0..n_features {
            let zz_inv = zz[j]
                .clone()
                .try_inverse()
                .ok_or(PPCAError::MatrixError("Cannot invert second-moment matrix for feature".to_string()))?;
            let w_j = zz_inv * &xz[j];
            for l in 0..d {
                W[(j, l)] = w_j[l];
            }
        }

        // sigma2 update (using NEW W, OLD M_i^{-1}):
        //   sigma2_new = (1/N_obs) * Σ_{i,j∈O_i}
        //                  [ (x_ij - W_j^T mu_i)^2  +  sigma2_old * W_j^T M_i^{-1} W_j ]
        let mut sigma2_num = 0.0_f64;
        let mut n_observed = 0usize;

        for i in 0..n_samples {
            let obs_idx = &obs_by_sample[i];
            if obs_idx.is_empty() {
                continue;
            }

            let mu_i = &expectations[i];
            let M_inv_i = &posterior_covs[i];
            let n_obs = obs_idx.len();

            let mut W_obs = DMatrix::zeros(n_obs, d);
            let mut x_obs = DVector::zeros(n_obs);
            for (k, &j) in obs_idx.iter().enumerate() {
                x_obs[k] = X_centered[(i, j)];
                for l in 0..d {
                    W_obs[(k, l)] = W[(j, l)];
                }
            }

            // Squared residuals: ||x_obs - W_obs mu_i||^2
            let residual = &x_obs - &W_obs * mu_i;
            sigma2_num += residual.norm_squared();

            // Trace correction: sigma2_old * trace(M_i^{-1} W_obs^T W_obs)
            let WtW_obs = W_obs.transpose() * &W_obs;
            sigma2_num += sigma2 * (M_inv_i * WtW_obs).trace();

            n_observed += n_obs;
        }

        let sigma2_new = if n_observed > 0 {
            (sigma2_num / n_observed as f64).max(1e-6)
        } else {
            sigma2
        };

        Ok((W, sigma2_new))
    }

    fn compute_explained_variance(
        &self,
        W: &DMatrix<f64>,
        sigma2: f64,
        n_features: usize,
    ) -> Result<DVector<f64>> {
        let WtW = W.transpose() * W;
        let svd = SVD::new(WtW, false, false);

        // Eigenvalues of W^T W = variance attributable to each latent direction.
        // Total model variance = trace(W^T W) + p * sigma2  (always sums to <= 1 by construction).
        let signal_variances = svd.singular_values;
        let total_variance = signal_variances.iter().sum::<f64>() + n_features as f64 * sigma2;
        let variance_ratio = if total_variance > 0.0 {
            signal_variances / total_variance
        } else {
            DVector::from_element(signal_variances.len(), 1.0 / signal_variances.len() as f64)
        };

        Ok(variance_ratio)
    }

    fn matrix_inverse(&self, M: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        M.clone().try_inverse()
            .ok_or(PPCAError::MatrixError("Matrix is singular".to_string()))
    }

    /// Get the explained variance ratio from the fitted model
    pub fn explained_variance_ratio(&self) -> Result<&DVector<f64>> {
        self.result.as_ref()
            .map(|r| &r.explained_variance_ratio)
            .ok_or(PPCAError::NoDimensionality)
    }

    /// Get the noise variance from the fitted model
    pub fn noise_variance(&self) -> Result<f64> {
        self.result.as_ref()
            .map(|r| r.sigma2)
            .ok_or(PPCAError::NoDimensionality)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ppca_fit_simple() {
        let data = DMatrix::new_random(10, 5);
        let mask = DMatrix::from_element(10, 5, false);
        let mut ppca = PPCA::new(2);
        let result = ppca.fit(&data, &mask);
        assert!(result.is_ok());
    }

    #[test]
    fn test_ppca_transform() {
        let data = DMatrix::new_random(10, 5);
        let mask = DMatrix::from_element(10, 5, false);
        let mut ppca = PPCA::new(2);
        ppca.fit(&data, &mask).unwrap();
        let transformed = ppca.transform(&data);
        assert!(transformed.is_ok());
        let Y = transformed.unwrap();
        assert_eq!(Y.nrows(), 10);
        assert_eq!(Y.ncols(), 2);
    }

    #[test]
    fn test_ppca_with_missing() {
        let mut data = DMatrix::new_random(10, 5);
        let mut mask = DMatrix::from_element(10, 5, false);
        // Mark some values as missing
        mask[(0, 0)] = true;
        mask[(1, 2)] = true;
        mask[(3, 4)] = true;

        let mut ppca = PPCA::new(2);
        let result = ppca.fit(&data, &mask);
        assert!(result.is_ok());
    }
}
