//! Probabilistic PCA implementation using the EM algorithm
//!
//! Reference: Tipping & Bishop, "Probabilistic Principal Component Analysis", JMLR, 1999

use nalgebra::{DMatrix, DVector, SVD};
use crate::errors::{PPCAError, Result};
use std::f64;

/// Configuration for PPCA training
#[derive(Debug, Clone)]
pub struct PPCAConfig {
    pub n_components: usize,
    pub max_iterations: usize,
    pub tol: f64,
    pub random_state: Option<u64>,
}

impl Default for PPCAConfig {
    fn default() -> Self {
        PPCAConfig {
            n_components: 2,
            max_iterations: 100,
            tol: 1e-4,
            random_state: None,
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

        // Initialize parameters with random values
        let mut loadings = DMatrix::from_fn(n_features, self.config.n_components, |_, _| {
            rand::random::<f64>()
        });
        let mut sigma2 = 1.0;

        // Run EM algorithm
        for _iteration in 0..self.config.max_iterations {
            let old_sigma2 = sigma2;

            // E-step: compute expectations
            let (expectations, C) = self.e_step(&X_centered, &loadings, sigma2, mask)?;

            // M-step: update parameters
            let (new_loadings, new_sigma2) =
                self.m_step(&X_centered, &expectations, &C, sigma2, mask)?;

            loadings = new_loadings;
            sigma2 = new_sigma2;

            // Check convergence
            if (old_sigma2 - sigma2).abs() < self.config.tol {
                break;
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

    fn e_step(
        &self,
        X_centered: &DMatrix<f64>,
        W: &DMatrix<f64>,
        sigma2: f64,
        mask: &DMatrix<bool>,
    ) -> Result<(Vec<DVector<f64>>, DMatrix<f64>)> {
        let (n_samples, n_features) = X_centered.shape();
        let d = self.config.n_components;

        // C = W^T W + sigma2 I
        let WtW = W.transpose() * W;
        let sigma2_I = DMatrix::from_diagonal(&DVector::from_element(d, sigma2));
        let C = WtW + sigma2_I.clone();

        // Compute C^-1
        let C_inv = self.matrix_inverse(&C)?;

        let mut expectations = vec![];
        for i in 0..n_samples {
            let x = X_centered.row(i).transpose();

            // Handle missing values: E[z_i | x_i^obs]
            let x_obs_idx: Vec<usize> = (0..n_features)
                .filter(|&j| !mask[(i, j)])
                .collect();

            if x_obs_idx.is_empty() {
                expectations.push(DVector::zeros(d));
                continue;
            }

            // Partial data
            let mut x_obs = DVector::zeros(x_obs_idx.len());
            for (k, &j) in x_obs_idx.iter().enumerate() {
                x_obs[k] = x[j];
            }

            // Partial loading matrix
            let mut W_obs = DMatrix::zeros(x_obs_idx.len(), d);
            for (k, &j) in x_obs_idx.iter().enumerate() {
                for l in 0..d {
                    W_obs[(k, l)] = W[(j, l)];
                }
            }

            // M = (W_obs^T W_obs + sigma2 I)^-1
            let WobtWo = W_obs.transpose() * &W_obs;
            let M = (WobtWo + sigma2_I.clone()).try_inverse()
                .ok_or(PPCAError::MatrixError("Cannot invert M matrix".to_string()))?;

            // E[z_i | x_i^obs] = M W_obs^T x_obs
            let expectation = &M * W_obs.transpose() * x_obs;
            expectations.push(expectation);
        }

        Ok((expectations, C_inv))
    }

    fn m_step(
        &self,
        X_centered: &DMatrix<f64>,
        expectations: &[DVector<f64>],
        C_inv: &DMatrix<f64>,
        sigma2: f64,
        mask: &DMatrix<bool>,
    ) -> Result<(DMatrix<f64>, f64)> {
        let (n_samples, n_features) = X_centered.shape();
        let d = self.config.n_components;

        // Per-feature W update: each feature j uses only the samples where
        // j is observed, so the denominator scales correctly with missing data.
        let mut W = DMatrix::zeros(n_features, d);
        for j in 0..n_features {
            let mut xz_j = DVector::<f64>::zeros(d);
            let mut zz_j = DMatrix::<f64>::zeros(d, d);
            let mut count_j = 0usize;

            for i in 0..n_samples {
                if !mask[(i, j)] {
                    let x_ij = X_centered[(i, j)];
                    let z = &expectations[i];
                    for l in 0..d {
                        xz_j[l] += x_ij * z[l];
                    }
                    for l1 in 0..d {
                        for l2 in 0..d {
                            zz_j[(l1, l2)] += z[l1] * z[l2];
                        }
                    }
                    count_j += 1;
                }
            }

            // sum_{i in S_j} E[z_i z_i^T] = Z_j^T Z_j + |S_j| * sigma2 * C_inv
            let zz_j_full = &zz_j + count_j as f64 * sigma2 * C_inv;
            let zz_j_inv = zz_j_full.try_inverse()
                .ok_or(PPCAError::MatrixError("Cannot invert per-feature ZtZ".to_string()))?;
            let w_j = &zz_j_inv * &xz_j;
            for l in 0..d {
                W[(j, l)] = w_j[l];
            }
        }

        // Precompute trace(C_inv W^T W) for the sigma2 correction
        let WtW = W.transpose() * &W;
        let C_inv_WtW = C_inv * &WtW;
        let trace_correction = C_inv_WtW.trace();

        // Update sigma2 — only over observed entries
        let mut tr_sum = 0.0;
        let mut n_observed = 0usize;
        for i in 0..n_samples {
            let z = &expectations[i];

            for j in 0..n_features {
                if !mask[(i, j)] {
                    let x_ij = X_centered[(i, j)];
                    let mut reconstruction = 0.0;
                    for l in 0..d {
                        reconstruction += W[(j, l)] * z[l];
                    }
                    tr_sum += (x_ij - reconstruction).powi(2);
                    n_observed += 1;
                }
            }

            // Trace correction: sigma2 * trace(M^{-1} W^T W) accounts for
            // posterior uncertainty in z beyond the point estimate.
            tr_sum += sigma2 * trace_correction;
        }

        let sigma2_new = if n_observed > 0 {
            tr_sum / n_observed as f64
        } else {
            sigma2
        };

        Ok((W, sigma2_new.max(1e-6)))
    }

    fn compute_explained_variance(
        &self,
        W: &DMatrix<f64>,
        sigma2: f64,
        n_features: usize,
    ) -> Result<DVector<f64>> {
        let WtW = W.transpose() * W;
        let svd = SVD::new(WtW, false, false);

        // Total variance under the PPCA model is
        //   trace(C) = trace(W W^T + sigma2 I) = sum(eig(W^T W)) + n_features * sigma2
        let signal_variances = svd.singular_values;
        let total_variance: f64 = signal_variances.iter().sum::<f64>() + n_features as f64 * sigma2;

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
