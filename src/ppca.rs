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

        // Center the data
        let mean = self.compute_mean(X, mask)?;
        let mut X_centered = X.clone();
        for i in 0..n_samples {
            X_centered.set_row(i, &(X.row(i) - mean.transpose()));
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
                self.m_step(&X_centered, &expectations, &C, sigma2)?;

            loadings = new_loadings;
            sigma2 = new_sigma2;

            // Check convergence
            if (old_sigma2 - sigma2).abs() < self.config.tol {
                break;
            }
        }

        // Compute explained variance
        let explained_variance = self.compute_explained_variance(&loadings, sigma2)?;

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
                centered.set_row(i, &(X.row(i) - result.mean.transpose()));
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
        _sigma2: f64,
    ) -> Result<(DMatrix<f64>, f64)> {
        let (n_samples, n_features) = X_centered.shape();
        let d = self.config.n_components;

        // Compute Z and related quantities
        let mut Z = DMatrix::zeros(n_samples, d);
        for (i, z) in expectations.iter().enumerate() {
            Z.set_row(i, &z.transpose());
        }

        let ZtZ: DMatrix<f64> = Z.transpose() * &Z + n_samples as f64 * C_inv;

        // Update W
        let XtZ = X_centered.transpose() * &Z;
        let ZtZ_inv = self.matrix_inverse(&ZtZ)?;
        let W = &XtZ * &ZtZ_inv;

        // Update sigma2
        let mut tr_sum = 0.0;
        for i in 0..n_samples {
            let x = X_centered.row(i).transpose();
            let z = &expectations[i];

            let x_dot_x = x.dot(&x);
            let z_dot_WtWz = (z.transpose() * W.transpose() * &W * z)[0];

            tr_sum += x_dot_x - 2.0 * (W.transpose() * &x).dot(z) + z_dot_WtWz;
        }

        let sigma2_new = tr_sum / (n_samples as f64 * n_features as f64);

        Ok((W, sigma2_new.max(1e-6)))
    }

    fn compute_explained_variance(
        &self,
        W: &DMatrix<f64>,
        sigma2: f64,
    ) -> Result<DVector<f64>> {
        let WtW = W.transpose() * W;
        let svd = SVD::new(WtW, true, true);

        let singular_values = svd.singular_values.clone();
        let variance = singular_values.map(|v| (v - sigma2).max(0.0));

        let sum_variance: f64 = variance.iter().sum();
        let variance_ratio = variance / sum_variance;

        Ok(variance_ratio)
    }

    fn matrix_inverse(&self, M: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        M.clone().try_inverse()
            .ok_or(PPCAError::MatrixError("Matrix is singular".to_string()))
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
