//! Probabilistic Principal Component Analysis (PPCA) implementation using EM algorithm
//!
//! This crate provides an efficient implementation of PPCA using the EM algorithm,
//! with support for missing values.

pub mod ppca;
pub mod errors;

pub use ppca::{PPCA, PPCAConfig, PPCAResult};
pub use errors::PPCAError;

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    #[test]
    fn test_ppca_basic() {
        // Basic test to ensure library loads
        let config = PPCAConfig::default();
        assert_eq!(config.n_components, 2);
    }
}
