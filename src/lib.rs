//! Probabilistic Principal Component Analysis (PPCA) with missing value support
//!
//! This module provides both the core algorithm and Python bindings.

pub mod bindings;
pub mod errors;
pub mod ppca;

pub use errors::PPCAError;
pub use ppca::{NoiseType, PPCAConfig, PPCAResult, PPCA};

// Re-export for convenience
pub type Result<T> = std::result::Result<T, PPCAError>;

// Re-export Python bindings at crate root for PyO3
pub use bindings::PPCARust;

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    #[test]
    fn test_ppca_basic() {
        let config = PPCAConfig::default();
        assert_eq!(config.n_components, 2);
    }
}
