//! Error types for PPCA operations

use std::fmt;

#[derive(Debug, Clone)]
pub enum PPCAError {
    InvalidDimensions { expected: usize, got: usize },
    NoDimensionality,
    NoConvergence,
    MatrixError(String),
    InvalidComponents { n_components: usize, n_features: usize },
}

impl fmt::Display for PPCAError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            PPCAError::InvalidDimensions { expected, got } => {
                write!(f, "Invalid dimensions: expected {}, got {}", expected, got)
            }
            PPCAError::NoDimensionality => write!(f, "Data has no dimensionality"),
            PPCAError::NoConvergence => write!(f, "EM algorithm did not converge"),
            PPCAError::MatrixError(msg) => write!(f, "Matrix error: {}", msg),
            PPCAError::InvalidComponents { n_components, n_features } => {
                write!(f, "n_components ({}) must be <= n_features ({})", n_components, n_features)
            }
        }
    }
}

impl std::error::Error for PPCAError {}

pub type Result<T> = std::result::Result<T, PPCAError>;
