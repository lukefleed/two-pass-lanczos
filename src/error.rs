//! This module defines the custom error types for the library.
//!
//! This module centralizes all possible error conditions that
//! can arise within the Lanczos algorithm implementations into a single, comprehensive
//! enum: [`LanczosError`].
//!
//! Using the [`thiserror`] crate allows us to create idiomatic error types with minimal
//! boilerplate. Note that [`faer::linalg::evd::EvdError`] does not implement the standard
//! [`std::error::Error`] trait, so we wrap it manually to provide a compatible error type.
use thiserror::Error;

/// Represents all possible errors that can occur during a Lanczos process.
///
#[derive(Error, Debug)]
#[error(transparent)]
pub struct LanczosError(#[from] LanczosErrorKind);

/// Private enum containing the distinct kinds of errors.
/// This separation allows for a clean `Display` implementation via [`thiserror`]
/// while handling non-standard error types manually.
#[derive(Error, Debug, PartialEq)]
pub(crate) enum LanczosErrorKind {
    /// Occurs when the Lanczos iteration terminates prematurely because the beta
    /// coefficient becomes zero (or numerically indistinguishable from zero).
    #[error(
        "Lanczos iteration breakdown at step {k}: Beta coefficient is zero. The Krylov subspace is invariant."
    )]
    Breakdown { k: usize },

    /// Indicates that the dimensions of the operator and the input vector are
    /// incompatible for a matrix-vector product.
    #[error(
        "Dimension mismatch: operator has {operator_cols} columns but vector has {vector_rows} rows."
    )]
    DimensionMismatch {
        operator_cols: usize,
        vector_rows: usize,
    },

    /// Indicates that an invalid input parameter was provided to a function.
    #[error("Invalid input parameter: {0}")]
    InputError(String),

    /// Wraps an error originating from [`faer`]'s eigendecomposition module.
    #[error("A numerical error occurred during the eigendecomposition of T_k: {0:?}")]
    EvdError(faer::linalg::evd::EvdError),
}

// Manually implement PartialEq for the public error type.
// We compare the inner `LanczosErrorKind`.
impl PartialEq for LanczosError {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

// Unit tests to ensure error messages are formatted correctly.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_breakdown_error_message() {
        let error = LanczosError(LanczosErrorKind::Breakdown { k: 42 });
        let expected_message = "Lanczos iteration breakdown at step 42: Beta coefficient is zero. The Krylov subspace is invariant.";
        assert_eq!(error.to_string(), expected_message);
    }

    #[test]
    fn test_dimension_mismatch_error_message() {
        let error = LanczosError(LanczosErrorKind::DimensionMismatch {
            operator_cols: 100,
            vector_rows: 99,
        });
        let expected_message =
            "Dimension mismatch: operator has 100 columns but vector has 99 rows.";
        assert_eq!(error.to_string(), expected_message);
    }

    #[test]
    fn test_input_error_message() {
        let error = LanczosError(LanczosErrorKind::InputError(
            "The initial vector `b` must not be a zero vector.".to_string(),
        ));
        let expected_message =
            "Invalid input parameter: The initial vector `b` must not be a zero vector.";
        assert_eq!(error.to_string(), expected_message);
    }

    #[test]
    fn test_evd_error_message() {
        let evd_error = faer::linalg::evd::EvdError::NoConvergence;
        let error = LanczosError(LanczosErrorKind::EvdError(evd_error));
        // Note: The message now uses the `Debug` format for the inner error.
        let expected_message =
            "A numerical error occurred during the eigendecomposition of T_k: NoConvergence";
        assert_eq!(error.to_string(), expected_message);
    }
}
