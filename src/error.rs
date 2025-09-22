//! Error types for Lanczos algorithm failures.
//!
//! This module defines error conditions that can occur during Lanczos iterations.
//! The main type [`LanczosError`] covers numerical breakdowns, dimension mismatches,
//! invalid parameters, and solver failures.

use thiserror::Error;

/// Represents all possible errors that can occur during a Lanczos process.
///
#[derive(Error, Debug)]
#[error(transparent)]
pub struct LanczosError(#[from] pub(crate) LanczosErrorKind);

/// Private enum containing the distinct kinds of errors.
/// This separation allows for a clean `Display` implementation via [`thiserror`]
/// while handling non-standard error types manually.
#[allow(dead_code)]
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

    /// Indicates a mismatch in the dimensions of parameters provided to a function,
    /// for reasons other than an invalid matrix-vector product.
    #[error("Parameter mismatch: `{param_name}` expects size {expected}, but got {actual}.")]
    ParameterMismatch {
        param_name: String,
        expected: usize,
        actual: usize,
    },

    /// Wraps an error originating from [`faer`]'s eigendecomposition module.
    #[error("A numerical error occurred during the eigendecomposition of T_k: {0:?}")]
    EvdError(faer::linalg::evd::EvdError),

    /// Wraps an error returned by the user-provided solver for `f(T_k)`.
    #[error("The user-provided f(T_k) solver failed: {0}")]
    SolverError(String),
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
    fn test_parameter_mismatch_error_message() {
        let error = LanczosError(LanczosErrorKind::ParameterMismatch {
            param_name: "y_k".to_string(),
            expected: 10,
            actual: 9,
        });
        let expected_message = "Parameter mismatch: `y_k` expects size 10, but got 9.";
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
        let expected_message =
            "A numerical error occurred during the eigendecomposition of T_k: NoConvergence";
        assert_eq!(error.to_string(), expected_message);
    }

    #[test]
    fn test_solver_error_message() {
        let error = LanczosError(LanczosErrorKind::SolverError(
            "Custom solver failed".to_string(),
        ));
        let expected_message = "The user-provided f(T_k) solver failed: Custom solver failed";
        assert_eq!(error.to_string(), expected_message);
    }
}
