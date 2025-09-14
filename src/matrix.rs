//! This module defines the core abstraction for linear operators.
//!
//! In numerical linear algebra, many iterative algorithms, such as the Lanczos
//! process, do not require direct access to the individual elements of a matrix.
//! Instead, their fundamental operation is the matrix-vector product. This observation
//! allows for a powerful abstraction: the algorithm can be written to operate on any
//! object that can perform this action, known as a "linear operator."
//!
//! This "matrix-free" approach offers significant advantages:
//! 1.  **Generality**: The Lanczos algorithm can be implemented once and used with
//!     dense matrices, sparse matrices, or even functions that compute the product
//!     without explicitly storing a matrix. This is common when the operator represents
//!     a complex physical simulation or is the result of other matrix compositions.
//! 2.  **Testability**: The same algorithm can be tested on simple, small dense
//!     matrices for which results can be easily verified, and then deployed on large,
//!     complex sparse matrices without changing the core logic.
//! 3.  **Encapsulation**: The complexities of matrix storage and the specifics of the
//!     matrix-vector product are hidden behind a clean interface, leading to more
//!     modular and maintainable code.
//!
//! The central piece of this module is the `LinearOperator` trait, which formalizes
//! this contract.

use faer::{Mat, MatMut, MatRef, prelude::Reborrow, traits::ComplexField};

/// Represents a linear operator that can be applied to a vector (or a matrix).
///
/// This trait provides an abstraction for the matrix-vector product, which is the
/// fundamental operation required by Krylov subspace methods like the Lanczos algorithm.
/// By depending on this trait rather than a concrete matrix type, algorithms can be
/// written in a generic, "matrix-free" manner.
///
/// # Type Parameters
///
/// *   `T`: The scalar type, which must implement `ComplexField`. This trait from `faer`
///     provides the necessary arithmetic operations for `f32`, `f64`, and their complex
///     counterparts.
///
/// # Example
///
/// A generic function that uses a `LinearOperator`.
///
/// rust
/// ```
/// use faer::{Mat, MatRef};
/// use faer_traits::ComplexField;
/// use prelude::LinearOperator;
///
/// fn some_iterative_solver<T: ComplexField>(
///     operator: &impl LinearOperator<T>,
///     initial_vector: MatRef<T>,
/// ) -> Mat<T> {
///     // The function only needs to know the operator's dimensions and
///     // how to apply it.
///     assert_eq!(operator.ncols(), initial_vector.nrows());
///
///     // Perform one step of an iterative method, e.g., power iteration.
///     operator.apply(initial_vector)
/// }
/// ```
pub trait LinearOperator<T: ComplexField> {
    /// Returns the number of rows of the operator.
    fn nrows(&self) -> usize;

    /// Returns the number of columns of the operator.
    fn ncols(&self) -> usize;

    /// Applies the linear operator to a matrix `rhs`.
    ///
    /// In the context of the Lanczos algorithm, `rhs` will be a single-column matrix (a vector).
    /// The implementation must return an owned matrix (`Mat<T>`) containing the result of
    /// the operation `A * rhs`.
    ///
    /// # Panics
    ///
    /// This method is expected to panic if the inner dimension of the operator does not match
    /// the number of rows of `rhs`.
    fn apply(&self, rhs: MatRef<'_, T>) -> Mat<T>;
}

/// Implementation of `LinearOperator` for `faer`'s immutable dense matrix view (`MatRef`).
/// This is the primary concrete implementation that the generic algorithm will be tested against.
impl<'a, T: ComplexField> LinearOperator<T> for MatRef<'a, T> {
    #[inline]
    fn nrows(&self) -> usize {
        self.nrows()
    }

    #[inline]
    fn ncols(&self) -> usize {
        self.ncols()
    }

    #[inline]
    fn apply(&self, rhs: MatRef<'_, T>) -> Mat<T> {
        // Ensure dimensional compatibility for the matrix product.
        // This check is crucial for correctness in numerical code.
        assert_eq!(
            self.ncols(),
            rhs.nrows(),
            "Dimension mismatch: operator columns ({}) do not match vector rows ({}).",
            self.ncols(),
            rhs.nrows(),
        );

        // Defer to faer's highly optimized matrix multiplication routine.
        // The `*` operator on `MatRef` produces an owned `Mat`.
        self * rhs
    }
}

/// Implementation of `LinearOperator` for `faer`'s mutable dense matrix view (`MatMut`).
/// This implementation delegates to the `MatRef` implementation via a reborrow.
impl<'a, T: ComplexField> LinearOperator<T> for MatMut<'a, T> {
    #[inline]
    fn nrows(&self) -> usize {
        self.rb().nrows()
    }

    #[inline]
    fn ncols(&self) -> usize {
        self.rb().ncols()
    }

    #[inline]
    fn apply(&self, rhs: MatRef<'_, T>) -> Mat<T> {
        // Reborrow as an immutable view (`MatRef`) and call its `apply` method.
        // This avoids code duplication and leverages the core implementation.
        self.rb().apply(rhs)
    }
}

/// Implementation of `LinearOperator` for `faer`'s owned dense matrix (`Mat`).
/// This implementation delegates to the `MatRef` implementation via a reference.
impl<T: ComplexField> LinearOperator<T> for Mat<T> {
    #[inline]
    fn nrows(&self) -> usize {
        self.as_ref().nrows()
    }

    #[inline]
    fn ncols(&self) -> usize {
        self.as_ref().ncols()
    }

    #[inline]
    fn apply(&self, rhs: MatRef<'_, T>) -> Mat<T> {
        // Create an immutable view (`MatRef`) and call its `apply` method.
        self.as_ref().apply(rhs)
    }
}

// Unit tests to verify the correctness of the LinearOperator trait and its implementations.
#[cfg(test)]
mod tests {
    use super::*;
    use faer::mat;

    #[test]
    fn test_linear_operator_for_mat() {
        // Define a concrete matrix and vector for testing.
        let matrix: Mat<f64> = mat![[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0],];
        let vector: Mat<f64> = mat![[1.0], [2.0], [3.0]];

        // Expected result from manual calculation or direct multiplication.
        let expected_result = &matrix * &vector;

        // Test the `LinearOperator` implementation for `Mat<f64>`.
        let operator: &dyn LinearOperator<f64> = &matrix;
        let result = operator.apply(vector.as_ref());

        // Verify that the abstraction produces the correct result.
        assert_eq!(result, expected_result);
        assert_eq!(operator.nrows(), 3);
        assert_eq!(operator.ncols(), 3);
    }

    #[test]
    fn test_linear_operator_for_mat_ref_and_mut() {
        let mut matrix: Mat<f64> = mat![[1.0, 2.0], [3.0, 4.0]];
        let vector: Mat<f64> = mat![[1.0], [1.0]];

        // Calculate the expected result once.
        let expected = &matrix * &vector;

        // Test the implementation for `MatRef`.
        let operator_ref: &dyn LinearOperator<f64> = &matrix.as_ref();
        let result_ref = operator_ref.apply(vector.as_ref());
        assert_eq!(result_ref, expected);

        // Test the implementation for `MatMut`.
        let operator_mut: &dyn LinearOperator<f64> = &matrix.as_mut();
        let result_mut = operator_mut.apply(vector.as_ref());
        assert_eq!(result_mut, expected);
    }

    #[test]
    #[should_panic(
        expected = "Dimension mismatch: operator columns (2) do not match vector rows (3)."
    )]
    fn test_dimension_mismatch_panic() {
        let matrix: Mat<f64> = mat![[1.0, 0.0], [0.0, 1.0]];
        let vector: Mat<f64> = mat![[1.0], [2.0], [3.0]]; // Incorrect dimension

        // This call should panic due to the assertion inside `apply`.
        let operator: &dyn LinearOperator<f64> = &matrix;
        operator.apply(vector.as_ref());
    }
}
