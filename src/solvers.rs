//! This module provides a high-level, user-friendly API for executing Lanczos-based
//! algorithms to compute the action of a matrix function on a vector, `f(A)b`.

use crate::{
    algorithms::lanczos::{lanczos_pass_one, lanczos_pass_two, lanczos_standard},
    error::{LanczosError, LanczosErrorKind},
};
use faer::{
    Accum,
    dyn_stack::MemStack,
    linalg::matmul::matmul,
    matrix_free::LinOp,
    prelude::*,
    traits::{ComplexField, RealField},
};

/// Computes an approximation to `f(A)b` using the standard one-pass Lanczos method.
///
/// This method generates and stores an orthonormal basis for the Krylov subspace,
/// resulting in a memory complexity of O(nk), where `n` is the dimension of the
/// operator and `k` is the number of iterations.
///
/// # Arguments
/// * `operator`: A linear operator `A` that implements `faer::matrix_free::LinOp`.
/// * `b`: The starting vector. Must not be a zero vector.
/// * `k`: The number of iterations to perform.
/// * `stack`: Memory stack for temporary allocations.
/// * `f_tk_solver`: A closure that takes the `alphas` and `betas` defining the
///   tridiagonal matrix `T_k` and returns the vector `f(T_k) * e_1`.
///
/// # Returns
/// A `Result` containing the final approximate solution vector `x_k`, or a `LanczosError`.
// Computes an approximation to `f(A)b` using the standard one-pass Lanczos method.
///
/// This method generates and stores an orthonormal basis for the Krylov subspace,
/// resulting in a memory complexity of O(nk), where `n` is the dimension of the
/// operator and `k` is the number of iterations.
///
/// # Arguments
/// * `operator`: A linear operator `A` that implements `faer::matrix_free::LinOp`.
/// * `b`: The starting vector. Must not be a zero vector.
/// * `k`: The number of iterations to perform.
/// * `stack`: Memory stack for temporary allocations.
/// * `f_tk_solver`: A closure that takes the `alphas` and `betas` defining the
///   tridiagonal matrix `T_k` and returns the vector `f(T_k) * e_1`.
///
/// # Returns
/// A `Result` containing the final approximate solution vector `x_k`, or a `LanczosError`.
pub fn lanczos<T, O, F>(
    operator: &O,
    b: MatRef<'_, T>,
    k: usize,
    stack: &mut MemStack,
    mut f_tk_solver: F,
) -> Result<Mat<T>, LanczosError>
where
    T: ComplexField,
    T::Real: RealField,
    O: LinOp<T>,
    F: FnMut(&[T::Real], &[T::Real]) -> Result<Mat<T>, anyhow::Error>,
{
    // Perform the standard one-pass Lanczos iteration. This is memory-intensive
    // as it stores the full basis matrix `v_k`.
    let standard_output = lanczos_standard(operator, b, k, stack, None)?;

    // If no steps were taken (e.g., input vector was zero), return a zero vector.
    if standard_output.decomposition.steps_taken == 0 {
        return Ok(Mat::zeros(b.nrows(), 1));
    }

    // Invoke the user-provided closure to solve for `f(T_k) * e_1`.
    let y_k_prime = f_tk_solver(
        &standard_output.decomposition.alphas,
        &standard_output.decomposition.betas,
    )
    .map_err(|e| LanczosError::from(LanczosErrorKind::SolverError(e.to_string())))?;

    // Validate the dimensions of the vector returned by the user's solver.
    if y_k_prime.nrows() != standard_output.decomposition.steps_taken || y_k_prime.ncols() != 1 {
        return Err(LanczosErrorKind::ParameterMismatch {
            param_name: "y_k_prime".to_string(),
            expected: standard_output.decomposition.steps_taken,
            actual: y_k_prime.nrows(),
        }
        .into());
    }

    // Scale the result by the norm of the initial vector `b` to get the final `y_k`.
    let y_k = &y_k_prime * Scale(T::from_real_impl(&standard_output.decomposition.b_norm));

    // --- SOLUTION RECONSTRUCTION ---
    // Pre-allocate the destination vector `x_k` for the final result.
    // This gives us explicit control over the memory allocation.
    let mut x_k = Mat::<T>::zeros(standard_output.v_k.nrows(), 1);

    // Use the explicit `matmul` function instead of the `*` operator.
    // This function writes the result directly into the pre-allocated `x_k`
    // without performing any intermediate allocations for the output.
    // This prevents the temporary memory spike observed in the experiments.
    matmul(
        x_k.as_mut(),
        Accum::Replace, // Overwrite the destination: x_k = alpha * V_k * y_k
        standard_output.v_k.as_ref(), // lhs
        y_k.as_ref(),   // rhs
        T::one_impl(),  // alpha = 1.0
        Par::Seq,       // Use sequential execution for consistency
    );

    Ok(x_k)
}

/// Computes an approximation to `f(A)b` using the memory-efficient two-pass Lanczos method.
///
/// This method avoids storing the full basis matrix, resulting in a memory complexity of O(n).
/// It performs a first pass to compute the scalar coefficients of `T_k`, and a second pass
/// to reconstruct the solution vector on-the-fly.
///
/// # Arguments
/// * `operator`: A linear operator `A` that implements `faer::matrix_free::LinOp`.
/// * `b`: The starting vector. Must not be a zero vector.
/// * `k`: The number of iterations to perform.
/// * `stack`: Memory stack for temporary allocations.
/// * `f_tk_solver`: A closure that takes the `alphas` and `betas` defining the
///   tridiagonal matrix `T_k` and returns the vector `f(T_k) * e_1`.
///
/// # Returns
/// A `Result` containing the final approximate solution vector `x_k`, or a `LanczosError`.
pub fn lanczos_two_pass<T, O, F>(
    operator: &O,
    b: MatRef<'_, T>,
    k: usize,
    stack: &mut MemStack,
    mut f_tk_solver: F,
) -> Result<Mat<T>, LanczosError>
where
    T: ComplexField,
    T::Real: RealField,
    O: LinOp<T>,
    F: FnMut(&[T::Real], &[T::Real]) -> Result<Mat<T>, anyhow::Error>,
{
    // Perform the first pass, which only computes the scalar decomposition and
    // uses minimal memory (O(n)).
    let decomposition = lanczos_pass_one(operator, b, k, stack)?;

    // If no steps were taken, return a zero vector.
    if decomposition.steps_taken == 0 {
        return Ok(Mat::zeros(b.nrows(), 1));
    }

    // Invoke the user-provided closure to solve for `f(T_k) * e_1`.
    let y_k_prime = f_tk_solver(&decomposition.alphas, &decomposition.betas)
        .map_err(|e| LanczosError::from(LanczosErrorKind::SolverError(e.to_string())))?;

    // Validate the dimensions of the vector returned by the user's solver.
    if y_k_prime.nrows() != decomposition.steps_taken || y_k_prime.ncols() != 1 {
        return Err(LanczosErrorKind::ParameterMismatch {
            param_name: "y_k_prime".to_string(),
            expected: decomposition.steps_taken,
            actual: y_k_prime.nrows(),
        }
        .into());
    }

    // Scale the result by the norm of the initial vector `b`.
    let y_k = &y_k_prime * Scale(T::from_real_impl(&decomposition.b_norm));

    // Perform the second pass, which reconstructs the solution vector on-the-fly
    // without storing the full basis.
    lanczos_pass_two(operator, b, &decomposition, y_k.as_ref(), stack)
}
