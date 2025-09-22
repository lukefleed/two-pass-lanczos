//! High-level solvers for computing the action of a matrix function on a vector.
//!
//! This module provides a user-friendly API for executing Lanczos-based algorithms to
//! compute $\mathbf{x} \approx f(\mathbf{A})\mathbf{b}$. It abstracts the internal details of the
//! Lanczos passes and exposes two primary functions, [`lanczos`] and [`lanczos_two_pass`],
//! which represent the standard one-pass and the memory-efficient two-pass methods, respectively.

use crate::{
    algorithms::{
        lanczos::lanczos_standard,
        lanczos_two_pass::{lanczos_pass_one, lanczos_pass_two},
    },
    error::{LanczosError, LanczosErrorKind},
};
use faer::{
    Accum, Par,
    dyn_stack::MemStack,
    linalg::matmul::matmul,
    matrix_free::LinOp,
    prelude::*,
    traits::{ComplexField, RealField},
};

/// Computes an approximation to $f(\mathbf{A})\mathbf{b}$ using the standard one-pass Lanczos method.
///
/// This method generates and stores an orthonormal basis $\mathbf{V}_k$ for the Krylov subspace,
/// resulting in a memory complexity of $O(nk)$. It is generally preferred when $k$ is small
/// enough that the memory required to store $\mathbf{V}_k$ is not prohibitive.
///
/// # Type Parameters
/// * `T`: The complex field type (e.g., `c64`).
/// * `O`: A type that implements `faer::matrix_free::LinOp<T>`.
/// * `F`: A closure type for the projected problem solver.
///
/// # Arguments
/// * `operator`: A linear operator $\mathbf{A}$.
/// * `b`: The starting vector. Must not be a zero vector.
/// * `k`: The number of Lanczos iterations to perform.
/// * `stack`: A `MemStack` for temporary allocations.
/// * `f_tk_solver`: A closure that takes the coefficients $(\alpha_j, \beta_j)$ defining the
///   tridiagonal matrix $\mathbf{T}_k$ and returns the vector $f(\mathbf{T}_k) \mathbf{e}_1$.
///   This decouples the Lanczos iteration from the specifics of the function $f$.
///
/// # Returns
/// A `Result` containing the final approximate solution vector $\mathbf{x}_k$, or a `LanczosError`.
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
    // 1. Perform the standard one-pass Lanczos iteration. This is memory-intensive
    // as it materializes the full basis matrix `v_k` in memory.
    let standard_output = lanczos_standard(operator, b, k, stack, None)?;

    // Handle the case where the iteration terminates immediately (e.g., zero input vector).
    if standard_output.decomposition.steps_taken == 0 {
        return Ok(Mat::zeros(b.nrows(), 1));
    }

    // 2. Solve the projected problem. Invoke the user-provided closure to compute
    // y'_k = f(T_k) * e_1. This is the coefficient vector for the solution in the
    // Lanczos basis, prior to scaling.
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

    // --- 3. Solution Reconstruction ---
    // Pre-allocate the destination vector `x_k`. This explicit allocation avoids
    // intermediate copies and gives the optimizer better visibility.
    let mut x_k = Mat::<T>::zeros(standard_output.v_k.nrows(), 1);

    // Compute x_k = V_k * y_k_prime * ||b|| using [`faer::linalg::matmul`].
    // This high-performance kernel is optimized for memory access patterns and can leverage
    // SIMD instructions. The scaling by ||b|| is handled efficiently by the `alpha`
    // parameter of `matmul`, avoiding an extra allocation for a scaled coefficient vector.
    matmul(
        x_k.as_mut(),
        Accum::Replace,
        standard_output.v_k.as_ref(),
        y_k_prime.as_ref(),
        // The final scaling factor.
        T::from_real_impl(&standard_output.decomposition.b_norm),
        Par::Seq,
    );

    Ok(x_k)
}

/// Computes an approximation to $f(\mathbf{A})\mathbf{b}$ using the memory-efficient two-pass Lanczos method.
///
/// This method avoids storing the full basis matrix, resulting in a memory complexity of $O(n)$.
/// It performs a first pass to compute the scalar coefficients of $\mathbf{T}_k$, and a second pass
/// to reconstruct the solution vector by regenerating the basis vectors on-the-fly. This approach
/// is ideal for problems where $k$ is large and memory is a constraint.
///
///
/// # Type Parameters
/// * `T`: The complex field type (e.g., `c64`).
/// * `O`: A type that implements `faer::matrix_free::LinOp<T>`.
/// * `F`: A closure type for the projected problem solver.
///
/// # Arguments
/// * `operator`: A linear operator $\mathbf{A}$.
/// * `b`: The starting vector. Must not be a zero vector.
/// * `k`: The number of Lanczos iterations to perform.
/// * `stack`: A `MemStack` for temporary allocations.
/// * `f_tk_solver`: A closure that takes the coefficients $(\alpha_j, \beta_j)$ defining the
///   tridiagonal matrix $\mathbf{T}_k$ and returns the vector $f(\mathbf{T}_k) \mathbf{e}_1$.
///   This decouples the Lanczos iteration from the specifics of the function $f$.
///
/// # Returns
/// A `Result` containing the final approximate solution vector $\mathbf{x}_k$, or a `LanczosError`.
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
    // 1. Perform the first pass, which is memory-light. It computes the scalar
    // decomposition and uses only a constant number of n-dimensional vectors.
    let decomposition = lanczos_pass_one(operator, b, k, stack)?;

    if decomposition.steps_taken == 0 {
        return Ok(Mat::zeros(b.nrows(), 1));
    }

    // 2. Solve the projected problem, identical to the one-pass method.
    let y_k_prime = f_tk_solver(&decomposition.alphas, &decomposition.betas)
        .map_err(|e| LanczosError::from(LanczosErrorKind::SolverError(e.to_string())))?;

    if y_k_prime.nrows() != decomposition.steps_taken || y_k_prime.ncols() != 1 {
        return Err(LanczosErrorKind::ParameterMismatch {
            param_name: "y_k_prime".to_string(),
            expected: decomposition.steps_taken,
            actual: y_k_prime.nrows(),
        }
        .into());
    }

    // 3. Scale the result by the norm of the initial vector `b` to get the final
    // coefficient vector for reconstruction.
    let y_k = &y_k_prime * Scale(T::from_real_impl(&decomposition.b_norm));

    // 4. Perform the second pass. This reconstructs the solution vector on-the-fly
    // by regenerating the basis vectors one at a time and accumulating the result,
    // thereby avoiding the storage of the full basis matrix.
    lanczos_pass_two(operator, b, &decomposition, y_k.as_ref(), stack)
}
