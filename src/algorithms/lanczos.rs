//! Standard one-pass symmetric Lanczos algorithm implementation.
//!
//! ** NOTE: We recommend using the high-level method [`crate::solvers::lanczos`] instead. This
//! module is intended for use cases where fine-grained control over the Lanczos process is required.
//!
//! This module implements the traditional Lanczos method that generates and stores the
//! full orthonormal basis V_k during iteration. The main function [`lanczos_standard`]
//! executes the Lanczos recurrence while maintaining all basis vectors in memory.
//!
//! Memory usage scales as O(nk) where n is the problem dimension and k is the number
//! of iterations. This approach is suitable when k is small enough that storing the
//! n×k basis matrix does not create memory constraints. The stored basis enables
//! direct solution reconstruction via matrix multiplication.
//!
//! ## When to use this module directly
//!
//! - You need access to the intermediate Lanczos basis matrix V_k
//! - You want to implement custom iteration callbacks
//! - You're conducting algorithm benchmarking
//!
//! For normal usage, prefer [`crate::solvers::lanczos`] which provides a simpler interface.

use super::{
    LanczosCallback, LanczosDecomposition, LanczosError, LanczosIteration, LanczosOutput,
    TridiagonalSystemView, breakdown_tolerance,
};
use faer::{
    dyn_stack::MemStack,
    matrix_free::LinOp,
    prelude::*,
    traits::{ComplexField, RealField},
};

/// Performs the standard one-pass symmetric Lanczos algorithm.
///
/// This function executes up to `k` steps of the Lanczos process to generate an
/// orthonormal basis $\mathbf{V}_k$ for the Krylov subspace $\mathcal{K}_k(\mathbf{A}, \mathbf{b})$.
/// It stores the generated basis vectors in the columns of the matrix `v_k`, leading to
/// an $O(nk)$ memory complexity.
///
/// An optional callback function can be provided to inspect the algorithm's state at
/// each iteration, allowing for custom convergence monitoring.
///
/// # Arguments
/// * `operator`: A linear operator implementing [`faer::matrix_free::LinOp`].
/// * `b`: The starting vector. Must not be a zero vector.
/// * `k`: The maximum number of iterations to perform.
/// * `stack`: A [`MemStack`] for temporary allocations.
/// * `callback`: An optional mutable reference to a callback function invoked at each iteration.
///
/// # Returns
/// A [`Result`] containing the [`LanczosOutput`] on success, or a [`LanczosError`] on failure.
/// The output includes the basis matrix $\mathbf{V}_k$ and the scalar decomposition
/// defining the tridiagonal matrix $\mathbf{T}_k$.
pub fn lanczos_standard<T: ComplexField>(
    operator: &impl LinOp<T>,
    b: MatRef<'_, T>,
    k: usize,
    stack: &mut MemStack,
    mut callback: Option<&mut LanczosCallback<T>>,
) -> Result<LanczosOutput<T>, LanczosError>
where
    T::Real: RealField,
{
    let b_norm = b.norm_l2();

    // Pre-allocate the basis matrix V_k. This is a key aspect of the one-pass
    // approach. While it reserves a potentially large contiguous block of memory,
    // it avoids incremental resizing, which would be highly inefficient.
    let mut v_k = Mat::<T>::zeros(operator.nrows(), k);

    // Pre-allocate vectors for the scalar coefficients with a capacity hint to
    // prevent reallocations during the main loop.
    let mut alphas = Vec::with_capacity(k);
    let mut betas = Vec::with_capacity(k - 1);

    // Initialize the stateful Lanczos iterator.
    let mut lanczos_iter = LanczosIteration::new(operator, b, k, T::Real::copy_impl(&b_norm))?;

    // The first Lanczos vector is the normalized input vector `b`.
    v_k.col_mut(0)
        .copy_from(lanczos_iter.v_curr.as_ref().col(0));

    let mut steps_taken = 0;

    for i in 0..k {
        if let Some(step) = lanczos_iter.next_step(stack) {
            alphas.push(step.alpha);
            steps_taken += 1;

            // If a callback is provided, invoke it with the current state. This allows
            // for external logic to monitor the process without modifying the core algorithm.
            if let Some(ref mut cb) = callback {
                // We provide a view into the valid portion of the basis matrix.
                let current_v_k = v_k.as_ref().get(.., 0..steps_taken);
                let t_k_view = TridiagonalSystemView {
                    alphas: &alphas,
                    betas: &betas,
                    steps_taken,
                };

                // The callback can signal for an early, graceful stop.
                if !cb(steps_taken, current_v_k, &t_k_view) {
                    break;
                }
            }

            // A zero (or numerically zero) beta indicates that breakdown has occurred.
            // The Krylov subspace is invariant, and the iteration must terminate.
            let tolerance = breakdown_tolerance::<T::Real>();
            if step.beta <= tolerance {
                break;
            }

            // Store the off-diagonal element and the newly computed basis vector.
            // This is skipped in the final iteration as v_{k+1} is not needed.
            if i < k - 1 {
                betas.push(step.beta);
                // The `lanczos_iter` has already updated its internal state, so `v_curr`
                // now holds the next orthonormal vector, v_{i+1}.
                v_k.col_mut(i + 1)
                    .copy_from(lanczos_iter.v_curr.as_ref().col(0));
            }
        } else {
            // This branch is taken if the iterator terminates because k >= max_k.
            break;
        }
    }

    // --- Finalize the basis matrix V_k ---
    // This logic is critical for both correctness and memory efficiency.
    // If the algorithm terminated early (due to breakdown or callback), the pre-allocated
    // `v_k` matrix is larger than the number of valid basis vectors. We must return a
    // matrix of the correct dimensions.
    let final_v_k = if steps_taken == k {
        // The algorithm completed all k steps. We can move ownership of the correctly-sized
        // matrix `v_k` directly to the output structure. This is a zero-cost operation
        // that avoids a potentially very expensive clone of a large matrix.
        v_k
    } else {
        // Termination was early. We must allocate a new, smaller matrix and copy only the
        // valid columns. This slicing operation (`get`) creates a view, and `.to_owned()`
        // performs the allocation and copy.
        v_k.as_ref().get(.., 0..steps_taken).to_owned()
    };

    Ok(LanczosOutput {
        v_k: final_v_k,
        decomposition: LanczosDecomposition {
            alphas,
            betas,
            steps_taken,
            b_norm,
        },
    })
}
