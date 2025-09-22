//! Symmetric Lanczos algorithm implementations for matrix function computations.
//!
//! This crate implements the symmetric Lanczos process for computing f(A)b, where f is a
//! matrix function, A is a Hermitian linear operator, and b is a vector. The implementations
//! target large-scale sparse problems and provide two algorithmic variants with different
//! memory-computation trade-offs.
//!
//! Built on the [`faer`] linear algebra framework, the algorithms operate on matrix-free
//! linear operators ([`faer::matrix_free::LinOp`]) and do not require explicit matrix storage.
//!
//! ## Algorithms
//!
//! **Standard Lanczos** ([`lanczos`]): Generates and stores the full Krylov basis V_k.
//! Uses O(nk) memory but requires only k matrix-vector products. Suitable when memory
//! permits storing the n×k basis matrix.
//!
//! **Two-Pass Lanczos** ([`lanczos_two_pass`]): Memory-efficient variant requiring only O(n) storage.
//! Executes two phases:
//! - First pass: Computes tridiagonal matrix T_k coefficients without storing basis vectors
//! - Second pass: Reconstructs solution by regenerating basis vectors on demand
//!
//! The two-pass method doubles the matrix-vector products to 2k but reduces memory from O(nk) to O(n).
//!
//! ## Example Usage
//!
//! The following example demonstrates how to solve a simple symmetric linear system
//! $\mathbf{A}\mathbf{x} = \mathbf{b}$ using both the standard one-pass and the two-pass
//! Lanczos solvers. This problem is equivalent to computing the action of the matrix
//! function $f(\mathbf{A}) = \mathbf{A}^{-1}$ on the vector $\mathbf{b}$.
//!
//! The core of the example lies in the definition of the `linear_solver` closure,
//! which constructs the tridiagonal matrix T_k from the Lanczos coefficients
//! and solves the small linear system T_k * y = e_1 to obtain the projected solution.
//!
//! ```rust
//! use faer::{Mat, dyn_stack::{MemBuffer, MemStack}};
//! use lanczos_project::{lanczos, lanczos_two_pass};
//! use faer::matrix_free::LinOp;
//! use faer::prelude::Solve;
//!
//! // Create a simple symmetric matrix
//! let a = Mat::from_fn(4, 4, |i, j| {
//!     if i == j { 2.0 }
//!     else if (i as isize - j as isize).abs() == 1 { -1.0 }
//!     else { 0.0 }
//! });
//!
//! // Right-hand side vector
//! let b = Mat::from_fn(4, 1, |i, _| (i + 1) as f64);
//!
//! // Allocate workspace
//! let mut mem = MemBuffer::new(a.as_ref().apply_scratch(1, faer::Par::Seq));
//! let mut stack = MemStack::new(&mut mem);
//!
//! // Define solver for f(T_k) = T_k^(-1) (linear system)
//! let linear_solver = |alphas: &[f64], betas: &[f64]| -> Result<Mat<f64>, anyhow::Error> {
//!     let k = alphas.len();
//!     let mut t_k = Mat::zeros(k, k);
//!
//!     // Build tridiagonal matrix
//!     for i in 0..k {
//!         t_k[(i, i)] = alphas[i];
//!         if i < k - 1 {
//!             t_k[(i, i + 1)] = betas[i];
//!             t_k[(i + 1, i)] = betas[i];
//!         }
//!     }
//!
//!     // Solve T_k * y = e_1
//!     let mut e1 = Mat::zeros(k, 1);
//!     e1[(0, 0)] = 1.0;
//!     Ok(t_k.partial_piv_lu().solve(&e1))
//! };
//!
//! let num_steps = 3;
//! // Standard one-pass method
//! let x1 = lanczos(&a.as_ref(), b.as_ref(), num_steps, &mut stack, &linear_solver).unwrap();
//!
//! // Two-pass method (same result, less memory)
//! let x2 = lanczos_two_pass(&a.as_ref(), b.as_ref(), num_steps, &mut stack, linear_solver).unwrap();
//!
//! // Results should be nearly identical
//! assert!((x1.as_ref() - x2.as_ref()).norm_l2() < 1e-12);
//! ```
//!
//! ## Performance Characteristics
//!
//! The algorithms use optimized SIMD kernels from [`faer`] for vector operations and
//! stack-allocated workspaces ([`faer::dyn_stack::MemStack`]) to minimize allocation overhead.
//! Matrix-free design supports implicit operators where A is defined by its action rather
//! than explicit storage.

// Declare the modules that form the crate's API structure.
pub mod algorithms;
pub mod error;
pub mod solvers;
pub mod utils;

// Re-export the main API from solvers for convenient access.
// These are the primary functions that users should use.
pub use solvers::{lanczos, lanczos_two_pass};
