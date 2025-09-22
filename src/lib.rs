//! A high-performance Rust implementation of the symmetric Lanczos algorithm.
//!
//! This crate provides implementations of the symmetric
//! Lanczos process for computing the action of a matrix function on a vector, an
//! operation denoted as $\mathbf{x} = f(\mathbf{A})\mathbf{b}$. It is designed for
//! large-scale, sparse Hermitian linear operators and offers two primary algorithmic
//! variants to address different memory and computational constraints.
//!
//! The core of the library is built upon the [`faer`] linear algebra framework, leveraging
//! its matrix-free ([`faer::matrix_free::LinOp`]) capabilities to operate on linear operators that may not
//! have an explicit matrix representation.
//!
//! ## Core Features
//!
//! The library exposes two primary solvers, each tailored to a specific use case:
//!
//! 1.  **Standard One-Pass Lanczos**: A direct implementation that generates and stores
//!     the full orthonormal basis for the Krylov subspace. This method is computationally
//!     optimal in terms of matrix-vector products but requires $O(nk)$ memory, where
//!     $n$ is the problem dimension and $k$ is the number of iterations. It is suitable
//!     for problems where memory is not a limiting factor.
//!
//! 2.  **Two-Pass Lanczos**: A memory-efficient variant designed for large-scale problems
//!     where storing the full basis is infeasible. This algorithm executes two passes:
//!     - The **first pass** computes the scalar coefficients of the projected tridiagonal
//!       matrix $\mathbf{T}_k$ without storing the basis vectors, operating with a minimal
//!       $O(n)$ memory footprint.
//!     - The **second pass** reconstructs the final solution by regenerating the basis
//!       vectors on-the-fly, using the coefficients from the first pass.
//!
//! This two-pass strategy reduces the memory requirement to $O(n)$ at the cost of
//! doubling the number of matrix-vector products to $2k$.
//!
//! ## Design and Architecture
//!
//! The implementation is structured to be both performant and extensible:
//!
//! - **Matrix-Free Operations**: By building on the [`faer::matrix_free::LinOp`] trait,
//!   the algorithms are generic over any linear operator, enabling applications where
//!   the matrix $\mathbf{A}$ is defined implicitly.
//!
//! - **High-Performance Kernels**: Vector operations leverage the optimized, SIMD-enabled
//!   kernels provided by [`faer`], ensuring high computational throughput.
//!
//! - **Efficient Memory Management**: Temporary workspaces are allocated on the stack
//!   using [`faer::dyn_stack::MemStack`], minimizing heap allocation overhead within the
//!   main iterative loops, a critical factor for performance in numerical algorithms.

// Declare the modules that form the public API of the crate.
pub mod algorithms;
pub mod error;
pub mod solvers;
pub mod utils;

// Re-export key types to the top level of the crate for easier access.
pub use solvers::{lanczos, lanczos_two_pass};
