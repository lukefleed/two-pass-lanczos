//! Integration test suite to verify the mathematical correctness of the Lanczos algorithms.
//!
//! # Test Methodology
//!
//! The core principle of this test suite is to validate the Lanczos approximations by comparing
//! them against a ground-truth solution that can be computed analytically. This is a standard
//! validation technique in numerical analysis for iterative methods.
//!
//! The methodology consists of the following steps:
//! 1.  **Construct a Test Problem `(A, b)`:** A symmetric matrix `A` is constructed such that
//!     the action of a function `f(A)` can be easily and accurately computed. We use a
//!     diagonal matrix, for which `f(A)` is simply the function `f` applied to its
//!     diagonal entries.
//! 2.  **Compute the Ground Truth:** The exact solution vector, `x_true = f(A)b`, is computed
//!     analytically using the known spectral properties of `A`.
//! 3.  **Compute the Lanczos Approximation:** The implemented Lanczos algorithm (either
//!     one-pass or two-pass) is executed for `k` iterations to produce an approximate
//!     solution, `x_k`.
//! 4.  **Verify Accuracy:** The relative error `||x_k - x_true|| / ||x_true||` is computed
//!     and asserted to be within a predefined tolerance.
//!
//! This process is repeated for a set of carefully chosen functions `f` to ensure the
//! algorithm's robustness and correctness across different mathematical domains.

use anyhow::{Result, anyhow, ensure};
use faer::{
    Par, Side,
    dyn_stack::{MemBuffer, MemStack},
    matrix_free::LinOp,
    prelude::*,
    sparse::{SparseColMat, Triplet},
};
use lanczos_project::solvers::{lanczos, lanczos_two_pass};
use rand::{Rng, SeedableRng, rngs::StdRng};

/// A tolerance for the relative error against the ground truth for non-polynomial functions.
///
/// For general analytic functions, the Lanczos method provides an approximation whose accuracy
/// depends on how well a polynomial of degree `k-1` can approximate `f` on the spectrum of `A`.
/// This is fundamentally linked to the theory of Gauss quadrature, which is not exact in general.
/// Therefore, we expect a small but non-zero error.
const APPROX_TOLERANCE: f64 = 1e-3;

/// A tighter tolerance for polynomial functions where the method should be nearly exact.
///
/// If `f` is a polynomial of degree `d`, then the exact solution `f(A)b` lies within the
/// Krylov subspace `K_{d+1}(A, b)`. The Lanczos method, by construction, finds the optimal
/// approximation within `K_k(A, b)`. Thus, for `k > d`, the algorithm should find the exact
/// solution up to machine precision. This test uses `k=30` for a degree-2 polynomial, so
/// we expect a very high-precision result.
const EXACT_TOLERANCE: f64 = 1e-12;

/// Assembles a dense `faer::Mat` from the Lanczos coefficients.
///
/// The symmetric Lanczos process generates two sequences of scalars, `alphas` (α_j) and
/// `betas` (β_j), which define the `k x k` real symmetric tridiagonal matrix `T_k`:
///
///     T_k = | α_1 β_1  0  ... |
///           | β_1 α_2 β_2 ... |
///           |  0  β_2 α_3 ... |
///           | ... ... ... ... |
///
/// This helper function constructs the explicit dense representation of `T_k`, which is
/// required for computing `f(T_k)`.
fn assemble_tridiagonal(alphas: &[f64], betas: &[f64]) -> Mat<f64> {
    let steps = alphas.len();
    if steps == 0 {
        return Mat::zeros(0, 0);
    }
    let mut t_k = Mat::zeros(steps, steps);
    for (i, &alpha) in alphas.iter().enumerate() {
        t_k.as_mut()[(i, i)] = alpha;
    }
    for (i, &beta) in betas.iter().enumerate() {
        t_k.as_mut()[(i, i + 1)] = beta;
        t_k.as_mut()[(i + 1, i)] = beta;
    }
    t_k
}

/// Creates a test problem with a diagonal sparse matrix and a random vector `b`.
///
/// A diagonal matrix `A = diag(λ_1, ..., λ_n)` is chosen because its spectral properties are
/// trivial. The action of any function `f` on `A` is defined as `f(A) = diag(f(λ_1), ..., f(λ_n))`.
/// This allows for the direct and highly accurate computation of the ground-truth solution vector
/// `x_true = f(A)b`, as each component is simply `(x_true)_i = f(λ_i) * b_i`.
///
/// A random starting vector `b` is used to ensure that its projection onto the eigenspaces
/// of `A` is non-trivial for many eigenvalues. This prevents premature breakdown of the Lanczos
/// iteration and ensures the Krylov subspace is sufficiently rich for a meaningful test.
/// Using a fixed seed for the random number generator makes the tests deterministic.
fn create_diagonal_problem(n: usize) -> (SparseColMat<usize, f64>, Mat<f64>, Vec<f64>) {
    let mut triplets = Vec::with_capacity(n);
    let mut eigs = Vec::with_capacity(n);

    // Create a diagonal matrix with well-spaced, positive eigenvalues.
    for i in 0..n {
        let val = (i + 1) as f64;
        triplets.push(Triplet {
            row: i,
            col: i,
            val,
        });
        eigs.push(val);
    }
    let a = SparseColMat::try_new_from_triplets(n, n, &triplets).unwrap();

    // Create a reproducible random vector.
    let mut rng = StdRng::seed_from_u64(42);
    let b = Mat::from_fn(n, 1, |_, _| rng.random());

    (a, b, eigs)
}

/// A macro to generate the boilerplate for each correctness test.
///
/// This macro abstracts the common test structure:
/// 1. Set up the problem dimensions `n` and `k`.
/// 2. Call `create_diagonal_problem` to get `A`, `b`, and the eigenvalues.
/// 3. Compute the analytical ground-truth solution `x_true`.
/// 4. Execute the specified Lanczos solver (`$solver_logic`).
/// 5. Calculate the relative error of the approximation and assert it is within tolerance.
macro_rules! generate_correctness_test {
    ($test_name:ident, $solver_logic:expr, $f:expr, $tolerance:expr, $error_msg_prefix:expr) => {
        #[test]
        fn $test_name() -> Result<()> {
            let n = 100;
            let k = 30;
            let (a, b, eigs) = create_diagonal_problem(n);

            // Analytically compute the ground truth solution `x_true = f(A)b`.
            // For a diagonal matrix A, this simplifies to a component-wise product.
            let mut x_true = Mat::zeros(n, 1);
            for (i, &eig) in eigs.iter().enumerate() {
                x_true.as_mut()[(i, 0)] = $f(eig) * b.as_ref()[(i, 0)];
            }

            let mut mem = MemBuffer::new(a.as_ref().apply_scratch(1, Par::Seq));
            let mut stack = MemStack::new(&mut mem);

            // Run the specified Lanczos solver to get the approximate solution x_k.
            let x_k = $solver_logic(&a.as_ref(), b.as_ref(), k, &mut stack)?;

            // Compute the relative error, a standard metric for vector approximation accuracy.
            let rel_err = (&x_k - &x_true).norm_l2() / x_true.norm_l2();

            ensure!(
                rel_err < $tolerance,
                "{} error too high: {}",
                $error_msg_prefix,
                rel_err
            );

            Ok(())
        }
    };
}

// --- Test Suite ---
// The following tests validate the Lanczos implementations for three distinct functions,
// each chosen to probe a different mathematical property of the algorithm.

// Test Case 1: Linear System Solve via f(z) = 1/z.
// This is the most fundamental application of Krylov subspace methods.
generate_correctness_test!(
    test_linear_solve_standard,
    |a, b, k, stack| {
        // The solver for the projected problem computes y'_k = T_k^{-1} * e_1.
        // We use a general-purpose LU decomposition with partial pivoting, which is a
        // numerically stable method for solving dense linear systems.
        let f_tk_solver = |alphas: &[f64], betas: &[f64]| -> Result<Mat<f64>, anyhow::Error> {
            let t_k = assemble_tridiagonal(alphas, betas);
            if t_k.nrows() == 0 {
                return Ok(Mat::zeros(0, 1));
            }
            let mut e1 = Mat::zeros(t_k.nrows(), 1);
            e1.as_mut()[(0, 0)] = 1.0;
            Ok(t_k.as_ref().partial_piv_lu().solve(&e1))
        };
        lanczos(a, b, k, stack, f_tk_solver)
    },
    |z: f64| 1.0 / z,
    APPROX_TOLERANCE,
    "One-pass linear solve"
);

generate_correctness_test!(
    test_linear_solve_two_pass,
    |a, b, k, stack| {
        let f_tk_solver = |alphas: &[f64], betas: &[f64]| -> Result<Mat<f64>, anyhow::Error> {
            let t_k = assemble_tridiagonal(alphas, betas);
            if t_k.nrows() == 0 {
                return Ok(Mat::zeros(0, 1));
            }
            let mut e1 = Mat::zeros(t_k.nrows(), 1);
            e1.as_mut()[(0, 0)] = 1.0;
            Ok(t_k.as_ref().partial_piv_lu().solve(&e1))
        };
        lanczos_two_pass(a, b, k, stack, f_tk_solver)
    },
    |z: f64| 1.0 / z,
    APPROX_TOLERANCE,
    "Two-pass linear solve"
);

// Test Case 2: Matrix Exponential via f(z) = exp(z).
// This validates the algorithm for a transcendental function, common in the solution of ODEs.
generate_correctness_test!(
    test_matrix_exp_standard,
    |a, b, k, stack| {
        // The solver for the projected problem computes y'_k = exp(T_k) * e_1.
        // Since T_k is symmetric, exp(T_k) can be stably computed via its spectral
        // decomposition: exp(T_k) = Q * exp(D) * Q^T.
        let f_tk_solver = |alphas: &[f64], betas: &[f64]| -> Result<Mat<f64>, anyhow::Error> {
            let t_k = assemble_tridiagonal(alphas, betas);
            let steps = t_k.nrows();
            if steps == 0 {
                return Ok(Mat::zeros(0, 1));
            }
            // 1. Compute the eigendecomposition T_k = Q * D * Q^T.
            let evd = t_k
                .as_ref()
                .self_adjoint_eigen(Side::Upper)
                .map_err(|e| anyhow!("EVD failed: {:?}", e))?;
            let q_tk = evd.U();
            let d_lambda = evd.S();
            // 2. Compute exp(D) by applying exp() to the eigenvalues.
            let f_d = Mat::from_fn(
                steps,
                steps,
                |i, j| {
                    if i == j { d_lambda[i].exp() } else { 0.0 }
                },
            );
            // 3. Reconstruct exp(T_k) = Q * exp(D) * Q^T.
            let f_t_k = q_tk * &f_d * q_tk.adjoint();
            let mut e1 = Mat::zeros(steps, 1);
            e1.as_mut()[(0, 0)] = 1.0;
            // 4. Compute the final result vector.
            Ok(&f_t_k * &e1)
        };
        lanczos(a, b, k, stack, f_tk_solver)
    },
    |z: f64| z.exp(),
    APPROX_TOLERANCE,
    "One-pass matrix exponential"
);

generate_correctness_test!(
    test_matrix_exp_two_pass,
    |a, b, k, stack| {
        let f_tk_solver = |alphas: &[f64], betas: &[f64]| -> Result<Mat<f64>, anyhow::Error> {
            let t_k = assemble_tridiagonal(alphas, betas);
            let steps = t_k.nrows();
            if steps == 0 {
                return Ok(Mat::zeros(0, 1));
            }
            let evd = t_k
                .as_ref()
                .self_adjoint_eigen(Side::Upper)
                .map_err(|e| anyhow!("EVD failed: {:?}", e))?;
            let q_tk = evd.U();
            let d_lambda = evd.S();
            let f_d = Mat::from_fn(
                steps,
                steps,
                |i, j| {
                    if i == j { d_lambda[i].exp() } else { 0.0 }
                },
            );
            let f_t_k = q_tk * &f_d * q_tk.adjoint();
            let mut e1 = Mat::zeros(steps, 1);
            e1.as_mut()[(0, 0)] = 1.0;
            Ok(&f_t_k * &e1)
        };
        lanczos_two_pass(a, b, k, stack, f_tk_solver)
    },
    |z: f64| z.exp(),
    APPROX_TOLERANCE,
    "Two-pass matrix exponential"
);

// Test Case 3: Matrix Square via f(z) = z^2.
// This validates the algorithm for a polynomial, for which the result should be nearly exact.
generate_correctness_test!(
    test_matrix_square_standard,
    |a, b, k, stack| {
        // The solver for the projected problem computes y'_k = T_k^2 * e_1.
        // This can be computed directly by matrix multiplication.
        let f_tk_solver = |alphas: &[f64], betas: &[f64]| -> Result<Mat<f64>, anyhow::Error> {
            let t_k = assemble_tridiagonal(alphas, betas);
            if t_k.nrows() == 0 {
                return Ok(Mat::zeros(0, 1));
            }
            let f_t_k = &t_k * &t_k;
            let mut e1 = Mat::zeros(t_k.nrows(), 1);
            e1.as_mut()[(0, 0)] = 1.0;
            Ok(&f_t_k * &e1)
        };
        lanczos(a, b, k, stack, f_tk_solver)
    },
    |z: f64| z.powi(2),
    EXACT_TOLERANCE,
    "One-pass matrix square"
);

generate_correctness_test!(
    test_matrix_square_two_pass,
    |a, b, k, stack| {
        let f_tk_solver = |alphas: &[f64], betas: &[f64]| -> Result<Mat<f64>, anyhow::Error> {
            let t_k = assemble_tridiagonal(alphas, betas);
            if t_k.nrows() == 0 {
                return Ok(Mat::zeros(0, 1));
            }
            let f_t_k = &t_k * &t_k;
            let mut e1 = Mat::zeros(t_k.nrows(), 1);
            e1.as_mut()[(0, 0)] = 1.0;
            Ok(&f_t_k * &e1)
        };
        lanczos_two_pass(a, b, k, stack, f_tk_solver)
    },
    |z: f64| z.powi(2),
    EXACT_TOLERANCE,
    "Two-pass matrix square"
);

// --- Solution Equivalence Test ---
// Verifies that the one-pass and two-pass algorithms produce identical solution vectors,
// confirming that the memory-efficient two-pass variant is mathematically equivalent.

#[test]
fn test_one_pass_two_pass_solution_equivalence() -> Result<()> {
    let n = 100;
    let k = 30;
    let (a, b, _eigs) = create_diagonal_problem(n);

    let f_tk_solver = |alphas: &[f64], betas: &[f64]| -> Result<Mat<f64>, anyhow::Error> {
        let t_k = assemble_tridiagonal(alphas, betas);
        if t_k.nrows() == 0 {
            return Ok(Mat::zeros(0, 1));
        }
        let mut e1 = Mat::zeros(t_k.nrows(), 1);
        e1.as_mut()[(0, 0)] = 1.0;
        Ok(t_k.as_ref().partial_piv_lu().solve(&e1))
    };

    let mut mem1 = MemBuffer::new(a.as_ref().apply_scratch(1, Par::Seq));
    let mut stack1 = MemStack::new(&mut mem1);
    let x1 = lanczos(&a.as_ref(), b.as_ref(), k, &mut stack1, &f_tk_solver)?;

    let mut mem2 = MemBuffer::new(a.as_ref().apply_scratch(1, Par::Seq));
    let mut stack2 = MemStack::new(&mut mem2);
    let x2 = lanczos_two_pass(&a.as_ref(), b.as_ref(), k, &mut stack2, f_tk_solver)?;

    let diff = (&x1 - &x2).norm_l2();
    ensure!(diff < 1e-12, "One-pass vs two-pass differ by {}", diff);
    Ok(())
}

// --- Regression Test ---
// This test locks down the exact Lanczos tridiagonal coefficients for a fixed problem,
// ensuring that any refactoring that alters floating-point summation order is detected.

#[test]
fn test_golden_value_lanczos_coefficients() -> Result<()> {
    use lanczos_project::algorithms::lanczos::lanczos_standard;

    let n = 100;
    let k = 10;
    let (a, b, _eigs) = create_diagonal_problem(n);
    let mut mem = MemBuffer::new(a.as_ref().apply_scratch(1, Par::Seq));
    let mut stack = MemStack::new(&mut mem);

    let output = lanczos_standard(&a.as_ref(), b.as_ref(), k, &mut stack, None)?;

    ensure!(
        output.decomposition.steps_taken == k,
        "Expected {} steps, got {}",
        k,
        output.decomposition.steps_taken
    );

    // Golden values captured from a known-good run. Updated after fusing the
    // beta-subtraction and dot-product passes (4 sweeps -> 3), which changed
    // the FP accumulation order for alpha/beta values at ULP level.
    let golden_alphas: [f64; 5] = [
        52.97782051430146311,
        51.63983040769662125,
        47.56379230698905758,
        52.26404007970732124,
        49.98428350718381097,
    ];
    let golden_betas: [f64; 5] = [
        28.97606231351560879,
        26.11302677401963379,
        24.96008410770487984,
        24.57230990302882390,
        25.27009528124148119,
    ];

    let tol = 1e-14;
    for (i, (&actual, &expected)) in output
        .decomposition
        .alphas
        .iter()
        .zip(golden_alphas.iter())
        .enumerate()
    {
        ensure!(
            (actual - expected).abs() < tol,
            "Alpha mismatch at index {}: got {}, expected {} (diff = {})",
            i,
            actual,
            expected,
            (actual - expected).abs()
        );
    }

    for (i, (&actual, &expected)) in output
        .decomposition
        .betas
        .iter()
        .zip(golden_betas.iter())
        .enumerate()
    {
        ensure!(
            (actual - expected).abs() < tol,
            "Beta mismatch at index {}: got {}, expected {} (diff = {})",
            i,
            actual,
            expected,
            (actual - expected).abs()
        );
    }

    Ok(())
}
