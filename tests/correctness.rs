//! Integration test suite to verify the mathematical correctness of the
//! Lanczos algorithms by comparing their output against an analytically
//! known ground truth.

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

// A tolerance for the relative error against the ground truth for non-polynomial functions.
const APPROX_TOLERANCE: f64 = 1e-3;
// A tighter tolerance for polynomial functions where the method should be nearly exact.
const EXACT_TOLERANCE: f64 = 1e-12;

/// Creates a test problem with a diagonal sparse matrix, for which f(A) is
/// easily computed, and a random vector b.
fn create_diagonal_problem(n: usize) -> (SparseColMat<usize, f64>, Mat<f64>, Vec<f64>) {
    let mut triplets = Vec::with_capacity(n);
    let mut eigs = Vec::with_capacity(n);

    // Create a diagonal matrix with well-spaced eigenvalues.
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

/// Macro to generate correctness tests for both one-pass and two-pass algorithms.
macro_rules! generate_correctness_test {
    ($test_name:ident, $solver_logic:expr, $f:expr, $tolerance:expr, $error_msg_prefix:expr) => {
        #[test]
        fn $test_name() -> Result<()> {
            let n = 100;
            let k = 30;
            let (a, b, eigs) = create_diagonal_problem(n);

            // Compute ground truth solution
            let mut x_true = Mat::zeros(n, 1);
            for i in 0..n {
                x_true.as_mut()[(i, 0)] = $f(eigs[i]) * b.as_ref()[(i, 0)];
            }

            let mut mem = MemBuffer::new(a.as_ref().apply_scratch(1, Par::Seq));
            let mut stack = MemStack::new(&mut mem);

            // Run the specified Lanczos solver
            let x_k = $solver_logic(&a.as_ref(), b.as_ref(), k, &mut stack)?;
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

// --- Test Definitions using the macro ---

// Linear System Solve: f(z) = 1/z
generate_correctness_test!(
    test_linear_solve_standard,
    |a, b, k, stack| {
        let f_tk_solver = |alphas: &[f64], betas: &[f64]| -> Result<Mat<f64>, anyhow::Error> {
            let steps = alphas.len();
            let mut t_k = Mat::zeros(steps, steps);
            for i in 0..steps {
                t_k.as_mut()[(i, i)] = alphas[i];
            }
            for i in 0..steps - 1 {
                t_k.as_mut()[(i, i + 1)] = betas[i];
                t_k.as_mut()[(i + 1, i)] = betas[i];
            }
            let mut e1 = Mat::zeros(steps, 1);
            e1.as_mut()[(0, 0)] = 1.0;
            let y_k_prime = t_k.as_ref().partial_piv_lu().solve(&e1);
            Ok(y_k_prime)
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
            let steps = alphas.len();
            let mut t_k = Mat::zeros(steps, steps);
            for i in 0..steps {
                t_k.as_mut()[(i, i)] = alphas[i];
            }
            for i in 0..steps - 1 {
                t_k.as_mut()[(i, i + 1)] = betas[i];
                t_k.as_mut()[(i + 1, i)] = betas[i];
            }
            let mut e1 = Mat::zeros(steps, 1);
            e1.as_mut()[(0, 0)] = 1.0;
            let y_k_prime = t_k.as_ref().partial_piv_lu().solve(&e1);
            Ok(y_k_prime)
        };
        lanczos_two_pass(a, b, k, stack, f_tk_solver)
    },
    |z: f64| 1.0 / z,
    APPROX_TOLERANCE,
    "Two-pass linear solve"
);

// Matrix Exponential: f(z) = exp(z)
generate_correctness_test!(
    test_matrix_exp_standard,
    |a, b, k, stack| {
        let f_tk_solver = |alphas: &[f64], betas: &[f64]| -> Result<Mat<f64>, anyhow::Error> {
            let steps = alphas.len();
            let mut t_k = Mat::zeros(steps, steps);
            for i in 0..steps {
                t_k.as_mut()[(i, i)] = alphas[i];
            }
            for i in 0..steps - 1 {
                t_k.as_mut()[(i, i + 1)] = betas[i];
                t_k.as_mut()[(i + 1, i)] = betas[i];
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
                |i, j| if i == j { d_lambda[i].exp() } else { 0.0 },
            );
            let f_t_k = &q_tk * &f_d * q_tk.adjoint();
            let mut e1 = Mat::zeros(steps, 1);
            e1.as_mut()[(0, 0)] = 1.0;
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
            let steps = alphas.len();
            let mut t_k = Mat::zeros(steps, steps);
            for i in 0..steps {
                t_k.as_mut()[(i, i)] = alphas[i];
            }
            for i in 0..steps - 1 {
                t_k.as_mut()[(i, i + 1)] = betas[i];
                t_k.as_mut()[(i + 1, i)] = betas[i];
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
                |i, j| if i == j { d_lambda[i].exp() } else { 0.0 },
            );
            let f_t_k = &q_tk * &f_d * q_tk.adjoint();
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

// Matrix Square: f(z) = z^2
generate_correctness_test!(
    test_matrix_square_standard,
    |a, b, k, stack| {
        let f_tk_solver = |alphas: &[f64], betas: &[f64]| -> Result<Mat<f64>, anyhow::Error> {
            let steps = alphas.len();
            let mut t_k = Mat::zeros(steps, steps);
            for i in 0..steps {
                t_k.as_mut()[(i, i)] = alphas[i];
            }
            for i in 0..steps - 1 {
                t_k.as_mut()[(i, i + 1)] = betas[i];
                t_k.as_mut()[(i + 1, i)] = betas[i];
            }
            let f_t_k = &t_k * &t_k;
            let mut e1 = Mat::zeros(steps, 1);
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
            let steps = alphas.len();
            let mut t_k = Mat::zeros(steps, steps);
            for i in 0..steps {
                t_k.as_mut()[(i, i)] = alphas[i];
            }
            for i in 0..steps - 1 {
                t_k.as_mut()[(i, i + 1)] = betas[i];
                t_k.as_mut()[(i + 1, i)] = betas[i];
            }
            let f_t_k = &t_k * &t_k;
            let mut e1 = Mat::zeros(steps, 1);
            e1.as_mut()[(0, 0)] = 1.0;
            Ok(&f_t_k * &e1)
        };
        lanczos_two_pass(a, b, k, stack, f_tk_solver)
    },
    |z: f64| z.powi(2),
    EXACT_TOLERANCE,
    "Two-pass matrix square"
);
