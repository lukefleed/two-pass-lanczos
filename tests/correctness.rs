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
use lanczos_project::algorithms::lanczos::{lanczos_pass_two, lanczos_standard};
use rand::{Rng, SeedableRng, rngs::StdRng};

// A tolerance for the relative error against the ground truth.
// This is higher than the property test tolerance because it accounts for
// approximation errors inherent to the Krylov method for a fixed number of
// iterations, not just floating-point drift. A value of k=30 for an n=100
// problem is not expected to yield results accurate to machine precision
// for non-polynomial functions.
const TOLERANCE: f64 = 1e-3;

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

#[test]
fn test_linear_system_solve() -> Result<()> {
    let n = 100;
    let k = 30; // Number of Lanczos iterations.
    let (a, b, eigs) = create_diagonal_problem(n);

    // Compute ground truth: x_true = A^-1 * b
    let mut x_true = Mat::zeros(n, 1);
    for i in 0..n {
        x_true.as_mut()[(i, 0)] = b.as_ref()[(i, 0)] / eigs[i];
    }

    let mut mem = MemBuffer::new(a.as_ref().apply_scratch(1, Par::Seq));
    let mut stack = MemStack::new(&mut mem);

    // Run one-pass Lanczos.
    let standard_output = lanczos_standard(&a.as_ref(), b.as_ref(), k, &mut stack, None)?;
    let v_k = standard_output.v_k;
    let decomp = standard_output.decomposition;

    // Build T_k matrix from decomposition.
    let t_k_mat = {
        let mut t = Mat::zeros(k, k);
        for i in 0..k {
            t.as_mut()[(i, i)] = decomp.alphas[i];
        }
        for i in 0..k - 1 {
            t.as_mut()[(i, i + 1)] = decomp.betas[i];
            t.as_mut()[(i + 1, i)] = decomp.betas[i];
        }
        t
    };

    // For f(A) = A^-1, the coefficient vector y_k is the solution to T_k*y = ||b||*e_1.
    let mut rhs = Mat::zeros(k, 1);
    rhs.as_mut()[(0, 0)] = decomp.b_norm;
    let y_k = t_k_mat.as_ref().partial_piv_lu().solve(&rhs);

    // Reconstruct solution x_k = V_k * y_k.
    let x_k_std = &v_k * &y_k;

    // Run two-pass Lanczos.
    let x_k_2pass = lanczos_pass_two(&a.as_ref(), b.as_ref(), &decomp, y_k.as_ref(), &mut stack)?;

    let err_std = (&x_k_std - &x_true).norm_l2() / x_true.norm_l2();
    let err_2pass = (&x_k_2pass - &x_true).norm_l2() / x_true.norm_l2();

    ensure!(
        err_std < TOLERANCE,
        "One-pass linear solve error too high: {}",
        err_std
    );
    ensure!(
        err_2pass < TOLERANCE,
        "Two-pass linear solve error too high: {}",
        err_2pass
    );

    Ok(())
}

#[test]
fn test_matrix_exponential() -> Result<()> {
    let n = 100;
    let k = 30;
    let (a, b, eigs) = create_diagonal_problem(n);
    let f = |z: f64| z.exp();

    // Compute ground truth: x_true = exp(A) * b
    let mut x_true = Mat::zeros(n, 1);
    for i in 0..n {
        x_true.as_mut()[(i, 0)] = f(eigs[i]) * b.as_ref()[(i, 0)];
    }

    let mut mem = MemBuffer::new(a.as_ref().apply_scratch(1, Par::Seq));
    let mut stack = MemStack::new(&mut mem);

    let standard_output = lanczos_standard(&a.as_ref(), b.as_ref(), k, &mut stack, None)?;
    let v_k = standard_output.v_k;
    let decomp = standard_output.decomposition;

    let t_k_mat = {
        let mut t = Mat::zeros(k, k);
        for i in 0..k {
            t.as_mut()[(i, i)] = decomp.alphas[i];
        }
        for i in 0..k - 1 {
            t.as_mut()[(i, i + 1)] = decomp.betas[i];
            t.as_mut()[(i + 1, i)] = decomp.betas[i];
        }
        t
    };

    // Compute y_k = f(T_k) * ||b|| * e_1 via eigendecomposition of T_k.
    let evd = t_k_mat
        .as_ref()
        .self_adjoint_eigen(Side::Upper)
        .map_err(|e| anyhow!("EVD failed: {:?}", e))?;
    let q_tk = evd.U();
    let d_lambda = evd.S();
    let f_d = Mat::from_fn(k, k, |i, j| if i == j { f(d_lambda[i]) } else { 0.0 });
    let f_t_k = &q_tk * &f_d * q_tk.adjoint();

    let mut e1 = Mat::zeros(k, 1);
    e1.as_mut()[(0, 0)] = 1.0;
    let y_k = &f_t_k * &e1 * Scale(decomp.b_norm);

    let x_k_std = &v_k * &y_k;
    let x_k_2pass = lanczos_pass_two(&a.as_ref(), b.as_ref(), &decomp, y_k.as_ref(), &mut stack)?;

    let err_std = (&x_k_std - &x_true).norm_l2() / x_true.norm_l2();
    let err_2pass = (&x_k_2pass - &x_true).norm_l2() / x_true.norm_l2();

    ensure!(
        err_std < TOLERANCE,
        "One-pass exponential error too high: {}",
        err_std
    );
    ensure!(
        err_2pass < TOLERANCE,
        "Two-pass exponential error too high: {}",
        err_2pass
    );

    Ok(())
}

#[test]
fn test_matrix_square() -> Result<()> {
    let n = 100;
    let k = 30;
    let (a, b, eigs) = create_diagonal_problem(n);
    let f = |z: f64| z.powi(2);

    // Compute ground truth: x_true = A^2 * b
    let mut x_true = Mat::zeros(n, 1);
    for i in 0..n {
        x_true.as_mut()[(i, 0)] = f(eigs[i]) * b.as_ref()[(i, 0)];
    }

    let mut mem = MemBuffer::new(a.as_ref().apply_scratch(1, Par::Seq));
    let mut stack = MemStack::new(&mut mem);

    let standard_output = lanczos_standard(&a.as_ref(), b.as_ref(), k, &mut stack, None)?;
    let v_k = standard_output.v_k;
    let decomp = standard_output.decomposition;

    let t_k_mat = {
        let mut t = Mat::zeros(k, k);
        for i in 0..k {
            t.as_mut()[(i, i)] = decomp.alphas[i];
        }
        for i in 0..k - 1 {
            t.as_mut()[(i, i + 1)] = decomp.betas[i];
            t.as_mut()[(i + 1, i)] = decomp.betas[i];
        }
        t
    };

    // Compute y_k = f(T_k) * ||b|| * e_1 = T_k^2 * ||b|| * e_1
    let f_t_k = &t_k_mat * &t_k_mat;
    let mut e1 = Mat::zeros(k, 1);
    e1.as_mut()[(0, 0)] = 1.0;
    let y_k = &f_t_k * &e1 * Scale(decomp.b_norm);

    let x_k_std = &v_k * &y_k;
    let x_k_2pass = lanczos_pass_two(&a.as_ref(), b.as_ref(), &decomp, y_k.as_ref(), &mut stack)?;

    let err_std = (&x_k_std - &x_true).norm_l2() / x_true.norm_l2();
    let err_2pass = (&x_k_2pass - &x_true).norm_l2() / x_true.norm_l2();

    ensure!(
        err_std < TOLERANCE,
        "One-pass matrix square error too high: {}",
        err_std
    );
    ensure!(
        err_2pass < TOLERANCE,
        "Two-pass matrix square error too high: {}",
        err_2pass
    );

    Ok(())
}
