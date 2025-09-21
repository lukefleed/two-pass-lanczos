//! Experiment Runner for the Numerical Stability and Accuracy Analysis.
//!
//! This executable conducts an analysis of the Lanczos algorithms by
//! comparing their computed solutions against a known analytical "ground truth".
//! It evaluates both the standard one-pass and the memory-efficient two-pass
//! variants to demonstrate their numerical equivalence and accuracy across four
//! distinct problem scenarios.

use anyhow::{Result, anyhow};
use clap::{Parser, ValueEnum};
use faer::{
    Side,
    dyn_stack::{MemBuffer, MemStack},
    matrix_free::LinOp,
    prelude::*,
    sparse::{SparseColMat, SymbolicSparseColMat, Triplet},
};
use lanczos_project::solvers::{lanczos, lanczos_two_pass};
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::Serialize;
use std::path::PathBuf;

/// Type alias for matrix function f(z).
type MatrixFunctionType = Box<dyn Fn(f64) -> f64>;

/// Type alias for T_k solver function.
type TkSolverType = Box<dyn Fn(&[f64], &[f64]) -> Result<Mat<f64>>>;

/// The matrix function `f` to be approximated in the experiment.
#[derive(ValueEnum, Clone, Debug, Copy)]
enum MatrixFunction {
    /// Corresponds to solving a linear system, f(A)b = A^-1 b.
    Inv,
    /// Corresponds to computing the action of the matrix exponential, f(A)b = exp(A)b.
    Exp,
}

/// The spectral properties of the test matrix A.
#[derive(ValueEnum, Clone, Debug, Copy)]
enum ProblemScenario {
    /// A well-conditioned problem where eigenvalues are well-separated from any
    /// function singularities.
    WellConditioned,
    /// An ill-conditioned problem, either due to a wide spectrum or eigenvalues
    /// very close to a singularity.
    IllConditioned,
}

/// Command-line arguments for the numerical stability and accuracy experiment.
#[derive(Parser, Debug)]
#[clap(
    name = "accuracy-runner",
    about = "Runs a numerical accuracy analysis for Lanczos methods against a ground truth."
)]
struct AccuracyArgs {
    /// The matrix function to test.
    #[clap(long, value_enum)]
    function: MatrixFunction,

    /// The spectral scenario for the test problem.
    #[clap(long, value_enum)]
    scenario: ProblemScenario,

    /// Dimension of the test matrix.
    #[clap(long, default_value_t = 1000)]
    n: usize,

    /// Minimum number of Lanczos iterations (k) to test.
    #[clap(long, default_value_t = 5)]
    k_min: usize,

    /// Maximum number of Lanczos iterations (k) to test.
    #[clap(long, default_value_t = 200)]
    k_max: usize,

    /// Step size for iterating k.
    #[clap(long, default_value_t = 5)]
    k_step: usize,

    /// Path to the output CSV file where results will be written.
    #[clap(long, value_name = "PATH")]
    output: PathBuf,
}

/// Represents a single row of data for the accuracy analysis CSV.
#[derive(Debug, Serialize)]
struct AccuracyResult {
    /// The number of Lanczos iterations.
    k: usize,
    /// Relative error of the standard one-pass method vs. ground truth.
    relative_error_standard: f64,
    /// Relative error of the two-pass method vs. ground truth.
    relative_error_two_pass: f64,
    /// L2-norm of the difference between the two final solution vectors.
    solution_deviation_l2: f64,
}

/// Creates a synthetic sparse diagonal matrix with controlled eigenvalues.
fn create_diagonal_problem(
    n: usize,
    scenario: ProblemScenario,
    func: MatrixFunction,
) -> (SparseColMat<usize, f64>, Vec<f64>) {
    let mut triplets = Vec::with_capacity(n);
    let mut eigs = Vec::with_capacity(n);

    match (func, scenario) {
        (MatrixFunction::Exp, ProblemScenario::WellConditioned) => {
            // Eigenvalues on a compact interval, well-suited for exp(z).
            // Values are in [-10, -0.1] to avoid large exponentials.
            for i in 0..n {
                let val = -10.0 + (9.9 / (n - 1).max(1) as f64) * i as f64;
                eigs.push(val);
            }
        }
        (MatrixFunction::Exp, ProblemScenario::IllConditioned) => {
            // A very wide spectrum, making polynomial approximation difficult for exp(z).
            // Values are in [-1000, -0.1].
            for i in 0..n {
                let val = -1000.0 + (999.9 / (n - 1).max(1) as f64) * i as f64;
                eigs.push(val);
            }
        }
        (MatrixFunction::Inv, ProblemScenario::WellConditioned) => {
            // A positive definite spectrum, well-conditioned for inversion.
            // Values are in [0.1, 100].
            for i in 0..n {
                let val = 0.1 + (99.9 / (n - 1).max(1) as f64) * i as f64;
                eigs.push(val);
            }
        }
        (MatrixFunction::Inv, ProblemScenario::IllConditioned) => {
            // An indefinite, quasi-singular spectrum, ill-conditioned for inversion.
            // Values are in [-1, -0.1] U [0.1, 1], with one eigenvalue very close to zero.
            let mid = n / 2;
            for i in 0..n {
                let val = if i < mid {
                    0.1 + (0.9 / (mid - 1).max(1) as f64) * i as f64
                } else {
                    -1.0 + (0.9 / (n - mid - 1).max(1) as f64) * (i - mid) as f64
                };
                eigs.push(val);
            }
            eigs[mid] = 1e-8; // The critical eigenvalue
        }
    }

    for (i, &eig) in eigs.iter().enumerate() {
        triplets.push(Triplet {
            row: i,
            col: i,
            val: eig,
        });
    }

    let a = SparseColMat::try_new_from_triplets(n, n, &triplets).unwrap();
    (a, eigs)
}

/// Solver for $f(\mathbf{T}_k)\mathbf{e}_1$ where $f(z) = z^{-1}$.
/// This solves the system $\mathbf{T}_k \mathbf{y}' = \mathbf{e}_1$. Since $\mathbf{T}_k$ is small and tridiagonal, a sparse LU decomposition is extremely efficient ($O(k)$) and numerically stable.
fn inv_tk_solver(alphas: &[f64], betas: &[f64]) -> Result<Mat<f64>> {
    let steps = alphas.len();
    if steps == 0 {
        return Ok(Mat::zeros(0, 1));
    }
    let t_k_sparse = assemble_tridiagonal_sparse(alphas, betas)?;
    let mut e1 = Mat::zeros(steps, 1);
    e1.as_mut()[(0, 0)] = 1.0;
    Ok(t_k_sparse.as_ref().sp_lu()?.solve(e1.as_ref()))
}

/// Solver for $f(\mathbf{T}_k)\mathbf{e}_1$ where $f(z) = \exp(z)$.
/// Since $\mathbf{T}_k$ is symmetric, the matrix exponential is computed via
/// eigendecomposition: $\exp(\mathbf{T}_k) = \mathbf{Q} \exp(\mathbf{D}) \mathbf{Q}^T$.
fn exp_tk_solver(alphas: &[f64], betas: &[f64]) -> Result<Mat<f64>> {
    let steps = alphas.len();
    if steps == 0 {
        return Ok(Mat::zeros(0, 1));
    }
    let t_k = assemble_tridiagonal_dense(alphas, betas);
    let evd = t_k.as_ref().self_adjoint_eigen(Side::Upper).unwrap();
    let q_tk = evd.U();
    let d_lambda = evd.S();
    let f_d = Mat::from_fn(
        steps,
        steps,
        |i, j| if i == j { d_lambda[i].exp() } else { 0.0 },
    );
    let f_t_k = q_tk * &f_d * q_tk.adjoint();
    let mut e1 = Mat::zeros(steps, 1);
    e1.as_mut()[(0, 0)] = 1.0;
    Ok(&f_t_k * &e1)
}

/// Helper to assemble a sparse `faer::SparseColMat` from Lanczos coefficients.
fn assemble_tridiagonal_sparse(alphas: &[f64], betas: &[f64]) -> Result<SparseColMat<usize, f64>> {
    let steps = alphas.len();
    if steps == 0 {
        let symbolic = SymbolicSparseColMat::<usize>::new_checked(0, 0, vec![0], None, vec![]);
        return Ok(SparseColMat::new(symbolic, vec![]));
    }
    let mut triplets = Vec::with_capacity(3 * steps - 2);
    for (i, &alpha) in alphas.iter().enumerate() {
        triplets.push(Triplet {
            row: i,
            col: i,
            val: alpha,
        });
    }
    for (i, &beta) in betas.iter().enumerate() {
        triplets.push(Triplet {
            row: i,
            col: i + 1,
            val: beta,
        });
        triplets.push(Triplet {
            row: i + 1,
            col: i,
            val: beta,
        });
    }
    SparseColMat::try_new_from_triplets(steps, steps, &triplets)
        .map_err(|e| anyhow!("Failed to construct sparse T_k: {:?}", e))
}

/// Helper to assemble a dense `faer::Mat` from Lanczos coefficients.
fn assemble_tridiagonal_dense(alphas: &[f64], betas: &[f64]) -> Mat<f64> {
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

/// The main entry point for the accuracy experiment.
fn main() -> Result<()> {
    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Info)
        .try_init()?;
    let args = AccuracyArgs::parse();
    log::info!(
        "Starting numerical accuracy analysis for function: {:?}, scenario: {:?}",
        args.function,
        args.scenario
    );

    // 1. Setup the test problem and ground truth.
    let (a, eigs) = create_diagonal_problem(args.n, args.scenario, args.function);
    let mut rng = StdRng::seed_from_u64(42);
    let b = Mat::from_fn(args.n, 1, |_, _| rng.random());

    let (f, f_tk_solver): (MatrixFunctionType, TkSolverType) = match args.function {
        MatrixFunction::Exp => (Box::new(|z| z.exp()), Box::new(exp_tk_solver)),
        MatrixFunction::Inv => (Box::new(|z| 1.0 / z), Box::new(inv_tk_solver)),
    };

    // Compute the analytical ground-truth solution x_true = f(A)b.
    // Since A is diagonal with eigenvalues on the diagonal, f(A) is also diagonal
    // with f(eigenvalue_i) on the diagonal. For a diagonal matrix, f(A)b simply
    // scales each component b_i by f(eigenvalue_i), giving us the exact solution
    // without any approximation error.
    let x_true = Mat::from_fn(args.n, 1, |i, _| f(eigs[i]) * b.as_ref()[(i, 0)]);
    let x_true_norm = x_true.norm_l2();

    let mut results = Vec::new();
    let mut stack_mem = MemBuffer::new(a.as_ref().apply_scratch(1, faer::Par::Seq));

    // 2. Iterate through k and run both algorithms.
    for k in (args.k_min..=args.k_max).step_by(args.k_step) {
        if k == 0 {
            continue;
        }
        log::info!("Running for k = {}...", k);
        // A fresh memory stack is used for each value of k to ensure that allocations
        // from one iteration do not interfere with the next
        let stack = MemStack::new(&mut stack_mem);

        let x_k_standard = match lanczos(&a.as_ref(), b.as_ref(), k, stack, &*f_tk_solver) {
            Ok(x) => x,
            Err(_) => {
                log::warn!("Standard Lanczos failed at k={}. Stopping.", k);
                break;
            }
        };

        let x_k_two_pass = match lanczos_two_pass(&a.as_ref(), b.as_ref(), k, stack, &*f_tk_solver)
        {
            Ok(x) => x,
            Err(_) => {
                log::warn!("Two-pass Lanczos failed at k={}. Stopping.", k);
                break;
            }
        };

        // 3. Compute and store metrics.
        results.push(AccuracyResult {
            k,
            relative_error_standard: (&x_k_standard - &x_true).norm_l2() / x_true_norm,
            relative_error_two_pass: (&x_k_two_pass - &x_true).norm_l2() / x_true_norm,
            solution_deviation_l2: (&x_k_standard - &x_k_two_pass).norm_l2(),
        });
    }

    // 4. Write results to CSV.
    log::info!("Writing results to {:?}...", &args.output);
    let mut writer = csv::Writer::from_path(&args.output)?;
    for record in results {
        writer.serialize(record)?;
    }
    writer.flush()?;

    log::info!("Accuracy analysis complete.");
    Ok(())
}
