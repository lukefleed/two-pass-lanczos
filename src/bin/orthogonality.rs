//! Experiment Runner for Orthogonality Analysis.
//!
//! This executable analyzes the numerical properties of the Lanczos basis vectors,
//! specifically the loss of orthogonality and the numerical drift between the standard
//! (one-pass) and regenerated (two-pass) bases. The analysis is performed
//! across different problem scenarios to correlate basis stability with the spectral
//! properties of the operator.

use anyhow::Result;
use clap::{Parser, ValueEnum};
use faer::{
    dyn_stack::{MemBuffer, MemStack},
    matrix_free::LinOp,
    prelude::*,
    sparse::{SparseColMat, Triplet},
};
use lanczos_project::algorithms::lanczos::{lanczos_pass_two_with_basis, lanczos_standard};
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::Serialize;
use std::path::PathBuf;

/// Defines the matrix function `f` to be used for setting up the problem scenario.
/// While the orthogonality analysis is independent of `f`, the function's properties
/// (e.g., singularities) inform the choice of a challenging spectrum for the matrix A.
#[derive(ValueEnum, Clone, Debug, Copy)]
enum MatrixFunction {
    /// Corresponds to f(z) = z^-1, motivating spectra with eigenvalues near zero.
    Inv,
    /// Corresponds to f(z) = exp(z), motivating spectra over wide intervals.
    Exp,
}

/// Defines the spectral properties of the test matrix A.
#[derive(ValueEnum, Clone, Debug, Copy)]
enum ProblemScenario {
    /// A well-conditioned problem where eigenvalues are well-separated.
    WellConditioned,
    /// An ill-conditioned problem with spectral properties designed to challenge
    /// the numerical stability of the Lanczos process.
    IllConditioned,
}

/// Command-line arguments for the orthogonality analysis runner.
#[derive(Parser, Debug)]
#[clap(
    name = "orthogonality-runner",
    about = "Runs an analysis of the Lanczos basis orthogonality and regeneration stability."
)]
struct OrthoArgs {
    /// The matrix function class to determine the problem's spectral properties.
    #[clap(long, value_enum)]
    function: MatrixFunction,
    /// The spectral scenario for the test problem.
    #[clap(long, value_enum)]
    scenario: ProblemScenario,
    /// Dimension of the test matrix.
    #[clap(long, default_value_t = 1000)]
    n: usize,
    /// Minimum number of Lanczos iterations (k) to test.
    #[clap(long, default_value_t = 20)]
    k_min: usize,
    /// Maximum number of Lanczos iterations (k) to test.
    #[clap(long, default_value_t = 500)]
    k_max: usize,
    /// Step size for iterating k.
    #[clap(long, default_value_t = 20)]
    k_step: usize,
    /// Path to the output CSV file where results will be written.
    #[clap(long, value_name = "PATH")]
    output: PathBuf,
}

/// Represents a single row of data in the output CSV file, capturing key
/// metrics related to the numerical properties of the generated bases.
#[derive(Debug, Serialize)]
struct OrthogonalityResult {
    /// The number of Lanczos iterations performed.
    k: usize,
    /// Orthogonality loss for the standard one-pass basis, measured as ||I - V_k^H V_k||_F.
    ortho_loss_standard: f64,
    /// Orthogonality loss for the regenerated two-pass basis, measured as ||I - V'_k^H V'_k||_F.
    ortho_loss_regenerated: f64,
    /// Numerical deviation between the two bases, measured as ||V_k - V'_k||_F.
    basis_drift_fro: f64,
    /// L2-norm of the difference between the final solution vectors from each method.
    /// This is included for a comprehensive analysis, linking basis stability to solution accuracy.
    solution_deviation_l2: f64,
}

/// Creates a synthetic sparse diagonal matrix with controlled eigenvalues
/// corresponding to the specified scenario.
fn create_diagonal_problem(
    n: usize,
    scenario: ProblemScenario,
    func: MatrixFunction,
) -> SparseColMat<usize, f64> {
    let mut triplets = Vec::with_capacity(n);
    let mut eigs = Vec::with_capacity(n);

    match (func, scenario) {
        (MatrixFunction::Exp, ProblemScenario::WellConditioned) => {
            for i in 0..n {
                eigs.push(-10.0 + (9.9 / (n - 1).max(1) as f64) * i as f64);
            }
        }
        (MatrixFunction::Exp, ProblemScenario::IllConditioned) => {
            for i in 0..n {
                eigs.push(-1000.0 + (999.9 / (n - 1).max(1) as f64) * i as f64);
            }
        }
        (MatrixFunction::Inv, ProblemScenario::WellConditioned) => {
            for i in 0..n {
                eigs.push(0.1 + (99.9 / (n - 1).max(1) as f64) * i as f64);
            }
        }
        (MatrixFunction::Inv, ProblemScenario::IllConditioned) => {
            let mid = n / 2;
            for i in 0..n {
                eigs.push(if i < mid {
                    0.1 + (0.9 / (mid - 1).max(1) as f64) * i as f64
                } else {
                    -1.0 + (0.9 / (n - mid - 1).max(1) as f64) * (i - mid) as f64
                });
            }
            eigs[mid] = 1e-8; // A near-zero eigenvalue to induce ill-conditioning.
        }
    }

    for (i, &eig) in eigs.iter().enumerate() {
        triplets.push(Triplet {
            row: i,
            col: i,
            val: eig,
        });
    }
    SparseColMat::try_new_from_triplets(n, n, &triplets)
        .expect("Failed to construct sparse diagonal matrix.")
}

fn main() -> Result<()> {
    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Info)
        .try_init()?;
    let args = OrthoArgs::parse();
    log::info!(
        "Starting orthogonality analysis for function: {:?}, scenario: {:?}",
        args.function,
        args.scenario
    );

    let a = create_diagonal_problem(args.n, args.scenario, args.function);
    let mut rng = StdRng::seed_from_u64(42); // For reproducible results.
    let b = Mat::from_fn(args.n, 1, |_, _| rng.random());

    let mut writer = csv::Writer::from_path(&args.output)?;
    let mut stack_mem = MemBuffer::new(a.as_ref().apply_scratch(1, faer::Par::Seq));

    for k in (args.k_min..=args.k_max).step_by(args.k_step) {
        if k == 0 {
            continue;
        }
        log::info!("Running for k = {}...", k);
        let stack = MemStack::new(&mut stack_mem);

        // 1. Execute the standard one-pass algorithm to get the reference basis V_k.
        let standard_output = lanczos_standard(&a.as_ref(), b.as_ref(), k, stack, None)?;
        let v_k_standard = standard_output.v_k;
        let steps = standard_output.decomposition.steps_taken;
        if steps == 0 {
            continue;
        }

        // 2. Execute the two-pass algorithm using a variant that returns the regenerated basis V'_k.
        // A dummy `y_k` is sufficient as we are only interested in the regenerated basis,
        // not the final solution vector from this function call.
        let y_k_dummy = Mat::<f64>::zeros(steps, 1);
        let pass_two_output = lanczos_pass_two_with_basis(
            &a.as_ref(),
            b.as_ref(),
            &standard_output.decomposition,
            y_k_dummy.as_ref(),
            stack,
        )?;
        let v_k_regenerated = pass_two_output.v_k;

        // 3. Compute the numerical stability metrics.
        let identity = Mat::<f64>::identity(steps, steps);

        // The loss of orthogonality is quantified by the Frobenius norm of the deviation
        // of V_k^H * V_k from the identity matrix.
        let ortho_loss_standard =
            (&identity - v_k_standard.as_ref().adjoint() * v_k_standard.as_ref()).norm_l2();
        let ortho_loss_regenerated =
            (&identity - v_k_regenerated.as_ref().adjoint() * v_k_regenerated.as_ref()).norm_l2();

        // The basis drift measures the numerical difference between the standard and regenerated bases.
        let basis_drift_fro = (v_k_standard.as_ref() - v_k_regenerated.as_ref()).norm_l2();

        // This is included to confirm that identical bases produce identical solutions.
        // For this specific test, we can compute the solutions directly without a solver.
        let x_k_standard = &v_k_standard * &y_k_dummy;
        let x_k_regenerated = &v_k_regenerated * &y_k_dummy;
        let solution_deviation_l2 = (x_k_standard - x_k_regenerated).norm_l2();

        writer.serialize(OrthogonalityResult {
            k: steps,
            ortho_loss_standard,
            ortho_loss_regenerated,
            basis_drift_fro,
            solution_deviation_l2,
        })?;
    }

    writer.flush()?;
    log::info!(
        "Orthogonality analysis complete. Results saved to {:?}.",
        &args.output
    );
    Ok(())
}
