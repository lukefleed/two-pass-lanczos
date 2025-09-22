//! Orthogonality analysis for Lanczos basis stability measurement.
//!
//! This executable analyzes the numerical stability of Lanczos basis vectors by measuring
//! orthogonality loss and basis drift between standard and two-pass methods. Computes
//! ||I - V_k^H V_k||_F to quantify orthogonality deterioration and ||V_k - V'_k||_F
//! to measure basis regeneration accuracy. Tests different spectral conditions to
//! assess stability under various numerical challenges.

use anyhow::Result;
use clap::{Parser, ValueEnum};
use faer::{
    dyn_stack::{MemBuffer, MemStack},
    matrix_free::LinOp,
    prelude::*,
    sparse::{SparseColMat, Triplet},
};
use lanczos_project::algorithms::{
    lanczos::lanczos_standard, lanczos_two_pass::lanczos_pass_two_with_basis,
};
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::Serialize;
use std::path::PathBuf;

/// Defines the matrix function `f` used to motivate the spectral properties of the test matrix.
/// While the orthogonality of the Lanczos basis is independent of `f`, the function's analytical
/// properties (e.g., singularities) guide the construction of challenging spectra for $\mathbf{A}$.
#[derive(ValueEnum, Clone, Debug, Copy)]
enum MatrixFunction {
    /// Corresponds to $f(z) = z^{-1}$. This motivates test spectra with eigenvalues near zero,
    /// which is a known stress test for the stability of the projected system inversion.
    Inv,
    /// Corresponds to $f(z) = \exp(z)$. This motivates wide spectra where high-degree
    /// polynomial approximation is challenging, requiring a large number of iterations.
    Exp,
}

/// Defines the spectral properties of the test matrix $\mathbf{A}$.
#[derive(ValueEnum, Clone, Debug, Copy)]
enum ProblemScenario {
    /// A well-conditioned problem where eigenvalues are well-separated and bounded away
    /// from any function singularities.
    WellConditioned,
    /// An ill-conditioned problem with spectral properties
    /// designed to challenge the numerical stability
    /// of the Lanczos process.
    IllConditioned,
}

/// Command-line arguments for the orthogonality analysis runner.
#[derive(Parser, Debug)]
#[clap(
    name = "orthogonality-runner",
    about = "Runs an analysis of Lanczos basis orthogonality and regeneration stability."
)]
struct OrthoArgs {
    /// The matrix function class to determine the problem's spectral properties.
    #[clap(long, value_enum)]
    function: MatrixFunction,
    /// The spectral scenario for the test problem.
    #[clap(long, value_enum)]
    scenario: ProblemScenario,
    /// Dimension of the synthetic test matrix.
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

/// Represents a single row of data in the output CSV
#[derive(Debug, Serialize)]
struct OrthogonalityResult {
    /// The number of Lanczos iterations performed.
    k: usize,
    /// Orthogonality loss for the standard one-pass basis, measured as $\|I - V_k^H V_k\|_F$.
    /// Note: `faer`'s `norm_l2` on a matrix computes the Frobenius norm.
    ortho_loss_standard: f64,
    /// Orthogonality loss for the regenerated two-pass basis, measured as $\|I - V'_k^H V'_k\|_F$.
    ortho_loss_regenerated: f64,
    /// Numerical deviation (or "drift") between the two bases, measured as $\|V_k - V'_k\|_F$.
    /// A small drift validates the numerical faithfulness of the regeneration process.
    basis_drift_fro: f64,
    /// L2-norm of the difference between solution vectors computed with a dummy coefficient vector.
    /// This metric links basis stability to the final solution space, confirming that
    /// numerically similar bases produce similar outputs.
    solution_deviation_l2: f64,
}

/// Creates a synthetic sparse diagonal matrix with a controlled eigenvalue distribution.
///
/// This is a standard methodology in numerical analysis for creating test problems where the
/// exact action of a matrix function $f(\mathbf{A})$ can be computed analytically to machine
/// precision, providing an unambiguous ground truth for error analysis.
fn create_diagonal_problem(
    n: usize,
    scenario: ProblemScenario,
    func: MatrixFunction,
) -> SparseColMat<usize, f64> {
    let mut triplets = Vec::with_capacity(n);
    let mut eigs = Vec::with_capacity(n);

    match (func, scenario) {
        // A compact, well-separated spectrum.
        // Values are in the range [-10, -0.1].
        (MatrixFunction::Exp, ProblemScenario::WellConditioned) => {
            for i in 0..n {
                eigs.push(-10.0 + (9.9 / (n - 1).max(1) as f64) * i as f64);
            }
        }
        // A wide spectrum, challenging for polynomial approximation.
        // Values are in the range [-1000, -0.1].
        (MatrixFunction::Exp, ProblemScenario::IllConditioned) => {
            for i in 0..n {
                eigs.push(-1000.0 + (999.9 / (n - 1).max(1) as f64) * i as f64);
            }
        }
        // A positive definite spectrum, bounded away from the singularity at zero.
        // Values are in the range [0.1, 100].
        (MatrixFunction::Inv, ProblemScenario::WellConditioned) => {
            for i in 0..n {
                eigs.push(0.1 + (99.9 / (n - 1).max(1) as f64) * i as f64);
            }
        }
        // An indefinite, quasi-singular spectrum. Values are in the ranges [0.1, 1.0] and
        // [-1.0, -0.1], with a near-zero eigenvalue introduced to induce severe ill-conditioning.
        (MatrixFunction::Inv, ProblemScenario::IllConditioned) => {
            let mid = n / 2;
            for i in 0..n {
                eigs.push(if i < mid {
                    0.1 + (0.9 / (mid - 1).max(1) as f64) * i as f64
                } else {
                    -1.0 + (0.9 / (n - mid - 1).max(1) as f64) * (i - mid) as f64
                });
            }
            // Introduce a near-zero eigenvalue to induce severe ill-conditioning.
            eigs[mid] = 1e-8;
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

    // --- 1. Experimental Setup ---
    let a = create_diagonal_problem(args.n, args.scenario, args.function);
    // A random vector ensures non-trivial components in many eigenspaces,
    // avoiding premature breakdown.
    let mut rng = StdRng::seed_from_u64(42);
    let b = Mat::from_fn(args.n, 1, |_, _| rng.random());

    let mut writer = csv::Writer::from_path(&args.output)?;
    // Pre-allocate a memory buffer for `faer`'s stack to reuse across iterations.
    let mut stack_mem = MemBuffer::new(a.as_ref().apply_scratch(1, faer::Par::Seq));

    // --- 2. Iterate over Krylov subspace dimensions ---
    for k in (args.k_min..=args.k_max).step_by(args.k_step) {
        if k == 0 {
            continue;
        }
        log::info!("Running for k = {}...", k);
        let stack = MemStack::new(&mut stack_mem);

        // --- 3. Generate Bases ---
        // a. Execute the standard one-pass algorithm to get the reference basis V_k.
        let standard_output = lanczos_standard(&a.as_ref(), b.as_ref(), k, stack, None)?;
        let v_k_standard = standard_output.v_k;
        let steps = standard_output.decomposition.steps_taken;
        if steps == 0 {
            continue;
        }

        // b. Execute the two-pass variant that returns the regenerated basis V'_k.
        // A dummy `y_k` vector is sufficient here, as we are only interested in the
        // properties of the regenerated basis `V'_k`, not the final solution vector.
        let y_k_dummy = Mat::<f64>::zeros(steps, 1);
        let pass_two_output = lanczos_pass_two_with_basis(
            &a.as_ref(),
            b.as_ref(),
            &standard_output.decomposition,
            y_k_dummy.as_ref(),
            stack,
        )?;
        let v_k_regenerated = pass_two_output.v_k;

        // --- 4. Compute and Serialize Numerical Stability Metrics ---
        let identity = Mat::<f64>::identity(steps, steps);

        // a. Loss of orthogonality for both bases.
        let ortho_loss_standard =
            (&identity - v_k_standard.as_ref().adjoint() * v_k_standard.as_ref()).norm_l2();
        let ortho_loss_regenerated =
            (&identity - v_k_regenerated.as_ref().adjoint() * v_k_regenerated.as_ref()).norm_l2();

        // b. Direct deviation (drift) between the two bases.
        let basis_drift_fro = (v_k_standard.as_ref() - v_k_regenerated.as_ref()).norm_l2();

        // c. Sanity check: confirm that identical bases produce identical solutions.
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
