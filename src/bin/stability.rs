//! Experiment Runner for the Numerical Stability Analysis (Experiment 3).
//!
//! This executable performs a monolithic experiment to analyze the numerical
//! stability of the two-pass Lanczos basis regeneration. It compares the basis
//! from the standard one-pass algorithm (`V_k`) with the regenerated basis
//! from the second pass (`V'_k`) and computes several key error metrics.

use anyhow::{Context, Result, anyhow};
use clap::Parser;
use faer::{
    dyn_stack::{MemBuffer, MemStack},
    matrix_free::LinOp,
    prelude::*,
    sparse::{SparseColMat, SymbolicSparseColMat, Triplet},
};
use lanczos_project::{
    algorithms::lanczos::{LanczosDecomposition, lanczos_pass_two_with_basis, lanczos_standard},
    utils::data_loader::load_kkt_system,
};
use serde::Serialize;
use std::hint::black_box;
use std::path::{Path, PathBuf};

/// Command-line arguments for the numerical stability experiment.
#[derive(Parser, Debug)]
#[clap(
    name = "stability-runner",
    about = "Runs the numerical stability analysis for the Lanczos two-pass method."
)]
struct StabilityArgs {
    /// Path to the directory containing the test instance files.
    #[clap(long, value_name = "PATH")]
    instance_dir: PathBuf,

    /// Maximum number of Lanczos iterations (k) to test.
    #[clap(long, default_value_t = 500)]
    k_max: usize,

    /// Step size for iterating k.
    #[clap(long, default_value_t = 10)]
    k_step: usize,

    /// Path to the output CSV file where results will be written.
    #[clap(long, value_name = "PATH")]
    output: PathBuf,
}

/// Represents a single row of data for the stability analysis CSV.
#[derive(Debug, Serialize)]
struct StabilityResult {
    /// The number of Lanczos iterations.
    k: usize,
    /// Loss of orthogonality for the standard basis: ||I - V_k^H V_k||_F.
    ortho_loss_standard: f64,
    /// Loss of orthogonality for the regenerated basis: ||I - (V'_k)^H V'_k||_F.
    ortho_loss_regenerated: f64,
    /// Frobenius norm of the difference between the two bases: ||V_k - V'_k||_F.
    basis_drift_fro: f64,
    /// L2-norm of the difference between the final solution vectors: ||x_k - x'_k||_2.
    solution_deviation_l2: f64,
}

/// Helper to find the first file with a given extension in a directory.
fn find_file_by_extension(dir: &Path, ext: &str) -> Result<PathBuf> {
    let entries =
        std::fs::read_dir(dir).with_context(|| format!("Failed to read directory: {:?}", dir))?;
    for entry in entries {
        let path = entry?.path();
        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some(ext) {
            return Ok(path);
        }
    }
    Err(anyhow!("No .{} file found in directory {:?}", ext, dir))
}

/// Helper to assemble a sparse `faer::SparseColMat` from Lanczos coefficients.
fn assemble_tridiagonal_sparse(alphas: &[f64], betas: &[f64]) -> Result<SparseColMat<usize, f64>> {
    let steps = alphas.len();
    if steps == 0 {
        let symbolic = SymbolicSparseColMat::<usize>::new_checked(0, 0, vec![0], None, vec![]);
        return Ok(SparseColMat::new(symbolic, vec![]));
    }
    let mut triplets = Vec::with_capacity(3 * steps - 2);
    for i in 0..steps {
        triplets.push(Triplet {
            row: i,
            col: i,
            val: alphas[i],
        });
    }
    for i in 0..steps - 1 {
        triplets.push(Triplet {
            row: i,
            col: i + 1,
            val: betas[i],
        });
        triplets.push(Triplet {
            row: i + 1,
            col: i,
            val: betas[i],
        });
    }
    SparseColMat::try_new_from_triplets(steps, steps, &triplets)
        .map_err(|e| anyhow!("Failed to construct sparse T_k: {:?}", e))
}

/// The f(T_k) solver for this experiment. f(z)=z.
fn f_tk_solver(alphas: &[f64], betas: &[f64]) -> Result<Mat<f64>> {
    let t_k_sparse = assemble_tridiagonal_sparse(alphas, betas)?;
    if t_k_sparse.nrows() == 0 {
        return Ok(Mat::zeros(0, 1));
    }
    let mut e1 = Mat::zeros(t_k_sparse.nrows(), 1);
    e1.as_mut()[(0, 0)] = 1.0;
    Ok(t_k_sparse * e1)
}

/// Executes the standard one-pass Lanczos process to get reference results.
fn run_standard_path(
    a: &impl LinOp<f64>,
    b: MatRef<f64>,
    k: usize,
    stack: &mut MemStack,
) -> Result<(Mat<f64>, Mat<f64>, LanczosDecomposition<f64>)> {
    let standard_output = lanczos_standard(a, b, k, stack, None)?;
    let v_k = standard_output.v_k.clone();
    let decomposition = standard_output.decomposition;
    let y_k = f_tk_solver(&decomposition.alphas, &decomposition.betas)? * decomposition.b_norm;
    let x_k = &v_k * &y_k;
    Ok((v_k, x_k, decomposition))
}

/// Executes the two-pass regeneration process.
fn run_regenerated_path(
    a: &impl LinOp<f64>,
    b: MatRef<f64>,
    decomposition: &LanczosDecomposition<f64>,
    stack: &mut MemStack,
) -> Result<(Mat<f64>, Mat<f64>)> {
    let y_k = f_tk_solver(&decomposition.alphas, &decomposition.betas)? * decomposition.b_norm;
    let pass_two_output = lanczos_pass_two_with_basis(a, b, decomposition, y_k.as_ref(), stack)?;
    let v_k_regenerated = pass_two_output.v_k;
    let x_k_regenerated = pass_two_output.x_k;
    Ok((v_k_regenerated, x_k_regenerated))
}

/// The main entry point for the stability experiment.
fn main() -> Result<()> {
    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Info)
        .try_init()?;

    let args = StabilityArgs::parse();
    log::info!("Starting numerical stability experiment...");

    let dmx_path = find_file_by_extension(&args.instance_dir, "dmx")?;
    let qfc_path = find_file_by_extension(&args.instance_dir, "qfc")?;
    let kkt_system = load_kkt_system(dmx_path, qfc_path)?;
    let a = &kkt_system.a;
    let n = a.nrows();
    let b = Mat::<f64>::from_fn(n, 1, |_, _| 1.0 / (n as f64).sqrt());

    let mut results = Vec::new();
    let mut stack_mem = MemBuffer::new(a.as_ref().apply_scratch(1, faer::Par::Seq));

    for k in (args.k_step..=args.k_max).step_by(args.k_step) {
        if k == 0 {
            continue;
        }
        log::info!("Running stability analysis for k = {}...", k);

        let mut stack = MemStack::new(&mut stack_mem);

        // 1. Run the standard path to get reference V_k, x_k, and decomposition.
        let (v_k, x_k_standard, decomposition) =
            run_standard_path(&a.as_ref(), b.as_ref(), k, &mut stack)?;

        if decomposition.steps_taken != k {
            log::warn!(
                "Standard pass broke down at {} steps, requested k={}",
                decomposition.steps_taken,
                k
            );
            break;
        }

        // 2. Create an optimization barrier to ensure independent computations.
        let opaque_decomposition = black_box(decomposition);

        // 3. Run the regeneration path using the now "opaque" decomposition.
        let (v_k_regenerated, x_k_regenerated) =
            run_regenerated_path(&a.as_ref(), b.as_ref(), &opaque_decomposition, &mut stack)?;

        // 4. Calculate Metrics.
        let id_k = Mat::<f64>::identity(k, k);
        let ortho_loss_standard = (&id_k - v_k.adjoint() * &v_k).norm_l2();
        let ortho_loss_regenerated =
            (&id_k - v_k_regenerated.adjoint() * &v_k_regenerated).norm_l2();
        let basis_drift_fro = (&v_k - &v_k_regenerated).norm_l2();
        let solution_deviation_l2 = (&x_k_standard - &x_k_regenerated).norm_l2();

        // 5. Store the result.
        results.push(StabilityResult {
            k,
            ortho_loss_standard,
            ortho_loss_regenerated,
            basis_drift_fro,
            solution_deviation_l2,
        });
    }

    log::info!(
        "Experiment finished. Writing results to {:?}...",
        &args.output
    );
    let mut writer = csv::Writer::from_path(&args.output)?;
    for record in results {
        writer.serialize(record)?;
    }
    writer.flush()?;

    log::info!("Stability analysis complete.");
    Ok(())
}
