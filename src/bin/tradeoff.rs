//! Experiment Runner for the Memory-Computation Trade-off Analysis (Experiment 1).
//!
//! This executable acts as an orchestrator for the trade-off experiment. When run,
//! it spawns isolated child processes for each Lanczos variant (`standard`, `two-pass`)
//! to ensure accurate, independent memory measurements. The main process then
//! collects and consolidates the results into a single CSV file.
use anyhow::{Context, Result, anyhow};
use clap::{Parser, ValueEnum};
use faer::{
    dyn_stack::{MemBuffer, MemStack},
    matrix_free::LinOp,
    prelude::*,
    sparse::{SparseColMat, SymbolicSparseColMat, Triplet},
};
use lanczos_project::utils::perf::get_peak_rss_kb;
use lanczos_project::{
    solvers::{lanczos, lanczos_two_pass},
    utils::data_loader::load_kkt_system,
};
use serde::{Deserialize, Serialize};
use std::{
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::Instant,
};

/// Defines command-line arguments that are common to all experiments.
/// By placing them in this shared module, we ensure consistency and avoid
/// code duplication across the different runner executables.
#[derive(Parser, Debug)]
pub struct CommonArgs {
    /// Path to the directory containing the test instance files.
    /// The directory is expected to contain 'netgen.dmx' and 'netgen.qfc'.
    #[clap(long, value_name = "PATH")]
    pub instance_dir: PathBuf,

    /// Path to the output CSV file where results will be written.
    #[clap(long, value_name = "PATH")]
    pub output: PathBuf,
}

/// An environment variable used for internal communication between the orchestrator
/// process and the worker child processes.
const VARIANT_ENV_VAR: &str = "LANCZOS_EXPERIMENT_VARIANT";

/// Defines the Lanczos algorithm variant to be run in a child process.
#[derive(ValueEnum, Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
enum LanczosVariant {
    Standard,
    TwoPass,
}

/// Command-line arguments for the trade-off experiment orchestrator.
#[derive(Parser, Debug)]
#[clap(
    name = "tradeoff-runner",
    about = "Runs the memory-computation trade-off experiment for the Lanczos methods."
)]
struct TradeoffArgs {
    #[clap(flatten)]
    common: CommonArgs,
    #[clap(long, default_value_t = 50)]
    k_start: usize,
    #[clap(long, default_value_t = 1000)]
    k_end: usize,
    #[clap(long, default_value_t = 50)]
    k_step: usize,
}

/// Represents a single row of data produced by a worker process.
#[derive(Debug, Serialize, Deserialize)]
struct TradeoffResult {
    variant: LanczosVariant,
    k: usize,
    time_s: f64,
    rss_kb: u64,
}

/// Helper function to find the first file with a given extension in a directory.
fn find_file_by_extension(dir: &Path, ext: &str) -> Result<PathBuf> {
    let entries =
        std::fs::read_dir(dir).with_context(|| format!("Failed to read directory: {dir:?}"))?;

    for entry in entries {
        let path = entry?.path();
        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some(ext) {
            return Ok(path);
        }
    }
    Err(anyhow!("No .{} file found in directory {:?}", ext, dir))
}

/// A helper function to assemble a sparse `faer::SparseColMat` from the Lanczos coefficients.
/// This uses the triplet representation for efficient sparse matrix construction.
fn assemble_tridiagonal_sparse(alphas: &[f64], betas: &[f64]) -> Result<SparseColMat<usize, f64>> {
    let steps = alphas.len();
    if steps == 0 {
        let symbolic = SymbolicSparseColMat::<usize>::new_checked(0, 0, vec![0], None, vec![]);
        return Ok(SparseColMat::new(symbolic, vec![]));
    }

    // A tridiagonal matrix of size `k x k` has at most 3k - 2 non-zero entries.
    let mut triplets = Vec::with_capacity(3 * steps - 2);

    // Add diagonal elements (alphas)
    for (i, &alpha) in alphas.iter().enumerate() {
        triplets.push(Triplet {
            row: i,
            col: i,
            val: alpha,
        });
    }

    // Add off-diagonal elements (betas)
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

/// The main entry point.
/// It distinguishes between being the orchestrator or a worker based on an environment variable.
fn main() -> Result<()> {
    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Info)
        .try_init()
        .map_err(|e| anyhow!("Failed to initialize logger: {}", e))?;

    if let Ok(variant_str) = std::env::var(VARIANT_ENV_VAR) {
        let variant = LanczosVariant::from_str(&variant_str, true)
            .map_err(|_| anyhow!("Invalid variant string in env var: {}", variant_str))?;
        run_worker(&variant)
    } else {
        run_orchestrator()
    }
}

/// This function runs in the main orchestrator process.
/// It spawns child processes for each variant and collects the results.
fn run_orchestrator() -> Result<()> {
    let args = TradeoffArgs::parse();
    log::info!("Orchestrator starting experiment...");

    let variants_to_run = [LanczosVariant::Standard, LanczosVariant::TwoPass];
    let mut child_handles = Vec::new();

    for variant in &variants_to_run {
        log::info!("Spawning worker for variant: {variant:?}");
        let current_exe = std::env::current_exe()?;
        let child = Command::new(current_exe)
            .args(std::env::args_os().skip(1))
            .env(
                VARIANT_ENV_VAR,
                variant
                    .to_possible_value()
                    .expect("Variant must have a corresponding clap value")
                    .get_name(),
            )
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| format!("Failed to spawn worker for variant {variant:?}"))?;
        child_handles.push((variant, child));
    }

    let mut all_results = Vec::new();
    for (variant, handle) in child_handles {
        log::info!("Waiting for worker {variant:?} to complete...");
        let output = handle.wait_with_output()?;

        if !output.status.success() {
            return Err(anyhow!(
                "Worker process for variant {:?} failed with status: {}",
                variant,
                output.status
            ));
        }

        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_reader(output.stdout.as_slice());
        for result in rdr.deserialize() {
            let record: TradeoffResult = result?;
            all_results.push(record);
        }
    }

    log::info!(
        "All workers finished. Consolidating results into {:?}...",
        &args.common.output
    );
    let mut writer = csv::Writer::from_path(&args.common.output)?;
    for record in all_results {
        writer.serialize(record)?;
    }
    writer.flush()?;

    log::info!("Experiment complete.");
    Ok(())
}

/// This function runs in an isolated child process.
/// It performs the measurements for a single Lanczos variant and prints the results
/// as CSV to standard output, which is then captured by the orchestrator.
fn run_worker(variant: &LanczosVariant) -> Result<()> {
    let args = TradeoffArgs::parse();
    log::info!("Worker for {variant:?} started.");

    let dmx_path = find_file_by_extension(&args.common.instance_dir, "dmx")?;
    let qfc_path = find_file_by_extension(&args.common.instance_dir, "qfc")?;
    let kkt_system = load_kkt_system(dmx_path, qfc_path)?;
    let a: &SparseColMat<usize, f64> = &kkt_system.a;

    let n = a.nrows();
    let x_true = Mat::<f64>::from_fn(n, 1, |_, _| 1.0 / (n as f64).sqrt());
    let b = a * &x_true;

    let mut writer = csv::WriterBuilder::new()
        .has_headers(false)
        .from_writer(std::io::stdout());

    // This closure now uses the highly efficient sparse LU solver from faer.
    let f_tk_solver = |alphas: &[f64], betas: &[f64]| -> Result<Mat<f64>> {
        let t_k_sparse = assemble_tridiagonal_sparse(alphas, betas)?;
        if t_k_sparse.nrows() == 0 {
            return Ok(Mat::zeros(0, 1));
        }

        let mut e1 = Mat::zeros(t_k_sparse.nrows(), 1);
        e1.as_mut()[(0, 0)] = 1.0;

        // Use the sparse LU solver. This is now an O(k) operation for a tridiagonal matrix.
        let lu = t_k_sparse.as_ref().sp_lu()?;
        Ok(lu.solve(e1.as_ref()))
    };

    let mut stack_mem = MemBuffer::new(a.as_ref().apply_scratch(1, Par::Seq));

    for k in (args.k_start..=args.k_end).step_by(args.k_step) {
        log::info!("Worker {variant:?}: Running for k = {k}...");

        let (time_s, rss_kb) = match variant {
            LanczosVariant::Standard => {
                let start_time = Instant::now();
                let _ = lanczos(
                    &a.as_ref(),
                    b.as_ref(),
                    k,
                    MemStack::new(&mut stack_mem),
                    &f_tk_solver,
                )?;
                (start_time.elapsed().as_secs_f64(), get_peak_rss_kb())
            }
            LanczosVariant::TwoPass => {
                let start_time = Instant::now();
                let _ = lanczos_two_pass(
                    &a.as_ref(),
                    b.as_ref(),
                    k,
                    MemStack::new(&mut stack_mem),
                    &f_tk_solver,
                )?;
                (start_time.elapsed().as_secs_f64(), get_peak_rss_kb())
            }
        };

        writer.serialize(TradeoffResult {
            variant: variant.clone(),
            k,
            time_s,
            rss_kb,
        })?;
    }

    writer.flush()?;
    log::info!("Worker for {variant:?} finished.");
    Ok(())
}
