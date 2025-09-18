//! Experiment Runner for the Scalability Analysis (Experiment 2).
//!
//! This executable orchestrates the scalability experiment. The primary process,
//! the "orchestrator," iterates through a range of problem sizes. For each size,
//! it first invokes the `datagen` binary to generate the required KKT system files.
//! It then spawns isolated "worker" child processes for each Lanczos variant
//! (`standard`, `two-pass`) to run the computation and measure performance.
//!
//! This orchestrator/worker model is crucial for obtaining accurate memory
//! measurements (Peak RSS), as each worker runs in a separate process,
//! preventing its memory usage from being conflated with that of the orchestrator
//! or other workers. The orchestrator captures the single-row CSV output from
//! each worker and aggregates all results into a final CSV file.

use anyhow::{Context, Result, anyhow};
use clap::{Parser, ValueEnum};
use faer::{
    dyn_stack::{MemBuffer, MemStack},
    matrix_free::LinOp,
    prelude::*,
    sparse::{SparseColMat, Triplet},
};
use lanczos_project::{
    solvers::{lanczos, lanczos_two_pass},
    utils::{data_loader::load_kkt_system, perf::get_peak_rss_kb},
};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::Instant,
};

/// Environment variable to differentiate between orchestrator and worker processes.
/// If this is set, the process runs in worker mode for the specified variant.
const VARIANT_ENV_VAR: &str = "LANCZOS_SCALABILITY_VARIANT";

/// Defines the Lanczos algorithm variant to be run in a worker process.
#[derive(ValueEnum, Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Copy)]
#[serde(rename_all = "kebab-case")]
enum LanczosVariant {
    Standard,
    TwoPass,
}

/// Command-line arguments for the main orchestrator process.
#[derive(Parser, Debug)]
#[clap(
    name = "scalability-runner",
    about = "Runs the scalability analysis for the Lanczos methods."
)]
struct ScalabilityArgs {
    /// The fixed number of Lanczos iterations (k) to use for all runs.
    #[clap(long)]
    k_fixed: usize,
    /// The starting number of arcs for the generated network problems.
    #[clap(long)]
    arcs_start: usize,
    /// The ending number of arcs for the generated network problems.
    #[clap(long)]
    arcs_end: usize,
    /// The step size for increasing the number of arcs.
    #[clap(long)]
    arcs_step: usize,
    /// The rho parameter for the `datagen` utility, controlling problem structure.
    #[clap(long)]
    rho: u32,
    /// Path to the output CSV file for storing aggregated results.
    #[clap(long, value_name = "PATH")]
    output: PathBuf,
}

/// Command-line arguments for the isolated worker processes.
#[derive(Parser, Debug)]
struct WorkerArgs {
    /// The path to the directory containing the instance data for this specific run.
    #[clap(long)]
    instance_dir: PathBuf,
    /// The fixed number of Lanczos iterations (k) to use.
    #[clap(long)]
    k_fixed: usize,
}

/// Represents a single row of data in the final output CSV.
/// This struct captures the result from a single worker run.
#[derive(Debug, Serialize, Deserialize)]
struct ScalabilityResult {
    variant: LanczosVariant,
    n: usize,
    k: usize,
    time_s: f64,
    rss_kb: u64,
}

/// Main entry point.
///
/// The logic dispatches to either the orchestrator or a worker based on the
/// presence of the `LANCZOS_SCALABILITY_VARIANT` environment variable.
fn main() -> Result<()> {
    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Info)
        .try_init()
        .map_err(|e| anyhow!("Failed to initialize logger: {}", e))?;

    if let Ok(variant_str) = std::env::var(VARIANT_ENV_VAR) {
        let variant = LanczosVariant::from_str(&variant_str, true)
            .map_err(|_| anyhow!("Invalid variant string in env var: {}", variant_str))?;
        run_worker(variant)
    } else {
        run_orchestrator()
    }
}

/// Orchestrator logic.
///
/// This function iterates through the specified range of problem sizes (`arcs`),
/// calls `datagen` to create the test problems, and then spawns worker processes
/// to execute the Lanczos algorithms and collect performance data.
/// Results are written to the output CSV file incrementally to ensure data is
/// saved even if a run fails mid-way through the experiment.
fn run_orchestrator() -> Result<()> {
    let args = ScalabilityArgs::parse();
    log::info!("Orchestrator starting scalability experiment...");

    // Create the CSV writer. Opening the file here will truncate it if it exists.
    let mut writer = csv::Writer::from_path(&args.output)
        .with_context(|| format!("Failed to create CSV writer for {:?}", &args.output))?;

    let variants_to_run = [LanczosVariant::Standard, LanczosVariant::TwoPass];

    for arcs in (args.arcs_start..=args.arcs_end).step_by(args.arcs_step) {
        log::info!("Processing problem size: {arcs} arcs");

        // 1. Create a unique directory for this problem instance's data.
        let instance_dir = PathBuf::from(format!("data/scalability/arcs_{arcs}_rho_{}", args.rho));
        fs::create_dir_all(&instance_dir)
            .with_context(|| format!("Failed to create instance directory: {instance_dir:?}"))?;

        // 2. Invoke the `datagen` binary to generate the KKT system files.
        log::info!("Invoking datagen to generate instance data...");
        let datagen_status = Command::new("./target/release/datagen")
            .arg("--arcs")
            .arg(arcs.to_string())
            .arg("--rho")
            .arg(args.rho.to_string())
            .arg("--output-dir")
            .arg(&instance_dir)
            .status()
            .context(
                "Failed to execute datagen binary. Did you build with `cargo build --release`?",
            )?;

        if !datagen_status.success() {
            // Log the error and continue to the next problem size.
            log::error!(
                "datagen process failed with status: {}. Skipping this problem size.",
                datagen_status
            );
            continue;
        }
        log::info!("Datagen completed successfully.");

        // 3. Spawn worker processes for each Lanczos variant.
        for &variant in &variants_to_run {
            log::info!("Spawning worker for variant: {variant:?}");
            let current_exe = std::env::current_exe()?;
            let child = Command::new(current_exe)
                .arg("--instance-dir")
                .arg(&instance_dir)
                .arg("--k-fixed")
                .arg(args.k_fixed.to_string())
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

            // 4. Capture and parse the worker's output.
            let output = child.wait_with_output()?;
            if !output.status.success() {
                // Log the error and continue. This makes the experiment more robust.
                log::error!(
                    "Worker process for variant {:?} on instance {:?} failed with status: {}. Skipping.",
                    variant,
                    instance_dir,
                    output.status
                );
                continue;
            }

            let mut rdr = csv::ReaderBuilder::new()
                .has_headers(false)
                .from_reader(output.stdout.as_slice());

            if let Some(result) = rdr.deserialize::<ScalabilityResult>().next() {
                match result {
                    Ok(record) => {
                        log::info!(
                            "Worker finished. Result: n={}, k={}, time={:.2}s, rss={}KB",
                            record.n,
                            record.k,
                            record.time_s,
                            record.rss_kb
                        );
                        // 5. Write the record to the CSV file immediately and flush.
                        writer.serialize(&record)?;
                        writer.flush()?;
                    }
                    Err(e) => {
                        log::error!(
                            "Failed to parse worker output as CSV: {}. Skipping record.",
                            e
                        );
                    }
                }
            } else {
                log::warn!(
                    "Worker for {:?} produced no output. Skipping record.",
                    variant
                );
            }
        }
    }

    log::info!(
        "Scalability experiment complete. Results saved to {:?}.",
        &args.output
    );
    Ok(())
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

/// Worker logic.
///
/// This function runs in an isolated child process. It loads a single KKT system,
/// runs the specified Lanczos variant for a fixed `k`, measures performance,
/// and prints a single `ScalabilityResult` to stdout as a CSV row.
fn run_worker(variant: LanczosVariant) -> Result<()> {
    let args = WorkerArgs::parse();
    log::info!("Worker for {variant:?} started.");

    // Load the KKT system from the directory prepared by the orchestrator.
    let dmx_path = find_file_by_extension(&args.instance_dir, "dmx")?;
    let qfc_path = find_file_by_extension(&args.instance_dir, "qfc")?;
    let kkt_system = load_kkt_system(dmx_path, qfc_path)?;
    let a: &SparseColMat<usize, f64> = &kkt_system.a;
    let n = a.nrows();

    // The vector `b` is constructed from a known solution for consistency, though
    // for this experiment, we are not checking the solution's correctness.
    let x_true = Mat::<f64>::from_fn(n, 1, |_, _| 1.0 / (n as f64).sqrt());
    let b = a * &x_true;

    // A solver for the tridiagonal system T_k y = e_1. This uses `faer`'s
    // efficient sparse LU decomposition, which is O(k) for a tridiagonal system.
    let f_tk_solver = |alphas: &[f64], betas: &[f64]| -> Result<Mat<f64>> {
        let steps = alphas.len();
        if steps == 0 {
            return Ok(Mat::zeros(0, 1));
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
        let t_k_sparse = SparseColMat::try_new_from_triplets(steps, steps, &triplets)
            .map_err(|e| anyhow!("Failed to construct sparse T_k: {:?}", e))?;

        let mut e1 = Mat::zeros(steps, 1);
        e1.as_mut()[(0, 0)] = 1.0;

        let lu = t_k_sparse.as_ref().sp_lu()?;
        Ok(lu.solve(e1.as_ref()))
    };

    let mut stack_mem = MemBuffer::new(a.apply_scratch(1, Par::Seq));

    // Execute the specified Lanczos variant and measure performance.
    let (time_s, rss_kb) = match variant {
        LanczosVariant::Standard => {
            let start_time = Instant::now();
            let _ = lanczos(
                a,
                b.as_ref(),
                args.k_fixed,
                MemStack::new(&mut stack_mem),
                &f_tk_solver,
            )?;
            (start_time.elapsed().as_secs_f64(), get_peak_rss_kb())
        }
        LanczosVariant::TwoPass => {
            let start_time = Instant::now();
            let _ = lanczos_two_pass(
                a,
                b.as_ref(),
                args.k_fixed,
                MemStack::new(&mut stack_mem),
                &f_tk_solver,
            )?;
            (start_time.elapsed().as_secs_f64(), get_peak_rss_kb())
        }
    };

    // Serialize the single result to stdout.
    let mut writer = csv::WriterBuilder::new()
        .has_headers(false)
        .from_writer(std::io::stdout());
    writer.serialize(ScalabilityResult {
        variant,
        n,
        k: args.k_fixed,
        time_s,
        rss_kb,
    })?;
    writer.flush()?;

    log::info!("Worker for {variant:?} finished.");
    Ok(())
}
