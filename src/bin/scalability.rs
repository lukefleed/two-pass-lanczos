//! Experiment Runner for Scalability Analysis.
//!
//! This executable orchestrates the scalability experiment by employing an
//! orchestrator/worker process model. The main process, the "orchestrator," iterates
//! through a range of problem sizes (`n`). For each size, it first invokes the
//! `datagen` binary to generate the required KKT system files. It then spawns
//! isolated "worker" child processes for each Lanczos variant (`standard`, `two-pass`)
//! to run the computation and measure performance.
//!
//! By isolating each computational task in a separate process, we ensure that its
//! memory usage is not conflated with that of the orchestrator or other workers.
//! Communication is handled via environment variables (to assign roles) and stdout
//! (for workers to report results).

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
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::Instant,
};

/// Environment variable used for orchestrator/worker process differentiation.
/// Its presence and value determine if the current process executes as a worker
/// for a specific Lanczos variant.
const VARIANT_ENV_VAR: &str = "LANCZOS_SCALABILITY_VARIANT";
/// Maximum number of attempts to generate a valid instance for each problem size.
/// This provides robustness against stochastic failures in the external `netgen` tool.
const MAX_DATAGEN_ATTEMPTS: u32 = 5;

/// Defines the Lanczos algorithm variant to be executed in a worker process.
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
    /// The starting number of arcs for the generated network problems, defining the initial problem size.
    #[clap(long)]
    arcs_start: usize,
    /// The ending number of arcs for the generated network problems.
    #[clap(long)]
    arcs_end: usize,
    /// The step size for increasing the number of arcs between runs.
    #[clap(long)]
    arcs_step: usize,
    /// The rho parameter for the `datagen` utility, controlling problem topology.
    #[clap(long)]
    rho: u32,
    /// Path to the output CSV file for storing aggregated results.
    #[clap(long, value_name = "PATH")]
    output: PathBuf,
}

/// Command-line arguments parsed by the isolated worker processes.
#[derive(Parser, Debug)]
struct WorkerArgs {
    /// The path to the directory containing the instance data files (.dmx, .qfc).
    #[clap(long)]
    instance_dir: PathBuf,
    /// The fixed number of Lanczos iterations (k) to use.
    #[clap(long)]
    k_fixed: usize,
}

/// Represents the data contract between a worker and the orchestrator.
/// Each worker serializes one instance of this struct to stdout as a single CSV row.
#[derive(Debug, Serialize, Deserialize)]
struct ScalabilityResult {
    variant: LanczosVariant,
    n: usize,
    k: usize,
    time_s: f64,
    rss_kb: u64,
}

/// Main entry point: dispatches to orchestrator or worker logic.
///
/// This function acts as a dispatcher. It checks for the presence of the
/// `LANCZOS_SCALABILITY_VARIANT` environment variable to determine the process's role.
/// If the variable is set, it runs as a worker; otherwise, it runs as the orchestrator.
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

/// Orchestrator logic for managing the experiment lifecycle.
///
/// This function iterates through problem sizes, generates data for each, and then
/// spawns worker processes for each algorithm variant. Results are written to the
/// output CSV file incrementally.
fn run_orchestrator() -> Result<()> {
    let args = ScalabilityArgs::parse();
    log::info!("Orchestrator starting scalability experiment...");

    let mut writer = csv::Writer::from_path(&args.output)
        .with_context(|| format!("Failed to create CSV writer for {:?}", &args.output))?;

    for arcs in (args.arcs_start..=args.arcs_end).step_by(args.arcs_step) {
        log::info!("Processing problem size: {arcs} arcs");

        let instance_dir = PathBuf::from(format!("data/scalability/arcs_{arcs}_rho_{}", args.rho));

        // Generate and validate the instance, with built-in retry logic.
        let generation_result =
            generate_and_validate_instance(arcs, args.rho, &instance_dir, MAX_DATAGEN_ATTEMPTS);

        let valid_instance_dir = match generation_result {
            Ok(Some(dir)) => dir,
            Ok(None) => {
                log::error!(
                    "Failed to generate a valid instance for {arcs} arcs after {} attempts. Skipping.",
                    MAX_DATAGEN_ATTEMPTS
                );
                continue;
            }
            Err(e) => {
                log::error!(
                    "Unrecoverable error during data generation for {arcs} arcs: {}. Skipping.",
                    e
                );
                continue;
            }
        };

        // If instance is valid, spawn worker processes for each variant.
        for &variant in &[LanczosVariant::Standard, LanczosVariant::TwoPass] {
            log::info!("Spawning worker for variant: {variant:?}");
            // Re-spawn the current executable to run in worker mode.
            let current_exe = std::env::current_exe()?;
            let child = Command::new(current_exe)
                .arg("--instance-dir")
                .arg(&valid_instance_dir)
                .arg("--k-fixed")
                .arg(args.k_fixed.to_string())
                .env(
                    VARIANT_ENV_VAR,
                    variant.to_possible_value().unwrap().get_name(),
                )
                .stdout(Stdio::piped()) // Capture stdout to receive the result.
                .stderr(Stdio::inherit())
                .spawn()
                .with_context(|| format!("Failed to spawn worker for variant {variant:?}"))?;

            let output = child.wait_with_output()?;
            if !output.status.success() {
                log::error!(
                    "Worker for {:?} on {:?} failed with status: {}. Skipping.",
                    variant,
                    valid_instance_dir,
                    output.status
                );
                continue;
            }

            // Parse the single-line CSV from the worker's stdout.
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
                        writer.serialize(&record)?;
                        // Flush after each record to ensure data is saved incrementally.
                        writer.flush()?;
                    }
                    Err(e) => log::error!("Failed to parse worker CSV output: {}. Skipping.", e),
                }
            } else {
                log::warn!("Worker for {:?} produced no output. Skipping.", variant);
            }
        }
    }

    log::info!(
        "Scalability experiment complete. Results saved to {:?}.",
        &args.output
    );
    Ok(())
}

/// Generates and validates a test instance, with a retry mechanism.
///
/// This function calls the `datagen` binary and validates its output. The `netgen`
/// tool sometimes produces invalid DIMACS files with 0-based node indices, which
/// our loader rejects. This function checks for that specific error and retries with a
/// different random seed (`instance_id`) if validation fails.
fn generate_and_validate_instance(
    arcs: usize,
    rho: u32,
    instance_dir: &Path,
    max_attempts: u32,
) -> Result<Option<PathBuf>> {
    for attempt in 1..=max_attempts {
        if instance_dir.exists() {
            fs::remove_dir_all(instance_dir)?;
        }
        fs::create_dir_all(instance_dir)?;

        log::info!(
            "Invoking datagen (Attempt {}/{}) to generate instance...",
            attempt,
            max_attempts
        );

        let datagen_status = Command::new("./target/release/datagen")
            .arg("--arcs")
            .arg(arcs.to_string())
            .arg("--rho")
            .arg(rho.to_string())
            // Use attempt number as instance ID to effectively change the random seed.
            .arg("--instance-id")
            .arg(attempt.to_string())
            .arg("--output-dir")
            .arg(instance_dir)
            .status()
            .context("Failed to execute datagen. Did you build with `cargo build --release`?")?;

        if !datagen_status.success() {
            log::warn!(
                "Datagen process failed with status: {}. Retrying...",
                datagen_status
            );
            continue;
        }

        let dmx_path = match find_file_by_extension(instance_dir, "dmx") {
            Ok(path) => path,
            Err(_) => {
                log::warn!("Datagen produced no .dmx file. Retrying...");
                continue;
            }
        };

        if validate_dmx_file(&dmx_path)? {
            log::info!("Datagen completed with valid output.");
            return Ok(Some(instance_dir.to_path_buf()));
        } else {
            log::warn!(
                "Validation failed for {:?}: found invalid node index. Retrying...",
                dmx_path
            );
            continue;
        }
    }
    Ok(None)
}

/// Validates a DMX file by checking for invalid (0-based) node indices in arc definitions.
/// The DIMACS format requires 1-based indexing.
fn validate_dmx_file(path: &Path) -> Result<bool> {
    let file = fs::File::open(path)?;
    for line in BufReader::new(file).lines() {
        let line = line?;
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.first() == Some(&"a") && parts.len() >= 3 {
            // Check if either node index is "0".
            if parts[1] == "0" || parts[2] == "0" {
                return Ok(false); // Invalid index found.
            }
        }
    }
    Ok(true) // No invalid indices found.
}

/// Utility to find the first file with a given extension in a directory.
fn find_file_by_extension(dir: &Path, ext: &str) -> Result<PathBuf> {
    let entries =
        fs::read_dir(dir).with_context(|| format!("Failed to read directory: {dir:?}"))?;
    for entry in entries {
        let path = entry?.path();
        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some(ext) {
            return Ok(path);
        }
    }
    Err(anyhow!("No .{} file found in directory {:?}", ext, dir))
}

/// Worker logic for a single experimental run.
///
/// This function is the "payload" of the experiment, executed in an isolated process.
/// It loads a single KKT system, runs the specified Lanczos variant for a fixed `k`,
/// measures performance (wall-clock time and peak RSS), and prints a single
/// `ScalabilityResult` to stdout as a CSV row to be captured by the orchestrator.
fn run_worker(variant: LanczosVariant) -> Result<()> {
    let args = WorkerArgs::parse();
    log::info!(
        "Worker for {variant:?} started on instance {:?}.",
        &args.instance_dir
    );

    // Load the KKT system data.
    let dmx_path = find_file_by_extension(&args.instance_dir, "dmx")?;
    let qfc_path = find_file_by_extension(&args.instance_dir, "qfc")?;
    let kkt_system = load_kkt_system(dmx_path, qfc_path)?;
    let a: &SparseColMat<usize, f64> = &kkt_system.a;
    let n = a.nrows();

    // To evaluate the algorithm, we construct a problem `Ax = b` for which the
    // exact solution is known a priori. This is achieved by first defining a ground-truth
    // solution vector `x_true`, and then computing the corresponding right-hand side
    // vector as `b := A * x_true`. This allows for precise error calculation against a
    // verifiable reference.

    // We choose `x_true` to be a generic, unit-norm vector. A vector with all equal
    // components `1.0 / sqrt(n)` is selected because its L2 norm is exactly 1
    // and it has no zero components, ensuring it is not
    // pathologically aligned with any specific eigenspace of A.
    let x_true = Mat::<f64>::from_fn(n, 1, |_, _| 1.0 / (n as f64).sqrt());

    // Now, compute the right-hand side `b` that corresponds to our chosen `x_true`. The
    // resulting pair (A, b) now constitutes a complete test problem with a known solution.
    let b = a * &x_true;

    // Define the solver for the projected problem f(T_k)e_1. For this experiment, we
    // solve a linear system, so f(z)=z^-1. We use a sparse LU decomposition on the
    // tridiagonal T_k, which is an efficient O(k) operation.
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
        let t_k_sparse = SparseColMat::try_new_from_triplets(steps, steps, &triplets)?;
        let mut e1 = Mat::zeros(steps, 1);
        e1.as_mut()[(0, 0)] = 1.0;
        Ok(t_k_sparse.as_ref().sp_lu()?.solve(e1.as_ref()))
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

    // Serialize the single result to stdout as a headerless CSV row.
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
