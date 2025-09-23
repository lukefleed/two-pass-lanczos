//! Memory-computation trade-off analysis with robust statistical sampling.
//!
//! This executable measures memory usage and execution time as iteration count `k`
//! increases for a fixed problem size. It uses an isolated worker process for each
//! data point to ensure accurate memory profiling, and statistical sampling to ensure
//! the robustness of the results.
//!
//! ## Methodology
//!
//! 1.  **Fixed Problem**: A single, large KKT system is used for the entire experiment to
//!     ensure that the only independent variable is the iteration count `k`.
//! 2.  **Statistical Sampling**: For each value of `k`, the orchestrator runs `S`
//!     independent samples to capture system noise and scheduling variability.
//! 3.  **Process Isolation**: Each sample for each `k` is run in a separate worker
//!     process. This guarantees that Peak Resident Set Size (RSS) measurements are
//!     uncontaminated by previous runs.
//! 4.  **Statistical Aggregation**: The results from the `S` samples are aggregated to
//!     produce the median (robust to outliers) and standard deviation for both
//!     wall-clock time and memory usage.

use anyhow::{Context, Result, anyhow};
use clap::{Parser, ValueEnum};
use faer::{
    dyn_stack::{MemBuffer, MemStack},
    matrix_free::LinOp,
    prelude::*,
    sparse::{SparseColMat, SymbolicSparseColMat, Triplet},
};
use lanczos_project::{
    solvers::{lanczos, lanczos_two_pass},
    utils::{data_loader::load_kkt_system, perf::get_peak_rss_kb},
};
use serde::{Deserialize, Serialize};
use statrs::statistics::{Data, Distribution, Median};
use std::{
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::Instant,
};

/// Environment variable used to differentiate between orchestrator and worker.
const VARIANT_ENV_VAR: &str = "LANCZOS_TRADEOFF_VARIANT";

/// Defines the Lanczos algorithm variant to be executed in a worker process.
#[derive(ValueEnum, Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Copy)]
#[serde(rename_all = "kebab-case")]
enum LanczosVariant {
    Standard,
    TwoPass,
}

/// Command-line arguments for the orchestrator process.
#[derive(Parser, Debug)]
#[clap(
    name = "tradeoff-runner",
    about = "Runs the memory-computation trade-off experiment for the Lanczos methods."
)]
struct TradeoffArgs {
    /// Path to the directory containing the test instance files.
    #[clap(long, value_name = "PATH")]
    instance_dir: PathBuf,
    /// Path to the output CSV file where results will be written.
    #[clap(long, value_name = "PATH")]
    output: PathBuf,
    /// The starting number of Lanczos iterations.
    #[clap(long, default_value_t = 50)]
    k_start: usize,
    /// The final number of Lanczos iterations.
    #[clap(long, default_value_t = 1000)]
    k_end: usize,
    /// The step size for increasing `k`.
    #[clap(long, default_value_t = 50)]
    k_step: usize,
    /// The number of independent samples to run for each value of `k`.
    #[clap(long, default_value_t = 5)]
    num_samples: u32,
}

/// Command-line arguments for the isolated worker processes.
#[derive(Parser, Debug)]
struct WorkerArgs {
    /// Path to the directory containing the test instance.
    #[clap(long)]
    instance_dir: PathBuf,
    /// The specific number of iterations `k` this worker should run.
    #[clap(long)]
    k: usize,
}

/// Data contract for a single sample, passed from worker to orchestrator.
#[derive(Debug, Serialize, Deserialize)]
struct TradeoffResult {
    variant: LanczosVariant,
    k: usize,
    time_s: f64,
    rss_kb: u64,
}

/// Data structure for the final aggregated results written to the output CSV.
#[derive(Debug, Serialize)]
struct AggregatedResult {
    variant: LanczosVariant,
    k: usize,
    time_s_median: f64,
    time_s_stddev: f64,
    rss_kb_median: f64,
    rss_kb_stddev: f64,
}

/// Main entry point: dispatches to orchestrator or worker logic.
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
fn run_orchestrator() -> Result<()> {
    let args = TradeoffArgs::parse();
    log::info!("Orchestrator starting trade-off experiment...");

    let mut writer = csv::Writer::from_path(&args.output)
        .with_context(|| format!("Failed to create CSV writer for {:?}", &args.output))?;

    for k in (args.k_start..=args.k_end).step_by(args.k_step) {
        if k == 0 {
            continue;
        }
        log::info!("Processing k = {} with {} samples", k, args.num_samples);

        let mut all_sample_results: Vec<TradeoffResult> = Vec::new();

        // --- Sampling Loop ---
        for sample_id in 1..=args.num_samples {
            log::info!(
                "--- Sample {}/{} for k = {} ---",
                sample_id,
                args.num_samples,
                k
            );

            for &variant in &[LanczosVariant::Standard, LanczosVariant::TwoPass] {
                log::info!("Spawning worker for variant: {:?}", variant);
                let current_exe = std::env::current_exe()?;
                let child = Command::new(current_exe)
                    // Pass only the arguments expected by the worker.
                    .arg("--instance-dir")
                    .arg(&args.instance_dir)
                    .arg("--k")
                    .arg(k.to_string())
                    .env(
                        VARIANT_ENV_VAR,
                        variant.to_possible_value().unwrap().get_name(),
                    )
                    .stdout(Stdio::piped())
                    .stderr(Stdio::inherit())
                    .spawn()
                    .with_context(|| format!("Failed to spawn worker for variant {:?}", variant))?;

                let output = child.wait_with_output()?;
                if !output.status.success() {
                    log::error!(
                        "Worker for {:?} at k={} failed. Skipping sample.",
                        variant,
                        k
                    );
                    continue;
                }

                let mut rdr = csv::ReaderBuilder::new()
                    .has_headers(false)
                    .from_reader(output.stdout.as_slice());
                if let Some(result) = rdr.deserialize::<TradeoffResult>().next() {
                    all_sample_results.push(result?);
                }
            }
        }

        // --- Aggregation and Serialization ---
        if all_sample_results.is_empty() {
            log::warn!("No successful samples for k = {}. Skipping aggregation.", k);
            continue;
        }

        for variant_to_process in [LanczosVariant::Standard, LanczosVariant::TwoPass] {
            let samples: Vec<_> = all_sample_results
                .iter()
                .filter(|r| r.variant == variant_to_process)
                .collect();

            if samples.is_empty() {
                log::warn!("No data for variant {:?} at k = {}.", variant_to_process, k);
                continue;
            }

            let times: Vec<f64> = samples.iter().map(|r| r.time_s).collect();
            let rsss: Vec<f64> = samples.iter().map(|r| r.rss_kb as f64).collect();

            let time_data = Data::new(times);
            let rss_data = Data::new(rsss);

            let time_s_stddev = if time_data.len() > 1 {
                time_data.std_dev().unwrap_or(0.0)
            } else {
                0.0
            };
            let rss_kb_stddev = if rss_data.len() > 1 {
                rss_data.std_dev().unwrap_or(0.0)
            } else {
                0.0
            };

            let agg_result = AggregatedResult {
                variant: variant_to_process,
                k: samples[0].k,
                time_s_median: time_data.median(),
                time_s_stddev,
                rss_kb_median: rss_data.median(),
                rss_kb_stddev,
            };
            writer.serialize(&agg_result)?;
        }
        writer.flush()?;
    }

    log::info!(
        "Trade-off experiment complete. Results saved to {:?}.",
        &args.output
    );
    Ok(())
}

/// Worker logic for a single experimental run.
fn run_worker(variant: LanczosVariant) -> Result<()> {
    // The worker only needs to parse arguments relevant to its task.
    // Clap will correctly ignore any other arguments passed by the orchestrator
    // if they are not defined in `WorkerArgs`.
    let args = WorkerArgs::parse();
    log::info!("Worker for {:?} started for k={}.", variant, args.k);

    let dmx_path = find_file_by_extension(&args.instance_dir, "dmx")?;
    let qfc_path = find_file_by_extension(&args.instance_dir, "qfc")?;
    let kkt_system = load_kkt_system(dmx_path, qfc_path)?;
    let a: &SparseColMat<usize, f64> = &kkt_system.a;

    let n = a.nrows();
    let x_true = Mat::<f64>::from_fn(n, 1, |_, _| 1.0 / (n as f64).sqrt());
    let b = a * &x_true;

    let f_tk_solver = |alphas: &[f64], betas: &[f64]| -> Result<Mat<f64>> {
        let steps = alphas.len();
        if steps == 0 {
            return Ok(Mat::zeros(0, 1));
        }
        let t_k_sparse = assemble_tridiagonal_sparse(alphas, betas)?;
        let mut e1 = Mat::zeros(steps, 1);
        e1.as_mut()[(0, 0)] = 1.0;
        Ok(t_k_sparse.as_ref().sp_lu()?.solve(e1.as_ref()))
    };

    let mut stack_mem = MemBuffer::new(a.apply_scratch(1, Par::Seq));

    let (time_s, rss_kb) = match variant {
        LanczosVariant::Standard => {
            let start_time = Instant::now();
            let _ = lanczos(
                a,
                b.as_ref(),
                args.k,
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
                args.k,
                MemStack::new(&mut stack_mem),
                &f_tk_solver,
            )?;
            (start_time.elapsed().as_secs_f64(), get_peak_rss_kb())
        }
    };

    let mut writer = csv::WriterBuilder::new()
        .has_headers(false)
        .from_writer(std::io::stdout());
    writer.serialize(TradeoffResult {
        variant,
        k: args.k,
        time_s,
        rss_kb,
    })?;
    writer.flush()?;

    log::info!("Worker for {:?} at k={} finished.", variant, args.k);
    Ok(())
}

/// Utility to find the first file with a given extension in a directory.
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
