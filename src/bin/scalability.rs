//! Scalability analysis with robust statistical sampling.
//!
//! This executable measures how memory and time scale with problem dimension, using a
//! methodology inspired by statistical benchmarking tools like Criterion.rs. It employs
//! an orchestrator/worker process model to ensure measurement accuracy.
//!
//! ## Methodology
//!
//! 1.  **Statistical Sampling**: For each problem size `n`, the orchestrator runs `S`
//!     independent samples. Each sample uses a stochastically generated test instance
//!     to capture performance variability.
//! 2.  **Process Isolation**: Each of the `S` samples is executed in an isolated worker
//!     process. This is critical for obtaining clean, independent Peak Resident Set Size
//!     (RSS) measurements, preventing memory usage from one run from affecting the next.
//! 3.  **Statistical Aggregation**: Results from all `S` samples are collected. Instead of
//!     a single data point, the final output for each `n` is a statistical summary,
//!     including the median (for robustness to outliers) and standard deviation (to
//!     quantify variance) for both wall-clock time and memory usage.
//!
//! This robust approach produces smoother performance curves and provides a clearer
//! understanding of the algorithm's typical performance and its variability.

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
use statrs::statistics::{Data, Distribution, Median};
use std::{
    fs,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::Instant,
};

/// Environment variable used for orchestrator/worker process differentiation.
const VARIANT_ENV_VAR: &str = "LANCZOS_SCALABILITY_VARIANT";

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
    about = "Runs the scalability analysis for the Lanczos methods with statistical sampling."
)]
struct ScalabilityArgs {
    /// The fixed number of Lanczos iterations (k) to use for all runs.
    #[clap(long)]
    k_fixed: usize,
    /// The starting number of arcs, defining the initial problem size.
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
    /// The number of independent samples to run for each problem size.
    #[clap(long, default_value_t = 5)]
    num_samples: u32,
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

/// Data contract for a single sample, passed from worker to orchestrator via stdout.
#[derive(Debug, Serialize, Deserialize)]
struct ScalabilityResult {
    variant: LanczosVariant,
    n: usize,
    k: usize,
    time_s: f64,
    rss_kb: u64,
}

/// Data structure for the final aggregated results written to the output CSV.
#[derive(Debug, Serialize)]
struct AggregatedResult {
    variant: LanczosVariant,
    n: usize,
    k: usize,
    time_s_median: f64,
    time_s_stddev: f64,
    rss_kb_median: f64,
    rss_kb_stddev: f64,
}

/// Main entry point: dispatches to orchestrator or worker logic based on an environment variable.
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
    let args = ScalabilityArgs::parse();
    log::info!("Orchestrator starting scalability experiment...");

    let mut writer = csv::Writer::from_path(&args.output)
        .with_context(|| format!("Failed to create CSV writer for {:?}", &args.output))?;

    for arcs in (args.arcs_start..=args.arcs_end).step_by(args.arcs_step) {
        log::info!(
            "Processing problem size: {} arcs with {} samples",
            arcs,
            args.num_samples
        );

        let mut all_sample_results: Vec<ScalabilityResult> = Vec::new();

        // --- Sampling Loop ---
        for sample_id in 1..=args.num_samples {
            log::info!(
                "--- Sample {}/{} for {} arcs ---",
                sample_id,
                args.num_samples,
                arcs
            );

            let instance_dir = PathBuf::from(format!(
                "data/scalability/arcs_{}_rho_{}/sample_{}",
                arcs, args.rho, sample_id
            ));
            let generation_result =
                generate_and_validate_instance(arcs, args.rho, &instance_dir, sample_id);

            let valid_instance_dir = match generation_result {
                Ok(Some(dir)) => dir,
                Ok(None) => {
                    log::warn!(
                        "Failed to generate valid instance for sample_id {}. Skipping sample.",
                        sample_id
                    );
                    continue;
                }
                Err(e) => {
                    log::error!(
                        "Error during data generation for sample_id {}: {}. Skipping sample.",
                        sample_id,
                        e
                    );
                    continue;
                }
            };

            for &variant in &[LanczosVariant::Standard, LanczosVariant::TwoPass] {
                log::info!("Spawning worker for variant: {:?}", variant);
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
                    .stdout(Stdio::piped())
                    .stderr(Stdio::inherit())
                    .spawn()
                    .with_context(|| format!("Failed to spawn worker for variant {:?}", variant))?;

                let output = child.wait_with_output()?;
                if !output.status.success() {
                    log::error!("Worker for {:?} failed. Skipping.", variant);
                    continue;
                }

                let mut rdr = csv::ReaderBuilder::new()
                    .has_headers(false)
                    .from_reader(output.stdout.as_slice());
                if let Some(result) = rdr.deserialize::<ScalabilityResult>().next() {
                    all_sample_results.push(result?);
                }
            }
        }

        // --- Aggregation and Serialization ---
        if all_sample_results.is_empty() {
            log::warn!(
                "No successful samples for {} arcs. Skipping aggregation.",
                arcs
            );
            continue;
        }

        for variant_to_process in [LanczosVariant::Standard, LanczosVariant::TwoPass] {
            let samples: Vec<_> = all_sample_results
                .iter()
                .filter(|r| r.variant == variant_to_process)
                .collect();

            if samples.is_empty() {
                log::warn!(
                    "No data for variant {:?} at {} arcs.",
                    variant_to_process,
                    arcs
                );
                continue;
            }

            let times: Vec<f64> = samples.iter().map(|r| r.time_s).collect();
            let rsss: Vec<f64> = samples.iter().map(|r| r.rss_kb as f64).collect();

            let time_data = Data::new(times);
            let rss_data = Data::new(rsss);

            // Handle cases with fewer than 2 samples to avoid NaN.
            // By convention, the standard deviation of a single point is 0.
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
                n: samples[0].n,
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
        "Scalability experiment complete. Results saved to {:?}.",
        &args.output
    );
    Ok(())
}

/// Generates and validates a single test instance for a given sample.
fn generate_and_validate_instance(
    arcs: usize,
    rho: u32,
    instance_dir: &Path,
    instance_id: u32,
) -> Result<Option<PathBuf>> {
    if instance_dir.exists() {
        fs::remove_dir_all(instance_dir)?;
    }
    fs::create_dir_all(instance_dir)?;

    log::info!(
        "Invoking datagen (instance_id: {}) to generate instance...",
        instance_id
    );

    let datagen_status = Command::new("./target/release/datagen")
        .arg("--arcs")
        .arg(arcs.to_string())
        .arg("--rho")
        .arg(rho.to_string())
        .arg("--instance-id")
        .arg(instance_id.to_string())
        .arg("--output-dir")
        .arg(instance_dir)
        .status()
        .context("Failed to execute datagen. Build with `cargo build --release`.")?;

    if !datagen_status.success() {
        log::warn!("Datagen process failed with status: {}", datagen_status);
        return Ok(None);
    }

    let dmx_path = match find_file_by_extension(instance_dir, "dmx") {
        Ok(path) => path,
        Err(_) => {
            log::warn!("Datagen produced no .dmx file.");
            return Ok(None);
        }
    };

    if validate_dmx_file(&dmx_path)? {
        log::info!("Datagen completed with valid output.");
        Ok(Some(instance_dir.to_path_buf()))
    } else {
        log::warn!(
            "Validation failed for {:?}: found invalid node index.",
            dmx_path
        );
        Ok(None)
    }
}

/// Validates a DMX file by checking for invalid (0-based) node indices.
fn validate_dmx_file(path: &Path) -> Result<bool> {
    let file = fs::File::open(path)?;
    for line in BufReader::new(file).lines() {
        let line = line?;
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.first() == Some(&"a") && parts.len() >= 3 && (parts[1] == "0" || parts[2] == "0") {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Utility to find the first file with a given extension in a directory.
fn find_file_by_extension(dir: &Path, ext: &str) -> Result<PathBuf> {
    let entries =
        fs::read_dir(dir).with_context(|| format!("Failed to read directory: {:?}", dir))?;
    for entry in entries {
        let path = entry?.path();
        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some(ext) {
            return Ok(path);
        }
    }
    Err(anyhow!("No .{} file found in directory {:?}", ext, dir))
}

/// Worker logic for a single experimental run.
fn run_worker(variant: LanczosVariant) -> Result<()> {
    let args = WorkerArgs::parse();
    log::info!(
        "Worker for {:?} started on instance {:?}",
        variant,
        &args.instance_dir
    );

    let dmx_path = find_file_by_extension(&args.instance_dir, "dmx")?;
    let qfc_path = find_file_by_extension(&args.instance_dir, "qfc")?;
    let kkt_system = load_kkt_system(dmx_path, qfc_path)?;
    let a: &SparseColMat<usize, f64> = &kkt_system.a;
    let n = a.nrows();

    // Construct a problem with a known ground-truth solution for validation.
    let x_true = Mat::<f64>::from_fn(n, 1, |_, _| 1.0 / (n as f64).sqrt());
    let b = a * &x_true;

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

    log::info!("Worker for {:?} finished.", variant);
    Ok(())
}
