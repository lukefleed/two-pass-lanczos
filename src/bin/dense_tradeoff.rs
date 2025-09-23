//! Dense matrix performance analysis for the memory-computation trade-off.
//!
//! This executable serves as a "proof-of-concept" to validate the performance
//! analysis from the main report. It measures the execution time of both Lanczos
//! variants on a dense matrix, where the O(n^2) cost of the matrix-vector product
//! is expected to be the dominant computational factor. This helps confirm that
//! the performance characteristics observed for sparse matrices are attributable
//! to memory access patterns and cache effects.

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
    utils::perf::get_peak_rss_kb,
};
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use std::{
    process::{Command, Stdio},
    time::Instant,
};

/// Environment variable used for orchestrator/worker process differentiation.
const VARIANT_ENV_VAR: &str = "LANCZOS_DENSE_EXPERIMENT_VARIANT";

/// Defines the Lanczos algorithm variant to be executed in a worker process.
#[derive(ValueEnum, Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
enum LanczosVariant {
    Standard,
    TwoPass,
}

/// Command-line arguments for the dense matrix trade-off experiment orchestrator.
#[derive(Parser, Debug)]
#[clap(
    name = "dense-tradeoff-runner",
    about = "Runs the memory-computation trade-off experiment for Lanczos methods on a dense matrix."
)]
struct DenseTradeoffArgs {
    /// Dimension of the dense square matrix.
    #[clap(long)]
    n: usize,
    /// The starting number of Lanczos iterations.
    #[clap(long, default_value_t = 50)]
    k_start: usize,
    /// The ending number of Lanczos iterations.
    #[clap(long, default_value_t = 1000)]
    k_end: usize,
    /// The step size for increasing the number of iterations.
    #[clap(long, default_value_t = 50)]
    k_step: usize,
    /// Path to the output CSV file where results will be written.
    #[clap(long)]
    output: String,
}

/// Represents a single row of data produced by a worker process.
#[derive(Debug, Serialize, Deserialize)]
struct DenseTradeoffResult {
    variant: LanczosVariant,
    k: usize,
    time_s: f64,
    rss_kb: u64,
}

/// The main entry point, dispatching to orchestrator or worker logic based on an environment variable.
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

/// Orchestrator logic: spawns worker processes for each variant and collects results.
fn run_orchestrator() -> Result<()> {
    let args = DenseTradeoffArgs::parse();
    log::info!("Orchestrator starting dense matrix experiment...");

    let variants_to_run = [LanczosVariant::Standard, LanczosVariant::TwoPass];
    let mut child_handles = Vec::new();

    for variant in &variants_to_run {
        log::info!("Spawning worker for variant: {variant:?}");
        let current_exe = std::env::current_exe()?;
        let child = Command::new(current_exe)
            .args(std::env::args_os().skip(1))
            .env(
                VARIANT_ENV_VAR,
                variant.to_possible_value().unwrap().get_name(),
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
            let record: DenseTradeoffResult = result?;
            all_results.push(record);
        }
    }

    log::info!(
        "All workers finished. Consolidating results into {}...",
        &args.output
    );
    let mut writer = csv::Writer::from_path(&args.output)?;
    for record in all_results {
        writer.serialize(record)?;
    }
    writer.flush()?;

    log::info!("Dense matrix experiment complete.");
    Ok(())
}

/// Worker logic: generates a dense matrix and runs a single Lanczos variant.
fn run_worker(variant: &LanczosVariant) -> Result<()> {
    let args = DenseTradeoffArgs::parse();
    log::info!("Worker for {variant:?} started for n={}.", args.n);

    // 1. Generate a dense, symmetric, random matrix A.
    // We use a fixed seed for reproducibility.
    let mut rng = StdRng::seed_from_u64(42);
    let b_rand = Mat::from_fn(args.n, args.n, |_, _| rng.random::<f64>());
    let a = &b_rand + b_rand.transpose();

    // 2. Create a known-solution problem: b = A * x_true.
    let x_true = Mat::<f64>::from_fn(args.n, 1, |_, _| 1.0 / (args.n as f64).sqrt());
    let b = &a * &x_true;

    // 3. Define the solver for the projected problem f(T_k) = T_k^-1.
    let f_tk_solver = |alphas: &[f64], betas: &[f64]| -> Result<Mat<f64>> {
        let t_k_sparse = assemble_tridiagonal_sparse(alphas, betas)?;
        if t_k_sparse.nrows() == 0 {
            return Ok(Mat::zeros(0, 1));
        }
        let mut e1 = Mat::zeros(t_k_sparse.nrows(), 1);
        e1.as_mut()[(0, 0)] = 1.0;
        Ok(t_k_sparse.as_ref().sp_lu()?.solve(e1.as_ref()))
    };

    let mut writer = csv::WriterBuilder::new()
        .has_headers(false)
        .from_writer(std::io::stdout());
    let mut stack_mem = MemBuffer::new(a.as_ref().apply_scratch(1, faer::Par::Seq));

    // 4. Iterate over k, run the specified algorithm, and record performance.
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

        writer.serialize(DenseTradeoffResult {
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
