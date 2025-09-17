//! Experiment Runner for the Memory-Computation Trade-off Analysis (Experiment 1).
//!
//! This executable runs a controlled experiment to validate the hypothesis that the
//! two-pass Lanczos method trades a doubled computational cost for a constant
//! memory footprint with respect to the number of iterations `k`.

mod common;

use crate::common::{CommonArgs, get_peak_rss_kb};
use anyhow::{Context, Result, anyhow};
use clap::{Parser, ValueEnum};
use faer::{
    dyn_stack::{MemBuffer, MemStack},
    matrix_free::LinOp,
    prelude::*,
    sparse::SparseColMat,
};
use lanczos_project::{
    solvers::{lanczos, lanczos_two_pass},
    utils::data_loader::load_kkt_system,
};
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Defines the Lanczos algorithm variant to be run.
#[derive(ValueEnum, Clone, Debug, Serialize)]
#[serde(rename_all = "kebab-case")]
enum LanczosVariant {
    /// The standard one-pass algorithm (O(nk) memory).
    Standard,
    /// The memory-efficient two-pass algorithm (O(n) memory).
    TwoPass,
}

/// Command-line arguments specific to the trade-off experiment.
#[derive(Parser, Debug)]
#[clap(
    name = "tradeoff-runner",
    about = "Runs a specified Lanczos method variant for the memory-computation trade-off experiment."
)]
struct TradeoffArgs {
    #[clap(flatten)]
    common: CommonArgs,
    /// The Lanczos algorithm variant to execute.
    #[clap(long, value_enum)]
    variant: LanczosVariant,
    #[clap(long, default_value_t = 50)]
    k_start: usize,
    #[clap(long, default_value_t = 1000)]
    k_end: usize,
    #[clap(long, default_value_t = 50)]
    k_step: usize,
}

/// Represents a single row of the output CSV file.
#[derive(Debug, Serialize)]
struct TradeoffResult {
    variant: LanczosVariant,
    k: usize,
    time_s: f64,
    rss_kb: u64,
}

/// Helper function to find the first file with a given extension in a directory.
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

fn main() -> Result<()> {
    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Info)
        .try_init()
        .map_err(|e| anyhow!("Failed to initialize logger: {}", e))?;

    let args = TradeoffArgs::parse();
    log::info!(
        "Starting trade-off experiment for variant: {:?}",
        args.variant
    );
    log::info!(
        "Parameters: k_start={}, k_end={}, k_step={}",
        args.k_start,
        args.k_end,
        args.k_step
    );

    log::info!(
        "Loading test instance from {:?}...",
        &args.common.instance_dir
    );

    let dmx_path = find_file_by_extension(&args.common.instance_dir, "dmx")?;
    let qfc_path = find_file_by_extension(&args.common.instance_dir, "qfc")?;
    log::info!("Found instance files: {:?} and {:?}", dmx_path, qfc_path);

    let kkt_system = load_kkt_system(dmx_path, qfc_path)?;
    let a: &SparseColMat<usize, f64> = &kkt_system.a;

    let n = a.nrows();
    let x_true = Mat::<f64>::from_fn(n, 1, |_, _| 1.0 / (n as f64).sqrt());
    let b = a * &x_true;

    // Open the CSV writer in append mode if the file already exists,
    // otherwise create it and write the header.
    let output_exists = args.common.output.exists();
    let output_file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .append(output_exists)
        .open(&args.common.output)?;

    let mut writer = csv::WriterBuilder::new()
        .has_headers(!output_exists)
        .from_writer(output_file);

    // A generic solver for the tridiagonal system f(T_k)e_1.
    // This is shared by both Lanczos variants.
    let f_tk_solver = |alphas: &[f64], betas: &[f64]| -> Result<Mat<f64>> {
        let steps = alphas.len();
        if steps == 0 {
            return Ok(Mat::zeros(0, 1));
        }
        let mut t_k = Mat::zeros(steps, steps);
        for i in 0..steps {
            t_k.as_mut()[(i, i)] = alphas[i];
        }
        for i in 0..steps - 1 {
            t_k.as_mut()[(i, i + 1)] = betas[i];
            t_k.as_mut()[(i + 1, i)] = betas[i];
        }
        let mut e1 = Mat::zeros(steps, 1);
        e1.as_mut()[(0, 0)] = 1.0;
        Ok(t_k.as_ref().partial_piv_lu().solve(&e1))
    };

    let mut stack_mem = MemBuffer::new(a.as_ref().apply_scratch(1, Par::Seq));

    for k in (args.k_start..=args.k_end).step_by(args.k_step) {
        log::info!("Running variant {:?} for k = {}...", &args.variant, k);

        let (time_s, rss_kb) = match args.variant {
            LanczosVariant::Standard => {
                let start_time = Instant::now();
                let _ = lanczos(
                    &a.as_ref(),
                    b.as_ref(),
                    k,
                    &mut MemStack::new(&mut stack_mem),
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
                    &mut MemStack::new(&mut stack_mem),
                    &f_tk_solver,
                )?;
                (start_time.elapsed().as_secs_f64(), get_peak_rss_kb())
            }
        };

        writer.serialize(TradeoffResult {
            variant: args.variant.clone(),
            k,
            time_s,
            rss_kb,
        })?;
    }

    writer.flush()?;
    log::info!(
        "Experiment run for variant {:?} complete. Results saved to {:?}",
        &args.variant,
        &args.common.output
    );
    Ok(())
}
