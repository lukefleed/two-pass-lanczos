//! Experiment Runner for the Memory-Computation Trade-off Analysis (Experiment 1).
//!
//! This executable runs a controlled experiment to validate the hypothesis that the
//! two-pass Lanczos method trades a doubled computational cost for a constant
//! memory footprint with respect to the number of iterations `k`.

mod common;

use crate::common::{CommonArgs, get_peak_rss_kb};
use anyhow::{Context, Result, anyhow};
use clap::Parser;
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

/// Command-line arguments specific to the trade-off experiment.
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

/// Represents a single row of the output CSV file.
#[derive(Debug, Serialize)]
struct TradeoffResult {
    k: usize,
    time_standard_s: f64,
    rss_standard_kb: u64,
    time_two_pass_s: f64,
    rss_two_pass_kb: u64,
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
    log::info!("Starting trade-off experiment...");
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

    // --- Corrected File Loading Logic ---
    let dmx_path = find_file_by_extension(&args.common.instance_dir, "dmx")?;
    let qfc_path = find_file_by_extension(&args.common.instance_dir, "qfc")?;
    log::info!("Found instance files: {:?} and {:?}", dmx_path, qfc_path);

    let kkt_system = load_kkt_system(dmx_path, qfc_path)?;
    let a: &SparseColMat<usize, f64> = &kkt_system.a;

    let n = a.nrows();
    let x_true = Mat::<f64>::from_fn(n, 1, |_, _| 1.0 / (n as f64).sqrt());
    let b = a * &x_true;

    let mut writer = csv::Writer::from_path(&args.common.output)?;

    for k in (args.k_start..=args.k_end).step_by(args.k_step) {
        log::info!("Running for k = {}...", k);

        let mut stack_mem = MemBuffer::new(a.as_ref().apply_scratch(1, Par::Seq));

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

        let start_std = Instant::now();
        let _ = lanczos(
            &a.as_ref(),
            b.as_ref(),
            k,
            &mut MemStack::new(&mut stack_mem),
            &f_tk_solver,
        )?;
        let time_standard_s = start_std.elapsed().as_secs_f64();
        let rss_standard_kb = get_peak_rss_kb();

        let start_2p = Instant::now();
        let _ = lanczos_two_pass(
            &a.as_ref(),
            b.as_ref(),
            k,
            &mut MemStack::new(&mut stack_mem),
            &f_tk_solver,
        )?;
        let time_two_pass_s = start_2p.elapsed().as_secs_f64();
        let rss_two_pass_kb = get_peak_rss_kb();

        writer.serialize(TradeoffResult {
            k,
            time_standard_s,
            rss_standard_kb,
            time_two_pass_s,
            rss_two_pass_kb,
        })?;
    }

    writer.flush()?;
    log::info!(
        "Experiment complete. Results saved to {:?}",
        &args.common.output
    );
    Ok(())
}
