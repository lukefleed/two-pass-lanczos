//! A data generation utility for creating KKT system test instances.
//!
//! This binary acts as a command-line driven wrapper that orchestrates a
//! three-stage generation pipeline using external, pre-compiled tools.
//! It executes `pargen`, `netgen`, and `qfcgen` in sequence to produce
//! a complete KKT test instance, consisting of a `.dmx` (topology) and a `.qfc` (cost) file.

use anyhow::{Context, Result, anyhow, ensure};
use clap::{Parser, ValueEnum};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Defines the cost level for fixed and quadratic costs, corresponding to the
/// character arguments required by the external `pargen` utility.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
enum CostLevel {
    /// Low costs (maps to 'b' in the `pargen` interface).
    #[clap(name = "b")]
    Low,
    /// High costs (maps to 'a' in the `pargen` interface).
    #[clap(name = "a")]
    High,
}

impl CostLevel {
    /// Returns the single-character string representation required by `pargen`.
    fn to_arg_str(&self) -> &'static str {
        match self {
            CostLevel::Low => "b",
            CostLevel::High => "a",
        }
    }
}

/// Defines whether to scale arc capacities, corresponding to the string arguments
/// required by the `pargen` utility.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
enum Scaling {
    /// Scaled capacities (maps to 's').
    #[clap(name = "s")]
    Scaled,
    /// Not scaled capacities (maps to 'ns').
    #[clap(name = "ns")]
    NotScaled,
}

impl Scaling {
    /// Returns the string representation required by `pargen`.
    fn to_arg_str(&self) -> &'static str {
        match self {
            Scaling::Scaled => "s",
            Scaling::NotScaled => "ns",
        }
    }
}

/// Command-line interface for the data generation orchestrator.
///
/// This utility orchestrates the pargen -> netgen -> qfcgen pipeline to generate
/// reproducible KKT test instances for min-cost flow problems.
#[derive(Parser, Debug)]
#[clap(
    name = "datagen",
    about = "Orchestrates the pargen -> netgen -> qfcgen pipeline to generate test instances."
)]
struct DataGenArgs {
    /// The number of arcs in the generated network graph.
    #[clap(long)]
    arcs: u32,
    /// A structural parameter (`rho`) for the network generator, controlling topology.
    #[clap(long, value_parser = clap::value_parser!(u32).range(1..=3))]
    rho: u32,
    /// A unique identifier for the instance, used as a seed for the random generators.
    #[clap(long, default_value_t = 1)]
    instance_id: u32,
    /// The level of fixed costs for the arcs.
    #[clap(long, value_enum, default_value_t = CostLevel::High)]
    fixed_cost: CostLevel,
    /// The level of quadratic costs for the arcs.
    #[clap(long, value_enum, default_value_t = CostLevel::High)]
    quadratic_cost: CostLevel,
    /// Specifies whether arc capacities should be scaled.
    #[clap(long, value_enum, default_value_t = Scaling::NotScaled)]
    scaling: Scaling,
    /// The directory where the final instance files (.dmx, .qfc) will be saved.
    #[clap(long)]
    output_dir: PathBuf,
}

fn main() -> Result<()> {
    env_logger::Builder::new()
        .filter_level(log::LevelFilter::Info)
        .try_init()
        .map_err(|e| anyhow!("Failed to initialize logger: {}", e))?;

    let args = DataGenArgs::parse();
    log::info!(
        "Starting data generation pipeline with parameters: {:?}",
        &args
    );

    std::fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("Failed to create output directory: {:?}", &args.output_dir))?;

    // Construct the base filename following the exact convention of the external tools.
    // This is crucial for ensuring that `qfcgen` can find the output from `netgen`.
    let base_name = format!(
        "netgen-{}-{}-{}-{}-{}-{}",
        args.arcs,
        args.rho,
        args.instance_id,
        args.fixed_cost.to_arg_str(),
        args.quadratic_cost.to_arg_str(),
        args.scaling.to_arg_str(),
    );

    let par_path = args.output_dir.join(format!("{}.par", base_name));
    let dmx_path = args.output_dir.join(format!("{}.dmx", base_name));
    let qfc_path = args.output_dir.join(format!("{}.qfc", base_name));

    // Execute the three stages of the pipeline in sequence.
    run_pargen(&args, &par_path)?;
    run_netgen(&par_path, &dmx_path)?;
    run_qfcgen(&dmx_path, &qfc_path)?;

    log::info!("Data generation pipeline completed successfully.");
    log::info!("Generated files:\n  - {:?}\n  - {:?}", dmx_path, qfc_path);

    Ok(())
}

/// Pipeline Stage 1: Executes the `pargen` command to generate a parameter file.
/// This `.par` file serves as the input for the `netgen` utility.
fn run_pargen(args: &DataGenArgs, par_path: &Path) -> Result<()> {
    log::info!("Step 1: Running pargen to create parameter file...");
    let pargen_exe = std::fs::canonicalize("data/qcnd/pargen").context(
        "Could not find 'data/qcnd/pargen'. Make sure you are running from the project root.",
    )?;

    let pargen_status = Command::new(pargen_exe)
        .arg(args.arcs.to_string())
        .arg(args.rho.to_string())
        .arg(args.instance_id.to_string())
        .arg(args.fixed_cost.to_arg_str())
        .arg(args.quadratic_cost.to_arg_str())
        .arg(args.scaling.to_arg_str())
        .current_dir(&args.output_dir)
        .status()
        .context("Failed to execute `pargen`.")?;

    ensure!(
        pargen_status.success(),
        "`pargen` command failed with status: {}",
        pargen_status
    );
    ensure!(
        par_path.exists(),
        "pargen ran but did not create the expected .par file at {:?}",
        par_path
    );
    log::info!("Successfully created parameter file: {:?}", par_path);
    Ok(())
}

/// Pipeline Stage 2: Executes the `netgen` command to generate the network topology.
/// This tool reads from the `.par` file via stdin and writes the graph in DIMACS
/// format (`.dmx`) to stdout.
fn run_netgen(par_path: &Path, dmx_path: &Path) -> Result<()> {
    log::info!("Step 2: Running netgen to generate graph topology...");
    // Open files for I/O redirection.
    let par_file = File::open(par_path)
        .with_context(|| format!("Failed to open parameter file for reading: {:?}", par_path))?;
    let dmx_file = File::create(dmx_path)
        .with_context(|| format!("Failed to create DMX output file at: {:?}", dmx_path))?;

    let netgen_exe = std::fs::canonicalize("data/netgen/src/netgen").context(
        "Could not find 'data/netgen/netgen'. Make sure you are running from the project root.",
    )?;

    // Execute the command, redirecting standard input and output.
    let netgen_status = Command::new(netgen_exe)
        .stdin(Stdio::from(par_file))
        .stdout(Stdio::from(dmx_file))
        .stderr(Stdio::inherit()) // Pass through stderr for debugging.
        .status()
        .context("Failed to execute `netgen`.")?;

    ensure!(
        netgen_status.success(),
        "`netgen` command failed with status: {}",
        netgen_status
    );
    log::info!("Successfully generated DMX file: {:?}", dmx_path);
    Ok(())
}

/// Pipeline Stage 3: Executes the `qfcgen` command to generate the cost file.
/// This tool reads the `.dmx` file and generates a corresponding `.qfc` file
/// containing the fixed and quadratic cost coefficients.
fn run_qfcgen(dmx_path: &Path, qfc_path: &Path) -> Result<()> {
    log::info!("Step 3: Running qfcgen to generate cost file...");

    // `qfcgen` requires the filename as an argument and must be run in the
    // same directory as the input file.
    let dmx_filename = dmx_path.file_name().unwrap().to_str().unwrap();
    let working_dir = dmx_path.parent().unwrap();

    let qfcgen_exe = std::fs::canonicalize("data/qcnd/qfcgen").context(
        "Could not find 'data/qcnd/qfcgen'. Make sure you are running from the project root.",
    )?;

    let qfcgen_status = Command::new(qfcgen_exe)
        .arg(dmx_filename)
        .current_dir(working_dir)
        .status()
        .context("Failed to execute `qfcgen`.")?;

    ensure!(
        qfcgen_status.success(),
        "`qfcgen` command failed with status: {}",
        qfcgen_status
    );
    ensure!(
        qfc_path.exists(),
        "qfcgen ran but did not create the expected .qfc file at {:?}",
        qfc_path
    );

    log::info!("Successfully generated QFC file: {:?}", qfc_path);
    Ok(())
}
