//! This module provides shared utilities for the experiment runner executables.
//! It includes common command-line argument definitions.

use clap::Parser;
use std::path::PathBuf;

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
