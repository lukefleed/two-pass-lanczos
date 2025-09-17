//! This module provides shared utilities for the experiment runner executables.
//! It includes common command-line argument definitions and helper functions
//! for performance metric collection.

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

/// Reads the peak resident set size (VmPeak) from /proc/self/status on Linux.
///
/// This provides a reliable way to measure the maximum memory usage of the process
/// for a given experiment run. The implementation is platform-specific.
///
/// # Returns
/// The peak memory usage in kilobytes (KB), or 0 if the value cannot be read
/// or the platform is not Linux.
#[cfg(target_os = "linux")]
pub fn get_peak_rss_kb() -> u64 {
    let status_content = match std::fs::read_to_string("/proc/self/status") {
        Ok(content) => content,
        Err(_) => return 0,
    };

    for line in status_content.lines() {
        if line.starts_with("VmPeak:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if let Some(value_str) = parts.get(1) {
                return value_str.parse().unwrap_or(0);
            }
        }
    }
    0
}

/// A dummy implementation for non-Linux platforms to ensure the code compiles.
#[cfg(not(target_os = "linux"))]
pub fn get_peak_rss_kb() -> u64 {
    // We log a warning once to inform the user.
    use std::sync::Once;
    static WARN_ONCE: Once = Once::new();
    WARN_ONCE.call_once(|| {
        log::warn!("Peak RSS measurement is only supported on Linux; returning 0.");
    });
    0
}
