//! Common utilities for data loading and performance measurement.
//!
//! This module provides helper functions used across the experimental binaries.
//! It is organized into two submodules:
//!
//! - **`data_loader`**: Handles parsing of test problem files, specifically the
//!   DIMACS (.dmx) and quadratic cost (.qfc) formats used for min-cost flow problems.
//!   It constructs the full KKT system matrix from these files.
//!
//! - **`perf`**: Contains platform-specific utilities for performance analysis.
//!   Currently, it provides a function to read the peak resident set size (RSS)
//!   on Linux systems, which is crucial for memory usage experiments.
//!

pub mod data_loader;
pub mod perf;
