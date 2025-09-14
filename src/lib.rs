//! # Two-Pass Lanczos Library

//! A library for the two-pass Lanczos method for computing matrix functions.
//! This crate provides the core algorithms and data structure abstractions.

// Declare the modules that form the public API of the crate.
pub mod algorithms;
pub mod error;
pub mod matrix;

// Re-export key types to the top level of the crate for easier access.
pub use error::LanczosError;
pub use matrix::LinearOperator;
