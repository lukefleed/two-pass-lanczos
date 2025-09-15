//! This module provides utilities for loading test problems from files.
//!
//! It includes functions to parse DIMACS format files for network topologies
//! and custom `.qfc` files for quadratic cost coefficients, ultimately
//! assembling them into a complete Karush-Kuhn-Tucker (KKT) system matrix.

use faer::sparse::{SparseColMat, Triplet};
use std::{
    fs::File,
    io::{self, BufRead, BufReader},
    path::Path,
};
use thiserror::Error;

/// Represents all possible errors that can occur during data loading and parsing.
#[derive(Error, Debug)]
pub enum DataLoaderError {
    /// Wraps a standard I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    /// Occurs when a string cannot be parsed into an integer.
    #[error("Parse error: Failed to parse integer from '{0}'")]
    ParseInt(String),
    /// Occurs when a string cannot be parsed into a float.
    #[error("Parse error: Failed to parse float from '{0}'")]
    ParseFloat(String),
    /// Occurs if the DIMACS problem line 'p min ...' is missing or malformed.
    #[error("Format error: The 'p min' problem line was not found or was malformed.")]
    ProblemLineMissing,
    /// Occurs when the end of a file is reached unexpectedly during parsing.
    #[error("Format error: Unexpected end of file while reading data.")]
    UnexpectedEof,
    /// Occurs if the number of arcs in the .dmx and .qfc files do not match.
    #[error("Dimension mismatch: qfc file specifies {qfc_arcs} arcs, but dmx file has {dmx_arcs}.")]
    ArcCountMismatch { qfc_arcs: usize, dmx_arcs: usize },
    /// Occurs if the sparse matrix construction fails internally.
    #[error("Internal error: Failed to construct the sparse matrix from triplets.")]
    SparseMatrixConstructionError,
}

/// A struct containing the fully assembled KKT system and its metadata.
///
/// # Implementation Notes
///
/// We use `usize` for the matrix indices (`I` in `SparseColMat<I, T>`),
/// as it aligns with Rust's memory addressing and is efficient for this use case.
pub struct KKTSystem {
    /// The complete KKT matrix A = [[D, E^T], [E, 0]], stored in a sparse column format.
    pub a: SparseColMat<usize, f64>,
    /// The number of nodes (p) in the network graph.
    pub num_nodes: usize,
    /// The number of arcs (m) in the network graph.
    pub num_arcs: usize,
}

/// Parses a `.dmx` file in DIMACS format to construct the node-arc incidence matrix E.
///
/// # Arguments
/// * `path`: The path to the `.dmx` file.
///
/// # Returns
/// A tuple containing the number of nodes (p), number of arcs (m), and the
/// node-arc incidence matrix E as a sparse matrix.
fn parse_dmx(
    path: impl AsRef<Path>,
) -> Result<(usize, usize, SparseColMat<usize, f64>), DataLoaderError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut num_nodes = 0;
    let mut num_arcs = 0;
    let mut triplets: Vec<Triplet<usize, usize, f64>> = Vec::new();
    let mut arc_counter = 0;
    let mut problem_line_found = false;

    // Read the file line by line, building a list of triplets for the sparse matrix.
    for line in reader.lines() {
        let line = line?;
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        match parts[0] {
            "c" => continue, // Comment line, ignore.
            "p" => {
                if parts.len() >= 4 && parts[1] == "min" {
                    num_nodes = parts[2]
                        .parse::<usize>()
                        .map_err(|_| DataLoaderError::ParseInt(parts[2].to_string()))?;
                    num_arcs = parts[3]
                        .parse::<usize>()
                        .map_err(|_| DataLoaderError::ParseInt(parts[3].to_string()))?;
                    problem_line_found = true;
                } else {
                    return Err(DataLoaderError::ProblemLineMissing);
                }
            }
            "a" => {
                // An arc line defines a column in the incidence matrix E.
                // Node indices in DIMACS are 1-based, so we convert them to 0-based.
                let u: usize = parts[1]
                    .parse::<usize>()
                    .map_err(|_| DataLoaderError::ParseInt(parts[1].to_string()))?
                    - 1;
                let v: usize = parts[2]
                    .parse::<usize>()
                    .map_err(|_| DataLoaderError::ParseInt(parts[2].to_string()))?
                    - 1;

                // For the j-th arc (column), set +1 for the outgoing node u (row u)
                // and -1 for the incoming node v (row v).
                triplets.push(Triplet {
                    row: u,
                    col: arc_counter,
                    val: 1.0,
                });
                triplets.push(Triplet {
                    row: v,
                    col: arc_counter,
                    val: -1.0,
                });
                arc_counter += 1;
            }
            _ => continue,
        }
    }

    if !problem_line_found {
        return Err(DataLoaderError::ProblemLineMissing);
    }

    // In debug builds, validate that the number of arcs found matches the declaration.
    // This is a zero-cost abstraction in release builds.
    debug_assert_eq!(
        arc_counter, num_arcs,
        "The number of parsed arcs does not match the count in the problem line."
    );

    // `try_new_from_triplets` is the most efficient way to build the sparse matrix.
    // We handle the potential error instead of panicking.
    let e_matrix = SparseColMat::try_new_from_triplets(num_nodes, num_arcs, &triplets)
        .map_err(|_| DataLoaderError::SparseMatrixConstructionError)?;

    Ok((num_nodes, num_arcs, e_matrix))
}

/// Parses a `.qfc` file to extract the quadratic cost coefficients.
///
/// # Arguments
/// * `path`: The path to the `.qfc` file.
/// * `expected_arcs`: The number of arcs read from the `.dmx` file, for validation.
///
/// # Returns
/// A vector containing the quadratic cost for each arc, used for the diagonal of D.
fn parse_qfc(path: impl AsRef<Path>, expected_arcs: usize) -> Result<Vec<f64>, DataLoaderError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    // First line must be the number of arcs, used for validation.
    let m_from_file: usize = lines
        .next()
        .ok_or(DataLoaderError::UnexpectedEof)??
        .parse::<usize>()
        .map_err(|_| DataLoaderError::ParseInt("m".to_string()))?;

    if m_from_file != expected_arcs {
        return Err(DataLoaderError::ArcCountMismatch {
            qfc_arcs: m_from_file,
            dmx_arcs: expected_arcs,
        });
    }

    // The quadratic costs start after the 'm' fixed cost lines.
    let quadratic_cost_lines = lines.skip(expected_arcs);

    let mut quadratic_costs = Vec::with_capacity(expected_arcs);
    for line in quadratic_cost_lines.take(expected_arcs) {
        let line = line?;
        let cost: f64 = line
            .parse::<f64>()
            .map_err(|_| DataLoaderError::ParseFloat(line.to_string()))?;
        quadratic_costs.push(cost);
    }

    Ok(quadratic_costs)
}

/// Loads a KKT system from DIMACS and QFC files into a single sparse matrix.
///
/// This is the main entry point of the module. It orchestrates the parsing
/// and assembles the final block KKT matrix `A = [[D, E^T], [E, 0]]`.
///
/// # Arguments
/// * `dmx_path`: Path to the `.dmx` file describing the network topology (matrix E).
/// * `qfc_path`: Path to the `.qfc` file containing cost coefficients (matrix D).
///
/// # Returns
/// A `KKTSystem` struct containing the assembled sparse matrix `A` and its dimensions.
pub fn load_kkt_system(
    dmx_path: impl AsRef<Path>,
    qfc_path: impl AsRef<Path>,
) -> Result<KKTSystem, DataLoaderError> {
    // 1. Parse the .dmx file to get the node-arc incidence matrix E.
    let (num_nodes, num_arcs, e_matrix) = parse_dmx(dmx_path)?;

    // 2. Parse the .qfc file to get the diagonal entries for matrix D.
    let quadratic_costs = parse_qfc(qfc_path, num_arcs)?;

    // 3. Assemble the final KKT matrix A from its blocks.
    let n = num_nodes + num_arcs;
    let mut triplets: Vec<Triplet<usize, usize, f64>> = Vec::new();

    // Add block D (top-left, size m x m).
    for (i, cost) in quadratic_costs.iter().enumerate() {
        triplets.push(Triplet {
            row: i,
            col: i,
            val: *cost,
        });
    }

    // Add blocks E (bottom-left) and E^T (top-right) in a single pass.
    for triplet in e_matrix.triplet_iter() {
        // Add entry for E (size p x m), shifting rows by num_arcs.
        triplets.push(Triplet {
            row: triplet.row + num_arcs,
            col: triplet.col,
            val: *triplet.val,
        });
        // Add entry for E^T (size m x p), swapping row/col and shifting the new column.
        triplets.push(Triplet {
            row: triplet.col,
            col: triplet.row + num_arcs,
            val: *triplet.val,
        });
    }

    // Construct the final sparse matrix from the combined list of triplets.
    let a_matrix = SparseColMat::try_new_from_triplets(n, n, &triplets)
        .map_err(|_| DataLoaderError::SparseMatrixConstructionError)?;

    Ok(KKTSystem {
        a: a_matrix,
        num_nodes,
        num_arcs,
    })
}
