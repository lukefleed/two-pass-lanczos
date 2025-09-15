//! Integration test suite for verifying the mathematical correctness of the Lanczos algorithms.
//!
//! This suite uses the `rstest` crate to automatically discover all test instances
//! in the `data/` directory and generate a distinct test case for each one. This
//! provides granular feedback and avoids manual maintenance of the test list.

use anyhow::{Result, ensure};
use faer::{
    Mat, Par,
    dyn_stack::{MemBuffer, MemStack},
    matrix_free::LinOp,
};
use glob::glob;
use lanczos_project::{
    algorithms::lanczos::{lanczos_pass_one, lanczos_pass_two, lanczos_standard},
    utils::data_loader::load_kkt_system,
};
use rand::{Rng, SeedableRng, rngs::StdRng};
use rstest::rstest;
use std::path::PathBuf;

// A tolerance for floating-point comparisons in correctness tests.
const TOLERANCE: f64 = 1e-12;

/// A helper struct to hold paths to a complete test instance.
#[derive(Debug)]
pub struct TestInstance {
    pub name: String,
    pub dmx_path: PathBuf,
    pub qfc_path: PathBuf,
}

/// A test case generator function for `rstest`.
///
/// This function is executed by `rstest` at test collection time. It scans the
/// `data/` directory for all pairs of `.dmx` and `.qfc` files and returns a
/// Vec of `TestInstance`, which then becomes the source of parameters for the test.
fn get_all_instances() -> Vec<TestInstance> {
    let data_dirs = ["data/1000", "data/2000", "data/3000"];
    data_dirs
        .into_iter()
        .flat_map(|dir| glob(&format!("{}/*.dmx", dir)).expect("Failed to read glob pattern"))
        .filter_map(|entry| {
            if let Ok(dmx_path) = entry {
                let qfc_path = dmx_path.with_extension("qfc");
                if qfc_path.exists() {
                    let name = dmx_path.file_stem().unwrap().to_string_lossy().to_string();
                    return Some(TestInstance {
                        name,
                        dmx_path,
                        qfc_path,
                    });
                }
            }
            None
        })
        .collect()
}

/// The main correctness test, parametrized by `rstest`.

#[rstest]
#[case::all_instances(get_all_instances())]
fn test_lanczos_correctness(#[case] instances: Vec<TestInstance>) -> Result<()> {
    for instance in instances {
        test_single_instance(instance)?;
    }
    Ok(())
}

fn test_single_instance(instance: TestInstance) -> Result<()> {
    // Number of Lanczos iterations to perform.
    let k = 30;

    // Load the matrix from the instance files.
    let kkt_system = load_kkt_system(&instance.dmx_path, &instance.qfc_path)?;
    let a = kkt_system.a;
    let n = a.nrows();

    // Create a reproducible random vector.
    let mut rng = StdRng::seed_from_u64(42);
    let b = Mat::from_fn(n, 1, |_, _| rng.random());

    // Allocate memory for faer operations.
    let mut mem = MemBuffer::new(a.as_ref().apply_scratch(k + 1, Par::Seq));
    let mut stack = MemStack::new(&mut mem);

    // --- Verification 1: Consistency between one-pass and two-pass decomposition ---
    let standard_output = lanczos_standard(&a.as_ref(), b.as_ref(), k, &mut stack, None)?;
    let pass_one_output = lanczos_pass_one(&a.as_ref(), b.as_ref(), k, &mut stack)?;
    ensure!(
        standard_output.decomposition.alphas == pass_one_output.alphas,
        "Alphas mismatch on instance '{}'",
        instance.name
    );
    ensure!(
        standard_output.decomposition.betas == pass_one_output.betas,
        "Betas mismatch on instance '{}'",
        instance.name
    );

    // --- Verification 2: Reconstruction correctness ---
    let steps = standard_output.decomposition.steps_taken;
    let y_k = Mat::from_fn(steps, 1, |_, _| rng.random());
    let x_k_expected = &standard_output.v_k * &y_k;
    let x_k_reconstructed = lanczos_pass_two(
        &a.as_ref(),
        b.as_ref(),
        &pass_one_output,
        y_k.as_ref(),
        &mut stack,
    )?;
    let recon_error = (&x_k_expected - &x_k_reconstructed).norm_l2();
    ensure!(
        recon_error < TOLERANCE,
        "Reconstruction error on instance '{}' is too high: {}",
        instance.name,
        recon_error
    );

    // --- Verification 3: Orthonormality of the basis ---
    let v_k = standard_output.v_k.as_ref();
    let identity = Mat::<f64>::identity(steps, steps);
    let ortho_error = (&identity - v_k.adjoint() * v_k).norm_l2();
    ensure!(
        ortho_error < TOLERANCE,
        "Basis on instance '{}' is not orthonormal. Error norm: {}",
        instance.name,
        ortho_error
    );

    Ok(())
}
