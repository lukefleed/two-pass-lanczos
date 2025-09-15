//! Integration tests for verifying the mathematical correctness of the Lanczos algorithms.
//!
//! These tests load real problem instances from the commalab dataset and verify that
//! the outputs of the implemented algorithms adhere to the theoretical properties
//! of the symmetric Lanczos process. Each assertion is a direct verification of a
//! key mathematical identity derived in the theory. A macro-based approach is used
//! to easily run the same suite of tests across multiple data files.

use anyhow::Result;
use faer::{
    Mat,
    dyn_stack::{MemBuffer, MemStack},
    matrix_free::LinOp,
};
use lanczos_project::{
    algorithms::lanczos::{lanczos_pass_one, lanczos_pass_two, lanczos_standard},
    utils::data_loader::load_kkt_system,
};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::path::Path;

// A tight tolerance for floating-point comparisons in correctness tests,
// reflecting expectations within machine precision.
const TOLERANCE: f64 = 1e-12;

/// A helper struct to hold paths to a complete test instance.
struct TestInstance {
    name: &'static str,
    dmx_path: &'static str,
    qfc_path: &'static str,
}

// A comprehensive list of test instances. This array can be easily expanded
// to increase test coverage across different problem sizes and structures.
const TEST_INSTANCES: &[TestInstance] = &[
    TestInstance {
        name: "1000-1-1-b-b-ns",
        dmx_path: "data/1000/netgen-1000-1-1-b-b-ns.dmx",
        qfc_path: "data/1000/netgen-1000-1-1-b-b-ns.qfc",
    },
    TestInstance {
        name: "2000-2-3-a-b-s",
        dmx_path: "data/2000/netgen-2000-2-3-a-b-s.dmx",
        qfc_path: "data/2000/netgen-2000-2-3-a-b-s.qfc",
    },
    TestInstance {
        name: "3000-3-5-b-a-ns",
        dmx_path: "data/3000/netgen-3000-3-5-b-a-ns.dmx",
        qfc_path: "data/3000/netgen-3000-3-5-b-a-ns.qfc",
    },
];

/// Generates a random vector of dimension `n` for testing purposes.
fn random_vector(n: usize, seed: u64) -> Mat<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    // Explicitly specify the type for `gen` to resolve ambiguity with the Rust 2024 `gen` keyword.
    Mat::from_fn(n, 1, |_, _| rng.random())
}

/// Macro to generate a full suite of correctness tests for a given instance.
/// This avoids code duplication and allows for easy expansion of test cases.
macro_rules! generate_correctness_test {
    ($test_name:ident, $instance:expr) => {
        #[test]
        fn $test_name() -> Result<()> {
            let instance = $instance;
            let k = 20; // A moderate number of iterations for all tests.

            // --- Test 1: Decomposition and Reconstruction Consistency ---
            {
                let kkt_system =
                    load_kkt_system(Path::new(instance.dmx_path), Path::new(instance.qfc_path))?;
                let a = kkt_system.a;
                let n = a.nrows();
                let b = random_vector(n, 42);

                let mut mem = MemBuffer::new(a.as_ref().apply_scratch(k, faer::Par::Seq));
                let mut stack = MemStack::new(&mut mem);

                let standard_output =
                    lanczos_standard(&a.as_ref(), b.as_ref(), k, &mut stack, None)?;
                let v_k_ref = standard_output.v_k;
                let decomp_ref = standard_output.decomposition;

                let pass_one_decomp = lanczos_pass_one(&a.as_ref(), b.as_ref(), k, &mut stack)?;

                assert_eq!(
                    decomp_ref.steps_taken, pass_one_decomp.steps_taken,
                    "Mismatch in steps_taken on instance '{}'",
                    instance.name
                );
                assert!(
                    (decomp_ref.b_norm - pass_one_decomp.b_norm).abs() < TOLERANCE,
                    "Mismatch in b_norm on instance '{}'",
                    instance.name
                );

                let y_k = random_vector(decomp_ref.steps_taken, 99);
                let x_k_expected = &v_k_ref * &y_k;
                let x_k_reconstructed = lanczos_pass_two(
                    &a.as_ref(),
                    b.as_ref(),
                    &pass_one_decomp,
                    y_k.as_ref(),
                    &mut stack,
                )?;
                let reconstruction_error = (&x_k_expected - &x_k_reconstructed).norm_l2();
                assert!(
                    reconstruction_error < TOLERANCE,
                    "Reconstruction error on instance '{}' is too high: {}",
                    instance.name,
                    reconstruction_error
                );
            }

            // --- Test 2: Mathematical Properties ---
            {
                let kkt_system =
                    load_kkt_system(Path::new(instance.dmx_path), Path::new(instance.qfc_path))?;
                let a = kkt_system.a;
                let n = a.nrows();
                let b = random_vector(n, 123);

                let mut mem = MemBuffer::new(a.as_ref().apply_scratch(k + 1, faer::Par::Seq));
                let mut stack = MemStack::new(&mut mem);

                let standard_output =
                    lanczos_standard(&a.as_ref(), b.as_ref(), k, &mut stack, None)?;
                let v_k = standard_output.v_k;
                let decomp = standard_output.decomposition;
                let steps = decomp.steps_taken;

                let identity = Mat::<f64>::identity(steps, steps);
                let v_k_t_v_k = v_k.as_ref().adjoint() * v_k.as_ref();
                let ortho_error = (&identity - &v_k_t_v_k).norm_l2();
                assert!(
                    ortho_error < TOLERANCE,
                    "Basis on instance '{}' is not orthonormal. Error norm: {}",
                    instance.name,
                    ortho_error
                );

                if steps == k {
                    let next_step_output =
                        lanczos_standard(&a.as_ref(), b.as_ref(), k + 1, &mut stack, None)?;
                    if next_step_output.decomposition.steps_taken > k {
                        let v_k_plus_1 = next_step_output.v_k.as_ref().col(k);
                        let beta_k = next_step_output.decomposition.betas[k - 1];

                        let mut t_k = Mat::zeros(k, k);
                        for i in 0..k {
                            t_k[(i, i)] = decomp.alphas[i];
                        }
                        for i in 0..k - 1 {
                            t_k[(i, i + 1)] = decomp.betas[i];
                            t_k[(i + 1, i)] = decomp.betas[i];
                        }

                        let residual_matrix = &a * &v_k - &v_k * &t_k;
                        let mut e_k_t = Mat::zeros(1, k);
                        e_k_t[(0, k - 1)] = 1.0;
                        let expected_residual = (v_k_plus_1 * e_k_t.as_ref().row(0)) * beta_k;
                        let relation_error = (&residual_matrix - &expected_residual).norm_l2();
                        assert!(
                            relation_error < TOLERANCE,
                            "Lanczos relation on instance '{}' does not hold. Error norm: {}",
                            instance.name,
                            relation_error
                        );
                    }
                }
            }
            Ok(())
        }
    };
}

// Generate the tests for each instance defined in the constant array.
generate_correctness_test!(correctness_test_1000_grid, &TEST_INSTANCES[0]);
generate_correctness_test!(correctness_test_2000_rnd, &TEST_INSTANCES[1]);
generate_correctness_test!(correctness_test_3000_grid, &TEST_INSTANCES[2]);
