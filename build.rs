// build.rs

use glob::glob;
use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

/// A lightweight version of the TestInstance struct for the build script.
#[derive(Debug)]
struct TestInstance {
    pub name: String,
    pub dmx_path: PathBuf,
    pub qfc_path: PathBuf,
}

/// Discovers all test instances by scanning the `data/` directory.
/// This logic is replicated from the original test file.
fn get_all_instances() -> Vec<TestInstance> {
    let data_dirs = ["data/1000", "data/2000", "data/3000"];
    data_dirs
        .into_iter()
        .flat_map(|dir| glob(&format!("{}/*.dmx", dir)).expect("Failed to read glob pattern"))
        .filter_map(|entry| {
            if let Ok(dmx_path) = entry {
                let qfc_path = dmx_path.with_extension("qfc");
                if qfc_path.exists() {
                    let name = dmx_path
                        .file_stem()
                        .unwrap()
                        .to_string_lossy()
                        .to_string()
                        .replace('-', "_");
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

fn main() {
    // Get the Cargo output directory where we will place the generated code.
    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("correctness_tests.rs");
    let mut file = BufWriter::new(File::create(&dest_path).unwrap());

    // Discover all test instances.
    let instances = get_all_instances();

    // Generate a separate `#[test]` function for each instance.
    for instance in instances {
        let test_fn_name = format!("correctness_test_{}", instance.name);
        let dmx_path_str = instance.dmx_path.to_str().unwrap();
        let qfc_path_str = instance.qfc_path.to_str().unwrap();

        // Write the test function code to the output file.
        writeln!(
            file,
            r#"
#[test]
fn {fn_name}() -> anyhow::Result<()> {{
    // Re-create the TestInstance struct for this specific test.
    let instance = TestInstance {{
        name: "{name}".to_string(),
        dmx_path: "{dmx_path}".into(),
        qfc_path: "{qfc_path}".into(),
    }};
    // Call the main test runner function. A failure will be propagated
    // via the `Result` and cause the test to fail.
    run_correctness_test_for_instance(&instance)
}}
"#,
            fn_name = test_fn_name,
            name = instance.name,
            dmx_path = dmx_path_str.escape_default(),
            qfc_path = qfc_path_str.escape_default()
        )
        .unwrap();
    }
}
