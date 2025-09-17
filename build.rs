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
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=data/");

    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("lanczos_properties_tests.rs");
    let mut file = BufWriter::new(File::create(&dest_path).unwrap());

    let instances = get_all_instances();

    for instance in instances {
        let test_fn_name_base = &instance.name;
        let dmx_path_str = instance.dmx_path.to_str().unwrap();
        let qfc_path_str = instance.qfc_path.to_str().unwrap();

        let test_template = |test_type: &str, runner_fn: &str| {
            format!(
                r#"
#[test]
fn property_{test_type}_{fn_name_base}() -> anyhow::Result<()> {{
    let instance = TestInstance {{
        name: "{name}".to_string(),
        dmx_path: "{dmx_path}".into(),
        qfc_path: "{qfc_path}".into(),
    }};
    {runner_fn}(&instance)
}}
"#,
                test_type = test_type,
                fn_name_base = test_fn_name_base,
                name = instance.name,
                dmx_path = dmx_path_str.escape_default(),
                qfc_path = qfc_path_str.escape_default(),
                runner_fn = runner_fn
            )
        };

        writeln!(
            file,
            "{}",
            test_template(
                "decomposition_consistency",
                "run_decomposition_consistency_test_for_instance"
            )
        )
        .unwrap();
        writeln!(
            file,
            "{}",
            test_template("lanczos_relation", "run_lanczos_relation_test_for_instance")
        )
        .unwrap();
        writeln!(
            file,
            "{}",
            test_template("orthonormality", "run_orthonormality_test_for_instance")
        )
        .unwrap();
        writeln!(
            file,
            "{}",
            test_template(
                "reconstruction_stability",
                "run_reconstruction_stability_test_for_instance"
            )
        )
        .unwrap();
    }
}
