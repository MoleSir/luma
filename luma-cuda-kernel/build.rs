use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let kernels: &[(&str, &str)] = &[
        ("binary", "BINARY"),
        ("unary", "UNARY"),
        ("reduce", "REDUCE"),
        ("cast", "CAST"),
        ("binary_scalar", "BINARY_SCALAR"),
        ("copy", "COPY"),
        ("indexing", "INDEXING"),
        ("pick", "PICK"),
        ("allclose", "ALLCLOSE"),
        ("nn", "NN"),
    ];

    if cuda_available() {
        let builder = bindgen_cuda::Builder::default()
            .compute_cap(52)
            .include_paths(vec!["./kernels/utils.cuh"])
            .kernel_paths(kernels.iter().map(|(name, _)| format!("./kernels/{name}.cu")).collect());
        println!("cargo:info={builder:?}");
        let bindings = builder.build_ptx().unwrap();
        bindings.write("src/ptx.rs").unwrap();
    } else {
        println!("cargo:warning=CUDA not found, building with stub PTX (GPU support disabled)");

        let out_dir = env::var("OUT_DIR").unwrap();

        // Generate empty PTX stub files
        for (name, _) in kernels {
            let ptx_path = PathBuf::from(&out_dir).join(format!("{name}.ptx"));
            fs::write(&ptx_path, "").unwrap();
        }

        // Generate ptx.rs referencing the stub PTX files
        let ptx_rs: String = kernels
            .iter()
            .map(|(name, const_name)| {
                format!("pub const {const_name}: &str = include_str!(concat!(env!(\"OUT_DIR\"), \"/{name}.ptx\"));\n")
            })
            .collect();
        fs::write("src/ptx.rs", ptx_rs).unwrap();
    }
}

fn cuda_available() -> bool {
    // Check CUDA_PATH environment variable
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        if PathBuf::from(&cuda_path).exists() {
            return true;
        }
    }

    // Check common CUDA installation paths
    for candidate in &["/usr/local/cuda", "/opt/cuda", "/usr/lib/cuda"] {
        if PathBuf::from(candidate).exists() {
            return true;
        }
    }

    // Check if nvcc is available in PATH
    Command::new("nvcc")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}
