//! What the GPU tests share: building a C guest through the LLVM on-ramp, and running it on each
//! engine. A host with no adapter fails every test, saying what to install, rather than skipping
//! it (#2665): a GPU test that passes because nothing ran is no test.

use std::path::PathBuf;
use std::process::Command;

pub const ENGINES: [temen_run::Backend; 3] = [
    temen_run::Backend::TreeWalk,
    temen_run::Backend::Bytecode,
    temen_run::Backend::Jit,
];

/// Fails the test unless this host has a WebGPU adapter.
pub fn require_adapter() {
    assert!(
        temen_webgpu::adapter_available(),
        "no WebGPU adapter: install a Vulkan driver (lavapipe: `mesa-vulkan-drivers`)"
    );
}

/// The demo `crates/temen-run/demos/webgpu/<name>.c`, built for temen and instantiated.
pub fn guest(name: &str) -> temen_run::Instance {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let demo = root.join(format!("../temen-run/demos/webgpu/{name}.c"));
    let ll = std::env::temp_dir().join(format!("temen_webgpu_{name}_{}.ll", std::process::id()));
    let status = Command::new("clang")
        .args([
            "-O2",
            "-emit-llvm",
            "-S",
            "-DTEMEN_GUEST",
            "-fno-vectorize",
            "-fno-slp-vectorize",
        ])
        .arg(format!("-I{}", root.join("../temen-gpu").display()))
        .arg(&demo)
        .arg("-o")
        .arg(&ll)
        .status()
        .expect("clang, to build the GPU demos");
    assert!(status.success(), "clang failed on {}", demo.display());
    let t = temen_llvm::translate_ll_path(&ll).expect("translate");
    let _ = std::fs::remove_file(&ll);
    temen_run::instantiate(t.module).expect("instantiate")
}

/// Runs `inst` on `backend` with `caps`, and returns its stdout.
pub fn run(
    inst: &temen_run::Instance,
    backend: temen_run::Backend,
    caps: &[(&str, temen_run::HostCap)],
) -> String {
    let out = inst
        .run_with_caps(backend, &temen_run::RunConfig::default(), caps)
        .unwrap_or_else(|e| panic!("run ({backend:?}): {e}"));
    String::from_utf8_lossy(&out.stdout).into_owned()
}
