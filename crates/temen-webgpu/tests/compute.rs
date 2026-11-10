//! End-to-end WebGPU compute demo: a guest C program (translated through the LLVM on-ramp) uploads
//! a buffer, runs a WGSL compute shader on the host GPU (lavapipe in CI), reads it back, and checks
//! the result against a CPU reference **in-guest** — on all three Temen engines. The capability is
//! TEMEN-only (no native symbol), so the assertion is the guest's own self-check (`ALL MATCH cpu`),
//! like the async/JIT capability demos.

mod common;

#[test]
fn demo_webgpu_compute() {
    common::require_adapter();
    let inst = common::guest("wgpu_demo");
    for backend in common::ENGINES {
        let caps = [(
            "webgpu",
            temen_webgpu::webgpu_cap(temen_webgpu::Surface::None),
        )];
        let stdout = common::run(&inst, backend, &caps);
        assert!(
            stdout.contains("compute1 (mul-add, 1 buffer): OK")
                && stdout.contains("compute2 (a*a+i, 2 buffers): OK")
                && stdout.contains("webgpu compute: ALL MATCH cpu"),
            "webgpu demo ({backend:?}) self-check failed; stdout:\n{stdout}"
        );
    }
}
