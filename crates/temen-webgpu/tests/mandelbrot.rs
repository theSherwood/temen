//! WebGPU **compute → image → PNG** demo (LLVM.md WebGPU demo 3). A guest C program (translated
//! through the LLVM on-ramp) runs a WGSL Mandelbrot compute shader on the host GPU (lavapipe in CI),
//! reads the RGBA image back, self-validates it (exact top/bottom mirror symmetry — a
//! float-implementation-independent invariant — plus an in-set/escaped sanity pair), and writes the
//! raw pixels out through the granted `fs` capability. The test drives it on all three Temen engines,
//! asserts the guest's self-check, and encodes the pixels to a PNG (exercising the readback → image
//! path). Exercises the `webgpu` and `fs` capabilities together. Fails with no GPU adapter.

mod common;

const W: u32 = 320;
const H: u32 = 240;

#[test]
fn demo_webgpu_mandelbrot() {
    common::require_adapter();
    let inst = common::guest("mandelbrot");
    for backend in common::ENGINES {
        let root = std::env::temp_dir().join(format!(
            "temen_webgpu_mandel_{}_{:?}",
            std::process::id(),
            backend
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("out dir");
        let caps = [
            (
                "webgpu",
                temen_webgpu::webgpu_cap(temen_webgpu::Surface::None),
            ),
            ("fs", temen_run::fs::host_fs(root.clone())),
        ];
        let stdout = common::run(&inst, backend, &caps);
        assert!(
            stdout.contains("symmetric=1 center_in=1 corner_out=1")
                && stdout.contains("webgpu mandelbrot: ALL OK"),
            "mandelbrot ({backend:?}) self-check failed; stdout:\n{stdout}"
        );

        // The guest wrote the raw RGBA through the fs cap; encode a PNG to prove the readback → image
        // path (and that the file landed on disk through the capability).
        let rgba = std::fs::read(root.join("mandel.rgba")).expect("guest-written mandel.rgba");
        assert_eq!(rgba.len(), (W * H * 4) as usize, "rgba size");
        let png = root.join("mandelbrot.png");
        image::save_buffer(&png, &rgba, W, H, image::ExtendedColorType::Rgba8).expect("encode png");
        assert!(std::fs::metadata(&png)
            .map(|m| m.len() > 0)
            .unwrap_or(false));
        let _ = std::fs::remove_dir_all(&root);
    }
}
