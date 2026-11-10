//! WebGPU **render → present**, checked pixel for pixel (#2665). A guest C program renders one frame
//! into the host's offscreen surface: instanced quads from index and vertex buffers with additive
//! blending, a scissored draw, and a texture it uploads and samples. It reads the frame back and
//! compares every pixel with its own CPU model, and checks that a malformed stream is refused whole
//! and that a WGSL error comes back as a value. The host checks that the frame it saw presented is
//! the one the guest read back. On all three engines.

mod common;

use temen_webgpu::{Frames, Surface};

fn fnv(bytes: &[u8]) -> u32 {
    bytes
        .iter()
        .fold(2166136261u32, |h, &b| (h ^ b as u32).wrapping_mul(16777619))
}

#[test]
fn demo_webgpu_render() {
    common::require_adapter();
    let inst = common::guest("render");
    for backend in common::ENGINES {
        let frames = Frames::default();
        let surface = Surface::Offscreen {
            width: 64,
            height: 64,
            format: temen_gpu::TextureFormat::Rgba8Unorm,
            frames: frames.clone(),
        };
        let stdout = common::run(
            &inst,
            backend,
            &[("webgpu", temen_webgpu::webgpu_cap(surface))],
        );
        assert!(
            stdout.contains("surface: 64x64 format 3: OK")
                && stdout.contains("refused: OK")
                && stdout.contains("wgsl error reported: OK")
                && stdout.contains("frame: OK, 0 pixels differ")
                && stdout.contains("render: OK"),
            "render ({backend:?}) self-check failed; stdout:\n{stdout}"
        );
        let (presented, frame) = frames.last();
        let frame = frame.expect("a presented frame");
        assert_eq!((presented, frame.width, frame.height), (1, 64, 64));
        let digest = format!("digest={:08x}", fnv(&frame.texels));
        assert!(
            stdout.contains(&digest),
            "the presented frame ({digest}) is the one the guest read back; stdout:\n{stdout}"
        );
    }
}
