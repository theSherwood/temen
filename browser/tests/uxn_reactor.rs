//! **Uxn runs in the browser reactor** — the playground's Uxn card (`crates/temen-run/demos/uxn`)
//! exercised natively over the exact Rust the wasm exports wrap: the committed `web/assets/uxn.temen`
//! guest opened with the committed demo ROM served as `boot.rom` through the `fs` capability
//! (`OnrampReactor::open_with_fs`, the `temen_onramp_open_fs` path), then driven one `tick` per frame
//! with key events pushed through `keyboard`. The pixels themselves are pinned byte-exact to a native
//! build of the same C by `crates/temen-llvm/tests/uxn_diff.rs`; this proves the *reactor wiring*: the
//! ROM arrives, the Screen vector runs each frame, the swarm animates, and key events reach the
//! Controller device (a letter cycles the palette, the arrows move the player), and pointer events
//! pushed through `mouse` reach the Mouse device (a click puts the player under the pointer).
//!
//! Both assets are code-coupled: rebuild them with `ONLY=uxn bash scripts/rebuild-assets.sh` after an
//! IR/wire change (they decode as `BadOpcode` otherwise) or a change to the demo sources.

use temen_browser::{Frame, OnrampReactor, STATUS_OK};

const W: u32 = 256;
const H: u32 = 192;

fn open() -> OnrampReactor {
    let temen = include_bytes!("../web/assets/uxn.temen");
    let rom = include_bytes!("../web/assets/uxn_demo.rom");
    let m = temen_encode::decode_module(temen).expect("decode uxn.temen");
    OnrampReactor::open_with_fs(&m, "boot.rom".to_string(), rom.to_vec())
        .expect("_start reads boot.rom through fs and runs the reset vector")
}

fn step(r: &mut OnrampReactor) -> Frame {
    let (status, _stdout) = r.frame();
    assert_eq!(status, STATUS_OK, "tick keeps going");
    r.take_frame()
        .expect("the swarm moves every frame, so every tick presents")
}

fn pixel(f: &Frame, x: u32, y: u32) -> [u8; 4] {
    let i = ((y * f.width + x) * 4) as usize;
    [f.rgba[i], f.rgba[i + 1], f.rgba[i + 2], f.rgba[i + 3]]
}

#[test]
fn uxn_boots_and_animates() {
    let mut r = open();
    let f1 = step(&mut r);
    assert_eq!(
        (f1.width, f1.height),
        (W, H),
        "demo.tal sets a 256x192 screen"
    );
    assert_eq!(f1.rgba.len(), (W * H * 4) as usize);
    // The top band is color 1 of theme 0 (r/g/b nibbles 2/3/4, replicated), painted by the fill op.
    assert_eq!(
        pixel(&f1, 0, 0),
        [0x22, 0x33, 0x44, 0xff],
        "background band, theme 0"
    );
    // The second band down is color 2.
    assert_eq!(
        pixel(&f1, 0, 16),
        [0x33, 0x44, 0x66, 0xff],
        "second band, theme 0"
    );
    // The title row (y = 8..16, x = 80..176) has white (color 3) glyph pixels.
    let title_white = (80..176).any(|x| pixel(&f1, x, 10) == [0xff, 0xff, 0xff, 0xff]);
    assert!(title_white, "the 1bpp title glyphs are drawn in color 3");
    // Frame 2 differs: the swarm bounces on the foreground layer every frame.
    let f2 = step(&mut r);
    assert_ne!(f1.rgba, f2.rgba, "the sprites move between frames");
}

#[test]
fn keys_reach_the_controller() {
    // A letter key cycles the palette: after 'b' (JS keyCode 66) the same band is theme 1's color 1.
    let mut r = open();
    step(&mut r);
    r.push_key(66, 1);
    let f = step(&mut r);
    assert_eq!(
        pixel(&f, 0, 0),
        [0xee, 0x66, 0x22, 0xff],
        "theme 1 after a letter key"
    );

    // Holding Right (keyCode 39) moves the player, so the frames diverge from an input-free run at the
    // same frame count — with the rest of the scene (the swarm) identical in both.
    let mut steered = open();
    let mut idle = open();
    steered.push_key(39, 1);
    let (mut fs, mut fi) = (step(&mut steered), step(&mut idle));
    for _ in 0..20 {
        fs = step(&mut steered);
        fi = step(&mut idle);
    }
    assert_ne!(fs.rgba, fi.rgba, "the arrow key steers the player");
    // Sanity: without the key, two fresh runs are identical (the whole thing is deterministic).
    let mut again = open();
    let mut fa = step(&mut again);
    for _ in 0..20 {
        fa = step(&mut again);
    }
    assert_eq!(fa.rgba, fi.rgba, "an input-free run is deterministic");
}

#[test]
fn mouse_reaches_the_mouse_device() {
    // A left click at (200, 100) — kind 0, payload (buttons << 24) | (x << 12) | y — moves the player
    // under the pointer: its white (color 3) outline appears in the 8x8 box at (196, 96) on the next
    // frame, where an input-free run has only background.
    let click = |buttons: i32| (buttons << 24) | (200 << 12) | 100;
    let mut idle = open();
    let mut clicked = open();
    step(&mut idle);
    step(&mut clicked);
    clicked.push_mouse(0, click(1));
    clicked.push_mouse(0, click(0));
    let fi = step(&mut idle);
    let fc = step(&mut clicked);
    let white = |f: &Frame| {
        (96..104)
            .flat_map(|y| (196..204).map(move |x| (x, y)))
            .filter(|&(x, y)| pixel(f, x, y) == [0xff, 0xff, 0xff, 0xff])
            .count()
    };
    assert_eq!(
        white(&fi),
        0,
        "nothing white under the pointer without a click"
    );
    assert!(
        white(&fc) >= 8,
        "the player's outline is under the pointer after the click"
    );
    // A wheel notch (kind 1, dy = 1) cycles the palette, like a letter key.
    clicked.push_mouse(1, 1);
    let f = step(&mut clicked);
    assert_eq!(
        pixel(&f, 0, 0),
        [0xee, 0x66, 0x22, 0xff],
        "theme 1 after a wheel notch"
    );
}

#[test]
fn tal_source_assembles_in_the_guest() {
    // The same guest served the demo's SOURCE as boot.tal assembles it in the sandbox (uxnasm_core.c)
    // and runs it: its first frame is byte-identical to the committed ROM's.
    let temen = include_bytes!("../web/assets/uxn.temen");
    let tal = include_bytes!("../../crates/temen-run/demos/uxn/demo.tal");
    let m = temen_encode::decode_module(temen).expect("decode uxn.temen");
    let mut from_tal = OnrampReactor::open_with_fs(&m, "boot.tal".to_string(), tal.to_vec())
        .expect("_start assembles boot.tal");
    let mut from_rom = open();
    let (a, b) = (step(&mut from_tal), step(&mut from_rom));
    assert_eq!((a.width, a.height), (W, H));
    assert_eq!(
        a.rgba, b.rgba,
        "assembled-in-guest ROM renders like the committed ROM"
    );
}

#[test]
fn bad_tal_reports_the_error_and_exits() {
    // An unassemblable boot.tal: the first tick prints `uxnasm: line N: …` on stdout (the page shows
    // it) and exits, ending the reactor loop.
    let temen = include_bytes!("../web/assets/uxn.temen");
    let m = temen_encode::decode_module(temen).expect("decode uxn.temen");
    let src = b"|0100 #01 #02 ADD\n,&nope JMP BRK\n".to_vec();
    let mut r = OnrampReactor::open_with_fs(&m, "boot.tal".to_string(), src)
        .expect("_start survives a bad source (the error is reported by the first tick)");
    let (status, stdout) = r.frame();
    assert_eq!(
        status,
        temen_browser::STATUS_EXIT,
        "the guest exits after reporting"
    );
    let text = String::from_utf8_lossy(&stdout);
    assert_eq!(
        text.trim(),
        "uxnasm: line 2: unknown reference: on-reset/nope",
        "the error names the line and the unresolved label"
    );
}
