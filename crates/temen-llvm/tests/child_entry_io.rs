//! **#1011 slice 3c (2b) — a §14 child-entry phase does real I/O through a re-granted cap.** A compiler
//! phase spawned via op-13 must reach its `stdout`/`fs` — which arrive as **manifest imports** bound at
//! spawn to the child's re-granted named caps (`Host::bind_child_manifest`). This proves that binding
//! for an on-ramp child-entry module: a real Rust guest compiled `--child-entry` that calls `write(1,
//! …)` is `instantiate_module_named` (op 13)'d with `stdout` re-granted, and its bytes land in the
//! shared sink — the exact hand-off a JIT'd `nifler` uses to reach its `fs`. (The cooperative engine
//! binds the child manifest inline; wiring the same into the resumable/tier-up path is a follow-up.)

#![cfg(target_os = "linux")]

use temen_interp::{run_with_host, Host, StreamRole, Value};

// A Rust guest that writes "hi" to fd 1. `write` lowers to a `Stream` manifest import, which (a) forces
// a synthesized powerbox `_start` and (b) must bind to the re-granted `stdout` at spawn. Compiled
// `--child-entry`, its `_start` is the §14 child ABI.
const GUEST: &str = r##"
#![no_std]
#![allow(internal_features)]
#[panic_handler]
fn ph(_: &core::panic::PanicInfo) -> ! { loop {} }
#[no_mangle]
pub extern "C" fn rust_eh_personality() {}
extern "C" {
    fn write(fd: i64, buf: *const u8, n: i64) -> i64;
}
static MSG: [u8; 2] = *b"hi";
#[no_mangle]
pub extern "C" fn main() -> i32 {
    unsafe { write(1, MSG.as_ptr(), 2); }
    0
}
"##;

fn emit_ll(src: &std::path::Path, ll: &std::path::Path) -> bool {
    std::process::Command::new("rustc")
        .args([
            "--edition",
            "2021",
            "-O",
            "-Cpanic=abort",
            "--emit=llvm-ir",
            "--crate-type=cdylib",
        ])
        .arg(src)
        .arg("-o")
        .arg(ll)
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[test]
fn child_entry_writes_through_a_regranted_stdout() {
    let dir = std::env::temp_dir().join(format!("ce_io_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("g.rs");
    let ll = dir.join("g.ll");
    std::fs::write(&src, GUEST).unwrap();
    if !emit_ll(&src, &ll) {
        eprintln!("note: skipping (rustc unavailable)");
        return;
    }

    let opts = temen_llvm::TranslateOptions {
        child_entry: true,
        ..Default::default()
    };
    let child = temen_llvm::translate_ll_path_with_options(&ll, opts)
        .expect("translate child-entry")
        .module;
    temen_verify::verify_module(&child).expect("child verifies");
    let sl = child.memory.expect("child window").size_log2;

    // The `"stdout"` name packed little-endian into one i64, and the 16-byte grant record's first word
    // (`name_off:u32=2048 | name_len:u32=6`). The parent lays these in its window, then op-13-spawns the
    // child re-granting the `stdout` handle (arg `v2`) under that name.
    let name_i64: u64 = b"stdout"
        .iter()
        .enumerate()
        .fold(0u64, |acc, (i, &b)| acc | ((b as u64) << (8 * i)));
    let word0: u64 = 2048 | (6u64 << 32);
    let carve_off: u64 = 1u64 << sl;
    let parent_src = format!(
        r#"memory {psl}
func (i32, i32, i32) -> (i64) {{
block 0 (v0: i32, v1: i32, v2: i32) {{
  vname = i64.const {name_i64}
  vnoff = i64.const 2048
  i64.store vnoff vname
  vrec0 = i64.const {word0}
  vrecoff = i64.const 1024
  i64.store vrecoff vrec0
  vsh = i64.extend_i32_u v2
  vrec1off = i64.const 1032
  i64.store vrec1off vsh
  vmh = i64.extend_i32_u v1
  vgptr = i64.const 1024
  vgn = i64.const 1
  ventry = i64.const 0
  voff = i64.const {carve_off}
  vsl = i64.const {sl}
  vq = i64.const 0
  vh = call.cap 6 13 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vmh, vgptr, vgn, ventry, voff, vsl, vq)
  vr = call.cap 6 1 (i32) -> (i64) v0 (vh)
  return vr
  }}
}}
"#,
        psl = sl + 1,
    );
    let parent = temen_text::parse_module(&parent_src).expect("parse parent");
    temen_verify::verify_module(&parent).expect("verify parent");

    let mut host = Host::new();
    let sink = host.shared_stdout(); // the shared Out sink we read after the run
    let out_h = host.grant_stream(StreamRole::Out);
    let win = 1u64 << (sl + 1);
    let inst = host.grant_instantiator(0, win);
    let modh = host.grant_module(&child);

    let mut fuel = 200_000_000u64;
    let r = run_with_host(
        &parent,
        0,
        &[Value::I32(inst), Value::I32(modh), Value::I32(out_h)],
        &mut fuel,
        &mut host,
    )
    .expect("parent run");

    // The child's `main` returned 0, joined back through op-1.
    assert!(
        matches!(r.as_slice(), [Value::I64(0)] | [Value::I32(0)]),
        "child status 0 joined back: {r:?}"
    );
    assert_eq!(
        &*sink.lock().unwrap(),
        b"hi",
        "the child-entry guest's write bound to the re-granted stdout and reached the shared sink"
    );
}
