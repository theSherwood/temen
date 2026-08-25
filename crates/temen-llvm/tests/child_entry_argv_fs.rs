//! **#1011 slice 3c — the full real-nifler op-13 shape: an on-ramp `main(argc, argv)` child reads its
//! input file and writes its output, spawned with argv seeded into its carve and a shared memfs
//! re-granted.** This is the last mechanism gap before dropping in the real `nifler` asset. The pieces
//! were each proven apart: `child_entry_argv` had a synthesized `synth_start_argv` parse a seeded
//! `POWERBOX_ARGS_BASE` buffer, but driven *directly*, not through an op-13 spawn; `child_entry_fs`
//! (temen-run) did op-13 + a re-granted memfs + read/write, but with a *hand-written text-IR* child and
//! *hard-coded* paths, no argv.
//! This composes them on a **real on-ramp child**: a Rust `main(argc, argv)` compiled `--child-entry`
//! (so func 0 is `synth_start_argv`), `instantiate_module`'d (op 13) into a carve whose
//! `POWERBOX_ARGS_BASE` the parent seeded with `nifler`-shaped argv `["prog","/in.nim","/out.nif"]`,
//! with a forkable `mem_fs_shared_factory` re-granted as `"fs"`. The child resolves `"fs"`, opens the
//! `argv[1]` the parent named, reads it, writes it to `argv[2]`, and the parent reads that file back out
//! of its shared handle — exactly `nifler p <in> <out>`, with a copy stub standing in for the parse.
//!
//! Why a copy stub and not real `nifler`: building the real child-entry asset needs the nimony
//! toolchain (`nim` for the C backend), which isn't in per-PR CI. This proves every seam the real asset
//! rides — argv-in-carve feeding `synth_start_argv`, the fs re-grant, the memfs hand-back — with only
//! `rustc`, so the real-nifler swap is a build-script change, not a mechanism unknown.
//!
//! The parent pre-seeds argv into the carve because op-13 gives the child a `nested_view` that *aliases*
//! the parent's window (it does not zero the carve — only the child's own data segments materialize on
//! top, and the on-ramp keeps them clear of `[128, argv_end)`). The guest strips a leading `/` from each
//! path because the memfs cap is relative-only (`EACCES` on absolute) — the same normalization the real
//! `os_shim.c` does for nifler. Window confinement (invariant 2) is untouched: the shared authority is
//! the granted cap (§3), and every `open`/`read`/`write` buffer is masked to the child's carve.

#![cfg(target_os = "linux")]

use temen_interp::{run_with_host, ForkedProc, Host, HostProc, HostProcFork, Value};

// A child-entry `main(argc, argv)` mini-nifler: resolve `"fs"`, open `argv[1]` (O_READ), read up to 256
// bytes, open `argv[2]` (O_WRITE|O_CREATE|O_TRUNC = 26), write them back, close both, return the byte
// count. Reaches the memfs through the raw `__vm_cap_resolve`/`__vm_host_call` seam (op must be a
// constant), stripping a leading `/` (the memfs is relative-only). `main(argc, argv)` forces
// `synth_start_argv` as func 0 — the argv-parsing child entry.
const GUEST: &str = r##"
#![no_std]
#![allow(internal_features)]
#[panic_handler]
fn ph(_: &core::panic::PanicInfo) -> ! { loop {} }
#[no_mangle]
pub extern "C" fn rust_eh_personality() {}
extern "C" {
    fn __vm_cap_resolve(name: *const u8, len: i64) -> i32;
    fn __vm_host_call(h: i32, op: i32, a: i64, b: i64, c: i64, d: i64) -> i64;
    fn snprintf(buf: *mut u8, n: usize, fmt: *const u8, ...) -> i32;
}
unsafe fn strip(p: *const u8) -> *const u8 { if *p == b'/' { p.add(1) } else { p } }
unsafe fn clen(p: *const u8) -> i64 {
    let mut n = 0i64;
    while *p.add(n as usize) != 0 { n += 1; }
    n
}
#[no_mangle]
pub extern "C" fn main(argc: i32, argv: *const *const u8) -> i32 {
    // Force a synthesized powerbox `_start` (so `--child-entry` has an entry to shape): `snprintf`
    // writes the format scratch the powerbox layout reserves. A guest that reaches its caps only through
    // raw `__vm_*` intrinsics needs no powerbox, so the on-ramp would otherwise synthesize no `_start`
    // at all — exactly what real `nifler` (a full libc program) never hits. The `& 0` keeps the call
    // live without perturbing the result.
    let mut fb = [0u8; 8];
    unsafe { snprintf(fb.as_mut_ptr(), 8, b"%d\0".as_ptr(), argc); }
    let keep = (unsafe { core::ptr::read_volatile(fb.as_ptr()) } as i32) & 0;
    if argc < 3 { return -1; }
    unsafe {
        let fs = __vm_cap_resolve(b"fs".as_ptr(), 2);
        let inp = strip(*argv.add(1));
        let outp = strip(*argv.add(2));
        let fin = __vm_host_call(fs, 0, inp as i64, clen(inp), 1, 0);
        if fin < 0 { return fin as i32; }
        let mut buf = [0u8; 256];
        let n = __vm_host_call(fs, 1, fin, buf.as_mut_ptr() as i64, 256, 0);
        let fout = __vm_host_call(fs, 0, outp as i64, clen(outp), 26, 0);
        if fout < 0 { return fout as i32; }
        __vm_host_call(fs, 2, fout, buf.as_mut_ptr() as i64, n, 0);
        __vm_host_call(fs, 4, fin, 0, 0, 0);
        __vm_host_call(fs, 4, fout, 0, 0, 0);
        (n as i32) + keep
    }
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
fn on_ramp_child_reads_argv_paths_and_copies_a_file_over_a_regranted_memfs() {
    let dir = std::env::temp_dir().join(format!("ce_argv_fs_{}", std::process::id()));
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
        .expect("translate child-entry argv")
        .module;
    temen_verify::verify_module(&child).expect("child verifies");
    // The synthesized argv `_start` is inserted at func 0 (the on-ramp does not reorder past it), so it
    // is the child entry: a starter cap in (`[I64]`, or `[I64, I64]` when the module also manages its own
    // pages), an i64 status out (`child_entry_ok`). Both shapes are valid; op-13 dispatches whichever it
    // declares. (`snprintf` in the guest forces this `_start` to exist — see GUEST.)
    use temen_ir::ValType::I64 as V;
    let entry = 0u32;
    let esig = &child.funcs[entry as usize];
    assert!(
        matches!(esig.params.as_slice(), [V] | [V, V]) && esig.results == [V],
        "func 0 is a valid §14 child entry: {:?} -> {:?}",
        esig.params,
        esig.results
    );
    let sl = child.memory.expect("child window").size_log2;
    assert!(
        sl >= 12,
        "child window must clear the parent's grant scratch (got 2^{sl})"
    );

    // The parent seeds `POWERBOX_ARGS_BASE` (128) inside the child's carve with `{argc=3, envc=0}` +
    // packed `"prog\0/in.nim\0/out.nif\0"` — exactly what `synth_start_argv` parses into `argv[]`. It
    // also lays a one-entry grant record `{name_off:2048, name_len:2} → fs` (the `fs` handle is `v2`),
    // then op-13-spawns the child into the carve and op-1-joins it. The carve aliases these seeded
    // bytes, so the child sees both the argv and (by name) the memfs.
    let carve_off: u64 = 1u64 << sl;
    // #964/#1094: a guarded child reads argv one guard up — key off `module_args_base` (the grant
    // records/cap-names below stay in the parent window, read by the op-13 handler, never by the child).
    let argv_off = carve_off + temen_ir::module_args_base(&child);
    let word0: u64 = 2048 | (2u64 << 32);
    // `\x03\x00\x00\x00` = argc 3, `\x00\x00\x00\x00` = envc 0, then the NUL-separated args.
    let argv_data = "\\x03\\x00\\x00\\x00\\x00\\x00\\x00\\x00prog\\x00/in.nim\\x00/out.nif\\x00";
    let parent_src = format!(
        r#"memory {psl}
data 2048 "fs"
data {argv_off} "{argv_data}"
func (i32, i32, i32) -> (i64) {{
block 0 (v0: i32, v1: i32, v2: i32) {{
  vrec0 = i64.const {word0}
  vrecoff = i64.const 1024
  i64.store vrecoff vrec0
  vsh = i64.extend_i32_u v2
  vrec1off = i64.const 1032
  i64.store vrec1off vsh
  vmh = i64.extend_i32_u v1
  vgptr = i64.const 1024
  vgn = i64.const 1
  ventry = i64.const {entry}
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

    // A cross-domain shared memfs seeded with the input the child will read as `/in.nim` (key `in.nim`,
    // after the guest strips the leading `/`). The parent's `MemFsHandle` observes the same store.
    let input = b"proc main() = echo 42".to_vec();
    let (factory, handle) =
        temen_run::fs::mem_fs_shared_factory(vec![("in.nim".to_string(), input.clone())], vec![]);
    let factory = std::sync::Arc::new(factory);

    let mut host = Host::new();
    let init: HostProc = (*factory)();
    let fork: HostProcFork = {
        let factory = std::sync::Arc::clone(&factory);
        std::sync::Arc::new(move |_pid| ForkedProc::shared((*factory)()))
    };
    let fs_h = host.grant_host_proc_forkable(init, fork);
    let inst = host.grant_instantiator(0, 1u64 << (sl + 1));
    let modh = host.grant_module(&child);

    let mut fuel = 200_000_000u64;
    let r = run_with_host(
        &parent,
        0,
        &[Value::I32(inst), Value::I32(modh), Value::I32(fs_h)],
        &mut fuel,
        &mut host,
    )
    .expect("parent run");

    // The child returned the byte count it copied (== the input length), joined through op-1.
    let n = input.len() as i64;
    let got = match r.as_slice() {
        [Value::I64(m)] => *m,
        [Value::I32(m)] => *m as i64,
        other => panic!("unexpected join result: {other:?}"),
    };
    assert_eq!(
        got, n,
        "child parsed argv, opened argv[1], and copied {n} bytes to argv[2]; status joined back"
    );

    // Read `out.nif` back out of the shared store — the parent half of the `nifler p <in> <out>`
    // hand-off, with the emitted bytes now present.
    let (files, _dirs) = handle.seed();
    let emitted = files
        .into_iter()
        .find(|(name, _)| name == "out.nif")
        .map(|(_, bytes)| bytes)
        .expect("child wrote `out.nif` into the re-granted shared memfs");
    assert_eq!(
        emitted, input,
        "the on-ramp child read the parent-seeded `/in.nim` named by argv[1] and wrote it to the \
         `/out.nif` named by argv[2] through the re-granted memfs — the full `nifler p <in> <out>` \
         shape, argv-in-carve and all"
    );
}
