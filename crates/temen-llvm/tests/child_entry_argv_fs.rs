//! **#1011 slice 3c — the full real-nifler spawn shape: an on-ramp `main(argc, argv)` child reads its
//! input file and writes its output, spawned with argv as the spawn's args payload and a shared memfs
//! re-granted.** This is the last mechanism gap before dropping in the real `nifler` asset. The pieces
//! were each proven apart: `child_entry_argv` had a synthesized `synth_start_argv` parse a seeded args
//! buffer, but driven *directly*, not through a spawn; `child_entry_fs` (temen-run) did a spawn + a
//! re-granted memfs + read/write, but with a *hand-written text-IR* child and *hard-coded* paths, no
//! argv.
//! This composes them on a **real on-ramp child**: a Rust `main(argc, argv)` compiled `--child-entry`
//! (so func 0 is `synth_start_argv`), spawned by `temen_run::conductor` — a window of its own, with
//! `nifler`-shaped argv `["prog","/in.nim","/out.nif"]` landing at its `module_args_base` — with a
//! forkable `mem_fs_shared_factory` re-granted as `"fs"`. The child resolves `"fs"`, opens the
//! `argv[1]` the parent named, reads it, writes it to `argv[2]`, and the parent reads that file back
//! out of its shared handle — exactly `nifler p <in> <out>`, with a copy stub standing in for the parse.
//!
//! Why a copy stub and not real `nifler`: building the real child-entry asset needs the nimony
//! toolchain (`nim` for the C backend), which isn't in per-PR CI. This proves every seam the real asset
//! rides — the args payload feeding `synth_start_argv`, the fs re-grant, the memfs hand-back — with
//! only `rustc`, so the real-nifler swap is a build-script change, not a mechanism unknown.
//!
//! The guest strips a leading `/` from each path because the memfs cap is relative-only (`EACCES` on
//! absolute) — the same normalization the real `os_shim.c` does for nifler. Window confinement
//! (invariant 2) is untouched: the shared authority is the granted cap (§3), and every
//! `open`/`read`/`write` buffer is masked to the child's own window.

#![cfg(target_os = "linux")]

use temen_interp::{run_with_host, ForkedProc, Host, HostProcFork, Value};

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
    // The parent spawns the child with `{"fs"}` re-granted and argv `prog /in.nim /out.nif` — exactly
    // what `synth_start_argv` parses into `argv[]` — and joins it.
    let parent = temen_run::conductor(&["fs"], &["prog", "/in.nim", "/out.nif"]);

    // A cross-domain shared memfs seeded with the input the child will read as `/in.nim` (key `in.nim`,
    // after the guest strips the leading `/`). The parent's `MemFsHandle` observes the same store.
    let input = b"proc main() = echo 42".to_vec();
    let (factory, handle) =
        temen_run::fs::mem_fs_shared_factory(vec![("in.nim".to_string(), input.clone())], vec![]);
    let factory = std::sync::Arc::new(factory);

    let mut host = Host::new();
    let (init, init_state) = (*factory)();
    let fork: HostProcFork = {
        let factory = std::sync::Arc::clone(&factory);
        std::sync::Arc::new(move |_pid| {
            let (h, s) = (*factory)();
            ForkedProc::shared(h, s)
        })
    };
    let fs_h = host.grant_host_proc_forkable(init, fork, init_state);
    let (inst, modh, budget) = temen_run::grant_conductor(&mut host, &child);

    let mut fuel = 200_000_000u64;
    let r = run_with_host(
        &parent,
        0,
        &[
            Value::I32(inst),
            Value::I32(modh),
            Value::I32(budget),
            Value::I32(fs_h),
        ],
        &mut fuel,
        &mut host,
    )
    .expect("parent run");

    // The child returned the byte count it copied (== the input length), joined back.
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
         shape, argv and all"
    );
}
