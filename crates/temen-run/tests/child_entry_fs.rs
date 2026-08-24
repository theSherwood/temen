//! **#1011 slice 3c — a §14 child-entry phase hands a file back through a re-granted shared memfs.**
//!
//! The nim compiler driver spawns each phase (`nifler → nimsem → hexer → …`) as an op-13 child and
//! reads the `.nif` that phase *wrote* — the whole point of the fan-out. Every prior child-entry test
//! re-granted only a `stdout` **Stream** (`child_entry_io*`), or proved a child *resolves* a granted
//! cap over shared state without going through op-13 (`confined_child_grant`). This composes the three
//! for the first time on the real filesystem seam:
//!
//!   1. the parent holds a **forkable** `mem_fs_shared_factory` host proc (the re-grantable shape —
//!      `can_regrant` requires `fork.is_some()`), whose store its own `MemFsHandle` observes;
//!   2. it **op-13-spawns** a child-entry module with a one-entry grant list `{"fs" → that handle}`,
//!      so the spawn `regrant_into_child`s the memfs (re-minting the handler over the *same* store)
//!      and `register_cap_name("fs", …)` in the child powerbox;
//!   3. the child resolves `"fs"` by name, `open`/`write`/`close`s `/out.bin` through it, and returns
//!      the byte count — which joins back to the parent;
//!   4. the parent reads `out.bin` back out of its shared handle and sees the child's bytes.
//!
//! That is exactly the hand-back a JIT'd `nifler` uses to emit its `.nif` into a memfs its parent (the
//! Rust-on-SVM driver guest) then reads — proven here with a hand-written text-IR child so it needs no
//! toolchain and pins the plumbing, not the phase. Window confinement (invariant 2) is untouched: the
//! child's `open`/`write` name *window-relative* buffers, masked to its carve; the shared authority is
//! the granted cap (§3), not any carve address.

use std::sync::Arc;

use temen_interp::{run_with_host, ForkedProc, Host, HostProc, HostProcFork, Value};
use temen_text::parse_module;

// A child-entry guest (its `Instantiator` starter arrives as `v0`, unused): resolve `"fs"` by name,
// `open("out.bin", O_WRITE|O_CREATE|O_TRUNC = 2|16|8 = 26)`, `write` its 9-byte `"hello.nif"` data
// segment, `close`, and return the bytes written. The memfs cap is relative-only (it refuses absolute
// paths, `EACCES`), so the guest names `out.bin` — the same key the parent reads back. Fs ops are
// `cap.call HOST_PROC(=13) <op>` — op 0 open, 2 write, 4 close (fs.rs op protocol).
const CHILD: &str = r#"
memory 16
data 0 "fs"
data 8 "out.bin"
data 32 "hello.nif"
func (i64) -> (i64) {
block 0 (vstarter: i64) {
  vfp = i64.const 0
  vfl = i64.const 2
  vfs = cap.self.resolve vfp vfl
  vpath = i64.const 8
  vplen = i64.const 7
  vflags = i64.const 26
  vzero = i64.const 0
  vfd = cap.call 13 0 (i64, i64, i64, i64) -> (i64) vfs (vpath, vplen, vflags, vzero)
  vbuf = i64.const 32
  vblen = i64.const 9
  vn = cap.call 13 2 (i64, i64, i64, i64) -> (i64) vfs (vfd, vbuf, vblen, vzero)
  vc = cap.call 13 4 (i64, i64, i64, i64) -> (i64) vfs (vfd, vzero, vzero, vzero)
  return vn
  }
}
"#;

// The parent: `main(inst, module, fs)` lays a one-entry grant record `{name_off:2048, name_len:2}
// → fs` at window offset 1024, op-13 (`cap.call INSTANTIATOR(=6) 13`) spawns the child module into a
// 64 KiB carve at `[65536, 131072)` (its declared `memory 16`, off = 1<<16), then op-1 joins it. The
// grant name `"fs"` is a data segment at 2048; the record's second word carries the `fs` handle.
const PARENT: &str = r#"
memory 17
data 2048 "fs"
func (i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32) {
  vrec0 = i64.const 8589936640
  vrecoff = i64.const 1024
  i64.store vrecoff vrec0
  vsh = i64.extend_i32_u v2
  vrec1off = i64.const 1032
  i64.store vrec1off vsh
  vmh = i64.extend_i32_u v1
  vgptr = i64.const 1024
  vgn = i64.const 1
  ventry = i64.const 0
  voff = i64.const 65536
  vsl = i64.const 16
  vq = i64.const 0
  vh = cap.call 6 13 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vmh, vgptr, vgn, ventry, voff, vsl, vq)
  vr = cap.call 6 1 (i32) -> (i64) v0 (vh)
  return vr
  }
}
"#;

#[test]
fn child_entry_writes_a_file_through_a_regranted_memfs() {
    let child = parse_module(CHILD).expect("parse child");
    temen_verify::verify_module(&child).expect("child verifies");
    // The grant record's `name_off:2048 | name_len:2<<32` and the carve geometry below assume the
    // child declares a 64 KiB window; assert it so a change to the module can't silently mis-carve.
    assert_eq!(child.memory.expect("child window").size_log2, 16);
    let parent = parse_module(PARENT).expect("parse parent");
    temen_verify::verify_module(&parent).expect("parent verifies");

    // A cross-domain shared memfs: every `HostProc` the factory yields (the parent's grant and the
    // child's re-mint) closes over one store, which this `MemFsHandle` also observes — so the file the
    // child writes is readable here after the run (the phase-to-phase `.nif` hand-off).
    let (factory, handle) = temen_run::fs::mem_fs_shared_factory(vec![], vec![]);
    let factory = Arc::new(factory);

    let mut host = Host::new();
    // Grant the parent a *forkable* memfs host proc: the initial handler plus a fork factory minting a
    // fresh handler over the same store. `regrant_into_child` needs `fork.is_some()` to carry it into a
    // child (a factory-less host proc fails `can_regrant`, fail-closed).
    let init: HostProc = (*factory)();
    let fork: HostProcFork = {
        let factory = Arc::clone(&factory);
        Arc::new(move |_pid| ForkedProc::shared((*factory)()))
    };
    let fs_h = host.grant_host_proc_forkable(init, fork);
    let inst = host.grant_instantiator(0, 1u64 << 17);
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

    // The child's `write` returned 9, joined back through op-1.
    assert!(
        matches!(r.as_slice(), [Value::I64(9)] | [Value::I32(9)]),
        "child wrote 9 bytes, status joined back: {r:?}"
    );

    // Read the emitted file back out of the shared store — the parent half of the hand-off.
    let (files, _dirs) = handle.seed();
    let emitted = files
        .into_iter()
        .find(|(name, _)| name == "out.bin")
        .map(|(_, bytes)| bytes)
        .expect("child wrote `out.bin` into the re-granted shared memfs");
    assert_eq!(
        emitted, b"hello.nif",
        "the child-entry guest wrote its payload through the re-granted memfs; the parent read it \
         back via the shared handle — the `.nif` hand-back a spawned nifler uses"
    );
}
