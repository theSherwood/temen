//! **#1011 slice 3c — a §14 child-entry phase reads and writes files through a re-granted shared memfs.**
//!
//! The nim compiler driver spawns each phase (`nifler → nimsem → hexer → …`) as an op-13 child that
//! *reads* its input file and *writes* its output — `nifler p <in> <out>` parses `<in>.nim` into
//! `<out>.nif`, and the next phase reads that `.nif`. Every prior child-entry test re-granted only a
//! `stdout` **Stream** (`child_entry_io*`), or proved a child *resolves* a granted cap over shared
//! state without going through op-13 (`confined_child_grant`). These two tests compose the pieces on
//! the real filesystem seam:
//!
//!   1. the parent holds a **forkable** `mem_fs_shared_factory` host proc (the re-grantable shape —
//!      `can_regrant` requires `fork.is_some()`), whose store its own `MemFsHandle` observes;
//!   2. it **op-13-spawns** a child-entry module with a one-entry grant list `{"fs" → that handle}`,
//!      so the spawn `regrant_into_child`s the memfs (re-minting the handler over the *same* store)
//!      and `register_cap_name("fs", …)` in the child powerbox;
//!   3. the child resolves `"fs"` by name and does file I/O through it — [`child_entry_writes_a_file…`]
//!      just writes (the emit half), [`child_entry_copies_a_file…`] `read`s a parent-seeded input and
//!      `write`s it back out (the full read→write shape a phase performs);
//!   4. the parent reads the emitted file back out of its shared handle and sees the child's bytes.
//!
//! That is exactly the hand-off a JIT'd `nifler` uses: it reads its `<in>.nim` and emits its `<out>.nif`
//! into a memfs its parent (the Rust-on-SVM driver guest) seeded and then reads. Proven here with
//! hand-written text-IR children so it needs no toolchain and pins the plumbing, not the phase. Window
//! confinement (invariant 2) is untouched: the children's `open`/`read`/`write` name *window-relative*
//! buffers, masked to their carve; the shared authority is the granted cap (§3), not any carve address.
//! The memfs cap is relative-only (it refuses absolute paths, `EACCES`), so the guests name `in.bin` /
//! `out.bin` — the same keys the parent seeds and reads back.

use std::sync::Arc;

use temen_interp::{run_with_host, ForkedProc, Host, HostProc, HostProcFork, Value};
use temen_ir::Module;
use temen_run::fs::MemFsHandle;
use temen_text::parse_module;

// The parent: `main(inst, module, fs)` lays a one-entry grant record `{name_off:2048, name_len:2}
// → fs` at window offset 1024, op-13 (`cap.call INSTANTIATOR(=6) 13`) spawns the child module into a
// 64 KiB carve at `[65536, 131072)` (its declared `memory 16`, off = 1<<16), then op-1 joins it. The
// grant name `"fs"` is a data segment at 2048; the record's second word carries the `fs` handle. Both
// children below share this parent — only their own file I/O differs.
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

// Op-13-spawn `child` under [`PARENT`], re-granting a forkable memfs (seeded with `seed`) as `"fs"`.
// Returns the child's joined status and the shared `MemFsHandle` (seed it before / read it after).
fn spawn_child_over_memfs(
    child: &Module,
    seed: Vec<(String, Vec<u8>)>,
) -> (Vec<Value>, MemFsHandle) {
    temen_verify::verify_module(child).expect("child verifies");
    // The grant record's `name_off:2048 | name_len:2<<32` and the carve geometry in PARENT assume the
    // child declares a 64 KiB window; assert it so a change to a module can't silently mis-carve.
    assert_eq!(child.memory.expect("child window").size_log2, 16);
    let parent = parse_module(PARENT).expect("parse parent");
    temen_verify::verify_module(&parent).expect("parent verifies");

    // A cross-domain shared memfs: every `HostProc` the factory yields (the parent's grant and the
    // child's re-mint) closes over one store, which this `MemFsHandle` also observes — so a file the
    // child writes is readable here after the run, and a file seeded here is readable by the child.
    let (factory, handle) = temen_run::fs::mem_fs_shared_factory(seed, vec![]);
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
    let modh = host.grant_module(child);

    let mut fuel = 200_000_000u64;
    let r = run_with_host(
        &parent,
        0,
        &[Value::I32(inst), Value::I32(modh), Value::I32(fs_h)],
        &mut fuel,
        &mut host,
    )
    .expect("parent run");
    (r, handle)
}

fn read_back(handle: &MemFsHandle, key: &str) -> Vec<u8> {
    let (files, _dirs) = handle.seed();
    files
        .into_iter()
        .find(|(name, _)| name == key)
        .map(|(_, bytes)| bytes)
        .unwrap_or_else(|| panic!("child wrote no `{key}` into the shared memfs"))
}

// A child-entry guest (its `Instantiator` starter arrives as `v0`, unused): resolve `"fs"` by name,
// `open("out.bin", O_WRITE|O_CREATE|O_TRUNC = 2|16|8 = 26)`, `write` its 9-byte `"hello.nif"` data
// segment, `close`, and return the bytes written. Fs ops are `cap.call HOST_PROC(=13) <op>` — op 0
// open, 2 write, 4 close (fs.rs op protocol).
const WRITER: &str = r#"
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

#[test]
fn child_entry_writes_a_file_through_a_regranted_memfs() {
    let child = parse_module(WRITER).expect("parse writer");
    let (r, handle) = spawn_child_over_memfs(&child, vec![]);

    // The child's `write` returned 9, joined back through op-1.
    assert!(
        matches!(r.as_slice(), [Value::I64(9)] | [Value::I32(9)]),
        "child wrote 9 bytes, status joined back: {r:?}"
    );
    assert_eq!(
        read_back(&handle, "out.bin"),
        b"hello.nif",
        "the child-entry guest wrote its payload through the re-granted memfs; the parent read it \
         back via the shared handle — the `.nif` hand-back a spawned nifler uses"
    );
}

// A child-entry guest that does the full read→write of a phase: resolve `"fs"`, `open("in.bin", O_READ
// = 1)`, `read` up to 256 bytes into a scratch buffer at window offset 256, `open("out.bin",
// O_WRITE|O_CREATE|O_TRUNC = 26)`, `write` exactly those bytes back, close both, and return the byte
// count. Copying `in.bin → out.bin` proves the input the *parent* seeded flowed through the child and
// back out — the data path `nifler <in> <out>` rides. Ops: 0 open, 1 read, 2 write, 4 close.
const COPIER: &str = r#"
memory 16
data 0 "fs"
data 8 "in.bin"
data 16 "out.bin"
func (i64) -> (i64) {
block 0 (vstarter: i64) {
  vfp = i64.const 0
  vfl = i64.const 2
  vfs = cap.self.resolve vfp vfl
  vzero = i64.const 0
  vinp = i64.const 8
  vinl = i64.const 6
  vrd = i64.const 1
  vfin = cap.call 13 0 (i64, i64, i64, i64) -> (i64) vfs (vinp, vinl, vrd, vzero)
  vbuf = i64.const 256
  vcap = i64.const 256
  vn = cap.call 13 1 (i64, i64, i64, i64) -> (i64) vfs (vfin, vbuf, vcap, vzero)
  voutp = i64.const 16
  voutl = i64.const 7
  vwr = i64.const 26
  vfout = cap.call 13 0 (i64, i64, i64, i64) -> (i64) vfs (voutp, voutl, vwr, vzero)
  vw = cap.call 13 2 (i64, i64, i64, i64) -> (i64) vfs (vfout, vbuf, vn, vzero)
  vc1 = cap.call 13 4 (i64, i64, i64, i64) -> (i64) vfs (vfin, vzero, vzero, vzero)
  vc2 = cap.call 13 4 (i64, i64, i64, i64) -> (i64) vfs (vfout, vzero, vzero, vzero)
  return vw
  }
}
"#;

#[test]
fn child_entry_copies_a_parent_seeded_file_through_a_regranted_memfs() {
    let child = parse_module(COPIER).expect("parse copier");
    let input = b"hello nifler".to_vec(); // 12 bytes the parent seeds into the shared store
    let (r, handle) = spawn_child_over_memfs(&child, vec![("in.bin".to_string(), input.clone())]);

    // The child read 12 bytes and wrote all 12 back; the count joined through op-1.
    let n = input.len() as i64;
    let got = match r.as_slice() {
        [Value::I64(m)] => *m,
        [Value::I32(m)] => *m as i64,
        other => panic!("unexpected join result: {other:?}"),
    };
    assert_eq!(got, n, "child copied {n} bytes, status joined back");
    assert_eq!(
        read_back(&handle, "out.bin"),
        input,
        "the child read the parent-seeded `in.bin` and wrote it to `out.bin` through the re-granted \
         memfs — the full read→write a spawned nifler does over its `<in>`/`<out>`"
    );
}
