//! **#1025 — run-in-guest: `ModuleLoader.from_bytes` (iface 7).** The missing input side of the §14
//! nesting primitive. Every prior op-13 test spawned a child from a module the *host* granted
//! (`Host::grant_module`) — but a compiler driver running *on* the sandbox produces its program itself
//! (the linker emits a module into guest memory) and must be able to run its own output without the
//! host pre-granting it. This proves the primitive that closes that gap: a guest hands the host a
//! wire-encoded module from its window, the host **decodes + verifies** it (the trusted floor — the
//! same `verify_module` a host-granted module passes; the decode is the copy, so no wire byte is ever
//! executed and the result never aliases the guest's mutable buffer) and mints a `Module` handle the
//! guest then op-13-spawns exactly like a host-granted one.
//!
//! Hand-written text-IR, so it needs **no toolchain**. The child is the same re-granted-memfs writer
//! `child_entry_fs.rs` proves; the only difference here is *where its module handle comes from* — the
//! parent's `from_bytes(ptr, len)` over the child's encoded bytes (seeded into the parent window as a
//! data segment), not a host grant. Window confinement (invariant 2) is untouched: the minted module's
//! code is host-owned and immutable, its data materializes into the child's carve, and the child's I/O
//! is masked to that carve.

use std::sync::Arc;

use temen_interp::{run_with_host, ForkedProc, Host, HostProc, HostProcFork, Value};
use temen_ir::{Data, Module};
use temen_run::fs::MemFsHandle;
use temen_text::parse_module;

/// Where the parent's `from_bytes` reads the child's encoded module from: above the grant record
/// (17408) and the `"fs"` name (18432), below the child's carve `[65536, 131072)`. A small module is
/// well under the ~45 KiB of headroom.
const CHILD_BYTES_OFF: u64 = 20480;

// The parent: `main(inst, loader, fs, blen)`. It (1) `from_bytes(CHILD_BYTES_OFF, blen)` over the
// child's encoded module in its window — `call.cap MODULE_LOADER(=7) 0` — to mint a `Module` handle,
// then (2) op-13-spawns that handle into a 64 KiB carve at `[65536, 131072)` with a one-entry grant
// list `{"fs" -> fs}` (the record at 17408, the name at 18432), and (3) op-1 joins it. Identical to
// `child_entry_fs.rs`'s parent from step (2) on — only the module handle's *origin* differs (minted
// from bytes here, host-granted there).
const PARENT: &str = r#"
memory 17
data 18432 "fs"
func (i32, i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32, v3: i32) {
  vptr = i64.const 20480
  vblen = i64.extend_i32_u v3
  vmh = call.cap 7 0 (i64, i64) -> (i64) v1 (vptr, vblen)
  vrec0 = i64.const 8589953024
  vrecoff = i64.const 17408
  i64.store vrecoff vrec0
  vsh = i64.extend_i32_u v2
  vrec1off = i64.const 17416
  i64.store vrec1off vsh
  vgptr = i64.const 17408
  vgn = i64.const 1
  ventry = i64.const 0
  voff = i64.const 65536
  vsl = i64.const 16
  vq = i64.const 0
  vh = call.cap 6 13 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vmh, vgptr, vgn, ventry, voff, vsl, vq)
  vr = call.cap 6 1 (i32) -> (i64) v0 (vh)
  return vr
  }
}
"#;

// The child (identical to `child_entry_fs.rs`'s WRITER): resolve `"fs"` by name, `open("out.bin",
// O_WRITE|O_CREATE|O_TRUNC = 26)`, `write` its 9-byte `"hello.nif"`, `close`, return the byte count.
const WRITER: &str = r#"
memory 16
data 16384 "fs"
data 16392 "out.bin"
data 16416 "hello.nif"
func (i64) -> (i64) {
block 0 (vstarter: i64) {
  vfp = i64.const 16384
  vfl = i64.const 2
  vfs = self.resolve vfp vfl
  vpath = i64.const 16392
  vplen = i64.const 7
  vflags = i64.const 26
  vzero = i64.const 0
  vfd = call.cap 13 0 (i64, i64, i64, i64) -> (i64) vfs (vpath, vplen, vflags, vzero)
  vbuf = i64.const 16416
  vblen = i64.const 9
  vn = call.cap 13 2 (i64, i64, i64, i64) -> (i64) vfs (vfd, vbuf, vblen, vzero)
  vc = call.cap 13 4 (i64, i64, i64, i64) -> (i64) vfs (vfd, vzero, vzero, vzero)
  return vn
  }
}
"#;

/// Build the parent with `child_bytes` seeded at [`CHILD_BYTES_OFF`] as a data segment.
fn parent_with_child_bytes(child_bytes: &[u8]) -> Module {
    let mut parent = parse_module(PARENT).expect("parse parent");
    parent.data.push(Data {
        offset: CHILD_BYTES_OFF,
        readonly: true,
        bytes: child_bytes.to_vec(),
    });
    temen_verify::verify_module(&parent).expect("parent verifies");
    parent
}

/// Run [`PARENT`] over `child_bytes` with a forkable memfs granted as `"fs"`, a `ModuleLoader`, and an
/// `Instantiator`. Returns the parent's joined result and the shared `MemFsHandle`.
fn run_parent(child_bytes: &[u8]) -> (Vec<Value>, MemFsHandle) {
    let parent = parent_with_child_bytes(child_bytes);

    let (factory, handle) = temen_run::fs::mem_fs_shared_factory(vec![], vec![]);
    let factory = Arc::new(factory);

    let mut host = Host::new();
    let init: HostProc = (*factory)();
    let fork: HostProcFork = {
        let factory = Arc::clone(&factory);
        Arc::new(move |_pid| ForkedProc::shared((*factory)()))
    };
    let fs_h = host.grant_host_proc_forkable(init, fork);
    let inst = host.grant_instantiator(0, 1u64 << 17);
    let loader = temen_run::grant_module_loader(&mut host);

    let mut fuel = 200_000_000u64;
    let r = run_with_host(
        &parent,
        0,
        &[
            Value::I32(inst),
            Value::I32(loader),
            Value::I32(fs_h),
            Value::I32(child_bytes.len() as i32),
        ],
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

#[test]
fn guest_mints_a_module_from_bytes_then_spawns_and_runs_it() {
    // The child, encoded exactly as a host-granted module would be — the bytes a linker running
    // in-guest would emit.
    let child = parse_module(WRITER).expect("parse writer");
    temen_verify::verify_module(&child).expect("child verifies");
    assert_eq!(child.memory.expect("child window").size_log2, 16);
    let child_bytes = temen_encode::encode_module(&child);

    let (r, handle) = run_parent(&child_bytes);

    // The child ran: its `write` returned 9, joined back through op-1 — the whole loop, from guest
    // bytes to a running confined child, closed.
    assert!(
        matches!(r.as_slice(), [Value::I64(9)] | [Value::I32(9)]),
        "child (minted from bytes) wrote 9 bytes, status joined back: {r:?}"
    );
    assert_eq!(
        read_back(&handle, "out.bin"),
        b"hello.nif",
        "the child minted via `from_bytes` wrote its payload through the re-granted memfs; the parent \
         read it back — a module the guest promoted from its own bytes ran exactly like a host-granted one"
    );
}

// The parent-probe: `main(loader, blen)` returns the raw `from_bytes` result (the module handle, or a
// negative errno) so a caller can assert the fail-closed path without spawning.
const PROBE: &str = r#"
memory 17
func (i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32) {
  vptr = i64.const 20480
  vblen = i64.extend_i32_u v1
  vmh = call.cap 7 0 (i64, i64) -> (i64) v0 (vptr, vblen)
  return vmh
  }
}
"#;

fn probe(bytes: &[u8]) -> i64 {
    let mut m = parse_module(PROBE).expect("parse probe");
    m.data.push(Data {
        offset: CHILD_BYTES_OFF,
        readonly: true,
        bytes: bytes.to_vec(),
    });
    temen_verify::verify_module(&m).expect("probe verifies");
    let mut host = Host::new();
    let loader = temen_run::grant_module_loader(&mut host);
    let mut fuel = 50_000_000u64;
    let r = run_with_host(
        &m,
        0,
        &[Value::I32(loader), Value::I32(bytes.len() as i32)],
        &mut fuel,
        &mut host,
    )
    .expect("probe run");
    match r.as_slice() {
        [Value::I64(v)] => *v,
        [Value::I32(v)] => *v as i64,
        other => panic!("probe returned {other:?}"),
    }
}

#[test]
fn from_bytes_mints_a_handle_for_a_valid_module_and_fails_closed_on_garbage() {
    // A valid module decodes+verifies -> a non-negative `Module` handle.
    let good = temen_encode::encode_module(&parse_module(WRITER).expect("parse writer"));
    let h = probe(&good);
    assert!(
        h >= 0,
        "a valid module must mint a non-negative handle, got {h}"
    );

    // Garbage that is not a decodable/verifiable module fails closed to a negative errno — nothing
    // minted, no trap (guest-visible, non-fatal). Use bytes long enough to fill the read but that do
    // not decode as a module.
    let garbage = vec![0xFFu8; good.len().max(64)];
    let g = probe(&garbage);
    assert!(
        g < 0,
        "an unverifiable blob must fail closed to a negative errno, got {g}"
    );
}
