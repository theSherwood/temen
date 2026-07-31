//! FORK.md §8.5 — a program forking on svm with **real posix libc**, end to end under the manager
//! topology. This is the capstone of Track 2 minus the chibicc frontend: it exercises the same
//! wiring a compiled-C `fork()` will use — forkable libc (slice 1), libc re-granted into a nested
//! child (slice 3), and the manager/server/guest topology — with a hand-written-IR guest (so the
//! guest resolves its caps by name, sidestepping the `__px_` import-manifest binder that is the one
//! remaining piece for pure compiled-C).
//!
//! Topology (all one module; task ids are deterministic: manager root = 0, server = 1, guest = 2,
//! twin = 3):
//! - **manager** (func 0, args = instantiator + the granted libc handle): spawns the **server**
//!   (func 1), mints a `child_offer` over its `fork` export, then spawns the **guest** (func 3) via
//!   `instantiate_named`, re-granting BOTH the fork offer (as `"fork"`) and the libc (as `"libc"`)
//!   into it; joins the guest and returns its result.
//! - **server** (func 1): a `svc.wait` loop whose handler (func 2) runs **pid-mode `clone_caller`**.
//! - **guest** (func 3): resolves `"libc"` + `"fork"` by name, calls `fork()`, then BOTH copies
//!   `write(1, &ret, 8)` their fork return through the shared libc, and return it.
//!
//! The guest's fork() returns the twin's task id (3) in the original and 0 in the twin — POSIX
//! parent-sees-pid / child-sees-0 — and both writes land in the ONE shared libc memfs/stdout (the
//! twin's libc is the parent's, re-minted over the same `Inner`: fork-shares-open-file-descriptions).
//! Interp only: the serve-loop / caller-parking substrate `fork()` rides is eval-loop-only (as for
//! every `clone_caller` test).

use std::sync::Arc;

use svm_interp::{run_with_host, Host, Value};
use svm_text::parse_module;
use svm_verify::verify_module;

/// `"libc"` and `"fork"` as little-endian `i64`s, for the guest to stage in its window and resolve.
const LIBC_LE: i64 = 0x6362_696c; // b"libc"
const FORK_LE: i64 = 0x6b72_6f66; // b"fork"

const SRC: &str = r#"
memory 19
type 0 func (i64) -> (i64)
type 1 interface { op: 0 }
export 0 interface "fork" 1 { op: 2 }
data 300 "fork"
data 310 "libc"
func (i32, i32) -> (i64) {
block 0 (v0: i32, vlibc: i32) {
  ve1 = i64.const 1
  voffs = i64.const 262144
  vlog = i64.const 12
  vq = i64.const 0
  vs = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (ve1, voffs, vlog, vq)
  vz0 = i64.const 0
  vforkoff = cap.call 6 14 (i32, i64) -> (i32) v0 (vs, vz0)
  va0 = i64.const 256
  vnp0 = i32.const 300
  i32.store va0 vnp0
  va1 = i64.const 260
  vfour = i32.const 4
  i32.store va1 vfour
  va2 = i64.const 264
  i32.store va2 vforkoff
  va3 = i64.const 272
  vnp1 = i32.const 310
  i32.store va3 vnp1
  va4 = i64.const 276
  i32.store va4 vfour
  va5 = i64.const 280
  i32.store va5 vlibc
  vgp = i64.const 256
  vgn = i64.const 2
  ve3 = i64.const 3
  voffg = i64.const 266240
  vg = cap.call 6 11 (i64, i64, i64, i64, i64, i64) -> (i32) v0 (vgp, vgn, ve3, voffg, vlog, vq)
  vjg = cap.call 6 1 (i32) -> (i64) v0 (vg)
  return vjg
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vz = i32.const 0
  vn = cap.call 4294967295 10 () -> (i64) vz ()
  return vn
  }
}
func (i64) -> (i64) {
block 0 (vx: i64) {
  vz = i32.const 0
  vzero = i64.const 0
  vt = cap.call 4294967295 11 (i64) -> (i64) vz (vzero)
  return vt
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vln = i64.const 1667393900
  vz0 = i64.const 0
  i64.store vz0 vln
  vp0 = i64.const 0
  vl4 = i64.const 4
  vlibc = cap.self.resolve vp0 vl4
  vfn = i64.const 1802661734
  vz8 = i64.const 8
  i64.store vz8 vfn
  vp8 = i64.const 8
  vfork = cap.self.resolve vp8 vl4
  varg = i64.const 0
  vr = cap.call 268435456 0 (i64) -> (i64) vfork (varg)
  vp16 = i64.const 16
  i64.store vp16 vr
  vfd1 = i64.const 1
  veight = i64.const 8
  vw = cap.call 13 0 (i64, i64, i64) -> (i64) vlibc (vfd1, vp16, veight)
  return vr
  }
}
"#;

#[test]
fn a_guest_forks_with_real_libc_and_both_copies_write_through_the_shared_memfs() {
    // The two staged names must match the data-segment names the manager re-grants under.
    assert_eq!(LIBC_LE, 1_667_393_900);
    assert_eq!(FORK_LE, 1_802_661_734);

    let m = Arc::new(parse_module(SRC).expect("parse"));
    verify_module(&m).expect("verify");

    let mut host = Host::new();
    host.set_self_module(&m);
    let win = 1u64 << 19;
    // Forkable posix libc on the manager's host (slice 1). The guest inherits it re-granted (slice 3).
    let (libc, posix) = svm_posix::grant(&mut host, win / 2, win, Vec::new());
    let inst = host.grant_instantiator(0, win);

    let mut fuel = 40_000_000u64;
    let r = run_with_host(
        &m,
        0,
        &[Value::I32(inst), Value::I32(libc)],
        &mut fuel,
        &mut host,
    )
    .expect("run");

    // The manager returns the guest's fork() = the twin's task id (3) — POSIX parent-sees-pid.
    assert_eq!(
        r,
        vec![Value::I64(3)],
        "the original guest's fork() returns the twin's pid (task id 3)"
    );

    // Both copies wrote their 8-byte fork return through the ONE shared libc (the twin's libc is the
    // parent's, re-minted over the same Inner) — so the captured stdout holds both {0, 3}.
    let out = posix.stdout();
    assert_eq!(out.len(), 16, "two 8-byte writes reached the shared libc stdout");
    let mut vals: Vec<i64> = out
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    vals.sort();
    assert_eq!(
        vals,
        vec![0, 3],
        "child wrote 0, parent wrote its pid (3) — through the SAME forked libc; fork-shares-fds"
    );
}
