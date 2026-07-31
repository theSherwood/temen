//! FORK.md §8.5 slice 2 — the fork offer bound to a **separate-module child's named `fork` import** by
//! the child-manifest binder, rather than discovered with `cap.self.resolve`. This is the binding half a
//! compiled-C `fork()` needs: chibicc emits a `call.import "fork"` in a *separate* module (its own
//! program), and the nested spawn's `bind_child_manifest` must route that slot to the live fork offer.
//! The capstone (`fork_manager.rs`) proved the *substrate*; this proves the *binding*, so swapping the
//! hand-written guest for a chibicc one is a frontend change, not a runtime one.
//!
//! Two things had to be true and one was not:
//! - The guest must be a **separate module** (`instantiate_module_named`, op 13) so the manifest binder
//!   runs at all — a same-module `instantiate_named` (op 11) child just inherits the parent's bindings.
//!   A compiled-C command is always its own module, so this is the realistic shape.
//! - `bind_child_manifest`'s named-offer step used to resolve only wired `offer_func`s
//!   (`Binding::Offer`). The fork offer minted by `child_offer` is a **live-callee** offer
//!   (`Binding::LiveImpl` — a call parks the caller on the server's serve loop), which `resolve_offer`
//!   does not return, so a named `fork` import could not bind to it. The binder now also matches a
//!   LiveImpl offer by signature (it is name-less at that layer) and binds the slot; `call.import` then
//!   rides the same caller-parking path `cap.self.resolve` + `cap.call` did.
//!
//! Topology (manager root = 0, server = 1, guest = 2, twin = 3): the manager (its own module) spawns the
//! server (a same-module `svc.wait` loop whose handler runs pid-mode `clone_caller`), mints a
//! `child_offer` over its `fork` export, then spawns the **guest module** via op 13 re-granting BOTH the
//! fork offer (named `"fork"`, the guest's import name) and the forkable posix libc (named `"libc"`), and
//! joins. The guest calls `fork()` **through its import**, both copies `write(1, &ret, 8)` through the
//! shared libc, and it returns the fork result. Parent sees the twin's pid (3), the twin sees 0; both
//! writes land in the ONE shared libc memfs. Interp only (the serve substrate is eval-loop-only).

use std::sync::Arc;

use svm_interp::{run_with_host, Host, Value};
use svm_text::parse_module;
use svm_verify::verify_module;

/// `"libc"` as a little-endian `i64`, staged by the guest for `cap.self.resolve` (fork is an import, so
/// only libc is discovered dynamically).
const LIBC_LE: i64 = 0x6362_696c; // b"libc"

/// The manager program: `main(inst, libc, guestmod)` spawns the server, mints the fork offer, builds a
/// 2-entry grant list {`"fork"` → offer, `"libc"` → libc} at window offset 256, spawns the guest module
/// via op 13 into a 4 KiB carve at 266240, and joins it. Server = func 1, handler = func 2.
const MANAGER: &str = r#"
memory 19
type 0 func (i64) -> (i64)
type 1 interface { op: 0 }
export 0 interface "fork" 1 { op: 2 }
data 300 "fork"
data 310 "libc"
func (i32, i32, i64) -> (i64) {
block 0 (v0: i32, vlibc: i32, vgmod: i64) {
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
  ve0 = i64.const 0
  voffg = i64.const 266240
  vg = cap.call 6 13 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vgmod, vgp, vgn, ve0, voffg, vlog, vq)
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
"#;

/// The guest program (its own module, entry 0): resolve `"libc"` by name, call `fork()` **through the
/// named import**, write the 8-byte result through libc fd 1, return it.
const GUEST: &str = r#"
memory 12
type 0 func (i64) -> (i64)
import 0 func "fork" 0
func (i64) -> (i64) {
block 0 (v0: i64) {
  vln = i64.const 1667393900
  vz0 = i64.const 0
  i64.store vz0 vln
  vp0 = i64.const 0
  vl4 = i64.const 4
  vlibc = cap.self.resolve vp0 vl4
  varg = i64.const 0
  vr = call.import 0 (varg)
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
fn a_childs_named_fork_import_binds_to_the_live_fork_offer_and_forks_twice() {
    assert_eq!(LIBC_LE, 1_667_393_900);

    let manager = Arc::new(parse_module(MANAGER).expect("parse manager"));
    verify_module(&manager).expect("verify manager");
    let guest = parse_module(GUEST).expect("parse guest");
    verify_module(&guest).expect("verify guest");

    let mut host = Host::new();
    host.set_self_module(&manager);
    let win = 1u64 << 19;
    let (libc, posix) = svm_posix::grant(&mut host, win / 2, win, Vec::new());
    let inst = host.grant_instantiator(0, win);
    let gmod = host.grant_module(&guest);

    let mut fuel = 40_000_000u64;
    let r = run_with_host(
        &manager,
        0,
        &[Value::I32(inst), Value::I32(libc), Value::I64(gmod as i64)],
        &mut fuel,
        &mut host,
    )
    .expect("run");

    assert_eq!(
        r,
        vec![Value::I64(3)],
        "the guest's imported fork() returns the twin's pid (task id 3)"
    );

    let out = posix.stdout();
    assert_eq!(
        out.len(),
        16,
        "two 8-byte writes reached the shared libc stdout"
    );
    let mut vals: Vec<i64> = out
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    vals.sort();
    assert_eq!(
        vals,
        vec![0, 3],
        "child wrote 0, parent wrote its pid (3) — fork bound through the child module's named import"
    );
}
