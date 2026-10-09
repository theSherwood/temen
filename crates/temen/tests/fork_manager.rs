//! FORK.md §8.5 — a program forking on temen with **real posix libc**, end to end under the manager
//! topology. This is the capstone of Track 2 minus the chibicc frontend: it exercises the same
//! wiring a compiled-C `fork()` will use — forkable libc (slice 1), libc re-granted into a child
//! (slice 3), and the manager/server/guest topology — with a hand-written-IR guest (so the
//! guest resolves its caps by name, sidestepping the `__px_` import-manifest binder that is the one
//! remaining piece for pure compiled-C).
//!
//! Topology (one program, whose server and guest run as its child images, #2219; task ids are
//! deterministic: manager root = 0, server = 1, guest = 2, twin = 3):
//! - **manager** (func 0, args = instantiator + the granted libc handle + a budget + the two
//!   images): spawns the **server** (func 1), mints a `child_offer` over its `fork` export, then
//!   spawns the **guest** (func 3), re-granting BOTH the fork offer (as `"fork"`) and the libc (as
//!   `"libc"`) into it; joins the guest and returns its result. Both children are detached, paid
//!   from the budget.
//! - **server** (func 1): a `svc.wait` loop whose handler (func 2) runs **pid-mode `clone_caller`**.
//! - **guest** (func 3): resolves `"libc"` + `"fork"` by name, calls `fork()` (retrying on the
//!   `-EAGAIN` serve/park race — the realistic `while ((pid = fork()) < 0)` shell idiom, see
//!   ISSUES.md I53), then BOTH copies `write(1, &ret, 8)` their fork return through the shared libc,
//!   and return it. The retry is deterministic-safe: a failed fork never consumes a task id, so the
//!   winning fork still mints twin id 3.
//!
//! The guest's fork() returns the twin's task id (3) in the original and 0 in the twin — POSIX
//! parent-sees-pid / child-sees-0 — and both writes land in the ONE shared libc memfs/stdout (the
//! twin's libc is the parent's, re-minted over the same `Inner`: fork-shares-open-file-descriptions).
//! Interp only: the serve-loop / caller-parking substrate `fork()` rides is eval-loop-only (as for
//! every `clone_caller` test).

#[path = "../../temen-interp/tests/support/rec.rs"]
mod rec;

use std::sync::Arc;

use temen_interp::{run_with_host, Host, Value};
use temen_ir::SpawnRec;
use temen_text::parse_module;
use temen_verify::verify_module;

const SRC: &str = r#"
memory 19
type 0 func (i64) -> (i64)
type 1 interface { op: 0 }
export 0 interface "fork" 1 { op: 2 }
data 16684 "fork"
data 16694 "libc"
func (i32, i32, i32, i32, i32) -> (i64) {
block 0 (v0: i32, vlibc: i32, vbud: i32, vsrv: i32, vgst: i32) {
  q0m = i64.const 17560
  i32.store q0m vsrv
  q0b = i64.const 17564
  i32.store q0b vbud
  q0p = i64.const 17536
  vs = call.cap 6 17 (i64) -> (i32) v0 (q0p)
  vz0 = i64.const 0
  vforkoff = call.cap 6 14 (i32, i64) -> (i32) v0 (vs, vz0)
  va0 = i64.const 16640
  vnp0 = i32.const 16684
  i32.store va0 vnp0
  va1 = i64.const 16644
  vfour = i32.const 4
  i32.store va1 vfour
  va2 = i64.const 16648
  i32.store va2 vforkoff
  va3 = i64.const 16656
  vnp1 = i32.const 16694
  i32.store va3 vnp1
  va4 = i64.const 16660
  i32.store va4 vfour
  va5 = i64.const 16664
  i32.store va5 vlibc
  q1m = i64.const 17688
  i32.store q1m vgst
  q1b = i64.const 17692
  i32.store q1b vbud
  q1p = i64.const 17664
  vg = call.cap 6 17 (i64) -> (i32) v0 (q1p)
  vjg = call.cap 6 1 (i32) -> (i64) v0 (vg)
  return vjg
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  br 1()
  }
block 1 () {
  vz = i32.const 0
  vn = call.cap 4294967295 10 () -> (i64) vz ()
  br 1()
  }
}
func (i64) -> (i64) {
block 0 (vx: i64) {
  vz = i32.const 0
  vzero = i64.const 0
  vt = call.cap 4294967295 11 (i64) -> (i64) vz (vzero)
  return vt
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vp0 = i64.const 16694
  vl4 = i64.const 4
  vlibc = self.resolve vp0 vl4
  vp8 = i64.const 16684
  vfork = self.resolve vp8 vl4
  br 1(vlibc, vfork)
}
block 1 (vlibc: i32, vfork: i32) {
  varg = i64.const 0
  vr = call.cap 268435456 0 (i64) -> (i64) vfork (varg)
  vzero = i64.const 0
  vforkfail = i64.lt_s vr vzero
  br_if vforkfail 1(vlibc, vfork) 2(vlibc, vr)
}
block 2 (vlibc: i32, vr: i64) {
  vp16 = i64.const 17408
  i64.store vp16 vr
  vfd1 = i64.const 1
  veight = i64.const 8
  vw = call.cap 13 0 (i64, i64, i64) -> (i64) vlibc (vfd1, vp16, veight)
  return vr
  }
}
"#;

/// [`SRC`] with its spawn records, both detached and paid from the manager's budget: the server at
/// 17536, and the guest at 17664, granted `"fork"` and `"libc"` by the list at 16640. Each is a
/// child image of the manager's program (#2219) the root is handed: the server of func 1, the guest
/// of func 3 ([`images`]).
fn src() -> String {
    let guest = SpawnRec {
        grants_ptr: 16640,
        grants_n: 2,
        ..SpawnRec::v1(0)
    };
    format!(
        "{SRC}{}{}",
        rec::segment(17536, &SpawnRec::v1(0)),
        rec::segment(17664, &guest)
    )
}

/// The manager's child images (#2219) of its server (func 1) and its guest (func 3), granted in
/// that order: the root's last two arguments.
fn images(host: &mut Host, m: &temen_ir::Module) -> [Value; 2] {
    [1, 3].map(|f| {
        let image = temen_ir::child_image_at(m, f).expect("child image");
        Value::I32(host.grant_module(&image))
    })
}

#[test]
fn a_guest_forks_with_real_libc_and_both_copies_write_through_the_shared_memfs() {
    let m = Arc::new(parse_module(&src()).expect("parse"));
    verify_module(&m).expect("verify");

    let mut host = Host::new();
    let win = 1u64 << 19;
    // Forkable posix libc on the manager's host (slice 1). The guest inherits it re-granted (slice 3).
    let (libc, posix) = temen_posix::grant(&mut host, win / 2, win, Vec::new());
    let inst = host.grant_instantiator(0, win);
    let budget = host.grant_budget(-1, 64 << 20, -1);
    let [server, guest] = images(&mut host, &m);

    let mut fuel = 40_000_000u64;
    let r = run_with_host(
        &m,
        0,
        &[
            Value::I32(inst),
            Value::I32(libc),
            Value::I32(budget),
            server,
            guest,
        ],
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
        "child wrote 0, parent wrote its pid (3) — through the SAME forked libc; fork-shares-fds"
    );
}

/// #2106 — the fork topology above with a guest that reads its budget around a fork. The manager
/// also serves `wait` (export 1, func 4, `reap`), and the guest (func 3) resolves `"budget"`, the node
/// that paid for its window (and the server's). It forks, retrying the serve/park race at most 64
/// times; if none succeeds it reports the node's `mem` room and returns the last refusal. Otherwise
/// the twin reports the room while it lives, and the parent reaps it, then reports the room again
/// and returns it.
const BUDGET_SRC: &str = r#"
memory 19
type 0 func (i64) -> (i64)
type 1 interface { op: 0 }
export 0 interface "fork" 1 { op: 2 }
export 1 interface "wait" 1 { op: 4 }
data 16800 "fork"
data 16810 "libc"
data 16820 "wait"
data 16830 "budget"
func (i32, i32, i32, i32, i32) -> (i64) {
block 0 (v0: i32, vlibc: i32, vbud: i32, vsrv: i32, vgst: i32) {
  q0m = i64.const 17560
  i32.store q0m vsrv
  q0b = i64.const 17564
  i32.store q0b vbud
  q0p = i64.const 17536
  vs = call.cap 6 17 (i64) -> (i32) v0 (q0p)
  vz0 = i64.const 0
  vforkoff = call.cap 6 14 (i32, i64) -> (i32) v0 (vs, vz0)
  vo1 = i64.const 1
  vwaitoff = call.cap 6 14 (i32, i64) -> (i32) v0 (vs, vo1)
  vfour = i32.const 4
  va0 = i64.const 16640
  vnp0 = i32.const 16800
  i32.store va0 vnp0
  va1 = i64.const 16644
  i32.store va1 vfour
  va2 = i64.const 16648
  i32.store va2 vforkoff
  va3 = i64.const 16656
  vnp1 = i32.const 16810
  i32.store va3 vnp1
  va4 = i64.const 16660
  i32.store va4 vfour
  va5 = i64.const 16664
  i32.store va5 vlibc
  va6 = i64.const 16672
  vnp2 = i32.const 16820
  i32.store va6 vnp2
  va7 = i64.const 16676
  i32.store va7 vfour
  va8 = i64.const 16680
  i32.store va8 vwaitoff
  q1m = i64.const 17688
  i32.store q1m vgst
  q1b = i64.const 17692
  i32.store q1b vbud
  q1p = i64.const 17664
  vg = call.cap 6 17 (i64) -> (i32) v0 (q1p)
  vjg = call.cap 6 1 (i32) -> (i64) v0 (vg)
  return vjg
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  br 1()
  }
block 1 () {
  vz = i32.const 0
  vn = call.cap 4294967295 10 () -> (i64) vz ()
  br 1()
  }
}
func (i64) -> (i64) {
block 0 (vx: i64) {
  vz = i32.const 0
  vzero = i64.const 0
  vt = call.cap 4294967295 11 (i64) -> (i64) vz (vzero)
  return vt
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vl4 = i64.const 4
  vpl = i64.const 16810
  vlibc = self.resolve vpl vl4
  vpf = i64.const 16800
  vfork = self.resolve vpf vl4
  vpw = i64.const 16820
  vwait = self.resolve vpw vl4
  vpb = i64.const 16830
  vl6 = i64.const 6
  vbud = self.resolve vpb vl6
  vtries = i64.const 0
  br 1(vlibc, vfork, vwait, vbud, vtries)
}
block 1 (vlibc: i32, vfork: i32, vwait: i32, vbud: i32, vn: i64) {
  varg = i64.const 0
  vr = call.cap 268435456 0 (i64) -> (i64) vfork (varg)
  vzr = i64.const 0
  vfail = i64.lt_s vr vzr
  br_if vfail 2(vlibc, vfork, vwait, vbud, vn, vr) 3(vlibc, vwait, vbud, vr)
}
block 2 (vlibc2: i32, vfork2: i32, vwait2: i32, vbud2: i32, vn2: i64, vr2: i64) {
  vone2 = i64.const 1
  vnext = i64.add vn2 vone2
  vmax = i64.const 64
  vmore = i64.lt_s vnext vmax
  br_if vmore 1(vlibc2, vfork2, vwait2, vbud2, vnext) 7(vr2, vlibc2, vbud2)
}
block 3 (vlibc3: i32, vwait3: i32, vbud3: i32, vpid: i64) {
  vz3 = i64.const 0
  vtwin = i64.eq vpid vz3
  br_if vtwin 4(vlibc3, vbud3) 5(vlibc3, vwait3, vbud3, vpid)
}
block 4 (vlibc4: i32, vbud4: i32) {
  vone4 = i64.const 1
  vroom4 = call.cap 14 1 (i64) -> (i64) vbud4 (vone4)
  vp4 = i64.const 17408
  i64.store vp4 vroom4
  vfd4 = i64.const 1
  veight4 = i64.const 8
  vw4 = call.cap 13 0 (i64, i64, i64) -> (i64) vlibc4 (vfd4, vp4, veight4)
  vz4 = i64.const 0
  return vz4
}
block 5 (vlibc5: i32, vwait5: i32, vbud5: i32, vpid5: i64) {
  vs5 = call.cap 268435456 0 (i64) -> (i64) vwait5 (vpid5)
  vz5 = i64.const 0
  vwfail = i64.lt_s vs5 vz5
  br_if vwfail 5(vlibc5, vwait5, vbud5, vpid5) 6(vlibc5, vbud5)
}
block 6 (vlibc6: i32, vbud6: i32) {
  vone6 = i64.const 1
  vroom6 = call.cap 14 1 (i64) -> (i64) vbud6 (vone6)
  vp6 = i64.const 17416
  i64.store vp6 vroom6
  vfd6 = i64.const 1
  veight6 = i64.const 8
  vw6 = call.cap 13 0 (i64, i64, i64) -> (i64) vlibc6 (vfd6, vp6, veight6)
  return vroom6
}
block 7 (vlast: i64, vlibc7: i32, vbud7: i32) {
  vone7 = i64.const 1
  vroom7 = call.cap 14 1 (i64) -> (i64) vbud7 (vone7)
  vp7 = i64.const 17424
  i64.store vp7 vroom7
  vfd7 = i64.const 1
  veight7 = i64.const 8
  vw7 = call.cap 13 0 (i64, i64, i64) -> (i64) vlibc7 (vfd7, vp7, veight7)
  return vlast
}
}
func (i64) -> (i64) {
block 0 (vpid: i64) {
  vz = i32.const 0
  vt = call.cap 4294967295 12 (i64) -> (i64) vz (vpid)
  return vt
  }
}
"#;

/// The guest's window, and the server's: both run [`BUDGET_SRC`], whose window is 512 KiB.
const WINDOW: i64 = 1 << 19;

/// Run [`BUDGET_SRC`] with its server and guest paid from a budget whose `mem` ceiling is `ceiling`,
/// on the tree-walker or the bytecode engine (both serve `clone_caller`): the manager's result, the
/// rooms the guest's copies wrote while the run was live, and the `mem` the budget still holds once
/// the run is over. The fork server is still live at the run's end, and ends with it (#2006).
fn run_budgeted(ceiling: i64, bytecode: bool) -> (Vec<Value>, Vec<i64>, i64) {
    let guest = SpawnRec {
        grants_ptr: 16640,
        grants_n: 3,
        ..SpawnRec::v1(0)
    };
    let src = format!(
        "{BUDGET_SRC}{}{}",
        rec::segment(17536, &SpawnRec::v1(0)),
        rec::segment(17664, &guest)
    );
    let m = Arc::new(parse_module(&src).expect("parse"));
    verify_module(&m).expect("verify");
    let mut host = Host::new();
    let win = WINDOW as u64;
    let (libc, posix) = temen_posix::grant(&mut host, win / 2, win, Vec::new());
    let inst = host.grant_instantiator(0, win);
    let budget = host.grant_budget(-1, ceiling, -1);
    let [server, guest] = images(&mut host, &m);
    let mut fuel = 40_000_000u64;
    let args = [
        Value::I32(inst),
        Value::I32(libc),
        Value::I32(budget),
        server,
        guest,
    ];
    let r = if bytecode {
        temen_interp::bytecode::compile_and_run_with_host(&m, 0, &args, &mut fuel, &mut host)
            .expect("the fork module runs natively on the bytecode engine")
    } else {
        run_with_host(&m, 0, &args, &mut fuel, &mut host)
    }
    .expect("run");
    let wrote = posix
        .stdout()
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let held = host
        .capture_durable_budgets()
        .into_iter()
        .find(|n| n.parent.is_none())
        .expect("the granted budget")
        .used
        .mem;
    (r, wrote, held)
}

#[test]
fn a_fork_twin_pays_for_its_copy_of_the_window_while_it_lives() {
    let ceiling = 8 * WINDOW;
    for bytecode in [false, true] {
        let (r, wrote, held) = run_budgeted(ceiling, bytecode);
        assert_eq!(
            wrote,
            vec![ceiling - 3 * WINDOW, ceiling - 2 * WINDOW],
            "bytecode={bytecode}: the live twin's copy is charged with the server's and the guest's \
             windows; reaping it hands the copy back"
        );
        assert_eq!(
            r,
            vec![Value::I64(ceiling - 2 * WINDOW)],
            "bytecode={bytecode}"
        );
        assert_eq!(
            held, 0,
            "bytecode={bytecode}: the run's end hands every window back"
        );
    }
}

#[test]
fn a_fork_with_no_room_for_the_twins_window_is_refused() {
    let ceiling = 3 * WINDOW - 1;
    for bytecode in [false, true] {
        let (r, wrote, held) = run_budgeted(ceiling, bytecode);
        assert_eq!(
            r,
            vec![Value::I64(temen_ir::errno::EAGAIN)],
            "bytecode={bytecode}: every fork is refused, as the server's and the guest's windows \
             leave no room for a copy"
        );
        assert_eq!(
            wrote,
            vec![ceiling - 2 * WINDOW],
            "bytecode={bytecode}: the refusals charged nothing"
        );
        assert_eq!(
            held, 0,
            "bytecode={bytecode}: the run's end hands every window back"
        );
    }
}
