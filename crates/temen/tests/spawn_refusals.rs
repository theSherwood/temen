//! #2864 — a refused spawn record (op 17) answers its cause's own code, the same on every interpreter
//! driver and the Cranelift JIT ([`SpawnRefusal`]), and charges nothing. So a spawner can say why its
//! child did not start: a program's language decides its imports, and a grant list that misses one
//! answers `-ENOENT`, not the `-EINVAL` every refusal used to share.
//!
//! #2338 — a spawn's args payload is the guest's to size, so admission checks its length before it
//! reads a byte of it, on every engine. A length past the args region refuses `-EINVAL`: 2^32 and up
//! included, which a read confined at a `u32` width used to truncate and then allocate whole,
//! aborting the host. A payload the region holds but the window does not faults as any window read
//! does. The module resolves first on every engine, so a forged module faults the same way beside a
//! bad payload.
//!
//! A durable spawner's refusals are pinned beside its freezes (`durable_detached_jit.rs`), and a
//! pre-mapped region's in `crates/temen-interp/tests/detached_premap.rs`.

#[path = "../../temen-interp/tests/support/drivers.rs"]
mod drivers;
#[path = "../../temen-interp/tests/support/rec.rs"]
mod rec;

use temen_interp::{cap_id, Host, MemLayout, Trap, Value};
use temen_ir::{Module, SpawnRec, SpawnRefusal};
use temen_jit::JitOutcome;

fn module(src: &str) -> Module {
    let m = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// The child: a 64 KiB window, which holds the args region; its entry returns its first args byte.
const CHILD: &str = "memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  va = i64.const 16512
  vb = i32.load8_u va
  vr = i64.extend_i32_u vb
  return vr
  }
}
";

/// A child whose function 0 takes an `i32`: no child entry ([`temen_ir::child_entry_ok`]).
const CHILD_NO_ENTRY: &str = "memory 16
func (i32) -> (i64) {
block 0 (v0: i32) {
  vr = i64.const 7
  return vr
  }
}
";

/// A child that imports `exit`, which no grant binds.
const CHILD_IMPORTS_EXIT: &str = "memory 16
import 0 \"exit\" (i32) -> ()
func (i64) -> (i64) {
block 0 (v0: i64) {
  vr = i64.const 42
  return vr
  }
}
";

/// Where the parent keeps its spawn record.
const REC: u64 = 17408;
/// Where the parent's data segment keeps a one-byte payload, `*` (42).
const PAYLOAD: u64 = 20480;
/// The args region's size: a payload one byte longer refuses.
const ROOM: u64 = 16384 - 128;
/// An address past the parent's whole reservation.
const OUTSIDE: u64 = 1 << 40;
/// The child's window, which a budget of exactly this much pays for.
const WINDOW: i64 = 1 << 16;

/// The tail of a parent whose spawn returned `vh`: a refusal (negative) returns its `-errno`, and an
/// admitted child is joined and its result returned.
const JOIN_OR_ERRNO: &str = "  vz = i32.const 0
  vneg = i32.lt_s vh vz
  br_if vneg 1(vh) 2(vh, vi)
}
block 1 (ve: i32) {
  vr = i64.extend_i32_s ve
  return vr
}
block 2 (vc: i32, vi2: i32) {
  vj_st, vj = call.cap 6 1 (i32) -> (i64, i64) vi2 (vc)
  return vj
  }
}
";

/// #1975 — the tail of a parent that reports what a refused spawn left in its budget: the `mem` room,
/// or `-1` if the spawn was admitted after all.
const ROOM_AFTER_REFUSAL: &str = "  vz = i32.const 0
  vneg = i32.lt_s vh vz
  br_if vneg 1(vb) 2()
}
block 1 (vb1: i32) {
  vf = i64.const 1
  vroom = call.cap 14 1 (i64) -> (i64) vb1 (vf)
  return vroom
}
block 2 () {
  vm = i64.const -1
  return vm
  }
}
";

/// A parent `(inst, module, budget) -> i64` that makes the spawn `call`, which leaves its answer in
/// `vh`, then ends in `tail`. `call` sees `vi`, `vmod` and `vb` (the `Instantiator`, the module and
/// the `Budget`) and `vp` (the record).
fn parent(call: &str, tail: &str, data: &str) -> Module {
    module(&format!(
        "memory 17
func (i32, i32, i32) -> (i64) {{
block 0 (vi: i32, vmod: i32, vb: i32) {{
  vp = i64.const {REC}
{call}
{tail}data {PAYLOAD} \"*\"
{data}"
    ))
}

/// A parent spawning by `rec`, its module the granted one (or the forged handle 999) and its budget
/// the handle in register `budget` (`vb`, or `vi` for a capability that is not a `Budget`).
fn spawning(rec: SpawnRec, forged: bool, budget: &str, tail: &str) -> Module {
    let modh = if forged { "vm999" } else { "vmod" };
    parent(
        &format!(
            "  vm999 = i32.const 999
  vma = i64.const {}
  i32.store vma {modh}
  vba = i64.const {}
  i32.store vba {budget}
  vh = call.cap 6 17 (i64) -> (i32) vi (vp)",
            REC + SpawnRec::MODULE_AT,
            REC + SpawnRec::BUDGET_AT,
        ),
        tail,
        &rec::segment(REC, &rec),
    )
}

/// A parent spawning by a record whose payload is `(ptr, len)`, naming the granted module (or the
/// forged handle 999) and the `Budget`.
fn by_record(ptr: u64, len: u64, forged: bool) -> Module {
    let rec = SpawnRec {
        args: (ptr, len),
        ..SpawnRec::default()
    };
    spawning(rec, forged, "vb", JOIN_OR_ERRNO)
}

/// A parent spawning the child its powerbox holds by a record with a one-byte payload.
fn plain(budget: &str, tail: &str) -> Module {
    by_record_then(SpawnRec::default(), budget, tail)
}

fn by_record_then(rec: SpawnRec, budget: &str, tail: &str) -> Module {
    let rec = SpawnRec {
        args: (PAYLOAD, 1),
        ..rec
    };
    spawning(rec, false, budget, tail)
}

/// What the parent's powerbox holds: the child it spawns, the `mem` of the `Budget` it pays with, and
/// its lanes (`Some((cap, lane))`: the spawner's lane cap, and the lane its budget carries).
#[derive(Clone, Copy)]
struct Powerbox {
    child: &'static str,
    mem: i64,
    lanes: Option<(i64, i64)>,
}

const PLAIN: Powerbox = Powerbox {
    child: CHILD,
    mem: 1 << 20,
    lanes: None,
};

/// The parent's powerbox: an `Instantiator`, the child module, and a `Budget` its window spends.
fn powerbox(p: Powerbox) -> (Host, [i32; 3]) {
    let mut host = Host::new();
    let inst = host.grant_instantiator();
    let child = host.grant_module(&module(p.child));
    let mut budget = host.grant_budget(-1, p.mem, -1);
    if let Some((cap, lane)) = p.lanes {
        // A budget split off with its own lane, then the spawner's cap, which the lane is wider than.
        let split = host.cap_call(
            cap_id::BUDGET,
            0,
            budget,
            &[-1, p.mem, -1, -1, lane],
            1,
            None,
        );
        budget = split.expect("the split")[0] as i32;
        host.set_lane_cap(cap);
    }
    (host, [inst, child, budget])
}

/// `m` on the Cranelift JIT, as an embedder drives it (`temen_run::jit_cap_run`).
fn cranelift(m: &Module, p: Powerbox) -> Result<i64, Trap> {
    let (mut host, h) = powerbox(p);
    let out = temen_run::jit_cap_run(
        m,
        0,
        &h.map(i64::from),
        &MemLayout::image(Vec::new()),
        temen_ir::DEFAULT_RESERVED_LOG2,
        0,
        &mut host,
        None,
    )
    .expect("the JIT compiles the module")
    .0;
    match out {
        JitOutcome::Returned(v) => Ok(v[0]),
        JitOutcome::Trapped(k) => Err(Trap::from_code(k.code()).expect("a trap kind is a trap")),
        o => panic!("unexpected JIT outcome {o:?}"),
    }
}

/// `m` ends as `want` on every interpreter driver and the Cranelift JIT, over powerbox `p`.
fn each_engine_over(what: &str, m: &Module, p: Powerbox, want: Result<i64, Trap>) {
    let setup = || {
        let (host, h) = powerbox(p);
        (host, h.map(Value::I32).to_vec())
    };
    for driver in drivers::ALL {
        let ran = drivers::run_on(driver, m, &setup)
            .unwrap_or_else(|| panic!("{what}: {driver:?} declined the module"));
        let got = ran.result.map(|v| match v[..] {
            [Value::I64(r)] => r,
            _ => panic!("{what}: {driver:?} returned {v:?}"),
        });
        assert_eq!(got, want, "{what} on {driver:?}");
    }
    assert_eq!(cranelift(m, p), want, "{what} on the Cranelift JIT");
}

fn each_engine(what: &str, m: &Module, want: Result<i64, Trap>) {
    each_engine_over(what, m, PLAIN, want);
}

const fn refused(r: SpawnRefusal) -> Result<i64, Trap> {
    Ok(r.errno())
}

const EINVAL: Result<i64, Trap> = refused(SpawnRefusal::Record);

/// The control: the child each refusal below changes one thing about starts on every engine.
#[test]
fn a_record_with_nothing_wrong_spawns() {
    each_engine("a one-byte payload", &plain("vb", JOIN_OR_ERRNO), Ok(42));
}

#[test]
fn each_refusal_answers_its_own_code_on_every_engine() {
    let window = SpawnRec {
        size_log2: 17,
        ..SpawnRec::default()
    };
    each_engine(
        "a window the module does not declare",
        &by_record_then(window, "vb", JOIN_OR_ERRNO),
        refused(SpawnRefusal::Record),
    );
    each_engine_over(
        "a module whose function 0 is no child entry",
        &plain("vb", JOIN_OR_ERRNO),
        Powerbox {
            child: CHILD_NO_ENTRY,
            ..PLAIN
        },
        refused(SpawnRefusal::Module),
    );
    each_engine(
        "a capability that is not a Budget",
        &plain("vi", JOIN_OR_ERRNO),
        refused(SpawnRefusal::Budget),
    );
    each_engine_over(
        "a budget one byte short of the window",
        &plain("vb", JOIN_OR_ERRNO),
        Powerbox {
            mem: WINDOW - 1,
            ..PLAIN
        },
        refused(SpawnRefusal::Room),
    );
    each_engine_over(
        "a budget whose lane is wider than the spawner's",
        &plain("vb", JOIN_OR_ERRNO),
        Powerbox {
            lanes: Some((1, 2)),
            ..PLAIN
        },
        refused(SpawnRefusal::Lanes),
    );
    each_engine_over(
        "an import no grant binds",
        &plain("vb", JOIN_OR_ERRNO),
        Powerbox {
            child: CHILD_IMPORTS_EXIT,
            ..PLAIN
        },
        refused(SpawnRefusal::Unbound),
    );
}

/// A record with two causes answers the first, in the order every engine checks: the entry before
/// the window, and the record before the budget.
#[test]
fn a_record_with_two_causes_answers_the_first_on_every_engine() {
    let window = SpawnRec {
        size_log2: 17,
        ..SpawnRec::default()
    };
    each_engine_over(
        "no child entry, and a window the module does not declare",
        &by_record_then(window, "vb", JOIN_OR_ERRNO),
        Powerbox {
            child: CHILD_NO_ENTRY,
            ..PLAIN
        },
        refused(SpawnRefusal::Module),
    );
    each_engine(
        "a window the module does not declare, and no Budget",
        &by_record_then(window, "vi", JOIN_OR_ERRNO),
        refused(SpawnRefusal::Record),
    );
    each_engine_over(
        "an import no grant binds, and a budget that cannot pay",
        &plain("vb", JOIN_OR_ERRNO),
        Powerbox {
            child: CHILD_IMPORTS_EXIT,
            mem: WINDOW - 1,
            lanes: None,
        },
        refused(SpawnRefusal::Room),
    );
}

/// A refusal charges nothing. The imports bind after the budget pays for the window, so a refusal
/// there hands the window back: a budget of exactly one window still has all its room.
#[test]
fn a_refusal_after_the_budget_paid_charges_nothing_on_every_engine() {
    each_engine_over(
        "an import no grant binds",
        &plain("vb", ROOM_AFTER_REFUSAL),
        Powerbox {
            child: CHILD_IMPORTS_EXIT,
            mem: WINDOW,
            lanes: None,
        },
        Ok(WINDOW),
    );
}

#[test]
fn a_length_past_the_args_region_refuses_before_anything_is_read() {
    each_engine(
        "args_len 2^34 + 1",
        &by_record(PAYLOAD, (1 << 34) + 1, false),
        EINVAL,
    );
    each_engine("args_len 2^63", &by_record(PAYLOAD, 1 << 63, false), EINVAL);
    each_engine(
        "args_len 2^64 - 1",
        &by_record(PAYLOAD, u64::MAX, false),
        EINVAL,
    );
    each_engine(
        "args_len one past the region",
        &by_record(PAYLOAD, ROOM + 1, false),
        EINVAL,
    );
    // Nothing is read, so a range outside the window is not reached.
    each_engine(
        "an over-long payload outside the window",
        &by_record(OUTSIDE, ROOM + 1, false),
        EINVAL,
    );
}

#[test]
fn a_payload_the_window_does_not_hold_faults_after_the_module_resolves() {
    each_engine(
        "a payload outside the window",
        &by_record(OUTSIDE, 8, false),
        Err(Trap::MemoryFault),
    );
    each_engine(
        "a forged module beside a payload outside the window",
        &by_record(OUTSIDE, 8, true),
        Err(Trap::CapFault),
    );
}

/// The control: a payload the region holds reaches the child on every engine.
#[test]
fn a_payload_the_region_holds_reaches_the_child() {
    each_engine("a one-byte payload", &by_record(PAYLOAD, 1, false), Ok(42));
}

/// A parent spawning the child its powerbox holds by a record granting the forged handle 999 as
/// `stdout`.
fn forged_grant() -> Module {
    let (at, name_at) = (18432u64, 18560u64);
    let grant = [name_at as u32, 6, 999, 0].map(u32::to_le_bytes).concat();
    let grant: String = grant.iter().map(|b| format!("\\x{b:02x}")).collect();
    let rec = SpawnRec {
        args: (PAYLOAD, 1),
        grants_ptr: at,
        grants_n: 1,
        ..SpawnRec::default()
    };
    let data = format!(
        "{}data {name_at} \"stdout\"\ndata {at} \"{grant}\"\n",
        rec::segment(REC, &rec)
    );
    parent(
        &format!(
            "  vma = i64.const {}
  i32.store vma vmod
  vba = i64.const {}
  i32.store vba vb
  vh = call.cap 6 17 (i64) -> (i32) vi (vp)",
            REC + SpawnRec::MODULE_AT,
            REC + SpawnRec::BUDGET_AT,
        ),
        JOIN_OR_ERRNO,
        &data,
    )
}

/// The grant list is checked before any refusal, so a forged grant traps whatever else the record
/// says, on every engine. The JIT read its grants only once the budget had paid, so beside a budget
/// that cannot pay it answered the refusal where the interpreters trapped (#2864).
#[test]
fn a_forged_grant_traps_before_any_refusal_on_every_engine() {
    let m = forged_grant();
    each_engine("a forged grant", &m, Err(Trap::CapFault));
    each_engine_over(
        "a forged grant, and a budget that cannot pay",
        &m,
        Powerbox {
            mem: WINDOW - 1,
            ..PLAIN
        },
        Err(Trap::CapFault),
    );
}
