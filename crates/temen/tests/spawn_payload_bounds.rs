//! #2338 — a spawn's args payload is the guest's to size, so admission checks its length before it
//! reads a byte of it, on every engine. A length past the args region refuses `-EINVAL`: 2^32 and up
//! included, which a read confined at a `u32` width used to truncate and then allocate whole,
//! aborting the host. A payload the region holds but the window does not faults as any window read
//! does. The module resolves first on every engine, so a forged module faults the same way beside a
//! bad payload.

#[path = "../../temen-interp/tests/support/drivers.rs"]
mod drivers;
#[path = "../../temen-interp/tests/support/rec.rs"]
mod rec;

use temen_interp::{Host, MemLayout, Trap, Value};
use temen_ir::{Module, SpawnRec};
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

/// Where the parent keeps its spawn record.
const REC: u64 = 17408;
/// Where the parent's data segment keeps a one-byte payload, `*` (42).
const PAYLOAD: u64 = 20480;
/// The args region's size: a payload one byte longer refuses.
const ROOM: u64 = 16384 - 128;
/// An address past the parent's whole reservation.
const OUTSIDE: u64 = 1 << 40;

/// A parent `(inst, module, budget) -> i64` that makes the spawn `call`, then returns the child's
/// result, or the spawn's errno when it refuses. `call` sees `vi`, `vmod` and `vb` (the
/// `Instantiator`, the module and the `Budget`) and `vp` (the record).
fn parent(call: &str, data: &str) -> Module {
    module(&format!(
        "memory 17
func (i32, i32, i32) -> (i64) {{
block 0 (vi: i32, vmod: i32, vb: i32) {{
  vp = i64.const {REC}
{call}
  vz = i32.const 0
  vneg = i32.lt_s vh vz
  br_if vneg 1(vh) 2(vh, vi)
}}
block 1 (ve: i32) {{
  vr = i64.extend_i32_s ve
  return vr
}}
block 2 (vc: i32, vi2: i32) {{
  vj_st, vj = call.cap 6 1 (i32) -> (i64, i64) vi2 (vc)
  return vj
  }}
}}
data {PAYLOAD} \"*\"
{data}"
    ))
}

/// A parent spawning by a version-1 record whose payload is `(ptr, len)`, naming the granted module
/// (or the forged handle 999) and the `Budget`.
fn by_record(ptr: u64, len: u64, forged: bool) -> Module {
    let modh = if forged { "vm999" } else { "vmod" };
    parent(
        &format!(
            "  vm999 = i32.const 999
  vma = i64.const {}
  i32.store vma {modh}
  vba = i64.const {}
  i32.store vba vb
  vh = call.cap 6 17 (i64) -> (i32) vi (vp)",
            REC + rec::MODULE_AT,
            REC + rec::BUDGET_AT,
        ),
        &rec::segment(
            REC,
            &SpawnRec {
                args: (ptr, len),
                ..SpawnRec::v1(0)
            },
        ),
    )
}

/// A parent spawning by op 15's positional form, its payload `(ptr, len)`.
fn by_op15(ptr: u64, len: i64) -> Module {
    parent(
        &format!(
            "  vb64 = i64.extend_i32_u vb
  vm64 = i64.extend_i32_u vmod
  vz64 = i64.const 0
  vap = i64.const {ptr}
  val = i64.const {len}
  vh = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64, i64, i64) -> (i32) vi (vb64, vm64, vz64, vz64, vz64, vz64, vz64, vap, val)"
        ),
        "",
    )
}

/// The parent's powerbox: an `Instantiator`, the child module, and a `Budget` its window spends.
fn powerbox() -> (Host, [i32; 3]) {
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, 1 << 17);
    let child = host.grant_module(&module(CHILD));
    let budget = host.grant_budget(-1, 1 << 20, -1);
    (host, [inst, child, budget])
}

fn setup() -> (Host, Vec<Value>) {
    let (host, h) = powerbox();
    (host, h.map(Value::I32).to_vec())
}

/// `m` on the Cranelift JIT, as an embedder drives it (`temen_run::jit_cap_run`).
fn cranelift(m: &Module) -> Result<i64, Trap> {
    let (mut host, h) = powerbox();
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

/// `m` ends as `want` on every interpreter driver and the Cranelift JIT.
fn each_engine(what: &str, m: &Module, want: Result<i64, Trap>) {
    for driver in drivers::ALL {
        let ran = drivers::run_on(driver, m, &setup)
            .unwrap_or_else(|| panic!("{what}: {driver:?} declined the module"));
        let got = ran.result.map(|v| match v[..] {
            [Value::I64(r)] => r,
            _ => panic!("{what}: {driver:?} returned {v:?}"),
        });
        assert_eq!(got, want, "{what} on {driver:?}");
    }
    assert_eq!(cranelift(m), want, "{what} on the Cranelift JIT");
}

const EINVAL: Result<i64, Trap> = Ok(-22);

#[test]
fn a_length_past_the_args_region_refuses_before_anything_is_read() {
    each_engine(
        "args_len 2^34 + 1",
        &by_record(PAYLOAD, (1 << 34) + 1, false),
        EINVAL,
    );
    each_engine("args_len 2^63", &by_record(PAYLOAD, 1 << 63, false), EINVAL);
    each_engine("op 15's args_len -1", &by_op15(PAYLOAD, -1), EINVAL);
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
    each_engine("op 15's one-byte payload", &by_op15(PAYLOAD, 1), Ok(42));
}
