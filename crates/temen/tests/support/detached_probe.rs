//! #1867 decision 3 — the escape oracle's **detached-child placement**. A small root spawns a module's
//! entry as a detached child through a v1 record, so the child runs in a window of its own (base 0, its
//! declared size). The root waits for the child and joins it if it returned, then reports how the
//! child ended and whether its own window changed across the child's life: the canary. Every engine is
//! held to the same report, the bytecode drivers included, which return no window snapshot.
//!
//! Included by `irgen.rs` (the differential's detached pass) and `escape_oracle.rs` (the hand-written
//! pins).
#![allow(dead_code)] // each including binary uses a different subset

use temen_interp::{run_capture_reserved_with_host, Host, Trap, Value};
use temen_ir::Module;
use temen_jit::{compile_and_run_capture_reserved_with_host_ex, JitError, JitOutcome};

/// The root's declared window, `log2`.
pub const ROOT_LOG2: u8 = 16;

/// Where the root builds its spawn record: above the NULL guard and the durable control words.
const REC_AT: u64 = 17408;

/// The text of a function `() -> (i64)` that digests its module's window `[guard, size)`: an FNV-1a
/// over the window's i64 words, so a change to any byte changes the digest. The NULL guard below is
/// never mapped, so it is not read.
pub fn digest_func(size: u64) -> String {
    let guard = temen_ir::POWERBOX_NULL_GUARD;
    format!(
        "func () -> (i64) {{
block 0 () {{
  da = i64.const {guard}
  dh = i64.const {basis}
  br 1(da, dh)
}}
block 1 (dp: i64, dx: i64) {{
  dend = i64.const {size}
  dmore = i64.lt_u dp dend
  br_if dmore 2(dp, dx) 3(dx)
}}
block 2 (dq: i64, dy: i64) {{
  dw = i64.load dq
  dz = i64.xor dy dw
  dprime = i64.const 1099511628211
  dy2 = i64.mul dz dprime
  deight = i64.const 8
  dq2 = i64.add dq deight
  br 1(dq2, dy2)
}}
block 3 (dr: i64) {{
  return dr
  }}
}}
",
        basis = 0xcbf2_9ce4_8422_2325u64 as i64,
    )
}

/// The root module: `(instantiator, module, budget) -> (trapped, value, canary)`. It fills the granted
/// `module` and `budget` into a v1 record for `entry` (the module's declared window, no grants),
/// digests its own window, spawns the child, and waits for it (`Instantiator.wait`, which answers how
/// the child ended without inheriting its trap). For a child that returned it joins the child: `trapped`
/// is 0 and `value` the child's result. For one that trapped, `trapped` is 1 and `value` the trap's
/// wire code. `canary` is the root's window digest before the spawn xor after the child's life: 0 when
/// the child never touched the root's window.
pub fn root(entry: u32) -> Module {
    let rec = temen_ir::SpawnRec::v1(entry).encode();
    let rec: String = rec.iter().map(|b| format!("\\x{b:02x}")).collect();
    let src = format!(
        "memory {ROOT_LOG2}
data {REC_AT} \"{rec}\"
func (i32, i32, i32) -> (i64, i64, i64) {{
block 0 (vi: i32, vm: i32, vb: i32) {{
  vr = i64.const {REC_AT}
  i32.store vr vm offset=24
  i32.store vr vb offset=28
  vd0 = call 1 ()
  vc = call.cap 6 17 (i64) -> (i32) vi (vr)
  vw = call.cap 6 18 (i32) -> (i64) vi (vc)
  vz = i64.const 0
  vok = i64.eq vw vz
  br_if vok 1(vi, vc, vd0) 2(vw, vd0)
}}
block 1 (vi1: i32, vc1: i32, vd01: i64) {{
  vv = call.cap 6 1 (i32) -> (i64) vi1 (vc1)
  vd1 = call 1 ()
  vcan = i64.xor vd01 vd1
  vk0 = i64.const 0
  return vk0, vv, vcan
}}
block 2 (vt: i64, vd02: i64) {{
  vd2 = call 1 ()
  vcan2 = i64.xor vd02 vd2
  vk1 = i64.const 1
  return vk1, vt, vcan2
  }}
}}
{digest}",
        digest = digest_func(1 << ROOT_LOG2),
    );
    let m = temen_text::parse_module(&src).expect("the probe root parses");
    temen_verify::verify_module(&m).expect("the probe root verifies");
    m
}

/// The powerbox the root runs against, granted in one fixed order so every engine's host holds the
/// same handles: an `Instantiator` over the root's window, `child` as a `Module`, and an unbounded
/// `Budget`. Returns the host and the root's arguments.
pub fn powerbox(child: &Module) -> (Host, [i32; 3]) {
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, 1 << ROOT_LOG2);
    let module = host.grant_module(child);
    let budget = host.grant_budget(-1, -1, -1);
    (host, [inst, module, budget])
}

/// What the root reports: how its child ended (`Ok(value)`, or `Err(trap code)`) and the canary.
#[derive(Debug, PartialEq, Eq)]
pub struct Report {
    pub child: Result<i64, i64>,
    pub canary: i64,
}

impl Report {
    /// Read the root's three results.
    pub fn of(values: &[i64]) -> Report {
        let [trapped, value, canary] = values else {
            panic!("the probe root returns three values, got {values:?}");
        };
        Report {
            child: if *trapped == 0 {
                Ok(*value)
            } else {
                Err(*value)
            },
            canary: *canary,
        }
    }

    /// [`Report::of`] over interpreter values.
    pub fn of_values(values: &[Value]) -> Report {
        let raw: Vec<i64> = values
            .iter()
            .map(|v| match v {
                Value::I64(x) => *x,
                other => panic!("the probe root returns i64s, got {other:?}"),
            })
            .collect();
        Report::of(&raw)
    }
}

/// Run [`root`] over `child` on the tree-walk oracle, `init` seeding the root's window, with `fuel`:
/// the root's report (or its own trap) and its final window.
pub fn run_oracle(
    root: &Module,
    child: &Module,
    init: &[u8],
    fuel: u64,
) -> (Result<Report, Trap>, Vec<u8>) {
    let (mut host, args) = powerbox(child);
    let mut fuel = fuel;
    let (r, mem) = run_capture_reserved_with_host(
        root,
        0,
        &args.map(Value::I32),
        &mut fuel,
        init,
        0,
        &mut host,
    );
    (r.map(|v| Report::of_values(&v)), mem)
}

/// [`run_oracle`] on the Cranelift JIT, through temen-run's production spawn hooks. The JIT arms no
/// fuel here, so a child the oracle runs out of fuel may still finish on it.
pub fn run_jit(
    root: &Module,
    child: &Module,
    init: &[u8],
) -> Result<(JitOutcome, Vec<u8>), JitError> {
    let (mut host, args) = powerbox(child);
    let hp = &mut host as *mut Host;
    compile_and_run_capture_reserved_with_host_ex(
        root,
        0,
        &args.map(i64::from),
        init,
        0,
        temen_run::cap_thunk,
        hp as *mut core::ffi::c_void,
        Some(temen_run::module_resolver),
        Some(temen_run::production_grant_hooks(temen_run::CapCtx::Raw(
            hp,
        ))),
    )
}
