//! #2964 — `svc.fail(errno)` (self op 20): a serve handler answers the dispatch it serves as a
//! transport failure, and its caller gets INVARIANTS 8's reply, whatever the handler then returns.
//! The servicer lives. Every driver must agree.
//!
//! The parent serves `calc { add, fail, pair, wait }` through the live self-serve grant (#744) to
//! the child it spawns, serves until `svc.wait` answers 0, and joins it:
//!
//! - `fail(x)` calls `svc.fail(-12)` and returns `x`: the caller gets `-12`.
//! - `pair()` fails with `-12` and returns `(7, its own handle 0)`: the caller gets `(-12, 0)`, the
//!   zero untranslated (no handle of the servicer's reaches the caller).
//! - `add(x, y)` calls `svc.fail(5)`, a non-negative errno, which answers `-EINVAL` and fails
//!   nothing: the caller gets `x + y + (-22)`.
//! - `wait(x)` fails with `-7`, then parks (a timed futex wait no one notifies) and returns `x`:
//!   the failure rides the park, and the caller gets `-7`.
//!
//! The parent calls `svc.fail` itself, outside any handler, and gets `-EINVAL`.

#[path = "../../temen-interp/tests/support/rec.rs"]
mod rec;

use std::sync::Arc;

use temen_interp::Value;
use temen_ir::SpawnRec;
use temen_run::{instantiate, Backend, HostCap, Outcome, RunConfig};
use temen_text::parse_module;
use temen_verify::verify_module;

const IFACE: &str = "type 0 func (i64, i64) -> (i64)
type 1 func (i64) -> (i64)
type 2 func () -> (i64, cap)
type 3 interface { add: 0, fail: 1, pair: 2, wait: 1 }
";

/// The parent: spawn the child with `calc` as its export 0, serve until it is gone, and answer
/// `join + 10⁶ · (its own svc.fail)`.
fn parent() -> String {
    let child = SpawnRec {
        grants_ptr: 16640,
        grants_n: 1,
        ..SpawnRec::default()
    };
    format!(
        "memory 17
data 16384 \"vm\"
data 16400 \"budget\"
data 16448 \"child\"
data 16684 \"calc\"
{rec}{IFACE}export 0 interface \"calc\" 3 {{ add: 1, fail: 2, pair: 3, wait: 4 }}
func 0 () -> (i64) {{
block 0 () {{
  vp = i64.const 16384
  vl = i64.const 2
  vh = self.resolve vp vl
  vbp = i64.const 16400
  vbl = i64.const 6
  vb = self.resolve vbp vbl
  vba = i64.const {budget_at}
  i32.store vba vb
  vcp = i64.const 16448
  vcl = i64.const 5
  vchild = self.resolve vcp vcl
  vma = i64.const {module_at}
  i32.store vma vchild
  vgh = i32.const {tag}
{grant}  vrp = i64.const 17408
  vch = call.cap 6 17 (i64) -> (i32) vh (vrp)
  vz32 = i32.const 0
  verrno = i64.const -12
  vself = call.cap 4294967295 20 (i64) -> (i64) vz32 (verrno)
  br 1(vh, vch, vself)
}}
block 1 (vh1: i32, vch1: i32, vself1: i64) {{
  vz1 = i32.const 0
  vw = svc.wait vz1
  vzero = i64.const 0
  vdone = i64.eq vw vzero
  br_if vdone 2(vh1, vch1, vself1) 1(vh1, vch1, vself1)
}}
block 2 (vh2: i32, vch2: i32, vself2: i64) {{
  vj_st, vj = call.cap 6 1 (i32) -> (i64, i64) vh2 (vch2)
  vk6 = i64.const 1000000
  vsm = i64.mul vself2 vk6
  vr = i64.add vj vsm
  vz64 = i64.const 0
  vbad = i64.ne vj_st vz64
  vfail = i64.const -1
  vout = select vbad vfail vr
  return vout
}}
}}
func 1 (i64, i64) -> (i64) {{
block 0 (va: i64, vb: i64) {{
  vz = i32.const 0
  vfive = i64.const 5
  ve = call.cap 4294967295 20 (i64) -> (i64) vz (vfive)
  s = i64.add va vb
  r = i64.add s ve
  return r
  }}
}}
func 2 (i64) -> (i64) {{
block 0 (vx: i64) {{
  vz = i32.const 0
  vno = i64.const -12
  ve = call.cap 4294967295 20 (i64) -> (i64) vz (vno)
  return vx
  }}
}}
func 3 () -> (i64, cap) {{
block 0 () {{
  vz = i32.const 0
  vno = i64.const -12
  ve = call.cap 4294967295 20 (i64) -> (i64) vz (vno)
  vseven = i64.const 7
  vh = i32.const 0
  return vseven, vh
  }}
}}
func 4 (i64) -> (i64) {{
block 0 (vx: i64) {{
  vz = i32.const 0
  vno = i64.const -7
  ve = call.cap 4294967295 20 (i64) -> (i64) vz (vno)
  vgate = i64.const 16900
  vms = i64.const 1000000
  vwk = i32.atomic.wait vgate vz vms
  return vx
  }}
}}
",
        rec = rec::segment(17408, &child),
        grant = rec::grant("ge", 16640, 16684, 4, "vgh"),
        tag = temen_interp::GRANT_SERVE_LIVE_TAG as i32,
        budget_at = 17408 + SpawnRec::BUDGET_AT,
        module_at = 17408 + SpawnRec::MODULE_AT,
    )
}

/// The child: `fail(3)`, `pair()`, `add(1, 2)` and `wait(9)`, each in base 100 of the last.
fn child() -> String {
    format!(
        "memory 17
{IFACE}import 0 interface \"calc\" 3
func 0 (i64) -> (i64) {{
block 0 (v0: i64) {{
  v3 = i64.const 3
  vf = call.import 0.fail (v3)
  vp0, vp1 = call.import 0.pair ()
  vp164 = i64.extend_i32_s vp1
  v1 = i64.const 1
  v2 = i64.const 2
  vadd = call.import 0.add (v1, v2)
  v9 = i64.const 9
  vw = call.import 0.wait (v9)
  vk = i64.const 100
  t1 = i64.mul vf vk
  t2 = i64.add t1 vp0
  t3 = i64.mul t2 vk
  t4 = i64.add t3 vp164
  t5 = i64.mul t4 vk
  t6 = i64.add t5 vadd
  t7 = i64.mul t6 vk
  t8 = i64.add t7 vw
  return t8
  }}
}}
"
    )
}

fn module(src: &str) -> temen_ir::Module {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    m
}

fn run(backend: Option<Backend>) -> Result<i64, String> {
    let c = Arc::new(module(&child()));
    let inst = instantiate(module(&parent())).expect("instantiate");
    let caps = [
        ("vm", HostCap::custom(6, 0, |h, _| h.grant_instantiator())),
        ("budget", HostCap::detached_budget(1 << 20)),
        (
            "child",
            HostCap::custom(temen_ir::cap_id::MODULE, 0, move |h, _| h.grant_module(&c)),
        ),
    ];
    let cfg = RunConfig::default();
    let r = match backend {
        Some(b) => inst.run_with_caps(b, &cfg, &caps),
        None => inst.run_with_caps_parallel(&cfg, &caps),
    }
    .map_err(|e| e.to_string())?;
    match r.outcome {
        Outcome::Returned(v) => match v[..] {
            [Value::I64(x)] => Ok(x),
            _ => Err(format!("result {v:?}")),
        },
        o => Err(format!("outcome {o:?}")),
    }
}

#[test]
fn a_handler_answers_its_dispatch_as_a_transport_failure() {
    // fail → -12, pair → (-12, 0), add → 1 + 2 - 22, wait → -7; the parent's own svc.fail → -22.
    let child: i64 = (((-12 * 100 - 12) * 100) * 100 - 19) * 100 - 7;
    let expected = child - 22 * 1_000_000;
    for b in [
        Some(Backend::TreeWalk),
        Some(Backend::Bytecode),
        Some(Backend::Jit),
    ] {
        assert_eq!(run(b), Ok(expected), "{b:?}");
    }
    // The parallel driver pins the bytecode engine, which declines a server whose handler parks.
    let r = run(None);
    assert!(
        r == Ok(expected)
            || matches!(&r, Err(e) if e.contains("outside the parallel engine's subset")),
        "the parallel driver agrees or declines: {r:?}"
    );
}
