//! #1469 — a §14 child that uses §12 fibers runs on the **Cranelift JIT**, with a fiber registry of
//! its own, as on the tree-walk oracle. Carved (op 5) and detached (op 15) children run as executor
//! tasks, and each task carries its own fiber runtime: the child's `cont.*` handles number from 0
//! and never reach the parent's fibers. The parent parks a fiber of its own across the spawn, so a
//! registry shared across domains shows up as a different result (the programs are
//! `temen`'s `bytecode_spawn_fibers.rs`, the bytecode engine's half of the same guarantee).
#![cfg(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
))]

use core::ffi::c_void;
use temen_interp::{run_capture_reserved_with_host, Host, Value};
use temen_ir::Module;
use temen_jit::{compile_and_run_capture_reserved_with_host_ex, JitOutcome};

const PARENT_LOG2: u8 = 23;
const CHILD_LOG2: u8 = 16;

/// A generator fiber: resumed with `a`, suspends with `a + 1`; resumed with `b`, returns `b + 5`.
const FIBER: &str = "func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  v0 = i64.const 1
  v1 = i64.add varg v0
  v2 = suspend v1
  v3 = i64.const 5
  v4 = i64.add v2 v3
  return v4
  }
}
";

/// The child: its first fiber's handle × 100, plus that fiber's two results (2 + 8).
fn child_src() -> String {
    format!(
        "memory {CHILD_LOG2}
func (i64) -> (i64) {{
block 0 (vs: i64) {{
  v0 = ref.func 1
  v1 = i64.const 0
  h = cont.new v0 v1
  a = i64.const 1
  s1, r1 = cont.resume h a
  b = i64.const 3
  s2, r2 = cont.resume h b
  c = i64.const 100
  hh = i64.mul h c
  t = i64.add r1 r2
  out = i64.add hh t
  return out
  }}
}}
{FIBER}"
    )
}

/// The parent (args `(instantiator, child module, budget)`): parks a fiber at its suspend, spawns and
/// joins the child, resumes the fiber to its return. Result: child status × 1000 + 11 + 25.
fn parent_src(detached: bool) -> String {
    let spawn = if detached {
        "ch = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) vinst (vb, me, gz, gz, gz, sl, gz)"
    } else {
        "ch = call.cap 6 5 (i64, i64, i64, i64, i64) -> (i32) vinst (me, gz, off, sl, gz)"
    };
    format!(
        "memory {PARENT_LOG2}
func (i32, i32, i32) -> (i64) {{
block 0 (vinst: i32, vmod: i32, vbud: i32) {{
  f0 = ref.func 1
  z = i64.const 0
  pf = cont.new f0 z
  a = i64.const 10
  s1, r1 = cont.resume pf a
  me = i64.extend_i32_s vmod
  vb = i64.extend_i32_s vbud
  gz = i64.const 0
  off = i64.const 4194304
  sl = i64.const {CHILD_LOG2}
  {spawn}
  vr = call.cap 6 1 (i32) -> (i64) vinst (ch)
  b = i64.const 20
  s2, r2 = cont.resume pf b
  k = i64.const 1000
  x = i64.mul vr k
  y = i64.add r1 r2
  out = i64.add x y
  return out
  }}
}}
{FIBER}"
    )
}

fn parse(src: &str) -> Module {
    let m = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

fn powerbox(child: &Module) -> (Host, [i32; 3]) {
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, 1 << PARENT_LOG2);
    let module = host.grant_module(child);
    let budget = host.grant_budget(-1, -1, -1);
    (host, [inst, module, budget])
}

/// The tree-walk oracle's result, or its trap's name.
fn oracle(parent: &Module, child: &Module) -> Result<i64, String> {
    let (mut host, h) = powerbox(child);
    let mut fuel = 10_000_000u64;
    let init = vec![0u8; 1 << PARENT_LOG2];
    let args = h.map(Value::I32);
    match run_capture_reserved_with_host(parent, 0, &args, &mut fuel, &init, 0, &mut host).0 {
        Ok(v) => match v.as_slice() {
            [Value::I64(x)] => Ok(*x),
            other => panic!("unexpected oracle result {other:?}"),
        },
        Err(t) => Err(format!("{t:?}")),
    }
}

/// The JIT's result, or its trap's name.
fn jit(parent: &Module, child: &Module) -> Result<i64, String> {
    let (mut host, h) = powerbox(child);
    let args = h.map(i64::from);
    let hp = &mut host as *mut Host;
    let (jo, _) = compile_and_run_capture_reserved_with_host_ex(
        parent,
        0,
        &args,
        &[],
        PARENT_LOG2,
        temen_run::cap_thunk,
        hp as *mut c_void,
        Some(temen_run::module_resolver),
        Some(temen_run::production_grant_hooks(temen_run::CapCtx::Raw(
            hp,
        ))),
    )
    .expect("the JIT compiles a parent that spawns and uses fibers");
    match jo {
        JitOutcome::Returned(v) => Ok(v[0]),
        JitOutcome::Trapped(t) => Err(format!("{t:?}")),
        o => panic!("jit ended abnormally: {o:?}"),
    }
}

fn parity(parent: &str, child: &str) -> Result<i64, String> {
    let (p, c) = (parse(parent), parse(child));
    let want = oracle(&p, &c);
    let before = temen_jit::child_compiles();
    assert_eq!(jit(&p, &c), want, "the JIT matches the oracle");
    assert!(
        temen_jit::child_compiles() > before,
        "the child was JIT-compiled"
    );
    want
}

#[test]
fn a_carved_child_has_its_own_fiber_registry_on_the_jit() {
    assert_eq!(
        parity(&parent_src(false), &child_src()),
        Ok(10 * 1000 + 11 + 25),
        "the child's first fiber is its handle 0"
    );
}

#[test]
fn a_detached_child_has_its_own_fiber_registry_on_the_jit() {
    assert_eq!(
        parity(&parent_src(true), &child_src()),
        Ok(10 * 1000 + 11 + 25),
        "the child's first fiber is its handle 0"
    );
}

/// A parent with no fibers of its own: runs `prelude`, spawns the child, returns its join status.
fn spawner_src(detached: bool, prelude: &str) -> String {
    let spawn = if detached {
        "ch = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) vinst (vb, me, gz, gz, gz, sl, gz)"
    } else {
        "ch = call.cap 6 5 (i64, i64, i64, i64, i64) -> (i32) vinst (me, gz, off, sl, gz)"
    };
    format!(
        "memory {PARENT_LOG2}
func (i32, i32, i32) -> (i64) {{
block 0 (vinst: i32, vmod: i32, vbud: i32) {{
  {prelude}
  me = i64.extend_i32_s vmod
  vb = i64.extend_i32_s vbud
  gz = i64.const 0
  off = i64.const 4194304
  sl = i64.const {CHILD_LOG2}
  {spawn}
  vr = call.cap 6 1 (i32) -> (i64) vinst (ch)
  return vr
  }}
}}"
    )
}

/// A child root that `suspend`s is the root computation suspending: a `FiberFault`, never a yield to
/// the executor worker the task runs on.
#[test]
fn a_child_root_that_suspends_faults_as_on_the_oracle() {
    let child = format!(
        "memory {CHILD_LOG2}
func (i64) -> (i64) {{
block 0 (vs: i64) {{
  a = i64.const 7
  r = suspend a
  return r
  }}
}}"
    );
    for detached in [false, true] {
        let got = parity(&spawner_src(detached, ""), &child);
        assert!(
            got != Ok(7),
            "detached {detached}: the suspend did not fault"
        );
    }
}

/// The child's fiber waits on a window word with a 1 ms timeout. Its `cont.resume` reports
/// `FIBER_PARKED` (3) until the wait times out, so the root re-polls — or, with `block`, the root's
/// one `cont.resume.block` parks the whole task until the fiber finishes. Either way the fiber
/// returns the wait status (2 = timed out) + 7.
fn waiting_child(block: bool) -> String {
    let drive = if block {
        "s, r = cont.resume.block hh a
  br 2(r)"
    } else {
        "s, r = cont.resume hh a
  parked = i32.const 3
  again = i32.eq s parked
  br_if again 1(hh) 2(r)"
    };
    format!(
        "memory {CHILD_LOG2}
func (i64) -> (i64) {{
block 0 (vs: i64) {{
  f = ref.func 1
  z = i64.const 0
  h = cont.new f z
  br 1(h)
  }}
block 1 (hh: i64) {{
  a = i64.const 0
  {drive}
  }}
block 2 (res: i64) {{
  return res
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (vsp: i64, varg: i64) {{
  addr = i64.const 32768
  exp = i32.const 0
  t = i64.const 1000000
  w = i32.atomic.wait addr exp t
  w64 = i64.extend_i32_u w
  seven = i64.const 7
  out = i64.add w64 seven
  return out
  }}
}}"
    )
}

#[test]
fn a_childs_fiber_parks_on_a_timed_wait() {
    for detached in [false, true] {
        for block in [false, true] {
            assert_eq!(
                parity(&spawner_src(detached, ""), &waiting_child(block)),
                Ok(9),
                "detached {detached}, block {block}"
            );
        }
    }
}

/// `gc.roots` from a child's fiber reaches the child **root's** frames, which live on the task's
/// stack: the root holds `0x7ab8` (loaded from the carve, where the parent stored it, so it cannot
/// be rematerialized) live across the resume, and the fiber holds `heap_lo` (`0x7ab0`). The child
/// returns `count << 40 | buf[0] << 20 | buf[1]`.
#[test]
fn gc_roots_in_a_childs_fiber_scans_the_child_root() {
    let prelude = "pp = i64.const 4214792
  pv = i64.const 31416
  i64.store pp pv";
    let child = format!(
        "memory {CHILD_LOG2}
func (i64) -> (i64) {{
block 0 (vs: i64) {{
  pa = i64.const 20488
  vp = i64.load pa
  f = ref.func 1
  sz = i64.const 4096
  h = cont.new f sz
  z = i64.const 0
  s, r = cont.resume h z
  keep = i64.const 20480
  i64.store keep vp
  b0a = i64.const 16384
  b0 = i64.load b0a
  b1a = i64.const 16392
  b1 = i64.load b1a
  s40 = i64.const 40
  s20 = i64.const 20
  x = i64.shl r s40
  y = i64.shl b0 s20
  xy = i64.or x y
  out = i64.or xy b1
  return out
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (vsp: i64, varg: i64) {{
  lo = i64.const 31408
  hi = i64.const 31424
  buf = i64.const 16384
  cap = i64.const 4
  mask = i64.const -1
  n = gc.roots lo hi mask buf cap
  return n
  }}
}}"
    );
    assert_eq!(
        parity(&spawner_src(false, prelude), &child),
        Ok(2 << 40 | 0x7ab0 << 20 | 0x7ab8)
    );
}
