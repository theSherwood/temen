//! §3.6 **behavioral parity** — a serving domain behaves identically on all three backends.
//! The serve loop is a single implementation on the reference interpreter (the oracle);
//! bytecode reaches it by declining the serving ops at compile and falling back, and the JIT
//! by the `module_serves` fold in temen-run. This pins the end-to-end §3.6 story — spawn a
//! serving child, mint a live-callee offer (`Instantiator.child_offer`), call through it
//! (park), serve (`svc.wait`), reply-wake, join — producing the SAME observable on
//! TreeWalk, Bytecode, and Jit.

use temen_run::{instantiate_with_imports, Backend, HostCap, Imports, Outcome, RunConfig};
use temen_text::parse_module;

/// `_start`: resolve the granted `"vm"` Instantiator and `"budget"` by name, spawn the serving child
/// (func 1) detached through a v1 record, mint `child_offer(child, export 0)`, call `add(40, 2)`
/// through the live cap (parking until the child's `svc.wait` serves it), join, and exit with the
/// reply — 42.
const SERVING_PROGRAM: &str = "\
memory 17
data 16384 \"vm\"
data 16400 \"budget\"
type 0 func (i64, i64) -> (i64)
type 1 interface { add: 0 }
export 0 interface \"adder\" 1 { add: 2 }
import 0 \"exit\" (i32) -> ()

func 0 () -> () {
block 0 () {
  vp = i64.const 16384
  vl = i64.const 2
  vh = self.resolve vp vl
  vbp = i64.const 16400
  vbl = i64.const 6
  vbud = self.resolve vbp vbl
  ; a v1 record at 17536 (above the #1094 NULL guard and the durable control words): entry 1, the
  ; running module (-1), its declared window, no pager or pre-mapped region, paid from `budget`
  q0a0 = i64.const 17536
  q0v0 = i64.const 4294967297
  i64.store q0a0 q0v0
  q0v2 = i64.const -4294967296
  i64.store q0a0 q0v2 offset=16
  q0m = i32.const -1
  i32.store q0a0 q0m offset=24
  i32.store q0a0 vbud offset=28
  q0v9 = i64.const 4294967295
  i64.store q0a0 q0v9 offset=72
  v5 = call.cap 6 17 (i64) -> (i32) vh (q0a0)
  v6 = i64.const 0
  v7 = call.cap 6 14 (i32, i64) -> (i32) vh (v5, v6)
  va = i64.const 40
  vb = i64.const 2
  vr = call.cap 268435456 0 (i64, i64) -> (i64) v7 (va, vb)
  vj = call.cap 6 1 (i32) -> (i64) vh (v5)
  vc = i32.wrap_i64 vr
  call.import 0 (vc)
  unreachable
  }
}

func 1 (i64) -> (i64) {
block 0 (v0: i64) {
  vz = i32.const 0
  vn = svc.wait vz
  return vn
  }
}

func 2 (i64, i64) -> (i64) {
block 0 (va: i64, vb: i64) {
  vs = i64.add va vb
  return vs
  }
}
";

#[test]
fn a_serving_domain_behaves_identically_on_all_three_backends() {
    let m = parse_module(SERVING_PROGRAM).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let registry = Imports::new().provide("exit", HostCap::exit());
    let inst = instantiate_with_imports(m, registry).expect("instantiate");
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        let r = inst
            .run_with_caps(
                backend,
                &RunConfig::default(),
                &[
                    (
                        "vm",
                        HostCap::custom(6, 0, |h, win| h.grant_instantiator(0, win)),
                    ),
                    ("budget", HostCap::detached_budget(1 << 20)),
                ],
            )
            .unwrap_or_else(|e| panic!("{backend:?}: {e}"));
        assert_eq!(
            r.outcome,
            Outcome::Exited(42),
            "{backend:?}: spawn → child_offer → park → svc.wait-serve → reply → join → 42"
        );
    }
}
