//! §3.6 **behavioral parity** — a serving domain behaves identically on all three backends,
//! each of which runs it itself: the reference interpreter (the oracle), the cooperative bytecode
//! driver and the Cranelift JIT, which serve the children a domain spawns (#744). This pins the
//! end-to-end §3.6 story — spawn a serving child, mint a live-callee offer
//! (`Instantiator.child_offer`), call through it (park), serve (`svc.wait`), reply-wake, join —
//! producing the SAME observable on TreeWalk, Bytecode, and Jit.

use temen_interp::{Host, MemLayout, Value};
use temen_ir::Module;
use temen_jit::JitOutcome;
use temen_run::{instantiate_with_imports, Backend, HostCap, Imports, Outcome, RunConfig};
use temen_text::parse_module;

#[path = "../../temen-interp/tests/support/drivers.rs"]
mod drivers;

use drivers::{agree_on, Driver, Ran};

/// `_start`: resolve the granted `"vm"` Instantiator, `"budget"` and `"child"` by name, spawn the
/// serving child (the image of func 1) detached through a v1 record, mint
/// `child_offer(child, export 0)`, call `add(40, 2)` through the live cap (parking until the child's
/// `svc.wait` serves it), join, and return the reply — 42. The image keeps the `"adder"` export, so
/// the child serves it.
const SERVING_PROGRAM: &str = "\
memory 17
data 16384 \"vm\"
data 16400 \"budget\"
data 16416 \"child\"
type 0 func (i64, i64) -> (i64)
type 1 interface { add: 0 }
export 0 interface \"adder\" 1 { add: 2 }
export 0 func \"child\" 1

func 0 () -> (i64) {
block 0 () {
  vp = i64.const 16384
  vl = i64.const 2
  vh = self.resolve vp vl
  vbp = i64.const 16400
  vbl = i64.const 6
  vbud = self.resolve vbp vbl
  vcp = i64.const 16416
  vcl = i64.const 5
  vchild = self.resolve vcp vcl
  ; a v1 record at 17536 (above the #1094 NULL guard and the durable control words): entry 0 of
  ; the `child` image, its declared window, no pager or pre-mapped region, paid from `budget`
  q0a0 = i64.const 17536
  q0v0 = i64.const 1
  i64.store q0a0 q0v0
  q0v2 = i64.const -4294967296
  i64.store q0a0 q0v2 offset=16
  i32.store q0a0 vchild offset=24
  i32.store q0a0 vbud offset=28
  q0v9 = i64.const 4294967295
  i64.store q0a0 q0v9 offset=72
  v5 = call.cap 6 17 (i64) -> (i32) vh (q0a0)
  v6 = i64.const 0
  v7 = call.cap 6 14 (i32, i64) -> (i32) vh (v5, v6)
  va = i64.const 40
  vb = i64.const 2
  vr = call.cap 268435456 0 (i64, i64) -> (i64) v7 (va, vb)
  vj_st, vj = call.cap 6 1 (i32) -> (i64, i64) vh (v5)
  return vr
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
    assert!(
        temen_interp::bytecode::admits(&m),
        "the cooperative bytecode driver serves it, not the tree-walker"
    );
    let image = std::sync::Arc::new(temen_ir::child_image_at(&m, 1).expect("child image"));
    let inst = instantiate_with_imports(m, Imports::new()).expect("instantiate");
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        let image = std::sync::Arc::clone(&image);
        let r = inst
            .run_with_caps(
                backend,
                &RunConfig::default(),
                &[
                    ("vm", HostCap::custom(6, 0, |h, _| h.grant_instantiator())),
                    ("budget", HostCap::detached_budget(1 << 20)),
                    (
                        "child",
                        HostCap::custom(temen_ir::cap_id::MODULE, 0, move |h, _| {
                            h.grant_module(&image)
                        }),
                    ),
                ],
            )
            .unwrap_or_else(|e| panic!("{backend:?}: {e}"));
        assert_eq!(
            r.outcome,
            Outcome::Returned(vec![temen_interp::Value::I64(42)]),
            "{backend:?}: spawn → child_offer → park → svc.wait-serve → reply → join → 42"
        );
    }
}

/// #2166 — a root that serves and also spawns a thread runs over the locked powerbox on the JIT, and
/// is served there by the same loop as any other domain. It used to reach the child serve loop,
/// which only knew a child's registration, and its `svc.poll` answered `-EINVAL` where the other
/// backends serve. The `Instantiator` call (never reached) keeps the module off the oracle fold,
/// and the thread is never spawned: only the module's shape matters.
const THREADED_SERVING_ROOT: &str = "\
memory 16
type 0 func (i64) -> (i64)
type 1 interface { bump: 0 }
export 0 interface \"counter\" 1 { bump: 1 }
import 0 \"exit\" (i32) -> ()

func 0 () -> () {
block 0 () {
  vz = i32.const 0
  vn = call.cap 4294967295 9 () -> (i64) vz ()
  vc = i32.wrap_i64 vn
  call.import 0 (vc)
  unreachable
  }
}

func 1 (i64) -> (i64) {
block 0 (vx: i64) {
  return vx
  }
}

func 2 () -> (i64) {
block 0 () {
  vsp = i64.const 32768
  va = i64.const 0
  vh = thread.spawn 3 vsp va
  vz = i64.const 0
  return vz
  }
}

func 3 (i64, i64) -> (i64) {
block 0 (vsp: i64, va: i64) {
  vz = i64.const 0
  return vz
  }
}

func 4 (i32, i32) -> (i64) {
block 0 (vh: i32, vc: i32) {
  vj_st, vj = call.cap 6 1 (i32) -> (i64, i64) vh (vc)
  return vj
  }
}
";

#[test]
fn a_serving_root_with_a_thread_serves_on_all_three_backends() {
    let m = parse_module(THREADED_SERVING_ROOT).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let registry = Imports::new().provide("exit", HostCap::exit());
    let inst = instantiate_with_imports(m, registry).expect("instantiate");
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        let r = inst
            .run_with_caps(backend, &RunConfig::default(), &[])
            .unwrap_or_else(|e| panic!("{backend:?}: {e}"));
        assert_eq!(
            r.outcome,
            Outcome::Exited(0),
            "{backend:?}: an empty poll serves nothing and answers 0"
        );
    }
}

/// #2247 — a parent whose `"adder"` handler is its func 1 spawns a child of **another module**, the
/// `"kid"` grant ([`SEPARATE_CHILD`]), mints `child_offer(child, export 0)`, calls `add(40, 2)`
/// through it, joins, and returns the reply.
const SEPARATE_PARENT: &str = "\
memory 17
data 16384 \"vm\"
data 16400 \"budget\"
data 16416 \"kid\"
type 0 func (i64, i64) -> (i64)
type 1 interface { add: 0 }
export 0 interface \"adder\" 1 { add: 1 }

func 0 () -> (i64) {
block 0 () {
  vp = i64.const 16384
  vl = i64.const 2
  vh = self.resolve vp vl
  vbp = i64.const 16400
  vbl = i64.const 6
  vbud = self.resolve vbp vbl
  vkp = i64.const 16416
  vkl = i64.const 3
  vkid = self.resolve vkp vkl
  q0a0 = i64.const 17536
  q0v0 = i64.const 1
  i64.store q0a0 q0v0
  q0v2 = i64.const -4294967296
  i64.store q0a0 q0v2 offset=16
  i32.store q0a0 vkid offset=24
  i32.store q0a0 vbud offset=28
  q0v9 = i64.const 4294967295
  i64.store q0a0 q0v9 offset=72
  v5 = call.cap 6 17 (i64) -> (i32) vh (q0a0)
  v6 = i64.const 0
  v7 = call.cap 6 14 (i32, i64) -> (i32) vh (v5, v6)
  va = i64.const 40
  vb = i64.const 2
  vr = call.cap 268435456 0 (i64, i64) -> (i64) v7 (va, vb)
  vj_st, vj = call.cap 6 1 (i32) -> (i64, i64) vh (v5)
  return vr
  }
}

func 1 (i64, i64) -> (i64) {
block 0 (va: i64, vb: i64) {
  vs = i64.add va vb
  return vs
  }
}
";

/// The `"kid"` module: its func 0 serves one call; its `"adder"` handler is its func 2, which adds
/// 1000. Its func 1, at the parent's handler index, adds 2000, so a child served through the
/// parent's handler indices answers 2042 instead of 1042.
const SEPARATE_CHILD: &str = "\
memory 16
type 0 func (i64, i64) -> (i64)
type 1 interface { add: 0 }
export 0 interface \"adder\" 1 { add: 2 }

func 0 (i64) -> (i64) {
block 0 (v0: i64) {
  vz = i32.const 0
  vn = svc.wait vz
  return vn
  }
}

func 1 (i64, i64) -> (i64) {
block 0 (va: i64, vb: i64) {
  vs = i64.add va vb
  vk = i64.const 2000
  vt = i64.add vs vk
  return vt
  }
}

func 2 (i64, i64) -> (i64) {
block 0 (va: i64, vb: i64) {
  vs = i64.add va vb
  vk = i64.const 1000
  vt = i64.add vs vk
  return vt
  }
}
";

fn verified(src: &str) -> Module {
    let m = parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// [`SEPARATE_PARENT`]'s powerbox: the `"vm"` it spawns through, the `"budget"` that pays, and the
/// `"kid"` module.
fn separate_setup(kid: &Module) -> impl Fn() -> (Host, Vec<Value>) + '_ {
    move || {
        let mut h = Host::new();
        let vm = h.grant_instantiator();
        h.register_cap_name("vm", vm);
        let budget = h.grant_budget(-1, 1 << 20, -1);
        h.register_cap_name("budget", budget);
        let k = h.grant_module(kid);
        h.register_cap_name("kid", k);
        (h, Vec::new())
    }
}

/// **#2247: a child serves its own module's offer, at its own handler index.** The Cranelift JIT
/// compiled every child with its parent's handler indices, so this child had no serve trampoline
/// for its func 2, and the call answered `-EBADF`. The tree-walker and the scheduling bytecode
/// drivers serve it; the debug driver declines `child_offer` and the resumable `Vcpu` has no child
/// to offer over (#1732), so neither is asked.
#[test]
fn a_child_of_another_module_serves_its_own_offer_on_every_engine() {
    let parent = verified(SEPARATE_PARENT);
    let kid = verified(SEPARATE_CHILD);
    let want = Ran {
        result: Ok(vec![Value::I64(1042)]),
        stdout: Vec::new(),
        stderr: Vec::new(),
    };
    agree_on(
        &[Driver::Oracle, Driver::Coop, Driver::Parallel],
        "a child of another module serves",
        &parent,
        &separate_setup(&kid),
        &want,
    );

    // The Cranelift JIT, on a thread of its own, so a call the child never serves fails the test
    // instead of hanging it.
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let (mut host, _) = separate_setup(&kid)();
        let window = MemLayout::image(vec![0u8; 128 << 10]);
        let (jo, _) = temen_run::jit_cap_run(&parent, 0, &[], &window, 0, 0, &mut host, None)
            .expect("the JIT compiles it");
        let _ = tx.send(jo);
    });
    let jo = rx
        .recv_timeout(std::time::Duration::from_secs(60))
        .expect("the JIT run hung");
    assert_eq!(
        jo,
        JitOutcome::Returned(vec![1042]),
        "the Cranelift JIT: the kid's own handler answers add(40, 2)"
    );
}
