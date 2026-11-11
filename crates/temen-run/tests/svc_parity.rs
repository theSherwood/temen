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
#[path = "../../temen-interp/tests/support/rec.rs"]
mod rec;

use drivers::{agree_on, Driver, Ran};

/// `_start`: resolve the granted `"vm"` Instantiator, `"budget"` and `"child"` by name, spawn the
/// serving child (the image of func 1) detached through a spawn record, mint
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
  ; a spawn record at 17536 (above the #1094 NULL guard and the durable control words): the
  ; `child` image, its declared window, no pager or pre-mapped region, paid from `budget`
  q0a0 = i64.const 17536
  q0v0 = i64.const -4294967296
  i64.store q0a0 q0v0
  i32.store q0a0 vchild offset=8
  i32.store q0a0 vbud offset=12
  q0v6 = i64.const 4294967295
  i64.store q0a0 q0v6 offset=48
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

/// #2553 — the run-mode matrix's fork topology (`clone_caller.rs`'s twin), its powerbox resolved by
/// name. The root spawns a server S (func 1) and a caller C (func 4), handing C a live offer into S.
/// C's call reaches S's `fork` handler (func 2), whose `clone_caller` forks C the way `fork()` does:
/// the original resumes with the twin's pid and the twin with 0. The twin writes 200 and returns it.
/// The original reaps the twin by its pid through S's `wait` handler (func 3, `reap`) and returns
/// the status, 200, which the root's join of C returns. (The matrix's probe reaps pid 3, the twin's
/// task id in its runners; a run through `Instance` numbers its tasks differently.) C retries a
/// `fork` or `wait` that answers `-EAGAIN`, the serve/park race `clone_caller.rs` documents (I53),
/// and returns any other error, so a backend that can't fork fails the test rather than spinning.
const FORKING_PROGRAM: &str = r#"
memory 18
type 0 func (i64) -> (i64)
type 1 interface { fork: 0, wait: 0 }
export 0 interface "svc" 1 { fork: 2, wait: 3 }
data 16448 "vm"
data 16464 "budget"
data 16480 "server"
data 16496 "caller"
data 16512 "out"
data 16684 "svc"
data 16694 "o"
func () -> (i64) {
block 0 () {
  vpv = i64.const 16448
  vlv = i64.const 2
  v0 = self.resolve vpv vlv
  vpb = i64.const 16464
  vlb = i64.const 6
  vbud = self.resolve vpb vlb
  vps = i64.const 16480
  vls = i64.const 6
  vsm = self.resolve vps vls
  vpc = i64.const 16496
  vlc = i64.const 6
  vcm = self.resolve vpc vlc
  vpo = i64.const 16512
  vlo = i64.const 3
  vout = self.resolve vpo vlo
  qsm = i64.const 17544
  i32.store qsm vsm
  q1b = i64.const 17548
  i32.store q1b vbud
  q1p = i64.const 17536
  vs = call.cap 6 17 (i64) -> (i32) v0 (q1p)
  vz0 = i64.const 0
  vcap = call.cap 6 14 (i32, i64) -> (i32) v0 (vs, vz0)
  va0 = i64.const 16640
  vnp = i32.const 16684
  i32.store va0 vnp
  va1 = i64.const 16644
  vnl = i32.const 3
  i32.store va1 vnl
  va2 = i64.const 16648
  i32.store va2 vcap
  va3 = i64.const 16656
  vnp2 = i32.const 16694
  i32.store va3 vnp2
  va4 = i64.const 16660
  vnl2 = i32.const 1
  i32.store va4 vnl2
  va5 = i64.const 16664
  i32.store va5 vout
  qcm = i64.const 17672
  i32.store qcm vcm
  q2b = i64.const 17676
  i32.store q2b vbud
  q2p = i64.const 17664
  vc = call.cap 6 17 (i64) -> (i32) v0 (q2p)
  vjc_st, vjc = call.cap 6 1 (i32) -> (i64, i64) v0 (vc)
  return vjc
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
  vrt = i64.const 0
  vt = call.cap 4294967295 11 (i64) -> (i64) vz (vrt)
  return vt
  }
}
func (i64) -> (i64) {
block 0 (vpid: i64) {
  vz = i32.const 0
  vt = call.cap 4294967295 12 (i64) -> (i64) vz (vpid)
  return vt
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vp0 = i64.const 16684
  vl3 = i64.const 3
  vhsvc = self.resolve vp0 vl3
  vp8 = i64.const 16694
  vl1 = i64.const 1
  vho = self.resolve vp8 vl1
  br 1(vhsvc, vho)
  }
block 1 (vhsvc: i32, vho: i32) {
  varg = i64.const 7
  vpid = call.cap 268435456 0 (i64) -> (i64) vhsvc (varg)
  vea1 = i64.const -11
  vagain1 = i64.eq vpid vea1
  br_if vagain1 1(vhsvc, vho) 2(vpid, vhsvc, vho)
  }
block 2 (vpid: i64, vhsvc: i32, vho: i32) {
  vz2 = i64.const 0
  vistwin = i64.eq vpid vz2
  br_if vistwin 5(vho) 3(vpid, vhsvc)
  }
block 3 (vpid: i64, vhsvc: i32) {
  vz3 = i64.const 0
  vfailed = i64.lt_s vpid vz3
  br_if vfailed 6(vpid) 4(vpid, vhsvc)
  }
block 4 (vpid: i64, vhsvc: i32) {
  vstatus = call.cap 268435456 1 (i64) -> (i64) vhsvc (vpid)
  vea4 = i64.const -11
  vagain4 = i64.eq vstatus vea4
  br_if vagain4 4(vpid, vhsvc) 6(vstatus)
  }
block 5 (vho: i32) {
  v200 = i64.const 200
  vp16 = i64.const 17408
  i64.store vp16 v200
  vlen = i64.const 8
  vw = call.cap 0 1 (i64, i64) -> (i64) vho (vp16, vlen)
  return v200
  }
block 6 (vr: i64) {
  return vr
  }
}
"#;

/// #2553 — a forking server answers as the oracle does on all three backends, run as an embedder runs
/// it. The Cranelift JIT does not fork, so it declines the module (`temen_jit::admits`), and the run
/// folds down the ladder to the bytecode engine, which forks. The JIT used to compile the fork ops
/// to its generic cap thunk, and C's fork answered `-EINVAL`, so the run returned -22.
#[test]
fn a_forking_server_answers_as_the_oracle_on_all_three_backends() {
    let caller = temen_ir::SpawnRec {
        grants_ptr: 16640,
        grants_n: 2,
        ..temen_ir::SpawnRec::default()
    };
    let src = format!(
        "{FORKING_PROGRAM}{}{}",
        rec::segment(17536, &temen_ir::SpawnRec::default()),
        rec::segment(17664, &caller)
    );
    let m = parse_module(&src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    assert!(
        !temen_jit::admits(&m),
        "the Cranelift JIT declines a module that forks"
    );
    let image = |f| std::sync::Arc::new(temen_ir::child_image_at(&m, f).expect("child image"));
    let (server, caller) = (image(1), image(4));
    let inst = instantiate_with_imports(m, Imports::new()).expect("instantiate");
    // Queued calls: under direct handoff the tree-walker runs S's handler inline, with no parked
    // caller to fork, so C's `fork` answers `-EAGAIN` for good (#2854).
    let config = RunConfig {
        handoff: false,
        ..RunConfig::default()
    };
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        let module = |image: &std::sync::Arc<Module>| {
            let image = std::sync::Arc::clone(image);
            HostCap::custom(temen_ir::cap_id::MODULE, 0, move |h, _| {
                h.grant_module(&image)
            })
        };
        let r = inst
            .run_with_caps(
                backend,
                &config,
                &[
                    ("vm", HostCap::custom(6, 0, |h, _| h.grant_instantiator())),
                    ("budget", HostCap::detached_budget(8 << 20)),
                    ("server", module(&server)),
                    ("caller", module(&caller)),
                    ("out", HostCap::stdout()),
                ],
            )
            .unwrap_or_else(|e| panic!("{backend:?}: {e}"));
        assert_eq!(
            r.outcome,
            Outcome::Returned(vec![Value::I64(200)]),
            "{backend:?}: fork → the twin returns 200 → the original reaps it → 200"
        );
        assert_eq!(
            r.stdout,
            200i64.to_le_bytes(),
            "{backend:?}: the twin wrote through its copy of the powerbox"
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
  q0v0 = i64.const -4294967296
  i64.store q0a0 q0v0
  i32.store q0a0 vkid offset=8
  i32.store q0a0 vbud offset=12
  q0v6 = i64.const 4294967295
  i64.store q0a0 q0v6 offset=48
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
/// for its func 2, and the call answered `-EBADF`. The tree-walker and the cooperative and parallel
/// bytecode drivers serve it; the debug driver declines `child_offer`, so it is not asked.
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
