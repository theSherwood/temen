//! The run-mode matrix's **probes** (#2545): one small program per guest feature, the powerbox it
//! runs with, and the answer the oracle gives. `run_mode_conformance.rs` runs each on every runner
//! the manifest lists. Included there with `#[path = "support/run_probes.rs"] mod probes;`.
//!
//! The programs are the smallest each feature's own tests run on several engines, so a probe's
//! answer is one those tests already pin on the oracle.
//!
//! The embedder a runner plays ([`Embedder`], [`Stopper`], [`Outside`]) is here too, so
//! `axis_conformance.rs`, which runs the frontier's capability probes on the same runners, includes
//! this module for it.

// Each including binary uses a subset: the frontier's runs use only the embedder and its registrar.
#![allow(dead_code)]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use temen_durable::transform_module_assume_confined;
use temen_interp::{
    CapState, Host, HostProc, HostProcGrant, NamedCapGrant, OffloadOutcome, ProcHandler,
    SignalSource, StreamRole, Value,
};
use temen_ir::durable_abi::ShadowArena;
use temen_ir::{Module, SpawnRec};
use temen_parity::run_modes::Feature;

use crate::support::capability_probe;

/// One feature's probe.
#[derive(Clone, Copy)]
pub struct Probe {
    pub feature: Feature,
    pub src: fn() -> String,
    /// The powerbox and arguments, built fresh for each run from the run's module and the embedder
    /// that runs it.
    pub setup: fn(&Module, &Embedder) -> (Host, Vec<Value>),
    /// What the embedder does from outside the run, beside answering its calls.
    pub outside: Outside,
    /// The oracle's answer: what the entry returns.
    pub want: i64,
}

/// What a probe's embedder does from outside the run. Each runner does it the way its embedder
/// would.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Outside {
    /// Nothing.
    Nothing,
    /// It supplies these bytes on stdin after the run has read: a run it holds is fed when it waits
    /// on them, and any other run pulls them from a source that blocks until they arrive.
    Feed(&'static [u8]),
    /// It continues the process the probe stops ([`Embedder::stopper`]). A run it holds is
    /// continued when it waits on that. A run nobody holds faults when its embedder has not said
    /// it may act from outside, and waits for the continue when it has (#2529).
    Continue,
}

/// One run's embedder, as far as a probe reaches it.
#[derive(Clone, Default)]
pub struct Embedder {
    /// The job control it holds over the run's process.
    pub stopper: Arc<Stopper>,
    /// Whether it freezes and thaws the run. It then grants each capability in the form a thaw
    /// can grant again: a module instrumented and attested freezable ([`Self::image`],
    /// [`Self::grant_module`]), a host procedure under its name ([`named_proc`]).
    pub durable: bool,
}

impl Embedder {
    /// An embedder that freezes and thaws its run.
    pub fn durable() -> Embedder {
        Embedder {
            durable: true,
            ..Embedder::default()
        }
    }

    /// `m` as this embedder runs it: [`instrumented`] when it freezes and thaws the run.
    pub fn image(&self, m: Module) -> Module {
        if self.durable {
            instrumented(&m).expect("every probe's module instruments")
        } else {
            m
        }
    }

    /// Grant `m`, an [`image`](Self::image), for a child to run: attested freezable when this
    /// embedder freezes and thaws the run, since a durable domain spawns only that (DURABILITY.md
    /// §4).
    pub fn grant_module(&self, host: &mut Host, m: &Module) -> i32 {
        if self.durable {
            host.grant_durable_module(m)
        } else {
            host.grant_module(m)
        }
    }
}

/// The shadow arena a durable run's modules declare (INVARIANTS #16): four context regions, in a
/// part of the window no probe uses.
pub const ARENA: ShadowArena = ShadowArena::new(32768, 49152);

/// `m` made durable: its window, at least 64 KiB, declares [`ARENA`], and the `temen-durable`
/// transform instruments it on the guarantee that the guest leaves the arena alone, which every
/// probe does.
pub fn instrumented(m: &Module) -> Result<Module, String> {
    let mut m = m.clone();
    let size_log2 = m.memory.map_or(16, |w| w.size_log2.max(16));
    m.memory = Some(temen_ir::Memory {
        size_log2,
        shadow: Some(ARENA),
    });
    let inst = transform_module_assume_confined(&m)
        .map_err(|e| format!("the transform refuses the module: {e:?}"))?;
    temen_verify::verify_module(&inst).map_err(|e| format!("{e:?}"))?;
    Ok(inst)
}

/// The host procedures the probes' embedder serves, by the name it grants each under: what a
/// setup grants ([`grant_named`]), and what its registrar grants again when it thaws the run
/// (#1455). A registrar re-grants a synchronous procedure, so `"plus100"`, which two setups grant
/// as an offloadable one, comes back synchronous: a durable run waits on a pool call inline anyway
/// (#2504). The frontier matrix's capability probes run on this embedder too, so it also serves
/// their `HostProc` row's procedure.
pub fn named_proc(name: &str, e: &Embedder) -> Option<HostProc> {
    let s = Arc::clone(&e.stopper);
    Some(match name {
        capability_probe::ZERO => capability_probe::zero(),
        "plus100" => Box::new(|_op, args, _mem, _minter| Ok(vec![plus_100(args)])),
        "stop" => Box::new(move |_op, _args, _mem, _minter| {
            s.set(true);
            Ok(vec![0])
        }),
        "continued" => Box::new(move |_op, _args, _mem, _minter| {
            Ok(vec![i64::from(s.continued.load(Ordering::SeqCst))])
        }),
        _ => return None,
    })
}

/// [`named_proc`] as a thaw's registrar re-grants it: stateless, with no fork factory.
pub fn regrant(name: &str, e: &Embedder) -> Option<NamedCapGrant> {
    named_proc(name, e).map(|handler| NamedCapGrant {
        grant: HostProcGrant::new(ProcHandler::Sync(handler), CapState::Stateless),
        process: None,
    })
}

/// Grant [`named_proc`] `name` under its name.
fn grant_named(host: &mut Host, e: &Embedder, name: &str) -> i32 {
    let proc = named_proc(name, e).expect("a probe grants only what its embedder serves");
    let h = host.grant_host_proc(HostProcGrant::new(
        ProcHandler::Sync(proc),
        CapState::Stateless,
    ));
    host.register_cap_name(name, h);
    h
}

/// `"plus100"`'s answer: its argument plus 100.
fn plus_100(args: &[i64]) -> i64 {
    args.first().copied().unwrap_or(0) + 100
}

/// An embedder's job control with no personality behind it: the guest stops its own process
/// through a host call ([`stop_self`]), and the embedder continues it. Each change reaches the
/// engines through the doors they install, in the order the POSIX personality fires them: the
/// inline mirror (`set_stop_apply`), then the wake (`set_stop`). A mirror installed while the
/// process is stopped hears the stop at once, as the personality publishes it to a thawed run
/// (#1900); the state and the mirror change under one lock, as under the personality's. The
/// bytecode executors read [`SignalSource::stopped`].
#[derive(Default)]
pub struct Stopper {
    /// The inline mirror door. Its lock also orders the changes to `stopped`.
    apply: Mutex<Option<Door>>,
    wake: Mutex<Option<Door>>,
    stopped: AtomicBool,
    /// Whether the embedder has continued the process: what the probe's second call reports.
    continued: AtomicBool,
    /// Whether the process asked to stop, or the run is over, for [`Self::await_stop`].
    asked: (Mutex<bool>, Condvar),
}

type Door = Arc<dyn Fn(bool) + Send + Sync>;

impl Stopper {
    /// Stop the process (`true`) or continue it (`false`).
    pub fn set(&self, stopped: bool) {
        if !stopped {
            self.continued.store(true, Ordering::SeqCst);
        }
        {
            let apply = self.apply.lock().unwrap();
            self.stopped.store(stopped, Ordering::SeqCst);
            if let Some(apply) = apply.as_ref() {
                apply(stopped);
            }
        }
        if let Some(wake) = self.wake.lock().unwrap().clone() {
            wake(stopped);
        }
        if stopped {
            self.wake_waiter();
        }
    }

    pub fn is_stopped(&self) -> bool {
        self.stopped.load(Ordering::SeqCst)
    }

    /// Wait until the process has asked to stop, or [`Self::close`] says the run is over.
    pub fn await_stop(&self) {
        let (asked, cv) = &self.asked;
        let mut asked = asked.lock().unwrap();
        while !*asked {
            asked = cv.wait(asked).unwrap();
        }
    }

    /// The run is over: nothing waits for a stop any more.
    pub fn close(&self) {
        self.wake_waiter();
    }

    fn wake_waiter(&self) {
        *self.asked.0.lock().unwrap() = true;
        self.asked.1.notify_all();
    }
}

impl SignalSource for Stopper {
    fn take_deliverable(&self) -> Option<(i32, i32, u64)> {
        None
    }
    fn set_stop_apply(&self, apply: Door) {
        let mut door = self.apply.lock().unwrap();
        if self.is_stopped() {
            apply(true);
        }
        *door = Some(apply);
    }
    fn set_stop(&self, wake: Door) {
        *self.wake.lock().unwrap() = Some(wake);
    }
    fn stopped(&self) -> bool {
        self.is_stopped()
    }
}

pub fn module(src: &str) -> Module {
    let m = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

pub fn probes() -> Vec<Probe> {
    vec![
        Probe {
            feature: Feature::Threads,
            src: || THREADS.into(),
            setup: |_, _| bare(),
            outside: Outside::Nothing,
            want: 212,
        },
        Probe {
            feature: Feature::Fibers,
            src: || FIBERS.into(),
            setup: |_, _| bare(),
            outside: Outside::Nothing,
            want: 10,
        },
        Probe {
            feature: Feature::Children,
            src: children,
            setup: |_, e| child_powerbox(e),
            outside: Outside::Nothing,
            want: 42,
        },
        Probe {
            feature: Feature::Fork,
            src: fork,
            setup: |m, e| fork_powerbox(m, e),
            outside: Outside::Nothing,
            want: 100,
        },
        Probe {
            feature: Feature::Exec,
            src: || EXEC.into(),
            setup: |m, e| exec_powerbox(m, e),
            outside: Outside::Nothing,
            want: 42,
        },
        Probe {
            feature: Feature::HostAnswered,
            src: || TWO_CALLS.into(),
            setup: |_, _| host_answered(),
            outside: Outside::Nothing,
            want: 105_107,
        },
        Probe {
            feature: Feature::Offloaded,
            src: || TWO_CALLS.into(),
            setup: |_, _| offloaded(),
            outside: Outside::Nothing,
            want: 105_107,
        },
        Probe {
            feature: Feature::StdinLater,
            src: || READ_ONCE.into(),
            setup: |_, _| stdin(),
            outside: Outside::Feed(b"abc"),
            want: 3,
        },
        Probe {
            feature: Feature::Signals,
            src: || STOP_SELF.into(),
            setup: |_, e| stop_self(e),
            outside: Outside::Continue,
            want: 1001,
        },
        Probe {
            feature: Feature::GrowingWindow,
            src: || GROW.into(),
            setup: |_, _| memory(),
            outside: Outside::Nothing,
            want: 424_242,
        },
        Probe {
            feature: Feature::SharedRegion,
            src: || REGION.into(),
            setup: |_, _| region(),
            outside: Outside::Nothing,
            want: 41,
        },
        Probe {
            feature: Feature::GuestJit,
            src: guest_jit,
            setup: |m, e| jit_powerbox(m, e),
            outside: Outside::Nothing,
            want: 105_107,
        },
        Probe {
            feature: Feature::Serving,
            src: || SERVER.into(),
            setup: |m, _| serve_powerbox(m),
            outside: Outside::Nothing,
            want: 2042,
        },
    ]
}

// ---- threads and fibers ---------------------------------------------------------------------------

/// A powerbox with nothing granted.
fn bare() -> (Host, Vec<Value>) {
    (Host::new(), Vec::new())
}

/// The root spawns two threads, on 5 and 7, each returning `argument + 100`, and joins both.
const THREADS: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  vz = i64.const 0
  v5 = i64.const 5
  t0 = thread.spawn 1 vz v5
  v7 = i64.const 7
  t1 = thread.spawn 1 vz v7
  j0_st, j0 = thread.join t0
  j1_st, j1 = thread.join t1
  vs = i64.add j0 j1
  return vs
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  v100 = i64.const 100
  vr = i64.add varg v100
  return vr
  }
}
"#;

/// A generator fiber yields 1, 2 and 3, then returns 4, and the root sums what it delivers.
const FIBERS: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  v0 = ref.func 1
  v1 = i64.const 4096
  v2 = cont.new v0 v1
  v3 = i64.const 0
  br 1(v2, v3)
}
block 1 (v4: i64, v5: i64) {
  v6 = i64.const 0
  v7, v8 = cont.resume v4 v6
  v9 = i64.add v5 v8
  br_if v7 2(v9) 1(v4, v9)
}
block 2 (v10: i64) {
  return v10
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 1
  v3 = suspend v2
  v4 = i64.const 2
  v5 = suspend v4
  v6 = i64.const 3
  v7 = suspend v6
  v8 = i64.const 4
  return v8
  }
}
"#;

// ---- processes ------------------------------------------------------------------------------------

/// A separate child module whose entry returns 42.
const CHILD_42: &str = "memory 12
func (i64) -> (i64) {
block 0 (v0: i64) {
  vr = i64.const 42
  return vr
  }
}
";

/// Where the §14 probe lays its spawn record.
const REC_AT: u64 = 17536;

/// The root `(i32 inst, i32 module, i32 budget) -> i64` spawns `module` detached from the spawn
/// record at [`REC_AT`], paid from `budget`, and returns what the child's join gives: 42.
fn children() -> String {
    format!(
        "memory 17
func (i32, i32, i32) -> (i64) {{
block 0 (vinst: i32, vmod: i32, vbud: i32) {{
  vma = i64.const {m}
  i32.store vma vmod
  vba = i64.const {b}
  i32.store vba vbud
  vrp = i64.const {REC_AT}
  vch = call.cap 6 17 (i64) -> (i32) vinst (vrp)
  vst, vr = call.cap 6 1 (i32) -> (i64, i64) vinst (vch)
  return vr
  }}
}}
{rec}",
        m = REC_AT + SpawnRec::MODULE_AT,
        b = REC_AT + SpawnRec::BUDGET_AT,
        rec = crate::rec::segment(REC_AT, &SpawnRec::default()),
    )
}

/// The §14 probe's powerbox: the Instantiator, the child's module and a budget to pay for it.
fn child_powerbox(e: &Embedder) -> (Host, Vec<Value>) {
    let mut host = Host::new();
    let i = host.grant_instantiator();
    let c = e.grant_module(&mut host, &e.image(module(CHILD_42)));
    let b = host.grant_budget(5_000_000, -1, -1);
    (host, [i, c, b].map(Value::I32).to_vec())
}

/// Fork (`clone_caller.rs`'s `SRC_TWIN`): the root spawns a server S (func 1) and offers it to a
/// caller C (func 4). C's live call reaches S's handler (func 2), which forks its caller: `clone_caller`
/// answers the original 100 and the twin 200. Each writes its reply to `"o"`; the twin's ends, C
/// reaps it (re-forking on a lost twin) and returns 100, which the root's join of C returns.
fn fork() -> String {
    let caller = SpawnRec {
        grants_ptr: 16640,
        grants_n: 2,
        ..SpawnRec::default()
    };
    format!(
        "{FORK_TWIN}{}{}",
        crate::rec::segment(17536, &SpawnRec::default()),
        crate::rec::segment(17664, &caller)
    )
}

const FORK_TWIN: &str = r#"
memory 18
type 0 func (i64) -> (i64)
type 1 interface { fork: 0, wait: 0 }
export 0 interface "svc" 1 { fork: 2, wait: 3 }
data 16684 "svc"
data 16694 "o"
func (i32, i32, i32, i32, i32) -> (i64) {
block 0 (v0: i32, vout: i32, vbud: i32, vsm: i32, vcm: i32) {
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
  vro = i64.const 100
  vrt = i64.const 200
  vt = call.cap 4294967295 11 (i64, i64) -> (i64) vz (vro, vrt)
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
  vr = call.cap 268435456 0 (i64) -> (i64) vhsvc (varg)
  v200 = i64.const 200
  vistwin = i64.eq vr v200
  br_if vistwin 4(vr, vho) 2(vr, vhsvc, vho)
  }
block 2 (vr: i64, vhsvc: i32, vho: i32) {
  vpid3 = i64.const 3
  vstatus = call.cap 268435456 1 (i64) -> (i64) vhsvc (vpid3)
  veagain = i64.const -11
  viseagain = i64.eq vstatus veagain
  br_if viseagain 2(vr, vhsvc, vho) 3(vr, vstatus, vhsvc, vho)
  }
block 3 (vr: i64, vstatus: i64, vhsvc: i32, vho: i32) {
  vechild = i64.const -10
  visechild = i64.eq vstatus vechild
  br_if visechild 1(vhsvc, vho) 4(vr, vho)
  }
block 4 (vr: i64, vho: i32) {
  vp16 = i64.const 17408
  i64.store vp16 vr
  vlen = i64.const 8
  vw = call.cap 0 1 (i64, i64) -> (i64) vho (vp16, vlen)
  return vr
  }
}
"#;

/// The fork probe's powerbox: the Instantiator, `stdout`, a budget, and the server's and caller's
/// child images (funcs 1 and 4).
fn fork_powerbox(m: &Module, e: &Embedder) -> (Host, Vec<Value>) {
    let mut host = Host::new();
    let inst = host.grant_instantiator();
    let budget = host.grant_budget(-1, 8 << 20, -1);
    let _ = host.shared_stdout();
    let out = host.grant_stream(StreamRole::Out);
    let image = |host: &mut Host, f| {
        e.grant_module(host, &temen_ir::child_image_at(m, f).expect("child image"))
    };
    let (s, c) = (image(&mut host, 1), image(&mut host, 4));
    (host, [inst, out, budget, s, c].map(Value::I32).to_vec())
}

/// Exec (`execve.rs`'s `GUEST`): `main(inst, cmd, stream)` grants `"stdout"` and calls
/// `exec_module`, which replaces its image with `cmd`; the command writes and returns 42. The
/// trailing `return 99` runs only if the exec failed. Both programs keep their data above the NULL
/// guard, in a 64 KiB window a durable run's arena fits in.
const EXEC: &str = r#"
memory 16
data 16684 "stdout"
func (i32, i64, i32) -> (i64) {
block 0 (vinst: i32, vcmd: i64, vstream: i32) {
  va0 = i64.const 16640
  vnp = i32.const 16684
  i32.store va0 vnp
  va1 = i64.const 16644
  vlen = i32.const 6
  i32.store va1 vlen
  va2 = i64.const 16648
  i32.store va2 vstream
  vz = i32.const 0
  vgp = i64.const 16640
  vgn = i64.const 1
  ventry = i64.const 0
  vsl = i64.const 16
  vr = call.cap 4294967295 14 (i64, i64, i64, i64, i64) -> (i64) vz (vcmd, vgp, vgn, ventry, vsl)
  v99 = i64.const 99
  return v99
  }
}
"#;

/// The command `exec_module` runs: resolves `"stdout"`, writes `"EXEC"` to it and returns 42.
const EXEC_CMD: &str = r#"
memory 16
data 16484 "EXEC"
data 16584 "stdout"
func (i64) -> (i64) {
block 0 (vinst: i64) {
  vp = i64.const 16584
  vl = i64.const 6
  vh = self.resolve vp vl
  vwp = i64.const 16484
  vwl = i64.const 4
  vw = call.cap 0 1 (i64, i64) -> (i64) vh (vwp, vwl)
  v42 = i64.const 42
  return v42
  }
}
"#;

/// The exec probe's powerbox: the running module, the Instantiator, `stdout` and the command.
fn exec_powerbox(m: &Module, e: &Embedder) -> (Host, Vec<Value>) {
    let mut host = Host::new();
    host.set_self_module(&Arc::new(m.clone()));
    let inst = host.grant_instantiator();
    let stream = host.grant_stream(StreamRole::Out);
    let cmd = e.grant_module(&mut host, &e.image(module(EXEC_CMD)));
    (
        host,
        vec![
            Value::I32(inst),
            Value::I64(i64::from(cmd)),
            Value::I32(stream),
        ],
    )
}

// ---- host calls and input -------------------------------------------------------------------------

/// The root makes two calls to op 1 of the capability it is handed, on 5 and 7: `105 * 1000 + 107`.
const TWO_CALLS: &str = r#"memory 16
func (i32) -> (i64) {
block 0 (vh: i32) {
  vfive = i64.const 5
  vr0 = call.cap 13 1 (i64) -> (i64) vh (vfive)
  vseven = i64.const 7
  vr1 = call.cap 13 1 (i64) -> (i64) vh (vseven)
  vk = i64.const 1000
  vm = i64.mul vr0 vk
  vsum = i64.add vm vr1
  return vsum
  }
}
"#;

/// A powerbox whose op 1 the embedder answers, from a thread of its own, `argument + 100` after a
/// millisecond: the call waits on the world while the run goes on.
fn host_answered() -> (Host, Vec<Value>) {
    let mut host = Host::new();
    let h = host.grant_host_proc(HostProcGrant::new(
        ProcHandler::Offloadable(Box::new(|_op, args| {
            let a = plus_100(args);
            OffloadOutcome::Host(Box::new(move |c| {
                std::thread::spawn(move || {
                    std::thread::sleep(Duration::from_millis(1));
                    c.complete(a);
                });
            }))
        })),
        CapState::Stateless,
    ));
    host.register_cap_name("plus100", h);
    (host, vec![Value::I32(h)])
}

/// A powerbox whose op 1 a pool thread answers, `argument + 100`.
fn offloaded() -> (Host, Vec<Value>) {
    let mut host = Host::new();
    let h = host.grant_host_proc(HostProcGrant::new(
        ProcHandler::Offloadable(Box::new(|_op, args| {
            let a = plus_100(args);
            OffloadOutcome::Offload(Box::new(move || a))
        })),
        CapState::Stateless,
    ));
    host.register_cap_name("plus100", h);
    (host, vec![Value::I32(h)])
}

/// The root reads stdin once into a 64-byte buffer and returns how many bytes it got: 3, once the
/// embedder feeds `"abc"`.
const READ_ONCE: &str = r#"memory 16
func (i32) -> (i64) {
block 0 (vin: i32) {
  vptr = i64.const 17408
  vlen = i64.const 64
  vn = call.cap 0 0 (i64, i64) -> (i64) vin (vptr, vlen)
  return vn
  }
}
"#;

/// A powerbox holding stdin with nothing on it yet: the read waits until the runner supplies the
/// probe's `feed`.
fn stdin() -> (Host, Vec<Value>) {
    let mut host = Host::new();
    let vin = host.grant_stream(StreamRole::In);
    (host, vec![Value::I32(vin)])
}

// ---- job control ----------------------------------------------------------------------------------

/// The root stops its own process (`stop`), then asks whether its embedder has continued it
/// (`continued`): `1000 + 1` when nothing ran while it was stopped.
const STOP_SELF: &str = r#"memory 16
func (i32, i32) -> (i64) {
block 0 (vstop: i32, vcont: i32) {
  vz = i64.const 0
  vs = call.cap 13 0 (i64) -> (i64) vstop (vz)
  vc = call.cap 13 0 (i64) -> (i64) vcont (vz)
  vk = i64.const 1000
  vr = i64.add vk vc
  return vr
  }
}
"#;

/// The job-control probe's powerbox: the embedder's `stopper` as the process's signal source, a
/// host call that stops the process through it, and one that says whether the embedder has
/// continued it.
fn stop_self(e: &Embedder) -> (Host, Vec<Value>) {
    let mut host = Host::new();
    host.set_signal_source(e.stopper.clone(), Default::default());
    let stop = grant_named(&mut host, e, "stop");
    let cont = grant_named(&mut host, e, "continued");
    (host, vec![Value::I32(stop), Value::I32(cont)])
}

// ---- memory ---------------------------------------------------------------------------------------

/// The root maps the 64 KiB past its declared window through its `AddressSpace`, stores a marker in
/// the new page and loads it back: `0 * 10 + 424242` when the map succeeds.
const GROW: &str = r#"memory 16
func (i32) -> (i64) {
block 0 (vas: i32) {
  voff = i64.const 65536
  vlen = i64.const 65536
  vprot = i32.const 3
  vr = call.cap 5 0 (i64, i64, i32) -> (i64) vas (voff, vlen, vprot)
  vaddr = i64.const 65552
  vmark = i64.const 424242
  i64.store vaddr vmark
  vl = i64.load vaddr
  vk = i64.const 10
  vm = i64.mul vr vk
  vs = i64.add vm vl
  return vs
  }
}
"#;

/// A powerbox holding the whole window's `AddressSpace`.
fn memory() -> (Host, Vec<Value>) {
    let mut host = Host::new();
    let a = host.grant_memory();
    (host, vec![Value::I32(a)])
}

/// The root maps a shared region one map page long at two window offsets, stores 41 through the
/// first and loads it through the second: `0 * 1000 + 0 * 100 + 41` when both maps succeed. The map
/// page is the region's `page_size` (op 3), which a portable guest aligns its maps to (#1241): the
/// host page on unix, the 64 KiB allocation granularity on Windows, which the JIT maps by (#2565).
const REGION: &str = r#"memory 18
func (i32) -> (i64) {
block 0 (vrh: i32) {
  vlen = call.cap 4 3 () -> (i64) vrh ()
  vro = i64.const 0
  vprot = i32.const 3
  va = i64.const 65536
  vm0 = call.cap 4 0 (i64, i64, i64, i32) -> (i64) vrh (va, vro, vlen, vprot)
  vb = i64.add va vlen
  vm1 = call.cap 4 0 (i64, i64, i64, i32) -> (i64) vrh (vb, vro, vlen, vprot)
  vin = i64.const 41
  i64.store va vin
  vout = i64.load vb
  vk = i64.const 1000
  vh = i64.const 100
  vx = i64.mul vm0 vk
  vy = i64.mul vm1 vh
  vxy = i64.add vx vy
  vs = i64.add vxy vout
  return vs
  }
}
"#;

/// A powerbox holding one shared region, a map page long, over an OS shared-memory object
/// (`temen_run::new_shared_region`), which every engine can alias: the JIT maps its descriptor. A
/// region the host rebuilds, as a thaw does, is one too.
fn region() -> (Host, Vec<Value>) {
    let mut host = Host::new();
    host.set_region_factory(temen_run::new_shared_region);
    let page = temen_interp::host_region_granularity() as usize;
    let backing = temen_run::new_shared_region(page).expect("an OS shared-memory object");
    let r = host.grant_shared_region_backed(backing);
    (host, vec![Value::I32(r)])
}

// ---- code -----------------------------------------------------------------------------------------

/// The unit: `(x) -> x + 100`, over the probe's window (a unit declares the memory of the domain
/// that compiles it).
const UNIT: &str = r#"memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  v100 = i64.const 100
  vr = i64.add v0 v100
  return vr
  }
}
"#;

/// Where the guest-JIT probe keeps its unit's wire bytes.
const BLOB_AT: u64 = 20480;

/// The root compiles [`UNIT`] from the wire bytes at [`BLOB_AT`], installs it and calls it through
/// `call.dyn` on 5, then invokes it on 7: `105 * 1000 + 107`.
fn guest_jit() -> String {
    let blob = temen_encode::encode_module(&module(UNIT));
    let bytes: String = blob.iter().map(|b| format!("\\x{b:02x}")).collect();
    format!(
        "memory 16
data {BLOB_AT} \"{bytes}\"
func (i32) -> (i64) {{
block 0 (vjit: i32) {{
  vat = i64.const {BLOB_AT}
  vlen = i64.const {len}
  vc = call.cap 11 0 (i64, i64) -> (i64) vjit (vat, vlen)
  vslot = call.cap 11 3 (i64) -> (i64) vjit (vc)
  vs32 = i32.wrap_i64 vslot
  vfive = i64.const 5
  vr0 = call.dyn (i64) -> (i64) vs32 (vfive)
  vseven = i64.const 7
  vr1 = call.cap 11 1 (i64, i64) -> (i64) vjit (vc, vseven)
  vk = i64.const 1000
  vm = i64.mul vr0 vk
  vs = i64.add vm vr1
  return vs
  }}
}}
",
        len = blob.len(),
    )
}

/// The guest-JIT probe's powerbox: a `Jit` with an install table, as `temen_run::grant_jit` grants
/// one to every engine. An embedder that freezes and thaws the run grants the `Jit` that
/// instruments each unit as its program was (DURABILITY.md §12.5).
fn jit_powerbox(m: &Module, e: &Embedder) -> (Host, Vec<Value>) {
    let mut host = Host::new();
    let jit = if e.durable {
        temen_run::grant_jit_durable_confined(&mut host, m, 4)
    } else {
        temen_run::grant_jit(&mut host, m, 4)
    };
    (host, vec![Value::I32(jit)])
}

/// Serving (`jit_cap.rs`'s `SERVER`): offer `"counter"` op 0 is `bump(x) -> old`, adding `x` to the
/// live value at 16384. `main` seeds it with 7, serves what the embedder queued with `svc.poll`, and
/// returns `served * 1000 + value`.
const SERVER: &str = r#"
memory 16
type 0 func (i64) -> (i64)
type 1 interface { bump: 0 }
export 0 interface "counter" 1 { bump: 1 }

func () -> (i64) {
block 0 () {
  va = i64.const 16384
  vseed = i64.const 7
  i64.store va vseed
  vz = i32.const 0
  vn = call.cap 4294967295 9 () -> (i64) vz ()
  vafter = i64.load va
  vk = i64.const 1000
  vm = i64.mul vn vk
  vr = i64.add vm vafter
  return vr
  }
}

func (i64) -> (i64) {
block 0 (vx: i64) {
  va = i64.const 16384
  vold = i64.load va
  vnew = i64.add vold vx
  i64.store va vnew
  return vold
  }
}
"#;

/// The serving probe's powerbox: the running module, with two bumps (5 and 30) queued on its
/// offer, so `main` serves two and returns `2 * 1000 + 42`.
fn serve_powerbox(m: &Module) -> (Host, Vec<Value>) {
    let mut host = Host::new();
    host.set_self_module(&Arc::new(m.clone()));
    for x in [5, 30] {
        host.svc_enqueue(0, 0, vec![x])
            .expect("the offer's queue has room");
    }
    (host, Vec::new())
}
