//! #2695 step 2 — a domain its embedder **holds**, on any of the four executors ([`hold`]).
//!
//! The tree-walker, the pump and executor 2 hold one through [`HeldDomain::start`]. The Cranelift JIT
//! holds one here, where its host glue is: the domain's main thread runs the JIT run of its entry,
//! and once the entry returns the run rests instead of ending ([`temen_jit::CompiledModule::set_rest`]).
//! Its threads run on. Each call, the embedder's ([`HeldDomain::call`]) or one from inside its run
//! through a live offer ([`held_call`]), runs as an activation on the caller's thread at the domain's
//! [`HeldSite`], and one that parks is its caller's to carry. The domain ends when its embedder
//! ends it or it ends itself, and only then does its run's teardown begin.

use std::any::Any;
use std::sync::{Arc, Mutex, Weak};

use temen_interp::{
    CallerHost, Executor, HeldDomain, HeldEngine, Host, Reply, RootWindow, Trap, Value,
};
use temen_ir::errno::{EAGAIN, EINVAL};
use temen_ir::{FuncIdx, Module, ValType};
use temen_jit::{HandlerEnd, HeldCaller, HeldSite, JitOutcome, ServeRefused};

use crate::{jit_proc, CAP_REVOKED};

/// Hold `entry` of `m` with `args` over `host`, the domain's powerbox, on executor `on`, with `fuel`
/// for the domain's own work: its entry and its threads. Once the entry returns, the domain rests and
/// this answers the handle and the entry's results; if it ends first, the trap that ended it and the
/// powerbox. A durable domain cannot be held yet: it is refused with `ThreadFault`. The interpreted
/// executors hold it through [`HeldDomain::start`], which holds a module the bytecode engine declines
/// on the tree-walker; `Executor::Jit` holds it on the Cranelift JIT, unless a JIT run of it would fold
/// to the interpreters (`folds_to_oracle`), which then hold it. The JIT declines a module that
/// can serve its offers in a serve loop (`svc.poll`, `svc.wait`): a thread of it parked in `svc.wait`
/// takes its calls first, which the JIT's do not reach yet (#2895), so the tree-walker holds it
/// (INVARIANTS #9).
///
/// #3080 — a module that makes fibers is held on the JIT too: a fiber made in one call charges the
/// call that resumes it, as on the tree-walker, since each resume names the running activation's
/// cell (`VmCtx::fiber_switched_in`). Except where a fiber could call into a held domain
/// ([`fiber_may_call_in`]): the tree-walker holds those until #3164. So the JIT holds a domain where a
/// fiber can exist only when none of its code calls an interface a guest can implement and it cannot
/// reach `Jit`. Those interfaces include the built-in `Stream` and `Exit`, whose shape a guest's offer
/// can carry: a fiber-making module that prints or exits through its powerbox is the tree-walker's
/// until #3164.
pub fn hold(
    on: Executor,
    m: Arc<Module>,
    entry: FuncIdx,
    args: &[Value],
    fuel: u64,
    host: Host,
) -> Result<(HeldDomain, Vec<Value>), (Trap, Box<Host>)> {
    let jit = on == Executor::Jit
        && !crate::folds_to_oracle(&m)
        && !jit_proc::serves(&jit_proc::host_calls(&m), &host)
        && !fiber_may_call_in(&m, &host);
    if jit {
        hold_jit(m, entry, args, fuel, host)
    } else {
        HeldDomain::start(on, m, entry, args, fuel, host)
    }
}

/// #3080 (review B2, G1, G2, H1) — whether a guest fiber of the domain could call through an
/// interface a guest can implement, so into a held domain, its own included: on the JIT such a call
/// runs under the calling activation's gate and then gives it away, where the tree-walker keeps it
/// out until that activation parks or ends (#3164). It could when **a fiber can exist** (the module
/// makes one, or a unit it submits may, [`jit_proc::units_make_fibers`]: the compile stands a fiber
/// runtime up, by its own `cont.*` or `gc.roots` or by the hosting grant) **and code that can run in
/// one may make such a call**:
/// - the module's own code, through a `call.cap` on an interface a guest can implement
///   ([`temen_interp::guest_implementable`]: a guest-interned id, or a pre-seeded built-in shape,
///   `Stream` or `Exit`, which a guest's offer can carry), or through a bound import or a dynamic call
///   that may land on one ([`jit_proc::may_reach`]);
/// - any unit, when it can reach `Jit` ([`jit_proc::drives_jit`]): a unit `Jit.invoke`d from a fiber,
///   or one whose own fiber runs, is code not there to read before it arrives, and its `call.cap`
///   dispatches into the domain's powerbox.
fn fiber_may_call_in(m: &Module, host: &Host) -> bool {
    let calls = jit_proc::host_calls(m);
    let units = jit_proc::drives_jit(&calls, host);
    let fiber =
        m.funcs.iter().any(temen_ir::Func::uses_fibers) || jit_proc::units_make_fibers(m, host);
    let calls_a_guest = || {
        (calls.iter()).any(|&(t, o)| {
            jit_proc::may_reach(t, o, host, |t, _| temen_interp::guest_implementable(t))
        })
    };
    fiber && (units || calls_a_guest())
}

/// The domain's main thread: what its run answers, and its powerbox back.
type MainThread = std::thread::JoinHandle<(Result<JitOutcome, String>, Host)>;

/// [`hold`] on the Cranelift JIT.
fn hold_jit(
    m: Arc<Module>,
    entry: FuncIdx,
    args: &[Value],
    fuel: u64,
    mut host: Host,
) -> Result<(HeldDomain, Vec<Value>), (Trap, Box<Host>)> {
    if host.is_durable() {
        return Err((Trap::ThreadFault, Box::new(host)));
    }
    let Some(func) = m.funcs.get(entry as usize) else {
        return Err((Trap::Malformed, Box::new(host)));
    };
    let results: Vec<ValType> = func.results.clone();
    let slots: Vec<i64> = args.iter().copied().map(crate::value_slot).collect();
    host.hold_for_engine(true);
    let (rested, rests) = std::sync::mpsc::channel();
    let main: MainThread = std::thread::spawn(move || {
        let ran = run_held(&m, entry, &slots, fuel, &mut host, rested);
        host.hold_for_engine(false);
        // The domain has ended: what its window held goes back. The run's root: no spawn record
        // named it an end post.
        let no_end_post = host.release_memory();
        debug_assert!(no_end_post.is_none());
        host.release_pipe_ends();
        (ran, host)
    });
    match rests.recv() {
        Ok(Rested {
            site,
            cell,
            results: r,
        }) => {
            let held = JitHeld {
                site,
                cell,
                main: Mutex::new(Some(main)),
            };
            let values = results
                .iter()
                .zip(r)
                .map(|(&t, v)| crate::typed(t, v))
                .collect();
            Ok((HeldDomain::of(Box::new(held)), values))
        }
        // The run ended before its entry returned: it is never held.
        Err(_) => {
            let (ran, host) = main.join().unwrap_or_else(|p| std::panic::resume_unwind(p));
            Err((trap_of(ran).unwrap_or(Trap::ThreadFault), Box::new(host)))
        }
    }
}

/// What a held JIT run hands its starter once its entry has returned.
struct Rested {
    site: Arc<HeldSite>,
    /// The domain's powerbox cell, for a call's handler.
    cell: Weak<Mutex<Host>>,
    results: Vec<i64>,
}

/// The held run of `entry` of `m` with `slots` over `host`, on the domain's main thread: its code
/// charges fuel whatever `fuel` is, since each call lends its own, and it has a thread domain, so an
/// activation can park. Its rest publishes the domain's site in its powerbox, for a call from inside
/// its run, and to its starter.
fn run_held(
    m: &Module,
    entry: FuncIdx,
    slots: &[i64],
    fuel: u64,
    host: &mut Host,
    rested: std::sync::mpsc::Sender<Rested>,
) -> Result<JitOutcome, String> {
    let window = RootWindow::fresh(m, temen_ir::default_reserved_log2());
    let Some(node) = host.begin_activation(fuel, window) else {
        return Err(crate::no_room_for_run());
    };
    let chain = (node.fuel_room() >= 0)
        .then(|| Arc::new(crate::HostNode(node)) as Arc<dyn temen_jit::BudgetNode>);
    // What the cell drew and did not burn goes back to its node as it drops, with the run.
    let mut cell = chain.map_or_else(
        || temen_jit::FuelCell::fixed(u64::MAX),
        temen_jit::FuelCell::drawing,
    );
    let fuel_cell = &mut *cell as *mut temen_jit::FuelCell;
    let jit = jit_proc::drives_jit(&jit_proc::host_calls(m), host);
    // SAFETY: `host` is the run's alone; the cell outlives the run, which ends before this returns.
    let ran = unsafe {
        jit_proc::with_cap_ctx(host, true, |cc| {
            let mut cm = jit_proc::compile_image(cc, m, entry, None, jit, Some(fuel_cell), None)?;
            cm.enable_thread_hosting()?;
            let own = cc.with_host(|h| h.own_cell().map(|c| Arc::downgrade(&c)));
            // `set_rest` refuses a module with no nursery, which `compile_image` never makes: every
            // module it compiles has install room (#1726). Were it refused, the hold would end as a
            // failed run (`Trap::ThreadFault`, `hold_jit`), not decline to the interpreters.
            cm.set_rest(Some(Box::new(
                move |site: &Arc<HeldSite>, results: &[i64]| {
                    let cell = own.unwrap_or_default();
                    if let Some(c) = cell.upgrade() {
                        let mut g = c.lock().unwrap_or_else(|e| e.into_inner());
                        g.set_held_site(Some(Arc::clone(site) as Arc<dyn Any + Send + Sync>));
                        // A call from inside the run that waits for the rest looks again: `g` is the
                        // locked host of the live run's cell.
                        crate::ring(&g);
                    }
                    let _ = rested.send(Rested {
                        site: Arc::clone(site),
                        cell,
                        results: results.to_vec(),
                    });
                },
            )))?;
            jit_proc::run_image(
                cc,
                cm,
                None,
                jit_proc::Entry::Fresh {
                    args: slots,
                    init_mem: None,
                    snapshot_cap: None,
                },
            )
        })
    };
    ran.map(|run| run.outcome)
        .map_err(|e| format!("JIT compile failed: {e:?}"))
}

/// The trap a held JIT run ended with: `None` for a clean end, its embedder's.
fn trap_of(ran: Result<JitOutcome, String>) -> Option<Trap> {
    match ran {
        Ok(JitOutcome::Returned(_)) => None,
        Ok(JitOutcome::Exited(code)) => Some(Trap::Exit(code)),
        Ok(JitOutcome::Trapped(kind)) => Some(jit_proc::trap_of(kind)),
        Ok(JitOutcome::HostUnwound) | Err(_) => Some(Trap::ThreadFault),
    }
}

/// A domain the Cranelift JIT holds ([`hold`]): its rest's site, its powerbox cell, and its main
/// thread, until the handle ends it.
struct JitHeld {
    site: Arc<HeldSite>,
    cell: Weak<Mutex<Host>>,
    main: Mutex<Option<MainThread>>,
}

impl HeldEngine for JitHeld {
    fn call(&self, export: u32, op: u32, args: &[i64], fuel: &mut u64) -> Reply {
        let func = self.cell.upgrade().and_then(|c| {
            let g = c.lock().unwrap_or_else(|e| e.into_inner());
            g.svc_handler(export, op)
        });
        let Some(func) = func else {
            return Reply::Failed(if self.site.closed() {
                CAP_REVOKED
            } else {
                EINVAL
            });
        };
        // SAFETY: an outside call has no caller context; `activate` may be called whether or not the
        // run has returned, and a closed site refuses it touching nothing of the run (#2990).
        reply(unsafe { self.site.activate(func, args, HeldCaller::Outside(fuel)) })
            .unwrap_or(Reply::Failed(CAP_REVOKED))
    }

    fn finish(&mut self) -> Option<(Host, Option<Trap>)> {
        let main = self
            .main
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .take()?;
        self.site.end();
        let (ran, host) = main.join().unwrap_or_else(|p| std::panic::resume_unwind(p));
        Some((host, trap_of(ran)))
    }

    fn executor(&self) -> Executor {
        Executor::Jit
    }
}

/// An activation's reply: its results, or a failed call's errno. `None` when the caller must
/// unwind: its own domain ended while its call waited.
fn reply(ran: Result<Option<HandlerEnd>, ServeRefused>) -> Option<Reply> {
    match ran {
        Ok(Some(HandlerEnd::Returned(results))) => Some(Reply::Results(results)),
        // It ended the domain, or found it ended: a call into a dead domain (INVARIANTS #8).
        Ok(Some(HandlerEnd::Trapped(_))) => Some(Reply::Failed(CAP_REVOKED)),
        Ok(None) => None,
        Err(ServeRefused::NoHandler) => Some(Reply::Failed(EINVAL)),
        // The OS refused the activation's stack: probeable backpressure.
        Err(ServeRefused::NoStack) => Some(Reply::Failed(EAGAIN)),
    }
}

/// A call from inside the run into `callee`, a domain the JIT holds: an activation on the caller's
/// thread at its site, once it rests. While its entry runs, the caller waits on the domain's bell,
/// which its rest rings. A guest fiber's call parks its fiber once first, as every call through a
/// live offer does (#1952). INVARIANTS #8: the call's handles are granted into the callee first.
/// `None` when the caller must unwind (its trap cell set).
///
/// # Safety
/// As [`crate::live_reply`]'s: `vm` is the caller's live context.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn held_call(
    caller: &mut CallerHost,
    callee: &Mutex<Host>,
    export: u32,
    op: u32,
    args: &[i64],
    params: &[ValType],
    vm: &temen_jit::VmCtx,
) -> Option<Reply> {
    let lock = || callee.lock().unwrap_or_else(|e| e.into_inner());
    let mut parked_once = !temen_jit::in_guest_fiber();
    let (site, func) = loop {
        let g = lock();
        if !g.held_for_engine() {
            return Some(Reply::Failed(CAP_REVOKED));
        }
        let bell = g.svc_bell().cloned()?;
        let seen = crate::bell_now(Some(&bell));
        let site = g
            .held_site()
            .and_then(|s| s.downcast::<HeldSite>().ok())
            .filter(|_| parked_once);
        if let Some(site) = site {
            let Some(func) = g.svc_handler(export, op) else {
                return Some(Reply::Failed(EINVAL));
            };
            break (site, func);
        }
        if !parked_once {
            // Rung now, so the fiber's one park ends at its next resume.
            crate::ring(&g);
            parked_once = true;
        }
        drop(g);
        if !crate::park_until_rung(&bell, seen, temen_jit::Parks::Fiber, None, vm) {
            return None;
        }
    };
    let granted;
    let args = if params.contains(&ValType::Cap) {
        let mut a = args.to_vec();
        if temen_interp::grant_live_args(caller, callee, params, &mut a).is_err() {
            vm.raise(temen_jit::TrapKind::CapFault as i64);
            return None;
        }
        granted = a;
        &granted[..]
    } else {
        args
    };
    reply(site.activate(func, args, HeldCaller::Inside(vm)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What a [`decline_case`] module's one function holds.
    #[derive(Clone, Copy, Default)]
    struct Holds {
        cont: bool,
        gc_roots: bool,
        jit: bool,
        guest_call: bool,
        import: Option<temen_interp::BoundImport>,
        rebindable: bool,
        dynamic: bool,
    }

    /// A module whose function 0 holds the ops `h` names (a `cont.new` over function 1, a
    /// `gc.roots`, a `call.cap` on `Jit`, one on a guest-interned id, a `call.import` through import
    /// 0, a `call.import.dyn`), over a host with import 0 bound as `h.import` says. Parsed and
    /// verified, never run: [`fiber_may_call_in`] reads only the code and the bindings.
    fn decline_case(h: Holds) -> bool {
        let mut body = String::new();
        if h.cont {
            body += "  vf = i32.const 1\n  vsp = i64.const 4096\n  vk = cont.new vf vsp\n";
        }
        if h.gc_roots {
            body += "  g0 = i64.const 0\n  g1 = i64.const 4096\n  gm = i64.const -1\n  g2 = i64.const 0\n  g3 = i64.const 8\n  vg = gc.roots g0 g1 gm g2 g3\n";
        }
        if h.jit {
            body += "  vjh = i32.const 0\n  vjz = i64.const 0\n  vj = call.cap 11 3 (i64) -> (i64) vjh (vjz)\n";
        }
        if h.guest_call {
            body += "  vqh = i32.const 0\n  vq = call.cap 268435456 0 () -> (i64) vqh ()\n";
        }
        if h.import.is_some() {
            body += "  vi = call.import 0 ()\n";
        }
        if h.dynamic {
            body += "  vdh = i32.const 0\n  vd = call.import.dyn 1.kept vdh ()\n";
        }
        let src = format!(
            "memory 16\ntype 0 func () -> (i64)\ntype 1 interface {{ kept: 0 }}\nimport 0 \"k\" () -> (i64){mode}\nfunc () -> (i64) {{\nblock 0 () {{\n{body}  vr = i64.const 0\n  return vr\n  }}\n}}\nfunc (i64, i64) -> (i64) {{\nblock 0 (vsp: i64, varg: i64) {{\n  return varg\n  }}\n}}\n",
            mode = if h.rebindable { " rebindable" } else { "" },
        );
        let m = temen_text::parse_module(&src).expect("parse");
        temen_verify::verify_module(&m).expect("verify");
        let mut host = Host::new();
        host.set_import_bindings(h.import.into_iter().collect());
        fiber_may_call_in(&m, &host)
    }

    /// #3080 (review T1) — [`fiber_may_call_in`]'s truth table, read on the rule itself. `hold` asks
    /// `jit_proc::serves` first, which declines a rebindable slot and a `call.import.dyn` through the
    /// same `may_reach` arms, so a `hold` row cannot reach the fiber rule's own arms for them.
    #[test]
    fn a_fiber_may_call_in_when_a_fiber_can_exist_and_its_code_may_call_a_guest() {
        use temen_interp::BoundImport;
        use temen_ir::cap_id::{GUEST_IMPL_BASE, HOST_PROC};
        let fibers = Holds {
            cont: true,
            ..Holds::default()
        };
        let cases = [
            // H1's arm, its accept side: `gc.roots` stands the runtime up, but no unit can arrive.
            (
                "gc.roots, a guest call, no Jit",
                Holds {
                    gc_roots: true,
                    guest_call: true,
                    ..Holds::default()
                },
                false,
            ),
            (
                "gc.roots and Jit",
                Holds {
                    gc_roots: true,
                    jit: true,
                    ..Holds::default()
                },
                true,
            ),
            ("cont.*, no Jit, no guest call", fibers, false),
            (
                "Jit, no cont.* and no gc.roots, no hosting grant",
                Holds {
                    jit: true,
                    ..Holds::default()
                },
                false,
            ),
            (
                "cont.* and a rebindable slot",
                Holds {
                    import: Some(BoundImport {
                        type_id: GUEST_IMPL_BASE,
                        op: 0,
                        handle: 0,
                        bound: false,
                        rebindable: true,
                    }),
                    rebindable: true,
                    ..fibers
                },
                true,
            ),
            (
                "cont.* and a call.import.dyn",
                Holds {
                    dynamic: true,
                    ..fibers
                },
                true,
            ),
            (
                "cont.* and an import bound to a guest id",
                Holds {
                    import: Some(BoundImport::required(GUEST_IMPL_BASE, 0, 1)),
                    ..fibers
                },
                true,
            ),
            (
                "cont.* and an import bound for good to a host capability",
                Holds {
                    import: Some(BoundImport::required(HOST_PROC, 0, 1)),
                    ..fibers
                },
                false,
            ),
        ];
        for (name, holds, declines) in cases {
            assert_eq!(decline_case(holds), declines, "{name}");
        }
    }

    /// #2990 — an embedder keeps the `Arc<HeldSite>` past the run, so it may call `activate` after
    /// the run returned and its root's context, window and module are gone. The call is refused, and
    /// reads nothing the run owned: before the fix, the activation's guard read the dead root's end
    /// word as it left.
    #[test]
    fn a_call_through_the_site_after_the_run_returned_reads_nothing_of_the_run() {
        let m = temen_text::parse_module(
            "memory 16
func () -> (i64) {
block 0 () {
  vr = i64.const 1
  return vr
  }
}
",
        )
        .expect("parse");
        temen_verify::verify_module(&m).expect("verify");
        let mut host = Host::new();
        host.hold_for_engine(true);
        let (rested, rests) = std::sync::mpsc::channel();
        let main = std::thread::spawn(move || run_held(&m, 0, &[], u64::MAX, &mut host, rested));
        let site = rests
            .recv()
            .expect("the entry returned and the domain rests")
            .site;
        site.end();
        let ran = main.join().expect("the run's thread");
        assert!(matches!(ran, Ok(JitOutcome::Returned(_))), "{ran:?}");
        assert!(site.closed());
        let mut fuel = u64::MAX;
        for _ in 0..8 {
            // SAFETY: the corrected contract: a closed site touches only what it owns.
            let end = unsafe { site.activate(0, &[], HeldCaller::Outside(&mut fuel)) };
            assert!(
                matches!(end, Ok(Some(HandlerEnd::Trapped(_)))),
                "a call into a dead domain is refused"
            );
        }
    }
}
