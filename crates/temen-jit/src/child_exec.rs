//! **Child-domain executor** (DESIGN.md §23, D66) — every non-durable §14 child (the carve children
//! of Instantiator ops 0/5/8/11/13 and the detached children of op 15) runs as a **task on a bounded
//! pool of platform workers**, not one OS thread each. A task is a platform-owned fiber ([`FiberSlot::platform`]) carrying the child's own window,
//! trap cell and fiber execution context; any worker may resume it (migration under the same
//! single-owner claim guest fibers use), and a task that parks — a futex `wait` inside the child —
//! hands its worker back instead of holding an OS thread, so a lane-bounded parent can run N
//! children on L lanes without a parked child wedging a runnable one.
//!
//! **Parallelism is a granted resource, bounded at dispatch** (INVARIANTS #3, D66): every resume
//! first takes one slot in each bounded lane along the task's chain (`temen_ir::lanes` — the same
//! arithmetic the interpreter's two drivers gate on), and gives them back when the task yields or
//! finishes. A child of a cap-1 parent therefore never overlaps a sibling, whatever the worker count.
//! Workers are spawned on demand up to the widest lane a task could use (one per task when
//! unbounded), so an unbounded run keeps today's thread-per-child parallelism; only the *shape*
//! changed (migrating tasks), never the ceiling.
//!
//! What this does **not** do (the JIT frontier, tracked on #1600): preempt. A task that never parks
//! holds its lane until it returns; the interpreter's quantum round-robin has no JIT twin. And a
//! domain's *own* `thread.spawn` vCPUs — a child task's included (#1469) — stay 1:1 OS threads,
//! counted by `max_vcpus`; they are not tasks.
//!
//! **Env-swap rules** (the D66 checklist R1–R5): each resume is its own `run_guarded_range` bracket
//! over the *task's* fault range (R1); the per-thread state the child reads through TLS — its fiber
//! runtime (`CURRENT_RT`), its thread domain (`CURRENT_DOMAIN`), `vcpu.tls`, the in-task flag — is
//! seeded on entry and saved/reset on exit of every resume, and every reader on a fiber stack is
//! `#[inline(never)]` (R2, #1466); a finished or faulted task is never resumed (R3: the slot's
//! `finish` closes the claim); a trap is attributed to the task's own cell, never the run's (R4);
//! the window base is baked at first entry and the window moves with the task (R5 —
//! `GuestWindow: Send`).

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use temen_fiber::{Fiber, State};

use crate::fiber_rt::{self, FiberRuntime, FiberSlot, SharedFiberTable};
use crate::instantiator_rt::ChildDone;
use crate::os_thread_rt::{self, Domain};
use crate::{mem, vcpu_tls, CompiledModule, TrapKind, VmCtx};

/// A task's native control stack: the fiber arena's slot size (`temen-fiber` hands out fixed
/// 256 KiB slots), the same stack every guest fiber runs on. Less headroom than the 2 MiB OS thread
/// a detached child used to get — but bounded by the guest's own prologue checks now (the fiber's
/// low bound rides in as the child's `stack_limit`), so a child that recurses past it traps
/// `StackOverflow` cleanly instead of relying on the OS guard.
const TASK_STACK: usize = 1 << 18;

/// How often an idle worker re-offers the parked tasks so each re-checks its own predicate (the §5
/// kill cell, a torn-down domain) — the same bounded cadence a parked 1:1 vCPU uses
/// (`os_thread_rt::KILL_RECHECK`), one timer for the pool instead of one per parked task. A parked
/// task's own deadline comes sooner when it is sooner (#2012: [`Entry::deadline`]).
const RECHECK: Duration = Duration::from_millis(20);

/// The limit-taking buffer-ABI trampoline a task's fiber body enters the child through
/// (`build_trampoline(.., with_limit = true)`): the five `Entry` params plus the running stack's
/// low bound, which the fiber alone knows.
type LimitedEntry =
    extern "C" fn(*const i64, *mut i64, *mut u8, *const core::ffi::c_void, *mut i64, u64);

/// A **carve** child's copy-back: the parent is the superset, so the child's window image is
/// written back into its carve at finish. `None` for a detached child (its window is its own).
pub(crate) type CopyBack = Box<dyn FnOnce(&[u8]) + Send>;

/// A task's one-shot teardown: release the child powerbox, return its lane and its window's bytes.
/// Its argument says whether a freeze captured the child (#1971): a child that unwound for a freeze
/// is carried by the artifact, not ended, so its window stays charged, as on the interpreter.
pub(crate) type Teardown = Box<dyn FnOnce(bool) + Send>;

/// A raw pointer that crosses to a worker under the executor's ownership discipline (documented at
/// each construction site).
struct SendRaw<T>(T);
// SAFETY: every `SendRaw` here wraps a pointer whose pointee is either immutable for the run
// (compiled code, the parent host the grant hooks were registered with) or owned by exactly one
// task and touched only by the worker holding that task's claim.
unsafe impl<T> Send for SendRaw<T> {}

/// One detached child as the executor sees it: everything a resume needs, owned here (not on the
/// fiber stack) so a task abandoned by a fault still tears down from the worker's own frame.
pub(crate) struct ChildTask {
    /// The platform fiber slot (owns the `Fiber`; the single-owner claim arbitrates workers).
    slot: Arc<FiberSlot>,
    /// The child's own fiber execution context — published as `CURRENT_RT` for each resume so the
    /// futex thunk's "inside a fiber ⇒ park the fiber" arm sees this task.
    rt: Box<FiberRuntime>,
    /// The child's window — its own for a detached child, a private image of the parent's carve
    /// for a carve child (seeded by `init`, written back by `copy_back`). Moves with the task
    /// (`GuestWindow: Send`).
    window: mem::GuestWindow,
    fault: (usize, usize),
    /// The task's own instance context — the child's powerbox, the parent's kill-path cell, its own
    /// budget cell, and (field 0) its own trap cell (R4). Heap-stable: threaded into the child's
    /// frames at first entry. Shared with the task's [`Entry`] so teardown can reach the trap cell
    /// while a worker holds the task.
    vm: Arc<VmCtx>,
    /// The entry's result buffer (heap-stable, written by the trampoline at return).
    results: Box<[i64]>,
    /// The arguments, alive until the fiber body's first entry reads them.
    _args: Box<[i64]>,
    /// The compiled child, alive as long as its code can run.
    _code: Arc<CompiledModule>,
    /// D66 lane chain: `(domain, cap)` from the root down to this child (`-1` = unbounded).
    chain: Vec<(usize, i64)>,
    /// Saved `vcpu.tls` register between residencies (R2: task-level, not thread-level).
    tls: i64,
    done: Arc<ChildDone>,
    /// A **carve** child's copy-back: runs once at finish over the task's window image, before
    /// `teardown` — the parent (superset) then sees the child's writes. `None` for a detached child
    /// (its window is its own; nothing to copy anywhere).
    copy_back: Option<CopyBack>,
    /// Runs exactly once, after the last resume, while the window is still alive: releases the
    /// child powerbox and returns the child's lane to the parent.
    teardown: Option<Teardown>,
    /// #1469 — the domain the child's own `thread.spawn` vCPUs run in (their handle table, window,
    /// fiber registry and lane chain), sharing the run's futex, park count, lanes and live count.
    /// Built when the task is filed ([`ChildExec::spawn`]), for a child compiled with thread ops.
    dom: Option<Arc<Domain>>,
    /// #2001 — a detached child's own budget node, handed to [`Self::dom`] for its threads' charges.
    node: Option<Arc<dyn crate::BudgetNode>>,
    /// #1469 — the child's root has returned (its outcome is published) while vCPUs it spawned
    /// still run. As on the oracle, a root's return does not end its domain; the task stays filed,
    /// never resumed again, until the last of them ends, and only then frees its window and
    /// powerbox ([`ChildExec::finish`]).
    retiring: bool,
    /// What a thawed durable child's capture left beside its image: its fibers (#2031) and its
    /// `thread.spawn` vCPUs with its root's extent (#2010), re-created when it is filed
    /// ([`ChildExec::spawn`]). Empty for a fresh spawn.
    pub(crate) thaw: crate::DurableResidue,
}

// SAFETY: a task moves between workers only through the executor's queue, and is touched only by
// the worker holding its slot claim (`FiberSlot::claim_for_worker`) — one thread at a time, with
// the claim's acquire/release pairing carrying every field across the hand-off. The runtime's raw
// yielder pointers are live only during a residency (pushed and popped on the running worker).
unsafe impl Send for ChildTask {}

impl ChildTask {
    /// Build a task around a compiled detached child. `init` seeds the fresh window (data segments,
    /// payload) and may protect its pages (a detached child's readonly segments); `premap` aliases the op-15 pre-mapped region onto it (`false` ⇒ the child never
    /// runs: a `CapFault` outcome, as on the OS-thread path it replaces).
    ///
    /// # Safety
    /// `code` was compiled by `compile_child_windowed` for exactly `(mapped_log2, reserved_log2)`
    /// (so it carries a limit-taking trampoline); `args` matches the entry's arity.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn new(
        code: Arc<CompiledModule>,
        mapped_log2: u8,
        reserved_log2: u8,
        init: impl FnOnce(&mut mem::GuestWindow),
        premap: impl FnOnce(*mut u8, u64, u64) -> bool,
        args: Vec<i64>,
        n_results: usize,
        chain: Vec<(usize, i64)>,
        done: Arc<ChildDone>,
        copy_back: Option<CopyBack>,
        teardown: Teardown,
        node: Option<Arc<dyn crate::BudgetNode>>,
    ) -> Result<ChildTask, Teardown> {
        let mut window = mem::GuestWindow::new(1usize << mapped_log2, 1usize << reserved_log2);
        let base = window.base();
        init(&mut window);
        // #964/#1733: every child window reserves its own `[0, guard)`, as the interpreter's windows
        // do — last, after the host's seeding above; a window smaller than the guard skips, as the
        // interpreter's `seed_null_guard` does. `finish` re-asserts RW before the host reads it back.
        let guard = temen_ir::module_null_guard();
        if guard <= 1u64 << mapped_log2 {
            window.seed_null_guard(0, guard);
        }
        if !premap(base, 1u64 << mapped_log2, 1u64 << reserved_log2) {
            window.restore_rw();
            return Err(teardown);
        }
        let fault = window.fault_range();
        // The child runs as its own instance, the one its compile recorded (#1768).
        let vm = Arc::new(VmCtx::new(code.instance));
        let results: Box<[i64]> = vec![0i64; n_results.max(1)].into_boxed_slice();
        let args: Box<[i64]> = args.into_boxed_slice();
        // The child domain's own execution context and fiber table (#1469): its `cont.*` handles
        // live here, numbered from 0 and out of every other domain's reach, as the oracle's
        // per-domain registry. A child compiled without `cont.*` gets an unused one-slot table; the
        // runtime still carries the `yielders` / `active_slots` bookkeeping that
        // `fiber_event_park` and `current_fiber_slot` read. A durable child's table knows its shadow
        // arena, where its `thread.spawn` vCPUs reserve their contexts (#2010).
        let shadow = done
            .durable
            .as_ref()
            .map_or(temen_ir::durable_abi::ShadowArena::EMPTY, |d| d.shadow);
        let mut rt = match (code.fiber_cfg, code.call_tramp) {
            (Some((type_id, mask)), Some(t)) => {
                let table = Arc::new(SharedFiberTable::new(
                    temen_ir::Quota::default().max_fibers,
                    shadow,
                ));
                let mut rt = Box::new(FiberRuntime::new(table, type_id, mask));
                rt.set_call_tramp(t);
                rt
            }
            _ => {
                let table = Arc::new(SharedFiberTable::new(1, shadow));
                Box::new(FiberRuntime::new(table, 0, code.fn_table_mask))
            }
        };
        let tramp: LimitedEntry = core::mem::transmute(code.tramp_code_limited);
        let a = SendRaw(args.as_ptr());
        let r = SendRaw(results.as_ptr() as *mut i64);
        let b = SendRaw(base);
        let t = SendRaw(code.fn_table.as_ptr() as *const core::ffi::c_void);
        // The entry ABI spells the vmctx as its trap cell (`VmCtx` field 0).
        let c = SendRaw(Arc::as_ptr(&vm) as *mut i64);
        // The body: enter the child once through the limit-taking trampoline with this fiber's
        // stack low bound (§2b path B — the prologue checks guard the fiber stack). Every park inside
        // is a `fiber_event_park` yield from within this call; the body returns when the entry does.
        let fiber = fiber_rt::make_task_fiber(TASK_STACK, move |stack_low| {
            let (a, r, b, t, c) = (a, r, b, t, c);
            tramp(a.0, r.0, b.0, t.0, c.0, stack_low);
        });
        let Some(fiber) = fiber else {
            window.restore_rw();
            return Err(teardown);
        };
        // The child root's frames live on the task stack: its top is the high bound of the
        // `gc.roots` root-frame scan, as the OS-thread entry SP is a root's.
        rt.set_root_entry_sp(fiber.stack_top() as usize);
        // #2031 — a durable child's fiber switches swap shadow-SPs and mark consumed parks over its
        // own window, as a run's root's do (`run_inner`).
        rt.set_durable_env(base as u64, done.durable.is_some());
        Ok(ChildTask {
            slot: FiberSlot::platform(fiber),
            rt,
            window,
            fault,
            vm,
            results,
            _args: args,
            _code: code,
            chain,
            tls: 0,
            done,
            copy_back,
            teardown: Some(teardown),
            dom: None,
            node,
            retiring: false,
            thaw: crate::DurableResidue::default(),
        })
    }
}

impl ChildTask {
    /// The task's window base (its own window's byte 0) — published for a durable child's doorbell.
    pub(crate) fn window_base(&self) -> usize {
        self.window.base() as usize
    }

    /// Whether vCPUs this task's child spawned are still running.
    fn threads_live(&self) -> bool {
        self.dom.as_ref().is_some_and(|d| d.live_threads() > 0)
    }

    /// #1937, #2010 — what a run's teardown does with this task, found parked. A durable child whose
    /// own freeze word a freeze has rung unwinds and rides the artifact; poisoned, it would end in a
    /// trap and leave no image.
    fn at_teardown(&self) -> AtTeardown {
        let rung = self.done.durable.is_some()
            // SAFETY: the task's window is live while the task is; its first page holds the word.
            && unsafe { fiber_rt::window_is_unwinding(self.window.base() as u64) };
        // A retiring task's root has returned while its vCPUs run on. Under a freeze of its window
        // they unwind too, and the last one's end re-offers it; any other teardown ends them.
        if self.retiring {
            return if rung {
                AtTeardown::Wait
            } else {
                AtTeardown::Poison
            };
        }
        match self.slot.parked_on() {
            // Its futex wait or thread join ends on that word when re-checked.
            fiber_rt::ParkOn::Futex(_) | fiber_rt::ParkOn::Thread if rung => AtTeardown::Wake,
            // Its join waits on a child the same freeze rang (`Nursery::ring_detached`), whose end
            // re-offers it. Woken any sooner, it would only park again.
            fiber_rt::ParkOn::Child if rung => AtTeardown::Wait,
            // Any other park may never come back.
            _ => AtTeardown::Poison,
        }
    }

    /// #1361 step 4 — whether a durable child's root unwound for a freeze: it spilled past its
    /// frame base under its window's `UNWINDING`, the rule a thread vCPU's end follows
    /// (`os_thread_rt::run_child`). The word alone is not enough (#1937): a doorbell that lands after
    /// the child's last safepoint leaves one that returned normally with its word `UNWINDING`.
    fn root_unwound(&self) -> bool {
        let Some(d) = self.done.durable.as_ref() else {
            return false;
        };
        let base = self.window.base() as u64;
        self.vm.trap.load(Ordering::Relaxed) == 0
            // SAFETY: the window is live while the task is; its first page holds the freeze word at
            // `STATE_OFF`, and context 0's region the root's shadow-SP word.
            && unsafe { fiber_rt::window_is_unwinding(base) }
            && unsafe { fiber_rt::read_shadow_sp(base, d.shadow.region_base(0)) }
                > d.shadow.frame_base(0)
    }
}

/// What a run's teardown does with a parked task ([`ChildTask::at_teardown`]).
#[derive(Clone, Copy, PartialEq, Eq)]
enum AtTeardown {
    /// Poison its cell, so its park returns and its trailing guard unwinds it.
    Poison,
    /// Wake it: its park ends on its own freeze word.
    Wake,
    /// Leave it parked until the event it waits for.
    Wait,
}

/// In/out cell for the `Entry`-shaped [`resume_shim`]: the fiber to resume, and what it did.
struct ResumeCall {
    fib: *mut Fiber,
    state: Option<State>,
}

/// `Entry`-shaped so it runs under [`mem::run_guarded_range`] (the same shim shape as
/// `os_thread_rt::child_entry`): resume the task's fiber once. A guest fault inside the fiber
/// `longjmp`s past this frame to the guard; the fiber is then abandoned, never resumed (R3).
extern "C" fn resume_shim(
    a: *const i64,
    _r: *mut i64,
    _m: *mut u8,
    _t: *const core::ffi::c_void,
    _tc: *mut i64,
) {
    // SAFETY: `a` is the `&mut ResumeCall` the worker passed; `fib` is the claimed task's fiber.
    unsafe {
        let c = a as *mut ResumeCall;
        (*c).state = Some((*(*c).fib).resume(0));
    }
}

/// Per-task executor bookkeeping. `task` is `None` while a worker holds the task (running).
struct Entry {
    task: Option<Box<ChildTask>>,
    /// Parked (yielded on an event park) and not yet woken. Drives the cadence sweep and the
    /// teardown poison; **not** the same question as [`Self::counted`].
    parked: bool,
    /// #1631 — whether this park was counted in `Domain::parked`. A park that carries its own
    /// deadline is not, because that counter's one reader is the futex deadlock predicate
    /// `live > parked` and a task sleeping on a deadline is a potential notifier, not a blocked
    /// one. Tracked separately so the wake decrements exactly what the park incremented: keying
    /// the decrement off `parked` instead would underflow the counter on every timed park.
    counted: bool,
    /// #2012 — the deadline this park carries, if any: an idle worker wakes no later than the
    /// earliest one, so a timed wait inside a child ends on time rather than on the next
    /// [`RECHECK`] sweep.
    deadline: Option<Instant>,
    /// A wake arrived (possibly while the task was still running toward its park).
    woken: bool,
    /// A **carve** child's trap cell, for a teardown that ends the parent's domain: the parent's
    /// completion ends its nested children, running ones included (DESIGN §12 domain teardown),
    /// and a running task is reachable only through this — its `task` is on a worker. `None` for a
    /// detached child, which a parent's completion does not end.
    stop: Option<Arc<VmCtx>>,
    /// #1469 — the task's own domain, reachable while a worker holds the task: a poisoned task's
    /// parked vCPUs must be woken to observe it.
    dom: Option<Arc<Domain>>,
    /// #2074 — the task's instance (its trap cell in field 0), reachable while a worker holds the
    /// task, for [`ChildExec::kill`].
    vm: Arc<VmCtx>,
}

struct ExecState {
    tasks: HashMap<u64, Entry>,
    /// Runnable tasks, FIFO. A task whose lane is full stays here — [`ChildExec::pick`] scans past
    /// it rather than shuffling it to a second queue, so there is one runnable set, not two.
    runnable: VecDeque<u64>,
    next_id: u64,
    workers: Vec<std::thread::JoinHandle<()>>,
    idle_workers: usize,
    /// Run teardown began: a park now poisons the task (it unwinds through its trailing guard),
    /// and workers exit once no task remains.
    shutdown: bool,
    /// When an idle worker last re-offered every parked task (the [`RECHECK`] cadence).
    swept: Instant,
}

/// The executor (one per run, owned by the nursery). Workers are OS threads spawned on demand.
pub(crate) struct ChildExec {
    state: Mutex<ExecState>,
    /// Workers wait here for runnable tasks.
    cv: Condvar,
    /// `shutdown_and_join` waits here for the last task to finish.
    quiescent: Condvar,
    /// The run's `Domain` (`0` = none): live/parked accounting for the 1:1 vCPUs' deadlock
    /// predicate, and the futex broadcast a finished child issues.
    dom: usize,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl ChildExec {
    pub(crate) fn new(dom: usize) -> Arc<ChildExec> {
        Arc::new(ChildExec {
            state: Mutex::new(ExecState {
                tasks: HashMap::new(),
                runnable: VecDeque::new(),
                next_id: 1,
                workers: Vec::new(),
                idle_workers: 0,
                shutdown: false,
                swept: Instant::now(),
            }),
            cv: Condvar::new(),
            quiescent: Condvar::new(),
            dom,
        })
    }

    fn domain(&self) -> Option<&Domain> {
        // SAFETY: a nonzero `dom` is the run's live `Domain`, which outlives the nursery (and so
        // this executor and every task).
        (self.dom != 0).then(|| unsafe { &*(self.dom as *const Domain) })
    }

    /// File a task and make it runnable. Its `done` cell is filled when it finishes. Returns its id,
    /// which [`Self::kill`] takes.
    pub(crate) fn spawn(self: &Arc<Self>, mut task: ChildTask) -> u64 {
        // #1469 — a child compiled with thread ops gets its own domain over its window (the thread
        // thunks find it through `os_thread_rt::CURRENT_DOMAIN`, set for each residency below).
        // So does a durable child (#1937): a wait ends on a freeze of its domain's window, and a
        // durable child's is its own, the word its safepoints read. Its parent's domain, whose
        // futex it shares through the hub, names the parent's window.
        let durable = task.done.durable.is_some();
        // #2031 — a thawed durable child's fibers go back in its table first, as a run's root's do:
        // its re-attached vCPUs and its root's rewind resolve them there. A child compiled without
        // `cont.*` has no table to seed, so residue for one fails closed.
        let thaw = std::mem::take(&mut task.thaw);
        if !thaw.fibers.is_empty() {
            let seeded = task._code.call_tramp.is_some()
                // SAFETY: the task's runtime, window (seeded with its image), table and trap cell
                // are live, and none of its code has run yet.
                && unsafe {
                    fiber_rt::seed_frozen_fibers(
                        &mut *task.rt,
                        &thaw.fibers,
                        task.window.base() as u64,
                        task._code.fn_table.as_ptr() as u64,
                        Arc::as_ptr(&task.vm) as u64,
                    )
                };
            if !seeded {
                task.vm
                    .trap
                    .store(TrapKind::FiberFault as i64, Ordering::Relaxed);
            }
        }
        if task._code.thread.spawn_thunk != 0 || durable {
            if let Some(hub) = self.domain() {
                let d = Arc::new(Domain::new_child(hub, task.chain.clone(), task.node.take()));
                d.set_env(
                    task.window.base() as u64,
                    task._code.fn_table.as_ptr() as u64,
                    Arc::as_ptr(&task.vm), // the child's vCPUs are its instance
                    task._code.call_tramp,
                    task.fault,
                    task._code.fiber_cfg,
                    Some(task.rt.table()),
                    task._code.instance.epoch as usize,
                    durable,
                );
                // #2010 — a durable child's vCPUs each reserve a shadow context of their own, so a
                // freeze of its window unwinds each into its own region, wherever its root is. A
                // thaw re-spawns the ones its capture carried, rewinding, before its root runs.
                if durable {
                    d.engage_concurrent_durable();
                    if !thaw.vcpus.is_empty() {
                        // An extent-less capture is an empty root's, as for a run's root.
                        let root_sp = thaw.root_sp.unwrap_or(d.shadow().frame_base(0));
                        // SAFETY: the env just set is the task's: its window (seeded with its image)
                        // and code live until the task finishes, after its vCPUs are joined.
                        unsafe { d.thaw_reattach_and_run(&thaw.vcpus, root_sp) };
                    }
                }
                task.dom = Some(d);
            }
        }
        let mut g = lock(&self.state);
        let id = g.next_id;
        g.next_id += 1;
        // Workers: one more if none is idle and the pool could still use one — bounded by the
        // widest lane this task could run in (its tightest cap), or by the task count when its
        // chain is unbounded (today's thread-per-child parallelism, as a pool).
        // A task takes one slot of its own lane whatever happens, so only the *enclosing* lanes
        // bound how many tasks can run at once.
        let enclosing = task.chain.len().saturating_sub(1);
        let widest = task.chain[..enclosing]
            .iter()
            .filter(|&&(_, cap)| cap >= 0)
            .map(|&(_, cap)| cap as usize)
            .min()
            .unwrap_or(g.tasks.len() + 1);
        let stop = task.copy_back.is_some().then(|| Arc::clone(&task.vm));
        let dom = task.dom.clone();
        let vm = Arc::clone(&task.vm);
        g.tasks.insert(
            id,
            Entry {
                task: Some(Box::new(task)),
                parked: false,
                deadline: None,
                counted: false,
                woken: false,
                stop,
                dom,
                vm,
            },
        );
        g.runnable.push_back(id);
        self.ensure_worker(&mut g, widest);
        self.cv.notify_one();
        id
    }

    /// `Instantiator.kill` (#2074): end task `id` with `ThreadFault`, running or parked. Its trap
    /// cell stops its running vCPUs at their next entry or back-edge poll (`emit_domain_poll`), and a
    /// parked task (or a parked vCPU of its own) is woken to observe it, as a teardown's poison is.
    /// A task that has finished is gone from the table, and nothing happens.
    pub(crate) fn kill(&self, id: u64) {
        let mut g = lock(&self.state);
        let Some(e) = g.tasks.get_mut(&id) else {
            return;
        };
        // Never clobber a trap the child already recorded.
        let _ = e.vm.trap.compare_exchange(
            0,
            TrapKind::ThreadFault as i64,
            Ordering::Relaxed,
            Ordering::Relaxed,
        );
        if let Some(d) = &e.dom {
            d.wake_own_parked();
        }
        // A running task may be on its way to a park: latch the wake, as `wake_parked` does.
        e.woken = true;
        if std::mem::take(&mut e.parked) {
            e.deadline = None;
            if std::mem::take(&mut e.counted) {
                if let Some(d) = self.domain() {
                    d.task_unparked();
                }
            }
            g.runnable.push_back(id);
        }
        self.cv.notify_all();
    }

    fn ensure_worker(self: &Arc<Self>, g: &mut ExecState, want: usize) {
        if g.idle_workers > 0 || g.workers.len() >= want {
            return;
        }
        let me = Arc::clone(self);
        let worker = g.workers.len() as u64 + 1;
        let h = std::thread::Builder::new()
            .name("temen-child".into())
            .spawn(move || me.worker_loop(worker));
        // An `Err` is the OS refusing a thread: the existing workers (if any) still drain the
        // queue; with none, the task waits for a later spawn's worker or runs at shutdown on the
        // joining thread.
        if let Ok(h) = h {
            g.workers.push(h);
        }
    }

    /// Make every parked task runnable so it re-checks its own park predicate. This is the whole
    /// wake mechanism: a `notify`, a kill, a vCPU exit and run teardown all reach it through
    /// [`Domain::wake_all_parked`], and an idle worker's cadence sweep fires deadlines with it.
    /// A task that re-checks and is still unsatisfied simply parks again.
    /// D66 — a lane was given back somewhere in this domain (by a 1:1 vCPU, say): re-offer the
    /// queue, so a task that `pick` refused for want of a lane is reconsidered. Without this an idle
    /// worker sleeps on its condvar until something *else* wakes it, which for a run whose only
    /// other activity is the vCPU that just stepped aside is never.
    pub(crate) fn notify_workers(&self) {
        let _g = lock(&self.state);
        self.cv.notify_all();
    }

    pub(crate) fn wake_all_parked_locked(&self) {
        let mut g = lock(&self.state);
        self.wake_all_parked(&mut g);
    }

    fn wake_all_parked(&self, g: &mut ExecState) {
        self.wake_parked(g, None);
    }

    /// Make the parked tasks runnable — all of them, or with `due`, only those whose own deadline
    /// is at or before it (#2012: a deadline fired between two cadence sweeps).
    fn wake_parked(&self, g: &mut ExecState, due: Option<Instant>) {
        if due.is_some() {
            let ids: Vec<u64> = g
                .tasks
                .iter()
                .filter(|(_, e)| e.parked && e.deadline.is_some_and(|d| Some(d) <= due))
                .map(|(&id, _)| id)
                .collect();
            for id in ids {
                let e = g.tasks.get_mut(&id).expect("listed");
                e.parked = false;
                e.deadline = None;
                e.woken = true;
                // A deadline park is never counted (#1631), so there is no `Domain::parked` to undo.
                g.runnable.push_back(id);
            }
            self.cv.notify_all();
            return;
        }
        // #1711 — a running task may already be past its predicate check and yielding toward a
        // park that the worker has not filed yet. Latch the wake on it too, so the worker requeues
        // it instead of parking it on a cell this wake has already satisfied. At worst the task
        // re-checks once more than it needed to.
        for e in g.tasks.values_mut().filter(|e| e.task.is_none()) {
            e.woken = true;
        }
        let ids: Vec<u64> = g
            .tasks
            .iter()
            .filter(|(_, e)| e.parked)
            .map(|(&id, _)| id)
            .collect();
        for id in ids {
            let e = g.tasks.get_mut(&id).expect("listed");
            e.parked = false;
            e.deadline = None;
            e.woken = true;
            // #1631 — decrement exactly what the park incremented (see `Entry::counted`).
            if std::mem::take(&mut e.counted) {
                if let Some(d) = self.domain() {
                    d.task_unparked();
                }
            }
            g.runnable.push_back(id);
        }
        self.cv.notify_all();
    }

    /// The next runnable task whose whole lane chain has room, taking its lanes. A task refused a
    /// lane keeps its place in the queue; a later release simply makes the next scan admit it.
    fn pick(&self, g: &mut ExecState) -> Option<u64> {
        // The lane counts live on the `Domain`, shared with its own 1:1 vCPUs — a task of this
        // subtree and a `thread.spawn` vCPU of the parent draw on the *same* lane, so a private map
        // here would enforce the cap twice and hand out double the granted parallelism. Taking the
        // domain's lane lock while holding this one is the one lock order used anywhere; nothing
        // takes this lock while holding that one.
        let mut chains = g.runnable.iter().map(|id| {
            g.tasks
                .get(id)
                .and_then(|e| e.task.as_ref())
                .map(|t| t.chain.as_slice())
        });
        let admit = match self.domain() {
            Some(d) => d.lane_try_enter_first(chains)?,
            None => chains.position(|c| c.is_some())?, // no domain ⇒ no lanes to honour (the durable nested nursery)
        };
        let id = g.runnable.remove(admit).expect("position is in range");
        g.tasks.get_mut(&id).expect("scanned").woken = false;
        Some(id)
    }

    fn worker_loop(self: Arc<Self>, worker: u64) {
        mem::install_guard();
        loop {
            let (id, mut task) = {
                let mut g = lock(&self.state);
                let id = loop {
                    if let Some(id) = self.pick(&mut g) {
                        break id;
                    }
                    if g.shutdown && g.tasks.is_empty() {
                        return;
                    }
                    g.idle_workers += 1;
                    // A parked task has no resumer polling it, so the pool re-offers every parked
                    // task on a bounded cadence and each one re-checks its own predicate — a passed
                    // deadline, a fired kill cell, a torn-down domain — in the loop it already has
                    // (`fiber_futex_wait`). One sweep for all three, rather than a second copy of
                    // each condition here. A prompt wake (a `notify`, teardown) does not wait for
                    // the cadence: it arrives through `Domain::wake_all_parked`. Nor does a parked
                    // task's own deadline (#2012): the worker wakes for the earliest one, if it
                    // comes before the next sweep, and re-offers just the tasks it has passed.
                    let sweeping = g.tasks.values().any(|e| e.parked);
                    g = if sweeping {
                        let next = g
                            .tasks
                            .values()
                            .filter_map(|e| e.deadline.filter(|_| e.parked))
                            .fold(g.swept + RECHECK, Instant::min);
                        let wait = next.saturating_duration_since(Instant::now());
                        self.cv
                            .wait_timeout(g, wait)
                            .unwrap_or_else(|e| e.into_inner())
                            .0
                    } else {
                        self.cv.wait(g).unwrap_or_else(|e| e.into_inner())
                    };
                    g.idle_workers -= 1;
                    if sweeping {
                        let now = Instant::now();
                        if now >= g.swept + RECHECK {
                            g.swept = now;
                            self.wake_all_parked(&mut g);
                        } else {
                            self.wake_parked(&mut g, Some(now));
                        }
                    }
                };
                let task = g
                    .tasks
                    .get_mut(&id)
                    .and_then(|e| e.task.take())
                    .expect("picked task is present");
                (id, task)
            };
            // A retiring task is never resumed: it is offered only to re-check its vCPUs.
            let outcome = if task.retiring {
                Outcome::Retiring
            } else {
                self.run_once(&mut task, worker)
            };
            // The lane goes back before anything else: a peer 1:1 vCPU or sibling task may be
            // queued behind it. `lane_give_back` touches only the domain's lane lock, so it is safe
            // to call before taking this executor's.
            if let Some(d) = self.domain() {
                d.lane_give_back(&task.chain);
            }
            // #1469 — a root that returns while its vCPUs run publishes its outcome now and
            // retires; a retiring task finishes once they have all ended.
            let outcome = match outcome {
                Outcome::Finished if task.threads_live() => {
                    self.settle(&mut task);
                    Outcome::Retiring
                }
                Outcome::Retiring if !task.threads_live() => Outcome::Finished,
                o => o,
            };
            let mut task = Some(task);
            let mut g = lock(&self.state);
            match outcome {
                Outcome::Parked { deadline } => {
                    let shutdown = g.shutdown;
                    let e = g.tasks.get_mut(&id).expect("running task is filed");
                    if shutdown {
                        // Teardown: a park now would wait for a wake that can never come. Poison the
                        // task's cell so its wait returns and the trailing guard unwinds it — unless
                        // it parked after a freeze rang it, when its re-check ends its wait, or it
                        // waits on a child that freeze rang (`at_teardown`).
                        let t = task.as_ref().expect("held");
                        match t.at_teardown() {
                            AtTeardown::Poison => {
                                t.vm.trap
                                    .store(crate::DOMAIN_DONE_CODE as i64, Ordering::Relaxed);
                                if let Some(d) = &t.dom {
                                    d.wake_own_parked(); // its vCPUs share the cell
                                }
                                e.woken = true;
                            }
                            AtTeardown::Wake => e.woken = true,
                            AtTeardown::Wait => {}
                        }
                    }
                    e.task = task.take();
                    if e.woken {
                        g.runnable.push_back(id);
                    } else {
                        e.parked = true;
                        e.deadline = deadline;
                        // #1631 — a park that wakes on its own deadline is a potential notifier, so
                        // it stays out of the deadlock predicate's count. It is still `parked` for
                        // the cadence sweep, which is what fires that deadline.
                        if deadline.is_none() {
                            e.counted = true;
                            if let Some(d) = self.domain() {
                                d.task_parked();
                            }
                        }
                    }
                }
                Outcome::Retiring => {
                    let shutdown = g.shutdown;
                    let e = g.tasks.get_mut(&id).expect("running task is filed");
                    let t = task.take().expect("held");
                    if shutdown {
                        // Run teardown ends the domain its vCPUs outlived its root in, as the
                        // oracle's teardown sweep reaps them: poison the cell they share and wake
                        // the parked ones. Under a freeze of a durable child's window they are only
                        // woken, to unwind (#2010).
                        if t.at_teardown() == AtTeardown::Poison {
                            t.vm.trap
                                .store(crate::DOMAIN_DONE_CODE as i64, Ordering::Relaxed);
                        }
                        if let Some(d) = &t.dom {
                            d.wake_own_parked();
                        }
                    }
                    e.task = Some(t);
                    if e.woken {
                        g.runnable.push_back(id);
                    } else {
                        // Parked for the wakes and the cadence sweep, never counted: the task's
                        // root is no longer a vCPU (`settle` dropped it from `live`).
                        e.parked = true;
                    }
                }
                Outcome::Finished => {
                    g.tasks.remove(&id);
                    if g.tasks.is_empty() {
                        self.quiescent.notify_all();
                    }
                }
            }
            self.cv.notify_all();
            drop(g);
            if let Some(task) = task {
                self.finish(task);
            }
        }
    }

    /// One residency: claim the task, seed this thread's per-task state, resume its fiber under a
    /// guard bracket over its own fault range, classify the yield, reset the thread.
    fn run_once(&self, task: &mut ChildTask, worker: u64) -> Outcome {
        assert!(
            task.slot.claim_for_worker(worker),
            "child task dispatched while claimed"
        );
        let prev_rt = fiber_rt::set_current(&mut *task.rt as *mut FiberRuntime);
        // #1361 step 4 — a durable child spills into its own window's context-0 region; the register
        // is per OS thread, so seed it for this residency and give the worker's back after.
        let prev_shadow = task.done.durable.as_ref().map(|d| {
            let s = crate::durable_shadow::get();
            crate::durable_shadow::seed(d.shadow.region_base(0));
            s
        });
        let prev_tls = vcpu_tls::get();
        vcpu_tls::seed(task.tls);
        // #1469 — the child's thread thunks act in its own domain for this residency.
        let prev_dom = os_thread_rt::set_current_domain(
            task.dom.as_ref().map_or(std::ptr::null(), Arc::as_ptr),
        );
        task.rt.push_active(Arc::clone(&task.slot));
        let fib = task
            .slot
            .fiber_ptr()
            .expect("a claimed platform slot holds its fiber");
        let mut call = ResumeCall { fib, state: None };
        // SAFETY: `resume_shim` honours the `Entry` ABI; `call` outlives the bracket; the task's
        // window is live and its fault range is exactly this bracket's (R1). A fault abandons the
        // fiber, which is then dropped with the task, never resumed (R3).
        let faulted = unsafe {
            mem::run_guarded_range(
                resume_shim as *const () as *const u8,
                &mut call as *mut ResumeCall as *const i64,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null(),
                Arc::as_ptr(&task.vm),
                task.fault.0,
                task.fault.1,
            )
        };
        task.rt.pop_active();
        os_thread_rt::set_current_domain(prev_dom);
        task.tls = vcpu_tls::get();
        vcpu_tls::seed(prev_tls);
        if let Some(s) = prev_shadow {
            crate::durable_shadow::seed(s);
        }
        fiber_rt::set_current(prev_rt);
        if faulted {
            task.vm.trap.store(mem::FAULT_TRAP, Ordering::Relaxed);
            task.slot.finish_task();
            return Outcome::Finished;
        }
        match call.state {
            // A guest fiber's `suspend` yields to its resumer inside the task, and the child root's
            // traps (#1469), so the only yield reaching the worker is the task's event park.
            Some(State::Yielded(_)) if task.slot.took_event_park() => {
                // Read the park's own deadline before the slot goes back to the pool (#1631).
                let deadline = task.slot.took_park_deadline();
                task.slot.release_to_pool();
                Outcome::Parked { deadline }
            }
            Some(State::Yielded(_)) => {
                // Unreachable by construction; fail closed rather than resume an unknown yield.
                task.vm
                    .trap
                    .store(TrapKind::ThreadFault as i64, Ordering::Relaxed);
                task.slot.finish_task();
                Outcome::Finished
            }
            Some(State::Complete(_)) | None => {
                task.slot.finish_task();
                Outcome::Finished
            }
        }
    }

    /// After the last residency: free the window, run the teardown (powerbox release + lane
    /// give-back), publish the outcome, drop the domain's live count.
    fn finish(&self, mut task: Box<ChildTask>) {
        // #1469 — every vCPU the child spawned has ended (a retiring task waited for that): join
        // their OS threads before the window they ran on is freed.
        if let Some(d) = &task.dom {
            d.join_all();
        }
        let unwound = task.root_unwound();
        task.window.restore_rw();
        // #1361 step 4 — a durable child that unwound for a freeze leaves its window image for the
        // harvest (the parent's artifact carries it). Retire the base under the cell's lock first, so
        // no doorbell store can land on a window about to be freed.
        let mut captured = false;
        if let Some(d) = task.done.durable.as_ref() {
            *d.base.lock().unwrap_or_else(|e| e.into_inner()) = 0;
            if unwound {
                // #2031 — first flatten the root's parked fibers into their own regions, as a run's
                // root's are (`run_inner`): the image below carries their continuations. Its vCPUs
                // flattened the ones they found as they unwound.
                // SAFETY: the root has unwound and every vCPU has ended, so the table is at rest;
                // the runtime is armed over the live window, and the register names this window's
                // context-0 region while the fibers run, as in a residency.
                let mut fibers = unsafe {
                    let prev = crate::durable_shadow::get();
                    crate::durable_shadow::seed(d.shadow.region_base(0));
                    let rt = &mut *task.rt as *mut FiberRuntime;
                    fiber_rt::freeze_drive(rt, Arc::as_ptr(&task.vm) as u64);
                    crate::durable_shadow::seed(prev);
                    fiber_rt::take_frozen(rt)
                };
                // A fiber that traps as it flattens is the child's trap, as a run's root's is: its
                // outcome, not a capture.
                if task.vm.trap.load(Ordering::Relaxed) == 0 {
                    // #1854 — up to the child's high-water, as the root's capture reaches its own,
                    // so a page it grew through the Memory capability rides the artifact.
                    let mapped = task.window.rw_mut().len();
                    let reserved = 1usize << d.reserved_log2;
                    // SAFETY: the hook reads the child's own powerbox, alive until `teardown`.
                    let high = d.high_water.map_or(0, |(f, ctx)| unsafe {
                        f(ctx as *mut core::ffi::c_void, task.window.base() as usize) as usize
                    });
                    let image = if high > mapped {
                        task.window.read_low(high.min(reserved))
                    } else {
                        task.window.rw_mut().to_vec()
                    };
                    // #2010 — with what its vCPUs left as they ended: the ones that unwound, the
                    // ones that finished unjoined, and the fibers they flattened.
                    let mut vcpus = Vec::new();
                    if let Some(dom) = &task.dom {
                        vcpus = dom.take_frozen_vcpus();
                        vcpus.extend(dom.take_completed_children_residue());
                        fibers.extend(dom.take_frozen_fibers());
                    }
                    // #1684 — and every slot nobody flattened, so a thaw rebuilds the table slot
                    // for slot.
                    let flattened: Vec<usize> = fibers.iter().map(|f| f.slot).collect();
                    fibers.extend(task.rt.table().unflattened_for_freeze(&flattened));
                    // SAFETY: as `root_unwound`, which read the same word.
                    let root_sp = unsafe {
                        fiber_rt::read_shadow_sp(task.window.base() as u64, d.shadow.region_base(0))
                    };
                    *d.capture.lock().unwrap_or_else(|e| e.into_inner()) =
                        Some(crate::DetachedCapture {
                            image,
                            residue: crate::DurableResidue {
                                fibers,
                                vcpus,
                                nested: Vec::new(),
                                root_sp: Some(root_sp),
                            },
                        });
                    captured = true;
                }
            }
        }
        if let Some(c) = task.copy_back.take() {
            c(task.window.rw_mut());
        }
        if let Some(t) = task.teardown.take() {
            t(captured);
        }
        // A retiring task published at `settle`, unless its root unwound, whose capture this is.
        let trap = task.vm.trap.load(Ordering::Relaxed);
        let result = task.results.first().copied().unwrap_or(0);
        {
            let mut st = task.done.state.lock().unwrap_or_else(|e| e.into_inner());
            if st.is_none() {
                *st = Some((result, trap));
                task.done.cv.notify_all();
            }
        }
        let retiring = task.retiring;
        drop(task);
        // `settle` dropped a retiring task's live count when its root returned.
        if !retiring {
            if let Some(d) = self.domain() {
                d.child_finished();
            }
        }
    }

    /// #1469 — the root of a task whose vCPUs still run has returned: publish its outcome now, as
    /// the oracle does at the root's completion (a `join` of the child returns), and drop it from
    /// the run's live count. Its window and powerbox stay until the vCPUs end ([`Self::finish`]).
    /// A carve child's copy-back happens here, so the parent's join sees the child's writes; what
    /// the vCPUs write after it stays in the child's image. A durable root that unwound for a freeze
    /// has no outcome yet (#2010): its vCPUs are unwinding too, and the capture `finish` takes once
    /// they have ended is what its parent's join is owed.
    fn settle(&self, task: &mut ChildTask) {
        task.retiring = true;
        let trap = task.vm.trap.load(Ordering::Relaxed);
        let result = task.results.first().copied().unwrap_or(0);
        // A trap ends the child's domain: its parked vCPUs observe the shared cell and unwind.
        if trap != 0 {
            if let Some(d) = &task.dom {
                d.wake_own_parked();
            }
        }
        if let Some(c) = task.copy_back.take() {
            task.window.restore_rw();
            c(task.window.rw_mut());
        }
        if !task.root_unwound() {
            let mut st = task.done.state.lock().unwrap_or_else(|e| e.into_inner());
            *st = Some((result, trap));
            task.done.cv.notify_all();
        }
        if let Some(d) = self.domain() {
            d.child_finished();
        }
    }

    /// Run teardown (`Nursery::join_children`): poison every parked task so it unwinds, let the
    /// runnable ones finish, wait for quiescence, join the workers. One parked forever would have
    /// hung the join, and unwinds through its trailing guard instead — the interpreter's teardown
    /// sweep. A task that parks after this began is poisoned too (`worker_loop`).
    ///
    /// `end_domain` — the parent's domain is ending (not freezing): its **carve** children end with
    /// it, running or not yet started, as the oracle ends them. Their cell gets the completion
    /// sentinel, which their entry and back-edge polls observe (`emit_domain_poll`). A
    /// freeze skips this, so a child the freeze reaches still unwinds under its own freeze word.
    /// For the same reason a **durable** child under a word the freeze rang
    /// (`Nursery::ring_detached`) is never poisoned, here or when it parks later, and unwinds and
    /// rides, where a poisoned one would end in a trap and leave no image: its futex wait is woken
    /// (#1937), and its join is left for the joined child's end to re-offer (#2010,
    /// [`ChildTask::at_teardown`]).
    pub(crate) fn shutdown_and_join(self: &Arc<Self>, end_domain: bool) {
        let workers = {
            let mut g = lock(&self.state);
            g.shutdown = true;
            for e in g.tasks.values_mut() {
                let spared = e
                    .task
                    .as_ref()
                    .is_some_and(|t| t.at_teardown() != AtTeardown::Poison);
                let poisoned = if e.parked && !spared {
                    if let Some(t) = &e.task {
                        t.vm.trap
                            .store(crate::DOMAIN_DONE_CODE as i64, Ordering::Relaxed);
                    }
                    true
                } else if let Some(stop) = e.stop.as_ref().filter(|_| end_domain) {
                    // Never clobber a trap the child already recorded.
                    let _ = stop.trap.compare_exchange(
                        0,
                        crate::DOMAIN_DONE_CODE as i64,
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    );
                    true
                } else {
                    false
                };
                // #1469 — the child's own parked vCPUs share the poisoned cell; a spared child's
                // re-check the word the freeze rang (#2010).
                if poisoned || spared {
                    if let Some(d) = &e.dom {
                        d.wake_own_parked();
                    }
                }
            }
            self.wake_all_parked(&mut g);
            // No worker could be spawned for a filed task (thread refusal): drive it here.
            if g.workers.is_empty() && !g.tasks.is_empty() {
                drop(g);
                self.clone().worker_loop(0);
                g = lock(&self.state);
            }
            while !g.tasks.is_empty() {
                g = self
                    .quiescent
                    .wait_timeout(g, RECHECK)
                    .unwrap_or_else(|e| e.into_inner())
                    .0;
            }
            self.cv.notify_all();
            std::mem::take(&mut g.workers)
        };
        for w in workers {
            let _ = w.join();
        }
    }
}

enum Outcome {
    /// Event-parked, until `deadline` if the park carries one: #1631's question (does it come
    /// back on its own, so it should not count toward `Domain::parked`?) and #2012's (when?).
    Parked {
        deadline: Option<Instant>,
    },
    /// #1469 — the root has returned; vCPUs it spawned still run ([`ChildTask::retiring`]).
    Retiring,
    Finished,
}
