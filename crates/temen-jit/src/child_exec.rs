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
//! domain's *own* `thread.spawn` vCPUs stay 1:1 OS threads, counted by `max_vcpus`, not by lanes.
//!
//! **Env-swap rules** (the D66 checklist R1–R5): each resume is its own `run_guarded_range` bracket
//! over the *task's* fault range (R1); the per-thread state the child reads through TLS — its fiber
//! runtime (`CURRENT_RT`), `vcpu.tls`, the current-task word — is seeded on entry and saved/reset on
//! exit of every resume, and every reader on a fiber stack is `#[inline(never)]` (R2, #1466); a
//! finished or faulted task is never resumed (R3: the slot's `finish` closes the claim); a trap is
//! attributed to the task's own cell, never the run's (R4); the window base is baked at first entry
//! and the window moves with the task (R5 — `GuestWindow: Send`).

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Condvar, Mutex, Weak};
use std::time::{Duration, Instant};

use temen_fiber::{Fiber, State};
use temen_ir::lanes;

use crate::fiber_rt::{self, FiberRuntime, FiberSlot, SharedFiberTable};
use crate::instantiator_rt::ChildDone;
use crate::os_thread_rt::Domain;
use crate::{mem, vcpu_tls, CompiledModule, TrapKind};

/// A task's native control stack: the fiber arena's slot size (`temen-fiber` hands out fixed
/// 256 KiB slots), the same stack every guest fiber runs on. Less headroom than the 2 MiB OS thread
/// a detached child used to get — but bounded by the guest's own prologue checks now (the fiber's
/// low bound rides in as the child's `stack_limit`), so a child that recurses past it traps
/// `StackOverflow` cleanly instead of relying on the OS guard.
const TASK_STACK: usize = 1 << 18;

/// How often idle workers re-check the §5 kill cell and any parked task's deadline when no wake
/// arrives — the same cadence a parked 1:1 vCPU uses (`os_thread_rt::KILL_RECHECK`).
const RECHECK: Duration = Duration::from_millis(20);

/// The limit-taking buffer-ABI trampoline a task's fiber body enters the child through
/// (`build_trampoline(.., with_limit = true)`): the five `Entry` params plus the running stack's
/// low bound, which the fiber alone knows.
type LimitedEntry =
    extern "C" fn(*const i64, *mut i64, *mut u8, *const core::ffi::c_void, *mut i64, u64);

/// A **carve** child's copy-back: the parent is the superset, so the child's window image is
/// written back into its carve at finish. `None` for a detached child (its window is its own).
pub(crate) type CopyBack = Box<dyn FnOnce(&[u8]) + Send>;

/// A task's one-shot teardown: release the child powerbox, return its lane.
pub(crate) type Teardown = Box<dyn FnOnce() + Send>;

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
    /// The task's own trap cell (R4). Heap-stable: baked into the child's frames at first entry.
    trap: Box<AtomicI64>,
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
}

// SAFETY: a task moves between workers only through the executor's queue, and is touched only by
// the worker holding its slot claim (`FiberSlot::claim_for_worker`) — one thread at a time, with
// the claim's acquire/release pairing carrying every field across the hand-off. The runtime's raw
// yielder pointers are live only during a residency (pushed and popped on the running worker).
unsafe impl Send for ChildTask {}

impl ChildTask {
    /// Build a task around a compiled detached child. `init` seeds the fresh window (data segments,
    /// payload); `premap` aliases the op-15 pre-mapped region onto it (`false` ⇒ the child never
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
        init: impl FnOnce(&mut [u8]),
        premap: impl FnOnce(*mut u8, u64, u64) -> bool,
        args: Vec<i64>,
        n_results: usize,
        chain: Vec<(usize, i64)>,
        done: Arc<ChildDone>,
        copy_back: Option<CopyBack>,
        teardown: Teardown,
    ) -> Result<ChildTask, Teardown> {
        let mut window = mem::GuestWindow::new(1usize << mapped_log2, 1usize << reserved_log2);
        let base = window.base();
        init(window.rw_mut());
        if !premap(base, 1u64 << mapped_log2, 1u64 << reserved_log2) {
            window.restore_rw();
            return Err(teardown);
        }
        let fault = window.fault_range();
        let trap = Box::new(AtomicI64::new(0));
        let results: Box<[i64]> = vec![0i64; n_results.max(1)].into_boxed_slice();
        let args: Box<[i64]> = args.into_boxed_slice();
        // The child's own execution context over a private, unused fiber table: the child compiles
        // with no `cont.*` env, so the table holds nothing; the runtime exists for its `yielders` /
        // `active_slots` bookkeeping, which is what `fiber_event_park` and `current_fiber_slot` read.
        let table = Arc::new(SharedFiberTable::new(
            1,
            temen_ir::durable_abi::ShadowArena::EMPTY,
        ));
        let rt = Box::new(FiberRuntime::new(table, 0, code.fn_table_mask));
        let tramp: LimitedEntry = core::mem::transmute(code.tramp_code_limited);
        let a = SendRaw(args.as_ptr());
        let r = SendRaw(results.as_ptr() as *mut i64);
        let b = SendRaw(base);
        let t = SendRaw(code.fn_table.as_ptr() as *const core::ffi::c_void);
        let c = SendRaw(&*trap as *const AtomicI64 as *mut i64);
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
        Ok(ChildTask {
            slot: FiberSlot::platform(fiber),
            rt,
            window,
            fault,
            trap,
            results,
            _args: args,
            _code: code,
            chain,
            tls: 0,
            done,
            copy_back,
            teardown: Some(teardown),
        })
    }
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

/// A parked task's wake handle, attached to the futex cell it parks on ([`current_waker`]).
#[derive(Clone)]
pub(crate) struct TaskWaker {
    exec: Weak<ChildExec>,
    id: u64,
}

impl TaskWaker {
    /// Mark the task woken: runnable now if it is parked, or remembered so its in-flight park
    /// lands runnable (the executor's no-lost-wake discipline, see [`ChildExec::wake`]).
    pub(crate) fn fire(&self) {
        if let Some(e) = self.exec.upgrade() {
            e.wake(self.id);
        }
    }
}

thread_local! {
    /// The `(executor, task)` this worker is running on behalf of, or `(null, 0)`. Seeded around
    /// every resume so a park site inside the child can find its waker without being told.
    static CURRENT: std::cell::Cell<(usize, u64)> = const { std::cell::Cell::new((0, 0)) };
}

/// The waker for the task running on **this** thread, if any (`None` on the root, a 1:1 vCPU, or a
/// worker between tasks). `#[inline(never)]`: read on a fiber stack after a possible migration
/// (#1466).
#[inline(never)]
pub(crate) fn current_waker() -> Option<TaskWaker> {
    let (exec, id) = CURRENT.with(|c| c.get());
    if exec == 0 {
        return None;
    }
    // SAFETY: a nonzero `exec` is the `Arc<ChildExec>` the running worker holds (seeded around this
    // resume), so a weak ref minted from it is sound.
    let arc = unsafe { Arc::from_raw(exec as *const ChildExec) };
    let weak = Arc::downgrade(&arc);
    std::mem::forget(arc);
    Some(TaskWaker { exec: weak, id })
}

/// Register (`Some`) or clear (`None`) the running task's park deadline — a timed futex wait's
/// firing point, since no resumer polls a task: an idle worker re-dispatches it once due.
#[inline(never)]
pub(crate) fn set_park_deadline(deadline: Option<Instant>) {
    if let Some(w) = current_waker() {
        if let Some(e) = w.exec.upgrade() {
            let mut g = lock(&e.state);
            if let Some(t) = g.tasks.get_mut(&w.id) {
                t.deadline = deadline;
            }
        }
    }
}

/// Per-task executor bookkeeping. `task` is `None` while a worker holds the task (running).
struct Entry {
    task: Option<Box<ChildTask>>,
    /// Parked (yielded on an event park) and not yet woken — counted in `Domain::parked`.
    parked: bool,
    /// A wake arrived (possibly while the task was still running toward its park).
    woken: bool,
    /// A timed wait's deadline; the idle loop turns a due one into a wake.
    deadline: Option<Instant>,
}

struct ExecState {
    tasks: HashMap<u64, Entry>,
    runnable: VecDeque<u64>,
    /// D66 — per-domain running counts, the lane gate's state (`temen_ir::lanes`).
    lane_running: BTreeMap<usize, usize>,
    /// Runnable tasks refused a lane; re-offered whenever a lane is released.
    lane_waiters: VecDeque<u64>,
    next_id: u64,
    workers: Vec<std::thread::JoinHandle<()>>,
    idle_workers: usize,
    /// Run teardown began: a park now poisons the task (it unwinds through its trailing guard),
    /// and workers exit once no task remains.
    shutdown: bool,
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
    /// The §5 kill cell (`0` = unarmed): idle workers re-check it so a parked task unwinds on a kill.
    epoch_addr: usize,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// True iff the kill-path is armed and fired (mirror of `os_thread_rt::epoch_fired`).
fn epoch_fired(addr: usize) -> bool {
    // SAFETY: a non-zero `addr` is the run's live interrupt cell (an `AtomicU64`), outliving the run.
    addr != 0
        && unsafe { (*(addr as *const std::sync::atomic::AtomicU64)).load(Ordering::Relaxed) != 0 }
}

impl ChildExec {
    pub(crate) fn new(dom: usize, epoch_addr: usize) -> Arc<ChildExec> {
        Arc::new(ChildExec {
            state: Mutex::new(ExecState {
                tasks: HashMap::new(),
                runnable: VecDeque::new(),
                lane_running: BTreeMap::new(),
                lane_waiters: VecDeque::new(),
                next_id: 1,
                workers: Vec::new(),
                idle_workers: 0,
                shutdown: false,
            }),
            cv: Condvar::new(),
            quiescent: Condvar::new(),
            dom,
            epoch_addr,
        })
    }

    fn domain(&self) -> Option<&Domain> {
        // SAFETY: a nonzero `dom` is the run's live `Domain`, which outlives the nursery (and so
        // this executor and every task).
        (self.dom != 0).then(|| unsafe { &*(self.dom as *const Domain) })
    }

    /// File a task and make it runnable. Its `done` cell is filled when it finishes.
    pub(crate) fn spawn(self: &Arc<Self>, task: ChildTask) {
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
        g.tasks.insert(
            id,
            Entry {
                task: Some(Box::new(task)),
                parked: false,
                woken: false,
                deadline: None,
            },
        );
        g.runnable.push_back(id);
        self.ensure_worker(&mut g, widest);
        self.cv.notify_one();
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

    /// Wake task `id` (see [`TaskWaker::fire`]).
    pub(crate) fn wake(&self, id: u64) {
        let mut g = lock(&self.state);
        let Some(e) = g.tasks.get_mut(&id) else {
            return;
        };
        e.woken = true;
        if e.parked {
            e.parked = false;
            e.deadline = None;
            if let Some(d) = self.domain() {
                d.task_unparked();
            }
            g.runnable.push_back(id);
            self.cv.notify_one();
        }
    }

    /// Wake every parked task so it re-checks its park predicate (kill / teardown).
    fn wake_all_parked(&self, g: &mut ExecState) {
        let ids: Vec<u64> = g
            .tasks
            .iter()
            .filter(|(_, e)| e.parked)
            .map(|(&id, _)| id)
            .collect();
        for id in ids {
            let e = g.tasks.get_mut(&id).expect("listed");
            e.parked = false;
            e.woken = true;
            e.deadline = None;
            if let Some(d) = self.domain() {
                d.task_unparked();
            }
            g.runnable.push_back(id);
        }
        self.cv.notify_all();
    }

    /// Fire the deadlines that have passed (a timed wait's firing point).
    fn fire_due(&self, g: &mut ExecState, now: Instant) {
        let due: Vec<u64> = g
            .tasks
            .iter()
            .filter(|(_, e)| e.parked && e.deadline.is_some_and(|d| d <= now))
            .map(|(&id, _)| id)
            .collect();
        for id in due {
            let e = g.tasks.get_mut(&id).expect("listed");
            e.parked = false;
            e.woken = true;
            e.deadline = None;
            if let Some(d) = self.domain() {
                d.task_unparked();
            }
            g.runnable.push_back(id);
        }
    }

    /// Pop the next runnable task that can take its lanes now; a refused one waits on the lane.
    fn pick(&self, g: &mut ExecState) -> Option<u64> {
        while let Some(id) = g.runnable.pop_front() {
            let Some(e) = g.tasks.get_mut(&id) else {
                continue;
            };
            let Some(t) = e.task.as_ref() else {
                continue; // already held by a worker (a double wake): its re-insert requeues it
            };
            if lanes::enter(&mut g.lane_running, &t.chain) {
                e.woken = false;
                return Some(id);
            }
            g.lane_waiters.push_back(id);
        }
        None
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
                    let now = Instant::now();
                    let next = g
                        .tasks
                        .values()
                        .filter(|e| e.parked)
                        .filter_map(|e| e.deadline)
                        .min()
                        .map(|d| d.saturating_duration_since(now));
                    let armed = self.epoch_addr != 0;
                    g = match (next, armed) {
                        (None, false) => self.cv.wait(g).unwrap_or_else(|e| e.into_inner()),
                        (next, _) => {
                            let to = next.unwrap_or(RECHECK).min(RECHECK);
                            self.cv
                                .wait_timeout(g, to)
                                .unwrap_or_else(|e| e.into_inner())
                                .0
                        }
                    };
                    g.idle_workers -= 1;
                    self.fire_due(&mut g, Instant::now());
                    if epoch_fired(self.epoch_addr) {
                        self.wake_all_parked(&mut g);
                    }
                };
                let task = g
                    .tasks
                    .get_mut(&id)
                    .and_then(|e| e.task.take())
                    .expect("picked task is present");
                (id, task)
            };
            let outcome = self.run_once(id, &mut task, worker);
            let mut task = Some(task);
            let mut g = lock(&self.state);
            lanes::leave(&mut g.lane_running, &task.as_ref().expect("held").chain);
            // A released lane re-offers every lane waiter (the cheap wake-all; a refused one
            // simply re-queues).
            let waiters: Vec<u64> = g.lane_waiters.drain(..).collect();
            g.runnable.extend(waiters);
            match outcome {
                Outcome::Parked => {
                    let shutdown = g.shutdown;
                    let e = g.tasks.get_mut(&id).expect("running task is filed");
                    if shutdown {
                        // Teardown: a park now would wait for a wake that can never come. Poison the
                        // task's cell so its wait returns and the trailing guard unwinds it.
                        let t = task.as_ref().expect("held");
                        t.trap
                            .store(crate::DOMAIN_DONE_CODE as i64, Ordering::Relaxed);
                        e.woken = true;
                    }
                    e.task = task.take();
                    if e.woken {
                        g.runnable.push_back(id);
                    } else {
                        e.parked = true;
                        if let Some(d) = self.domain() {
                            d.task_parked();
                        }
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
    fn run_once(self: &Arc<Self>, id: u64, task: &mut ChildTask, worker: u64) -> Outcome {
        assert!(
            task.slot.claim_for_worker(worker),
            "child task dispatched while claimed"
        );
        let prev_rt = fiber_rt::set_current(&mut *task.rt as *mut FiberRuntime);
        let prev_tls = vcpu_tls::get();
        vcpu_tls::seed(task.tls);
        CURRENT.with(|c| c.set((Arc::as_ptr(self) as usize, id)));
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
                &*task.trap as *const AtomicI64 as *mut i64,
                task.fault.0,
                task.fault.1,
            )
        };
        task.rt.pop_active();
        CURRENT.with(|c| c.set((0, 0)));
        task.tls = vcpu_tls::get();
        vcpu_tls::seed(prev_tls);
        fiber_rt::set_current(prev_rt);
        if faulted {
            task.trap.store(mem::FAULT_TRAP, Ordering::Relaxed);
            task.slot.finish_task();
            return Outcome::Finished;
        }
        match call.state {
            // The child compiles with no `cont.*` env, so the only yield a task can make is the
            // futex thunk's event park.
            Some(State::Yielded(_)) if task.slot.took_event_park() => {
                task.slot.release_to_pool();
                Outcome::Parked
            }
            Some(State::Yielded(_)) => {
                // Unreachable by construction; fail closed rather than resume an unknown yield.
                task.trap
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
        let trap = task.trap.load(Ordering::Relaxed);
        let result = task.results.first().copied().unwrap_or(0);
        task.window.restore_rw();
        if let Some(c) = task.copy_back.take() {
            c(task.window.rw_mut());
        }
        if let Some(t) = task.teardown.take() {
            t();
        }
        {
            let mut st = task.done.state.lock().unwrap_or_else(|e| e.into_inner());
            *st = Some((result, trap));
            task.done.cv.notify_all();
        }
        drop(task);
        if let Some(d) = self.domain() {
            d.child_finished();
        }
    }

    /// Run teardown (`Nursery::join_children`): poison every parked task so it unwinds, let the
    /// runnable ones finish, wait for quiescence, join the workers. A task that never started runs
    /// to its end as before (the OS-thread path ran it too); one parked forever would have hung the
    /// join, and now unwinds through its trailing guard instead — the interpreter's teardown sweep.
    pub(crate) fn shutdown_and_join(self: &Arc<Self>) {
        let workers = {
            let mut g = lock(&self.state);
            g.shutdown = true;
            for (_, e) in g.tasks.iter_mut() {
                if e.parked {
                    if let Some(t) = &e.task {
                        t.trap
                            .store(crate::DOMAIN_DONE_CODE as i64, Ordering::Relaxed);
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
    Parked,
    Finished,
}
