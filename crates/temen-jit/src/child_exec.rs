//! **Child-domain executor** (DESIGN.md §23, D66) — every non-durable §14 child (a detached child of
//! Instantiator op 17) runs as a **task on a bounded
//! pool of platform workers**, not one OS thread each. A task is a platform-owned fiber ([`FiberSlot::platform`]) carrying the child's own window,
//! trap cell and fiber execution context; any worker may resume it (migration under the same
//! single-owner claim guest fibers use), and a task that parks — a futex `wait` inside the child —
//! hands its worker back instead of holding an OS thread, so a lane-bounded parent can run N
//! children on L lanes without a parked child wedging a runnable one.
//!
//! **Parallelism is a granted resource, bounded at dispatch** (INVARIANTS #3, D66): every resume
//! first takes one slot in each bounded lane along the task's chain (`temen_ir::lanes` — the same
//! arithmetic the interpreter's two drivers gate on), and gives them back when the task parks or
//! finishes. A task its quantum preempted keeps them, as on the interpreter (#2237). A child of a
//! cap-1 parent therefore never overlaps a sibling, whatever the worker count.
//!
//! **The pool is the driver's size, and preempts** (#1991), as the interpreter's is. Workers are
//! spawned as work waits with none free: up to the widest lane a task could use, and never past the
//! workers the host's driver has (`GrantChildHooks::workers`, by default the cores). A task's quantum
//! ends where a chunk of its fuel runs out: every task counts, metered or not. At each refill
//! ([`quantum_end`]) a task queued behind it gets a worker: a free one, a new one, or, once the pool
//! is full, this one, the running task parking as runnable at the back of the queue. So the cap
//! stays invisible to the guest: a task that never parks still gives way, and a parked task's
//! deadline, or the work queued behind a running one, comes due within a quantum.
//!
//! A domain's *own* `thread.spawn` vCPUs — a child task's included (#1469) — stay 1:1 OS threads,
//! counted against `MAX_VCPUS` and their domain's `Budget.spawn`; they are not tasks, and the OS
//! preempts them.
//!
//! **Env-swap rules** (the D66 checklist R1–R5): each resume is its own `run_guarded_range` bracket
//! over the *task's* fault range (R1); the per-thread state the child reads through TLS — its fiber
//! runtime (`CURRENT_RT`), its thread domain (`CURRENT_DOMAIN`), `vcpu.tls`, the in-task flag — is
//! seeded on entry and saved/reset on exit of every resume, and every reader on a fiber stack is
//! `#[inline(never)]` (R2, #1466); a finished or faulted task is never resumed (R3: the slot's
//! `finish` closes the claim); a trap is attributed to the task's own cell, never the run's (R4);
//! the window base is baked at first entry and the window moves with the task (R5 —
//! `GuestWindow: Send`).

use std::cell::Cell;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

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

/// #2722 — process-wide count of child-executor residencies ([`ChildExec::run_once`]): each time a
/// pool worker resumes a task's fiber. A parked task advances it only when something genuinely
/// re-offers it (a wake or its deadline), never on an idle cadence — the observable
/// [`crate::pool_resumes`] exposes so a test can prove the idle pool does not spin.
pub(crate) static POOL_RESUMES: AtomicU64 = AtomicU64::new(0);

/// #2757 — a waker registered on what a park waits for: firing it re-offers the one task waiting
/// there ([`Wake::Task`]). It holds the executor weakly, so a store that outlives the run wakes
/// nothing.
pub(crate) type TaskWaker = Arc<dyn Fn() + Send + Sync>;

thread_local! {
    /// #1991 — the executor whose task this worker is running, for the task's quantum end
    /// ([`quantum_end`]): the worker loop's own reference, published for one residency. Null
    /// outside one, as on a run's root and on a 1:1 vCPU.
    static RUNNING_FOR: Cell<*const Arc<ChildExec>> = const { Cell::new(std::ptr::null()) };
    /// #2695 — the pool's name for the task whose fiber this thread is running, innermost: a worker's
    /// task, or a held domain's activation its carrier resumes. `0` outside one. A call that task
    /// makes into a held domain is carried by it ([`Carrier`]).
    static RUNNING_TASK: Cell<u64> = const { Cell::new(0) };
}

/// The limit-taking buffer-ABI trampoline a task's fiber body enters the child through
/// (`build_trampoline(.., with_limit = true)`): the five `Entry` params plus the running stack's
/// low bound, which the fiber alone knows.
type LimitedEntry =
    extern "C" fn(*const i64, *mut i64, *mut u8, *const core::ffi::c_void, *mut i64, u64);

/// #1936 — a task's one-shot end: its domain ended, and no freeze captured it ([`crate::ChildEnded`]).
/// It answers the signal of the domain's end post, if it has one, for the task to give once its end
/// is published (#2800).
pub(crate) type Ended = Box<dyn FnOnce() -> Option<EndSignal> + Send>;

/// #2800 — the signal of a domain's end post ([`crate::ChildEndSignal`]): given once its end is
/// published, so a waiter it wakes finds the end, and before the task leaves the run's live count.
pub(crate) type EndSignal = Box<dyn FnOnce() + Send>;

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
    /// The child's own window (seeded by `init`). Moves with the task (`GuestWindow: Send`).
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
    /// Saved serve-handler count between residencies (R2), so a handler that parks resumes nested
    /// on whichever worker picks it up ([`vcpu_tls::serve_handlers`]).
    serve_handlers: u32,
    done: Arc<ChildDone>,
    /// Runs exactly once, after the last resume, while the window is still alive: releases the
    /// child powerbox and returns the child's lane to the parent.
    teardown: Option<Teardown>,
    /// #1936 — told once its domain ends ([`Self::end`]).
    ended: Option<Ended>,
    /// #1469 — the domain the child's own `thread.spawn` vCPUs run in (their handle table, window,
    /// fiber registry and lane chain), sharing the run's futex, park count, lanes and live count.
    /// Built when the task is filed ([`ChildExec::spawn`]), for a child compiled with thread ops.
    dom: Option<Arc<Domain>>,
    /// #2001 — a detached child's own budget node, handed to [`Self::dom`] for its threads' charges.
    node: Option<Arc<dyn crate::BudgetNode>>,
    /// #2043 — the child domain's child table, which its nursery and [`Self::dom`] share; its
    /// counters ride the child's capture.
    children: crate::os_thread_rt::ChildTable,
    /// #1991 — its quantum ended with another task waiting, and it keeps the lanes it ran in until it
    /// parks or finishes, as the interpreter's preempted vCPU does (#2237): its next resume takes
    /// none.
    holds_lanes: bool,
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
    /// #1936 — tell the task's powerbox, through `ended`, when its domain ends ([`Self::end`]).
    pub(crate) fn on_end(mut self, ended: Ended) -> ChildTask {
        self.ended = Some(ended);
        self
    }

    /// What the task's parent learns of its end: its root's first result, and the trap that ended its
    /// domain (`0` for none). `DOMAIN_DONE` is no trap: it is the end a returned domain's last handlers
    /// unwind on ([`ChildExec::file`]), or the run's teardown.
    fn outcome(&self) -> (i64, i64) {
        let trap = match self.vm.ended() {
            Some(code) if code != crate::DOMAIN_DONE_CODE as i64 => code,
            _ => 0,
        };
        (self.results.first().copied().unwrap_or(0), trap)
    }

    /// #1936 — the task's domain has ended, and no freeze captured it: tell its powerbox, once, while
    /// it is live. Answers its end post's signal, for the caller to give once the end is published
    /// and before the task leaves the run's live count (#2800).
    fn end(&mut self) -> Option<EndSignal> {
        self.ended.take().and_then(|ended| ended())
    }

    /// Build a task around a compiled detached child. `init` seeds the fresh window (data segments,
    /// payload) and may protect its pages (a detached child's readonly segments, a thawed one's page
    /// map), `false` when the OS refuses one; `premap` aliases the spawn's pre-mapped region onto it
    /// (`false` ⇒ the child never runs: a `CapFault` outcome, as on the OS-thread path it replaces).
    /// `Err` hands `teardown` back when the child never runs: its window, its seeding or its pre-map
    /// refused.
    ///
    /// # Safety
    /// `code` was compiled by `compile_child_windowed` for exactly `(mapped_log2, reserved_log2)`
    /// (so it carries a limit-taking trampoline); `args` matches the entry's arity.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn new(
        code: Arc<CompiledModule>,
        mapped_log2: u8,
        reserved_log2: u8,
        init: impl FnOnce(&mut mem::GuestWindow) -> bool,
        premap: impl FnOnce(*mut u8, u64, u64) -> bool,
        args: Vec<i64>,
        n_results: usize,
        chain: Vec<(usize, i64)>,
        done: Arc<ChildDone>,
        teardown: Teardown,
        node: Option<Arc<dyn crate::BudgetNode>>,
        children: crate::os_thread_rt::ChildTable,
    ) -> Result<ChildTask, Teardown> {
        // #2305/#2286 — a window the OS will not give refuses the child as a refused pre-map does:
        // no room for its reservation, or a commit or protection its seeding needs. It never runs,
        // and its spawn answers what a refused filing answers.
        let Some(mut window) =
            mem::GuestWindow::try_new(1usize << mapped_log2, 1usize << reserved_log2)
        else {
            return Err(teardown);
        };
        let base = window.base();
        // #964/#1733: every child window reserves its own `[0, guard)`, as the interpreter's windows
        // do — last, after the host's seeding; a window smaller than the guard skips, as the
        // interpreter's `seed_null_guard` does. `finish` re-asserts RW before the host reads it back.
        let guard = temen_ir::module_null_guard();
        let seeded =
            init(&mut window) && (guard > 1u64 << mapped_log2 || window.seed_null_guard(guard));
        if !seeded || !premap(base, 1u64 << mapped_log2, 1u64 << reserved_log2) {
            return Err(teardown);
        }
        let fault = window.fault_range();
        // The child runs as its own instance, the one its compile recorded (#1768).
        let vm = Arc::new(VmCtx::new(code.instance, code.window_at(base)));
        let results: Box<[i64]> = vec![0i64; n_results.max(1)].into_boxed_slice();
        let args: Box<[i64]> = args.into_boxed_slice();
        // The child domain's own execution context and fiber table (#1469): its `cont.*` handles
        // live here, numbered from 0 and out of every other domain's reach, as the oracle's
        // per-domain registry. A child compiled without `cont.*` gets an unused one-slot table; the
        // runtime still carries the `yielders` / `active_slots` bookkeeping that
        // `fiber_event_park` and `current_fiber_slot` read. A durable child's table knows its shadow
        // arena, where its `thread.spawn` vCPUs reserve their contexts (#2010), and a detached
        // child's its node, which its fibers are charged to (#2112).
        let shadow = done
            .durable
            .as_ref()
            .map_or(temen_ir::durable_abi::ShadowArena::EMPTY, |d| d.shadow);
        let mut rt = match (code.fiber_cfg, code.call_tramp) {
            (Some((type_id, mask)), Some(t)) => {
                let table = Arc::new(SharedFiberTable::new(
                    temen_ir::MAX_FIBERS,
                    shadow,
                    node.clone(),
                ));
                let mut rt = Box::new(FiberRuntime::new(table, type_id, mask));
                rt.set_call_tramp(t);
                rt
            }
            _ => {
                let table = Arc::new(SharedFiberTable::new(1, shadow, None));
                Box::new(FiberRuntime::new(table, 0, code.fn_table_mask))
            }
        };
        let tramp: LimitedEntry = core::mem::transmute(code.tramp_code_limited);
        let a = SendRaw(args.as_ptr());
        let r = SendRaw(results.as_ptr() as *mut i64);
        let b = SendRaw(base);
        let t = SendRaw(code.fn_table.base());
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
            serve_handlers: 0,
            done,
            teardown: Some(teardown),
            ended: None,
            dom: None,
            node,
            children,
            holds_lanes: false,
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
        // SAFETY: the task's window is live while the task is; its freeze word is in its arena.
        let rung = self.done.durable.as_ref().is_some_and(|d| unsafe {
            fiber_rt::window_is_unwinding(self.window.base() as u64, d.shadow)
        });
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
        self.vm.ended().is_none()
            // SAFETY: the window is live while the task is; its arena holds the freeze word and, in
            // context 0's region, the root's shadow-SP word.
            && unsafe { fiber_rt::window_is_unwinding(base, d.shadow) }
            && unsafe { fiber_rt::read_shadow_sp(base, d.shadow.region_base(0)) }
                > d.shadow.frame_base(0)
    }

    /// Open the window for the host's read of it: the guest may have closed pages through the Memory
    /// capability (`GuestWindow::restore_rw`). #2286 — when the OS refuses, the process cannot hold
    /// the child's window, so the child ends refused ([`Self::end_refused`]) and nothing reads the
    /// window: a capture needs a child that did not trap, and [`Self::root_unwound`] reads the trap
    /// first.
    fn open_for_host(&mut self) {
        if !self.window.restore_rw() {
            self.end_refused();
        }
    }

    /// The child ends with `ThreadFault`, unless it trapped already: the host could not read its
    /// window (#2286, #2318).
    fn end_refused(&self) {
        self.vm.raise(TrapKind::ThreadFault as i64);
    }

    /// #1361 step 4 — what a durable child whose root unwound for a freeze leaves for its parent's
    /// artifact: its window, with the `fibers` its root flattened and what its vCPUs left. `None`
    /// when the OS will not commit its grown tail for the read (#2318).
    fn capture(
        &mut self,
        d: &crate::instantiator_rt::DurableCell,
        mut fibers: Vec<crate::FrozenFiber>,
    ) -> Option<crate::DetachedCapture> {
        // #1854, #2427 — its declared window and each page it holds past that, as the root's capture
        // reads its own, so a page it grew through the Memory capability rides the artifact.
        let image = self.window.rw_mut().to_vec();
        let tail = match d.held_tail {
            // SAFETY: the hook reads the child's own powerbox, alive until `teardown`.
            Some((f, ctx)) => self.window.read_pages(&unsafe {
                f(
                    ctx as *mut core::ffi::c_void,
                    self.window.base() as usize,
                    image.len() as u64,
                )
            })?,
            None => Vec::new(),
        };
        // #2010 — with what its vCPUs left as they ended: the ones that unwound, the ones that
        // finished unjoined, and the fibers they flattened.
        let mut vcpus = Vec::new();
        if let Some(dom) = &self.dom {
            vcpus = dom.take_frozen_vcpus();
            vcpus.extend(dom.take_completed_children_residue());
            fibers.extend(dom.take_frozen_fibers());
        }
        // #1684 — and every slot nobody flattened, so a thaw rebuilds the table slot for slot.
        let flattened: Vec<usize> = fibers.iter().map(|f| f.slot).collect();
        fibers.extend(self.rt.table().unflattened_for_freeze(&flattened));
        // SAFETY: as `root_unwound`, which read the same word.
        let root_sp =
            unsafe { fiber_rt::read_shadow_sp(self.window.base() as u64, d.shadow.region_base(0)) };
        Some(crate::DetachedCapture {
            image,
            tail,
            residue: crate::DurableResidue {
                fibers,
                vcpus,
                root_sp: Some(root_sp),
                children: self.children.generations(),
            },
        })
    }
}

/// #2742 — a serving vCPU's **window** onto the pool: what its handlers share with it (DESIGN.md
/// §12a: every cross-domain call mints a handler fiber in the callee's world, and who runs it is a
/// scheduling decision). A handler runs only while its window is open, its vCPU inside a serve op,
/// and one at a time under the window's gate, which the serve op holds whenever it is not parked:
/// on the oracle the handlers are fibers of the serving vCPU, run by its serve op. The pool counts
/// them in the deadlock predicate only while the window is open ([`WindowCount`]).
pub(crate) struct Window {
    /// The vCPU's `vcpu.tls` word while its serve op runs, which its handlers read and write as the
    /// vCPU (R2: the word goes with execution).
    tls: std::sync::atomic::AtomicI64,
    /// Rings the serving domain's bell, once a handler of the window has ended (for whoever waits
    /// on its dispatch) or given its gate back on the pool (for a serve op waiting to take it).
    ring: Box<dyn Fn() + Send + Sync>,
    /// Dispatches its handlers served, wherever they ended, since its serve op last counted them:
    /// the serve op's own count (§10.2: a dispatch a caller's thread served counts too).
    served: AtomicU64,
    /// The context its handlers run as: a vCPU context of the serving domain (`VmCtx::vcpu`),
    /// charging the domain's fuel, so a trap ends the domain through its one end word (#2499). One
    /// does for every handler of the window, which run one at a time.
    vm: Arc<VmCtx>,
    /// Idle handler fibers, each with its slot and execution context, for the next dispatches. Only
    /// the thread whose context holds the gate touches them ([`Window::idle`]), so they need no lock.
    idle: std::cell::UnsafeCell<Vec<Shell>>,
    /// Its handlers the pool holds (promoted, [`ChildExec::promote`], and not yet ended): with none
    /// there, its serve op opens and shuts it and gives its gate back without the pool's lock.
    pooled: AtomicUsize,
    /// Its vCPU is inside a serve op. The pool's count of the window ([`WindowCount`]), which exists
    /// only while it holds handlers of it, takes it from here.
    open: std::sync::atomic::AtomicBool,
    /// Who holds its gate ([`fiber_rt::context_id`]), `0` while it is free: the serve op whenever it
    /// is not parked, or the thread of control running one of its handlers. A handler gives it back
    /// for each live call it makes and each event park ([`handler_steps_aside`]).
    gate: AtomicU64,
    /// How many wait to take the gate, parked on its key ([`Window::wait_take`]): a release wakes
    /// them only when there are some.
    gate_waiters: AtomicUsize,
    /// Its handlers' live calls into its own domain in flight, each from the gate its handler gave
    /// back for it until the handler takes the gate back ([`handler_steps_aside`]). Its serve op
    /// answers such a call, and on the oracle the answer wakes the handler at once, so the serve op
    /// resumes it before it returns. Here the handler may not be back from its call yet, nor filed
    /// with the pool, so the serve op cannot see it woken: it stays in its serve op until the
    /// handler is back ([`ServeSite::self_calls_in_flight`]).
    self_calls: AtomicUsize,
    /// #2695 — a held domain's window ([`HeldSite`]). Each of its handlers keeps a `vcpu.tls` word
    /// of its own, from the main thread's at rest, rather than the one the gate's last holder left.
    /// And its calls run one at a time, each until it parks (DESIGN.md §12a), a join included: a
    /// serve handler's join keeps the gate, as on the oracle it blocks the serving vCPU
    /// ([`Task::keep_lanes`]), but an activation is no serving vCPU's.
    held: bool,
    /// #2695 — where a thread outside the run waits for a held domain's gate ([`Window::wait_take_outside`]):
    /// an embedder's, whose wait is no park of the run, as its call is no task of it.
    outside: Mutex<()>,
    outside_cv: Condvar,
}

// SAFETY: a window is shared by its serve op and its handlers, wherever they run. Its fields are
// atomic, immutable or `Sync`, but its idle shells, which only the thread whose context holds the
// gate touches ([`Window::idle`]).
unsafe impl Sync for Window {}

/// #2742 — how many idle handler fibers a window keeps.
const IDLE_SHELLS: usize = 4;

/// #2742 — set in a domain's handler count ([`VmCtx::handlers`]) once its end waits for its
/// handlers: the last of them to end re-offers the domain's retiring task.
pub(crate) const HANDLERS_ENDING: usize = 1 << (usize::BITS - 1);

/// #2742 — the stack a direct handoff's handler needs left on the caller's to run there: a fresh
/// handler fiber's half ([`ServeSite::run_inline`]).
const HANDOFF_ROOM: u64 = TASK_STACK as u64 / 2;

/// #2742 — whether this stack has room for a handler run on it ([`HANDOFF_ROOM`]).
fn handoff_room() -> bool {
    let limit = crate::stack_check::thread_limit();
    let here = 0u8;
    let sp = &here as *const u8 as u64;
    limit != 0 && sp.saturating_sub(limit) >= HANDOFF_ROOM
}

/// #2742 — a serve trampoline's run on a handler fiber: its entry, and the arguments its guarded
/// entry shape takes.
#[derive(Clone, Copy)]
struct Job {
    entry: ServeEntry,
    args: *const i64,
    results: *mut i64,
    mem_base: *mut u8,
    fn_table: *const core::ffi::c_void,
    vm: *mut i64,
}

/// #2742 — the job a handler fiber runs at its next resume, left by its resumer.
struct JobCell(std::cell::UnsafeCell<Option<Job>>);

/// The body of every job a handler fiber runs ([`fiber_rt::make_handler_fiber`]): the handler's whole
/// run, counted as a serve handler, so a serve op inside it refuses (#2160). Its stack limit is the
/// fiber's (#2793).
///
/// # Safety
/// `cell` is the fiber's live [`JobCell`], its job left by the thread resuming it.
unsafe fn run_job(cell: *const (), _stack_low: u64) {
    let job = (*(*(cell as *const JobCell)).0.get()).expect("a job before each resume");
    vcpu_tls::set_serve_handlers(vcpu_tls::serve_handlers() + 1);
    (job.entry)(job.args, job.results, job.mem_base, job.fn_table, job.vm);
    vcpu_tls::set_serve_handlers(vcpu_tls::serve_handlers() - 1);
}

/// #2742 — a handler fiber with what it runs with: its pool slot, its execution context over the
/// callee's fiber table (so a `cont.*` it makes lives in the callee's registry, as on the oracle),
/// its job and its argument and result buffers. A window keeps idle ones for its next dispatches.
pub(crate) struct Shell {
    slot: Arc<FiberSlot>,
    /// The fiber its slot holds, for the slot's life, at an address its box keeps.
    fib: *mut Fiber,
    rt: Box<FiberRuntime>,
    job: Box<JobCell>,
    args: Vec<i64>,
    results: Vec<i64>,
}

// SAFETY: as `ChildTask`'s — its fiber, runtime and buffers move between threads only with the shell,
// and its job's pointers are left by the thread that resumes the fiber, which owns the shell, before
// the resume that reads them.
unsafe impl Send for Shell {}

impl Shell {
    /// A new one for the module `cc` over fiber table `table`; `None` when the OS refuses its stack.
    ///
    /// # Safety
    /// `cc` is live while the shell is.
    unsafe fn new(cc: &CompiledModule, table: &Arc<SharedFiberTable>) -> Option<Shell> {
        let mut rt = Box::new(FiberRuntime::new(
            Arc::clone(table),
            cc.fiber_cfg.map_or(0, |(type_id, _)| type_id),
            cc.fiber_cfg.map_or(cc.fn_table_mask, |(_, mask)| mask),
        ));
        if let Some(t) = cc.call_tramp.filter(|_| cc.fiber_cfg.is_some()) {
            rt.set_call_tramp(t);
        }
        let job = Box::new(JobCell(std::cell::UnsafeCell::new(None)));
        let cell = &*job as *const JobCell as *const ();
        let fiber = fiber_rt::make_handler_fiber(TASK_STACK, cell, run_job)?;
        let slot = FiberSlot::platform(fiber);
        let fib = slot.fiber_ptr().expect("a platform slot holds its fiber");
        Some(Shell {
            slot,
            fib,
            rt,
            job,
            args: Vec::new(),
            results: Vec::new(),
        })
    }
}

impl Window {
    /// The window's key: its address, tagged clear of every domain's parallelism lane. Unique while
    /// the window lives, and every holder of its gate keeps it alive.
    fn key(self: &Arc<Self>) -> usize {
        (1 << (usize::BITS - 2)) | Arc::as_ptr(self) as usize
    }

    /// Take the gate for `me` if it is free.
    fn take(&self, me: u64) -> bool {
        self.gate
            .compare_exchange(0, me, Ordering::SeqCst, Ordering::Acquire)
            .is_ok()
    }

    /// Whether `me` holds the gate.
    fn held_by(&self, me: u64) -> bool {
        self.gate.load(Ordering::Acquire) == me
    }

    /// Its idle handler fibers, for `me`, which holds the gate.
    ///
    /// # Safety
    /// `me` holds the gate as this thread's context or as a handler that ended on this thread, and
    /// the borrow ends before the gate goes to anyone else.
    #[allow(clippy::mut_from_ref)] // the gate is the exclusion
    unsafe fn idle(&self, me: u64) -> &mut Vec<Shell> {
        debug_assert!(
            self.held_by(me),
            "a window's idle shells are its gate holder's"
        );
        &mut *self.idle.get()
    }

    /// Hand the gate from `from`, which holds it, to `to`: a starter to the handler it runs, and back
    /// once the handler has ended there. `false` when `from` does not hold it.
    fn hand(&self, from: u64, to: u64) -> bool {
        self.gate
            .compare_exchange(from, to, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    /// `me` gives the gate back, and whoever waits to take it is woken. `false`, and nothing done,
    /// when `me` does not hold it: a release a handler's park already made is not made twice.
    fn release(&self, me: u64) -> bool {
        if self
            .gate
            .compare_exchange(me, 0, Ordering::SeqCst, Ordering::Relaxed)
            .is_err()
        {
            return false;
        }
        // Read after the release, as a waiter counts itself before it last looks at the gate: one
        // of the two sees the other. (loom models no host park, so no waiter parks there.)
        #[cfg(not(loom))]
        if self.gate_waiters.load(Ordering::SeqCst) != 0 {
            // SAFETY: the window's handler context names its live domain, whose hub its parks use.
            unsafe { os_thread_rt::wake_host_parks_in(self.vm.sched as usize, self.gate_key()) };
            if self.held {
                self.wake_outside();
            }
        }
        true
    }

    /// #2695 — wake the threads outside the run waiting for a held domain's gate, each to look again
    /// ([`Self::wait_take_outside`]): at a release, and at the domain's end.
    fn wake_outside(&self) {
        let _g = lock(&self.outside);
        self.outside_cv.notify_all();
    }

    /// #2695 — `me`, a thread outside the run (an embedder's), takes a held domain's gate, waiting while
    /// another holds it. Its wait is no park of the run, which its call is not a task of: the deadlock
    /// verdict never counts it. `false` once `ended` holds.
    fn wait_take_outside(&self, me: u64, ended: &dyn Fn() -> bool) -> bool {
        let free = || self.held_by(me) || (self.open.load(Ordering::Acquire) && self.take(me));
        if free() {
            return true;
        }
        let mut g = lock(&self.outside);
        loop {
            if free() {
                return true;
            }
            if ended() {
                return false;
            }
            // Counted before the last look, as a waiter in the run counts itself: a release before
            // the count is seen by the look, and one after it sees the count and wakes this wait.
            self.gate_waiters.fetch_add(1, Ordering::SeqCst);
            let taken = free();
            if !taken && !ended() {
                g = self.outside_cv.wait(g).unwrap_or_else(|e| e.into_inner());
            }
            self.gate_waiters.fetch_sub(1, Ordering::SeqCst);
            if taken {
                return true;
            }
        }
    }

    /// The key gate waiters park on, in the run's hub: the window's address, tagged clear of every
    /// bell's key.
    #[cfg(not(loom))]
    fn gate_key(&self) -> u64 {
        (1 << 62) | self as *const Window as u64
    }

    /// `me` takes the gate, waiting while the window is shut or another holds it, parked as a park
    /// of the run counts (as `vm`'s, the waiter's context). `false` when the window's domain has ended
    /// (`vm`'s trap cell then names its end) or no live vCPU could ever give the gate back (the
    /// deadlock verdict, raised as `ThreadFault`). A gate handed to `me` meanwhile counts as taken.
    ///
    /// # Safety
    /// `vm` is the live context of the waiter's call.
    unsafe fn wait_take(&self, me: u64, vm: &VmCtx) -> bool {
        loop {
            if self.held_by(me) || (self.open.load(Ordering::Acquire) && self.take(me)) {
                return true;
            }
            if vm.ended().is_some() {
                vm.raise(TrapKind::ThreadFault as i64);
                return false;
            }
            let free = || {
                (self.open.load(Ordering::Acquire) && self.gate.load(Ordering::SeqCst) == 0)
                    || self.held_by(me)
            };
            self.gate_waiters.fetch_add(1, Ordering::SeqCst);
            // Looked at again once counted, before parking: a release before the count is seen
            // here, and one after it sees the count and wakes this wait, parked or still on its way
            // to the park (a task's park checks only once woken, and a wake meanwhile is latched).
            let looks_again = free() || self.park_waiter(vm, &free);
            self.gate_waiters.fetch_sub(1, Ordering::SeqCst);
            if !looks_again {
                return false;
            }
        }
    }

    /// One park of a gate waiter, as `vm`'s (the waiter's context), until `free` may hold: `false`
    /// when no live vCPU could ever end it (the deadlock verdict) or the run ended it, `vm`'s trap
    /// cell then set.
    ///
    /// # Safety
    /// As [`Self::wait_take`].
    #[cfg(not(loom))]
    unsafe fn park_waiter(&self, vm: &VmCtx, free: &dyn Fn() -> bool) -> bool {
        let parked = os_thread_rt::park_host_call(
            vm,
            None,
            self.gate_key(),
            os_thread_rt::Parks::Fiber,
            None,
            || !free() && vm.ended().is_none(),
        );
        match parked {
            os_thread_rt::HostPark::Woken | os_thread_rt::HostPark::Frozen => true,
            os_thread_rt::HostPark::Deadlock | os_thread_rt::HostPark::Ended => {
                vm.raise(TrapKind::ThreadFault as i64);
                false
            }
        }
    }

    /// loom models no host park, nor the fibers handlers run on, so no gate waiter parks there; this
    /// arm keeps the crate compiling under `--cfg loom`, as `os_thread_rt`'s fiber arms do.
    #[cfg(loom)]
    unsafe fn park_waiter(&self, _vm: &VmCtx, _free: &dyn Fn() -> bool) -> bool {
        unreachable!("loom models no host park")
    }
}

/// #2742 — a handler's gate, given back for a wait ([`handler_steps_aside`]).
pub struct GateAside {
    window: *const Window,
    me: u64,
    /// The wait is a live call into the handler's own domain ([`Window::self_calls`]), in flight
    /// until this goes, after the handler has stepped back or its domain has ended.
    self_call: bool,
}

impl Drop for GateAside {
    fn drop(&mut self) {
        if self.self_call {
            // SAFETY: a handler context's window outlives every handler running as it, and this
            // lives in a handler's own frame.
            unsafe { (*self.window).self_calls.fetch_sub(1, Ordering::SeqCst) };
        }
    }
}

/// #2742 — code running as a serve handler parks on an event, or makes a live call: if it holds its
/// window's gate, the gate goes back for the wait, so the window's serve op and other handlers run
/// meanwhile, as on the oracle, where a handler's park frees its serving vCPU. A handler on a fiber
/// of its own and one on a caller's stack alike. `None` when `vm` is no handler's context, or the
/// gate is not held here (it was given back already, for the live call this park is part of).
/// `self_call`: the wait is a live call into the handler's own domain, which counts as in flight
/// until the handler steps back ([`Window::self_calls`]).
pub fn handler_steps_aside(vm: &VmCtx, self_call: bool) -> Option<GateAside> {
    let window = vm.handler_window.load(Ordering::Acquire);
    // A guest fiber's park parks the fiber alone, and the handler's own code runs on, as the fiber's
    // resumer, with the gate.
    if window.is_null() || crate::in_guest_fiber() {
        return None;
    }
    let me = fiber_rt::context_id();
    // SAFETY: a handler context's window outlives every handler running as it.
    let w = unsafe { &*window };
    if !w.held_by(me) {
        return None;
    }
    // #2842 — the `vcpu.tls` word goes with the gate, to whichever handler of the window or its
    // serve op takes it next (R2: the word goes with execution), and the call counts before the gate
    // goes, so the serve op that takes it next sees both. #2695 — a held domain's handler keeps its
    // own.
    if !w.held {
        w.tls.store(vcpu_tls::get(), Ordering::Relaxed);
    }
    if self_call {
        w.self_calls.fetch_add(1, Ordering::SeqCst);
    }
    let released = w.release(me);
    debug_assert!(released, "the gate's holder gives it back");
    Some(GateAside {
        window,
        me,
        self_call,
    })
}

/// #2742 — ...and takes it back after, before the handler's code runs on: waiting while the window
/// is shut or another holds the gate, the `vcpu.tls` word as the window's other handlers left it with
/// it. `false` when the handler's domain has ended meanwhile, its trap cell then set to unwind it. A
/// call into its own domain is back once the handler holds the gate again, or its domain has ended.
///
/// # Safety
/// `vm` is the context `aside` was given back from.
pub unsafe fn handler_steps_back(aside: GateAside, vm: &VmCtx) -> bool {
    let window = &*aside.window;
    if !window.wait_take(aside.me, vm) {
        return false;
    }
    if !window.held {
        vcpu_tls::seed(window.tls.load(Ordering::Relaxed));
    }
    true
}

/// #2742 — a window's handlers, as the run's deadlock predicate counts them: in the domain's `live`
/// and `parked` while the window is open, out of both while it is shut. On the oracle a handler is
/// a fiber of its serving vCPU, not a vCPU: the vCPU is blocked when its serve op idles with every
/// handler parked, or when it parks outside the serve op, whatever its handlers wait for.
#[derive(Default)]
struct WindowCount {
    /// The serving domain ([`VmCtx::domain_key`]).
    domain: usize,
    /// Started and not yet ended.
    handlers: usize,
    /// Of those, parked and counted blocked ([`Entry::counted`]).
    parked: usize,
    /// Its vCPU is inside a serve op.
    open: bool,
    /// Its domain has ended with these handlers in it ([`ChildExec::drain`]): they run once more,
    /// whatever the window and its gate, only to unwind, and count as an open window's do.
    draining: bool,
}

impl WindowCount {
    /// Whether its handlers count in the deadlock predicate: they may run.
    fn counts(&self) -> bool {
        self.open || self.draining
    }
}

/// #2742 — a serve handler running as a fiber of its own, of its serving vCPU's [`Window`]. Its
/// fiber enters the callee's serve trampoline over the callee's live window, as a vCPU of the
/// callee. It starts on the thread that admits it, the serve op's or a caller's
/// ([`ChildExec::run_here`]); a park files it with the pool, which resumes it when its wait ends,
/// on a worker, or on the serve op's thread when the serve op finds it woken.
pub(crate) struct HandlerTask {
    /// Its fiber, slot and execution context, and its arguments and results.
    shell: Shell,
    /// The callee window's fault range: its residencies' guard bracket.
    fault: (usize, usize),
    /// Its window.
    window: Arc<Window>,
    /// The context it runs as: its window's handler context ([`Window::vm`]), or, for a held
    /// domain's activation, one of its own, on the fuel its caller lends it ([`HeldSite::activate`]).
    vm: Arc<VmCtx>,
    /// #2695 — a held domain's activation's own `vcpu.tls` word, from the main thread's at rest; `None`
    /// for a serve handler, which runs with its window's ([`Window::held`]).
    tls: Option<i64>,
    /// #2695 — who carries it once it parks: its caller, which resumes it, not a worker ([`Carrier`]).
    carrier: Option<Arc<Carrier>>,
    /// #2695 — the fuel cell its context charges, when it has one of its own: a held domain's
    /// activation's, which its call funds. Kept for the context's life.
    _cell: Option<Box<crate::FuelCell>>,
    /// The serve-handler nesting count between residencies (#2160).
    serve_handlers: u32,
    /// It holds its window's gate: taken for a residency, and kept across a quantum's end and a
    /// whole-vCPU park, which on the oracle hold the serving vCPU.
    holds_lanes: bool,
    /// The thread domain the callee's thread thunks act in (`os_thread_rt::CURRENT_DOMAIN`); null
    /// when the callee's vmctx names it.
    domain: *const Domain,
    n_results: usize,
    /// Its last residency ended with its dispatch done and its fiber waiting for the next.
    reusable: bool,
    /// What its end tells the callee.
    done: Option<HandlerDone>,
}

// SAFETY: as `ChildTask`'s — the raw pointers a handler task holds (its window, its domain) are the
// callee's, live until the callee's end drains its handlers, and only the thread that owns the task
// (the one its window's gate admitted, or the pool's worker unwinding it) touches them.
unsafe impl Send for HandlerTask {}

/// How a serve handler ended ([`HandlerTask`]).
#[derive(Debug)]
pub enum HandlerEnd {
    /// It returned these results, every one the op declares (INVARIANTS #8).
    Returned(Vec<i64>),
    /// It trapped or faulted, ending the callee with this code.
    Trapped(i64),
}

/// What a handler's end runs, wherever it ends ([`HandlerTask`]): it settles the dispatch.
pub type HandlerDone = Box<dyn FnOnce(HandlerEnd) + Send>;

/// The guarded entry shape of a serve trampoline: `(args, results, mem_base, fn_table, vmctx)`.
type ServeEntry = extern "C" fn(*const i64, *mut i64, *mut u8, *const core::ffi::c_void, *mut i64);

impl HandlerTask {
    /// Its thread of control as its window's gate records it ([`fiber_rt::context_id`]): its shell's
    /// platform slot, which each residency makes the runtime's first active slot.
    fn id(&self) -> u64 {
        Arc::as_ptr(&self.shell.slot) as u64
    }

    /// How it ended, once its fiber has.
    fn end(&mut self) -> HandlerEnd {
        match self.vm.ended() {
            Some(code) => HandlerEnd::Trapped(code),
            None => HandlerEnd::Returned(self.shell.results[..self.n_results].to_vec()),
        }
    }

    /// It has ended: its `done` runs with its end, or, when it was given none (it ended on the
    /// thread that started it), its end comes back for that thread to deliver. It counts as served
    /// if it returned, and its fiber goes back to its window for the next dispatch, unless a fault
    /// left it unusable (`reusable`) or it ended without the gate (it unwound at its domain's end):
    /// the idle fibers are the gate holder's, the handler's or the thread's it handed the gate to.
    /// Its window comes back with its end.
    fn conclude(mut self, reusable: bool) -> (Option<HandlerEnd>, Arc<Window>) {
        let end = self.end();
        let served = matches!(end, HandlerEnd::Returned(_));
        let end = match self.done.take() {
            Some(done) => {
                done(end);
                None
            }
            None => Some(end),
        };
        // Counted after its settle, so a serve op that reads the count sees the settle too.
        if served {
            self.window.served.fetch_add(1, Ordering::AcqRel);
        }
        if reusable {
            let holder = [self.id(), fiber_rt::context_id()]
                .into_iter()
                .find(|&id| self.window.held_by(id));
            if let Some(holder) = holder {
                // SAFETY: `holder` holds the gate: the handler, which ended on this thread, or this
                // thread's context, which it handed the gate back to.
                let idle = unsafe { self.window.idle(holder) };
                if idle.len() < IDLE_SHELLS {
                    idle.push(self.shell);
                }
            }
        }
        (end, self.window)
    }
}

/// How a handler's residency on the thread that holds its gate went ([`ServeSite::start`],
/// [`ServeSite::resume_woken`]).
#[derive(Debug)]
pub struct Ran {
    /// The caller still holds the gate: the handler ended or parked alone. A handler preempted, or
    /// parked on a whole-vCPU wait, keeps it, and gives it back from the pool.
    pub held: bool,
    /// The handler ended: the caller rings for whoever waits on its dispatch.
    pub ended: bool,
    /// Its end, for the caller to deliver, when it ended where it started ([`ServeSite::start`]);
    /// a handler that parked first delivers its own, through the `done` it was given.
    pub end: Option<HandlerEnd>,
}

/// Why a dispatch did not start ([`ServeSite::start`]).
#[derive(Debug, PartialEq, Eq)]
pub enum ServeRefused {
    /// The module has no handler `func`, or not of the dispatch's arity.
    NoHandler,
    /// The OS refused the handler's stack.
    NoStack,
}

/// #2742 — a serving vCPU as its handlers run against it: the serving domain's module and live
/// window, the vCPU's own context, its fiber table and thread domain, the run's pool, and its
/// [`Window`]. Made at the vCPU's first serve op ([`ServeSite::of`]) and kept in its context. A
/// caller that finds the serve op parked starts its own dispatch's handler through the same site, on
/// its own thread.
pub struct ServeSite {
    cc: *const CompiledModule,
    mem_base: *mut u8,
    mem_reserved: u64,
    table: Arc<SharedFiberTable>,
    /// The serving vCPU's own thread domain, if it is not its vmctx's (a §14 task's).
    domain: *const Domain,
    exec: Arc<ChildExec>,
    window: Arc<Window>,
}

// SAFETY: the raw pointers are the serving domain's module, window, vCPU context and thread domain,
// live while it serves (its end drains its handlers before any of them goes); every mutation behind
// them is under their own locks.
unsafe impl Send for ServeSite {}
unsafe impl Sync for ServeSite {}

impl ServeSite {
    /// The site of the serve op running on this thread, the serving vCPU's: `cc` its registered serve
    /// context, `mem_base`/`mem_reserved` its window, `vm` its context, where the site is kept, and
    /// `ring` what rings its domain's bell. `None` when the run has no pool (no thread domain, or no
    /// nursery), so it has no §14 child and no live offer reaches it: its handlers run inline
    /// ([`crate::invoke_handler`]).
    ///
    /// # Safety
    /// Called from the serve op's thunk on the serving vCPU's thread; `cc` is the live serve context
    /// of the module `vm` runs, over the window at `mem_base`.
    pub unsafe fn of(
        vm: &VmCtx,
        cc: *const core::ffi::c_void,
        mem_base: *mut u8,
        mem_reserved: u64,
        ring: impl FnOnce() -> Box<dyn Fn() + Send + Sync>,
    ) -> Option<Arc<ServeSite>> {
        if let Some(site) = vm.serve_site.get() {
            // The vCPU serves one module over one window for its life.
            let same =
                std::ptr::eq(site.cc, cc as *const CompiledModule) && site.mem_base == mem_base;
            return same.then(|| Arc::clone(site));
        }
        if vm.sched.is_null() {
            return None;
        }
        let hub = os_thread_rt::current_domain(vm.sched as *const Domain).hub();
        // A run whose nursery runs no pool has no §14 child, so no live offer reaches it.
        let exec = hub.child_exec()?;
        let table = fiber_rt::current_table().unwrap_or_else(|| {
            Arc::new(SharedFiberTable::new(
                1,
                temen_ir::durable_abi::ShadowArena::EMPTY,
                None,
            ))
        });
        let window = Arc::new(Window {
            tls: std::sync::atomic::AtomicI64::new(0),
            ring: ring(),
            served: AtomicU64::new(0),
            vm: Arc::new(vm.vcpu((*(cc as *const CompiledModule)).instance.fuel)),
            idle: std::cell::UnsafeCell::new(Vec::new()),
            pooled: AtomicUsize::new(0),
            open: std::sync::atomic::AtomicBool::new(false),
            gate: AtomicU64::new(0),
            gate_waiters: AtomicUsize::new(0),
            self_calls: AtomicUsize::new(0),
            held: false,
            outside: Mutex::new(()),
            outside_cv: Condvar::new(),
        });
        // Its handler context names it, for the gate a handler's park gives back.
        window
            .vm
            .handler_window
            .store(Arc::as_ptr(&window) as *mut Window, Ordering::Release);
        let site = Arc::new(ServeSite {
            cc: cc as *const CompiledModule,
            mem_base,
            mem_reserved,
            table,
            domain: os_thread_rt::current_domain_ptr(),
            exec,
            window,
        });
        Some(Arc::clone(vm.serve_site.get_or_init(|| site)))
    }

    /// The serve op begins: take the gate, free while the window is shut, and open the window, the
    /// vCPU's `vcpu.tls` word with it.
    pub fn begin(&self) {
        self.window
            .tls
            .store(vcpu_tls::get(), std::sync::atomic::Ordering::Relaxed);
        let taken = self.window.take(fiber_rt::context_id());
        debug_assert!(taken, "a shut window's gate is free");
        self.set_open(true);
    }

    /// Open or shut the window: in the pool's count of it too, if the pool holds a handler of it.
    /// Only the gate's holder or its serve op opens, shuts and promotes, so no handler is promoted
    /// meanwhile.
    fn set_open(&self, open: bool) {
        self.window.open.store(open, Ordering::Release);
        if self.window.pooled.load(Ordering::Acquire) != 0 {
            self.exec.set_window_open(self.window.key(), open);
        }
    }

    /// The serve op ends: shut the window and give the gate back if it holds it (`held`; one that
    /// unwinds may not), the `vcpu.tls` word back to the vCPU. No handler of it runs until its next
    /// serve op.
    pub fn end(&self, held: bool) {
        self.set_open(false);
        if held {
            self.window.release(fiber_rt::context_id());
        }
        vcpu_tls::seed(self.window.tls.load(std::sync::atomic::Ordering::Relaxed));
    }

    /// Take the gate if it is free: no handler of this window runs, and the serve op is parked.
    pub fn try_enter(&self) -> bool {
        self.window.take(fiber_rt::context_id())
    }

    /// The serve op takes the gate back, waiting while a handler holds it: `false` when its domain
    /// has ended meanwhile, or no live vCPU could give the gate back (`vm`'s trap cell then says why).
    ///
    /// # Safety
    /// `vm` is the serve op's live context.
    pub unsafe fn wait_enter(&self, vm: &VmCtx) -> bool {
        self.window.wait_take(fiber_rt::context_id(), vm)
    }

    /// The serve op parks: give the gate back, so a woken handler of the window runs meanwhile on
    /// the pool, which is woken for it if one waits.
    pub fn step_aside(&self) {
        self.window.release(fiber_rt::context_id());
        if self.window.pooled.load(Ordering::Acquire) != 0 {
            self.exec.offer_gate(self.window.key());
        }
    }

    /// #2742 — the site's vCPU has ended, however it ended (owner, 2026-11-09): its window never opens
    /// again, so each handler of it the pool holds answers its call `CAP_REVOKED` now, as the
    /// oracle answers a dead vCPU's parked handlers. Their fibers unwind once their domain has no
    /// vCPU left ([`ChildExec::drain`]).
    pub(crate) fn revoke(&self) {
        if self.window.pooled.load(Ordering::Acquire) != 0 {
            self.exec.revoke_window(&self.window);
        }
    }

    /// The dispatches this window's handlers served since the last count, which the serve op adds to
    /// its own.
    pub fn take_served(&self) -> i64 {
        self.window.served.swap(0, Ordering::AcqRel) as i64
    }

    /// Whether the pool holds handlers of this window, one of which may be woken: what makes
    /// [`Self::resume_woken`] worth its pool lock.
    pub fn holds_handlers(&self) -> bool {
        self.window.pooled.load(Ordering::Acquire) != 0
    }

    /// Whether a handler of this window has a live call into this domain in flight, which the serve
    /// op answers: it does not leave its serve op until the handler is back ([`Window::self_calls`]).
    pub fn self_calls_in_flight(&self) -> bool {
        self.window.self_calls.load(Ordering::SeqCst) != 0
    }

    /// Resume a woken handler of this window here, holding the gate: what the serve op does first,
    /// as the oracle's resumes a woken handler before the next dispatch. `None` when none is woken;
    /// else whether the caller still holds the gate (a handler that parks on a whole-vCPU wait, or
    /// is preempted, keeps it).
    pub fn resume_woken(&self) -> Option<Ran> {
        let (id, task) = self.exec.take_woken(self.window.key())?;
        Some(self.exec.reside_here(id, task))
    }

    /// #2742 — whether a direct handoff may run handler `func` on the caller's own stack
    /// ([`Self::run_inline`]): nothing it reaches can park, so it runs to completion or traps and
    /// never gives its gate back, and this stack has room for it. A handler that can park runs on a
    /// fiber of its own ([`Self::start`]).
    pub fn inline_ok(&self, func: u32) -> bool {
        // SAFETY: the site's module is live while the site is.
        let never_parks = unsafe { (*self.cc).handler_never_parks(func) };
        never_parks && handoff_room()
    }

    /// #2742 — the direct handoff of a handler that cannot park ([`Self::inline_ok`]): run handler
    /// `func` with `args` on this thread's own stack, as the window's vCPU (its thread-local word,
    /// counted as a serve handler), for a caller whose call waits on it anyway. It never parks, so
    /// the gate the caller holds stays with it throughout. It counts in its domain's handlers until
    /// it ends, so the domain's end (a kill, a trap of another vCPU) waits for it before its window
    /// and code go.
    ///
    /// # Safety
    /// The caller holds the gate, the site's vCPU is parked in its serve op, and `inline_ok(func)`.
    pub unsafe fn run_inline(&self, func: u32, args: &[i64]) -> Result<HandlerEnd, ServeRefused> {
        self.run_inline_as(&self.window.vm, None, func, args)
    }

    /// [`Self::run_inline`] as context `vm`, with `tls` its own `vcpu.tls` word, or the window's
    /// (`None`): a held domain's activation runs as one of its own ([`HeldSite::activate`]).
    ///
    /// # Safety
    /// As [`Self::run_inline`]; `vm` is a context of the site's domain.
    unsafe fn run_inline_as(
        &self,
        vm: &VmCtx,
        tls: Option<i64>,
        func: u32,
        args: &[i64],
    ) -> Result<HandlerEnd, ServeRefused> {
        let cc = &*self.cc;
        let Some((code, n_params, n_results)) = cc.handler_tramp(func) else {
            return Err(ServeRefused::NoHandler);
        };
        if args.len() != n_params {
            return Err(ServeRefused::NoHandler);
        }
        vm.handlers().fetch_add(1, Ordering::SeqCst);
        let mut results = vec![0i64; n_results.max(1)];
        let caller_tls = vcpu_tls::get();
        vcpu_tls::seed(tls.unwrap_or_else(|| self.window.tls.load(Ordering::Relaxed)));
        let caller_handlers = vcpu_tls::serve_handlers();
        vcpu_tls::set_serve_handlers(caller_handlers + 1);
        let (lo, hi) = mem::fault_range_of(self.mem_base, self.mem_reserved as usize);
        let faulted = mem::run_guarded_range(
            code,
            args.as_ptr(),
            results.as_mut_ptr(),
            self.mem_base,
            cc.fn_table.base(),
            vm,
            lo,
            hi,
        );
        vcpu_tls::set_serve_handlers(caller_handlers);
        if tls.is_none() {
            self.window.tls.store(vcpu_tls::get(), Ordering::Relaxed);
        }
        vcpu_tls::seed(caller_tls);
        if faulted {
            vm.raise(mem::FAULT_TRAP);
        }
        let end = match vm.ended() {
            Some(code) => HandlerEnd::Trapped(code),
            None => {
                results.truncate(n_results);
                self.window.served.fetch_add(1, Ordering::AcqRel);
                HandlerEnd::Returned(results)
            }
        };
        self.exec.handler_ended(vm);
        Ok(end)
    }

    /// Start handler `func` with `args` on a fiber of its own, here, holding the gate. One that ends
    /// here gives its end back ([`Ran::end`]); one that parks is given the `done` that `make_done`
    /// makes then, which gets its end wherever the pool ends it. So the common call makes none.
    ///
    /// # Safety
    /// The gate is held, and the site's vCPU is in its serve op.
    pub unsafe fn start(
        &self,
        func: u32,
        args: &[i64],
        make_done: &mut dyn FnMut() -> HandlerDone,
    ) -> Result<Ran, ServeRefused> {
        let h = self.handler(func, args, &self.window.vm)?;
        Ok(self.exec.run_here(h, fiber_rt::context_id(), make_done))
    }

    /// Handler `func` with `args`, on a fiber of its own (an idle one of the window's, else a new
    /// one), to run as context `vm`: the window's vCPU, or a held domain's activation's own.
    ///
    /// # Safety
    /// This thread's context holds the gate: the window's idle fibers are its holder's.
    unsafe fn handler(
        &self,
        func: u32,
        args: &[i64],
        vm: &Arc<VmCtx>,
    ) -> Result<HandlerTask, ServeRefused> {
        let cc = &*self.cc;
        let Some((code, n_params, n_results)) = cc.handler_tramp(func) else {
            return Err(ServeRefused::NoHandler);
        };
        if args.len() != n_params {
            return Err(ServeRefused::NoHandler);
        }
        let idle = self.window.idle(fiber_rt::context_id()).pop();
        let mut shell = match idle {
            Some(shell) => shell,
            None => Shell::new(cc, &self.table).ok_or(ServeRefused::NoStack)?,
        };
        shell.args.clear();
        shell.args.extend_from_slice(args);
        shell.results.clear();
        shell.results.resize(n_results.max(1), 0);
        *shell.job.0.get() = Some(Job {
            entry: core::mem::transmute::<*const u8, ServeEntry>(code),
            args: shell.args.as_ptr(),
            results: shell.results.as_mut_ptr(),
            mem_base: self.mem_base,
            fn_table: cc.fn_table.base(),
            // The entry ABI spells the vmctx as its trap cell (`VmCtx` field 0).
            vm: Arc::as_ptr(vm) as *mut i64,
        });
        Ok(HandlerTask {
            shell,
            fault: mem::fault_range_of(self.mem_base, self.mem_reserved as usize),
            vm: Arc::clone(vm),
            window: Arc::clone(&self.window),
            tls: None,
            carrier: None,
            _cell: None,
            serve_handlers: 0,
            holds_lanes: true,
            domain: self.domain,
            n_results,
            reusable: false,
            done: None,
        })
    }
}

/// #2695 step 2 — who calls a held domain ([`HeldSite::activate`]).
pub enum HeldCaller<'a> {
    /// A thread outside the run, the embedder's, lending the call `fuel`: what it leaves comes back.
    Outside(&'a mut u64),
    /// A vCPU or task of the run, calling through a live offer, as context `vm`: it funds the call
    /// from its own budget.
    Inside(&'a VmCtx),
}

/// #2695 step 2 — who carries a held domain's activation once it parks ([`HeldSite::activate`]): the
/// thread of control of its call, which resumes it each time it is runnable again, as executor 2's
/// caller does. No worker of the pool runs it (owner, 2026-11-11).
pub(crate) struct Carrier {
    /// Set when the activation is runnable again: its carrier takes it back.
    woken: std::sync::atomic::AtomicBool,
    by: CarriedBy,
}

enum CarriedBy {
    /// An OS thread, the embedder's or a vCPU's: unparked.
    Thread(std::thread::Thread),
    /// A task of the pool: re-offered, under the pool's lock, which its wake runs under.
    Task(u64),
}

impl Carrier {
    /// The carrier of an activation called from this thread of control.
    fn here() -> Arc<Carrier> {
        let by = match RUNNING_TASK.with(Cell::get) {
            0 => CarriedBy::Thread(std::thread::current()),
            id => CarriedBy::Task(id),
        };
        Arc::new(Carrier {
            woken: std::sync::atomic::AtomicBool::new(false),
            by,
        })
    }

    /// The activation is runnable again: wake its carrier. Under the pool's lock (`g`).
    fn wake(&self, exec: &Arc<ChildExec>, g: &mut ExecState) {
        self.woken.store(true, Ordering::SeqCst);
        match &self.by {
            CarriedBy::Thread(t) => t.unpark(),
            CarriedBy::Task(id) => exec.wake_parked(g, Wake::Task(*id)),
        }
    }

    /// Wait until the activation is runnable again: on this thread, or, for a task of the pool, as its
    /// park (as `vm`'s, the caller's context), which frees its worker. `false` when the task's own
    /// domain ended meanwhile: it unwinds, and its activation runs on in the pool.
    ///
    /// # Safety
    /// `vm` is the live context of the call, for a task's carrier.
    unsafe fn wait(&self, vm: Option<&VmCtx>) -> bool {
        match (&self.by, vm) {
            (CarriedBy::Task(_), Some(vm)) => {
                // loom models no host park.
                #[cfg(loom)]
                let _ = vm;
                #[cfg(not(loom))]
                {
                    let key = (1 << 61) | self as *const Carrier as u64;
                    let parked = os_thread_rt::park_host_call(
                        vm,
                        None,
                        key,
                        os_thread_rt::Parks::Fiber,
                        None,
                        || !self.woken.load(Ordering::SeqCst),
                    );
                    if !matches!(parked, os_thread_rt::HostPark::Woken) {
                        return false;
                    }
                }
            }
            _ => {
                while !self.woken.load(Ordering::SeqCst) {
                    std::thread::park();
                }
            }
        }
        self.woken.store(false, Ordering::SeqCst);
        true
    }
}

/// #2695 step 2 — a held domain at rest, as its calls reach it ([`crate::CompiledModule::set_rest`]):
/// a serving vCPU parked for good, whose window is open while it rests. Each call, the embedder's or
/// one from inside its run, is an **activation**: a new fiber in the domain at the op's handler,
/// called with exactly the op's arguments, on the caller's thread, which funds it. The calls run
/// one at a time under the window's gate (`Single`), each until it parks, and each starts with the
/// main thread's `vcpu.tls` word. One that parks is the caller's to carry ([`Carrier`]).
pub struct HeldSite {
    site: ServeSite,
    /// The resting root's context, which each activation's is a vCPU context of. Live until the rest
    /// returns, and read only by an activation that counted itself in flight and then found the site
    /// open ([`Self::activate`], [`InFlight`]): the rest closes the site before it waits for the
    /// count, so such an activation is one it waits for. The site itself outlives the run (its
    /// embedder holds the `Arc`), and so may this pointer, dead (#2990).
    root: *const VmCtx,
    /// The main thread's `vcpu.tls` word at rest.
    tls: i64,
    /// Activations in flight: the rest waits for them before the domain's window and code go.
    inflight: AtomicUsize,
    /// The domain no longer takes calls: its rest has ended, or is ending.
    closed: std::sync::atomic::AtomicBool,
    /// Its embedder asked for the end ([`Self::end`]).
    end_asked: Mutex<bool>,
    /// Where the rest waits: for the end, then for the activations in flight.
    rest_cv: Condvar,
}

// SAFETY: as `ServeSite`'s; `root` is the resting root's context, live while the rest runs, and
// read by an activation only after it counted itself in flight and found the site open, which the
// rest waits for. A closed site touches only what it owns (#2990).
unsafe impl Send for HeldSite {}
unsafe impl Sync for HeldSite {}

/// An activation in flight ([`HeldSite::activate`]), counted out when it leaves.
struct InFlight<'a>(&'a HeldSite);

impl Drop for InFlight<'_> {
    fn drop(&mut self) {
        let site = self.0;
        // The rest waits for the domain's end, which an activation may have made, and once it is
        // ending, for the last activation. The end is read while this one still counts, so the
        // root's context is live; the rest's look at the count, under the lock this wake takes,
        // sees this one gone or this one sees the domain closed.
        //
        // #2990 — but only while the site is open. A call that counted itself in after the rest
        // had seen the count reach 0 and returned finds the site closed and leaves at once, and by
        // then the root's context may be gone: it must not be read. The rest sets `closed` before
        // it waits on the count, so an activation that sees the site open is one the rest waits
        // for, and one that sees it closed notifies below anyway.
        // SAFETY: the rest does not return while an activation counts in flight and the site is
        // open.
        let ended = !site.closed() && unsafe { (*site.root).ended().is_some() };
        site.inflight.fetch_sub(1, Ordering::SeqCst);
        if ended || site.closed() {
            let _g = lock(&site.end_asked);
            site.rest_cv.notify_all();
        }
    }
}

impl HeldSite {
    /// The site of a domain whose root `root` returned and now rests: its module `cc`, its window at
    /// `mem_base`, its fiber table `table` and thread domain `domain`, and `tls`, the main thread's
    /// word at rest. The run's pool, made now if the run has none, keeps an activation that parks.
    ///
    /// # Safety
    /// Called on the root's thread, after its entry returned and before its teardown; every pointer
    /// is live until [`Self::rest`] returns.
    pub(crate) unsafe fn new(
        root: &VmCtx,
        cc: &CompiledModule,
        mem_base: *mut u8,
        mem_reserved: u64,
        table: Arc<SharedFiberTable>,
        domain: &Domain,
        tls: i64,
    ) -> Arc<HeldSite> {
        let hub = domain.hub();
        let exec = hub.child_exec().unwrap_or_else(|| {
            let e = ChildExec::new(hub as *const Domain as usize);
            hub.set_child_exec(Arc::clone(&e));
            e
        });
        let window = Arc::new(Window {
            tls: std::sync::atomic::AtomicI64::new(tls),
            ring: Box::new(|| {}),
            served: AtomicU64::new(0),
            vm: Arc::new(root.vcpu(root.fuel)),
            idle: std::cell::UnsafeCell::new(Vec::new()),
            pooled: AtomicUsize::new(0),
            open: std::sync::atomic::AtomicBool::new(true),
            gate: AtomicU64::new(0),
            gate_waiters: AtomicUsize::new(0),
            self_calls: AtomicUsize::new(0),
            held: true,
            outside: Mutex::new(()),
            outside_cv: Condvar::new(),
        });
        window
            .vm
            .handler_window
            .store(Arc::as_ptr(&window) as *mut Window, Ordering::Release);
        Arc::new(HeldSite {
            site: ServeSite {
                cc: cc as *const CompiledModule,
                mem_base,
                mem_reserved,
                table,
                domain: domain as *const Domain,
                exec,
                window,
            },
            root: root as *const VmCtx,
            tls,
            inflight: AtomicUsize::new(0),
            closed: std::sync::atomic::AtomicBool::new(false),
            end_asked: Mutex::new(false),
            rest_cv: Condvar::new(),
        })
    }

    /// The domain rests: until its embedder ends it ([`Self::end`]) or it ends itself (a trap or an
    /// exit of an activation or a thread, a kill). Then it ends: it takes no more calls, its parks
    /// are woken to find the end, and the activations in flight unwind or return, which this waits
    /// for. On the root's thread, before its teardown.
    ///
    /// # Safety
    /// As [`Self::new`].
    pub(crate) unsafe fn rest(&self) {
        let root = &*self.root;
        // A vCPU's end, a kill and a freeze fire the run's wakers. (loom models no run.)
        #[cfg(not(loom))]
        let _woken = {
            let rest = self as *const HeldSite as usize;
            os_thread_rt::wake_on_run_event(
                root,
                Box::new(move || {
                    // SAFETY: the registration lives in `rest`, which outlives every firing of it.
                    let site = unsafe { &*(rest as *const HeldSite) };
                    let _g = lock(&site.end_asked);
                    site.rest_cv.notify_all();
                }),
            )
        };
        let killed = || os_thread_rt::epoch_fired(root.epoch as usize);
        {
            let mut asked = lock(&self.end_asked);
            while !*asked && root.ended().is_none() && !killed() {
                asked = self.rest_cv.wait(asked).unwrap_or_else(|e| e.into_inner());
            }
        }
        self.closed.store(true, Ordering::SeqCst);
        if killed() {
            root.raise(TrapKind::OutOfFuel as i64);
        }
        if root.ended().is_none() {
            root.raise(crate::DOMAIN_DONE_CODE as i64);
        }
        // Every park of the domain looks again and finds the end: its threads', and its parked
        // activations', which their carriers run once more to unwind.
        (*self.site.domain).wake_own_parked();
        self.site.exec.wake_locked(Wake::All);
        self.site.window.wake_outside();
        let mut g = lock(&self.end_asked);
        while self.inflight.load(Ordering::SeqCst) != 0 {
            g = self.rest_cv.wait(g).unwrap_or_else(|e| e.into_inner());
        }
        drop(g);
        self.site.set_open(false);
    }

    /// End the domain ([`Self::rest`]): its embedder lets it go.
    pub fn end(&self) {
        *lock(&self.end_asked) = true;
        self.rest_cv.notify_all();
    }

    /// Whether the domain has stopped taking calls: it has ended, or is ending.
    pub fn closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }

    /// Run handler `func` with `args` as an activation of the domain on this thread, for `caller`:
    /// its end, `Trapped` once the domain has ended (a call into a dead domain), or `None` when the
    /// caller's own domain ended while it waited (its trap cell set to unwind it). `Err` when the
    /// module has no such handler, or the OS refused its stack.
    ///
    /// May be called at any time, before or after the domain's run returned: its embedder holds the
    /// `Arc<HeldSite>` past the run. A closed site answers `Trapped` at once and touches only what
    /// the site owns; an open one is read only while this call counts in flight, which the rest waits
    /// for before the run's context, window and module go (#2990).
    ///
    /// # Safety
    /// `Inside`'s context is the caller's live one.
    pub unsafe fn activate(
        &self,
        func: u32,
        args: &[i64],
        caller: HeldCaller<'_>,
    ) -> Result<Option<HandlerEnd>, ServeRefused> {
        self.inflight.fetch_add(1, Ordering::SeqCst);
        let _out = InFlight(self);
        let revoked = Ok(Some(HandlerEnd::Trapped(crate::DOMAIN_DONE_CODE as i64)));
        if self.closed() {
            return revoked;
        }
        let root = &*self.root;
        let cc = &*self.site.cc;
        match cc.handler_tramp(func) {
            Some((_, n_params, _)) if n_params == args.len() => {}
            _ => return Err(ServeRefused::NoHandler),
        }
        // Its context: a vCPU context of the domain on the fuel the call funds it with. A cell of its
        // own holding the embedder's loan, or drawing on the inside caller's budget; the caller's
        // own cell, shared, when that draws on none (a fixed allowance); or, for a caller that
        // charges no cell, an unbounded one, as the domain's code charges one whatever it runs for.
        let (inside, lent) = match caller {
            HeldCaller::Outside(fuel) => (None, Some(fuel)),
            HeldCaller::Inside(vm) => (Some(vm), None),
        };
        // A live offer reaches the domain only from its own run (spawns wire them, and a held run
        // forks no twin), whose hub its gate waits park in and whose pool carries its parks.
        debug_assert!(inside.is_none_or(|vm| std::ptr::eq(
            os_thread_rt::current_domain(vm.sched as *const Domain).hub(),
            (*self.site.domain).hub()
        )));
        let mut cell = match (&lent, inside.and_then(|vm| vm.fuel.as_ref())) {
            (Some(fuel), _) => Some(crate::FuelCell::fixed(**fuel)),
            (None, Some(c)) => c.sibling(),
            (None, None) => Some(crate::FuelCell::fixed(u64::MAX)),
        };
        let fuel = match cell.as_deref_mut() {
            Some(c) => c as *mut crate::FuelCell,
            None => inside.map_or(root.fuel, |vm| vm.fuel),
        };
        let vm = Arc::new(root.vcpu(fuel));
        let window = &self.site.window;
        vm.handler_window
            .store(Arc::as_ptr(window) as *mut Window, Ordering::Release);
        let me = fiber_rt::context_id();
        let ended = || self.closed() || root.ended().is_some();
        let gated = match inside {
            None => window.wait_take_outside(me, &ended),
            Some(cvm) => window.wait_take(me, cvm),
        };
        if !gated {
            return match inside {
                Some(cvm) if cvm.ended().is_some() && !ended() => Ok(None),
                _ => revoked,
            };
        }
        // The domain ended while the call waited for its gate (a waiter inside the run waits on as
        // its own domain's park, which the callee's end does not stop): a call into a dead domain.
        if ended() {
            self.site.step_aside();
            return revoked;
        }
        if self.site.inline_ok(func) {
            let end = self.site.run_inline_as(&vm, Some(self.tls), func, args);
            // Back to its own caller: a direct call settles nothing in the pool.
            self.site.step_aside();
            if let (Some(fuel), Some(c)) = (lent, cell.as_ref()) {
                *fuel = c.can_burn();
            }
            return end.map(Some);
        }
        self.carry(&vm, cell.take(), func, args, inside, lent)
    }

    /// The activation of handler `func` on a fiber of its own, started here under the gate: run here
    /// until it ends, and each time it parks, here again once it is runnable. Its carrier is this
    /// thread of control ([`Carrier`]).
    ///
    /// # Safety
    /// As [`Self::activate`]; this thread's context holds the gate.
    unsafe fn carry(
        &self,
        vm: &Arc<VmCtx>,
        cell: Option<Box<crate::FuelCell>>,
        func: u32,
        args: &[i64],
        inside: Option<&VmCtx>,
        lent: Option<&mut u64>,
    ) -> Result<Option<HandlerEnd>, ServeRefused> {
        let site = &self.site;
        let mut h = match site.handler(func, args, vm) {
            Ok(h) => h,
            Err(refused) => {
                site.step_aside();
                return Err(refused);
            }
        };
        // Its end, wherever it ends: here, or in the pool's drain once its carrier is gone. With it,
        // what its cell has left, read while the cell is still the task's.
        let slot: Arc<Mutex<Option<(HandlerEnd, u64)>>> = Arc::new(Mutex::new(None));
        h.done = Some(Box::new({
            let (slot, vm) = (Arc::clone(&slot), Arc::clone(vm));
            move |end| {
                // SAFETY: a context's cell outlives its handler, which is ending now.
                let left = unsafe { vm.fuel.as_ref() }.map_or(0, crate::FuelCell::can_burn);
                *lock(&slot) = Some((end, left));
            }
        }));
        let carrier = Carrier::here();
        h.tls = Some(self.tls);
        h.carrier = Some(Arc::clone(&carrier));
        h._cell = cell;
        let me = fiber_rt::context_id();
        let (id, mut task) = site.exec.enlist(h);
        let mut hand = true;
        loop {
            let ran = site.exec.reside_on(id, task, hand);
            if ran.ended {
                if ran.held {
                    site.step_aside();
                }
                break;
            }
            // Parked, or preempted: take it back once it is runnable again, here.
            task = loop {
                if let Some(t) = site.exec.take_carried(id) {
                    break t;
                }
                if !carrier.wait(inside) {
                    // Its caller's domain ended: it unwinds, and the pool runs its activation on.
                    site.exec.uncarry(id, None);
                    return Ok(None);
                }
            };
            // A park on a whole-vCPU wait (a join) kept the gate ([`Task::keep_lanes`]). Any other
            // runs under the gate taken again, unless the domain has ended: it then runs once more,
            // to unwind.
            hand = !task.holds_lanes()
                && !self.closed()
                && match inside {
                    None => site.window.wait_take_outside(me, &|| self.closed()),
                    Some(cvm) => site.window.wait_take(me, cvm),
                };
            if !hand && !task.holds_lanes() && !self.closed() {
                // Its caller's domain ended while it waited for the gate.
                site.exec.uncarry(id, Some(task));
                return Ok(None);
            }
        }
        let ended = lock(&slot).take();
        Ok(ended.map(|(end, left)| {
            if let Some(fuel) = lent {
                *fuel = left;
            }
            end
        }))
    }
}

/// What the pool runs: a §14 child domain's vCPU, or a serve handler's fiber (#2742).
pub(crate) enum Task {
    Child(ChildTask),
    Handler(HandlerTask),
}

impl Task {
    fn slot(&self) -> &Arc<FiberSlot> {
        match self {
            Task::Child(t) => &t.slot,
            Task::Handler(h) => &h.shell.slot,
        }
    }

    /// Its lane chain: a child's; a handler's gate is its window's own ([`Window::gate`]).
    fn chain(&self) -> &[(usize, i64)] {
        match self {
            Task::Child(t) => &t.chain,
            Task::Handler(_) => &[],
        }
    }

    fn holds_lanes(&self) -> bool {
        match self {
            Task::Child(t) => t.holds_lanes,
            Task::Handler(h) => h.holds_lanes,
        }
    }

    /// After a residency that ended `outcome`: whether the task keeps its lanes. A task its quantum
    /// preempted does (#2237). So does a handler parked on a whole-vCPU wait (a join, a page fault):
    /// on the oracle that wait blocks the serving vCPU, so no other handler of its window runs. A
    /// held domain's activation gives its gate back at a join, as at any park ([`Window::held`]). A
    /// handler keeps only a gate it holds (one the teardown resumed holds none).
    fn keep_lanes(&mut self, outcome: &Outcome) -> bool {
        let keep = match (&*self, outcome) {
            (_, Outcome::Yielded) => true,
            (Task::Handler(h), Outcome::Parked { .. }) => {
                let on = h.shell.slot.parked_on();
                on.blocks_vcpu()
                    && !(h.window.held
                        && matches!(on, fiber_rt::ParkOn::Thread | fiber_rt::ParkOn::Child))
            }
            _ => false,
        };
        match self {
            Task::Child(t) => t.holds_lanes = keep,
            Task::Handler(h) => h.holds_lanes &= keep,
        }
        keep
    }

    /// #2695 — the task, no longer carried: the pool runs it ([`ChildExec::uncarry`]).
    fn take_carrier(mut self: Box<Self>) -> Box<Self> {
        if let Task::Handler(h) = &mut *self {
            h.carrier = None;
        }
        self
    }

    /// The window of a handler.
    fn window(&self) -> Option<&Arc<Window>> {
        match self {
            Task::Child(_) => None,
            Task::Handler(h) => Some(&h.window),
        }
    }

    fn vm(&self) -> &Arc<VmCtx> {
        match self {
            Task::Child(t) => &t.vm,
            Task::Handler(h) => &h.vm,
        }
    }

    /// The task's own thread domain, whose parked vCPUs a poison or a kill must wake.
    fn dom(&self) -> Option<&Arc<Domain>> {
        match self {
            Task::Child(t) => t.dom.as_ref(),
            Task::Handler(_) => None,
        }
    }

    fn retiring(&self) -> bool {
        matches!(self, Task::Child(t) if t.retiring)
    }

    fn threads_live(&self) -> bool {
        matches!(self, Task::Child(t) if t.threads_live())
    }

    /// What a run's teardown does with the task, found parked: a handler's park may never come
    /// back, so it is poisoned.
    fn at_teardown(&self) -> AtTeardown {
        match self {
            Task::Child(t) => t.at_teardown(),
            Task::Handler(_) => AtTeardown::Poison,
        }
    }
}

/// The per-thread state one residency seeds and restores ([`ChildExec::reside`]), borrowed from
/// either kind of task.
struct Residency<'a> {
    slot: &'a Arc<FiberSlot>,
    /// The fiber the slot holds, which the residency resumes: its task's, owned by this thread.
    fib: *mut Fiber,
    rt: &'a mut FiberRuntime,
    fault: (usize, usize),
    vm: &'a Arc<VmCtx>,
    /// The `vcpu.tls` word and serve-handler count the residency starts with.
    tls: i64,
    serve_handlers: u32,
    /// The thread domain the task's thread thunks act in (null: the vmctx's).
    domain: *const Domain,
    /// The durable shadow region the task spills into, if it is durable.
    shadow: Option<u64>,
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
    // SAFETY: `a` is the `&mut ResumeCall` the resumer passed; `fib` is its task's fiber.
    unsafe {
        let c = a as *mut ResumeCall;
        (*c).state = Some((*(*c).fib).resume(0));
    }
}

/// Per-task executor bookkeeping. `task` is `None` while a worker holds the task (running).
struct Entry {
    task: Option<Box<Task>>,
    /// Parked (yielded on an event park) and not yet woken: what a wake re-offers and the teardown
    /// poisons; **not** the same question as [`Self::counted`].
    parked: bool,
    /// #1631 — whether this park was counted in `Domain::parked`. A park that carries its own
    /// deadline is not, because that counter's one reader is the futex deadlock predicate
    /// `live > parked` and a task sleeping on a deadline is a potential notifier, not a blocked
    /// one. Tracked separately so the wake decrements exactly what the park incremented: keying
    /// the decrement off `parked` instead would underflow the counter on every timed park.
    counted: bool,
    /// #2012 — the deadline this park carries, if any, so a timed wait inside a child ends on time:
    /// the pool fires it between tasks, at a quantum's end, or waking idle for it
    /// ([`ExecState::next_due`]). A park with no deadline is re-offered only by a wake (#2757).
    deadline: Option<Instant>,
    /// A wake arrived (possibly while the task was still running toward its park).
    woken: bool,
    /// #1820 — a kill reached it ([`ChildExec::kill`]); counted in the run's `Domain::killing`
    /// until the task leaves the table.
    killed: bool,
    /// #1469 — the task's own domain, reachable while a worker holds the task: a poisoned task's
    /// parked vCPUs must be woken to observe it.
    dom: Option<Arc<Domain>>,
    /// #2074 — the task's instance (its trap cell in field 0), reachable while a worker holds the
    /// task, for [`ChildExec::kill`] and the run's teardown.
    vm: Arc<VmCtx>,
    /// #2695 — a held domain's activation's caller, which takes it back each time it is runnable:
    /// no worker picks it ([`Carrier`]).
    carrier: Option<Arc<Carrier>>,
}

struct ExecState {
    tasks: HashMap<u64, Entry>,
    /// Runnable tasks, FIFO. A task whose lane is full stays here — [`ChildExec::pick`] scans past
    /// it rather than shuffling it to a second queue, so there is one runnable set, not two.
    runnable: VecDeque<u64>,
    next_id: u64,
    workers: Vec<std::thread::JoinHandle<()>>,
    /// Workers not running a task: waiting for one, back from one, or spawned and not yet started.
    /// A task queued while none is free adds one, up to the pool's cap (#1991).
    free_workers: usize,
    /// #1991 — no later than the earliest deadline a parked task carries: when it passes, a worker
    /// fires the tasks due. A task woken before its deadline may leave it early, which a firing that
    /// finds nothing due corrects.
    next_due: Option<Instant>,
    /// Run teardown began: a park now poisons the task (it unwinds through its trailing guard),
    /// and workers exit once no task remains.
    shutdown: bool,
    /// #2742 — each serving vCPU's window with a handler started or open, by key.
    windows: HashMap<usize, WindowCount>,
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
    /// #1991 — the most workers the pool holds: the host's driver's (`GrantChildHooks::workers`),
    /// the cores until the run's hooks say.
    workers: AtomicUsize,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// How many workers the pool may hold on account of a task with lane chain `chain`, among `tasks`
/// filed, in a pool of at most `workers`: the widest lane it could run in (its tightest enclosing
/// cap), or one per task when its chain is unbounded. A task takes one slot of its own lane whatever
/// happens, so only the *enclosing* lanes bound how many run at once.
fn ceiling(chain: &[(usize, i64)], tasks: usize, workers: usize) -> usize {
    let enclosing = chain.len().saturating_sub(1);
    chain[..enclosing]
        .iter()
        .filter(|&&(_, cap)| cap >= 0)
        .map(|&(_, cap)| cap as usize)
        .min()
        .unwrap_or(tasks)
        .min(workers)
}

/// The host's cores, the pool's cap until a run's hooks name the driver's workers: what
/// `temen_interp::pool_workers` answers. Read once, since the read can go to cgroup files.
fn cores() -> usize {
    static CORES: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CORES.get_or_init(|| {
        std::thread::available_parallelism()
            .map_or(1, |n| n.get())
            .clamp(1, 32)
    })
}

/// #1991 — a pool task's quantum has ended: a chunk of its fuel ran out (`fuel::fuel_refill`). The
/// pool's due wakes run first ([`ChildExec::due`]). Then the task at the head of the queue, if one
/// waits, gets a worker ([`ChildExec::find_worker`]): a free one, or a new one up to the pool's cap.
/// Only a full pool, every worker busy, gives it this one: the task parks as runnable, at the back
/// of the queue, whatever fiber its quantum ended in ([`fiber_rt::ParkOn::Yield`]). Outside a
/// residency (a run's root, a 1:1 vCPU of a child sharing its cell) there is no quantum to end.
#[inline(never)] // R2 (#1466): a TLS read on a task's fiber stack
pub(crate) fn quantum_end() {
    let exec = RUNNING_FOR.with(Cell::get);
    if exec.is_null() {
        return;
    }
    {
        // SAFETY: a residency publishes its worker's reference, which outlives the residency.
        let exec = unsafe { &*exec };
        let mut g = lock(&exec.state);
        exec.due(&mut g, Instant::now());
        let Some(id) = g.runnable.front().copied() else {
            return;
        };
        if exec.find_worker(&mut g, id) {
            return;
        }
    }
    if let Some(slot) = fiber_rt::current_fiber_slot() {
        // SAFETY: inside the task's residency, on one of its fibers; the park hands the worker back
        // and returns when a worker resumes the task.
        unsafe { fiber_rt::fiber_event_park(&slot, None, fiber_rt::ParkOn::Yield) };
    }
}

impl ChildExec {
    pub(crate) fn new(dom: usize) -> Arc<ChildExec> {
        Arc::new(ChildExec {
            state: Mutex::new(ExecState {
                tasks: HashMap::new(),
                runnable: VecDeque::new(),
                next_id: 1,
                workers: Vec::new(),
                free_workers: 0,
                next_due: None,
                shutdown: false,
                windows: HashMap::new(),
            }),
            cv: Condvar::new(),
            quiescent: Condvar::new(),
            dom,
            workers: AtomicUsize::new(cores()),
        })
    }

    /// #1991 — hold no more than `n` workers (at least one): the host's driver's.
    pub(crate) fn set_workers(&self, n: usize) {
        self.workers.store(n.max(1), Ordering::Relaxed);
    }

    /// [`ceiling`] in this pool.
    fn ceiling(&self, chain: &[(usize, i64)], tasks: usize) -> usize {
        ceiling(chain, tasks, self.workers.load(Ordering::Relaxed))
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
                        task._code.fn_table.base() as u64,
                        Arc::as_ptr(&task.vm) as u64,
                    )
                };
            if !seeded {
                task.vm.raise(TrapKind::FiberFault as i64);
            }
        }
        if task._code.spawns || durable {
            if let Some(hub) = self.domain() {
                let d = Arc::new(Domain::new_child(
                    hub,
                    task.chain.clone(),
                    task.node.take(),
                    Arc::clone(&task.children),
                ));
                d.set_env(
                    task.window.base() as u64,
                    task._code.fn_table.base() as u64,
                    Arc::as_ptr(&task.vm), // the child's vCPUs are its instance
                    task._code.call_tramp,
                    task.fault,
                    task._code.fiber_cfg,
                    Some(task.rt.table()),
                    task._code.instance.epoch as usize,
                    durable,
                    task.rt.table().shadow,
                );
                // #2010 — a durable child's vCPUs each reserve a shadow context of their own (as every
                // durable spawn does, #2543), so a freeze of its window unwinds each into its own
                // region, wherever its root is. A thaw re-spawns the ones its capture carried,
                // rewinding, before its root runs.
                if durable && !thaw.vcpus.is_empty() {
                    // An extent-less capture is an empty root's, as for a run's root.
                    let root_sp = thaw.root_sp.unwrap_or(d.shadow().frame_base(0));
                    // SAFETY: the env just set is the task's: its window (seeded with its image)
                    // and code live until the task finishes, after its vCPUs are joined.
                    unsafe {
                        d.thaw_reattach_and_run(&thaw.vcpus, root_sp, &|f| {
                            task._code.thawed_spawn_entry(f)
                        })
                    };
                }
                task.dom = Some(d);
            }
        }
        let mut g = lock(&self.state);
        let id = g.next_id;
        g.next_id += 1;
        // #2757 — a durable child's freeze doorbell wakes this task. Registered under this lock,
        // before the task is runnable, so before it can park.
        if let Some(d) = task.done.durable.as_ref() {
            d.set_waker(self.waker(id));
        }
        let dom = task.dom.clone();
        let vm = Arc::clone(&task.vm);
        g.tasks.insert(
            id,
            Entry {
                task: Some(Box::new(Task::Child(task))),
                parked: false,
                deadline: None,
                counted: false,
                woken: false,
                killed: false,
                dom,
                vm,
                carrier: None,
            },
        );
        g.runnable.push_back(id);
        self.find_worker(&mut g, id);
        id
    }

    /// #2742 — run handler `h` here, on the thread that holds its window's gate: the serve op's, or
    /// a caller's that found the serve op parked. A handler that ends here, the common call, never
    /// meets the pool: its runner is a live vCPU, which the deadlock predicate counts already. One
    /// that parks or is preempted is promoted ([`Self::promote`]).
    pub(crate) fn run_here(
        self: &Arc<Self>,
        mut h: HandlerTask,
        me: u64,
        make_done: &mut dyn FnMut() -> HandlerDone,
    ) -> Ran {
        // The starter's gate (`me`'s) goes to the handler for its run, and comes back if it ends
        // here. Its domain's end waits for it from now, wherever it ends.
        let handed = h.window.hand(me, h.id());
        debug_assert!(handed, "a handler starts under its starter's gate");
        h.window.vm.handlers().fetch_add(1, Ordering::SeqCst);
        h.holds_lanes = true;
        let outcome = self.run_handler(&mut h, false);
        match outcome {
            Outcome::Done | Outcome::Finished => {
                let reusable = h.reusable;
                let held = h.window.hand(h.id(), me);
                let (end, window) = h.conclude(reusable);
                self.handler_ended(&window.vm);
                Ran {
                    held,
                    ended: true,
                    end,
                }
            }
            // It keeps the gate for a quantum's end or a whole-vCPU park, and gives it back for any
            // other (its park gave it back already, unless it parked again within the same wait).
            Outcome::Parked { .. } | Outcome::Yielded => {
                h.done = Some(make_done());
                let (window, hid) = (Arc::clone(&h.window), h.id());
                let mut task = Box::new(Task::Handler(h));
                if !task.keep_lanes(&outcome) {
                    window.release(hid);
                }
                self.promote(task);
                Ran {
                    held: false,
                    ended: false,
                    end: None,
                }
            }
            Outcome::Retiring => unreachable!("a handler never retires"),
        }
    }

    /// #2742 — a handler that parked (or was preempted) on the thread that started it becomes a task
    /// of the pool, one of its window's handlers until it ends. It is filed runnable, to take its
    /// park again from the pool: a wake that came while it ran unfiled latched on nothing, so it
    /// re-checks what it waits for once more, as a woken task does.
    fn promote(self: &Arc<Self>, task: Box<Task>) {
        let Task::Handler(h) = &*task else {
            unreachable!("only a handler is promoted");
        };
        let mut g = lock(&self.state);
        let id = self.admit(&mut g, h);
        g.tasks.get_mut(&id).expect("admitted").task = Some(task);
        g.runnable.push_back(id);
        self.offer(&mut g, id);
    }

    /// #2742 — handler `h` joins the pool's tasks, under its lock (`g`), one of its window's handlers
    /// until it ends: counted as the window's, and in the run's live tasks while the window counts.
    /// Its entry holds no task: its caller files it.
    fn admit(&self, g: &mut ExecState, h: &HandlerTask) -> u64 {
        h.window.pooled.fetch_add(1, Ordering::AcqRel);
        let id = g.next_id;
        g.next_id += 1;
        // #2845 — one whose domain has ended meanwhile (it parked on a caller's thread while a trap
        // or the root's end and its drain ran) is drained on arrival: its window never opens again,
        // and the domain's drain, which takes this lock after it records the end, may have run
        // without it, so nothing else would run it to unwind.
        let w = g
            .windows
            .entry(h.window.key())
            .or_insert_with(|| WindowCount {
                open: h.window.open.load(Ordering::Acquire),
                draining: h.window.vm.ended().is_some(),
                ..WindowCount::default()
            });
        w.domain = h.window.vm.domain_key();
        w.handlers += 1;
        if w.counts() {
            if let Some(d) = self.domain() {
                d.handler_started();
            }
        }
        g.tasks.insert(
            id,
            Entry {
                task: None,
                parked: false,
                deadline: None,
                counted: false,
                woken: false,
                killed: false,
                dom: None,
                vm: Arc::clone(&h.vm),
                carrier: h.carrier.clone(),
            },
        );
        id
    }

    /// #2742 — a woken handler of window `key`, taken out of the queue for the serve op, which holds
    /// the gate, to resume itself ([`ServeSite::resume_woken`]).
    fn take_woken(&self, key: usize) -> Option<(u64, Box<Task>)> {
        let mut g = lock(&self.state);
        let ExecState {
            tasks, runnable, ..
        } = &mut *g;
        let at = runnable.iter().position(|id| {
            let e = tasks.get(id).filter(|e| e.carrier.is_none());
            let t = e.and_then(|e| e.task.as_ref());
            t.and_then(|t| t.window()).is_some_and(|w| w.key() == key)
        })?;
        let id = runnable.remove(at).expect("position is in range");
        let e = tasks.get_mut(&id).expect("queued");
        e.woken = false;
        Some((id, e.task.take().expect("queued tasks are held")))
    }

    /// #2742 — one residency of handler task `id` here, lent the caller's gate. Its end runs its
    /// `done`; a park files it with the pool. One that keeps the gate (a quantum's end, a whole-vCPU
    /// park) takes it from the caller, and gives it back when it lets go of it on a worker.
    fn reside_here(self: &Arc<Self>, id: u64, task: Box<Task>) -> Ran {
        self.reside_on(id, task, true)
    }

    /// [`Self::reside_here`], `hand`ing the handler its resumer's gate or not: a carried handler
    /// that kept its gate through its park resumes with it, and one unwinding at its domain's end
    /// runs without one, as the pool runs either ([`Self::pick`]).
    fn reside_on(self: &Arc<Self>, id: u64, mut task: Box<Task>, hand: bool) -> Ran {
        let me = fiber_rt::context_id();
        let Task::Handler(h) = &mut *task else {
            unreachable!("only a handler is resumed here");
        };
        let (window, hid) = (Arc::clone(&h.window), h.id());
        if hand {
            let handed = window.hand(me, hid);
            debug_assert!(
                handed,
                "a woken handler is resumed under its resumer's gate"
            );
            h.holds_lanes = true;
        }
        let prev = RUNNING_TASK.with(|c| c.replace(id));
        let outcome = self.run_handler(h, false);
        RUNNING_TASK.with(|c| c.set(prev));
        let keep = task.keep_lanes(&outcome);
        let ended = matches!(outcome, Outcome::Finished | Outcome::Done);
        // One that ended gives the gate back to its resumer before its end is told; one that parked
        // gives it back, as on a worker, unless it keeps it.
        let held = ended && window.hand(hid, me);
        if !ended && !keep {
            window.release(hid);
        }
        if let Some(task) = self.file(id, task, outcome, false) {
            self.finish(task);
        }
        Ran {
            held,
            ended,
            end: None,
        }
    }

    /// #2695 — carried handler `id`, if it is runnable: taken out of the queue for its carrier to resume
    /// ([`HeldSite::activate`]). `None` while it is parked, or held by its carrier already.
    fn take_carried(&self, id: u64) -> Option<Box<Task>> {
        let mut g = lock(&self.state);
        let ExecState {
            tasks, runnable, ..
        } = &mut *g;
        let at = runnable.iter().position(|&r| r == id)?;
        runnable.remove(at);
        let e = tasks.get_mut(&id).expect("queued");
        e.woken = false;
        e.task.take()
    }

    /// #2695 — carried handler `id`'s carrier is gone (its caller's domain ended while it waited): the
    /// pool runs it on from here, as one it holds, so it still unwinds at its domain's end.
    fn uncarry(self: &Arc<Self>, id: u64, taken: Option<Box<Task>>) {
        let mut g = lock(&self.state);
        let Some(e) = g.tasks.get_mut(&id) else {
            return;
        };
        e.carrier = None;
        if let Some(task) = taken {
            // Taken back runnable, and given up before it ran: runnable in the pool's queue again.
            e.task = Some(task.take_carrier());
            g.runnable.push_back(id);
        } else if let Some(Task::Handler(h)) = e.task.as_deref_mut() {
            h.carrier = None;
        }
        if g.runnable.contains(&id) {
            self.find_worker(&mut g, id);
        }
    }

    /// #2695 — enlist handler `h`, a held domain's activation its caller carries, before its first
    /// residency ([`Self::admit`]): held by its carrier, not queued. Its domain's end waits for it
    /// from now, wherever it ends.
    fn enlist(self: &Arc<Self>, h: HandlerTask) -> (u64, Box<Task>) {
        h.vm.handlers().fetch_add(1, Ordering::SeqCst);
        let id = self.admit(&mut lock(&self.state), &h);
        (id, Box::new(Task::Handler(h)))
    }

    /// #2742 — handler `hid`'s worker gives its window's gate back, if the handler still holds it
    /// (its park gave it back already), and whoever waits for it is woken.
    fn release_gate(&self, window: Option<&Arc<Window>>, hid: u64) {
        if let Some(w) = window {
            w.release(hid);
        }
    }

    /// #2742 — window `key`'s gate was given back: a worker for a woken handler of it, if one waits
    /// in the queue for the gate. (A worker's own next pick re-offers the queue; a serve op or a
    /// caller giving the gate back is no worker.)
    fn offer_gate(self: &Arc<Self>, key: usize) {
        let mut g = lock(&self.state);
        let waiting = g.runnable.iter().copied().find(|id| {
            let e = g.tasks.get(id).filter(|e| e.carrier.is_none());
            let t = e.and_then(|e| e.task.as_ref());
            t.and_then(|t| t.window()).is_some_and(|w| w.key() == key)
        });
        if let Some(id) = waiting {
            self.find_worker(&mut g, id);
        }
    }

    /// #2742 — window `key`'s serve op begins (`open`) or ends: its handlers may run, and count in the
    /// deadlock predicate, only while it is open ([`WindowCount`]).
    fn set_window_open(&self, key: usize, open: bool) {
        let mut g = lock(&self.state);
        let Some(w) = g.windows.get_mut(&key) else {
            return; // its last pooled handler has just ended
        };
        let before = w.counts();
        w.open = open;
        let (counts, handlers, parked) = (w.counts(), w.handlers, w.parked);
        if let (true, Some(d)) = (counts != before, self.domain()) {
            d.count_window(counts, handlers, parked);
        }
    }

    /// `Instantiator.kill` (#2074): end task `id` with `ThreadFault`, running or parked. Its trap
    /// cell stops its running vCPUs at their next entry or back-edge poll (`emit_domain_poll`), and a
    /// parked task (or a parked vCPU of its own) is woken to observe it, as a teardown's poison is.
    /// A task that has finished is gone from the table, and nothing happens.
    pub(crate) fn kill(self: &Arc<Self>, id: u64) {
        let mut g = lock(&self.state);
        let Some(e) = g.tasks.get_mut(&id) else {
            return;
        };
        // #1820 — the run is not quiescent until this task has ended.
        if !std::mem::replace(&mut e.killed, true) {
            if let Some(d) = self.domain() {
                d.kill_in_flight(true);
            }
        }
        // Its next poll ends it, and so does a wait a parked vCPU of it re-checks; a trap the child
        // already raised outranks the kill (#2499).
        e.vm.raise(TrapKind::ThreadFault as i64);
        self.wake_parked(&mut g, Wake::Task(id));
    }

    /// #2757 — the waker of task `id`: what a store that can end its park fires.
    fn waker(self: &Arc<Self>, id: u64) -> TaskWaker {
        let exec = Arc::downgrade(self);
        Arc::new(move || {
            if let Some(exec) = exec.upgrade() {
                exec.wake_locked(Wake::Task(id));
            }
        })
    }

    /// A worker for task `id`, queued: a free one, woken, or else a new one, unless the pool
    /// already holds as many as the task could use ([`Self::ceiling`]). `false` when it does and
    /// every one is busy.
    fn find_worker(self: &Arc<Self>, g: &mut ExecState, id: u64) -> bool {
        if g.free_workers > 0 {
            self.cv.notify_one();
            return true;
        }
        let tasks = g.tasks.len();
        let want = (g.tasks.get(&id))
            .and_then(|e| e.task.as_ref())
            .map_or(1, |t| self.ceiling(t.chain(), tasks));
        if g.workers.len() >= want {
            return false;
        }
        let me = Arc::clone(self);
        let worker = g.workers.len() as u64 + 1;
        let h = std::thread::Builder::new()
            .name("temen-child".into())
            .spawn(move || me.worker_loop(worker));
        // An `Err` is the OS refusing a thread: the existing workers (if any) still drain the
        // queue; with none, the task waits for a later spawn's worker or runs at shutdown on the
        // joining thread.
        let Ok(h) = h else {
            return false;
        };
        g.workers.push(h);
        g.free_workers += 1;
        true
    }

    /// D66 — a lane was given back somewhere in this domain (by a 1:1 vCPU, say): re-offer the
    /// queue, so a task that `pick` refused for want of a lane is reconsidered. Without this an idle
    /// worker sleeps on its condvar until something *else* wakes it, which for a run whose only
    /// other activity is the vCPU that just stepped aside is never.
    pub(crate) fn notify_workers(&self) {
        let _g = lock(&self.state);
        self.cv.notify_all();
    }

    /// Make the parked tasks `which` names runnable, each to re-check its own park predicate. This
    /// is the whole wake mechanism: a `notify`, a kill, a vCPU's or a child's end and run teardown
    /// all reach it through [`Domain::wake_child_tasks`], passed deadlines ([`Self::due`]) through it
    /// too, and a store that can end one task's park through [`Wake::Task`] (#2757). A task that re-checks and is still unsatisfied
    /// simply parks again.
    pub(crate) fn wake_locked(self: &Arc<Self>, which: Wake) {
        let mut g = lock(&self.state);
        self.wake_parked(&mut g, which);
    }

    /// #2757 — wake the vCPUs parked in every task's domain, each to re-check what it waits on,
    /// once the caller has woken the run's own (`Domain::wake_own_parked`): a task domain's futex
    /// waits and lanes are its hub's, woken with them, so only its joiners are left. Under the
    /// executor's lock, which a task domain's wake may take after its own: never the other way round.
    fn wake_task_vcpus(g: &ExecState) {
        for d in g.tasks.values().filter_map(|e| e.dom.as_ref()) {
            d.wake_joiners();
        }
    }

    /// #2757 — a task's park, or a killed task's end, may have left the run quiescent: wake the
    /// vCPU parks that decide the deadlock verdict, on the run's domain and in every task's, as a
    /// vCPU's own park needs not (it decides the verdict itself). Tasks are not re-offered: a task
    /// cannot decide the verdict, and one re-offered would only park again.
    fn wake_vcpus_if_quiescent(&self, g: &ExecState) {
        if let Some(d) = self.domain().filter(|d| !d.peers_live()) {
            d.wake_own_parked();
            Self::wake_task_vcpus(g);
        }
    }

    /// Make the parked tasks `which` names runnable.
    fn wake_parked(self: &Arc<Self>, g: &mut ExecState, which: Wake) {
        match which {
            // #1711 — a running task may already be past its predicate check and yielding toward a
            // park that the worker has not filed yet. Latch the wake on it too, so the worker
            // requeues it instead of parking it on a cell this wake has already satisfied. At worst
            // the task re-checks once more than it needed to. [`Wake::Resolved`] needs no latch:
            // the worker files no futex park whose wait has fired (`worker_loop`).
            Wake::All => {
                for e in g.tasks.values_mut().filter(|e| e.task.is_none()) {
                    e.woken = true;
                }
            }
            Wake::Run => {
                for e in g.tasks.values_mut().filter(|e| e.task.is_none()) {
                    e.woken = true;
                }
                Self::wake_task_vcpus(g);
            }
            // #2757 — the one task a store names: latched the same way, and its own parked vCPUs
            // woken, since the store may end their waits too.
            Wake::Task(id) => {
                if let Some(e) = g.tasks.get_mut(&id) {
                    if e.task.is_none() {
                        e.woken = true;
                    }
                    if let Some(d) = &e.dom {
                        d.wake_own_parked();
                    }
                }
            }
            Wake::Due(_) | Wake::Resolved => {}
        }
        let ids: Vec<u64> = g
            .tasks
            .iter()
            .filter(|(id, e)| {
                e.parked
                    && match which {
                        Wake::All | Wake::Run => true,
                        Wake::Task(task) => **id == task,
                        Wake::Due(now) => e.deadline.is_some_and(|d| d <= now),
                        Wake::Resolved => e.task.as_ref().is_some_and(|t| t.slot().park_resolved()),
                    }
            })
            .map(|(&id, _)| id)
            .collect();
        for id in ids {
            let ExecState { tasks, windows, .. } = &mut *g;
            let e = tasks.get_mut(&id).expect("listed");
            e.parked = false;
            e.deadline = None;
            e.woken = true;
            // #1631 — decrement exactly what the park incremented (see `Entry::counted`); a park
            // with a deadline is never counted, so it has nothing to undo. #2742 — a handler's
            // counts through its window, and in the domain only while the window is open.
            if std::mem::take(&mut e.counted) {
                let window = e.task.as_ref().and_then(|t| t.window()).map(|w| w.key());
                let open = window.is_none_or(|k| {
                    let w = windows.get_mut(&k).expect("a handler's window is counted");
                    w.parked -= 1;
                    w.counts()
                });
                if let (true, Some(d)) = (open, self.domain()) {
                    d.task_unparked();
                }
            }
            g.runnable.push_back(id);
            self.offer(g, id);
        }
        // What is left parked bounds the next firing (`ExecState::next_due`), once a wake or a
        // firing has woken what was due.
        if !matches!(which, Wake::Resolved) {
            g.next_due = (g.tasks.values())
                .filter(|e| e.parked)
                .filter_map(|e| e.deadline)
                .min();
        }
    }

    /// #1991 — the pool's own wakes, due at `now`: each parked task whose deadline has passed. A
    /// worker fires them between tasks, at a quantum's end ([`quantum_end`]) and when it wakes idle,
    /// so they come within a quantum whatever runs. The pool sweeps nothing else (#2757): a park
    /// with no deadline is re-offered only by a wake.
    fn due(self: &Arc<Self>, g: &mut ExecState, now: Instant) {
        if g.next_due.is_some_and(|d| d <= now) {
            self.wake_parked(g, Wake::Due(now));
        }
    }

    /// #1991 — finds a worker for task `id`, just queued ([`Self::find_worker`]), unless the waker's
    /// own worker will take it. A task that wakes one sibling leaves it to its own worker, which runs
    /// it next, once the task parks or ends: a request and its reply stay on one worker, and no
    /// thread is woken. Waking another worker cost each hop of a keystroke through unir's store vat a
    /// thread's wake (98 µs a keystroke, where one worker took 53). A waker that runs on to the end
    /// of its quantum finds the sibling a worker then ([`quantum_end`]). A second task queued
    /// meanwhile, or a task queued from outside any task (the root's thread, a host's, a worker
    /// firing a deadline), finds one at once, which runs it beside the waker. Waking every free worker at each
    /// wake, as before, cost one switch per free worker.
    fn offer(self: &Arc<Self>, g: &mut ExecState, id: u64) {
        // #2695 — a carried handler goes to its carrier, never a worker.
        if let Some(carrier) = g.tasks.get(&id).and_then(|e| e.carrier.clone()) {
            carrier.wake(self, g);
            return;
        }
        if g.runnable.len() == 1 && self.on_own_worker() {
            return;
        }
        self.find_worker(g, id);
    }

    /// Whether this thread is one of this pool's workers in a residency, so it takes the next
    /// runnable task once its own parks or ends ([`Self::offer`]). #2742 — not a thread that runs a
    /// handler it admitted (a serve op's, a caller's): that thread takes no task of the pool's.
    fn on_own_worker(self: &Arc<Self>) -> bool {
        let p = RUNNING_FOR.with(Cell::get);
        // SAFETY: a residency publishes its worker's reference, which outlives the residency.
        !p.is_null() && Arc::ptr_eq(unsafe { &*p }, self)
    }

    /// The next runnable task whose whole lane chain has room, taking its lanes. A task refused a
    /// lane keeps its place in the queue; a later release simply makes the next scan admit it.
    fn pick(&self, g: &mut ExecState) -> Option<u64> {
        // The lane counts live on the `Domain`, shared with its own 1:1 vCPUs — a task of this
        // subtree and a `thread.spawn` vCPU of the parent draw on the *same* lane, so a private map
        // here would enforce the cap twice and hand out double the granted parallelism. Taking the
        // domain's lane lock while holding this one is the one lock order used anywhere; nothing
        // takes this lock while holding that one.
        // A task its quantum preempted still holds its lanes, and takes none. #2742 — a handler
        // runs only while its window is open, under its gate; the run's teardown runs it regardless,
        // poisoned, to unwind.
        let shutdown = g.shutdown;
        // A handler unwinding at its domain's end or the run's: no window, no gate.
        let unwinding = |t: &Task| {
            t.window()
                .is_some_and(|w| shutdown || g.windows.get(&w.key()).is_some_and(|w| w.draining))
        };
        // A handler's gate is its window's own: taken here for the handler `position` admits, the
        // first one taken (a handler's lane chain is empty, so the lanes admit it at once).
        let mut chains = g.runnable.iter().map(|id| {
            let e = g.tasks.get(id).filter(|e| e.carrier.is_none());
            e.and_then(|e| e.task.as_ref()).and_then(|t| {
                if let (Some(w), false) = (t.window(), t.holds_lanes() || unwinding(t)) {
                    let hid = Arc::as_ptr(t.slot()) as u64;
                    if !g.windows[&w.key()].open || !(w.held_by(hid) || w.take(hid)) {
                        return None;
                    }
                }
                Some(if t.holds_lanes() { &[][..] } else { t.chain() })
            })
        });
        let admit = match self.domain() {
            Some(d) => d.lane_try_enter_first(chains)?,
            None => chains.position(|c| c.is_some())?, // no domain ⇒ no lanes to honour (the durable nested nursery)
        };
        let id = g.runnable.remove(admit).expect("position is in range");
        let ExecState { tasks, windows, .. } = &mut *g;
        let e = tasks.get_mut(&id).expect("scanned");
        e.woken = false;
        // A handler picked to run, not to unwind, holds its gate now.
        if let Some(Task::Handler(h)) = e.task.as_deref_mut() {
            h.holds_lanes |= !shutdown && !windows[&h.window.key()].draining;
        }
        Some(id)
    }

    fn worker_loop(self: Arc<Self>, worker: u64) {
        mem::install_guard();
        loop {
            let (id, mut task) = {
                let mut g = lock(&self.state);
                let id = loop {
                    // #2757 — a parked task is re-offered only by a wake (a `notify`, a kill or a
                    // freeze through the run's doors, a vCPU's or a child's end, teardown, its freeze
                    // doorbell) or by its own deadline (#2012), fired as it passes: the pool runs
                    // no cadence. #1991 — a worker fires due deadlines before each task it takes,
                    // not only when idle, so they come due however busy the pool is.
                    self.due(&mut g, Instant::now());
                    if let Some(id) = self.pick(&mut g) {
                        break id;
                    }
                    if g.shutdown && g.tasks.is_empty() {
                        g.free_workers -= 1;
                        return;
                    }
                    // Idle: wake for the earliest parked deadline, if a parked task carries one;
                    // else only when a wake queues something.
                    g = match g.next_due {
                        Some(next) => {
                            let wait = next.saturating_duration_since(Instant::now());
                            let (g, _) = self
                                .cv
                                .wait_timeout(g, wait)
                                .unwrap_or_else(|e| e.into_inner());
                            g
                        }
                        None => self.cv.wait(g).unwrap_or_else(|e| e.into_inner()),
                    };
                };
                let task = g
                    .tasks
                    .get_mut(&id)
                    .and_then(|e| e.task.take())
                    .expect("picked task is present");
                g.free_workers -= 1;
                (id, task)
            };
            // A retiring task is never resumed: it is offered only to re-check its vCPUs.
            let outcome = if task.retiring() {
                Outcome::Retiring
            } else {
                let prev = RUNNING_TASK.with(|c| c.replace(id));
                let outcome = self.run_once(&mut task, worker, true);
                RUNNING_TASK.with(|c| c.set(prev));
                outcome
            };
            // The lane goes back before anything else: a peer 1:1 vCPU or sibling task may be
            // queued behind it. `lane_leave` touches only the domain's lane lock, so it is safe to
            // call before taking this executor's. It wakes no worker of this pool: this worker's
            // own next `pick` re-offers the queue. A task its quantum preempted keeps its lanes.
            // #2742 — a handler that parked gives its gate back before its park is counted (its park
            // gave it back already, unless it parked again within the same wait), else a waiter for
            // the gate could read every live task parked while the wake that frees it is still to
            // come, a false deadlock verdict. One that ended gives it back as it finishes, once its
            // dispatch is settled and counted, and before it leaves the live count
            // ([`Self::finish_handler`]). One its quantum preempted, or parked on a whole-vCPU
            // wait, keeps it.
            let held = task.holds_lanes();
            let keep = task.keep_lanes(&outcome);
            let gate = task.window().filter(|_| held && !keep).map(Arc::clone);
            let hid = Arc::as_ptr(task.slot()) as u64;
            let ends = matches!(outcome, Outcome::Finished | Outcome::Done);
            if !keep && task.window().is_none() {
                if let Some(d) = self.domain() {
                    d.lane_leave(task.chain());
                }
            }
            if !ends {
                self.release_gate(gate.as_ref(), hid);
            }
            // No wake here (#1991): this worker takes the next runnable task itself, a task it
            // requeued included, and every other task was queued with a wake of its own. A lane it
            // gave back admits at most one more task, which this worker's own `pick` takes.
            if let Some(task) = self.file(id, task, outcome, true) {
                self.finish(task);
            }
        }
    }

    /// After a residency of task `id` that ended `outcome`, on a pool worker (`worker`, which is
    /// free again) or on the thread a handler started on ([`Self::run_here`]): file the task by its
    /// outcome — runnable behind the queue, parked until a wake, retiring — or hand it back,
    /// finished, for [`Self::finish`].
    fn file(
        self: &Arc<Self>,
        id: u64,
        mut task: Box<Task>,
        outcome: Outcome,
        worker: bool,
    ) -> Option<Box<Task>> {
        // #1469 — a root that returns while its vCPUs run publishes its outcome now and
        // retires; a retiring task finishes once they have all ended. #2742 — and once the handlers
        // of its serve loops have: none of them can run again, so once its vCPUs have ended they
        // are ended too ([`Self::drain`]).
        // The look marks the domain ending, so the last of its handlers to end re-offers it, whether
        // the pool holds it or it runs on a caller's stack or thread.
        let handlers = match &*task {
            Task::Child(t) if matches!(outcome, Outcome::Finished | Outcome::Retiring) => {
                t.vm.handlers().fetch_or(HANDLERS_ENDING, Ordering::SeqCst) & !HANDLERS_ENDING != 0
            }
            _ => false,
        };
        // #2742 — the root has ended, however it ended (owner, 2026-11-09): the handlers it parked
        // can never run again, so their calls answer `CAP_REVOKED` now ([`ServeSite::revoke`]).
        if let (Task::Child(t), Outcome::Finished) = (&*task, &outcome) {
            if let Some(site) = t.vm.serve_site.get() {
                site.revoke();
            }
        }
        // Once none of its vCPUs is left, the domain's handlers run once more, to unwind on its end
        // ([`Self::drain`]): a trap's or a kill's, or, for a root that returned, `DOMAIN_DONE`,
        // which nothing of the domain is left to see but them. They count as live from before the
        // root leaves the count, so no waiter sees the domain all parked while they still have to
        // unwind.
        if handlers && !task.threads_live() {
            if task.vm().ended().is_none() {
                task.vm().raise(crate::DOMAIN_DONE_CODE as i64);
            }
            self.drain(task.vm().domain_key());
        }
        let outcome = match outcome {
            Outcome::Finished if task.threads_live() || handlers => {
                if let Task::Child(t) = &mut *task {
                    self.settle(t);
                }
                Outcome::Retiring
            }
            Outcome::Retiring if !task.threads_live() && !handlers => Outcome::Finished,
            o => o,
        };
        // #1820 — a join counts itself through its completion cell, so it is not counted here;
        // #2424 — nor is a wait on an answer from outside the run, which is not blocked.
        let uncounted = matches!(outcome, Outcome::Parked { .. }) && {
            let on = task.slot().parked_on();
            on.counts_itself() || matches!(on, fiber_rt::ParkOn::Outside)
        };
        let window = task.window().map(|w| w.key());
        let mut task = Some(task);
        let mut g = lock(&self.state);
        // Back from the task, free to take the next: a task this filing queues needs no worker. A
        // handler's admitter is no worker (#2742): a task it queues needs one ([`Self::offer`]).
        if worker {
            g.free_workers += 1;
        }
        let queued = g.runnable.len();
        match outcome {
            // #1991 — behind the task its quantum's end found waiting.
            Outcome::Yielded => {
                let e = g.tasks.get_mut(&id).expect("running task is filed");
                e.task = task.take();
                g.runnable.push_back(id);
            }
            Outcome::Parked { deadline } => {
                let shutdown = g.shutdown;
                // A counted park may leave the run quiescent (#2757).
                let mut quiet = false;
                let ExecState {
                    tasks,
                    windows,
                    runnable,
                    next_due,
                    ..
                } = &mut *g;
                let e = tasks.get_mut(&id).expect("running task is filed");
                if shutdown {
                    // Teardown: a park now would wait for a wake that can never come. Poison the
                    // task's cell so its wait returns and the trailing guard unwinds it — unless
                    // it parked after a freeze rang it, when its re-check ends its wait, or it
                    // waits on a child that freeze rang (`at_teardown`).
                    let t = task.as_ref().expect("held");
                    match t.at_teardown() {
                        AtTeardown::Poison => {
                            t.vm().raise(crate::DOMAIN_DONE_CODE as i64);
                            if let Some(d) = t.dom() {
                                d.wake_own_parked(); // its vCPUs share the cell
                            }
                            e.woken = true;
                        }
                        AtTeardown::Wake => e.woken = true,
                        AtTeardown::Wait => {}
                    }
                }
                // #1991 — a futex wait a `notify` claimed while the task was on its way here is
                // runnable already: that notify's wake ([`Wake::Resolved`]) found it running and
                // latched nothing, and found no park to end.
                let resolved = task.as_ref().is_some_and(|t| t.slot().park_resolved());
                e.task = task.take();
                if e.woken || resolved {
                    runnable.push_back(id);
                } else {
                    e.parked = true;
                    e.deadline = deadline;
                    // #1631 — a park that wakes on its own deadline is a potential notifier, so
                    // it stays out of the deadlock predicate's count. It is still `parked`, and
                    // its deadline is fired as it passes.
                    if deadline.is_none() && !uncounted {
                        e.counted = true;
                        // #2742 — a handler's, through its window, which is open: a handler runs only
                        // in an open window ([`WindowCount`]).
                        let open = window.is_none_or(|k| {
                            let w = windows.get_mut(&k).expect("a handler's window is counted");
                            w.parked += 1;
                            w.counts()
                        });
                        if let (true, Some(d)) = (open, self.domain()) {
                            d.task_parked();
                            quiet = true;
                        }
                    }
                    if let Some(d) = deadline {
                        *next_due = Some(next_due.map_or(d, |n| n.min(d)));
                    }
                }
                if quiet {
                    self.wake_vcpus_if_quiescent(&g);
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
                        t.vm().raise(crate::DOMAIN_DONE_CODE as i64);
                    }
                    if let Some(d) = t.dom() {
                        d.wake_own_parked();
                    }
                }
                e.task = Some(t);
                if e.woken {
                    g.runnable.push_back(id);
                } else {
                    // Parked until a wake (its vCPUs' exits re-offer it), never counted: the
                    // task's root is no longer a vCPU (`settle` dropped it from `live`).
                    e.parked = true;
                }
            }
            Outcome::Finished | Outcome::Done => {
                // Every vCPU of a killed task has ended by now, its parks settled with it.
                if g.tasks.remove(&id).is_some_and(|e| e.killed) {
                    if let Some(d) = self.domain() {
                        d.kill_in_flight(false);
                    }
                    self.wake_vcpus_if_quiescent(&g);
                }
                if g.tasks.is_empty() {
                    self.quiescent.notify_all();
                }
            }
        }
        if !worker && g.runnable.len() > queued {
            self.offer(&mut g, id);
        }
        drop(g);
        task
    }

    /// One residency of `task` on this thread: a pool worker's, whose quantum ends against this pool
    /// (`pool`), or the thread a handler starts on ([`Self::run_here`]).
    fn run_once(self: &Arc<Self>, task: &mut Task, worker: u64, pool: bool) -> Outcome {
        match task {
            Task::Child(t) => {
                let r = Residency {
                    slot: &t.slot,
                    fib: t.slot.fiber_ptr().expect("a platform slot holds its fiber"),
                    rt: &mut t.rt,
                    fault: t.fault,
                    vm: &t.vm,
                    tls: t.tls,
                    serve_handlers: t.serve_handlers,
                    // #1469 — the child's thread thunks act in its own domain for this residency.
                    domain: t.dom.as_ref().map_or(std::ptr::null(), Arc::as_ptr),
                    // #1361 step 4 — a durable child spills into its own window's context-0 region.
                    shadow: t.done.durable.as_ref().map(|d| d.shadow.region_base(0)),
                };
                // D66 — the single-owner claim for the residency: back to the pool after it, or
                // retired with the task.
                assert!(
                    t.slot.claim_for_worker(worker),
                    "pool task dispatched while claimed"
                );
                let (outcome, tls, handlers) = self.reside(r, pool);
                match outcome {
                    Outcome::Finished => t.slot.finish_task(),
                    _ => t.slot.release_to_pool(),
                }
                (t.tls, t.serve_handlers) = (tls, handlers);
                outcome
            }
            Task::Handler(h) => self.run_handler(h, pool),
        }
    }

    /// #2742 — one residency of handler `h`, which runs as its serving vCPU: its `vcpu.tls` word is
    /// its window's. It takes no slot claim: the thread running it owns its task, as its window's
    /// gate admits it ([`Self::pick`], [`Self::run_here`]).
    fn run_handler(self: &Arc<Self>, h: &mut HandlerTask, pool: bool) -> Outcome {
        let r = Residency {
            slot: &h.shell.slot,
            fib: h.shell.fib,
            rt: &mut h.shell.rt,
            fault: h.fault,
            vm: &h.vm,
            tls: h
                .tls
                .unwrap_or_else(|| h.window.tls.load(Ordering::Relaxed)),
            serve_handlers: h.serve_handlers,
            domain: h.domain,
            shadow: None,
        };
        let (outcome, tls, handlers) = self.reside(r, pool);
        // #2842 — the word is the gate holder's: one that gave the gate back for a wait gave the
        // word back with it ([`handler_steps_aside`]), and whoever holds the gate now may have
        // changed it since. #2695 — a held domain's activation keeps its own.
        if let Some(own) = h.tls.as_mut() {
            *own = tls;
        } else if h.window.held_by(h.id()) {
            h.window.tls.store(tls, Ordering::Relaxed);
        }
        h.serve_handlers = handlers;
        h.reusable = matches!(outcome, Outcome::Done);
        outcome
    }

    /// The residency itself: seed this thread's per-task state, resume the task's fiber under a guard
    /// bracket over its own fault range, classify the yield, reset the thread.
    fn reside(self: &Arc<Self>, r: Residency<'_>, pool: bool) -> (Outcome, i64, u32) {
        // #2722 — a pool worker is about to resume this task's fiber (the cost a cadence sweep used
        // to spend on parked tasks that could not advance). `pool_resumes` observes it.
        if pool {
            POOL_RESUMES.fetch_add(1, Ordering::Relaxed);
        }
        let prev_rt = fiber_rt::set_current(&mut *r.rt as *mut FiberRuntime);
        // The durable shadow register is per OS thread: seed it for this residency and give the
        // thread's back after.
        let prev_shadow = r.shadow.map(|region| {
            let s = crate::durable_shadow::get();
            crate::durable_shadow::seed(region);
            s
        });
        let prev_tls = vcpu_tls::get();
        vcpu_tls::seed(r.tls);
        let prev_handlers = vcpu_tls::serve_handlers();
        vcpu_tls::set_serve_handlers(r.serve_handlers);
        let prev_dom = os_thread_rt::set_current_domain(r.domain);
        r.rt.push_active(Arc::clone(r.slot));
        // #1991 — a worker's task's quantum ends against this pool ([`quantum_end`]); a handler on
        // the thread that admitted it runs on until it parks or returns.
        let prev_exec = pool.then(|| RUNNING_FOR.with(|c| c.replace(self as *const Arc<Self>)));
        let mut call = ResumeCall {
            fib: r.fib,
            state: None,
        };
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
                Arc::as_ptr(r.vm),
                r.fault.0,
                r.fault.1,
            )
        };
        if let Some(prev) = prev_exec {
            RUNNING_FOR.with(|c| c.set(prev));
        }
        r.rt.pop_active();
        os_thread_rt::set_current_domain(prev_dom);
        let tls = vcpu_tls::get();
        vcpu_tls::seed(prev_tls);
        let handlers = vcpu_tls::serve_handlers();
        vcpu_tls::set_serve_handlers(prev_handlers);
        if let Some(s) = prev_shadow {
            crate::durable_shadow::seed(s);
        }
        fiber_rt::set_current(prev_rt);
        if faulted {
            r.vm.raise(mem::FAULT_TRAP);
            return (Outcome::Finished, tls, handlers);
        }
        let outcome = match call.state {
            // A guest fiber's `suspend` yields to its resumer inside the task, and the root's and a
            // handler's trap (#1469), so the only yield reaching here is the task's event park.
            Some(State::Yielded(_)) if r.slot.took_event_park() => {
                // Read the park's own deadline before the slot goes back to the pool (#1631).
                let deadline = r.slot.took_park_deadline();
                let yielded = matches!(r.slot.parked_on(), fiber_rt::ParkOn::Yield);
                if yielded {
                    Outcome::Yielded
                } else {
                    Outcome::Parked { deadline }
                }
            }
            // #2742 — a handler fiber done with its dispatch, waiting for the next.
            Some(State::Yielded(fiber_rt::JOB_DONE)) if r.slot.is_platform() => Outcome::Done,
            Some(State::Yielded(_)) => {
                // Unreachable by construction; fail closed rather than resume an unknown yield.
                r.vm.raise(TrapKind::ThreadFault as i64);
                Outcome::Finished
            }
            Some(State::Complete(_)) | None => Outcome::Finished,
        };
        (outcome, tls, handlers)
    }

    /// After a task's last residency: a child's ([`Self::finish_child`]) or a handler's
    /// ([`Self::finish_handler`]).
    fn finish(self: &Arc<Self>, task: Box<Task>) {
        match *task {
            Task::Child(t) => self.finish_child(Box::new(t)),
            Task::Handler(h) => self.finish_handler(h),
        }
    }

    /// #2742 — a handler the pool ran to its end: hand its end to the callee ([`HandlerDone`]) and
    /// drop it from the run's live count.
    fn finish_handler(self: &Arc<Self>, h: HandlerTask) {
        let key = h.window.key();
        let hid = h.id();
        h.window.pooled.fetch_sub(1, Ordering::AcqRel);
        let reusable = h.reusable;
        // A pooled handler was given its `done`, unless its serving vCPU's end answered its call
        // already ([`ServeSite::revoke`]): that one unwound, and its end goes nowhere.
        let (_revoked, window) = h.conclude(reusable);
        // Whoever waits on its dispatch is woken before it leaves the live count, as a joiner is
        // settled before its joinee leaves it (#1820): else a waiter that re-checks the deadlock
        // predicate in between sees every live vCPU parked, its answer already there. So is whoever
        // waits for its gate, which goes back once its dispatch is settled and counted.
        (window.ring)();
        window.release(hid);
        let mut g = lock(&self.state);
        let w = g
            .windows
            .get_mut(&key)
            .expect("a handler's window is counted");
        w.handlers -= 1;
        let (open, draining) = (w.counts(), w.draining);
        if w.handlers == 0 {
            g.windows.remove(&key);
        }
        // Out of the run's live count while its window counts it: the run may be quiescent now,
        // so the vCPU parks that decide the deadlock verdict re-check (#2757). One its domain's end
        // drained re-offers the domain's retiring task, which waits for its last handler.
        if let (true, Some(d)) = (open, self.domain()) {
            d.handler_finished();
            self.wake_vcpus_if_quiescent(&g);
        }
        if window.vm.handlers().fetch_sub(1, Ordering::SeqCst) == HANDLERS_ENDING + 1 || draining {
            self.wake_parked(&mut g, Wake::All);
        }
    }

    /// #2742 — a handler that did not end in the pool has ended (one started here that ended here,
    /// [`Self::run_here`]; one run on a caller's stack, [`ServeSite::run_inline`]): out of its
    /// domain's count, the last of an ending domain's re-offering its retiring task.
    fn handler_ended(self: &Arc<Self>, vm: &VmCtx) {
        if vm.handlers().fetch_sub(1, Ordering::SeqCst) == HANDLERS_ENDING + 1 {
            self.wake_locked(Wake::All);
        }
    }

    /// #2742 — `window`'s vCPU has ended ([`ServeSite::revoke`]): each handler of it the pool holds
    /// answers its call `CAP_REVOKED`, and keeps its fiber parked, shut out of the window, until its
    /// domain drains it. Its `done` is run outside the pool's lock, as a handler's end runs it, and
    /// the window's bell rings for the callers.
    fn revoke_window(&self, window: &Arc<Window>) {
        let dones: Vec<HandlerDone> = {
            let mut g = lock(&self.state);
            g.tasks
                .values_mut()
                .filter_map(|e| match e.task.as_deref_mut() {
                    Some(Task::Handler(h)) if Arc::ptr_eq(&h.window, window) => h.done.take(),
                    _ => None,
                })
                .collect()
        };
        if dones.is_empty() {
            return;
        }
        for done in dones {
            done(HandlerEnd::Trapped(crate::DOMAIN_DONE_CODE as i64));
        }
        (window.ring)();
    }

    /// #2742 — domain `domain` has ended, every vCPU of it, with handlers of its serve loops still
    /// parked or queued: their windows are shut for good, so none can run on. Mark them draining:
    /// each runs once more, whatever its window and gate, to unwind on the domain's end, and its
    /// `done`, unless its vCPU's end ran it ([`Self::revoke_window`]), answers its caller
    /// `CAP_REVOKED`, as the oracle revokes such a callee's calls. They count as live meanwhile. The
    /// domain's task retires until the last has ended.
    fn drain(self: &Arc<Self>, domain: usize) {
        let mut g = lock(&self.state);
        let mut began = false;
        for w in g
            .windows
            .values_mut()
            .filter(|w| w.domain == domain && !w.draining)
        {
            let before = w.counts();
            w.draining = true;
            if let (false, Some(d)) = (before, self.domain()) {
                d.count_window(true, w.handlers, w.parked);
            }
            began = true;
        }
        // Woken now, before the root leaves the live count: their parks end on the recorded end,
        // and a waiter must never see them counted parked meanwhile (a false deadlock verdict).
        if began {
            self.wake_parked(&mut g, Wake::All);
        }
    }

    /// After a child's last residency: free the window, run the teardown (powerbox release + lane
    /// give-back), publish the outcome, drop the domain's live count.
    fn finish_child(&self, mut task: Box<ChildTask>) {
        // #1469 — every vCPU the child spawned has ended (a retiring task waited for that): join
        // their OS threads before the window they ran on is freed.
        if let Some(d) = &task.dom {
            d.join_all(false);
        }
        // The host reads a durable child's window, for its state and its capture.
        if task.done.durable.is_some() {
            task.open_for_host();
        }
        let unwound = task.root_unwound();
        // #1361 step 4 — a durable child that unwound for a freeze leaves its window image for the
        // harvest (the parent's artifact carries it). Retire the base under the cell's lock first, so
        // no doorbell store can land on a window about to be freed.
        let done = Arc::clone(&task.done);
        let mut captured = false;
        if let Some(d) = done.durable.as_ref() {
            *d.base.lock().unwrap_or_else(|e| e.into_inner()) = 0;
            if unwound {
                // #2031 — first flatten the root's parked fibers into their own regions, as a run's
                // root's are (`run_inner`): the image below carries their continuations. Its vCPUs
                // flattened the ones they found as they unwound.
                // SAFETY: the root has unwound and every vCPU has ended, so the table is at rest;
                // the runtime is armed over the live window, and the register names this window's
                // context-0 region while the fibers run, as in a residency.
                let fibers = unsafe {
                    let prev = crate::durable_shadow::get();
                    crate::durable_shadow::seed(d.shadow.region_base(0));
                    let rt = &mut *task.rt as *mut FiberRuntime;
                    fiber_rt::freeze_drive(rt, Arc::as_ptr(&task.vm) as u64);
                    crate::durable_shadow::seed(prev);
                    fiber_rt::take_frozen(rt)
                };
                // A fiber that traps as it flattens is the child's trap, as a run's root's is: its
                // outcome, not a capture.
                if task.vm.ended().is_none() {
                    match task.capture(d, fibers) {
                        Some(c) => {
                            *d.capture.lock().unwrap_or_else(|e| e.into_inner()) = Some(c);
                            captured = true;
                        }
                        // #2318 — a grown tail the OS will not commit for the read cannot ride:
                        // the child ends refused, as a window it cannot open does.
                        None => task.end_refused(),
                    }
                }
            }
        }
        // #2112 — the child's domain has ended: its fibers' charges go back before its parent can
        // learn of the end, as its window's do in `teardown`.
        task.rt.table().release_charges();
        // #1936 — and its powerbox hears of it, unless its root already ended (`settle`).
        let end = if captured { None } else { task.end() };
        if let Some(t) = task.teardown.take() {
            t(captured);
        }
        // A retiring task published at `settle`, unless its root unwound, whose capture this is.
        let (result, trap) = task.outcome();
        done.publish(result, trap, self.domain());
        // #2800 — its end post once the end is published, while the task still counts as live.
        if let Some(signal) = end {
            signal();
        }
        let retiring = task.retiring;
        drop(task);
        // `settle` dropped a retiring task's live count when its root returned.
        if !retiring {
            if let Some(d) = self.domain() {
                d.child_finished();
            }
        }
        // #2359 — nothing of the task is left: its parent's nursery may let go of its powerbox.
        done.torn_down.store(true, Ordering::Release);
    }

    /// #1469 — the root of a task whose vCPUs still run has returned: publish its outcome now, as
    /// the oracle does at the root's completion (a `join` of the child returns), and drop it from
    /// the run's live count. Its window and powerbox stay until the vCPUs end ([`Self::finish`]).
    /// A durable root that unwound for a freeze
    /// has no outcome yet (#2010): its vCPUs are unwinding too, and the capture `finish` takes once
    /// they have ended is what its parent's join is owed.
    fn settle(&self, task: &mut ChildTask) {
        task.retiring = true;
        let (result, trap) = task.outcome();
        // A trap ends the child's domain: its parked vCPUs observe the shared cell and unwind.
        if trap != 0 {
            if let Some(d) = &task.dom {
                d.wake_own_parked();
            }
        }
        if !task.root_unwound() {
            // #1936 — the domain ends with its root, as the oracle's does, so its powerbox hears
            // of it now, while the task still counts as live; and its end post once the end is
            // published (#2800).
            let end = task.end();
            task.done.publish(result, trap, self.domain());
            if let Some(signal) = end {
                signal();
            }
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
    /// `end_domain` — the run is ending (not freezing): every task ends with it, running or not yet
    /// started, as the oracle ends a child domain with the run (DESIGN §23, D66).
    /// Its context gets the completion sentinel raised, which its entry and back-edge polls observe
    /// (`emit_domain_poll`). A freeze skips this, so a child the freeze reaches still unwinds under
    /// its own freeze word.
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
                        t.vm().raise(crate::DOMAIN_DONE_CODE as i64);
                    }
                    true
                } else if end_domain && !spared {
                    // As `kill` ends a task: its next poll ends it, and so does a host-side wait it
                    // re-checks (a serve loop's park).
                    e.vm.raise(crate::DOMAIN_DONE_CODE as i64);
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
            self.wake_parked(&mut g, Wake::All);
            // No worker could be spawned for a filed task (thread refusal): drive it here, this
            // thread a free worker until the queue is done.
            if g.workers.is_empty() && !g.tasks.is_empty() {
                g.free_workers += 1;
                drop(g);
                self.clone().worker_loop(0);
                g = lock(&self.state);
            }
            // The task table's one removal notifies `quiescent` under this lock (#2757).
            while !g.tasks.is_empty() {
                g = self.quiescent.wait(g).unwrap_or_else(|e| e.into_inner());
            }
            self.cv.notify_all();
            std::mem::take(&mut g.workers)
        };
        for w in workers {
            let _ = w.join();
        }
    }
}

/// Which parked tasks a wake makes runnable ([`ChildExec::wake_locked`]).
#[derive(Clone, Copy)]
pub(crate) enum Wake {
    /// Every one: what changed may be any park's predicate — a vCPU's or a child's end, a host
    /// park's key, an answer from outside, teardown.
    All,
    /// #2757 — every one, and every vCPU parked in a task's domain too: a store that can end any
    /// park of the run fired, its kill door or a freeze.
    Run,
    /// #2757 — the one task a store that can end its park names: its kill, its freeze doorbell.
    /// Its own parked vCPUs are woken with it.
    Task(u64),
    /// #2012 — those whose own deadline is at or before this instant.
    Due(Instant),
    /// #1991 — those whose futex wait has fired: a `notify` ends the waits it claims and no other,
    /// so every other parked task would only re-check and park again.
    Resolved,
}

enum Outcome {
    /// Event-parked, until `deadline` if the park carries one: #1631's question (does it come
    /// back on its own, so it should not count toward `Domain::parked`?) and #2012's (when?).
    Parked {
        deadline: Option<Instant>,
    },
    /// #1991 — its quantum ended with another task waiting: runnable, behind that one.
    Yielded,
    /// #1469 — the root has returned; vCPUs it spawned still run ([`ChildTask::retiring`]).
    Retiring,
    Finished,
    /// #2742 — a handler's dispatch is done, its fiber waiting, reusable, for the next.
    Done,
}
