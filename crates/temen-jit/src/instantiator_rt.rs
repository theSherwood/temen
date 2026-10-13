//! The JIT host side of the §14 **`Instantiator`** capability — VM-in-VM nesting. A guest holding an
//! `Instantiator` `instantiate`s a child confined to a power-of-two sub-window of its own window and
//! `join`s it. Unlike the interpreter (which spawns a child vCPU on its M:N executor), the JIT bakes
//! confinement into machine code, so a child confined to a *different* sub-window needs its own
//! compilation — "**nesting cost is paid at setup, not at runtime**" (§14): [`instantiate`] re-compiles
//! the child entry with the child's `mask`/`sub_base` ([`crate::compile_child_and_run`]) and runs it
//! over the **parent's live window** (so the parent intrinsically sees the child's writes — the §14
//! superset), under the caller's already-installed detect-and-kill guard.
//!
//! Authority lives in the host capability table (the same `Host` the interpreter uses): `instantiate`
//! resolves its `Instantiator` handle through the run's `call.cap` thunk (op 0 → the carve range
//! `[base, base+size)`), so a forged/wrong handle is an inert `CapFault` exactly as for any cap. The
//! child gets an **empty powerbox** for now (an inert `call.cap`); attenuated child caps + recursion +
//! "park only the calling fiber" (vs. today's synchronous run-at-`instantiate`) are follow-ups.

use crate::{CapThunk, TrapKind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex};
use temen_ir::{Data, Func, FuncIdx, SpawnRec, TypeEntry};

/// #1726 — the program an **installed §22 unit** runs, kept alive by the `CompiledModule` that
/// defined it, so the unit's `Instantiator` call sites can bake its address as `self_prog`. A
/// same-module (**self**) child spawned from unit code runs the *unit's* functions — the spawning
/// frame's module, as on both interpreters — not module 0's. `self_prog == 0` names the nursery's own
/// program (module 0), which is every call site outside a unit.
pub(crate) struct UnitProg {
    /// The unit's `(table, unit)` on its host ([`crate::unit_ref`]), which the host callbacks resolve
    /// a detached self-spawn from the unit against (#2143); [`crate::NO_UNIT`] for a unit no host
    /// names.
    pub(crate) unit: i64,
}

/// Negative-errno an out-of-range carve returns (§3e D42) — the one shared table.
use temen_ir::errno::EINVAL;

/// One spawned child's **completion cell** (S1c): `(result, trap)` once the child has finished (`trap`
/// `0` = clean), or `None` while it is still running. An async child's OS thread fills it from the
/// child's *own* thread — until then `join` parks on `cv` and `poll` reports *running* (`0`); a
/// synchronous (durable) child's cell is `Some` before `instantiate` returns. `Arc` so `join` can
/// clone it and drop the `children` lock before parking (no lock held across a wait).
pub(crate) struct ChildDone {
    pub(crate) state: Mutex<Option<(i64, i64)>>,
    pub(crate) cv: Condvar,
    /// #1361 step 4 — `Some` for a **durable detached** child: what a freeze of its parent needs to
    /// reach it and to keep it (see [`DurableCell`]). `None` for every other child.
    pub(crate) durable: Option<DurableCell>,
    /// #1820 — `true` while this cell's joiner is counted in the run's `Domain::parked`
    /// (`Domain::join_park`); settled by [`Self::publish`] or by the joiner leaving.
    pub(crate) joiner_parked: AtomicBool,
}

impl ChildDone {
    pub(crate) fn new(state: Option<(i64, i64)>, durable: Option<DurableCell>) -> ChildDone {
        ChildDone {
            state: Mutex::new(state),
            cv: Condvar::new(),
            durable,
            joiner_parked: AtomicBool::new(false),
        }
    }

    /// The child's outcome is known (its root's, for a task whose vCPUs run on): publish
    /// `(result, trap)` unless an earlier publish stands, and wake its joiner. #1820 — a parked
    /// joiner stops counting here, under the cell's lock, before the child leaves `live`
    /// (`Domain::child_finished`), so no deadlock check sees it beside the dropped `live`. `dom` is
    /// the run's domain (`None` ⇒ the run has none, and nothing counted the joiner).
    pub(crate) fn publish(
        &self,
        result: i64,
        trap: i64,
        dom: Option<&crate::os_thread_rt::Domain>,
    ) {
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if st.is_none() {
            *st = Some((result, trap));
            self.cv.notify_all();
        }
        if let Some(d) = dom {
            d.join_settle(&self.joiner_parked);
        }
    }
}

/// #1361 step 4 — a durable detached child's freeze cell, shared by its join-table entry and its task.
/// The parent holds no pointer into the child's window otherwise (detachment severs *read*); this is
/// the lifecycle linkage a freeze uses, like the interpreter's doorbell.
pub(crate) struct DurableCell {
    /// The child's live window base while its window exists, `0` once its task frees it. The lock
    /// orders a freeze's doorbell store against that free.
    pub(crate) base: Mutex<usize>,
    /// What the child's task left for the harvest at finish, **iff** it unwound for a freeze.
    pub(crate) capture: Mutex<Option<crate::DetachedCapture>>,
    /// The child module's shadow arena: its context-0 region is where the child spills.
    pub(crate) shadow: temen_ir::durable_abi::ShadowArena,
    /// What the spawn knew that a thaw needs: the entry and the window geometry.
    pub(crate) entry: u32,
    pub(crate) mapped_log2: u8,
    pub(crate) reserved_log2: u8,
    /// #1854 — how far the child grew its window, asked of its own powerbox (`(hook, child ctx)`),
    /// so its image reaches its grown pages; `None` when no hook is installed.
    pub(crate) high_water: Option<(crate::HighWater, usize)>,
}

impl DurableCell {
    pub(crate) fn new(
        shadow: temen_ir::durable_abi::ShadowArena,
        entry: u32,
        mapped_log2: u8,
        reserved_log2: u8,
        high_water: Option<(crate::HighWater, usize)>,
    ) -> DurableCell {
        DurableCell {
            base: Mutex::new(0),
            capture: Mutex::new(None),
            shadow,
            entry,
            mapped_log2,
            reserved_log2,
            high_water,
        }
    }

    /// Ring the child's doorbell: store `UNWINDING` into its **own** freeze word (what
    /// [`crate::FreezeController::request_freeze`] does for the root), so it unwinds at its next poll
    /// and its task deposits its window image at finish. A no-op once its window is gone, which the
    /// answer says: whether the child was live.
    pub(crate) fn ring(&self) -> bool {
        let base = self.base.lock().unwrap_or_else(|e| e.into_inner());
        if *base != 0 {
            // SAFETY: a nonzero base is the child's live window (its task retires the base under
            // this lock before freeing it); `STATE_OFF` is within its first mapped page, and the
            // word is only ever accessed as an aligned `i32`.
            unsafe {
                (*((*base + temen_ir::durable_abi::STATE_OFF as usize)
                    as *const std::sync::atomic::AtomicI32))
                    .store(temen_ir::durable_abi::STATE_UNWINDING, Ordering::SeqCst);
            }
        }
        *base != 0
    }

    /// Whether the child unwound for a freeze (its task deposited its capture).
    pub(crate) fn unwound(&self) -> bool {
        self.capture
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_some()
    }
}

/// One spawned child's join-table entry: its completion cell plus whether it has been `join`ed (a
/// second join is inert — `ThreadFault`, matching the interpreter's once-only join).
struct Child {
    done: std::sync::Arc<ChildDone>,
    joined: bool,
    /// CALLS.md 5c.0 — the nursery-retained ref to this child's **shared powerbox**
    /// (`GrantChild::retained_ctx` as usize; `0` = the builder shared nothing, op 14 answers
    /// `-EINVAL`). Lets `child_offer` mint a live-impl over the child and keeps the child `Host`
    /// reachable after its thread exits (the interp's `child_hosts` retention, JIT twin).
    /// Released exactly once, at [`Nursery::join_children`], via the grant hooks' releaser.
    retained: usize,
    /// §4 — a durable §14 child's freeze record, and whether it unwound into its carve. Recorded when
    /// its parent unwinds with the child still unjoined ([`Nursery::freeze_unjoined`]).
    nested: Option<(crate::FrozenNested, bool)>,
    /// #2010 — the nursery this child spawns its own children through, if it has one (#1956; also
    /// held in [`Nursery::kids`]). A freeze reaches them through it, and its harvest nests theirs.
    kid: Option<std::sync::Arc<Nursery>>,
    /// #2044 — a detached child a thaw delivered as completed ([`crate::DetachedSeed::Completed`]):
    /// only its outcome exists, and a freeze before its `join` carries it again ([`Nursery::take_detached_harvest`]).
    delivered: bool,
    /// #2074 — the child's executor task, for `kill` (`0`: none, as for a child already finished at
    /// its filing).
    task: u64,
}

impl Child {
    /// #1361 step 4, #2010 — a freeze reached this child's parent: ring a durable child's doorbell,
    /// then its own children's, if the ring found it live or it already unwound for this freeze.
    /// The whole live subtree is rung before the run's teardown begins, so a grandchild parked under
    /// a word nobody rang yet is never poisoned there (`ChildExec::shutdown_and_join`). A child that
    /// already returned rings nothing: its children run on, as the oracle's do.
    fn ring(&self) {
        if let Some(d) = self.done.durable.as_ref() {
            if d.ring() || d.unwound() {
                if let Some(kid) = &self.kid {
                    kid.ring_detached();
                }
            }
        }
    }

    /// A child whose outcome is **already known** (the synchronous path): a cell pre-filled with
    /// `(result, trap)`.
    fn finished(result: i64, trap: i64) -> Child {
        Child {
            done: std::sync::Arc::new(ChildDone::new(Some((result, trap)), None)),
            joined: false,
            retained: 0,
            nested: None,
            kid: None,
            delivered: false,
            task: 0,
        }
    }

    /// A child whose OS thread is **still running** (the async path): the empty cell the thread fills on
    /// completion.
    fn pending(done: std::sync::Arc<ChildDone>) -> Child {
        Child {
            done,
            joined: false,
            retained: 0,
            nested: None,
            kid: None,
            delivered: false,
            task: 0,
        }
    }
}

/// How a task filing ended (see [`file_task`]).
enum Filed {
    /// Filed: the child's join-table slot.
    Slot(i32),
    /// The domain is at its §15 live-vCPU ceiling — nothing was taken.
    AtCeiling,
    /// Admitted, then refused (a task stack or a pre-map alias the platform could not give): the
    /// child never ran, its teardown already ran, the §15 reservation is released.
    Refused,
}

/// D66 — **the one child filing** for every non-durable §14 child (INVARIANTS #15): reserve the §15
/// slot, build the task around `code` in a fresh window (`init` seeds it — a carve image or data
/// segments + payload; `premap` aliases an op-15 region; `copy_back` writes a carve child's image
/// back at finish), register it in the join table with its retained powerbox ref, and hand it to
/// the executor. `teardown` (release the powerbox, return the lane) runs exactly once — after the
/// task's last residency, or right here if the filing is refused.
///
/// # Safety
/// `code` was compiled by `compile_child_windowed` for `(mapped_log2, reserved_log2)`; `args`
/// matches the entry's arity; the `SendRaw` pointers `init`/`premap`/`copy_back`/`teardown` capture
/// are live for the run (the parent window, joined-after; the child powerbox, owned by the task).
#[allow(clippy::too_many_arguments)]
unsafe fn file_task(
    rt: &Nursery,
    code: std::sync::Arc<crate::CompiledModule>,
    mapped_log2: u8,
    reserved_log2: u8,
    init: impl FnOnce(&mut crate::mem::GuestWindow),
    premap: impl FnOnce(*mut u8, u64, u64) -> bool,
    copy_back: Option<crate::child_exec::CopyBack>,
    args: Vec<i64>,
    n_results: usize,
    chain: Vec<(usize, i64)>,
    retained_ctx: usize,
    teardown: crate::child_exec::Teardown,
    // #1361 step 4 — a thaw re-files a captured detached child at its recorded join slot; every
    // spawn appends (`None`).
    slot: Option<usize>,
    // #1361 step 4 — a durable detached child's freeze cell (see [`DurableCell`]).
    durable: Option<DurableCell>,
    // #2001 — a detached child's own budget node, which its threads are charged to; `None` for a
    // carve child.
    node: Option<std::sync::Arc<dyn crate::BudgetNode>>,
    // #2010 — the nursery `code` was compiled against, if the child spawns (see [`Child::kid`]).
    kid: Option<std::sync::Arc<Nursery>>,
    // #2010 — what a thawed durable child's capture left beside its image (see
    // [`crate::child_exec::ChildTask::thaw`]); empty for a fresh spawn.
    thaw: crate::DurableResidue,
) -> Filed {
    let futex_sched = rt.futex_sched;
    // #1586 — reserve a §15 live-vCPU slot before filing, so a parent cannot hold more concurrency
    // than its ancestors granted (INVARIANTS #3). SAFETY: a nonzero `futex_sched` is the run's live
    // `Domain`, outliving every task.
    if futex_sched != 0
        && !unsafe { (*(futex_sched as *const crate::os_thread_rt::Domain)).try_child_start() }
    {
        teardown(false);
        return Filed::AtCeiling;
    }
    let done = std::sync::Arc::new(ChildDone::new(None, durable));
    let task = unsafe {
        crate::child_exec::ChildTask::new(
            code,
            mapped_log2,
            reserved_log2,
            init,
            premap,
            args,
            n_results,
            chain,
            std::sync::Arc::clone(&done),
            copy_back,
            teardown,
            node,
        )
    };
    let mut task = match task {
        Ok(t) => t,
        Err(teardown) => {
            teardown(false);
            if futex_sched != 0 {
                unsafe { (*(futex_sched as *const crate::os_thread_rt::Domain)).child_finished() };
            }
            return Filed::Refused;
        }
    };
    // #1361 step 4 — publish a durable child's window base before it can run, so a freeze's doorbell
    // reaches it from its first op.
    if let Some(d) = done.durable.as_ref() {
        *d.base.lock().unwrap_or_else(|e| e.into_inner()) = task.window_base();
    }
    let mut children = rt.children.lock().unwrap_or_else(|e| e.into_inner());
    let mut child = Child::pending(done);
    // 5c.0 — retain the shared child powerbox for `child_offer` (released at join_children).
    child.retained = retained_ctx;
    child.kid = kid;
    let slot = match slot {
        None => {
            children.push(child);
            children.len() - 1
        }
        Some(s) => {
            while children.len() <= s {
                children.push(Child::finished(0, 0));
            }
            children[s] = child;
            s
        }
    };
    drop(children);
    task.thaw = thaw;
    let id = rt.child_exec.spawn(task);
    if let Some(c) = rt
        .children
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_mut(slot)
    {
        c.task = id;
    }
    Filed::Slot(slot as i32)
}

/// A raw pointer a task's closures carry to a worker. SAFETY (for every use in this file): the
/// parent window outlives every task (`join_children` runs before it frees) and a carve child
/// touches only its own carve for copy-in / copy-back — disjoint from siblings and the parent's
/// live data, the §14 disjointness the guest owns; a child powerbox is handed over wholesale to
/// its task (`Host: Send`) and never touched by the parent after the spawn.
struct SendRaw<T>(T);
unsafe impl<T> Send for SendRaw<T> {}

/// #2010 — give back the powerbox refs a thaw's seed holds, and its children's, for a re-launch that
/// has no nursery to file them in.
///
/// # Safety
/// As [`Nursery::relaunch_detached`]'s seed: each holds two live counted refs, owned here.
unsafe fn release_seed(release: crate::GrantChildReleaser, seed: crate::DetachedSeed) {
    // A completed child's seed holds no powerbox.
    if let crate::DetachedSeed::Captured(seed) = seed {
        release(seed.child.ctx);
        release(seed.child.retained_ctx);
        for c in seed.children {
            release_seed(release, c);
        }
    }
}

/// D66 — a granted child's teardown: release its powerbox and, for a detached child, return its
/// lane to the parent through the `lane_give` hook (`-1` for a carve child: a no-op, it holds none).
/// `window` is the `(budget, bytes, runs)` lease a detached spawn's admission charged for the child's
/// window: it goes back to that budget here, when the window goes (owner, 2026-09-29 — `Budget`
/// accounts live windows). `None` for a carve child.
unsafe fn granted_teardown(
    rt: &Nursery,
    release: crate::GrantChildReleaser,
    gc_ctx: *mut core::ffi::c_void,
    lane: i64,
    window: Option<(i32, u64, u64)>,
    fuel: Option<Box<crate::FuelCell>>,
) -> crate::child_exec::Teardown {
    let (ctx, parent_ctx) = (SendRaw(gc_ctx), SendRaw(rt.grant_ctx()));
    let lane_give = rt.grant_lane_give.load(Ordering::Acquire);
    let mem_give = rt.grant_budget_mem_give.load(Ordering::Acquire);
    Box::new(move |captured| {
        let (ctx, parent_ctx) = (ctx, parent_ctx);
        // #1944 slice 3 — the child's fuel cell goes back as its task ends (or a freeze unwinds it):
        // what it drew and did not burn returns to its budget chain.
        drop(fuel);
        // #1971 — a child a freeze captured keeps its window charged: the artifact carries the
        // charge, and the thaw's relaunch files the lease again.
        let window = if captured { None } else { window };
        // SAFETY: the powerbox is freed exactly once, here, by the task that owned it, after the
        // window settles against it; the givers were loaded from `rt`, registered with the parent
        // host `parent_ctx` names.
        unsafe {
            give_back(parent_ctx.0, ctx.0, lane_give, mem_give, lane, window);
            release(ctx.0);
        }
    })
}

/// Hand a detached child's lane, window bytes and first vCPU back to its parent, and what its window
/// grew (#1909): at the child's end ([`granted_teardown`]), or when its spawn goes no further after
/// the admission took them ([`undo_admission`], with no `child`: it never ran).
///
/// # Safety
/// `lane_give` / `mem_give` are 0 or the embedder's registered [`crate::LaneGiver`] /
/// [`crate::BudgetMemGiver`], `parent_ctx` is the parent host they were registered with, and `child`
/// is null or the child's live powerbox.
unsafe fn give_back(
    parent_ctx: *mut core::ffi::c_void,
    child: *mut core::ffi::c_void,
    lane_give: usize,
    mem_give: usize,
    lane: i64,
    window: Option<(i32, u64, u64)>,
) {
    if lane_give != 0 && lane >= 0 {
        let give: crate::LaneGiver = core::mem::transmute(lane_give);
        give(parent_ctx, lane);
    }
    if let (Some((budget, bytes, runs)), true) = (window, mem_give != 0) {
        let give: crate::BudgetMemGiver = core::mem::transmute(mem_give);
        give(parent_ctx, child, budget, bytes, runs);
    }
}

/// #1975 — the undo of a detached spawn's admission (`budget_mem_take`), for a spawn that files no
/// child after it: its `lease` and its lane go back, so it charges nothing.
///
/// # Safety
/// `rt` is the live nursery whose `grant_ctx` the admission charged.
unsafe fn undo_admission(rt: &Nursery, lease: (i32, u64, u64), lane: i64) {
    give_back(
        rt.grant_ctx(),
        core::ptr::null_mut(),
        rt.grant_lane_give.load(Ordering::Acquire),
        rt.grant_budget_mem_give.load(Ordering::Acquire),
        lane,
        Some(lease),
    );
}

/// CALLS.md 5c.1b — register a granted child's serve context (its live `CompiledModule`) on its
/// shared powerbox. The same registration is the child's `Jit` native ctx (#1296).
unsafe fn register_serve(
    rt: &Nursery,
    gc_ctx: *mut core::ffi::c_void,
    code: &std::sync::Arc<crate::CompiledModule>,
) {
    let rs = rt.grant_register_serve.load(Ordering::Acquire);
    if rs != 0 {
        let rs: crate::ChildServeRegistrar = unsafe { core::mem::transmute(rs) };
        unsafe { rs(gc_ctx, std::sync::Arc::as_ptr(code) as usize) };
    }
}

/// The per-run §14 nesting runtime, baked into the module's `Instantiator` `call.cap` sites. Holds
/// what compiling + running a child needs: the module's functions, the run's `call.cap` thunk/ctx
/// (to resolve an `Instantiator` handle's authority), and — supplied post-finalize via [`set_env`] —
/// the live window's detect-and-kill fault range. Non-durable children (plain and granted) run
/// **asynchronously** on their own OS threads; outcomes land in per-child completion cells `join`
/// parks on. Only durable children still run synchronously at `instantiate`.
pub(crate) struct Nursery {
    /// CALLS.md 5c.1a — the parent module's impl-export **handler** funcidxs, threaded into every
    /// granted-child compile so the child gets serve trampolines (same-module children, ops 8/11;
    /// op-13 separate-module children pass their own — a later slice). Computed from
    /// `m.impl_exports` at construction.
    pub(crate) serve_handlers: Box<[u32]>,
    /// The module's own program, which a self child of its own code runs: empty when the module
    /// names no `Instantiator`, as a nursery stood up for §22 units alone (#2312), whose self
    /// children run their unit's program ([`UnitProg`]).
    funcs: std::sync::Arc<[Func]>,
    /// #922 — the parent module's type section, threaded into every same-module child compile so
    /// the child's interned `call.dyn` type indices resolve (a separate-module child resolves
    /// against its own resolved type section instead — see [`Nursery::resolve_child`]).
    types: std::sync::Arc<[TypeEntry]>,
    cap_thunk: CapThunk,
    cap_ctx: *mut core::ffi::c_void,
    /// §14 separate-module children: the host callback resolving a guest's `Module` handle to the
    /// granted module's code/data (`None` ⇒ module ops are an inert `CapFault`). Kept apart from the
    /// `call.cap` thunk so the host pointers it yields are never guest-reachable.
    resolve_module: Option<crate::ModuleResolver>,
    /// Address of the parent run's §5 kill-path interrupt cell (`0` ⇒ no kill-path armed). A nested
    /// JIT child is compiled to poll the **same** cell, so one host interrupt stops the parent *and*
    /// every child it spawned (a runaway child would otherwise hang the parent inside `instantiate` /
    /// `resume`, where the parent's own epoch checks can't fire).
    epoch_addr: usize,
    /// Address of the parent run's thread [`crate::os_thread_rt::Domain`] (`0` ⇒ none — the durable
    /// nested nursery). Children compile their `atomic.wait`/`notify` against this **shared** futex
    /// table, so concurrent children (and the parent's own vCPUs) rendezvous — the pipeline
    /// primitive; spawns also register in its live count for the wait/join deadlock detection.
    futex_sched: usize,
    children: Mutex<Vec<Child>>,
    /// D66 — the child-domain executor every non-durable child runs on: migrating tasks over a
    /// lane-bounded worker pool (`child_exec`), replacing one OS thread per child. The run drives it
    /// to quiescence at teardown ([`Nursery::join_children`]) before the parent window is freed.
    child_exec: std::sync::Arc<crate::child_exec::ChildExec>,
    /// DURABILITY.md §4: the run is **durable** (set by [`Nursery::set_durable`] at run entry —
    /// the durable flag is applied after compile, where this nursery is built). A durable run's
    /// `instantiate`/`coro_spawn` **fail closed** (`-EINVAL`): this child runner re-compiles and
    /// runs children with no durable state (no shadow init, no instrumented-admission check), so a
    /// child it spawned could never drain-then-unwind — silently breaking "the snapshot unit is
    /// the domain closed over its nesting subtree". The interpreter is the reference for durable
    /// nesting; JIT parity is a follow-up.
    durable: AtomicBool,
    /// #1693 — the run's async freeze controller, if any (set at run entry, like `durable`): each
    /// durable child registers its window with it, so a request reaches the child mid-run.
    freeze: Mutex<Option<std::sync::Arc<crate::FreezeController>>>,
    /// §4 depth-2: the **shared** subtree task-id counter — the next id to hand a nested child's
    /// nursery. Threaded down the subtree (every nursery in one freeze shares the `Arc`) so ids are
    /// assigned in instantiate order across all levels, matching the interpreter's dense scheme.
    task_counter: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    /// §4 freeze export: the §14 nested-child re-attach residue captured during a durable freeze — one
    /// [`crate::FrozenNested`] per child that unwound into its carve (`instantiate` records it when
    /// `compile_child_and_run` reports the child left `UNWINDING`). This is the **shared** subtree sink
    /// (every nursery in one freeze holds the same `Arc`), so a grandchild's residue — recorded by its
    /// parent-child's nursery — coalesces at the **root**, where the top-level run drains it
    /// (`take_frozen_nested`). The JIT analog of the interpreter's `VCpu::freeze_sink`.
    frozen_nested_sink: std::sync::Arc<Mutex<Vec<crate::FrozenNested>>>,
    /// PROCESS.md §5 / #1287 — the detached-child powerbox builder + the `Budget` taker
    /// ([`crate::GrantChildHooks::build_detached`] / [`crate::BudgetMemTaker`]; 0 = none ⇒ op 15 is an
    /// inert `CapFault`, like the other grant ops without hooks).
    grant_build_detached: std::sync::atomic::AtomicUsize,
    grant_budget_mem_take: std::sync::atomic::AtomicUsize,
    /// #1587 — [`crate::BudgetMemGiver`]: returns a taken quota when the OS-thread spawn fails.
    grant_budget_mem_give: std::sync::atomic::AtomicUsize,
    /// D66 — [`crate::LaneGiver`]: a reaped child task returns its lane to the parent.
    grant_lane_give: std::sync::atomic::AtomicUsize,
    /// Op-15 pre-mapped region hooks ([`crate::PremapAdmit`] / [`crate::PremapStage`] /
    /// [`crate::WindowReady`]; 0 = none ⇒ a spawn asking for one is an inert `CapFault`). The last is
    /// also every detached child's window-ready hook (#2268).
    grant_premap_admit: std::sync::atomic::AtomicUsize,
    grant_premap_stage: std::sync::atomic::AtomicUsize,
    grant_premap_apply: std::sync::atomic::AtomicUsize,
    /// #1944 slice 3 — the installed [`crate::ChildBudgetNode`] (0 = none: detached children run
    /// un-metered, and their threads uncharged).
    grant_budget_node: std::sync::atomic::AtomicUsize,
    /// #1854 — the installed child [`crate::HighWater`] (0 = none).
    grant_high_water: std::sync::atomic::AtomicUsize,
    grant_release: std::sync::atomic::AtomicUsize,
    grant_bind_imports: std::sync::atomic::AtomicUsize,
    /// CALLS.md 5c.0 — the `child_offer` mint hook ([`crate::ChildOfferMint`] as usize; 0 = none).
    grant_mint: std::sync::atomic::AtomicUsize,
    /// #2220 — the `grant` hook ([`crate::ChildGrant`] as usize; 0 = none).
    grant_into: std::sync::atomic::AtomicUsize,
    /// CALLS.md 5c.0 — the lock-taking cap thunk granted-child compiles run against
    /// ([`crate::CapThunk`] as usize; 0 ⇒ fall back to the run's `cap_thunk` — pre-5c.0 behavior,
    /// only correct for a builder that does not share the child `Host`).
    grant_thunk: std::sync::atomic::AtomicUsize,
    /// CALLS.md 5c.1b — the [`crate::ChildServeRegistrar`] hook (0 ⇒ none): registers a spawned
    /// granted child's module address on its shared powerbox so the locked thunk's serve arm
    /// can resolve + invoke handlers.
    grant_register_serve: std::sync::atomic::AtomicUsize,
    /// #1234 — the parent host pointer the registered hook family decodes
    /// ([`crate::GrantChildHooks::parent_ctx`]), stored with the family rather than re-read from
    /// `cap_ctx`: the run may have baked either shape there, and only the hooks' own registration
    /// knows which one this family expects. `0` when no hooks are registered.
    grant_parent_ctx: std::sync::atomic::AtomicUsize,
    /// #1956 — the lane chain above this nursery's domain: empty for a run's, the enclosing
    /// domains' for a detached child's. With its children's own two lanes it makes the whole chain
    /// a child's task is gated on ([`Nursery::lane_chain_of`]), as the oracle gates every vCPU.
    ancestors: Vec<(usize, i64)>,
    /// #1956 — the nurseries of this nursery's detached children that spawn in turn, held to the
    /// run's teardown. Their addresses are baked into their children's code, and what they retain
    /// is released only after every task has ended ([`Nursery::join_children`]): a grandchild's
    /// teardown hands its lane and window back to its parent's host, which only that retained ref
    /// keeps alive once the parent has ended. Each is held from its birth ([`Nursery::for_child`]),
    /// whether or not its child is ever filed: a thaw files its children into it first (#2010).
    kids: Mutex<Vec<std::sync::Arc<Nursery>>>,
    /// #1956 — the installed [`crate::AsParent`] (0 = none: a detached child spawns nothing).
    grant_as_parent: std::sync::atomic::AtomicUsize,
}

// SAFETY: the raw `cap_ctx` is the run's host pointer, valid for the whole run; the `Nursery` is
// only ever used on the run's threads while that host (and window) are alive. The interior tables
// are `Mutex`-guarded. (A child runs synchronously on the calling thread today, so there is in fact
// no cross-thread sharing yet; the bounds keep the door open for concurrent children later.)
unsafe impl Send for Nursery {}
unsafe impl Sync for Nursery {}

impl Nursery {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        funcs: std::sync::Arc<[Func]>,
        types: std::sync::Arc<[TypeEntry]>,
        cap_thunk: CapThunk,
        cap_ctx: *mut core::ffi::c_void,
        resolve_module: Option<crate::ModuleResolver>,
        epoch_addr: usize,
        futex_sched: usize,
        task_counter: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        frozen_nested_sink: std::sync::Arc<Mutex<Vec<crate::FrozenNested>>>,
        serve_handlers: Box<[u32]>,
        // #1956 — the executor this nursery files its children on: a run's own (`None`, made here),
        // or the run's, shared by a detached child's nursery.
        child_exec: Option<std::sync::Arc<crate::child_exec::ChildExec>>,
    ) -> Nursery {
        Nursery {
            funcs,
            types,
            serve_handlers,
            cap_thunk,
            cap_ctx,
            resolve_module,
            epoch_addr,
            futex_sched,
            children: Mutex::new(Vec::new()),
            child_exec: child_exec.unwrap_or_else(|| {
                let e = crate::child_exec::ChildExec::new(futex_sched);
                // D66 — the domain wakes parked tasks with its other parked waiters (a `notify`, a
                // vCPU exit, the kill path, teardown), so it needs a ref. SAFETY: a nonzero
                // `futex_sched` is the run's live `Domain`, which outlives this nursery.
                if futex_sched != 0 {
                    unsafe {
                        (*(futex_sched as *const crate::os_thread_rt::Domain))
                            .set_child_exec(std::sync::Arc::clone(&e))
                    };
                }
                e
            }),
            ancestors: Vec::new(),
            kids: Mutex::new(Vec::new()),
            grant_as_parent: std::sync::atomic::AtomicUsize::new(0),
            durable: AtomicBool::new(false),
            freeze: Mutex::new(None),
            task_counter,
            frozen_nested_sink,
            grant_build_detached: std::sync::atomic::AtomicUsize::new(0),
            grant_budget_mem_take: std::sync::atomic::AtomicUsize::new(0),
            grant_budget_mem_give: std::sync::atomic::AtomicUsize::new(0),
            grant_lane_give: std::sync::atomic::AtomicUsize::new(0),
            grant_premap_admit: std::sync::atomic::AtomicUsize::new(0),
            grant_premap_stage: std::sync::atomic::AtomicUsize::new(0),
            grant_premap_apply: std::sync::atomic::AtomicUsize::new(0),
            grant_budget_node: std::sync::atomic::AtomicUsize::new(0),
            grant_high_water: std::sync::atomic::AtomicUsize::new(0),
            grant_release: std::sync::atomic::AtomicUsize::new(0),
            grant_bind_imports: std::sync::atomic::AtomicUsize::new(0),
            grant_parent_ctx: std::sync::atomic::AtomicUsize::new(0),
            grant_register_serve: std::sync::atomic::AtomicUsize::new(0),
            grant_mint: std::sync::atomic::AtomicUsize::new(0),
            grant_into: std::sync::atomic::AtomicUsize::new(0),
            grant_thunk: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// PROCESS.md S2 (JIT parity): install the child host callbacks — build a detached child's
    /// powerbox and release it after the run. Called once at run entry (like [`Self::set_durable`]),
    /// before any spawn site can fire. `None` leaves them `0` (a spawn is an inert `CapFault`).
    pub(crate) fn set_grant_hooks(&self, hooks: Option<crate::GrantChildHooks>) {
        let (r, bi, m, t, rs, bd, mt, mg, lg, pc, pa, ps, pp) = match hooks {
            Some(h) => (
                h.release as usize,
                h.bind_imports as usize,
                h.mint as usize,
                h.thunk as usize,
                h.register_serve as usize,
                h.build_detached as usize,
                h.budget_mem_take as usize,
                h.budget_mem_give as usize,
                h.lane_give as usize,
                h.parent_ctx as usize,
                h.premap_admit as usize,
                h.premap_stage as usize,
                h.premap_apply as usize,
            ),
            None => (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        };
        self.grant_build_detached.store(bd, Ordering::Release);
        self.grant_budget_mem_take.store(mt, Ordering::Release);
        self.grant_budget_mem_give.store(mg, Ordering::Release);
        self.grant_lane_give.store(lg, Ordering::Release);
        self.grant_premap_admit.store(pa, Ordering::Release);
        self.grant_premap_stage.store(ps, Ordering::Release);
        self.grant_premap_apply.store(pp, Ordering::Release);
        self.grant_high_water.store(
            hooks.map_or(0, |h| h.high_water as usize),
            Ordering::Release,
        );
        self.grant_budget_node.store(
            hooks.map_or(0, |h| h.budget_node as usize),
            Ordering::Release,
        );
        self.grant_as_parent
            .store(hooks.map_or(0, |h| h.as_parent as usize), Ordering::Release);
        self.grant_register_serve.store(rs, Ordering::Release);
        self.grant_release.store(r, Ordering::Release);
        self.grant_bind_imports.store(bi, Ordering::Release);
        self.grant_mint.store(m, Ordering::Release);
        self.grant_into
            .store(hooks.map_or(0, |h| h.grant as usize), Ordering::Release);
        self.grant_thunk.store(t, Ordering::Release);
        self.grant_parent_ctx.store(pc, Ordering::Release);
    }

    /// #1234 — the parent host pointer to hand a §14 child hook: the one **registered with the
    /// hook family**, never the run's `cap_ctx`. `cap_ctx` is correct for `self.cap_thunk` (they
    /// are baked together and the thunk decodes its own shape); a hook decodes the pointer itself,
    /// so it must be given the shape its own family was built for.
    fn grant_ctx(&self) -> *mut core::ffi::c_void {
        self.grant_parent_ctx.load(Ordering::Acquire) as *mut core::ffi::c_void
    }

    /// #1854 — the installed child [`crate::HighWater`] over child powerbox `ctx`, for its freeze cell.
    fn child_high_water(&self, ctx: *mut core::ffi::c_void) -> Option<(crate::HighWater, usize)> {
        match self.grant_high_water.load(Ordering::Acquire) {
            0 => None,
            // SAFETY: a nonzero value is a `HighWater` stored by `set_grant_hooks`.
            f => Some((
                unsafe { core::mem::transmute::<usize, crate::HighWater>(f) },
                ctx as usize,
            )),
        }
    }

    /// D66 — the lane chain a granted child's task is gated on: the lanes above this nursery's
    /// domain, then its parent's over its own — the whole chain from the root, as the oracle's.
    fn lane_chain_of(&self, gc: &crate::GrantChild) -> Vec<(usize, i64)> {
        let mut chain = self.ancestors.clone();
        chain.push((gc.parent_domain as usize, gc.parent_lane_cap));
        chain.push((gc.domain as usize, gc.lane_cap));
        chain
    }

    /// #1956 — the nursery a detached child spawns its own children through, or `None` when it has
    /// no need of one (it names no `Instantiator` and can install no unit that would) or no means:
    /// no [`crate::AsParent`] installed, or a powerbox nothing retains. Its powerbox `gc` is the
    /// host its `Instantiator` resolves against and the parent its hooks charge, through the family
    /// `as_parent` builds over it; the executor, futex domain and kill cell are the run's. The
    /// program, shadow and fuel a carve child would take are empty: it files none. It is durable
    /// when this one is (#2010): a durable domain's children are durable, and so are theirs.
    ///
    /// # Safety
    /// `gc` is the live powerbox the installed hooks just built for the child.
    unsafe fn for_child(
        &self,
        gc: &crate::GrantChild,
        funcs: &[Func],
    ) -> Option<std::sync::Arc<Nursery>> {
        let as_parent = self.grant_as_parent.load(Ordering::Acquire);
        // A powerbox nothing retains dies with the child, while a grandchild's teardown still hands
        // its lane and window back to it.
        if as_parent == 0
            || gc.retained_ctx.is_null()
            || !(crate::funcs_use_instantiator(funcs) || gc.jit_table_log2 > 0)
        {
            return None;
        }
        let hooks = core::mem::transmute::<usize, crate::AsParent>(as_parent)(gc.ctx);
        let mut n = Nursery::new(
            Vec::new().into(),
            Vec::new().into(),
            hooks.thunk,
            gc.ctx,
            Some(hooks.resolve_module),
            self.epoch_addr,
            self.futex_sched,
            self.task_counter(),
            self.nested_sink(),
            self.serve_handlers.clone(),
            Some(std::sync::Arc::clone(&self.child_exec)),
        );
        n.ancestors = self.lane_chain_of(gc);
        n.ancestors.pop(); // the child's own lane is its children's parent lane
        n.set_grant_hooks(Some(hooks));
        n.set_durable(self.durable.load(Ordering::Acquire));
        let n = std::sync::Arc::new(n);
        self.kids
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(std::sync::Arc::clone(&n));
        Some(n)
    }

    /// #1944 slice 3 — the budget node child powerbox `ctx` holds as its own
    /// ([`crate::ChildBudgetNode`]), and the fuel cell its vCPUs draw from it. The node is `None` when
    /// no hook is installed. The cell draws whether or not this run is metered, so a child's budget
    /// caps it on an unmetered root too (#1705); it is `None`, compiled un-metered, when the chain is
    /// unbounded at the spawn (every level `-1`). The caller bakes the cell's address into the child
    /// and hands the cell to the child's task, whose end returns what it did not burn, and the node to
    /// its domain, which charges its threads (#2001).
    ///
    /// # Safety
    /// `ctx` is a detached child's powerbox the installed hooks built.
    unsafe fn detached_node(
        &self,
        ctx: *mut core::ffi::c_void,
    ) -> (
        Option<std::sync::Arc<dyn crate::BudgetNode>>,
        Option<Box<crate::FuelCell>>,
    ) {
        let addr = self.grant_budget_node.load(Ordering::Acquire);
        if addr == 0 {
            return (None, None);
        }
        let node = core::mem::transmute::<usize, crate::ChildBudgetNode>(addr)(ctx);
        let fuel = node.clone().and_then(crate::FuelCell::metering);
        (node, fuel)
    }

    /// A clone of the shared subtree residue sink + task counter, to hand to a nested child's nursery
    /// so its (and its descendants') freeze residue coalesces at the root.
    pub(crate) fn nested_sink(&self) -> std::sync::Arc<Mutex<Vec<crate::FrozenNested>>> {
        std::sync::Arc::clone(&self.frozen_nested_sink)
    }
    pub(crate) fn task_counter(&self) -> std::sync::Arc<std::sync::atomic::AtomicUsize> {
        std::sync::Arc::clone(&self.task_counter)
    }

    /// §4 — this nursery's owner is unwinding for a freeze: record each of its still-unjoined durable
    /// §14 children into the **shared** subtree sink (coalesces at the root). A child that unwound
    /// into its carve is re-attached on thaw; one that finished first carries its result, which the
    /// thaw delivers to the owner's rewound `join` without re-running it (#1692), as the interpreter
    /// does: its value, or the trap its `join` re-raises (#1674).
    pub(crate) fn freeze_unjoined(&self) {
        let children = self.children.lock().unwrap_or_else(|e| e.into_inner());
        let mut sink = self
            .frozen_nested_sink
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        for c in children.iter().filter(|c| !c.joined) {
            let Some((rec, unwound)) = &c.nested else {
                continue;
            };
            let mut rec = rec.clone();
            if !unwound {
                let (result, trap) = c
                    .done
                    .state
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .expect("a durable child runs synchronously, so it has finished");
                rec.completed_result = Some(if trap == 0 { Ok(result) } else { Err(trap) });
            }
            sink.push(rec);
        }
    }

    /// Drain the §14 nested-child freeze residue captured during a durable freeze (see
    /// [`Nursery::frozen_nested_sink`]). Called by the top-level run after the root unwinds; drains the
    /// whole subtree's residue (it coalesced here via the shared sink).
    pub(crate) fn take_frozen_nested(&self) -> Vec<crate::FrozenNested> {
        std::mem::take(
            &mut self
                .frozen_nested_sink
                .lock()
                .unwrap_or_else(|e| e.into_inner()),
        )
    }

    /// The module's functions (for a thaw to re-run a frozen §14 **same-module** child over its carve).
    pub(crate) fn funcs(&self) -> std::sync::Arc<[Func]> {
        std::sync::Arc::clone(&self.funcs)
    }

    /// The module's type section (#922 — resolves a re-run same-module child's interned call types).
    pub(crate) fn types(&self) -> std::sync::Arc<[TypeEntry]> {
        std::sync::Arc::clone(&self.types)
    }

    /// The parent run's §5 kill-path interrupt cell (a re-attached thaw child polls the same cell).
    pub(crate) fn epoch_addr(&self) -> usize {
        self.epoch_addr
    }

    /// §4 thaw: publish a re-attached child's (rewound) result at its join-table `slot`, so the
    /// parent's re-executed `join` resolves without re-running the child; and (#2041) a detached child
    /// that had completed but was not joined. The freeze recorded slots in ascending order; pad any gap
    /// with an inert placeholder to keep the index alignment.
    pub(crate) fn seed_child_result(&self, slot: usize, result: i64, trap: i64) {
        let mut children = self.children.lock().unwrap_or_else(|e| e.into_inner());
        while children.len() <= slot {
            children.push(Child::finished(0, 0));
        }
        children[slot] = Child::finished(result, trap);
    }

    /// S1c — join every child task. Called at run **teardown**, before the parent window is freed, so
    /// no child outlives the memory it copies to/from. A child still running ends with the run, as the
    /// oracle ends a child domain (DESIGN §23, D66): unless the run froze, its stop word ends it at its
    /// next poll, and the join waits for that.
    ///
    /// #1361 step 4 — a frozen run's teardown then takes the harvest: every unjoined durable detached
    /// child, with its powerbox handed over (see [`crate::DetachedHarvest`]). Empty unless `froze`.
    pub(crate) fn join_children(&self, froze: bool) -> Vec<crate::DetachedHarvest> {
        // D66 — drive the executor to quiescence: parked tasks are poisoned so they unwind, and
        // unless this is a freeze every running task ends with the run; then the workers are joined.
        // It is the whole run's, so every descendant's task has ended too (#1956).
        self.child_exec.shutdown_and_join(!froze);
        let harvest = if froze {
            self.take_detached_harvest()
        } else {
            Vec::new()
        };
        self.release_retained();
        harvest
    }

    /// CALLS.md 5c.0 — release each child's nursery-retained shared-powerbox ref (minted live-impls
    /// hold their own counted refs, so a parent-held offer handle stays valid at the host layer; the
    /// run is over regardless), and (#1956) every descendant nursery's. Only once the executor has
    /// drained, so no child task still runs against a `Host` while its last-but-one ref drops.
    /// Exactly once per child: `take` zeroes the field, as the harvest's does for the ones it hands
    /// over.
    fn release_retained(&self) {
        for kid in self.kids.lock().unwrap_or_else(|e| e.into_inner()).iter() {
            kid.release_retained();
        }
        let release_addr = self.grant_release.load(Ordering::Acquire);
        if release_addr != 0 {
            let release: crate::GrantChildReleaser = unsafe { core::mem::transmute(release_addr) };
            let mut children = self.children.lock().unwrap_or_else(|e| e.into_inner());
            for c in children.iter_mut() {
                let retained = std::mem::take(&mut c.retained);
                if retained != 0 {
                    // SAFETY: `retained` is a live `GrantChild::retained_ctx` this nursery owns,
                    // released exactly once here (spawn error paths released theirs before filing).
                    unsafe { release(retained as *mut core::ffi::c_void) };
                }
            }
        }
    }

    /// #1361 step 4 — a freeze reached this domain: ring every live durable detached child's doorbell,
    /// which is a store of `UNWINDING` into the child's **own** freeze word (what
    /// [`crate::FreezeController::request_freeze`] does for the root). The child unwinds at its next
    /// poll like any root and its task deposits its window image at finish. A child being joined is
    /// rung too (#2010): its joiner is a domain the same freeze reaches, which may itself be parked.
    pub(crate) fn ring_detached(&self) {
        for c in self
            .children
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
        {
            c.ring();
        }
    }

    /// The unjoined child at `slot`: the powerbox this nursery retains for it (`0`: its builder
    /// shared none) and whether its outcome is in, as [`poll`] reads it. `None` for a forged or
    /// joined handle — the interp's dead thread slot. What `child_offer` and `grant` reach a child by.
    fn unjoined(&self, slot: i32) -> Option<(usize, bool)> {
        let children = self.children.lock().unwrap_or_else(|e| e.into_inner());
        let c = children.get(slot as usize).filter(|c| !c.joined)?;
        let ended = c
            .done
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_some();
        Some((c.retained, ended))
    }

    /// `Instantiator.kill` (#2074): end the child at `slot` and every child under it, running or
    /// parked ([`crate::child_exec::ChildExec::kill`]). A child that has finished is left as it is.
    /// `false` if there is no such slot.
    fn kill_slot(&self, slot: usize) -> bool {
        let target = self
            .children
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(slot)
            .map(|c| (c.task, c.kid.clone()));
        let Some((task, kid)) = target else {
            return false;
        };
        self.child_exec.kill(task);
        if let Some(kid) = kid {
            kid.kill_all();
        }
        true
    }

    /// [`Nursery::kill_slot`] for every child of this nursery: its spawner was killed.
    fn kill_all(&self) {
        let n = self
            .children
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len();
        for slot in 0..n {
            self.kill_slot(slot);
        }
    }

    /// #2010 — ring the child at `slot` the way [`Nursery::ring_detached`] rings each one.
    fn ring_slot(&self, slot: usize) {
        if let Some(c) = self
            .children
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(slot)
        {
            c.ring();
        }
    }

    /// #1361 step 4 — every unjoined durable detached child after a frozen run's teardown, with its
    /// powerbox handed over, and (#2010) its own harvest nested in it. #2044 — and every one a thaw
    /// delivered as completed, as its outcome alone: it has no powerbox, window or children.
    fn take_detached_harvest(&self) -> Vec<crate::DetachedHarvest> {
        let mut children = self.children.lock().unwrap_or_else(|e| e.into_inner());
        let mut out = Vec::new();
        for (slot, c) in children.iter_mut().enumerate() {
            if c.joined {
                continue;
            }
            if c.delivered {
                out.push(crate::DetachedHarvest {
                    slot,
                    entry: 0,
                    mapped_log2: 0,
                    reserved_log2: 0,
                    powerbox: core::ptr::null_mut(),
                    capture: None,
                    outcome: *c.done.state.lock().unwrap_or_else(|e| e.into_inner()),
                    children: Vec::new(),
                });
                continue;
            }
            let Some(d) = c.done.durable.as_ref() else {
                continue;
            };
            out.push(crate::DetachedHarvest {
                slot,
                entry: d.entry,
                mapped_log2: d.mapped_log2,
                reserved_log2: d.reserved_log2,
                powerbox: std::mem::take(&mut c.retained) as *mut core::ffi::c_void,
                capture: d.capture.lock().unwrap_or_else(|e| e.into_inner()).take(),
                outcome: *c.done.state.lock().unwrap_or_else(|e| e.into_inner()),
                children: c
                    .kid
                    .as_ref()
                    .map_or_else(Vec::new, |k| k.take_detached_harvest()),
            });
        }
        out
    }

    /// #1361 step 4 — a **thaw**: re-launch a captured detached child at its recorded join slot, on a
    /// fresh window holding its image, restored to `REWINDING` (the freeze word `NORMAL`, context 0's
    /// thaw word `REWINDING`), running its own program over its restored powerbox. The parent's
    /// rewound `join` then parks on it as before the cut. `false` if its code will not compile or the
    /// executor refuses it; the slot is then left to fail closed at the join.
    ///
    /// #2010 — a child that spawned gets its nursery back, and its own children are re-created in
    /// it first, so its rewound `join` parks on them as before the cut too.
    ///
    /// #2041 — a child that had completed but was not joined is re-created as its outcome only: it
    /// owns no window to restore, so the rewound `join` reloads its result.
    ///
    /// # Safety
    /// A captured seed's `child` holds two live counted refs to the child's powerbox, owned from here
    /// on, and so does each of its captured `children`, at every depth.
    pub(crate) unsafe fn relaunch_detached(&self, seed: crate::DetachedSeed) -> bool {
        let seed = match seed {
            crate::DetachedSeed::Completed { slot, result, trap } => {
                self.seed_child_result(slot, result, trap);
                if let Some(c) = self
                    .children
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .get_mut(slot)
                {
                    c.delivered = true;
                }
                return true;
            }
            crate::DetachedSeed::Captured(seed) => *seed,
        };
        let crate::CapturedSeed {
            slot,
            entry,
            mapped_log2,
            reserved_log2,
            funcs,
            types,
            shadow,
            image,
            prots,
            child: gc,
            window,
            children,
            residue,
        } = seed;
        let release_addr = self.grant_release.load(Ordering::Acquire);
        if release_addr == 0 {
            return false;
        }
        let release: crate::GrantChildReleaser = core::mem::transmute(release_addr);
        let thunk_addr = self.grant_thunk.load(Ordering::Acquire);
        let child_thunk: crate::CapThunk = if thunk_addr != 0 {
            core::mem::transmute::<usize, crate::CapThunk>(thunk_addr)
        } else {
            self.cap_thunk
        };
        let kid = self.for_child(&gc, &funcs);
        for c in children {
            match &kid {
                Some(kid) => {
                    kid.relaunch_detached(c);
                }
                None => release_seed(release, c),
            }
        }
        // #1944 slice 3 — the thawed child draws from its budget node again, as at its spawn; what
        // it drew before the freeze and did not burn went back as the freeze unwound it.
        let (node, child_fuel) = self.detached_node(gc.ctx);
        let Ok(code) = crate::compile_child_windowed(
            &funcs,
            &types,
            entry as FuncIdx,
            mapped_log2,
            reserved_log2,
            child_thunk,
            gc.ctx,
            self.epoch_addr,
            child_fuel
                .as_deref()
                .map_or(0, |c| c as *const crate::FuelCell as usize),
            self.futex_sched,
            kid.as_deref()
                .map_or(crate::InstEnv::null(), crate::InstEnv::over),
            &self.serve_handlers,
            gc.jit_table_log2,
            shadow,
            // A thawed child runs as a durable executor task, as at its spawn.
            crate::ChildRun::DurableTask,
        ) else {
            release(gc.ctx);
            release(gc.retained_ctx);
            return false;
        };
        let n_args = funcs.get(entry as usize).map_or(1, |f| f.params.len());
        let n_results = funcs.get(entry as usize).map_or(1, |f| f.results.len());
        let code = std::sync::Arc::new(code);
        register_serve(self, gc.ctx, &code);
        let teardown = granted_teardown(self, release, gc.ctx, gc.lane_cap, window, child_fuel);
        let thaw_off = shadow.thaw_state_off(0) as usize;
        // #2025 — its §13 pages, aliased back onto their regions once its window exists, by the
        // same host hook that applies a spawn's op-15 pre-map.
        let apply_addr = self.grant_premap_apply.load(Ordering::Acquire);
        let premap_apply = (apply_addr != 0)
            .then(|| core::mem::transmute::<usize, crate::WindowReady>(apply_addr));
        let premap_ctx = SendRaw(gc.ctx);
        let filed = file_task(
            self,
            code,
            mapped_log2,
            reserved_log2,
            move |w| {
                let rw = w.rw_mut();
                let n = image.len().min(rw.len());
                rw[..n].copy_from_slice(&image[..n]);
                let s = temen_ir::durable_abi::STATE_OFF as usize;
                if let Some(st) = rw.get_mut(s..s + 4) {
                    st.copy_from_slice(&temen_ir::durable_abi::STATE_NORMAL.to_le_bytes());
                }
                if let Some(th) = rw.get_mut(thaw_off..thaw_off + 4) {
                    th.copy_from_slice(&temen_ir::durable_abi::STATE_REWINDING.to_le_bytes());
                }
                // Its protections as captured (the guard follows, as on every child window).
                let mapped = rw.len() as u64;
                w.apply_prots(0, &prots, mapped, &image);
            },
            // Its page map was restored with the window's (`prots` above), read-only pages and all.
            |base, mapped, reserved| match premap_apply {
                // SAFETY: `premap_ctx.0` is the child powerbox this task owns; the window is live.
                Some(apply) => apply(premap_ctx.0, base, mapped, reserved, [].as_ptr(), 0) != 0,
                None => true,
            },
            None,
            vec![0; n_args], // inert under a rewind: the prologue reloads spilled values
            n_results,
            self.lane_chain_of(&gc),
            gc.retained_ctx as usize,
            teardown,
            Some(slot),
            Some(DurableCell::new(
                shadow,
                entry,
                mapped_log2,
                reserved_log2,
                self.child_high_water(gc.retained_ctx),
            )),
            node,
            kid,
            residue,
        );
        matches!(filed, Filed::Slot(_))
    }

    /// Mark the run durable (DURABILITY.md §4) — see the [`Nursery::durable`] field: the nesting
    /// thunks then fail closed. Called at run entry (`run_code_raw`), after the entry wrappers
    /// have applied the compile-side durable flag.
    pub(crate) fn set_durable(&self, durable: bool) {
        self.durable.store(durable, Ordering::Release);
    }

    /// Set the run's async freeze controller (#1693) — see [`Nursery::freeze`].
    pub(crate) fn set_freeze(&self, fc: Option<std::sync::Arc<crate::FreezeController>>) {
        *self.freeze.lock().unwrap_or_else(|e| e.into_inner()) = fc;
    }

    /// The run's async freeze controller, for a durable child to register with (#1693).
    pub(crate) fn freeze(&self) -> Option<std::sync::Arc<crate::FreezeController>> {
        self.freeze
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Resolve a `Module` handle through the host (`temen-run`'s `module_resolver`) — including
    /// `SELF_MODULE` (`-1`), which the host resolves to the spawning code's program *with* its data
    /// segments and declared memory: the running module's, or from an installed unit's code (`unit`,
    /// [`crate::unit_ref`]) the unit's (#2143). A detached child needs those (its window is fresh),
    /// where a carve child of `-1` runs over the parent's bytes and takes only
    /// [`Self::self_program`] (#1863).
    #[allow(clippy::type_complexity)]
    unsafe fn resolve_granted(
        &self,
        module: i64,
        unit: i64,
        trap_out: *mut i64,
    ) -> Option<(
        &[Func],
        &[TypeEntry],
        Option<i32>,
        &[Data],
        temen_ir::durable_abi::ShadowArena,
    )> {
        let Some(resolver) = self.resolve_module else {
            *trap_out = TrapKind::CapFault as i64;
            return None;
        };
        let mut rm = core::mem::MaybeUninit::<crate::ResolvedModule>::zeroed().assume_init();
        if resolver(self.cap_ctx, module as i32, unit, &mut rm) == 0 || rm.n_funcs == 0 {
            *trap_out = TrapKind::CapFault as i64;
            return None;
        }
        let funcs = std::slice::from_raw_parts(rm.funcs, rm.n_funcs);
        let data = if rm.n_data == 0 {
            &[][..]
        } else {
            std::slice::from_raw_parts(rm.data, rm.n_data)
        };
        // #922: the child's type section (empty ⇒ a module with no interned call sites).
        let types = if rm.n_types == 0 {
            &[][..]
        } else {
            std::slice::from_raw_parts(rm.types, rm.n_types)
        };
        Some((funcs, types, Some(rm.memory_log2), data, rm.shadow))
    }

    /// #1501 — whether the module a detached spawn names is attested **freezable** (instrumented): the
    /// other half of what a durable domain may spawn. `-1` asks the host too, which attests neither the
    /// running module's grant nor a unit's, so a durable domain refuses it, as on the oracle.
    unsafe fn child_module_durable(&self, module: i64, unit: i64) -> bool {
        let Some(resolver) = self.resolve_module else {
            return false;
        };
        let mut rm = core::mem::MaybeUninit::<crate::ResolvedModule>::zeroed().assume_init();
        resolver(self.cap_ctx, module as i32, unit, &mut rm) != 0 && rm.durable
    }

    /// Resolve `handle` as this domain's `Instantiator` via the run's `call.cap` thunk, returning its
    /// carve range `[base, base+size)`. `None` (and `*trap_out` set) for a forged/closed/wrong handle.
    unsafe fn resolve(&self, mem_base: u64, handle: i32, trap_out: *mut i64) -> Option<(u64, u64)> {
        let mut out = [0i64; 2];
        // op 0 on an `Instantiator` binding returns `[base, size]` (see `cap_dispatch_slots`); a bad
        // handle sets `*trap_out` to a `CapFault` and we propagate by returning `None`.
        (self.cap_thunk)(
            self.cap_ctx,
            mem_base as *mut u8,
            0,
            0,
            temen_ir_iface_instantiator(),
            0,
            handle,
            core::ptr::null(),
            0,
            out.as_mut_ptr(),
            out.len() as u64,
            trap_out,
        );
        if unsafe { *trap_out } != 0 {
            return None;
        }
        Some((out[0] as u64, out[1] as u64))
    }
}

/// The `Instantiator` interface id (§3e), kept in lockstep with `temen_interp::cap_id::INSTANTIATOR`.
/// (`temen-jit` does not depend on `temen-interp`; the host dispatch on the other side checks the same
/// constant, and the cross-backend tests pin them equal.)
#[inline]
fn temen_ir_iface_instantiator() -> u32 {
    6
}

/// CONSOLIDATION.md §3c — the **config-record spawn** (`Instantiator` op 17,
/// `instantiate_rec(record_ptr)`) on the native JIT tier: parse + validate the record
/// ([`SpawnRec`]) from the guest window, then **delegate to [`instantiate_detached`]** — a record
/// is op 15 as data (#1863), so there is exactly one spawn implementation on this tier too. Field
/// handling mirrors the tree-walker's op-17 arm:
///
/// - a version other than 1, or a nonzero reserved field → `CapFault` (fail closed).
/// - `pager != u32::MAX` → `CapFault`. Sound, not a divergence: `temen-run` folds op-17 modules
///   **with** impl exports to the oracle, so a natively-running module has none — and the
///   interpreter fails its pager validation (`Module::is_pager_export`) identically.
///
/// # Safety
/// Same contract as [`instantiate_detached`]: called from JITted code on the spawning
/// vCPU's thread with the run's live nursery, window base/size, and a writable `trap_out`.
pub(crate) unsafe extern "C" fn instantiate_rec(
    rt: *const Nursery,
    self_prog: i64,
    mem_base: u64,
    mem_size: u64,
    handle: i32,
    record_ptr: i64,
    trap_out: *mut i64,
) -> i32 {
    let rp = record_ptr as u64;
    // #826: `mem_size` is the reserved span, so a record in a `map`-grown tail page is read where
    // the interpreter's live page map reads it; a record on a still-uncommitted page faults the
    // copy below through the SIGSEGV guard (`MemoryFault`), exactly like a guest load.
    if rp
        .checked_add(temen_ir::SPAWN_REC_LEN as u64)
        .is_none_or(|e| e > mem_size)
    {
        *trap_out = TrapKind::MemoryFault as i64;
        return 0;
    }
    // Copy the record out of the guest window, then share the layout decode with the two
    // interpreter tiers (#911).
    let mut buf = [0u8; temen_ir::SPAWN_REC_LEN];
    core::ptr::copy_nonoverlapping(
        (mem_base + rp) as *const u8,
        buf.as_mut_ptr(),
        temen_ir::SPAWN_REC_LEN,
    );
    let Some(sr) = SpawnRec::parse(&buf) else {
        *trap_out = TrapKind::CapFault as i64;
        return 0;
    };
    // No pager on this tier (above).
    if sr.pager != u32::MAX {
        *trap_out = TrapKind::CapFault as i64;
        return 0;
    }
    instantiate_detached(
        rt,
        self_prog,
        mem_base,
        mem_size,
        handle,
        sr.budget as i64,
        sr.modh as i64,
        sr.grants_ptr as i64,
        sr.grants_n as i64,
        sr.entry as i64,
        sr.size_log2,
        sr.quota,
        sr.args.0 as i64,
        sr.args.1 as i64,
        sr.region as i64,
        sr.child_off as i64,
        trap_out,
    )
}

/// `join(child_handle) -> result` — block on the child's completion (an async op-0/5/8/11/13 child
/// runs on its own OS thread; a durable child is already done) and return its `i64` result,
/// propagating a child trap as the parent's (`*trap_out`). A §5 host kill on the parent's interrupt
/// cell while parked here unwinds the waiter *as `OutOfFuel`* (see the loop below). A forged /
/// already-joined handle is inert (a `CapFault`), matching the interpreter's once-only join.
///
/// # Safety
/// As [`instantiate`]: `rt`/`trap_out` are the baked nursery + run trap cell, valid for the call.
/// #1287 — the detached twin of [`spawn_granted_child`]: the child runs on its own OS thread in a window
/// that is its own (`run_detached_child_then` — `mapped_log2` committed inside a `reserved_log2`
/// reservation, seeded from `seeds`), no carve, no copy-in/copy-back. Registered as a pending join-table
/// entry like every async child; the powerbox is released by the teardown hook while the window lives.
///
/// # Safety
/// As [`spawn_granted_child`] minus the carve: `code` was compiled against `gc_ctx` for exactly
/// `(mapped_log2, reserved_log2)`; `args` matches the entry arity.
#[allow(clippy::too_many_arguments)]
unsafe fn spawn_detached_child(
    rt: &Nursery,
    code: crate::CompiledModule,
    mapped_log2: u8,
    reserved_log2: u8,
    seeds: Vec<Seed>,
    window_ready: Option<crate::WindowReady>,
    args: Vec<i64>,
    n_results: usize,
    release: crate::GrantChildReleaser,
    gc: &crate::GrantChild,
    trap_out: *mut i64,
    // #1587 — the lease `instantiate_detached` already took from the funding budget, `(budget, bytes,
    // runs)`, so a spawn that fails *after* that commit can hand it back.
    lease: (i32, u64, u64),
    // #1944 slice 3 — the child's fuel cell, drawing from its own budget node (`None`: un-metered);
    // its task's teardown returns what it did not burn.
    fuel: Option<Box<crate::FuelCell>>,
    // #2001 — that node, which the child's threads are charged to.
    node: Option<std::sync::Arc<dyn crate::BudgetNode>>,
    // #1361 step 4 — `Some(arena)` for a durable parent's child: its window starts as a durable one
    // (context 0's shadow-SP word at its frame base) and carries a freeze cell recording `entry`.
    durable: Option<temen_ir::durable_abi::ShadowArena>,
    entry: u32,
    // #1760, #2010 — the spawner's window, whose freeze word a durable child's follows: one spawned
    // by a parent already unwinding for a freeze starts `UNWINDING` (see [`init_durable_words`]),
    // and one a freeze's ring missed while it was being filed is rung once it is.
    spawner_base: u64,
    // #1956 — the nursery `code` was compiled against, if the child spawns.
    kid: Option<std::sync::Arc<Nursery>>,
) -> i32 {
    // SAFETY: a durable spawner's window is live for the call, and its first page holds the word.
    let parent_freezing =
        || durable.is_some() && unsafe { crate::fiber_rt::window_is_unwinding(spawner_base) };
    let freezing = parent_freezing();
    let code = std::sync::Arc::new(code);
    register_serve(rt, gc.ctx, &code);
    let teardown = granted_teardown(rt, release, gc.ctx, gc.lane_cap, Some(lease), fuel);
    let premap_ctx = SendRaw(gc.ctx);
    // The segments that fit the window, and of them the read-only ranges its window-ready hook hands
    // the child's page map (#2268).
    let len = 1usize << mapped_log2;
    let seeds: Vec<Seed> = seeds
        .into_iter()
        .filter(|s| {
            (s.offset as usize)
                .checked_add(s.bytes.len())
                .is_some_and(|end| end <= len)
        })
        .collect();
    let ro: Vec<[u64; 2]> = seeds
        .iter()
        .filter(|s| s.readonly && !s.bytes.is_empty())
        .map(|s| [s.offset, s.offset + s.bytes.len() as u64])
        .collect();
    let filed = file_task(
        rt,
        code,
        mapped_log2,
        reserved_log2,
        // The window image: the module's data segments, then the payload at the args base; a durable
        // child's window also starts durable (`temen_durable::init_durable_window`'s one word). Then
        // the `readonly` segments go RO, as in a root's window (#1730): a detached child is
        // root-shaped, so a write to its const data faults.
        |w| {
            let rw = w.rw_mut();
            for s in &seeds {
                let off = s.offset as usize;
                rw[off..off + s.bytes.len()].copy_from_slice(&s.bytes);
            }
            if let Some(a) = durable {
                init_durable_words(rw, a, freezing);
            }
            for s in seeds.iter().filter(|s| s.readonly) {
                w.protect_ro(s.offset, s.bytes.len() as u64);
            }
        },
        // The child's page map takes its read-only pages, and the op-15 pre-mapped region, if any,
        // is aliased onto the fresh window (the child powerbox's own `map` path), by the host hook.
        |base, mapped, reserved| match window_ready {
            // SAFETY: `premap_ctx.0` is the child powerbox this task owns; the window is live.
            Some(ready) => ready(premap_ctx.0, base, mapped, reserved, ro.as_ptr(), ro.len()) != 0,
            None => true,
        },
        None,
        args,
        n_results,
        rt.lane_chain_of(gc),
        gc.retained_ctx as usize,
        teardown,
        None,
        durable.map(|a| {
            DurableCell::new(
                a,
                entry,
                mapped_log2,
                reserved_log2,
                rt.child_high_water(gc.retained_ctx),
            )
        }),
        node,
        kid,
        crate::DurableResidue::default(),
    );
    match filed {
        Filed::Slot(slot) => {
            // #2010 — a freeze that reached the spawner while the child was being filed rang its
            // children without this one: ring it now, as that ring would have.
            if parent_freezing() {
                rt.ring_slot(slot as usize);
            }
            slot
        }
        Filed::AtCeiling => {
            *trap_out = TrapKind::ThreadFault as i64;
            0
        }
        // The child never ran (a refused pre-map alias or task stack) — the one refusal on this
        // path that happens *after* `budget_mem_take` committed. Its teardown already ran, and that
        // un-spent the window bytes (#1587), so answer `-EINVAL` like every other admission failure
        // (INVARIANTS #5).
        Filed::Refused => EINVAL as i32,
    }
}

/// One run of bytes a detached child's fresh window starts with: a data segment, or the args
/// payload.
struct Seed {
    offset: u64,
    bytes: Vec<u8>,
    readonly: bool,
}

/// A fresh durable window's control words, as `temen_durable::init_durable_window` writes them: the
/// freeze word (`NORMAL`, unless `freezing`) and context 0's shadow-SP word at its frame base (the
/// empty stack).
///
/// #1760 — a child spawned by a parent that is already unwinding for a freeze starts `UNWINDING`,
/// as if the freeze's doorbell had rung at its birth: the freeze was requested before the child
/// existed, so the child freezes at its first poll rather than running on until the parent's
/// harvest reaches it — how far it got used to depend on how the OS scheduled its thread.
fn init_durable_words(rw: &mut [u8], a: temen_ir::durable_abi::ShadowArena, freezing: bool) {
    use temen_ir::durable_abi::{STATE_NORMAL, STATE_OFF, STATE_UNWINDING};
    let s = STATE_OFF as usize;
    let b = a.region_base(0) as usize;
    let state = if freezing {
        STATE_UNWINDING
    } else {
        STATE_NORMAL
    };
    if let Some(st) = rw.get_mut(s..s + 4) {
        st.copy_from_slice(&state.to_le_bytes());
    }
    if let Some(sp) = rw.get_mut(b..b + 8) {
        sp.copy_from_slice(&a.frame_base(0).to_le_bytes());
    }
}

/// PROCESS.md §5 / #1287 — `instantiate_detached(budget, module, grants_ptr, grants_n, entry,
/// size_log2, quota[, args_ptr, args_len[, region, child_off]]) -> child | -EINVAL` on the native JIT:
/// a separate-module
/// child in a **fresh window** — `1 << size_log2` committed inside a root-sized lazy reservation, no
/// carve, no alias — minted through the `Budget` `budget`. Admission is the interpreter's op-15
/// arm errno-for-errno: child entry shape, the window **equals** the module's declared memory (§14
/// transparency), the payload fits the args region, and the budget quota covers the window (a forged /
/// exhausted budget refuses, charging nothing). The child powerbox is the by-name grant list plus
/// starter caps spanning the **reservation** (a root's shape, so its `vm_map` grows the window);
/// it attests `window_exposed = false`. Data segments + the payload are seeded into the child's own
/// window before it starts. A durable run refuses outright (multi-window freeze is O6).
///
/// # Safety
/// `rt`/`mem_base`/`mem_size`/`trap_out` are the baked nursery, live parent window base, **reserved**
/// span, and run trap cell, valid for the call; a nonzero
/// `self_prog` is a live [`UnitProg`] ([`Nursery::self_program`]).
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe extern "C" fn instantiate_detached(
    rt: *const Nursery,
    self_prog: i64,
    mem_base: u64,
    mem_size: u64,
    handle: i32,
    budget: i64,
    module: i64,
    grants_ptr: i64,
    grants_n: i64,
    entry: i64,
    size_log2: i64,
    fuel: i64,
    args_ptr: i64,
    args_len: i64,
    premap_region: i64,
    premap_off: i64,
    trap_out: *mut i64,
) -> i32 {
    let rt = &*rt;
    // #1944 slice 3 — `quota` is retired: the budget is the one way to limit a child's fuel, so any
    // other value fails closed, checked first as the tree-walker's op-15 arm does.
    if fuel != 0 {
        *trap_out = TrapKind::CapFault as i64;
        return 0;
    }
    // #1361 step 4 — a durable parent's detached child is captured by the parent's freeze (its window
    // rides the artifact as its own), so it spawns durable: an attested-freezable module with a shadow
    // arena of its own (§4, #1501). The authority half — freeze authority over detached progeny, #1440 —
    // is the shared admission's (`Host::admit_detached_spawn`, via the budget take below).
    let durable = rt.durable.load(Ordering::Acquire);
    let build_addr = rt.grant_build_detached.load(Ordering::Acquire);
    let release_addr = rt.grant_release.load(Ordering::Acquire);
    let take_addr = rt.grant_budget_mem_take.load(Ordering::Acquire);
    if build_addr == 0 || release_addr == 0 || take_addr == 0 {
        *trap_out = TrapKind::CapFault as i64;
        return 0;
    }
    let build: crate::GrantDetachedChildBuilder = core::mem::transmute(build_addr);
    let release: crate::GrantChildReleaser = core::mem::transmute(release_addr);
    let take: crate::BudgetMemTaker = core::mem::transmute(take_addr);
    let thunk_addr = rt.grant_thunk.load(Ordering::Acquire);
    let child_thunk: crate::CapThunk = if thunk_addr != 0 {
        core::mem::transmute::<usize, crate::CapThunk>(thunk_addr)
    } else {
        rt.cap_thunk
    };
    // The Instantiator is the authority (a forged handle is a CapFault); its carve range is unused —
    // a detached child lives in no carve.
    if rt.resolve(mem_base, handle, trap_out).is_none() {
        return 0;
    }
    // The installed unit the spawning code is, if any: from it, `module = -1` names the unit's own
    // program (#2143). One no host names cannot spawn that program.
    let unit = if self_prog == 0 {
        crate::NO_UNIT
    } else {
        (*(self_prog as *const UnitProg)).unit
    };
    if module < 0 && self_prog != 0 && unit == crate::NO_UNIT {
        *trap_out = TrapKind::CapFault as i64;
        return 0;
    }
    let Some((child_funcs, child_types, mod_mem, child_data, child_shadow)) =
        rt.resolve_granted(module, unit, trap_out)
    else {
        return 0;
    };
    let entry = entry as u64;
    let size_log2 = temen_ir::detached_size_log2(size_log2, mod_mem.map(|m| m as u8));
    let child_size = if (0..64).contains(&size_log2) {
        1u64 << size_log2
    } else {
        0
    };
    let arity = child_funcs
        .get(entry as usize)
        .map_or(0, |f| f.params.len());
    // The one entry-shape rule every tier shares (#911, #1720): the child-entry ABI or a powerbox
    // `_start` — never a private copy that could admit what the oracle refuses, or refuse what it runs.
    let ok_entry = child_funcs
        .get(entry as usize)
        .is_some_and(|f| temen_ir::child_entry_ok(&f.params, &f.results));
    // §14 transparency: the detached window equals the module's declared memory (the interpreter's
    // op-15 `mod_ok`); it grows into its own reservation, so no superset room is needed.
    let mod_ok = mod_mem == Some(size_log2 as i32);
    // The optional payload, read from THIS window (reserved-bounded — an out-of-window range is a
    // MemoryFault, as any window read); an over-long one refuses probeably.
    let payload: Vec<u8> = if args_len > 0 {
        let (p, l) = (args_ptr as u64, args_len as u64);
        match p.checked_add(l) {
            Some(end) if end <= mem_size => {
                std::slice::from_raw_parts((mem_base as *const u8).add(p as usize), l as usize)
                    .to_vec()
            }
            _ => {
                *trap_out = TrapKind::MemoryFault as i64;
                return 0;
            }
        }
    } else {
        Vec::new()
    };
    let args_room = temen_ir::module_args_end() - temen_ir::module_args_base();
    let durable_ok = !durable
        || (rt.child_module_durable(module, unit)
            && child_shadow != temen_ir::durable_abi::ShadowArena::EMPTY);
    if !ok_entry
        || child_size == 0
        || !mod_ok
        || !durable_ok
        || payload.len() as u64 > args_room
        || size_log2 as u8 > crate::MAX_JIT_WINDOW_LOG2
    {
        return EINVAL as i32;
    }
    // The optional pre-mapped region (`premap_region < 0` ⇒ none): admitted host-side — the
    // interpreter's `Host::premap_admit`, a forged handle traps, bad geometry refuses — before the take.
    let premap = (premap_region >= 0).then_some((premap_region as i32, premap_off as u64));
    // #2268 — the child's window-ready hook, which records its read-only pages and applies the
    // pre-map, if any.
    let apply_addr = rt.grant_premap_apply.load(Ordering::Acquire);
    let window_ready =
        (apply_addr != 0).then(|| core::mem::transmute::<usize, crate::WindowReady>(apply_addr));
    if let Some((r, o)) = premap {
        let admit_addr = rt.grant_premap_admit.load(Ordering::Acquire);
        let stage_addr = rt.grant_premap_stage.load(Ordering::Acquire);
        if admit_addr == 0 || stage_addr == 0 || window_ready.is_none() {
            *trap_out = TrapKind::CapFault as i64;
            return 0;
        }
        let admit: crate::PremapAdmit = core::mem::transmute(admit_addr);
        match admit(rt.grant_ctx(), r, o, child_size, trap_out) {
            1 => {}
            0 => return EINVAL as i32,
            _ => return 0, // forged: `*trap_out` is set
        }
    }
    // Admission = the lane reserved and the budget's window charge (the commit; every refusal above
    // charged nothing).
    // #2268 — the lease: the window's bytes, its child's first vCPU, and the mappings its domain and
    // window will hold, bounded by its program's read-only segments and the pre-map.
    let lease = (
        budget as i32,
        child_size,
        temen_ir::window_runs_bound(child_data, premap.is_some()),
    );
    let lane = take(rt.grant_ctx(), lease.0, lease.1, lease.2);
    if lane == crate::ADMIT_REFUSED {
        return EINVAL as i32;
    }
    let reservation = 1u64 << temen_ir::DEFAULT_RESERVED_LOG2;
    let mut gc = crate::GrantChild {
        ctx: core::ptr::null_mut(),
        retained_ctx: core::ptr::null_mut(),
        inst_handle: 0,
        as_handle: 0,
        grant_handle: 0,
        jit_table_log2: 0,
        domain: 0,
        lane_cap: -1,
        parent_domain: 0,
        parent_lane_cap: -1,
    };
    if build(
        rt.grant_ctx(),
        mem_base as *mut u8,
        mem_size,
        grants_ptr as u64,
        grants_n as u64,
        reservation, // starter caps span the reservation — a root's shape
        lease.0,     // the lease `take` just charged: its budget the child's `"budget"`
        lease.1,
        lease.2,
        lane, // and the lane it reserved
        &mut gc,
        trap_out,
    ) == 0
    {
        undo_admission(rt, lease, lane);
        return 0;
    }
    // #1975 — every exit from here that files no child hands back what the admission took, as the
    // one above does.
    let bind_addr = rt.grant_bind_imports.load(Ordering::Acquire);
    if bind_addr != 0 {
        let bind: crate::ChildManifestBinder = core::mem::transmute(bind_addr);
        if bind(rt.grant_ctx(), gc.ctx, module, unit) != 0 {
            release(gc.ctx);
            release(gc.retained_ctx);
            undo_admission(rt, lease, lane);
            return EINVAL as i32;
        }
    }
    // Stage the pre-mapped region into the built child powerbox (a re-grant, as the grant list); the
    // alias itself is applied on the child thread once its window exists (`premap_apply` above).
    if let Some((r, o)) = premap {
        let stage: crate::PremapStage =
            core::mem::transmute(rt.grant_premap_stage.load(Ordering::Acquire));
        if stage(rt.grant_ctx(), gc.ctx, r, o) == 0 {
            release(gc.ctx);
            release(gc.retained_ctx);
            undo_admission(rt, lease, lane);
            *trap_out = TrapKind::CapFault as i64;
            return 0;
        }
    }
    // #1944 slice 3 — the child draws from the budget that paid for its window, its own node.
    let (node, child_fuel) = rt.detached_node(gc.ctx);
    let child_fuel_addr = child_fuel
        .as_deref()
        .map_or(0, |c| c as *const crate::FuelCell as usize);
    // #1956 — a child that spawns gets a nursery of its own; a durable child's is durable, and a
    // freeze reaches its children through it (#2010).
    let kid = rt.for_child(&gc, child_funcs);
    let compiled = crate::compile_child_windowed(
        child_funcs,
        child_types,
        entry as FuncIdx,
        size_log2 as u8,
        temen_ir::DEFAULT_RESERVED_LOG2,
        child_thunk,
        gc.ctx,
        rt.epoch_addr,
        child_fuel_addr,
        rt.futex_sched,
        kid.as_deref()
            .map_or(crate::InstEnv::null(), crate::InstEnv::over),
        &rt.serve_handlers,
        gc.jit_table_log2, // #1296: slots for the units a `Jit`-holding child installs,
        child_shadow,
        // An executor task (#1469), durable when its parent is: its vCPUs unwind with it (#2010).
        if durable {
            crate::ChildRun::DurableTask
        } else {
            crate::ChildRun::Task
        },
    );
    let code = match compiled {
        Ok(code) => code,
        Err(_) => {
            release(gc.ctx);
            release(gc.retained_ctx);
            undo_admission(rt, lease, lane);
            *trap_out = TrapKind::CapFault as i64;
            return 0;
        }
    };
    let args: Vec<i64> =
        temen_ir::child_entry_handles(arity, gc.inst_handle, gc.as_handle).collect();
    let n_results = child_funcs[entry as usize].results.len();
    // The window image: the module's data segments, then the payload at the args base.
    let mut seeds: Vec<Seed> = child_data
        .iter()
        .map(|d| Seed {
            offset: d.offset,
            bytes: d.bytes.clone(),
            readonly: d.readonly,
        })
        .collect();
    if !payload.is_empty() {
        seeds.push(Seed {
            offset: temen_ir::module_args_base(),
            bytes: payload,
            readonly: false,
        });
    }
    spawn_detached_child(
        rt,
        code,
        size_log2 as u8,
        temen_ir::DEFAULT_RESERVED_LOG2,
        seeds,
        window_ready,
        args,
        n_results,
        release,
        &gc,
        trap_out,
        lease,
        child_fuel,
        node,
        durable.then_some(child_shadow),
        entry as u32,
        mem_base,
        kid,
    )
}

/// CALLS.md 5c.0 — `child_offer` (Instantiator op 14) on the JIT: mint a **live-callee offer**
/// in the parent's powerbox over a spawned granted child's nursery-retained shared `Host`.
/// Semantics mirror the interp op-14 arm errno-for-errno: every miss — forged/joined handle, a
/// plain (non-shared) child, no mint hook, bad export — is the probeable `-EINVAL`, never a trap.
/// (A call *through* the minted handle still answers the host dispatch's probeable `-EINVAL`
/// until the 5c.1 transport lands; minting is the 5c.0 slice.)
///
/// # Safety
/// `rt` is the run's live nursery; `mem_base` its live window base; `trap_out` the live trap cell
/// (written only for a forged `Instantiator`).
pub(crate) unsafe extern "C" fn child_offer(
    rt: *const Nursery,
    mem_base: u64,
    inst: i32,
    child: i32,
    export: i64,
    trap_out: *mut i64,
) -> i32 {
    const EINVAL: i32 = -22;
    let rt = &*rt;
    if rt.resolve(mem_base, inst, trap_out).is_none() {
        return 0; // a forged `Instantiator` — `*trap_out` holds the CapFault (#1729)
    }
    let mint_addr = rt.grant_mint.load(Ordering::Acquire);
    if mint_addr == 0 {
        return EINVAL;
    }
    // A joined child mirrors the interp's dead thread-slot: nothing to offer. The nursery's lock is
    // dropped before the mint (which takes the child + parent powerbox locks in turn). No release
    // race: the retained ref is freed only at `join_children` (after guest code) or on a spawn error
    // path (before the child is ever filed).
    let retained = rt.unjoined(child).map_or(0, |(r, _)| r);
    if retained == 0 {
        return EINVAL;
    }
    let mint: crate::ChildOfferMint = core::mem::transmute(mint_addr);
    mint(rt.grant_ctx(), retained as *mut core::ffi::c_void, export)
}

/// #2220 — `grant` (Instantiator op 19) on the JIT: grant `handle` into a child that is running or
/// parked, through the powerbox the nursery retains for it, with the [`crate::ChildGrant`] hook —
/// the interp op-19 arm's body. A forged `Instantiator` or `handle` is a `CapFault`. A child whose
/// outcome is in (what [`poll`] reports: it returned, trapped, or unwound for a freeze) has ended,
/// and the hook is handed no child for it, nor for a forged or joined handle or a child that shared
/// no powerbox: `-EINVAL`, as op 14's misses. A run with no hook grants nothing: `-EINVAL`.
///
/// # Safety
/// As [`child_offer`]; `trap_out` is also written for a forged `handle`.
pub(crate) unsafe extern "C" fn child_grant(
    rt: *const Nursery,
    mem_base: u64,
    inst: i32,
    child: i32,
    handle: i64,
    trap_out: *mut i64,
) -> i32 {
    let rt = &*rt;
    if rt.resolve(mem_base, inst, trap_out).is_none() {
        return 0; // a forged `Instantiator` — `*trap_out` holds the CapFault (#1729)
    }
    let grant_addr = rt.grant_into.load(Ordering::Acquire);
    if grant_addr == 0 {
        return EINVAL as i32;
    }
    // As op 14's: the nursery's lock is dropped before the hook takes the powerbox locks.
    let retained = rt
        .unjoined(child)
        .filter(|&(_, ended)| !ended)
        .map_or(0, |(r, _)| r);
    let grant: crate::ChildGrant = core::mem::transmute(grant_addr);
    grant(
        rt.grant_ctx(),
        retained as *mut core::ffi::c_void,
        handle,
        trap_out,
    )
}

pub(crate) unsafe extern "C" fn join(
    rt: *const Nursery,
    mem_base: u64,
    inst: i32,
    handle: i32,
    trap_out: *mut i64,
) -> i64 {
    join_or_wait(rt, mem_base, inst, handle, trap_out, false)
}

/// `wait(child) -> 0 | trap code` (op 18, JIT): park as [`join`] does, then answer how the child
/// ended — 0 if it returned, the child left for its `join`; else its trap's wire code, the child
/// reaped — so the spawner does not inherit the trap. The oracle's `join_delivery` rule.
///
/// # Safety
/// As [`join`].
pub(crate) unsafe extern "C" fn wait(
    rt: *const Nursery,
    mem_base: u64,
    inst: i32,
    handle: i32,
    trap_out: *mut i64,
) -> i64 {
    join_or_wait(rt, mem_base, inst, handle, trap_out, true)
}

/// [`join`], or with `wait` [`wait`]: one park for both.
///
/// # Safety
/// As [`join`].
unsafe fn join_or_wait(
    rt: *const Nursery,
    mem_base: u64,
    inst: i32,
    handle: i32,
    trap_out: *mut i64,
    wait: bool,
) -> i64 {
    let rt = &*rt;
    if rt.resolve(mem_base, inst, trap_out).is_none() {
        return 0; // a forged `Instantiator` — `*trap_out` holds the CapFault (#1729)
    }
    let mut children = rt.children.lock().unwrap_or_else(|e| e.into_inner());
    let slot = handle as usize;
    let done = match children.get_mut(slot) {
        Some(c) if !c.joined => {
            // A join claims the child now; a wait only if it trapped (below).
            c.joined = !wait;
            c.done.clone() // clone the cell + drop the `children` lock before parking
        }
        _ => {
            // A forged or already-joined child handle: `ThreadFault`, as the oracle's `take_child`
            // answers — a child rides the §12 executor as a vCPU, so this is a thread-handle
            // forgery, not a capability one (#1573). The `Instantiator` operand itself was checked
            // above and is the `CapFault` case.
            *trap_out = TrapKind::ThreadFault as i64;
            return 0;
        }
    };
    drop(children);
    // #1956 — a detached child joins from a task, so the thread that would block is an executor
    // worker: the join parks the task instead, as `thread.join` does there, and the executor gives
    // the task's lanes back while it is parked. #2322 — from a guest fiber too: the fiber's event
    // park blocks the whole task, as a join blocks its vCPU on every engine, so its resumer neither
    // runs on nor sees `FIBER_PARKED` (`fiber_rt::fiber_resume` hands the park up to the task). An
    // outcome already in is returned below without parking, on any stack.
    let task = if crate::fiber_rt::in_task() {
        match crate::fiber_rt::current_fiber_slot() {
            Some(s) => Some(s),
            // Unreachable: a task runs on its platform slot, the first of the resumes this reads.
            None => {
                *trap_out = TrapKind::ThreadFault as i64;
                return 0;
            }
        }
    } else {
        None
    };
    // D66 — a parent joining its §14 child holds no lane while it waits. Load-bearing under a cap of
    // 1: the child task cannot be dispatched at all until the joining vCPU steps aside, so without
    // this every bounded `instantiate`-then-`join` would deadlock. Released before the completion
    // cell's lock, re-taken after — blocking for a lane while holding that lock would stop the child
    // from publishing the very outcome being waited for. They are the lanes of the joining vCPU's
    // own domain (#1956: a detached child's thread runs under the child's chain). SAFETY: a nonzero
    // `futex_sched` is the run's live `Domain`, which outlives every child.
    let dom = (task.is_none() && rt.futex_sched != 0).then(|| unsafe {
        crate::os_thread_rt::current_domain(rt.futex_sched as *const crate::os_thread_rt::Domain)
    });
    let lane = dom.map(|d| d.lane_chain()).unwrap_or_default();
    if let Some(d) = dom {
        d.lane_give_back(&lane);
    }
    // #1820 — the run's domain, whose deadlock count this join takes part in (`None`: the durable
    // nested nursery's, which has none). SAFETY: as `dom` above.
    let hub = (rt.futex_sched != 0)
        .then(|| unsafe { (*(rt.futex_sched as *const crate::os_thread_rt::Domain)).hub() });
    let result = (|| -> i64 {
        // Park on the completion cell until the child's OS thread fills it (S1c async children). A durable
        // child ran synchronously, so its cell is already `Some` and this returns without waiting; an
        // async (op-0/5/8/11/13) child parks here until its thread publishes the outcome, with a bounded
        // re-check so a §5 host interrupt on the parent's `epoch_addr` still unwinds a waiter (the child
        // bakes that same cell, so it unwinds too).
        let mut st = done.state.lock().unwrap_or_else(|e| e.into_inner());
        let mut _pg = None;
        let (result, trap) = loop {
            if let Some(outcome) = *st {
                break outcome;
            }
            // #1820 — count this joiner as blocked, once, under the cell's lock with no result yet:
            // the child's publish settles it before the child leaves `live` (`ChildDone::publish`).
            if let (None, Some(h)) = (&_pg, hub) {
                _pg = Some(h.join_park(&done.joiner_parked));
            }
            // #1904 — a freeze reached this parent while it waits: it drives the child it is joining,
            // as the freeze drives a `thread.join`'s, by ringing the child's doorbell.
            let freezing = rt.durable.load(Ordering::Acquire)
                && crate::fiber_rt::window_is_unwinding(mem_base);
            if let Some(d) = done.durable.as_ref().filter(|_| freezing) {
                d.ring();
            }
            // §5 kill-path: the host set the parent's interrupt cell — stop waiting and **propagate
            // `OutOfFuel` right here** (the child bakes the same cell, so it unwinds too, and is joined at
            // teardown). We must not return a bare `0` and lean on "the parent traps at its next epoch
            // poll": unlike a spinning caller, a parent that does `join` then `return` has **no** back-edge
            // or function-entry between this call and its `return`, so there is no next poll — the `0` would
            // flow straight out as a clean `Returned`, silently dropping the kill (ISSUES.md I33). Setting
            // the trap cell makes the outcome `Trapped(OutOfFuel)` regardless of a subsequent poll, matching
            // the child's own kill and the interpreter's runaway-nesting semantics.
            if epoch_fired(rt.epoch_addr) {
                *trap_out = TrapKind::OutOfFuel as i64;
                return 0;
            }
            // Owner decision 2026-07-24 (domain teardown; DESIGN.md §12, D37 death-is-revocation): a
            // trap/exit from any vCPU of the parent domain — or the root's completion (the internal
            // DOMAIN_DONE sentinel) — ends the domain; a sibling vCPU parked here joining a nested
            // child returns so its trailing trap-propagation guard (which checks this same cell)
            // unwinds it. Observed on the same bounded re-check cadence as the kill-path above; the
            // atomic load matches the cell's cross-thread contract (an `AtomicI64`'s storage).
            if (*(trap_out as *const core::sync::atomic::AtomicI64))
                .load(core::sync::atomic::Ordering::Relaxed)
                != 0
            {
                return 0;
            }
            // #1820 — the deadlock verdict, decided here as a futex wait decides it: with this
            // join's own cell seen empty under its lock, no live vCPU left uncounted means nothing
            // can ever end the child, so fail closed (`ThreadFault`, the oracle's answer) rather
            // than wait forever. A task's join leaves the verdict to an OS vCPU's park, as every
            // task park does; a freeze in flight is not a deadlock.
            if task.is_none() && !freezing && hub.is_some_and(|h| !h.peers_live()) {
                *trap_out = TrapKind::ThreadFault as i64;
                return 0;
            }
            st = match &task {
                // The child's end, or the cadence sweep, re-offers the task (`child_finished`).
                Some(fib) => {
                    drop(st);
                    crate::fiber_rt::fiber_event_park(fib, None, crate::fiber_rt::ParkOn::Child);
                    done.state.lock().unwrap_or_else(|e| e.into_inner())
                }
                None => {
                    done.cv
                        .wait_timeout(st, std::time::Duration::from_millis(20))
                        .unwrap_or_else(|e| e.into_inner())
                        .0
                }
            };
        };
        drop(st);
        // #1904 — a child the freeze unwound is not joined: it stays for the harvest (its window rides
        // the artifact), and the join is re-issued on thaw against the re-launched child.
        if done.durable.as_ref().is_some_and(DurableCell::unwound) {
            rt.children.lock().unwrap_or_else(|e| e.into_inner())[slot].joined = false;
            crate::os_thread_rt::mark_reissue(mem_base);
            return 0;
        }
        match (trap, wait) {
            (0, true) => 0, // returned: it stays for its `join`
            (0, false) => result,
            // A trapped child has nothing left to join: the wait reaps it.
            (trap, true) => {
                rt.children.lock().unwrap_or_else(|e| e.into_inner())[slot].joined = true;
                trap
            }
            (trap, false) => {
                *trap_out = trap; // a child trap propagates to the parent on join
                0
            }
        }
    })();
    // Back from the park: re-take the lane before the parent's guest code runs again.
    if let Some(d) = dom {
        if !d.lane_acquire(&lane, || {
            // SAFETY: the run's live interrupt cell / trap cell.
            epoch_fired(rt.epoch_addr)
                || unsafe {
                    (*(trap_out as *const core::sync::atomic::AtomicI64))
                        .load(core::sync::atomic::Ordering::Relaxed)
                        != 0
                }
        }) {
            *trap_out = TrapKind::ThreadFault as i64;
            return 0;
        }
    }
    result
}

/// PROCESS.md S3 `poll(child) -> 0 running | 1 returned | 2 trapped` (JIT). An **async** child (S1c,
/// ops 0/5/8/11/13) runs on its own OS thread, so `poll` reports the live cell state: `0` while its
/// thread is still executing, then `1` (clean) / `2` (trapped) once it publishes an outcome. A
/// synchronous (durable) child is already done, so it never reads `0`. Non-destructive: the slot + its
/// result stay for a later `join`. A forged / already-joined handle is a `CapFault` (matching this
/// runtime's `join`).
///
/// # Safety
/// As [`join`]: `rt`/`trap_out` are the baked nursery + run trap cell, valid for the call.
pub(crate) unsafe extern "C" fn poll(
    rt: *const Nursery,
    mem_base: u64,
    inst: i32,
    handle: i32,
    trap_out: *mut i64,
) -> i32 {
    let rt = &*rt;
    if rt.resolve(mem_base, inst, trap_out).is_none() {
        return 0; // a forged `Instantiator` — `*trap_out` holds the CapFault (#1729)
    }
    let children = rt.children.lock().unwrap_or_else(|e| e.into_inner());
    match children.get(handle as usize) {
        Some(c) if !c.joined => {
            let st = c.done.state.lock().unwrap_or_else(|e| e.into_inner());
            match *st {
                None => 0,             // still running (the OS-thread child hasn't finished)
                Some((_, 0)) => 1,     // returned cleanly
                Some((_, _trap)) => 2, // trapped
            }
        }
        _ => {
            *trap_out = TrapKind::CapFault as i64;
            0
        }
    }
}

/// PROCESS.md S3 `detach(child) -> 0` (JIT). Drop the parent's join claim; a later `join` is then inert.
/// The child's OS thread (if still running) is joined at run teardown (`join_children`), so a detached
/// async child never outlives the window. A forged / already-joined handle is a `CapFault`.
///
/// # Safety
/// As [`join`].
pub(crate) unsafe extern "C" fn detach(
    rt: *const Nursery,
    mem_base: u64,
    inst: i32,
    handle: i32,
    trap_out: *mut i64,
) -> i32 {
    let rt = &*rt;
    if rt.resolve(mem_base, inst, trap_out).is_none() {
        return 0; // a forged `Instantiator` — `*trap_out` holds the CapFault (#1729)
    }
    let mut children = rt.children.lock().unwrap_or_else(|e| e.into_inner());
    match children.get_mut(handle as usize) {
        Some(c) if !c.joined => {
            c.joined = true;
            0
        }
        _ => {
            *trap_out = TrapKind::CapFault as i64;
            0
        }
    }
}

/// PROCESS.md S3 `kill(child) -> 0` (JIT, #2074): end the child and every child under it with
/// `ThreadFault`, running or parked ([`Nursery::kill_slot`]), as the interpreter's op-12 arm does. A
/// child that has finished is left as it is. A forged / already-joined handle is a `CapFault`.
///
/// # Safety
/// As [`join`].
pub(crate) unsafe extern "C" fn kill(
    rt: *const Nursery,
    mem_base: u64,
    inst: i32,
    handle: i32,
    trap_out: *mut i64,
) -> i32 {
    let rt = &*rt;
    if rt.resolve(mem_base, inst, trap_out).is_none() {
        return 0; // a forged `Instantiator` — `*trap_out` holds the CapFault (#1729)
    }
    if !rt.kill_slot(handle as usize) {
        *trap_out = TrapKind::CapFault as i64;
    }
    0
}

/// Whether the §5 kill-path interrupt cell at `addr` has fired (`0` ⇒ no kill-path armed) — so a
/// `join` parked on a still-running child stops waiting and lets the parent unwind. Mirrors
/// `os_thread_rt::epoch_fired`; a wrong read only affects a wakeup, never confinement.
fn epoch_fired(addr: usize) -> bool {
    addr != 0
        && unsafe { (*(addr as *const std::sync::atomic::AtomicU64)).load(Ordering::Relaxed) != 0 }
}

/// A durable §14 child's baked `call.cap` thunk (DURABILITY.md §4, "JIT parity"): its powerbox holds
/// exactly one capability — an `Instantiator` over the child's **own full window** `[0, child_size)`,
/// so the child can carve and run a grandchild of its own. `Nursery::resolve` calls this with iface-6
/// op-0 to read the holder's `[base, size]`; the child is confined to its window by the masking
/// lowering and can forge no other cap, so any handle resolves to `[0, child_size]` (full authority
/// over its own window, and nothing beyond). Anything else is an inert `CapFault`, matching the
/// interpreter's single-binding child powerbox (`grant_instantiator(0, child_size)`).
///
/// # Safety
/// `ctx` points at a live `u64` (the child's window size) for the call; `results`/`trap_out` are the
/// call-site slot buffers (`Nursery::resolve` / the `call.cap` lowering guarantee them).
pub(crate) unsafe extern "C" fn child_instantiator_thunk(
    ctx: *mut core::ffi::c_void,
    _mem_base: *mut u8,
    _mem_size: u64,
    _mem_reserved: u64,
    type_id: u32,
    op: u32,
    _handle: i32,
    _args: *const i64,
    _n_args: u64,
    results: *mut i64,
    n_results: u64,
    trap_out: *mut i64,
) {
    if type_id == temen_ir_iface_instantiator() && op == 0 && n_results >= 2 {
        let child_size = *(ctx as *const u64);
        *results = 0; // base, window-relative — the child's own window starts at 0
        *results.add(1) = child_size as i64; // size
        *trap_out = 0;
    } else {
        *trap_out = TrapKind::CapFault as i64;
    }
}
