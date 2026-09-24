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
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex};
use temen_ir::{Data, Func, FuncIdx, SpawnRec, TypeEntry, ValType};

/// PROCESS.md S1: per-carve compile-cache key for a **non-durable** child — the identity the compiled
/// a compiled child ([`crate::CompiledModule`]) depends on. `funcs_ptr`/`n_funcs` name the module's function slice (stable
/// for the whole run per the [`crate::ModuleResolver`] contract — a held grant's storage outlives the
/// run and distinct live modules have distinct storage, so a stale-pointer collision cannot happen
/// within a run; the worst case of any mismatch is a miss, never wrong code). `entry` picks the
/// trampolined function; `size_log2` picks the baked window mask. The carve **base** is deliberately
/// absent — it is a runtime arg, so one entry serves every offset.
type ChildCodeKey = (usize, usize, u32, u8);

/// #1726 — the program an **installed §22 unit** runs, kept alive by the `CompiledModule` that
/// defined it, so the unit's `Instantiator` call sites can bake its address as `self_prog`. A
/// same-module (**self**) child spawned from unit code runs the *unit's* functions — the spawning
/// frame's module, as on both interpreters — not module 0's. `self_prog == 0` names the nursery's own
/// program (module 0), which is every call site outside a unit.
pub(crate) struct UnitProg {
    pub(crate) funcs: std::sync::Arc<[Func]>,
    pub(crate) types: std::sync::Arc<[TypeEntry]>,
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
}

/// #1361 step 4 — a durable detached child's freeze cell, shared by its join-table entry and its task.
/// The parent holds no pointer into the child's window otherwise (detachment severs *read*); this is
/// the lifecycle linkage a freeze uses, like the interpreter's doorbell.
pub(crate) struct DurableCell {
    /// The child's live window base while its window exists, `0` once its task frees it. The lock
    /// orders a freeze's doorbell store against that free.
    pub(crate) base: Mutex<usize>,
    /// The child's window image, deposited by its task at finish **iff** it unwound for a freeze.
    pub(crate) image: Mutex<Option<Vec<u8>>>,
    /// The child module's shadow arena: its context-0 region is where the child spills.
    pub(crate) shadow: temen_ir::durable_abi::ShadowArena,
    /// What the spawn knew that a thaw needs: the entry and the window geometry.
    pub(crate) entry: u32,
    pub(crate) mapped_log2: u8,
    pub(crate) reserved_log2: u8,
}

impl DurableCell {
    pub(crate) fn new(
        shadow: temen_ir::durable_abi::ShadowArena,
        entry: u32,
        mapped_log2: u8,
        reserved_log2: u8,
    ) -> DurableCell {
        DurableCell {
            base: Mutex::new(0),
            image: Mutex::new(None),
            shadow,
            entry,
            mapped_log2,
            reserved_log2,
        }
    }
}

/// One spawned child's join-table entry: its completion cell plus whether it has been `join`ed (a
/// second join is inert — `CapFault`, matching the interpreter's once-only join).
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
}

impl Child {
    /// A child whose outcome is **already known** (the synchronous path): a cell pre-filled with
    /// `(result, trap)`.
    fn finished(result: i64, trap: i64) -> Child {
        Child {
            done: std::sync::Arc::new(ChildDone {
                state: Mutex::new(Some((result, trap))),
                cv: Condvar::new(),
                durable: None,
            }),
            joined: false,
            retained: 0,
            nested: None,
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
    init: impl FnOnce(&mut [u8]),
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
) -> Filed {
    let futex_sched = rt.futex_sched;
    // #1586 — reserve a §15 live-vCPU slot before filing, so a parent cannot hold more concurrency
    // than its ancestors granted (INVARIANTS #3). SAFETY: a nonzero `futex_sched` is the run's live
    // `Domain`, outliving every task.
    if futex_sched != 0
        && !unsafe { (*(futex_sched as *const crate::os_thread_rt::Domain)).try_child_start() }
    {
        teardown();
        return Filed::AtCeiling;
    }
    let done = std::sync::Arc::new(ChildDone {
        state: Mutex::new(None),
        cv: Condvar::new(),
        durable,
    });
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
        )
    };
    let task = match task {
        Ok(t) => t,
        Err(teardown) => {
            teardown();
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
    rt.child_exec.spawn(task);
    Filed::Slot(slot as i32)
}

/// A raw pointer a task's closures carry to a worker. SAFETY (for every use in this file): the
/// parent window outlives every task (`join_children` runs before it frees) and a carve child
/// touches only its own carve for copy-in / copy-back — disjoint from siblings and the parent's
/// live data, the §14 disjointness the guest owns; a child powerbox is handed over wholesale to
/// its task (`Host: Send`) and never touched by the parent after the spawn.
struct SendRaw<T>(T);
unsafe impl<T> Send for SendRaw<T> {}

/// D66 — file a **carve** child (ops 0/5/8/11/13): its window is a private image of the parent's
/// carve `[parent_mem_base + sub_base, +2^size_log2)`, seeded at filing and written back at finish
/// (the parent is the superset). `chain` is the lane chain the executor gates it on.
#[allow(clippy::too_many_arguments)]
unsafe fn file_carve_task(
    rt: &Nursery,
    code: std::sync::Arc<crate::CompiledModule>,
    sub_base: u64,
    size_log2: u8,
    parent_mem_base: *mut u8,
    args: Vec<i64>,
    n_results: usize,
    chain: Vec<(usize, i64)>,
    retained_ctx: usize,
    teardown: crate::child_exec::Teardown,
) -> Filed {
    let size = 1usize << size_log2;
    let src = SendRaw(parent_mem_base);
    let dst = SendRaw(parent_mem_base);
    file_task(
        rt,
        code,
        size_log2,
        size_log2,
        move |rw| {
            let src = src;
            // SAFETY: the carve is committed parent memory (Instantiator-bounded), size = `size`.
            let carve = unsafe { std::slice::from_raw_parts(src.0.add(sub_base as usize), size) };
            rw[..size].copy_from_slice(carve);
        },
        |_, _, _| true,
        Some(Box::new(move |image: &[u8]| {
            let dst = dst;
            // SAFETY: as the copy-in; the parent window is alive until `join_children` returns.
            let carve =
                unsafe { std::slice::from_raw_parts_mut(dst.0.add(sub_base as usize), size) };
            carve.copy_from_slice(&image[..size]);
        })),
        args,
        n_results,
        chain,
        retained_ctx,
        teardown,
        None,
        None,
    )
}

/// D66 — a granted child's teardown: release its powerbox and, for a detached child, return its
/// lane to the parent through the `lane_give` hook (`-1` for a carve child: a no-op, it holds none).
unsafe fn granted_teardown(
    rt: &Nursery,
    release: crate::GrantChildReleaser,
    gc_ctx: *mut core::ffi::c_void,
    lane: i64,
) -> crate::child_exec::Teardown {
    let (ctx, parent_ctx) = (SendRaw(gc_ctx), SendRaw(rt.grant_ctx()));
    let lane_give = rt.grant_lane_give.load(Ordering::Acquire);
    Box::new(move || {
        let (ctx, parent_ctx) = (ctx, parent_ctx);
        // SAFETY: the powerbox is freed exactly once, here, by the task that owned it.
        unsafe { release(ctx.0) };
        if lane_give != 0 && lane >= 0 {
            // SAFETY: a nonzero address is the embedder's registered `LaneGiver`; `parent_ctx` is
            // the parent host it was registered with.
            let give: crate::LaneGiver = unsafe { core::mem::transmute(lane_give) };
            unsafe { give(parent_ctx.0, lane) };
        }
    })
}

/// D66 — the lane chain a granted child's task is gated on: its parent's lane over its own.
fn lane_chain_of(gc: &crate::GrantChild) -> Vec<(usize, i64)> {
    vec![
        (gc.parent_domain as usize, gc.parent_lane_cap),
        (gc.domain as usize, gc.lane_cap),
    ]
}

/// D66 — file a **granted** carve child (ops 8/11/13): register its serve context on the shared
/// powerbox first (CALLS.md 5c.1b — so a dispatch enqueued at any point of the child's life finds
/// it; the releaser clears it before the module drops), then file it with its powerbox teardown.
#[allow(clippy::too_many_arguments)]
unsafe fn file_granted_carve_task(
    rt: &Nursery,
    code: crate::CompiledModule,
    sub_base: u64,
    size_log2: u8,
    parent_mem_base: *mut u8,
    args: Vec<i64>,
    n_results: usize,
    release: crate::GrantChildReleaser,
    gc: &crate::GrantChild,
    trap_out: *mut i64,
) -> i32 {
    let code = std::sync::Arc::new(code);
    register_serve(rt, gc.ctx, &code);
    let teardown = granted_teardown(rt, release, gc.ctx, -1);
    match file_carve_task(
        rt,
        code,
        sub_base,
        size_log2,
        parent_mem_base,
        args,
        n_results,
        lane_chain_of(gc),
        gc.retained_ctx as usize,
        teardown,
    ) {
        Filed::Slot(slot) => slot,
        // #1586 — `ThreadFault` is what the interpreter raises at the same ceiling.
        Filed::AtCeiling => {
            *trap_out = TrapKind::ThreadFault as i64;
            0
        }
        // #1587 — a refused task is a probeable `-EINVAL`, never a host abort.
        Filed::Refused => EINVAL as i32,
    }
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
    /// Address of the parent run's **counted-fuel** cell (`0` ⇒ the parent isn't fuel-armed, so
    /// children stay un-metered — byte-identical to before). Read (not decremented) at each spawn to
    /// derive the child's budget `min(quota, *parent_fuel_addr)` — the JIT mirror of the interpreter's
    /// `child_fuel` contract (INTERP_PERF.md "Fuel unification" step 5). Same-thread as the spawning
    /// vCPU that owns this cell, so the read needs no synchronization.
    parent_fuel_addr: usize,
    /// Each fuel-armed child's own budget cell, kept alive here until run teardown (after
    /// [`Nursery::join_children`]) because an **async** child's OS thread — or a suspended **coro** —
    /// decrements it after the spawning thunk has returned. `Box<u64>` gives a stable heap address to
    /// bake into the child's code; the cells are never merged back into the parent (no credit-back,
    /// exactly like the interpreter's value-copy `child_fuel`).
    // `Box` is load-bearing: the baked address must survive a `push`, which a `Vec<u64>`'s realloc
    // would move — so the clippy `vec_box` "simplification" would dangle the address in a child's code.
    #[allow(clippy::vec_box)]
    child_fuel_cells: Mutex<Vec<Box<u64>>>,
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
    /// This nursery's own §14 domain **task id** — `0` for the root, and a subtree-unique id (from
    /// [`Nursery::task_counter`]) for each nested child's nursery. `instantiate` stamps it as the
    /// recorded child's `parent_task` (DURABILITY.md §4 depth-2), so a thaw can group residue by parent
    /// (the root's direct child carries `0`; a grandchild carries its parent-child's id).
    my_task: usize,
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
    /// PROCESS.md S1: **per-carve compile cache** for non-durable children. Keyed by
    /// [`ChildCodeKey`], each entry is a compiled [`crate::CompiledModule`] reused across spawns — so a
    /// shell respawning the same applet (any offset, same size) recompiles nothing. Held behind the
    /// nursery, alive for the run; the durable / nesting child bypasses it (its baked per-child
    /// nursery makes its code un-shareable). **`Arc`** (S1c): a cached child can be handed to an
    /// OS-thread child executor and run concurrently on several threads — sound because a compiled child is
    /// `Send + Sync` (its code arena + `fn_table` are immutable read-execute memory after
    /// `finalize_definitions`; the `unsafe impl` + compile-time assertion live in `lib.rs`). A lookup
    /// still drops the lock before the run. (Children run synchronously on the calling thread **today**;
    /// this makes the artifact ready for the async spawn slice that follows.)
    child_code: Mutex<HashMap<ChildCodeKey, std::sync::Arc<crate::CompiledModule>>>,
    /// PROCESS.md S2 (JIT parity): the host callbacks for `instantiate_granted` (op 8) — build a
    /// granted child's powerbox `Host` and free it after the run — stored as raw fn-pointer addresses
    /// (`0` ⇒ none, an inert `CapFault`, like a run that re-grants nothing). Set once at run entry via
    /// [`Nursery::set_grant_hooks`] (the same interior-mutability contract as [`Nursery::set_durable`]:
    /// written before the guest runs, then only read by the `instantiate_granted` thunk), so no new
    /// param threads through the compile pipeline.
    grant_build: std::sync::atomic::AtomicUsize,
    grant_build_named: std::sync::atomic::AtomicUsize,
    /// PROCESS.md §5 / #1287 — the detached-child powerbox builder + the `Budget` taker
    /// ([`crate::GrantChildHooks::build_detached`] / [`crate::BudgetMemTaker`]; 0 = none ⇒ op 15 is an
    /// inert `CapFault`, like the other grant ops without hooks).
    grant_build_detached: std::sync::atomic::AtomicUsize,
    grant_budget_mem_take: std::sync::atomic::AtomicUsize,
    /// #1587 — [`crate::BudgetMemGiver`]: returns a taken quota when the OS-thread spawn fails.
    grant_budget_mem_give: std::sync::atomic::AtomicUsize,
    /// D66 — [`crate::LaneGiver`]: a reaped child task returns its lane to the parent.
    grant_lane_give: std::sync::atomic::AtomicUsize,
    /// D66 — the parent domain's `(id, lane cap)` ([`crate::GrantChildHooks::parent_domain`] /
    /// `parent_lane_cap`): what a plain carve child's task is gated on.
    grant_parent_domain: std::sync::atomic::AtomicU64,
    grant_parent_lane_cap: std::sync::atomic::AtomicI64,
    /// Op-15 pre-mapped region hooks ([`crate::PremapAdmit`] / [`crate::PremapStage`] /
    /// [`crate::PremapApply`]; 0 = none ⇒ a spawn asking for one is an inert `CapFault`).
    grant_premap_admit: std::sync::atomic::AtomicUsize,
    grant_premap_stage: std::sync::atomic::AtomicUsize,
    grant_premap_apply: std::sync::atomic::AtomicUsize,
    /// §3c.2 — the installed [`crate::BudgetTaker`] (0 = none: budget records stay `-EINVAL`).
    grant_budget_take: std::sync::atomic::AtomicUsize,
    grant_release: std::sync::atomic::AtomicUsize,
    grant_bind_imports: std::sync::atomic::AtomicUsize,
    /// CALLS.md 5c.0 — the `child_offer` mint hook ([`crate::ChildOfferMint`] as usize; 0 = none).
    grant_mint: std::sync::atomic::AtomicUsize,
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
    /// #964: the run's NULL guard (`0` = unguarded), set once at run entry via
    /// [`Nursery::set_null_guard`] (same interior-mutability contract as `set_durable`). A carve
    /// overlapping `[0, guard)` of the window is refused `-EINVAL` — the host seeds/copies a
    /// child's carve outside the guarded call, so an mprotect-guarded carve would fault the host.
    null_guard: std::sync::atomic::AtomicU64,
    /// The parent module's declared shadow arena — a same-module (op-0 self) child places its
    /// contexts in *its own* window at the same declared offsets.
    shadow: temen_ir::durable_abi::ShadowArena,
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
        parent_fuel_addr: usize,
        futex_sched: usize,
        my_task: usize,
        task_counter: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        frozen_nested_sink: std::sync::Arc<Mutex<Vec<crate::FrozenNested>>>,
        serve_handlers: Box<[u32]>,
        shadow: temen_ir::durable_abi::ShadowArena,
    ) -> Nursery {
        Nursery {
            funcs,
            types,
            serve_handlers,
            shadow,
            cap_thunk,
            cap_ctx,
            resolve_module,
            epoch_addr,
            parent_fuel_addr,
            child_fuel_cells: Mutex::new(Vec::new()),
            futex_sched,
            children: Mutex::new(Vec::new()),
            child_exec: {
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
            },
            grant_parent_domain: std::sync::atomic::AtomicU64::new(0),
            grant_parent_lane_cap: std::sync::atomic::AtomicI64::new(-1),
            durable: AtomicBool::new(false),
            my_task,
            task_counter,
            frozen_nested_sink,
            child_code: Mutex::new(HashMap::new()),
            grant_build: std::sync::atomic::AtomicUsize::new(0),
            grant_build_named: std::sync::atomic::AtomicUsize::new(0),
            grant_build_detached: std::sync::atomic::AtomicUsize::new(0),
            grant_budget_mem_take: std::sync::atomic::AtomicUsize::new(0),
            grant_budget_mem_give: std::sync::atomic::AtomicUsize::new(0),
            grant_lane_give: std::sync::atomic::AtomicUsize::new(0),
            grant_premap_admit: std::sync::atomic::AtomicUsize::new(0),
            grant_premap_stage: std::sync::atomic::AtomicUsize::new(0),
            grant_premap_apply: std::sync::atomic::AtomicUsize::new(0),
            grant_budget_take: std::sync::atomic::AtomicUsize::new(0),
            grant_release: std::sync::atomic::AtomicUsize::new(0),
            grant_bind_imports: std::sync::atomic::AtomicUsize::new(0),
            grant_parent_ctx: std::sync::atomic::AtomicUsize::new(0),
            grant_register_serve: std::sync::atomic::AtomicUsize::new(0),
            grant_mint: std::sync::atomic::AtomicUsize::new(0),
            grant_thunk: std::sync::atomic::AtomicUsize::new(0),
            null_guard: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// #964: install the run's NULL guard (see the `null_guard` field). Called once at run entry,
    /// before any guest code can `instantiate`; child nurseries keep the `0` default (a child's
    /// own window is unguarded).
    pub(crate) fn set_null_guard(&self, guard: u64) {
        self.null_guard.store(guard, Ordering::Release);
    }

    /// PROCESS.md S2 (JIT parity): install the granted-child host callbacks — build a granted child's
    /// powerbox (positional op 8 / by-name op 11) and release it after the run. Called once at run entry
    /// (like [`Self::set_durable`]), before any `instantiate_granted`/`instantiate_named` site can fire.
    /// `None` leaves them `0` (both ops stay an inert `CapFault`).
    /// §3c.2 — install/clear the Budget taker (see [`crate::CompiledModule::set_budget_taker`]).
    pub(crate) fn set_budget_take(&self, addr: usize) {
        self.grant_budget_take.store(addr, Ordering::Release);
    }

    pub(crate) fn set_grant_hooks(&self, hooks: Option<crate::GrantChildHooks>) {
        let (b, bn, r, bi, m, t, rs, bd, mt, mg, lg, pc, pa, ps, pp) = match hooks {
            Some(h) => (
                h.build as usize,
                h.build_named as usize,
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
            None => (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        };
        self.grant_build.store(b, Ordering::Release);
        self.grant_build_named.store(bn, Ordering::Release);
        self.grant_build_detached.store(bd, Ordering::Release);
        self.grant_budget_mem_take.store(mt, Ordering::Release);
        self.grant_budget_mem_give.store(mg, Ordering::Release);
        self.grant_lane_give.store(lg, Ordering::Release);
        let (pd, pl) = hooks.map_or((0, -1), |h| (h.parent_domain, h.parent_lane_cap));
        self.grant_parent_domain.store(pd, Ordering::Release);
        self.grant_parent_lane_cap.store(pl, Ordering::Release);
        self.grant_premap_admit.store(pa, Ordering::Release);
        self.grant_premap_stage.store(ps, Ordering::Release);
        self.grant_premap_apply.store(pp, Ordering::Release);
        self.grant_register_serve.store(rs, Ordering::Release);
        self.grant_release.store(r, Ordering::Release);
        self.grant_bind_imports.store(bi, Ordering::Release);
        self.grant_mint.store(m, Ordering::Release);
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

    /// D66 — the lane chain a **plain** carve child (op 0/5) is gated on: its parent's lane alone.
    fn parent_lane_chain(&self) -> Vec<(usize, i64)> {
        vec![(
            self.grant_parent_domain.load(Ordering::Acquire) as usize,
            self.grant_parent_lane_cap.load(Ordering::Acquire),
        )]
    }

    /// Derive and allocate a child's counted-fuel cell, exactly as the interpreter derives `child_fuel`
    /// (INTERP_PERF.md "Fuel unification" step 5; the tree-walker at `lib.rs` "Quota: the child's fuel,
    /// sub-allocated from (and capped by) ours"): the child's budget is `min(quota, parent_remaining)`,
    /// or the parent's *entire* remaining fuel when `quota <= 0` (the "unspecified" sentinel). The
    /// operand the lowering passes to the instantiate thunks as `fuel` is that same `quota`. Returns the
    /// cell's stable address to bake into the child's code, or `0` when the parent isn't fuel-armed (the
    /// child stays un-metered, byte-identical to before this slice). The cell lives in the nursery until
    /// teardown so an async/coro child can decrement it after the spawning thunk has returned; it is
    /// never merged back into the parent (no credit-back — the interpreter's `child_fuel` is a value copy).
    ///
    /// # Safety
    /// Runs on the spawning vCPU's own thread, the sole writer of `*parent_fuel_addr`, so the read is
    /// race-free; `parent_fuel_addr` (when nonzero) is the live host-owned parent fuel cell.
    unsafe fn arm_child_fuel(&self, quota: i64) -> usize {
        if self.parent_fuel_addr == 0 {
            return 0; // parent un-metered ⇒ child un-metered
        }
        let parent_remaining = *(self.parent_fuel_addr as *const u64);
        let child_fuel = if quota <= 0 {
            parent_remaining
        } else {
            (quota as u64).min(parent_remaining)
        };
        let cell = Box::new(child_fuel);
        let addr = &*cell as *const u64 as usize;
        self.child_fuel_cells
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(cell);
        addr
    }

    /// This nursery's §14 domain task id (`0` = root) — `instantiate` records it as a child's
    /// `parent_task` (depth-2 grouping).
    pub(crate) fn my_task(&self) -> usize {
        self.my_task
    }

    /// Reserve the next subtree-unique task id for a nested child's nursery (shared counter).
    pub(crate) fn next_child_task(&self) -> usize {
        self.task_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
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
    /// parent's re-executed `join` resolves without re-running the child. The freeze recorded slots in
    /// ascending order; pad any gap with an inert placeholder to keep the index alignment.
    pub(crate) fn seed_child_result(&self, slot: usize, result: i64, trap: i64) {
        let mut children = self.children.lock().unwrap_or_else(|e| e.into_inner());
        while children.len() <= slot {
            children.push(Child::finished(0, 0));
        }
        children[slot] = Child::finished(result, trap);
    }

    /// S1c — join every async child OS thread. Called at run **teardown**, before the parent window is
    /// freed, so no child thread outlives the memory it copies to/from. A well-behaved child has already
    /// finished (a `join`/`detach` waited on it, or it ran to completion and the run is ending); a still-
    /// running **detached** child blocks here exactly as a detached `thread.spawn` vCPU does at
    /// `Domain::join_all` — the run's contract is that every vCPU/child is joined before the window dies.
    pub(crate) fn join_children(&self, froze: bool) {
        // D66 — drive the executor to quiescence: parked tasks are poisoned so they unwind,
        // runnable ones finish, then the workers are joined.
        self.child_exec.shutdown_and_join();
        // CALLS.md 5c.0 — release each child's nursery-retained shared-powerbox ref (minted
        // live-impls hold their own counted refs, so a parent-held offer handle stays valid at the
        // host layer; the run is over regardless). After the joins above, so no child thread still
        // runs against the `Host` while its last-but-one ref drops. Exactly once per child: `take`
        // zeroes the field.
        let release_addr = self.grant_release.load(Ordering::Acquire);
        if release_addr != 0 {
            let release: crate::GrantChildReleaser = unsafe { core::mem::transmute(release_addr) };
            let mut children = self.children.lock().unwrap_or_else(|e| e.into_inner());
            for c in children.iter_mut() {
                // #1361 step 4 — a frozen run keeps each unjoined durable detached child's powerbox:
                // it rides the artifact ([`Nursery::take_detached_harvest`] hands it over).
                if froze && !c.joined && c.done.durable.is_some() {
                    continue;
                }
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
    /// poll like any root and its task deposits its window image at finish.
    pub(crate) fn ring_detached(&self) {
        let children = self.children.lock().unwrap_or_else(|e| e.into_inner());
        for c in children.iter().filter(|c| !c.joined) {
            let Some(d) = c.done.durable.as_ref() else {
                continue;
            };
            let base = d.base.lock().unwrap_or_else(|e| e.into_inner());
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
        }
    }

    /// #1361 step 4 — after a frozen run's [`Nursery::join_children`]: every unjoined durable detached
    /// child, with its powerbox handed over (see [`crate::DetachedHarvest`]).
    pub(crate) fn take_detached_harvest(&self) -> Vec<crate::DetachedHarvest> {
        let mut children = self.children.lock().unwrap_or_else(|e| e.into_inner());
        let mut out = Vec::new();
        for (slot, c) in children.iter_mut().enumerate() {
            if c.joined {
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
                image: d.image.lock().unwrap_or_else(|e| e.into_inner()).take(),
                outcome: *c.done.state.lock().unwrap_or_else(|e| e.into_inner()),
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
    /// # Safety
    /// `seed.child` holds two live counted refs to the child's powerbox, owned from here on.
    pub(crate) unsafe fn relaunch_detached(&self, seed: crate::DetachedSeed) -> bool {
        let crate::DetachedSeed {
            slot,
            entry,
            mapped_log2,
            reserved_log2,
            funcs,
            types,
            shadow,
            image,
            child: gc,
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
        let Ok(code) = crate::compile_child_windowed(
            &funcs,
            &types,
            entry as FuncIdx,
            mapped_log2,
            reserved_log2,
            child_thunk,
            gc.ctx,
            self.epoch_addr,
            0, // a thawed child runs un-metered, as every durable JIT re-attach does
            self.futex_sched,
            crate::InstEnv::null(),
            &self.serve_handlers,
            gc.jit_table_log2,
            shadow,
        ) else {
            release(gc.ctx);
            release(gc.retained_ctx);
            return false;
        };
        let n_args = funcs.get(entry as usize).map_or(1, |f| f.params.len());
        let n_results = funcs.get(entry as usize).map_or(1, |f| f.results.len());
        let code = std::sync::Arc::new(code);
        register_serve(self, gc.ctx, &code);
        let teardown = granted_teardown(self, release, gc.ctx, gc.lane_cap);
        let thaw_off = shadow.thaw_state_off(0) as usize;
        let filed = file_task(
            self,
            code,
            mapped_log2,
            reserved_log2,
            move |rw| {
                let n = image.len().min(rw.len());
                rw[..n].copy_from_slice(&image[..n]);
                let s = temen_ir::durable_abi::STATE_OFF as usize;
                if let Some(st) = rw.get_mut(s..s + 4) {
                    st.copy_from_slice(&temen_ir::durable_abi::STATE_NORMAL.to_le_bytes());
                }
                if let Some(th) = rw.get_mut(thaw_off..thaw_off + 4) {
                    th.copy_from_slice(&temen_ir::durable_abi::STATE_REWINDING.to_le_bytes());
                }
            },
            |_, _, _| true,
            None,
            vec![0; n_args], // inert under a rewind: the prologue reloads spilled values
            n_results,
            lane_chain_of(&gc),
            gc.retained_ctx as usize,
            teardown,
            Some(slot),
            Some(DurableCell::new(shadow, entry, mapped_log2, reserved_log2)),
        );
        matches!(filed, Filed::Slot(_))
    }

    /// Mark the run durable (DURABILITY.md §4) — see the [`Nursery::durable`] field: the nesting
    /// thunks then fail closed. Called at run entry (`run_code_raw`), after the entry wrappers
    /// have applied the compile-side durable flag.
    pub(crate) fn set_durable(&self, durable: bool) {
        self.durable.store(durable, Ordering::Release);
    }

    /// The program a **self** child runs (#1726): the spawning code's own — `self_prog`, an installed
    /// unit's [`UnitProg`], or `0` for this nursery's module-0 program.
    ///
    /// # Safety
    /// A nonzero `self_prog` is the address of a [`UnitProg`] the defining `CompiledModule` keeps
    /// alive for the whole run (it is baked into that unit's code by `define_extra`).
    unsafe fn self_program(&self, self_prog: i64) -> (&[Func], &[TypeEntry]) {
        if self_prog == 0 {
            (&self.funcs, &self.types)
        } else {
            let u = &*(self_prog as *const UnitProg);
            (&u.funcs, &u.types)
        }
    }

    /// Resolve a spawn's child source (§14): `module < 0` ⇒ a **self** child (the parent's own
    /// functions, no data segments, no declared-memory constraint); otherwise a host-granted
    /// **`Module` handle** resolved via [`Nursery::resolve_module`] — the child runs *that* verified
    /// module's code, its data segments materialize into the carve, and the carve must equal its
    /// declared memory. `None` (with `*trap_out` set to a `CapFault`) for a forged handle or a run
    /// with no resolver.
    ///
    /// # Safety
    /// `trap_out` is the live trap cell. The returned slices borrow host-owned storage valid for the
    /// run (the [`ModuleResolver`](crate::ModuleResolver) contract).
    #[allow(clippy::type_complexity)]
    unsafe fn resolve_child(
        &self,
        module: i64,
        self_prog: i64,
        trap_out: *mut i64,
    ) -> Option<(
        &[Func],
        &[TypeEntry],
        Option<i32>,
        &[Data],
        temen_ir::durable_abi::ShadowArena,
    )> {
        if module < 0 {
            let (funcs, types) = self.self_program(self_prog);
            return Some((funcs, types, None, &[], self.shadow));
        }
        let Some(resolver) = self.resolve_module else {
            *trap_out = TrapKind::CapFault as i64;
            return None;
        };
        let mut rm = core::mem::MaybeUninit::<crate::ResolvedModule>::zeroed().assume_init();
        if resolver(self.cap_ctx, module as i32, &mut rm) == 0 || rm.n_funcs == 0 {
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

    /// #1501 — whether a `Module` grant is attested **freezable** (instrumented): the other half of what
    /// a durable domain may spawn. A self child runs this domain's own (durable) program.
    unsafe fn child_module_durable(&self, module: i64) -> bool {
        if module < 0 {
            return true;
        }
        let Some(resolver) = self.resolve_module else {
            return false;
        };
        let mut rm = core::mem::MaybeUninit::<crate::ResolvedModule>::zeroed().assume_init();
        resolver(self.cap_ctx, module as i32, &mut rm) != 0 && rm.durable
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

/// Materialize a §14 separate-module child's **data segments** into its carve `[abs_base, …+size)`
/// of the live parent window — exactly as if the child wrote them (the parent sees them, the §14
/// superset; the verifier bounded each segment to the child's declared window == the carve, with a
/// defensive re-check here). RO protection of `readonly` segments is skipped for nested children
/// (intra-domain self-corruption is a §1 non-goal).
///
/// # Safety
/// `[mem_base+abs_base, …+child_size)` is committed parent-window memory (the Instantiator bounded
/// the carve to the holder's range), valid for the call.
unsafe fn write_data_segments(data: &[Data], mem_base: u64, abs_base: u64, child_size: u64) {
    for d in data {
        if d.offset.saturating_add(d.bytes.len() as u64) <= child_size {
            core::ptr::copy_nonoverlapping(
                d.bytes.as_ptr(),
                (mem_base as *mut u8).add((abs_base + d.offset) as usize),
                d.bytes.len(),
            );
        }
    }
}

/// `instantiate(handle, [module,] entry, off, size_log2, fuel) -> child_handle` — the §14 nesting op
/// (`module < 0` ⇒ a self child, op 0; a `Module` handle ⇒ a **separate-module child**, op 5 — the
/// "plugin"). Resolves the holder's carve range, validates the requested power-of-two sub-window fits
/// within it (`-EINVAL` otherwise; a module child's carve must **equal its declared memory** — §14
/// transparency), materializes a module child's data segments into the carve, then **re-compiles**
/// the child entry confined to its own window and runs it (seeded from / copied back to the carve),
/// stashing its outcome for `join`. Returns a child handle (a table index), or `-EINVAL`. A child
/// that cannot be compiled (it uses §12 fibers/threads) or a forged module handle is a `CapFault`.
///
/// # Safety
/// Called from JIT'd code with `rt` the baked [`Nursery`], `mem_base` the live parent window base, and
/// `trap_out` the run's trap cell. All must be valid for the call (the JIT lowering guarantees it).
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe extern "C" fn instantiate(
    rt: *const Nursery,
    self_prog: i64,
    mem_base: u64,
    handle: i32,
    module: i64,
    entry: i64,
    off: i64,
    size_log2: i64,
    fuel: i64,
    trap_out: *mut i64,
) -> i32 {
    let rt_ptr = rt;
    let rt = &*rt;
    let durable = rt.durable.load(Ordering::Acquire);
    // §14 op-5 — a **separate-module** child (`module >= 0`) gets a full attenuated powerbox: an
    // Instantiator + AddressSpace + its bound import manifest, so it can `vm_map` (malloc heap growth)
    // and nest — matching the interpreter, which funnels op 0/5/11/13 through one powerbox path. An
    // op-0 *self* child (`module < 0`, below) instead runs the parent's own funcs compute-only in its
    // carve (no powerbox — the durable-nesting design relies on a self child `call.cap` fail-closing).
    // Route op-5 through the op-13 builder with an **empty** grant list — the exact path the
    // `child_entry_malloc_binds_vm_map_on_the_jit` gate already proves (it spawns with `grants_n = 0`).
    // `mem_size = 0` is safe: with no grants the builder reads no grant records from the window. A
    // durable run is rejected inside `instantiate_module_named` just as the `mod_mem.is_some()` guard
    // below rejects it here.
    //
    // Only when the granted-spawn hooks are installed (`grant_build_named != 0`): a run without them
    // (the plain `jit_separate_module` differential — a *compute* separate-module child that makes no
    // `call.cap`) keeps the empty-powerbox path below, matching the interpreter for a capless child
    // (both tiers return the same value). Delegating unconditionally would `CapFault` such a run at the
    // absent builder.
    if module >= 0 && rt.grant_build_named.load(Ordering::Acquire) != 0 {
        return instantiate_module_named(
            rt_ptr, self_prog, mem_base, /*mem_size (unused: 0 grants)*/ 0, handle, module,
            /*grants_ptr*/ 0, /*grants_n*/ 0, entry, off, size_log2, fuel, trap_out,
        );
    }
    let Some((base, size)) = rt.resolve(mem_base, handle, trap_out) else {
        return 0; // `*trap_out` already holds the CapFault
    };
    let Some((child_funcs, child_types, mod_mem, child_data, child_shadow)) =
        rt.resolve_child(module, self_prog, trap_out)
    else {
        return 0; // forged Module handle / no resolver — CapFault set
    };
    // Only an op-0 **self** child (`module < 0`, so `mod_mem = None`) reaches here — a separate-module
    // op-5 child was delegated to `instantiate_module_named` above. A self child runs the parent's own
    // funcs compute-only in its carve (empty powerbox: a `call.cap` fail-closes with `CapFault`).
    // §4 (DURABILITY.md, "JIT parity" slice 1): a durable run may nest such a same-module child — a
    // pure-compute (non-may-suspend) func with no poll sites runs atomically to completion, no durable
    // control-word setup needed; freezing a *live* nested child on the JIT (ctx-0 control words + shadow
    // base seeded to match the interpreter) is a later slice. The `mod_mem.is_some()` guard is a dead
    // defensive backstop now (a self child is always `None`), kept for safety.
    if durable && mod_mem.is_some() {
        return EINVAL as i32;
    }

    // The carve must be a power-of-two-aligned sub-window within `[0, size)` — a child can only get what
    // the holder sub-allocates (§14/D19). `mod_ok` is trivially true here (`mod_mem = None` for a self
    // child); the op-5 carve-vs-declared-window relaxation (`declared <= carve`, FORK.md §8.6 / #773)
    // lives in `instantiate_module_named`, which op-5 delegates to. Bad entry index / size / alignment
    // ⇒ `-EINVAL`.
    let entry = entry as u64;
    let child_size = if (0..64).contains(&size_log2) {
        1u64 << size_log2
    } else {
        0
    };
    let off = off as u64;
    let mod_ok = mod_mem.is_none_or(|ml| ml <= size_log2 as i32);
    let fits = child_size != 0
        && child_size <= size
        && off & (child_size - 1) == 0
        && off.checked_add(child_size).is_some_and(|e| e <= size)
        // #964: a carve may not dip into the reserved NULL region `[0, guard)` (interp twin).
        && base + off >= rt.null_guard.load(Ordering::Acquire)
        && (entry as usize) < child_funcs.len();
    if !fits || !mod_ok {
        return EINVAL as i32;
    }

    // A module child's data segments materialize into the carve now — `compile_child_and_run` seeds
    // the child's window from the carve, so they arrive exactly like the interpreter's shared-backing
    // writes at spawn.
    write_data_segments(child_data, mem_base, base + off, child_size);

    // The child entry takes its starter caps as `i64` args; with an empty powerbox today they are
    // unused, so pass zeros of the right arity (the entry is a fixed `(i64[, i64]) -> i64`).
    let nargs = child_funcs[entry as usize].params.len();
    let args = vec![0i64; nargs];

    // Fuel unification (step 5): derive this child's own budget cell from the `fuel` operand (the
    // interpreter's `quota`) clamped to the parent's remaining fuel — `0` when the parent isn't armed,
    // leaving the child un-metered as before. Owned by the nursery until teardown (an async child
    // decrements it on its own thread). SAFETY: on the spawning vCPU's thread, sole writer of the cell.
    let child_fuel_addr = rt.arm_child_fuel(fuel);

    // Durable children stay **synchronous** (their baked per-child nursery + freeze residue can't ride
    // the cached OS-thread path yet): re-compile + run inline, record the outcome (and any freeze
    // unwind), and return the join slot.
    if durable {
        // §4 depth-2: reserve this child's subtree-unique domain task id (shared counter, instantiate
        // order), stamped as its nursery's `my_task` so a grandchild it records carries a non-zero
        // `parent_task`. The child inherits the **shared** residue sink + counter, so its descendants'
        // freeze residue coalesces at the root.
        let child_task = rt.next_child_task();
        let (result, trap, unwound) = match crate::compile_child_and_run(
            child_funcs,
            child_types,
            entry as FuncIdx,
            base + off,
            size_log2 as u8,
            mem_base as *mut u8,
            &args,
            rt.epoch_addr, // §5: the child polls the parent's kill-path cell, so one interrupt kills both
            child_fuel_addr, // §5 fuel: the child decrements its own clamped budget cell
            durable, // §4: seed the child's carve control words + give it an Instantiator powerbox
            child_shadow,
            false, // not a thaw re-attach — a live `instantiate` (seed fresh / inherit the parent phase)
            child_task,
            rt.nested_sink(),
            rt.task_counter(),
            &[], // a live `instantiate` re-attaches no frozen residue (that is the thaw path)
        ) {
            Ok(outcome) => outcome,
            Err(_) => {
                // A child we cannot compile (fibers/threads, or a backend error) is a CapFault, not a
                // silent success — the guest learns its nesting request was refused.
                *trap_out = TrapKind::CapFault as i64;
                return 0;
            }
        };
        let mut children = rt.children.lock().unwrap_or_else(|e| e.into_inner());
        let slot = children.len();
        let mut child = Child::finished(result, trap);
        // §4 freeze export: what a freeze records for this child if it is still unjoined when this
        // nursery's owner unwinds ([`Nursery::freeze_unjoined`]).
        child.nested = Some((
            crate::FrozenNested {
                parent_task: rt.my_task(),
                task: child_task,
                slot,
                carve_off: base + off,
                size_log2: size_log2 as u8,
                entry: entry as u32,
                completed_result: None,
            },
            unwound,
        ));
        children.push(child);
        return slot as i32;
    }

    // PROCESS.md S1c — the common (non-durable) path is **asynchronous**: compile once per
    // `(module, entry, size)` (cached, position-independent), then run the child on its **own OS thread
    // in its own guarded window** and return immediately. `join`/`poll` resolve through the child's
    // completion cell; the thread is joined at run teardown (`join_children`). This is what lets two
    // children run concurrently — a pipeline — where the synchronous path serialized them.
    let key: ChildCodeKey = (
        child_funcs.as_ptr() as usize,
        child_funcs.len(),
        entry as u32,
        size_log2 as u8,
    );
    // A fuel-armed child bakes its **per-spawn** budget-cell address into its code, so that code can't
    // be shared across spawns — compile fresh and bypass the cache. (Fuel arming is opt-in, so the
    // common un-armed production path keeps the cache byte-identically.)
    let compile_fresh = || {
        crate::compile_nondurable_child(
            child_funcs,
            child_types,
            entry as FuncIdx,
            size_log2 as u8,
            rt.epoch_addr, // §5: the child polls the parent's kill-path cell (one interrupt kills both)
            child_fuel_addr, // §5 fuel: 0 ⇒ un-metered (cacheable); nonzero ⇒ per-spawn (not cached)
            rt.futex_sched,  // wait/notify against the parent domain's shared futex,,
            child_shadow,
        )
    };
    let code = if child_fuel_addr != 0 {
        match compile_fresh() {
            Ok(cc) => std::sync::Arc::new(cc),
            Err(_) => {
                *trap_out = TrapKind::CapFault as i64;
                return 0;
            }
        }
    } else {
        let mut cache = rt.child_code.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(c) = cache.get(&key) {
            std::sync::Arc::clone(c)
        } else {
            match compile_fresh() {
                Ok(cc) => {
                    let a = std::sync::Arc::new(cc);
                    cache.insert(key, std::sync::Arc::clone(&a));
                    a
                }
                Err(_) => {
                    // Un-compilable child (fibers/threads/setjmp, or a backend error) → CapFault.
                    *trap_out = TrapKind::CapFault as i64;
                    return 0;
                }
            }
        }
    };
    let n_results = child_funcs[entry as usize].results.len();
    match file_carve_task(
        rt,
        code,
        base + off,
        size_log2 as u8,
        mem_base as *mut u8,
        args,
        n_results,
        rt.parent_lane_chain(),
        0,
        Box::new(|| ()), // an empty powerbox: nothing to release
    ) {
        Filed::Slot(slot) => slot,
        // #1586 — at the §15 live-vCPU ceiling (`ThreadFault` is what the interpreter raises when
        // its scheduler refuses a §14 spawn for the same reason), or — #1587 — a task the platform
        // refused: op 0 answers both as a value, never a host abort.
        Filed::AtCeiling | Filed::Refused => {
            *trap_out = TrapKind::ThreadFault as i64;
            0
        }
    }
}

/// PROCESS.md S2 (JIT parity) — `instantiate_named(grants_ptr, grants_n, entry, off, size_log2, quota)`
/// (Instantiator op 11): the multi-cap, by-name form of [`instantiate_granted`]. The child powerbox is
/// built host-side by [`Nursery::grant_build_named`], which reads `grants_n` 16-byte grant records from
/// the **parent** window (`[mem_base, mem_base+mem_size)`) and re-grants each copyable handle under its
/// name; the child finds them by `self.resolve` (lowered to the run's `call.cap` thunk with the
/// child host as ctx, so name resolution "just works"). The child entry is the 1- or 2-arg form
/// (`Instantiator` [, `AddressSpace`]) — no positional grant arg. Same non-durable / uncached /
/// non-nesting shape as [`instantiate_granted`]; a bad record/name is a `MemoryFault`, a non-copyable
/// grant a `CapFault`, a bad carve/entry `-EINVAL` — all matching the interpreter's op-11 path.
///
/// # Safety
/// As [`instantiate`]: `rt`/`mem_base`/`trap_out` are the baked nursery, live parent window base, and
/// run trap cell, valid for the call; `mem_size` is the parent window's **reserved** span (#826 — a
/// grant record in a `map`-grown tail page is admitted; an uncommitted page faults through the
/// SIGSEGV guard as `MemoryFault`, exactly like a guest access).
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe extern "C" fn instantiate_named(
    rt: *const Nursery,
    self_prog: i64,
    mem_base: u64,
    mem_size: u64,
    handle: i32,
    grants_ptr: i64,
    grants_n: i64,
    entry: i64,
    off: i64,
    size_log2: i64,
    fuel: i64,
    trap_out: *mut i64,
) -> i32 {
    let rt = &*rt;
    if rt.durable.load(Ordering::Acquire) {
        return EINVAL as i32;
    }
    let build_addr = rt.grant_build_named.load(Ordering::Acquire);
    let release_addr = rt.grant_release.load(Ordering::Acquire);
    if build_addr == 0 || release_addr == 0 {
        *trap_out = TrapKind::CapFault as i64;
        return 0;
    }
    let build: crate::GrantNamedChildBuilder = core::mem::transmute(build_addr);
    let release: crate::GrantChildReleaser = core::mem::transmute(release_addr);
    // 5c.0 — a builder that shares the child `Host` supplies the lock-taking thunk; granted-child
    // code must synchronize its call.cap calls once the parent can reach the same powerbox. Fall back
    // to the run's thunk only for a legacy non-sharing builder.
    let thunk_addr = rt.grant_thunk.load(Ordering::Acquire);
    let child_thunk: crate::CapThunk = if thunk_addr != 0 {
        core::mem::transmute::<usize, crate::CapThunk>(thunk_addr)
    } else {
        rt.cap_thunk
    };

    let Some((base, size)) = rt.resolve(mem_base, handle, trap_out) else {
        return 0; // `*trap_out` already holds the CapFault
    };
    // #922/#1726: a same-module named child runs (and resolves types against) the spawning code's own
    // program — an installed unit's, or module 0's.
    let (child_funcs, child_types) = rt.self_program(self_prog);
    let entry = entry as u64;
    let child_size = if (0..64).contains(&size_log2) {
        1u64 << size_log2
    } else {
        0
    };
    let off = off as u64;
    // A named child receives no positional grant, so its entry is the 1- or 2-arg form (`Instantiator`
    // [, `AddressSpace`]) returning `i64` — it discovers its granted caps by name.
    let want_as = child_funcs
        .get(entry as usize)
        .is_some_and(|f| f.params.len() >= 2);
    let ok_entry = child_funcs.get(entry as usize).is_some_and(|f| {
        f.results.as_slice() == [ValType::I64]
            && (f.params.len() == 1 || f.params.len() == 2)
            && f.params.iter().all(|p| *p == ValType::I64)
    });
    let fits = child_size != 0
        && child_size <= size
        && off & (child_size - 1) == 0
        && off.checked_add(child_size).is_some_and(|e| e <= size)
        // #964: a carve may not dip into the reserved NULL region `[0, guard)` (interp twin).
        && base + off >= rt.null_guard.load(Ordering::Acquire);
    if !ok_entry || !fits {
        return EINVAL as i32;
    }

    // Build the child powerbox host-side from the grant records; a bad record/name sets `*trap_out`
    // (MemoryFault / CapFault) and fails the whole spawn closed.
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
        child_size,
        &mut gc,
        trap_out,
    ) == 0
    {
        return 0; // `*trap_out` already set by the builder
    }

    // #1234 — bind the child's import manifest against the powerbox just built, exactly as the op-13
    // and op-15 thunks do below. A same-module child's manifest is the *parent's* own running module,
    // which the binder fetches with the record's `-1 = self` module selector — so a guest that nests a
    // confined copy of itself reaches its granted `stdout`/`jit` through `call.import` instead of
    // `CapFault`ing, on this backend exactly as on the two interpreter tiers (§3.3 withhold: a
    // `required` slot with nothing to bind fails the spawn closed, probeable `-EINVAL`).
    //
    // **Only when the spawn handed the child caps by name** — same gate as the interpreter arms. A
    // grant-less child was given nothing to bind, so its slots stay empty and fail closed on use,
    // exactly as before this existed; binding it anyway would refuse spawns that used to work.
    let bind_addr = rt.grant_bind_imports.load(Ordering::Acquire);
    if bind_addr != 0 && grants_n > 0 {
        let bind: crate::ChildManifestBinder = core::mem::transmute(bind_addr);
        if bind(rt.grant_ctx(), gc.ctx, -1) != 0 {
            release(gc.ctx);
            release(gc.retained_ctx);
            return EINVAL as i32;
        }
    }

    let child_fuel_addr = rt.arm_child_fuel(fuel); // §5 fuel: clamp to parent-remaining (0 ⇒ un-metered)
    let compiled = crate::compile_child(
        child_funcs,
        child_types,
        entry as FuncIdx,
        size_log2 as u8,
        child_thunk,
        gc.ctx,
        rt.epoch_addr,
        child_fuel_addr, // §5 fuel: the child decrements its own clamped budget cell
        rt.futex_sched,  // wait/notify against the parent domain's shared futex
        crate::InstEnv::null(),
        &rt.serve_handlers,
        gc.jit_table_log2, // #1296: slots for the units a `Jit`-holding child installs,,
        rt.shadow,
    );
    let code = match compiled {
        Ok(code) => code,
        Err(_) => {
            release(gc.ctx);
            release(gc.retained_ctx);
            *trap_out = TrapKind::CapFault as i64;
            return 0;
        }
    };
    let mut args = vec![gc.inst_handle as i64];
    if want_as {
        args.push(gc.as_handle as i64);
    }
    let n_results = child_funcs[entry as usize].results.len();
    // Async (S1c): the child runs on its own OS thread — two named-grant children can pipeline
    // through a granted `SharedRegion` — and its powerbox host is released from that thread.
    file_granted_carve_task(
        rt,
        code,
        base + off,
        size_log2 as u8,
        mem_base as *mut u8,
        args,
        n_results,
        release,
        &gc,
        trap_out,
    )
}

/// CONSOLIDATION.md §3c — the **config-record spawn** (`Instantiator` op 17,
/// `instantiate_rec(record_ptr)`) on the native JIT tier: parse + validate the 56-byte record
/// from the guest window, then **delegate to the existing spawn thunks** (the same bodies
/// ops 0/5/11/13 call), so there is exactly one spawn implementation per shape on this tier
/// too. Field handling mirrors the tree-walker's op-17 arm:
///
/// - `version != 0` → `CapFault` (fail closed).
/// - `pager != u32::MAX` → `CapFault`. Sound, not a divergence: `temen-run` folds op-17 modules
///   **with** impl exports to the oracle, so a natively-running module has none — and the
///   interpreter fails its pager validation (`self_module.impl_exports.get(..)`) identically.
/// - `budget != 0` → probeable `-EINVAL`: the Budget-funded spawn is an interpreter-first
///   feature (§3b); JIT parity is a follow-up (§3c.2), exactly like durable nesting above —
///   the interpreter is the reference. `budget` and `quota` are mutually exclusive either way.
/// - module `-1` = self / else a granted `Module` handle; named grants `(grants_ptr, grants_n)`.
///
/// # Safety
/// Same contract as [`instantiate_module_named`]: called from JITted code on the spawning
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
    if rp.checked_add(56).is_none_or(|e| e > mem_size) {
        *trap_out = TrapKind::MemoryFault as i64;
        return 0;
    }
    let base = (mem_base + rp) as *const u8;
    // Copy the 56 bytes out of the guest window once, then share the layout decode with the two
    // interpreter tiers (#911). `parse` returns `None` on a nonzero version word — fail closed.
    let mut buf = [0u8; 56];
    core::ptr::copy_nonoverlapping(base, buf.as_mut_ptr(), 56);
    let sr = match SpawnRec::parse(&buf) {
        Some(s) => s,
        None => {
            *trap_out = TrapKind::CapFault as i64;
            return 0;
        }
    };
    let entry = sr.entry as i64;
    let off = sr.off as i64;
    let size_log2 = sr.size_log2;
    let pager = sr.pager;
    let modh = sr.modh;
    let budget = sr.budget;
    let quota = sr.quota;
    let grants_ptr = sr.grants_ptr as i64;
    let grants_n = sr.grants_n as i64;
    if pager != u32::MAX {
        *trap_out = TrapKind::CapFault as i64;
        return 0;
    }
    let mut quota = quota;
    if budget != 0 {
        if quota != 0 {
            *trap_out = TrapKind::CapFault as i64;
            return 0;
        }
        let taker_addr = (*rt).grant_budget_take.load(Ordering::Acquire);
        if taker_addr == 0 {
            // No taker installed (bare harnesses): the §3c gap — probeable, budget untouched.
            return EINVAL as i32;
        }
        let taker: crate::BudgetTaker = core::mem::transmute(taker_addr);
        if !(0..64).contains(&size_log2) {
            return EINVAL as i32; // same refusal the delegate would give, before any drain
        }
        let child_size = 1u64 << size_log2;
        let mut taken = crate::BudgetTaken {
            fuel: -1,
            spawn: -1,
        };
        // Peek first (validate + mem-quota gate, budget intact on refusal)…
        match taker(
            (*rt).grant_ctx(),
            budget,
            child_size,
            0,
            &mut taken,
            trap_out,
        ) {
            1 => {}
            2 => return EINVAL as i32,
            _ => return 0, // CapFault set
        }
        if taken.spawn >= 0 || taken.fuel == 0 {
            // Bounded spawn ceilings and bounded-zero fuel need child-quota threading this
            // tier doesn't have yet — the narrowed §3c.2 gap (probeable, budget intact; the
            // interpreter is the reference).
            return EINVAL as i32;
        }
        // …then drain at commit: past every local guard, the delegate's own guards repeat the
        // peek's (same size/entry/carve math), so a post-drain refusal is unreachable in
        // practice; a same-domain sibling racing `split` between peek and drain gets
        // either-or (its own doing — the armed fuel is the peeked value).
        match taker(
            (*rt).grant_ctx(),
            budget,
            child_size,
            1,
            &mut taken,
            trap_out,
        ) {
            1 => {}
            2 => return EINVAL as i32,
            _ => return 0,
        }
        quota = if taken.fuel < 0 { 0 } else { taken.fuel };
    }
    // §3d: the record folds the plain and named spawn shapes onto one op — and the interpreter
    // retains EVERY child's powerbox (its `child_offer`, op 14, mints over any live child), so
    // when the embedder installed the grant hooks (temen-run always does, at both production
    // sites) an empty grant list still routes through the NAMED path: the retained,
    // interpreter-parity shape. A bare hookless harness keeps the plain path — its children
    // cannot mint offers, the documented hookless gap
    // (`jit_child_offer.rs::child_offer_on_a_hookless_child_refuses` pins it).
    let hooked = (*rt).grant_build_named.load(Ordering::Acquire) != 0
        && (*rt).grant_release.load(Ordering::Acquire) != 0;
    match (modh >= 0, grants_n > 0 || hooked) {
        (false, false) => instantiate(
            rt, self_prog, mem_base, handle, -1, entry, off, size_log2, quota, trap_out,
        ),
        (true, false) => instantiate(
            rt,
            self_prog,
            mem_base,
            handle,
            modh as i64,
            entry,
            off,
            size_log2,
            quota,
            trap_out,
        ),
        (false, true) => instantiate_named(
            rt, self_prog, mem_base, mem_size, handle, grants_ptr, grants_n, entry, off, size_log2,
            quota, trap_out,
        ),
        (true, true) => instantiate_module_named(
            rt,
            self_prog,
            mem_base,
            mem_size,
            handle,
            modh as i64,
            grants_ptr,
            grants_n,
            entry,
            off,
            size_log2,
            quota,
            trap_out,
        ),
    }
}

/// STAGE1.md — `instantiate_module_named(module, grants_ptr, grants_n, entry, off, size_log2, quota)`
/// (Instantiator op 13): the **shell exec** primitive — the union of [`instantiate`]'s separate-module
/// path (op 5: resolve + compile a host-granted `Module`, materialize its data into the carve) and
/// [`instantiate_named`]'s by-name grant list (op 11: re-grant caps into the child's powerbox). It is
/// the only op that runs a foreign program *and* hands it capabilities, so a compiled command (its own
/// module) can resolve an inherited `stdout` by name and do real I/O. The child ctx is per-spawn, so
/// the code is compiled uncached (like op 11). A forged module / non-copyable grant / bad record fails
/// closed exactly as ops 5 and 11 do individually.
///
/// # Safety
/// As [`instantiate`]/[`instantiate_named`]: `rt`/`mem_base`/`mem_size`/`trap_out` are the baked
/// nursery, live parent window base, **reserved** span (#826, as for `instantiate_named`), and run
/// trap cell, valid for the call.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe extern "C" fn instantiate_module_named(
    rt: *const Nursery,
    self_prog: i64,
    mem_base: u64,
    mem_size: u64,
    handle: i32,
    module: i64,
    grants_ptr: i64,
    grants_n: i64,
    entry: i64,
    off: i64,
    size_log2: i64,
    fuel: i64,
    trap_out: *mut i64,
) -> i32 {
    let rt = &*rt;
    // A durable run may not spawn a separate-module child (host-supplied identity + freeze residue are
    // a later slice), matching the `instantiate` op-5 path.
    if rt.durable.load(Ordering::Acquire) {
        return EINVAL as i32;
    }
    let build_addr = rt.grant_build_named.load(Ordering::Acquire);
    let release_addr = rt.grant_release.load(Ordering::Acquire);
    if build_addr == 0 || release_addr == 0 {
        *trap_out = TrapKind::CapFault as i64;
        return 0;
    }
    let build: crate::GrantNamedChildBuilder = core::mem::transmute(build_addr);
    let release: crate::GrantChildReleaser = core::mem::transmute(release_addr);
    // 5c.0 — a builder that shares the child `Host` supplies the lock-taking thunk; granted-child
    // code must synchronize its call.cap calls once the parent can reach the same powerbox. Fall back
    // to the run's thunk only for a legacy non-sharing builder.
    let thunk_addr = rt.grant_thunk.load(Ordering::Acquire);
    let child_thunk: crate::CapThunk = if thunk_addr != 0 {
        core::mem::transmute::<usize, crate::CapThunk>(thunk_addr)
    } else {
        rt.cap_thunk
    };

    let Some((base, size)) = rt.resolve(mem_base, handle, trap_out) else {
        return 0; // `*trap_out` already holds the CapFault
    };
    // Resolve the granted separate module (op 5): its funcs, declared memory, and data segments.
    let Some((child_funcs, child_types, mod_mem, child_data, child_shadow)) =
        rt.resolve_child(module, self_prog, trap_out)
    else {
        return 0; // forged Module handle / no resolver — CapFault set
    };
    let entry = entry as u64;
    let child_size = if (0..64).contains(&size_log2) {
        1u64 << size_log2
    } else {
        0
    };
    let off = off as u64;
    // A named child receives no positional grant, so its entry is the 1- or 2-arg form (a compiled
    // command's `--child-entry` `_start` is the 1-arg starter form; it finds granted caps by name).
    let want_as = child_funcs
        .get(entry as usize)
        .is_some_and(|f| f.params.len() >= 2);
    let ok_entry = child_funcs.get(entry as usize).is_some_and(|f| {
        f.results.as_slice() == [ValType::I64]
            && (f.params.len() == 1 || f.params.len() == 2)
            && f.params.iter().all(|p| *p == ValType::I64)
    });
    // A separate-module child's carve must be **at least** its declared memory (FORK.md §8.6 /
    // #773 — the interpreter twin at `instantiate_rec`'s `mod_ok`): a larger window is a safe
    // superset (confinement, invariant 2, still masks every access to the actual carve), and a
    // malloc child *needs* it — its synthesized bump allocator's `heap_base` is `1<<declared` and
    // grows the heap up into `[1<<declared, carve)`, so a carve equal to the declared window leaves
    // no heap room. The child is compiled fully-mapped over the whole carve (`compile_child` /
    // `run_child_code_then`), so those heap pages are already committed and the allocator's `vm_map`
    // commits are no-ops — matching the interp's `nested_view` (mapped == carve).
    let mod_ok = mod_mem.is_none_or(|ml| ml <= size_log2 as i32);
    let fits = child_size != 0
        && child_size <= size
        && off & (child_size - 1) == 0
        && off.checked_add(child_size).is_some_and(|e| e <= size)
        // #964: a carve may not dip into the reserved NULL region `[0, guard)` (interp twin).
        && base + off >= rt.null_guard.load(Ordering::Acquire);
    if !ok_entry || !fits || !mod_ok {
        return EINVAL as i32;
    }
    // Materialize the module's data segments into the carve (op 5) before the grants + run.
    write_data_segments(child_data, mem_base, base + off, child_size);

    // Build the child powerbox host-side from the grant records (op 11); a bad record/name sets
    // `*trap_out` and fails the whole spawn closed.
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
        child_size,
        &mut gc,
        trap_out,
    ) == 0
    {
        return 0; // `*trap_out` already set by the builder
    }

    // IMPORTS.md phase 3 / S2.1: bind the child module's import manifest against the powerbox just
    // built, so its `call.import`s dispatch through instance bindings (the interpreter's inline
    // spawn does the same via `Host::bind_child_manifest` — differential lockstep).
    let bind_addr = rt.grant_bind_imports.load(Ordering::Acquire);
    if bind_addr != 0 {
        let bind: crate::ChildManifestBinder = core::mem::transmute(bind_addr);
        // §3.3 withhold: a `required` import with nothing to bind fails the spawn closed —
        // probeable `-EINVAL`, before compiling or running any child code (the interpreter's
        // inline spawn takes the same early exit).
        if bind(rt.grant_ctx(), gc.ctx, module) != 0 {
            release(gc.ctx);
            release(gc.retained_ctx);
            return EINVAL as i32;
        }
    }

    // Compile the foreign module's entry confined to the carve, with the child powerbox ctx so its
    // `self.resolve(name)` routes to the granted caps. Per-spawn ctx ⇒ uncached (like op 11).
    let child_fuel_addr = rt.arm_child_fuel(fuel); // §5 fuel: clamp to parent-remaining (0 ⇒ un-metered)
    let compiled = crate::compile_child(
        child_funcs,
        child_types,
        entry as FuncIdx,
        size_log2 as u8,
        child_thunk,
        gc.ctx,
        rt.epoch_addr,
        child_fuel_addr, // §5 fuel: the child decrements its own clamped budget cell
        rt.futex_sched,  // wait/notify against the parent domain's shared futex
        crate::InstEnv::null(),
        &rt.serve_handlers,
        gc.jit_table_log2, // #1296: slots for the units a `Jit`-holding child installs,,
        child_shadow,
    );
    let code = match compiled {
        Ok(code) => code,
        Err(_) => {
            release(gc.ctx);
            release(gc.retained_ctx);
            *trap_out = TrapKind::CapFault as i64;
            return 0;
        }
    };
    let mut args = vec![gc.inst_handle as i64];
    if want_as {
        args.push(gc.as_handle as i64);
    }
    let n_results = child_funcs[entry as usize].results.len();
    // Async (S1c): a spawned command runs on its own OS thread — the shell-exec primitive can
    // pipeline (`cmd1 | cmd2` over a granted region ring or pipe) instead of serializing.
    file_granted_carve_task(
        rt,
        code,
        base + off,
        size_log2 as u8,
        mem_base as *mut u8,
        args,
        n_results,
        release,
        &gc,
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
    seeds: Vec<(u64, Vec<u8>)>,
    premap_apply: Option<crate::PremapApply>,
    args: Vec<i64>,
    n_results: usize,
    release: crate::GrantChildReleaser,
    gc: &crate::GrantChild,
    trap_out: *mut i64,
    // #1587 — the funding budget + the window bytes `instantiate_detached` already took from it, so
    // a spawn that fails *after* that commit can hand them back.
    budget: i32,
    child_size: u64,
    // #1361 step 4 — `Some(arena)` for a durable parent's child: its window starts as a durable one
    // (context 0's shadow-SP word at its frame base) and carries a freeze cell recording `entry`.
    durable: Option<temen_ir::durable_abi::ShadowArena>,
    entry: u32,
) -> i32 {
    let code = std::sync::Arc::new(code);
    register_serve(rt, gc.ctx, &code);
    let teardown = granted_teardown(rt, release, gc.ctx, gc.lane_cap);
    let premap_ctx = SendRaw(gc.ctx);
    let filed = file_task(
        rt,
        code,
        mapped_log2,
        reserved_log2,
        // The window image: the module's data segments, then the payload at the args base; a durable
        // child's window also starts durable (`temen_durable::init_durable_window`'s one word).
        |rw| {
            for (off, bytes) in &seeds {
                let off = *off as usize;
                if let Some(end) = off.checked_add(bytes.len()) {
                    if end <= rw.len() {
                        rw[off..end].copy_from_slice(bytes);
                    }
                }
            }
            if let Some(a) = durable {
                init_durable_words(rw, a);
            }
        },
        // The op-15 pre-mapped region, aliased onto the fresh window by the host hook (the child
        // powerbox's own `map` path); none staged ⇒ nothing to do.
        |base, mapped, reserved| match premap_apply {
            // SAFETY: `premap_ctx.0` is the child powerbox this task owns; the window is live.
            Some(apply) => apply(premap_ctx.0, base, mapped, reserved) != 0,
            None => true,
        },
        None,
        args,
        n_results,
        lane_chain_of(gc),
        gc.retained_ctx as usize,
        teardown,
        None,
        durable.map(|a| DurableCell::new(a, entry, mapped_log2, reserved_log2)),
    );
    match filed {
        Filed::Slot(slot) => slot,
        Filed::AtCeiling => {
            *trap_out = TrapKind::ThreadFault as i64;
            0
        }
        // The child never ran (a refused pre-map alias or task stack) — the one refusal on this
        // path that happens *after* `budget_mem_take` committed, so un-spend the window bytes
        // (#1587) and answer `-EINVAL` like every other admission failure (INVARIANTS #5).
        Filed::Refused => {
            let give_addr = rt.grant_budget_mem_give.load(Ordering::Acquire);
            if give_addr != 0 {
                // SAFETY: a nonzero address is the embedder's registered `BudgetMemGiver`.
                let give: crate::BudgetMemGiver = unsafe { core::mem::transmute(give_addr) };
                unsafe { give(rt.grant_ctx(), budget, child_size) };
            }
            EINVAL as i32
        }
    }
}

/// A fresh durable window's control words, as `temen_durable::init_durable_window` writes them: the
/// freeze word `NORMAL` and context 0's shadow-SP word at its frame base (the empty stack).
fn init_durable_words(rw: &mut [u8], a: temen_ir::durable_abi::ShadowArena) {
    use temen_ir::durable_abi::{STATE_NORMAL, STATE_OFF};
    let s = STATE_OFF as usize;
    let b = a.region_base(0) as usize;
    if let Some(st) = rw.get_mut(s..s + 4) {
        st.copy_from_slice(&STATE_NORMAL.to_le_bytes());
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
/// As [`instantiate_module_named`]: `rt`/`mem_base`/`mem_size`/`trap_out` are the baked nursery, live
/// parent window base, **reserved** span, and run trap cell, valid for the call.
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
    let build: crate::GrantNamedChildBuilder = core::mem::transmute(build_addr);
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
    let Some((child_funcs, child_types, mod_mem, child_data, child_shadow)) =
        rt.resolve_child(module, self_prog, trap_out)
    else {
        return 0;
    };
    let entry = entry as u64;
    let child_size = if (0..64).contains(&size_log2) {
        1u64 << size_log2
    } else {
        0
    };
    let want_as = child_funcs
        .get(entry as usize)
        .is_some_and(|f| f.params.len() >= 2);
    let ok_entry = child_funcs.get(entry as usize).is_some_and(|f| {
        f.results.as_slice() == [ValType::I64]
            && (f.params.len() == 1 || f.params.len() == 2)
            && f.params.iter().all(|p| *p == ValType::I64)
    });
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
        || (rt.child_module_durable(module)
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
    let mut premap_apply: Option<crate::PremapApply> = None;
    if let Some((r, o)) = premap {
        let admit_addr = rt.grant_premap_admit.load(Ordering::Acquire);
        let stage_addr = rt.grant_premap_stage.load(Ordering::Acquire);
        let apply_addr = rt.grant_premap_apply.load(Ordering::Acquire);
        if admit_addr == 0 || stage_addr == 0 || apply_addr == 0 {
            *trap_out = TrapKind::CapFault as i64;
            return 0;
        }
        let admit: crate::PremapAdmit = core::mem::transmute(admit_addr);
        match admit(rt.grant_ctx(), r, o, child_size, trap_out) {
            1 => {}
            0 => return EINVAL as i32,
            _ => return 0, // forged: `*trap_out` is set
        }
        premap_apply = Some(core::mem::transmute::<usize, crate::PremapApply>(
            apply_addr,
        ));
    }
    // Admission = the budget's quota take (the commit; every refusal above charged nothing).
    if take(rt.grant_ctx(), budget as i32, child_size) == 0 {
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
        &mut gc,
        trap_out,
    ) == 0
    {
        return 0;
    }
    let bind_addr = rt.grant_bind_imports.load(Ordering::Acquire);
    if bind_addr != 0 {
        let bind: crate::ChildManifestBinder = core::mem::transmute(bind_addr);
        if bind(rt.grant_ctx(), gc.ctx, module) != 0 {
            release(gc.ctx);
            release(gc.retained_ctx);
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
            *trap_out = TrapKind::CapFault as i64;
            return 0;
        }
    }
    let child_fuel_addr = rt.arm_child_fuel(fuel);
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
        crate::InstEnv::null(),
        &rt.serve_handlers,
        gc.jit_table_log2, // #1296: slots for the units a `Jit`-holding child installs,
        child_shadow,
    );
    let code = match compiled {
        Ok(code) => code,
        Err(_) => {
            release(gc.ctx);
            release(gc.retained_ctx);
            *trap_out = TrapKind::CapFault as i64;
            return 0;
        }
    };
    let mut args = vec![gc.inst_handle as i64];
    if want_as {
        args.push(gc.as_handle as i64);
    }
    let n_results = child_funcs[entry as usize].results.len();
    // The window image: the module's data segments, then the payload at the args base.
    let mut seeds: Vec<(u64, Vec<u8>)> = child_data
        .iter()
        .map(|d| (d.offset, d.bytes.clone()))
        .collect();
    if !payload.is_empty() {
        seeds.push((temen_ir::module_args_base(), payload));
    }
    spawn_detached_child(
        rt,
        code,
        size_log2 as u8,
        temen_ir::DEFAULT_RESERVED_LOG2,
        seeds,
        premap_apply,
        args,
        n_results,
        release,
        &gc,
        trap_out,
        budget as i32,
        child_size,
        durable.then_some(child_shadow),
        entry as u32,
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
/// `rt` is the run's live nursery; `trap_out` the live trap cell (untouched — no trap paths).
pub(crate) unsafe extern "C" fn child_offer(
    rt: *const Nursery,
    child: i32,
    export: i64,
    _trap_out: *mut i64,
) -> i32 {
    const EINVAL: i32 = -22;
    let rt = &*rt;
    let mint_addr = rt.grant_mint.load(Ordering::Acquire);
    if mint_addr == 0 {
        return EINVAL;
    }
    let retained = {
        let children = rt.children.lock().unwrap_or_else(|e| e.into_inner());
        match children.get(child as usize) {
            // A joined child mirrors the interp's dead thread-slot: nothing to offer.
            Some(c) if !c.joined => c.retained,
            _ => 0,
        }
        // Lock dropped before the mint (which takes the child + parent powerbox locks in turn).
        // No release race: the retained ref is freed only at `join_children` (after guest code)
        // or on a spawn error path (before the child is ever filed).
    };
    if retained == 0 {
        return EINVAL;
    }
    let mint: crate::ChildOfferMint = core::mem::transmute(mint_addr);
    mint(rt.grant_ctx(), retained as *mut core::ffi::c_void, export)
}

pub(crate) unsafe extern "C" fn join(rt: *const Nursery, handle: i32, trap_out: *mut i64) -> i64 {
    let rt = &*rt;
    let mut children = rt.children.lock().unwrap_or_else(|e| e.into_inner());
    let slot = handle as usize;
    let done = match children.get_mut(slot) {
        Some(c) if !c.joined => {
            c.joined = true;
            c.done.clone() // clone the cell + drop the `children` lock before parking
        }
        _ => {
            *trap_out = TrapKind::CapFault as i64; // forged or already-joined handle
            return 0;
        }
    };
    drop(children);
    // D66 — a parent joining its §14 child holds no lane while it waits. Load-bearing under a cap of
    // 1: the child task cannot be dispatched at all until the joining vCPU steps aside, so without
    // this every bounded `instantiate`-then-`join` would deadlock. Released before the completion
    // cell's lock, re-taken after — blocking for a lane while holding that lock would stop the child
    // from publishing the very outcome being waited for. SAFETY: a nonzero `futex_sched` is the
    // run's live `Domain`, which outlives every child.
    let dom = (rt.futex_sched != 0)
        .then(|| unsafe { &*(rt.futex_sched as *const crate::os_thread_rt::Domain) });
    let lane = dom.map(|d| d.lane_chain()).unwrap_or_default();
    if let Some(d) = dom {
        d.lane_give_back(&lane);
    }
    let result = (|| -> i64 {
        // Park on the completion cell until the child's OS thread fills it (S1c async children). A durable
        // child ran synchronously, so its cell is already `Some` and this returns without waiting; an
        // async (op-0/5/8/11/13) child parks here until its thread publishes the outcome, with a bounded
        // re-check so a §5 host interrupt on the parent's `epoch_addr` still unwinds a waiter (the child
        // bakes that same cell, so it unwinds too).
        let mut st = done.state.lock().unwrap_or_else(|e| e.into_inner());
        let (result, trap) = loop {
            if let Some(outcome) = *st {
                break outcome;
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
            st = done
                .cv
                .wait_timeout(st, std::time::Duration::from_millis(20))
                .unwrap_or_else(|e| e.into_inner())
                .0;
        };
        if trap != 0 {
            *trap_out = trap; // a child trap propagates to the parent on join
            0
        } else {
            result
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
pub(crate) unsafe extern "C" fn poll(rt: *const Nursery, handle: i32, trap_out: *mut i64) -> i32 {
    let rt = &*rt;
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
pub(crate) unsafe extern "C" fn detach(rt: *const Nursery, handle: i32, trap_out: *mut i64) -> i32 {
    let rt = &*rt;
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

/// PROCESS.md S3 `kill(child) -> 0` (JIT). Acknowledges the request as a success. A synchronous child is
/// already finished; an **async** op-0/5 child (S1c) still runs on its own thread — it is reached only by
/// the run-wide §5 kill-path (the parent's `epoch_addr` cell, which the child bakes), not yet by a
/// per-child targeted interrupt (deferred: that lands with the confinement-codegen kill point). A forged
/// / already-joined handle is a `CapFault`.
///
/// # Safety
/// As [`join`].
pub(crate) unsafe extern "C" fn kill(rt: *const Nursery, handle: i32, trap_out: *mut i64) -> i32 {
    let rt = &*rt;
    let children = rt.children.lock().unwrap_or_else(|e| e.into_inner());
    match children.get(handle as usize) {
        Some(_) => 0,
        None => {
            *trap_out = TrapKind::CapFault as i64;
            0
        }
    }
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
