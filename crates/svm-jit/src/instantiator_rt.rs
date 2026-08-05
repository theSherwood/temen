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
//! resolves its `Instantiator` handle through the run's `cap.call` thunk (op 0 → the carve range
//! `[base, base+size)`), so a forged/wrong handle is an inert `CapFault` exactly as for any cap. The
//! child gets an **empty powerbox** for now (an inert `cap.call`); attenuated child caps + recursion +
//! "park only the calling fiber" (vs. today's synchronous run-at-`instantiate`) are follow-ups.

use crate::{mem, CapThunk, TrapKind};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex};
use svm_ir::{Data, Func, FuncIdx, ValType};

/// PROCESS.md S1: per-carve compile-cache key for a **non-durable** child — the identity the compiled
/// [`crate::ChildCode`] depends on. `funcs_ptr`/`n_funcs` name the module's function slice (stable
/// for the whole run per the [`crate::ModuleResolver`] contract — a held grant's storage outlives the
/// run and distinct live modules have distinct storage, so a stale-pointer collision cannot happen
/// within a run; the worst case of any mismatch is a miss, never wrong code). `entry` picks the
/// trampolined function; `size_log2` picks the baked window mask. The carve **base** is deliberately
/// absent — it is a runtime arg, so one entry serves every offset.
type ChildCodeKey = (usize, usize, u32, u8);

/// Negative-errno an out-of-range carve returns (matches the interpreter's `EINVAL`, §3e D42).
const EINVAL: i64 = -22;

/// One spawned child's **completion cell** (S1c): `(result, trap)` once the child has finished (`trap`
/// `0` = clean), or `None` while it is still running. An async child's OS thread fills it from the
/// child's *own* thread — until then `join` parks on `cv` and `poll` reports *running* (`0`); a
/// synchronous (durable) child's cell is `Some` before `instantiate` returns. `Arc` so `join` can
/// clone it and drop the `children` lock before parking (no lock held across a wait).
struct ChildDone {
    state: Mutex<Option<(i64, i64)>>,
    cv: Condvar,
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
}

impl Child {
    /// A child whose outcome is **already known** (the synchronous path): a cell pre-filled with
    /// `(result, trap)`.
    fn finished(result: i64, trap: i64) -> Child {
        Child {
            done: std::sync::Arc::new(ChildDone {
                state: Mutex::new(Some((result, trap))),
                cv: Condvar::new(),
            }),
            joined: false,
            retained: 0,
        }
    }

    /// A child whose OS thread is **still running** (the async path): the empty cell the thread fills on
    /// completion.
    fn pending(done: std::sync::Arc<ChildDone>) -> Child {
        Child {
            done,
            joined: false,
            retained: 0,
        }
    }
}

/// S1c — spawn `code` on its **own OS thread** in the child's own guarded window: the thread arms its
/// detect-and-kill recovery ([`mem::install_guard`]), runs [`crate::run_child_code`] (which allocates a
/// fresh `2^child_size_log2` window, seeds it from the carve, runs the child confined, and copies back),
/// and publishes `(result, trap)` into `done`. Returns the `JoinHandle` the nursery joins at teardown so
/// no child thread outlives the parent window. This is the concurrency primitive: `instantiate` returns
/// immediately, so a parent can spawn a second child (or its own work) while this one runs.
#[allow(clippy::too_many_arguments)] // a child spawn threads its full carve/completion/futex context
fn spawn_child_on_thread(
    code: std::sync::Arc<crate::ChildCode>,
    sub_base: u64,
    child_size_log2: u8,
    parent_mem_base: *mut u8,
    args: Vec<i64>,
    n_results: usize,
    done: std::sync::Arc<ChildDone>,
    futex_sched: usize,
) -> std::thread::JoinHandle<()> {
    struct SendPtr(*mut u8);
    // SAFETY: `parent_mem_base` is the parent window, which outlives every child (`join_children` runs
    // before it frees). The child thread touches only its **own** carve `[sub_base, +size)` for copy-in
    // / copy-back — disjoint from siblings and from the parent's live data — so crossing the pointer to
    // the thread races nothing (`ChildCode` is `Send + Sync`; the carve model is the disjointness the
    // guest owns, exactly like sibling `thread.spawn` accesses to one window).
    unsafe impl Send for SendPtr {}
    let base = SendPtr(parent_mem_base);
    // Count the child live in the parent domain's futex accounting for the wait/join deadlock
    // detection — before the spawn returns, so a wait issued right after already sees it. SAFETY:
    // a nonzero `futex_sched` is the run's live `Domain`, which outlives every child (children are
    // joined at run teardown, before the domain drops).
    if futex_sched != 0 {
        unsafe { (*(futex_sched as *const crate::os_thread_rt::Domain)).child_started() };
    }
    std::thread::Builder::new()
        .name("svm-child".into())
        .spawn(move || {
            let base = base; // move the wrapper into the thread
            mem::install_guard();
            // SAFETY: `code` is a live `Arc<ChildCode>` held by this closure; the carve is committed
            // parent memory the Instantiator bounded; `args` matches the entry arity (caller-checked).
            let (r, t) = unsafe {
                crate::run_child_code(&code, sub_base, child_size_log2, base.0, &args, n_results)
            };
            let mut st = done.state.lock().unwrap_or_else(|e| e.into_inner());
            *st = Some((r, t));
            done.cv.notify_all();
            // SAFETY: as `child_started` above — the domain outlives this (joined) thread.
            if futex_sched != 0 {
                unsafe { (*(futex_sched as *const crate::os_thread_rt::Domain)).child_finished() };
            }
        })
        .expect("spawn a §14 child OS thread")
}

/// S1c for **granted** children (Instantiator ops 8/11/13) — spawn the per-spawn-compiled child on
/// its own OS thread and register it as a pending join-table entry, returning its slot. Like
/// [`spawn_child_on_thread`], but the child owns a powerbox `Host` (`gc_ctx`) that must be freed
/// when it finishes: the thread releases it via [`run_child_code_then`]'s teardown hook — **after**
/// the copy-back but **while the child window is still alive** — so the host's region-canon purge
/// guard (which covers the child window's VA range) can never erase entries a later window at a
/// reused address just recorded. This is what lets two granted children run **concurrently** — a
/// pipeline over a granted `SharedRegion` ring — where the synchronous path serialized them.
///
/// # Safety
/// `code` is compiled against `gc_ctx` (the live child powerbox `Host`, exclusively owned by the
/// spawned thread from here until `release` frees it — `Host` is `Send`; state it shares with the
/// parent host rides `Sync` internals). The carve `[parent_mem_base + sub_base, +2^child_size_log2)`
/// is committed parent-window memory the Instantiator bounded; `args` matches the entry arity.
#[allow(clippy::too_many_arguments)]
unsafe fn spawn_granted_child(
    rt: &Nursery,
    code: crate::ChildCode,
    sub_base: u64,
    child_size_log2: u8,
    parent_mem_base: *mut u8,
    args: Vec<i64>,
    n_results: usize,
    release: crate::GrantChildReleaser,
    gc_ctx: *mut core::ffi::c_void,
    retained_ctx: *mut core::ffi::c_void,
) -> i32 {
    struct SendRaw<T>(T);
    // SAFETY: `parent_mem_base` outlives every child (`join_children` runs before it frees) and the
    // child thread touches only its own carve (the §14 disjointness the guest owns, as in
    // `spawn_child_on_thread`); `gc_ctx` is a heap `Host` handed over wholesale to the child thread
    // (`Host: Send` — checked where svm-run builds it), untouched by the parent after this call.
    unsafe impl<T> Send for SendRaw<T> {}
    let base = SendRaw(parent_mem_base);
    let ctx = SendRaw(gc_ctx);
    let code = std::sync::Arc::new(code);
    // CALLS.md 5c.1b — register the child's serve context on its shared powerbox before the child
    // thread starts (so a dispatch enqueued at any point of the child's life finds it). The
    // `ChildCode` Arc lives until the child thread ends, and the releaser clears the ctx before
    // that Arc drops (the teardown hook runs `release` first) — no stale read window.
    {
        let rs = rt.grant_register_serve.load(Ordering::Acquire);
        if rs != 0 {
            let rs: crate::ChildServeRegistrar = unsafe { core::mem::transmute(rs) };
            unsafe { rs(gc_ctx, std::sync::Arc::as_ptr(&code) as usize) };
        }
    }
    let done = std::sync::Arc::new(ChildDone {
        state: Mutex::new(None),
        cv: Condvar::new(),
    });
    let done2 = std::sync::Arc::clone(&done);
    let futex_sched = rt.futex_sched;
    // Count the child live for the parent domain's wait/join deadlock detection (see
    // [`spawn_child_on_thread`]). SAFETY: a nonzero `futex_sched` is the run's live `Domain`,
    // outliving every (teardown-joined) child.
    if futex_sched != 0 {
        unsafe { (*(futex_sched as *const crate::os_thread_rt::Domain)).child_started() };
    }
    let handle = std::thread::Builder::new()
        .name("svm-child".into())
        .spawn(move || {
            let (base, ctx) = (base, ctx);
            mem::install_guard();
            // SAFETY: per this function's contract; the teardown frees the child powerbox exactly
            // once, from the only thread still holding it.
            let (r, t) = unsafe {
                crate::run_child_code_then(
                    &code,
                    sub_base,
                    child_size_log2,
                    base.0,
                    &args,
                    n_results,
                    || release(ctx.0),
                )
            };
            let mut st = done2.state.lock().unwrap_or_else(|e| e.into_inner());
            *st = Some((r, t));
            done2.cv.notify_all();
            // SAFETY: as `child_started` above — the domain outlives this (joined) thread.
            if futex_sched != 0 {
                unsafe { (*(futex_sched as *const crate::os_thread_rt::Domain)).child_finished() };
            }
        })
        .expect("spawn a §14 granted-child OS thread");
    let mut children = rt.children.lock().unwrap_or_else(|e| e.into_inner());
    let slot = children.len();
    let mut child = Child::pending(done);
    // 5c.0 — retain the shared child powerbox for `child_offer` (released at join_children).
    child.retained = retained_ctx as usize;
    children.push(child);
    drop(children);
    rt.child_threads
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(handle);
    slot as i32
}

/// The per-run §14 nesting runtime, baked into the module's `Instantiator` `cap.call` sites. Holds
/// what compiling + running a child needs: the module's functions, the run's `cap.call` thunk/ctx
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
    cap_thunk: CapThunk,
    cap_ctx: *mut core::ffi::c_void,
    /// §14 separate-module children: the host callback resolving a guest's `Module` handle to the
    /// granted module's code/data (`None` ⇒ module ops are an inert `CapFault`). Kept apart from the
    /// `cap.call` thunk so the host pointers it yields are never guest-reachable.
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
    /// S1c: the OS threads spawned for **async** non-durable children (each runs `run_child_code` in the
    /// child's own guarded window and fills its completion cell). Tracked so the run **joins them all at
    /// teardown** ([`Nursery::join_children`]) before the parent window is freed — no child thread may
    /// outlive the window it copies to/from. Empty on a run with only synchronous children.
    child_threads: Mutex<Vec<std::thread::JoinHandle<()>>>,
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
    /// [`ChildCodeKey`], each entry is a compiled [`crate::ChildCode`] reused across spawns — so a
    /// shell respawning the same applet (any offset, same size) recompiles nothing. Held behind the
    /// nursery, alive for the run; the durable / nesting child bypasses it (its baked per-child
    /// nursery makes its code un-shareable). **`Arc`** (S1c): a cached child can be handed to an
    /// OS-thread child executor and run concurrently on several threads — sound because `ChildCode` is
    /// `Send + Sync` (its code arena + `fn_table` are immutable read-execute memory after
    /// `finalize_definitions`; the `unsafe impl` + compile-time assertion live in `lib.rs`). A lookup
    /// still drops the lock before the run. (Children run synchronously on the calling thread **today**;
    /// this makes the artifact ready for the async spawn slice that follows.)
    child_code: Mutex<HashMap<ChildCodeKey, std::sync::Arc<crate::ChildCode>>>,
    /// PROCESS.md S2 (JIT parity): the host callbacks for `instantiate_granted` (op 8) — build a
    /// granted child's powerbox `Host` and free it after the run — stored as raw fn-pointer addresses
    /// (`0` ⇒ none, an inert `CapFault`, like a run that re-grants nothing). Set once at run entry via
    /// [`Nursery::set_grant_hooks`] (the same interior-mutability contract as [`Nursery::set_durable`]:
    /// written before the guest runs, then only read by the `instantiate_granted` thunk), so no new
    /// param threads through the compile pipeline.
    grant_build: std::sync::atomic::AtomicUsize,
    grant_build_named: std::sync::atomic::AtomicUsize,
    grant_release: std::sync::atomic::AtomicUsize,
    grant_bind_imports: std::sync::atomic::AtomicUsize,
    /// CALLS.md 5c.0 — the `child_offer` mint hook ([`crate::ChildOfferMint`] as usize; 0 = none).
    grant_mint: std::sync::atomic::AtomicUsize,
    /// CALLS.md 5c.0 — the lock-taking cap thunk granted-child compiles run against
    /// ([`crate::CapThunk`] as usize; 0 ⇒ fall back to the run's `cap_thunk` — pre-5c.0 behavior,
    /// only correct for a builder that does not share the child `Host`).
    grant_thunk: std::sync::atomic::AtomicUsize,
    /// CALLS.md 5c.1b — the [`crate::ChildServeRegistrar`] hook (0 ⇒ none): registers a spawned
    /// granted child's `ChildCode` address on its shared powerbox so the locked thunk's serve arm
    /// can resolve + invoke handlers.
    grant_register_serve: std::sync::atomic::AtomicUsize,
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
    ) -> Nursery {
        Nursery {
            funcs,
            serve_handlers,
            cap_thunk,
            cap_ctx,
            resolve_module,
            epoch_addr,
            parent_fuel_addr,
            child_fuel_cells: Mutex::new(Vec::new()),
            futex_sched,
            children: Mutex::new(Vec::new()),
            child_threads: Mutex::new(Vec::new()),
            durable: AtomicBool::new(false),
            my_task,
            task_counter,
            frozen_nested_sink,
            child_code: Mutex::new(HashMap::new()),
            grant_build: std::sync::atomic::AtomicUsize::new(0),
            grant_build_named: std::sync::atomic::AtomicUsize::new(0),
            grant_release: std::sync::atomic::AtomicUsize::new(0),
            grant_bind_imports: std::sync::atomic::AtomicUsize::new(0),
            grant_register_serve: std::sync::atomic::AtomicUsize::new(0),
            grant_mint: std::sync::atomic::AtomicUsize::new(0),
            grant_thunk: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// PROCESS.md S2 (JIT parity): install the granted-child host callbacks — build a granted child's
    /// powerbox (positional op 8 / by-name op 11) and release it after the run. Called once at run entry
    /// (like [`Self::set_durable`]), before any `instantiate_granted`/`instantiate_named` site can fire.
    /// `None` leaves them `0` (both ops stay an inert `CapFault`).
    pub(crate) fn set_grant_hooks(&self, hooks: Option<crate::GrantChildHooks>) {
        let (b, bn, r, bi, m, t, rs) = match hooks {
            Some(h) => (
                h.build as usize,
                h.build_named as usize,
                h.release as usize,
                h.bind_imports as usize,
                h.mint as usize,
                h.thunk as usize,
                h.register_serve as usize,
            ),
            None => (0, 0, 0, 0, 0, 0, 0),
        };
        self.grant_build.store(b, Ordering::Release);
        self.grant_build_named.store(bn, Ordering::Release);
        self.grant_register_serve.store(rs, Ordering::Release);
        self.grant_release.store(r, Ordering::Release);
        self.grant_bind_imports.store(bi, Ordering::Release);
        self.grant_mint.store(m, Ordering::Release);
        self.grant_thunk.store(t, Ordering::Release);
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

    /// Push a captured §14 nested-child re-attach record into the **shared** subtree sink (coalesces at
    /// the root). `instantiate` calls this when a child left its carve `UNWINDING`.
    pub(crate) fn push_frozen_nested(&self, rec: crate::FrozenNested) {
        self.frozen_nested_sink
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(rec);
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
    pub(crate) fn join_children(&self) {
        let handles: Vec<std::thread::JoinHandle<()>> = {
            let mut g = self.child_threads.lock().unwrap_or_else(|e| e.into_inner());
            std::mem::take(&mut *g)
        };
        for h in handles {
            let _ = h.join();
        }
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
                let retained = std::mem::take(&mut c.retained);
                if retained != 0 {
                    // SAFETY: `retained` is a live `GrantChild::retained_ctx` this nursery owns,
                    // released exactly once here (spawn error paths released theirs before filing).
                    unsafe { release(retained as *mut core::ffi::c_void) };
                }
            }
        }
    }

    /// Mark the run durable (DURABILITY.md §4) — see the [`Nursery::durable`] field: the nesting
    /// thunks then fail closed. Called at run entry (`run_code_raw`), after the entry wrappers
    /// have applied the compile-side durable flag.
    pub(crate) fn set_durable(&self, durable: bool) {
        self.durable.store(durable, Ordering::Release);
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
    unsafe fn resolve_child(
        &self,
        module: i64,
        trap_out: *mut i64,
    ) -> Option<(&[Func], Option<i32>, &[Data])> {
        if module < 0 {
            return Some((&self.funcs, None, &[]));
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
        Some((funcs, Some(rm.memory_log2), data))
    }

    /// Resolve `handle` as this domain's `Instantiator` via the run's `cap.call` thunk, returning its
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
            svm_ir_iface_instantiator(),
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

/// The `Instantiator` interface id (§3e), kept in lockstep with `svm_interp::cap_id::INSTANTIATOR`.
/// (`svm-jit` does not depend on `svm-interp`; the host dispatch on the other side checks the same
/// constant, and the cross-backend tests pin them equal.)
#[inline]
fn svm_ir_iface_instantiator() -> u32 {
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
    mem_base: u64,
    handle: i32,
    module: i64,
    entry: i64,
    off: i64,
    size_log2: i64,
    fuel: i64,
    trap_out: *mut i64,
) -> i32 {
    let rt = &*rt;
    let durable = rt.durable.load(Ordering::Acquire);
    let Some((base, size)) = rt.resolve(mem_base, handle, trap_out) else {
        return 0; // `*trap_out` already holds the CapFault
    };
    let Some((child_funcs, mod_mem, child_data)) = rt.resolve_child(module, trap_out) else {
        return 0; // forged Module handle / no resolver — CapFault set
    };
    // §4 (DURABILITY.md, "JIT parity" slice 1): a durable run may now nest a **same-module** child.
    // Its funcs are the parent's own instrumented funcs; a runnable same-module child on the JIT is a
    // pure-compute (non-may-suspend) func — it has no poll sites, so it runs atomically to completion
    // in its carve with no durable control-word setup needed (a would-be *instrumented* child hits a
    // `cap.call` against its empty powerbox → `CapFault`, so it never reaches an unwind). Freezing a
    // *live* nested child on the JIT — which needs the carve's ctx-0 control words + shadow base seeded
    // to match the interpreter — is the next slice. A durable **separate-module** child (`mod_mem =
    // Some`) stays fail-closed (host-supplied module identity + freeze residue are a later slice), as
    // does `coro_spawn`. Guest-reachable errno, like a bad carve.
    if durable && mod_mem.is_some() {
        return EINVAL as i32;
    }

    // The carve must be a power-of-two-aligned sub-window within `[0, size)` — a child can only get
    // what the holder sub-allocates (§14/D19) — and a module child's carve must equal its declared
    // memory. Bad entry index / size / alignment ⇒ `-EINVAL`.
    let entry = entry as u64;
    let child_size = if (0..64).contains(&size_log2) {
        1u64 << size_log2
    } else {
        0
    };
    let off = off as u64;
    let mod_ok = mod_mem.is_none_or(|ml| ml == size_log2 as i32);
    let fits = child_size != 0
        && child_size <= size
        && off & (child_size - 1) == 0
        && off.checked_add(child_size).is_some_and(|e| e <= size)
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
            entry as FuncIdx,
            base + off,
            size_log2 as u8,
            mem_base as *mut u8,
            &args,
            rt.epoch_addr, // §5: the child polls the parent's kill-path cell, so one interrupt kills both
            child_fuel_addr, // §5 fuel: the child decrements its own clamped budget cell
            durable, // §4: seed the child's carve control words + give it an Instantiator powerbox
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
        children.push(Child::finished(result, trap));
        drop(children);
        // §4 freeze export: the child left its carve `UNWINDING` — record its re-attach residue into the
        // shared subtree sink tagged with this nursery's task, so a thaw re-creates the child domain.
        if unwound {
            rt.push_frozen_nested(crate::FrozenNested {
                parent_task: rt.my_task(),
                slot,
                carve_off: base + off,
                size_log2: size_log2 as u8,
                entry: entry as u32,
            });
        }
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
            entry as FuncIdx,
            size_log2 as u8,
            rt.epoch_addr, // §5: the child polls the parent's kill-path cell (one interrupt kills both)
            child_fuel_addr, // §5 fuel: 0 ⇒ un-metered (cacheable); nonzero ⇒ per-spawn (not cached)
            rt.futex_sched,  // wait/notify against the parent domain's shared futex
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
    let done = std::sync::Arc::new(ChildDone {
        state: Mutex::new(None),
        cv: Condvar::new(),
    });
    let handle = spawn_child_on_thread(
        code,
        base + off,
        size_log2 as u8,
        mem_base as *mut u8,
        args,
        n_results,
        std::sync::Arc::clone(&done),
        rt.futex_sched,
    );
    let mut children = rt.children.lock().unwrap_or_else(|e| e.into_inner());
    let slot = children.len();
    children.push(Child::pending(done));
    drop(children);
    rt.child_threads
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(handle);
    slot as i32
}

/// PROCESS.md S2 (JIT parity) — `instantiate_granted(grant_handle, entry, off, size_log2, quota)`
/// (Instantiator op 8): exactly [`instantiate`] (a **same-module** child confined to the carve), but
/// the child's powerbox is not empty — one of the parent's own coordinate-free capabilities
/// (`Stream`/`Exit`/`Clock`, named by `grant_handle`) is re-granted into a fresh child `Host`, which
/// the child receives as its **third** entry arg (after `Instantiator`, `AddressSpace`). The child
/// `Host` is built host-side by [`Nursery::grant_build`] (svm-run knows the `Host` type; svm-jit keeps
/// it opaque, exactly as the [`crate::ModuleResolver`] seam does for module storage), the child runs
/// against the run's own `cap.call` thunk with that host as its ctx, and the host is freed with
/// [`Nursery::grant_release`] afterwards.
///
/// A granted child is **not** cached (its baked per-spawn child-host ctx makes the code un-shareable —
/// the same exclusion the durable child takes) and, like every non-durable JIT child today, gets no
/// nesting `InstEnv` — its `Instantiator` arg is inert (recursive nesting of a *granted* child is a
/// follow-up, tied to JIT async children, S1c). A durable run fails closed (`-EINVAL`), a run that
/// re-grants nothing / a forged-or-non-copyable handle is a `CapFault`, and a bad carve/entry is
/// `-EINVAL` — all matching the interpreter's op-8 path.
///
/// # Safety
/// As [`instantiate`]: `rt`/`mem_base`/`trap_out` are the baked nursery, live parent window base, and
/// run trap cell, valid for the call.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe extern "C" fn instantiate_granted(
    rt: *const Nursery,
    mem_base: u64,
    handle: i32,
    grant_handle: i32,
    entry: i64,
    off: i64,
    size_log2: i64,
    fuel: i64,
    trap_out: *mut i64,
) -> i32 {
    let rt = &*rt;
    // §4: a durable run can't yet freeze a granted child's separate powerbox host — fail closed, like
    // the durable separate-module child. Guest-reachable errno.
    if rt.durable.load(Ordering::Acquire) {
        return EINVAL as i32;
    }
    // The host callbacks that build/free the child powerbox. A run that installed none re-grants
    // nothing → an inert `CapFault` (matching a host with no such capability to give).
    let build_addr = rt.grant_build.load(Ordering::Acquire);
    let release_addr = rt.grant_release.load(Ordering::Acquire);
    if build_addr == 0 || release_addr == 0 {
        *trap_out = TrapKind::CapFault as i64;
        return 0;
    }
    let build: crate::GrantChildBuilder = core::mem::transmute(build_addr);
    let release: crate::GrantChildReleaser = core::mem::transmute(release_addr);
    // 5c.0 — a builder that shares the child `Host` supplies the lock-taking thunk; granted-child
    // code must synchronize its cap.calls once the parent can reach the same powerbox. Fall back
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
    // A granted child is a **same-module** child (its funcs are the parent's own); its entry must be
    // the 3-arg form `(i64, i64, i64) -> (i64)` so it actually receives the re-granted handle.
    let child_funcs = &rt.funcs;
    let entry = entry as u64;
    let child_size = if (0..64).contains(&size_log2) {
        1u64 << size_log2
    } else {
        0
    };
    let off = off as u64;
    let ok_entry = child_funcs.get(entry as usize).is_some_and(|f| {
        f.results.as_slice() == [ValType::I64]
            && f.params.len() == 3
            && f.params.iter().all(|p| *p == ValType::I64)
    });
    let fits = child_size != 0
        && child_size <= size
        && off & (child_size - 1) == 0
        && off.checked_add(child_size).is_some_and(|e| e <= size);
    if !ok_entry || !fits {
        return EINVAL as i32;
    }

    // Build the child powerbox host-side (resolve the copyable grant against the parent host, mint the
    // child's `Instantiator`/`AddressSpace`/grant handles, share stdout/stderr sinks). A forged or
    // non-copyable handle fails the whole spawn closed.
    let mut gc = crate::GrantChild {
        ctx: core::ptr::null_mut(),
        retained_ctx: core::ptr::null_mut(),
        inst_handle: 0,
        as_handle: 0,
        grant_handle: 0,
    };
    if build(rt.cap_ctx, grant_handle, child_size, &mut gc) == 0 {
        *trap_out = TrapKind::CapFault as i64;
        return 0;
    }

    // Compile the child against the run's own `cap.call` thunk with the **child host** as ctx (so its
    // `Stream`/`Exit`/`Clock` cap.calls reach the re-granted cap, not the parent's powerbox), then run
    // it **asynchronously** on its own OS thread (S1c — granted children are concurrent, so a spawned
    // pair can pipeline). Uncached — the per-spawn child ctx is baked into the code. The child gets no
    // nesting `InstEnv` (like every non-durable JIT child today).
    let child_fuel_addr = rt.arm_child_fuel(fuel); // §5 fuel: clamp to parent-remaining (0 ⇒ un-metered)
    let compiled = crate::compile_child(
        child_funcs,
        entry as FuncIdx,
        size_log2 as u8,
        child_thunk,
        gc.ctx,
        rt.epoch_addr,
        child_fuel_addr, // §5 fuel: the child decrements its own clamped budget cell
        rt.futex_sched,  // wait/notify against the parent domain's shared futex
        crate::InstEnv::null(),
        &rt.serve_handlers,
    );
    let code = match compiled {
        Ok(code) => code,
        Err(_) => {
            // An un-compilable child (fibers/threads/setjmp, or a backend error) is a CapFault, like
            // the plain `instantiate` path. Free the powerbox host it will never run against.
            release(gc.ctx);
            release(gc.retained_ctx);
            *trap_out = TrapKind::CapFault as i64;
            return 0;
        }
    };
    let args = vec![
        gc.inst_handle as i64,
        gc.as_handle as i64,
        gc.grant_handle as i64,
    ];
    let n_results = child_funcs[entry as usize].results.len();
    spawn_granted_child(
        rt,
        code,
        base + off,
        size_log2 as u8,
        mem_base as *mut u8,
        args,
        n_results,
        release,
        gc.ctx,
        gc.retained_ctx,
    )
}

/// PROCESS.md S2 (JIT parity) — `instantiate_named(grants_ptr, grants_n, entry, off, size_log2, quota)`
/// (Instantiator op 11): the multi-cap, by-name form of [`instantiate_granted`]. The child powerbox is
/// built host-side by [`Nursery::grant_build_named`], which reads `grants_n` 16-byte grant records from
/// the **parent** window (`[mem_base, mem_base+mem_size)`) and re-grants each copyable handle under its
/// name; the child finds them by `cap.self.resolve` (lowered to the run's `cap.call` thunk with the
/// child host as ctx, so name resolution "just works"). The child entry is the 1- or 2-arg form
/// (`Instantiator` [, `AddressSpace`]) — no positional grant arg. Same non-durable / uncached /
/// non-nesting shape as [`instantiate_granted`]; a bad record/name is a `MemoryFault`, a non-copyable
/// grant a `CapFault`, a bad carve/entry `-EINVAL` — all matching the interpreter's op-11 path.
///
/// # Safety
/// As [`instantiate`]: `rt`/`mem_base`/`trap_out` are the baked nursery, live parent window base, and
/// run trap cell, valid for the call; `mem_size` is the parent window's mapped byte count.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe extern "C" fn instantiate_named(
    rt: *const Nursery,
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
    // code must synchronize its cap.calls once the parent can reach the same powerbox. Fall back
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
    let child_funcs = &rt.funcs;
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
        && off.checked_add(child_size).is_some_and(|e| e <= size);
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
    };
    if build(
        rt.cap_ctx,
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

    let child_fuel_addr = rt.arm_child_fuel(fuel); // §5 fuel: clamp to parent-remaining (0 ⇒ un-metered)
    let compiled = crate::compile_child(
        child_funcs,
        entry as FuncIdx,
        size_log2 as u8,
        child_thunk,
        gc.ctx,
        rt.epoch_addr,
        child_fuel_addr, // §5 fuel: the child decrements its own clamped budget cell
        rt.futex_sched,  // wait/notify against the parent domain's shared futex
        crate::InstEnv::null(),
        &rt.serve_handlers,
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
    spawn_granted_child(
        rt,
        code,
        base + off,
        size_log2 as u8,
        mem_base as *mut u8,
        args,
        n_results,
        release,
        gc.ctx,
        gc.retained_ctx,
    )
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
/// nursery, live parent window base, mapped byte count, and run trap cell, valid for the call.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe extern "C" fn instantiate_module_named(
    rt: *const Nursery,
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
    // code must synchronize its cap.calls once the parent can reach the same powerbox. Fall back
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
    let Some((child_funcs, mod_mem, child_data)) = rt.resolve_child(module, trap_out) else {
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
    // A module child's carve must equal its declared memory (§14 transparency), as in op 5.
    let mod_ok = mod_mem.is_none_or(|ml| ml == size_log2 as i32);
    let fits = child_size != 0
        && child_size <= size
        && off & (child_size - 1) == 0
        && off.checked_add(child_size).is_some_and(|e| e <= size);
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
    };
    if build(
        rt.cap_ctx,
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
        if bind(rt.cap_ctx, gc.ctx, module) != 0 {
            release(gc.ctx);
            release(gc.retained_ctx);
            return EINVAL as i32;
        }
    }

    // Compile the foreign module's entry confined to the carve, with the child powerbox ctx so its
    // `cap.self.resolve(name)` routes to the granted caps. Per-spawn ctx ⇒ uncached (like op 11).
    let child_fuel_addr = rt.arm_child_fuel(fuel); // §5 fuel: clamp to parent-remaining (0 ⇒ un-metered)
    let compiled = crate::compile_child(
        child_funcs,
        entry as FuncIdx,
        size_log2 as u8,
        child_thunk,
        gc.ctx,
        rt.epoch_addr,
        child_fuel_addr, // §5 fuel: the child decrements its own clamped budget cell
        rt.futex_sched,  // wait/notify against the parent domain's shared futex
        crate::InstEnv::null(),
        &rt.serve_handlers,
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
    spawn_granted_child(
        rt,
        code,
        base + off,
        size_log2 as u8,
        mem_base as *mut u8,
        args,
        n_results,
        release,
        gc.ctx,
        gc.retained_ctx,
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
    mint(rt.cap_ctx, retained as *mut core::ffi::c_void, export)
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

/// A durable §14 child's baked `cap.call` thunk (DURABILITY.md §4, "JIT parity"): its powerbox holds
/// exactly one capability — an `Instantiator` over the child's **own full window** `[0, child_size)`,
/// so the child can carve and run a grandchild of its own. `Nursery::resolve` calls this with iface-6
/// op-0 to read the holder's `[base, size]`; the child is confined to its window by the masking
/// lowering and can forge no other cap, so any handle resolves to `[0, child_size]` (full authority
/// over its own window, and nothing beyond). Anything else is an inert `CapFault`, matching the
/// interpreter's single-binding child powerbox (`grant_instantiator(0, child_size)`).
///
/// # Safety
/// `ctx` points at a live `u64` (the child's window size) for the call; `results`/`trap_out` are the
/// call-site slot buffers (`Nursery::resolve` / the `cap.call` lowering guarantee them).
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
    if type_id == svm_ir_iface_instantiator() && op == 0 && n_results >= 2 {
        let child_size = *(ctx as *const u64);
        *results = 0; // base, window-relative — the child's own window starts at 0
        *results.add(1) = child_size as i64; // size
        *trap_out = 0;
    } else {
        *trap_out = TrapKind::CapFault as i64;
    }
}
