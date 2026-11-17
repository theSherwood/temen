//! The **thaw residue** and the one pass that admits it (DURABILITY.md §3, #3034).
//!
//! A snapshot's residue sections are structured data every engine's thaw acts on, some of it with raw
//! host stores at offsets the values pick. An artifact is untrusted input (owner, 2026-11-13), so the
//! residue crosses into an engine only as a [`Thaw`], which only [`validate`] builds: every value in it
//! is one a run of the module could have made. An engine takes it apart with [`Thaw::into_parts`] and
//! destructures the [`ThawParts`] whole, so a residue field no engine consumes cannot be dropped
//! silently: it does not compile until each engine consumes it or declines the thaw.
//!
//! The records are the ones a freeze writes ([`FrozenFiber`], [`FrozenVCpu`], …), one definition for
//! every engine (INVARIANTS #13/#15). A live detached child or fork twin is a domain of its own, with a
//! window and a powerbox this crate cannot name: its payload is a type parameter (`D`, `T`), and the
//! pass reads only the keys beside it. Its own residue is admitted by the same pass, at its own depth,
//! against its own module, when the codec restores its nested artifact.

use alloc::vec::Vec;

use crate::durable_abi::{ShadowArena, ShadowContexts, MAX_SHADOW_CONTEXTS};
use crate::{handles, Module, ValType};

/// The largest task id, pid or ticket a thaw accepts: room for every `+ 1` a run makes after it.
pub const TASK_MAX: u64 = i64::MAX as u64;

/// A fiber flattened for freeze (DURABILITY.md §12.8 slice 3.1.4–5), as the snapshot carries it. Its
/// continuation is bytes in its in-window shadow region `[frame_base(slot+1), shadow_sp)`; this is the
/// small host-side residue: where it sits and how to re-enter it on thaw. Re-entry recreates it as a
/// `Pending` fiber so a thaw `cont.resume` runs its entry under `REWINDING`, rebuilding then
/// re-parking it.
///
/// Every slot of the fiber table rides (#1684), so a thaw rebuilds the table slot for slot: a
/// **fresh** fiber (never resumed) is a record whose extent is its empty frame base — the resume that
/// thaws it starts it from its entry — and a **free** slot is [`FrozenFiber::free`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrozenFiber {
    /// Registry slot = the guest fiber handle (so re-seeding preserves handle values).
    pub slot: usize,
    /// The fiber's entry funcref, re-entered on thaw. A funcref names a dispatch-table slot, which a
    /// guest-JIT unit may have installed, so the pass leaves it to the resume, which masks and
    /// type-checks it on every engine (`FiberFault`).
    pub func: i32,
    /// The entry's first argument, `cont.new`'s `arg0`. Inert for rewind.
    pub arg0: i64,
    /// Window offset of the flattened shadow-SP — the extent of its frozen continuation. `0` for a
    /// free slot: no shadow region lies at offset 0.
    pub shadow_sp: u64,
    /// The slot's **generation** counter at freeze: a full `u64`, masked only where a handle is
    /// compared, so any value is one a long enough run makes.
    pub generation: u64,
    /// #1538 — the fiber's last `suspend` value was already **consumed** by a resumer that ran on, so
    /// a thaw claim delivers its argument at the re-executed `suspend` instead of re-parking.
    pub consumed: bool,
}

impl FrozenFiber {
    /// #1684 — a free slot (its fiber finished): only its generation rides.
    pub fn free(slot: usize, generation: u64) -> FrozenFiber {
        FrozenFiber {
            slot,
            func: 0,
            arg0: 0,
            shadow_sp: 0,
            generation,
            consumed: false,
        }
    }

    /// Whether this is a [`FrozenFiber::free`] slot.
    pub fn is_free(&self) -> bool {
        self.shadow_sp == 0
    }
}

/// A **spawned vCPU** (a `thread.spawn` child) flattened for freeze (DURABILITY.md §12.8 slice 3.2.1):
/// the host-side residue needed to reconstruct it on thaw. The root vCPU needs none — its entry is the
/// thaw caller's. A completed-but-unjoined child (#1685) rides with its result and inert entry fields.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrozenVCpu {
    /// The child's task id at freeze, preserved on thaw so the §12.6 canonical re-freeze records the
    /// same task.
    pub task: usize,
    /// The task that spawned it (slice 3.4). Always below `task`. It need not name any record: the
    /// spawner may have finished, joined or not, before the freeze (#3034 D3); the child is still in
    /// the domain's one child table (#2043).
    pub parent_task: usize,
    /// Its slot in the domain's child table (#1685, #2043); `0` for one no handle names.
    pub slot: usize,
    /// #2043 — whether a handle still names it.
    pub joinable: bool,
    /// Its entry function (the `thread.spawn` target, a module function of the spawn signature).
    pub func: i32,
    /// Its spawn args (`[arg0, arg1]`), replayed on re-spawn.
    pub args: Vec<i64>,
    /// Window offset of its flattened shadow-SP, inside the context it holds.
    pub shadow_sp: u64,
    /// `Some(result)` for a completed-but-unjoined child; `func`, `args`, `shadow_sp` are then inert
    /// (`0`, empty, `0`).
    pub completed_result: Option<i64>,
}

/// A trap's wire code ([`crate::trap_code`]), as a join outcome carries it.
pub type TrapCode = i64;

/// #1361 step 2 — a **completed-but-unjoined detached §14 child** captured at a subtree freeze: only
/// its `thread.join` outcome crosses, at its spawner's join slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrozenDetached {
    /// The task that spawned it.
    pub parent_task: usize,
    /// The join slot it was minted at.
    pub slot: usize,
    /// Its join outcome: its value, or the code of the trap its `join` answers (#1674).
    pub completed_result: Result<i64, TrapCode>,
}

/// #2571 — a servicer's **fork twin that exited** before a freeze, its outcome waiting for its
/// parent's `wait`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrozenTwin {
    /// Its pid, which the parent's `wait` names.
    pub pid: u64,
    /// How it ended: its value, or its trap's code.
    pub outcome: Result<i64, TrapCode>,
}

/// #1361 step 4 — a **live** detached child: the keys the pass checks, beside its domain (`payload`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LiveDetached<D> {
    /// The task that spawned it.
    pub parent_task: usize,
    /// Its spawner's join slot. Read only while its spawner can join it ([`Self::joinable`]).
    pub slot: usize,
    /// #2465 — whether its spawner still holds its join claim.
    pub joinable: bool,
    /// Its own task id in the frozen run.
    pub task: u64,
    /// Its D66 lane cap (`-1` = unbounded).
    pub lane: i64,
    /// Its restored domain, and what else a thaw needs to launch it.
    pub payload: D,
}

/// #1688 — a **live** fork twin: the keys the pass checks, beside its domain (`payload`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LiveTwin<T> {
    /// The pid its parent's `fork()` returned.
    pub pid: u64,
    /// The program function its continuation re-enters: the forking vCPU's entry.
    pub entry: u32,
    /// The shadow context it runs in: the forking vCPU's.
    pub ctx: u32,
    /// Its restored domain, and what else a thaw needs to re-enter it.
    pub payload: T,
}

/// A domain's thaw residue as the codec decodes it, **not yet admitted**: no engine takes one. Hand it
/// to [`validate`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawResidue<D, T> {
    pub fibers: Vec<FrozenFiber>,
    pub vcpus: Vec<FrozenVCpu>,
    /// The root's flattened shadow-SP extent: present whenever `vcpus` is non-empty (the codec carries
    /// it only then; an in-memory freeze records it for a lone root too).
    pub root_sp: Option<u64>,
    /// The domain's child-handle table: every slot's counter (#2043).
    pub children: Vec<u32>,
    pub detached: Vec<FrozenDetached>,
    pub live_detached: Vec<LiveDetached<D>>,
    pub twins: Vec<LiveTwin<T>>,
    pub exited_twins: Vec<FrozenTwin>,
}

impl<D, T> Default for RawResidue<D, T> {
    fn default() -> Self {
        RawResidue {
            fibers: Vec::new(),
            vcpus: Vec::new(),
            root_sp: None,
            children: Vec::new(),
            detached: Vec::new(),
            live_detached: Vec::new(),
            twins: Vec::new(),
            exited_twins: Vec::new(),
        }
    }
}

/// An admitted thaw residue: every value in it is one a run of the module it was validated against
/// could have made. Only [`validate`] builds a non-empty one.
#[derive(Debug)]
pub struct Thaw<D, T>(RawResidue<D, T>);

/// A [`Thaw`] taken apart, for the engine that consumes it. Destructure it whole.
pub type ThawParts<D, T> = RawResidue<D, T>;

impl<D, T> Default for Thaw<D, T> {
    fn default() -> Self {
        Thaw(RawResidue::default())
    }
}

impl<D, T> Thaw<D, T> {
    /// No residue: a fresh run, a freeze, or a single-vCPU thaw whose window holds everything.
    pub fn none() -> Self {
        Self::default()
    }

    /// Whether it carries nothing.
    pub fn is_empty(&self) -> bool {
        let r = &self.0;
        r.fibers.is_empty()
            && r.vcpus.is_empty()
            && r.root_sp.is_none()
            && r.children.is_empty()
            && r.detached.is_empty()
            && r.live_detached.is_empty()
            && r.twins.is_empty()
            && r.exited_twins.is_empty()
    }

    /// Read it.
    pub fn parts(&self) -> &ThawParts<D, T> {
        &self.0
    }

    /// Take it apart, to consume it.
    pub fn into_parts(self) -> ThawParts<D, T> {
        self.0
    }

    /// The same residue with each live domain's payload rebuilt for another engine, from its record
    /// (keys and payload). The keys stay as the pass admitted them.
    pub fn map<D2, T2>(
        self,
        mut detached: impl FnMut(LiveDetached<D>) -> D2,
        mut twin: impl FnMut(LiveTwin<T>) -> T2,
    ) -> Thaw<D2, T2> {
        let r = self.0;
        Thaw(RawResidue {
            fibers: r.fibers,
            vcpus: r.vcpus,
            root_sp: r.root_sp,
            children: r.children,
            detached: r.detached,
            live_detached: r
                .live_detached
                .into_iter()
                .map(|l| {
                    let (parent_task, slot, joinable, task, lane) =
                        (l.parent_task, l.slot, l.joinable, l.task, l.lane);
                    LiveDetached {
                        parent_task,
                        slot,
                        joinable,
                        task,
                        lane,
                        payload: detached(l),
                    }
                })
                .collect(),
            twins: r
                .twins
                .into_iter()
                .map(|t| {
                    let (pid, entry, ctx) = (t.pid, t.entry, t.ctx);
                    LiveTwin {
                        pid,
                        entry,
                        ctx,
                        payload: twin(t),
                    }
                })
                .collect(),
            exited_twins: r.exited_twins,
        })
    }
}

/// The residue names a value no run of the module could have made.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Malformed;

/// Whether `f` is a module function `thread.spawn` can name: `(i64, i64) -> (i64)`.
fn spawn_entry(m: &Module, f: i32) -> bool {
    usize::try_from(f)
        .ok()
        .and_then(|f| m.funcs.get(f))
        .is_some_and(|f| f.params == [ValType::I64, ValType::I64] && f.results == [ValType::I64])
}

/// Whether `sp` is an extent inside context `ctx`'s region: at or past its empty frame base, at most
/// its end.
fn in_region(a: ShadowArena, ctx: usize, sp: u64) -> bool {
    a.region_fits(ctx) && a.frame_base(ctx) <= sp && sp <= a.region_base(ctx) + a.stride
}

/// Whether child-table `slot` is one the table issued.
fn issued(children: &[u32], slot: usize) -> bool {
    children.get(slot).is_some_and(|&g| g != 0)
}

/// Admit `raw`, the residue of a domain of `m` whose root runs in shadow context `root_ctx` (`0`, or a
/// fork twin's forking context), or refuse it. The one check every engine's thaw goes through
/// (#3034); the ranges are the ones a freeze writes.
pub fn validate<D, T>(
    m: &Module,
    root_ctx: usize,
    mut raw: RawResidue<D, T>,
) -> Result<Thaw<D, T>, Malformed> {
    // The order every engine seeds in (and the codec writes): fibers by slot, vCPUs by task. A freeze
    // that hands its residue over in memory records them in the order they unwound.
    raw.fibers.sort_by_key(|f| f.slot);
    raw.vcpus.sort_by_key(|v| v.task);
    let ok = |c: bool| if c { Ok(()) } else { Err(Malformed) };
    let a = m
        .memory
        .and_then(|mem| mem.shadow)
        .unwrap_or(ShadowArena::EMPTY);

    // A module without a shadow arena is not freezable (INVARIANTS #16): it has no residue at all.
    if !a.region_fits(root_ctx) {
        ok(Thaw(raw).is_empty())?;
        return Ok(Thaw::none());
    }
    let r = &raw;

    // ---- The child-handle table (#2043): at most `SLOTS` counters, the last one issued, and each one
    // a counter `Table::issue` can leave (it skips a generation of 0).
    ok(r.children.len() <= handles::SLOTS)?;
    ok(r.children.last() != Some(&0))?;
    ok(r.children
        .iter()
        .all(|&g| g == 0 || g & handles::GEN_MASK != 0))?;

    // Every join slot a record names is issued, and no two name one slot.
    let mut slots: Vec<usize> = r
        .vcpus
        .iter()
        .filter(|v| v.joinable)
        .map(|v| v.slot)
        .chain(r.detached.iter().map(|d| d.slot))
        .chain(
            r.live_detached
                .iter()
                .filter(|l| l.joinable)
                .map(|l| l.slot),
        )
        .collect();
    ok(slots.iter().all(|&s| issued(&r.children, s)))?;
    slots.sort_unstable();
    ok(slots.windows(2).all(|w| w[0] != w[1]))?;

    // ---- Spawned vCPUs: ascending task ids below `TASK_MAX`, each spawned by a lower id. A live one
    // holds a context of the vCPU pool, its own, and an extent inside it, and runs a spawn entry; a
    // completed one carries its result and nothing else.
    let mut held = 0u64; // the contexts the live vCPUs hold (each below 64: `vcpu_of_sp`)
    let mut last = 0usize;
    for v in &r.vcpus {
        ok(v.task > last && v.task as u64 <= TASK_MAX && v.parent_task < v.task)?;
        last = v.task;
        ok(v.joinable || (v.slot == 0 && v.completed_result.is_none()))?;
        if v.completed_result.is_some() {
            ok(v.func == 0 && v.args.is_empty() && v.shadow_sp == 0)?;
            continue;
        }
        let ctx = ShadowContexts::vcpu_of_sp(a, v.shadow_sp).ok_or(Malformed)?;
        ok(ctx != root_ctx && in_region(a, ctx, v.shadow_sp))?;
        ok(spawn_entry(m, v.func) && v.args.len() == 2)?;
        ok(held & (1 << ctx) == 0)?;
        held |= 1 << ctx;
    }
    let mut contexts = ShadowContexts::default();
    contexts.seed(a, held);
    // The root's extent rides beside every vCPU (the window's shared word holds the last context to
    // run), inside the root's own region. A freeze records it for a lone root too.
    ok(r.vcpus.is_empty() || r.root_sp.is_some())?;
    ok(r.root_sp.is_none_or(|sp| in_region(a, root_ctx, sp)))?;

    // ---- Fibers: one record per slot, dense from 0, each slot's context below every live vCPU's and
    // inside the arena (`cont.new`'s own rule). A free slot carries only its generation; a live one
    // an extent inside its own region.
    for (i, f) in r.fibers.iter().enumerate() {
        ok(f.slot == i && contexts.fiber_fits(a, i) && i + 1 != root_ctx)?;
        if f.is_free() {
            ok(f.func == 0 && f.arg0 == 0 && !f.consumed)?;
        } else {
            ok(in_region(a, i + 1, f.shadow_sp))?;
        }
    }

    // ---- Detached children: a live one's own id and lane are ones a spawn makes.
    ok(r.detached.iter().all(|d| d.parent_task as u64 <= TASK_MAX))?;
    for l in &r.live_detached {
        ok(l.parent_task as u64 <= TASK_MAX)?;
        // A JIT freeze records its live children's own id as 0 (each is the root of its domain).
        ok(l.task <= TASK_MAX && l.lane >= -1)?;
    }

    // ---- Fork twins: a pid a fork returns; a live one re-enters a module function in a context
    // the forking vCPU could hold.
    for t in &r.twins {
        ok(t.pid >= 1 && t.pid <= TASK_MAX && (t.entry as usize) < m.funcs.len())?;
        let ctx = t.ctx as usize;
        ok(ctx < MAX_SHADOW_CONTEXTS && a.region_fits(ctx))?;
    }
    ok(r.exited_twins
        .iter()
        .all(|t| t.pid >= 1 && t.pid <= TASK_MAX))?;

    Ok(Thaw(raw))
}

#[cfg(test)]
mod tests;
