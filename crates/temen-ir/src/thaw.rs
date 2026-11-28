//! The **thaw residue** and the one pass that admits it (DURABILITY.md §3, #3034).
//!
//! A snapshot's residue sections are structured data every engine's thaw acts on, some of it with raw
//! host stores at offsets the values pick. An artifact is untrusted input (owner, 2026-11-13), so the
//! residue crosses into an engine's thaw only as a [`Thaw`], which only [`validate`] builds: every value
//! in it is one a run of the module could have made. A freeze's own residue is engine state, which
//! rides unsealed until a thaw admits it; the one value in it a guest writes, each extent, the freeze
//! checks where it reads it ([`extent_fits`]). An engine gets the parts only by admitting it for the
//! code it runs ([`admit`], [`Thaw::seed`], against [`CodeFacts`]), so a `Thaw` validated for one
//! module cannot seed another, and destructures the [`ThawParts`] whole, so a residue field no engine
//! consumes cannot be dropped silently: it does not compile until each engine consumes it or declines
//! the thaw. A domain's serve state ([`Serve`]) is admitted beside its residue by the same pass.
//!
//! The records are the ones a freeze writes ([`FrozenFiber`], [`FrozenVCpu`], …), one definition for
//! every engine (INVARIANTS #13/#15). A live detached child or fork twin is a domain of its own, with a
//! window and a powerbox this crate cannot name: its payload is a type parameter (`D`, `T`), and the
//! pass reads only the keys beside it. Its own residue is admitted by the same pass, at its own depth,
//! against its own module, when the codec restores its nested artifact.

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::durable_abi::{ShadowArena, ShadowContexts, MAX_SHADOW_CONTEXTS};
use crate::{cap_id, handles, trap_code, Func, ImplExport, Inst, Module, ValType};

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

/// §3.6 slice 2 — one queued inbound dispatch (the serve-loop core): a call targeting this
/// domain's impl-export `export`, op `op`, args in the i64-slot ABI, completion posted under
/// `ticket`. Admitted as a handler over the domain's **one world** at a `svc.poll` service
/// point — run-to-completion this slice (handler parking rides the fiber-admission slice),
/// which is exactly the passive instance's serialized observable behavior (the oracle).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SvcDispatch {
    pub export: u32,
    pub op: u32,
    pub args: Vec<i64>,
    pub ticket: u64,
}

/// INVARIANTS #8 — what a call through a live offer (a call queued into another running domain)
/// answers: the handler's results, or a failure of the transport. The caller lays it out by its
/// own call's signature.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Reply {
    /// The handler returned: its results as slots, any handle among them still the callee's.
    Results(Vec<i64>),
    /// The transport failed — a full queue, a refused dispatch, a callee that ended unserved: a
    /// negative errno.
    Failed(i64),
}

impl Reply {
    /// The reply as one status, for a call whose op answers one integer (a pager's `page`): its first
    /// result, or the failure's errno.
    pub fn status(&self) -> i64 {
        match self {
            Reply::Results(r) => r.first().copied().unwrap_or(0),
            Reply::Failed(errno) => *errno,
        }
    }
}

/// #3034 — a domain's serve state as a snapshot carries it (DURABILITY.md §13.4, #1676, #1901): the
/// dispatches queued and the replies settled, the ticket counter, the abandoned reply waits
/// `(context, ticket)`, and the frozen handlers `(fiber slot, ticket)`. Admitted beside the residue
/// ([`validate`]), whose fibers its handlers name; a powerbox holds it in its live serve state, not
/// beside the residue at rest.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Serve {
    pub queue: Vec<SvcDispatch>,
    pub results: Vec<(u64, Reply)>,
    pub next_ticket: u64,
    pub waits: Vec<(u32, u64)>,
    pub handlers: Vec<(u32, u64)>,
}

impl Serve {
    /// Whether it carries nothing: a domain that serves nothing, or a fresh one.
    pub fn is_empty(&self) -> bool {
        *self == Serve::default()
    }
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

impl<D, T> RawResidue<D, T> {
    /// Whether it carries nothing.
    pub fn is_empty(&self) -> bool {
        self.fibers.is_empty()
            && self.vcpus.is_empty()
            && self.root_sp.is_none()
            && self.children.is_empty()
            && self.detached.is_empty()
            && self.live_detached.is_empty()
            && self.twins.is_empty()
            && self.exited_twins.is_empty()
    }
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

/// An admitted thaw residue, and the serve state beside it: every value in them is one a run of the
/// code it was validated against could have made. Only [`validate`] builds a non-empty one. It keeps
/// the root context it was admitted for, and whether the domain installs code ([`CodeFacts`]'s
/// `installs`), so it reaches an engine only through [`Thaw::seed`], which admits it again for that
/// engine's own code: a `Thaw` validated for one module cannot seed another (#3034 slice 2).
#[derive(Debug)]
pub struct Thaw<D, T> {
    residue: RawResidue<D, T>,
    serve: Serve,
    installs: bool,
}

/// A [`Thaw`] taken apart, for the engine that consumes it. Destructure it whole.
pub type ThawParts<D, T> = RawResidue<D, T>;

impl<D, T> Default for Thaw<D, T> {
    fn default() -> Self {
        Thaw {
            residue: RawResidue::default(),
            serve: Serve::default(),
            installs: false,
        }
    }
}

impl<D, T> Thaw<D, T> {
    /// No residue: a fresh run, a freeze, or a single-vCPU thaw whose window holds everything.
    pub fn none() -> Self {
        Self::default()
    }

    /// Whether it carries nothing.
    pub fn is_empty(&self) -> bool {
        self.residue.is_empty() && self.serve.is_empty()
    }

    /// Read the residue.
    pub fn parts(&self) -> &ThawParts<D, T> {
        &self.residue
    }

    /// Read the serve state.
    pub fn serve(&self) -> &Serve {
        &self.serve
    }

    /// The parts, for an engine that seeds `facts`' code with its root in context `root_ctx`: this
    /// residue admitted again for exactly that code and context (the check [`validate`] ran, against
    /// the engine's own facts and its own root context, the arguments [`admit`] takes). Refused, and
    /// handed back whole, otherwise, and when it carries serve state: an engine seeds none, since a
    /// powerbox holds it ([`Thaw::into_held`]).
    pub fn seed(self, facts: &CodeFacts, root_ctx: usize) -> Result<ThawParts<D, T>, Box<Self>> {
        let ok = self.serve.is_empty()
            && (self.residue.is_empty()
                || check(facts, self.installs, root_ctx, &self.residue, &self.serve).is_ok());
        if ok {
            Ok(self.residue)
        } else {
            Err(Box::new(self))
        }
    }

    /// The residue and the serve state, for a powerbox to hold: the serve state in its live serve
    /// state, the residue at rest until an engine's thaw admits it again ([`admit`]).
    pub fn into_held(self) -> (RawResidue<D, T>, Serve) {
        (self.residue, self.serve)
    }

    /// #3099 — the residue as a live detached child's own root sees it: the JIT runs the child as the
    /// root of its domain (task 0), where the tree-walker records the child's scheduler id `task`, so a
    /// vCPU it spawned names `task` as its spawner. Each such `parent_task` becomes 0, which is below
    /// every vCPU's task, so the residue stays one the pass admits.
    pub fn rerooted(mut self, task: u64) -> Self {
        if task != 0 {
            for v in &mut self.residue.vcpus {
                if v.parent_task as u64 == task {
                    v.parent_task = 0;
                }
            }
        }
        self
    }

    /// The same residue with each live domain's payload rebuilt for another engine, from its record
    /// (keys and payload). The keys stay as the pass admitted them.
    pub fn map<D2, T2>(
        self,
        mut detached: impl FnMut(LiveDetached<D>) -> D2,
        mut twin: impl FnMut(LiveTwin<T>) -> T2,
    ) -> Thaw<D2, T2> {
        let r = self.residue;
        let residue = RawResidue {
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
        };
        Thaw {
            residue,
            serve: self.serve,
            installs: self.installs,
        }
    }
}

/// How many tests hand a thaw's seed hostile values on purpose right now ([`expect_backstops`]).
static BACKSTOPS_EXPECTED: core::sync::atomic::AtomicUsize =
    core::sync::atomic::AtomicUsize::new(0);

/// #3034 — a refusal behind the pass, in front of a raw host store or an unchecked index: an engine's
/// seed refuses `what` there whatever residue reaches it. A residue the pass admitted never does, so
/// reaching one is a hole in the pass, loud in a debug build (it panics) unless a test is handing a
/// seed hostile values on purpose ([`expect_backstops`]). In release it does nothing: the refusal is
/// the caller's.
pub fn backstop(what: &str) {
    if cfg!(debug_assertions) && BACKSTOPS_EXPECTED.load(core::sync::atomic::Ordering::Relaxed) == 0
    {
        panic!("{what}");
    }
}

/// Say that this test hands thaw seeds hostile values on purpose, so a refusal behind the pass is the
/// behaviour under test rather than a hole in it ([`backstop`]), for as long as the guard lives. A test
/// hook. The seeds run on the engines' own threads, so the switch is the process's, not the calling
/// thread's: it is scoped in time, and a test running beside one in the same binary is quiet only
/// while that guard is held.
#[doc(hidden)]
#[must_use = "the backstops are expected only while the guard is held"]
pub fn expect_backstops() -> BackstopsExpected {
    BACKSTOPS_EXPECTED.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    BackstopsExpected(())
}

/// The guard [`expect_backstops`] returns: dropping it makes the backstops loud again.
#[doc(hidden)]
pub struct BackstopsExpected(());

impl Drop for BackstopsExpected {
    fn drop(&mut self) {
        BACKSTOPS_EXPECTED.fetch_sub(1, core::sync::atomic::Ordering::Relaxed);
    }
}

/// The residue names a value no run of the module could have made.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Malformed;

/// A residue [`validate`] refused, handed back whole: a caller that declines the thaw leaves it where
/// it found it. The serve state passed beside it is not handed back: a caller that declines no longer
/// holds it.
pub struct Refused<D, T> {
    pub residue: Box<RawResidue<D, T>>,
}

impl<D, T> core::fmt::Debug for Refused<D, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("Refused")
    }
}

impl<D, T> From<Refused<D, T>> for Malformed {
    fn from(_: Refused<D, T>) -> Malformed {
        Malformed
    }
}

/// The record kinds the code of a domain can make (#3034): a fiber needs a `cont.new` or a serve op
/// (`svc.poll`/`svc.wait`, whose handlers are fibers), a vCPU a `thread.spawn`, and a detached child an
/// `Instantiator` call. A domain that holds a `Jit` grant (`installs`) runs installed code in its
/// frames, which may make any of them (DESIGN.md §22). An engine that stood up no runtime for a kind
/// its residue holds declines the thaw whole (`temen_run`'s JIT hosts an installed unit's fibers and
/// threads only under the grant's hosting, #3104).
///
/// `installs` is read from the restored domain, whose `Jit` domains come from the artifact's
/// Section 5 whether or not the restoring host grants one: a forged Section 5 turns this rule off for
/// every kind. Tying it to the re-granted grant is #3096 A1.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Makes {
    fibers: bool,
    vcpus: bool,
    detached: bool,
}

/// #3034 — what [`validate`] needs to know about the code a residue is admitted against, owned: which
/// record kinds it makes, which of its functions `thread.spawn` can name, how many it has, its shadow
/// arena, and each impl export's handlers' arities. Built once from the code ([`CodeFacts::of`]), so
/// an engine that keeps no IR (the JIT's compiled module) answers the check from what it was compiled
/// from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CodeFacts {
    makes: Makes,
    /// Whether the code uses `cont.*` or `gc.roots`: the JIT stands up its fiber runtime for it.
    uses_fibers: bool,
    /// Whether the code uses a thread op (`thread.spawn`/`join`, `wait`/`notify`): the JIT runs it
    /// under its thread scheduler.
    uses_threads: bool,
    /// Per function, whether it is a spawn entry, `(i64, i64) -> (i64)`.
    spawn: Vec<bool>,
    arena: ShadowArena,
    /// Per impl export, per op: whether its handler is a function in range, as the runtime's
    /// enqueue requires (`svc_admits`).
    handlers: Vec<Vec<bool>>,
}

impl Default for CodeFacts {
    /// The facts of no code: it makes nothing and has no arena.
    fn default() -> Self {
        CodeFacts {
            makes: Makes::default(),
            uses_fibers: false,
            uses_threads: false,
            spawn: Vec::new(),
            arena: ShadowArena::EMPTY,
            handlers: Vec::new(),
        }
    }
}

impl CodeFacts {
    /// The facts of `code`, in one pass over it.
    pub fn of<'a>(code: impl Into<Code<'a>>) -> CodeFacts {
        let code = code.into();
        let mut makes = Makes::default();
        let (mut uses_fibers, mut uses_threads) = (false, false);
        for f in code.funcs {
            uses_fibers |= f.uses_fibers();
            uses_threads |= f.uses_threads() || f.uses_futex();
            for i in f.blocks.iter().flat_map(|b| &b.insts) {
                match i {
                    Inst::ContNew { .. } => makes.fibers = true,
                    // A serve op runs each dispatch as a handler fiber of the serving task (#2742); a
                    // durable domain reaches one only by a direct `call.cap` (an import never binds
                    // to one).
                    Inst::CapCall(c) if crate::durable_abi::is_serve_op(c.type_id, c.op) => {
                        makes.fibers = true
                    }
                    Inst::ThreadSpawn { .. } => makes.vcpus = true,
                    Inst::CapCall(c) if c.type_id == cap_id::INSTANTIATOR => makes.detached = true,
                    _ => {}
                }
            }
        }
        CodeFacts {
            makes,
            uses_fibers,
            uses_threads,
            spawn: code
                .funcs
                .iter()
                .map(|f| f.params == [ValType::I64, ValType::I64] && f.results == [ValType::I64])
                .collect(),
            arena: code.arena,
            handlers: code
                .impl_exports
                .iter()
                .map(|e| {
                    e.ops
                        .iter()
                        .map(|&f| (f as usize) < code.funcs.len())
                        .collect()
                })
                .collect(),
        }
    }

    /// The record kinds `[fibers, vcpus, detached]` a JIT instance of this code stands up a runtime
    /// for, when it hosts a thread domain: a fiber runtime for `cont.*`/`gc.roots`, a nursery for an
    /// `Instantiator`. Not [`validate`]'s rule: installed code may make a kind its module's runtime
    /// does not host (#3104).
    pub fn reseeds(&self) -> [bool; 3] {
        [self.uses_fibers, true, self.makes.detached]
    }

    /// Its shadow arena.
    pub fn arena(&self) -> ShadowArena {
        self.arena
    }

    /// Whether the code uses `cont.*` or `gc.roots` (the JIT's fiber runtime).
    pub fn uses_fibers(&self) -> bool {
        self.uses_fibers
    }

    /// Whether the code uses a thread op (the JIT's thread scheduler).
    pub fn uses_threads(&self) -> bool {
        self.uses_threads
    }

    /// Whether the code calls an `Instantiator` (it makes detached children; the JIT's nursery).
    pub fn uses_instantiator(&self) -> bool {
        self.makes.detached
    }
}

/// Whether `code` is a trap's wire code as a join outcome carries it: one a guest can see
/// ([`trap_code::DIV_BY_ZERO`] to [`trap_code::STACK_OVERFLOW`]), with a status only on an exit.
pub fn surfaced_trap(code: TrapCode) -> bool {
    let low = code & 0xffff_ffff;
    (trap_code::DIV_BY_ZERO..=trap_code::STACK_OVERFLOW).contains(&low)
        && (low == trap_code::EXIT || code == low)
}

/// #3034 — whether `sp` is an extent inside context `ctx`'s region: at or past its empty frame base,
/// at most its end. The one rule over a value a guest writes: each context's shadow-SP word is guest
/// memory, which the transform's push check bounds only at that context's own unwind. An engine's
/// freeze applies it to every extent it reads out of the window, where it reads it, and a failure is
/// that context's trap, `Unreachable` (DURABILITY.md §3); the thaw pass applies it to every extent an
/// artifact carries.
pub fn extent_fits(a: ShadowArena, ctx: usize, sp: u64) -> bool {
    in_region(a, ctx, sp)
}

fn in_region(a: ShadowArena, ctx: usize, sp: u64) -> bool {
    a.region_fits(ctx) && a.frame_base(ctx) <= sp && sp <= a.region_base(ctx) + a.stride
}

/// Whether child-table `slot` is one the table issued.
fn issued(children: &[u32], slot: usize) -> bool {
    children.get(slot).is_some_and(|&g| g != 0)
}

/// The code a residue is admitted against: a module's functions and its shadow arena. A [`Module`]
/// is one; an engine's freeze, which holds the functions it runs rather than their module, builds one.
#[derive(Clone, Copy, Debug)]
pub struct Code<'a> {
    pub funcs: &'a [Func],
    pub arena: ShadowArena,
    /// Its impl exports, whose handlers a serve state's dispatches name. Only [`validate`] with serve
    /// state reads them; [`admit`], which takes none, does not.
    pub impl_exports: &'a [ImplExport],
}

impl<'a> From<&'a Module> for Code<'a> {
    fn from(m: &'a Module) -> Code<'a> {
        Code {
            funcs: &m.funcs,
            impl_exports: &m.impl_exports,
            arena: m
                .memory
                .and_then(|mem| mem.shadow)
                .unwrap_or(ShadowArena::EMPTY),
        }
    }
}

/// The code a residue is admitted against: borrowed, its facts built for the one check, or the facts
/// an engine already holds ([`CodeFacts`]).
#[derive(Clone, Copy, Debug)]
pub enum CodeRef<'a> {
    Code(Code<'a>),
    Facts(&'a CodeFacts),
}

impl<'a> From<Code<'a>> for CodeRef<'a> {
    fn from(c: Code<'a>) -> Self {
        CodeRef::Code(c)
    }
}

impl<'a> From<&'a Module> for CodeRef<'a> {
    fn from(m: &'a Module) -> Self {
        CodeRef::Code(m.into())
    }
}

impl<'a> From<&'a CodeFacts> for CodeRef<'a> {
    fn from(f: &'a CodeFacts) -> Self {
        CodeRef::Facts(f)
    }
}

/// Admit `raw`, the residue of a domain running `code` whose root runs in shadow context `root_ctx`
/// (`0`, or a fork twin's forking context), and `serve`, the domain's serve state, or refuse them,
/// handing the residue back. `installs`: the domain holds a `Jit` grant ([`Makes`]). The one check
/// every engine's thaw, and every restore, goes through (#3034); the ranges are the ones a freeze
/// writes.
pub fn validate<'a, D, T>(
    code: impl Into<CodeRef<'a>>,
    installs: bool,
    root_ctx: usize,
    mut raw: RawResidue<D, T>,
    serve: Serve,
) -> Result<Thaw<D, T>, Refused<D, T>> {
    // The order every engine seeds in (and the codec writes): fibers by slot, vCPUs by task. A freeze
    // that hands its residue over in memory records them in the order they unwound.
    raw.fibers.sort_by_key(|f| f.slot);
    raw.vcpus.sort_by_key(|v| v.task);
    let thaw = Thaw {
        residue: raw,
        serve,
        installs,
    };
    if thaw.is_empty() {
        return Ok(thaw);
    }
    let owned;
    let facts = match code.into() {
        CodeRef::Facts(f) => f,
        CodeRef::Code(c) => {
            owned = CodeFacts::of(c);
            &owned
        }
    };
    match check(facts, installs, root_ctx, &thaw.residue, &thaw.serve) {
        Ok(()) => Ok(thaw),
        Err(Malformed) => Err(Refused {
            residue: Box::new(thaw.residue),
        }),
    }
}

/// [`validate`] for a residue with no serve state, taken apart for the engine that seeds it: what an
/// engine's thaw does with the residue a powerbox holds at rest.
pub fn admit<'a, D, T>(
    code: impl Into<CodeRef<'a>>,
    installs: bool,
    root_ctx: usize,
    raw: RawResidue<D, T>,
) -> Result<ThawParts<D, T>, Refused<D, T>> {
    validate(code, installs, root_ctx, raw, Serve::default()).map(|t| t.residue)
}

/// [`validate`]'s checks, over a residue in seeding order.
fn check<D, T>(
    code: &CodeFacts,
    installs: bool,
    root_ctx: usize,
    r: &RawResidue<D, T>,
    serve: &Serve,
) -> Result<(), Malformed> {
    let ok = |c: bool| if c { Ok(()) } else { Err(Malformed) };
    let a = code.arena;
    let makes = Makes {
        fibers: installs || code.makes.fibers,
        vcpus: installs || code.makes.vcpus,
        detached: installs || code.makes.detached,
    };

    // A module without a shadow arena is not freezable (INVARIANTS #16): it has no residue at all.
    if !a.region_fits(root_ctx) {
        return ok(r.is_empty() && serve_ok(code, root_ctx, 0, r, serve));
    }

    // Only a kind of record the domain's code can make.
    ok(makes.fibers || r.fibers.is_empty())?;
    ok(makes.vcpus || r.vcpus.is_empty())?;
    ok(makes.detached || (r.detached.is_empty() && r.live_detached.is_empty()))?;

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
        let entry = usize::try_from(v.func).ok().and_then(|f| code.spawn.get(f));
        ok(entry == Some(&true) && v.args.len() == 2)?;
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
        ok(f.slot == i && ShadowContexts::default().fiber_fits(a, i) && i + 1 != root_ctx)?;
        // Not a context a vCPU's extent placed a vCPU in.
        ok(contexts.fiber_fits(a, i))?;
        if f.is_free() {
            ok(f.func == 0 && f.arg0 == 0 && !f.consumed)?;
        } else {
            ok(in_region(a, i + 1, f.shadow_sp))?;
        }
    }

    // ---- Detached children: a completed one's outcome is a value or a trap a guest sees; a live one's
    // own id and lane are ones a spawn makes.
    ok(r.detached.iter().all(|d| {
        d.parent_task as u64 <= TASK_MAX && d.completed_result.err().is_none_or(surfaced_trap)
    }))?;
    for l in &r.live_detached {
        ok(l.parent_task as u64 <= TASK_MAX)?;
        // A JIT freeze records its live children's own id as 0 (each is the root of its domain).
        ok(l.task <= TASK_MAX && l.lane >= -1)?;
    }

    // ---- Fork twins: a pid a fork returns; a live one re-enters a module function in a context
    // the forking vCPU could hold.
    for t in &r.twins {
        ok(t.pid >= 1 && t.pid <= TASK_MAX && (t.entry as usize) < code.spawn.len())?;
        let ctx = t.ctx as usize;
        ok(ctx < MAX_SHADOW_CONTEXTS && a.region_fits(ctx))?;
    }
    ok(r.exited_twins
        .iter()
        .all(|t| t.pid >= 1 && t.pid <= TASK_MAX && t.outcome.err().is_none_or(surfaced_trap)))?;

    ok(serve_ok(code, root_ctx, held, r, serve))
}

/// [`check`]'s serve rules (#1676, #1901), arena or none (a freeze writes serve state for any domain
/// whose ticket counter moved): each queued dispatch names an op the code serves, as the runtime's
/// enqueue requires (its argument count is the serve loop's to answer, `EINVAL`); each ticket the
/// domain minted is distinct and below its counter, which a `+ 1` does not overflow; a frozen handler
/// sits in a live fiber; a reply wait belongs to a context of this domain that a freeze abandons a
/// call in (the root's, a live vCPU's in `held`, a live fiber's). A wait's ticket is its callee's,
/// and the record does not say which callee, so no rule here or in the codec ties the two (#3034).
fn serve_ok<D, T>(
    code: &CodeFacts,
    root_ctx: usize,
    held: u64,
    r: &RawResidue<D, T>,
    s: &Serve,
) -> bool {
    let servable = s.queue.iter().all(|d| {
        code.handlers
            .get(d.export as usize)
            .and_then(|ops| ops.get(d.op as usize))
            .is_some_and(|&ok| ok)
    });
    let mut minted: Vec<u64> = s
        .queue
        .iter()
        .map(|d| d.ticket)
        .chain(s.results.iter().map(|r| r.0))
        .chain(s.handlers.iter().map(|h| h.1))
        .collect();
    let below = minted.iter().all(|&t| t < s.next_ticket);
    minted.sort_unstable();
    let live_fiber = |slot: usize| r.fibers.get(slot).is_some_and(|f| !f.is_free());
    s.next_ticket <= TASK_MAX
        && servable
        && below
        && minted.windows(2).all(|w| w[0] != w[1])
        && s.handlers
            .iter()
            .all(|&(slot, _)| live_fiber(slot as usize))
        && s.waits.iter().all(|&(ctx, _)| {
            let ctx = ctx as usize;
            ctx == root_ctx
                || (ctx < 64 && held & (1 << ctx) != 0)
                || (ctx >= 1 && live_fiber(ctx - 1))
        })
}

#[cfg(test)]
mod tests;
