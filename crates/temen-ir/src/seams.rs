//! The concurrency and park **seams** some functions' instructions touch, found in one linear scan
//! ([`scan_seams`]). Every engine's routing reads them from here: the bytecode engine's compile
//! gate and its escapes, the serve qualification the JIT's routing folds on
//! (`temen_interp::bytecode::serve_qualifies`), and whether a serve handler can park at all
//! ([`handler_never_parks`]), which decides where the JIT runs a direct handoff's handler. So the
//! engines cannot drift over what counts as a seam (INVARIANTS #9 and #15: one definition).

use crate::{cap_id, Func, FuncIdx, Inst, Terminator};
use crate::{CAP_IMPORT_ATTACH_TYPE_ID, CAP_IMPORT_TYPE_ID, CAP_SELF_TYPE_ID};

/// What [`scan_seams`] finds in some functions.
#[derive(Default)]
pub struct Seams {
    pub has_fiber: bool,
    pub has_thread: bool,
    pub has_instantiate: bool,
    pub has_gc: bool,
    pub has_svc: bool,
    pub has_park_seam: bool,
    /// I38's **timed** `svc.wait`: the scheduler's deadline machinery, oracle-only (it is a
    /// `has_park_seam` too).
    pub has_timed_wait: bool,
    /// FORK.md §9 — `clone_caller` / `reap` (self-ops 11, 12): the fork servicer primitives. A
    /// distinct seam because the **bytecode** engine services them natively (its fork-serving
    /// escape) while the serve qualification vetoes them ([`Seams::svc_park_veto`] keeps it). The
    /// tree-walker never serves a call inline into a domain that has them (#2854): both need the
    /// caller parked on the dispatch.
    pub has_fork: bool,
    /// #744 — a call whose callee is unknown until it runs: an indirect call, a dynamic-mode import,
    /// or a `call.cap` on an interface outside the built-ins that never park (a live offer parks
    /// its caller on a ticket, an embedder's `HostProc` or the `Blocking` cap can punt, a `Jit`
    /// invoke runs any unit). Read over a serve handler's closure only.
    pub has_opaque_call: bool,
}

impl Seams {
    /// The **serve-qualification veto**: a service point (`svc.poll` / `svc.wait`) coexisting with
    /// any seam that could park or unwind a handler mid-dispatch. This is the single definition of
    /// that disjunction — consulted by both the bytecode compile gate and the serve qualification
    /// temen-run's JIT routing folds on — so the two backends can never drift over which modules
    /// serve natively vs. decline to the tree-walk oracle. Adding a new park-capable seam means
    /// extending this one list. (INVARIANTS.md §9: one veto predicate, one definition.)
    ///
    /// `has_fork` is in the disjunction, so a forking server is not serve-qualified. That is not
    /// what keeps fork off the Cranelift JIT: a forking topology nests, and temen-run's serve fold
    /// leaves a nesting module on the JIT. The JIT declines fork itself (`temen_jit::admits`,
    /// #2553). The **bytecode** engine services `clone_caller`/`reap` natively; its compile gate
    /// takes its fork-serving escape past this veto (a per-backend split, the one the two-predicate
    /// structure exists to allow).
    pub fn svc_park_veto(&self) -> bool {
        self.has_svc
            && (self.has_park_seam
                || self.has_fiber
                || self.has_thread
                || self.has_instantiate
                || self.has_gc
                || self.has_fork)
    }
}

/// Whether a `call.cap` on interface `type_id` can never park its caller, by the interface alone:
/// a built-in whose every op runs to completion (the parking ops of the stream, instantiator and
/// self interfaces are classified by op, before this is asked). Any other interface is a live
/// offer, a wired guest impl or embedder-defined, and may park.
pub fn iface_never_parks(type_id: u32) -> bool {
    use cap_id::*;
    matches!(
        type_id,
        STREAM
            | EXIT
            | CLOCK
            | SHARED_REGION
            | ADDRESS_SPACE
            | MODULE_LOADER
            | MODULE
            | JIT_CODE
            | BUDGET
            | FREEZE_AUTHORITY
            | CAP_SELF_TYPE_ID
            | CAP_IMPORT_ATTACH_TYPE_ID
    )
}

/// The seams `funcs` touch, in one linear pass over their instructions.
pub fn scan_seams<'a>(funcs: impl IntoIterator<Item = &'a Func>) -> Seams {
    let mut s = Seams::default();
    for f in funcs {
        for b in &f.blocks {
            if let Terminator::ReturnCallIndirect { .. } = b.term {
                s.has_opaque_call = true;
            }
            for inst in &b.insts {
                match inst {
                    // Every live op (`crate::served_contract`) drives executor children,
                    // scheduler-driven. Every other INSTANTIATOR op traps `CapFault`, so it is no
                    // seam.
                    Inst::CapCall(c)
                        if c.type_id == cap_id::INSTANTIATOR
                            && crate::served_contract(cap_id::INSTANTIATOR, c.op).is_some() =>
                    {
                        s.has_instantiate = true
                    }
                    Inst::CapCall(c) if c.type_id == cap_id::INSTANTIATOR => {}
                    // I38's **timed** `svc.wait` (op 10 with the optional timeout arg) needs
                    // the scheduler's deadline machinery — oracle-only; veto like a park seam
                    // so both fast backends decline the module.
                    Inst::CapCall(c)
                        if c.type_id == CAP_SELF_TYPE_ID && c.op == 10 && !c.args.is_empty() =>
                    {
                        s.has_svc = true;
                        s.has_park_seam = true;
                        s.has_timed_wait = true;
                    }
                    // §3.6 service points (I36 slice 1): svc.poll/svc.wait sites — natively
                    // servable only when nothing in the module could park a handler (below).
                    Inst::CapCall(c) if c.type_id == CAP_SELF_TYPE_ID && matches!(c.op, 9 | 10) => {
                        s.has_svc = true
                    }
                    // FORK.md §9 — `clone_caller` (11) / `reap` (12): the fork servicer
                    // primitives, the `has_fork` seam. The bytecode engine services both natively,
                    // the Cranelift JIT declines them, and the tree-walker serves no call inline
                    // into a domain that has them.
                    Inst::CapCall(c)
                        if c.type_id == CAP_SELF_TYPE_ID && matches!(c.op, 11 | 12) =>
                    {
                        s.has_fork = true
                    }
                    // A blocking stream `read` (type 0 op 0) can stdin-park, and an import call
                    // can be *bound* to one at spawn — either inside a handler would need the
                    // tree-walker's FIBER_PARKED (completed-but-not-replied) machinery.
                    Inst::CapCall(c)
                        if (c.type_id, c.op) == (cap_id::STREAM, 0)
                            || c.type_id == CAP_IMPORT_TYPE_ID =>
                    {
                        s.has_park_seam = true
                    }
                    Inst::CallImport(_) | Inst::SetJmp { .. } | Inst::LongJmp { .. } => {
                        s.has_park_seam = true
                    }
                    Inst::ContNew { .. }
                    | Inst::ContResume { .. } // I48: `block` flag is advisory here
                    | Inst::Suspend { .. } => s.has_fiber = true,
                    Inst::ThreadSpawn { .. }
                    | Inst::ThreadJoin { .. }
                    | Inst::MemoryWait { .. }
                    | Inst::MemoryNotify { .. } => s.has_thread = true,
                    // #1936 — a post's `signal` is a notify: it stops for its driver, which wakes
                    // the word's waiters.
                    Inst::CapCall(c) if (c.type_id, c.op) == (cap_id::SHARED_REGION, 7) => {
                        s.has_thread = true
                    }
                    Inst::GcRoots { .. } => s.has_gc = true,
                    Inst::CallIndirect(_) | Inst::CallImportDyn(_) => s.has_opaque_call = true,
                    Inst::CapCall(c) if !iface_never_parks(c.type_id) => s.has_opaque_call = true,
                    _ => {}
                }
            }
        }
    }
    s
}

/// #2742 — whether serve handler `func` of `funcs` can never park, whatever runs it: nothing it
/// reaches ([`crate::reachable_in`]) holds a park seam, a spawn or join, a call whose callee is
/// unknown until it runs, a fiber or thread op, gc roots, a service point or a fork primitive. It
/// runs to completion or traps. The JIT runs a direct handoff of such a handler on the caller's
/// own stack. An out-of-range `func` answers `false`, the safe answer.
pub fn handler_never_parks(funcs: &[Func], func: FuncIdx) -> bool {
    let Ok(reach) = crate::reachable_in(funcs, [func]) else {
        return false;
    };
    let s = scan_seams(funcs.iter().zip(reach).filter_map(|(f, r)| r.then_some(f)));
    !(s.has_park_seam
        || s.has_instantiate
        || s.has_opaque_call
        || s.has_fiber
        || s.has_thread
        || s.has_gc
        || s.has_svc
        || s.has_fork)
}

/// #2958 — a §12 op an engine's native code makes, for the scheduling core it runs on to apply as an
/// interpreted op's stop (`temen_interp::bytecode::EmittedCall::park`): the Cranelift JIT's, when its
/// vCPUs run as executor 2's tasks. Addresses are window-relative, as the guest wrote them.
///
/// #3078 slice 3 — and two a `cont.resume` of a kernel-parked **fiber** makes: its claim, or a block of
/// its task on it, through the rules ([`NativeStop::FiberClaim`], [`NativeStop::FiberBlock`]). Each
/// names the fiber's incarnation, its slot and generation, so a stale handle to a recycled slot never
/// reaches another fiber's park ([`ParkWord`]). The park itself, which carries the slot's word, is a
/// [`FiberWait`] of its own, so a stop stays `Copy` and as small as it was: every §12 op makes one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeStop {
    Spawn {
        func: u32,
        arg0: i64,
        arg1: i64,
    },
    Join {
        handle: i32,
    },
    /// `timeout` in ns, negative for none; `wide` is a 64-bit wait.
    Wait {
        addr: u64,
        expected: u64,
        wide: bool,
        timeout: i64,
    },
    Notify {
        addr: u64,
        count: i32,
    },
    /// Claim fiber `slot` (generation `gen`), kernel-parked: `[status, CLAIMED]` with its wait's
    /// status once it can be claimed (it is the claimant's to switch into), `[0, PARKED]` while it is
    /// still parked, `[0, NOT_PARKED]` when no park of that incarnation is filed.
    FiberClaim {
        slot: u32,
        gen: u32,
    },
    /// Block the task until fiber `slot` (generation `gen`), kernel-parked, can be claimed: answers
    /// at once when it already can, or when no park of that incarnation is filed.
    FiberBlock {
        slot: u32,
        gen: u32,
    },
}

/// #3078 slice 3 — the futex wait native fiber `slot` (generation `gen`) made before it switched out,
/// which its resumer files with the rules as the fiber's park: its record in the run's registry, its
/// entry in the wait index, and `word`, the slot's parked word, set for pollers. The wait's operands
/// are [`NativeStop::Wait`]'s.
#[derive(Clone)]
pub struct FiberWait {
    pub slot: u32,
    pub gen: u32,
    pub addr: u64,
    pub expected: u64,
    pub wide: bool,
    pub timeout: i64,
    pub word: ParkRef,
}

/// #3078 slice 3 — what holds a native fiber's parked word: its slot, which keeps the word inline
/// beside the slot's own claim word, so a poll reads one line it reads anyway. The rules hold the
/// slot through [`ParkRef`] while the park is filed, and write the word through it.
pub trait ParkSite: Send + Sync {
    fn park_word(&self) -> &ParkWord;
}

/// A word on its own is its own site: the models' and the unit rows'.
impl ParkSite for ParkWord {
    fn park_word(&self) -> &ParkWord {
        self
    }
}

/// The rules' hold of a parked word's site ([`ParkSite`]).
pub type ParkRef = alloc::sync::Arc<dyn ParkSite>;

/// [`NativeStop::FiberClaim`]'s answer in `[1]`: the claim took the park, and `[0]` is its status.
pub const FIBER_CLAIMED: i64 = 1;
/// [`NativeStop::FiberClaim`]'s answer in `[1]`: the fiber is still parked.
pub const FIBER_STILL_PARKED: i64 = 0;
/// [`NativeStop::FiberClaim`]'s answer in `[1]`: no park of the named incarnation is filed (it was
/// claimed by another, or the handle is stale).
pub const FIBER_NOT_PARKED: i64 = 2;

#[cfg(not(feature = "loom-park"))]
use core::sync::atomic::{AtomicU64, Ordering};
#[cfg(feature = "loom-park")]
use loom::sync::atomic::{AtomicU64, Ordering};

/// #3078 slice 3 — a native fiber's **parked word**: whether its kernel park can be claimed, read by
/// a `cont.resume` without the run lock. The rules write it, only under the run lock: as they file
/// the park ([`Self::park`]), as a notify or a timeout wakes it ([`Self::wake`]), and as a claim takes
/// it ([`Self::idle`]). A poller reads it once ([`Self::poll`]) and takes the run lock only for a park
/// it can claim.
///
/// It names the fiber's **incarnation**: `word` is `generation << 2 | state`, so a handle whose slot
/// has been recycled to another fiber never reads that fiber's park as its own. A finite deadline
/// sits beside it, stored before the word's release, and is read only for a timed park, so an untimed
/// park's poll reads no clock.
///
/// A stale read is legal either way. A pending read of a park since woken answers one transient
/// `FIBER_PARKED`, as the oracle's register-then-recheck may. An idle read of a park since filed sends
/// the poller to the slot's own claim, which a kernel-parked slot never grants (it is never published),
/// and so answers as a claim of a running fiber does.
#[derive(Debug)]
pub struct ParkWord {
    word: AtomicU64,
    /// The park's real deadline on [`wall_now`]'s clock, while the word says it is timed.
    deadline: AtomicU64,
}

/// What a poller does with a fiber's park ([`ParkWord::poll`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParkPoll {
    /// No park of this incarnation is filed: the slot's own claim decides.
    Idle,
    /// Still parked: `FIBER_PARKED`, no lock.
    Parked,
    /// Woken, or past its deadline: claim it through the rules.
    Claim,
}

const IDLE: u64 = 0;
const UNTIMED: u64 = 1;
const TIMED: u64 = 2;
const WOKEN: u64 = 3;
const STATE: u64 = 3;

impl Default for ParkWord {
    fn default() -> ParkWord {
        ParkWord::new()
    }
}

impl ParkWord {
    pub fn new() -> ParkWord {
        ParkWord {
            word: AtomicU64::new(IDLE),
            deadline: AtomicU64::new(0),
        }
    }

    /// The park of incarnation `gen` is filed (under the run lock): woken at once (the park-time
    /// re-check fired), or pending until `deadline`, if any.
    pub fn park(&self, gen: u64, woken: bool, deadline: Option<u64>) {
        let state = match (woken, deadline) {
            (true, _) => WOKEN,
            (false, None) => UNTIMED,
            (false, Some(d)) => {
                // Release: a poller that reads this deadline then sees every earlier store to the
                // word, so its re-read tells an earlier incarnation's park from this one's.
                self.deadline.store(d, Ordering::Release);
                TIMED
            }
        };
        self.word.store(gen << 2 | state, Ordering::Release);
    }

    /// The filed park is woken (under the run lock): a notify took it, or its deadline passed.
    pub fn wake(&self) {
        let w = self.word.load(Ordering::Relaxed);
        debug_assert_ne!(w & STATE, IDLE, "only a filed park is woken");
        self.word.store(w | WOKEN, Ordering::Release);
    }

    /// The park is claimed (under the run lock): the word names none.
    pub fn idle(&self) {
        self.word.store(IDLE, Ordering::Release);
    }

    /// Whether the word names no park: what a slot holds when no kernel park of it is filed.
    pub fn is_idle(&self) -> bool {
        self.word.load(Ordering::Acquire) & STATE == IDLE
    }

    /// What a `cont.resume` of incarnation `gen` does with its park, `now` read only for a timed one.
    ///
    /// A timed park's deadline is a second word, which a later incarnation of the slot may have
    /// written by the time it is read: so the word is read again after it, and a park that changed
    /// meanwhile is not this incarnation's to answer (`Idle`: the slot's own claim decides).
    pub fn poll(&self, gen: u64, now: impl FnOnce() -> u64) -> ParkPoll {
        self.poll_checked(gen, now, true)
    }

    /// [`Self::poll`] without its re-read of the word: the model's broken variant of it.
    #[cfg(feature = "loom-park")]
    #[doc(hidden)]
    pub fn poll_without_recheck(&self, gen: u64, now: impl FnOnce() -> u64) -> ParkPoll {
        self.poll_checked(gen, now, false)
    }

    fn poll_checked(&self, gen: u64, now: impl FnOnce() -> u64, recheck: bool) -> ParkPoll {
        let w = self.word.load(Ordering::Acquire);
        if w & STATE == IDLE || w >> 2 != gen {
            return ParkPoll::Idle;
        }
        match w & STATE {
            WOKEN => ParkPoll::Claim,
            TIMED => {
                // Acquire, pairing with the deadline's release ([`Self::park`]).
                let d = self.deadline.load(Ordering::Acquire);
                if recheck && self.word.load(Ordering::Acquire) != w {
                    return ParkPoll::Idle;
                }
                match now() >= d {
                    true => ParkPoll::Claim,
                    false => ParkPoll::Parked,
                }
            }
            _ => ParkPoll::Parked,
        }
    }
}

/// #3078 slice 3 — the clock a kernel fiber park's real deadline is on: nanoseconds since the
/// process first read it. The bytecode engine's waits read it, and the Cranelift JIT's poll of a
/// parked fiber reads the same one ([`ParkWord::poll`]), so one deadline means one instant.
#[cfg(feature = "clock")]
pub fn wall_now() -> u64 {
    use std::time::Instant;
    static BASE: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    BASE.get_or_init(Instant::now).elapsed().as_nanos() as u64
}
