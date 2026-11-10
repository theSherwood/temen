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
    /// escape), while the Cranelift routing still folds them ([`Seams::svc_park_veto`] keeps it).
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
    /// **Fork rides this veto for Cranelift** (`has_fork` is in the disjunction), so temen-run's JIT
    /// routing still folds a forking module to the oracle — the Cranelift fork slice is unbuilt
    /// (FORK.md §9.1). The **bytecode** engine, in contrast, services `clone_caller`/`reap` natively;
    /// its compile gate takes its fork-serving escape past this veto (a per-backend split, the one
    /// the two-predicate structure exists to allow — the bytecode gate and the serve qualification
    /// legitimately diverge on fork until Cranelift catches up).
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
                    // primitives. The bytecode engine services both natively, so they are the
                    // `has_fork` seam — folded for Cranelift (`svc_park_veto` keeps `has_fork`)
                    // but run natively on bytecode.
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
