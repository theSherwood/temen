//! Core kernel identifiers, structural [`TypeId`], [`Trap`], [`ContextKind`]/[`ContextState`],
//! [`PollFlags`], [`FuelCell`], [`Step`], [`KernelOps`], and [`LifecycleHooks`].

use temen_ir::{trap_code, FuncType, ValType};

/// Opaque identifier of a [`crate::Domain`] within a runtime instance.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct DomainId(pub u32);

/// Opaque identifier of a runnable context (`Root`, `Thread`, `GuestFiber`, or `ServeHandler`)
/// within a [`crate::Domain`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct ContextId(pub u32);

/// Opaque identifier of a Carrier (an OS worker thread or cooperative driver pump).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct CarrierId(pub u32);

/// Opaque identifier of a kernel waitable object (thread join, child join, cap I/O, offer reply,
/// pager request, doorbell).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct WaitableId(pub u64);

/// Canonical identity of a linear memory address space (a domain's anonymous window or a
/// `SharedRegion`), so cross-domain futex keys over aliased regions match identically.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct MemSpaceId(pub u64);

/// Identifier of an inbound cross-domain offer call served by a `ServeHandler` context.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct CallId(pub u64);

/// Canonical 64-bit structural hash of a [`FuncType`] (Invariant 10).
///
/// Identical across independently compiled modules and Stage 4 compile-once PIC shared libraries,
/// so indirect calls (`call.dyn` / `return_call.dyn`) and fiber entries check signatures with a
/// single integer comparison without a global nominal type registry.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TypeId(pub u64);

impl TypeId {
    /// Deterministic 64-bit structural hash of `ft` (FNV-1a with splitmix64 finalizer).
    pub fn of_func_type(ft: &FuncType) -> TypeId {
        Self::of_sig(&ft.params, &ft.results)
    }

    /// Deterministic 64-bit structural hash of a `(params, results)` signature.
    pub fn of_sig(params: &[ValType], results: &[ValType]) -> TypeId {
        const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

        let mix = |mut h: u64, b: u8| -> u64 {
            h ^= u64::from(b);
            h.wrapping_mul(FNV_PRIME)
        };
        let mix_u64 = |mut h: u64, v: u64| -> u64 {
            for byte in v.to_le_bytes() {
                h = mix(h, byte);
            }
            h
        };
        let val_tag = |t: ValType| -> u8 {
            match t {
                ValType::I32 => 1,
                ValType::I64 => 2,
                ValType::F32 => 3,
                ValType::F64 => 4,
                ValType::Ref => 5,
                ValType::Cap => 6,
                ValType::V128 => 7,
            }
        };

        let mut h = FNV_OFFSET;
        h = mix_u64(h, params.len() as u64);
        for &p in params {
            h = mix(h, val_tag(p));
        }
        h = mix(h, 0xFF); // domain separator between params and results
        h = mix_u64(h, results.len() as u64);
        for &r in results {
            h = mix(h, val_tag(r));
        }
        // Splitmix64 avalanche finalizer for uniform bit distribution.
        h ^= h >> 30;
        h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        h ^= h >> 27;
        h = h.wrapping_mul(0x94d0_49bb_1331_11eb);
        h ^= h >> 31;
        TypeId(h)
    }

    /// Canonical structural `TypeId` of a guest fiber entry: `(i64, i64) -> (i64)`.
    pub fn fiber_entry() -> TypeId {
        Self::of_sig(&[ValType::I64, ValType::I64], &[ValType::I64])
    }

    /// Canonical structural `TypeId` of a spawned thread entry: `(i64) -> (i64)`.
    pub fn thread_entry() -> TypeId {
        Self::of_sig(&[ValType::I64], &[ValType::I64])
    }
}

/// Guest-observable trap / termination reasons (aligned 1:1 with `temen_ir::trap_code`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Trap {
    OutOfFuel,
    DivByZero,
    IntOverflow,
    MemoryFault,
    StackOverflow,
    IndirectCallType,
    Unreachable,
    BadConversion,
    CapFault,
    Exit(i32),
    FiberFault,
    ThreadFault,
    Malformed,
}

impl Trap {
    /// Stable name matching the existing trap taxonomy.
    pub const fn name(self) -> &'static str {
        match self {
            Trap::OutOfFuel => "OutOfFuel",
            Trap::DivByZero => "DivByZero",
            Trap::IntOverflow => "IntOverflow",
            Trap::MemoryFault => "MemoryFault",
            Trap::StackOverflow => "StackOverflow",
            Trap::IndirectCallType => "IndirectCallType",
            Trap::Unreachable => "Unreachable",
            Trap::BadConversion => "BadConversion",
            Trap::CapFault => "CapFault",
            Trap::Exit(_) => "Exit",
            Trap::FiberFault => "FiberFault",
            Trap::ThreadFault => "ThreadFault",
            Trap::Malformed => "Malformed",
        }
    }

    /// Shared wire code from [`temen_ir::trap_code`].
    pub const fn code(self) -> i64 {
        match self {
            Trap::DivByZero => trap_code::DIV_BY_ZERO,
            Trap::IntOverflow => trap_code::INT_OVERFLOW,
            Trap::BadConversion => trap_code::BAD_CONVERSION,
            Trap::Unreachable => trap_code::UNREACHABLE,
            Trap::IndirectCallType => trap_code::INDIRECT_CALL_TYPE,
            Trap::CapFault => trap_code::CAP_FAULT,
            Trap::Exit(status) => trap_code::exit(status),
            Trap::MemoryFault => trap_code::MEMORY_FAULT,
            Trap::FiberFault => trap_code::FIBER_FAULT,
            Trap::ThreadFault => trap_code::THREAD_FAULT,
            Trap::OutOfFuel => trap_code::OUT_OF_FUEL,
            Trap::Malformed => trap_code::MALFORMED,
            Trap::StackOverflow => trap_code::STACK_OVERFLOW,
        }
    }

    /// Decode a wire code from [`temen_ir::trap_code`]. Returns `None` for 0 or JIT-internal codes.
    pub const fn from_code(code: i64) -> Option<Trap> {
        let tag = code & 0xFFFF_FFFF;
        match tag {
            trap_code::DIV_BY_ZERO => Some(Trap::DivByZero),
            trap_code::INT_OVERFLOW => Some(Trap::IntOverflow),
            trap_code::BAD_CONVERSION => Some(Trap::BadConversion),
            trap_code::UNREACHABLE => Some(Trap::Unreachable),
            trap_code::INDIRECT_CALL_TYPE => Some(Trap::IndirectCallType),
            trap_code::CAP_FAULT => Some(Trap::CapFault),
            trap_code::EXIT => Some(Trap::Exit((code >> 32) as i32)),
            trap_code::MEMORY_FAULT => Some(Trap::MemoryFault),
            trap_code::FIBER_FAULT => Some(Trap::FiberFault),
            trap_code::THREAD_FAULT => Some(Trap::ThreadFault),
            trap_code::OUT_OF_FUEL => Some(Trap::OutOfFuel),
            trap_code::MALFORMED => Some(Trap::Malformed),
            trap_code::STACK_OVERFLOW => Some(Trap::StackOverflow),
            _ => None,
        }
    }
}

/// Which kind of runnable stack a [`Context`] represents (retiring separate `VTask`,
/// `FiberState`, `HandlerTask`, and `HeldSite` taxonomies).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ContextKind {
    /// The domain's entry computation (`main` / `_start`).
    Root,
    /// A concurrent vCPU created by `thread.spawn`.
    Thread,
    /// A cooperative guest fiber created by `cont.new` (migratable across Carriers).
    GuestFiber,
    /// An inbound cross-domain offer handler executing in this domain's world.
    ServeHandler(CallId),
}

/// Unified lifecycle state of any [`Context`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ContextState {
    Runnable,
    Running {
        carrier: CarrierId,
    },
    /// A `GuestFiber` suspended via `suspend` (awaiting `cont.resume`).
    Suspended,
    /// Parked in [`crate::WaitIndex`] on a futex or kernel object.
    Parked {
        key: WaitKey,
        deadline: Option<u64>,
        interruptible: bool,
    },
    /// Paused by `DomainControl::set_paused(true)` or a freeze quiesce.
    Paused,
    /// Terminal completion of this context.
    Done,
}

/// Terminal outcome of a [`Context`].
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ContextExit {
    Returned(Vec<i64>),
    Trapped(Trap),
    Exited(i32),
}

impl ContextExit {
    /// Whether this exit terminates the enclosing domain (Invariant 6: any trap or `exit`).
    pub fn is_domain_terminal(&self) -> bool {
        matches!(self, ContextExit::Trapped(_) | ContextExit::Exited(_))
    }

    /// Convert into a `Trap` if this exit did not return normally.
    pub fn as_trap(&self) -> Option<Trap> {
        match self {
            ContextExit::Returned(_) => None,
            ContextExit::Trapped(t) => Some(*t),
            ContextExit::Exited(code) => Some(Trap::Exit(*code)),
        }
    }
}

/// Unified safepoint poll word bits (lives in [`crate::Domain`]'s shared `AtomicU64`, pointed to
/// by `VmCtx.domain_poll` at offset 0 — red-team refinement #2).
///
/// Zero on the steady-state hot path: a single relaxed load + branch-not-zero covers domain end,
/// epoch kill, job-control/debugger pause, freeze quiesce, and async upcall delivery.
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub struct PollFlags(pub u64);

impl PollFlags {
    pub const EMPTY: PollFlags = PollFlags(0);
    /// Domain ended (root returned, trap, or `exit`).
    pub const ENDED: u64 = 1 << 0;
    /// External kill / epoch interrupt.
    pub const KILL: u64 = 1 << 1;
    /// Paused by `DomainControl::set_paused(true)` (job control or debugger).
    pub const PAUSED: u64 = 1 << 2;
    /// Durable freeze quiesce requested.
    pub const FREEZE: u64 = 1 << 3;
    /// Personality armed an async upcall via `DomainControl::arm_upcall()`.
    pub const UPCALL: u64 = 1 << 4;

    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    #[inline]
    pub const fn contains(self, bit: u64) -> bool {
        (self.0 & bit) != 0
    }
}

/// Carrier-owned fuel counter (Decision 30).
///
/// Fuel lives on the **Carrier**, never per-`Context`. Mounting any `Context` points execution at
/// the Carrier's `FuelCell` (drawn in chunks from the context's `Domain` budget node), and unmounts
/// refund the unburned remainder. This keeps IR-anchored safepoint metering (Invariant 9)
/// identical across `Bytecode`, `Cranelift`, and `OpOracle` while eliminating per-fiber fuel
/// stranding.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FuelCell {
    /// Remaining fuel in the current chunk (checked at IR-anchored safepoints).
    pub remaining: u64,
    /// Whether this cell is actively metered (`false` = unlimited, no chunk draws).
    pub metered: bool,
}

impl FuelCell {
    /// Create a metered fuel cell initialized with `remaining` units.
    #[inline]
    pub const fn metered(remaining: u64) -> FuelCell {
        FuelCell {
            remaining,
            metered: true,
        }
    }

    /// Create an unmetered fuel cell (infinite fuel).
    #[inline]
    pub const fn unlimited() -> FuelCell {
        FuelCell {
            remaining: u64::MAX,
            metered: false,
        }
    }

    /// Charge one IR-anchored safepoint (entry, taken back-edge, or `cont.resume` — Invariant 9).
    /// Returns `true` if admitted, `false` if this chunk is exhausted (`OutOfFuel` unless refilled).
    #[inline]
    pub fn charge_one(&mut self) -> bool {
        if !self.metered {
            return true;
        }
        if self.remaining == 0 {
            return false;
        }
        self.remaining -= 1;
        true
    }

    /// Drain and return any unburned fuel (for refunding back to the domain's budget node).
    #[inline]
    pub fn take_remaining(&mut self) -> u64 {
        if !self.metered {
            return 0;
        }
        let left = self.remaining;
        self.remaining = 0;
        left
    }
}

/// Unified key for all blocking waits in [`crate::WaitIndex`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum WaitKey {
    /// Shared-memory futex keyed on canonical `(MemSpaceId, byte_offset)`.
    Futex { space: MemSpaceId, offset: u64 },
    /// Kernel waitable object (thread join, child join, cap I/O, offer reply, pager, doorbell).
    Object(WaitableId),
}

/// Why a parked context was woken from [`crate::WaitIndex`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WakeReason {
    /// Notified (`memory.notify`) or waitable object became ready.
    Ready,
    /// Guest-stated deadline elapsed.
    Timeout,
    /// Interrupted by `DomainControl::interrupt_parks()` (`-EINTR`).
    Interrupted,
    /// Target domain ended or capability was revoked (`-EPIPE` / `-ECANCELED`).
    Revoked,
    /// `DomainControl::set_paused(true)` — context should transition to `Paused`.
    Paused,
    /// Freeze quiesce requested — context should unwind or snapshot.
    Freeze,
}

/// Outcome of attempting to park in [`KernelOps::futex_wait`] or [`KernelOps::park_object`].
///
/// Red-team refinement #3: makes non-blocking guest-fiber parks (`FIBER_PARKED`, status 2)
/// explicit in the kernel contract so neither native thunks nor cooperative Carriers ever block an
/// OS thread on a `GuestFiber`'s wait.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ParkDecision {
    /// Did not park (e.g., futex word was not equal -> `1`, or object already signaled -> `0`).
    Immediate(i64),
    /// Context owns its Carrier (`Root` / `Thread` / `ServeHandler`) and blocked until woken.
    Blocked(WakeReason),
    /// Context is a `GuestFiber`: its park is registered in `WaitIndex`; the Carrier must
    /// immediately return `FIBER_PARKED` (status 2) to the resuming caller without blocking.
    /// On a later resume (potentially on another Carrier), the fiber re-executes per Invariant 7.
    YieldFiber,
}

/// Why a stackless engine yielded back to its Carrier loop.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum YieldReason {
    /// Cooperative quantum exhausted (round-robin handoff to sibling contexts).
    Quantum,
    /// Non-zero [`PollFlags`] observed at a safepoint.
    Poll(PollFlags),
    /// Fuel chunk hit zero; Carrier should attempt to refill from the domain's budget node
    /// (and trap `OutOfFuel` if the budget is empty).
    FuelRefill,
}

/// Maximum scalar argument count for a capability call (matches `temen_ir::CAP_MAX_ARITY`).
pub const MAX_CAP_ARGS: usize = temen_ir::CAP_MAX_ARITY;

/// Standard guest-visible `cont.resume` status codes (`DESIGN.md` §12, `#2667`).
pub const FIBER_SUSPENDED: i32 = 0;
pub const FIBER_RETURNED: i32 = 1;
pub const FIBER_PARKED: i32 = 3;
/// `-EBADF`: returned by `cont.resume` / `thread.join` on a spent handle (Invariant 5, `#2667`).
pub const ERRNO_EBADF: i32 = -9;

/// Standard guest-visible `memory.wait` status codes (`DESIGN.md` §12).
pub const WAIT_OK: i32 = 0;
pub const WAIT_NOT_EQUAL: i32 = 1;
pub const WAIT_TIMED_OUT: i32 = 2;

/// Returned by stackless execution engines (`temen-bytecode`, `OpOracle`) to the Carrier loop.
///
/// Stackless engines never block OS threads or mutate global runtime tables directly — they
/// perform pure computation up to a boundary and return a `Step` describing the kernel action.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Step {
    /// Context finished (returned values, trapped, or called exit).
    Finished(ContextExit),
    /// Context hit a safepoint poll, fuel refill, or cooperative quantum boundary.
    Yield(YieldReason),
    /// `memory.wait`: compare the 32- or 64-bit word at `addr` against `expected` and park until
    /// notified or `timeout_ns` (`< 0` = infinite).
    FutexWait {
        addr: u64,
        expected: i64,
        width64: bool,
        timeout_ns: i64,
    },
    /// `memory.notify`: wake up to `count` waiters on `addr`.
    FutexNotify { addr: u64, count: u32 },
    /// `call.cap`: invoke capability `handle` (`type_id`, `op`) with `args[..nargs]`.
    CallCap {
        handle: i32,
        type_id: u32,
        op: u32,
        args: [i64; MAX_CAP_ARGS],
        nargs: u8,
        nresults: u8,
    },
    /// `cont.new`: allocate a new `GuestFiber` starting at `func_idx` with initial `arg0`.
    FiberNew { func_idx: u32, arg0: i64 },
    /// `cont.resume` / `cont.resume.block`: resume guest fiber `handle` delivering `val`.
    FiberResume { handle: i64, val: i64, block: bool },
    /// `suspend`: yield from the current `GuestFiber` back to its resumer delivering `val`.
    FiberSuspend { val: i64 },
    /// `thread.spawn`: spawn a new `Thread` context in this domain running `func_idx(arg0, arg1)`.
    ThreadSpawn { func_idx: u32, arg0: i64, arg1: i64 },
    /// `thread.join`: wait for sibling thread `handle` to finish and return `(status, value)`.
    ThreadJoin { handle: i32 },
    /// Generic kernel-object park (used when a capability or runtime primitive parks on an object).
    Park {
        key: WaitKey,
        deadline: Option<u64>,
        interruptible: bool,
    },
}

/// Outcome of a synchronous or parking capability call invoked via [`KernelOps::call_cap`].
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum CapOutcome {
    /// Completed synchronously with result words.
    Ready(Vec<i64>),
    /// Capability chose to park this context on `key` (re-executed on wake per Invariant 7).
    Park {
        key: WaitKey,
        deadline: Option<u64>,
        interruptible: bool,
    },
    /// Capability raised a trap (e.g., `CapFault` on forged handle, `Exit(code)`).
    Trap(Trap),
}

/// Action the kernel directs a context to take after observing non-zero [`PollFlags`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PollAction {
    /// False alarm or flag already handled — continue running.
    Continue,
    /// Transition to `Paused` and wait until unpaused.
    Pause,
    /// Quiesce for durable freeze.
    Freeze,
    /// Invoke a personality-armed async upcall at this safepoint.
    Upcall {
        func_idx: u32,
        arg: i64,
        alt_data_sp: u64,
    },
    /// Domain ended or killed — terminate this context immediately.
    Terminate(Trap),
}

/// Shared state-transition seam invoked by BOTH stackless Carrier loops (`temen-bytecode`,
/// `OpOracle`) and native runtime thunks (`temen-jit` via `VmCtx`).
///
/// One implementation in `temen-runtime` serves every engine (Invariant 15).
pub trait KernelOps: Send + Sync {
    /// Atomically compare-and-park on a futex word (`check_eq` runs after the Dekker `SeqCst`
    /// fence so no wake can be lost).
    fn futex_wait(
        &self,
        domain: DomainId,
        ctx: ContextId,
        key: WaitKey,
        deadline: Option<u64>,
        check_eq: &dyn Fn() -> bool,
    ) -> ParkDecision;

    /// Wake up to `count` FIFO waiters on `key`. Returns number of waiters woken.
    fn futex_notify(&self, key: WaitKey, count: u32) -> u32;

    /// Park `ctx` on a kernel waitable object `id`.
    fn park_object(
        &self,
        domain: DomainId,
        ctx: ContextId,
        id: WaitableId,
        deadline: Option<u64>,
        interruptible: bool,
    ) -> ParkDecision;

    /// Dispatch a capability call.
    fn call_cap(
        &self,
        domain: DomainId,
        ctx: ContextId,
        handle: i32,
        type_id: u32,
        op: u32,
        args: &[i64],
    ) -> CapOutcome;

    /// Interpret non-zero [`PollFlags`] observed at a safepoint.
    fn on_poll(&self, domain: DomainId, ctx: ContextId, flags: PollFlags) -> PollAction;
}

/// Four OS-neutral lifecycle hooks connecting a personality (e.g., `temen-posix`) to the kernel,
/// replacing the 21-method POSIX-shaped `SignalSource` trait (Invariant 4 / 12).
pub trait LifecycleHooks: Send + Sync {
    /// Armed by a personality when an async upcall is ready (e.g. Bash `SIGINT` trap).
    /// Returns `(funcref_slot, arg, alt_data_sp)` to invoke at the next safepoint.
    fn take_upcall(&self, _domain: DomainId) -> Option<(u32, i64, u64)> {
        None
    }

    /// Called when an async upcall returns normally or is unwound past by `longjmp`.
    fn upcall_finished(&self, _domain: DomainId) {}

    /// Whether an interrupted park on `domain` returns `-EINTR` (`false`) or auto-restarts
    /// (`true`, e.g., `SA_RESTART`).
    fn should_restart_park(&self, _domain: DomainId) -> bool {
        false
    }

    /// Notification that `domain` has reached its single terminal [`ContextExit`].
    fn on_domain_exit(&self, _domain: DomainId, _exit: &ContextExit) {}
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;

    #[test]
    fn type_id_is_structural_and_distinct() {
        let a = TypeId::of_sig(&[ValType::I32, ValType::I64], &[ValType::I64]);
        let b = TypeId::of_func_type(&FuncType {
            params: vec![ValType::I32, ValType::I64],
            results: vec![ValType::I64],
        });
        assert_eq!(a, b);
        // Domain separator distinguishes (i32, i64) -> () from (i32) -> (i64).
        let c = TypeId::of_sig(&[ValType::I32, ValType::I64], &[]);
        let d = TypeId::of_sig(&[ValType::I32], &[ValType::I64]);
        assert_ne!(c, d);
        assert_ne!(a, c);
        assert_ne!(TypeId::fiber_entry(), TypeId::thread_entry());
    }

    #[test]
    fn trap_wire_codes_round_trip() {
        let traps = [
            Trap::OutOfFuel,
            Trap::DivByZero,
            Trap::IntOverflow,
            Trap::MemoryFault,
            Trap::StackOverflow,
            Trap::IndirectCallType,
            Trap::Unreachable,
            Trap::BadConversion,
            Trap::CapFault,
            Trap::Exit(0),
            Trap::Exit(42),
            Trap::Exit(-1),
            Trap::FiberFault,
            Trap::ThreadFault,
            Trap::Malformed,
        ];
        for t in traps {
            assert_eq!(Trap::from_code(t.code()), Some(t), "round-trip for {t:?}");
        }
        assert_eq!(Trap::from_code(0), None);
        assert_eq!(Trap::from_code(trap_code::DOMAIN_DONE), None);
    }

    #[test]
    fn fuel_cell_charges_and_drains() {
        let mut c = FuelCell::metered(2);
        assert!(c.charge_one());
        assert_eq!(c.remaining, 1);
        assert!(c.charge_one());
        assert_eq!(c.remaining, 0);
        assert!(!c.charge_one());

        let mut u = FuelCell::unlimited();
        assert!(u.charge_one());
        assert_eq!(u.take_remaining(), 0);
    }
}
