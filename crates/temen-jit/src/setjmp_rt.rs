//! Host-side `setjmp`/`longjmp` runtime for the JIT — **Option B** (LLVM.md §"JIT `longjmp`").
//!
//! The JIT lowers [`Inst::SetJmp`]/[`Inst::LongJmp`] to libc `_setjmp`/`_longjmp` called **inline in
//! the guest JIT frame** — the very frame `longjmp` returns to — with the `jmp_buf` held **host-side**
//! in this per-instance table. The host `jmp_buf` holds the native stack pointer / return address /
//! callee-saved registers; it MUST NOT live in the guest window (leaking those defeats ASLR).
//!
//! **One rule on every engine (#2877).** A checkpoint is named by the **token** its `setjmp` writes
//! into the guest `jmp_buf`'s first 8 bytes (#1062: a memcpy'd buffer carries its identity), and it
//! lives exactly as long as the **activation** that took it. It dies when that activation returns or
//! tail-calls ([`rt_setjmp_leave`], which the JIT emits before every exit of a function that takes
//! checkpoints), when a `longjmp` unwinds past it ([`rt_longjmp`]), or when the guarded call it ran
//! under ends — normally or by a fault, which skips every frame's exit ([`Scope`]). A `longjmp` to
//! anything else raises [`TrapKind::SetjmpFault`] (the interpreters' `Trap::Malformed`) and never
//! jumps: not to a returned frame, not to an earlier run's, and not over the host frames between a
//! nested guarded call and an outer checkpoint.
//!
//! So the live checkpoints are a **stack**: each is taken in the current, deepest live activation,
//! and each dies before the frames above it do, so they are ordered by frame pointer (outermost
//! first; a frame's own checkpoints in the order it took them). The table is a flat `Vec`, popped
//! from the top.
//!
//! **The inline-call invariant.** `_setjmp` is emitted inline in the guest frame, never wrapped in a
//! Rust thunk that `setjmp`s and returns — that helper's frame would be gone by `longjmp` time. So
//! the `SetJmp` lowering is *two* calls: [`rt_setjmp_slot`] (which writes the token, records the
//! checkpoint and returns its host `jmp_buf`) then an inline `_setjmp(slot)`. `LongJmp` is *one*,
//! [`rt_longjmp`], which reads the token, checks the checkpoint and `_longjmp`s from its own frame,
//! which holds nothing to drop. The code never holds a host `jmp_buf` pointer it could forge (#2534).
//!
//! **One stack.** A module that mixes `setjmp` with fibers or threads is declined at compile time
//! (`CompiledModule::compile`), so an instance's checkpoints all live on the one native stack its
//! guarded calls run on, which the frame-pointer order relies on.
//!
//! Unix-only (the `setjmp_rt` cfg = `fiber_rt && unix`): `libc` is a unix-only dependency here and the
//! plain `_setjmp`/`_longjmp` symbols are unix; Windows `setjmp` is SEH-coupled (a follow-on).

#![cfg(setjmp_rt)]

use crate::TrapKind;
use std::cell::UnsafeCell;
use std::os::raw::{c_int, c_void};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};

extern "C" {
    /// libc `_setjmp`/`_longjmp` — the **non**-signal-mask variants (no `sigprocmask`
    /// save/restore), matching the interpreters' semantics. Real exported symbols in glibc/musl.
    /// The `SetJmp`/`LongJmp` sites call them *inline in the guest frame* (so `_setjmp` saves that
    /// frame's SP and `_longjmp` returns into it).
    fn _setjmp(env: *mut c_void) -> c_int;
    fn _longjmp(env: *mut c_void, val: c_int) -> !;
}

/// Address of libc `_setjmp`, which each `SetJmp` site calls inline (`HostFn::Setjmp`).
pub(crate) fn setjmp_addr() -> *const () {
    _setjmp as *const ()
}

/// The source of `setjmp` tokens: process-wide and monotonic, so a token an earlier run left in a
/// window never names a checkpoint a later one takes. Opaque to the guest — only equality matters.
static TOKEN: AtomicU64 = AtomicU64::new(1);

/// One host-owned libc `jmp_buf`, over-aligned and generously sized (glibc/musl `jmp_buf` ≈ 200 B; the
/// 512 B here covers every gated target with margin). Only `_setjmp`/`_longjmp` touch its bytes, through
/// the pointer [`Self::as_ptr`] hands out; `Box`ed so that address is **stable** while the table grows.
#[repr(C, align(16))]
struct JmpBufCell(UnsafeCell<[u8; 512]>);

impl JmpBufCell {
    fn boxed() -> Box<JmpBufCell> {
        Box::new(JmpBufCell(UnsafeCell::new([0u8; 512])))
    }
    fn as_ptr(&self) -> *mut c_void {
        self.0.get().cast()
    }
}

thread_local! {
    /// The `jmp_buf` a `setjmp` whose token store faulted saves into: its site's trap check bails
    /// before anything could jump to it.
    static SCRATCH: Box<JmpBufCell> = JmpBufCell::boxed();
}

/// A checkpoint: the token naming it, the frame pointer of the activation that took it, and the host
/// `jmp_buf` its inline `_setjmp` saved into.
struct Checkpoint {
    token: u64,
    fp: usize,
    env: Box<JmpBufCell>,
}

/// The table's locked state.
#[derive(Default)]
struct Checkpoints {
    /// The first [`Self::live`] are the live checkpoints, outermost frame first (frame pointers
    /// non-increasing); the rest are dead ones, whose cells the next checkpoints reuse.
    slots: Vec<Checkpoint>,
    live: usize,
    /// The first checkpoint the current guarded call took: the ones below it belong to calls that
    /// enclose it, across host frames no `longjmp` may cross ([`Scope`]).
    floor: usize,
}

impl Checkpoints {
    /// The live checkpoints.
    fn live(&self) -> &[Checkpoint] {
        &self.slots[..self.live]
    }

    /// Record a checkpoint and return its host `jmp_buf`.
    fn push(&mut self, token: u64, fp: usize) -> *mut c_void {
        if self.live == self.slots.len() {
            self.slots.push(Checkpoint {
                token,
                fp,
                env: JmpBufCell::boxed(),
            });
        } else {
            let c = &mut self.slots[self.live];
            (c.token, c.fp) = (token, fp);
        }
        self.live += 1;
        self.slots[self.live - 1].env.as_ptr()
    }

    /// Drop the live checkpoints from index `keep` up (never below the floor).
    fn truncate(&mut self, keep: usize) {
        self.live = self.live.min(keep.max(self.floor));
    }

    /// Drop the checkpoints taken at or below frame pointer `fp` (the leaving frame's and any deeper
    /// one's).
    fn leave(&mut self, fp: usize) {
        let deeper = self.live().iter().rev().take_while(|c| c.fp <= fp).count();
        self.truncate(self.live - deeper);
    }
}

/// Per-instance `setjmp` checkpoint table. Stood up only when the module uses `setjmp`
/// (`module_uses_setjmp`), one per instance; the instance's `SetJmp`/`LongJmp` sites and its guarded
/// calls find it through the `vmctx` (`VmCtx::setjmp`, temen#2389), and the `CompiledModule` owns it.
pub(crate) struct SetjmpRuntime {
    state: Mutex<Checkpoints>,
}

impl SetjmpRuntime {
    pub(crate) fn new() -> SetjmpRuntime {
        SetjmpRuntime {
            state: Mutex::new(Checkpoints::default()),
        }
    }

    fn lock(&self) -> MutexGuard<'_, Checkpoints> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// The table `vm`'s instance names, if its module uses `setjmp`.
///
/// # Safety
/// `vm.setjmp` is null or the instance's live [`SetjmpRuntime`].
unsafe fn runtime(vm: &crate::VmCtx) -> Option<&SetjmpRuntime> {
    (vm.setjmp as *const SetjmpRuntime).as_ref()
}

/// `rt_setjmp_slot(vm, fp, buf)` — take a checkpoint in the activation whose frame pointer is `fp`:
/// write a fresh token into the guest `jmp_buf` at window offset `buf` (confined, paged and probed as
/// a guest store, so it faults where the interpreters' token store does), record the checkpoint, and
/// return its host `jmp_buf` for the inline `_setjmp`. When the store faults it raises (the site's
/// trap check bails) and returns a scratch `jmp_buf` instead, recording nothing.
///
/// # Safety
/// `vm` is the calling instance's live context, whose code uses `setjmp`, so its table is set; `fp` is
/// the calling JIT frame's frame pointer.
pub(crate) unsafe extern "C" fn rt_setjmp_slot(
    vm: &crate::VmCtx,
    fp: u64,
    buf: u64,
) -> *mut c_void {
    let token = TOKEN.fetch_add(1, Ordering::Relaxed);
    // Before the lock: a store past the backed extent faults out of this frame.
    let Some(word) = crate::mem::guest_word(vm, buf, true) else {
        return SCRATCH.with(|s| s.as_ptr());
    };
    word.cast::<u64>().write_unaligned(token);
    let Some(rt) = runtime(vm) else {
        vm.raise(TrapKind::SetjmpFault as i64);
        return SCRATCH.with(|s| s.as_ptr());
    };
    let mut st = rt.lock();
    // A checkpoint deeper than this frame is dead (its exit already dropped it).
    st.leave((fp as usize).saturating_sub(1));
    st.push(token, fp as usize)
}

/// `rt_setjmp_leave(vm, fp)` — the activation whose frame pointer is `fp` is returning or tail-calling:
/// its checkpoints (and any deeper frame's) die. Emitted before each exit of a function that takes
/// checkpoints.
///
/// # Safety
/// As [`rt_setjmp_slot`].
pub(crate) unsafe extern "C" fn rt_setjmp_leave(vm: &crate::VmCtx, fp: u64) {
    if let Some(rt) = runtime(vm) {
        rt.lock().leave(fp as usize);
    }
}

/// `rt_longjmp(vm, fp, buf, val)` — read the token from the guest `jmp_buf` at window offset `buf`
/// (confined as a guest load) and `_longjmp(val)` to its checkpoint, unwinding the deeper frames'
/// checkpoints with them. The checkpoint must be live, taken under the current guarded call, and in
/// this frame (`fp`) or one that encloses it; otherwise — a forged token, a returned activation, an
/// earlier run's, an enclosing call's — raise [`TrapKind::SetjmpFault`] and return, and the JIT's
/// trap-propagate check bails. The lock is released before the jump, so this frame holds nothing
/// `_longjmp` would skip dropping.
///
/// # Safety
/// As [`rt_setjmp_slot`].
pub(crate) unsafe extern "C" fn rt_longjmp(vm: &crate::VmCtx, fp: u64, buf: u64, val: c_int) {
    let Some(word) = crate::mem::guest_word(vm, buf, false) else {
        return;
    };
    let token = word.cast::<u64>().read_unaligned();
    let env = runtime(vm).and_then(|rt| {
        let mut st = rt.lock();
        let i = st.floor
            + st.live()[st.floor..]
                .iter()
                .rposition(|c| c.token == token)?;
        let target = st.slots[i].fp;
        if target < fp as usize {
            return None; // a frame this one is not inside
        }
        st.leave(target.saturating_sub(1));
        Some(st.slots[i].env.as_ptr())
    });
    match env {
        Some(env) => _longjmp(env, val),
        None => vm.raise(TrapKind::SetjmpFault as i64),
    }
}

/// The checkpoints of one guarded call into JIT code (`mem::run_guarded_range`, the one entry every
/// run, nested `invoke_extra` and fiber resume takes): on entry it raises the table's floor, so a
/// `longjmp` cannot reach a checkpoint an enclosing call took (that would jump over the host frames
/// between them); on drop — after a normal return or a fault, which skips every frame's exit — it
/// drops the checkpoints the call took and restores the floor. A no-op for a module without `setjmp`.
pub(crate) struct Scope<'a> {
    rt: Option<&'a SetjmpRuntime>,
    floor: usize,
}

impl Scope<'_> {
    /// # Safety
    /// `vm` is null or a live context whose `setjmp` field is null or its instance's live table, and
    /// both outlive the scope.
    pub(crate) unsafe fn enter(vm: *const crate::VmCtx) -> Self {
        let rt = vm.as_ref().and_then(|vm| runtime(vm));
        let floor = rt.map_or(0, |rt| {
            let mut st = rt.lock();
            let outer = st.floor;
            st.floor = st.live;
            outer
        });
        Scope { rt, floor }
    }
}

impl Drop for Scope<'_> {
    fn drop(&mut self) {
        if let Some(rt) = self.rt {
            let mut st = rt.lock();
            let mine = st.floor;
            st.truncate(mine);
            st.floor = self.floor;
        }
    }
}
