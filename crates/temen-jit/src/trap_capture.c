/* Cross-platform trap-time backtrace capture (DEBUGGING.md §5 W3), compiled on every target that has
 * the JIT trap-backtrace feature (unix + windows). The platform-specific *trap detection* lives
 * elsewhere — the unix SIGSEGV/SIGBUS handler in `trap_shim.c`, the windows Vectored Exception Handler
 * in `mem.rs` — but the capture *state* and the frame-pointer walk are shared here so a div-by-zero /
 * `unreachable` / `OutOfFuel` / indirect-call-type trap is captured identically on both platforms.
 *
 * The host reads the captured `(pc, return addresses)` via `temen_take_trap_frame` and only *symbolizes*
 * them (pure arithmetic, no stack reads). Thread-local: the trap is attributed to the faulting worker,
 * and the host takes the capture before re-running anything on that thread.
 *
 * MSVC has no `__builtin_frame_address`/`__builtin_return_address`, so: the trapping frame pointer is
 * passed *in* (the JIT computes it with Cranelift `get_frame_pointer` and threads it as an argument),
 * and the trap-site return address comes from `_ReturnAddress()` (MSVC) / `__builtin_return_address(0)`
 * (GCC/Clang) — neither needs the helper to have its own frame pointer. */
#include <stdint.h>

#ifdef _MSC_VER
#include <intrin.h>
#pragma intrinsic(_ReturnAddress)
#define TEMEN_TLS __declspec(thread)
#define TEMEN_RETURN_ADDRESS() _ReturnAddress()
#else
#define TEMEN_TLS _Thread_local
#define TEMEN_RETURN_ADDRESS() __builtin_return_address(0)
#endif

#define TEMEN_TRAP_MAXFRAMES 64
static TEMEN_TLS volatile int g_trap_valid = 0;
static TEMEN_TLS uintptr_t g_trap_pc = 0;
static TEMEN_TLS uintptr_t g_trap_rets[TEMEN_TRAP_MAXFRAMES];
/* `volatile`: the walk publishes this as it goes, and the walk's recovery point (below) reads it
 * after a `siglongjmp`/`__except` out of the middle of the loop. */
static TEMEN_TLS volatile int g_trap_nrets = 0;

/* Per-fiber trap attribution (DEBUGGING.md §5 W3 / §23-D57). `g_current_fiber` is the guest handle of
 * the fiber executing on *this* OS thread right now, or `TEMEN_NO_FIBER` (root, no fiber). The fiber
 * runtime publishes it with stack discipline across the resume seam (`temen_set_current_fiber`), so under
 * work-stealing migration — where a fiber may resume on a different vCPU thread than it suspended on —
 * a trap is attributed to the fiber *running at the trap instant*, not inferred from the thread. The
 * capture functions copy it into `g_trap_fiber` (signal-safe: a plain TLS read), and the host reads it
 * back via `temen_take_trap_fiber`. */
#define TEMEN_NO_FIBER (-1)
static TEMEN_TLS int64_t g_current_fiber = TEMEN_NO_FIBER;
static TEMEN_TLS int64_t g_trap_fiber = TEMEN_NO_FIBER;

/* Publish the fiber now running on this thread; returns the previous value so the caller can restore it
 * when the resume returns (the same save/restore the durable shadow-SP swap uses). */
int64_t temen_set_current_fiber(int64_t handle) {
    int64_t prev = g_current_fiber;
    g_current_fiber = handle;
    return prev;
}


/* Walk the frame-pointer chain from `fp` toward the stack base, appending each frame's return address
 * to `g_trap_rets`. The JIT's `preserve_frame_pointers` gives every guest frame a
 * `{ saved_fp, ret_addr }` record: `*fp` is the caller's saved frame pointer, `*(fp+1)` the return
 * address. The walk moves *up* (increasing addresses, away from the low-address stack guard a fault
 * sits near) and its links are bounded — aligned, non-null, strictly-increasing, within a generous
 * span, and the frame cap — so a corrupt chain terminates instead of looping. The host stops at the
 * first return address that isn't guest code, so a few trailing host frames are harmless. Reads-only.
 *
 * **Those bounds constrain the arithmetic, not the mapping.** Nothing here establishes that a link is
 * mapped, and it cannot: the walk reads the guest's own stack, and the guest chooses what is in it.
 * The bad link is not even hypothetical — the JIT entry trampoline spills `mem_base` into a frame
 * slot, so at the depth where the walk steps out of guest frames it can read the *window base*, whose
 * first page is the `PROT_NONE` null guard. That value is non-null, aligned, and increasing, so every
 * test above passes and the deref faults. `temen_guarded_walk` is what makes that survivable; see it
 * for why the bound is not tightened instead. (#1487)
 *
 * `g_trap_nrets` is published after each append rather than returned at the end, so a walk cut short
 * by such a fault still hands back every frame it did collect. */
static void temen_walk_fp_chain(uintptr_t fp) {
    uintptr_t cur = fp;
    const uintptr_t start = fp;
    const uintptr_t span = 8u * 1024 * 1024; /* don't chase a corrupt chain off the stack */
    int n = g_trap_nrets;
    while (n < TEMEN_TRAP_MAXFRAMES && cur != 0 && (cur & (sizeof(uintptr_t) - 1)) == 0 &&
           cur >= start && cur - start < span) {
        uintptr_t next = *(uintptr_t *)cur;
        uintptr_t ret = *(uintptr_t *)(cur + sizeof(uintptr_t));
        g_trap_rets[n++] = ret;
        g_trap_nrets = n; /* publish as we go: a faulting link truncates the backtrace, never voids it */
        if (next <= cur) /* frame pointers grow toward the base; a non-increasing link is the end */
            break;
        cur = next;
    }
}

/* ---- the walk's own fault recovery (#1487) ---------------------------------------------------
 *
 * A fault *inside* the walk used to kill the host. On unix the walk runs from the SIGSEGV handler,
 * which has already disarmed itself by then, so the nested fault fell through to `temen_chain` →
 * `SIG_DFL` → `raise` — a recoverable guest `MemoryFault` became a dead host process. Since the guest
 * controls its own stack, it controls whether that happens: this is an availability break the guest
 * can reach, on the confinement *recovery* path whose entire job is to kill the guest and not the
 * host. And the walk is a debugging nicety; a fault while collecting a backtrace must at worst cost
 * the backtrace.
 *
 * So the walk gets a recovery point of its own, and the platform trap detector (the unix signal
 * handler in `trap_shim.c`, the windows VEH in `mem.rs`) asks `temen_walk_in_progress` *before* its
 * own range test and stands down: unix jumps back here, windows lets the fault fall through to the
 * `__except` below. Either way the capture stands with whatever the walk had published, and the
 * *original* guest trap goes on to be reported normally.
 *
 * Why this rather than bounding the walk to the thread's stack (`pthread_getattr_np` /
 * `GetCurrentThreadStackLimits`): a guest frame chain does not have to be on the thread stack. Fibers
 * switch stacks, and trap attribution across that seam is a feature here — `g_current_fiber` exists
 * precisely so a trap is attributed to the fiber running at the trap instant. Clamping to the thread
 * stack would silently truncate every backtrace taken on a fiber. A recovery point costs nothing and
 * is correct for *any* faulting address, which a bound never is. */
#if defined(_MSC_VER)
/* MSVC compiles this file natively on windows (the mingw cross-check doesn't build it at all — see
 * `build.rs`), so SEH is the idiom. `EXCEPTION_EXECUTE_HANDLER` spelled out to avoid <windows.h>. */
#define TEMEN_EXECUTE_HANDLER 1
#else
#include <setjmp.h>
static TEMEN_TLS sigjmp_buf g_walk_buf;
#endif

static TEMEN_TLS volatile int g_walking = 0;

/* Is *this thread* inside the frame-pointer walk right now? The trap detectors call this first. */
int temen_walk_in_progress(void) {
    return g_walking;
}

#ifndef _MSC_VER
/* Abandon a faulting walk: back to `temen_guarded_walk`, which returns to the detector as if the walk
 * had simply ended. Called from the signal handler; does not return. */
void temen_walk_abort(void) {
    g_walking = 0;
    siglongjmp(g_walk_buf, 1);
}
#endif

/* Walk from `fp`, surviving a fault on any link. */
static void temen_guarded_walk(uintptr_t fp) {
#if defined(_MSC_VER)
    __try {
        g_walking = 1;
        temen_walk_fp_chain(fp);
    } __except (TEMEN_EXECUTE_HANDLER) {
        /* a link pointed off a mapped page — keep what the walk published */
    }
#else
    if (sigsetjmp(g_walk_buf, 1) == 0) {
        g_walking = 1;
        temen_walk_fp_chain(fp);
    }
#endif
    g_walking = 0;
}

/* Store a **memory-fault** capture: the trap detector (unix signal handler / windows VEH) extracts the
 * faulting `(pc, fp)` from its platform context and calls this to walk + stash. `pc` is the exact
 * faulting instruction (the host symbolizes it directly). Async-signal-safe: stack reads + TLS writes.
 *
 * Everything but the frames is published *before* the walk, so a walk cut short by its own fault still
 * yields a valid capture — the faulting pc, the fiber, and the frames collected so far. */
void temen_store_trap_frame(uintptr_t pc, uintptr_t fp) {
    g_trap_pc = pc;
    g_trap_nrets = 0;
    g_trap_fiber = g_current_fiber;
    g_trap_valid = 1;
    temen_guarded_walk(fp);
}

/* Capture an **explicit-check** trap (§5 W3 Stage 2). The JIT calls this from a trap site (div-by-zero,
 * `unreachable`, `OutOfFuel`, indirect-call-type) *before* storing the trap kind and returning — those
 * returns unwind every guest frame, so the chain must be walked here, while it is live. `guest_fp` is
 * the trapping function's frame pointer (Cranelift `get_frame_pointer`, threaded in by the JIT); the
 * trap site is this helper's own return address (just past the `call`), recorded as `rets[0]`
 * (symbolized at `ret - 1`, like every caller) with `pc` left 0.
 *
 * Unlike the memory-fault path this runs in ordinary context, not a signal handler — but it walks the
 * same guest-controlled stack, so it takes the same guarded walk. */
void temen_capture_explicit_trap(uintptr_t guest_fp) {
    g_trap_pc = 0;
    g_trap_rets[0] = (uintptr_t)TEMEN_RETURN_ADDRESS();
    g_trap_nrets = 1;
    g_trap_fiber = g_current_fiber;
    g_trap_valid = 1;
    temen_guarded_walk(guest_fp);
}

/* Read and clear the captured trap stack (the host calls this after a guarded run reports a trap).
 * Fills `*pc` and up to `max` return addresses into `rets`; returns the number written, or -1 if
 * nothing was captured. */
int temen_take_trap_frame(uintptr_t *pc, uintptr_t *rets, int max) {
    if (!g_trap_valid)
        return -1;
    g_trap_valid = 0;
    *pc = g_trap_pc;
    int n = g_trap_nrets;
    if (n > max)
        n = max;
    for (int i = 0; i < n; i++)
        rets[i] = g_trap_rets[i];
    return n;
}

/* The guest fiber handle captured with the most recent trap (paired with `temen_take_trap_frame`), or
 * `TEMEN_NO_FIBER` when the root computation (no fiber) trapped. Not cleared — read it right after a
 * successful `temen_take_trap_frame`. */
int64_t temen_take_trap_fiber(void) {
    return g_trap_fiber;
}
