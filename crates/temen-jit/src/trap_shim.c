/* Detect-and-kill trap recovery for the JIT (DESIGN.md §4/§5), unix only.
 *
 * The guest window is bracketed by a PROT_NONE guard page (allocated in Rust); the masking
 * lowering confines every access to [0, size), so a fault in the guard region is a
 * width-overrun at the very top of the window or — defense-in-depth — a masking/elision
 * bug. We catch SIGSEGV/SIGBUS, and if the faulting address is inside the *currently armed*
 * window's guarded range we siglongjmp back out of the JIT call, reporting a memory fault to
 * the host instead of crashing it. Anything else chains to the previous disposition.
 *
 * setjmp/sigsetjmp are macros that need the compiler's `returns_twice` handling, so this
 * lives in C (calling them via raw FFI from Rust is unsound). The recovery state is
 * thread-local, so concurrent JIT runs on different threads are independent: the handler
 * runs on the faulting thread and reads that thread's state.
 */
/* Feature-test macros must precede every include. `_XOPEN_SOURCE` exposes the (macOS-deprecated)
 * ucontext routines + `ucontext_t` mcontext on Apple SDKs; `_GNU_SOURCE` gates glibc's REG_RIP/REG_RBP
 * in <sys/ucontext.h>. Both are harmless where unneeded. */
#define _XOPEN_SOURCE 700
#define _GNU_SOURCE
#include <setjmp.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <ucontext.h>
#include <unistd.h>

/* #3167 — the guarded calls open on this thread, as a chain: each `temen_run_guarded` frame holds its
 * scope (its recovery point and its window's fault range) and a link to the scope that was innermost
 * when it was entered, and `g_scope` names the innermost. Null means disarmed. A scope lives on the
 * stack its call runs on, so nothing is copied: entry links, exit unlinks.
 *
 * R-TLS (#3167): a thread-local's address computed before a possible stack switch must not be used
 * after it. `fn` may switch stacks, and the compiler may keep `&g_scope` in a callee-saved register
 * across the call (the probe at bf6d7aa showed `temen_run_guarded` doing exactly that with `g_buf`),
 * so every access that can follow `fn` goes through the `noipa` accessors below, which resolve the
 * thread-local inside themselves, on the thread that calls them. The handler runs on the faulting
 * thread with no switch inside it, so it reads `g_scope` directly. */
struct temen_scope {
    sigjmp_buf buf;
    uintptr_t lo;
    uintptr_t hi;
    struct temen_scope *parent;
};
static _Thread_local struct temen_scope *g_scope = NULL;

__attribute__((noinline, noipa)) static struct temen_scope *temen_scope_top(void) {
    return g_scope;
}

__attribute__((noinline, noipa)) static void temen_scope_set(struct temen_scope *s) {
    g_scope = s;
}

/* The trap-time backtrace capture *state* and the frame-pointer walk live in `trap_capture.c` (shared
 * with the windows VEH and the explicit-trap helper). The signal handler below extracts the faulting
 * `(pc, fp)` from the ucontext and hands them to `temen_store_trap_frame`, which walks the chain and
 * stashes it in a thread-local for the host to symbolize (DEBUGGING.md §5 W3). The walk must happen
 * here, while the guest stack is intact: a siglongjmp unwinds back onto the same stack the post-fault
 * host code then reuses, so the frames would be gone by the time the host could walk them. */
extern void temen_store_trap_frame(uintptr_t pc, uintptr_t fp);
/* The walk's own recovery (#1487). It reads the guest's stack, and the guest chooses what is in it, so
 * a planted frame link can steer it onto an unmapped page. `temen_walk_in_progress` says this thread is
 * mid-walk; `temen_walk_abort` returns to the walk's recovery point with the backtrace truncated. */
extern int temen_walk_in_progress(void);
extern void temen_walk_abort(void);

/* The faulting (pc, fp) from the signal ucontext; both 0 on an arch we don't decode. */
static void temen_fault_pc_fp(void *uc, uintptr_t *pc_out, uintptr_t *fp_out) {
    uintptr_t pc = 0, fp = 0;
#if defined(__linux__) && defined(__x86_64__)
    ucontext_t *c = (ucontext_t *)uc;
    pc = (uintptr_t)c->uc_mcontext.gregs[REG_RIP];
    fp = (uintptr_t)c->uc_mcontext.gregs[REG_RBP];
#elif defined(__linux__) && defined(__aarch64__)
    ucontext_t *c = (ucontext_t *)uc;
    pc = (uintptr_t)c->uc_mcontext.pc;
    fp = (uintptr_t)c->uc_mcontext.regs[29]; /* AAPCS64 frame pointer */
#elif defined(__APPLE__) && defined(__x86_64__)
    ucontext_t *c = (ucontext_t *)uc;
    pc = (uintptr_t)c->uc_mcontext->__ss.__rip;
    fp = (uintptr_t)c->uc_mcontext->__ss.__rbp;
#elif defined(__APPLE__) && defined(__aarch64__)
    ucontext_t *c = (ucontext_t *)uc;
    pc = (uintptr_t)c->uc_mcontext->__ss.__pc;
    fp = (uintptr_t)c->uc_mcontext->__ss.__fp;
#else
    (void)uc; /* an arch we don't decode: pc/fp stay 0 → the host yields an empty backtrace */
#endif
    *pc_out = pc;
    *fp_out = fp;
}

/* Memory-fault capture: extract the faulting (pc, fp) from the signal ucontext and hand off the walk.
 * Async-signal-safe: only the ucontext read + the (stack-reading, TLS-writing) store. */
static void temen_capture_frame(void *uc) {
    uintptr_t pc, fp;
    temen_fault_pc_fp(uc, &pc, &fp);
    temen_store_trap_frame(pc, fp); /* walk + stash in the shared (trap_capture.c) thread-local */
}

/* #2996 — which code faulted. A guarded fault is the guest's only when its pc is JIT-emitted code
 * (`code_ranges.rs`, loads only). `temen_fault_strict` says whether a guarded fault anywhere else is
 * fatal, which it is in a build with debug assertions; in a release build it is unwound as before. */
extern int temen_pc_is_jit(uintptr_t pc);
extern int temen_fault_strict(void);

/* Async-signal-safe: hex into a stack buffer, then write(2). */
static char *temen_hex(char *p, uintptr_t v) {
    *p++ = '0';
    *p++ = 'x';
    int started = 0;
    for (int shift = (int)(sizeof v * 8) - 4; shift >= 0; shift -= 4) {
        unsigned d = (unsigned)(v >> shift) & 0xf;
        if (d || started || shift == 0) {
            *p++ = (char)(d < 10 ? '0' + d : 'a' + d - 10);
            started = 1;
        }
    }
    return p;
}

static void temen_report_host_fault(int sig, uintptr_t pc, uintptr_t addr) {
    static const char head[] = "temen-jit: host code faulted on guest memory inside a guarded call: pc=";
    char buf[192];
    char *p = buf;
    memcpy(p, head, sizeof head - 1);
    p += sizeof head - 1;
    p = temen_hex(p, pc);
    memcpy(p, " addr=", 6);
    p += 6;
    p = temen_hex(p, addr);
    memcpy(p, " sig=", 5);
    p += 5;
    p = temen_hex(p, (uintptr_t)sig);
    *p++ = '\n';
    ssize_t n = write(2, buf, (size_t)(p - buf));
    (void)n;
}

/* Whether the guarded fault at `pc` is a host bug to die of (#2996): `pc` is decoded, is not JIT
 * code, and the build is strict. Reported here, before the caller chains to the default action. */
static int temen_host_fault(int sig, void *uc, uintptr_t addr) {
    uintptr_t pc, fp;
    temen_fault_pc_fp(uc, &pc, &fp);
    if (pc == 0 || temen_pc_is_jit(pc) || !temen_fault_strict())
        return 0;
    temen_report_host_fault(sig, pc, addr);
    return 1;
}

/* ---- #2996: a host access to the window that may fault --------------------------------------
 *
 * Some host functions the guest calls read or write its window where only the hardware knows whether
 * the page is there (a guest's own unmap or protect, the null guard, the uncommitted tail, a racing
 * vCPU). They run the access through `temen_try_call`, which gives it a recovery point of its own:
 * a fault on the window during it comes back here, and the call returns 0, so the caller raises the
 * guest's `MemoryFault` and returns normally, with its locks released by their own destructors. The
 * jump skips only `f`'s frame, which holds nothing to drop. A fault outside `[lo, hi)` (the window's
 * fault range) during the try is not recovered: it is a host bug. */
static _Thread_local sigjmp_buf g_try_buf;
static _Thread_local volatile int g_trying = 0;
static _Thread_local volatile uintptr_t g_try_lo = 0;
static _Thread_local volatile uintptr_t g_try_hi = 0;

int temen_try_call(void (*f)(void *), void *arg, uintptr_t lo, uintptr_t hi) {
    /* Not nested: a try runs one access and nothing that could start another. */
    g_try_lo = lo;
    g_try_hi = hi;
    /* No mask saved, as at the guarded entry (#2805): the handler runs with the mask the fault found. */
    if (sigsetjmp(g_try_buf, 0)) {
        g_trying = 0;
        return 0;
    }
    g_trying = 1;
    f(arg);
    g_trying = 0;
    return 1;
}

static struct sigaction g_old_segv;
static struct sigaction g_old_bus;
static struct sigaction g_old_ill;

static void temen_chain(struct sigaction *old, int sig, siginfo_t *info, void *uc) {
    if (old->sa_flags & SA_SIGINFO) {
        if (old->sa_sigaction)
            old->sa_sigaction(sig, info, uc);
    } else if (old->sa_handler != SIG_DFL && old->sa_handler != SIG_IGN) {
        old->sa_handler(sig);
    } else {
        /* No useful previous handler: restore the default and re-raise so the process dies
         * with the usual diagnostics (this is a genuine host fault, not a guest one). */
        signal(sig, SIG_DFL);
        raise(sig);
    }
}

static void temen_handler(int sig, siginfo_t *info, void *uc) {
    uintptr_t addr = (uintptr_t)info->si_addr;
    /* Our own backtrace walk faulted (#1487). This test comes **first**, before the arming and range
     * tests: by the time the walk runs we have already cleared `g_scope`, so the nested fault would
     * otherwise fall through to `temen_chain` → `SIG_DFL` → `raise` and kill the host on what was a
     * recoverable guest fault. Nor would keeping `g_scope` set across the walk do: that only recovers
     * a walk fault landing inside the scope's range, and it would re-enter its `siglongjmp` from a
     * half-written capture. Unwind the walk instead; the guest's original trap is reported below on
     * the way out, with whatever frames the walk had published. */
    if (temen_walk_in_progress())
        temen_walk_abort(); /* does not return */
    /* A host access to the window that may fault (#2996): back to its own recovery point. First, for
     * the walk's reason: it is not a guest fault, and the guarded call's state is not its to unwind. */
    if (g_trying && addr >= g_try_lo && addr < g_try_hi) {
        g_trying = 0;
        siglongjmp(g_try_buf, 1); /* back to temen_try_call; does not return */
    }
    /* SIGILL = the confinement bounds-check trap (Cranelift `trapnz` → `ud2`/`udf`, the memory-fault
     * lowering, DESIGN.md §4/§5). Unlike a guard-page SIGSEGV, `si_addr` is the faulting *instruction*
     * PC (in JIT code), not a window data address, so the [lo,hi) range test does not apply: an
     * illegal instruction inside an armed guarded guest call is always our emitted trap. Recover
     * exactly like a guarded data fault (capture the frame, then siglongjmp out reporting a memory
     * fault). Not armed ⇒ a genuine host illegal instruction: chain to the previous disposition. */
    struct temen_scope *s = g_scope;
    if (sig == SIGILL) {
        if (s && !temen_host_fault(sig, uc, (uintptr_t)info->si_addr)) {
            g_scope = NULL; /* disarmed for the capture; the landing re-links the parent */
            g_trying = 0; /* a try open on the frames this abandons goes with them (#2996) */
            temen_capture_frame(uc);
            siglongjmp(s->buf, 1);
        }
        temen_chain(&g_old_ill, sig, info, uc);
        return;
    }
    if (s && addr >= s->lo && addr < s->hi && !temen_host_fault(sig, uc, addr)) {
        g_scope = NULL; /* as for SIGILL, above */
        g_trying = 0; /* as for SIGILL, above */
        temen_capture_frame(uc); /* stash the faulting frame before the stack unwinds (§5 W3) */
        siglongjmp(s->buf, 1);  /* back to temen_run_guarded; does not return */
    }
    temen_chain(sig == SIGBUS ? &g_old_bus : &g_old_segv, sig, info, uc);
}

/* Install the handler. **Idempotent, and it has to be**: `sigaction` hands back the *previous*
 * disposition, so a second install saves our own handler as `g_old_*`, and `temen_chain` then calls
 * `temen_handler` from `temen_handler` — an unkillable spin on the first fault we decline, which is
 * far worse than the crash it replaces.
 *
 * `mem.rs`'s `install_guard` orders the real call behind a `std::sync::Once`, which is what makes the
 * install itself well-defined; the test-and-set here is so that a *direct* caller (a second one, or
 * two at once) cannot reach the self-chaining state. A loser returns without waiting: under the
 * `Once` there are no losers, and for a caller outside that contract "possibly not installed yet" is
 * a far better failure than an unkillable spin. */
static atomic_flag g_installed = ATOMIC_FLAG_INIT;

void temen_install_trap_handler(void) {
    if (atomic_flag_test_and_set(&g_installed))
        return;
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = temen_handler;
    /* SA_NODEFER: don't block SIGSEGV/SIGBUS while the handler runs. The backtrace walk it runs can
     * itself fault (#1487), and that nested fault must reach the handler, which aborts the walk
     * (`temen_walk_abort`): a blocked synchronous signal would kill the process instead. Nothing else
     * in the handler faults (it touches only thread-locals), so re-entry is not a recursion hazard.
     * #2805 — and load-bearing for the guarded entry, which saves no signal mask: the jump out of the
     * handler leaves the mask as the handler ran with, so the handler must not block the signal it
     * takes (`tests/fault_mask.rs`).
     * (No demand fault is handled here: the Cranelift JIT pages a child by a software check before
     * the access, never by a fault, #2164.) */
    sa.sa_flags = SA_SIGINFO | SA_ONSTACK | SA_NODEFER;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, &g_old_segv);
    sigaction(SIGBUS, &sa, &g_old_bus);
    /* SIGILL: the Cranelift `trapnz` confinement trap (see temen_handler). Same handler/flags. */
    sigaction(SIGILL, &sa, &g_old_ill);
}

/* #3167 — a guarded call returned with another scope innermost: some stack switched out from under
 * its scope with a scope of its own still open, so this thread's recovery point names a frame on a
 * stack that is not running here (#3158). The yield floor (#3162) is the rule that keeps any stack
 * from doing so; this is its backstop, in every build. Fatal: no recovery point on this thread can be
 * trusted. Async-signal-safe, as the host-fault report is. */
static void temen_scope_open_above(struct temen_scope *found, struct temen_scope *own) {
    static const char head[] =
        "temen-jit: a guarded call returned with another recovery scope innermost (a stack switched out under it, #3167): found=";
    char buf[224];
    char *p = buf;
    memcpy(p, head, sizeof head - 1);
    p += sizeof head - 1;
    p = temen_hex(p, (uintptr_t)found);
    memcpy(p, " own=", 5);
    p += 5;
    p = temen_hex(p, (uintptr_t)own);
    *p++ = '\n';
    ssize_t n = write(2, buf, (size_t)(p - buf));
    (void)n;
    abort();
}

/* Run `fn(a, r, m, t, tc)` (a JIT entry trampoline) with faults in [lo, hi) caught.
 * Returns 0 if it ran to completion, 1 if a guarded fault was caught and unwound.
 *
 * **Re-entrant** (§14 nesting): a guest may run a *child* guest (in its own window) from inside its
 * own guarded call. This call's scope links to the one innermost at entry and unlinks at exit, so a
 * child's guard nests cleanly and the parent's recovery point is intact afterwards. A child fault
 * unwinds to *this* (the child's) frame; the parent's is untouched. */
int temen_run_guarded(void (*fn)(const long *, long *, unsigned char *, const void *, long *),
                    const long *a, long *r, unsigned char *m, const void *t, long *tc,
                    uintptr_t lo, uintptr_t hi) {
    struct temen_scope scope;
    scope.lo = lo;
    scope.hi = hi;
    scope.parent = temen_scope_top();
    /* #2805 — armed without saving the signal mask: saving it is an `rt_sigprocmask` syscall on
     * every guarded entry (about 120 ns, and a point where a woken peer takes a contended core),
     * and the jump never needs it restored. The handler runs with the mask the fault found
     * (`SA_NODEFER`, an empty `sa_mask`), and it jumps only out of a fault in the armed window range
     * or out of an emitted trap: guest code, which cannot change the mask, or a host thunk touching
     * guest memory, and no host code here changes the thread's mask (a guest's `sigprocmask` is
     * emulated state). So the mask at the jump is the mask at this entry.
     * `scope` is not written between here and the jump, so its fields are well-defined after it. */
    if (sigsetjmp(scope.buf, 0)) {
        /* A guarded fault unwound back here: the handler disarmed; link the parent back in. */
        temen_scope_set(scope.parent);
        return 1;
    }
    temen_scope_set(&scope);
    fn(a, r, m, t, tc);
    /* Ran to completion: this call's scope must be the innermost, on whatever thread `fn` returned
     * on (R-TLS: read here, not through an address taken before `fn`). */
    struct temen_scope *top = temen_scope_top();
    if (top != &scope)
        temen_scope_open_above(top, &scope);
    temen_scope_set(scope.parent);
    return 0;
}
