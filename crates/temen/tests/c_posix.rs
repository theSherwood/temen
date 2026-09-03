//! End-to-end C linking against the **POSIX personality** (`temen-posix`) through §7 named imports:
//! each import name binds to `(HOST_PROC, op)` on the granted personality handle as an
//! instantiation-time **slot binding** ([`temen_posix::resolve`] + `Host::set_import_bindings`) —
//! the module bytes are never rewritten (IMPORTS.md phase 4).
//!
//! The module's **import section is its capability manifest** — the discoverable contract between
//! guest and host. There is no positional agreement anywhere: no powerbox slot for the personality,
//! no `__vm_cap(n)`, no implicit slot numbering shared out-of-band. A tiny guest libc **shim**
//! (guest code) gives each libc call its **real C signature** — `write(fd, buf, n)`, `open(path,
//! flags)`, `getenv(name)`, `exit(code)` — adapting NUL-terminated strings to the personality's
//! explicit-length `(ptr, len)` ABI (POSIX.md §4), and forwards to a `__px_`-prefixed undefined
//! extern whose first argument is a literal `0`: a **dummy** handle operand, vestigial in static
//! dispatch (the slot binding carries the granted handle — IMPORTS.md §2.5). Grant happens
//! *before* binding (the §7 "binding happens once, at instantiation" ordering); an unknown name
//! fails closed.
//!
//! The shim uses the **standard libc names** `write`/`read`/`exit` — its *definitions* shadow
//! chibicc's Stream/Exit builtins (PROCESS.md S15 (b): a guest definition beats a compiler builtin),
//! so `write(1, buf, n)` reaches the personality with `fd` preserved rather than the fd-dropping
//! powerbox Stream call. The frontend `_start` is now paramless (S15 (c2)): these personality-only
//! programs grant no powerbox, so `_start`'s by-name resolves of it stash `-errno` and are never
//! loaded — the libc reaches the personality through its slots (`bind_shim`), and the entry runs `&[]`.
//!
//! Each program runs `_start` (function 0) on **both** the interpreter and the JIT under an identical
//! host, asserting they agree on the result *and* the observable personality state (captured stdout,
//! the memfs) — so it doubles as a cross-backend differential, capability effects included. The
//! personality's `HostProc` dispatches through the same `cap_dispatch_slots` the JIT's `call.cap` thunk
//! calls, so parity comes for free. Requires a unix C toolchain (`make` + `cc`) to build the chibicc
//! fork, so the suite is gated to `#![cfg(unix)]` (like `c_frontend.rs`).
#![cfg(unix)]

#[path = "support/repo_root.rs"]
mod repo_root_mod;
use repo_root_mod::repo_root;

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use core::ffi::c_void;
use temen_interp::{run_with_host, Host, Trap, Value};
use temen_jit::{compile_and_run_with_host, JitOutcome};
use temen_posix::Posix;
use temen_run::cap_thunk;
use temen_text::parse_module as parse_module_raw;
use temen_verify::verify_module;

/// Build the chibicc fork once per test binary, returning the path to its binary.
fn chibicc() -> &'static Path {
    static CC: OnceLock<PathBuf> = OnceLock::new();
    CC.get_or_init(|| {
        let dir = repo_root().join("frontend/chibicc");
        let status = Command::new("make")
            .arg("-s")
            .current_dir(&dir)
            .status()
            .expect("run `make` to build the chibicc fork");
        assert!(status.success(), "chibicc build failed");
        dir.join("chibicc")
    })
    .as_path()
}

/// Compile a C source string to our text IR via the frontend.
fn c_to_ir(src: &str) -> String {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let id = N.fetch_add(1, Ordering::Relaxed);
    let base = std::env::temp_dir().join(format!("temen_cposix_{}_{id}", std::process::id()));
    let cfile = base.with_extension("c");
    let irfile = base.with_extension("temen");
    std::fs::write(&cfile, src).unwrap();
    let status = Command::new(chibicc())
        .args([
            "-cc1",
            "--emit-ir",
            "-cc1-input",
            cfile.to_str().unwrap(),
            "-cc1-output",
            irfile.to_str().unwrap(),
            cfile.to_str().unwrap(),
        ])
        .status()
        .expect("run chibicc");
    assert!(status.success(), "chibicc failed on:\n{src}");
    std::fs::read_to_string(&irfile).unwrap()
}

/// Install the shim's import-slot bindings: strip the `__px_` prefix (which keeps the shim's
/// externs clear of chibicc's builtin names) and map the bare libc name through
/// [`temen_posix::resolve`] to `(HOST_PROC, op)` on the granted personality `handle` — the phase-4
/// no-rewrite binding (`Host::set_import_bindings`). A name outside the personality leaves its
/// slot unbound (a dispatch through it is a fail-closed `CapFault`).
fn bind_shim(m: &temen_ir::Module, host: &mut Host, handle: i32) {
    let bindings = m
        .imports
        .iter()
        .map(
            |i| match i.name.strip_prefix("__px_").and_then(temen_posix::resolve) {
                Some(c) => temen_interp::BoundImport::required(c.type_id, c.op, handle),
                None => temen_interp::BoundImport::rebindable(0, 0, None),
            },
        )
        .collect();
    host.set_import_bindings(bindings);
}

/// Grant the POSIX personality on `host`, with a window-heap region in the upper half of the guest
/// window (clear of chibicc's low data image + data stack). These programs are **personality-only**:
/// the paramless `_start` (S15 (c2)) resolves the fixed powerbox by name, but this host grants none
/// of it — those resolves stash `-errno` and are never loaded, since the libc shim reaches the
/// personality, bound by name via [`resolver`]. The personality handle is **not** an entry argument;
/// it binds at resolve. Returns a [`Posix`] handle to the captured state + the granted handle.
fn setup(host: &mut Host, win: u64) -> (Posix, i32) {
    // Heap at the top quarter: the frontend guarantees data ends by `win/2` (it sizes the window to
    // 2x data + reserve), but the C data stack grows UP from data_end with no marked ceiling — a
    // heap granted at `win/2` sits right where a big program's stack frames land (the regex
    // differential found this: its ~32.7 KiB of data in a 64 KiB window put main's frame 104 bytes
    // past the old heap base, and deep matcher recursion smashed live heap blocks). `3*win/4`
    // leaves the stack >= win/4 of headroom above data while the heap keeps win/4.
    let (px, posix) = temen_posix::grant(host, 3 * win / 4, win, Vec::new());
    (posix, px)
}

/// What a program did on one backend: either `main` returned values or the personality's `exit` op
/// terminated it (`exited`), plus the captured stdout and the memfs contents of file `"f"`.
struct Effects {
    result: Vec<Value>,
    exited: Option<i32>,
    stdout: Vec<u8>,
    file_f: Option<Vec<u8>>,
}

/// Compile a C program, **grant first** on two identical hosts (binding needs the granted
/// handle — the §7 instantiation ordering), bind its import slots through [`bind_shim`], verify,
/// then run `_start` on **both** backends and return each backend's observable effects for the caller to
/// compare. `prep` stages each backend's personality identically before the run (seed the
/// environment / memfs); pass a no-op when there is nothing to stage. Panics with the IR on a
/// parse/verify/trap so failures are legible.
fn run_both(src: &str, prep: impl Fn(&Posix)) -> (Effects, Effects) {
    let ir = c_to_ir(src);
    let raw = parse_module_raw(&ir)
        .unwrap_or_else(|e| panic!("parse IR failed: {e:?}\n--- IR ---\n{ir}"));
    let win = 1u64
        << raw
            .memory
            .expect("the frontend declares a window")
            .size_log2;

    // Grant before resolve, identically on both hosts; deterministic grant order gives both
    // backends the same handle value, so one resolved module serves both.
    let mut ih = Host::new();
    let (iposix, ipx) = setup(&mut ih, win);
    let mut jh = Host::new();
    let (jposix, jpx) = setup(&mut jh, win);
    assert_eq!(ipx, jpx, "identical grant order → identical handle");
    prep(&iposix);
    prep(&jposix);

    // Phase 4: no rewrite — the manifest stays and each slot binds to the personality.
    verify_module(&raw).unwrap_or_else(|e| panic!("verify failed: {e:?}\n--- IR ---\n{ir}"));
    bind_shim(&raw, &mut ih, ipx);
    bind_shim(&raw, &mut jh, jpx);
    let m = raw;

    // Interpreter — a normal return yields values; the personality's `exit` op is `Trap::Exit(code)`.
    // The paramless `_start` takes no entry args.
    let mut fuel = 50_000_000u64;
    let (iresult, iexited) = match run_with_host(&m, 0, &[], &mut fuel, &mut ih) {
        Ok(v) => (v, None),
        Err(Trap::Exit(c)) => (Vec::new(), Some(c)),
        Err(e) => panic!("interp trapped: {e:?}\n--- IR ---\n{ir}"),
    };
    let interp = Effects {
        result: iresult,
        exited: iexited,
        stdout: iposix.stdout(),
        file_f: iposix.read_file("f"),
    };

    // JIT.
    let jout =
        compile_and_run_with_host(&m, 0, &[], cap_thunk, &mut jh as *mut Host as *mut c_void)
            .expect("jit compiles");
    let (jresult, jexited) = match jout {
        JitOutcome::Returned(s) => (s.iter().map(|&x| Value::I64(x)).collect(), None),
        JitOutcome::Exited(c) => (Vec::new(), Some(c)),
        other => panic!("jit ended abnormally: {other:?}\n--- IR ---\n{ir}"),
    };
    let jit = Effects {
        result: jresult,
        exited: jexited,
        stdout: jposix.stdout(),
        file_f: jposix.read_file("f"),
    };

    (interp, jit)
}

/// #796 L2 — like [`run_both`] but **interpreter-only**. Async signal delivery (the safepoint redirect)
/// lives only in the interpreter, so a program that relies on it can't run on the JIT (which would never
/// deliver, and here would spin forever). Returns just the interpreter's effects.
fn run_interp_only(src: &str, prep: impl Fn(&Posix)) -> Effects {
    let ir = c_to_ir(src);
    let raw = parse_module_raw(&ir)
        .unwrap_or_else(|e| panic!("parse IR failed: {e:?}\n--- IR ---\n{ir}"));
    let win = 1u64
        << raw
            .memory
            .expect("the frontend declares a window")
            .size_log2;
    let mut ih = Host::new();
    let (iposix, ipx) = setup(&mut ih, win);
    prep(&iposix);
    verify_module(&raw).unwrap_or_else(|e| panic!("verify failed: {e:?}\n--- IR ---\n{ir}"));
    bind_shim(&raw, &mut ih, ipx);
    let mut fuel = 200_000_000u64;
    let (result, exited) = match run_with_host(&raw, 0, &[], &mut fuel, &mut ih) {
        Ok(v) => (v, None),
        Err(Trap::Exit(c)) => (Vec::new(), Some(c)),
        Err(e) => panic!("interp trapped: {e:?}\n--- IR ---\n{ir}"),
    };
    Effects {
        result,
        exited,
        stdout: iposix.stdout(),
        file_f: iposix.read_file("f"),
    }
}

/// The **bytecode-engine** twin of [`run_interp_only`]: the same guest + personality wiring, driven by
/// the cooperative `drive` (the tier the browser playground runs on). Used to differential the
/// personality `fork()`/`waitpid()` park engine (#1080 rung 3) against the tree-walker oracle.
fn run_bytecode_only(src: &str, prep: impl Fn(&Posix)) -> Effects {
    run_bytecode_setup(src, |_host, posix| prep(posix))
}

/// [`run_bytecode_only`] with a `|host, posix|` setup callback (e.g. `stage_executable` to register a
/// `/bin` command before the run) — the bytecode-engine twin of [`run_interp_setup`]. Differentials the
/// whole external-command flow (fork → execve → waitpid) on the cooperative driver (#1080 rungs 2+3).
fn run_bytecode_setup(src: &str, extra: impl Fn(&mut Host, &Posix)) -> Effects {
    let ir = c_to_ir(src);
    let raw = parse_module_raw(&ir)
        .unwrap_or_else(|e| panic!("parse IR failed: {e:?}\n--- IR ---\n{ir}"));
    let win = 1u64
        << raw
            .memory
            .expect("the frontend declares a window")
            .size_log2;
    let mut ih = Host::new();
    let (iposix, ipx) = setup(&mut ih, win);
    extra(&mut ih, &iposix);
    verify_module(&raw).unwrap_or_else(|e| panic!("verify failed: {e:?}\n--- IR ---\n{ir}"));
    bind_shim(&raw, &mut ih, ipx);
    let mut fuel = 200_000_000u64;
    let ran = temen_interp::bytecode::compile_and_run_with_host(&raw, 0, &[], &mut fuel, &mut ih)
        .expect("the bytecode engine compiles this module (no declining op)");
    let (result, exited) = match ran {
        Ok(v) => (v, None),
        Err(Trap::Exit(c)) => (Vec::new(), Some(c)),
        Err(e) => panic!("bytecode trapped: {e:?}\n--- IR ---\n{ir}"),
    };
    Effects {
        result,
        exited,
        stdout: iposix.stdout(),
        file_f: iposix.read_file("f"),
    }
}

/// The **parallel-driver** twin of [`run_bytecode_only`] (#748): the same guest + personality wiring,
/// driven by `drive_parallel` — every fork twin a real OS thread over a private window copy, blocking
/// `waitpid` a real condvar wait. The cooperative engine and the tree-walker are its oracles: on these
/// deterministic-output guests the results must be identical (the dual-driver principle — every
/// fork/wait-using program runs correctly under both coop and parallel, though scheduling interleaves
/// differ).
fn run_bytecode_parallel_only(src: &str, prep: impl Fn(&Posix)) -> Effects {
    run_bytecode_parallel_setup(src, |_host, posix| prep(posix))
}

/// [`run_bytecode_parallel_only`] with a `|host, posix|` setup callback (e.g. `stage_executable`) —
/// the parallel-driver twin of [`run_bytecode_setup`], for the fork → execve → waitpid and
/// pipe-through-exec differentials (#748 rungs 2+3).
fn run_bytecode_parallel_setup(src: &str, extra: impl Fn(&mut Host, &Posix)) -> Effects {
    let ir = c_to_ir(src);
    let raw = parse_module_raw(&ir)
        .unwrap_or_else(|e| panic!("parse IR failed: {e:?}\n--- IR ---\n{ir}"));
    let win = 1u64
        << raw
            .memory
            .expect("the frontend declares a window")
            .size_log2;
    let mut ih = Host::new();
    let (iposix, ipx) = setup(&mut ih, win);
    extra(&mut ih, &iposix);
    verify_module(&raw).unwrap_or_else(|e| panic!("verify failed: {e:?}\n--- IR ---\n{ir}"));
    bind_shim(&raw, &mut ih, ipx);
    // An 8-aligned zeroed buffer + a `Region::shared` over it (the bytecode_parallel.rs harness
    // shape): the root runs over this shared backing; each fork twin copies off it (`fork_private`).
    let layout = std::alloc::Layout::from_size_align(win as usize, 8).unwrap();
    // SAFETY: non-zero layout; the buffer is `win` valid 8-aligned bytes owned here, used only as
    // this run's window until freed below, after the region (and every vCPU borrow) is dropped.
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!base.is_null());
    // SAFETY: `base` is `win` valid 8-aligned bytes, exclusively this window's, freed only after.
    let back = std::sync::Arc::new(unsafe { temen_interp::Region::shared(base, win) });
    let mut fuel = 200_000_000u64;
    let ran = temen_interp::bytecode::compile_and_run_capture_over_parallel_with_host(
        &raw,
        0,
        &[],
        &mut fuel,
        &[],
        std::sync::Arc::clone(&back),
        &mut ih,
    )
    .expect("the bytecode engine compiles this module (no declining op)");
    drop(back);
    // SAFETY: same layout; the region and all borrows of `base` are gone (the scope joined all vCPUs).
    unsafe { std::alloc::dealloc(base, layout) };
    let (result, exited) = match ran.0 {
        Ok(v) => (v, None),
        Err(Trap::Exit(c)) => (Vec::new(), Some(c)),
        Err(e) => panic!("bytecode-parallel trapped: {e:?}\n--- IR ---\n{ir}"),
    };
    Effects {
        result,
        exited,
        stdout: iposix.stdout(),
        file_f: iposix.read_file("f"),
    }
}

/// #796 L2 — **async delivery to a running loop**: a signal raised while the guest is compute-bound is
/// delivered to its handler at a safepoint, with **no `sigcheck` poll** in the loop. The handler sets a
/// global; the loop (which never polls) observes it and exits. This is the headline "async" win — the
/// interpreter redirects a running fiber into the handler `void(int)` on its registered signal stack and
/// resumes. Interpreter-only (the JIT has no safepoint injection yet).
#[test]
fn c_async_signal_interrupts_a_compute_loop() {
    let src = r#"
long __px_signal(int cap, long signum, long handler);
long __px_kill(int cap, long pid, long sig);
long __px_sigaltstack(int cap, long sp, long size);
static char sigstk[16384];
static volatile int fired;
static void handler(int sig) { fired = sig; }
int main(void) {
  __px_signal(0, 2, (long)handler);           /* catch SIGINT(2) with `handler` */
  __px_sigaltstack(0, (long)sigstk, 16384);   /* register the dedicated handler stack */
  __px_kill(0, 0, 2);                          /* raise SIGINT -- armed for async delivery */
  long spins = 0;
  while (!fired) {                             /* compute loop -- NEVER calls sigcheck */
    spins = spins + 1;
    if (spins > 100000000) return -1;          /* safety: fail if the handler never fired */
  }
  return fired;                                /* 2, set asynchronously by the handler */
}
"#;
    let e = run_interp_only(src, |_| {});
    assert_eq!(
        e.result,
        vec![Value::I32(2)],
        "the async handler ran during the compute loop (no poll) and set `fired = SIGINT`"
    );
    // #1146 — the same async delivery now runs on the bytecode engine's per-op safepoint, on BOTH the
    // cooperative driver (the browser tier) and `drive_parallel`. Each redirects the running vCPU into
    // `handler` mid-compute-loop and returns 2, byte-identical to the tree-walker oracle above.
    let b = run_bytecode_only(src, |_| {});
    assert_eq!(
        b.result,
        vec![Value::I32(2)],
        "#1146: the cooperative bytecode engine delivers the async handler mid-loop too"
    );
    let p = run_bytecode_parallel_only(src, |_| {});
    assert_eq!(
        p.result,
        vec![Value::I32(2)],
        "#1146: the parallel bytecode driver delivers the async handler mid-loop too"
    );
}

/// #796 L2 — async delivery **honors the mask**: a blocked signal is held (not delivered to a running
/// loop) until unblocked, then delivered asynchronously. Encodes both checks: `a` (fired while blocked,
/// must be 0) in the thousands, the post-unblock `fired` (2) in the units → 2. A mask that leaked would
/// read 2002 instead.
#[test]
fn c_async_delivery_respects_the_mask() {
    let src = r#"
long __px_signal(int cap, long signum, long handler);
long __px_kill(int cap, long pid, long sig);
long __px_sigaltstack(int cap, long sp, long size);
long __px_sigprocmask(int cap, long how, long set, long oldset);
static char sigstk[16384];
static volatile int fired;
static unsigned long mask;
static void handler(int sig) { fired = sig; }
int main(void) {
  __px_signal(0, 2, (long)handler);
  __px_sigaltstack(0, (long)sigstk, 16384);
  mask = (1UL << 2);
  __px_sigprocmask(0, 0, (long)&mask, 0);      /* SIG_BLOCK SIGINT */
  __px_kill(0, 0, 2);                          /* raise -- blocked, must NOT deliver */
  long i = 0;
  while (i < 2000000) i = i + 1;               /* spin while blocked: handler must stay silent */
  int a = fired;                               /* 0 */
  __px_sigprocmask(0, 1, (long)&mask, 0);      /* SIG_UNBLOCK -- now deliverable async */
  while (!fired) { i = i + 1; if (i > 100000000) return -1; }
  return a * 1000 + fired;                     /* 2 */
}
"#;
    let e = run_interp_only(src, |_| {});
    assert_eq!(
        e.result,
        vec![Value::I32(2)],
        "a blocked signal is not delivered async until unblocked"
    );
    // #1146 — the mask logic lives in the shared personality (`take_deliverable`), so the bytecode
    // engine's safepoint delivery honors it identically: held while blocked, delivered on unblock.
    assert_eq!(
        run_bytecode_only(src, |_| {}).result,
        vec![Value::I32(2)],
        "#1146: the bytecode engine honors the block mask for async delivery too"
    );
}

/// #796 block-during-handler — **a handler is never reentered by its own signal**: the handler
/// re-raises SIGINT at itself and spins a window; the delivery mask (the delivered signal is
/// blocked for the handler's duration, POSIX) holds it — a leak would re-enter and bump `count`
/// inside the window. On return the mask restores and the held raise delivers (a second, NON-nested
/// handler run): `count` reaches 2 with `leaked` still 0. Interpreter-only (no JIT safepoints).
#[test]
fn c_a_handler_is_never_reentered_by_its_own_signal() {
    let src = r#"
long __px_signal(int cap, long signum, long handler);
long __px_kill(int cap, long pid, long sig);
long __px_sigaltstack(int cap, long sp, long size);
static char sigstk[16384];
static volatile int count;
static volatile int leaked;
static void handler(int sig) {
  count = count + 1;
  if (count == 1) {
    __px_kill(0, 0, 2);                        /* re-raise SIGINT inside its own handler */
    long i = 0;
    while (i < 2000000) i = i + 1;             /* the leak window: a reentry would bump count */
    if (count != 1) leaked = 1;
  }
}
int main(void) {
  __px_signal(0, 2, (long)handler);
  __px_sigaltstack(0, (long)sigstk, 16384);
  __px_kill(0, 0, 2);
  long i = 0;
  while (count < 2) {                          /* the held raise delivers at handler return */
    i = i + 1;
    if (i > 100000000) return -1;
  }
  return leaked * 100 + count;                 /* 2: delivered twice, never nested */
}
"#;
    let e = run_interp_only(src, |_| {});
    assert_eq!(
        e.result,
        vec![Value::I32(2)],
        "the in-handler re-raise was held (blocked) and delivered exactly once after return"
    );
    // #1146 — block-during-handler on the bytecode tier: the injected handler window blocks its own
    // signal (via `take_deliverable`'s mask push) and `Op::Ret` fires `handler_returned` to restore
    // it, so the held re-raise delivers exactly once after return — never reentrant.
    assert_eq!(
        run_bytecode_only(src, |_| {}).result,
        vec![Value::I32(2)],
        "#1146: the bytecode engine never reenters a handler by its own signal"
    );
}

/// #796 nested delivery — **a different unmasked signal interrupts a running handler**: SIGINT's
/// handler raises SIGUSR1 at itself and spins until USR1's handler runs — which can only happen
/// NESTED (the old one-in-handler guard would spin this forever / return the stuck marker).
/// `saw10_in2 = 1` witnesses USR1's handler running while SIGINT's was live.
#[test]
fn c_a_different_signal_nests_into_a_running_handler() {
    let src = r#"
long __px_signal(int cap, long signum, long handler);
long __px_kill(int cap, long pid, long sig);
long __px_sigaltstack(int cap, long sp, long size);
static char sigstk[16384];
static volatile int in2;
static volatile int fired10;
static volatile int saw10_in2;
static volatile int stuck;
static void h10(int sig) { saw10_in2 = in2; fired10 = 1; }
static void h2(int sig) {
  in2 = 1;
  __px_kill(0, 0, 10);                         /* raise USR1 inside SIGINT's handler */
  long i = 0;
  while (!fired10) {                           /* only a NESTED delivery can break this */
    i = i + 1;
    if (i > 100000000) { stuck = 1; break; }
  }
  in2 = 0;
}
int main(void) {
  __px_signal(0, 2, (long)h2);
  __px_signal(0, 10, (long)h10);
  __px_sigaltstack(0, (long)sigstk, 16384);
  __px_kill(0, 0, 2);
  long i = 0;
  while (!fired10) { i = i + 1; if (i > 100000000) return -1; }
  return stuck * 100 + saw10_in2 * 10 + fired10;  /* 11: nested, not stuck */
}
"#;
    let e = run_interp_only(src, |_| {});
    assert_eq!(
        e.result,
        vec![Value::I32(11)],
        "USR1's handler ran nested inside SIGINT's (saw10_in2 = 1), no stuck marker"
    );
    // #1146 — nested delivery on the bytecode tier: the `sig_handler_stack` guard admits a *different*
    // unmasked signal while a handler is live (up to `MAX_SIG_HANDLER_NEST`), so USR1 nests into
    // SIGINT's running handler exactly as on the tree-walker.
    assert_eq!(
        run_bytecode_only(src, |_| {}).result,
        vec![Value::I32(11)],
        "#1146: a different signal nests into a running handler on the bytecode engine too"
    );
}

/// #796 `sa_mask` — **a sigaction-masked signal is held for the handler's duration**: SIGINT is
/// installed via `sigaction` with `sa_mask` blocking USR1; the handler raises USR1 and spins a
/// window — it must NOT nest (held by the handler mask); it delivers after the handler returns.
/// The complement of the nesting test: same shape, `sa_mask` flips the outcome.
#[test]
fn c_sa_mask_holds_a_signal_for_the_handlers_duration() {
    let src = r#"
long __px_signal(int cap, long signum, long handler);
long __px_kill(int cap, long pid, long sig);
long __px_sigaltstack(int cap, long sp, long size);
long __px_sigaction(int cap, long signum, long act, long oldact);
static char sigstk[16384];
static long act[3];
static volatile int in2;
static volatile int fired10;
static volatile int saw10_in2;
static volatile int held;
static void h10(int sig) { saw10_in2 = in2; fired10 = 1; }
static void h2(int sig) {
  in2 = 1;
  __px_kill(0, 0, 10);                         /* raise USR1 -- sa_mask blocks it here */
  long i = 0;
  while (i < 2000000) i = i + 1;               /* the window: a leak would nest h10 */
  held = !fired10;                             /* still held at the end of the handler */
  in2 = 0;
}
int main(void) {
  act[0] = (long)h2;
  act[1] = (1L << 10);                         /* sa_mask: block USR1 while h2 runs */
  act[2] = 0;
  __px_sigaction(0, 2, (long)act, 0);
  __px_signal(0, 10, (long)h10);
  __px_sigaltstack(0, (long)sigstk, 16384);
  __px_kill(0, 0, 2);
  long i = 0;
  while (!fired10) { i = i + 1; if (i > 100000000) return -1; }  /* delivers at h2's return */
  return held * 100 + saw10_in2 * 10 + fired10;  /* 101: held during h2, delivered after, not nested */
}
"#;
    let e = run_interp_only(src, |_| {});
    assert_eq!(
        e.result,
        vec![Value::I32(101)],
        "sa_mask held USR1 through the handler (held=1), delivered non-nested after return"
    );
    // #1146 — `sa_mask` on the bytecode tier: `take_deliverable` folds the action mask into the
    // handler mask, so USR1 is held through SIGINT's handler and delivered non-nested after return.
    assert_eq!(
        run_bytecode_only(src, |_| {}).result,
        vec![Value::I32(101)],
        "#1146: the bytecode engine honors sa_mask for the handler's duration too"
    );
}

/// A tiny guest libc shim (guest code) binding C's libc calls to the POSIX personality by **name
/// only**. Each `__px_` extern's first argument is a literal `0` — a dummy handle operand,
/// vestigial in static dispatch (the slot binding carries the handle); no `__vm_cap`, no stash.
/// The wrappers expose the **real C signatures** — `write(fd, buf, n)`, `open(path, flags)`,
/// `getenv(name)`, `exit(code)` — adapting C's NUL-terminated `char*` convention to the personality's
/// explicit-length `(ptr, len)` ABI (POSIX.md §4); the adaptation is guest code. `write`/`read`/`exit`
/// are the standard libc names: they *define* those functions, which now **shadows** chibicc's Stream
/// builtin (PROCESS.md S15 (b)) — so a program's `write(1, buf, n)` reaches the personality with `fd`
/// preserved, not the fd-dropping powerbox Stream call.
const SHIM: &str = r#"
long __px_write(int cap, long fd, long buf, long len);
long __px_read(int cap, long fd, long buf, long len);
long __px_malloc(int cap, long size);
long __px_open(int cap, long path, long len, long flags);
long __px_lseek(int cap, long fd, long off, long whence);
long __px_getcwd(int cap, long buf, long size);
long __px_chdir(int cap, long path, long len);
long __px_getenv(int cap, long name, long len);
void __px_exit(int cap, int code);

static long slen(char *s) { long n = 0; while (s[n]) n = n + 1; return n; }

void *malloc(long size) { return (void *)__px_malloc(0, size); }
long open(char *path, long flags) { return __px_open(0, (long)path, slen(path), flags); }
long write(long fd, void *buf, long n) { return __px_write(0, fd, (long)buf, n); }
long read(long fd, void *buf, long n) { return __px_read(0, fd, (long)buf, n); }
long lseek(long fd, long off, long whence) { return __px_lseek(0, fd, off, whence); }
char *getcwd(char *buf, long size) { return __px_getcwd(0, (long)buf, size) > 0 ? buf : 0; }
long chdir(char *path) { return __px_chdir(0, (long)path, slen(path)); }
char *getenv(char *name) { return (char *)__px_getenv(0, (long)name, slen(name)); }
void exit(int code) { __px_exit(0, code); }
"#;

/// The full round-trip: `malloc` a buffer, write it to the personality's **stdout** (fd 1), `open`
/// a memfs file and write the same bytes there, then `lseek` to 0 and `read` them back into a second
/// buffer and echo *that* to stdout. Proves malloc, fd routing (stdout vs a file fd), open, write,
/// lseek, and read all reach the personality from compiled C — identically on both backends.
#[test]
fn c_links_libc_to_posix_personality_roundtrip() {
    let src = format!(
        "{SHIM}\n\
int main() {{\n\
  char *msg = \"hi\\n\";\n\
  long n = slen(msg);\n\
  char *buf = (char *)malloc(32);\n\
  for (long i = 0; i < n; i = i + 1) buf[i] = msg[i];\n\
  write(1, buf, n);          /* fd 1 -> captured stdout */\n\
  long fd = open(\"f\", 66);    /* O_CREAT|O_RDWR */\n\
  write(fd, buf, n);         /* -> memfs file \"f\" */\n\
  lseek(fd, 0, 0);              /* SEEK_SET 0 */\n\
  char *buf2 = (char *)malloc(32);\n\
  long r = read(fd, buf2, 32); /* read the file back */\n\
  write(1, buf2, r);         /* echo it to stdout again */\n\
  return (int)fd;               /* the first file fd is 3 */\n\
}}\n"
    );
    let (interp, jit) = run_both(&src, |_| {});

    // Interpreter reference: first file fd is 3; stdout got "hi\n" twice; the memfs file holds "hi\n".
    assert_eq!(
        interp.result,
        vec![Value::I32(3)],
        "interp: main returns fd 3"
    );
    assert_eq!(interp.stdout, b"hi\nhi\n", "interp: two writes to stdout");
    assert_eq!(
        interp.file_f.as_deref(),
        Some(&b"hi\n"[..]),
        "interp: the memfs file was written"
    );

    // JIT parity — same personality path, so identical result + effects (result slots are i64).
    assert_eq!(jit.result, vec![Value::I64(3)], "jit: fd must match interp");
    assert_eq!(jit.stdout, interp.stdout, "jit: stdout must match interp");
    assert_eq!(jit.file_f, interp.file_f, "jit: memfs must match interp");
}

/// #799 slice 1 — **coexistence: one guest, both worlds, one Host.** A single compiled program links the
/// temen-posix **personality** (memfs/stdio via `__px_*` named imports — World A) AND drives the
/// **capability** path (`__vm_pipe`/`__vm_read`/`__vm_write`, the `cap.self`/Stream builtins — World B) in
/// one run. The worlds use **disjoint** host state (a HOST_PROC handle + import bindings vs. the Stream
/// powerbox slots) and **disjoint** guest dispatch (named imports vs. `call.cap` builtins), so they
/// compose on one `Host` with no conflict — `temen_posix::grant` claims nothing the capability grants touch,
/// and the `__vm_*` builtins generate no import entries, so the manifest is pure `__px_*`. This is the
/// bridge #799 is built on: a personality-linked program (eventually bash) that also holds capability
/// handles. Interp-only — the capability pipe path needs the `Real` scheduler (`CAP_SELF_PIPE`).
const DUAL_WORLD_SRC: &str = r#"
long __vm_pipe(int *fds);
long __vm_read(int fd, void *buf, long len);
long __vm_write(int fd, void *buf, long len);
int main(void) {
  char *msg = "hi\n";
  long n = slen(msg);
  /* World A — the personality: create a memfs file, write it, rewind, read it back. */
  long fd = open("f", 66);            /* O_CREAT|O_RDWR -> __px_open; the first file fd is 3 */
  write(fd, (void *)msg, n);          /* __px_write -> memfs "f" */
  lseek(fd, 0, 0);
  char a[8];
  long ra = read(fd, a, 8);           /* __px_read -> "hi\n", ra = 3 */
  /* World B — the capability path: mint a pipe (two Stream ends), write then read the same bytes.
     One fiber, so the write lands before the read drains it — no park needed for the coexistence proof. */
  int fds[2];
  __vm_pipe(fds);                     /* CAP_SELF_PIPE mints two Stream handles into the powerbox */
  __vm_write(fds[1], a, ra);          /* STREAM call.cap write */
  char b[8];
  long rb = __vm_read(fds[0], b, ra); /* STREAM call.cap read -> ra bytes */
  return (int)(fd + rb);              /* 3 + 3 = 6 : both worlds ran on one Host */
}
"#;

#[test]
fn c_a_guest_links_the_personality_and_the_capability_pipe_on_one_host() {
    let src = format!("{SHIM}\n{DUAL_WORLD_SRC}");
    let e = run_interp_only(&src, |_| {});
    // Both worlds ran on one Host: World A's memfs holds "hi\n" (the personality write), and main returns
    // 6 = the personality file fd (3) + the bytes the capability pipe round-tripped (3).
    assert_eq!(
        e.result,
        vec![Value::I32(6)],
        "one guest reached both the personality (fd 3) and the capability pipe (3 bytes) on one Host"
    );
    assert_eq!(
        e.file_f.as_deref(),
        Some(&b"hi\n"[..]),
        "World A (the personality) wrote the memfs file"
    );
}

/// #799 slice 2 — **a personality-linked guest genuinely *blocks* on a capability pipe read.** Slice 1
/// proved coexistence; this proves the parking bridge: `main` (personality-linked) mints a pipe and does a
/// blocking `__vm_read` on the empty read end while a live writer is open — so it **parks**
/// (`Blocked::PipeRead`) — and a spawned capability `writer` thread wakes it. This is the shape every
/// blocking syscall (and, next, `EINTR`) rides: a personality-linked program that can actually suspend on
/// a capability-world park. It also does one personality op (`open`/`write` a memfs file) to confirm both
/// worlds still compose. Interp-only (`Real` scheduler: `CAP_SELF_PIPE`, `thread.spawn`, the park/wake).
const DUAL_WORLD_PARK_SRC: &str = r#"
long __vm_pipe(int *fds);
long __vm_read(int fd, void *buf, long len);
long __vm_write(int fd, void *buf, long len);
int  __vm_thread_spawn(long (*fn)(long), void *stack, long arg);
long __vm_thread_join(int h);
static char msg[] = "GO\n";
long g_wfd;
long writer(long arg) {
  /* burn a little first so `main` reaches its read and parks before we write */
  volatile long acc = 0;
  for (long i = 0; i < 2000000; i = i + 1) acc = acc + i;
  __vm_write(g_wfd, msg, 3);   /* wakes main's parked read */
  return 0;
}
int main(void) {
  /* World A — one personality op, so both worlds still compose. */
  long fd = open("f", 66);
  write(fd, msg, 3);           /* __px_write -> memfs "f" = "GO\n" */
  /* World B — block on a capability pipe read, woken by the writer thread. */
  int fds[2];
  __vm_pipe(fds);
  g_wfd = fds[1];
  int h = __vm_thread_spawn(writer, (void *)0, 0);
  char b[8];
  long n = __vm_read(fds[0], b, 8);  /* empty FIFO, writer open -> PARKS; woken -> drains 3 */
  __vm_thread_join(h);
  return (int)(fd + n);              /* 3 + 3 = 6 */
}
"#;

#[test]
fn c_a_personality_guest_blocks_on_a_capability_pipe_read() {
    let src = format!("{SHIM}\n{DUAL_WORLD_PARK_SRC}");
    let e = run_interp_only(&src, |_| {});
    // main's blocking `__vm_read` parked and was woken by the writer thread (drained 3 bytes); the
    // personality op ran too (fd 3, memfs "f" = "GO\n"). 3 + 3 = 6.
    assert_eq!(
        e.result,
        vec![Value::I32(6)],
        "the personality-linked guest parked on the capability pipe read and was woken"
    );
    assert_eq!(
        e.file_f.as_deref(),
        Some(&b"GO\n"[..]),
        "World A (the personality) still ran alongside the capability park"
    );
}

/// #799 slice 3 — **faithful EINTR: a *delivered* signal interrupts a blocked capability syscall.** The
/// payoff of the bridge with the signal split (PROCESS.md §9 / #799): the guest catches SIGINT via the
/// **personality** (`__px_signal` + `__px_sigaltstack` — policy), then blocks on a **capability**
/// `__vm_read` (parks on `Blocked::PipeRead`). A sibling thread raises SIGINT via the personality's
/// `__px_kill`; the *personality* decides it is deliverable (caught + unmasked), and only then does the
/// **core** interrupt the parked read (`interrupt_interruptible_parks`, driven from the L2 safepoint) so it
/// returns `-EINTR` (sentinel 42). Nothing signal-specific lives in temen — the interrupt fires exactly when
/// the personality hands over a delivery. The raiser *retries* (deterministic under the M:N executor): a
/// `kill` before `main` parks is a no-op interrupt; `main` sets `done` once its read takes EINTR.
/// #1157 — **a yield-free shared-memory spin makes progress on the cooperative driver.** `main`
/// spawns a sibling that sets a flag and exits, then spins `while (!flag) {}` — reaching **no** park
/// point (no syscall, no futex). Under real threads (parallel) or the preemptive tree-walker the
/// sibling runs concurrently and `main` observes the store. The cooperative bytecode driver runs a
/// task to park-or-completion, so without a preemption quantum `main` monopolizes the single thread
/// and the sibling never runs → deadlock. With the #1157 op-count quantum (armed only while ≥2 tasks
/// are runnable) `main` is preempted, the sibling runs, and the store becomes visible. The
/// dual-driver differential now holds for this shape: all three engines return 7.
const SPIN_FLAG_SRC: &str = r#"
long __vm_atomic_add(void *p, long v);
long __vm_atomic_load(void *p);
int  __vm_thread_spawn(long (*fn)(long), void *stack, long arg);
long __vm_thread_join(int h);
long flag;
long setter(long arg) {
  __vm_atomic_add(&flag, 1);   /* set the flag, then exit — no wait, no park */
  return 0;
}
int main(void) {
  int h = __vm_thread_spawn(setter, (void *)0, 0);
  while (__vm_atomic_load(&flag) == 0) { }   /* SPIN yield-free until the sibling's store lands */
  __vm_thread_join(h);
  return 7;
}
"#;

#[test]
fn c_yield_free_spin_makes_progress_on_the_cooperative_driver() {
    let i = run_interp_only(SPIN_FLAG_SRC, |_| {});
    assert_eq!(
        i.result,
        vec![Value::I32(7)],
        "tree-walker (preemptive): the spin observed the store"
    );
    let p = run_bytecode_parallel_only(SPIN_FLAG_SRC, |_| {});
    assert_eq!(
        p.result,
        vec![Value::I32(7)],
        "parallel (real OS threads): the spin observed the store"
    );
    let c = run_bytecode_only(SPIN_FLAG_SRC, |_| {});
    assert_eq!(
        c.result,
        vec![Value::I32(7)],
        "coop bytecode: the #1157 preemption quantum let the sibling run so the spin observed the store"
    );
}

const EINTR_CAUGHT_SRC: &str = r#"
long __px_signal(int cap, long signum, long handler);
long __px_kill(int cap, long pid, long sig);
long __px_sigaltstack(int cap, long sp, long size);
long __vm_pipe(int *fds);
long __vm_read(int fd, void *buf, long len);
long __vm_atomic_add(void *p, long v);
long __vm_atomic_load(void *p);
int  __vm_thread_spawn(long (*fn)(long), void *stack, long arg);
long __vm_thread_join(int h);
static char sigstk[16384];
static volatile long fired;
static void handler(int sig) { fired = sig; }
long done;
long raiser(long arg) {
  while (__vm_atomic_load(&done) == 0)
    __px_kill(0, 0, 2);   /* raise SIGINT via the personality (policy: caught -> deliverable) */
  return 0;
}
int main(void) {
  __px_signal(0, 2, (long)handler);          /* catch SIGINT */
  __px_sigaltstack(0, (long)sigstk, 16384);  /* async delivery on */
  int fds[2];
  __vm_pipe(fds);                            /* main holds both ends -> a live writer keeps the read blocked */
  int h = __vm_thread_spawn(raiser, (void *)0, 0);
  char b[8];
  long n = __vm_read(fds[0], b, 8);          /* PARKS; a delivered SIGINT interrupts it -> -EINTR */
  __vm_atomic_add(&done, 1);
  __vm_thread_join(h);
  if (n == -4) return 42;                    /* -EINTR: the caught signal interrupted the blocked read */
  return (int)n;
}
"#;

#[test]
fn c_a_caught_signal_interrupts_a_blocked_capability_read_with_eintr() {
    let e = run_interp_only(EINTR_CAUGHT_SRC, |_| {});
    assert_eq!(
        e.result,
        vec![Value::I32(42)],
        "tree-walker: a caught, delivered SIGINT interrupted the guest's blocked capability read"
    );
    // #1146 slice 2 (parallel) — the `drive_parallel` twin: the raiser is a real OS thread raising
    // SIGINT concurrently while `main`'s thread blocks in the pipe-read poll loop. The loop observes
    // the deliverable, non-SA_RESTART signal (`park_interrupted`), sets its host EINTR flag, and
    // breaks; the re-run completes `-EINTR` at the park site and the handler fires (slice 1) — 42.
    // Before this the parked read polled forever with no writer and the run deadlocked.
    //
    // #1173 — the parallel leg is timing-nondeterministic (the sleep-polled pipe read races the
    // interrupt; the EINTR-vs-EOF ordering under real threads is not yet airtight — condvar doors on
    // the shared pipe backing are the follow-up), so it intermittently returns EOF(0) instead of
    // EINTR(42). Retry a bounded number of times: a genuine regression never yields 42 across all
    // attempts, while the intermittent race is tolerated. The interp (above) and coop bytecode twins
    // stay strict. Drop this retry when #1173's race-free pipe read lands.
    let mut p = run_bytecode_parallel_only(EINTR_CAUGHT_SRC, |_| {});
    for _ in 0..8 {
        if p.result == vec![Value::I32(42)] {
            break;
        }
        p = run_bytecode_parallel_only(EINTR_CAUGHT_SRC, |_| {});
    }
    assert_eq!(
        p.result,
        vec![Value::I32(42)],
        "parallel bytecode: the concurrent raiser's SIGINT broke the blocked read with -EINTR — matching the oracle (retried per #1173)"
    );
}

/// #799 slice 3 — **disposition-gated: an *ignored* signal does NOT interrupt a blocked syscall.** The
/// mirror of the caught case, proving the gate is the personality's policy, not temen's: `main` sets SIGINT
/// to `SIG_IGN` (1) and blocks on `__vm_read`; the raiser raises SIGINT — undeliverable, so the
/// personality's `take_deliverable` returns `None` and the core never interrupts — and *then writes* the
/// pipe, so `main`'s read completes with **data** (3), never `-EINTR` (42).
const EINTR_IGNORED_SRC: &str = r#"
long __px_signal(int cap, long signum, long handler);
long __px_kill(int cap, long pid, long sig);
long __px_sigaltstack(int cap, long sp, long size);
long __vm_pipe(int *fds);
long __vm_read(int fd, void *buf, long len);
long __vm_write(int fd, void *buf, long len);
int  __vm_thread_spawn(long (*fn)(long), void *stack, long arg);
long __vm_thread_join(int h);
static char sigstk[16384];
static char msg[] = "GO\n";
long g_wfd;
long raiser(long arg) {
  volatile long acc = 0;
  for (long i = 0; i < 2000000; i = i + 1) acc = acc + i;  /* let main reach its read and park */
  __px_kill(0, 0, 2);          /* SIGINT is SIG_IGN -> undeliverable -> the core never interrupts */
  __vm_write(g_wfd, msg, 3);   /* so main's read completes with DATA, proving no spurious EINTR */
  return 0;
}
int main(void) {
  __px_signal(0, 2, 1);        /* SIG_IGN(1): ignore SIGINT */
  __px_sigaltstack(0, (long)sigstk, 16384);
  int fds[2];
  __vm_pipe(fds);
  g_wfd = fds[1];
  int h = __vm_thread_spawn(raiser, (void *)0, 0);
  char b[8];
  long n = __vm_read(fds[0], b, 8);  /* PARKS; the ignored SIGINT must NOT interrupt -> woken by the write */
  __vm_thread_join(h);
  if (n == -4) return 42;            /* a wrong EINTR — must not happen */
  return (int)n;                     /* 3: the read completed with data */
}
"#;

#[test]
fn c_an_ignored_signal_does_not_interrupt_a_blocked_capability_read() {
    let e = run_interp_only(EINTR_IGNORED_SRC, |_| {});
    assert_eq!(
        e.result,
        vec![Value::I32(3)],
        "tree-walker: an ignored SIGINT must not interrupt the blocked read; it completed with data"
    );
    // #1146 slice 2 (parallel) — SIG_IGN is not deliverable, so `park_interrupted` is false: the poll
    // loop keeps waiting and the raiser's later write completes the read with DATA — no spurious EINTR.
    let p = run_bytecode_parallel_only(EINTR_IGNORED_SRC, |_| {});
    assert_eq!(
        p.result,
        vec![Value::I32(3)],
        "parallel bytecode: an ignored SIGINT left the blocked read to complete with data — matching the oracle"
    );
}

/// #796 `SA_RESTART` — **a restart-flagged delivery re-issues the blocked read instead of `-EINTR`**:
/// SIGINT is installed via `sigaction` with `SA_RESTART`; the raiser interrupts `main`'s blocked
/// pipe read — the handler runs (fired = 2, promptly: the woken read re-executes and the per-op
/// poll delivers first), but the read RESTARTS (re-parks) instead of surfacing `-EINTR`, and later
/// completes with the raiser's DATA. The exact complement of the `-EINTR` test above — the
/// `sa_flags` bit flips the outcome.
const RESTART_READ_SRC: &str = r#"
long __px_kill(int cap, long pid, long sig);
long __px_sigaltstack(int cap, long sp, long size);
long __px_sigaction(int cap, long signum, long act, long oldact);
long __vm_pipe(int *fds);
long __vm_read(int fd, void *buf, long len);
long __vm_write(int fd, void *buf, long len);
int  __vm_thread_spawn(long (*fn)(long), void *stack, long arg);
long __vm_thread_join(int h);
static char sigstk[16384];
static long act[3];
static char msg[] = "GO\n";
static volatile int fired;
static void handler(int sig) { fired = sig; }
long g_wfd;
long raiser(long arg) {
  volatile long acc = 0;
  for (long i = 0; i < 2000000; i = i + 1) acc = acc + i;  /* let main reach its read and park */
  __px_kill(0, 0, 2);          /* SA_RESTART'd SIGINT: handler runs, the read must RESTART */
  for (long i = 0; i < 2000000; i = i + 1) acc = acc + i;  /* window: an EINTR would surface here */
  __vm_write(g_wfd, msg, 3);   /* the restarted read completes with data */
  return 0;
}
int main(void) {
  act[0] = (long)handler;
  act[1] = 0;
  act[2] = 0x10000000;         /* SA_RESTART */
  __px_sigaction(0, 2, (long)act, 0);
  __px_sigaltstack(0, (long)sigstk, 16384);
  int fds[2];
  __vm_pipe(fds);
  g_wfd = fds[1];
  int h = __vm_thread_spawn(raiser, (void *)0, 0);
  char b[8];
  long n = __vm_read(fds[0], b, 8);  /* PARKS; interrupted -> handler -> RESTARTS -> data */
  __vm_thread_join(h);
  if (n == -4) return 42;            /* -EINTR would mean SA_RESTART was ignored */
  return fired * 10 + (int)n;        /* 23: the handler ran AND the read completed with data */
}
"#;

#[test]
fn c_sa_restart_reissues_an_interrupted_blocked_read() {
    let e = run_interp_only(RESTART_READ_SRC, |_| {});
    assert_eq!(
        e.result,
        vec![Value::I32(23)],
        "tree-walker: the SA_RESTART'd SIGINT ran its handler but the read restarted and returned data"
    );
    // #1146 slice 2 (parallel) — SA_RESTART makes `park_interrupted` false (it honors `syscall_restart`),
    // so the poll loop does NOT break on the signal: the handler runs (fired=2) but the read keeps
    // waiting and the raiser's write completes it with data — result 23, never 42.
    let p = run_bytecode_parallel_only(RESTART_READ_SRC, |_| {});
    assert_eq!(
        p.result,
        vec![Value::I32(23)],
        "parallel bytecode: SA_RESTART kept the read waiting (handler ran, data returned) — matching the oracle"
    );
}

/// #799 slice 5 — **the embedder `^C`: a signal from *outside* the run interrupts a blocked syscall.** The
/// interactive terminal case: the guest catches SIGINT and blocks on a capability `__vm_read` in a
/// **single** fiber — so when it parks (`Blocked::PipeRead`), *every* fiber is parked and no per-op
/// safepoint fires. A background "terminal" thread calls `Posix::raise_signal(SIGINT)` (the embedder's
/// signal authority over the guest — the `Posix` handle shares the personality `Inner` with the running
/// guest); the personality decides it is deliverable and invokes the interp's scheduler-wake closure, so
/// the parked read completes `-EINTR` (sentinel 42). The "terminal" holds fire until the guest's
/// readiness byte lands on stdout — #796 default actions made a pre-handler `^C` FATAL (terminate is
/// SIGINT's `SIG_DFL` action); after the handler is installed each raise stays harmless.
const EINTR_EMBEDDER_SRC: &str = r#"
long __px_signal(int cap, long signum, long handler);
long __px_sigaltstack(int cap, long sp, long size);
long __px_write(int cap, long fd, long buf, long len);
long __vm_pipe(int *fds);
long __vm_read(int fd, void *buf, long len);
static char sigstk[16384];
static volatile long fired;
static void handler(int sig) { fired = sig; }
int main(void) {
  __px_signal(0, 2, (long)handler);          /* catch SIGINT */
  __px_sigaltstack(0, (long)sigstk, 16384);  /* async delivery on */
  __px_write(0, 1, (long)sigstk, 1);         /* readiness byte: the terminal may open fire (#796) */
  int fds[2];
  __vm_pipe(fds);                            /* main holds both ends -> a live writer keeps the read blocked */
  char b[8];
  long n = __vm_read(fds[0], b, 8);          /* single fiber PARKS -> all parked, no safepoint; ^C -> -EINTR */
  if (n == -4) return 42;                    /* -EINTR: the terminal ^C interrupted the blocked read */
  return (int)n;
}
"#;

#[test]
fn c_an_embedder_signal_interrupts_a_blocked_read_ctrl_c() {
    use std::sync::atomic::{AtomicBool, Ordering};
    let ir = c_to_ir(EINTR_EMBEDDER_SRC);
    let raw = parse_module_raw(&ir).unwrap_or_else(|e| panic!("parse IR failed: {e:?}\n{ir}"));
    let win = 1u64
        << raw
            .memory
            .expect("the frontend declares a window")
            .size_log2;
    let mut ih = Host::new();
    let (posix, px) = setup(&mut ih, win);
    verify_module(&raw).unwrap_or_else(|e| panic!("verify failed: {e:?}\n{ir}"));
    bind_shim(&raw, &mut ih, px);
    // A background "terminal" raises SIGINT until the guest takes EINTR and the run returns. `raise_signal`
    // shares the personality `Inner` with the running guest (via the `Posix` handle), so it reaches it
    // mid-run and — when deliverable — pokes the interp's scheduler-wake to interrupt the parked read.
    let posix2 = posix.clone();
    let done = std::sync::Arc::new(AtomicBool::new(false));
    let done2 = std::sync::Arc::clone(&done);
    let terminal = std::thread::spawn(move || {
        // Hold fire until the guest's readiness byte (#796 — a pre-handler ^C is fatal).
        while !done2.load(Ordering::Relaxed) && posix2.stdout().is_empty() {
            std::thread::yield_now();
        }
        while !done2.load(Ordering::Relaxed) {
            posix2.raise_signal(2);
            std::thread::yield_now();
        }
    });
    let mut fuel = 200_000_000u64;
    let r = run_with_host(&raw, 0, &[], &mut fuel, &mut ih);
    done.store(true, Ordering::Relaxed);
    terminal.join().unwrap();
    match r {
        Ok(v) => assert_eq!(
            v.as_slice(),
            [Value::I32(42)],
            "an embedder ^C (raise_signal from another thread) interrupted the guest's blocked read"
        ),
        Err(e) => panic!("interp trapped: {e:?}\n{ir}"),
    }
}

/// #796 slice D — **the interactive prompt case: a `^C` interrupts a blocked STDIN read.** The
/// guest catches SIGINT and blocks reading a granted `Stream{In}` with **interactive stdin**
/// (`stdin_block = true`: no data ⇒ park, not EOF) — the `Blocked::CapRead` park, a different
/// park kind from the pipe reads the earlier tests cover. The embedder "terminal" raises SIGINT;
/// the sweep completes the parked stream read with `-EINTR` (the `cap_revoke` injection shape)
/// and the handler runs. This is a shell sitting at its prompt taking a `^C`.
#[test]
fn c_a_ctrl_c_interrupts_a_blocked_interactive_stdin_read() {
    use std::sync::atomic::{AtomicBool, Ordering};
    let mut ih = Host::new();
    // Grant the interactive stdin FIRST (independent of the window size) so its handle can be
    // spliced into the guest source; the personality lands after, once the window is known.
    let stdin_h = ih.grant_stream(temen_interp::StreamRole::In);
    ih.stdin_block = true; // interactive: an exhausted stdin parks the read instead of EOF
    let src = format!(
        r#"
long __px_signal(int cap, long signum, long handler);
long __px_sigaltstack(int cap, long sp, long size);
long __px_write(int cap, long fd, long buf, long len);
long __vm_read(int fd, void *buf, long len);
static char sigstk[16384];
static volatile long fired;
static void handler(int sig) {{ fired = sig; }}
int main(void) {{
  __px_signal(0, 2, (long)handler);
  __px_sigaltstack(0, (long)sigstk, 16384);
  __px_write(0, 1, (long)sigstk, 1);   /* readiness byte: the terminal may open fire */
  char b[8];
  long n = __vm_read({stdin_h}, b, 8); /* the PROMPT: interactive stdin, no data -> PARKS */
  long i = 0;
  while (!fired) {{ i = i + 1; if (i > 100000000) return -1; }}  /* handler landing window */
  if (n == -4) return 40 + fired;      /* 42: -EINTR and the SIGINT handler ran */
  return (int)n;
}}
"#
    );
    let ir = c_to_ir(&src);
    let raw = parse_module_raw(&ir).unwrap_or_else(|e| panic!("parse IR failed: {e:?}\n{ir}"));
    let win = 1u64
        << raw
            .memory
            .expect("the frontend declares a window")
            .size_log2;
    let (posix, px) = setup(&mut ih, win);
    verify_module(&raw).unwrap_or_else(|e| panic!("verify failed: {e:?}\n{ir}"));
    bind_shim(&raw, &mut ih, px);
    let posix2 = posix.clone();
    let done = std::sync::Arc::new(AtomicBool::new(false));
    let done2 = std::sync::Arc::clone(&done);
    let terminal = std::thread::spawn(move || {
        // Hold fire until the guest's readiness byte (#796 — a pre-handler ^C is fatal).
        while !done2.load(Ordering::Relaxed) && posix2.stdout().is_empty() {
            std::thread::yield_now();
        }
        while !done2.load(Ordering::Relaxed) {
            posix2.raise_signal(2);
            std::thread::yield_now();
        }
    });
    let mut fuel = 200_000_000u64;
    let r = run_with_host(&raw, 0, &[], &mut fuel, &mut ih);
    done.store(true, Ordering::Relaxed);
    terminal.join().unwrap();
    match r {
        Ok(v) => assert_eq!(
            v.as_slice(),
            [Value::I32(42)],
            "the ^C completed the blocked interactive stdin read with -EINTR and ran the handler"
        ),
        Err(e) => panic!("interp trapped: {e:?}\n{ir}"),
    }
}

/// #799 — **the two-world merge, witnessed in one program**: a PERSONALITY-ONLY guest (no
/// manager, no offers, no powerbox topology — the exact link shape bash gets) runs return-twice
/// `fork()` and a blocking `waitpid()` through nothing but named personality imports. The fork
/// rides the caller-request door (`ParkEvent::ForkSelf` → the core's fork engine); the twin is
/// table-registered by the fork factory at mint (so `getppid` and `waitpid` know it instantly);
/// the parent's single un-looped `waitpid` blocks through the twin's slow spin. Interp-only (the
/// fork/park doors are interp-tier; on other tiers the op answers `-ENOSYS`, "fork unavailable").
#[test]
fn c_a_personality_only_guest_forks_and_blocks_in_waitpid() {
    let src = r#"
long __px_fork(int cap, long a);
long __px_waitpid(int cap, long pid, long status, long opts);
long __px_getpid(int cap, long a);
long __px_getppid(int cap, long a);
static int status;
static volatile long acc;
static long me;
static long pid;
static long h;
int main(void) {
  me = __px_getpid(0, 0);
  pid = __px_fork(0, 0);                       /* return-twice, no offer anywhere in sight */
  if (pid < 0) return 1;                       /* -ENOSYS/-EAGAIN: the door failed */
  if (pid == 0) {
    if (__px_getppid(0, 0) != me) return 9;    /* the twin knows its forking parent */
    for (long i = 0; i < 30000; i = i + 1) acc = acc + 1;  /* slow: the parent must BLOCK */
    return 7;
  }
  if (pid == me) return 2;                     /* the twin got a fresh pid */
  h = __px_waitpid(0, pid, (long)&status, 0);  /* ONE call — blocks until the twin exits */
  if (h != pid) return 3;
  if ((status & 0x7f) != 0) return 4;          /* clean exit, not a signal death */
  if (((status >> 8) & 0xff) != 7) return 5;   /* WEXITSTATUS = the twin's 7 */
  return 42;
}
"#;
    let e = run_interp_only(src, |_| {});
    assert_eq!(
        e.result,
        vec![Value::I32(42)],
        "a personality-only guest forked (return-twice) and blocked in waitpid until the twin's \
         real exit — the capability fork engine reached through named imports alone"
    );
}

/// #1080 rung 3 — the **same fork + blocking-waitpid witness on the bytecode engine** (the playground
/// tier). Exercises the ported personality-fork park engine end to end: `fork()` (`ParkEvent::ForkSelf`
/// → the cooperative driver self-forks a twin), the twin's slow spin, its exit firing the personality
/// exit hooks (Live → Zombie), and the parent's blocking `waitpid` (`ParkEvent::TaskExit` → park →
/// re-execute) reading the retired twin's `WEXITSTATUS`. Must agree with the tree-walker oracle above.
#[test]
fn c_a_personality_fork_and_waitpid_on_bytecode() {
    let src = r#"
long __px_fork(int cap, long a);
long __px_waitpid(int cap, long pid, long status, long opts);
long __px_getpid(int cap, long a);
long __px_getppid(int cap, long a);
static int status;
static volatile long acc;
static long me;
static long pid;
static long h;
int main(void) {
  me = __px_getpid(0, 0);
  pid = __px_fork(0, 0);
  if (pid < 0) return 1;
  if (pid == 0) {
    if (__px_getppid(0, 0) != me) return 9;
    for (long i = 0; i < 30000; i = i + 1) acc = acc + 1;
    return 7;
  }
  if (pid == me) return 2;
  h = __px_waitpid(0, pid, (long)&status, 0);
  if (h != pid) return 3;
  if ((status & 0x7f) != 0) return 4;
  if (((status >> 8) & 0xff) != 7) return 5;
  return 42;
}
"#;
    let e = run_bytecode_only(src, |_| {});
    assert_eq!(
        e.result,
        vec![Value::I32(42)],
        "the bytecode engine forked, blocked in waitpid, and read the twin's exit 7 — matching the \
         tree-walker (#1080 rung 3: fork/reap park + exit-hook firing on the cooperative driver)"
    );
}

/// #798 — **job-control stop/continue across every engine** (the `^Z` semantics bash's foreground
/// wait relies on). A parent forks a child that catches signal 10 (a `sigcheck` token) and spins on
/// it; the parent `SIGTSTP`s the child (as a terminal `^Z` does to the foreground job) and
/// `waitpid(WUNTRACED)` reports the **stop** (`(20 << 8) | 0x7f`, `WIFSTOPPED`) — once. A signal sent
/// **while the child is stopped** is HELD, proven by a long busy-wait: a still-running child would
/// consume its token and exit, but `waitpid(WNOHANG)` keeps returning `-ECHILD` (alive, not exited).
/// `SIGCONT` resumes it, the held 10 delivers (`sigcheck → 7`), the child exits `7`, and the parent
/// reaps `WEXITSTATUS == 7`. This is the `ctrl_z_stops_a_forked_child_and_fg_resumes_it` oracle
/// (`c_fork.rs`, VM-cap band) carried onto the `__px_*` personality band and **differentialled across
/// the tree-walker, the cooperative bytecode driver (the browser tier), and the parallel driver** —
/// the dual-driver principle: the stop/continue bookkeeping (`stopped_sig`/`stop_fresh`/`cont_fresh`,
/// held-while-stopped) must agree on all three. All return 42.
///
/// This locks the *tractable* half of job control. Stopping a job that is **parked** when the stop
/// arrives (e.g. a foreground `cat` blocked on `read`, with the shell already blocked in
/// `waitpid(WUNTRACED)`) — the interactive browser `^Z` — needs the stop to interrupt the park and
/// wake the shell, which is a cross-engine gap tracked in #1171 (the tree-walker hangs on it too).
const STOP_CONT_SRC: &str = r#"
long __px_fork(int cap, long a);
long __px_waitpid(int cap, long pid, long status, long opts);
long __px_kill(int cap, long pid, long sig);
long __px_signal(int cap, long signum, long handler);
long __px_sigcheck(int cap, long a);
static int status;
static long pid;
static long i;
static volatile long sink;
int main(void) {
  /* Pre-install a caught token (7) for signal 10, inherited by the twin, so the held-while-stopped
   * 10 is a caught delivery (not SIG_DFL, which would terminate at continue instead of delivering). */
  __px_signal(0, 10, 7);
  pid = __px_fork(0, 0);
  if (pid < 0) return 1;
  if (pid == 0) {
    __px_signal(0, 10, 7);                     /* the twin re-installs its own token */
    while (__px_sigcheck(0, 0) != 7);          /* spin until the held 10 delivers post-continue */
    return 7;
  }
  if (__px_kill(0, pid, 20) != 0) return 2;         /* SIGTSTP the child (^Z): default -> stop */
  long h = __px_waitpid(0, pid, (long)&status, 2);  /* waitpid(pid, WUNTRACED): report the stop */
  if (h != pid) return 3;
  if ((status & 0xff) != 0x7f) return 4;            /* WIFSTOPPED: low byte 0x7f */
  if (((status >> 8) & 0xff) != 20) return 5;       /* the stop signal = SIGTSTP(20) */
  if (__px_kill(0, pid, 10) != 0) return 6;         /* the 10 lands while stopped: HELD */
  for (i = 0; i < 200000; i = i + 1) sink = i;      /* every chance to run, were it runnable */
  if (__px_waitpid(0, pid, (long)&status, 1) != -10) return 7;  /* WNOHANG: still alive (-ECHILD), truly stopped */
  if (__px_kill(0, pid, 18) != 0) return 8;         /* SIGCONT: resume, the held 10 delivers */
  h = __px_waitpid(0, pid, (long)&status, 0);       /* reap the resumed child's real exit */
  if (h != pid) return 9;
  if ((status & 0x7f) != 0) return 11;              /* a clean exit, not a signal death */
  if (((status >> 8) & 0xff) != 7) return 10;       /* WEXITSTATUS = the child's 7 */
  return 42;
}
"#;

#[test]
fn c_job_control_stop_continue_reports_wuntraced_then_reaps_the_exit() {
    let e = run_interp_only(STOP_CONT_SRC, |_| {});
    assert_eq!(
        e.result,
        vec![Value::I32(42)],
        "tree-walker: SIGTSTP stopped the child (waitpid WUNTRACED reported it), the held signal stayed \
         held through the busy-wait, SIGCONT resumed it, and the parent reaped exit 7"
    );
    let b = run_bytecode_only(STOP_CONT_SRC, |_| {});
    assert_eq!(
        b.result,
        vec![Value::I32(42)],
        "coop bytecode (the browser tier): the stop/continue bookkeeping reported the stop via WUNTRACED, \
         held the mid-stop signal, resumed on SIGCONT, and reaped exit 7 — matching the oracle"
    );
    let p = run_bytecode_parallel_only(STOP_CONT_SRC, |_| {});
    assert_eq!(
        p.result,
        vec![Value::I32(42)],
        "parallel driver: same stop/continue semantics over real OS threads — matching the oracle"
    );
}

/// #1080 rung 3 — the **any-child** blocking wait (`waitpid(-1)` → `ParkEvent::TaskExitAny` → the
/// driver's `BlockedReapPersonality { child: None }` park) on the bytecode engine. The parent forks a
/// slow twin and blocks in `waitpid(-1)`; the settle scan wakes it when ANY forked child completes, and
/// the re-executed wait returns the twin's pid + `WEXITSTATUS`. Differentialled against the tree-walker
/// (both engines must return 42) — covers the wildcard reap path job control uses.
#[test]
fn c_a_personality_fork_and_waitpid_any_child_differential() {
    let src = r#"
long __px_fork(int cap, long a);
long __px_waitpid(int cap, long pid, long status, long opts);
static int status;
static volatile long acc;
static long pid;
int main(void) {
  pid = __px_fork(0, 0);
  if (pid < 0) return 1;
  if (pid == 0) {
    for (long i = 0; i < 30000; i = i + 1) acc = acc + 1;  /* slow: the parent must BLOCK */
    return 7;
  }
  long h = __px_waitpid(0, -1, (long)&status, 0);          /* ANY child, blocking */
  if (h != pid) return 3;                                   /* reaped the twin, got its pid */
  if ((status & 0x7f) != 0) return 4;                       /* clean exit, not a signal death */
  if (((status >> 8) & 0xff) != 7) return 5;                /* WEXITSTATUS = the twin's 7 */
  return 42;
}
"#;
    let interp = run_interp_only(src, |_| {});
    assert_eq!(
        interp.result,
        vec![Value::I32(42)],
        "tree-walker any-child reap"
    );
    let byte = run_bytecode_only(src, |_| {});
    assert_eq!(
        byte.result,
        vec![Value::I32(42)],
        "bytecode any-child (waitpid(-1)) reap matched the oracle"
    );
}

/// #1080 pipeline rung — a **nested fork + blocking wait**: a forked twin ITSELF forks a grandchild
/// and blocks in `waitpid` for it, then the root reaps the twin. This is exactly the shape a real-bash
/// pipeline stage takes (the shell forks a subshell per stage; a stage running an external command forks
/// again and waits), and it is the first differential where a **forked child** — not the root — is the
/// one that parks in `waitpid`. On the bytecode cooperative engine the reproduction target is the
/// browser pipeline deadlock (`echo | cat`), where the subshell's `waitpid` busy-loops on `-ECHILD`
/// instead of parking. Both engines must return 42.
#[test]
fn c_a_nested_fork_twin_blocks_on_grandchild_differential() {
    let src = r#"
long __px_fork(int cap, long a);
long __px_waitpid(int cap, long pid, long status, long opts);
static int st1;
static int st2;
static volatile long acc;
static long p1;
static long p2;
int main(void) {
  p1 = __px_fork(0, 0);                       /* root forks twin1 (the "subshell") */
  if (p1 < 0) return 1;
  if (p1 == 0) {
    p2 = __px_fork(0, 0);                      /* twin1 forks a grandchild (the "command") */
    if (p2 < 0) return 21;
    if (p2 == 0) {
      for (long i = 0; i < 30000; i = i + 1) acc = acc + 1;  /* slow: twin1 must BLOCK */
      return 7;                                /* grandchild exits 7 */
    }
    long h2 = __px_waitpid(0, p2, (long)&st2, 0);  /* twin1 blocks on its grandchild */
    if (h2 != p2) return 22;
    if ((st2 & 0x7f) != 0) return 23;
    if (((st2 >> 8) & 0xff) != 7) return 24;
    return 11;                                 /* twin1 exits 11 */
  }
  long h1 = __px_waitpid(0, p1, (long)&st1, 0);    /* root blocks on twin1 */
  if (h1 != p1) return 3;
  if ((st1 & 0x7f) != 0) return 4;
  if (((st1 >> 8) & 0xff) != 11) return 5;
  return 42;
}
"#;
    let interp = run_interp_only(src, |_| {});
    assert_eq!(
        interp.result,
        vec![Value::I32(42)],
        "tree-walker: twin1 forked a grandchild, blocked in waitpid for it, then the root reaped twin1"
    );
    let byte = run_bytecode_only(src, |_| {});
    assert_eq!(
        byte.result,
        vec![Value::I32(42)],
        "bytecode: a forked twin's OWN blocking waitpid on its grandchild must park + reap like the \
         oracle (the pipeline-subshell shape behind the `echo | cat` browser deadlock)"
    );
}

/// #748 rung 0+1 — personality `fork()` + blocking `waitpid(pid)` on the **parallel** bytecode
/// driver: the twin runs on a real OS thread over a private window copy; the parent's `waitpid`
/// either finds the zombie directly (the twin raced ahead) or blocks on the fork-exit condvar and
/// re-issues on the wake. Same guest, same 42, as the cooperative and tree-walker oracles.
#[test]
fn c_a_personality_fork_and_waitpid_on_parallel_driver() {
    let src = r#"
long __px_fork(int cap, long a);
long __px_waitpid(int cap, long pid, long status, long opts);
long __px_getpid(int cap, long a);
long __px_getppid(int cap, long a);
static int status;
static volatile long acc;
static long me;
static long pid;
static long h;
int main(void) {
  me = __px_getpid(0, 0);
  pid = __px_fork(0, 0);
  if (pid < 0) return 1;
  if (pid == 0) {
    if (__px_getppid(0, 0) != me) return 9;
    for (long i = 0; i < 30000; i = i + 1) acc = acc + 1;
    return 7;
  }
  if (pid == me) return 2;
  h = __px_waitpid(0, pid, (long)&status, 0);
  if (h != pid) return 3;
  if ((status & 0x7f) != 0) return 4;
  if (((status >> 8) & 0xff) != 7) return 5;
  return 42;
}
"#;
    let e = run_bytecode_parallel_only(src, |_| {});
    assert_eq!(
        e.result,
        vec![Value::I32(42)],
        "the parallel driver forked a real-OS-thread twin, blocked in waitpid, and read the twin's \
         exit 7 — matching the cooperative + tree-walker oracles (#748 rung 0+1)"
    );
}

/// #748 rung 1 — the **any-child** blocking wait (`waitpid(-1)`) on the parallel driver: the wake is
/// the exit-generation condvar ([`ThreadRegistry::wait_fork_exit`]), whose newer-than-consumed
/// protocol is the condvar analogue of the cooperative driver's consumed-Done-twin prune (the #1112
/// livelock class must not re-appear here). Both engines return 42.
#[test]
fn c_a_personality_fork_and_waitpid_any_child_on_parallel_driver() {
    let src = r#"
long __px_fork(int cap, long a);
long __px_waitpid(int cap, long pid, long status, long opts);
static int status;
static volatile long acc;
static long pid;
int main(void) {
  pid = __px_fork(0, 0);
  if (pid < 0) return 1;
  if (pid == 0) {
    for (long i = 0; i < 30000; i = i + 1) acc = acc + 1;  /* slow: the parent usually BLOCKS */
    return 7;
  }
  long h = __px_waitpid(0, -1, (long)&status, 0);          /* ANY child, blocking */
  if (h != pid) return 3;
  if ((status & 0x7f) != 0) return 4;
  if (((status >> 8) & 0xff) != 7) return 5;
  return 42;
}
"#;
    let e = run_bytecode_parallel_only(src, |_| {});
    assert_eq!(
        e.result,
        vec![Value::I32(42)],
        "parallel-driver any-child (waitpid(-1)) reap matched the oracles"
    );
}

/// #748 rung 0+1 × nested fork — a forked twin ITSELF forks a grandchild and blocks in `waitpid` for
/// it (the bash pipeline-subshell shape), all on real OS threads: the twin's own park door
/// (`wire_park_door` at fork, the #1112 lesson) and its own powerbox cell make the grandchild fork +
/// nested wait work exactly as on the cooperative driver. Both engines return 42.
#[test]
fn c_a_nested_fork_twin_blocks_on_grandchild_on_parallel_driver() {
    let src = r#"
long __px_fork(int cap, long a);
long __px_waitpid(int cap, long pid, long status, long opts);
static int st1;
static int st2;
static volatile long acc;
static long p1;
static long p2;
int main(void) {
  p1 = __px_fork(0, 0);                       /* root forks twin1 (the "subshell") */
  if (p1 < 0) return 1;
  if (p1 == 0) {
    p2 = __px_fork(0, 0);                      /* twin1 forks a grandchild (the "command") */
    if (p2 < 0) return 21;
    if (p2 == 0) {
      for (long i = 0; i < 30000; i = i + 1) acc = acc + 1;
      return 7;                                /* grandchild exits 7 */
    }
    long h2 = __px_waitpid(0, p2, (long)&st2, 0);  /* twin1 blocks on its grandchild */
    if (h2 != p2) return 22;
    if ((st2 & 0x7f) != 0) return 23;
    if (((st2 >> 8) & 0xff) != 7) return 24;
    return 11;                                 /* twin1 exits 11 */
  }
  long h1 = __px_waitpid(0, p1, (long)&st1, 0);    /* root blocks on twin1 */
  if (h1 != p1) return 3;
  if ((st1 & 0x7f) != 0) return 4;
  if (((st1 >> 8) & 0xff) != 11) return 5;
  return 42;
}
"#;
    let e = run_bytecode_parallel_only(src, |_| {});
    assert_eq!(
        e.result,
        vec![Value::I32(42)],
        "parallel driver: a forked twin's OWN blocking waitpid on its grandchild parked + reaped \
         like the cooperative + tree-walker oracles"
    );
}

/// #1080 pipeline rung — **reap ownership**: `waitpid(-1)` must reap only the CALLER's own children,
/// never a sibling's. The root forks two twins; twinA exits (a zombie owned by the root), and twinB —
/// which has no children — calls `waitpid(-1)`. POSIX: twinB gets `-ECHILD` and does NOT steal twinA's
/// zombie, so the root still reaps twinA itself. This is the shape behind the browser `echo | cat`
/// deadlock: a pipeline stage's stray `waitpid(-1)` stealing a sibling stage's reap from the shell.
#[test]
fn c_a_waitpid_any_child_does_not_steal_a_siblings_reap_differential() {
    let src = r#"
long __px_fork(int cap, long a);
long __px_waitpid(int cap, long pid, long status, long opts);
static int stA;
static int stB;
static volatile long acc;
static long pa;
static long pb;
int main(void) {
  pa = __px_fork(0, 0);
  if (pa < 0) return 1;
  if (pa == 0) return 7;                        /* twinA: exit 7 -> a zombie owned by the ROOT */
  pb = __px_fork(0, 0);
  if (pb < 0) return 2;
  if (pb == 0) {
    for (long i = 0; i < 20000; i = i + 1) acc = acc + 1;  /* let twinA retire to a zombie first */
    long h = __px_waitpid(0, -1, (long)&stB, 0);           /* twinB has NO children */
    if (h > 0) return 20 + (int)h;                          /* BUG: twinB stole a sibling's zombie */
    return 11;                                              /* h < 0 (-ECHILD): correct */
  }
  long hb = __px_waitpid(0, pb, (long)&stB, 0);   /* root blocks on twinB FIRST: twinA becomes a
                                                     zombie while twinB spins, so twinB's waitpid(-1)
                                                     below has a stealable sibling zombie present */
  if (hb != pb) return 5;
  if (((stB >> 8) & 0xff) != 11) return 6;        /* twinB returned 11 (did not steal twinA) */
  long ha = __px_waitpid(0, pa, (long)&stA, 0);   /* root reaps twinA — still there iff not stolen */
  if (ha != pa) return 3;
  if (((stA >> 8) & 0xff) != 7) return 4;
  return 42;
}
"#;
    let interp = run_interp_only(src, |_| {});
    assert_eq!(
        interp.result,
        vec![Value::I32(42)],
        "tree-walker: waitpid(-1) in a childless sibling does not steal the root's zombie"
    );
    let byte = run_bytecode_only(src, |_| {});
    assert_eq!(
        byte.result,
        vec![Value::I32(42)],
        "bytecode: waitpid(-1) must be scoped to the caller's own children (no sibling reap-stealing)"
    );
}

/// #799 — **fork × default-actions × blocking-waitpid, personality-only**: the parent forks a
/// runaway twin (spins forever, no handlers), kills it with an unhandled `SIGTERM` (#796's
/// default-terminate through the kill door — the twin's doors minted at fork), and the blocking
/// `waitpid` wakes at the death reporting `WIFSIGNALED(15)`. Three subsystems from three
/// different slices composing in one guest with no test-specific wiring.
#[test]
fn c_a_forked_twin_dies_by_unhandled_sigterm_and_the_blocked_wait_reports_it() {
    let src = r#"
long __px_fork(int cap, long a);
long __px_waitpid(int cap, long pid, long status, long opts);
long __px_kill(int cap, long pid, long sig);
static int status;
static volatile long acc;
static long pid;
static long h;
int main(void) {
  pid = __px_fork(0, 0);
  if (pid < 0) return 1;
  if (pid == 0) {
    while (1) acc = acc + 1;                   /* runaway: no handler, no exit */
  }
  if (__px_kill(0, pid, 15) != 0) return 2;    /* SIGTERM: SIG_DFL = terminate the twin */
  h = __px_waitpid(0, pid, (long)&status, 0);  /* blocks until the kill lands */
  if (h != pid) return 3;
  if ((status & 0x7f) != 15) return 4;         /* WIFSIGNALED: the terminating signal */
  if (((status >> 8) & 0xff) != 0) return 5;   /* not an exit-code encode */
  return 42;
}
"#;
    let e = run_interp_only(src, |_| {});
    assert_eq!(
        e.result,
        vec![Value::I32(42)],
        "kill(twin, SIGTERM) terminated the runaway fork twin and the blocked personality \
         waitpid reported WIFSIGNALED(15)"
    );
}

/// #799 — **personality pipes are inherited across `fork`** (the audit's works-today witness):
/// the fd table clones share the pipe buffer (`Arc`-backed open file descriptions, POSIX), so
/// bytes the parent wrote before forking are drained by the twin through ITS copy of the read
/// end, and `dup2` re-plumbing survives the fork. This is the *sequential* pattern; the
/// concurrent write-while-blocked-reader pattern needs the personality-pipe blocking rework
/// (false-EOF-on-empty today — tracked on #799, the pipe-unification slice).
#[test]
fn c_personality_pipes_are_inherited_across_fork() {
    let src = r#"
long __px_fork(int cap, long a);
long __px_waitpid(int cap, long pid, long status, long opts);
long __px_pipe(int cap, long fdp);
long __px_write(int cap, long fd, long buf, long len);
long __px_read(int cap, long fd, long buf, long len);
static int fds[2];
static int status;
static char b[8];
static long pid;
static long h;
int main(void) {
  if (__px_pipe(0, (long)fds) != 0) return 1;
  if (__px_write(0, fds[1], (long)"hi!", 3) != 3) return 2;  /* parent fills BEFORE forking */
  pid = __px_fork(0, 0);
  if (pid < 0) return 3;
  if (pid == 0) {
    long n = __px_read(0, fds[0], (long)b, 8);   /* the twin drains ITS inherited copy */
    if (n != 3) return 8;
    if (b[0] != 'h' || b[1] != 'i' || b[2] != '!') return 9;
    return 7;
  }
  h = __px_waitpid(0, pid, (long)&status, 0);    /* blocking waitpid (#953) */
  if (h != pid) return 4;
  if (((status >> 8) & 0xff) != 7) return 5;     /* the twin saw the parent's bytes */
  return 42;
}
"#;
    let e = run_interp_only(src, |_| {});
    assert_eq!(
        e.result,
        vec![Value::I32(42)],
        "the fork twin drained the parent's pre-fork pipe bytes through its inherited fd copy"
    );
}

/// #1080 rung 4 — a **blocking pipe read across fork on the bytecode engine** (the `echo … | cat`
/// shape): the parent forks a writer twin and closes its own write copy; the twin spins slow then
/// writes and exits. The parent's first `read` BLOCKS on the empty FIFO (`Blocked::PipeRead` → the
/// cooperative driver's `BlockedPipeRead`, woken by the settle-scan readiness poll when the twin's
/// bytes land); the second `read` BLOCKS then EOFs (0) when the twin's exit drops its write end
/// (writers → 0). Differentialled against the tree-walker (both engines return 42).
#[test]
fn c_a_blocking_pipe_read_across_fork_on_bytecode() {
    // Uses the #972 CorePipe (`PIPE_SHIM`: `pipe`/`read`/`write`/`close` over `__vm_*` with parking) —
    // bash's actual pipe path — not the personality `__px_pipe` (no writer-count blocking).
    let src = format!(
        "{WIN_PAD_17}{PIPE_SHIM}\n\
long __px_fork(int cap, long a);\n\
long __px_waitpid(int cap, long pid, long status, long opts);\n\
static int fds[2];\n\
static int status;\n\
static char b[8];\n\
static volatile long acc;\n\
static long pid;\n\
int main(void) {{\n\
  if (pipe(fds) != 0) return 1;\n\
  pid = __px_fork(0, 0);\n\
  if (pid < 0) return 2;\n\
  if (pid == 0) {{\n\
    close(fds[0]);                                       /* twin: drop the read end */\n\
    for (long i = 0; i < 30000; i = i + 1) acc = acc + 1; /* slow: the parent's read must BLOCK */\n\
    write(fds[1], \"hi!\", 3);\n\
    return 7;                                            /* exit closes the twin's write end -> EOF */\n\
  }}\n\
  long n = read(fds[0], b, 8);                           /* BLOCKS (parent+twin hold write ends) */\n\
  if (n != 3) return 300 + (int)n;\n\
  if (b[0] != 'h' || b[1] != 'i' || b[2] != '!') return 4;\n\
  close(fds[1]);                                         /* drop the parent's write end; the twin exited */\n\
  long m = read(fds[0], b, 8);                           /* EOF (0): no writers remain */\n\
  if (m != 0) return 400 + (int)m;\n\
  if (__px_waitpid(0, pid, (long)&status, 0) != pid) return 6;\n\
  if (((status >> 8) & 0xff) != 7) return 8;\n\
  return 42;\n\
}}\n"
    );
    let interp = run_interp_only(&src, |_| {});
    assert_eq!(
        interp.result,
        vec![Value::I32(42)],
        "tree-walker blocking pipe read across fork"
    );
    let byte = run_bytecode_only(&src, |_| {});
    assert_eq!(
        byte.result,
        vec![Value::I32(42)],
        "bytecode: blocking pipe read parked + woken by the writer twin, EOF on its exit — matching the oracle"
    );
    // #748 rung 3 — the same on the **parallel driver**: the parent's read blocks its own OS thread
    // (the level-triggered readiness poll) until the twin thread's bytes land, then EOFs when the
    // twin's exit releases its write end.
    let par = run_bytecode_parallel_only(&src, |_| {});
    assert_eq!(
        par.result,
        vec![Value::I32(42)],
        "parallel driver: blocking pipe read across a real fork thread + EOF on twin exit, matching both oracles"
    );
}

/// #1080 rung 4 (backpressure) — a **blocking pipe WRITE across fork on the bytecode engine** (the
/// `yes | head` shape's backpressure): the writer twin pushes 96 KiB (> `PIPE_CAP` = 64 KiB) into the
/// FIFO, so once it fills the twin PARKS (`Blocked::PipeWrite` → the cooperative driver's
/// `BlockedPipeWrite`, re-admitted by the settle poll when the reader opens room); the parent drains
/// all of it. Both engines return 42 (the full 96 KiB round-tripped only because the write parked +
/// resumed — a lost wake would deadlock or short the count).
#[test]
fn c_a_blocking_pipe_write_backpressure_across_fork_on_bytecode() {
    let src = format!(
        "{WIN_PAD_17}{PIPE_SHIM}\n\
long __px_fork(int cap, long a);\n\
long __px_waitpid(int cap, long pid, long status, long opts);\n\
static int fds[2];\n\
static int status;\n\
static char buf[4096];\n\
static char rb[4096];\n\
static long pid;\n\
int main(void) {{\n\
  if (pipe(fds) != 0) return 1;\n\
  for (int i = 0; i < 4096; i = i + 1) buf[i] = 'x';\n\
  pid = __px_fork(0, 0);\n\
  if (pid < 0) return 2;\n\
  if (pid == 0) {{\n\
    close(fds[0]);                                       /* twin: writer only */\n\
    long total = 0;\n\
    while (total < 98304) {{                             /* 96 KiB > PIPE_CAP -> PARKS when full */\n\
      long want = 98304 - total; if (want > 4096) want = 4096;\n\
      long w = write(fds[1], buf, want);                /* partial near-full; parks at full */\n\
      if (w <= 0) return 50;\n\
      total = total + w;\n\
    }}\n\
    return 7;\n\
  }}\n\
  close(fds[1]);                                         /* parent: reader only */\n\
  long got = 0;\n\
  for (;;) {{\n\
    long n = read(fds[0], rb, 4096);\n\
    if (n <= 0) break;                                  /* EOF when the twin exits */\n\
    got = got + n;\n\
  }}\n\
  if (got != 98304) return 300 + (int)(got >> 12);\n\
  if (__px_waitpid(0, pid, (long)&status, 0) != pid) return 6;\n\
  if (((status >> 8) & 0xff) != 7) return 8;\n\
  return 42;\n\
}}\n"
    );
    let interp = run_interp_only(&src, |_| {});
    assert_eq!(
        interp.result,
        vec![Value::I32(42)],
        "tree-walker pipe-write backpressure across fork"
    );
    let byte = run_bytecode_only(&src, |_| {});
    assert_eq!(
        byte.result,
        vec![Value::I32(42)],
        "bytecode: the writer twin parked on the full FIFO and resumed as the parent drained — 96 KiB round-tripped"
    );
    // #748 rung 3 — backpressure on the **parallel driver**: the writer twin's OS thread blocks on
    // the full FIFO while the parent drains concurrently; a lost wake would deadlock the run.
    let par = run_bytecode_parallel_only(&src, |_| {});
    assert_eq!(
        par.result,
        vec![Value::I32(42)],
        "parallel driver: the writer thread parked at PIPE_CAP and resumed as the parent drained — 96 KiB round-tripped"
    );
}

/// #1080 rung 4 (EPIPE) — a parked writer **wakes to `-EPIPE` when the reader closes** (the `yes | head`
/// tail): the writer twin fills the FIFO and PARKS; the parent drains one chunk then closes the read end
/// (readers → 0). The settle poll re-admits the parked writer via `pipe_write_ready`'s readers-gone arm,
/// and its re-issued `write` returns `-EPIPE` (the twin ignores SIGPIPE, so it sees the errno, not death)
/// — the reader-gone wake path the backpressure test does not hit. Differentialled against the oracle.
#[test]
fn c_a_parked_writer_epipes_when_the_reader_closes_on_bytecode() {
    let src = format!(
        "{WIN_PAD_17}{PIPE_SHIM}\n\
long __px_fork(int cap, long a);\n\
long __px_waitpid(int cap, long pid, long status, long opts);\n\
long __px_signal(int cap, long signum, long handler);\n\
static int fds[2];\n\
static int status;\n\
static char buf[4096];\n\
static char rb[4096];\n\
static long pid;\n\
int main(void) {{\n\
  if (pipe(fds) != 0) return 1;\n\
  pid = __px_fork(0, 0);\n\
  if (pid < 0) return 2;\n\
  if (pid == 0) {{\n\
    __px_signal(0, 13, 1);                              /* SIG_IGN SIGPIPE: write returns -EPIPE */\n\
    close(fds[0]);\n\
    for (;;) {{\n\
      long w = write(fds[1], buf, 4096);               /* fills -> PARKS; reader-close -> -EPIPE */\n\
      if (w == -32) return 9;                           /* EPIPE detected -> clean exit 9 */\n\
      if (w <= 0) return 50;\n\
    }}\n\
  }}\n\
  if (read(fds[0], rb, 4096) <= 0) return 3;            /* drain one chunk (wakes the twin to refill) */\n\
  close(fds[0]);                                        /* reader gone -> the twin's parked write EPIPEs */\n\
  close(fds[1]);\n\
  if (__px_waitpid(0, pid, (long)&status, 0) != pid) return 6;\n\
  if (((status >> 8) & 0xff) != 9) return 300 + (int)((status >> 8) & 0xff);\n\
  return 42;\n\
}}\n"
    );
    let interp = run_interp_only(&src, |_| {});
    assert_eq!(
        interp.result,
        vec![Value::I32(42)],
        "tree-walker parked-writer EPIPE on reader close"
    );
    let byte = run_bytecode_only(&src, |_| {});
    assert_eq!(
        byte.result,
        vec![Value::I32(42)],
        "bytecode: the parked writer woke to -EPIPE when the reader closed — matching the oracle"
    );
    // #748 rung 3 — the reader-gone wake on the **parallel driver**: `pipe_write_ready`'s
    // readers==0 arm trips the blocked writer thread's poll and its re-issued write `-EPIPE`s.
    let par = run_bytecode_parallel_only(&src, |_| {});
    assert_eq!(
        par.result,
        vec![Value::I32(42)],
        "parallel driver: the parked writer thread woke to -EPIPE when the reader closed, matching both oracles"
    );
}

/// #799 — **`waitpid(-pgid)` on the personality table**: two twins `setpgid` into one group led
/// by the first; both are killed with one `kill(-pgid)`; two `waitpid(-pgid)` calls group-reap
/// exactly them (the zombie entries retain their pgid), and a third finds the group empty.
/// Everything through personality ops — the group-wait shape the core `__wait(-pgid)` offer
/// serves on the capability path, now available to a personality-linked shell.
#[test]
fn c_waitpid_by_group_reaps_a_personality_job() {
    let src = r#"
long __px_fork(int cap, long a);
long __px_waitpid(int cap, long pid, long status, long opts);
long __px_setpgid(int cap, long pid, long pgid);
long __px_kill(int cap, long pid, long sig);
long __px_signal(int cap, long signum, long handler);
static int status;
static long p1;
static long p2;
static long h;
static volatile long acc;
int main(void) {
  __px_signal(0, 10, 6);                          /* pre-install: the group kill must not
                                                     default-terminate a slow-to-install twin
                                                     (#796 discipline) — inherited by both */
  p1 = __px_fork(0, 0);
  if (p1 < 0) return 1;
  if (p1 == 0) { while (1) acc = acc + 1; }       /* member 1: runs until killed */
  p2 = __px_fork(0, 0);
  if (p2 < 0) return 2;
  if (p2 == 0) { while (1) acc = acc + 1; }       /* member 2 */
  if (__px_setpgid(0, p1, p1) != 0) return 3;     /* the job: group led by p1 */
  if (__px_setpgid(0, p2, p1) != 0) return 4;
  if (__px_kill(0, -p1, 9) != 0) return 5;        /* one SIGKILL fells the group (#796) */
  h = __px_waitpid(0, -p1, (long)&status, 0);     /* group-reap #1 */
  while (h == -10) h = __px_waitpid(0, -p1, (long)&status, 0);
  if (h != p1 && h != p2) return 6;
  long first = h;
  h = __px_waitpid(0, -p1, (long)&status, 0);     /* group-reap #2 */
  while (h == -10) h = __px_waitpid(0, -p1, (long)&status, 0);
  if (h == first || (h != p1 && h != p2)) return 7;
  if (__px_waitpid(0, -p1, (long)&status, 0) != -10) return 8;  /* the group is empty */
  return 42;
}
"#;
    let e = run_interp_only(src, |_| {});
    assert_eq!(
        e.result,
        vec![Value::I32(42)],
        "one kill(-pgid) felled the personality job and waitpid(-pgid) group-reaped exactly its \
         two members from the pgid-retaining zombie entries"
    );
}

/// #796 — guest wrappers for the signal ops, matching the `__px_` (dummy-handle-first) shim convention.
/// `sigprocmask`/`sigaction` take pointers to this personality's simple ABI: a `sigset_t` is a `u64`
/// bitset; a `struct sigaction` is `{ long sa_handler; unsigned long sa_mask; long sa_flags; }` (24 bytes).
const SIG_SHIM: &str = r#"
long __px_signal(int cap, long signum, long handler);
long __px_kill(int cap, long pid, long sig);
long __px_sigcheck(int cap, long z);
long __px_sigprocmask(int cap, long how, long set, long oldset);
long __px_sigaction(int cap, long signum, long act, long oldact);
static long signal_(long signum, long handler) { return __px_signal(0, signum, handler); }
static long raise_(long sig) { return __px_kill(0, 0, sig); }
static long sigcheck_(void) { return __px_sigcheck(0, 0); }
static long sigprocmask_(long how, void *set, void *oldset) { return __px_sigprocmask(0, how, (long)set, (long)oldset); }
static long sigaction_(long signum, void *act, void *oldact) { return __px_sigaction(0, signum, (long)act, (long)oldact); }
"#;

/// #796 — `sigprocmask` blocks a signal: a raised-but-blocked signal is **held** (not delivered by the
/// doorbell poll) until it is unblocked. Both checks are encoded in the return value: `a` (the poll while
/// blocked, which must be 0) in the thousands place, `b` (the poll after unblock, which must be the caught
/// handler 999) in the units → 999. A broken mask that delivered while blocked would read 999000 instead.
#[test]
fn c_sigprocmask_holds_a_blocked_signal() {
    let src = format!(
        "{SIG_SHIM}\n\
static unsigned long mask;\n\
int main(void) {{\n\
  signal_(2, 999);                 /* catch SIGINT */\n\
  mask = (1UL << 2);               /* the SIGINT bit */\n\
  sigprocmask_(0, &mask, 0);       /* SIG_BLOCK */\n\
  raise_(2);                       /* raise SIGINT -- blocked, so held */\n\
  long a = sigcheck_();            /* 0: masked */\n\
  sigprocmask_(1, &mask, 0);       /* SIG_UNBLOCK */\n\
  long b = sigcheck_();            /* 999: now deliverable */\n\
  return (int)(a * 1000 + b);      /* 999 */\n\
}}\n"
    );
    let (interp, jit) = run_both(&src, |_| {});
    assert_eq!(
        interp.result,
        vec![Value::I32(999)],
        "interp: a blocked signal is held, then delivered after unblock"
    );
    assert_eq!(jit.result, vec![Value::I64(999)], "jit parity");
}

/// #796 — `sigaction` records a disposition (delivered by the doorbell exactly like `signal`) and
/// round-trips the whole action through `oldact`. Returns the delivered handler (4242) plus 1 iff `oldact`
/// preserved `sa_handler` and `sa_flags` → 4243.
#[test]
fn c_sigaction_installs_and_round_trips() {
    let src = format!(
        "{SIG_SHIM}\n\
struct sigaction {{ long sa_handler; unsigned long sa_mask; long sa_flags; }};\n\
static struct sigaction act, old;\n\
int main(void) {{\n\
  act.sa_handler = 4242;\n\
  act.sa_mask = 0;\n\
  act.sa_flags = 7;\n\
  sigaction_(5, &act, 0);          /* install for signal 5 */\n\
  sigaction_(5, 0, &old);          /* read it back into old */\n\
  raise_(5);\n\
  long h = sigcheck_();            /* 4242 delivered */\n\
  int rt = (old.sa_handler == 4242 && old.sa_flags == 7) ? 1 : 0;\n\
  return (int)(h + rt);            /* 4243 */\n\
}}\n"
    );
    let (interp, jit) = run_both(&src, |_| {});
    assert_eq!(
        interp.result,
        vec![Value::I32(4243)],
        "interp: sigaction installs a caught handler and round-trips the action"
    );
    assert_eq!(jit.result, vec![Value::I64(4243)], "jit parity");
}

/// The **environment + cwd** surface from compiled C: `getenv` a variable the embedder staged, echo
/// its value; then `chdir` and read the new directory back with `getcwd`, echo that. Proves the
/// host-side env map and cwd (POSIX.md §3) are reachable through the same named-import path — the
/// pieces a shell needs for `$PATH` / `cd` / `pwd`. Both backends must agree on the echoed bytes.
#[test]
fn c_reads_env_and_cwd_through_the_personality() {
    let src = format!(
        "{SHIM}\n\
int main() {{\n\
  char *p = getenv(\"PATH\");     /* staged host-side as \"/bin\" */\n\
  if (p) write(1, p, slen(p)); /* -> \"/bin\" */\n\
  chdir(\"/tmp\");\n\
  char *buf = (char *)malloc(64);\n\
  getcwd(buf, 64);                /* NUL-terminated new cwd */\n\
  write(1, buf, slen(buf));    /* -> \"/tmp\" */\n\
  return 0;\n\
}}\n"
    );
    // Stage `PATH=/bin` in each backend's personality before the run (the embedder's environment).
    let (interp, jit) = run_both(&src, |px| px.set_env("PATH", "/bin"));

    assert_eq!(interp.result, vec![Value::I32(0)], "interp: main returns 0");
    assert_eq!(
        interp.stdout, b"/bin/tmp",
        "interp: getenv(PATH) then getcwd after chdir"
    );
    assert_eq!(
        jit.result,
        vec![Value::I64(0)],
        "jit: result must match interp"
    );
    assert_eq!(jit.stdout, interp.stdout, "jit: stdout must match interp");
}

/// A plain `write` then `exit(code)` from compiled C — both **standard libc names** whose guest
/// definitions shadow chibicc's Stream/Exit builtins (PROCESS.md S15 (b)), reaching the personality
/// (fd-routed write; `exit` → `Trap::Exit`). Proves the shadowing hook end to end: the program writes
/// to the personality's stdout with the real `fd` and terminates with the given code, identically on
/// both backends. The `return` after `exit` is dead (the personality's `exit` op never returns).
#[test]
fn c_write_then_exit_through_the_personality() {
    let src = format!(
        "{SHIM}\n\
int main() {{\n\
  write(1, \"bye\\n\", 4);   /* fd 1 -> captured stdout, via the shadowing wrapper */\n\
  exit(7);                  /* -> the personality's exit op (Trap::Exit) */\n\
  return 99;                /* dead: exit does not return */\n\
}}\n"
    );
    let (interp, jit) = run_both(&src, |_| {});

    assert_eq!(
        interp.exited,
        Some(7),
        "interp: exit(7) terminated the program"
    );
    assert_eq!(
        interp.stdout, b"bye\n",
        "interp: the write flushed before exit"
    );
    assert_eq!(jit.exited, Some(7), "jit: exit code must match interp");
    assert_eq!(jit.stdout, interp.stdout, "jit: stdout must match interp");
}

/// #800 — the guest-libc modules (`crates/temen-run/demos/posix_libc/`): real-libc functions that are
/// pure guest code (POSIX.md §1's split — semantics guest-side, never in the core), concatenated into
/// the test program as one translation unit.
const FNMATCH_C: &str = include_str!("../../temen-run/demos/posix_libc/fnmatch.c");
const POSIX_MISC_C: &str = include_str!("../../temen-run/demos/posix_libc/posix_misc.c");
const REGEX_C: &str = include_str!("../../temen-run/demos/posix_libc/regex.c");

/// Render `s` as a C string literal (escaping `\` and `"`).
fn c_str_lit(s: &str) -> String {
    let mut o = String::from("\"");
    for ch in s.chars() {
        match ch {
            '\\' => o.push_str("\\\\"),
            '"' => o.push_str("\\\""),
            _ => o.push(ch),
        }
    }
    o.push('"');
    o
}

/// #800 — the guest `fnmatch(3)` differential-tested against the **host libc's** `fnmatch(3)` over a
/// portable case table (the flags rendered per-platform through the `libc` crate). Guest flag values are
/// glibc's (fnmatch.c); the host side maps them to its own constants, so the same table drives both. The
/// guest prints one `'1'`/`'0'` per case; the expected string is computed by calling the real host
/// `fnmatch`. Cases stick to POSIX-specified behavior where glibc and the BSDs agree (the glibc-specific
/// fallbacks are hand-asserted in the next test). Both backends must match the host byte-for-byte.
#[test]
fn c_fnmatch_matches_the_host_libc() {
    // (pattern, string, guest flags: PATHNAME=1 NOESCAPE=2 PERIOD=4)
    const CASES: &[(&str, &str, i32)] = &[
        ("*", "abc", 0),
        ("*", "", 0),
        ("", "", 0),
        ("", "a", 0),
        ("*.c", "foo.c", 0),
        ("*.c", "foo.h", 0),
        ("*c", "c", 0),
        ("a**b", "ab", 0),
        ("a*b*c", "aXbYc", 0),
        ("a*b*c", "acb", 0),
        ("?at", "cat", 0),
        ("?at", "at", 0),
        ("[abc]at", "bat", 0),
        ("[abc]at", "dat", 0),
        ("[!abc]at", "dat", 0),
        ("[!abc]at", "bat", 0),
        ("[^abc]at", "dat", 0),
        ("[a-m]x", "gx", 0),
        ("[a-m]x", "px", 0),
        ("[]ab]", "]", 0),
        ("[]ab]", "c", 0),
        ("[[:digit:]][[:alpha:]]", "7x", 0),
        ("[[:digit:]][[:alpha:]]", "xx", 0),
        ("[![:space:]]", "q", 0),
        ("[[:xdigit:]]", "f", 0),
        ("foo/*", "foo/bar", 0),
        ("*", "foo/bar", 1),
        ("*/*", "foo/bar", 1),
        ("*/*", "foo/bar/baz", 1),
        ("foo?bar", "foo/bar", 1),
        ("foo*baz", "foo/baz", 1),
        ("foo/*", "foo/bar", 1),
        ("*", ".hidden", 4),
        (".*", ".hidden", 4),
        ("?idden", ".hidden", 4),
        ("*/*", "foo/.bar", 5),
        ("*/.*", "foo/.bar", 5),
        ("\\*", "*", 0),
        ("\\*", "x", 0),
        ("\\*", "\\anything", 2),
        ("a\\?c", "a?c", 0),
        ("a\\?c", "abc", 0),
    ];
    let n = CASES.len();

    // Host oracle: the same table through the platform's real fnmatch(3).
    fn host_flags(g: i32) -> i32 {
        let mut f = 0;
        if g & 1 != 0 {
            f |= libc::FNM_PATHNAME;
        }
        if g & 2 != 0 {
            f |= libc::FNM_NOESCAPE;
        }
        if g & 4 != 0 {
            f |= libc::FNM_PERIOD;
        }
        f
    }
    let expected: Vec<u8> = CASES
        .iter()
        .map(|&(pat, s, g)| {
            let p = std::ffi::CString::new(pat).unwrap();
            let c = std::ffi::CString::new(s).unwrap();
            let r = unsafe { libc::fnmatch(p.as_ptr(), c.as_ptr(), host_flags(g)) };
            if r == 0 {
                b'1'
            } else {
                b'0'
            }
        })
        .collect();

    let pats: Vec<String> = CASES.iter().map(|c| c_str_lit(c.0)).collect();
    let strs: Vec<String> = CASES.iter().map(|c| c_str_lit(c.1)).collect();
    let flgs: Vec<String> = CASES.iter().map(|c| c.2.to_string()).collect();
    let src = format!(
        "{SHIM}\n{FNMATCH_C}\n\
static char *pats[] = {{ {} }};\n\
static char *strs[] = {{ {} }};\n\
static int flgs[] = {{ {} }};\n\
static char out[{n}];\n\
int main(void) {{\n\
  int i;\n\
  for (i = 0; i < {n}; i = i + 1)\n\
    out[i] = fnmatch(pats[i], strs[i], flgs[i]) == 0 ? '1' : '0';\n\
  write(1, out, {n});\n\
  return 0;\n\
}}\n",
        pats.join(", "),
        strs.join(", "),
        flgs.join(", "),
    );
    let (interp, jit) = run_both(&src, |_| {});

    if interp.stdout != expected {
        let i = interp
            .stdout
            .iter()
            .zip(&expected)
            .position(|(a, b)| a != b)
            .unwrap_or(0);
        panic!(
            "guest fnmatch diverged from the host libc at case {i}: {:?} (guest {:?}, host {:?})",
            CASES[i], interp.stdout[i] as char, expected[i] as char
        );
    }
    assert_eq!(jit.stdout, expected, "jit parity with the host fnmatch");
}

/// #800 — the `fnmatch` behaviors the differential can't carry portably: `FNM_CASEFOLD` (the GNU/BSD
/// extension bash's nocaseglob/nocasematch use; the `libc` crate doesn't expose it everywhere) and the
/// glibc malformed-bracket fallback (an unterminated `[` matches literally — BSDs differ). Hand-asserted
/// expectations, both backends.
#[test]
fn c_fnmatch_casefold_and_bracket_fallback() {
    // (pattern, string, guest flags, expected-match) — CASEFOLD=16
    const CASES: &[(&str, &str, i32, bool)] = &[
        ("*.C", "foo.c", 16, true),
        ("abc", "ABC", 16, true),
        ("abc", "ABD", 16, false),
        ("[a-f]x", "Dx", 16, true),
        ("[A-F]x", "dx", 16, true),
        ("[a-f]x", "gx", 16, false),
        ("a[b", "a[b", 0, true),
        ("[", "[", 0, true),
        ("a[b", "ab", 0, false),
    ];
    let n = CASES.len();
    let expected: Vec<u8> = CASES
        .iter()
        .map(|c| if c.3 { b'1' } else { b'0' })
        .collect();
    let pats: Vec<String> = CASES.iter().map(|c| c_str_lit(c.0)).collect();
    let strs: Vec<String> = CASES.iter().map(|c| c_str_lit(c.1)).collect();
    let flgs: Vec<String> = CASES.iter().map(|c| c.2.to_string()).collect();
    let src = format!(
        "{SHIM}\n{FNMATCH_C}\n\
static char *pats[] = {{ {} }};\n\
static char *strs[] = {{ {} }};\n\
static int flgs[] = {{ {} }};\n\
static char out[{n}];\n\
int main(void) {{\n\
  int i;\n\
  for (i = 0; i < {n}; i = i + 1)\n\
    out[i] = fnmatch(pats[i], strs[i], flgs[i]) == 0 ? '1' : '0';\n\
  write(1, out, {n});\n\
  return 0;\n\
}}\n",
        pats.join(", "),
        strs.join(", "),
        flgs.join(", "),
    );
    let (interp, jit) = run_both(&src, |_| {});
    if interp.stdout != expected {
        let i = interp
            .stdout
            .iter()
            .zip(&expected)
            .position(|(a, b)| a != b)
            .unwrap_or(0);
        panic!(
            "fnmatch casefold/fallback diverged at case {i}: {:?} (guest {:?}, want {:?})",
            CASES[i], interp.stdout[i] as char, expected[i] as char
        );
    }
    assert_eq!(jit.stdout, expected, "jit parity");
}

/// #800 — `putenv` as real libc over the env ops: `KEY=VALUE` sets (overwriting a staged variable),
/// a bare `KEY` removes (the glibc behavior bash relies on). Round-tripped through `getenv` and echoed,
/// both backends.
#[test]
fn c_putenv_sets_overwrites_and_removes() {
    let src = format!(
        "{SHIM}\n{POSIX_MISC_C}\n\
int main(void) {{\n\
  if (putenv(\"NEWVAR=hello\") != 0) return 1;\n\
  char *v = getenv(\"NEWVAR\");\n\
  if (!v) return 2;\n\
  write(1, v, slen(v));                    /* -> \"hello\" */\n\
  if (putenv(\"PATH=/override\") != 0) return 3;\n\
  char *p = getenv(\"PATH\");\n\
  write(1, p, slen(p));                    /* -> \"/override\", the staged \"/bin\" replaced */\n\
  if (putenv(\"NEWVAR\") != 0) return 4;     /* bare name removes (glibc) */\n\
  if (getenv(\"NEWVAR\")) return 5;\n\
  return 42;\n\
}}\n"
    );
    let (interp, jit) = run_both(&src, |px| px.set_env("PATH", "/bin"));
    assert_eq!(
        interp.result,
        vec![Value::I32(42)],
        "interp: putenv set, overwrote, and removed"
    );
    assert_eq!(
        interp.stdout, b"hello/override",
        "interp: getenv round-trip"
    );
    assert_eq!(jit.result, vec![Value::I64(42)], "jit parity");
    assert_eq!(jit.stdout, interp.stdout, "jit stdout parity");
}

/// #800 — `wait4`/`wait3` as real libc over op 28: `wait4(pid)` rides the #799 **blocking** reap and
/// zeroes the caller's `rusage` (the personality meters fuel, not rusage — all-zero is the POSIX "no
/// information" answer; the buffer is pre-poisoned to prove the zeroing); `wait3` is the any-child form
/// (non-blocking on the table, so it polls like every `-1` wait). Two fork twins, reaped one by each.
#[test]
fn c_wait4_and_wait3_reap_fork_twins() {
    let src = format!(
        "{POSIX_MISC_C}\n\
long __px_fork(int cap, long a);\n\
static int status;\n\
static char ru[144];\n\
static long pid; static long pid2; static long h;\n\
int main(void) {{\n\
  int i;\n\
  for (i = 0; i < 144; i = i + 1) ru[i] = 0x55;   /* poison: wait4 must zero it */\n\
  pid = __px_fork(0, 0);\n\
  if (pid < 0) return 1;\n\
  if (pid == 0) return 7;\n\
  h = wait4(pid, &status, 0, ru);                 /* specific pid: the blocking reap */\n\
  if (h != pid) return 2;\n\
  if (((status >> 8) & 0xff) != 7) return 3;\n\
  for (i = 0; i < 144; i = i + 1) if (ru[i]) return 4;\n\
  pid2 = __px_fork(0, 0);\n\
  if (pid2 < 0) return 5;\n\
  if (pid2 == 0) return 9;\n\
  while ((h = wait3(&status, 0, ru)) == -10) {{}}  /* any-child: poll until the twin retires */\n\
  if (h != pid2) return 6;\n\
  if (((status >> 8) & 0xff) != 9) return 8;\n\
  return 42;\n\
}}\n"
    );
    let e = run_interp_only(&src, |_| {});
    assert_eq!(
        e.result,
        vec![Value::I32(42)],
        "wait4 blocked on the twin, zeroed rusage; wait3 reaped the second twin any-child"
    );
}

/// #800 — the guest `regcomp`/`regexec` (posix_libc/regex.c) differential-tested against the **host
/// libc's** POSIX regex over a case table: for every case both sides print the same encoding — `E;`
/// (compile error), `N;` (no match), or `so,eo` for the whole match and every capture group
/// (`nmatch = re_nsub+1`, so the group *count* is differential too) — and must agree byte-for-byte.
/// This pins the POSIX **leftmost-longest** discipline (`(a|ab)` against `"ab"` must span 0..2 where a
/// first-match backtracker stops at 0..1), capture spans including the last-repetition rule and unset
/// `(-1,-1)` groups, ICASE folding, and NOTBOL. Both guest backends must match the host.
#[test]
fn c_regex_matches_the_host_libc() {
    // (pattern, string, icase, notbol)
    const CASES: &[(&str, &str, bool, bool)] = &[
        ("abc", "xabcy", false, false),
        ("abc", "xy", false, false),
        ("a.c", "abc", false, false),
        ("^abc$", "abc", false, false),
        ("^abc$", "xabc", false, false),
        ("a*", "aaa", false, false),
        ("a*", "b", false, false),
        ("(a|ab)", "ab", false, false),
        ("(a|ab)(c|bcd)", "abcd", false, false),
        ("(a|b)+", "abab", false, false),
        ("a+b+", "aabbb", false, false),
        ("colou?r", "color", false, false),
        ("colou?r", "colour", false, false),
        ("[0-9]+", "abc123def", false, false),
        ("[^0-9]+", "abc123", false, false),
        ("[[:alpha:]]+", "ab12", false, false),
        ("[[:xdigit:]]+", "xfa9z", false, false),
        ("(ab)*", "ababx", false, false),
        ("a{2,3}", "aaaa", false, false),
        ("a{2}", "aaaa", false, false),
        ("a{2,}", "aaaaa", false, false),
        ("(a+)(b*)(c?)", "aabcc", false, false),
        ("x|yz|w", "byzq", false, false),
        ("\\.", "a.b", false, false),
        ("a$", "ba", false, false),
        ("(a)(b)?", "a", false, false),
        ("[]a]x", "]x", false, false),
        ("[a-]x", "-x", false, false),
        ("abc", "xABCy", true, false),
        ("[a-f]+", "DEaf", true, false),
        ("[[:upper:]]+", "aBCd", true, false),
        ("^a", "abc", false, true),
        ("a(", "aa", false, false),
        ("[z-a]", "x", false, false),
    ];
    let n = CASES.len();

    // In ERE every unescaped `(` outside a bracket expression opens a group; the libc crate keeps
    // `regex_t.__re_nsub` private, so count groups from the pattern — the encoding's pair count then
    // cross-checks the guest's own `re_nsub` against this.
    fn nsub_of(pat: &str) -> usize {
        let b = pat.as_bytes();
        let (mut n, mut i, mut in_br) = (0, 0, false);
        while i < b.len() {
            match b[i] {
                b'\\' if !in_br => i += 1,
                b'[' if !in_br => in_br = true,
                b']' if in_br => in_br = false,
                b'(' if !in_br => n += 1,
                _ => {}
            }
            i += 1;
        }
        n
    }

    // Host oracle: the identical encoding through the platform's real regcomp/regexec.
    let mut expected = String::new();
    for &(pat, s, icase, notbol) in CASES {
        unsafe {
            let mut re: libc::regex_t = std::mem::zeroed();
            let p = std::ffi::CString::new(pat).unwrap();
            let c = std::ffi::CString::new(s).unwrap();
            let cf = libc::REG_EXTENDED | if icase { libc::REG_ICASE } else { 0 };
            if libc::regcomp(&mut re, p.as_ptr(), cf) != 0 {
                expected.push_str("E;");
                continue;
            }
            let nsub = nsub_of(pat);
            let mut m: [libc::regmatch_t; 33] = std::mem::zeroed();
            let ef = if notbol { libc::REG_NOTBOL } else { 0 };
            if libc::regexec(&re, c.as_ptr(), nsub + 1, m.as_mut_ptr(), ef) != 0 {
                expected.push_str("N;");
            } else {
                for g in m.iter().take(nsub + 1) {
                    expected.push_str(&format!("{},{} ", g.rm_so, g.rm_eo));
                }
                expected.push(';');
            }
            libc::regfree(&mut re);
        }
    }

    let pats: Vec<String> = CASES.iter().map(|c| c_str_lit(c.0)).collect();
    let strs: Vec<String> = CASES.iter().map(|c| c_str_lit(c.1)).collect();
    let cfs: Vec<String> = CASES
        .iter()
        .map(|c| if c.2 { "2" } else { "0" }.to_string())
        .collect();
    let efs: Vec<String> = CASES
        .iter()
        .map(|c| if c.3 { "1" } else { "0" }.to_string())
        .collect();
    let src = format!(
        "{SHIM}\n{REGEX_C}\n\
static char *pats[] = {{ {} }};\n\
static char *strs[] = {{ {} }};\n\
static int cfs[] = {{ {} }};\n\
static int efs[] = {{ {} }};\n\
static void put_num(long v) {{\n\
  char b[24];\n\
  int i = 24;\n\
  int neg = v < 0;\n\
  if (neg) v = -v;\n\
  if (v == 0) {{ i = i - 1; b[i] = '0'; }}\n\
  while (v) {{ i = i - 1; b[i] = '0' + (v - v / 10 * 10); v = v / 10; }}\n\
  if (neg) {{ i = i - 1; b[i] = '-'; }}\n\
  write(1, b + i, 24 - i);\n\
}}\n\
int main(void) {{\n\
  int i;\n\
  for (i = 0; i < {n}; i = i + 1) {{\n\
    regex_t re;\n\
    if (regcomp(&re, pats[i], cfs[i] | 1) != 0) {{ write(1, \"E;\", 2); continue; }}\n\
    regmatch_t m[4];\n\
    if (regexec(&re, strs[i], re.re_nsub + 1, m, efs[i]) != 0) {{\n\
      write(1, \"N;\", 2);\n\
    }} else {{\n\
      long k;\n\
      for (k = 0; k <= re.re_nsub; k = k + 1) {{\n\
        put_num(m[k].rm_so); write(1, \",\", 1); put_num(m[k].rm_eo); write(1, \" \", 1);\n\
      }}\n\
      write(1, \";\", 1);\n\
    }}\n\
    regfree(&re);\n\
  }}\n\
  return 0;\n\
}}\n",
        pats.join(", "),
        strs.join(", "),
        cfs.join(", "),
        efs.join(", "),
    );
    let (interp, jit) = run_both(&src, |_| {});

    let got = String::from_utf8_lossy(&interp.stdout);
    if got != expected {
        let gi: Vec<&str> = got.split(';').collect();
        let ei: Vec<&str> = expected.split(';').collect();
        let i = gi.iter().zip(&ei).position(|(a, b)| a != b).unwrap_or(0);
        panic!(
            "guest regex diverged from the host libc at case {i}: {:?}\n  guest: {:?}\n  host:  {:?}",
            CASES[i],
            gi.get(i),
            ei.get(i)
        );
    }
    assert_eq!(
        String::from_utf8_lossy(&jit.stdout),
        expected,
        "jit parity with the host regex"
    );
}

/// #800 — the `[[ =~ ]]` shape end to end: compile once, match, and read the whole match + groups the
/// way bash fills `BASH_REMATCH` — hand-asserted (documents the intended use), both backends.
#[test]
fn c_regex_bash_rematch_shape() {
    let src = format!(
        "{SHIM}\n{REGEX_C}\n\
int main(void) {{\n\
  regex_t re;\n\
  /* the bash idiom: [[ \"2026-08-18\" =~ ([0-9]+)-([0-9]+)-([0-9]+) ]] */\n\
  if (regcomp(&re, \"([0-9]+)-([0-9]+)-([0-9]+)\", 1) != 0) return 1;\n\
  if (re.re_nsub != 3) return 2;\n\
  regmatch_t m[4];\n\
  if (regexec(&re, \"date: 2026-08-18!\", 4, m, 0) != 0) return 3;\n\
  if (m[0].rm_so != 6 || m[0].rm_eo != 16) return 4;   /* BASH_REMATCH[0] */\n\
  if (m[1].rm_so != 6 || m[1].rm_eo != 10) return 5;   /* 2026 */\n\
  if (m[2].rm_so != 11 || m[2].rm_eo != 13) return 6;  /* 08 */\n\
  if (m[3].rm_so != 14 || m[3].rm_eo != 16) return 7;  /* 18 */\n\
  if (regexec(&re, \"no digits here\", 4, m, 0) == 0) return 8;\n\
  /* no regfree */\n\
  return 42;\n\
}}\n"
    );
    let (interp, jit) = run_both(&src, |_| {});
    assert_eq!(
        interp.result,
        vec![Value::I32(42)],
        "the =~ shape: compile, match, BASH_REMATCH-style group spans"
    );
    assert_eq!(jit.result, vec![Value::I64(42)], "jit parity");
}

const GLOB_C: &str = include_str!("../../temen-run/demos/posix_libc/glob.c");

/// #800 — `glob(3)` over the memfs: absolute patterns walk segments via opendir/readdir with
/// slice 1's `fnmatch` (`FNM_PERIOD` — `*` skips dotfiles, the shell rule), results sorted; a magic
/// middle segment (`/*/x.c`) fans out across directories; `GLOB_MARK` marks directories with `/`;
/// no match is `GLOB_NOMATCH` unless `GLOB_NOCHECK` returns the pattern itself; `globfree` releases.
/// Each sub-check writes its joined `gl_pathv` to stdout, and the whole battery must agree across
/// backends. Hand-asserted (the host's glob walks a real fs, not this memfs — no oracle to share).
#[test]
fn c_glob_expands_over_the_memfs() {
    let src = format!(
        "{SHIM}\n{FNMATCH_C}\n{GLOB_C}\n\
static glob_t g;\n\
static void dump(void) {{\n\
  long i;\n\
  for (i = 0; i < g.gl_pathc; i = i + 1) {{\n\
    char *s = g.gl_pathv[g.gl_offs + i];\n\
    write(1, s, slen(s));\n\
    write(1, \" \", 1);\n\
  }}\n\
  write(1, \";\", 1);\n\
}}\n\
int main(void) {{\n\
  if (glob(\"/*.c\", 0, 0, &g) != 0) return 1;      /* sorted, dotfiles skipped */\n\
  dump();\n\
  globfree(&g);\n\
  if (glob(\"/dir/*.c\", 0, 0, &g) != 0) return 2;  /* .hidden.c excluded */\n\
  dump();\n\
  globfree(&g);\n\
  if (glob(\"/dir/.*.c\", 0, 0, &g) != 0) return 3; /* explicit dot matches it */\n\
  dump();\n\
  globfree(&g);\n\
  if (glob(\"/*/x.c\", 0, 0, &g) != 0) return 4;    /* magic middle segment */\n\
  dump();\n\
  globfree(&g);\n\
  if (glob(\"/d*\", 2, 0, &g) != 0) return 5;       /* GLOB_MARK: trailing / on dirs */\n\
  dump();\n\
  globfree(&g);\n\
  if (glob(\"/nope*\", 0, 0, &g) != 3) return 6;    /* GLOB_NOMATCH */\n\
  if (glob(\"/nope*\", 16, 0, &g) != 0) return 7;   /* GLOB_NOCHECK: the pattern back */\n\
  dump();\n\
  globfree(&g);\n\
  return 42;\n\
}}\n"
    );
    let (interp, jit) = run_both(&src, |px| {
        px.write_file("/b.c", b"b");
        px.write_file("/a.c", b"a");
        px.write_file("/ab.txt", b"t");
        px.write_file("/.dot.c", b"d");
        px.write_file("/dir/x.c", b"x");
        px.write_file("/dir/y.h", b"y");
        px.write_file("/dir/.hidden.c", b"h");
        px.write_file("/dpl/x.c", b"x2");
    });
    assert_eq!(
        interp.result,
        vec![Value::I32(42)],
        "every glob sub-check passed"
    );
    let want: &[u8] = b"/a.c /b.c ;\
/dir/x.c ;\
/dir/.hidden.c ;\
/dir/x.c /dpl/x.c ;\
/dir/ /dpl/ ;\
/nope* ;";
    assert_eq!(
        String::from_utf8_lossy(&interp.stdout),
        String::from_utf8_lossy(want),
        "the expansions, sorted, dot-rules and MARK applied"
    );
    assert_eq!(jit.result, vec![Value::I64(42)], "jit parity");
    assert_eq!(jit.stdout, interp.stdout, "jit stdout parity");
}

/// #972 slice 1 — the **pipe-unification shim**: `pipe()` composes the guest's own core mint
/// (`__vm_pipe`, CAP_SELF_PIPE) with the personality's adopt op; `read`/`write`/`close` follow the
/// personality's handle-carrying tag (`PX_TAG_BASE - handle`, disjoint from every errno) to the core
/// cap-call path — blocking parks, EINTR, true EOF, `-EPIPE` — and `write` raises SIGPIPE through
/// the personality on `-EPIPE` (disposition-gated; the write still returns `-EPIPE`). The guest
/// definitions shadow chibicc's powerbox builtins of the same names (PROCESS.md S15 (b)).
const PIPE_SHIM: &str = r#"
long __vm_pipe(int *fds);
long __vm_read(int fd, void *buf, long len);
long __vm_write(int fd, void *buf, long len);
long __vm_close(int h);
long __px_pipe_adopt(int cap, long rh, long wh, long fdp);
long __px_read(int cap, long fd, long buf, long len);
long __px_write(int cap, long fd, long buf, long len);
long __px_close(int cap, long fd);
long __px_kill(int cap, long pid, long sig);

static long px_h_(long r) { return r <= -1048576 ? -(r + 1048576) : -1; }
long pipe(int *fds) {
  int h[2];
  long r = __vm_pipe(h);
  if (r != 0) return r;              /* off the park tier: the probeable decline surfaces here */
  return __px_pipe_adopt(0, h[0], h[1], (long)fds);
}
long read(long fd, void *buf, long n) {
  long r = __px_read(0, fd, (long)buf, n);
  long h = px_h_(r);
  if (h < 0) return r;
  return __vm_read((int)h, buf, n);  /* core path: parks on empty-with-writers, EOFs at count 0 */
}
long write(long fd, void *buf, long n) {
  long r = __px_write(0, fd, (long)buf, n);
  long h = px_h_(r);
  if (h < 0) return r;
  r = __vm_write((int)h, buf, n);
  if (r == -32) __px_kill(0, 0, 13); /* -EPIPE: raise SIGPIPE per disposition */
  return r;
}
long close(long fd) {
  long r = __px_close(0, fd);
  long h = px_h_(r);
  if (h < 0) return r;
  __vm_close((int)h);                /* last dup: release the end -> EOF/EPIPE wakes */
  return 0;
}
"#;

/// #972 slice 1 — **THE witness the #967 audit couldn't write**: a concurrent pipeline across fork
/// twins. The parent pipes, forks, closes its own copy of the write end, and **blocks** in `read`
/// on the empty pipe — the twin's inherited write end keeps the writer count > 0, so this parks
/// (`Blocked::PipeRead`) instead of the old `PipeBuf` false-EOF. The twin (spinning first so the
/// parent is genuinely parked) writes through its fd, then exits — its domain teardown releases its
/// ends, so after the parent drains the bytes the next `read` returns **true EOF** (0). All through
/// libc names over the unified fd table. Interp-only (fork + the park machinery).
#[test]
fn c_core_pipe_concurrent_pipeline_across_fork() {
    let src = format!(
        "{PIPE_SHIM}\n\
long __px_fork(int cap, long a);\n\
long __px_waitpid(int cap, long pid, long status, long opts);\n\
static int fds[2];\n\
static int status;\n\
static char b[8];\n\
static volatile long acc;\n\
static long pid; static long h;\n\
int main(void) {{\n\
  if (pipe(fds) != 0) return 1;\n\
  pid = __px_fork(0, 0);\n\
  if (pid < 0) return 2;\n\
  if (pid == 0) {{\n\
    long i;\n\
    for (i = 0; i < 2000000; i = i + 1) acc = acc + i;  /* let the parent park first */\n\
    if (write(fds[1], \"GO!\", 3) != 3) return 8;\n\
    return 7;                                 /* exit releases the twin's ends */\n\
  }}\n\
  close(fds[1]);                              /* parent's write end gone: twin's keeps it open */\n\
  long n = read(fds[0], b, 8);                /* PARKS (writer alive), woken by the twin's write */\n\
  if (n != 3) return 3;\n\
  if (b[0] != 'G' || b[1] != 'O' || b[2] != '!') return 4;\n\
  n = read(fds[0], b, 8);                     /* twin exited: writer count 0 -> true EOF */\n\
  if (n != 0) return 5;\n\
  h = __px_waitpid(0, pid, (long)&status, 0);\n\
  if (h != pid) return 6;\n\
  if (((status >> 8) & 0xff) != 7) return 9;\n\
  return 42;\n\
}}\n"
    );
    let e = run_interp_only(&src, |_| {});
    assert_eq!(
        e.result,
        vec![Value::I32(42)],
        "blocked reader fed by a live writer across fork twins, true EOF on last-writer-close"
    );
}

/// #972 slice 1 — **dup refcounting + release-fires-EPIPE + SIGPIPE**: `dup2` shares the token, so
/// closing the original read fd releases nothing (the write side stays writable); closing the LAST
/// read dup releases the read end — proven by the writer immediately getting `-EPIPE` (the reader
/// count hit 0 in the core, so the release really fired), with SIGPIPE raised through the
/// personality (caught handler pending on the doorbell). Interp-only (`CAP_SELF_PIPE`).
#[test]
fn c_core_pipe_dup_close_refcount_and_epipe() {
    let src = format!(
        "{PIPE_SHIM}\n\
long __px_signal(int cap, long signum, long handler);\n\
long __px_sigcheck(int cap, long z);\n\
static int fds[2];\n\
static char b[8];\n\
int main(void) {{\n\
  __px_signal(0, 13, 777);                    /* catch SIGPIPE so the raise is observable */\n\
  if (pipe(fds) != 0) return 1;\n\
  if (write(fds[1], \"hi\", 2) != 2) return 2;\n\
  if (__px_close(0, 99) != -9) return 3;      /* plain errno stays plain: -EBADF, never a tag */\n\
  long d = 9;\n\
  long __px_dup2(int cap, long o, long n);\n\
  if (__px_dup2(0, fds[0], d) != d) return 4;\n\
  if (close(fds[0]) != 0) return 5;           /* NOT the last dup: no release */\n\
  if (read(d, b, 8) != 2) return 6;           /* the dup still drains the pipe */\n\
  if (close(d) != 0) return 7;                /* LAST dup: releases the read end */\n\
  if (write(fds[1], \"x\", 1) != -32) return 8; /* reader count 0 in the core: -EPIPE */\n\
  if (__px_sigcheck(0, 0) != 777) return 9;   /* the shim raised SIGPIPE; caught -> pending */\n\
  return 42;\n\
}}\n"
    );
    let e = run_interp_only(&src, |_| {});
    assert_eq!(
        e.result,
        vec![Value::I32(42)],
        "dup shares, last close releases (write sees -EPIPE), SIGPIPE raised per disposition"
    );
}

/// #972 slice 1 — **invariant-9 decline pin**: the pipe path refuses probeably where the park
/// machinery is absent. On the interpreter (Real scheduler) `pipe()` works; on the JIT the mint
/// self-op declines (`-EINVAL`) and the shim surfaces it as a clean errno from `pipe()` — no trap,
/// no hang, no false success. The same program, both backends, each asserting its own honest
/// answer — the divergence is *toward refusal*, the fail-closed direction invariant 9 sanctions.
#[test]
fn c_core_pipe_declines_probeably_off_the_park_tier() {
    let src = format!(
        "{PIPE_SHIM}\n\
static int fds[2];\n\
static char b[4];\n\
int main(void) {{\n\
  long r = pipe(fds);\n\
  if (r == -22) return 22;                    /* the decline: clean -EINVAL through the shim */\n\
  if (r != 0) return 9;\n\
  if (write(fds[1], \"ok\", 2) != 2) return 8; /* park tier: drain sequentially, no park needed */\n\
  if (read(fds[0], b, 4) != 2) return 7;\n\
  return 100;\n\
}}\n"
    );
    let (interp, jit) = run_both(&src, |_| {});
    assert_eq!(
        interp.result,
        vec![Value::I32(100)],
        "interp (park tier): the unified pipe works end to end"
    );
    assert_eq!(
        jit.result,
        vec![Value::I64(22)],
        "jit (no park machinery): pipe() declines with a clean -EINVAL, never a hang or false success"
    );
}

/// #972 slice 1 — **tag ABI discipline pins** (invariants 5/11): the raw op return on a CorePipe fd
/// is a value in the reserved range (`<= PX_TAG_BASE`, disjoint from every errno since errnos are
/// `> -4096`); a naive caller treating it as an error fails closed (no bytes moved — the buffer is
/// untouched); and the shim's decode round-trips (the decoded handle really reads the pipe's bytes).
/// The Rust side pins the constant the C shim hardcodes, so drift breaks here first.
#[test]
fn c_core_pipe_tag_range_and_fail_closed() {
    assert_eq!(
        temen_posix::PX_TAG_BASE,
        -(1 << 20),
        "the C shim hardcodes -1048576; keep them in lockstep"
    );
    let src = format!(
        "{PIPE_SHIM}\n\
static int fds[2];\n\
static char b[4];\n\
int main(void) {{\n\
  if (pipe(fds) != 0) return 1;\n\
  if (write(fds[1], \"ab\", 2) != 2) return 2;\n\
  b[0] = 'Z';\n\
  long t = __px_read(0, fds[0], (long)b, 4);  /* RAW op, no shim: the tag, not bytes */\n\
  if (t > -1048576) return 3;                 /* in the reserved range, below every errno */\n\
  if (b[0] != 'Z') return 4;                  /* fail-closed: nothing moved */\n\
  long h = -(t + 1048576);\n\
  if (__vm_read((int)h, b, 4) != 2) return 5; /* the decoded handle drains the real bytes */\n\
  if (b[0] != 'a' || b[1] != 'b') return 6;\n\
  return 42;\n\
}}\n"
    );
    let e = run_interp_only(&src, |_| {});
    assert_eq!(
        e.result,
        vec![Value::I32(42)],
        "tag in range, fail-closed on raw use, decode round-trips"
    );
}

/// #972 slice 1 — **freeze witness** (invariant 7-adjacent): after the personality makes pipe ends
/// reachable from libc (`pipe()` = mint + adopt), a freeze of the domain still hits the existing
/// clean refusal — `capture_durable_handles` reports `NonDurableKind::Pipe`, never a partial
/// snapshot or a new failure mode.
#[test]
fn c_core_pipe_freeze_refuses_nondurable() {
    let src = format!(
        "{PIPE_SHIM}\n\
static int fds[2];\n\
int main(void) {{ return pipe(fds) == 0 ? 42 : 9; }}\n"
    );
    let ir = c_to_ir(&src);
    let raw = parse_module_raw(&ir).expect("parse");
    let win = 1u64 << raw.memory.expect("window").size_log2;
    let mut ih = Host::new();
    let (_posix, ipx) = setup(&mut ih, win);
    verify_module(&raw).expect("verify");
    bind_shim(&raw, &mut ih, ipx);
    let mut fuel = 50_000_000u64;
    let r = run_with_host(&raw, 0, &[], &mut fuel, &mut ih).expect("run");
    assert_eq!(r, vec![Value::I32(42)], "the guest minted + adopted a pipe");
    // The refusal reports the FIRST non-durable slot: the personality's own HostProc handle sits
    // below the pipe ends, so a personality domain was non-durable before pipes and stays so —
    // the same clean refusal, no new failure mode.
    let err = ih
        .capture_durable_handles()
        .expect_err("a personality domain holding pipe ends must refuse durable capture");
    assert_eq!(
        err.kind,
        temen_interp::NonDurableKind::HostProc,
        "the personality slot refuses first (it precedes the pipe ends in the table)"
    );
    // And the pipe ends refuse in their own right: a bare host whose only non-durable slots are a
    // minted pipe's two ends reports NonDurableKind::Pipe.
    let mut bare = Host::new();
    let (_w, _r) = bare.grant_pipe();
    let err = bare
        .capture_durable_handles()
        .expect_err("a live pipe end alone must refuse durable capture");
    assert_eq!(
        err.kind,
        temen_interp::NonDurableKind::Pipe,
        "the pipe end's own refusal kind"
    );
}

const EXEC_C: &str = include_str!("../../temen-run/demos/posix_libc/exec.c");

/// #801 — like [`run_interp_only`] but with an `extra` hook that can grant command modules on the
/// `Host` and register them with the personality (grant order: personality first, then commands, so
/// handle values are deterministic). The execve tests use it to stage `/bin` executables.
fn run_interp_setup(src: &str, extra: impl Fn(&mut Host, &Posix)) -> Effects {
    let ir = c_to_ir(src);
    let raw = parse_module_raw(&ir)
        .unwrap_or_else(|e| panic!("parse IR failed: {e:?}\n--- IR ---\n{ir}"));
    let win = 1u64
        << raw
            .memory
            .expect("the frontend declares a window")
            .size_log2;
    let mut ih = Host::new();
    let (iposix, ipx) = setup(&mut ih, win);
    extra(&mut ih, &iposix);
    verify_module(&raw).unwrap_or_else(|e| panic!("verify failed: {e:?}\n--- IR ---\n{ir}"));
    bind_shim(&raw, &mut ih, ipx);
    let mut fuel = 200_000_000u64;
    let (result, exited) = match run_with_host(&raw, 0, &[], &mut fuel, &mut ih) {
        Ok(v) => (v, None),
        Err(Trap::Exit(c)) => (Vec::new(), Some(c)),
        Err(e) => panic!("interp trapped: {e:?}\n--- IR ---\n{ir}"),
    };
    Effects {
        result,
        exited,
        stdout: iposix.stdout(),
        file_f: iposix.read_file("f"),
    }
}

/// Compile a C source with the §14 **`--child-entry`** ABI — the `(i64 starter) -> (i64)` entry
/// shape `exec_module`'s admissibility requires of a command (`child_entry_ok`); the plain powerbox
/// `() -> (i32)` entry is refused. Commands are child-entry; shells stay plain.
fn c_to_ir_child(src: &str) -> String {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let id = N.fetch_add(1, Ordering::Relaxed);
    let base = std::env::temp_dir().join(format!("temen_cposixcmd_{}_{id}", std::process::id()));
    let cfile = base.with_extension("c");
    let irfile = base.with_extension("temen");
    std::fs::write(&cfile, src).unwrap();
    let status = Command::new(chibicc())
        .args([
            "-cc1",
            "--emit-ir",
            "--child-entry",
            "-cc1-input",
            cfile.to_str().unwrap(),
            "-cc1-output",
            irfile.to_str().unwrap(),
            cfile.to_str().unwrap(),
        ])
        .status()
        .expect("run chibicc");
    assert!(status.success(), "chibicc failed on:\n{src}");
    std::fs::read_to_string(&irfile).unwrap()
}

/// Compile a C command source, grant it as a `Module`, and register it as a filesystem executable
/// at `path` — the embedder half of #801's userland ("each coreutil compiled to a module,
/// presented in the fs as a file").
fn stage_executable(host: &mut Host, posix: &Posix, path: &str, src: &str) {
    let ir = c_to_ir_child(src);
    let m = parse_module_raw(&ir).expect("parse command");
    verify_module(&m).expect("verify command");
    let wl = m.memory.expect("command window").size_log2;
    let h = host.grant_module(&m);
    posix.register_executable(path, h, wl);
}

/// #801 slice A — **the POSIX trinity over the personality**: `fork` → `execve("/bin/rc")` →
/// `waitpid`. The twin image-replaces itself with a registered command module (resolved by
/// filesystem path through op 53, exec'd by the guest's own `CAP_SELF_EXEC` call — zero new core
/// surface); its `return 99` after `execve` **never runs** (the image was truly replaced); the
/// command's argv arrives through the preserved args region; and the parent's blocking `waitpid`
/// reaps the **command's** status under the twin's pid — POSIX's exec-keeps-the-pid, end to end.
#[test]
fn c_fork_execve_wait_runs_a_filesystem_command() {
    const CMD: &str = r#"
int main(int argc, char **argv) {
  if (argc != 2) return 90;
  return argv[1][0];        /* '4' = 52: argv crossed the exec */
}
"#;
    let src = format!(
        "{WIN_PAD_17}{EXEC_C}\n\
long __px_fork(int cap, long a);\n\
long __px_waitpid(int cap, long pid, long status, long opts);\n\
static char *av[] = {{ \"rc\", \"4\", 0 }};\n\
static int status;\n\
static long pid; static long h;\n\
int main(void) {{\n\
  pid = __px_fork(0, 0);\n\
  if (pid < 0) return 1;\n\
  if (pid == 0) {{\n\
    execve(\"/bin/rc\", av, 0);\n\
    return 99;                       /* must never run: the image is replaced */\n\
  }}\n\
  h = __px_waitpid(0, pid, (long)&status, 0);\n\
  if (h != pid) return 2;\n\
  if (((status >> 8) & 0xff) != 52) return 3;\n\
  return 42;\n\
}}\n"
    );
    let e = run_interp_setup(&src, |host, posix| {
        stage_executable(host, posix, "/bin/rc", CMD);
    });
    assert_eq!(
        e.result,
        vec![Value::I32(42)],
        "fork -> execve(path) -> waitpid: the command's argv-derived status reaped under the twin's pid"
    );
}

/// #801 slice A — **`execvp` walks PATH** (`getenv("PATH")`, `:`-separated, ENOENT continues the
/// walk) and the **errno split + exec bit** hold: an absent path is `-ENOENT`, a plain memfs file
/// without the executable registration is `-EACCES`, and `stat` reports the exec bits only on
/// registered executables.
#[test]
fn c_execvp_walks_path_and_the_errno_split_holds() {
    const CMD: &str = r#"
int main(void) { return 7; }
"#;
    let src = format!(
        "{WIN_PAD_17}{EXEC_C}\n\
long __px_fork(int cap, long a);\n\
long __px_waitpid(int cap, long pid, long status, long opts);\n\
long __px_stat(int cap, long path, long len, long buf);\n\
static char *av[] = {{ \"tool\", 0 }};\n\
static long st[2];\n\
static int status;\n\
static long pid; static long h;\n\
int main(void) {{\n\
  if (execve(\"/no/such\", av, 0) != -2) return 1;      /* ENOENT */\n\
  if (execve(\"/plain.txt\", av, 0) != -13) return 2;   /* a file, no exec bit: EACCES */\n\
  if (__px_stat(0, (long)\"/bin/tool\", 9, (long)st) != 0) return 3;\n\
  if ((st[0] & 0111) == 0) return 4;                    /* exec bits on the registered file */\n\
  if (__px_stat(0, (long)\"/plain.txt\", 10, (long)st) != 0) return 5;\n\
  if ((st[0] & 0111) != 0) return 6;                    /* none on the plain file */\n\
  pid = __px_fork(0, 0);\n\
  if (pid < 0) return 7;\n\
  if (pid == 0) {{\n\
    execvp(\"tool\", av);                 /* PATH=/nowhere:/bin -> ENOENT then the hit */\n\
    return 99;\n\
  }}\n\
  h = __px_waitpid(0, pid, (long)&status, 0);\n\
  if (h != pid) return 8;\n\
  if (((status >> 8) & 0xff) != 7) return 9;\n\
  return 42;\n\
}}\n"
    );
    let e = run_interp_setup(&src, |host, posix| {
        stage_executable(host, posix, "/bin/tool", CMD);
        posix.write_file("/plain.txt", b"not a program");
        posix.set_env("PATH", "/nowhere:/bin");
    });
    assert_eq!(
        e.result,
        vec![Value::I32(42)],
        "ENOENT/EACCES split, stat exec bits, and the PATH walk landing on /bin/tool"
    );
}

/// #801 rung 1 — **a `__px_`-linked command execs, its libc bound in-loop by the vtable**: the
/// registered command's manifest is pure `__px_*` imports; at exec the personality moves verbatim
/// into the new powerbox (same process — same Proc, same World), its vtable registers the op names
/// in the new image's directory, and `bind_child_manifest` binds them through the coverage walk —
/// signature-checked, no external resolver. The command writes through its own bound `write`,
/// reads an env var the shell's process carried across the exec (same-Proc witness), and exits
/// with argc; the parent reaps that status under the twin's pid.
#[test]
fn c_execve_runs_a_px_linked_command() {
    const CMD: &str = r#"
long __px_write(int cap, long fd, long buf, long len);
long __px_getenv(int cap, long name, long len);
int main(int argc, char **argv) {
  __px_write(0, 1, (long)"CMD!", 4);
  char *v = (char *)__px_getenv(0, (long)"MARK", 4);
  if (!v || v[0] != 'y') return 90;   /* the process (env) crossed the exec */
  return argc;                        /* 2: argv crossed too */
}
"#;
    let src = format!(
        "{WIN_PAD_17}{EXEC_C}\n\
long __px_fork(int cap, long a);\n\
long __px_waitpid(int cap, long pid, long status, long opts);\n\
static char *av[] = {{ \"px\", \"z\", 0 }};\n\
static int status;\n\
static long pid; static long h;\n\
int main(void) {{\n\
  pid = __px_fork(0, 0);\n\
  if (pid < 0) return 1;\n\
  if (pid == 0) {{\n\
    execve(\"/bin/px\", av, 0);\n\
    return 99;\n\
  }}\n\
  h = __px_waitpid(0, pid, (long)&status, 0);\n\
  if (h != pid) return 2;\n\
  if (((status >> 8) & 0xff) != 2) return 200 + ((status >> 8) & 0xff);\n\
  return 42;\n\
}}\n"
    );
    let e = run_interp_setup(&src, |host, posix| {
        stage_executable(host, posix, "/bin/px", CMD);
        posix.set_env("MARK", "y");
    });
    assert_eq!(
        e.result,
        vec![Value::I32(42)],
        "the px-linked command's libc bound in-loop; env + argv crossed; status reaped"
    );
    assert_eq!(
        e.stdout, b"CMD!",
        "the command's write dispatched to the SAME personality (captured stdout)"
    );
}

/// #801 rung 1 — **signature drift is a clean bind-time refusal, and the caller survives it**: a
/// command declaring `__px_write` with the wrong arity fails the vtable's sig check, `execve`
/// returns `-EINVAL` (POSIX: only on failure), and — the restore-path witness — the caller's own
/// personality still works afterwards (the moved host-proc entries were given back).
#[test]
fn c_execve_refuses_sig_drift_and_the_caller_survives() {
    const BAD: &str = r#"
long __px_write(int cap, long fd, long buf);   /* wrong arity: 2 args, canonical is 3 */
int main(void) {
  __px_write(0, 1, (long)"X");
  return 7;
}
"#;
    let src = format!(
        "{WIN_PAD_17}{EXEC_C}\n\
long __px_write(int cap, long fd, long buf, long len);\n\
long write2(long fd, void *b, long n) {{ return __px_write(0, fd, (long)b, n); }}\n\
static char *av[] = {{ \"bad\", 0 }};\n\
int main(void) {{\n\
  int r = execve(\"/bin/bad\", av, 0);\n\
  if (r != -22) return 1;              /* the bind refusal, surfaced as -EINVAL */\n\
  if (write2(1, \"OK\", 2) != 2) return 2; /* the caller's personality survived the refusal */\n\
  return 42;\n\
}}\n"
    );
    let e = run_interp_setup(&src, |host, posix| {
        stage_executable(host, posix, "/bin/bad", BAD);
    });
    assert_eq!(
        e.result,
        vec![Value::I32(42)],
        "sig drift refused at bind; the caller kept running with its personality intact"
    );
    assert_eq!(e.stdout, b"OK", "the post-refusal write proves the restore");
}

/// #972 exec carry — **the pipeline crosses the exec boundary**: the shell pipes, forks, and the
/// twin `dup2`s the write end onto fd 1, closes its originals, and `execve`s a `__px_`-linked
/// command. The image-replace carries the twin's pipe ends into the fresh powerbox (counts bumped
/// before the old ends release — no EOF/EPIPE dip) and fires the personality's exec-remap hook,
/// re-pointing the fd-1 token at the carried end's new handle — so the exec'd command's plain
/// `write(1, …)` flows through the tag redirect into the pipe. The parent, parked in `read`, gets
/// the command's bytes, then **true EOF** when the command exits (its teardown releases the carried
/// write end), and reaps the command's status. `cmd | shell`-shape, personality-native.
#[test]
fn c_pipeline_crosses_the_exec_boundary() {
    const WR: &str = r#"
long __px_write(int cap, long fd, long buf, long len);
long __vm_write(int fd, void *buf, long len);
static long px_h_(long r) { return r <= -1048576 ? -(r + 1048576) : -1; }
static long wr(long fd, void *b, long n) {
  long r = __px_write(0, fd, (long)b, n);
  long h = px_h_(r);
  if (h < 0) return r;
  return __vm_write((int)h, b, n);
}
int main(void) {
  if (wr(1, "EXEC>", 5) != 5) return 90;  /* fd 1 is the dup2'd pipe: tag -> remapped handle */
  return 6;
}
"#;
    let src = format!(
        "{WIN_PAD_17}{PIPE_SHIM}\n{EXEC_C}\n\
long __px_fork(int cap, long a);\n\
long __px_waitpid(int cap, long pid, long status, long opts);\n\
long __px_dup2(int cap, long o, long n);\n\
static char *av[] = {{ \"wr\", 0 }};\n\
static int fds[2];\n\
static int status;\n\
static char b[8];\n\
static long pid; static long h;\n\
int main(void) {{\n\
  if (pipe(fds) != 0) return 1;\n\
  pid = __px_fork(0, 0);\n\
  if (pid < 0) return 2;\n\
  if (pid == 0) {{\n\
    if (__px_dup2(0, fds[1], 1) != 1) return 91;\n\
    close(fds[0]);                        /* the twin's read end: released */\n\
    close(fds[1]);                        /* not the last dup: fd 1 keeps the write end */\n\
    execve(\"/bin/wr\", av, 0);\n\
    return 99;\n\
  }}\n\
  close(fds[1]);                          /* parent's write end gone: the twin's keeps it open */\n\
  long n = read(fds[0], b, 8);            /* PARKS until the exec'd command writes */\n\
  if (n != 5) return 3;\n\
  if (b[0] != 'E' || b[4] != '>') return 4;\n\
  n = read(fds[0], b, 8);                 /* command exited: carried end released -> true EOF */\n\
  if (n != 0) return 5;\n\
  h = __px_waitpid(0, pid, (long)&status, 0);\n\
  if (h != pid) return 7;\n\
  if (((status >> 8) & 0xff) != 6) return 8;\n\
  return 42;\n\
}}\n"
    );
    let e = run_interp_setup(&src, |host, posix| {
        stage_executable(host, posix, "/bin/wr", WR);
    });
    assert_eq!(
        e.result,
        vec![Value::I32(42)],
        "the exec'd command's write flowed through the carried, remapped pipe end; EOF and status reaped"
    );
    // #1080 rung 4 — the same on the **bytecode engine**: pipe-through-exec (the real `echo … | cmd`
    // shape). Combines #1086's exec pipe-end carry (the twin `dup2`s + `execve`s, the image-replace
    // re-installs the write end + fires the exec-remap hook) with the CorePipe read park (the parent
    // blocks on the empty FIFO) and EOF-on-exit (the exec'd command's teardown releases the end).
    let eb = run_bytecode_setup(&src, |host, posix| {
        stage_executable(host, posix, "/bin/wr", WR);
    });
    assert_eq!(
        eb.result,
        vec![Value::I32(42)],
        "bytecode: the exec'd command wrote through the carried pipe end, the parent parked + woke, EOF + reap — matching the oracle"
    );
    // #748 rungs 2+3 composed on the **parallel driver**: the twin thread dup2s + execs (pipe-end
    // carry + exec-remap across the in-place host swap), the parent thread blocks reading, EOFs
    // when the command's teardown releases the carried end, and condvar-reaps the status.
    let ep = run_bytecode_parallel_setup(&src, |host, posix| {
        stage_executable(host, posix, "/bin/wr", WR);
    });
    assert_eq!(
        ep.result,
        vec![Value::I32(42)],
        "parallel driver: the pipeline crossed the exec boundary — carried end, blocking read, EOF, reap — matching both oracles"
    );
}

/// #801 — **`#!` scripts, one level**: `execve` of a memfs file whose first line is
/// `#!/bin/echo6 X` re-execs the interpreter with argv spliced per POSIX —
/// `[interp, optarg, scriptpath, argv[1..]]` — so the (registered, `__px_`-linked) interpreter
/// command sees all four and the parent reaps its status through the twin's pid.
#[test]
fn c_execve_runs_a_hashbang_script() {
    const INTERP: &str = r#"
int main(int argc, char **argv) {
  if (argc != 4) return 80 + argc;
  if (argv[1][0] != 'X') return 91;      /* the #! optional arg */
  if (argv[2][0] != '/') return 92;      /* the script path */
  if (argv[3][0] != 't') return 93;      /* the caller's argv[1] */
  return 6;
}
"#;
    let src = format!(
        "{WIN_PAD_17}{EXEC_C}\n\
long __px_fork(int cap, long a);\n\
long __px_waitpid(int cap, long pid, long status, long opts);\n\
static char *av[] = {{ \"s\", \"tail\", 0 }};\n\
static int status;\n\
static long pid; static long h;\n\
int main(void) {{\n\
  pid = __px_fork(0, 0);\n\
  if (pid < 0) return 1;\n\
  if (pid == 0) {{\n\
    execve(\"/s.sh\", av, 0);\n\
    return 99;\n\
  }}\n\
  h = __px_waitpid(0, pid, (long)&status, 0);\n\
  if (h != pid) return 2;\n\
  if (((status >> 8) & 0xff) != 6) return 200 + ((status >> 8) & 0xff);\n\
  return 42;\n\
}}\n"
    );
    let e = run_interp_setup(&src, |host, posix| {
        stage_executable(host, posix, "/bin/echo6", INTERP);
        posix.write_file("/s.sh", b"#!/bin/echo6 X\necho hi\n");
    });
    assert_eq!(
        e.result,
        vec![Value::I32(42)],
        "the #! line re-exec'd the interpreter with [interp, X, /s.sh, tail]"
    );
}

/// #797 — like [`run_interp_only`] but with the **controlling terminal** enabled and a feeder
/// thread delivering keystrokes (and winsize changes) on best-effort delays while the guest runs.
/// Ordering is best-effort only — a feed landing before the guest's read means the read drains
/// without parking, which is equally correct; the assertions are on results, not on parking.
fn run_interp_terminal(
    src: &str,
    feeds: Vec<(u64, Vec<u8>)>,
    winsize: Option<(u64, i32, i32)>,
) -> Effects {
    let ir = c_to_ir(src);
    let raw = parse_module_raw(&ir)
        .unwrap_or_else(|e| panic!("parse IR failed: {e:?}\n--- IR ---\n{ir}"));
    let win = 1u64
        << raw
            .memory
            .expect("the frontend declares a window")
            .size_log2;
    let mut ih = Host::new();
    let (iposix, ipx) = setup(&mut ih, win);
    iposix.enable_terminal(&mut ih);
    verify_module(&raw).unwrap_or_else(|e| panic!("verify failed: {e:?}\n--- IR ---\n{ir}"));
    bind_shim(&raw, &mut ih, ipx);
    let feeder = {
        let px = iposix.clone();
        std::thread::spawn(move || {
            for (delay, bytes) in feeds {
                std::thread::sleep(std::time::Duration::from_millis(delay));
                px.feed_terminal(&bytes);
            }
            if let Some((delay, r, c)) = winsize {
                std::thread::sleep(std::time::Duration::from_millis(delay));
                px.set_winsize(r, c);
            }
        })
    };
    let mut fuel = 200_000_000u64;
    let (result, exited) = match run_with_host(&raw, 0, &[], &mut fuel, &mut ih) {
        Ok(v) => (v, None),
        Err(Trap::Exit(c)) => (Vec::new(), Some(c)),
        Err(e) => panic!("interp trapped: {e:?}\n--- IR ---\n{ir}"),
    };
    feeder.join().expect("feeder thread");
    Effects {
        result,
        exited,
        stdout: iposix.stdout(),
        file_f: iposix.read_file("f"),
    }
}

/// #1122 — the **cooperative bytecode** twin of [`run_interp_terminal`]: the same controlling
/// terminal + timed feeder thread, driven by the coop `drive` with the external-wake doorbell
/// armed ([`Host::arm_external_wake`]). Without the doorbell the first parked terminal read is the
/// pump's all-parked deadlock (`ThreadFault`) — the doorbell block at that point, woken by the
/// feed-time pipe-wake ring, is exactly what this harness witnesses.
fn run_bytecode_terminal(src: &str, feeds: Vec<(u64, Vec<u8>)>) -> Effects {
    let ir = c_to_ir(src);
    let raw = parse_module_raw(&ir)
        .unwrap_or_else(|e| panic!("parse IR failed: {e:?}\n--- IR ---\n{ir}"));
    let win = 1u64
        << raw
            .memory
            .expect("the frontend declares a window")
            .size_log2;
    let mut ih = Host::new();
    let (iposix, ipx) = setup(&mut ih, win);
    iposix.enable_terminal(&mut ih);
    ih.arm_external_wake();
    verify_module(&raw).unwrap_or_else(|e| panic!("verify failed: {e:?}\n--- IR ---\n{ir}"));
    bind_shim(&raw, &mut ih, ipx);
    let feeder = {
        let px = iposix.clone();
        std::thread::spawn(move || {
            for (delay, bytes) in feeds {
                std::thread::sleep(std::time::Duration::from_millis(delay));
                px.feed_terminal(&bytes);
            }
        })
    };
    let mut fuel = 200_000_000u64;
    let ran = temen_interp::bytecode::compile_and_run_with_host(&raw, 0, &[], &mut fuel, &mut ih)
        .expect("the bytecode engine compiles this module (no declining op)");
    let (result, exited) = match ran {
        Ok(v) => (v, None),
        Err(Trap::Exit(c)) => (Vec::new(), Some(c)),
        Err(e) => panic!("bytecode trapped: {e:?}\n--- IR ---\n{ir}"),
    };
    feeder.join().expect("feeder thread");
    Effects {
        result,
        exited,
        stdout: iposix.stdout(),
        file_f: iposix.read_file("f"),
    }
}

/// [`run_interp_terminal`] with a `|host, posix|` setup callback (e.g. `stage_executable`) — the
/// controlling-terminal harness for guests that `fork` + `execve` a `/bin` command on the terminal.
fn run_interp_terminal_setup(
    src: &str,
    feeds: Vec<(u64, Vec<u8>)>,
    extra: impl Fn(&mut Host, &Posix),
) -> Effects {
    let ir = c_to_ir(src);
    let raw = parse_module_raw(&ir)
        .unwrap_or_else(|e| panic!("parse IR failed: {e:?}\n--- IR ---\n{ir}"));
    let win = 1u64
        << raw
            .memory
            .expect("the frontend declares a window")
            .size_log2;
    let mut ih = Host::new();
    let (iposix, ipx) = setup(&mut ih, win);
    iposix.enable_terminal(&mut ih);
    extra(&mut ih, &iposix);
    verify_module(&raw).unwrap_or_else(|e| panic!("verify failed: {e:?}\n--- IR ---\n{ir}"));
    bind_shim(&raw, &mut ih, ipx);
    let feeder = {
        let px = iposix.clone();
        std::thread::spawn(move || {
            for (delay, bytes) in feeds {
                std::thread::sleep(std::time::Duration::from_millis(delay));
                px.feed_terminal(&bytes);
            }
        })
    };
    let mut fuel = 200_000_000u64;
    let (result, exited) = match run_with_host(&raw, 0, &[], &mut fuel, &mut ih) {
        Ok(v) => (v, None),
        Err(Trap::Exit(c)) => (Vec::new(), Some(c)),
        Err(e) => panic!("interp trapped: {e:?}\n--- IR ---\n{ir}"),
    };
    feeder.join().expect("feeder thread");
    Effects {
        result,
        exited,
        stdout: iposix.stdout(),
        file_f: iposix.read_file("f"),
    }
}

/// [`run_bytecode_terminal`] with a `|host, posix|` setup callback (e.g. `stage_executable`) — the
/// cooperative-bytecode twin for `fork` + `execve` on the controlling terminal (the browser tier).
fn run_bytecode_terminal_setup(
    src: &str,
    feeds: Vec<(u64, Vec<u8>)>,
    extra: impl Fn(&mut Host, &Posix),
) -> Effects {
    let ir = c_to_ir(src);
    let raw = parse_module_raw(&ir)
        .unwrap_or_else(|e| panic!("parse IR failed: {e:?}\n--- IR ---\n{ir}"));
    let win = 1u64
        << raw
            .memory
            .expect("the frontend declares a window")
            .size_log2;
    let mut ih = Host::new();
    let (iposix, ipx) = setup(&mut ih, win);
    iposix.enable_terminal(&mut ih);
    ih.arm_external_wake();
    extra(&mut ih, &iposix);
    verify_module(&raw).unwrap_or_else(|e| panic!("verify failed: {e:?}\n--- IR ---\n{ir}"));
    bind_shim(&raw, &mut ih, ipx);
    let feeder = {
        let px = iposix.clone();
        std::thread::spawn(move || {
            for (delay, bytes) in feeds {
                std::thread::sleep(std::time::Duration::from_millis(delay));
                px.feed_terminal(&bytes);
            }
        })
    };
    let mut fuel = 200_000_000u64;
    let ran = temen_interp::bytecode::compile_and_run_with_host(&raw, 0, &[], &mut fuel, &mut ih)
        .expect("the bytecode engine compiles this module (no declining op)");
    let (result, exited) = match ran {
        Ok(v) => (v, None),
        Err(Trap::Exit(c)) => (Vec::new(), Some(c)),
        Err(e) => panic!("bytecode trapped: {e:?}\n--- IR ---\n{ir}"),
    };
    feeder.join().expect("feeder thread");
    Effects {
        result,
        exited,
        stdout: iposix.stdout(),
        file_f: iposix.read_file("f"),
    }
}

/// #1122 — **a parked terminal read on the cooperative bytecode driver blocks for the feeder**
/// instead of faulting: the guest's `read(0)` parks on the empty terminal pipe with every task
/// parked (pre-doorbell this was the pump's deadlock → `ThreadFault`), the feeder's line lands
/// 60ms later (deposit + pipe-wake ring → the doorbell block returns → the settle re-polls →
/// the rewound read drains the line), and a `^D` on an empty line EOFs the second read.
/// Differentialled against the tree-walker terminal harness on the same guest + feed script.
#[test]
fn c_a_terminal_read_blocks_for_the_feeder_on_bytecode() {
    let src = format!(
        "{PIPE_SHIM}\n\
static char b[16];\n\
int main(void) {{\n\
  long n = read(0, b, 16);            /* parks: ALL tasks blocked until the feeder types */\n\
  if (n != 3) return (int)(100 + n);\n\
  if (b[0] != 'h' || b[1] != 'i' || b[2] != 10) return 2;\n\
  long m = read(0, b, 16);            /* parks again; ^D on an empty line -> EOF (0) */\n\
  if (m != 0) return (int)(200 + m);\n\
  return 42;\n\
}}\n"
    );
    let feeds = || vec![(60, b"hi\n".to_vec()), (60, b"\x04".to_vec())];
    let interp = run_interp_terminal(&src, feeds(), None);
    assert_eq!(
        interp.result,
        vec![Value::I32(42)],
        "tree-walker: the parked terminal read drained the fed line, then EOF'd on ^D"
    );
    let byte = run_bytecode_terminal(&src, feeds());
    assert_eq!(
        byte.result,
        vec![Value::I32(42)],
        "coop bytecode: the all-parked pump blocked on the doorbell and the feed woke it — matching the oracle"
    );
    assert_eq!(
        byte.stdout, interp.stdout,
        "the terminal echo is identical across the two engines"
    );
}

/// #797 — **a parked terminal read wakes on a completed canonical line, edited and echoed**: the
/// guest blocks in `read(0)` (the tag redirect parks on the empty input pipe — a real prompt);
/// the feeder types `hI`, erases the `I` (`VERASE`), types `i\n` — the guest receives the EDITED
/// line `hi\n`, and the echo (including backspace-space-backspace for the erase) landed on stdout.
#[test]
fn c_terminal_canonical_read_line_editing_and_echo() {
    let src = format!(
        "{PIPE_SHIM}\n\
static char b[16];\n\
int main(void) {{\n\
  long n = read(0, b, 16);            /* parks until the line completes */\n\
  if (n != 3) return (int)(100 + n);\n\
  if (b[0] != 'h' || b[1] != 'i' || b[2] != 10) return 2;\n\
  return 42;\n\
}}\n"
    );
    let e = run_interp_terminal(&src, vec![(60, b"hI\x7fi\n".to_vec())], None);
    assert_eq!(e.result, vec![Value::I32(42)], "the edited line arrived");
    assert_eq!(
        e.stdout, b"hI\x08 \x08i\n",
        "echo mirrored the typing, erase as backspace-space-backspace"
    );
}

/// #797 — **`^C` interrupts a parked terminal read as a signal, not data**: the guest catches
/// SIGINT and blocks in `read(0)`; the fed `VINTR` byte never enters the stream — the discipline
/// fires the #798 group kill at the foreground group, the #796 EINTR path wakes the park, the
/// read returns `-EINTR` (plain `signal()`, SysV no-restart), and the doorbell holds the handler.
///
/// #1146 slice 2 — the **cooperative bytecode** twin now matches: the all-parked pump sees the
/// `^C`-raised deliverable SIGINT, sweeps the parked read (`Host::set_sig_interrupt` + re-admit),
/// the rewound read completes `-EINTR`, and the slice-1 safepoint redirect runs the handler. This
/// is the parked-read half of the async-signal gap that the compute-loop witnesses (slice 1) left.
#[test]
fn c_terminal_ctrl_c_interrupts_a_parked_read() {
    let src = format!(
        "{PIPE_SHIM}\n\
long __px_signal(int cap, long signum, long handler);\n\
long __px_sigaltstack(int cap, long sp, long size);\n\
static char sigstk[16384];\n\
static volatile int fired;\n\
static void handler(int sig) {{ fired = sig; }}\n\
static char b[8];\n\
int main(void) {{\n\
  __px_signal(0, 2, (long)handler);   /* catch SIGINT (plain signal(): no SA_RESTART) */\n\
  __px_sigaltstack(0, (long)sigstk, 16384); /* async delivery on (the #796 policy gate) */\n\
  long n = read(0, b, 8);             /* parks; ^C must interrupt, not feed */\n\
  if (n != -4) return (int)(100 - n); /* -EINTR */\n\
  if (fired != 2) return 2;           /* the handler ran on the delivery */\n\
  return 42;\n\
}}\n"
    );
    let e = run_interp_terminal(&src, vec![(60, b"\x03".to_vec())], None);
    assert_eq!(
        e.result,
        vec![Value::I32(42)],
        "tree-walker: ^C -> SIGINT at the fg group -> EINTR from the parked read, handler pending"
    );
    let byte = run_bytecode_terminal(&src, vec![(60, b"\x03".to_vec())]);
    assert_eq!(
        byte.result,
        vec![Value::I32(42)],
        "coop bytecode: the all-parked EINTR sweep interrupted the parked read and delivered the handler — matching the oracle"
    );
}

/// #797 — **`^D` on an empty line is true one-shot EOF**: the held writer count drops to 0 and the
/// parked reader wakes to `read() == 0` — bash's exit-on-EOF, through a real park.
#[test]
fn c_terminal_ctrl_d_is_true_eof() {
    let src = format!(
        "{PIPE_SHIM}\n\
static char b[8];\n\
int main(void) {{\n\
  long n = read(0, b, 8);\n\
  if (n != 0) return (int)(100 + n);\n\
  return 42;\n\
}}\n"
    );
    let e = run_interp_terminal(&src, vec![(60, b"\x04".to_vec())], None);
    assert_eq!(e.result, vec![Value::I32(42)], "^D on an empty line: EOF");
}

/// #797 — **raw mode + winsize + SIGWINCH**: the guest reads the default termios, flips off
/// ICANON/ECHO (keeping ISIG) via `tcsetattr`, reads a single byte fed with no newline (raw
/// delivery), checks the default 80×24 winsize, then waits for the embedder's `set_winsize` —
/// SIGWINCH (caught) rings the doorbell and `tcgetwinsize` reports the new size. Nothing echoed
/// (ECHO off before any input).
#[test]
fn c_terminal_raw_mode_winsize_and_sigwinch() {
    let src = format!(
        "{PIPE_SHIM}\n\
long __px_tcgetattr(int cap, long fd, long p);\n\
long __px_tcsetattr(int cap, long fd, long p);\n\
long __px_tcgetwinsize(int cap, long fd, long p);\n\
long __px_signal(int cap, long signum, long handler);\n\
long __px_sigcheck(int cap, long z);\n\
static long t[4];\n\
static int ws[2];\n\
static char b[4];\n\
int main(void) {{\n\
  __px_signal(0, 28, 555);                    /* catch SIGWINCH */\n\
  if (__px_tcgetattr(0, 0, (long)t) != 0) return 1;\n\
  if ((t[0] & 02) == 0 || (t[0] & 010) == 0) return 2;  /* canonical+echo default */\n\
  t[0] = t[0] & ~02L & ~010L;                 /* raw: ICANON|ECHO off, ISIG stays */\n\
  if (__px_tcsetattr(0, 0, (long)t) != 0) return 3;\n\
  if (__px_tcgetwinsize(0, 0, (long)ws) != 0) return 4;\n\
  if (ws[0] != 24 || ws[1] != 80) return 5;\n\
  long n = read(0, b, 1);                      /* raw: a single byte, no newline needed */\n\
  if (n != 1 || b[0] != 'x') return 6;\n\
  long h = 0;\n\
  while (h != 555) h = __px_sigcheck(0, 0);    /* poll until SIGWINCH lands */\n\
  if (__px_tcgetwinsize(0, 0, (long)ws) != 0) return 7;\n\
  if (ws[0] != 50 || ws[1] != 120) return 8;\n\
  return 42;\n\
}}\n"
    );
    let e = run_interp_terminal(&src, vec![(60, b"x".to_vec())], Some((150, 50, 120)));
    assert_eq!(
        e.result,
        vec![Value::I32(42)],
        "raw single-byte read, default and updated winsize, SIGWINCH doorbell"
    );
    assert_eq!(
        e.stdout, b"",
        "ECHO was off before any input: nothing echoed"
    );
}

// ---------------------------------------------------------------------------
// #801 — coreutils staging: a real /bin of registered executables
// (crates/temen-run/demos/posix_utils/). Each tool is its own command module —
// util.c (the #972 tag-protocol fd runtime) + the tool source, grep also
// carrying posix_libc/regex.c — registered under /bin so execvp finds it on
// PATH and fork→exec pipelines run genuine multi-command workloads.
// ---------------------------------------------------------------------------

const UTIL_C: &str = include_str!("../../temen-run/demos/posix_utils/util.c");

/// Pads a witness DRIVER's window to ml >= 17: `exec` runs a command **in the caller's window**,
/// so the caller's committed extent must fit the command's declared image — and the /bin tools
/// (util.c + their statics) declare ml=17. Without the pad a small driver's exec is a clean
/// `-EINVAL` refusal (the admissibility gate bounds by the committed extent), which is honest but
/// not what these witnesses are for.
const WIN_PAD_17: &str = "static char xs_win_pad_[65536];\n";

/// Compile and register `/bin/<name>` for each requested tool. Staging only
/// what a witness needs keeps the per-test chibicc cost proportional.
fn stage_coreutils(host: &mut Host, posix: &Posix, names: &[&str]) {
    const TOOLS: &[(&str, &str, bool)] = &[
        (
            "true",
            include_str!("../../temen-run/demos/posix_utils/true.c"),
            false,
        ),
        (
            "false",
            include_str!("../../temen-run/demos/posix_utils/false.c"),
            false,
        ),
        (
            "echo",
            include_str!("../../temen-run/demos/posix_utils/echo.c"),
            false,
        ),
        (
            "cat",
            include_str!("../../temen-run/demos/posix_utils/cat.c"),
            false,
        ),
        (
            "seq",
            include_str!("../../temen-run/demos/posix_utils/seq.c"),
            false,
        ),
        (
            "head",
            include_str!("../../temen-run/demos/posix_utils/head.c"),
            false,
        ),
        (
            "wc",
            include_str!("../../temen-run/demos/posix_utils/wc.c"),
            false,
        ),
        (
            "sort",
            include_str!("../../temen-run/demos/posix_utils/sort.c"),
            false,
        ),
        (
            "uniq",
            include_str!("../../temen-run/demos/posix_utils/uniq.c"),
            false,
        ),
        (
            "grep",
            include_str!("../../temen-run/demos/posix_utils/grep.c"),
            true,
        ),
        (
            "ls",
            include_str!("../../temen-run/demos/posix_utils/ls.c"),
            false,
        ),
        (
            "pwd",
            include_str!("../../temen-run/demos/posix_utils/pwd.c"),
            false,
        ),
    ];
    for want in names {
        let (_, src, rx) = TOOLS
            .iter()
            .find(|(n, _, _)| n == want)
            .unwrap_or_else(|| panic!("no such coreutil: {want}"));
        let tu = if *rx {
            format!("{UTIL_C}\n{REGEX_C}\n{src}")
        } else {
            format!("{UTIL_C}\n{src}")
        };
        stage_executable(host, posix, &format!("/bin/{want}"), &tu);
    }
}

/// #801 coreutils — **the three-stage pipeline, every stage a real exec'd
/// program**: `seq 100 | head -n 10 | wc -l` with all plumbing done the Unix
/// way (pipe + fork + dup2 + execvp-on-PATH). The parent reads exactly "10\n"
/// from the tail pipe, gets true EOF once wc exits, and reaps three zero
/// statuses — carried pipe ends, PATH resolution, and argv delivery all
/// crossing three exec boundaries at once.
#[test]
fn c_coreutils_pipeline_seq_head_wc() {
    let src = format!(
        "{WIN_PAD_17}{PIPE_SHIM}\n{EXEC_C}\n\
long __px_fork(int cap, long a);\n\
long __px_waitpid(int cap, long pid, long status, long opts);\n\
long __px_dup2(int cap, long o, long n);\n\
long __px_setenv(int cap, long name, long nlen, long val, long vlen, long ow);\n\
static char *av1[] = {{ \"seq\", \"100\", 0 }};\n\
static char *av2[] = {{ \"head\", \"-n\", \"10\", 0 }};\n\
static char *av3[] = {{ \"wc\", \"-l\", 0 }};\n\
static int pa[2]; static int pb[2]; static int pc[2];\n\
static int st1; static int st2; static int st3;\n\
static char b[16];\n\
static long p1; static long p2; static long p3;\n\
static void shut(void) {{\n\
  close(pa[0]); close(pa[1]); close(pb[0]); close(pb[1]); close(pc[0]); close(pc[1]);\n\
}}\n\
int main(void) {{\n\
  __px_setenv(0, (long)\"PATH\", 4, (long)\"/bin\", 4, 1);\n\
  if (pipe(pa) != 0 || pipe(pb) != 0 || pipe(pc) != 0) return 1;\n\
  p1 = __px_fork(0, 0);\n\
  if (p1 < 0) return 2;\n\
  if (p1 == 0) {{\n\
    __px_dup2(0, pa[1], 1); shut();\n\
    execvp(\"seq\", av1); return 99;\n\
  }}\n\
  p2 = __px_fork(0, 0);\n\
  if (p2 < 0) return 2;\n\
  if (p2 == 0) {{\n\
    __px_dup2(0, pa[0], 0); __px_dup2(0, pb[1], 1); shut();\n\
    execvp(\"head\", av2); return 99;\n\
  }}\n\
  p3 = __px_fork(0, 0);\n\
  if (p3 < 0) return 2;\n\
  if (p3 == 0) {{\n\
    __px_dup2(0, pb[0], 0); __px_dup2(0, pc[1], 1); shut();\n\
    execvp(\"wc\", av3); return 99;\n\
  }}\n\
  close(pa[0]); close(pa[1]); close(pb[0]); close(pb[1]); close(pc[1]);\n\
  long got = 0;\n\
  for (;;) {{\n\
    long n = read(pc[0], b + got, 16 - got);\n\
    if (n < 0) return 3;\n\
    if (n == 0) break;                       /* true EOF: wc exited, all dups gone */\n\
    got = got + n;\n\
  }}\n\
  if (got != 3) return 4;\n\
  if (b[0] != '1' || b[1] != '0' || b[2] != '\\n') return 5;\n\
  close(pc[0]);\n\
  if (__px_waitpid(0, p1, (long)&st1, 0) != p1) return 6;\n\
  if (__px_waitpid(0, p2, (long)&st2, 0) != p2) return 7;\n\
  if (__px_waitpid(0, p3, (long)&st3, 0) != p3) return 8;\n\
  if (((st1 >> 8) & 0xff) != 0) return 9;\n\
  if (((st2 >> 8) & 0xff) != 0) return 10;\n\
  if (((st3 >> 8) & 0xff) != 0) return 11;\n\
  return 42;\n\
}}\n"
    );
    let e = run_interp_setup(&src, |host, posix| {
        stage_coreutils(host, posix, &["seq", "head", "wc"]);
    });
    assert_eq!(
        e.result,
        vec![Value::I32(42)],
        "seq | head | wc across three exec boundaries: \"10\\n\", true EOF, three zero statuses"
    );
    // #1080 rung 4 — the whole three-stage pipeline on the **bytecode engine**: three forks, three
    // `execve`d coreutils, three CorePipes, all the read parks + carried pipe ends + EOF-on-exit +
    // reaps composing at once. This is bash's `seq 100 | head -n 10 | wc -l` minus bash itself.
    let eb = run_bytecode_setup(&src, |host, posix| {
        stage_coreutils(host, posix, &["seq", "head", "wc"]);
    });
    assert_eq!(
        eb.result,
        vec![Value::I32(42)],
        "bytecode: seq | head | wc — three exec'd stages piped together, \"10\\n\" + true EOF + three reaps, matching the oracle"
    );
    // #748 — the full pipeline on the **parallel driver**: three fork threads each exec'ing a
    // coreutil, three CorePipes with real cross-thread blocking at every stage, three reaps. The
    // heaviest composition the dual-driver principle demands of rungs 0-3.
    let ep = run_bytecode_parallel_setup(&src, |host, posix| {
        stage_coreutils(host, posix, &["seq", "head", "wc"]);
    });
    assert_eq!(
        ep.result,
        vec![Value::I32(42)],
        "parallel driver: seq | head | wc across three exec'd OS-thread stages, matching both oracles"
    );
}

/// #801 coreutils — **parent-fed `sort | uniq -c`**: the parent writes an
/// unsorted, duplicated stream into the head pipe, closes it, and reads the
/// collapsed counted output from the tail — sort's whole-input buffering
/// (park until the feed closes) and uniq's adjacent-run collapse both running
/// as exec'd programs.
#[test]
fn c_coreutils_sort_uniq_counts() {
    let src = format!(
        "{WIN_PAD_17}{PIPE_SHIM}\n{EXEC_C}\n\
long __px_fork(int cap, long a);\n\
long __px_waitpid(int cap, long pid, long status, long opts);\n\
long __px_dup2(int cap, long o, long n);\n\
long __px_setenv(int cap, long name, long nlen, long val, long vlen, long ow);\n\
static char *av1[] = {{ \"sort\", 0 }};\n\
static char *av2[] = {{ \"uniq\", \"-c\", 0 }};\n\
static int pa[2]; static int pb[2]; static int pc[2];\n\
static int st1; static int st2;\n\
static char b[16];\n\
static char exp[9] = \"3 a\\n2 b\\n\";\n\
static long p1; static long p2;\n\
static void shut(void) {{\n\
  close(pa[0]); close(pa[1]); close(pb[0]); close(pb[1]); close(pc[0]); close(pc[1]);\n\
}}\n\
int main(void) {{\n\
  __px_setenv(0, (long)\"PATH\", 4, (long)\"/bin\", 4, 1);\n\
  if (pipe(pa) != 0 || pipe(pb) != 0 || pipe(pc) != 0) return 1;\n\
  p1 = __px_fork(0, 0);\n\
  if (p1 < 0) return 2;\n\
  if (p1 == 0) {{\n\
    __px_dup2(0, pa[0], 0); __px_dup2(0, pb[1], 1); shut();\n\
    execvp(\"sort\", av1); return 99;\n\
  }}\n\
  p2 = __px_fork(0, 0);\n\
  if (p2 < 0) return 2;\n\
  if (p2 == 0) {{\n\
    __px_dup2(0, pb[0], 0); __px_dup2(0, pc[1], 1); shut();\n\
    execvp(\"uniq\", av2); return 99;\n\
  }}\n\
  close(pa[0]); close(pb[0]); close(pb[1]); close(pc[1]);\n\
  if (write(pa[1], \"b\\na\\nb\\na\\na\\n\", 10) != 10) return 3;\n\
  close(pa[1]);                              /* sort's stdin EOFs: it can sort and emit */\n\
  long got = 0;\n\
  for (;;) {{\n\
    long n = read(pc[0], b + got, 16 - got);\n\
    if (n < 0) return 4;\n\
    if (n == 0) break;\n\
    got = got + n;\n\
  }}\n\
  if (got != 8) return 5;\n\
  long i;\n\
  for (i = 0; i < 8; i = i + 1) if (b[i] != exp[i]) return 6;\n\
  close(pc[0]);\n\
  if (__px_waitpid(0, p1, (long)&st1, 0) != p1) return 7;\n\
  if (__px_waitpid(0, p2, (long)&st2, 0) != p2) return 8;\n\
  if (((st1 >> 8) & 0xff) != 0) return 9;\n\
  if (((st2 >> 8) & 0xff) != 0) return 10;\n\
  return 42;\n\
}}\n"
    );
    let e = run_interp_setup(&src, |host, posix| {
        stage_coreutils(host, posix, &["sort", "uniq"]);
    });
    assert_eq!(
        e.result,
        vec![Value::I32(42)],
        "parent-fed sort | uniq -c: \"3 a\\n2 b\\n\" and two zero statuses"
    );
    // #1080 rung 4 — the same on the **bytecode engine**: the parent feeds the head pipe and closes
    // it; `sort` (exec'd) blocks reading its whole input until that EOF, then emits — a long read park
    // resolved by the writer's close — and `uniq -c` collapses it, both across exec boundaries.
    let eb = run_bytecode_setup(&src, |host, posix| {
        stage_coreutils(host, posix, &["sort", "uniq"]);
    });
    assert_eq!(
        eb.result,
        vec![Value::I32(42)],
        "bytecode: parent-fed sort | uniq -c parked until the feed closed, then collapsed the run — matching the oracle"
    );
    // #748 — the same on the **parallel driver**: `sort`'s whole-input read blocks its OS thread
    // until the parent's close flips `pipe_read_ready`'s writers-gone arm.
    let ep = run_bytecode_parallel_setup(&src, |host, posix| {
        stage_coreutils(host, posix, &["sort", "uniq"]);
    });
    assert_eq!(
        ep.result,
        vec![Value::I32(42)],
        "parallel driver: parent-fed sort | uniq -c across exec'd threads, matching both oracles"
    );
}

/// #801 coreutils — **grep is the posix_libc ERE engine running as a
/// program**: an anchored alternation over a memfs file prints exactly the
/// matching lines and exits 0; a miss prints nothing and exits 1 (the
/// grep(1) exit contract shells' `if grep -q ...` lives on).
#[test]
fn c_coreutils_grep_matches_and_exit_codes() {
    let src = format!(
        "{WIN_PAD_17}{PIPE_SHIM}\n{EXEC_C}\n\
long __px_fork(int cap, long a);\n\
long __px_waitpid(int cap, long pid, long status, long opts);\n\
long __px_dup2(int cap, long o, long n);\n\
long __px_setenv(int cap, long name, long nlen, long val, long vlen, long ow);\n\
static char *av1[] = {{ \"grep\", \"^(al|gam)\", \"/data.txt\", 0 }};\n\
static char *av2[] = {{ \"grep\", \"zeta\", \"/data.txt\", 0 }};\n\
static int fds[2];\n\
static int st1; static int st2;\n\
static char b[32];\n\
static char exp[17] = \"alpha 1\\ngamma 3\\n\";\n\
static long p1; static long p2;\n\
int main(void) {{\n\
  __px_setenv(0, (long)\"PATH\", 4, (long)\"/bin\", 4, 1);\n\
  if (pipe(fds) != 0) return 1;\n\
  p1 = __px_fork(0, 0);\n\
  if (p1 < 0) return 2;\n\
  if (p1 == 0) {{\n\
    __px_dup2(0, fds[1], 1); close(fds[0]); close(fds[1]);\n\
    execvp(\"grep\", av1); return 99;\n\
  }}\n\
  close(fds[1]);\n\
  long got = 0;\n\
  for (;;) {{\n\
    long n = read(fds[0], b + got, 32 - got);\n\
    if (n < 0) return 3;\n\
    if (n == 0) break;\n\
    got = got + n;\n\
  }}\n\
  close(fds[0]);\n\
  if (got != 16) return 4;\n\
  long i;\n\
  for (i = 0; i < 16; i = i + 1) if (b[i] != exp[i]) return 5;\n\
  if (__px_waitpid(0, p1, (long)&st1, 0) != p1) return 6;\n\
  if (((st1 >> 8) & 0xff) != 0) return 7;      /* hit: exit 0 */\n\
  p2 = __px_fork(0, 0);\n\
  if (p2 < 0) return 8;\n\
  if (p2 == 0) {{\n\
    execvp(\"grep\", av2); return 99;\n\
  }}\n\
  if (__px_waitpid(0, p2, (long)&st2, 0) != p2) return 9;\n\
  if (((st2 >> 8) & 0xff) != 1) return 10;     /* miss: exit 1 */\n\
  return 42;\n\
}}\n"
    );
    let e = run_interp_setup(&src, |host, posix| {
        stage_coreutils(host, posix, &["grep"]);
        posix.write_file("/data.txt", b"alpha 1\nbeta 2\ngamma 3\n");
    });
    assert_eq!(
        e.result,
        vec![Value::I32(42)],
        "grep over memfs: anchored ERE prints the two hits and exits 0; a miss exits 1"
    );
}

/// #801 coreutils — **the smoke sweep**: true/false exit codes, echo's argv
/// join, cat of a memfs file, ls's sorted directory listing, and pwd — six
/// tools exec'd through one capture helper, each output byte-checked. This is
/// the "real /bin" witness: a populated userland any shell loop can walk.
#[test]
fn c_coreutils_smoke_echo_cat_ls_pwd_true_false() {
    let src = format!(
        "{WIN_PAD_17}{PIPE_SHIM}\n{EXEC_C}\n\
long __px_fork(int cap, long a);\n\
long __px_waitpid(int cap, long pid, long status, long opts);\n\
long __px_dup2(int cap, long o, long n);\n\
long __px_setenv(int cap, long name, long nlen, long val, long vlen, long ow);\n\
static char ob[64];\n\
static long olen;\n\
static int runtool(char **av, long want) {{\n\
  int fds[2];\n\
  if (pipe(fds) != 0) return 1;\n\
  long pid = __px_fork(0, 0);\n\
  if (pid < 0) return 2;\n\
  if (pid == 0) {{\n\
    __px_dup2(0, fds[1], 1); close(fds[0]); close(fds[1]);\n\
    execvp(av[0], av);\n\
    return 99;                                /* exec failed: exits the twin via main */\n\
  }}\n\
  close(fds[1]);\n\
  olen = 0;\n\
  for (;;) {{\n\
    long n = read(fds[0], ob + olen, 64 - olen);\n\
    if (n < 0) return 3;\n\
    if (n == 0) break;\n\
    olen = olen + n;\n\
  }}\n\
  close(fds[0]);\n\
  int st;\n\
  if (__px_waitpid(0, pid, (long)&st, 0) != pid) return 4;\n\
  if (((st >> 8) & 0xff) != want) return 5;\n\
  return 0;\n\
}}\n\
static int outis(char *s) {{\n\
  long i = 0;\n\
  while (s[i]) {{ if (i >= olen || ob[i] != s[i]) return 0; i = i + 1; }}\n\
  return i == olen;\n\
}}\n\
static char *avt[] = {{ \"true\", 0 }};\n\
static char *avf[] = {{ \"false\", 0 }};\n\
static char *ave[] = {{ \"echo\", \"hi\", \"there\", 0 }};\n\
static char *avc[] = {{ \"cat\", \"/f.txt\", 0 }};\n\
static char *avl[] = {{ \"ls\", \"/d\", 0 }};\n\
static char *avp[] = {{ \"pwd\", 0 }};\n\
int main(void) {{\n\
  __px_setenv(0, (long)\"PATH\", 4, (long)\"/bin\", 4, 1);\n\
  int r;\n\
  r = runtool(avt, 0); if (r) return 10 + r;\n\
  if (!outis(\"\")) return 19;\n\
  r = runtool(avf, 1); if (r) return 20 + r;\n\
  if (!outis(\"\")) return 29;\n\
  r = runtool(ave, 0); if (r) return 30 + r;\n\
  if (!outis(\"hi there\\n\")) return 39;\n\
  r = runtool(avc, 0); if (r) return 40 + r;\n\
  if (!outis(\"xyz\\n\")) return 49;\n\
  r = runtool(avl, 0); if (r) return 50 + r;\n\
  if (!outis(\"a\\nb\\n\")) return 59;\n\
  r = runtool(avp, 0); if (r) return 60 + r;\n\
  if (!outis(\"/\\n\")) return 69;\n\
  return 42;\n\
}}\n"
    );
    let e = run_interp_setup(&src, |host, posix| {
        stage_coreutils(host, posix, &["true", "false", "echo", "cat", "ls", "pwd"]);
        posix.write_file("/f.txt", b"xyz\n");
        posix.write_file("/d/a", b"1");
        posix.write_file("/d/b", b"2");
    });
    assert_eq!(
        e.result,
        vec![Value::I32(42)],
        "true/false statuses, echo join, cat of memfs, sorted ls, pwd — all as exec'd /bin tools"
    );
}

/// #801 coreutils (regression) — **argv survives repeated fork→exec rounds**: the args-region
/// pack in exec.c must stage in private scratch, because a non-child caller's own statics (these
/// `av` arrays included) legitimately live inside `[128, 16384)` — the in-place pack used to
/// trample them mid-loop on the second round (caught by the coreutils grep witness).
#[test]
fn c_exec_argv_survives_a_second_twin() {
    const CMD: &str = r#"
int main(int argc, char **argv) {
  if (argc != 2) return 80 + argc;
  if (argv[1][0] != 'x') return 79;
  return 7;
}
"#;
    let src = format!(
        "{WIN_PAD_17}{PIPE_SHIM}\n{EXEC_C}\n\
long __px_fork(int cap, long a);\n\
long __px_waitpid(int cap, long pid, long status, long opts);\n\
long __px_setenv(int cap, long name, long nlen, long val, long vlen, long ow);\n\
static char *av[] = {{ \"c\", \"x\", 0 }};\n\
static int st;\n\
int main(void) {{\n\
  __px_setenv(0, (long)\"PATH\", 4, (long)\"/bin\", 4, 1);\n\
  long pid = __px_fork(0, 0);\n\
  if (pid < 0) return 1;\n\
  if (pid == 0) {{ execve(\"/bin/c\", av, 0); return 99; }}\n\
  if (__px_waitpid(0, pid, (long)&st, 0) != pid) return 2;\n\
  if (((st >> 8) & 0xff) != 7) return 3;\n\
  pid = __px_fork(0, 0);\n\
  if (pid < 0) return 4;\n\
  if (pid == 0) {{ execve(\"/bin/c\", av, 0); return 99; }}\n\
  if (__px_waitpid(0, pid, (long)&st, 0) != pid) return 5;\n\
  if (((st >> 8) & 0xff) != 7) return 6;\n\
  return 42;\n\
}}\n"
    );
    let e = run_interp_setup(&src, |host, posix| {
        stage_executable(host, posix, "/bin/c", CMD);
    });
    assert_eq!(
        e.result,
        vec![Value::I32(42)],
        "argv survives a second twin's execve"
    );
}

/// #1080 rungs 2+3 combined on the **bytecode engine** — the whole external-command flow: `fork()` (a
/// twin, rung 3's fork park), the twin `execve`s a staged `/bin` command (rung 2's `env: Some`
/// image-replace + personality carry — the path only the ignored browser test covered until now), and
/// the parent `waitpid`s for its exit (rung 3's reap park). This is the native, disk-cheap proof that
/// bash's fork→exec→wait runs on the playground tier, differentialled against the tree-walker oracle.
#[test]
fn c_fork_execve_waitpid_on_bytecode() {
    const CMD: &str = r#"
int main(int argc, char **argv) {
  if (argc != 2) return 80 + argc;
  if (argv[1][0] != 'x') return 79;
  return 7;
}
"#;
    let src = format!(
        "{WIN_PAD_17}{PIPE_SHIM}\n{EXEC_C}\n\
long __px_fork(int cap, long a);\n\
long __px_waitpid(int cap, long pid, long status, long opts);\n\
static char *av[] = {{ \"c\", \"x\", 0 }};\n\
static int st;\n\
int main(void) {{\n\
  long pid = __px_fork(0, 0);\n\
  if (pid < 0) return 1;\n\
  if (pid == 0) {{ execve(\"/bin/c\", av, 0); return 99; }}\n\
  if (__px_waitpid(0, pid, (long)&st, 0) != pid) return 2;\n\
  if (((st >> 8) & 0xff) != 7) return 3;\n\
  return 42;\n\
}}\n"
    );
    let e = run_bytecode_setup(&src, |host, posix| {
        stage_executable(host, posix, "/bin/c", CMD);
    });
    assert_eq!(
        e.result,
        vec![Value::I32(42)],
        "the bytecode engine forked, the twin execve'd /bin/c (env:Some image-replace), and the parent \
         reaped its exit 7 — rungs 2+3 combined, matching the tree-walker"
    );
    // #748 rung 2 — the same flow on the **parallel driver**: the fork twin is a real OS thread
    // that `execve`s in place (its host cell's contents swap to the command powerbox, its dispatch
    // table to the command's), exits 7, and the parent's condvar `waitpid` reaps it.
    let ep = run_bytecode_parallel_setup(&src, |host, posix| {
        stage_executable(host, posix, "/bin/c", CMD);
    });
    assert_eq!(
        ep.result,
        vec![Value::I32(42)],
        "parallel driver: fork → execve → waitpid across a real OS thread, matching both oracles"
    );
}

/// #802 slice 1 — **the bash keystone idiom: `longjmp` OUT of a signal handler** back to a
/// `setjmp` re-entry point (`throw_to_top_level` from `sigint_sighandler` — bash's entire
/// error/interrupt model). Three things must hold, in bash's exact shape:
/// 1. the escape: a handler delivered against a **parked terminal read** longjmps to the
///    top-level `setjmp`, abandoning the interrupted read's own EINTR return path;
/// 2. re-armed delivery: a **second** `^C` after the escape delivers again — the unwind must not
///    leave the signal marked handler-in-progress (#796 block-during-handler), or bash would eat
///    exactly one interrupt per session;
/// 3. liveness: after two escapes the terminal still works — a real line arrives.
#[test]
fn c_longjmp_out_of_a_signal_handler_bash_shape() {
    let src = format!(
        "{PIPE_SHIM}\n\
long __px_signal(int cap, long signum, long handler);\n\
long __px_sigaltstack(int cap, long sp, long size);\n\
typedef long jmp_buf[8];\n\
int setjmp(jmp_buf env);\n\
void longjmp(jmp_buf env, int val);\n\
static jmp_buf top_level;\n\
static char sigstk[16384];\n\
static volatile int throws;\n\
static void handler(int sig) {{\n\
  if (sig != 2) return;\n\
  throws = throws + 1;\n\
  longjmp(top_level, throws);      /* bash: throw_to_top_level */\n\
}}\n\
static char b[8];\n\
int main(void) {{\n\
  __px_signal(0, 2, (long)handler);        /* catch SIGINT */\n\
  __px_sigaltstack(0, (long)sigstk, 16384); /* async delivery on (the #796 policy gate) */\n\
  int r = setjmp(top_level);               /* bash's reader-loop re-entry point */\n\
  if (r == 0) {{\n\
    read(0, b, 8);                         /* parks; first ^C -> handler -> longjmp(1) */\n\
    return 90;                             /* unreachable: the escape abandons the EINTR path */\n\
  }}\n\
  if (r == 1) {{\n\
    read(0, b, 8);                         /* parks again; the second ^C must deliver */\n\
    return 89;                             /* unreachable */\n\
  }}\n\
  if (r != 2) return 80 + r;\n\
  long n = read(0, b, 8);                  /* liveness: a real line still arrives */\n\
  if (n != 2 || b[0] != 'x') return 70;\n\
  return 42;\n\
}}\n"
    );
    let e = run_interp_terminal(
        &src,
        vec![
            (60, b"\x03".to_vec()),
            (160, b"\x03".to_vec()),
            (260, b"x\n".to_vec()),
        ],
        None,
    );
    assert_eq!(
        e.result,
        vec![Value::I32(42)],
        "handler longjmp escaped twice, delivery re-armed, terminal alive"
    );
}

/// #1171 — **a stopped foreground job wakes the shell's blocked `waitpid`, and only the shell's**
/// (the core of the browser `^Z`). Three domains, no terminal: the shell `A` catches `SIGCHLD` (job
/// control) and forks the job `B` (which **parks on a pipe read** — exactly `cat` on stdin, blocked
/// with a live writer) and a stopper `C`. `A` blocks in `waitpid(B, WUNTRACED)` **first**; `C`
/// `SIGTSTP`s `B`. `B`'s stop must (1) leave `B` parked — a stopped process makes no progress — and
/// (2) `SIGCHLD` the shell, waking its parked `waitpid` so it reports the stop (`(20 << 8) | 0x7f`).
///
/// The cooperative bytecode driver's all-parked signal sweep used to interrupt **every** pipe-parked
/// task whenever the root host had a deliverable signal — so `A`'s `SIGCHLD` `-EINTR`'d `B`'s own
/// blocked read, running `B` off the end (it exits 7) and `A` reaped that exit instead of the stop
/// (`0x0700`, `WEXITSTATUS 7` — the bug). #1171 makes the sweep **domain-scoped** (invariant 12: a
/// signal to `A` never sweeps `B`'s park) and extends it to wake a `waitpid` park whose own domain has
/// a pending signal, so the shell's `SIGCHLD` wakes its `waitpid` (which re-runs and finds `B`'s fresh
/// stop) while `B` stays parked. The tree-walker already did this via its domain-scoped park interrupt;
/// this is the bytecode twin. Both engines return 42.
const STOP_WAKES_SHELL_WAIT_SRC: &str = r#"
long __px_signal(int cap, long signum, long handler);
long __px_sigaltstack(int cap, long sp, long size);
long __px_fork(int cap, long a);
long __px_waitpid(int cap, long pid, long status, long opts);
long __px_kill(int cap, long pid, long sig);
long __vm_pipe(int *fds);
long __vm_read(int fd, void *buf, long len);
long __px_pipe_adopt(int cap, long rh, long wh, long fdp);
long __px_read(int cap, long fd, long buf, long len);
static char sigstk[16384];
static volatile long chld;
static void on_chld(int s) { chld = chld + 1; }   /* the shell's SIGCHLD handler (job control) */
static int status;
static long bpid, cpid, i;
static volatile long acc;
static int fds[2];
static char buf[8];
static long px_h_(long r) { return r <= -1048576 ? -(r + 1048576) : -1; }
int main(void) {
  __px_signal(0, 17, (long)on_chld);         /* catch SIGCHLD(17) */
  __px_sigaltstack(0, (long)sigstk, 16384);  /* async delivery on */
  bpid = __px_fork(0, 0);
  if (bpid < 0) return 1;
  if (bpid == 0) {                           /* B — the foreground job: park on a pipe read forever */
    int h[2];
    __vm_pipe(h);
    __px_pipe_adopt(0, h[0], h[1], (long)fds);
    long r = __px_read(0, fds[0], (long)buf, 8);   /* PARKS (empty, B holds the write end) */
    long hh = px_h_(r);
    if (hh >= 0) __vm_read((int)hh, buf, 8);
    return 7;                                /* only reached if B is wrongly run off its stopped read */
  }
  cpid = __px_fork(0, 0);
  if (cpid < 0) return 2;
  if (cpid == 0) {                           /* C — the stopper (its own quantum) */
    for (i = 0; i < 40000; i = i + 1) acc = acc + 1;  /* let the shell bench in waitpid first */
    __px_kill(0, bpid, 20);                  /* SIGTSTP B while the shell is parked in its wait */
    return 3;
  }
  long h;
  while ((h = __px_waitpid(0, bpid, (long)&status, 2)) == -4) {   /* waitpid(B, WUNTRACED), retry on EINTR */
  }
  if (h != bpid) return 4;
  if ((status & 0xff) != 0x7f) return 2000 + (status & 0xffff);  /* WIFSTOPPED, not a reaped exit */
  if (((status >> 8) & 0xff) != 20) return 5;                     /* by SIGTSTP(20) */
  return 42;
}
"#;

#[test]
fn c_a_stopped_foreground_job_wakes_the_shells_parked_waitpid() {
    let e = run_interp_only(STOP_WAKES_SHELL_WAIT_SRC, |_| {});
    assert_eq!(
        e.result,
        vec![Value::I32(42)],
        "tree-walker: the job's stop SIGCHLD'd the shell, waking its parked waitpid(WUNTRACED) to \
         report the stop — the job stayed parked (domain-scoped park interrupt)"
    );
    let b = run_bytecode_only(STOP_WAKES_SHELL_WAIT_SRC, |_| {});
    assert_eq!(
        b.result,
        vec![Value::I32(42)],
        "coop bytecode (the browser tier): the domain-scoped all-parked sweep woke the shell's \
         waitpid on its SIGCHLD without EINTR'ing the job's own stopped read — matching the oracle"
    );
}

/// #1171 — **the interactive `^Z` end to end through the controlling terminal** (the browser flow,
/// reduced to the personality band). A shell catches `SIGCHLD` (job control), forks a job, foregrounds
/// it (`setpgid` + `tcsetpgrp`), and blocks in `waitpid(job, WUNTRACED)`. The job — once it confirms it
/// IS the terminal foreground (spin on `tcgetpgrp`, so no `SIGTTIN` background-read race) — blocks on
/// the terminal `read`. The feeder types `^Z`: the feed-time line discipline raises `SIGTSTP` at the
/// terminal's foreground group (the job's), the job **stops** while parked on its read, and the
/// resulting `SIGCHLD` wakes the shell's parked `waitpid`, which reports the stop (`(20 << 8) | 0x7f`).
/// The shell reclaims the terminal, `SIGCONT`s the job, the feeder's line lands, the resumed read
/// returns it, and the shell reaps the job's exit `7`.
///
/// This exercises the whole browser `^Z` path over the #797 terminal on **both** the tree-walker and
/// the cooperative bytecode driver (the tier the interactive card runs). It only passes on bytecode
/// because of the domain-scoped all-parked sweep (this PR): before it, the shell's `SIGCHLD` `-EINTR`'d
/// the job's own stopped read, running it off, so the shell reaped an exit instead of the stop. The
/// job runs on its own (fork-twin) domain, so the sweep must interrupt the shell's `waitpid` without
/// touching the job's parked read.
const TERM_CTRL_Z_SRC: &str = r#"
long __px_signal(int cap, long signum, long handler);
long __px_sigaltstack(int cap, long sp, long size);
long __px_fork(int cap, long a);
long __px_waitpid(int cap, long pid, long status, long opts);
long __px_kill(int cap, long pid, long sig);
long __px_setpgid(int cap, long pid, long pgid);
long __px_tcsetpgrp(int cap, long fd, long pgid);
long __vm_pipe(int *fds);
long __vm_read(int fd, void *buf, long len);
long __vm_write(int fd, void *buf, long len);
long __px_pipe_adopt(int cap, long rh, long wh, long fdp);
long __px_read(int cap, long fd, long buf, long len);
long __px_write(int cap, long fd, long buf, long len);
static char sigstk[16384];
static volatile long chld;
static void on_chld(int s) { chld = chld + 1; }
static int status;
static long cpid;
static int sync_fds[2];
static char buf[8];
static long px_h_(long r) { return r <= -1048576 ? -(r + 1048576) : -1; }
static long rd(void) {
  for (;;) {
    long r = __px_read(0, 0, (long)buf, 8);
    long hh = px_h_(r);
    if (hh >= 0) return __vm_read((int)hh, buf, 8);   /* parked-read result */
    if (r == -85 || r == -4) continue;                /* ERESTART / EINTR: re-issue */
    return r;
  }
}
int main(void) {
  __px_signal(0, 17, (long)on_chld);         /* the shell's SIGCHLD handler (job control) */
  __px_sigaltstack(0, (long)sigstk, 16384);  /* async delivery on */
  int h2[2];
  __vm_pipe(h2);
  __px_pipe_adopt(0, h2[0], h2[1], (long)sync_fds);  /* a sync pipe: "you are the foreground now" */
  cpid = __px_fork(0, 0);
  if (cpid < 0) return 1;
  if (cpid == 0) {                           /* the foreground job */
    __px_setpgid(0, 0, 0);                   /* own group (race-safe with the shell) */
    char g;                                  /* PARK on the sync pipe until the shell foregrounds us: */
    long r = __px_read(0, sync_fds[0], (long)&g, 1);   /* no busy-wait, no SIGTTIN race */
    long hh = px_h_(r);
    if (hh >= 0) __vm_read((int)hh, &g, 1);
    long n = rd();                           /* block on the terminal read; ^Z stops us here */
    if (n <= 0) return 20;
    return 7;                                /* the fed byte arrived after the SIGCONT resume */
  }
  __px_setpgid(0, cpid, cpid);               /* race-safe */
  __px_tcsetpgrp(0, 0, cpid);                /* hand the terminal to the job */
  char g = 'g';                              /* release the job onto the (now foreground) terminal read */
  long wr = __px_write(0, sync_fds[1], (long)&g, 1);
  long wh = px_h_(wr);
  if (wh >= 0) __vm_write((int)wh, &g, 1);
  long h;
  while ((h = __px_waitpid(0, cpid, (long)&status, 2)) == -4) { }  /* waitpid(WUNTRACED), retry EINTR */
  if (h != cpid) return 4;
  if ((status & 0xff) != 0x7f) return 2000 + (status & 0xffff);    /* WIFSTOPPED, not a reaped exit */
  if (((status >> 8) & 0xff) != 20) return 5;                       /* stopped by SIGTSTP(20) */
  __px_tcsetpgrp(0, 0, 1);                   /* the shell reclaims the terminal */
  __px_kill(0, cpid, 18);                    /* SIGCONT the job (`fg`) */
  while ((h = __px_waitpid(0, cpid, (long)&status, 0)) == -4) { }
  if (h != cpid) return 6;
  if (((status >> 8) & 0xff) != 7) return 2100 + (status & 0xffff); /* WEXITSTATUS = 7 */
  return 42;
}
"#;

#[test]
fn c_terminal_ctrl_z_stops_the_foreground_job_and_the_shell_reports_and_resumes_it() {
    // The job PARKS on a sync pipe until the shell foregrounds it and writes "go" (no busy-wait, so the
    // setup is not preemption-timing-dependent); it is on the terminal read well before the ^Z. The ^Z
    // stops it; the fed line lands after the SIGCONT resume. Feed delays are generous for slow CI.
    let feeds = || vec![(150u64, b"\x1a".to_vec()), (600u64, b"x\n".to_vec())];
    let e = run_interp_terminal(TERM_CTRL_Z_SRC, feeds(), None);
    assert_eq!(
        e.result,
        vec![Value::I32(42)],
        "tree-walker: ^Z stopped the foreground job on its terminal read, the shell's waitpid(WUNTRACED) \
         reported the stop, SIGCONT resumed it, and the shell reaped exit 7"
    );
    let b = run_bytecode_terminal(TERM_CTRL_Z_SRC, feeds());
    assert_eq!(
        b.result,
        vec![Value::I32(42)],
        "coop bytecode (the browser tier): the same ^Z round-trip — the job's stop woke the shell's \
         parked waitpid via the domain-scoped sweep without EINTR'ing the job's read — matching the oracle"
    );
}

const CATR_CMD: &str = r#"
long __vm_read(int fd, void *buf, long len);
long __px_read(int cap, long fd, long buf, long len);
static char b[8];
static long px_h_(long r){ return r <= -1048576 ? -(r+1048576) : -1; }
int main(void){
  for(;;){
    long r = __px_read(0, 0, (long)b, 8);
    long hh = px_h_(r);
    if (hh >= 0){ long n = __vm_read((int)hh, b, 8); return n <= 0 ? 20 : 7; }
    if (r == -85 || r == -4) continue;   /* ERESTART/EINTR: re-issue */
    return 30;
  }
}
"#;

fn zexec_src() -> String {
    format!(
        "{WIN_PAD_17}{EXEC_C}\n\
long __px_signal(int cap, long signum, long handler);\n\
long __px_sigaltstack(int cap, long sp, long size);\n\
long __px_fork(int cap, long a);\n\
long __px_waitpid(int cap, long pid, long status, long opts);\n\
long __px_kill(int cap, long pid, long sig);\n\
long __px_setpgid(int cap, long pid, long pgid);\n\
long __px_tcsetpgrp(int cap, long fd, long pgid);\n\
long __vm_pipe(int *fds);\n\
long __vm_read(int fd, void *buf, long len);\n\
long __vm_write(int fd, void *buf, long len);\n\
long __px_pipe_adopt(int cap, long rh, long wh, long fdp);\n\
long __px_write(int cap, long fd, long buf, long len);\n\
static char sigstk[16384];\n\
static volatile long chld;\n\
static void on_chld(int s){{ chld = chld + 1; }}\n\
static int status;\n\
static long cpid;\n\
static int sync_fds[2];\n\
static char *av[] = {{ \"catr\", 0 }};\n\
static long ph_(long r){{ return r <= -1048576 ? -(r+1048576) : -1; }}\n\
int main(void){{\n\
  __px_signal(0, 17, (long)on_chld);\n\
  __px_sigaltstack(0, (long)sigstk, 16384);\n\
  int h2[2]; __vm_pipe(h2); __px_pipe_adopt(0, h2[0], h2[1], (long)sync_fds);\n\
  cpid = __px_fork(0, 0);\n\
  if (cpid < 0) return 1;\n\
  if (cpid == 0) {{\n\
    __px_setpgid(0, 0, 0);\n\
    char g; long r = __px_read(0, sync_fds[0], (long)&g, 1);\n\
    long hh = ph_(r); if (hh >= 0) __vm_read((int)hh, &g, 1);\n\
    execve(\"/bin/catr\", av, 0);\n\
    return 99;\n\
  }}\n\
  __px_setpgid(0, cpid, cpid);\n\
  __px_tcsetpgrp(0, 0, cpid);\n\
  char g = 'g'; long wr = __px_write(0, sync_fds[1], (long)&g, 1);\n\
  long wh = ph_(wr); if (wh >= 0) __vm_write((int)wh, &g, 1);\n\
  long h;\n\
  while ((h = __px_waitpid(0, cpid, (long)&status, 2)) == -4) {{ }}\n\
  if (h != cpid) return 4;\n\
  if ((status & 0xff) != 0x7f) return 2000 + (status & 0xffff);\n\
  if (((status>>8)&0xff) != 20) return 5;\n\
  __px_tcsetpgrp(0, 0, 1);\n\
  __px_kill(0, cpid, 18);\n\
  while ((h = __px_waitpid(0, cpid, (long)&status, 0)) == -4) {{ }}\n\
  if (h != cpid) return 6;\n\
  if (((status>>8)&0xff) != 7) return 2100 + (status & 0xffff);\n\
  return 42;\n\
}}\n"
    )
}

/// #1171 — **the interactive `^Z` when the foreground job is a real `fork` + `execve` command** (the
/// browser shape: bash suspends `cat`, not a fork twin of itself). The shell foregrounds a job that,
/// after a sync-pipe handshake, `execve`s `/bin/catr` — a staged command that blocks reading the
/// terminal. `^Z` stops the exec'd job on its terminal read, its `SIGCHLD` wakes the shell's parked
/// `waitpid(WUNTRACED)` (reporting the stop), `SIGCONT` resumes it, the fed line returns, and the shell
/// reaps exit 7. Proves the stop/report/resume path survives an exec image-replace — the twin's own
/// signal doors are re-wired at exec, its pgid/ppid carry, and the domain-scoped sweep still wakes only
/// the shell's `waitpid`. Both engines return 42.
#[test]
fn c_terminal_ctrl_z_stops_an_execd_foreground_command_and_the_shell_reports_it() {
    let feeds = || vec![(150u64, b"\x1a".to_vec()), (600u64, b"x\n".to_vec())];
    let e = run_interp_terminal_setup(&zexec_src(), feeds(), |host, posix| {
        stage_executable(host, posix, "/bin/catr", CATR_CMD);
    });
    assert_eq!(
        e.result,
        vec![Value::I32(42)],
        "tree-walker: ^Z stopped the exec'd foreground command on its terminal read, the shell's \
         waitpid(WUNTRACED) reported the stop, SIGCONT resumed it, and the shell reaped exit 7"
    );
    let b = run_bytecode_terminal_setup(&zexec_src(), feeds(), |host, posix| {
        stage_executable(host, posix, "/bin/catr", CATR_CMD);
    });
    assert_eq!(
        b.result,
        vec![Value::I32(42)],
        "coop bytecode (the browser tier): the same ^Z round-trip across a fork+execve job — stop \
         reported, resumed, reaped — matching the oracle"
    );
}

/// #1171 — **a stopped child wakes a shell blocked in `waitpid(WUNTRACED)` even with NO async SIGCHLD
/// delivery** — the bash shape. Real bash installs a `SIGCHLD` handler but no sigaltstack, so its
/// `SIGCHLD` is poll-only: a parked `waitpid` cannot be woken by the async-delivery door, and a
/// foreground job stopped on a parked read never enters a core stop-park that would drain the reap
/// waiters. So the shell here catches `SIGCHLD` but registers **no** `sigaltstack`; a sibling `SIGTSTP`s
/// the job (parked on a pipe read); the shell's blocked `waitpid` must still wake and report the stop.
/// Before #1171 this HUNG on both engines. The wake now rides the one-shot child-transition edge
/// (`notify_parent_chld`/`chld_to` → the parent's run-wake + `reap_wake`; the coop all-parked sweep
/// re-admits the reap park via `reap_pending`). Both engines return 42.
const STOP_WAKES_NOALT_SRC: &str = r#"
long __px_signal(int cap, long signum, long handler);
long __px_fork(int cap, long a);
long __px_waitpid(int cap, long pid, long status, long opts);
long __px_kill(int cap, long pid, long sig);
long __vm_pipe(int *fds);
long __vm_read(int fd, void *buf, long len);
long __px_pipe_adopt(int cap, long rh, long wh, long fdp);
long __px_read(int cap, long fd, long buf, long len);
static volatile long chld;
static void on_chld(int s){ chld = chld + 1; }   /* SIGCHLD handler, but NO sigaltstack (async off) */
static int status;
static long bpid, cpid, i;
static volatile long acc;
static int fds[2];
static char buf[8];
static long px_h_(long r){ return r <= -1048576 ? -(r+1048576) : -1; }
int main(void){
  __px_signal(0, 17, (long)on_chld);
  bpid = __px_fork(0,0);
  if (bpid < 0) return 1;
  if (bpid == 0) {                        /* the job: park on a pipe read (writer held open) */
    int h[2]; __vm_pipe(h); __px_pipe_adopt(0, h[0], h[1], (long)fds);
    long r = __px_read(0, fds[0], (long)buf, 8);
    long hh = px_h_(r); if (hh >= 0) __vm_read((int)hh, buf, 8);
    return 7;
  }
  cpid = __px_fork(0,0);
  if (cpid < 0) return 2;
  if (cpid == 0) { for(i=0;i<40000;i++) acc=acc+1; __px_kill(0, bpid, 20); return 3; }  /* SIGTSTP the job */
  long h;
  while ((h = __px_waitpid(0, bpid, (long)&status, 2)) == -4) { }  /* blocks; the stop must wake it */
  if (h != bpid) return 4;
  if ((status & 0xff) != 0x7f) return 2000 + (status & 0xffff);
  if (((status>>8)&0xff) != 20) return 5;
  return 42;
}
"#;

#[test]
fn c_a_stopped_job_wakes_a_shell_wait_without_async_sigchld() {
    let e = run_interp_only(STOP_WAKES_NOALT_SRC, |_| {});
    assert_eq!(
        e.result,
        vec![Value::I32(42)],
        "tree-walker: the job's stop woke the shell's parked waitpid(WUNTRACED) via the child-transition \
         run-wake — no async SIGCHLD delivery (no sigaltstack) needed"
    );
    let b = run_bytecode_only(STOP_WAKES_NOALT_SRC, |_| {});
    assert_eq!(
        b.result,
        vec![Value::I32(42)],
        "coop bytecode (the browser tier): the all-parked sweep re-admitted the shell's reap park on the \
         one-shot reap_pending edge — matching the oracle (this is the bash-shape browser ^Z wake)"
    );
}

fn zfg2_src() -> String {
    format!(
        "{WIN_PAD_17}{EXEC_C}\n\
long __px_signal(int cap, long signum, long handler);\n\
long __px_sigaltstack(int cap, long sp, long size);\n\
long __px_fork(int cap, long a);\n\
long __px_waitpid(int cap, long pid, long status, long opts);\n\
long __px_kill(int cap, long pid, long sig);\n\
long __px_setpgid(int cap, long pid, long pgid);\n\
long __px_tcsetpgrp(int cap, long fd, long pgid);\n\
long __vm_pipe(int *fds);\n\
long __vm_read(int fd, void *buf, long len);\n\
long __vm_write(int fd, void *buf, long len);\n\
long __px_pipe_adopt(int cap, long rh, long wh, long fdp);\n\
long __px_read(int cap, long fd, long buf, long len);\n\
long __px_write(int cap, long fd, long buf, long len);\n\
static char sigstk[16384];\n\
static volatile long chld;\n\
static void on_chld(int s){{ chld = chld + 1; }}\n\
static int status;\n\
static long cpid;\n\
static int sync_fds[2];\n\
static char *av[] = {{ \"catr\", 0 }};\n\
static long ph_(long r){{ return r <= -1048576 ? -(r+1048576) : -1; }}\n\
int main(void){{\n\
  __px_signal(0, 17, (long)on_chld);\n\
  __px_sigaltstack(0, (long)sigstk, 16384);\n\
  int h2[2]; __vm_pipe(h2); __px_pipe_adopt(0, h2[0], h2[1], (long)sync_fds);\n\
  cpid = __px_fork(0, 0);\n\
  if (cpid < 0) return 1;\n\
  if (cpid == 0) {{\n\
    __px_setpgid(0, 0, 0);\n\
    char g; long r = __px_read(0, sync_fds[0], (long)&g, 1);\n\
    long hh = ph_(r); if (hh >= 0) __vm_read((int)hh, &g, 1);\n\
    execve(\"/bin/catr\", av, 0);\n\
    return 99;\n\
  }}\n\
  __px_setpgid(0, cpid, cpid);\n\
  __px_tcsetpgrp(0, 0, cpid);\n\
  char g = 'g'; long wr = __px_write(0, sync_fds[1], (long)&g, 1);\n\
  long wh = ph_(wr); if (wh >= 0) __vm_write((int)wh, &g, 1);\n\
  long h;\n\
  while ((h = __px_waitpid(0, cpid, (long)&status, 2)) == -4) {{ }}\n\
  if (h != cpid) return 4;\n\
  if ((status & 0xff) != 0x7f) return 2000 + (status & 0xffff);\n\
  __px_tcsetpgrp(0, 0, 1);\n\
  /* fg: re-foreground + SIGCONT + block again in waitpid (resume-to-block).\n\
     No data is fed after CONT, so the resumed catr re-blocks on its read and\n\
     the shell's waitpid must PARK — the real bash `fg` shape. A second ^Z then\n\
     re-stops it so the park wakes and the probe terminates. */\n\
  __px_tcsetpgrp(0, 0, cpid);\n\
  __px_kill(0, -cpid, 18);\n\
  while ((h = __px_waitpid(0, cpid, (long)&status, 2)) == -4) {{ }}\n\
  if (h != cpid) return 6;\n\
  if ((status & 0xff) != 0x7f) return 2100 + (status & 0xffff);\n\
  return 42;\n\
}}\n"
    )
}

// #1171 — the `fg` resume-to-block shape, both engines. After ^Z stops the exec'd foreground `catr`,
// the shell reclaims the terminal, re-foregrounds the job, and SIGCONTs it via a *guest* `kill(-pgid,
// SIGCONT)` (exactly what bash's `fg`/`start_job` does — unlike the ^Z stop, which arrives inline
// through the line discipline). The continue-fire routes through the per-op syscall handler's deferred
// `wake_after`: on wasm32 that must fire inline (no `thread::spawn`; firing it on a detached thread
// panicked bash's `fg` to a bare `unreachable` in the browser). No data is fed after the CONT, so the
// resumed catr re-blocks on its read and the shell's second waitpid must PARK; a second ^Z then re-stops
// it, waking the park with a fresh WUNTRACED — the resume-to-block round-trip.
#[test]
fn c_terminal_fg_resumes_a_stopped_execd_job_then_restops_it() {
    let feeds = || vec![(150u64, b"\x1a".to_vec()), (900u64, b"\x1a".to_vec())];
    let e = run_interp_terminal_setup(&zfg2_src(), feeds(), |host, posix| {
        stage_executable(host, posix, "/bin/catr", CATR_CMD);
    });
    assert_eq!(
        e.result,
        vec![Value::I32(42)],
        "tree-walker: `fg` SIGCONT resumed the stopped job, its read re-blocked, and the second ^Z \
         re-stopped it — the shell's re-parked waitpid(WUNTRACED) woke on the fresh stop"
    );
    let b = run_bytecode_terminal_setup(&zfg2_src(), feeds(), |host, posix| {
        stage_executable(host, posix, "/bin/catr", CATR_CMD);
    });
    assert_eq!(
        b.result,
        vec![Value::I32(42)],
        "coop bytecode (the browser tier): the guest `kill(-pgid, SIGCONT)` fired its deferred continue \
         wake and re-parked the shell's reap — matching the oracle (this is the bash `fg` resume shape)"
    );
}

fn mjobs_src() -> String {
    format!(
        "{WIN_PAD_17}\
long __px_fork(int cap, long a);\n\
long __px_waitpid(int cap, long pid, long status, long opts);\n\
long __px_kill(int cap, long pid, long sig);\n\
long __px_setpgid(int cap, long pid, long pgid);\n\
long __vm_pipe(int *fds);\n\
long __vm_read(int fd, void *buf, long len);\n\
long __vm_write(int fd, void *buf, long len);\n\
long __px_pipe_adopt(int cap, long rh, long wh, long fdp);\n\
long __px_read(int cap, long fd, long buf, long len);\n\
long __px_write(int cap, long fd, long buf, long len);\n\
static int status;\n\
static long pa, pb;\n\
static int fa[2]; static int fb[2];\n\
static char buf[8];\n\
static long ph_(long r){{ return r <= -1048576 ? -(r+1048576) : -1; }}\n\
static long rd1(int fd){{ long r=__px_read(0,fd,(long)buf,1); long h=ph_(r); if(h>=0) return __vm_read((int)h,buf,1); return r; }}\n\
static void wr1(int fd){{ char g='x'; long r=__px_write(0,fd,(long)&g,1); long h=ph_(r); if(h>=0) __vm_write((int)h,&g,1); }}\n\
int main(void){{\n\
  int ha[2]; __vm_pipe(ha); __px_pipe_adopt(0, ha[0], ha[1], (long)fa);\n\
  int hb[2]; __vm_pipe(hb); __px_pipe_adopt(0, hb[0], hb[1], (long)fb);\n\
  pa = __px_fork(0,0);\n\
  if (pa < 0) return 1;\n\
  if (pa == 0){{ __px_setpgid(0,0,0); if (rd1(fa[0])<=0) return 90; return 7; }}\n\
  pb = __px_fork(0,0);\n\
  if (pb < 0) return 2;\n\
  if (pb == 0){{ __px_setpgid(0,0,0); if (rd1(fb[0])<=0) return 91; return 9; }}\n\
  __px_setpgid(0, pa, pa);\n\
  __px_setpgid(0, pb, pb);\n\
  /* stop BOTH background groups; the per-op stop poll parks each job stopped even though the stop\n\
     races their first read. The shell then collects two WUNTRACED reports across the two pgids. */\n\
  __px_kill(0, -pa, 20);\n\
  __px_kill(0, -pb, 20);\n\
  int sa=0, sb=0, k;\n\
  for (k=0;k<2;k++){{\n\
    long h; while ((h = __px_waitpid(0, -1, (long)&status, 2)) == -4){{}}\n\
    if (h < 0) return 100+k;\n\
    if ((status & 0xff) != 0x7f) return 2000 + (status & 0xffff);\n\
    if (h==pa) sa++; else if (h==pb) sb++; else return 200+k;\n\
  }}\n\
  if (sa!=1 || sb!=1) return 300;\n\
  __px_kill(0,-pa,18); __px_kill(0,-pb,18);\n\
  wr1(fa[1]); wr1(fb[1]);\n\
  int ea=0, eb=0;\n\
  for (k=0;k<2;k++){{\n\
    long h; while ((h = __px_waitpid(0, -1, (long)&status, 0)) == -4){{}}\n\
    if (h < 0) return 4000 + (int)(-h);\n\
    if ((status & 0x7f) != 0) return 500 + (status & 0xffff);\n\
    int code = (status>>8)&0xff;\n\
    if (h==pa){{ if(code!=7) return 601; ea++; }}\n\
    else if (h==pb){{ if(code!=9) return 602; eb++; }}\n\
    else return 603;\n\
  }}\n\
  if (ea!=1 || eb!=1) return 700;\n\
  return 42;\n\
}}\n"
    )
}

// #798 multiple concurrent jobs — the personality carries SEVERAL jobs at once, each in its own
// process group, and the shell manages them by group: two background jobs (each `setpgid`'d into its
// own pgid, then parked on a pipe read) are BOTH stopped via `kill(-pgid, SIGTSTP)`, the shell
// collects two `waitpid(-1, WUNTRACED)` stop reports across the two groups (the one-shot reap-wake
// edge must survive draining two pending transitions, not just one), then BOTH are continued via
// `kill(-pgid, SIGCONT)`, fed a byte so their reads complete, and reaped — each exit reported exactly
// once through the wildcard `waitpid(-1)`. This is the job-table-depth shape behind `cmd1 & cmd2 &`
// + `^Z`/`bg`/`fg %n`: the single-job #1171 path generalized to N groups. The waits retry `EINTR`
// (as real bash's reap loop does): a continue transition can spuriously wake a wildcard wait, and the
// tree-walker surfaces that as `EINTR` where the coop engine re-blocks — both converge once retried.
#[test]
fn c_multiple_background_jobs_stop_and_continue_across_process_groups() {
    let e = run_interp_only(&mjobs_src(), |_| {});
    assert_eq!(
        e.result,
        vec![Value::I32(42)],
        "tree-walker: two jobs in two pgids each stopped + reported via waitpid(-1, WUNTRACED), then \
         continued and reaped — one exit each through the wildcard wait"
    );
    let b = run_bytecode_only(&mjobs_src(), |_| {});
    assert_eq!(
        b.result,
        vec![Value::I32(42)],
        "coop bytecode (the browser tier): the all-parked sweep drained two pending stop transitions on \
         the one-shot reap edge and re-parked between them — matching the oracle"
    );
    let p = run_bytecode_parallel_only(&mjobs_src(), |_| {});
    assert_eq!(
        p.result,
        vec![Value::I32(42)],
        "parallel driver: the same two-group stop/continue/reap over real OS threads — matching the oracle"
    );
}

// #797 — ^D (VEOF) is a ONE-SHOT EOF, not a permanent terminal close. The guest reads the terminal
// twice: a ^D on the first (empty) read returns 0 (EOF), and a SECOND read then BLOCKS for fresh
// input — a later `x\n` completes it. Before the fix, the empty-line VEOF dropped the terminal's
// writer count to 0 permanently, so the second read EOF'd immediately (returned 0) instead of
// blocking — the bug that made a foreground `cat`'s ^D also EOF the shell's next prompt read.
const TERM_ONESHOT_EOF_SRC: &str = r#"
long __px_read(int cap, long fd, long buf, long len);
long __vm_read(int fd, void *buf, long len);
static char b[8];
static long ph_(long r){ return r <= -1048576 ? -(r+1048576) : -1; }
static long rterm(long n){ long r=__px_read(0,0,(long)b,n); long h=ph_(r); if(h>=0) return __vm_read((int)h,b,n); return r; }
int main(void){
  long n1 = rterm(8);                 /* ^D on an empty line -> EOF -> 0 */
  if (n1 != 0) return 100;
  long n2 = rterm(8);                 /* MUST block for input; permanent-EOF would return 0 here */
  if (n2 <= 0) return 200;
  if (b[0] != 'x') return 300;
  return 42;
}
"#;

#[test]
fn c_terminal_ctrl_d_eof_is_one_shot_not_a_permanent_close() {
    let feeds = || vec![(150u64, b"\x04".to_vec()), (600u64, b"x\n".to_vec())];
    let e = run_interp_terminal_setup(TERM_ONESHOT_EOF_SRC, feeds(), |_, _| {});
    assert_eq!(
        e.result,
        vec![Value::I32(42)],
        "tree-walker: the empty-line ^D EOF'd only the pending read; the second read re-armed the \
         terminal writer and blocked for the later `x` instead of inheriting the EOF"
    );
    let b = run_bytecode_terminal_setup(TERM_ONESHOT_EOF_SRC, feeds(), |_, _| {});
    assert_eq!(
        b.result,
        vec![Value::I32(42)],
        "coop bytecode (the browser tier): same one-shot ^D EOF — the shell's prompt read no longer \
         inherits a foreground job's ^D"
    );
}

fn ttin_bg_src() -> String {
    format!(
        "{WIN_PAD_17}\
long __px_fork(int cap, long a);\n\
long __px_waitpid(int cap, long pid, long status, long opts);\n\
long __px_setpgid(int cap, long pid, long pgid);\n\
long __px_read(int cap, long fd, long buf, long len);\n\
long __vm_read(int fd, void *buf, long len);\n\
static int status;\n\
static long pb;\n\
static char b[8];\n\
static long ph_(long r){{ return r <= -1048576 ? -(r+1048576) : -1; }}\n\
int main(void){{\n\
  pb = __px_fork(0,0);\n\
  if (pb < 0) return 1;\n\
  if (pb == 0){{ __px_setpgid(0,0,0); long r; do {{ r=__px_read(0,0,(long)b,8); }} while(r==-85); long h=ph_(r); if(h>=0) __vm_read((int)h,b,8); return 7; }}\n\
  __px_setpgid(0, pb, pb);\n\
  long h; while ((h=__px_waitpid(0, pb, (long)&status, 2)) == -4){{}}\n\
  if (h != pb) return 100;\n\
  if ((status & 0xff) == 0x7f) return 4000 + ((status>>8)&0xff);\n\
  return 5000 + ((status>>8)&0xff);\n\
}}\n"
    )
}

// #798/#1198 — a **background** process reading the controlling terminal is stopped by SIGTTIN. A
// forked job in its own process group (never `tcsetpgrp`'d to the foreground — the shell's group
// stays foreground) reads fd 0; `tty_background_check` raises SIGTTIN (default: stop), the read
// returns `ERESTART`, and the reader — retrying `ERESTART`, as a libc read wrapper does — parks
// stopped rather than draining input. The shell's `waitpid(WUNTRACED)` reports the stop with the
// stop signal SIGTTIN (21), so `(status & 0xff) == 0x7f` and `(status >> 8) == 21` → 4021 on both
// engines. This is the mechanism a correct `bg` of a terminal-reading job relies on (#1198): the
// resumed reader must SIGTTIN-stop, not steal the shell's input.
#[test]
fn c_a_background_terminal_read_is_stopped_by_sigttin() {
    let e = run_interp_terminal_setup(&ttin_bg_src(), vec![], |_, _| {});
    assert_eq!(
        e.result,
        vec![Value::I32(4021)],
        "tree-walker: the background read raised SIGTTIN, the reader re-issued on ERESTART and parked \
         stopped, and waitpid(WUNTRACED) reported the SIGTTIN(21) stop"
    );
    let b = run_bytecode_terminal_setup(&ttin_bg_src(), vec![], |_, _| {});
    assert_eq!(
        b.result,
        vec![Value::I32(4021)],
        "coop bytecode (the browser tier): same SIGTTIN background-read stop — matching the oracle"
    );
}

fn bgexec_src() -> String {
    format!(
        "{WIN_PAD_17}{EXEC_C}\n\
long __px_fork(int cap, long a);\n\
long __px_waitpid(int cap, long pid, long status, long opts);\n\
long __px_setpgid(int cap, long pid, long pgid);\n\
long __vm_pipe(int *fds);\n\
long __vm_read(int fd, void *buf, long len);\n\
long __vm_write(int fd, void *buf, long len);\n\
long __px_pipe_adopt(int cap, long rh, long wh, long fdp);\n\
long __px_read(int cap, long fd, long buf, long len);\n\
long __px_write(int cap, long fd, long buf, long len);\n\
static int status;\n\
static long cpid;\n\
static int sync_fds[2];\n\
static char *av[] = {{ \"catr\", 0 }};\n\
static long ph_(long r){{ return r <= -1048576 ? -(r+1048576) : -1; }}\n\
int main(void){{\n\
  int h2[2]; __vm_pipe(h2); __px_pipe_adopt(0, h2[0], h2[1], (long)sync_fds);\n\
  cpid = __px_fork(0, 0);\n\
  if (cpid < 0) return 1;\n\
  if (cpid == 0) {{\n\
    __px_setpgid(0, 0, 0);\n\
    char g; long r = __px_read(0, sync_fds[0], (long)&g, 1);\n\
    long hh = ph_(r); if (hh >= 0) __vm_read((int)hh, &g, 1);\n\
    execve(\"/bin/catr\", av, 0);\n\
    return 99;\n\
  }}\n\
  __px_setpgid(0, cpid, cpid);\n\
  /* do NOT tcsetpgrp: catr is a BACKGROUND job (the shell's group stays foreground) */\n\
  char g = 'g'; long wr = __px_write(0, sync_fds[1], (long)&g, 1);\n\
  long wh = ph_(wr); if (wh >= 0) __vm_write((int)wh, &g, 1);\n\
  long h; while ((h = __px_waitpid(0, cpid, (long)&status, 2)) == -4) {{ }}\n\
  if (h != cpid) return 100;\n\
  if ((status & 0xff) == 0x7f) return 4000 + ((status>>8)&0xff);\n\
  return 5000 + ((status>>8)&0xff);\n\
}}\n"
    )
}

// #1198 — the same SIGTTIN background-terminal stop as
// `c_a_background_terminal_read_is_stopped_by_sigttin`, but the reader is an **exec'd** coreutil
// (`execve("/bin/catr")`), not a fork twin still running the parent's image. This is the shape a
// real `bg cat` takes in the browser: the shell forks, the child `setpgid`s into its own group and
// `execve`s the command, and the command — never `tcsetpgrp`'d to the foreground — reads the
// terminal. `catr` retries `-ERESTART` in a `while` loop, exactly as a libc `read` wrapper does.
//
// On the tree-walker this already worked (a per-op `stop_flag` safepoint benches the stopped domain
// after one retry turn). On the cooperative bytecode engine — the browser tier — the exec'd reader
// SPUN: the engine has no per-op stop poll, so the `-ERESTART` retry loop re-issued `read` forever
// inside one `step_vcpu` call, never yielding, so the pick could never bench it and the parent's
// `waitpid` never woke (#1198). The fix yields to the pump at the syscall boundary when the domain
// stopped itself, and the round-robin pick skips a stopped domain — so both engines now bench the
// reader and report the SIGTTIN(21) stop word 4021.
#[test]
fn c_a_execd_background_terminal_read_is_stopped_by_sigttin() {
    let feeds = || vec![(400u64, b"x\n".to_vec())];
    let e = run_interp_terminal_setup(&bgexec_src(), feeds(), |host, posix| {
        stage_executable(host, posix, "/bin/catr", CATR_CMD);
    });
    assert_eq!(
        e.result,
        vec![Value::I32(4021)],
        "tree-walker: the exec'd background `catr` read raised SIGTTIN, retried `-ERESTART`, parked \
         stopped, and the parent's waitpid(WUNTRACED) reported the SIGTTIN(21) stop"
    );
    let b = run_bytecode_terminal_setup(&bgexec_src(), feeds(), |host, posix| {
        stage_executable(host, posix, "/bin/catr", CATR_CMD);
    });
    assert_eq!(
        b.result,
        vec![Value::I32(4021)],
        "coop bytecode (the browser tier): the exec'd background reader yields at the syscall stop \
         boundary and is benched by the pick — no ERESTART spin — matching the oracle"
    );
}

fn ttou_bg_src() -> String {
    format!(
        "{WIN_PAD_17}\
long __px_fork(int cap, long a);\n\
long __px_waitpid(int cap, long pid, long status, long opts);\n\
long __px_setpgid(int cap, long pid, long pgid);\n\
long __px_write(int cap, long fd, long buf, long len);\n\
static int status;\n\
static long pb;\n\
static char msg[3];\n\
int main(void){{\n\
  msg[0]='h'; msg[1]='i'; msg[2]='\\n';\n\
  pb = __px_fork(0,0);\n\
  if (pb < 0) return 1;\n\
  if (pb == 0){{ __px_setpgid(0,0,0); __px_write(0,1,(long)msg,3); return 7; }}\n\
  __px_setpgid(0, pb, pb);\n\
  long h; while ((h=__px_waitpid(0, pb, (long)&status, 2)) == -4){{}}\n\
  if (h != pb) return 100;\n\
  if ((status & 0xff) == 0x7f) return 4000 + ((status>>8)&0xff);\n\
  return 5000 + ((status>>8)&0xff);\n\
}}\n"
    )
}

// #1198 — the TTOU counterpart to the SIGTTIN stop: a **background write** to the terminal must NOT
// stop when `TOSTOP` is off — and bash leaves it off, so `cmd &` that prints (e.g. `seq 3 &`) runs to
// completion. The personality raised SIGTTOU unconditionally on a background write; before the coop
// engine enforced stops that was invisible, but once #1198 benches a stopped domain an unconditional
// SIGTTOU would freeze every writing `cmd &` after its first line. The write now proceeds silently
// (TOSTOP clear), so the forked background writer exits 7 (`5007`), not stopped (`4022`), on BOTH
// engines. (A background *read* still always SIGTTIN-stops — see the sibling tests above.)
#[test]
fn c_a_background_terminal_write_does_not_stop_without_tostop() {
    let e = run_interp_terminal_setup(&ttou_bg_src(), vec![], |_, _| {});
    assert_eq!(
        e.result,
        vec![Value::I32(5007)],
        "tree-walker: the background write proceeded (TOSTOP off — the default), so the writer ran to \
         its own exit 7 instead of SIGTTOU-stopping"
    );
    let b = run_bytecode_terminal_setup(&ttou_bg_src(), vec![], |_, _| {});
    assert_eq!(
        b.result,
        vec![Value::I32(5007)],
        "coop bytecode (the browser tier): same — a writing `cmd &` is not frozen by an unconditional \
         SIGTTOU; the write goes through and the job completes, matching the oracle"
    );
}
