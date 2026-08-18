//! End-to-end C linking against the **POSIX personality** (`svm-posix`) through §7 named imports:
//! each import name binds to `(HOST_PROC, op)` on the granted personality handle as an
//! instantiation-time **slot binding** ([`svm_posix::resolve`] + `Host::set_import_bindings`) —
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
//! personality's `HostProc` dispatches through the same `cap_dispatch_slots` the JIT's `cap.call` thunk
//! calls, so parity comes for free. Requires a unix C toolchain (`make` + `cc`) to build the chibicc
//! fork, so the suite is gated to `#![cfg(unix)]` (like `c_frontend.rs`).
#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use core::ffi::c_void;
use svm_interp::{run_with_host, Host, Trap, Value};
use svm_jit::{compile_and_run_with_host, JitOutcome};
use svm_posix::Posix;
use svm_run::cap_thunk;
use svm_text::parse_module as parse_module_raw;
use svm_verify::verify_module;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

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
    let base = std::env::temp_dir().join(format!("svm_cposix_{}_{id}", std::process::id()));
    let cfile = base.with_extension("c");
    let irfile = base.with_extension("svm");
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
/// [`svm_posix::resolve`] to `(HOST_PROC, op)` on the granted personality `handle` — the phase-4
/// no-rewrite binding (`Host::set_import_bindings`). A name outside the personality leaves its
/// slot unbound (a dispatch through it is a fail-closed `CapFault`).
fn bind_shim(m: &svm_ir::Module, host: &mut Host, handle: i32) {
    let bindings = m
        .imports
        .iter()
        .map(
            |i| match i.name.strip_prefix("__px_").and_then(svm_posix::resolve) {
                Some(c) => svm_interp::BoundImport::required(c.type_id, c.op, handle),
                None => svm_interp::BoundImport::rebindable(0, 0, None),
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
    let (px, posix) = svm_posix::grant(host, win / 2, win, Vec::new());
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
/// svm-posix **personality** (memfs/stdio via `__px_*` named imports — World A) AND drives the
/// **capability** path (`__vm_pipe`/`__vm_read`/`__vm_write`, the `cap.self`/Stream builtins — World B) in
/// one run. The worlds use **disjoint** host state (a HOST_PROC handle + import bindings vs. the Stream
/// powerbox slots) and **disjoint** guest dispatch (named imports vs. `cap.call` builtins), so they
/// compose on one `Host` with no conflict — `svm_posix::grant` claims nothing the capability grants touch,
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
  __vm_write(fds[1], a, ra);          /* STREAM cap.call write */
  char b[8];
  long rb = __vm_read(fds[0], b, ra); /* STREAM cap.call read -> ra bytes */
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
/// returns `-EINTR` (sentinel 42). Nothing signal-specific lives in svm — the interrupt fires exactly when
/// the personality hands over a delivery. The raiser *retries* (deterministic under the M:N executor): a
/// `kill` before `main` parks is a no-op interrupt; `main` sets `done` once its read takes EINTR.
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
        "a caught, delivered SIGINT interrupted the personality-linked guest's blocked capability read"
    );
}

/// #799 slice 3 — **disposition-gated: an *ignored* signal does NOT interrupt a blocked syscall.** The
/// mirror of the caught case, proving the gate is the personality's policy, not svm's: `main` sets SIGINT
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
        "an ignored SIGINT must not interrupt the blocked read; it completed with data instead"
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
        "the SA_RESTART'd SIGINT ran its handler but the blocked read restarted and returned data"
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
    let stdin_h = ih.grant_stream(svm_interp::StreamRole::In);
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
