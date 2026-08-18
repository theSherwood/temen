//! FORK.md §8.5 slices 4+5 — **a chibicc-compiled C program calling `fork()`**, forking for real under
//! the manager topology. This is the frontend capstone: the substrate (`fork_manager.rs`) and the
//! named-import binding (`fork_import.rs`) are proven with hand-written IR; here the guest is *ordinary
//! C*, compiled by the chibicc fork, spawned as its own module via `instantiate_module_named` (op 13).
//!
//! The guest is a normal separate-module command (the `c_shell_exec.rs` `--child-entry` shape). Its libc
//! shim is two externs — chibicc drops the leading "handle" arg of a cap-style call, so `write(1, &x, 8)`
//! lowers to `Stream.write(&x, 8)` (bound to the granted `"stdout"` stream by the manifest reference
//! policy) and `__fork(0, 0)` lowers to a `(i64)->(i64)` call bound to the live **fork offer** by the
//! named-offer step of `bind_child_manifest` (the `fork_import.rs` change). The C wrapper
//! `long fork(void){ return __fork(0,0); }` is the whole POSIX face.
//!
//! Topology (manager root = 0, server = 1, guest = 2, twin = 3): the manager (hand-written IR) spawns the
//! server (a `svc.wait` loop whose handler runs pid-mode `clone_caller`), mints a `child_offer` over its
//! `fork` export, then spawns the **compiled guest module** via op 13 re-granting the fork offer (as
//! `"__fork"`, the guest's import name) and the shared stdout stream (as `"stdout"`), and joins. The
//! guest's `fork()` returns the twin's pid (3) in the original and 0 in the twin; both copies
//! `write(1, &slot, 8)` their result to the one shared stdout sink. Interp only (the serve substrate is
//! eval-loop-only, as for every fork test). Gated `#![cfg(unix)]` (needs the chibicc toolchain).
#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, OnceLock};

use svm_interp::{run_with_host, Host, StreamRole, Value};
use svm_text::parse_module as parse_module_raw;
use svm_verify::verify_module;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

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

/// Compile `src` to text IR with the §14 spawnable `--child-entry` ABI.
fn c_to_ir(src: &str) -> String {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let id = N.fetch_add(1, Ordering::Relaxed);
    let base = std::env::temp_dir().join(format!("svm_cfork_{}_{id}", std::process::id()));
    let cfile = base.with_extension("c");
    let irfile = base.with_extension("svm");
    std::fs::write(&cfile, src).unwrap();
    let cin = cfile.to_str().unwrap().to_string();
    let cout = irfile.to_str().unwrap().to_string();
    let status = Command::new(chibicc())
        .args([
            "-cc1",
            "--emit-ir",
            "--child-entry",
            "-cc1-input",
            &cin,
            "-cc1-output",
            &cout,
            &cin,
        ])
        .status()
        .expect("run chibicc");
    assert!(status.success(), "chibicc failed on:\n{src}");
    std::fs::read_to_string(&irfile).unwrap()
}

/// The reusable guest-side process libc (`fork`/`wait_pid`/`setpgid`/`execvp`/`getenv`) over the
/// fork/exec/wait surface — the layer a shell links so it writes the idiomatic fork/exec/wait loop
/// instead of hand-marshalling the args buffer. `#[allow(dead_code)]`-style: all entry points are
/// `static`, so chibicc drops the ones a given program doesn't call. Prepended to a guest's C source.
const FORK_SHIM: &str = include_str!("fork_shim.c");

/// The guest: `fork()` (through the `__fork` import → the fork offer), retrying while it returns `< 0`
/// (the `-EAGAIN` serve/park race — the `while ((pid = fork()) < 0)` shell idiom, ISSUES.md I68), then
/// write the 8-byte fork return to fd 1 (the granted stdout stream) — in BOTH the original and the twin.
/// `slot` is a `static` so its address is a plain data pointer (no stack-array codegen). `__fork` takes a
/// leading dummy arg chibicc drops, so the lowered call is `(i64)->(i64)` — matching the fork offer op.
const GUEST_SRC: &str = r#"
long write(long fd, void *buf, long n);
long __fork(int h, long a);
long fork(void) { return __fork(0, 0); }
static long slot;
int main(int argc, char **argv) {
  while ((slot = fork()) < 0);
  write(1, &slot, 8);
  return slot;
}
"#;

/// The manager program: `main(inst, stream, guestmod)` spawns the server, mints the fork offer, builds a
/// 2-entry grant list {`"stdout"` → stream, `"__fork"` → offer} at window offset 256, spawns the guest
/// module via op 13 into a 128 KiB carve at 131072, and joins it. Server = func 1, handler = func 2.
const MANAGER: &str = r#"
memory 19
type 0 func (i64) -> (i64)
type 1 interface { op: 0 }
export 0 interface "fork" 1 { op: 2 }
data 300 "__fork"
data 310 "stdout"
func (i32, i32, i64) -> (i64) {
block 0 (v0: i32, vstream: i32, vgmod: i64) {
  vq = i64.const 0
  ; spawn via record (op 17): entry=1 off=262144 sl=12 quota=0
  q0v0 = i64.const 4294967296
  q0v1 = i64.const 262144
  q0v2 = i64.const -4294967284
  q0v3 = i64.const 4294967295
  q0v4 = i64.const 0
  q0a0 = i64.const 1152
  i64.store q0a0 q0v0
  q0a1 = i64.const 1160
  i64.store q0a1 q0v1
  q0a2 = i64.const 1168
  i64.store q0a2 q0v2
  q0a3 = i64.const 1176
  i64.store q0a3 q0v3
  q0a4 = i64.const 1184
  i64.store q0a4 q0v4
  q0a5 = i64.const 1192
  i64.store q0a5 q0v4
  q0a6 = i64.const 1200
  i64.store q0a6 q0v4
  vs = cap.call 6 17 (i64) -> (i32) v0 (q0a0)
  vz0 = i64.const 0
  vforkoff = cap.call 6 14 (i32, i64) -> (i32) v0 (vs, vz0)
  va0 = i64.const 256
  vnp0 = i32.const 310
  i32.store va0 vnp0
  va1 = i64.const 260
  vsix = i32.const 6
  i32.store va1 vsix
  va2 = i64.const 264
  i32.store va2 vstream
  va3 = i64.const 272
  vnp1 = i32.const 300
  i32.store va3 vnp1
  va4 = i64.const 276
  i32.store va4 vsix
  va5 = i64.const 280
  i32.store va5 vforkoff
  vgp = i64.const 256
  vgn = i64.const 2
  ve0 = i64.const 0
  voffg = i64.const 131072
  vsl = i64.const 17
  vg = cap.call 6 13 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vgmod, vgp, vgn, ve0, voffg, vsl, vq)
  vjg = cap.call 6 1 (i32) -> (i64) v0 (vg)
  return vjg
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  br 1()
  }
block 1 () {
  vz = i32.const 0
  vn = cap.call 4294967295 10 () -> (i64) vz ()
  br 1()
  }
}
func (i64) -> (i64) {
block 0 (vx: i64) {
  vz = i32.const 0
  vzero = i64.const 0
  vt = cap.call 4294967295 11 (i64) -> (i64) vz (vzero)
  return vt
  }
}
"#;

/// FORK.md §8.6 — **the shell's command loop in one compiled-C program: `fork()` → `exec` → `wait()`.**
/// The guest runs the classic `pid = fork(); if (pid == 0) exec(cmd); else wait(pid);`:
///
/// - **fork** through `__fork` (the fork offer), retrying on the `-EAGAIN` serve/park race (I68).
/// - **exec** is **BusyBox-multicall applet dispatch** (STAGE1.md — "a shell's `exec` and BusyBox's
///   applet dispatch are the same shape"): the child transfers to the selected command entry
///   (`applet(1)`) and *becomes* it, exiting with the command's status (`42`). This is the achievable
///   first rung; true cross-module image-replace `execve` is the separate capstone (§8.6 remaining).
/// - **wait** through `__wait` (the wait offer → `reap`), retrying on `-EAGAIN` too — the parent reaps
///   the twin and observes the exec'd command's status, which it writes to stdout.
///
/// The status `42` **originates in the exec'd command**, flows through the child's exit into the
/// scheduler's `results[twin]`, and is reaped by the parent's `wait` — the full fork→exec→wait data
/// path, end to end, in ordinary compiled C. The server serves **two** verbs over two offers: `fork`
/// (export 0, op → func 2 `clone_caller`) and `wait` (export 1, op → func 3 `reap`). Interp only, like
/// every fork test.
const EXEC_GUEST_SRC: &str = r#"
long write(long fd, void *buf, long n);
long __fork(int h, long a);
long __wait(int h, long pid);
long fork(void) { return __fork(0, 0); }
long wait_pid(long pid) { return __wait(0, pid); }
int applet(int which) {
  if (which == 1) return 42;
  return 7;
}
static long pid;
static long status;
int main(int argc, char **argv) {
  while ((pid = fork()) < 0);
  if (pid == 0) {
    return applet(1);
  }
  while ((status = wait_pid(pid)) < 0);
  write(1, &status, 8);
  return status;
}
"#;

/// The manager for the fork→exec→wait guest: same shape as `MANAGER`, but the server serves **two**
/// offers over two exports — `fork` (export 0 → func 2 `clone_caller`) and `wait` (export 1 → func 3
/// `reap`) — and the 3-entry grant list re-grants `{"stdout" → stream, "__fork" → fork offer, "__wait"
/// → wait offer}`. One `svc.wait` loop (func 1) serves both, dispatched by `(export, op)`.
const EXEC_MANAGER: &str = r#"
memory 19
type 0 func (i64) -> (i64)
type 1 interface { op: 0 }
export 0 interface "fork" 1 { op: 2 }
export 1 interface "wait" 1 { op: 3 }
data 300 "__fork"
data 310 "stdout"
data 320 "__wait"
func (i32, i32, i64) -> (i64) {
block 0 (v0: i32, vstream: i32, vgmod: i64) {
  vq = i64.const 0
  ; spawn via record (op 17): entry=1 off=262144 sl=12 quota=0
  q0v0 = i64.const 4294967296
  q0v1 = i64.const 262144
  q0v2 = i64.const -4294967284
  q0v3 = i64.const 4294967295
  q0v4 = i64.const 0
  q0a0 = i64.const 1152
  i64.store q0a0 q0v0
  q0a1 = i64.const 1160
  i64.store q0a1 q0v1
  q0a2 = i64.const 1168
  i64.store q0a2 q0v2
  q0a3 = i64.const 1176
  i64.store q0a3 q0v3
  q0a4 = i64.const 1184
  i64.store q0a4 q0v4
  q0a5 = i64.const 1192
  i64.store q0a5 q0v4
  q0a6 = i64.const 1200
  i64.store q0a6 q0v4
  vs = cap.call 6 17 (i64) -> (i32) v0 (q0a0)
  vz0 = i64.const 0
  vforkoff = cap.call 6 14 (i32, i64) -> (i32) v0 (vs, vz0)
  v1c = i64.const 1
  vwaitoff = cap.call 6 14 (i32, i64) -> (i32) v0 (vs, v1c)
  va0 = i64.const 256
  vnp0 = i32.const 310
  i32.store va0 vnp0
  va1 = i64.const 260
  vsix = i32.const 6
  i32.store va1 vsix
  va2 = i64.const 264
  i32.store va2 vstream
  va3 = i64.const 272
  vnp1 = i32.const 300
  i32.store va3 vnp1
  va4 = i64.const 276
  i32.store va4 vsix
  va5 = i64.const 280
  i32.store va5 vforkoff
  va6 = i64.const 288
  vnp2 = i32.const 320
  i32.store va6 vnp2
  va7 = i64.const 292
  i32.store va7 vsix
  va8 = i64.const 296
  i32.store va8 vwaitoff
  vgp = i64.const 256
  vgn = i64.const 3
  ve0 = i64.const 0
  voffg = i64.const 131072
  vsl = i64.const 17
  vg = cap.call 6 13 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vgmod, vgp, vgn, ve0, voffg, vsl, vq)
  vjg = cap.call 6 1 (i32) -> (i64) v0 (vg)
  return vjg
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  br 1()
  }
block 1 () {
  vz = i32.const 0
  vn = cap.call 4294967295 10 () -> (i64) vz ()
  br 1()
  }
}
func (i64) -> (i64) {
block 0 (vx: i64) {
  vz = i32.const 0
  vzero = i64.const 0
  vt = cap.call 4294967295 11 (i64) -> (i64) vz (vzero)
  return vt
  }
}
func (i64) -> (i64) {
block 0 (vpid: i64) {
  vz = i32.const 0
  vt = cap.call 4294967295 12 (i64) -> (i64) vz (vpid)
  return vt
  }
}
"#;

#[test]
fn a_compiled_c_program_runs_fork_exec_wait_end_to_end() {
    let manager = Arc::new(parse_module_raw(EXEC_MANAGER).expect("parse exec manager"));
    verify_module(&manager).expect("verify exec manager");
    let guest = parse_module_raw(&c_to_ir(EXEC_GUEST_SRC)).expect("parse exec guest");
    verify_module(&guest).expect("verify exec guest");

    let mut host = Host::new();
    host.set_self_module(&manager);
    let _sink = host.shared_stdout();
    let win = 1u64 << 19;
    let stream = host.grant_stream(StreamRole::Out);
    let inst = host.grant_instantiator(0, win);
    let gmod = host.grant_module(&guest);

    let mut fuel = 80_000_000u64;
    let r = run_with_host(
        &manager,
        0,
        &[
            Value::I32(inst),
            Value::I32(stream),
            Value::I64(gmod as i64),
        ],
        &mut fuel,
        &mut host,
    )
    .expect("run");

    // The parent forked, waited on the twin, and returned the reaped status — the exec'd command's 42.
    assert_eq!(
        r,
        vec![Value::I64(42)],
        "the parent's wait(pid) reaped the exec'd applet's exit status (42)"
    );

    // Exactly ONE write reached stdout: the parent's reaped status. The child took the exec branch
    // (applet → exit 42) and never wrote — the status flowed command → child-exit → wait → parent.
    let out = host.stdout_bytes();
    assert_eq!(
        out.len(),
        8,
        "only the parent wrote — the exec'd child exited"
    );
    let status = i64::from_le_bytes(out[..8].try_into().unwrap());
    assert_eq!(
        status, 42,
        "the reaped status is the exec'd command's exit code"
    );
}

/// FORK.md §8.6 — **`waitpid(-1)`: reap *any* child.** The shell's `wait`-for-any-child loop, in
/// compiled C. The guest forks **twice** (two children exiting `3` and `4`), then reaps both with
/// `wait(-1)` — passing `pid == -1` so the servicer reaps whichever twin finishes next, not a named
/// one — accumulating their statuses. `3 + 4 = 7` proves both children were reaped through the
/// any-child path regardless of completion order. Reuses `EXEC_MANAGER` (grants `{stdout, __fork,
/// __wait}`); `__wait(0, -1)` lowers to the wait offer with `pid = -1`, which the `reap` self-op
/// routes to `reap_any_parked_caller`. The `s < 0` retry absorbs the `-EAGAIN` serve/park race.
const WAITANY_GUEST_SRC: &str = r#"
long write(long fd, void *buf, long n);
long __fork(int h, long a);
long __wait(int h, long pid);
long fork(void) { return __fork(0, 0); }
long wait_any(void) { return __wait(0, -1); }
static long pid1, pid2, s, total, n;
int main(int argc, char **argv) {
  while ((pid1 = fork()) < 0);
  if (pid1 == 0) return 3;
  while ((pid2 = fork()) < 0);
  if (pid2 == 0) return 4;
  total = 0;
  n = 0;
  while (n < 2) {
    s = wait_any();
    if (s < 0) continue;
    total = total + s;
    n = n + 1;
  }
  write(1, &total, 8);
  return total;
}
"#;

#[test]
fn a_compiled_c_program_reaps_two_children_with_waitpid_minus_one() {
    let manager = Arc::new(parse_module_raw(EXEC_MANAGER).expect("parse exec manager"));
    verify_module(&manager).expect("verify exec manager");
    let guest = parse_module_raw(&c_to_ir(WAITANY_GUEST_SRC)).expect("parse wait-any guest");
    verify_module(&guest).expect("verify wait-any guest");

    let mut host = Host::new();
    host.set_self_module(&manager);
    let _sink = host.shared_stdout();
    let win = 1u64 << 19;
    let stream = host.grant_stream(StreamRole::Out);
    let inst = host.grant_instantiator(0, win);
    let gmod = host.grant_module(&guest);

    let mut fuel = 120_000_000u64;
    let r = run_with_host(
        &manager,
        0,
        &[
            Value::I32(inst),
            Value::I32(stream),
            Value::I64(gmod as i64),
        ],
        &mut fuel,
        &mut host,
    )
    .expect("run");

    // Both children (exit 3 and 4) were reaped via wait(-1); the parent summed their statuses.
    assert_eq!(
        r,
        vec![Value::I64(7)],
        "the parent reaped both children through wait(-1): 3 + 4 = 7"
    );
    // Exactly one write — the parent's summed total. Neither child wrote (both returned before write).
    let out = host.stdout_bytes();
    assert_eq!(out.len(), 8, "only the parent wrote its summed total");
    let total = i64::from_le_bytes(out[..8].try_into().unwrap());
    assert_eq!(total, 7, "3 + 4, both reaped via the any-child wait");
}

/// FORK.md §8.6 — `waitpid(-1)` with **no children** is `-ECHILD`, not a hang. A shell's wait loop
/// must terminate when every child has been reaped; the any-child reap fails closed on an empty
/// `forked_twins` set *before* claiming the caller, so this is deterministic even under the serve
/// race (there is no twin to wait for). The guest never forks — its single `wait(-1)` sees `-10`
/// (`ECHILD`) and branches on it (returning `55`, so the check survives `int main`'s truncation of
/// a negative `long`).
const WAITANY_NOCHILD_GUEST_SRC: &str = r#"
long __wait(int h, long pid);
long wait_any(void) { return __wait(0, -1); }
static long s;
int main(int argc, char **argv) {
  s = wait_any();
  if (s == -10) return 55;
  return 66;
}
"#;

#[test]
fn waitpid_minus_one_with_no_children_is_echild() {
    let manager = Arc::new(parse_module_raw(EXEC_MANAGER).expect("parse exec manager"));
    verify_module(&manager).expect("verify exec manager");
    let guest =
        parse_module_raw(&c_to_ir(WAITANY_NOCHILD_GUEST_SRC)).expect("parse no-child guest");
    verify_module(&guest).expect("verify no-child guest");

    let mut host = Host::new();
    host.set_self_module(&manager);
    let _sink = host.shared_stdout();
    let win = 1u64 << 19;
    let stream = host.grant_stream(StreamRole::Out);
    let inst = host.grant_instantiator(0, win);
    let gmod = host.grant_module(&guest);

    let mut fuel = 80_000_000u64;
    let r = run_with_host(
        &manager,
        0,
        &[
            Value::I32(inst),
            Value::I32(stream),
            Value::I64(gmod as i64),
        ],
        &mut fuel,
        &mut host,
    )
    .expect("run");

    // wait(-1) with no children returned -ECHILD at once (the guest saw it and took the `55` branch)
    // — the shell's wait loop terminates rather than hanging.
    assert_eq!(
        r,
        vec![Value::I64(55)],
        "wait(-1) with no children is -ECHILD, so the guest took the `s == -10` branch"
    );
}

/// FORK.md §8.6 — **per-parent child scoping**: `wait` only reaps a domain's *own* children. The
/// twin table (`forked_twins`) records each twin's forking parent, so a `wait(-1)` sees only the
/// twins that parent forked — even though the fork/wait offer (and thus the servicer) is shared.
///
/// The guest forks a child; the **child** then calls `wait(-1)` with no children of its own. The
/// global twin table is *not* empty (it holds the child itself, under the parent's key), so without
/// parent scoping the child's `wait(-1)` would block forever (nothing of its to reap) and deadlock
/// the whole run. With scoping the child gets `-ECHILD` at once and returns `77`; the parent then
/// reaps the child through *its* `wait(-1)` and returns that `77`. So `r == 77` proves both halves:
/// the child was correctly told it has no children, and the parent reaped only its own. Both the
/// child and parent retry on `-EAGAIN` (`-11`, the serve/park race); the child stops on `-ECHILD`.
const WAIT_SCOPE_GUEST_SRC: &str = r#"
long __fork(int h, long a);
long __wait(int h, long pid);
long fork(void) { return __fork(0, 0); }
long wait_any(void) { return __wait(0, -1); }
static long pid, s, cs;
int main(int argc, char **argv) {
  while ((pid = fork()) < 0);
  if (pid == 0) {
    do { cs = wait_any(); } while (cs == -11);
    if (cs == -10) return 77;
    return 66;
  }
  while ((s = wait_any()) < 0);
  return s;
}
"#;

#[test]
fn wait_only_reaps_a_domains_own_children() {
    let manager = Arc::new(parse_module_raw(EXEC_MANAGER).expect("parse exec manager"));
    verify_module(&manager).expect("verify exec manager");
    let guest = parse_module_raw(&c_to_ir(WAIT_SCOPE_GUEST_SRC)).expect("parse scope guest");
    verify_module(&guest).expect("verify scope guest");

    let mut host = Host::new();
    host.set_self_module(&manager);
    let _sink = host.shared_stdout();
    let win = 1u64 << 19;
    let stream = host.grant_stream(StreamRole::Out);
    let inst = host.grant_instantiator(0, win);
    let gmod = host.grant_module(&guest);

    let mut fuel = 120_000_000u64;
    let r = run_with_host(
        &manager,
        0,
        &[
            Value::I32(inst),
            Value::I32(stream),
            Value::I64(gmod as i64),
        ],
        &mut fuel,
        &mut host,
    )
    .expect("run");

    // The child's wait(-1) got -ECHILD (it has no children of its own → returned 77), and the parent
    // reaped only its own child (returning that 77). A cross-reap or a hang would not yield 77.
    assert_eq!(
        r,
        vec![Value::I64(77)],
        "the child saw -ECHILD (no children of its own) and the parent reaped only its own child"
    );
}

/// FORK.md §8.6 — **process groups**: `setpgid` + `waitpid(-pgid)`, the job-control primitive a shell
/// uses to reap a whole pipeline as one group. Each forked child starts as its own group leader
/// (`pgid == pid`); `__vm_setpgid(pid, pgid)` (the self-op the parent drives directly) reassigns it.
///
/// The guest forks two children (A exits `10`, B exits `20`), then `setpgid`s B **into A's group**
/// (`pgid = a_pid`). It then reaps that group twice with `wait(-a_pid)` (`__wait(0, -a_pid)` →
/// `waitpid(-pgid)`), retrying only on `-EAGAIN` (`-11`) so a stray `-ECHILD` would corrupt the sum
/// rather than spin. Both A and B are now in group `a_pid`, so the two reaps sum to `30`. Had
/// `setpgid` not moved B, the second `wait(-a_pid)` would return `-ECHILD` (B still in its own group)
/// and the sum would be `10 + (-10) = 0` — so `r == 30` proves B really joined A's group.
const PGID_GUEST_SRC: &str = r#"
long __fork(int h, long a);
long __wait(int h, long pid);
long __vm_setpgid(long pid, long pgid);
long fork(void) { return __fork(0, 0); }
long wait_group(long pgid) { return __wait(0, -pgid); }
static long a_pid, b_pid, s, total, n;
int main(int argc, char **argv) {
  while ((a_pid = fork()) < 0);
  if (a_pid == 0) return 10;
  while ((b_pid = fork()) < 0);
  if (b_pid == 0) return 20;
  __vm_setpgid(b_pid, a_pid);
  total = 0;
  n = 0;
  while (n < 2) {
    do { s = wait_group(a_pid); } while (s == -11);
    total = total + s;
    n = n + 1;
  }
  return total;
}
"#;

#[test]
fn setpgid_groups_children_and_waitpid_reaps_the_group() {
    let manager = Arc::new(parse_module_raw(EXEC_MANAGER).expect("parse exec manager"));
    verify_module(&manager).expect("verify exec manager");
    let guest = parse_module_raw(&c_to_ir(PGID_GUEST_SRC)).expect("parse pgid guest");
    verify_module(&guest).expect("verify pgid guest");

    let mut host = Host::new();
    host.set_self_module(&manager);
    let _sink = host.shared_stdout();
    let win = 1u64 << 19;
    let stream = host.grant_stream(StreamRole::Out);
    let inst = host.grant_instantiator(0, win);
    let gmod = host.grant_module(&guest);

    let mut fuel = 160_000_000u64;
    let r = run_with_host(
        &manager,
        0,
        &[
            Value::I32(inst),
            Value::I32(stream),
            Value::I64(gmod as i64),
        ],
        &mut fuel,
        &mut host,
    )
    .expect("run");

    // Both children were reaped through wait(-a_pid): B joined A's group via setpgid, so the group
    // held both and the two reaps summed to 30 (10 + 20). A failed setpgid would have summed to 0.
    assert_eq!(
        r,
        vec![Value::I64(30)],
        "setpgid moved B into A's group; waitpid(-a_pid) reaped both (10 + 20 = 30)"
    );
}

/// FORK.md §8.6 — **`waitpid(-pgid)` is group-scoped**: it never reaps a child in a *different* group.
/// The dual of the grouping test. The guest forks A (`10`) and B (`20`) and leaves them in their
/// **default** groups (each its own leader: `pgid == pid`). `wait(-a_pid)` reaps only A; a second
/// `wait(-a_pid)` then returns `-ECHILD` (`-10`) because group `a_pid` is now empty — B is in group
/// `b_pid`, *not* reaped by A's group. B is finally reaped by `wait(-b_pid)`. The guest checks the
/// exact triple (`10`, `-10`, `20`) and returns `99` only if all hold — so a group over-reap (B
/// wrongly reaped by `wait(-a_pid)`) would not yield `99`.
const PGID_SCOPE_GUEST_SRC: &str = r#"
long __fork(int h, long a);
long __wait(int h, long pid);
long fork(void) { return __fork(0, 0); }
long wait_group(long pgid) { return __wait(0, -pgid); }
static long a_pid, b_pid, s1, s2, s3;
int main(int argc, char **argv) {
  while ((a_pid = fork()) < 0);
  if (a_pid == 0) return 10;
  while ((b_pid = fork()) < 0);
  if (b_pid == 0) return 20;
  do { s1 = wait_group(a_pid); } while (s1 == -11);
  s2 = wait_group(a_pid);
  while (s2 == -11) s2 = wait_group(a_pid);
  do { s3 = wait_group(b_pid); } while (s3 == -11);
  if (s1 == 10 && s2 == -10 && s3 == 20) return 99;
  return 1;
}
"#;

#[test]
fn waitpid_by_group_does_not_reap_other_groups() {
    let manager = Arc::new(parse_module_raw(EXEC_MANAGER).expect("parse exec manager"));
    verify_module(&manager).expect("verify exec manager");
    let guest = parse_module_raw(&c_to_ir(PGID_SCOPE_GUEST_SRC)).expect("parse pgid-scope guest");
    verify_module(&guest).expect("verify pgid-scope guest");

    let mut host = Host::new();
    host.set_self_module(&manager);
    let _sink = host.shared_stdout();
    let win = 1u64 << 19;
    let stream = host.grant_stream(StreamRole::Out);
    let inst = host.grant_instantiator(0, win);
    let gmod = host.grant_module(&guest);

    let mut fuel = 160_000_000u64;
    let r = run_with_host(
        &manager,
        0,
        &[
            Value::I32(inst),
            Value::I32(stream),
            Value::I64(gmod as i64),
        ],
        &mut fuel,
        &mut host,
    )
    .expect("run");

    // wait(-a_pid) reaped only A (10); group a_pid was then empty (-ECHILD, B is in group b_pid);
    // B was reaped by wait(-b_pid) (20). The guest saw the exact triple → 99. A cross-group reap fails.
    assert_eq!(
        r,
        vec![Value::I64(99)],
        "waitpid(-a_pid) reaped only A's group, never B (which was in b_pid's group)"
    );
}

/// FORK.md §8.6 — **`execve` delivers argv to the command.** A shell running `echo hello` must hand
/// the command its argument vector. The §3e powerbox args buffer lives at
/// `[POWERBOX_ARGS_BASE=128, POWERBOX_ARGS_END=16384)` as `{argc:u32, envc:u32}` + packed
/// NUL-terminated strings, and a `--child-entry` `main(argc, argv)` `_start` parses it. `execve`
/// replaces the image in the caller's *own* window in place, and a `main(argc,argv)` command's globals
/// start at `POWERBOX_ARGS_END`, so a forked child seeds the buffer *before* `execve` and the new image
/// reads it. Proven end to end: the child seeds `argv = {"cmd", "42"}` then `execve`s a command that
/// returns `argv[1][0]` (`'4'` = 52) — pointer-array delivery, not a flat read (the image-replace
/// preserves the args window; the command's data lands above it). This is the primitive the microshell
/// below builds on; env (`envc > 0`) is not yet parsed by the `_start` (the next frontier).
const EXECVE_ARGV_GUEST_SRC: &str = r#"
long __fork(int h, long a);
long __wait(int h, long pid);
long __vm_resolve(const char *name, long len);
long __vm_exec_module(long mod, long grants, long n, long entry, long sl);
long fork(void) { return __fork(0, 0); }
long wait_pid(long pid) { return __wait(0, pid); }
struct grant { int name_off; int name_len; int handle; int pad; };
static struct grant grec;
static char stdout_name[] = "stdout";
static char cmd_name[] = "cmd";
static long pid, status;
int main(int argc, char **argv) {
  while ((pid = fork()) < 0);
  if (pid == 0) {
    int *hdr = (int *)128;      /* POWERBOX_ARGS_BASE */
    hdr[0] = 2;                 /* argc */
    hdr[1] = 0;                 /* envc */
    char *s = (char *)136;      /* packed strings: base + 8 */
    s[0] = 'c'; s[1] = 'm'; s[2] = 'd'; s[3] = 0;
    s[4] = '4'; s[5] = '2'; s[6] = 0;
    long cmd = __vm_resolve(cmd_name, 3);
    long soh = __vm_resolve(stdout_name, 6);
    grec.name_off = (int)(long)stdout_name;
    grec.name_len = 6;
    grec.handle = (int)soh;
    __vm_exec_module(cmd, (long)&grec, 1, 0, 17);
    return -1;
  }
  while ((status = wait_pid(pid)) < 0);
  return status;
}
"#;

/// The command: return the first byte of `argv[1]` — `'4'` (52) if argv was delivered.
const EXECVE_ARGV_CMD_SRC: &str = r#"
int main(int argc, char **argv) {
  return argv[1][0];
}
"#;

#[test]
fn execve_delivers_argv_to_the_command() {
    let manager = Arc::new(parse_module_raw(EXECVE_MANAGER).expect("parse execve manager"));
    verify_module(&manager).expect("verify execve manager");
    let guest = parse_module_raw(&c_to_ir(EXECVE_ARGV_GUEST_SRC)).expect("parse argv guest");
    verify_module(&guest).expect("verify argv guest");
    let cmd = parse_module_raw(&c_to_ir(EXECVE_ARGV_CMD_SRC)).expect("parse argv command");
    verify_module(&cmd).expect("verify argv command");

    let mut host = Host::new();
    host.set_self_module(&manager);
    let _sink = host.shared_stdout();
    let win = 1u64 << 19;
    let stream = host.grant_stream(StreamRole::Out);
    let inst = host.grant_instantiator(0, win);
    let gmod = host.grant_module(&guest);
    let cmod = host.grant_module(&cmd);

    let mut fuel = 120_000_000u64;
    let r = run_with_host(
        &manager,
        0,
        &[
            Value::I32(inst),
            Value::I32(stream),
            Value::I64(gmod as i64),
            Value::I64(cmod as i64),
        ],
        &mut fuel,
        &mut host,
    )
    .expect("run");

    assert_eq!(
        r,
        vec![Value::I64(52)],
        "the exec'd command read argv[1][0] = '4' (52) — argv was delivered through execve"
    );
}

/// MICROSHELL (pointing a real shell at the fork/exec/wait surface) — a shell's whole command loop in
/// ordinary compiled C: a reusable `run(name, arg)` that **forks**, has the child build an `argv` from
/// runtime strings, **resolves the command module by name** (dynamic dispatch — the shell's PATH is the
/// name→module grant map), **execve**s it, and the parent **waitpid**s for the status. `main` runs two
/// *different* named commands in sequence and sums their statuses. This is the real shape — not a single
/// hardcoded exec — exercising dynamic command lookup, runtime argv marshalling from inside a fork
/// twin's own window, and repeated fork→exec→wait, all end to end.
const MICROSHELL_GUEST_SRC: &str = r#"
long __fork(int h, long a);
long __wait(int h, long pid);
long __vm_resolve(const char *name, long len);
long __vm_exec_module(long mod, long grants, long n, long entry, long sl);
long fork(void) { return __fork(0, 0); }
long wait_pid(long pid) { return __wait(0, pid); }
struct grant { int name_off; int name_len; int handle; int pad; };
static struct grant grec;
static char stdout_name[] = "stdout";
static long slen(const char *s) { long n = 0; while (s[n]) n = n + 1; return n; }
static long run(const char *name, const char *arg) {
  long pid, st, i, k;
  while ((pid = fork()) < 0);
  if (pid == 0) {
    int *hdr = (int *)128;
    hdr[0] = 2; hdr[1] = 0;                 /* argc = 2, envc = 0 */
    char *s = (char *)136;
    i = 0;
    for (k = 0; name[k]; k = k + 1) { s[i] = name[k]; i = i + 1; } s[i] = 0; i = i + 1;
    for (k = 0; arg[k]; k = k + 1) { s[i] = arg[k]; i = i + 1; } s[i] = 0;
    long cmd = __vm_resolve(name, slen(name));
    long soh = __vm_resolve(stdout_name, 6);
    grec.name_off = (int)(long)stdout_name;
    grec.name_len = 6;
    grec.handle = (int)soh;
    __vm_exec_module(cmd, (long)&grec, 1, 0, 17);
    return -1;
  }
  while ((st = wait_pid(pid)) < 0);
  return st;
}
static char n_one[] = "one";
static char n_two[] = "two";
static char a_five[] = "5";
int main(int argc, char **argv) {
  long r1 = run(n_one, a_five);   /* command "one": returns argv[1][0] = '5' = 53 */
  long r2 = run(n_two, a_five);   /* command "two": returns argv[1][0] + 1 = 54 */
  return r1 + r2;                 /* 53 + 54 = 107 */
}
"#;

/// The two commands the microshell dispatches — each reads its `argv[1]` (proving per-command argv
/// delivery) and returns a distinct function of it, so the summed status pins that *both* ran with the
/// right arguments through *different* name lookups.
const MICROSHELL_CMD_ONE_SRC: &str = r#"
int main(int argc, char **argv) { return argv[1][0]; }
"#;
const MICROSHELL_CMD_TWO_SRC: &str = r#"
int main(int argc, char **argv) { return argv[1][0] + 1; }
"#;

/// The manager: spawns the fork/wait server, then the microshell guest via op 13 with a 5-entry grant
/// list `{stdout, __fork, __wait, "one"→mod1, "two"→mod2}` (the two commands are the shell's PATH), and
/// joins. `main(inst, stream, guestmod, mod1, mod2)`.
const MICROSHELL_MANAGER: &str = r#"
memory 19
type 0 func (i64) -> (i64)
type 1 interface { op: 0 }
export 0 interface "fork" 1 { op: 2 }
export 1 interface "wait" 1 { op: 3 }
data 400 "__fork"
data 410 "stdout"
data 420 "__wait"
data 430 "one"
data 440 "two"
func (i32, i32, i64, i64, i64) -> (i64) {
block 0 (v0: i32, vstream: i32, vgmod: i64, vmod1: i64, vmod2: i64) {
  vq = i64.const 0
  q0v0 = i64.const 4294967296
  q0v1 = i64.const 262144
  q0v2 = i64.const -4294967284
  q0v3 = i64.const 4294967295
  q0v4 = i64.const 0
  q0a0 = i64.const 1152
  i64.store q0a0 q0v0
  q0a1 = i64.const 1160
  i64.store q0a1 q0v1
  q0a2 = i64.const 1168
  i64.store q0a2 q0v2
  q0a3 = i64.const 1176
  i64.store q0a3 q0v3
  q0a4 = i64.const 1184
  i64.store q0a4 q0v4
  q0a5 = i64.const 1192
  i64.store q0a5 q0v4
  q0a6 = i64.const 1200
  i64.store q0a6 q0v4
  vs = cap.call 6 17 (i64) -> (i32) v0 (q0a0)
  vz0 = i64.const 0
  vforkoff = cap.call 6 14 (i32, i64) -> (i32) v0 (vs, vz0)
  v1c = i64.const 1
  vwaitoff = cap.call 6 14 (i32, i64) -> (i32) v0 (vs, v1c)
  vmod1_32 = i32.wrap_i64 vmod1
  vmod2_32 = i32.wrap_i64 vmod2
  va0 = i64.const 256
  vnp0 = i32.const 410
  i32.store va0 vnp0
  va1 = i64.const 260
  vsix = i32.const 6
  i32.store va1 vsix
  va2 = i64.const 264
  i32.store va2 vstream
  va3 = i64.const 272
  vnp1 = i32.const 400
  i32.store va3 vnp1
  va4 = i64.const 276
  i32.store va4 vsix
  va5 = i64.const 280
  i32.store va5 vforkoff
  va6 = i64.const 288
  vnp2 = i32.const 420
  i32.store va6 vnp2
  va7 = i64.const 292
  i32.store va7 vsix
  va8 = i64.const 296
  i32.store va8 vwaitoff
  va9 = i64.const 304
  vnp3 = i32.const 430
  i32.store va9 vnp3
  va10 = i64.const 308
  vthree = i32.const 3
  i32.store va10 vthree
  va11 = i64.const 312
  i32.store va11 vmod1_32
  va12 = i64.const 320
  vnp4 = i32.const 440
  i32.store va12 vnp4
  va13 = i64.const 324
  i32.store va13 vthree
  va14 = i64.const 328
  i32.store va14 vmod2_32
  vgp = i64.const 256
  vgn = i64.const 5
  ve0 = i64.const 0
  voffg = i64.const 131072
  vsl = i64.const 17
  vg = cap.call 6 13 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vgmod, vgp, vgn, ve0, voffg, vsl, vq)
  vjg = cap.call 6 1 (i32) -> (i64) v0 (vg)
  return vjg
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  br 1()
  }
block 1 () {
  vz = i32.const 0
  vn = cap.call 4294967295 10 () -> (i64) vz ()
  br 1()
  }
}
func (i64) -> (i64) {
block 0 (vx: i64) {
  vz = i32.const 0
  vzero = i64.const 0
  vt = cap.call 4294967295 11 (i64) -> (i64) vz (vzero)
  return vt
  }
}
func (i64) -> (i64) {
block 0 (vpid: i64) {
  vz = i32.const 0
  vt = cap.call 4294967295 12 (i64) -> (i64) vz (vpid)
  return vt
  }
}
"#;

#[test]
fn a_microshell_dispatches_two_named_commands_through_fork_exec_wait() {
    let manager = Arc::new(parse_module_raw(MICROSHELL_MANAGER).expect("parse microshell manager"));
    verify_module(&manager).expect("verify microshell manager");
    let guest = parse_module_raw(&c_to_ir(MICROSHELL_GUEST_SRC)).expect("parse microshell guest");
    verify_module(&guest).expect("verify microshell guest");
    let one = parse_module_raw(&c_to_ir(MICROSHELL_CMD_ONE_SRC)).expect("parse cmd one");
    verify_module(&one).expect("verify cmd one");
    let two = parse_module_raw(&c_to_ir(MICROSHELL_CMD_TWO_SRC)).expect("parse cmd two");
    verify_module(&two).expect("verify cmd two");

    let mut host = Host::new();
    host.set_self_module(&manager);
    let _sink = host.shared_stdout();
    let win = 1u64 << 19;
    let stream = host.grant_stream(StreamRole::Out);
    let inst = host.grant_instantiator(0, win);
    let gmod = host.grant_module(&guest);
    let mod1 = host.grant_module(&one);
    let mod2 = host.grant_module(&two);

    let mut fuel = 200_000_000u64;
    let r = run_with_host(
        &manager,
        0,
        &[
            Value::I32(inst),
            Value::I32(stream),
            Value::I64(gmod as i64),
            Value::I64(mod1 as i64),
            Value::I64(mod2 as i64),
        ],
        &mut fuel,
        &mut host,
    )
    .expect("run");

    // Both commands ran through the shell's fork→resolve-by-name→execve→wait loop with their own argv:
    // "one" returned '5' (53), "two" returned '6' (54); the shell summed them → 107.
    assert_eq!(
        r,
        vec![Value::I64(107)],
        "the microshell dispatched two named commands via fork/exec/wait (53 + 54 = 107)"
    );
}

/// FORK.md §8.6 — **`execve` delivers the environment.** The gap the microshell surfaced: `envc` was
/// always 0 and chibicc's `_start` never parsed the env portion of the args buffer, so a command could
/// not read an exported variable. Now a `main(argc, argv, envp)` entry (3 params) makes `_start` also
/// parse the `envc` env strings — which follow the argv strings in the §3e buffer — into an `envp[]`
/// pointer array, passed as `main`'s third argument. Here a forked child seeds `argc=1, envc=1` with
/// `argv={"c"}` and `env={"V=7"}`, then `execve`s a command that returns `envp[0][2]` (`'7'` = 55) —
/// proving the env vector reaches the exec'd image, indexed through the pointer array.
const EXECVE_ENV_GUEST_SRC: &str = r#"
long __fork(int h, long a);
long __wait(int h, long pid);
long __vm_resolve(const char *name, long len);
long __vm_exec_module(long mod, long grants, long n, long entry, long sl);
long fork(void) { return __fork(0, 0); }
long wait_pid(long pid) { return __wait(0, pid); }
struct grant { int name_off; int name_len; int handle; int pad; };
static struct grant grec;
static char stdout_name[] = "stdout";
static char cmd_name[] = "cmd";
static long pid, status;
int main(int argc, char **argv) {
  while ((pid = fork()) < 0);
  if (pid == 0) {
    int *hdr = (int *)128;
    hdr[0] = 1;                 /* argc = 1 */
    hdr[1] = 1;                 /* envc = 1 */
    char *s = (char *)136;
    s[0] = 'c'; s[1] = 0;                        /* argv[0] = "c" */
    s[2] = 'V'; s[3] = '='; s[4] = '7'; s[5] = 0; /* env[0]  = "V=7" */
    long cmd = __vm_resolve(cmd_name, 3);
    long soh = __vm_resolve(stdout_name, 6);
    grec.name_off = (int)(long)stdout_name;
    grec.name_len = 6;
    grec.handle = (int)soh;
    __vm_exec_module(cmd, (long)&grec, 1, 0, 17);
    return -1;
  }
  while ((status = wait_pid(pid)) < 0);
  return status;
}
"#;

/// The command: a 3-arg `main` that returns the third byte of its first env string — `'7'` (55) for
/// `env[0] = "V=7"`, if the environment was delivered through `execve`.
const EXECVE_ENV_CMD_SRC: &str = r#"
int main(int argc, char **argv, char **envp) {
  return envp[0][2];
}
"#;

#[test]
fn execve_delivers_the_environment_to_the_command() {
    let manager = Arc::new(parse_module_raw(EXECVE_MANAGER).expect("parse execve manager"));
    verify_module(&manager).expect("verify execve manager");
    let guest = parse_module_raw(&c_to_ir(EXECVE_ENV_GUEST_SRC)).expect("parse env guest");
    verify_module(&guest).expect("verify env guest");
    let cmd = parse_module_raw(&c_to_ir(EXECVE_ENV_CMD_SRC)).expect("parse env command");
    verify_module(&cmd).expect("verify env command");

    let mut host = Host::new();
    host.set_self_module(&manager);
    let _sink = host.shared_stdout();
    let win = 1u64 << 19;
    let stream = host.grant_stream(StreamRole::Out);
    let inst = host.grant_instantiator(0, win);
    let gmod = host.grant_module(&guest);
    let cmod = host.grant_module(&cmd);

    let mut fuel = 120_000_000u64;
    let r = run_with_host(
        &manager,
        0,
        &[
            Value::I32(inst),
            Value::I32(stream),
            Value::I64(gmod as i64),
            Value::I64(cmod as i64),
        ],
        &mut fuel,
        &mut host,
    )
    .expect("run");

    assert_eq!(
        r,
        vec![Value::I64(55)],
        "the exec'd command read envp[0][2] = '7' (55) — the environment was delivered through execve"
    );
}

/// FORK.md §8.6 — **the process libc shim in action.** The microshell above hand-marshalled the args
/// buffer and drove the raw self-ops; a real shell links `fork_shim.c` and writes the idiomatic loop:
/// `pid = fork(); if (pid == 0) execvp(cmd, argv); else wait_pid(pid);` with a NUL-terminated `argv[]`.
/// The shell points `environ` at its own env, `execvp` marshals both argv and env into the §3e buffer,
/// and the command — also linking the shim — sets `environ = envp` and reads a var back with `getenv`.
/// The shim's `static` entry points mean the command drags in only `getenv`/`strlen`, **not**
/// `fork`/`execvp` and their `__fork`/`__wait` offer imports (which it was never granted). The command
/// returns `argv[1][0] + getenv("V")[0]` = `'4'`(52) + `'7'`(55) = 107.
const SHIM_SHELL_SRC: &str = r#"
static char *sh_env[] = { "V=7", 0 };
static char *cmd_argv[] = { "cmd", "42", 0 };
int main(int argc, char **argv) {
  environ = sh_env;
  long pid = fork();
  while (pid < 0) pid = fork();
  if (pid == 0) { execvp("cmd", cmd_argv); return 127; }
  return wait_pid(pid);
}
"#;

/// The command, also over the shim: wire `environ` from `envp`, then return a function of both its
/// argument (`argv[1]`) and its environment (`getenv("V")`) — so a correct result proves *both* the
/// argv and the env crossed `execvp`, and that `getenv` reads the delivered env.
const SHIM_CMD_SRC: &str = r#"
int main(int argc, char **argv, char **envp) {
  environ = envp;
  char *v = getenv("V");
  return argv[1][0] + (v ? v[0] : 0);
}
"#;

#[test]
fn a_shell_linking_the_process_libc_runs_execvp_with_argv_and_env() {
    let manager = Arc::new(parse_module_raw(EXECVE_MANAGER).expect("parse execve manager"));
    verify_module(&manager).expect("verify execve manager");
    let shell_src = format!("{FORK_SHIM}\n{SHIM_SHELL_SRC}");
    let cmd_src = format!("{FORK_SHIM}\n{SHIM_CMD_SRC}");
    let guest = parse_module_raw(&c_to_ir(&shell_src)).expect("parse shim shell");
    verify_module(&guest).expect("verify shim shell");
    let cmd = parse_module_raw(&c_to_ir(&cmd_src)).expect("parse shim command");
    verify_module(&cmd).expect("verify shim command");

    let mut host = Host::new();
    host.set_self_module(&manager);
    let _sink = host.shared_stdout();
    let win = 1u64 << 19;
    let stream = host.grant_stream(StreamRole::Out);
    let inst = host.grant_instantiator(0, win);
    let gmod = host.grant_module(&guest);
    let cmod = host.grant_module(&cmd);

    let mut fuel = 160_000_000u64;
    let r = run_with_host(
        &manager,
        0,
        &[
            Value::I32(inst),
            Value::I32(stream),
            Value::I64(gmod as i64),
            Value::I64(cmod as i64),
        ],
        &mut fuel,
        &mut host,
    )
    .expect("run");

    // The shell forked, execvp'd "cmd" with argv={"cmd","42"} and env={"V=7"}, and waited: the command
    // read argv[1][0]='4' (52) + getenv("V")[0]='7' (55) = 107 through the libc shim.
    assert_eq!(
        r,
        vec![Value::I64(107)],
        "the shim shell ran execvp with argv + env; the command read both (52 + 55 = 107)"
    );
}

/// FORK.md §8.6 — **a pipe between two forked commands: `one | two`.** The shell mints a pipe with the
/// `pipe()` self-op (via the shim), forks the producer with the pipe **write** end wired to its
/// `stdout`, waits, then forks the consumer with the pipe **read** end wired to its `stdin` and the
/// shell's own `stdout` as its output. The two commands use plain `write(1)` / `read(0)` — they never
/// know a pipe is there; the shim's `exec_io` re-grants the pipe ends by name, and the FIFO backing
/// aliases across each `execve`, so the producer's bytes reach the consumer transparently. Sequential
/// (fill-then-drain): the producer completes before the consumer runs. After reaping it the shell
/// **closes its own write end** (`close(fds[1])`) so the pipe has no writers left — the blocking read
/// then EOFs once the buffered bytes drain. `one` writes `"hi\n"`; `two` uppercases stdin to stdout →
/// the sink holds `"HI\n"` and `two` returns the 3 bytes it moved.
const PIPE_SHELL_SRC: &str = r#"
static char *a_one[] = { "one", 0 };
static char *a_two[] = { "two", 0 };
int main(int argc, char **argv) {
  int shell_out = (int)__vm_resolve("stdout", 6);
  int fds[2];
  pipe(fds);                       /* fds[0] = read end, fds[1] = write end */
  long p1 = fork();
  while (p1 < 0) p1 = fork();
  if (p1 == 0) { exec_io("one", a_one, fds[1], 0); return 127; }
  wait_pid(p1);                    /* producer fills the pipe, then exits */
  close(fds[1]);                   /* no writers left → the consumer's read EOFs after draining */
  long p2 = fork();
  while (p2 < 0) p2 = fork();
  if (p2 == 0) { exec_io("two", a_two, shell_out, fds[0]); return 127; }
  return wait_pid(p2);             /* consumer drains the pipe → shell stdout */
}
"#;

/// The producer `one`: write `"hi\n"` to its stdout (the pipe write end). Plain `write(1, …)`.
const PIPE_PRODUCER_SRC: &str = r#"
long write(long fd, void *buf, long n);
static char msg[] = "hi\n";
int main(int argc, char **argv) {
  write(1, msg, 3);
  return 0;
}
"#;

/// The consumer `two`: read stdin (the pipe read end) to EOF, uppercase it, write to stdout (the
/// shell's sink). Returns the byte count. Plain `read(0, …)` / `write(1, …)`.
const PIPE_CONSUMER_SRC: &str = r#"
long read(long fd, void *buf, long n);
long write(long fd, void *buf, long n);
int main(int argc, char **argv) {
  char buf[64];
  long total = 0, n, k;
  while ((n = read(0, buf, 64)) > 0) {
    for (k = 0; k < n; k = k + 1) {
      char c = buf[k];
      if (c >= 'a' && c <= 'z') c = c - 32;
      buf[k] = c;
    }
    write(1, buf, n);
    total = total + n;
  }
  return total;
}
"#;

#[test]
fn a_shell_pipes_the_output_of_one_forked_command_into_another() {
    let manager = Arc::new(parse_module_raw(MICROSHELL_MANAGER).expect("parse microshell manager"));
    verify_module(&manager).expect("verify microshell manager");
    let shell_src = format!("{FORK_SHIM}\n{PIPE_SHELL_SRC}");
    let guest = parse_module_raw(&c_to_ir(&shell_src)).expect("parse pipe shell");
    verify_module(&guest).expect("verify pipe shell");
    let producer = parse_module_raw(&c_to_ir(PIPE_PRODUCER_SRC)).expect("parse producer");
    verify_module(&producer).expect("verify producer");
    let consumer = parse_module_raw(&c_to_ir(PIPE_CONSUMER_SRC)).expect("parse consumer");
    verify_module(&consumer).expect("verify consumer");

    let mut host = Host::new();
    host.set_self_module(&manager);
    let _sink = host.shared_stdout();
    let win = 1u64 << 19;
    let stream = host.grant_stream(StreamRole::Out);
    let inst = host.grant_instantiator(0, win);
    let gmod = host.grant_module(&guest);
    let mod1 = host.grant_module(&producer); // granted as "one"
    let mod2 = host.grant_module(&consumer); // granted as "two"

    let mut fuel = 200_000_000u64;
    let r = run_with_host(
        &manager,
        0,
        &[
            Value::I32(inst),
            Value::I32(stream),
            Value::I64(gmod as i64),
            Value::I64(mod1 as i64),
            Value::I64(mod2 as i64),
        ],
        &mut fuel,
        &mut host,
    )
    .expect("run");

    // The consumer read all 3 bytes the producer piped to it (returned 3) and uppercased them to the
    // shell's stdout: `one` wrote "hi\n", `two` turned it into "HI\n".
    assert_eq!(
        r,
        vec![Value::I64(3)],
        "the consumer drained 3 bytes through the pipe from the producer"
    );
    let out = host.stdout_bytes();
    assert_eq!(
        &out, b"HI\n",
        "one | two: the producer's output flowed through the pipe and the consumer uppercased it"
    );
}

/// FORK.md §8.6 — **concurrent pipe: `one | two` with both stages live at once.** Unlike the sequential
/// test above (fork producer, *wait*, fork consumer), the shell forks **both** stages, closes its own
/// copies of the pipe ends, and only then waits — so the consumer starts reading before the producer
/// has written. This exercises the **blocking read**: the producer `one` first burns a long compute
/// loop, so the consumer `two` reaches its `read(0)` on an empty FIFO and **parks** (writers still open)
/// rather than seeing a false EOF; when `one` finally writes and exits (the last write end closing), the
/// parked read wakes — draining the bytes, then EOF. Were the read non-blocking (the pre-slice
/// behavior) the consumer would EOF immediately and emit nothing. The producer's write end reaches it
/// through fork+exec; the shell's `close(fds…)` leaves `one` the sole writer so EOF arrives on its exit.
/// `one` writes `"go\n"`; `two` uppercases → the sink holds `"GO\n"` and `two` returns 3.
const CONCURRENT_PIPE_SHELL_SRC: &str = r#"
static char *a_one[] = { "one", 0 };
static char *a_two[] = { "two", 0 };
int main(int argc, char **argv) {
  int shell_out = (int)__vm_resolve("stdout", 6);
  int fds[2];
  pipe(fds);
  long p1 = fork();
  while (p1 < 0) p1 = fork();
  if (p1 == 0) { exec_io("one", a_one, fds[1], 0); return 127; }
  long p2 = fork();
  while (p2 < 0) p2 = fork();
  if (p2 == 0) { exec_io("two", a_two, shell_out, fds[0]); return 127; }
  /* Both stages are live; the shell drops its own ends so `one` is the sole writer (EOF on its exit). */
  close(fds[0]);
  close(fds[1]);
  wait_pid(p1);
  return wait_pid(p2);   /* the consumer's byte count */
}
"#;

/// The concurrent producer `one`: burn a compute loop *first* (so the consumer parks on an empty read),
/// then write `"go\n"`. The `volatile` accumulator keeps the loop from folding away.
const CONCURRENT_PRODUCER_SRC: &str = r#"
long write(long fd, void *buf, long n);
static char msg[] = "go\n";
int main(int argc, char **argv) {
  volatile long acc = 0;
  long i;
  for (i = 0; i < 3000000; i = i + 1) acc = acc + i;
  write(1, msg, 3);
  return 0;
}
"#;

#[test]
fn a_shell_runs_a_concurrent_pipe_with_a_blocking_read() {
    let manager = Arc::new(parse_module_raw(MICROSHELL_MANAGER).expect("parse microshell manager"));
    verify_module(&manager).expect("verify microshell manager");
    let shell_src = format!("{FORK_SHIM}\n{CONCURRENT_PIPE_SHELL_SRC}");
    let guest = parse_module_raw(&c_to_ir(&shell_src)).expect("parse concurrent pipe shell");
    verify_module(&guest).expect("verify concurrent pipe shell");
    let producer = parse_module_raw(&c_to_ir(CONCURRENT_PRODUCER_SRC)).expect("parse producer");
    verify_module(&producer).expect("verify producer");
    // Reuse the uppercasing consumer from the sequential test.
    let consumer = parse_module_raw(&c_to_ir(PIPE_CONSUMER_SRC)).expect("parse consumer");
    verify_module(&consumer).expect("verify consumer");

    let mut host = Host::new();
    host.set_self_module(&manager);
    let _sink = host.shared_stdout();
    let win = 1u64 << 19;
    let stream = host.grant_stream(StreamRole::Out);
    let inst = host.grant_instantiator(0, win);
    let gmod = host.grant_module(&guest);
    let mod1 = host.grant_module(&producer); // "one"
    let mod2 = host.grant_module(&consumer); // "two"

    let mut fuel = 400_000_000u64;
    let r = run_with_host(
        &manager,
        0,
        &[
            Value::I32(inst),
            Value::I32(stream),
            Value::I64(gmod as i64),
            Value::I64(mod1 as i64),
            Value::I64(mod2 as i64),
        ],
        &mut fuel,
        &mut host,
    )
    .expect("run");

    // The consumer blocked on the empty pipe, was woken by the producer's write, drained all 3 bytes
    // (returned 3), and — after the producer exited (EOF) — uppercased "go\n" to "GO\n" on the sink.
    assert_eq!(
        r,
        vec![Value::I64(3)],
        "the concurrent consumer blocked, woke on the write, and drained 3 bytes"
    );
    let out = host.stdout_bytes();
    assert_eq!(
        &out, b"GO\n",
        "concurrent one | two: the blocking read delivered the producer's output in full"
    );
}

/// FORK.md §8.6 (backpressure) — **a full pipe bounds a write to the capacity.** The pipe FIFO is no
/// longer unbounded: a `write` never grows it past `PIPE_CAP` (64 KiB). This shell mints a pipe, holds
/// both ends itself (so there is a reader — no EPIPE — and the write does not block), and writes a
/// 100 000-byte buffer in one call; the write **short-returns** the 65 536 bytes that fit, leaving the
/// rest for a later call once the reader drains. Proves the buffer is bounded (a runaway producer can't
/// balloon host memory) without needing a second live stage.
const BACKPRESSURE_BOUND_SHELL_SRC: &str = r#"
long __vm_write(int fd, void *buf, long len);
/* The source is a modest buffer; the write `len` intentionally exceeds both it and PIPE_CAP. The extra
 * source span is backed (zero-filled) window memory — the point of the test is the *return value*: the
 * bounded pipe accepts only PIPE_CAP bytes in one call and short-returns, so the FIFO can never balloon. */
static char src[256];
int main(int argc, char **argv) {
  int fds[2];
  pipe(fds);
  return (int)__vm_write(fds[1], src, 100000);   /* fills to PIPE_CAP, short-returns 65536 */
}
"#;

#[test]
fn a_full_pipe_write_is_bounded_to_the_capacity() {
    let manager = Arc::new(parse_module_raw(MICROSHELL_MANAGER).expect("parse microshell manager"));
    verify_module(&manager).expect("verify microshell manager");
    let shell_src = format!("{FORK_SHIM}\n{BACKPRESSURE_BOUND_SHELL_SRC}");
    let guest = parse_module_raw(&c_to_ir(&shell_src)).expect("parse backpressure shell");
    verify_module(&guest).expect("verify backpressure shell");
    // The shell uses no commands; grant the guest itself as the (unused) one/two module slots.
    let filler = parse_module_raw(&c_to_ir(EPIPE_CONSUMER_SRC)).expect("parse filler");
    verify_module(&filler).expect("verify filler");

    let mut host = Host::new();
    host.set_self_module(&manager);
    let _sink = host.shared_stdout();
    let win = 1u64 << 19;
    let stream = host.grant_stream(StreamRole::Out);
    let inst = host.grant_instantiator(0, win);
    let gmod = host.grant_module(&guest);
    let mod1 = host.grant_module(&filler);
    let mod2 = host.grant_module(&filler);

    let mut fuel = 60_000_000u64;
    let r = run_with_host(
        &manager,
        0,
        &[
            Value::I32(inst),
            Value::I32(stream),
            Value::I64(gmod as i64),
            Value::I64(mod1 as i64),
            Value::I64(mod2 as i64),
        ],
        &mut fuel,
        &mut host,
    )
    .expect("run");

    // The 100 000-byte write filled the FIFO to exactly PIPE_CAP (64 KiB) and short-returned — the
    // buffer never grew past the bound.
    assert_eq!(
        r,
        vec![Value::I64(65536)],
        "a write to a bounded pipe returns only the bytes that fit (PIPE_CAP = 64 KiB)"
    );
}

/// FORK.md §8.6 (SIGPIPE + backpressure) — the **`yes | head`** story: a producer that writes forever
/// into a consumer that reads a little and quits. The producer `one` writes 1000-byte chunks in a loop,
/// **checking the return** — so it blocks (backpressure) once the pipe fills, resumes as the consumer
/// drains, and finally gets `-EPIPE` (returns 88) the moment the consumer's read end is gone. The
/// consumer `two` reads *once* (up to 10 bytes) and exits, closing the last read end. Without EPIPE the
/// producer would spin to its 1 000 000-iteration safety cap (returning 99); without backpressure it
/// would balloon the FIFO to megabytes before the consumer exits. The shell forks both, drops its own
/// ends, reaps the consumer, then returns the **producer's** status — 88, proving EPIPE terminated it.
const EPIPE_PRODUCER_SRC: &str = r#"
long write(long fd, void *buf, long n);
static char buf[1000];
int main(int argc, char **argv) {
  long i, r;
  for (i = 0; i < 1000000; i = i + 1) {
    r = write(1, buf, 1000);
    if (r < 0) return 88;   /* -EPIPE: the consumer's read end is closed */
  }
  return 99;                /* safety cap — never reached if EPIPE works */
}
"#;

/// The `yes | head` consumer `two`: read *once* (up to 10 bytes) and exit — dropping the last read end,
/// which flips the producer's next write to `-EPIPE`.
const EPIPE_CONSUMER_SRC: &str = r#"
long read(long fd, void *buf, long n);
static char buf[10];
int main(int argc, char **argv) {
  return (int)read(0, buf, 10);
}
"#;

const EPIPE_SHELL_SRC: &str = r#"
static char *a_one[] = { "one", 0 };
static char *a_two[] = { "two", 0 };
int main(int argc, char **argv) {
  int shell_out = (int)__vm_resolve("stdout", 6);
  int fds[2];
  pipe(fds);
  long p1 = fork();
  while (p1 < 0) p1 = fork();
  if (p1 == 0) { exec_io("one", a_one, fds[1], 0); return 127; }
  long p2 = fork();
  while (p2 < 0) p2 = fork();
  if (p2 == 0) { exec_io("two", a_two, shell_out, fds[0]); return 127; }
  close(fds[0]);
  close(fds[1]);
  wait_pid(p2);              /* reap the consumer (its exit closes the last read end) */
  return wait_pid(p1);       /* the producer's status: 88 = it got EPIPE */
}
"#;

#[test]
fn a_producer_gets_epipe_when_its_consumer_exits() {
    let manager = Arc::new(parse_module_raw(MICROSHELL_MANAGER).expect("parse microshell manager"));
    verify_module(&manager).expect("verify microshell manager");
    let shell_src = format!("{FORK_SHIM}\n{EPIPE_SHELL_SRC}");
    let guest = parse_module_raw(&c_to_ir(&shell_src)).expect("parse epipe shell");
    verify_module(&guest).expect("verify epipe shell");
    let producer = parse_module_raw(&c_to_ir(EPIPE_PRODUCER_SRC)).expect("parse producer");
    verify_module(&producer).expect("verify producer");
    let consumer = parse_module_raw(&c_to_ir(EPIPE_CONSUMER_SRC)).expect("parse consumer");
    verify_module(&consumer).expect("verify consumer");

    let mut host = Host::new();
    host.set_self_module(&manager);
    let _sink = host.shared_stdout();
    let win = 1u64 << 19;
    let stream = host.grant_stream(StreamRole::Out);
    let inst = host.grant_instantiator(0, win);
    let gmod = host.grant_module(&guest);
    let mod1 = host.grant_module(&producer); // "one"
    let mod2 = host.grant_module(&consumer); // "two"

    let mut fuel = 400_000_000u64;
    let r = run_with_host(
        &manager,
        0,
        &[
            Value::I32(inst),
            Value::I32(stream),
            Value::I64(gmod as i64),
            Value::I64(mod1 as i64),
            Value::I64(mod2 as i64),
        ],
        &mut fuel,
        &mut host,
    )
    .expect("run");

    // The producer blocked when the pipe filled (backpressure), resumed as the consumer drained, and
    // got -EPIPE (returned 88) once the consumer exited — it did not spin to its safety cap (99).
    assert_eq!(
        r,
        vec![Value::I64(88)],
        "yes | head: the producer got EPIPE when the consumer's read end closed"
    );
}

/// FORK.md §8.6 — **file redirection `cmd > out.txt`**, built as a *pump* over the concurrent-pipe + fs
/// substrate (no new mechanism). The shell wires the command's stdout to a pipe **write** end, then —
/// instead of forking a second stage to drain it — the shell itself is the reader: it closes its own
/// copies of the pipe ends (leaving `cmd` the sole writer), then loops `read_fd(pipe_read)` and forwards
/// each chunk to a memfs file through the granted `vm_fs` cap (`FS_WRITE`), until the read returns EOF
/// (0) — which arrives when `cmd` exits and its write end closes (the writer count hits 0, waking the
/// blocking read). The read drains through the **direct `cap.call`** route (`__vm_read`), so this also
/// exercises the pipe-read park on that arm. `cmd` writes `"redirected!"`; the file ends up holding it.
const REDIRECT_CMD_SRC: &str = r#"
long write(long fd, void *buf, long n);
static char msg[] = "redirected!";
int main(int argc, char **argv) {
  write(1, msg, 11);
  return 11;
}
"#;

/// The redirect shell (links `FORK_SHIM`): fork `cmd` with its stdout on the pipe write end, drop both of
/// the shell's own pipe ends, `FS_OPEN` the target (`O_CREATE|O_WRITE`), pump pipe → file until EOF, then
/// close + reap. Returns the byte count pumped. `__vm_fs(op,a,b,c,d)` is the flat `call.sym "vm_fs"` the
/// manifest binds to the granted memfs; `read_fd`/`__vm_read` is the handle-specific blocking pipe read.
const REDIRECT_SHELL_SRC: &str = r#"
long __vm_fs(long op, long a, long b, long c, long d);
static char *a_cmd[] = { "cmd", 0 };
static char outpath[] = "out.txt";
static char buf[64];
int main(int argc, char **argv) {
  int fds[2];
  pipe(fds);
  long p = fork();
  while (p < 0) p = fork();
  if (p == 0) { exec_io("cmd", a_cmd, fds[1], 0); return 127; }
  close(fds[1]);                              /* the shell drops its write end: `cmd` is the sole writer */
  long wfd = __vm_fs(0, (long)outpath, 7, 18, 0);  /* FS_OPEN(out.txt, len=7, O_CREATE|O_WRITE = 16|2) */
  if (wfd < 0) return -1;
  long total = 0, n;
  while ((n = read_fd(fds[0], buf, 64)) > 0) {
    __vm_fs(2, wfd, (long)buf, n, 0);         /* FS_WRITE(wfd, buf, n) */
    total = total + n;
  }
  close(fds[0]);
  __vm_fs(4, wfd, 0, 0, 0);                   /* FS_CLOSE(wfd) */
  wait_pid(p);
  return total;
}
"#;

#[test]
fn a_shell_redirects_a_command_output_to_a_file() {
    let manager = Arc::new(parse_module_raw(FS_FORK_MANAGER).expect("parse fs-fork manager"));
    verify_module(&manager).expect("verify fs-fork manager");
    let shell_src = format!("{FORK_SHIM}\n{REDIRECT_SHELL_SRC}");
    let guest = parse_module_raw(&c_to_ir(&shell_src)).expect("parse redirect shell");
    verify_module(&guest).expect("verify redirect shell");
    let cmd = parse_module_raw(&c_to_ir(REDIRECT_CMD_SRC)).expect("parse redirect command");
    verify_module(&cmd).expect("verify redirect command");

    let mut host = Host::new();
    host.set_self_module(&manager);
    let _sink = host.shared_stdout();
    let win = 1u64 << 19;
    let stream = host.grant_stream(StreamRole::Out);
    let inst = host.grant_instantiator(0, win);
    let gmod = host.grant_module(&guest);
    let cmod = host.grant_module(&cmd);

    // A fresh shared memfs (no seed files): the shell creates `out.txt` and the redirect fills it.
    let (factory, memfs) = svm_run::fs::mem_fs_shared_factory(vec![], vec![]);
    let factory = std::sync::Arc::new(factory);
    let make: svm_interp::HostProcFork = {
        let factory = factory.clone();
        std::sync::Arc::new(move |_pid| -> svm_interp::ForkedProc {
            let mut inner = factory();
            svm_interp::ForkedProc::shared(Box::new(move |_slot_op, args, mem, minter| {
                inner(args[0] as u32, &args[1..], mem, minter)
            }))
        })
    };
    let fs_cap = host.grant_host_proc_forkable(make(0).handler, make.clone());

    let mut fuel = 120_000_000u64;
    let r = run_with_host(
        &manager,
        0,
        &[
            Value::I32(inst),
            Value::I32(stream),
            Value::I64(gmod as i64),
            Value::I64(cmod as i64),
            Value::I32(fs_cap),
        ],
        &mut fuel,
        &mut host,
    )
    .expect("run");

    // The shell pumped all 11 bytes of `cmd`'s output through the pipe into the file, then reaped `cmd`.
    assert_eq!(
        r,
        vec![Value::I64(11)],
        "the redirect shell pumped 11 bytes (`cmd`'s output) from the pipe into the file"
    );
    // `cmd`'s stdout went to the pipe, not the shell's stdout, so the shared sink stays empty.
    assert!(
        host.stdout_bytes().is_empty(),
        "the redirected command wrote to the pipe, not the shell's stdout"
    );
    // The redirect target holds exactly `cmd`'s output — a real file written by draining a real pipe.
    let (files, _dirs) = memfs.seed();
    let out = files
        .iter()
        .find(|(name, _)| name == "out.txt")
        .expect("the shell created out.txt in the shared memfs");
    assert_eq!(
        out.1, b"redirected!",
        "`cmd > out.txt`: the file holds the command's output, pumped through the pipe"
    );
}

/// `cmd >> file` (append). Identical pump to `>` except the shell opens the target with `O_APPEND`
/// (`O_CREATE|O_WRITE|O_APPEND` = 22), so the command's output lands **after** the file's existing bytes
/// rather than truncating. Reuses `REDIRECT_CMD_SRC` (writes `"redirected!"`); the memfs is seeded with a
/// `log.txt` already holding `"existing\n"`, so the result is the concatenation.
const APPEND_SHELL_SRC: &str = r#"
long __vm_fs(long op, long a, long b, long c, long d);
static char *a_cmd[] = { "cmd", 0 };
static char logpath[] = "log.txt";
static char buf[64];
int main(int argc, char **argv) {
  int fds[2];
  pipe(fds);
  long p = fork();
  while (p < 0) p = fork();
  if (p == 0) { exec_io("cmd", a_cmd, fds[1], 0); return 127; }
  close(fds[1]);
  long wfd = __vm_fs(0, (long)logpath, 7, 22, 0);  /* FS_OPEN(log.txt, len=7, O_CREATE|O_WRITE|O_APPEND) */
  if (wfd < 0) return -1;
  long total = 0, n;
  while ((n = read_fd(fds[0], buf, 64)) > 0) {
    __vm_fs(2, wfd, (long)buf, n, 0);
    total = total + n;
  }
  close(fds[0]);
  __vm_fs(4, wfd, 0, 0, 0);
  wait_pid(p);
  return total;
}
"#;

#[test]
fn a_shell_appends_a_command_output_to_a_file() {
    let manager = Arc::new(parse_module_raw(FS_FORK_MANAGER).expect("parse fs-fork manager"));
    verify_module(&manager).expect("verify fs-fork manager");
    let shell_src = format!("{FORK_SHIM}\n{APPEND_SHELL_SRC}");
    let guest = parse_module_raw(&c_to_ir(&shell_src)).expect("parse append shell");
    verify_module(&guest).expect("verify append shell");
    let cmd = parse_module_raw(&c_to_ir(REDIRECT_CMD_SRC)).expect("parse append command");
    verify_module(&cmd).expect("verify append command");

    let mut host = Host::new();
    host.set_self_module(&manager);
    let _sink = host.shared_stdout();
    let win = 1u64 << 19;
    let stream = host.grant_stream(StreamRole::Out);
    let inst = host.grant_instantiator(0, win);
    let gmod = host.grant_module(&guest);
    let cmod = host.grant_module(&cmd);

    // Seed `log.txt` with existing content — the append must preserve it and add after it.
    let (factory, memfs) = svm_run::fs::mem_fs_shared_factory(
        vec![("log.txt".into(), b"existing\n".to_vec())],
        vec![],
    );
    let factory = std::sync::Arc::new(factory);
    let make: svm_interp::HostProcFork = {
        let factory = factory.clone();
        std::sync::Arc::new(move |_pid| -> svm_interp::ForkedProc {
            let mut inner = factory();
            svm_interp::ForkedProc::shared(Box::new(move |_slot_op, args, mem, minter| {
                inner(args[0] as u32, &args[1..], mem, minter)
            }))
        })
    };
    let fs_cap = host.grant_host_proc_forkable(make(0).handler, make.clone());

    let mut fuel = 120_000_000u64;
    let r = run_with_host(
        &manager,
        0,
        &[
            Value::I32(inst),
            Value::I32(stream),
            Value::I64(gmod as i64),
            Value::I64(cmod as i64),
            Value::I32(fs_cap),
        ],
        &mut fuel,
        &mut host,
    )
    .expect("run");

    assert_eq!(
        r,
        vec![Value::I64(11)],
        "the append shell pumped 11 bytes (`cmd`'s output) into the file"
    );
    let (files, _dirs) = memfs.seed();
    let out = files
        .iter()
        .find(|(name, _)| name == "log.txt")
        .expect("log.txt survives the append");
    assert_eq!(
        out.1, b"existing\nredirected!",
        "`cmd >> log.txt`: the command's output is appended after the pre-existing content"
    );
}

/// `cmd < file` (input redirection). The **reverse** pump: the shell opens the source (`O_READ`), reads
/// it in chunks, and writes each into a pipe whose *read* end is the command's `stdin`; when the file is
/// drained the shell closes the write end, so the command's `stdin` read EOFs. `cmd` echoes its stdin to
/// stdout, so the shell's own stdout ends up holding the file's bytes — routed in through a real pipe.
const INPUT_CMD_SRC: &str = r#"
long read(long fd, void *buf, long n);
long write(long fd, void *buf, long n);
static char buf[64];
int main(int argc, char **argv) {
  long n;
  while ((n = read(0, buf, 64)) > 0) write(1, buf, n);  /* echo stdin -> stdout (fd ignored: stdin/stdout) */
  return 0;
}
"#;

/// The input-redirect shell (links `FORK_SHIM`): mint a pipe, fork `cmd` with the pipe **read** end as
/// its `stdin`, then pump `in.txt` → pipe **write** end until the file EOFs, close it, and reap.
const INPUT_SHELL_SRC: &str = r#"
long __vm_fs(long op, long a, long b, long c, long d);
static char *a_cmd[] = { "cmd", 0 };
static char inpath[] = "in.txt";
static char buf[64];
int main(int argc, char **argv) {
  int fds[2];
  pipe(fds);                                  /* fds[0] = read (command's stdin), fds[1] = write (shell pumps) */
  int shell_out = (int)__vm_resolve("stdout", 6);
  long p = fork();
  while (p < 0) p = fork();
  if (p == 0) { exec_io("cmd", a_cmd, shell_out, fds[0]); return 127; }
  close(fds[0]);                              /* the shell drops the read end: `cmd` is the sole reader */
  long rfd = __vm_fs(0, (long)inpath, 6, 1, 0);  /* FS_OPEN(in.txt, len=6, O_READ) */
  if (rfd < 0) return -1;
  long n;
  while ((n = __vm_fs(1, rfd, (long)buf, 64, 0)) > 0)  /* FS_READ the file... */
    write_fd(fds[1], buf, n);                          /* ...and pump it into the command's stdin pipe */
  __vm_fs(4, rfd, 0, 0, 0);
  close(fds[1]);                              /* EOF for the command's stdin read */
  return wait_pid(p);
}
"#;

#[test]
fn a_shell_redirects_a_file_into_a_command_stdin() {
    let manager = Arc::new(parse_module_raw(FS_FORK_MANAGER).expect("parse fs-fork manager"));
    verify_module(&manager).expect("verify fs-fork manager");
    let shell_src = format!("{FORK_SHIM}\n{INPUT_SHELL_SRC}");
    let guest = parse_module_raw(&c_to_ir(&shell_src)).expect("parse input shell");
    verify_module(&guest).expect("verify input shell");
    let cmd = parse_module_raw(&c_to_ir(INPUT_CMD_SRC)).expect("parse input command");
    verify_module(&cmd).expect("verify input command");

    let mut host = Host::new();
    host.set_self_module(&manager);
    let _sink = host.shared_stdout();
    let win = 1u64 << 19;
    let stream = host.grant_stream(StreamRole::Out);
    let inst = host.grant_instantiator(0, win);
    let gmod = host.grant_module(&guest);
    let cmod = host.grant_module(&cmd);

    // Seed the source file the shell reads and pumps into the command's stdin.
    let (factory, _memfs) = svm_run::fs::mem_fs_shared_factory(
        vec![("in.txt".into(), b"hello from a file".to_vec())],
        vec![],
    );
    let factory = std::sync::Arc::new(factory);
    let make: svm_interp::HostProcFork = {
        let factory = factory.clone();
        std::sync::Arc::new(move |_pid| -> svm_interp::ForkedProc {
            let mut inner = factory();
            svm_interp::ForkedProc::shared(Box::new(move |_slot_op, args, mem, minter| {
                inner(args[0] as u32, &args[1..], mem, minter)
            }))
        })
    };
    let fs_cap = host.grant_host_proc_forkable(make(0).handler, make.clone());

    let mut fuel = 120_000_000u64;
    let r = run_with_host(
        &manager,
        0,
        &[
            Value::I32(inst),
            Value::I32(stream),
            Value::I64(gmod as i64),
            Value::I64(cmod as i64),
            Value::I32(fs_cap),
        ],
        &mut fuel,
        &mut host,
    )
    .expect("run");

    assert_eq!(r, vec![Value::I64(0)], "the shell reaped `cmd`'s exit (0)");
    assert_eq!(
        host.stdout_bytes(),
        b"hello from a file",
        "`cmd < in.txt`: the file's bytes were pumped through a pipe into the command's stdin and echoed out"
    );
}

/// `cmd 2> file` (stderr redirection). The command writes normal output to `stdout` (the frontend `write`
/// builtin) and diagnostics to a **distinct** `stderr` handle it resolves by name (`__vm_write`). The
/// shell wires that stderr end to a pipe (via `exec_io3`), pumps it to `err.txt`, and leaves the command's
/// stdout on the shell's own stdout — so the two streams land in different places, the point of `2>`.
const STDERR_CMD_SRC: &str = r#"
long __vm_resolve(const char *name, long len);
long __vm_write(int h, void *buf, long len);
long write(long fd, void *buf, long n);
static char out_msg[] = "stdout-line";
static char err_msg[] = "stderr-line";
int main(int argc, char **argv) {
  int e = (int)__vm_resolve("stderr", 6);
  write(1, out_msg, 11);            /* normal output -> stdout (the ambient stream) */
  __vm_write(e, err_msg, 11);       /* diagnostics -> the granted stderr handle */
  return 0;
}
"#;

/// The stderr-redirect shell (links `FORK_SHIM`): fork `cmd` with its `stdout` on the shell's stdout and
/// its `stderr` on a pipe write end (`exec_io3`), then pump the stderr pipe → `err.txt` and reap.
const STDERR_SHELL_SRC: &str = r#"
long __vm_fs(long op, long a, long b, long c, long d);
static char *a_cmd[] = { "cmd", 0 };
static char errpath[] = "err.txt";
static char buf[64];
int main(int argc, char **argv) {
  int fds[2];
  pipe(fds);
  int shell_out = (int)__vm_resolve("stdout", 6);
  long p = fork();
  while (p < 0) p = fork();
  if (p == 0) { exec_io3("cmd", a_cmd, shell_out, 0, fds[1]); return 127; }  /* stderr -> pipe write end */
  close(fds[1]);
  long wfd = __vm_fs(0, (long)errpath, 7, 18, 0);  /* FS_OPEN(err.txt, len=7, O_CREATE|O_WRITE) */
  if (wfd < 0) return -1;
  long total = 0, n;
  while ((n = read_fd(fds[0], buf, 64)) > 0) {
    __vm_fs(2, wfd, (long)buf, n, 0);
    total = total + n;
  }
  close(fds[0]);
  __vm_fs(4, wfd, 0, 0, 0);
  wait_pid(p);
  return total;
}
"#;

#[test]
fn a_shell_redirects_a_command_stderr_to_a_file() {
    let manager = Arc::new(parse_module_raw(FS_FORK_MANAGER).expect("parse fs-fork manager"));
    verify_module(&manager).expect("verify fs-fork manager");
    let shell_src = format!("{FORK_SHIM}\n{STDERR_SHELL_SRC}");
    let guest = parse_module_raw(&c_to_ir(&shell_src)).expect("parse stderr shell");
    verify_module(&guest).expect("verify stderr shell");
    let cmd = parse_module_raw(&c_to_ir(STDERR_CMD_SRC)).expect("parse stderr command");
    verify_module(&cmd).expect("verify stderr command");

    let mut host = Host::new();
    host.set_self_module(&manager);
    let _sink = host.shared_stdout();
    let win = 1u64 << 19;
    let stream = host.grant_stream(StreamRole::Out);
    let inst = host.grant_instantiator(0, win);
    let gmod = host.grant_module(&guest);
    let cmod = host.grant_module(&cmd);

    let (factory, memfs) = svm_run::fs::mem_fs_shared_factory(vec![], vec![]);
    let factory = std::sync::Arc::new(factory);
    let make: svm_interp::HostProcFork = {
        let factory = factory.clone();
        std::sync::Arc::new(move |_pid| -> svm_interp::ForkedProc {
            let mut inner = factory();
            svm_interp::ForkedProc::shared(Box::new(move |_slot_op, args, mem, minter| {
                inner(args[0] as u32, &args[1..], mem, minter)
            }))
        })
    };
    let fs_cap = host.grant_host_proc_forkable(make(0).handler, make.clone());

    let mut fuel = 120_000_000u64;
    let r = run_with_host(
        &manager,
        0,
        &[
            Value::I32(inst),
            Value::I32(stream),
            Value::I64(gmod as i64),
            Value::I64(cmod as i64),
            Value::I32(fs_cap),
        ],
        &mut fuel,
        &mut host,
    )
    .expect("run");

    // The shell pumped the 11 stderr bytes into the file; the command's stdout went elsewhere.
    assert_eq!(
        r,
        vec![Value::I64(11)],
        "the shell pumped 11 bytes of the command's stderr into the file"
    );
    // Stream separation: stdout landed on the shell's stdout, stderr in the file.
    assert_eq!(
        host.stdout_bytes(),
        b"stdout-line",
        "the command's stdout stayed on the shell's stdout (not redirected)"
    );
    let (files, _dirs) = memfs.seed();
    let out = files
        .iter()
        .find(|(name, _)| name == "err.txt")
        .expect("the shell created err.txt");
    assert_eq!(
        out.1, b"stderr-line",
        "`cmd 2> err.txt`: the file holds exactly the command's stderr, pumped through the pipe"
    );
}

/// #773 (FORK.md §8.6) — **a forked child execs a command with a large static/BSS buffer.** A command
/// that declares more memory than the default 128 KiB window (its window directive is `memory 18`, 256
/// KiB, because of the 60 KB array) used to be un-`exec`able: the fork/exec surface required the child's
/// declared window to **equal** the caller's carve, but the shell shim hardcodes the exec's window hint
/// (17), so the command was rejected — and a manager that then joined the failed spawn wedged. The fix
/// admits a command whose declared memory **fits** the inherited window (a larger window is a safe
/// superset, still masked to its actual size by invariant 2). This shell is spawned into a 256 KiB
/// window (the manager passes carve size 18 at a 256 KiB-aligned offset), so its forked child inherits
/// the room to exec the 256 KiB command. The command touches the *far end* of its 60 KB buffer (proving
/// the big window is genuinely mapped and usable) and writes a byte from it; the shell reaps its exit.
const BIGEXEC_CMD_SRC: &str = r#"
long write(long fd, void *buf, long n);
static char big[60000];
int main(int argc, char **argv) {
  big[59999] = 'Z';           /* touch the far end of the 60 KB BSS (only reachable in the big window) */
  write(1, &big[59999], 1);   /* prove the large window is mapped + usable */
  return 0;
}
"#;

const BIGEXEC_SHELL_SRC: &str = r#"
static char *a_one[] = { "one", 0 };
int main(int argc, char **argv) {
  int shell_out = (int)__vm_resolve("stdout", 6);
  long p = fork();
  while (p < 0) p = fork();
  if (p == 0) { exec_io("one", a_one, shell_out, 0); return 127; }
  return wait_pid(p);         /* the large command's exit status (0) */
}
"#;

#[test]
fn a_shell_execs_a_command_with_a_large_bss_buffer() {
    // Spawn the shell into a 256 KiB window (carve size 18 at a 256 KiB-aligned offset) instead of the
    // default 128 KiB (size 17 at 128 KiB) — a `MICROSHELL_MANAGER` with the two carve constants bumped.
    // The small shell (`memory 17`) into a size-18 carve exercises op-13's window-≥-declared admission;
    // the forked child's exec of the `memory 18` command exercises op-14's.
    let bigwin = MICROSHELL_MANAGER
        .replace("voffg = i64.const 131072", "voffg = i64.const 262144")
        .replace("vsl = i64.const 17", "vsl = i64.const 18");
    let manager = Arc::new(parse_module_raw(&bigwin).expect("parse bigwin manager"));
    verify_module(&manager).expect("verify bigwin manager");
    let shell_src = format!("{FORK_SHIM}\n{BIGEXEC_SHELL_SRC}");
    let guest = parse_module_raw(&c_to_ir(&shell_src)).expect("parse bigexec shell");
    verify_module(&guest).expect("verify bigexec shell");
    let cmd = parse_module_raw(&c_to_ir(BIGEXEC_CMD_SRC)).expect("parse bigexec command");
    verify_module(&cmd).expect("verify bigexec command");

    let mut host = Host::new();
    host.set_self_module(&manager);
    let _sink = host.shared_stdout();
    let win = 1u64 << 19;
    let stream = host.grant_stream(StreamRole::Out);
    let inst = host.grant_instantiator(0, win);
    let gmod = host.grant_module(&guest);
    let mod1 = host.grant_module(&cmd); // "one"
    let mod2 = host.grant_module(&cmd); // "two" (unused)

    let mut fuel = 120_000_000u64;
    let r = run_with_host(
        &manager,
        0,
        &[
            Value::I32(inst),
            Value::I32(stream),
            Value::I64(gmod as i64),
            Value::I64(mod1 as i64),
            Value::I64(mod2 as i64),
        ],
        &mut fuel,
        &mut host,
    )
    .expect("run");

    // The command ran to completion in the big inherited window and the shell reaped its exit (0) —
    // no wedge, no exec refusal.
    assert_eq!(
        r,
        vec![Value::I64(0)],
        "the shell reaped the large-BSS command's exit status (0)"
    );
    // It genuinely used the far end of its 60 KB buffer and wrote from it.
    assert_eq!(
        host.stdout_bytes(),
        b"Z",
        "the exec'd command touched its 60 KB buffer (only reachable in the enlarged window) and wrote from it"
    );
}

/// Isolation (no fork/wait): a **nested op-13-spawned** compiled-C guest resolves a re-granted command
/// module `"cmd"` by name and `execve`s into it — testing the module-regrant + `__vm_resolve` +
/// `__vm_exec_module` builtins + nested-child image-replace, without the fork/wait topology.
const NEXEC_GUEST_SRC: &str = r#"
long __vm_resolve(const char *name, long len);
long __vm_exec_module(long mod, long grants, long n, long entry, long sl);
struct grant { int name_off; int name_len; int handle; int pad; };
static struct grant grec;
static char stdout_name[] = "stdout";
static char cmd_name[] = "cmd";
int main(int argc, char **argv) {
  long cmd = __vm_resolve(cmd_name, 3);
  long soh = __vm_resolve(stdout_name, 6);
  grec.name_off = (int)(long)stdout_name;
  grec.name_len = 6;
  grec.handle = (int)soh;
  __vm_exec_module(cmd, (long)&grec, 1, 0, 17);
  return -1;
}
"#;

const NEXEC_MANAGER: &str = r#"
memory 19
data 310 "stdout"
data 330 "cmd"
func (i32, i32, i64, i64) -> (i64) {
block 0 (v0: i32, vstream: i32, vgmod: i64, vcmod: i64) {
  vq = i64.const 0
  vcmod32 = i32.wrap_i64 vcmod
  va0 = i64.const 256
  vnp0 = i32.const 310
  i32.store va0 vnp0
  va1 = i64.const 260
  vsix = i32.const 6
  i32.store va1 vsix
  va2 = i64.const 264
  i32.store va2 vstream
  va3 = i64.const 272
  vnp1 = i32.const 330
  i32.store va3 vnp1
  va4 = i64.const 276
  vthree = i32.const 3
  i32.store va4 vthree
  va5 = i64.const 280
  i32.store va5 vcmod32
  vgp = i64.const 256
  vgn = i64.const 2
  ve0 = i64.const 0
  voffg = i64.const 131072
  vsl = i64.const 17
  vg = cap.call 6 13 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vgmod, vgp, vgn, ve0, voffg, vsl, vq)
  vjg = cap.call 6 1 (i32) -> (i64) v0 (vg)
  return vjg
  }
}
"#;

#[test]
fn a_nested_compiled_c_guest_execs_a_separate_command() {
    let manager = Arc::new(parse_module_raw(NEXEC_MANAGER).expect("parse nexec manager"));
    verify_module(&manager).expect("verify nexec manager");
    let guest = parse_module_raw(&c_to_ir(NEXEC_GUEST_SRC)).expect("parse nexec guest");
    verify_module(&guest).expect("verify nexec guest");
    let cmd = parse_module_raw(&c_to_ir(EXECVE_CMD_SRC)).expect("parse nexec command");
    verify_module(&cmd).expect("verify nexec command");

    let mut host = Host::new();
    host.set_self_module(&manager);
    let _sink = host.shared_stdout();
    let win = 1u64 << 19;
    let stream = host.grant_stream(StreamRole::Out);
    let inst = host.grant_instantiator(0, win);
    let gmod = host.grant_module(&guest);
    let cmod = host.grant_module(&cmd);

    let mut fuel = 120_000_000u64;
    let r = run_with_host(
        &manager,
        0,
        &[
            Value::I32(inst),
            Value::I32(stream),
            Value::I64(gmod as i64),
            Value::I64(cmod as i64),
        ],
        &mut fuel,
        &mut host,
    )
    .expect("run");

    assert_eq!(
        r,
        vec![Value::I64(42)],
        "the nested guest exec'd the command (exit 42)"
    );
    let out = host.stdout_bytes();
    assert_eq!(
        &out, b"EXEC",
        "the exec'd separate command wrote through the inherited stdout"
    );
}

/// FORK.md §8.6 (increment 3a) — **a real command reading a real file**: the isolation slice of the
/// "run a real command end-to-end" milestone. A nested op-13-spawned compiled-C guest `execve`s into a
/// small `cat`-shaped command that `open`/`read`/`close`s a file from a granted **`vm_fs` capability**
/// (the shared in-memory filesystem, `crates/svm-fs`) and writes the bytes to stdout — no fork/wait yet,
/// so this proves the fs-cap-through-exec plumbing on its own. Three caps ride the exec grant list:
/// `"stdout"` (the inherited stream), `"cmd"` (the command module — a module regrant), and `"vm_fs"`
/// (the memfs `HostProc`, re-granted by name so `bind_child_manifest` binds the command's
/// `call.sym "vm_fs"` straight to the closure — the op-in-arg0 fs protocol). The manager seeds
/// `greeting = "HELLO"` host-side; the command reads it back through the cap and echoes it.
const FS_CAT_CMD_SRC: &str = r#"
long write(long fd, void *buf, long n);
long __vm_fs(long op, long a, long b, long c, long d);
static char path[] = "greeting";
static char buf[128];
int main(int argc, char **argv) {
  long fd = __vm_fs(0, (long)path, 8, 1, 0);   /* FS_OPEN(path, len=8, O_READ) */
  if (fd < 0) return 1;
  long n = __vm_fs(1, fd, (long)buf, 128, 0);  /* FS_READ(fd, buf, cap) */
  if (n > 0) write(1, buf, n);
  __vm_fs(4, fd, 0, 0, 0);                     /* FS_CLOSE(fd) */
  return (int)n;
}
"#;

/// The nested guest (no fork/wait): resolve the three re-granted caps by name, build a **2-entry** exec
/// grant list `{"stdout" → stream, "vm_fs" → memfs}` (the command inherits both), and `execve` into the
/// `"cmd"` command. Mirrors `NEXEC_GUEST_SRC` but carries the fs cap forward to the exec'd image.
const FS_GUEST_SRC: &str = r#"
long __vm_resolve(const char *name, long len);
long __vm_exec_module(long mod, long grants, long n, long entry, long sl);
struct grant { int name_off; int name_len; int handle; int pad; };
static struct grant grecs[2];
static char stdout_name[] = "stdout";
static char cmd_name[] = "cmd";
static char fs_name[] = "vm_fs";
int main(int argc, char **argv) {
  long cmd = __vm_resolve(cmd_name, 3);
  long soh = __vm_resolve(stdout_name, 6);
  long fsh = __vm_resolve(fs_name, 5);
  grecs[0].name_off = (int)(long)stdout_name;
  grecs[0].name_len = 6;
  grecs[0].handle = (int)soh;
  grecs[1].name_off = (int)(long)fs_name;
  grecs[1].name_len = 5;
  grecs[1].handle = (int)fsh;
  __vm_exec_module(cmd, (long)grecs, 2, 0, 17);
  return -1;
}
"#;

/// The manager: `main(inst, stream, guestmod, cmdmod, fscap)` spawns the guest via op 13 with a
/// **3-entry** grant list `{"stdout" → stream, "cmd" → cmdmod, "vm_fs" → fscap}` (the fs `HostProc`
/// re-granted by name — `regrant_into_child` re-mints its forkable closure over the shared store), then
/// joins the guest. Like `NEXEC_MANAGER` with the fs cap added as a third entry.
const FS_MANAGER: &str = r#"
memory 19
data 310 "stdout"
data 330 "cmd"
data 340 "vm_fs"
func (i32, i32, i64, i64, i32) -> (i64) {
block 0 (v0: i32, vstream: i32, vgmod: i64, vcmod: i64, vfs: i32) {
  vq = i64.const 0
  vcmod32 = i32.wrap_i64 vcmod
  va0 = i64.const 256
  vnp0 = i32.const 310
  i32.store va0 vnp0
  va1 = i64.const 260
  vsix = i32.const 6
  i32.store va1 vsix
  va2 = i64.const 264
  i32.store va2 vstream
  va3 = i64.const 272
  vnp1 = i32.const 330
  i32.store va3 vnp1
  va4 = i64.const 276
  vthree = i32.const 3
  i32.store va4 vthree
  va5 = i64.const 280
  i32.store va5 vcmod32
  va6 = i64.const 288
  vnp2 = i32.const 340
  i32.store va6 vnp2
  va7 = i64.const 292
  vfive = i32.const 5
  i32.store va7 vfive
  va8 = i64.const 296
  i32.store va8 vfs
  vgp = i64.const 256
  vgn = i64.const 3
  ve0 = i64.const 0
  voffg = i64.const 131072
  vsl = i64.const 17
  vg = cap.call 6 13 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vgmod, vgp, vgn, ve0, voffg, vsl, vq)
  vjg = cap.call 6 1 (i32) -> (i64) v0 (vg)
  return vjg
  }
}
"#;

#[test]
fn a_nested_compiled_c_command_reads_a_file_through_a_granted_fs_cap() {
    let manager = Arc::new(parse_module_raw(FS_MANAGER).expect("parse fs manager"));
    verify_module(&manager).expect("verify fs manager");
    let guest = parse_module_raw(&c_to_ir(FS_GUEST_SRC)).expect("parse fs guest");
    verify_module(&guest).expect("verify fs guest");
    let cmd = parse_module_raw(&c_to_ir(FS_CAT_CMD_SRC)).expect("parse fs command");
    verify_module(&cmd).expect("verify fs command");

    let mut host = Host::new();
    host.set_self_module(&manager);
    let _sink = host.shared_stdout();
    let win = 1u64 << 19;
    let stream = host.grant_stream(StreamRole::Out);
    let inst = host.grant_instantiator(0, win);
    let gmod = host.grant_module(&guest);
    let cmod = host.grant_module(&cmd);

    // The shared in-memory filesystem, seeded `greeting = "HELLO"`. The cap is granted **forkable**
    // (a factory that re-mints the closure over the one shared store), the shape `regrant_into_child`
    // carries into a spawned child; the op-in-arg0 wrapper drops the dispatch op and forwards `args[0]`
    // as the fs op (matching the compiled-C `call.sym "vm_fs"` protocol, as in `c_link.rs`).
    let (factory, _memfs) = svm_run::fs::mem_fs_shared_factory(
        vec![("greeting".to_string(), b"HELLO".to_vec())],
        vec![],
    );
    let factory = std::sync::Arc::new(factory);
    let make: svm_interp::HostProcFork = {
        let factory = factory.clone();
        std::sync::Arc::new(move |_pid| -> svm_interp::ForkedProc {
            let mut inner = factory();
            svm_interp::ForkedProc::shared(Box::new(move |_slot_op, args, mem, minter| {
                inner(args[0] as u32, &args[1..], mem, minter)
            }))
        })
    };
    let fs_cap = host.grant_host_proc_forkable(make(0).handler, make.clone());

    let mut fuel = 120_000_000u64;
    let r = run_with_host(
        &manager,
        0,
        &[
            Value::I32(inst),
            Value::I32(stream),
            Value::I64(gmod as i64),
            Value::I64(cmod as i64),
            Value::I32(fs_cap),
        ],
        &mut fuel,
        &mut host,
    )
    .expect("run");

    assert_eq!(
        r,
        vec![Value::I64(5)],
        "the command read all 5 bytes of `greeting` and returned the count"
    );
    let out = host.stdout_bytes();
    assert_eq!(
        &out, b"HELLO",
        "the real `cat` command echoed the file it read through the granted fs cap"
    );
}

/// FORK.md §8.6 — **the shell command loop with a *real* `execve`**: `fork()` → **`execve` (image
/// -replace of a separate command module)** → `wait()`, all in ordinary compiled C. This upgrades the
/// multicall stand-in above to the true capstone — the forked child *becomes a different program*.
///
/// - **fork** through `__fork` (retry on the `-EAGAIN` serve/park race).
/// - **execve** through `__vm_exec_module` (the `CAP_SELF_EXEC` self-op): the child resolves the
///   command module `"cmd"` (re-granted to it by name — modules are now regrantable) and the inherited
///   `"stdout"` handle (`__vm_resolve`), builds a 1-entry grant list, and replaces its image with the
///   command. Its `TaskId` is preserved, so it *is* the command now.
/// - **wait** through `__wait` (`reap`): the parent reaps the twin — which is running the command — and
///   returns the command's exit status.
///
/// The **separate command module** writes `"EXEC"` to the inherited stdout and exits `42`. So the sink
/// holds exactly `"EXEC"` (a different program did that I/O, as the child's task) and the run returns
/// `42` (the parent reaped the command's exit through the child's pid). Interp only, like every fork test.
const EXECVE_GUEST_SRC: &str = r#"
long write(long fd, void *buf, long n);
long __fork(int h, long a);
long __wait(int h, long pid);
long __vm_resolve(const char *name, long len);
long __vm_exec_module(long mod, long grants, long n, long entry, long sl);
long fork(void) { return __fork(0, 0); }
long wait_pid(long pid) { return __wait(0, pid); }
struct grant { int name_off; int name_len; int handle; int pad; };
static struct grant grec;
static char stdout_name[] = "stdout";
static char cmd_name[] = "cmd";
static long pid;
static long status;
int main(int argc, char **argv) {
  while ((pid = fork()) < 0);
  if (pid == 0) {
    long cmd = __vm_resolve(cmd_name, 3);
    long soh = __vm_resolve(stdout_name, 6);
    grec.name_off = (int)(long)stdout_name;
    grec.name_len = 6;
    grec.handle = (int)soh;
    __vm_exec_module(cmd, (long)&grec, 1, 0, 17);
    return -1;
  }
  while ((status = wait_pid(pid)) < 0);
  return status;
}
"#;

/// The separate command the child `execve`s into: write `"EXEC"` to the inherited stdout (its `write`
/// import binds to the regranted `"stdout"` at exec) and exit `42`.
const EXECVE_CMD_SRC: &str = r#"
long write(long fd, void *buf, long n);
static char msg[] = "EXEC";
int main(int argc, char **argv) {
  write(1, msg, 4);
  return 42;
}
"#;

/// The manager: like `EXEC_MANAGER` but `main(inst, stream, guestmod, cmdmod)` re-grants a **4th** entry
/// `{"cmd" → command module}` (a module regrant) so the guest resolves it by name and `execve`s it.
const EXECVE_MANAGER: &str = r#"
memory 19
type 0 func (i64) -> (i64)
type 1 interface { op: 0 }
export 0 interface "fork" 1 { op: 2 }
export 1 interface "wait" 1 { op: 3 }
data 400 "__fork"
data 410 "stdout"
data 420 "__wait"
data 430 "cmd"
func (i32, i32, i64, i64) -> (i64) {
block 0 (v0: i32, vstream: i32, vgmod: i64, vcmod: i64) {
  vq = i64.const 0
  q0v0 = i64.const 4294967296
  q0v1 = i64.const 262144
  q0v2 = i64.const -4294967284
  q0v3 = i64.const 4294967295
  q0v4 = i64.const 0
  q0a0 = i64.const 1152
  i64.store q0a0 q0v0
  q0a1 = i64.const 1160
  i64.store q0a1 q0v1
  q0a2 = i64.const 1168
  i64.store q0a2 q0v2
  q0a3 = i64.const 1176
  i64.store q0a3 q0v3
  q0a4 = i64.const 1184
  i64.store q0a4 q0v4
  q0a5 = i64.const 1192
  i64.store q0a5 q0v4
  q0a6 = i64.const 1200
  i64.store q0a6 q0v4
  vs = cap.call 6 17 (i64) -> (i32) v0 (q0a0)
  vz0 = i64.const 0
  vforkoff = cap.call 6 14 (i32, i64) -> (i32) v0 (vs, vz0)
  v1c = i64.const 1
  vwaitoff = cap.call 6 14 (i32, i64) -> (i32) v0 (vs, v1c)
  vcmod32 = i32.wrap_i64 vcmod
  va0 = i64.const 256
  vnp0 = i32.const 410
  i32.store va0 vnp0
  va1 = i64.const 260
  vsix = i32.const 6
  i32.store va1 vsix
  va2 = i64.const 264
  i32.store va2 vstream
  va3 = i64.const 272
  vnp1 = i32.const 400
  i32.store va3 vnp1
  va4 = i64.const 276
  i32.store va4 vsix
  va5 = i64.const 280
  i32.store va5 vforkoff
  va6 = i64.const 288
  vnp2 = i32.const 420
  i32.store va6 vnp2
  va7 = i64.const 292
  i32.store va7 vsix
  va8 = i64.const 296
  i32.store va8 vwaitoff
  va9 = i64.const 304
  vnp3 = i32.const 430
  i32.store va9 vnp3
  va10 = i64.const 308
  vthree = i32.const 3
  i32.store va10 vthree
  va11 = i64.const 312
  i32.store va11 vcmod32
  vgp = i64.const 256
  vgn = i64.const 4
  ve0 = i64.const 0
  voffg = i64.const 131072
  vsl = i64.const 17
  vg = cap.call 6 13 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vgmod, vgp, vgn, ve0, voffg, vsl, vq)
  vjg = cap.call 6 1 (i32) -> (i64) v0 (vg)
  return vjg
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  br 1()
  }
block 1 () {
  vz = i32.const 0
  vn = cap.call 4294967295 10 () -> (i64) vz ()
  br 1()
  }
}
func (i64) -> (i64) {
block 0 (vx: i64) {
  vz = i32.const 0
  vzero = i64.const 0
  vt = cap.call 4294967295 11 (i64) -> (i64) vz (vzero)
  return vt
  }
}
func (i64) -> (i64) {
block 0 (vpid: i64) {
  vz = i32.const 0
  vt = cap.call 4294967295 12 (i64) -> (i64) vz (vpid)
  return vt
  }
}
"#;

#[test]
fn a_compiled_c_program_runs_fork_execve_wait_with_a_separate_command() {
    let manager = Arc::new(parse_module_raw(EXECVE_MANAGER).expect("parse execve manager"));
    verify_module(&manager).expect("verify execve manager");
    let guest = parse_module_raw(&c_to_ir(EXECVE_GUEST_SRC)).expect("parse execve guest");
    verify_module(&guest).expect("verify execve guest");
    let cmd = parse_module_raw(&c_to_ir(EXECVE_CMD_SRC)).expect("parse execve command");
    verify_module(&cmd).expect("verify execve command");

    let mut host = Host::new();
    host.set_self_module(&manager);
    let _sink = host.shared_stdout();
    let win = 1u64 << 19;
    let stream = host.grant_stream(StreamRole::Out);
    let inst = host.grant_instantiator(0, win);
    let gmod = host.grant_module(&guest);
    let cmod = host.grant_module(&cmd);

    let mut fuel = 120_000_000u64;
    let r = run_with_host(
        &manager,
        0,
        &[
            Value::I32(inst),
            Value::I32(stream),
            Value::I64(gmod as i64),
            Value::I64(cmod as i64),
        ],
        &mut fuel,
        &mut host,
    )
    .expect("run");

    // The parent reaped the twin — which had replaced its image with the command — and returned the
    // command's exit status (42).
    assert_eq!(
        r,
        vec![Value::I64(42)],
        "the parent's wait reaped the exec'd separate command's exit status (42)"
    );
    // The separate command module wrote "EXEC" to the inherited stdout, as the child's task.
    let out = host.stdout_bytes();
    assert_eq!(
        &out, b"EXEC",
        "the exec'd separate command did real I/O through the inherited stdout"
    );
}

/// FORK.md §8.6 (increment 3b) — **run a real command end-to-end**: the capstone of the milestone.
/// `fork()` → `execve` (into a separate command) → `wait()`, all compiled C — *and* the command the
/// child becomes is the real `cat` from 3a, doing genuine file I/O through a granted `vm_fs` cap. This
/// is the full POSIX shell loop over a real program: the parent forks, the twin replaces its image with
/// `cat`, `cat` `open`/`read`/`close`s `greeting` from the shared memfs and writes `"HELLO"` to the
/// inherited stdout, and the parent reaps the twin's exit status (the byte count). The manager re-grants
/// **five** caps (fork/wait offers, stdout, cmd, and now `vm_fs`); the forked child carries `stdout` +
/// `vm_fs` into its execve grant list so the exec'd `cat` inherits the filesystem authority.
const FS_FORK_GUEST_SRC: &str = r#"
long write(long fd, void *buf, long n);
long __fork(int h, long a);
long __wait(int h, long pid);
long __vm_resolve(const char *name, long len);
long __vm_exec_module(long mod, long grants, long n, long entry, long sl);
long fork(void) { return __fork(0, 0); }
long wait_pid(long pid) { return __wait(0, pid); }
struct grant { int name_off; int name_len; int handle; int pad; };
static struct grant grecs[2];
static char stdout_name[] = "stdout";
static char cmd_name[] = "cmd";
static char fs_name[] = "vm_fs";
static long pid;
static long status;
int main(int argc, char **argv) {
  while ((pid = fork()) < 0);
  if (pid == 0) {
    long cmd = __vm_resolve(cmd_name, 3);
    long soh = __vm_resolve(stdout_name, 6);
    long fsh = __vm_resolve(fs_name, 5);
    grecs[0].name_off = (int)(long)stdout_name;
    grecs[0].name_len = 6;
    grecs[0].handle = (int)soh;
    grecs[1].name_off = (int)(long)fs_name;
    grecs[1].name_len = 5;
    grecs[1].handle = (int)fsh;
    __vm_exec_module(cmd, (long)grecs, 2, 0, 17);
    return -1;
  }
  while ((status = wait_pid(pid)) < 0);
  return status;
}
"#;

/// The manager: like `EXECVE_MANAGER` but `main(inst, stream, guestmod, cmdmod, fscap)` re-grants a
/// **5th** entry `{"vm_fs" → fscap}` (the memfs `HostProc`) so the forked child can carry it into its
/// execve grant list. Grant list: `{stdout, __fork, __wait, cmd, vm_fs}`.
const FS_FORK_MANAGER: &str = r#"
memory 19
type 0 func (i64) -> (i64)
type 1 interface { op: 0 }
export 0 interface "fork" 1 { op: 2 }
export 1 interface "wait" 1 { op: 3 }
data 400 "__fork"
data 410 "stdout"
data 420 "__wait"
data 430 "cmd"
data 440 "vm_fs"
func (i32, i32, i64, i64, i32) -> (i64) {
block 0 (v0: i32, vstream: i32, vgmod: i64, vcmod: i64, vfs: i32) {
  vq = i64.const 0
  q0v0 = i64.const 4294967296
  q0v1 = i64.const 262144
  q0v2 = i64.const -4294967284
  q0v3 = i64.const 4294967295
  q0v4 = i64.const 0
  q0a0 = i64.const 1152
  i64.store q0a0 q0v0
  q0a1 = i64.const 1160
  i64.store q0a1 q0v1
  q0a2 = i64.const 1168
  i64.store q0a2 q0v2
  q0a3 = i64.const 1176
  i64.store q0a3 q0v3
  q0a4 = i64.const 1184
  i64.store q0a4 q0v4
  q0a5 = i64.const 1192
  i64.store q0a5 q0v4
  q0a6 = i64.const 1200
  i64.store q0a6 q0v4
  vs = cap.call 6 17 (i64) -> (i32) v0 (q0a0)
  vz0 = i64.const 0
  vforkoff = cap.call 6 14 (i32, i64) -> (i32) v0 (vs, vz0)
  v1c = i64.const 1
  vwaitoff = cap.call 6 14 (i32, i64) -> (i32) v0 (vs, v1c)
  vcmod32 = i32.wrap_i64 vcmod
  va0 = i64.const 256
  vnp0 = i32.const 410
  i32.store va0 vnp0
  va1 = i64.const 260
  vsix = i32.const 6
  i32.store va1 vsix
  va2 = i64.const 264
  i32.store va2 vstream
  va3 = i64.const 272
  vnp1 = i32.const 400
  i32.store va3 vnp1
  va4 = i64.const 276
  i32.store va4 vsix
  va5 = i64.const 280
  i32.store va5 vforkoff
  va6 = i64.const 288
  vnp2 = i32.const 420
  i32.store va6 vnp2
  va7 = i64.const 292
  i32.store va7 vsix
  va8 = i64.const 296
  i32.store va8 vwaitoff
  va9 = i64.const 304
  vnp3 = i32.const 430
  i32.store va9 vnp3
  va10 = i64.const 308
  vthree = i32.const 3
  i32.store va10 vthree
  va11 = i64.const 312
  i32.store va11 vcmod32
  va12 = i64.const 320
  vnp4 = i32.const 440
  i32.store va12 vnp4
  va13 = i64.const 324
  vfive = i32.const 5
  i32.store va13 vfive
  va14 = i64.const 328
  i32.store va14 vfs
  vgp = i64.const 256
  vgn = i64.const 5
  ve0 = i64.const 0
  voffg = i64.const 131072
  vsl = i64.const 17
  vg = cap.call 6 13 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vgmod, vgp, vgn, ve0, voffg, vsl, vq)
  vjg = cap.call 6 1 (i32) -> (i64) v0 (vg)
  return vjg
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  br 1()
  }
block 1 () {
  vz = i32.const 0
  vn = cap.call 4294967295 10 () -> (i64) vz ()
  br 1()
  }
}
func (i64) -> (i64) {
block 0 (vx: i64) {
  vz = i32.const 0
  vzero = i64.const 0
  vt = cap.call 4294967295 11 (i64) -> (i64) vz (vzero)
  return vt
  }
}
func (i64) -> (i64) {
block 0 (vpid: i64) {
  vz = i32.const 0
  vt = cap.call 4294967295 12 (i64) -> (i64) vz (vpid)
  return vt
  }
}
"#;

#[test]
fn a_compiled_c_program_forks_execs_a_real_command_that_reads_a_file_and_waits() {
    let manager = Arc::new(parse_module_raw(FS_FORK_MANAGER).expect("parse fs-fork manager"));
    verify_module(&manager).expect("verify fs-fork manager");
    let guest = parse_module_raw(&c_to_ir(FS_FORK_GUEST_SRC)).expect("parse fs-fork guest");
    verify_module(&guest).expect("verify fs-fork guest");
    let cmd = parse_module_raw(&c_to_ir(FS_CAT_CMD_SRC)).expect("parse fs-fork command");
    verify_module(&cmd).expect("verify fs-fork command");

    let mut host = Host::new();
    host.set_self_module(&manager);
    let _sink = host.shared_stdout();
    let win = 1u64 << 19;
    let stream = host.grant_stream(StreamRole::Out);
    let inst = host.grant_instantiator(0, win);
    let gmod = host.grant_module(&guest);
    let cmod = host.grant_module(&cmd);

    let (factory, _memfs) = svm_run::fs::mem_fs_shared_factory(
        vec![("greeting".to_string(), b"HELLO".to_vec())],
        vec![],
    );
    let factory = std::sync::Arc::new(factory);
    let make: svm_interp::HostProcFork = {
        let factory = factory.clone();
        std::sync::Arc::new(move |_pid| -> svm_interp::ForkedProc {
            let mut inner = factory();
            svm_interp::ForkedProc::shared(Box::new(move |_slot_op, args, mem, minter| {
                inner(args[0] as u32, &args[1..], mem, minter)
            }))
        })
    };
    let fs_cap = host.grant_host_proc_forkable(make(0).handler, make.clone());

    let mut fuel = 120_000_000u64;
    let r = run_with_host(
        &manager,
        0,
        &[
            Value::I32(inst),
            Value::I32(stream),
            Value::I64(gmod as i64),
            Value::I64(cmod as i64),
            Value::I32(fs_cap),
        ],
        &mut fuel,
        &mut host,
    )
    .expect("run");

    // The parent forked, the twin exec'd `cat`, and the parent reaped its exit status (5 bytes read).
    assert_eq!(
        r,
        vec![Value::I64(5)],
        "the parent's wait reaped the forked+exec'd `cat`'s exit status (5 bytes read)"
    );
    // A real command — a different program running as the child's task — read a real file from the
    // shared memfs and echoed it through the inherited stdout.
    let out = host.stdout_bytes();
    assert_eq!(
        &out, b"HELLO",
        "the forked+exec'd `cat` read `greeting` through the inherited fs cap and wrote it to stdout"
    );
}

#[test]
fn a_compiled_c_program_forks_for_real_and_both_copies_write_through_the_shared_stream() {
    let manager = Arc::new(parse_module_raw(MANAGER).expect("parse manager"));
    verify_module(&manager).expect("verify manager");
    // Parse the guest RAW so its `write`/`__fork` call.syms stay manifest imports the op-13 spawn binds.
    let guest = parse_module_raw(&c_to_ir(GUEST_SRC)).expect("parse guest");
    verify_module(&guest).expect("verify guest");

    let mut host = Host::new();
    host.set_self_module(&manager);
    let _sink = host.shared_stdout(); // unify the stdout stream + the re-granted child stream into one sink
    let win = 1u64 << 19;
    let stream = host.grant_stream(StreamRole::Out);
    let inst = host.grant_instantiator(0, win);
    let gmod = host.grant_module(&guest);

    let mut fuel = 60_000_000u64;
    let r = run_with_host(
        &manager,
        0,
        &[
            Value::I32(inst),
            Value::I32(stream),
            Value::I64(gmod as i64),
        ],
        &mut fuel,
        &mut host,
    )
    .expect("run");

    assert_eq!(
        r,
        vec![Value::I64(3)],
        "the compiled guest's fork() returns the twin's pid (task id 3)"
    );

    let out = host.stdout_bytes();
    assert_eq!(
        out.len(),
        16,
        "two 8-byte writes reached the shared stdout stream"
    );
    let mut vals: Vec<i64> = out
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    vals.sort();
    assert_eq!(
        vals,
        vec![0, 3],
        "child wrote 0, parent wrote its pid (3) — a real compiled-C fork() through the shared stream"
    );
}

/// #863 slice 2 — **`kill(pid, SIGUSR1)` from a compiled-C parent to its forked child, by the pid
/// `fork()` returned.** The one-pid-space integration pin: the pid the core hands the parent (the
/// twin's `TaskId`) must be the SAME pid the personality registered in its process table at mint —
/// if they disagreed, the parent's `kill` would be `-ESRCH`. The child installs a SIGUSR1 handler
/// (the L0 doorbell) and spins on `sigcheck`; the parent kills an unknown pid first (`-ESRCH`),
/// then the child's; the child's own `sigcheck` — and only the child's — delivers the handler
/// token, and the child's exit (42) flows back through `wait`. Impossible before the World/Proc
/// split: parent and child shared one pending set, so per-child targeting had no address.
///
/// Reuses `FS_FORK_MANAGER` (fork + wait offers, one re-granted forkable cap): the `"__vm_fs"`
/// import slot here carries the op-shifted **posix** cap — same wire shape (`args[0]` = op), only
/// the personality behind it differs.
const KILL_BY_PID_SRC: &str = r#"
long __vm_fs(long op, long a, long b, long c, long d);
static long pid;
static long st;
static long h;
int main(int argc, char **argv) {
  /* Pre-install a caught disposition for 10 — inherited by the fork twin (POSIX), so the kill
   * below can never land on SIG_DFL (#796: that would terminate the child, not pend). */
  __vm_fs(30, 10, 6, 0, 0);
  while ((pid = fork()) < 0);
  if (pid == 0) {
    __vm_fs(30, 10, 7, 0, 0);                    /* signal(10, its own handler-token 7) */
    while ((h = __vm_fs(32, 0, 0, 0, 0)) == 0);  /* sigcheck: spin until the kill lands */
    if (h == 7) return 42;
    return 9;
  }
  if (__vm_fs(31, 99, 10, 0, 0) != -3) return 1; /* kill(unknown pid) is -ESRCH */
  if (__vm_fs(31, pid, 10, 0, 0) != 0) return 2; /* kill(child, 10) — by the fork() pid */
  if (__vm_fs(32, 0, 0, 0, 0) != 0) return 3;    /* the PARENT's own sigcheck stays empty */
  while ((st = wait_pid(pid)) < 0);
  return st;                                     /* 42: the child saw the pid-targeted signal */
}
"#;

/// Op-shift a posix fork factory into the manager re-grant wire shape (`args[0]` = op, like the
/// fs-cap tests), preserving the #863 extras — the per-process signal door rides along untouched,
/// and the replacement factory is re-wrapped so fork-of-fork keeps the shape.
fn opshift_fork(base: svm_interp::HostProcFork) -> svm_interp::HostProcFork {
    Arc::new(move |pid| {
        let forked = base(pid);
        let mut inner = forked.handler;
        svm_interp::ForkedProc {
            handler: Box::new(move |_slot_op, args, mem, minter| {
                inner(args[0] as u32, &args[1..], mem, minter)
            }),
            signal: forked.signal,
            refork: forked.refork.map(opshift_fork),
            exit: forked.exit,
        }
    })
}

#[test]
fn a_compiled_c_parent_kills_its_forked_child_by_pid() {
    let manager = Arc::new(parse_module_raw(FS_FORK_MANAGER).expect("parse fs-fork manager"));
    verify_module(&manager).expect("verify fs-fork manager");
    let guest_src = format!("{FORK_SHIM}\n{KILL_BY_PID_SRC}");
    let guest = parse_module_raw(&c_to_ir(&guest_src)).expect("parse kill-by-pid guest");
    verify_module(&guest).expect("verify kill-by-pid guest");

    let mut host = Host::new();
    host.set_self_module(&manager);
    let _sink = host.shared_stdout();
    let win = 1u64 << 19;
    let stream = host.grant_stream(StreamRole::Out);
    let inst = host.grant_instantiator(0, win);
    let gmod = host.grant_module(&guest);

    let (posix, make) = svm_posix::cap(4096, 1 << 16, Vec::new());
    let make = Arc::new(make);
    let px_handler: svm_interp::HostProc = {
        let mut inner = make();
        Box::new(move |_slot_op, args, mem, minter| inner(args[0] as u32, &args[1..], mem, minter))
    };
    let px_cap = host.grant_host_proc_forkable(
        px_handler,
        opshift_fork(svm_posix::cap_fork_factory(&posix)),
    );

    let mut fuel = 120_000_000u64;
    let r = run_with_host(
        &manager,
        0,
        &[
            Value::I32(inst),
            Value::I32(stream),
            Value::I64(gmod as i64),
            Value::I64(gmod as i64), // the cmd-module slot — unused by this guest
            Value::I32(px_cap),
        ],
        &mut fuel,
        &mut host,
    )
    .expect("run");

    assert_eq!(
        r,
        vec![Value::I64(42)],
        "the child saw the parent's pid-targeted SIGUSR1 (its sigcheck delivered the handler \
         token) and its exit rode back through wait — fork pid, table pid, and kill pid are ONE"
    );
}

/// #796 default actions — **an unhandled fatal signal really terminates, compiled C, end to end**:
/// the child spins forever (no handler for anything — a runaway job); the parent's plain
/// `kill(child, SIGTERM)` must actually kill it (`SIG_DFL` action = terminate through the
/// `SignalSource::set_kill` door → the domain's term flag → `ThreadFault` at the next per-op
/// poll), and the personality's `waitpid` must report the death in the `WIFSIGNALED` shape —
/// the signal in the low 7 bits, NOT an exit-code encode. Without the kill the child's infinite
/// spin would burn the run's whole fuel budget: the test passing quickly IS the death witness.
/// Also probes the discard side: `SIGCHLD` (default-ignore) at the child beforehand is a no-op.
const DEFAULT_TERMINATE_SRC: &str = r#"
long __vm_fs(long op, long a, long b, long c, long d);
static long pid;
static long r;
static int status;
static volatile long sink;
int main(int argc, char **argv) {
  while ((pid = fork()) < 0);
  if (pid == 0) {
    while (1) sink = sink + 1;                      /* runaway: no handler, no exit */
  }
  if (__vm_fs(31, pid, 17, 0, 0) != 0) return 1;    /* SIGCHLD: default-ignore, a no-op */
  if (__vm_fs(31, pid, 15, 0, 0) != 0) return 2;    /* SIGTERM: SIG_DFL = terminate */
  while ((r = __vm_fs(28, pid, (long)&status, 0, 0)) == -10);  /* reap the killed child */
  if (r != pid) return 3;
  if ((status & 0x7f) != 15) return 4;              /* WIFSIGNALED: the terminating signal */
  if (((status >> 8) & 0xff) != 0) return 5;        /* not an exit-code encode */
  return 42;
}
"#;

#[test]
fn an_unhandled_sigterm_kills_a_runaway_forked_child_for_real() {
    let manager = Arc::new(parse_module_raw(FS_FORK_MANAGER).expect("parse fs-fork manager"));
    verify_module(&manager).expect("verify fs-fork manager");
    let guest_src = format!("{FORK_SHIM}\n{DEFAULT_TERMINATE_SRC}");
    let guest = parse_module_raw(&c_to_ir(&guest_src)).expect("parse default-terminate guest");
    verify_module(&guest).expect("verify default-terminate guest");

    let mut host = Host::new();
    host.set_self_module(&manager);
    let _sink = host.shared_stdout();
    let win = 1u64 << 19;
    let stream = host.grant_stream(StreamRole::Out);
    let inst = host.grant_instantiator(0, win);
    let gmod = host.grant_module(&guest);

    let (posix, make) = svm_posix::cap(4096, 1 << 16, Vec::new());
    let make = Arc::new(make);
    let px_handler: svm_interp::HostProc = {
        let mut inner = make();
        Box::new(move |_slot_op, args, mem, minter| inner(args[0] as u32, &args[1..], mem, minter))
    };
    let px_cap = host.grant_host_proc_forkable(
        px_handler,
        opshift_fork(svm_posix::cap_fork_factory(&posix)),
    );

    let mut fuel = 120_000_000u64;
    let r = run_with_host(
        &manager,
        0,
        &[
            Value::I32(inst),
            Value::I32(stream),
            Value::I64(gmod as i64),
            Value::I64(gmod as i64), // the cmd-module slot — unused by this guest
            Value::I32(px_cap),
        ],
        &mut fuel,
        &mut host,
    )
    .expect("run");

    assert_eq!(
        r,
        vec![Value::I64(42)],
        "kill(child, SIGTERM) with no handler installed actually terminated the runaway child \
         (the default action, through the set_kill door), and waitpid reported the death as \
         WIFSIGNALED(15) — not an exit code"
    );
}

/// #799 — **blocking `waitpid`, the API-level witness**: ONE un-looped personality `waitpid`
/// call returns the child's real status. Impossible before this slice — the op was a poll, and a
/// parent reaching it while the child still ran got `-ECHILD` (every guest carried a retry loop).
/// Now the op requests a bench on the child's exit (`ParkEvent::TaskExit` through the park door),
/// the core rewinds+parks this vCPU, the twin-completion wake re-runs the op against the retired
/// table entry, and the single call answers. The child's long spin makes the parent reach the
/// wait first with certainty in practice — on the old semantics this guest returns 1.
const BLOCKING_WAITPID_SRC: &str = r#"
long __vm_fs(long op, long a, long b, long c, long d);
static long pid;
static long h;
static int status;
static volatile long acc;
int main(int argc, char **argv) {
  while ((pid = fork()) < 0);
  if (pid == 0) {
    for (long i = 0; i < 50000; i = i + 1) acc = acc + i;  /* be slow: the parent must WAIT */
    return 7;
  }
  h = __vm_fs(28, pid, (long)&status, 0, 0);   /* ONE call — no retry loop anywhere */
  if (h != pid) return 1;                      /* -10 here = the op is still a poll */
  if ((status & 0x7f) != 0) return 2;          /* a clean exit, not a signal death */
  if (((status >> 8) & 0xff) != 7) return 3;   /* WEXITSTATUS = the child's 7 */
  return 42;
}
"#;

#[test]
fn a_single_waitpid_call_blocks_until_the_child_exits() {
    let manager = Arc::new(parse_module_raw(FS_FORK_MANAGER).expect("parse fs-fork manager"));
    verify_module(&manager).expect("verify fs-fork manager");
    let guest_src = format!("{FORK_SHIM}\n{BLOCKING_WAITPID_SRC}");
    let guest = parse_module_raw(&c_to_ir(&guest_src)).expect("parse blocking-waitpid guest");
    verify_module(&guest).expect("verify blocking-waitpid guest");

    let mut host = Host::new();
    host.set_self_module(&manager);
    let _sink = host.shared_stdout();
    let win = 1u64 << 19;
    let stream = host.grant_stream(StreamRole::Out);
    let inst = host.grant_instantiator(0, win);
    let gmod = host.grant_module(&guest);

    let (posix, make) = svm_posix::cap(4096, 1 << 16, Vec::new());
    let make = Arc::new(make);
    let px_handler: svm_interp::HostProc = {
        let mut inner = make();
        Box::new(move |_slot_op, args, mem, minter| inner(args[0] as u32, &args[1..], mem, minter))
    };
    let px_cap = host.grant_host_proc_forkable(
        px_handler,
        opshift_fork(svm_posix::cap_fork_factory(&posix)),
    );

    let mut fuel = 120_000_000u64;
    let r = run_with_host(
        &manager,
        0,
        &[
            Value::I32(inst),
            Value::I32(stream),
            Value::I64(gmod as i64),
            Value::I64(gmod as i64), // the cmd-module slot — unused by this guest
            Value::I32(px_cap),
        ],
        &mut fuel,
        &mut host,
    )
    .expect("run");

    assert_eq!(
        r,
        vec![Value::I64(42)],
        "one un-looped personality waitpid blocked through the child's slow spin and returned \
         its real WEXITSTATUS — the park door end to end"
    );
}

/// #799 — **a `^C` interrupts a BLOCKED personality `waitpid` with `-EINTR`**: the parent parks
/// in the op-28 bench (not the core wait offer — the other channel the #863 capstone covers);
/// the terminal's SIGINT EINTRs it through the reap-bench sweep arm; after `SIG_IGN` the retried
/// wait blocks cleanly again, the released child exits, and the reap collects it. The signals ×
/// blocking-waitpid composition, on the personality channel.
const BLOCKING_WAITPID_EINTR_SRC: &str = r#"
long __vm_fs(long op, long a, long b, long c, long d);
static char sigstk[4096];
static volatile long got;
static void on_int(int sig) { got = sig; }
static volatile long usr;
static void on_usr(int sig) { usr = sig; }
static long pid;
static long h;
static int status;
int main(int argc, char **argv) {
  __vm_fs(30, 2, (long)on_int, 0, 0);      /* catch SIGINT */
  __vm_fs(30, 10, (long)on_usr, 0, 0);     /* pre-install a REAL handler for 10 (inherited): the
                                              twin runs with async delivery ON (inherited stack),
                                              so an L0 token would be table-masked and dropped at
                                              a safepoint — the #798-capstone hazard */
  __vm_fs(42, (long)sigstk, 4096, 0, 0);   /* sigaltstack: async delivery on */
  __vm_fs(0, 1, (long)&h, 1, 0);           /* readiness byte: the terminal may open fire */
  while ((pid = fork()) < 0);
  if (pid == 0) {
    while (!usr);                          /* spin until the parent's kill delivers async */
    return 5;
  }
  h = __vm_fs(28, pid, (long)&status, 0, 0);   /* BLOCKS in the op-28 bench; ^C -> -EINTR */
  if (h != -4) return 1;                       /* the interrupted wait returned EINTR */
  if (got != 2) return 2;                      /* the SIGINT handler ran */
  __vm_fs(30, 2, 1, 0, 0);                     /* SIG_IGN: the terminal keeps firing harmlessly */
  __vm_fs(31, pid, 10, 0, 0);                  /* release the child */
  while ((h = __vm_fs(28, pid, (long)&status, 0, 0)) == -10);  /* blocks again; reaps for real */
  if (h != pid) return 3;
  if (((status >> 8) & 0xff) != 5) return 4;
  return 42;
}
"#;

#[test]
fn a_ctrl_c_interrupts_a_blocked_personality_waitpid() {
    use std::sync::atomic::{AtomicBool, Ordering};

    let manager = Arc::new(parse_module_raw(FS_FORK_MANAGER).expect("parse fs-fork manager"));
    verify_module(&manager).expect("verify fs-fork manager");
    let guest_src = format!("{FORK_SHIM}\n{BLOCKING_WAITPID_EINTR_SRC}");
    let guest = parse_module_raw(&c_to_ir(&guest_src)).expect("parse waitpid-eintr guest");
    verify_module(&guest).expect("verify waitpid-eintr guest");

    let mut host = Host::new();
    host.set_self_module(&manager);
    let _sink = host.shared_stdout();
    let win = 1u64 << 19;
    let stream = host.grant_stream(StreamRole::Out);
    let inst = host.grant_instantiator(0, win);
    let gmod = host.grant_module(&guest);

    let (posix, make) = svm_posix::cap(4096, 1 << 16, Vec::new());
    let make = Arc::new(make);
    let px_handler: svm_interp::HostProc = {
        let mut inner = make();
        Box::new(move |_slot_op, args, mem, minter| inner(args[0] as u32, &args[1..], mem, minter))
    };
    let px_cap = host.grant_host_proc_forkable(
        px_handler,
        opshift_fork(svm_posix::cap_fork_factory(&posix)),
    );

    let done = Arc::new(AtomicBool::new(false));
    let done2 = Arc::clone(&done);
    let posix2 = posix.clone();
    let terminal = std::thread::spawn(move || {
        // Hold fire until the parent's readiness byte (#796 — a pre-handler ^C is fatal).
        while !done2.load(Ordering::Relaxed) && posix2.stdout().is_empty() {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        while !done2.load(Ordering::Relaxed) {
            posix2.kill_pid(1000, 2);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    });

    let mut fuel = 120_000_000u64;
    let r = run_with_host(
        &manager,
        0,
        &[
            Value::I32(inst),
            Value::I32(stream),
            Value::I64(gmod as i64),
            Value::I64(gmod as i64), // the cmd-module slot — unused by this guest
            Value::I32(px_cap),
        ],
        &mut fuel,
        &mut host,
    );
    done.store(true, Ordering::Relaxed);
    terminal.join().unwrap();
    let r = r.expect("run");

    assert_eq!(
        r,
        vec![Value::I64(42)],
        "the ^C EINTR'd the op-28 bench (handler ran), and after SIG_IGN the retried blocking \
         wait reaped the released child's real 5"
    );
}

/// #800 — **`isatty` + `getppid`, the shell's interactive-mode and `$PPID` probes**: `isatty`
/// answers the same proto-terminal test the `tc*` ops gate on (stdio fds 1, a pipe fd 0 — the
/// discrimination bash's "am I interactive?" check needs), and a fork twin's `getppid()` is the
/// parent's own `getpid()` (recorded at twin mint), with the parent's `getppid()` reporting its
/// granting root. The child returns its checks as an exit code the parent folds into its own.
const ISATTY_PPID_SRC: &str = r#"
long __vm_fs(long op, long a, long b, long c, long d);
static long pid;
static long me;
static long st;
static int fds[2];
int main(int argc, char **argv) {
  me = __vm_fs(44, 0, 0, 0, 0);                       /* getpid */
  if (__vm_fs(49, 1, 0, 0, 0) != 1) return 1;         /* stdout IS the proto-terminal */
  if (__vm_fs(49, 0, 0, 0, 0) != 1) return 2;         /* stdin too */
  if (__vm_fs(23, (long)fds, 0, 0, 0) != 0) return 3; /* pipe(fds) */
  if (__vm_fs(49, fds[0], 0, 0, 0) != 0) return 4;    /* a pipe fd is NOT a tty */
  while ((pid = fork()) < 0);
  if (pid == 0) {
    if (__vm_fs(50, 0, 0, 0, 0) != me) return 9;      /* child's getppid == parent's getpid */
    return 5;
  }
  while ((st = wait_pid(pid)) < 0);
  if (st != 5) return 6;
  if (__vm_fs(50, 0, 0, 0, 0) == me) return 7;        /* parent's ppid is its granter, not itself */
  return 42;
}
"#;

#[test]
fn isatty_discriminates_the_proto_terminal_and_getppid_names_the_forking_parent() {
    let manager = Arc::new(parse_module_raw(FS_FORK_MANAGER).expect("parse fs-fork manager"));
    verify_module(&manager).expect("verify fs-fork manager");
    let guest_src = format!("{FORK_SHIM}\n{ISATTY_PPID_SRC}");
    let guest = parse_module_raw(&c_to_ir(&guest_src)).expect("parse isatty-ppid guest");
    verify_module(&guest).expect("verify isatty-ppid guest");

    let mut host = Host::new();
    host.set_self_module(&manager);
    let _sink = host.shared_stdout();
    let win = 1u64 << 19;
    let stream = host.grant_stream(StreamRole::Out);
    let inst = host.grant_instantiator(0, win);
    let gmod = host.grant_module(&guest);

    let (posix, make) = svm_posix::cap(4096, 1 << 16, Vec::new());
    let make = Arc::new(make);
    let px_handler: svm_interp::HostProc = {
        let mut inner = make();
        Box::new(move |_slot_op, args, mem, minter| inner(args[0] as u32, &args[1..], mem, minter))
    };
    let px_cap = host.grant_host_proc_forkable(
        px_handler,
        opshift_fork(svm_posix::cap_fork_factory(&posix)),
    );

    let mut fuel = 120_000_000u64;
    let r = run_with_host(
        &manager,
        0,
        &[
            Value::I32(inst),
            Value::I32(stream),
            Value::I64(gmod as i64),
            Value::I64(gmod as i64), // the cmd-module slot — unused by this guest
            Value::I32(px_cap),
        ],
        &mut fuel,
        &mut host,
    )
    .expect("run");

    assert_eq!(
        r,
        vec![Value::I64(42)],
        "isatty told the proto-terminal from a pipe fd, and getppid reported the forking \
         parent from inside the twin"
    );
}

/// #796 `SA_RESTART` — **a restart-flagged delivery leaves a parked `wait` parked**: the parent
/// installs SIGUSR1 via `sigaction` with `SA_RESTART` (real handler, async stack) and parks in
/// `wait_pid(child)`; the child raises USR1 at the parent (pid 1) and only THEN exits 5. Without
/// the flag that raise EINTRs the wait (the #863 capstone proves it); with it the wait must ride
/// through — never `-4` — and return the child's real 5, with the handler having run by then
/// (delivered at the parent's first post-wait safepoint; for the shell's `SIGCHLD`-with-restart
/// case the wait completes at the same moment the signal lands, so the delay is invisible).
const RESTART_WAIT_SRC: &str = r#"
long __vm_fs(long op, long a, long b, long c, long d);
static char sigstk[4096];
static long act[3];
static volatile long got;
static void on_usr(int sig) { got = sig; }
static long pid;
static long ppid;
static long st;
static volatile long acc;
int main(int argc, char **argv) {
  act[0] = (long)on_usr;
  act[1] = 0;
  act[2] = 0x10000000;                     /* SA_RESTART */
  __vm_fs(41, 10, (long)act, 0, 0);        /* sigaction(SIGUSR1, restart) — BEFORE fork */
  __vm_fs(42, (long)sigstk, 4096, 0, 0);   /* sigaltstack: async delivery on */
  ppid = __vm_fs(44, 0, 0, 0, 0);          /* getpid, saved pre-fork: the twin's copy = its parent */
  while ((pid = fork()) < 0);
  if (pid == 0) {
    __vm_fs(31, ppid, 10, 0, 0);           /* kill(parent, USR1) while it is (or will be) parked */
    for (long i = 0; i < 200000; i = i + 1) acc = acc + i;  /* a window for a wrong EINTR */
    return 5;
  }
  st = wait_pid(pid);                      /* PARKS; the USR1 must NOT interrupt it (restart) */
  if (st == -4) return 1;                  /* -EINTR = SA_RESTART was ignored */
  if (st != 5) return 2;                   /* the child's real exit rode through */
  for (long i = 0; got != 10 && i < 100000000; i = i + 1) acc = acc + 1;  /* async landing window */
  if (got != 10) return 3;                 /* the held handler delivered after the wait */
  return 42;
}
"#;

#[test]
fn sa_restart_rides_a_parked_wait_through_a_delivered_signal() {
    let manager = Arc::new(parse_module_raw(FS_FORK_MANAGER).expect("parse fs-fork manager"));
    verify_module(&manager).expect("verify fs-fork manager");
    let guest_src = format!("{FORK_SHIM}\n{RESTART_WAIT_SRC}");
    let guest = parse_module_raw(&c_to_ir(&guest_src)).expect("parse restart-wait guest");
    verify_module(&guest).expect("verify restart-wait guest");

    let mut host = Host::new();
    host.set_self_module(&manager);
    let _sink = host.shared_stdout();
    let win = 1u64 << 19;
    let stream = host.grant_stream(StreamRole::Out);
    let inst = host.grant_instantiator(0, win);
    let gmod = host.grant_module(&guest);

    let (posix, make) = svm_posix::cap(4096, 1 << 16, Vec::new());
    let make = Arc::new(make);
    let px_handler: svm_interp::HostProc = {
        let mut inner = make();
        Box::new(move |_slot_op, args, mem, minter| inner(args[0] as u32, &args[1..], mem, minter))
    };
    let px_cap = host.grant_host_proc_forkable(
        px_handler,
        opshift_fork(svm_posix::cap_fork_factory(&posix)),
    );

    let mut fuel = 120_000_000u64;
    let r = run_with_host(
        &manager,
        0,
        &[
            Value::I32(inst),
            Value::I32(stream),
            Value::I64(gmod as i64),
            Value::I64(gmod as i64), // the cmd-module slot — unused by this guest
            Value::I32(px_cap),
        ],
        &mut fuel,
        &mut host,
    )
    .expect("run");

    assert_eq!(
        r,
        vec![Value::I64(42)],
        "the SA_RESTART'd USR1 left the parent's wait parked (no -EINTR): it returned the \
         child's real 5, and the handler had run by then"
    );
}

/// #863 slice 3 — **the waitpid-EINTR capstone: an embedder `^C` interrupts a forked shell blocked
/// in `wait`, on the RIGHT process.** The compiled-C parent catches SIGINT (real handler + signal
/// stack — async delivery on), forks a child (which spins on its `sigcheck` doorbell), and parks in
/// `wait(child)` — the core reap park (`join_waiters`). A background "terminal" thread calls
/// [`svm_posix::Posix::kill_pid`]`(1000, SIGINT)` — the parent's personality pid (the first
/// anonymous re-grant clone draws 1000 from the spawn allocator) — so the personality sets the
/// PARENT's pending bit and fires the parent's **domain-scoped weak run-wake**: the reap park
/// completes `-EINTR` while the child's spin is untouched (invariant 12 — a signal to A never
/// sweeps B). The parent then observes the handler ran, ignores further SIGINTs (`SIG_IGN` — the
/// terminal thread keeps firing; an undeliverable raise must never wake), releases the child with
/// `kill(child_pid, 10)`, and reaps it for real: the classic interactive-shell loop —
/// `wait → ^C → EINTR → handle → wait again` — end to end in ordinary C.
const WAIT_EINTR_SRC: &str = r#"
long __vm_fs(long op, long a, long b, long c, long d);
static char sigstk[4096];  /* small enough to keep the module inside the 2^17 spawn carve */
static volatile long got;
static void on_int(int sig) { got = sig; }
static volatile long usr;
static void on_usr(int sig) { usr = sig; }
static long pid;
static long st;
int main(int argc, char **argv) {
  __vm_fs(30, 2, (long)on_int, 0, 0);      /* signal(SIGINT, handler): catch */
  __vm_fs(30, 10, (long)on_usr, 0, 0);     /* pre-install 10 too — inherited by the twin, so the
                                              parent's kill(child, 10) can never land on SIG_DFL
                                              (#796: that would terminate it, not pend) */
  __vm_fs(42, (long)sigstk, 4096, 0, 0);   /* sigaltstack: async delivery on */
  __vm_fs(0, 1, (long)&st, 1, 0);          /* readiness byte: the terminal may open fire (#796 —
                                              a pre-handler ^C is now FATAL, the default action) */
  while ((pid = fork()) < 0);
  if (pid == 0) {
    /* The fork twin INHERITS the signal stack (POSIX) — async delivery is ON here too, so its
     * handler must be real (an L0 token would be table-masked and dispatched). Its own async
     * SIGUSR1 breaks the spin: the child-side half of the capstone. */
    __vm_fs(30, 10, (long)on_usr, 0, 0);   /* child: signal(10, handler) */
    while (!usr);                          /* spin until the parent's kill delivers async */
    return 5;
  }
  st = wait_pid(pid);                      /* PARKS in the core reap; the terminal ^C -> -EINTR */
  if (st != -4) return 1;                  /* the interrupted wait returned EINTR */
  if (got != 2) return 2;                  /* the SIGINT handler ran on the PARENT */
  __vm_fs(30, 2, 1, 0, 0);                 /* SIG_IGN: the terminal keeps firing; no more wakes */
  __vm_fs(31, pid, 10, 0, 0);              /* kill(child, 10): release the spinner */
  while ((st = wait_pid(pid)) < 0);        /* wait again — reap for real (retries a raced EINTR) */
  return st;                               /* 5: the child's exit, after the ^C round-trip */
}
"#;

#[test]
fn a_terminal_ctrl_c_interrupts_a_forked_parent_blocked_in_wait() {
    use std::sync::atomic::{AtomicBool, Ordering};

    let manager = Arc::new(parse_module_raw(FS_FORK_MANAGER).expect("parse fs-fork manager"));
    verify_module(&manager).expect("verify fs-fork manager");
    let guest_src = format!("{FORK_SHIM}\n{WAIT_EINTR_SRC}");
    let guest = parse_module_raw(&c_to_ir(&guest_src)).expect("parse wait-eintr guest");
    verify_module(&guest).expect("verify wait-eintr guest");

    let mut host = Host::new();
    host.set_self_module(&manager);
    let _sink = host.shared_stdout();
    let win = 1u64 << 19;
    let stream = host.grant_stream(StreamRole::Out);
    let inst = host.grant_instantiator(0, win);
    let gmod = host.grant_module(&guest);

    let (posix, make) = svm_posix::cap(4096, 1 << 16, Vec::new());
    let make = Arc::new(make);
    let px_handler: svm_interp::HostProc = {
        let mut inner = make();
        Box::new(move |_slot_op, args, mem, minter| inner(args[0] as u32, &args[1..], mem, minter))
    };
    let px_cap = host.grant_host_proc_forkable(
        px_handler,
        opshift_fork(svm_posix::cap_fork_factory(&posix)),
    );

    // The "terminal": raise SIGINT at the parent (personality pid 1000) until the run returns.
    // It holds fire until the guest's readiness byte lands on the captured stdout — #796 default
    // actions made a pre-handler `^C` FATAL (terminate is the `SIG_DFL` action), so the storm may
    // only start once the handler is installed. After that it stays robust to ordering: raises
    // are caught (pending bit + handler), and after the guest's SIG_IGN they are discarded.
    // Domain scoping keeps every one of these away from the child's spin.
    let done = Arc::new(AtomicBool::new(false));
    let done2 = Arc::clone(&done);
    let posix2 = posix.clone();
    let terminal = std::thread::spawn(move || {
        while !done2.load(Ordering::Relaxed) && posix2.stdout().is_empty() {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        while !done2.load(Ordering::Relaxed) {
            posix2.kill_pid(1000, 2);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    });

    let mut fuel = 120_000_000u64;
    let r = run_with_host(
        &manager,
        0,
        &[
            Value::I32(inst),
            Value::I32(stream),
            Value::I64(gmod as i64),
            Value::I64(gmod as i64), // the cmd-module slot — unused by this guest
            Value::I32(px_cap),
        ],
        &mut fuel,
        &mut host,
    );
    done.store(true, Ordering::Relaxed);
    terminal.join().unwrap();
    let r = r.expect("run");

    assert_eq!(
        r,
        vec![Value::I64(5)],
        "the ^C EINTR'd the parent's wait (handler ran, child's park untouched), and the \
         re-issued wait reaped the child's real exit — the interactive shell loop, end to end"
    );
}

/// #863 slice 3 — **the invariant-12 scoping witness: a `^C` at the parent never sweeps the
/// child's park.** Three generations: the parent waits on the child, the child waits on a
/// grandchild (which spins on its doorbell), so TWO processes of one world are parked in `wait`
/// simultaneously when the terminal `^C` hits the PARENT. Only the parent's wait may return
/// `-EINTR`: the child's park must survive untouched (`gst != 5 → 1` would surface a stray EINTR —
/// exactly what the pre-slice-3 run-global `interrupt_interruptible_parks` sweep did). The parent
/// then releases the leaf **across two generations** — `kill(4, 12)` straight to the grandchild's
/// own async handler — and the exits cascade back: grandchild 5 → child 6 → parent. Also
/// exercises fork-of-fork (the twin's replacement factory) and the grandchild's wake install at
/// its own mint.
const WAIT_SCOPED_SRC: &str = r#"
long __vm_fs(long op, long a, long b, long c, long d);
static char sigstk[4096];
static volatile long got;
static void on_int(int sig) { got = sig; }
static volatile long usr;
static void on_usr(int sig) { usr = sig; }
static long pid;
static long st;
int main(int argc, char **argv) {
  __vm_fs(30, 2, (long)on_int, 0, 0);      /* signal(SIGINT, handler): catch */
  __vm_fs(30, 12, (long)on_usr, 0, 0);     /* pre-install 12 — inherited down BOTH generations, so
                                              the cross-generation kill(4, 12) can never land on
                                              SIG_DFL in a grandchild that has not re-installed */
  __vm_fs(42, (long)sigstk, 4096, 0, 0);   /* sigaltstack: async delivery on (inherited by all) */
  __vm_fs(0, 1, (long)&st, 1, 0);          /* readiness byte: the terminal may open fire (#796 —
                                              a pre-handler ^C is now FATAL, the default action) */
  while ((pid = fork()) < 0);              /* child: pid 3 (task ids are deterministic here) */
  if (pid == 0) {
    long gpid;
    long gst;
    while ((gpid = fork()) < 0);           /* fork-of-fork: grandchild, pid 4 */
    if (gpid == 0) {
      __vm_fs(30, 12, (long)on_usr, 0, 0); /* grandchild: catch signal 12 */
      while (!usr);                        /* spin until the PARENT's cross-generation kill */
      return 5;
    }
    gst = wait_pid(gpid);                  /* PARKS — the parent's ^C must NOT touch this */
    if (gst != 5) return 1;                /* a stray -EINTR here = the unscoped-sweep bug */
    return 6;
  }
  st = wait_pid(pid);                      /* PARKS; the terminal ^C -> -EINTR, parent only */
  if (st != -4) return 2;
  if (got != 2) return 3;                  /* the SIGINT handler ran on the parent */
  __vm_fs(30, 2, 1, 0, 0);                 /* SIG_IGN: the terminal keeps firing harmlessly */
  __vm_fs(31, 4, 12, 0, 0);                /* kill(grandchild, 12): two generations down */
  while ((st = wait_pid(pid)) < 0);        /* reap the CHILD (retries a raced EINTR) */
  return st;                               /* 6: the child reaped 5 cleanly, unswept */
}
"#;

#[test]
fn a_ctrl_c_at_the_parent_never_sweeps_the_childs_wait_park() {
    use std::sync::atomic::{AtomicBool, Ordering};

    let manager = Arc::new(parse_module_raw(FS_FORK_MANAGER).expect("parse fs-fork manager"));
    verify_module(&manager).expect("verify fs-fork manager");
    let guest_src = format!("{FORK_SHIM}\n{WAIT_SCOPED_SRC}");
    let guest = parse_module_raw(&c_to_ir(&guest_src)).expect("parse scoped-wait guest");
    verify_module(&guest).expect("verify scoped-wait guest");

    let mut host = Host::new();
    host.set_self_module(&manager);
    let _sink = host.shared_stdout();
    let win = 1u64 << 19;
    let stream = host.grant_stream(StreamRole::Out);
    let inst = host.grant_instantiator(0, win);
    let gmod = host.grant_module(&guest);

    let (posix, make) = svm_posix::cap(4096, 1 << 16, Vec::new());
    let make = Arc::new(make);
    let px_handler: svm_interp::HostProc = {
        let mut inner = make();
        Box::new(move |_slot_op, args, mem, minter| inner(args[0] as u32, &args[1..], mem, minter))
    };
    let px_cap = host.grant_host_proc_forkable(
        px_handler,
        opshift_fork(svm_posix::cap_fork_factory(&posix)),
    );

    let done = Arc::new(AtomicBool::new(false));
    let done2 = Arc::clone(&done);
    let posix2 = posix.clone();
    let terminal = std::thread::spawn(move || {
        // Hold fire until the parent's readiness byte (#796 — a pre-handler ^C is fatal).
        while !done2.load(Ordering::Relaxed) && posix2.stdout().is_empty() {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        while !done2.load(Ordering::Relaxed) {
            posix2.kill_pid(1000, 2);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    });

    let mut fuel = 120_000_000u64;
    let r = run_with_host(
        &manager,
        0,
        &[
            Value::I32(inst),
            Value::I32(stream),
            Value::I64(gmod as i64),
            Value::I64(gmod as i64), // the cmd-module slot — unused by this guest
            Value::I32(px_cap),
        ],
        &mut fuel,
        &mut host,
    );
    done.store(true, Ordering::Relaxed);
    terminal.join().unwrap();
    let r = r.expect("run");

    assert_eq!(
        r,
        vec![Value::I64(6)],
        "the parent's ^C EINTR'd only ITS wait; the child's park survived (6 = the child \
         reaped the grandchild's clean 5) — the domain-scoped sweep, witnessed across three \
         generations"
    );
}

/// #863 hygiene — **`waitpid` over a fork twin, through the personality**: the parent reaps its
/// forked child with the POSIX `waitpid` op (a non-blocking `-ECHILD` poll), never touching the
/// core's wait offer. The chain under test: the twin's task completes → the core fires the fork
/// factory's exit hook → the personality flips the twin `Live` → `Zombie` (wait-encoded) → the
/// parent's poll stops seeing `-ECHILD` and reaps the status. One pid, two reap channels — this
/// pins the personality one; the wait-offer one is every other test in this file.
const WAITPID_TWIN_SRC: &str = r#"
long __vm_fs(long op, long a, long b, long c, long d);
static long pid;
static long r;
static int status;
int main(int argc, char **argv) {
  while ((pid = fork()) < 0);
  if (pid == 0) {
    return 7;
  }
  while ((r = __vm_fs(28, pid, (long)&status, 0, 0)) == -10);  /* waitpid poll: ECHILD until exit */
  if (r != pid) return 1;
  if (((status >> 8) & 0xff) != 7) return 2;                   /* WEXITSTATUS */
  if (__vm_fs(31, pid, 15, 0, 0) != -3) return 3;              /* reaped: kill(pid) is -ESRCH */
  return 42;
}
"#;

#[test]
fn a_compiled_c_parent_reaps_its_fork_twin_through_posix_waitpid() {
    let manager = Arc::new(parse_module_raw(FS_FORK_MANAGER).expect("parse fs-fork manager"));
    verify_module(&manager).expect("verify fs-fork manager");
    let guest_src = format!("{FORK_SHIM}\n{WAITPID_TWIN_SRC}");
    let guest = parse_module_raw(&c_to_ir(&guest_src)).expect("parse waitpid-twin guest");
    verify_module(&guest).expect("verify waitpid-twin guest");

    let mut host = Host::new();
    host.set_self_module(&manager);
    let _sink = host.shared_stdout();
    let win = 1u64 << 19;
    let stream = host.grant_stream(StreamRole::Out);
    let inst = host.grant_instantiator(0, win);
    let gmod = host.grant_module(&guest);

    let (posix, make) = svm_posix::cap(4096, 1 << 16, Vec::new());
    let make = Arc::new(make);
    let px_handler: svm_interp::HostProc = {
        let mut inner = make();
        Box::new(move |_slot_op, args, mem, minter| inner(args[0] as u32, &args[1..], mem, minter))
    };
    let px_cap = host.grant_host_proc_forkable(
        px_handler,
        opshift_fork(svm_posix::cap_fork_factory(&posix)),
    );

    let mut fuel = 120_000_000u64;
    let r = run_with_host(
        &manager,
        0,
        &[
            Value::I32(inst),
            Value::I32(stream),
            Value::I64(gmod as i64),
            Value::I64(gmod as i64), // the cmd-module slot — unused by this guest
            Value::I32(px_cap),
        ],
        &mut fuel,
        &mut host,
    )
    .expect("run");

    assert_eq!(
        r,
        vec![Value::I64(42)],
        "the exit hook flipped the twin to a zombie: posix waitpid reaped it (WEXITSTATUS 7) \
         and the pid is gone afterwards"
    );
}

/// #798 slice 1 — **the shell's job-control loop, compiled C, fully in-guest and deterministic**:
/// fork a two-process "pipeline job", `setpgid` both members into one group led by the first,
/// `tcsetpgrp` the job to the foreground (readback via `tcgetpgrp`), observe the now-background
/// shell's own stdout write ring its `SIGTTOU` doorbell (the write proceeds — the L0
/// approximation until slice 2's stop), **Ctrl-C the job with one `kill(-pgid)`** (both members'
/// doorbells ring, each exits with its own status), take the terminal back, and reap both members
/// through `waitpid`. Everything is L0 polling — no signal stack anywhere, so nothing dispatches
/// async and no embedder thread is needed; the run is single-threaded deterministic.
const JOB_CONTROL_SRC: &str = r#"
long __vm_fs(long op, long a, long b, long c, long d);
static long p1;
static long p2;
static long h;
static long msg;
static int status;
int main(int argc, char **argv) {
  /* Catch 10 BEFORE forking — the disposition is inherited (POSIX), so the group ^C below can
   * never find SIG_DFL in a member that has not yet re-installed its own token. #796 made that
   * matter: an uncaught signal 10 now runs its default action (terminate) instead of pending. */
  __vm_fs(30, 10, 6, 0, 0);
  while ((p1 = fork()) < 0);
  if (p1 == 0) {
    __vm_fs(30, 10, 7, 0, 0);                 /* member 1: re-catch with its own L0 token */
    while (__vm_fs(32, 0, 0, 0, 0) != 7);     /* spin on the doorbell */
    return 5;
  }
  while ((p2 = fork()) < 0);
  if (p2 == 0) {
    __vm_fs(30, 10, 8, 0, 0);                 /* member 2: its own token */
    while (__vm_fs(32, 0, 0, 0, 0) != 8);
    return 6;
  }
  /* Build the job: both children into a group led by the first; foreground it. */
  if (__vm_fs(45, p1, p1, 0, 0) != 0) return 1;   /* setpgid(c1, c1): group leader */
  if (__vm_fs(45, p2, p1, 0, 0) != 0) return 2;   /* setpgid(c2, c1): join the job */
  if (__vm_fs(46, p2, 0, 0, 0) != p1) return 3;   /* getpgid(c2) == the job group */
  if (__vm_fs(48, 1, p1, 0, 0) != 0) return 4;    /* tcsetpgrp(stdout, job) */
  if (__vm_fs(47, 1, 0, 0, 0) != p1) return 7;    /* tcgetpgrp readback */
  /* The shell is background now: its own stdout write rings SIGTTOU (and proceeds). */
  __vm_fs(30, 22, 9, 0, 0);                       /* catch SIGTTOU (token 9) */
  msg = 0x0a24;                                   /* "$\n" */
  if (__vm_fs(0, 1, (long)&msg, 2, 0) != 2) return 8;
  if (__vm_fs(32, 0, 0, 0, 0) != 9) return 9;     /* the TTOU doorbell rang */
  /* ^C the job: ONE group kill reaches both members. */
  if (__vm_fs(31, -p1, 10, 0, 0) != 0) return 10;
  /* Take the terminal back (own group, occupied by us) and reap the job. */
  if (__vm_fs(48, 1, 1, 0, 0) != 0) return 11;
  while ((h = __vm_fs(28, p1, (long)&status, 0, 0)) == -10);
  if (h != p1 || ((status >> 8) & 0xff) != 5) return 12;
  while ((h = __vm_fs(28, p2, (long)&status, 0, 0)) == -10);
  if (h != p2 || ((status >> 8) & 0xff) != 6) return 13;
  return 42;
}
"#;

#[test]
fn a_compiled_c_shell_runs_the_job_control_loop() {
    let manager = Arc::new(parse_module_raw(FS_FORK_MANAGER).expect("parse fs-fork manager"));
    verify_module(&manager).expect("verify fs-fork manager");
    let guest_src = format!("{FORK_SHIM}\n{JOB_CONTROL_SRC}");
    let guest = parse_module_raw(&c_to_ir(&guest_src)).expect("parse job-control guest");
    verify_module(&guest).expect("verify job-control guest");

    let mut host = Host::new();
    host.set_self_module(&manager);
    let _sink = host.shared_stdout();
    let win = 1u64 << 19;
    let stream = host.grant_stream(StreamRole::Out);
    let inst = host.grant_instantiator(0, win);
    let gmod = host.grant_module(&guest);

    let (posix, make) = svm_posix::cap(4096, 1 << 16, Vec::new());
    let make = Arc::new(make);
    let px_handler: svm_interp::HostProc = {
        let mut inner = make();
        Box::new(move |_slot_op, args, mem, minter| inner(args[0] as u32, &args[1..], mem, minter))
    };
    let px_cap = host.grant_host_proc_forkable(
        px_handler,
        opshift_fork(svm_posix::cap_fork_factory(&posix)),
    );

    let mut fuel = 120_000_000u64;
    let r = run_with_host(
        &manager,
        0,
        &[
            Value::I32(inst),
            Value::I32(stream),
            Value::I64(gmod as i64),
            Value::I64(gmod as i64), // the cmd-module slot — unused by this guest
            Value::I32(px_cap),
        ],
        &mut fuel,
        &mut host,
    )
    .expect("run");

    assert_eq!(
        r,
        vec![Value::I64(42)],
        "the job-control loop end to end: setpgid the job, tcsetpgrp foreground, the background \
         shell's TTOU doorbell, kill(-pgid) reaching BOTH members, terminal back, both reaped"
    );
    // The backgrounded shell's write proceeded (the L0 doorbell, not a stop): it reached stdout.
    assert_eq!(
        posix.stdout(),
        b"$\n",
        "the background write went through to the captured terminal"
    );
}

/// #798 slice 2 — **Ctrl-Z / `fg` in compiled C: the stop actually stops the domain.** The parent
/// forks a child that spins on its `sigcheck` doorbell, then:
///
/// 1. `kill(child, SIGTSTP)` — default disposition ⇒ the child's domain STOPS (the core parks it
///    at its next per-op poll).
/// 2. `waitpid(-1, WUNTRACED)` reports the stop (`SIGTSTP<<8 | 0x7f`) — and only once.
/// 3. `kill(child, 10)` — delivered **while stopped**, so it must be HELD: the child, were it
///    secretly still running, would consume it and exit. A long busy-wait gives the cooperative
///    scheduler every chance to run it; `waitpid(child, 0)` still `-ECHILD` is the proof of
///    stopped-ness (the interp would happily have scheduled a runnable child during the wait).
/// 4. `kill(child, SIGCONT)` — the domain resumes, the HELD signal delivers, the child exits 5;
///    `waitpid(WCONTINUED)` reports the continue (`0xffff`), then the reap collects the 5.
///
/// Fully in-guest and deterministic (L0 polling, no signal stacks, no embedder thread).
const CTRL_Z_SRC: &str = r#"
long __vm_fs(long op, long a, long b, long c, long d);
static long pid;
static long r;
static int status;
static long i;
static volatile long sink;
int main(int argc, char **argv) {
  /* Pre-install a caught disposition for 10 — inherited by the twin, so the held-while-stopped
   * 10 below can never sit as SIG_DFL (#796: the SIGCONT would then run the default action —
   * terminate — instead of delivering it). The child re-installs its own token before reading. */
  __vm_fs(30, 10, 6, 0, 0);
  while ((pid = fork()) < 0);
  if (pid == 0) {
    __vm_fs(30, 10, 7, 0, 0);                  /* catch 10 (its own L0 token) */
    while (__vm_fs(32, 0, 0, 0, 0) != 7);      /* spin until it delivers */
    return 5;
  }
  if (__vm_fs(31, pid, 20, 0, 0) != 0) return 1;        /* kill(child, SIGTSTP): stop */
  r = __vm_fs(28, -1, (long)&status, 2, 0);             /* waitpid(-1, WUNTRACED) */
  if (r != pid) return 2;
  if ((status & 0xff) != 0x7f) return 3;                /* stopped marker */
  if (((status >> 8) & 0xff) != 20) return 4;           /* by SIGTSTP */
  if (__vm_fs(28, -1, (long)&status, 2, 0) != -10) return 6;  /* report-once */
  if (__vm_fs(31, pid, 10, 0, 0) != 0) return 7;        /* the 10 lands while stopped: HELD */
  for (i = 0; i < 200000; i = i + 1) sink = i;          /* every chance to run, were it runnable */
  if (__vm_fs(28, pid, (long)&status, 1, 0) != -10) return 8;  /* WNOHANG probe (#799 — a plain
                                              waitpid now BLOCKS, and a stopped child without
                                              WUNTRACED would park this probe forever): still
                                              alive, truly stopped */
  if (__vm_fs(31, pid, 18, 0, 0) != 0) return 9;        /* kill(child, SIGCONT): resume */
  while ((r = __vm_fs(28, pid, (long)&status, 8, 0)) == -10);  /* waitpid(pid, WCONTINUED) */
  if (r == pid && status == 0xffff) {
    while ((r = __vm_fs(28, pid, (long)&status, 0, 0)) == -10);  /* now the real reap */
  }
  if (r != pid) return 10;
  if (((status >> 8) & 0xff) != 5) return 11;           /* the held 10 delivered post-continue */
  return 42;
}
"#;

#[test]
fn ctrl_z_stops_a_forked_child_and_fg_resumes_it() {
    let manager = Arc::new(parse_module_raw(FS_FORK_MANAGER).expect("parse fs-fork manager"));
    verify_module(&manager).expect("verify fs-fork manager");
    let guest_src = format!("{FORK_SHIM}\n{CTRL_Z_SRC}");
    let guest = parse_module_raw(&c_to_ir(&guest_src)).expect("parse ctrl-z guest");
    verify_module(&guest).expect("verify ctrl-z guest");

    let mut host = Host::new();
    host.set_self_module(&manager);
    let _sink = host.shared_stdout();
    let win = 1u64 << 19;
    let stream = host.grant_stream(StreamRole::Out);
    let inst = host.grant_instantiator(0, win);
    let gmod = host.grant_module(&guest);

    let (posix, make) = svm_posix::cap(4096, 1 << 16, Vec::new());
    let make = Arc::new(make);
    let px_handler: svm_interp::HostProc = {
        let mut inner = make();
        Box::new(move |_slot_op, args, mem, minter| inner(args[0] as u32, &args[1..], mem, minter))
    };
    let px_cap = host.grant_host_proc_forkable(
        px_handler,
        opshift_fork(svm_posix::cap_fork_factory(&posix)),
    );

    let mut fuel = 220_000_000u64;
    let r = run_with_host(
        &manager,
        0,
        &[
            Value::I32(inst),
            Value::I32(stream),
            Value::I64(gmod as i64),
            Value::I64(gmod as i64), // the cmd-module slot — unused by this guest
            Value::I32(px_cap),
        ],
        &mut fuel,
        &mut host,
    )
    .expect("run");

    assert_eq!(
        r,
        vec![Value::I64(42)],
        "Ctrl-Z: the child STOPPED (a signal sent while stopped was held through a long busy-wait \
         — a runnable child would have consumed it and exited), WUNTRACED reported it once, \
         SIGCONT resumed it, WCONTINUED reported, and the held signal delivered post-continue"
    );
}
