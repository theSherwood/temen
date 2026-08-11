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
        std::sync::Arc::new(move || -> svm_interp::HostProc {
            let mut inner = factory();
            Box::new(move |_slot_op, args, mem, minter| {
                inner(args[0] as u32, &args[1..], mem, minter)
            })
        })
    };
    let fs_cap = host.grant_host_proc_forkable(make(), make.clone());

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
        std::sync::Arc::new(move || -> svm_interp::HostProc {
            let mut inner = factory();
            Box::new(move |_slot_op, args, mem, minter| {
                inner(args[0] as u32, &args[1..], mem, minter)
            })
        })
    };
    let fs_cap = host.grant_host_proc_forkable(make(), make.clone());

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
