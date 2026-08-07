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
