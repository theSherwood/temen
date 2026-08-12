//! Stage 1 slice 2.5 (STAGE1.md) — the POSIX **spawn/wait** surface driven end to end by compiled C,
//! with a *real* instantiated child. A compiled-C shell calls `spawn("up")` + `waitpid` (the new
//! personality ops 27/28, reached by name through the same `__px_` shim as every other libc call); the
//! embedder wires the personality's spawn delegate ([`svm_posix::Posix::set_spawn`]) to **instantiate and
//! run a separate compiled-C command module** on its own Posix personality, capturing its stdout + exit
//! status. Because a spawn is authority the libc personality does not itself hold, the delegate is where
//! the embedder grants it — here by running a child domain (the domain-exec shape), not a Rust stub.
//!
//! This closes the loop the host-level `spawn_waitpid_over_the_delegate` unit left open: there the
//! delegate was a Rust closure; here it runs an actual second guest. It proves **fd inheritance across a
//! real spawn** — the child's `write(1, …)` follows the shell's fd 1 (the shell's stdout sink) — and
//! **status propagation** — the child's `main` return reaches the shell's `waitpid`. Differential
//! interp==JIT on the shell; the child runs on the reference interpreter either way (its backend is
//! independent, deterministic under fuel — the same choice `domain_exec` makes).
//!
//! Gated `#![cfg(unix)]` (needs the chibicc toolchain).
#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, OnceLock};

use core::ffi::c_void;
use svm_interp::{run_with_host, Host, Trap, Value};
use svm_jit::{compile_and_run_with_host, JitOutcome};
use svm_posix::SpawnResult;
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
    let base = std::env::temp_dir().join(format!("svm_cpxspawn_{}_{id}", std::process::id()));
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

/// Bind the shim's `__px_`-prefixed import slots to the granted personality `handle`
/// ([`svm_posix::resolve`] → `(HOST_PROC, op)`), phase-4 no-rewrite. Same helper as `c_posix.rs`.
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

/// The guest libc shim — the `c_posix.rs` set plus `spawn`/`waitpid` wrappers over the new ops. Each
/// `__px_` extern's first arg is a dummy `0` handle operand (the slot binding carries the real handle).
const SHIM: &str = r#"
long __px_write(int cap, long fd, long buf, long len);
long __px_read(int cap, long fd, long buf, long len);
long __px_spawn(int cap, long name, long nlen, long argv, long alen);
long __px_waitpid(int cap, long pid, long status, long opts);
void __px_exit(int cap, int code);

static long slen(char *s) { long n = 0; while (s[n]) n = n + 1; return n; }

long write(long fd, void *buf, long n) { return __px_write(0, fd, (long)buf, n); }
long read(long fd, void *buf, long n) { return __px_read(0, fd, (long)buf, n); }
long spawn(char *name) { return __px_spawn(0, (long)name, slen(name), 0, 0); }
long waitpid(long pid, int *status, int opts) { return __px_waitpid(0, pid, (long)status, opts); }
void exit(int code) { __px_exit(0, code); }
"#;

/// The command `up`: an uppercasing stdin→stdout filter, returning `42` (a distinctive exit code the
/// shell's `waitpid` must see). A `static` buffer avoids leaning on stack-array codegen.
fn child_src() -> String {
    format!(
        "{SHIM}\n\
static char buf[256];\n\
int main() {{\n\
  long n;\n\
  while ((n = read(0, buf, 256)) > 0) {{\n\
    for (long i = 0; i < n; i = i + 1) {{ char c = buf[i]; if (c >= 'a' && c <= 'z') buf[i] = c - 32; }}\n\
    write(1, buf, n);\n\
  }}\n\
  return 42;\n\
}}\n"
    )
}

/// The shell: write a `">>"` prefix, `spawn("up")` (which inherits the shell's fd 0 as the child's stdin
/// and routes the child's stdout to the shell's fd 1), `waitpid`, and return `WEXITSTATUS`.
fn shell_src() -> String {
    format!(
        "{SHIM}\n\
int main() {{\n\
  write(1, \">>\", 2);\n\
  long pid = spawn(\"up\");\n\
  int st = 0;\n\
  waitpid(pid, &st, 0);\n\
  return (st >> 8) & 255;\n\
}}\n"
    )
}

/// Build a spawn delegate that runs the compiled `up` command as a real child domain: grant a fresh
/// Posix on a fresh host (seeding the inherited `stdin`), bind the child's libc shim, run `_start` on the
/// interpreter, and return the child's captured stdout + exit status. A name that is not `"up"` is a
/// `127` "not found" (empty output) — the registry *is* the attenuation.
fn up_delegate(
    child: Arc<svm_ir::Module>,
) -> impl FnMut(&str, &[String], &[u8]) -> SpawnResult + Send {
    move |name: &str, _argv: &[String], stdin: &[u8]| {
        if name != "up" {
            return SpawnResult {
                stdout: Vec::new(),
                status: 127,
                ..Default::default()
            };
        }
        let win = 1u64 << child.memory.expect("child window").size_log2;
        let mut ch = Host::new();
        let (px, cposix) = svm_posix::grant(&mut ch, win / 2, win, stdin.to_vec());
        bind_shim(&child, &mut ch, px);
        let mut fuel = 50_000_000u64;
        let status = match run_with_host(&child, 0, &[], &mut fuel, &mut ch) {
            Ok(v) => match v.first() {
                Some(Value::I32(x)) => *x,
                Some(Value::I64(x)) => *x as i32,
                _ => 0,
            },
            Err(Trap::Exit(c)) => c,
            Err(e) => panic!("child trapped: {e:?}"),
        };
        SpawnResult {
            stdout: cposix.stdout(),
            stderr: cposix.stderr(),
            status,
        }
    }
}

/// Grant the personality (window-heap in the upper half, clear of the low data image), preload the
/// shell's `stdin`, wire the spawn delegate against the compiled child, bind the shell's shim, and run
/// `_start`. Returns (result-or-exit-code, captured stdout).
fn run_shell(
    shell: &svm_ir::Module,
    child: &Arc<svm_ir::Module>,
    stdin: &[u8],
    jit: bool,
) -> (i64, Vec<u8>) {
    let win = 1u64 << shell.memory.expect("shell window").size_log2;
    let mut host = Host::new();
    let (px, posix) = svm_posix::grant(&mut host, win / 2, win, stdin.to_vec());
    posix.set_spawn(up_delegate(Arc::clone(child)));
    bind_shim(shell, &mut host, px);

    if jit {
        let jo = compile_and_run_with_host(
            shell,
            0,
            &[],
            cap_thunk,
            &mut host as *mut Host as *mut c_void,
        )
        .expect("jit compiles");
        let code = match jo {
            JitOutcome::Returned(ref s) => s.first().copied().unwrap_or(0),
            JitOutcome::Exited(c) => c as i64,
            ref o => panic!("jit ended abnormally: {o:?}"),
        };
        (code, posix.stdout())
    } else {
        let mut fuel = 100_000_000u64;
        let code = match run_with_host(shell, 0, &[], &mut fuel, &mut host) {
            Ok(ref v) => match v.first() {
                Some(Value::I32(x)) => *x as i64,
                Some(Value::I64(x)) => *x,
                _ => 0,
            },
            Err(Trap::Exit(c)) => c as i64,
            Err(e) => panic!("interp trapped: {e:?}"),
        };
        (code, posix.stdout())
    }
}

/// The compiled shell spawns the compiled `up` filter through the POSIX `spawn`/`waitpid` ops: the
/// child inherits the shell's stdin (`"hello"`), uppercases it, and its `"HELLO"` follows fd 1 into the
/// shell's stdout right after the shell's own `">>"` prefix; `waitpid` carries the child's `42` back as
/// the shell's exit status. Identical on both backends.
#[test]
fn compiled_shell_spawns_compiled_child_via_posix_spawn_wait() {
    let shell = parse_module_raw(&c_to_ir(&shell_src())).expect("parse shell");
    verify_module(&shell).expect("verify shell");
    let child = Arc::new(parse_module_raw(&c_to_ir(&child_src())).expect("parse child"));
    verify_module(&child).expect("verify child");

    let (ic, iout) = run_shell(&shell, &child, b"hello", false);
    let (jc, jout) = run_shell(&shell, &child, b"hello", true);

    assert_eq!(
        iout, b">>HELLO",
        "interp: shell prefix then the child's inherited-stdin output"
    );
    assert_eq!(ic, 42, "interp: waitpid propagated the child's exit status");
    assert_eq!(jout, iout, "jit: spawn output must match interp");
    assert_eq!(jc, ic, "jit: waitpid status must match interp");
}
