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

/// The guest: `fork()` (through the `__fork` import → the fork offer), then write the 8-byte fork return
/// to fd 1 (the granted stdout stream) — in BOTH the original and the twin. `slot` is a `static` so its
/// address is a plain data pointer (no stack-array codegen). `__fork` takes a leading dummy arg chibicc
/// drops, so the lowered call is `(i64)->(i64)` — matching the fork offer op.
const GUEST_SRC: &str = r#"
long write(long fd, void *buf, long n);
long __fork(int h, long a);
long fork(void) { return __fork(0, 0); }
static long slot;
int main(int argc, char **argv) {
  slot = fork();
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
  ve1 = i64.const 1
  voffs = i64.const 262144
  vlog = i64.const 12
  vq = i64.const 0
  vs = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (ve1, voffs, vlog, vq)
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
  vz = i32.const 0
  vn = cap.call 4294967295 10 () -> (i64) vz ()
  return vn
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
