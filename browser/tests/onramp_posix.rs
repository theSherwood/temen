//! The **POSIX personality** on-ramp entry (`onramp_posix_exec` / the wasm `svm_run_onramp_posix`
//! export) — STAGE1.md "real posix shell in the playground", slice 1. Where `onramp_exec` runs a
//! `.svmb` under the fixed §3e powerbox (stdout/stdin/exit/memory), this runs it under the full
//! `svm-posix` personality: one `HostFn` capability implementing the libc/memfs surface, with the
//! module's manifest imports bound to it **by name** (`svm_posix::bind`). This is the seam the real
//! `svm-posix` shell (and any chibicc program linking the personality libc) runs through.
//!
//! Slice 1 proves the personality reaches the browser runtime and the crate still builds for wasm:
//! a hand-written manifest module reads its input from the personality's stdin and echoes it to the
//! personality's stdout — the shell's exact I/O path — plus the fail-closed gates. The full shell
//! `.svmb` asset + the playground card are slice 2.

use svm_browser::{onramp_posix_exec, STATUS_OK, STATUS_UNSUPPORTED};

/// A paramless `_start` (the phase-4 manifest shape) that `read(0, buf, 256)`s the personality's
/// preloaded stdin into the guest window, then `write(1, buf, n)`s it back — an `echo` of stdin
/// through the personality. Imports (first-occurrence order) are `read`/`write`, bound by name to
/// `svm_posix`'s `OP_READ`/`OP_WRITE`. Returns the byte count `write` reports.
const ECHO_STDIN: &str = r#"memory 15
export 0 func "_start" 0
func () -> (i64) {
block 0 () {
  hr = i32.const 0
  fd0 = i64.const 0
  buf = i64.const 1024
  cap = i64.const 256
  n = call.sym "read" (i64, i64, i64) -> (i64) hr (fd0, buf, cap)
  hw = i32.const 0
  fd1 = i64.const 1
  w = call.sym "write" (i64, i64, i64) -> (i64) hw (fd1, buf, n)
  return w
  }
}
"#;

#[test]
fn posix_personality_echoes_stdin_through_the_playground_entry() {
    let m = svm_text::parse_module(ECHO_STDIN).expect("parse echo module");
    assert_eq!(
        m.imports.iter().map(|i| i.name.as_str()).collect::<Vec<_>>(),
        ["read", "write"],
        "the manifest carries the two POSIX imports, in declaration order",
    );
    let out = onramp_posix_exec(&m, b"hello posix\n");
    assert_eq!(out.status, STATUS_OK, "the personality guest runs cleanly");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "hello posix\n",
        "read(0,…) drains the personality's stdin and write(1,…) reaches its captured stdout",
    );
    assert_eq!(out.value, 12, "write returns the byte count it forwarded");
}

/// An import the personality does not provide (`add_seven` is not a libc name) makes
/// `svm_posix::bind` fail closed — the run is `STATUS_UNSUPPORTED`, nothing is bound. (The
/// `Instantiator`/ring imports the shell's concurrent paths use land the same way until a later
/// slice teaches this entry those names.)
const NON_POSIX_IMPORT: &str = r#"memory 15
export 0 func "_start" 0
func () -> (i64) {
block 0 () {
  h = i32.const 0
  x = i64.const 5
  r = call.sym "add_seven" (i64) -> (i64) h (x)
  return r
  }
}
"#;

#[test]
fn a_non_posix_import_fails_closed() {
    let m = svm_text::parse_module(NON_POSIX_IMPORT).expect("parse module");
    let out = onramp_posix_exec(&m, b"");
    assert_eq!(
        out.status, STATUS_UNSUPPORTED,
        "a module importing a non-POSIX name binds nothing and fails closed",
    );
}

/// A pre-manifest **legacy positional** entry (func 0 takes its handle as a parameter, not exported
/// `_start`) fails the phase-4 shape gate ([`onramp_check`]) — same as the fixed on-ramp entry.
const LEGACY_POSITIONAL: &str = r#"memory 15
func (i32) -> (i64) {
block 0 (v0: i32) {
  fd1 = i64.const 1
  buf = i64.const 0
  len = i64.const 5
  w = call.sym "write" (i64, i64, i64) -> (i64) v0 (fd1, buf, len)
  return w
  }
}
"#;

#[test]
fn legacy_positional_entry_is_rejected() {
    let m = svm_text::parse_module(LEGACY_POSITIONAL).expect("parse module");
    let out = onramp_posix_exec(&m, b"");
    assert_eq!(
        out.status, STATUS_UNSUPPORTED,
        "an import-bearing module without the paramless `_start` shape fails closed",
    );
}
