//! The **on-ramp powerbox** entry (`onramp_exec` / the wasm `temen_run_onramp` export): a `.temen`
//! straight off `temen-llvm-translate` runs under the fixed §3e `VM_CAP_*` grant prefix the LLVM
//! on-ramp's synthesized `_start` expects — the seam that lets the browser run real C/C++ guests
//! (Lua, SQLite) the same way `temen-run` does natively.
//!
//! The fixture `fixtures/hello_onramp.temen` is `crates/temen-run/demos/hello.c` compiled with stock
//! `clang -O2 -emit-llvm` and translated (`temen-llvm-translate hello.bc -o hello_onramp.temen
//! --host-page 65536 --null-guard` — the #964 guarded layout, so this fixture also pins the
//! marked powerbox path in the browser). The
//! current on-ramp emits the **by-name** paramless `_start` (S15), whose manifest imports
//! (`write`/`exit`) bind to slot bindings at instantiation (`grant_onramp_caps`; IMPORTS.md phase 4
//! — the positional handle-args entry form is gone, and an import-bearing module without the
//! manifest entry shape fails closed). Larger guests (malloc → the memory cap, Lua, SQLite Phase A)
//! are verified out-of-tree via `cargo run --example run_onramp`.

use temen_browser::{onramp_exec, STATUS_OK, STATUS_UNSUPPORTED};

/// A pre-manifest **legacy positional** entry shape: func 0 takes its `write` handle as a
/// parameter (slot-order delivery) and is not exported as `_start`. Phase 4 deleted the
/// `resolve_imports` rewrite that used to accept this, so an import-bearing module without the
/// manifest entry shape (paramless func 0 exported `_start`) must **fail closed**.
const LEGACY_POSITIONAL: &str = r#"memory 16
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const 0
  v2 = i64.const 5
  v3 = call.sym "write" (i64, i64) -> (i64) v0 (v1, v2)
  return v3
  }
}
"#;

#[test]
fn legacy_positional_import_blob_is_rejected() {
    let m = temen_text::parse_module(LEGACY_POSITIONAL).expect("parse legacy module");
    assert!(
        !m.imports.is_empty(),
        "the fixture must actually carry an import manifest"
    );
    let out = onramp_exec(&m, b"");
    assert_eq!(
        out.status, STATUS_UNSUPPORTED,
        "an import-bearing module without the manifest entry shape fails closed (phase 4)"
    );
}

#[test]
fn hello_onramp_prints_through_the_powerbox() {
    let bytes = include_bytes!("fixtures/hello_onramp.temen");
    let m = temen_encode::decode_module(bytes).expect("decode hello_onramp.temen");
    let out = onramp_exec(&m, b"");
    assert_eq!(out.status, STATUS_OK, "on-ramp guest should run cleanly");
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "hello, sandbox!\n",
        "the guest's write(1, …) must reach the captured stdout stream",
    );
}

/// A manifest import no on-ramp grant answers to **refuses the run before it starts** (IMPORTS.md
/// §2.1, #2579), as `temen-run` refuses it natively and a child's spawn refuses it (#2219): the
/// guest never runs, so it never reaches the import that would have faulted, and the run's notes
/// name the import (`temen_notes_len`). Until #2579 the on-ramp left the slot empty and ran the
/// program until its first call through it.
#[test]
fn an_import_nothing_grants_refuses_the_run_and_the_notes_name_it() {
    let src = r#"memory 16
type 0 func (i64) -> (i64)
import 0 func "frobnicate" 0
export 0 func "_start" 0
func () -> (i64) {
block 0 () {
  v0 = i64.const 7
  return v0
  }
}
"#;
    let m = temen_text::parse_module(src).expect("parse");
    let out = onramp_exec(&m, b"");
    assert_eq!(out.status, STATUS_UNSUPPORTED, "refused before it starts");
    assert_eq!(
        temen_browser::last_notes(),
        vec!["refused the run: no grant satisfies its import `frobnicate`".to_string()]
    );
}

/// The POSIX open binds through the same binder (#2579): an import neither the personality's
/// ops (`__px_<op>`) nor the core's memory ops serve refuses the run, as `temen_posix::bind`
/// refuses it natively. It used to bind empty and fault at the call.
#[test]
fn a_posix_program_with_an_import_the_personality_lacks_is_refused() {
    let src = r#"memory 16
type 0 func (i64) -> (i64)
import 0 func "__px_frobnicate" 0
export 0 func "_start" 0
func () -> (i64) {
block 0 () {
  v0 = i64.const 0
  return v0
  }
}
"#;
    let m = temen_text::parse_module(src).expect("parse");
    let out = temen_browser::bash_exec(&m, &[b"prog"], b"");
    assert_eq!(out.status, STATUS_UNSUPPORTED, "refused before it starts");
    assert_eq!(
        temen_browser::last_notes(),
        vec!["refused the run: no grant satisfies its import `__px_frobnicate`".to_string()]
    );
}
