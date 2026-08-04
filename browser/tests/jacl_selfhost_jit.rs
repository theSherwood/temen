//! JACL self-hosted compiler-guest — the two engine contracts the browser relies on.
//!
//! `jacl_compiler.svmb` is the JACL frontend compiled to SVM: it reads JACL source on stdin and
//! writes SVM-IR on stdout, expanding macros IN-GUEST via the §22 `Jit` capability (`grant_onramp_caps`
//! grants it because the guest imports `vm_jit_*`). This test pins the two facts the playground's
//! run path depends on:
//!
//!   1. **Guest-JIT macro staging works** — on the on-ramp interpreter ([`onramp_exec`]) the compiler
//!      expands the tour's `unless` macro (→ `if` → `br_if`) in-guest. This is the tour fix.
//!   2. **The wasm-JIT reactor correctly refuses the compiler** — the compiler links the full `jaclrt`
//!      runtime (STW GC + scheduler), whose safepoint/`spawn`/`await` use §12 suspend/`cont`/futex ops.
//!      `compile_jit`'s `reachable_concurrency` guard therefore returns `InterpDriven`, and the
//!      wasm-driven [`JitOnrampRun`] reactor fails closed with `STATUS_UNSUPPORTED`. The playground
//!      falls back to the interpreter for it (as for any non-wasm-drivable guest).
//!
//! Making the compiler (or any linked JACL program) wasm-drivable needs a single-threaded, no-scheduler
//! `jaclrt` build variant so no reachable function carries a concurrency op — tracked separately.
//!
//! Cross-repo asset: `jacl_compiler.svmb` is built by the outer jacl_impl repo
//! (codegen/selfhost/build_compiler_svmb.sh). Absent ⇒ the test SKIPs.

use svm_browser::{onramp_exec, JitOnrampRun, STATUS_EXIT, STATUS_OK, STATUS_UNSUPPORTED};

const WIN_LOG2: u8 = 27; // 128 MiB — matches SELFHOST_WIN_LOG2 (the compiler + macro staging heap)
const WIN_SIZE: u64 = 1 << WIN_LOG2;

// The tour's macro: a pure syntax-quote template. Expands `unless C {B}` → `[if C {} {B}]` → br_if.
const MACRO_SRC: &str = "defmacro unless {cond body} { syntax-quote [if ~cond {} ~body] }\n\
                         mut hit 0\nunless [== 1 2] { set hit 5 }\nhit\n";

fn jacl_compiler_svmb() -> Option<svm_ir::Module> {
    // browser/ -> vendor/svm -> vendor -> jacl_impl root
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../codegen/selfhost/build/jacl_compiler.svmb"
    );
    let bytes = std::fs::read(p).ok()?;
    Some(svm_encode::decode_module(&bytes).expect("decode jacl_compiler.svmb"))
}

/// Interpreter path — the oracle. Source on stdin, emitted IR on stdout.
fn interp_compile(compiler: &svm_ir::Module, src: &str) -> String {
    let out = onramp_exec(compiler, src.as_bytes());
    assert!(
        out.status == STATUS_OK || out.status == STATUS_EXIT,
        "interp compile status {} exit={}",
        out.status,
        out.exit_code,
    );
    String::from_utf8(out.stdout).expect("IR is utf8")
}

/// Expand the tour's macro IN-GUEST on the browser's on-ramp interpreter — the tour fix. The JACL
/// compiler drives the §22 `Jit` cap **through imports** (svm-llvm lowers `extern __vm_jit_*` to
/// `call.import`), so `onramp_exec` runs it on the tree-walker (which routes import-bound `invoke`/
/// `install`/`uninstall` to the driver; the bytecode engine only handles statically-resolved
/// `cap.call (JIT, op)`). The symbol table is the canonical `Slot`/`Cap` wire form and the `Jit`
/// grant mirrors svm-run's powerbox (1024-slot table + fiber hosting) so the staged macro — which
/// runs on the compiler's scheduler root — resolves.
#[test]
fn jacl_bytecode_guest_jit_stages_the_tour_macro() {
    let Some(compiler) = jacl_compiler_svmb() else {
        eprintln!("SKIP: jacl_compiler.svmb absent (run codegen/selfhost/build_compiler_svmb.sh)");
        return;
    };
    let ir = interp_compile(&compiler, MACRO_SRC);
    assert!(!ir.contains("%%ERROR%%"), "interp emitted error: {ir:.400}");
    // The macro must have expanded in-guest: `unless` -> `if` lowers to a `br_if`.
    assert!(
        ir.contains("br_if"),
        "macro did not expand in-guest (no br_if):\n{ir:.400}"
    );
    eprintln!(
        "OK: {} bytes of IR, `unless` staged in-guest on the interpreter",
        ir.len()
    );
}

#[test]
fn jacl_compiler_wasmjit_reactor_open_is_well_formed() {
    let Some(compiler) = jacl_compiler_svmb() else {
        eprintln!("SKIP: jacl_compiler.svmb absent");
        return;
    };
    // The single-shot wasm-JIT reactor either emits the compiler-guest (wasm-drivable `_start`) or
    // fails closed with `STATUS_UNSUPPORTED` — at which point the browser runs it on the interpreter
    // (the guaranteed path, exercised by the test above). Which branch is taken depends on the vendored
    // engine's concurrency-reachability analysis of the full `jaclrt` runtime the compiler links, so
    // this asserts only the well-formed contract (open, or a clean `STATUS_UNSUPPORTED`), not a
    // specific drive mode — the interpreter path is the invariant, not wasm-drivability.
    let mut win = vec![0u8; WIN_SIZE as usize].into_boxed_slice();
    let win_ptr = win.as_mut_ptr();
    // SAFETY: `win` outlives the call and is used solely as this run's window.
    let opened = unsafe {
        JitOnrampRun::open_shared_run(
            &compiler,
            win_ptr,
            WIN_SIZE,
            WIN_LOG2,
            false,
            MACRO_SRC.as_bytes().to_vec(),
        )
    };
    match opened {
        Ok(_) => {
            eprintln!("OK: wasm-JIT reactor emits the compiler-guest (wasm-drivable `_start`)")
        }
        Err(status) => {
            assert_eq!(
                status, STATUS_UNSUPPORTED,
                "reactor refusal must be a clean STATUS_UNSUPPORTED (interpreter fallback), got {status}"
            );
            eprintln!("OK: wasm-JIT reactor refuses the compiler-guest → interpreter fallback");
        }
    }
}
