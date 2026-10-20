//! JACL self-hosted compiler-guest — the engine contract the browser relies on.
//!
//! `jacl_compiler.temen` is the JACL frontend compiled to Temen: it reads JACL source on stdin and
//! writes TEMEN-IR on stdout, expanding macros IN-GUEST via the §22 `Jit` capability (`grant_onramp_caps`
//! grants it because the guest imports `vm_jit_*`). This test pins the fact the playground's run path
//! depends on: **guest-JIT macro staging works** — on the on-ramp interpreter ([`onramp_exec`]) the
//! compiler expands the tour's `unless` macro (→ `if` → `br_if`) in-guest. This is the tour fix. Its
//! run on the emitted tier, over the pump, is `coop_tierup_driver.rs`'s
//! `coop_jacl_compiler_runs_through_the_driver`.
//!
//! Cross-repo asset: `jacl_compiler.temen` is built by the outer jacl_impl repo
//! (codegen/selfhost/build_compiler_temen.sh). Absent ⇒ the test SKIPs.

use temen_browser::{onramp_exec, STATUS_EXIT, STATUS_OK};

// The tour's macro: a pure syntax-quote template. Expands `unless C {B}` → `[if C {} {B}]` → br_if.
const MACRO_SRC: &str = "defmacro unless {cond body} { syntax-quote [if ~cond {} ~body] }\n\
                         mut hit 0\nunless [== 1 2] { set hit 5 }\nhit\n";

fn jacl_compiler_temen() -> Option<temen_ir::Module> {
    // browser/ -> vendor/temen -> vendor -> jacl_impl root
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../codegen/selfhost/build/jacl_compiler.temen"
    );
    let bytes = std::fs::read(p).ok()?;
    Some(temen_encode::decode_module(&bytes).expect("decode jacl_compiler.temen"))
}

/// Interpreter path — the oracle. Source on stdin, emitted IR on stdout.
fn interp_compile(compiler: &temen_ir::Module, src: &str) -> String {
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
/// compiler drives the §22 `Jit` cap **through imports** (temen-llvm lowers `extern __vm_jit_*` to
/// `call.import`), so `onramp_exec` runs it on the tree-walker (which routes import-bound `invoke`/
/// `install`/`uninstall` to the driver; the bytecode engine only handles statically-resolved
/// `call.cap (JIT, op)`). The symbol table is the canonical `Slot`/`Cap` wire form and the `Jit`
/// grant mirrors temen-run's powerbox (1024-slot table + fiber hosting) so the staged macro — which
/// runs on the compiler's scheduler root — resolves.
#[test]
fn jacl_bytecode_guest_jit_stages_the_tour_macro() {
    let Some(compiler) = jacl_compiler_temen() else {
        eprintln!(
            "SKIP: jacl_compiler.temen absent (run codegen/selfhost/build_compiler_temen.sh)"
        );
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
