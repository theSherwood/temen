//! **A release run says which trap it ended in, and where a memory fault was** (#1714).
//!
//! `onramp_exec` — the plain, non-debug run an embedder uses for speed — mapped every trap to
//! `STATUS_TRAP` and dropped which one it was. c_interpret runs Instant-speed Runs this way, so every
//! "this program crashes" lesson reported **"Error"** where it teaches **"Segmentation fault (addr
//! 0x0)"**. The debug session has carried the trap kind and fault address since #1190; this is the
//! same for the release path, through `PbOutcome` and the `temen_trap_*` / `temen_fault_addr`
//! exports. Fail-soft: SKIPs if `chibicc.temen` isn't built.

use temen_browser::{
    onramp_exec, onramp_fs_exec, playground_include_files, STATUS_EXIT, STATUS_OK, STATUS_TRAP,
};
use temen_interp::Trap;

fn chibicc_temen() -> Option<temen_ir::Module> {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/web/assets/chibicc.temen");
    let bytes = std::fs::read(p).ok()?;
    Some(temen_encode::decode_module(&bytes).expect("decode chibicc.temen"))
}

fn compile(chibicc: &temen_ir::Module, src: &str) -> temen_ir::Module {
    let mut files: Vec<(String, Vec<u8>)> = playground_include_files();
    files.push(("in.c".to_string(), src.as_bytes().to_vec()));
    let image = temen_fs::encode_image(&files, &["include".to_string()]);
    let compiled = onramp_fs_exec(
        chibicc,
        &image,
        &[b"chibicc", b"--data-page", b"65536", b"/in.c"],
        b"",
    );
    assert!(
        compiled.status == STATUS_OK || compiled.status == STATUS_EXIT,
        "compile status {} — stderr: {}",
        compiled.status,
        String::from_utf8_lossy(&compiled.stderr)
    );
    let ir = String::from_utf8(compiled.stdout).expect("IR is utf8");
    temen_text::parse_module(&ir).unwrap_or_else(|e| panic!("parse IR: {e:?}"))
}

#[test]
fn a_null_dereference_reports_memory_fault_at_address_zero() {
    let Some(chibicc) = chibicc_temen() else {
        eprintln!("SKIP: chibicc.temen absent");
        return;
    };
    let m = compile(&chibicc, "int main() { int *p = 0; return *p; }\n");
    let run = onramp_exec(&m, b"");
    assert_eq!(run.status, STATUS_TRAP, "a NULL dereference traps");
    assert_eq!(
        run.trap,
        Some(Trap::MemoryFault),
        "and says it was a memory fault"
    );
    assert_eq!(run.fault_addr, Some(0), "at the NULL address");
}

#[test]
fn a_trap_that_is_not_a_memory_fault_carries_no_address() {
    let Some(chibicc) = chibicc_temen() else {
        eprintln!("SKIP: chibicc.temen absent");
        return;
    };
    let m = compile(
        &chibicc,
        "int d(int x) { return 10 / x; }\nint main() { return d(0); }\n",
    );
    let run = onramp_exec(&m, b"");
    assert_eq!(run.status, STATUS_TRAP);
    assert_eq!(run.trap, Some(Trap::DivByZero));
    assert_eq!(run.fault_addr, None, "only a memory fault has an address");
}

/// And a clean run after a faulting one reports neither — the per-run slot is cleared, not left
/// holding the earlier run's fault.
#[test]
fn a_clean_run_reports_no_trap() {
    let Some(chibicc) = chibicc_temen() else {
        eprintln!("SKIP: chibicc.temen absent");
        return;
    };
    let crash = compile(&chibicc, "int main() { int *p = 0; return *p; }\n");
    assert_eq!(onramp_exec(&crash, b"").fault_addr, Some(0));
    let ok = compile(&chibicc, "int main() { return 7; }\n");
    let run = onramp_exec(&ok, b"");
    assert_eq!(run.status, STATUS_OK);
    assert_eq!(run.trap, None);
    assert_eq!(run.fault_addr, None);
}
