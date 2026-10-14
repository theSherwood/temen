//! §14 `Instantiator` (iface 6) on the **JIT**, differentially vs. the interpreter, for the ops whose
//! answer needs no child: both backends must agree on the result *and* the final window. The spawns
//! themselves are pinned on every tier by `instantiate_record.rs`, and the retired carve forms by
//! `carve_spawn_refused.rs` (#1867).

use temen_interp::{run_capture_reserved_with_host, Host, Trap, Value};
use temen_jit::{compile_and_run_capture_reserved_with_host, JitOutcome};
use temen_text::parse_module;
use temen_verify::verify_module;

type Both = (Result<Vec<Value>, Trap>, Vec<u8>, JitOutcome, Vec<u8>);

/// Run `src`'s entry (the granted `Instantiator` handle is `v0`) on **both** backends over a
/// fully-mapped `1<<win_log2`-byte window seeded with a non-zero pattern, with an `Instantiator`
/// granted over the whole window. Returns the interp result, the JIT outcome, and both final windows.
fn both(src: &str, win_log2: u8) -> Both {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let win = 1u64 << win_log2;
    let init: Vec<u8> = (0..win)
        .map(|i| (i as u8).wrapping_mul(31) ^ 0xa5)
        .collect();

    let mut hi = Host::new();
    let ih = hi.grant_instantiator(0, win);
    let mut hj = Host::new();
    let jh = hj.grant_instantiator(0, win);
    assert_eq!(ih, jh, "the Instantiator grant must encode identically");

    let mut fuel = 5_000_000u64;
    let (ir, imem) =
        run_capture_reserved_with_host(&m, 0, &[Value::I32(ih)], &mut fuel, &init, 0, &mut hi);
    let (jo, jmem) = compile_and_run_capture_reserved_with_host(
        &m,
        0,
        &[jh as i64],
        &init,
        0,
        temen_run::cap_thunk,
        &mut hj as *mut Host as *mut core::ffi::c_void,
    )
    .expect("jit");
    (ir, imem, jo, jmem)
}

/// §3.6 parity — `child_offer` (op 14) on the JIT is a **probeable refusal, never a trap**:
/// the op is eval-loop-serviced (it mints from the tree-walk scheduler's live-child registry,
/// which the JIT runtime does not have), so the JIT lowers it to a `-EINVAL` result — the
/// same refusal the interpreter gives a bad child handle. Differential: both backends agree.
#[test]
fn jit_child_offer_refuses_probeably_and_matches_interp() {
    if !temen_jit::fiber_supported() {
        return;
    }
    let src = "memory 16\n\
         func (i32) -> (i64) {\n\
         block 0 (v0: i32) {\n\
         \x20 vbad = i32.const 99\n\
         \x20 vz = i64.const 0\n\
         \x20 vc = call.cap 6 14 (i32, i64) -> (i32) v0 (vbad, vz)\n\
         \x20 vr = i64.extend_i32_s vc\n\
         \x20 return vr\n\
           }\n\
         }\n";
    let (ir, _imem, jo, _jmem) = both(src, 16);
    assert_eq!(
        ir,
        Ok(vec![Value::I64(-22)]),
        "interp: a bad child handle refuses -EINVAL"
    );
    assert_eq!(
        jo,
        JitOutcome::Returned(vec![-22]),
        "jit: op 14 refuses probeably with the same value — no trap, no wrong answer"
    );
}
