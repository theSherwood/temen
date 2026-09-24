//! #1728 — a `join` on a spent, negative or out-of-range child handle traps in the guest
//! (`ThreadFault`, the oracle's `resolve_thread` rule) on the browser's op-13 drivers, as it does on
//! the tree-walker and the cooperative scheduler.
//!
//! Two driver loops keep their own child tables:
//! - the op-13 JIT driver's root loop (`temen_op13jit_step`) answered a re-join with the banked result
//!   again, and an out-of-range handle with `Malformed`;
//! - the interpreter loop a declined child runs on (`nimc::drive_op13`) indexed its table with the
//!   guest's handle, so an out-of-range handle panicked the host.
//!
//! Both now resolve through `temen_interp::take_child`. The loop reports a trap as `OP13JIT_TRAP`
//! without its kind; `take_child`'s own unit test pins the kind.

use std::sync::Mutex;

use temen_browser::{
    temen_op13jit_close, temen_op13jit_open_named, temen_op13jit_result, temen_op13jit_step,
    OP13JIT_DONE, OP13JIT_TRAP,
};

// The op-13 loop state is process-global (`OP13_JIT`): serialize the tests.
static LOCK: Mutex<()> = Mutex::new(());

/// The driver (`memory 16`, entry `(inst, module, fs)`): op 13 spawns the child into the 32-KiB carve
/// at 32768 with no grants, joins it into `vr`, then runs `tail`, which must leave the result in `vo`.
fn driver(tail: &str) -> Vec<u8> {
    let src = format!(
        r#"memory 16
func (i32, i32, i32) -> (i64) {{
block 0 (v0: i32, v1: i32, v2: i32) {{
  vmh = i64.extend_i32_u v1
  vg = i64.const 0
  ventry = i64.const 0
  voff = i64.const 32768
  vsl = i64.const 15
  vq = i64.const 0
  vh = call.cap 6 13 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vmh, vg, vg, ventry, voff, vsl, vq)
  vr = call.cap 6 1 (i32) -> (i64) v0 (vh)
{tail}
  return vo
  }}
}}
"#
    );
    let m = temen_text::parse_module(&src).expect("parse driver");
    temen_verify::verify_module(&m).expect("verify driver");
    temen_encode::encode_module(&m)
}

/// The child (`memory 15`, the carve): runs `body`, then returns 42. An unreachable §13
/// `SharedRegion` op declines the emit, so the child runs on the interpreter loop inline.
fn child(body: &str) -> Vec<u8> {
    let src = format!(
        r#"memory 15
func (i64, i64) -> (i64) {{
block 0 (vsp: i64, vas: i64) {{
{body}
  vr = i64.const 42
  return vr
  }}
}}
func () -> (i64) {{
block 0 () {{
  vh = i32.const 0
  va = i64.const 0
  vr = call.cap 4 0 (i64, i64) -> (i64) vh (va, va)
  return vr
  }}
}}
"#
    );
    let m = temen_text::parse_module(&src).expect("parse child");
    temen_verify::verify_module(&m).expect("verify child");
    temen_encode::encode_module(&m)
}

/// Open the loop over `driver` and `child`, step it to its end, and return the status and result.
fn run(driver: &[u8], child: &[u8]) -> (i32, i64) {
    // SAFETY: live byte slices for the duration of the call.
    let st = unsafe {
        temen_op13jit_open_named(driver.as_ptr(), driver.len(), child.as_ptr(), child.len())
    };
    assert_eq!(st, 0, "the op-13 loop opens");
    let out = (temen_op13jit_step(), temen_op13jit_result());
    temen_op13jit_close();
    out
}

#[test]
fn a_first_join_returns_the_childs_result() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    assert_eq!(
        run(&driver("  vo = i64.add vr vr"), &child("")),
        (OP13JIT_DONE, 84),
        "control: the child returns 42 at the driver's single join"
    );
}

#[test]
fn a_rejoin_traps() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let (st, _) = run(
        &driver("  vo = call.cap 6 1 (i32) -> (i64) v0 (vh)"),
        &child(""),
    );
    assert_eq!(st, OP13JIT_TRAP, "the join retired the handle");
}

#[test]
fn a_negative_or_out_of_range_join_traps() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    for bad in [-1, 1, 7] {
        let tail = format!("  vb = i32.const {bad}\n  vo = call.cap 6 1 (i32) -> (i64) v0 (vb)");
        let (st, _) = run(&driver(&tail), &child(""));
        assert_eq!(st, OP13JIT_TRAP, "join on handle {bad}");
    }
}

/// The declined child joins a handle it never spawned. Its trap reaches the driver at its join.
#[test]
fn a_childs_join_on_a_handle_it_never_spawned_traps_not_panics() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    for bad in [-1, 0, 5] {
        let body = format!("  vb = i32.const {bad}\n  vj = thread.join vb");
        let (st, _) = run(&driver("  vo = i64.add vr vr"), &child(&body));
        assert_eq!(st, OP13JIT_TRAP, "the child's join on handle {bad}");
    }
}
