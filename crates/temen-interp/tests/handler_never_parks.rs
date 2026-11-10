//! #2742 — [`temen_ir::seams::handler_never_parks`], the one test of whether a serve handler can
//! park, which decides where the Cranelift JIT runs a direct handoff's handler: on the caller's own
//! stack when nothing the handler reaches can park, else on a fiber of its own. Each case is one
//! handler (function 1) of a module whose function 0 is an unused main; function 2, when present,
//! is a helper the handler calls.

use temen_ir::seams::handler_never_parks;

fn parks(handler: &str, helper: &str) -> bool {
    let src = format!(
        r#"memory 12
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  return v0
  }}
}}

func (i64, i64) -> (i64) {{
block 0 (vh: i64, vn: i64) {{
{handler}
  }}
}}
{helper}"#
    );
    let m = temen_text::parse_module(&src).unwrap_or_else(|e| panic!("parse: {e:?}\n{src}"));
    temen_verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}\n{src}"));
    !handler_never_parks(&m.funcs, 1)
}

const PURE_HELPER: &str = r#"
func (i64) -> (i64) {
block 0 (vx: i64) {
  vone = i64.const 1
  vy = i64.add vx vone
  return vy
  }
}"#;

const WAITING_HELPER: &str = r#"
func (i64) -> (i64) {
block 0 (vx: i64) {
  vaddr = i64.const 1024
  vexp = i32.const 0
  vto = i64.const -1
  vst = i32.atomic.wait vaddr vexp vto
  return vx
  }
}"#;

#[test]
fn arithmetic_never_parks() {
    assert!(!parks("  vs = i64.add vh vn\n  return vs", ""));
}

#[test]
fn a_call_to_a_helper_that_cannot_park_never_parks() {
    assert!(!parks("  vr = call 2 (vn)\n  return vr", PURE_HELPER));
}

#[test]
fn a_call_to_a_helper_that_waits_may_park() {
    assert!(parks("  vr = call 2 (vn)\n  return vr", WAITING_HELPER));
}

#[test]
fn a_live_call_may_park() {
    let live = "  vself = i32.wrap_i64 vh\n  vr = call.cap 268435456 0 (i64, i64) -> (i64) vself (vh, vn)\n  return vr";
    assert!(parks(live, ""));
}

#[test]
fn an_indirect_call_may_park() {
    let dynamic = "  vf = ref.func 2\n  vr = call.dyn (i64) -> (i64) vf (vn)\n  return vr";
    assert!(parks(dynamic, PURE_HELPER));
}

#[test]
fn a_clock_read_never_parks() {
    let clock = "  vc = i32.wrap_i64 vh\n  vt = call.cap 2 0 () -> (i64) vc ()\n  return vt";
    assert!(!parks(clock, ""));
}

#[test]
fn a_stream_read_may_park_and_a_write_never_does() {
    let read =
        "  vs = i32.wrap_i64 vh\n  vr = call.cap 0 0 (i64, i64) -> (i64) vs (vn, vn)\n  return vr";
    let write =
        "  vs = i32.wrap_i64 vh\n  vw = call.cap 0 1 (i64, i64) -> (i64) vs (vn, vn)\n  return vw";
    assert!(parks(read, ""));
    assert!(!parks(write, ""));
}

#[test]
fn a_serve_op_may_park() {
    let wait = "  vz = i32.const 0\n  vc = svc.wait vz\n  return vc";
    assert!(parks(wait, ""));
}

#[test]
fn an_out_of_range_handler_may_park() {
    let m =
        temen_text::parse_module("memory 12\nfunc () -> () {\nblock 0 () {\n  return\n  }\n}\n")
            .expect("parse");
    assert!(!handler_never_parks(&m.funcs, 7));
}
