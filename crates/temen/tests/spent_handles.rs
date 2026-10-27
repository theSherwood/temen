//! **#2667 — a spent handle answers `-EBADF`; a forged one traps, on every backend.**
//!
//! INVARIANTS #5 reserves traps for forgery: a handle carrying a generation never issued, or one
//! that names no slot. A handle that was issued and has since been spent — a thread already joined,
//! a fiber that already returned — is a lifecycle fact, and another party's lifecycle action must
//! never be a trap. So `thread.join` yields `(status, value)`, the shape of `Instantiator.join`:
//! `(0, its result)`, or `(-EBADF, 0)` for a spent handle; and a `cont.resume` of a spent fiber
//! yields `(-EBADF, 0)` beside the statuses it already had. Pinned on the tree-walk oracle, the
//! bytecode engine and the Cranelift JIT (INVARIANTS #9). The §14 child ops' answers are pinned on
//! every interpreter driver by temen-interp's `spawn_every_driver.rs`, and on the JIT by
//! `lifecycle.rs`.

use temen_interp::Value;
use temen_run::{instantiate, Backend, Outcome, RunConfig};

const ALL: [Backend; 3] = [Backend::TreeWalk, Backend::Bytecode, Backend::Jit];

/// Run `src` on `backend`: its results, or how it ended abnormally.
fn run_on(src: &str, backend: Backend) -> Result<Vec<Value>, String> {
    let module = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&module).expect("verify");
    let instance = instantiate(module).expect("instantiate");
    match instance.run(backend, &RunConfig::default()) {
        Ok(run) => match run.outcome {
            Outcome::Returned(vals) => Ok(vals),
            Outcome::Exited(code) => Err(format!("exited {code}")),
        },
        Err(e) => Err(e),
    }
}

/// Every backend returns `want` from `src`.
fn pin_returns(src: &str, what: &str, want: &[i64]) {
    let want: Vec<Value> = want.iter().map(|&v| Value::I64(v)).collect();
    for backend in ALL {
        assert_eq!(run_on(src, backend), Ok(want.clone()), "{backend:?}: {what}");
    }
}

/// Every backend ends `src` with the trap named `trap`.
fn pin_traps(src: &str, what: &str, trap: &str) {
    for backend in ALL {
        match run_on(src, backend) {
            Err(e) => assert!(e.contains(trap), "{backend:?}: {what}: want {trap}, got {e}"),
            Ok(vals) => panic!("{backend:?}: {what}: want {trap}, got {vals:?}"),
        }
    }
}

const EBADF: i64 = temen_ir::errno::EBADF;

/// A worker that returns its argument.
const WORKER: &str = "func (i64, i64) -> (i64) {\nblock 0 (vsp: i64, va: i64) {\n  return va\n  }\n}\n";

/// The second join of a thread finds its handle spent: `(-EBADF, 0)`, and the first join's
/// `(0, 5)` is untouched.
#[test]
fn a_second_thread_join_answers_ebadf() {
    let src = format!(
        "func () -> (i64, i64, i64, i64) {{
block 0 () {{
  vsp = i64.const 0
  va = i64.const 5
  vt = thread.spawn 1 vsp va
  vs1, vv1 = thread.join vt
  vs2, vv2 = thread.join vt
  return vs1 vv1 vs2 vv2
  }}
}}
{WORKER}"
    );
    pin_returns(&src, "a re-join", &[0, 5, EBADF, 0]);
}

/// A handle that names no slot is a forgery: negative, or past the table where the mask folds it
/// onto no slot (three threads span four slots, and slot 3 was never issued).
#[test]
fn a_forged_thread_handle_traps() {
    for (what, h) in [("join(-1)", "-1"), ("join(3)", "3")] {
        let src = format!(
            "func () -> (i64) {{
block 0 () {{
  vsp = i64.const 0
  va = i64.const 5
  vt0 = thread.spawn 1 vsp va
  vt1 = thread.spawn 1 vsp va
  vt2 = thread.spawn 1 vsp va
  vh = i32.const {h}
  vs, vv = thread.join vh
  vs0, vv0 = thread.join vt0
  vs1, vv1 = thread.join vt1
  vs2, vv2 = thread.join vt2
  return vv
  }}
}}
{WORKER}"
        );
        pin_traps(&src, what, "ThreadFault");
    }
}

/// A fiber that returns its argument plus 100.
const FIBER: &str = "func (i64, i64) -> (i64) {\nblock 0 (vsp: i64, va: i64) {\n  vc = i64.const 100\n  vr = i64.add va vc\n  return vr\n  }\n}\n";

/// A resume of a fiber that already returned runs nothing: `(-EBADF, 0)`, beside the first
/// resume's `(1 returned, 107)`.
#[test]
fn a_resume_of_a_returned_fiber_answers_ebadf() {
    let src = format!(
        "func () -> (i64, i64, i64, i64) {{
block 0 () {{
  vf = ref.func 1
  vsp = i64.const 0
  vk = cont.new vf vsp
  va = i64.const 7
  vs1, vv1 = cont.resume vk va
  vs2, vv2 = cont.resume vk va
  vw1 = i64.extend_i32_s vs1
  vw2 = i64.extend_i32_s vs2
  return vw1 vv1 vw2 vv2
  }}
}}
{FIBER}"
    );
    pin_returns(&src, "a re-resume", &[1, 107, EBADF, 0]);
}

/// A fiber handle carrying a generation its slot never issued is a forgery: `(5 << 24) | 0` names
/// slot 0, which has issued generation 0 only.
#[test]
fn a_forged_fiber_generation_traps() {
    let src = format!(
        "func () -> (i64) {{
block 0 () {{
  vf = ref.func 1
  vsp = i64.const 0
  vk = cont.new vf vsp
  vforged = i64.const 83886080
  va = i64.const 7
  vs, vv = cont.resume vforged va
  return vv
  }}
}}
{FIBER}"
    );
    pin_traps(&src, "a never-issued generation", "FiberFault");
}
