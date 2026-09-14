//! **#1412 / #1299 — the three engines agree about a durable domain's detached spawn.**
//!
//! `Instantiator.instantiate_detached` (op 15) under a **durable** host is the cell where the tree
//! silently diverged for six days: #1299 added a `!durable` gate to the resumable engine on
//! 2026-09-07 "like the other two engines", and #1289 R1 lifted that same gate from the tree-walker
//! on 2026-09-08 — touching only `lib.rs`. Both sides then carried comments asserting they agreed
//! with each other, so nothing in the tree pointed at the contradiction. The contradiction was not
//! *in* any file; it was between two files that each described the other.
//!
//! #1299's own note said "no test pins any of the three". This is that test.
//!
//! It exists because a per-engine assertion cannot catch this class: each engine was individually
//! self-consistent and individually documented. Only a **cross-engine** assertion, driven from one
//! module and one powerbox, can fail when a ruling moves one engine and not the others — which is
//! exactly what INVARIANTS #9 requires ("the bytecode interpreter is held bit-exact" against the
//! oracle) and what the frontier matrix (#1413) generalises.
//!
//! The current agreed answer is **refuse** (see `detached_windows.rs` for why this is interim, and
//! #1361 for the capture that reverses it). This test asserts *agreement*, and separately asserts
//! what they agree on — so when the capture lands, only the second assertion changes.

use std::sync::Arc;
use temen_interp::{bytecode, Host, Region, Trap, Value};

/// `v0` Instantiator, `v1` a granted `Module`, `v2` a detached-spawn `Budget`.
///
/// Spawns a detached child (op 15, 7-arg form) and returns the result sign-extended. A refusal is a
/// negative errno; an admission is a non-negative child slot. The guest does nothing else, so the
/// return value *is* the engine's answer.
const SPAWN: &str = r#"memory 16
func (i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32) {
  vmh = i64.extend_i32_u v1
  vmin = i64.extend_i32_u v2
  vz = i64.const 0
  ve = i64.const 0
  vlog = i64.const 15
  vq = i64.const 0
  vh = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vmin, vmh, vz, vz, ve, vlog, vq)
  vr = i64.extend_i32_s vh
  return vr
  }
}
"#;

/// The child: `memory 15` to match the op-15 `size_log2` (§14 transparency — a detached window equals
/// the module's declared memory).
const CHILD: &str = r#"memory 15
func (i64) -> (i64) {
block 0 (v0: i64) {
  return v0
  }
}
"#;

fn module(text: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(text).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// What an engine did with the spawn, reduced to the only distinction that matters here. Comparing
/// this rather than the raw value keeps the test about *agreement*: the engines need not return the
/// same child slot to agree that they admitted.
#[derive(PartialEq, Eq, Debug)]
enum Answer {
    Admitted,
    Declined(i64),
    Trapped(Trap),
}

fn classify(r: Result<Vec<Value>, Trap>) -> Answer {
    match r {
        Err(t) => Answer::Trapped(t),
        Ok(v) => match v.first() {
            Some(Value::I64(n)) if *n < 0 => Answer::Declined(*n),
            _ => Answer::Admitted,
        },
    }
}

/// Grant the three caps op 15 needs. The quota is generous on purpose: a quota miss also refuses with
/// `-EINVAL`, which would make a refusal assertion pass for the wrong reason.
fn powerbox(host: &mut Host, child: &temen_ir::Module) -> Vec<Value> {
    let inst = host.grant_instantiator(0, 1 << 16);
    let modh = host.grant_module(child);
    let budget = host.grant_budget(0, 1 << 20, 0);
    vec![Value::I32(inst), Value::I32(modh), Value::I32(budget)]
}

/// The tree-walk oracle (INVARIANTS #9: it defines guest-observable semantics).
fn oracle(durable: bool) -> Answer {
    let parent = module(SPAWN);
    let child = module(CHILD);
    let mut host = Host::new();
    host.set_durable(durable);
    let args = powerbox(&mut host, &child);
    let mut fuel = u64::MAX;
    classify(temen_interp::run_with_host(
        &parent, 0, &args, &mut fuel, &mut host,
    ))
}

/// The resumable `Vcpu` engine — the one the browser drives, and where #1299 put its gate. An
/// admission here surfaces as a `VcpuEvent::InstantiateDetached` for the host to service; a refusal
/// lands in the destination register and the run finishes.
fn resumable(durable: bool) -> Answer {
    let parent = module(SPAWN);
    let child = module(CHILD);
    let mut host = Host::new();
    host.set_durable(durable);
    let args = powerbox(&mut host, &child);
    let prog = bytecode::VcpuProgram::compile(&parent).expect("compile");
    let back = Arc::new(Region::new(1u64 << 16, 4096));
    let mut vcpu = bytecode::Vcpu::new_root_with_powerbox(&prog, 0, &args, back, &[], host)
        .expect("root vcpu");
    // One `run()` is the whole story: the guest issues exactly one op, so the first event it yields
    // *is* the engine's answer — either the spawn surfaced, or it refused and the run finished.
    match vcpu.run() {
        bytecode::VcpuEvent::Done(v) => classify(Ok(v)),
        bytecode::VcpuEvent::Trapped(t) => classify(Err(t)),
        // The engine did the authority-bearing work and handed the host a window to mint: an
        // admission, even though this test never services it.
        bytecode::VcpuEvent::InstantiateDetached { .. } => Answer::Admitted,
        _ => panic!("unexpected event from a guest that only issues op 15"),
    }
}

/// **The pin.** One module, one powerbox, both engines — they must give the same answer, durable or
/// not. This is the assertion that would have gone red on 2026-09-08.
#[test]
fn the_oracle_and_the_resumable_engine_agree_about_a_detached_spawn() {
    for durable in [false, true] {
        let a = oracle(durable);
        let b = resumable(durable);
        assert_eq!(
            a, b,
            "durable={durable}: the tree-walk oracle and the resumable engine must agree about op 15 \
             (INVARIANTS #9). A ruling that moves one engine must move the others in the same change."
        );
    }
}

/// What they agree *on*, asserted separately so the capture landing (#1361) changes this test and not
/// the agreement test above.
#[test]
fn a_durable_domain_is_refused_and_a_non_durable_one_is_admitted() {
    assert_eq!(
        oracle(false),
        Answer::Admitted,
        "a non-durable domain spawns detached children — the gate is about durability, not op 15"
    );
    assert!(
        matches!(oracle(true), Answer::Declined(_)),
        "a durable domain's detached spawn refuses, until the per-child capture lands (#1361)"
    );
}

/// The refusal is a **value**, never a trap — on either engine. INVARIANTS #5: a lifecycle constraint
/// the guest cannot see coming must be probeable on its own error path, not domain-killing.
#[test]
fn the_durable_refusal_is_a_value_on_both_engines() {
    for (name, answer) in [("oracle", oracle(true)), ("resumable", resumable(true))] {
        assert!(
            !matches!(answer, Answer::Trapped(_)),
            "{name}: a durable domain's detached spawn must refuse probeably, never trap — got {answer:?}"
        );
    }
}
