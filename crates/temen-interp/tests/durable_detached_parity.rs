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
fn powerbox(host: &mut Host, child: &temen_ir::Module, marked: bool) -> Vec<Value> {
    let inst = host.grant_instantiator(0, 1 << 16);
    // `marked`: the host attests the grant as freeze-instrumented (`grant_durable_module`) — the
    // DURABILITY.md §4 bit a durable domain's admission checks. Unmarked is the default everywhere
    // above; the #1501 tests below flip it.
    let modh = if marked {
        host.grant_durable_module(child)
    } else {
        host.grant_module(child)
    };
    let budget = host.grant_budget(0, 1 << 20, 0);
    vec![Value::I32(inst), Value::I32(modh), Value::I32(budget)]
}

/// The tree-walk oracle (INVARIANTS #9: it defines guest-observable semantics).
fn oracle(durable: bool) -> Answer {
    oracle_with(durable, false)
}

fn oracle_with(durable: bool, marked: bool) -> Answer {
    let parent = module(SPAWN);
    let child = module(CHILD);
    let mut host = Host::new();
    host.set_durable(durable);
    let args = powerbox(&mut host, &child, marked);
    let mut fuel = u64::MAX;
    classify(temen_interp::run_with_host(
        &parent, 0, &args, &mut fuel, &mut host,
    ))
}

/// The resumable `Vcpu` engine — the one the browser drives, and where #1299 put its gate. An
/// admission here surfaces as a `VcpuEvent::InstantiateDetached` for the host to service; a refusal
/// lands in the destination register and the run finishes.
fn resumable(durable: bool) -> Answer {
    resumable_with(durable, false)
}

fn resumable_with(durable: bool, marked: bool) -> Answer {
    let parent = module(SPAWN);
    let child = module(CHILD);
    let mut host = Host::new();
    host.set_durable(durable);
    let args = powerbox(&mut host, &child, marked);
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

// ---------------------------------------------------------------------------------------------
// #1501 — DURABILITY.md §4 on op 15: "a durable domain admits only freezable modules and may only
// spawn durable children." The nested spawn enforced both halves; the detached spawn, built
// fail-closed on `durable` from the start, enforced neither — the `!durable` gate was the only
// thing between a durable parent and a child no freeze could ever capture. R1 slice 1 lifted that
// gate without adding either half, which is one more reason it would have failed even with #1361's
// capture built.
//
// Both halves now land **behind** the interim gate, so today they are inert and the refusal arm is
// what fires. These tests are written as **tripwires**: each asserts the one outcome that is wrong
// in every era — admission of what §4 forbids — so they pass now, keep passing when the gate lifts
// with the halves in place, and go red if the gate lifts without them. The live control in each
// proves the probe itself works, so the tripwire half is not vacuous.
// ---------------------------------------------------------------------------------------------

/// **§4, second half:** a durable domain never admits an **un-instrumented** module detached — on
/// both engines. Today the interim `!durable` gate refuses first; when it lifts, `mod_durable_ok`
/// must refuse in its place. Either way, admission is the wrong answer.
/// One engine's answer to the spawn, parametrised on `(durable, marked)`.
type Engine = fn(bool, bool) -> Answer;

#[test]
fn a_durable_domain_never_admits_an_uninstrumented_module_detached() {
    let engines: [(&str, Engine); 2] = [("oracle", oracle_with), ("resumable", resumable_with)];
    for (name, run) in engines {
        // Live today: durability is what gates the un-instrumented module, not op 15 itself — a
        // non-durable domain admits it. Proves the mark is the only variable in play.
        assert_eq!(
            run(false, false),
            Answer::Admitted,
            "{name}: a non-durable domain admits an unmarked module detached"
        );
        // The tripwire: durable + unmarked must be a probeable refusal in every era.
        let a = run(true, false);
        assert!(
            matches!(a, Answer::Declined(_)),
            "{name}: a durable domain must refuse an unmarked module detached (a value, never \
             admission, never a trap) — got {a:?}"
        );
        // Durable + marked is what the re-lift will admit; until then it refuses with everything
        // else. No assertion on which — only that it is a value (INVARIANTS #5).
        let b = run(true, true);
        assert!(
            !matches!(b, Answer::Trapped(_)),
            "{name}: a durable domain's spawn of a marked module must never trap — got {b:?}"
        );
    }
}

/// The grandchild probe: a separate, **un-instrumented** module the detached child is handed by
/// name and tries to instantiate nested inside itself. `memory 15` — it carves as the top half of
/// the child's 64 KiB window.
const LEAF: &str = r#"memory 15
func (i64) -> (i64) {
block 0 (v0: i64) {
  return v0
  }
}
"#;

/// The detached child for the inheritance probe. Entry `(i64) -> (i64)`: `v0` is its own starter
/// `Instantiator` handle, as the op-15 arm passes it. It resolves the named grant `"m"` (LEAF),
/// tries to instantiate it as a nested grandchild (op 5: carve `[32 KiB, 64 KiB)` of its own
/// window, entry 0), and returns the grandchild's joined result (≥ 0) — or the spawn's `-EINVAL`
/// if the child's **own** durability refused the un-instrumented module (§4, the nested arm's
/// `mod_durable_ok`). So the value that comes back through the parent's join is a direct
/// observation of the bit the child inherited.
const PROBE_CHILD: &str = r#"memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  vinst = i32.wrap_i64 v0
  vnp = i64.const 20480
  vname = i64.const 109
  i64.store vnp vname
  vl = i64.const 1
  vmh = self.resolve vnp vl
  vm64 = i64.extend_i32_s vmh
  ve = i64.const 0
  voff = i64.const 32768
  vlog = i64.const 15
  vq = i64.const 0
  vs = call.cap 6 5 (i64, i64, i64, i64, i64) -> (i32) vinst (vm64, ve, voff, vlog, vq)
  z = i32.const 0
  vneg = i32.lt_s vs z
  br_if vneg 1(vs) 2(vinst, vs)
  }
block 1 (ve1: i32) {
  vr = i64.extend_i32_s ve1
  return vr
  }
block 2 (vi2: i32, vs2: i32) {
  vj = call.cap 6 1 (i32) -> (i64) vi2 (vs2)
  return vj
  }
}
"#;

/// The parent for the inheritance probe: lays one op-11-format grant record at 16384 —
/// `{name_off=16400, name_len=1, handle=v3, flags=0}` with `"m"` at 16400 — spawns `PROBE_CHILD`
/// detached (op 15: budget `v2`, module `v1`, grants `(16384, 1)`, entry 0, `memory 16`, no quota),
/// and returns the child's joined answer, or the spawn's own `-EINVAL` if op 15 refused.
/// Args: `(instantiator, child module, budget, leaf module)`.
const PROBE_PARENT: &str = r#"memory 16
func (i32, i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32, v3: i32) {
  r0 = i64.const 16384
  n16400 = i32.const 16400
  i32.store r0 n16400
  r4 = i64.const 16388
  n1 = i32.const 1
  i32.store r4 n1
  r8 = i64.const 16392
  i32.store r8 v3
  r12 = i64.const 16396
  z = i32.const 0
  i32.store r12 z
  q = i64.const 16400
  cm = i32.const 109
  i32.store8 q cm
  vmh = i64.extend_i32_u v1
  vmin = i64.extend_i32_u v2
  vgp = i64.const 16384
  vgn = i64.const 1
  ve = i64.const 0
  vlog = i64.const 16
  vq = i64.const 0
  vs = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vmin, vmh, vgp, vgn, ve, vlog, vq)
  vneg = i32.lt_s vs z
  br_if vneg 1(vs) 2(v0, vs)
  }
block 1 (ve1: i32) {
  vr = i64.extend_i32_s ve1
  return vr
  }
block 2 (vi2: i32, vs2: i32) {
  vj = call.cap 6 1 (i32) -> (i64) vi2 (vs2)
  return vj
  }
}
"#;

/// Run the inheritance probe on the tree-walk oracle (the engine that runs the detached child
/// inline, so its answer can come back through `join`). Returns the parent's i64: ≥ 0 iff the
/// grandchild was admitted.
fn probe(durable: bool) -> i64 {
    let parent = module(PROBE_PARENT);
    let child = module(PROBE_CHILD);
    let leaf = module(LEAF);
    let mut host = Host::new();
    host.set_durable(durable);
    let inst = host.grant_instantiator(0, 1 << 16);
    let ch = host.grant_module(&child);
    let budget = host.grant_budget(0, 1 << 20, 0);
    let lh = host.grant_module(&leaf); // un-instrumented on purpose: the thing §4 refuses
    let mut fuel = u64::MAX;
    let r = temen_interp::run_with_host(
        &parent,
        0,
        &[
            Value::I32(inst),
            Value::I32(ch),
            Value::I32(budget),
            Value::I32(lh),
        ],
        &mut fuel,
        &mut host,
    )
    .expect("the probe run completes");
    match r.first() {
        Some(Value::I64(n)) => *n,
        other => panic!("the probe returns one i64, got {other:?}"),
    }
}

/// **§4, first half:** a detached child's durability follows its parent's. Observed through the
/// grandchild probe above: a durable child refuses the un-instrumented LEAF, a non-durable one
/// admits it.
///
/// Today a durable parent's op 15 refuses before any child exists, so the durable arm reads the
/// parent's `-EINVAL`; after the re-lift it reads the *child's* `-EINVAL` — the same value, for the
/// right reason. What this pins is the third case: a re-lift that forgot `set_durable` would admit
/// the child non-durable, the child would admit the grandchild, and a slot would come back.
#[test]
fn a_detached_childs_durability_follows_its_parents() {
    // Live today, and the probe's own sanity check: a non-durable parent's detached child is
    // non-durable, admits the leaf grandchild, and its joined result comes back.
    let r = probe(false);
    assert!(
        r >= 0,
        "non-durable parent: the child must admit the un-instrumented grandchild — got {r}"
    );
    // The tripwire.
    let r = probe(true);
    assert!(
        r < 0,
        "durable parent: a detached child must inherit durability and refuse the un-instrumented \
         grandchild — a non-negative result means a re-lift spawned a non-durable child: {r}"
    );
}
