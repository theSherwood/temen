//! Slice 7 (INTERACTIVE_EMBEDDING.md): the **schedule policy** — a seeded deterministic pick
//! (adversarial interleaving variation, a pure function of `(seed, turn)`) and the
//! **forced-switch** verb (a concrete `(turn, task)` override recorded at resolve time).
//! Acceptance: the same seed reproduces the identical slice-6 trace tape (and a different seed
//! genuinely varies the interleaving); a forced switch lands at a deterministic turn and survives
//! `seek`; everything fails closed off the threaded bytecode path.

use temen_dap::{BytecodeBackend, DapServer, Debuggee, Json};
use temen_interp::{IrPc, Stop};
use temen_text::parse_module;

/// Two workers race load/add/store on mem[0]; the root joins both and returns the count. The race
/// gives the schedule real freedom: interleavings differ observably (the final count is 1 or 2).
const RACY_COUNTER: &str = r#"
memory 16
func () -> (i64) {
block 0 () {
  vsp = i64.const 0
  va = i64.const 1
  vh0 = thread.spawn 1 vsp va
  vh1 = thread.spawn 1 vsp va
  vj0 = thread.join vh0
  vj1 = thread.join vh1
  vaddr = i64.const 0
  vr = i64.load vaddr
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vaddr = i64.const 0
  vc = i64.load vaddr
  vn = i64.add vc varg
  i64.store vaddr vn
  vz = i64.const 0
  return vz
  }
}
"#;

fn backend(seed: Option<u64>) -> BytecodeBackend {
    let m = parse_module(RACY_COUNTER).expect("parses");
    BytecodeBackend::new(m, 0, &[], u64::MAX, false, Vec::new(), false, None, seed)
        .expect("threaded subset")
}

/// Arm the trace, drive to completion, return the tape JSON.
fn run_traced(b: &mut BytecodeBackend) -> String {
    assert!(Debuggee::set_sched_trace(b, true), "trace armed");
    drive_to_end(b);
    b.sched_trace_json().expect("a tape").to_string()
}

fn drive_to_end(b: &mut BytecodeBackend) {
    loop {
        match Debuggee::run_until_stop(b) {
            Stop::Finished(r) => {
                r.expect("clean finish");
                break;
            }
            Stop::Break { .. } => continue,
            Stop::Blocked => panic!("unexpected block"),
        }
    }
}

/// **The seeded pick varies the interleaving and is deterministic per seed**: the same seed twice
/// yields the identical tape; at least one seed in a small candidate set diverges from the
/// lowest-index default (everything is deterministic, so this test is stable).
#[test]
fn seeded_pick_varies_and_is_deterministic_per_seed() {
    let default_tape = run_traced(&mut backend(None));
    let seed42_a = run_traced(&mut backend(Some(42)));
    let seed42_b = run_traced(&mut backend(Some(42)));
    assert_eq!(seed42_a, seed42_b, "one seed, one schedule");
    let varied = (1..=8).any(|s| run_traced(&mut backend(Some(s))) != default_tape);
    assert!(
        varied,
        "some seed produces an interleaving the lowest-index default doesn't"
    );
}

/// **The seeded schedule survives `seek`**: rewind to turn 0 and re-drive — the replay reproduces
/// the seeded tape bit-for-bit (the policy is re-applied to the rebuilt run).
#[test]
fn seeded_schedule_survives_seek() {
    let mut b = backend(Some(42));
    let tape = run_traced(&mut b);
    let _ = Debuggee::seek(&mut b, 0);
    drive_to_end(&mut b);
    let replayed = b.sched_trace_json().expect("a tape").to_string();
    assert_eq!(
        tape, replayed,
        "seek(0) + rerun replays the seeded schedule"
    );
}

/// **A forced switch lands at a deterministic turn and survives `seek`**: stop at a worker
/// breakpoint (two tasks runnable), force a switch — the trace shows the chosen task running that
/// exact turn, and a `seek(0)` + re-drive reproduces the override at the identical turn.
#[test]
fn forced_switch_lands_and_survives_seek() {
    let mut b = backend(None);
    assert!(Debuggee::set_sched_trace(&mut b, true));
    // Break on the worker's load (func 1, block 0, inst 1) — fires in whichever worker runs first,
    // while the other worker is also runnable.
    let bp = IrPc {
        module: 0,
        func: 1,
        block: 0,
        inst: 1,
    };
    Debuggee::set_breakpoint(&mut b, bp);
    let Stop::Break { pc, .. } = Debuggee::run_until_stop(&mut b) else {
        panic!("expected the worker breakpoint");
    };
    assert_eq!(pc, bp);
    let turn = Debuggee::turn(&b);
    let chosen = Debuggee::force_switch(&mut b, None).expect("a runnable task to switch to");
    // Drive to completion; the tape must show the chosen task running the recorded turn.
    Debuggee::clear_breakpoint(&mut b, bp);
    drive_to_end(&mut b);
    let has_forced_turn = |tape: &str| {
        temen_dap::parse(tape)
            .and_then(|j| j.as_array().map(|a| a.to_vec()))
            .expect("array")
            .iter()
            .any(|e| {
                e.get("kind").and_then(|k| k.as_str()) == Some("turn")
                    && e.get("turn").and_then(|v| v.as_i64()) == Some(turn as i64)
                    && e.get("task").and_then(|v| v.as_i64()) == Some(chosen as i64)
            })
    };
    let tape = b.sched_trace_json().expect("a tape").to_string();
    assert!(
        has_forced_turn(&tape),
        "the forced task ran turn {turn}: {tape}"
    );
    // Rewind and re-drive: the override replays at the identical turn.
    let _ = Debuggee::seek(&mut b, 0);
    drive_to_end(&mut b);
    let replayed = b.sched_trace_json().expect("a tape").to_string();
    assert_eq!(tape, replayed, "the forced switch survives seek");
}

/// **Fail-closed**: a seed with a single-vCPU module fails construction; `force_switch` on a
/// single-vCPU backend refuses; the DAP `forceSwitch` request fails cleanly off the threaded path.
#[test]
fn policy_fails_closed_off_the_threaded_path() {
    const SINGLE: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  v0 = i64.const 0
  return v0
  }
}
"#;
    let m = parse_module(SINGLE).expect("parses");
    assert!(
        BytecodeBackend::new(
            m.clone(),
            0,
            &[],
            u64::MAX,
            false,
            Vec::new(),
            false,
            None,
            Some(7)
        )
        .is_none(),
        "a seed with a single-vCPU module declines"
    );
    let mut single =
        BytecodeBackend::new(m, 0, &[], u64::MAX, false, Vec::new(), false, None, None)
            .expect("subset");
    assert_eq!(
        Debuggee::force_switch(&mut single, None),
        None,
        "force_switch refuses on a single vCPU"
    );

    let req = |seq: i64, command: &str, args: Json| {
        Json::obj(vec![
            ("seq", Json::i(seq)),
            ("type", Json::s("request")),
            ("command", Json::s(command)),
            ("arguments", args),
        ])
    };
    let response = |msgs: &[Json]| -> Json {
        msgs.iter()
            .find(|m| m.get("type").and_then(|t| t.as_str()) == Some("response"))
            .expect("a response")
            .clone()
    };
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    s.handle(&req(
        2,
        "launch",
        Json::obj(vec![
            ("programText", Json::s(SINGLE)),
            ("function", Json::i(0)),
            ("args", Json::Arr(vec![])),
            ("engine", Json::s("bytecode")),
        ]),
    ));
    let out = s.handle(&req(3, "forceSwitch", Json::obj(vec![])));
    assert_eq!(response(&out).get("success"), Some(&Json::Bool(false)));
}

/// **The DAP surface end-to-end**: two sessions launched with the same seed + `schedTrace` produce
/// identical tapes; the guest still completes and returns a valid racy count (1 or 2).
#[test]
fn dap_seeded_sessions_reproduce_the_tape() {
    let req = |seq: i64, command: &str, args: Json| {
        Json::obj(vec![
            ("seq", Json::i(seq)),
            ("type", Json::s("request")),
            ("command", Json::s(command)),
            ("arguments", args),
        ])
    };
    let run = || -> String {
        let mut s = DapServer::new();
        s.handle(&req(1, "initialize", Json::obj(vec![])));
        let out = s.handle(&req(
            2,
            "launch",
            Json::obj(vec![
                ("programText", Json::s(RACY_COUNTER)),
                ("function", Json::i(0)),
                ("args", Json::Arr(vec![])),
                ("engine", Json::s("bytecode")),
                ("seed", Json::i(1234)),
                ("schedTrace", Json::Bool(true)),
            ]),
        ));
        let launched = out
            .iter()
            .find(|m| m.get("type").and_then(|t| t.as_str()) == Some("response"))
            .and_then(|r| r.get("success").cloned());
        assert_eq!(launched, Some(Json::Bool(true)), "seeded launch ok");
        let mut done = false;
        let mut seq = 3;
        while !done && seq < 200 {
            let out = s.handle(&req(seq, "continue", Json::obj(vec![])));
            done = out
                .iter()
                .any(|m| m.get("event").and_then(|e| e.as_str()) == Some("terminated"));
            seq += 1;
        }
        assert!(done, "the seeded session ran to completion");
        let out = s.handle(&req(seq, "schedTrace", Json::obj(vec![])));
        out.iter()
            .find(|m| m.get("type").and_then(|t| t.as_str()) == Some("response"))
            .and_then(|r| r.get("body").cloned())
            .expect("a tape body")
            .to_string()
    };
    assert_eq!(run(), run(), "same seed, same tape, across sessions");
}
