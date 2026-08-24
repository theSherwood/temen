//! Tests for [`temen_run::PowerboxProgram`] — the compile-once/run-many powerbox split.
//!
//! The central guarantee is **differential parity**: a program run through `PowerboxProgram`
//! (compiled once, reused) is byte-identical to the same program run through `run_powerbox`
//! (compiled per call), for the same `stdin`. That, plus independence between reuses and the
//! refusal of out-of-scope module shapes, is what makes the cache safe to hand a partial-evaluation
//! residual and run against many inputs without paying Cranelift codegen every time.

use temen_ir::Module;
use temen_run::{run_powerbox, Outcome, PowerboxProgram, Value};
use temen_text::parse_module;
use temen_verify::verify_module;

fn load(src: &str) -> Module {
    let m = parse_module(src).expect("parse text IR");
    verify_module(&m).expect("verify");
    m
}

/// read(stdin) into the window, then write those bytes back out — a stdin→stdout round-trip.
fn echo_module() -> Module {
    load(
        "memory 16\n\
         export 0 func \"_start\" 0\n\
         func () -> (i32) {\n\
         block 0 () {\n\
         \x20 v0 = i32.const 0\n\
         \x20 v1 = i64.const 0\n\
         \x20 v2 = i64.const 64\n\
         \x20 v3 = call.sym \"read\" (i64, i64) -> (i64) v0(v1, v2)\n\
         \x20 v4 = call.sym \"write\" (i64, i64) -> (i64) v0(v1, v3)\n\
         \x20 v5 = i32.const 0\n\
         \x20 return v5\n\
           }\n\
         }\n",
    )
}

#[test]
fn matches_run_powerbox_across_many_inputs() {
    // The whole point: PowerboxProgram (compiled once) is byte-identical to run_powerbox (compiled
    // per call) for the same stdin, over a batch of different inputs on ONE compiled program.
    let m = echo_module();
    let mut prog = PowerboxProgram::compile(m.clone()).expect("compile");
    for input in [
        &b""[..],
        &b"ping"[..],
        &b"a longer line of stdin\n"[..],
        &b"\x00\x01\x02binary\xff"[..],
        &b"ping"[..], // repeat: same input twice must still match
    ] {
        let want = run_powerbox(&m, input).expect("run_powerbox");
        let got = prog.run(input).expect("PowerboxProgram::run");
        assert_eq!(got.stdout, want.stdout, "stdout mismatch for {input:?}");
        assert_eq!(got.stderr, want.stderr, "stderr mismatch for {input:?}");
        assert_eq!(got.outcome, want.outcome, "outcome mismatch for {input:?}");
        // And directly against the oracle: an echo returns exactly its stdin.
        assert_eq!(got.stdout, input);
    }
}

#[test]
fn reuses_are_independent_no_state_leak() {
    // run_raw allocates a fresh window + re-applies data segments each call, and the host is reset
    // per run — so a big input followed by an empty one must NOT echo stale bytes from the first.
    let mut prog = PowerboxProgram::compile(echo_module()).expect("compile");
    let first = prog.run(b"stale-bytes-should-not-persist").expect("run 1");
    assert_eq!(first.stdout, b"stale-bytes-should-not-persist");
    let second = prog.run(b"").expect("run 2");
    assert!(
        second.stdout.is_empty(),
        "second run leaked state from the first: {:?}",
        second.stdout
    );
}

#[test]
fn matches_run_powerbox_for_write_and_exit() {
    // A no-stdin program that writes a constant and returns.
    let writer = load(
        "memory 16\n\
         data 16 \"hi\\n\"\n\
         export 0 func \"_start\" 0\n\
         func () -> (i32) {\n\
         block 0 () {\n\
         \x20 v0 = i32.const 0\n\
         \x20 v1 = i64.const 16\n\
         \x20 v2 = i64.const 3\n\
         \x20 v3 = call.sym \"write\" (i64, i64) -> (i64) v0(v1, v2)\n\
         \x20 v4 = i32.const 7\n\
         \x20 return v4\n\
           }\n\
         }\n",
    );
    let mut prog = PowerboxProgram::compile(writer.clone()).expect("compile");
    let got = prog.run(b"").expect("run");
    let oracle = run_powerbox(&writer, b"").expect("oracle");
    assert_eq!(got.stdout, b"hi\n");
    assert_eq!(got.outcome, Outcome::Returned(vec![Value::I32(7)]));
    assert_eq!(got.stdout, oracle.stdout);
    assert_eq!(got.outcome, oracle.outcome);

    // An Exit(5) through the manifest — the terminal outcome must survive the cache too.
    let exiter = load(
        "export 0 func \"_start\" 0\n\
         func () -> (i32) {\n\
         block 0 () {\n\
         \x20 v0 = i32.const 0\n\
         \x20 v1 = i32.const 5\n\
         \x20 call.sym \"exit\" (i32) -> () v0(v1)\n\
         \x20 unreachable\n\
           }\n\
         }\n",
    );
    let mut prog = PowerboxProgram::compile(exiter.clone()).expect("compile");
    assert_eq!(prog.run(b"").expect("run").outcome, Outcome::Exited(5));
    assert_eq!(prog.run(b"").expect("run").outcome, Outcome::Exited(5));
}

#[test]
fn compile_rejects_unverified_module_fail_closed() {
    // The escape gate is paid once, at compile — an unverified module must be refused here too, not
    // silently cached and run.
    let m = parse_module(
        "func () -> (i64) {\n\
         block 0 () {\n\
         \x20 v0 = i64.const 0\n\
         \x20 v1 = i64.load v0\n\
         \x20 return v1\n\
           }\n\
         }\n",
    )
    .expect("parse");
    assert!(
        verify_module(&m).is_err(),
        "module must be genuinely invalid"
    );
    let err = match PowerboxProgram::compile(m) {
        Ok(_) => panic!("must reject unverified module"),
        Err(e) => e,
    };
    assert!(err.contains("verification failed"), "got: {err}");
}

#[test]
fn compile_refuses_concurrent_module() {
    // A powerbox entry that spawns a vCPU needs the locked cap-thunk path, which the cache does not
    // implement — compile must refuse it (pointing at run_powerbox), not miscompile it.
    let m = load(
        "memory 16\n\
         export 0 func \"_start\" 0\n\
         func () -> () {\n\
         block 0 () {\n\
         \x20 v0 = i64.const 5\n\
         \x20 v1 = thread.spawn 1 v0 v0\n\
         \x20 v2 = thread.join v1\n\
         \x20 return\n\
           }\n\
         }\n\
         func (i64, i64) -> (i64) {\n\
         block 0 (vsp: i64, varg: i64) {\n\
         \x20 return varg\n\
           }\n\
         }\n",
    );
    let err = match PowerboxProgram::compile(m) {
        Ok(_) => panic!("must refuse concurrent module"),
        Err(e) => e,
    };
    assert!(
        err.contains("concurrency") && err.contains("run_powerbox"),
        "got: {err}"
    );
}
