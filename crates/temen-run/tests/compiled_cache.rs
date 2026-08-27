//! Tests for [`temen_run::CompiledCache`] — the content-keyed compiled-module cache (`WASM_AOT.md`
//! slice 1; the `ISSUES.md` build-once/run-many gap).
//!
//! The guarantees, mirroring `powerbox_program.rs`'s differential model:
//! 1. **Reuse by content** — the same module seen twice compiles once (a digest hit), and two
//!    independently-parsed but identical modules share the slot (identity is the encoded image, not
//!    the object).
//! 2. **Differential parity** — a cached run is byte-identical to `run_powerbox` (compiled per call).
//! 3. **No state leak** — caching *code* leaves each run's window/host fresh (INVARIANTS #6).
//! 4. **Distinct modules don't collide**, and a **refused** (out-of-scope) module is not cached and
//!    doesn't poison the cache.

use temen_ir::Module;
use temen_run::{run_powerbox, CompiledCache, Outcome, Value};
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
         \x20 v1 = i64.const 16384\n\
         \x20 v2 = i64.const 64\n\
         \x20 v3 = call.sym \"read\" (i64, i64) -> (i64) v0(v1, v2)\n\
         \x20 v4 = call.sym \"write\" (i64, i64) -> (i64) v0(v1, v3)\n\
         \x20 v5 = i32.const 0\n\
         \x20 return v5\n\
           }\n\
         }\n",
    )
}

/// A no-stdin program that writes a constant `data` segment and returns a distinct code — so it is
/// clearly a *different* module from `echo_module` (different digest, different output).
fn writer_module() -> Module {
    load(
        "memory 16\n\
         data 16400 \"hi\\n\"\n\
         export 0 func \"_start\" 0\n\
         func () -> (i32) {\n\
         block 0 () {\n\
         \x20 v0 = i32.const 0\n\
         \x20 v1 = i64.const 16400\n\
         \x20 v2 = i64.const 3\n\
         \x20 v3 = call.sym \"write\" (i64, i64) -> (i64) v0(v1, v2)\n\
         \x20 v4 = i32.const 7\n\
         \x20 return v4\n\
           }\n\
         }\n",
    )
}

#[test]
fn hit_reuses_without_recompiling() {
    let m = echo_module();
    let mut cache = CompiledCache::new();
    assert!(cache.is_empty());

    let first = cache.run(&m, b"ping").expect("run 1");
    assert_eq!(first.stdout, b"ping");
    assert_eq!(cache.compiles(), 1, "first sighting compiles");
    assert_eq!(cache.hits(), 0);
    assert_eq!(cache.len(), 1);
    assert!(cache.contains(&m));

    // Second run of the same module: served from the cache, no new compile.
    let second = cache.run(&m, b"pong").expect("run 2");
    assert_eq!(second.stdout, b"pong");
    assert_eq!(cache.compiles(), 1, "no recompile on the second sighting");
    assert_eq!(cache.hits(), 1);
    assert_eq!(cache.len(), 1);
}

#[test]
fn key_is_content_not_object_identity() {
    // Two independently-parsed but identical modules must share the slot: the second is a hit.
    let a = echo_module();
    let b = echo_module();
    assert_eq!(
        CompiledCache::key(&a),
        CompiledCache::key(&b),
        "identical modules hash identically"
    );
    let mut cache = CompiledCache::new();
    cache.run(&a, b"x").expect("run a");
    assert!(cache.contains(&b), "the clone hits the same digest slot");
    cache.run(&b, b"y").expect("run b");
    assert_eq!(cache.compiles(), 1, "the clone did not recompile");
    assert_eq!(cache.hits(), 1);
    assert_eq!(cache.len(), 1);
}

#[test]
fn matches_run_powerbox_across_inputs() {
    // A cached run is byte-identical to run_powerbox (compiled per call), over a batch of inputs on
    // ONE cached program — including a repeated input.
    let m = echo_module();
    let mut cache = CompiledCache::new();
    for input in [
        &b""[..],
        &b"ping"[..],
        &b"a longer line of stdin\n"[..],
        &b"\x00\x01\x02binary\xff"[..],
        &b"ping"[..], // repeat: same input twice must still match
    ] {
        let want = run_powerbox(&m, input).expect("run_powerbox");
        let got = cache.run(&m, input).expect("CompiledCache::run");
        assert_eq!(got.stdout, want.stdout, "stdout mismatch for {input:?}");
        assert_eq!(got.stderr, want.stderr, "stderr mismatch for {input:?}");
        assert_eq!(got.outcome, want.outcome, "outcome mismatch for {input:?}");
        assert_eq!(got.stdout, input, "echo returns exactly its stdin");
    }
    assert_eq!(
        cache.compiles(),
        1,
        "one module → one compile for all inputs"
    );
}

#[test]
fn reuses_are_independent_no_state_leak() {
    // Caching code must not cache window/guest state: a big input followed by an empty one must NOT
    // echo stale bytes from the first run (INVARIANTS #6 / DESIGN §21).
    let m = echo_module();
    let mut cache = CompiledCache::new();
    let first = cache
        .run(&m, b"stale-bytes-should-not-persist")
        .expect("run 1");
    assert_eq!(first.stdout, b"stale-bytes-should-not-persist");
    let second = cache.run(&m, b"").expect("run 2");
    assert!(
        second.stdout.is_empty(),
        "second cached run leaked state from the first: {:?}",
        second.stdout
    );
    assert_eq!(cache.compiles(), 1);
    assert_eq!(cache.hits(), 1);
}

#[test]
fn distinct_modules_do_not_collide() {
    let echo = echo_module();
    let writer = writer_module();
    assert_ne!(
        CompiledCache::key(&echo),
        CompiledCache::key(&writer),
        "different modules → different keys"
    );
    let mut cache = CompiledCache::new();

    let e = cache.run(&echo, b"abc").expect("echo run");
    assert_eq!(e.stdout, b"abc");
    let w = cache.run(&writer, b"").expect("writer run");
    assert_eq!(w.stdout, b"hi\n");
    assert_eq!(w.outcome, Outcome::Returned(vec![Value::I32(7)]));
    assert_eq!(cache.compiles(), 2, "two distinct modules compile twice");
    assert_eq!(cache.len(), 2);

    // Re-running each hits its own slot and stays correct (no cross-contamination).
    assert_eq!(cache.run(&echo, b"def").expect("echo 2").stdout, b"def");
    assert_eq!(cache.run(&writer, b"").expect("writer 2").stdout, b"hi\n");
    assert_eq!(cache.compiles(), 2, "still no recompiles");
    assert_eq!(cache.hits(), 2);
}

#[test]
fn refused_module_is_not_cached_and_does_not_poison() {
    // A concurrent module is out of PowerboxProgram's scope → run() surfaces the refusal, nothing is
    // cached, and a later cacheable module still compiles and runs cleanly.
    let concurrent = load(
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
    let mut cache = CompiledCache::new();
    let err = cache
        .run(&concurrent, b"")
        .expect_err("concurrent module must be refused");
    assert!(
        err.contains("concurrency") && err.contains("run_powerbox"),
        "got: {err}"
    );
    assert!(cache.is_empty(), "a refused module must not be cached");
    assert_eq!(cache.compiles(), 0);

    // The cache is not poisoned: a cacheable module afterwards works.
    let echo = echo_module();
    assert_eq!(cache.run(&echo, b"ok").expect("echo run").stdout, b"ok");
    assert_eq!(cache.compiles(), 1);
    assert_eq!(cache.len(), 1);
}
