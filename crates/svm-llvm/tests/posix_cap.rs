//! The **POSIX personality reached from the LLVM on-ramp** — the bridge between the two worlds
//! (`POSIX.md` / `STAGE1.md` slice 4). A C guest translated by `svm-llvm` resolves the personality by
//! name (`__vm_cap_resolve("posix")`) and drives the process/fd ABI through `__vm_host_call`, exactly as
//! `fs_cap.rs` drives the `fs` cap. This proves the ops a real shell needs — `pipe`/`dup2` and
//! `posix_spawn`/`waitpid` with fd inheritance — are reachable under the on-ramp, not just from the
//! chibicc/name-binding world (`crates/svm/tests/c_posix_spawn.rs`).
//!
//! The embedder wires `Posix::set_spawn` to an uppercasing delegate (exit 42); `spawn("up")` inherits the
//! guest's preloaded stdin (`"hello"`) as the child's input and routes the child's stdout to fd 1, so the
//! *personality's* captured stdout is `"HELLO"`. The guest's own `"posix probe ok"` marker goes to the
//! *powerbox* stdout (the run's stdout) via `printf` — the two host worlds side by side in one program.
//! Runs on all three engines.

use svm_posix::SpawnResult;
use svm_run::{Backend, Limits, Outcome, RunConfig, Value};

fn instance() -> svm_run::Instance {
    let bc = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/posix_probe.ll");
    let t = svm_llvm::translate_ll_path(bc).expect("translate posix probe");
    svm_run::instantiate(t.module).expect("instantiate")
}

fn config() -> RunConfig {
    RunConfig {
        limits: Limits {
            fuel: None,
            deadline: None,
            max_fibers: 0,
            max_vcpus: 0,
        },
        stdin: vec![],
        memory_size_log2: None,
        args: vec![],
        env: vec![],
    }
}

/// Build a fresh `posix` cap + handle (per backend, so state never leaks between runs), wire the
/// uppercasing spawn delegate, run the probe, and return (exit outcome, the run's powerbox stdout, the
/// personality's captured stdout). Heap `0,0`: the probe uses only static/stack storage, no personality
/// `malloc`.
fn run_probe(backend: Backend) -> (Outcome, Vec<u8>, Vec<u8>) {
    let (cap, posix) = svm_run::posix::posix_cap(0, 0, b"hello".to_vec());
    posix.set_spawn(|name: &str, _argv: &[String], stdin: &[u8]| {
        assert_eq!(name, "up", "the probe spawns \"up\"");
        SpawnResult {
            stdout: stdin.to_ascii_uppercase(),
            status: 42,
        }
    });
    let out = instance()
        .run_with_caps(backend, &config(), &[("posix", cap)])
        .expect("run posix probe");
    (out.outcome, out.stdout, posix.stdout())
}

fn check(backend: Backend) {
    let (outcome, run_stdout, personality_stdout) = run_probe(backend);
    assert_eq!(
        outcome,
        Outcome::Returned(vec![Value::I32(0)]),
        "{backend:?}: posix probe failed (exit = failing step)",
    );
    assert_eq!(
        personality_stdout, b"HELLO",
        "{backend:?}: the spawned child's inherited-stdin output, routed to the personality's fd 1",
    );
    assert_eq!(
        String::from_utf8_lossy(&run_stdout),
        "posix probe ok\n",
        "{backend:?}: the guest's own marker on the powerbox stdout",
    );
}

#[test]
fn posix_probe_tree_walker() {
    check(Backend::TreeWalk);
}

#[test]
fn posix_probe_bytecode() {
    check(Backend::Bytecode);
}

#[test]
fn posix_probe_jit() {
    check(Backend::Jit);
}
