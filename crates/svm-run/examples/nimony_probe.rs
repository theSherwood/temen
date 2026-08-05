//! Run the NIM.md Phase-1 codegen-shape probe (`demos/nimony/arc_probe.c`, on-ramped to a
//! `.svmb` by `demos/nimony/build_probe.sh`) on the SVM and forward its stdout + exit code, so the
//! build script can byte-diff every engine against the native reference. The probe is a plain
//! `main`-returning program over the synthesized powerbox (printf → `write`, `exit`), no caps —
//! the minimal cousin of `chibicc_run.rs`.
//!
//! ```text
//! SVM_PROBE_BACKEND={treewalk|bytecode|jit} \
//!   cargo run -q --release -p svm-run --example nimony_probe -- <probe.svmb>
//! ```

use std::io::Write;
use std::process::exit;

use svm_run::{Backend, Limits, Outcome, RunConfig, Value};

fn main() {
    let svmb = std::env::args()
        .nth(1)
        .expect("usage: nimony_probe <probe.svmb>");

    let backend = match std::env::var("SVM_PROBE_BACKEND").as_deref() {
        Ok("bytecode") => Backend::Bytecode,
        Ok("jit") => Backend::Jit,
        Ok("treewalk") | Err(_) => Backend::TreeWalk,
        Ok(other) => panic!("SVM_PROBE_BACKEND: unknown engine {other:?}"),
    };

    let bytes = std::fs::read(&svmb).unwrap_or_else(|e| panic!("read {svmb}: {e}"));
    let module = svm_encode::decode_module(&bytes).expect("decode .svmb");
    let inst = svm_run::instantiate(module).expect("instantiate (verifies)");

    let cfg = RunConfig {
        limits: Limits {
            fuel: None,
            deadline: None,
            max_fibers: 0,
            max_vcpus: 0,
        },
        stdin: vec![],
        memory_size_log2: None,
        args: vec![b"arc_probe".to_vec()],
        env: vec![],
        ..RunConfig::default()
    };
    let run = inst.run(backend, &cfg).expect("run probe guest");

    std::io::stdout().write_all(&run.stdout).unwrap();
    std::io::stderr().write_all(&run.stderr).unwrap();
    match run.outcome {
        Outcome::Returned(vals) => exit(match vals.first() {
            Some(Value::I32(n)) => *n & 0xff,
            _ => 0,
        }),
        Outcome::Exited(code) => exit(code & 0xff),
    }
}
