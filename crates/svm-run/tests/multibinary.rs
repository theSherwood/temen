//! W4 (NIM.md §3c) mechanism proof — a multi-*binary* compiler-driver shape on svm, decoupled
//! from the real nimony toolchain. nimony's `nifmake` spawns `nifler → nimony → hexer → lengc` as
//! subprocesses, each consuming the previous phase's output file. This test proves that
//! orchestration runs on svm today over an **existing seam** — the `exec` capability's
//! `domain_exec` backend (EXEC.md) — with **no new host op**: a driver module resolves `"exec"`,
//! runs phase `p1` with a seed input, drains its captured output, and feeds that output as the
//! **input** of phase `p2`, then emits `p2`'s result. Both phases and the driver are pure SVM
//! modules; each phase runs as an isolated child domain (own window, powerbox, fuel), exactly as a
//! spawned process would. This is the W4 analog of Path B's runtime shim: the mechanism, proven
//! with stand-in phases before the real phase binaries are compiled.
//!
//! The composition is content- and order-sensitive so it proves data *flowed*, not just that two
//! children ran: `p1` **doubles** its input, `p2` **echoes then appends `!`**. Seed `a` →
//! `p1` → `aa` → `p2` → `aa!`. If `p2` had been handed the original seed instead of `p1`'s output,
//! the result would be `a!` (2 bytes), not `aa!` (3) — so `aa!`/exit 3 witnesses the hand-off.

use std::sync::Arc;
use svm_run::exec::{domain_exec, DomainProgram};
use svm_run::{
    instantiate, instantiate_with_imports, Backend, HostCap, Imports, Limits, Outcome, RunConfig,
};
use svm_text::parse_module;

/// Phase `p1` — a child domain that **doubles** its stdin: read up to 32 bytes into 64, then write
/// them back twice. `read`/`write` bind to the child's seeded stdin / captured stdout (the domain
/// runner wires them, exactly as `ECHO_DOMAIN`/`CAT_DOMAIN` in `exec_cap.rs`).
const P1_DOUBLE: &str = "\
memory 16
import 0 \"read\" (i64, i64) -> (i64)
import 1 \"write\" (i64, i64) -> (i64)
func 0 () -> () {
block 0 () {
  vp = i64.const 64
  vc = i64.const 32
  vn = call.import 0 (vp, vc)
  vw1 = call.import 1 (vp, vn)
  vw2 = call.import 1 (vp, vn)
  return
  }
}
export 0 func \"_start\" 0
";

/// Phase `p2` — echo stdin, then append a fixed `!` from the phase's own data. `aa` → `aa!`.
const P2_ECHO_BANG: &str = "\
memory 16
data 0 \"!\"
import 0 \"read\" (i64, i64) -> (i64)
import 1 \"write\" (i64, i64) -> (i64)
func 0 () -> () {
block 0 () {
  vp = i64.const 64
  vc = i64.const 32
  vn = call.import 0 (vp, vc)
  vw1 = call.import 1 (vp, vn)
  vbp = i64.const 0
  vbl = i64.const 1
  vw2 = call.import 1 (vbp, vbl)
  return
  }
}
export 0 func \"_start\" 0
";

/// The driver — the `nifmake` analog. Resolve `"exec"`; run `p1` with stdin `a`; read `p1`'s output
/// into 32..; run `p2` with **that** buffer as stdin; read `p2`'s output into 64..; emit it and exit
/// with its length. `exec` ops (iface 13): 0 = run(argv,stdin), 1 = read_out(job,buf,cap).
const DRIVER: &str = "\
memory 16
data 0 \"exec\"
data 8 \"p1\"
data 12 \"p2\"
data 16 \"a\"
import 0 \"out\" (i64, i64) -> (i64)
import 1 \"exit\" (i32) -> ()
func 0 () -> () {
block 0 () {
  vp = i64.const 0
  vl = i64.const 4
  vh = cap.self.resolve vp vl
  vp1 = i64.const 8
  vp1l = i64.const 2
  vsp = i64.const 16
  vsl = i64.const 1
  vjob1 = cap.call 13 0 (i64, i64, i64, i64) -> (i64) vh (vp1, vp1l, vsp, vsl)
  vbuf1 = i64.const 32
  vcap = i64.const 16
  vn1 = cap.call 13 1 (i64, i64, i64) -> (i64) vh (vjob1, vbuf1, vcap)
  vp2 = i64.const 12
  vp2l = i64.const 2
  vjob2 = cap.call 13 0 (i64, i64, i64, i64) -> (i64) vh (vp2, vp2l, vbuf1, vn1)
  vbuf2 = i64.const 64
  vn2 = cap.call 13 1 (i64, i64, i64) -> (i64) vh (vjob2, vbuf2, vcap)
  vw = call.import 0 (vbuf2, vn2)
  vcode = i32.wrap_i64 vn2
  call.import 1 (vcode)
  unreachable
  }
}
export 0 func \"_start\" 0
";

fn registry() -> Imports {
    Imports::new()
        .provide("out", HostCap::stdout())
        .provide("exit", HostCap::exit())
}

fn phases() -> Vec<DomainProgram> {
    let p1 = parse_module(P1_DOUBLE).expect("parse p1");
    let p2 = parse_module(P2_ECHO_BANG).expect("parse p2");
    vec![
        DomainProgram {
            name: "p1".into(),
            instance: Arc::new(instantiate(p1).expect("instantiate p1")),
            limits: Limits::default(),
        },
        DomainProgram {
            name: "p2".into(),
            instance: Arc::new(instantiate(p2).expect("instantiate p2")),
            limits: Limits::default(),
        },
    ]
}

#[test]
fn driver_chains_two_phases_passing_output_to_input() {
    let m = parse_module(DRIVER).expect("parse driver");
    let inst = instantiate_with_imports(m, registry()).expect("instantiate driver");
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        let r = inst
            .run_with_caps(
                backend,
                &RunConfig::default(),
                &[("exec", domain_exec(phases()))],
            )
            .unwrap_or_else(|e| panic!("{backend:?}: {e}"));
        assert_eq!(
            r.stdout, b"aa!",
            "{backend:?}: p2 consumed p1's doubled output (a -> aa -> aa!)"
        );
        assert_eq!(
            r.outcome,
            Outcome::Exited(3),
            "{backend:?}: 1 -> 2 -> 3 bytes witnesses the phase hand-off"
        );
    }
}
