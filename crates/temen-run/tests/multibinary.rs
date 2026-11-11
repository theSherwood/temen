//! W4 (NIM.md §3c) mechanism proof — a multi-*binary* compiler-driver shape on temen, decoupled
//! from the real nimony toolchain. nimony's `nifmake` spawns `nifler → nimony → hexer → lengc` as
//! subprocesses, each consuming the previous phase's output file. This test proves that
//! orchestration runs on temen today over an **existing seam** — the `exec` capability's
//! `domain_exec` backend (EXEC.md) — with **no new host op**: a driver module resolves `"exec"`,
//! runs phase `p1` with a seed input, drains its captured output, and feeds that output as the
//! **input** of phase `p2`, then emits `p2`'s result. Both phases and the driver are pure Temen
//! modules; each phase runs as an isolated child domain (own window, powerbox, fuel), exactly as a
//! spawned process would. This is the W4 analog of Path B's runtime shim: the mechanism, proven
//! with stand-in phases before the real phase binaries are compiled.
//!
//! The composition is content- and order-sensitive so it proves data *flowed*, not just that two
//! children ran: `p1` **doubles** its input, `p2` **echoes then appends `!`**. Seed `a` →
//! `p1` → `aa` → `p2` → `aa!`. If `p2` had been handed the original seed instead of `p1`'s output,
//! the result would be `a!` (2 bytes), not `aa!` (3) — so `aa!`/exit 3 witnesses the hand-off.

use std::sync::Arc;
use temen_run::exec::{domain_exec, domain_exec_with_fs, DomainProgram};
use temen_run::{
    instantiate, instantiate_with_imports, Backend, HostCap, Imports, Limits, Outcome, RunConfig,
};
use temen_text::parse_module;

/// Phase `p1` — a child domain that **doubles** its stdin: read up to 32 bytes into 64, then write
/// them back twice. `read`/`write` bind to the child's seeded stdin / captured stdout (the domain
/// runner wires them, exactly as `ECHO_DOMAIN`/`CAT_DOMAIN` in `exec_cap.rs`).
const P1_DOUBLE: &str = "\
memory 16
import 0 \"read\" (i64, i64) -> (i64)
import 1 \"write\" (i64, i64) -> (i64)
func 0 () -> () {
block 0 () {
  vp = i64.const 16448
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
data 16384 \"!\"
import 0 \"read\" (i64, i64) -> (i64)
import 1 \"write\" (i64, i64) -> (i64)
func 0 () -> () {
block 0 () {
  vp = i64.const 16448
  vc = i64.const 32
  vn = call.import 0 (vp, vc)
  vw1 = call.import 1 (vp, vn)
  vbp = i64.const 16384
  vbl = i64.const 1
  vw2 = call.import 1 (vbp, vbl)
  return
  }
}
export 0 func \"_start\" 0
";

/// #2962 — every driver maps the `"exec.msg"` region the `exec` grant brings at window 65536 (the
/// upper half of a `memory 17` window) and passes the protocol region-relative offsets: op 0
/// `run(off, argv_len, stdin_len)` (argv at `off`, its stdin right after), 1 `read_out(job, off,
/// cap)`, 3 `status(job)`. The phase names sit in data
/// segments and are copied into the region; output lands in the region. `"exec.msg"` is the data at
/// 16384 (above the #1094 NULL guard), whose first four bytes spell `"exec"`.
///
/// The driver — the `nifmake` analog. Run `p1` (region 0) with stdin `a`; read `p1`'s output
/// **straight into `p2`'s stdin slot** (region 258, after `p2`'s name at 256), so the hand-off copies
/// nothing; run `p2`; read its output to region 512; emit it and exit with its length.
const DRIVER: &str = "\
memory 17
data 16384 \"exec.msg\"
data 16392 \"p1\"
data 16396 \"p2\"
data 16400 \"a\"
import 0 \"out\" (i64, i64) -> (i64)
import 1 \"exit\" (i32) -> ()
func 0 () -> () {
block 0 () {
  vp = i64.const 16384
  vl = i64.const 4
  vh = self.resolve vp vl
  vml = i64.const 8
  vmsg = self.resolve vp vml
  vwin = i64.const 65536
  vz = i64.const 0
  vlen = i64.const 65536
  vprot = i32.const 3
  vmapped = call.cap 4 0 (i64, i64, i64, i32) -> (i64) vmsg (vwin, vz, vlen, vprot)
  vs1 = i64.const 16392
  vtwo = i64.const 2
  mem.copy vwin vs1 vtwo
  vd1 = i64.const 65538
  vsa = i64.const 16400
  vone = i64.const 1
  mem.copy vd1 vsa vone
  vd2 = i64.const 65792
  vs2 = i64.const 16396
  mem.copy vd2 vs2 vtwo
  vrun = i64.const 0
  vjob1 = call.cap 13 0 (i64, i64, i64) -> (i64) vh (vz, vtwo, vone)
  vread = i64.const 1
  vin2 = i64.const 258
  vcap = i64.const 16
  vn1 = call.cap 13 1 (i64, i64, i64) -> (i64) vh (vjob1, vin2, vcap)
  vo2 = i64.const 256
  vjob2 = call.cap 13 0 (i64, i64, i64) -> (i64) vh (vo2, vtwo, vn1)
  vout = i64.const 512
  vn2 = call.cap 13 1 (i64, i64, i64) -> (i64) vh (vjob2, vout, vcap)
  vwout = i64.const 66048
  vw = call.import 0 (vwout, vn2)
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

// ---------------------------------------------------------------------------
// The full nifmake shape: four phases, run-and-check-exit, abort on failure.
//
// The two-phase test above proves the data hand-off. A real `nifmake` driver
// does more than pipe: it runs each phase, **checks its exit status, and stops
// the pipeline if a phase fails** (a broken `nifler` must not feed garbage to
// `nimony`). This test proves that control flow at nimony's real phase depth
// (four: nifler -> nimony -> hexer -> lengc), still entirely within the tested
// `exec` seam (ops: 0 = run, 1 = read_out, 3 = status) — no new infra.
//
// Each phase appends its own letter to its stdin, so the happy path composes
// `a -> ab -> abc -> abcd -> abcde` and witnesses the 4-deep hand-off. Swapping
// one phase for a version that exits non-zero makes the driver short-circuit:
// it emits nothing and exits with the failed phase's code, and the later phases
// never run. The driver module is identical across both — only the registry
// differs — so the abort is the driver reacting to status, not a different path.

/// A stand-in phase: echo stdin, then append `letter` (its own 1-byte data). Exit 0.
fn appender(letter: &str) -> String {
    format!(
        "\
memory 16
data 16384 \"{letter}\"
import 0 \"read\" (i64, i64) -> (i64)
import 1 \"write\" (i64, i64) -> (i64)
func 0 () -> () {{
block 0 () {{
  vp = i64.const 16448
  vc = i64.const 32
  vn = call.import 0 (vp, vc)
  vw1 = call.import 1 (vp, vn)
  vbp = i64.const 16384
  vbl = i64.const 1
  vw2 = call.import 1 (vbp, vbl)
  return
  }}
}}
export 0 func \"_start\" 0
"
    )
}

/// A phase that **fails**: its entry returns 7, so the child domain exits 7 (a
/// clean non-zero exit, not a trap) — the `domain_exec` outcome table maps the
/// first returned value to the exit code (`exec.rs:106`).
const P3_FAIL: &str = "\
memory 16
func 0 () -> (i64) {
block 0 () {
  vcode = i64.const 7
  return vcode
  }
}
export 0 func \"_start\" 0
";

/// The four-phase driver — the `nifmake` analog with abort-on-failure. Data (above the #1094 NULL
/// guard): phase names `p1`@16392 `p2`@16396 `p3`@16400 `p4`@16404; seed `a`@16408. Block 0 maps the
/// message region and lays out one 256-byte slot per phase: its name at `256k`, its stdin after it.
/// Phase `k` (block `k + 1`) runs slot `k`, reads its output straight into slot `k + 1`'s stdin, reads
/// its status, and `br_if status` to the abort block (6) or the next phase. Block 5 emits the final
/// output (slot 4's stdin) + exit 0; block 6 emits nothing and exits with the failed phase's status.
/// The `exec` handle is re-resolved per block (idempotent by name), so only the inter-phase length
/// is threaded.
const DRIVER4: &str = "\
memory 17
data 16384 \"exec.msg\"
data 16392 \"p1\"
data 16396 \"p2\"
data 16400 \"p3\"
data 16404 \"p4\"
data 16408 \"a\"
import 0 \"out\" (i64, i64) -> (i64)
import 1 \"exit\" (i32) -> ()
func 0 () -> () {
block 0 () {
  vp = i64.const 16384
  vl = i64.const 4
  vh = self.resolve vp vl
  vml = i64.const 8
  vmsg = self.resolve vp vml
  vwin = i64.const 65536
  vz = i64.const 0
  vlen = i64.const 65536
  vprot = i32.const 3
  vmapped = call.cap 4 0 (i64, i64, i64, i32) -> (i64) vmsg (vwin, vz, vlen, vprot)
  vn0 = i64.const 16392
  vd0 = i64.const 65536
  vtwo0 = i64.const 2
  mem.copy vd0 vn0 vtwo0
  vn1 = i64.const 16396
  vd1 = i64.const 65792
  vtwo1 = i64.const 2
  mem.copy vd1 vn1 vtwo1
  vn2 = i64.const 16400
  vd2 = i64.const 66048
  vtwo2 = i64.const 2
  mem.copy vd2 vn2 vtwo2
  vn3 = i64.const 16404
  vd3 = i64.const 66304
  vtwo3 = i64.const 2
  mem.copy vd3 vn3 vtwo3
  vsa = i64.const 16408
  vda = i64.const 65538
  vone = i64.const 1
  mem.copy vda vsa vone
  br 1(vone)
}
block 1 (q0: i64) {
  vp0 = i64.const 16384
  vl0 = i64.const 4
  vh0 = self.resolve vp0 vl0
  vrun0 = i64.const 0
  vo0 = i64.const 0
  vln0 = i64.const 2
  vj0 = call.cap 13 0 (i64, i64, i64) -> (i64) vh0 (vo0, vln0, q0)
  vread0 = i64.const 1
  vb0 = i64.const 258
  vc0 = i64.const 32
  vr0 = call.cap 13 1 (i64, i64, i64) -> (i64) vh0 (vj0, vb0, vc0)
  vst0 = i64.const 3
  vs0 = call.cap 13 3 (i64) -> (i64) vh0 (vj0)
  vs0w = i32.wrap_i64 vs0
  br_if vs0w 6(vs0) 2(vr0)
}
block 2 (q1: i64) {
  vp1 = i64.const 16384
  vl1 = i64.const 4
  vh1 = self.resolve vp1 vl1
  vrun1 = i64.const 0
  vo1 = i64.const 256
  vln1 = i64.const 2
  vj1 = call.cap 13 0 (i64, i64, i64) -> (i64) vh1 (vo1, vln1, q1)
  vread1 = i64.const 1
  vb1 = i64.const 514
  vc1 = i64.const 32
  vr1 = call.cap 13 1 (i64, i64, i64) -> (i64) vh1 (vj1, vb1, vc1)
  vst1 = i64.const 3
  vs1 = call.cap 13 3 (i64) -> (i64) vh1 (vj1)
  vs1w = i32.wrap_i64 vs1
  br_if vs1w 6(vs1) 3(vr1)
}
block 3 (q2: i64) {
  vp2 = i64.const 16384
  vl2 = i64.const 4
  vh2 = self.resolve vp2 vl2
  vrun2 = i64.const 0
  vo2 = i64.const 512
  vln2 = i64.const 2
  vj2 = call.cap 13 0 (i64, i64, i64) -> (i64) vh2 (vo2, vln2, q2)
  vread2 = i64.const 1
  vb2 = i64.const 770
  vc2 = i64.const 32
  vr2 = call.cap 13 1 (i64, i64, i64) -> (i64) vh2 (vj2, vb2, vc2)
  vst2 = i64.const 3
  vs2 = call.cap 13 3 (i64) -> (i64) vh2 (vj2)
  vs2w = i32.wrap_i64 vs2
  br_if vs2w 6(vs2) 4(vr2)
}
block 4 (q3: i64) {
  vp3 = i64.const 16384
  vl3 = i64.const 4
  vh3 = self.resolve vp3 vl3
  vrun3 = i64.const 0
  vo3 = i64.const 768
  vln3 = i64.const 2
  vj3 = call.cap 13 0 (i64, i64, i64) -> (i64) vh3 (vo3, vln3, q3)
  vread3 = i64.const 1
  vb3 = i64.const 1026
  vc3 = i64.const 32
  vr3 = call.cap 13 1 (i64, i64, i64) -> (i64) vh3 (vj3, vb3, vc3)
  vst3 = i64.const 3
  vs3 = call.cap 13 3 (i64) -> (i64) vh3 (vj3)
  vs3w = i32.wrap_i64 vs3
  br_if vs3w 6(vs3) 5(vr3)
}
block 5 (q4: i64) {
  vb4 = i64.const 66562
  vw4 = call.import 0 (vb4, q4)
  vz4 = i32.const 0
  call.import 1 (vz4)
  unreachable
}
block 6 (code: i64) {
  vcw = i32.wrap_i64 code
  call.import 1 (vcw)
  unreachable
}
}
export 0 func \"_start\" 0
";

/// Registry of four appender phases `p1..p4` (letters b,c,d,e). If `fail_at_p3`, `p3` is swapped for
/// the failing phase — same names, so the driver is unchanged.
fn phases4(fail_at_p3: bool) -> Vec<DomainProgram> {
    let mk = |name: &str, src: String| DomainProgram {
        name: name.into(),
        instance: Arc::new(
            instantiate(parse_module(&src).expect("parse phase")).expect("instantiate phase"),
        ),
        limits: Limits::default(),
    };
    let p3 = if fail_at_p3 {
        P3_FAIL.to_string()
    } else {
        appender("d")
    };
    vec![
        mk("p1", appender("b")),
        mk("p2", appender("c")),
        mk("p3", p3),
        mk("p4", appender("e")),
    ]
}

#[test]
fn four_phase_pipeline_composes_when_every_phase_succeeds() {
    let m = parse_module(DRIVER4).expect("parse driver4");
    let inst = instantiate_with_imports(m, registry()).expect("instantiate driver4");
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        let r = inst
            .run_with_caps(
                backend,
                &RunConfig::default(),
                &[("exec", domain_exec(phases4(false)))],
            )
            .unwrap_or_else(|e| panic!("{backend:?}: {e}"));
        assert_eq!(
            r.stdout, b"abcde",
            "{backend:?}: 4-deep hand-off a -> ab -> abc -> abcd -> abcde"
        );
        assert_eq!(r.outcome, Outcome::Exited(0), "{backend:?}: all phases ok");
    }
}

#[test]
fn pipeline_aborts_with_the_failed_phase_status_and_skips_later_phases() {
    let m = parse_module(DRIVER4).expect("parse driver4");
    let inst = instantiate_with_imports(m, registry()).expect("instantiate driver4");
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        let r = inst
            .run_with_caps(
                backend,
                &RunConfig::default(),
                &[("exec", domain_exec(phases4(true)))],
            )
            .unwrap_or_else(|e| panic!("{backend:?}: {e}"));
        assert_eq!(
            r.stdout, b"",
            "{backend:?}: aborted before the success emit — p4 never ran"
        );
        assert_eq!(
            r.outcome,
            Outcome::Exited(7),
            "{backend:?}: driver exits with the failed phase's status (p3 -> 7)"
        );
    }
}

// ---------------------------------------------------------------------------
// W4 step (iii): a *real* compiled program as an isolated exec child.
//
// The phases above are hand-written stand-ins — they prove the driver's control flow (hand-off,
// status, abort), but not that a program produced by the actual `Leng → TEMEN-IR` backend boots and
// runs correctly as a `domain_exec` child. This closes that gap on a real compiled binary: a Leng
// module with real control flow (a counted `while` loop over scalar locals) is lowered by
// `temen-leng`, verified, registered as a phase, and `exec`d by the same nifmake-shaped driver. Its
// `main()` returns 55 (the sum 1..10), and `domain_exec` maps that return to the child's exit code,
// which the driver reads via the status op (13/3) and re-exits with — so exit 55 witnesses the whole
// path: frontend output → verify → isolated child domain → correct compute → status back to driver.
//
// Scope: this is a **self-contained compute** program (no heap, no host imports beyond the child's
// default stdin/stdout). A phase that uses the real allocator/`system` runtime additionally needs
// either a self-seeding in-window allocator (the Path-B shim, made startup-seeding) or — for the
// mmap-backed `system` — the bottom-edge host caps granted to the child, which `domain_exec` does
// not do today (it runs children with only stdin/stdout). That child-capability grant is the same
// owner-reviewed infra as the shared-memfs hand-off (NIM.md §3c), so it is scoped there, not here.

/// A real Leng compute program: `main()` sums 1..10 with a `while` loop and returns 55. Its sole
/// proc becomes func 0 — the entry `domain_exec` runs — and it takes no params and imports nothing,
/// so it runs standalone in a fresh child window. (Same shape temen-leng's own `whole_module` tests
/// compile and run; here it runs *as an exec child*.)
const SUM_1_TO_10_LENG: &str = "\
(stmts
 (proc :main.0. . (i +64) .
  (stmts .
   (var :s.0 . (i +64) 0)
   (var :i.0 . (i +64) 1)
   (while (lt i.0 11)
    (stmts .
     (asgn s.0 (add (i +64) s.0 i.0))
     (asgn i.0 (add (i +64) i.0 1))))
   (ret s.0))))";

/// The nifmake-shaped driver for one real phase: resolve `exec`, run `compute` with no stdin, read
/// its exit status (op 3), and re-exit with it.
const DRIVER_REAL: &str = "\
memory 17
data 16384 \"exec.msg\"
data 16392 \"compute\"
import 0 \"exit\" (i32) -> ()
func 0 () -> () {
block 0 () {
  vp = i64.const 16384
  vl = i64.const 4
  vh = self.resolve vp vl
  vml = i64.const 8
  vmsg = self.resolve vp vml
  vwin = i64.const 65536
  vz = i64.const 0
  vlen = i64.const 65536
  vprot = i32.const 3
  vmapped = call.cap 4 0 (i64, i64, i64, i32) -> (i64) vmsg (vwin, vz, vlen, vprot)
  vname = i64.const 16392
  vnamel = i64.const 7
  mem.copy vwin vname vnamel
  vrun = i64.const 0
  vjob = call.cap 13 0 (i64, i64, i64) -> (i64) vh (vz, vnamel, vz)
  vst = i64.const 3
  vstatus = call.cap 13 3 (i64) -> (i64) vh (vjob)
  vcode = i32.wrap_i64 vstatus
  call.import 0 (vcode)
  unreachable
  }
}
export 0 func \"_start\" 0
";

/// Build the `compute` phase by compiling `SUM_1_TO_10_LENG` through the real temen-leng backend and
/// verifying it (the frontend is untrusted; the verifier re-checks — DESIGN.md §2a).
fn compiled_compute_phase() -> DomainProgram {
    let module = temen_leng::translate(SUM_1_TO_10_LENG).expect("temen-leng compiles the program");
    temen_verify::verify_module(&module).expect("the compiled program verifies");
    DomainProgram {
        name: "compute".into(),
        instance: Arc::new(instantiate(module).expect("instantiate compute")),
        limits: Limits::default(),
    }
}

#[test]
fn driver_runs_a_real_temen_leng_compiled_program_as_an_exec_child() {
    let m = parse_module(DRIVER_REAL).expect("parse real driver");
    let inst = instantiate_with_imports(m, registry()).expect("instantiate driver");
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        let r = inst
            .run_with_caps(
                backend,
                &RunConfig::default(),
                &[("exec", domain_exec(vec![compiled_compute_phase()]))],
            )
            .unwrap_or_else(|e| panic!("{backend:?}: {e}"));
        assert_eq!(
            r.outcome,
            Outcome::Exited(55),
            "{backend:?}: real compiled main() summed 1..10 in an isolated child; status reached the driver"
        );
    }
}

// ---------------------------------------------------------------------------
// W4, the faithful `nifmake` hand-off: phases pass an intermediate **file** through a shared memfs.
//
// The pipelines above hand off via stdout→stdin piping. Real `nifmake` phases pass *files*
// (`.p.nif`/`.s.nif`/…) down the chain, and this is that shape: `domain_exec_with_fs` grants every
// child one shared in-memory filesystem, so phase `gen` writes `mid` and phase `use` reads it — the
// data crosses the child boundary through the file, not a pipe. The child-fs grant is a deliberate,
// explicit widening of child authority (NIM.md §3c): plain `domain_exec` gives children only
// stdin/stdout; here the embedder hands the phases one seeded, in-memory store and nothing else.
//
// `gen` writes the bytes "OK" to `mid` and exits 0; `use` opens `mid`, reads it, echoes it to stdout,
// and exits with the byte count; the driver runs both, then emits what `use` produced. Driver stdout
// == "OK" witnesses that the file `gen` wrote reached `use` through the shared store — a fresh store
// per run, so `use` reads *this* run's write, not a leftover. (fs cap = iface 13, ops FS_OPEN=0,
// FS_READ=1, FS_WRITE=2, FS_CLOSE=4; open flags O_READ=1, O_CREATE|O_WRITE=18.)

/// Phase `gen`: create `mid` (O_CREATE|O_WRITE) and write "OK" into it.
const GEN_PHASE: &str = "\
memory 16
data 16384 \"fs\"
data 16388 \"mid\"
data 16392 \"OK\"
func 0 () -> (i64) {
block 0 () {
  vfp = i64.const 16384
  vfl = i64.const 2
  vfs = self.resolve vfp vfl
  vpath = i64.const 16388
  vplen = i64.const 3
  vflags = i64.const 18
  vfd = call.cap 13 0 (i64, i64, i64) -> (i64) vfs (vpath, vplen, vflags)
  vbuf = i64.const 16392
  vblen = i64.const 2
  vn = call.cap 13 2 (i64, i64, i64) -> (i64) vfs (vfd, vbuf, vblen)
  vcl = call.cap 13 4 (i64) -> (i64) vfs (vfd)
  vz = i64.const 0
  return vz
  }
}
export 0 func \"_start\" 0
";

/// Phase `use`: open `mid` (O_READ), read it into a buffer, echo it to stdout, exit with the count.
const USE_PHASE: &str = "\
memory 16
data 16384 \"fs\"
data 16388 \"mid\"
import 0 \"write\" (i64, i64) -> (i64)
func 0 () -> (i64) {
block 0 () {
  vfp = i64.const 16384
  vfl = i64.const 2
  vfs = self.resolve vfp vfl
  vpath = i64.const 16388
  vplen = i64.const 3
  vflags = i64.const 1
  vfd = call.cap 13 0 (i64, i64, i64) -> (i64) vfs (vpath, vplen, vflags)
  vbuf = i64.const 16448
  vcap = i64.const 32
  vn = call.cap 13 1 (i64, i64, i64) -> (i64) vfs (vfd, vbuf, vcap)
  vw = call.import 0 (vbuf, vn)
  vcl = call.cap 13 4 (i64) -> (i64) vfs (vfd)
  return vn
  }
}
export 0 func \"_start\" 0
";

/// The driver: run `gen` (writes `mid`), then `use` (reads `mid`), drain `use`'s output and emit it.
const DRIVER_FS: &str = "\
memory 17
data 16384 \"exec.msg\"
data 16392 \"gen\"
data 16396 \"use\"
import 0 \"out\" (i64, i64) -> (i64)
import 1 \"exit\" (i32) -> ()
func 0 () -> () {
block 0 () {
  vp = i64.const 16384
  vl = i64.const 4
  vh = self.resolve vp vl
  vml = i64.const 8
  vmsg = self.resolve vp vml
  vwin = i64.const 65536
  vz = i64.const 0
  vlen = i64.const 65536
  vprot = i32.const 3
  vmapped = call.cap 4 0 (i64, i64, i64, i32) -> (i64) vmsg (vwin, vz, vlen, vprot)
  vgen = i64.const 16392
  vthree = i64.const 3
  mem.copy vwin vgen vthree
  vd2 = i64.const 65792
  vuse = i64.const 16396
  mem.copy vd2 vuse vthree
  vrun = i64.const 0
  vjg = call.cap 13 0 (i64, i64, i64) -> (i64) vh (vz, vthree, vz)
  vst = i64.const 3
  vsg = call.cap 13 3 (i64) -> (i64) vh (vjg)
  vo2 = i64.const 256
  vju = call.cap 13 0 (i64, i64, i64) -> (i64) vh (vo2, vthree, vz)
  vread = i64.const 1
  vout = i64.const 512
  vcap = i64.const 32
  vnu = call.cap 13 1 (i64, i64, i64) -> (i64) vh (vju, vout, vcap)
  vwout = i64.const 66048
  vw = call.import 0 (vwout, vnu)
  vsu = call.cap 13 3 (i64) -> (i64) vh (vju)
  vcode = i32.wrap_i64 vsu
  call.import 1 (vcode)
  unreachable
  }
}
export 0 func \"_start\" 0
";

fn fs_phase(name: &str, src: &str) -> DomainProgram {
    DomainProgram {
        name: name.into(),
        instance: Arc::new(
            instantiate(parse_module(src).expect("parse fs phase")).expect("instantiate"),
        ),
        limits: Limits::default(),
    }
}

#[test]
fn driver_hands_off_a_file_between_phases_through_a_shared_memfs() {
    let m = parse_module(DRIVER_FS).expect("parse fs driver");
    let inst = instantiate_with_imports(m, registry()).expect("instantiate driver");
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        // A fresh shared store per run: both phase children are granted one grant each from the same
        // factory, so `gen`'s write and `use`'s read hit the same filesystem.
        let (factory, _handle) = temen_run::fs::mem_fs_shared_factory(vec![], vec![]);
        let child_fs = HostCap::host_proc(0, move |_, _| Some(factory()));
        let phases = vec![fs_phase("gen", GEN_PHASE), fs_phase("use", USE_PHASE)];
        let r = inst
            .run_with_caps(
                backend,
                &RunConfig::default(),
                &[("exec", domain_exec_with_fs(phases, child_fs))],
            )
            .unwrap_or_else(|e| panic!("{backend:?}: {e}"));
        assert_eq!(
            r.stdout, b"OK",
            "{backend:?}: 'use' read the file 'gen' wrote to the shared memfs (file hand-off)"
        );
    }
}

/// Confinement probe: resolve `"fs"` and return whether it was granted (1) or not (0). `self.resolve`
/// yields an i32 handle ≥ 0 on success, negative `-errno` when the name isn't granted.
const PROBE_PHASE: &str = "\
memory 16
data 16384 \"fs\"
func 0 () -> (i32) {
block 0 () {
  vfp = i64.const 16384
  vfl = i64.const 2
  vfs = self.resolve vfp vfl
  vzero = i32.const 0
  vok = i32.ge_s vfs vzero
  return vok
  }
}
export 0 func \"_start\" 0
";

/// A driver that runs `probe` and exits with its status (1 = it got `fs`, 0 = it didn't).
const DRIVER_PROBE: &str = "\
memory 17
data 16384 \"exec.msg\"
data 16392 \"probe\"
import 0 \"exit\" (i32) -> ()
func 0 () -> () {
block 0 () {
  vp = i64.const 16384
  vl = i64.const 4
  vh = self.resolve vp vl
  vml = i64.const 8
  vmsg = self.resolve vp vml
  vwin = i64.const 65536
  vz = i64.const 0
  vlen = i64.const 65536
  vprot = i32.const 3
  vmapped = call.cap 4 0 (i64, i64, i64, i32) -> (i64) vmsg (vwin, vz, vlen, vprot)
  vname = i64.const 16392
  vnamel = i64.const 5
  mem.copy vwin vname vnamel
  vrun = i64.const 0
  vj = call.cap 13 0 (i64, i64, i64) -> (i64) vh (vz, vnamel, vz)
  vst = i64.const 3
  vs = call.cap 13 3 (i64) -> (i64) vh (vj)
  vcode = i32.wrap_i64 vs
  call.import 0 (vcode)
  unreachable
  }
}
export 0 func \"_start\" 0
";

#[test]
fn a_child_gets_fs_only_when_the_parent_grants_it() {
    // The confinement invariant behind the file hand-off: `fs` is inherited by a child ONLY through an
    // explicit `domain_exec_with_fs` grant. A child under plain `domain_exec` cannot resolve `fs` — no
    // ambient authority, and a child cannot widen its own grant.
    let m = parse_module(DRIVER_PROBE).expect("parse probe driver");
    let inst = instantiate_with_imports(m, registry()).expect("instantiate driver");
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        let (factory, _h) = temen_run::fs::mem_fs_shared_factory(vec![], vec![]);
        let granted = inst
            .run_with_caps(
                backend,
                &RunConfig::default(),
                &[(
                    "exec",
                    domain_exec_with_fs(
                        vec![fs_phase("probe", PROBE_PHASE)],
                        HostCap::host_proc(0, move |_, _| Some(factory())),
                    ),
                )],
            )
            .unwrap_or_else(|e| panic!("{backend:?}: {e}"));
        assert_eq!(
            granted.outcome,
            Outcome::Exited(1),
            "{backend:?}: an explicitly-granted child resolves fs"
        );

        let default = inst
            .run_with_caps(
                backend,
                &RunConfig::default(),
                &[("exec", domain_exec(vec![fs_phase("probe", PROBE_PHASE)]))],
            )
            .unwrap_or_else(|e| panic!("{backend:?}: {e}"));
        assert_eq!(
            default.outcome,
            Outcome::Exited(0),
            "{backend:?}: a child under plain domain_exec has no fs (default confinement)"
        );
    }
}
