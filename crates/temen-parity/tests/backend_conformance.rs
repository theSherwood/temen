//! **The honesty pin for the frontier matrix's `backend` column (#1413).**
//!
//! INVARIANTS #14's first axis asks whether a capability is carried by every runtime backend:
//! *"bytecode interpreter, Cranelift JIT, wasm-JIT: each runs it identically or declines to the
//! oracle (invariant 9), never a limiting workaround."* The tree-walk interpreter **is** the oracle,
//! so it defines the answers and the other three are measured against it.
//!
//! `OPS_PARITY.md` already machine-checks this axis at **op** granularity, but it cannot answer the
//! capability question: `call.cap` is one row there, so the whole powerbox shares a single cell. What
//! that row does say is that the two interpreter tiers and Cranelift run `call.cap` while the wasm-JIT
//! declines it — it is a leaf accelerator that folds cap calls to the interpreter underneath. So this
//! test measures the three that run it, and pins the fourth's decline rather than restating it.
//!
//! ## What is compared
//!
//! Answers, not refusals — the same rule the concurrency column learned. A backend that returns a
//! *different value* is the interesting failure (`Answered(-22)` vs `Trapped(..)` are different
//! answers to the same question); a backend that declines to compile the module has folded to the
//! oracle exactly as invariant 9 allows, and conforms.
//!
//! Each `(op, argc, result type)` triple is run on each backend and compared to its own twin on the
//! oracle, because the lowerings are arity- and signature-guarded: folding each backend's *best*
//! result across arities would compare an arity one backend reached against a different one another
//! did, manufacturing divergences that are really the harness miscalling. A shape that cap-faults on
//! the oracle is the harness missing the arm, and is skipped.
//!
//! ## The audit rule
//!
//! As on the other columns, a row is scored only when the harness could actually mint its handle and
//! drive its ops. `Jit`, `JitCode`, `Offer` and `LiveImpl` need a live unit or a live peer, so they
//! stay `Unaudited` here exactly as they do on the debugger and concurrency columns.

mod support;
use std::ffi::c_void;
use support::capability_probe::{probe_module_typed, rows, Row, MAX_ARGC, PROBE_BEYOND};
use temen_interp::{bytecode, Host, Trap, Value};
use temen_parity::frontier::{capability_axes, Axis, Capability};
use temen_parity::Status;

/// What one backend did with one call shape. Compared for equality against the oracle's verdict, so
/// the answer's *content* matters.
///
/// The trap vocabularies differ by engine — the interpreter's `Trap` and the JIT's `TrapKind` are
/// separate types, and the JIT reports a guest `exit` as an outcome rather than a trap — so a verdict
/// names the *outcome*, not the engine's spelling of it. Comparing `Debug` strings across those two
/// universes reports `Exit(0)` against `Exited(0)` as a divergence, which is the harness talking.
#[derive(Clone, PartialEq, Eq, Debug)]
enum Verdict {
    /// The backend refused the module — it folds to the oracle (invariant 9). Conforming.
    Declined,
    /// The call never reached a dispatch arm: the shape does not match the op's signature, so the
    /// harness miscalled it. Reported as `CapFault` by one engine and `Malformed` by another, and
    /// skipped either way — never compared.
    Miscalled,
    /// The guest exited with this code. Terminal, but not an error.
    Exited(i32),
    /// The backend ran the op and it trapped, killing the domain.
    Trapped(String),
    /// The backend ran the op and it returned. A `-errno` counts: that is an answer.
    Answered(i64),
}

fn verdict(out: Result<Vec<Value>, Trap>) -> Verdict {
    match out {
        Err(Trap::CapFault) | Err(Trap::Malformed) => Verdict::Miscalled,
        Err(Trap::Exit(code)) => Verdict::Exited(code),
        Err(t) => Verdict::Trapped(format!("{t:?}")),
        Ok(v) => Verdict::Answered(match v.first() {
            Some(Value::I64(x)) => *x,
            Some(Value::I32(x)) => *x as i64,
            other => panic!("the probe entry returns a scalar, got {other:?}"),
        }),
    }
}

/// The tree-walk interpreter — the oracle (DESIGN §3): its answers define the axis.
fn oracle(mut host: Host, handle: i32, iface: u32, op: u32, argc: usize, res: &str) -> Verdict {
    let m = probe_module_typed(iface, op, argc, res);
    let mut fuel = 1_000_000u64;
    verdict(temen_interp::run_with_host(
        &m,
        0,
        &[Value::I32(handle)],
        &mut fuel,
        &mut host,
    ))
}

/// The bytecode interpreter, held bit-exact against the oracle.
fn bytecode(mut host: Host, handle: i32, iface: u32, op: u32, argc: usize, res: &str) -> Verdict {
    let m = probe_module_typed(iface, op, argc, res);
    let mut fuel = 1_000_000u64;
    match bytecode::compile_and_run_with_host(&m, 0, &[Value::I32(handle)], &mut fuel, &mut host) {
        None => Verdict::Declined, // outside the bytecode subset — folds to the oracle
        Some(out) => verdict(out),
    }
}

/// The Cranelift JIT, driven through the reference host trampoline an embedder supplies.
fn cranelift(mut host: Host, handle: i32, iface: u32, op: u32, argc: usize, res: &str) -> Verdict {
    let m = probe_module_typed(iface, op, argc, res);
    // SAFETY: `cap_thunk`'s contract — `ctx` is a live `*mut Host` that outlives the run below.
    let out = temen_jit::compile_and_run_with_host(
        &m,
        0,
        &[handle as i64],
        temen_run::cap_thunk,
        &mut host as *mut Host as *mut c_void,
    );
    match out {
        Err(temen_jit::JitError::Unsupported(_)) => Verdict::Declined,
        Err(e) => panic!("cranelift returned an unexpected (non-Unsupported) error: {e:?}"),
        Ok(temen_jit::JitOutcome::Returned(vals)) => {
            Verdict::Answered(vals.first().copied().unwrap_or(0))
        }
        Ok(temen_jit::JitOutcome::Exited(code)) => Verdict::Exited(code),
        Ok(temen_jit::JitOutcome::Trapped(temen_jit::TrapKind::CapFault)) => Verdict::Miscalled,
        Ok(temen_jit::JitOutcome::Trapped(t)) => Verdict::Trapped(format!("{t:?}")),
    }
}

/// Every `(argc, result type)` shape the sweep tries.
fn shapes() -> impl Iterator<Item = (usize, &'static str)> {
    (0..MAX_ARGC).flat_map(|a| ["i64", "i32"].map(move |r| (a, r)))
}

/// The call shapes of `op` on which a backend answers differently from the oracle, as
/// `(shape, backend name, oracle, backend)`.
fn divergences(row: &Row, op: u32) -> Vec<(String, &'static str, Verdict, Verdict)> {
    let mut out = Vec::new();
    for (argc, res) in shapes() {
        let (h, x) = (row.mint)();
        let want = oracle(h, x, row.iface, op, argc, res);
        // The oracle found no arm: the harness miscalled this shape, so there is nothing to compare.
        if want == Verdict::Miscalled {
            continue;
        }
        for (name, run) in [
            (
                "bytecode",
                bytecode as fn(Host, i32, u32, u32, usize, &str) -> Verdict,
            ),
            ("cranelift", cranelift),
        ] {
            let (h, x) = (row.mint)();
            let got = run(h, x, row.iface, op, argc, res);
            // Declining is conforming: the backend folded to the oracle (invariant 9).
            if got == Verdict::Declined || got == want {
                continue;
            }
            out.push((
                format!("op {op}, argc {argc}, -> {res}"),
                name,
                want.clone(),
                got,
            ));
        }
    }
    out
}

fn col(a: Axis) -> usize {
    Axis::ALL.iter().position(|x| *x == a).expect("axis listed")
}

/// **The pin.** For every drivable capability, the manifest's `backend` cell must match whether the
/// backends that run cap calls actually answer as the oracle does.
#[test]
fn the_backend_column_matches_what_the_backends_actually_do() {
    let k = col(Axis::RuntimeBackend);
    for row in rows() {
        let cell = capability_axes(row.cap)[k];
        let ops: Vec<u32> = if row.ops.is_empty() {
            // A capability with no callable ops: sweep a little way past its (empty) interface, so
            // "nothing to diverge about" is checked rather than assumed.
            (0..PROBE_BEYOND).collect()
        } else {
            row.ops.to_vec()
        };
        let found: Vec<_> = ops.iter().flat_map(|op| divergences(&row, *op)).collect();
        let observed = if found.is_empty() {
            Status::Full
        } else {
            Status::NotYet
        };
        assert_eq!(
            observed,
            cell.status,
            "{}: the frontier manifest says `{:?}` on the backend axis, but the backends {}.\n\
             Divergences found:\n{}",
            row.cap.name(),
            cell.status,
            if found.is_empty() {
                "answer as the oracle does on every call shape"
            } else {
                "diverge"
            },
            found
                .iter()
                .map(|(shape, who, want, got)| format!(
                    "  {shape}: oracle={want:?} {who}={got:?}\n"
                ))
                .collect::<String>(),
        );
    }
}

/// The divergence the column's first rendering found, pinned as a *specific* fact rather than left
/// as a bare `NotYet`: `join` (op 1) on a child handle that names no live child traps `ThreadFault`
/// on the oracle and `CapFault` under the Cranelift thunk. Invariant 9 lets a backend decline to the
/// oracle; it does not let one run the op and report a different failure, and `instantiator_rt.rs`
/// documents its arm as "matching the interpreter". Tracked as #1573; this keeps the shape of it
/// honest so a fix (or a regression) is visible here.
#[test]
fn join_traps_differently_on_the_oracle_and_the_cranelift_thunk() {
    let row = rows()
        .into_iter()
        .find(|r| r.cap == Capability::Instantiator)
        .expect("the Instantiator row");
    let found = divergences(&row, 1);
    assert!(
        !found.is_empty(),
        "join no longer diverges — if it was fixed, this pin and the Instantiator cell should move \
         together"
    );
    for (shape, who, want, got) in &found {
        assert_eq!(
            *want,
            Verdict::Trapped("ThreadFault".into()),
            "{shape}: the oracle should be the one raising ThreadFault, got {want:?}"
        );
        assert_eq!(
            (*who, got.clone()),
            ("cranelift", Verdict::Miscalled),
            "{shape}: the divergence should be cranelift reporting a cap fault, got {who}={got:?}"
        );
    }
}

/// The wasm-JIT's side of this axis, pinned rather than restated: it is a leaf accelerator that does
/// not emit `call.cap` at all, so every capability folds to the interpreter underneath it. That is a
/// decline in invariant 9's sense, and `OPS_PARITY.md`'s `call.cap` row is its op-granularity twin —
/// this keeps the capability-granularity claim from going stale if that ever changes.
#[test]
fn the_wasm_jit_declines_every_capability_call() {
    for row in rows() {
        let op = row.ops.first().copied().unwrap_or(0);
        let m = probe_module_typed(row.iface, op, 0, "i64");
        assert!(
            temen_wasm_jit::compile_module(&m).is_err(),
            "{}: the wasm-JIT emitted a module containing `call.cap` — if it now emits cap calls, \
             the backend column must measure it rather than record the fold",
            row.cap.name(),
        );
    }
}
