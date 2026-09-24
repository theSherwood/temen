//! **The honesty pin for the frontier matrix's `code origin` column (#1413 slice 4).**
//!
//! INVARIANTS #14's fourth axis asks whether a capability is "usable from a §22 guest-JIT unit as
//! from the base module". Both halves are drivable, so — like `concurrency` and unlike `target` —
//! the column can be *run* rather than asserted: mint the capability, make the identical
//! `call.cap`, once from a host-translated base module and once from inside a unit, and compare the
//! answers.
//!
//! ## Both routes into a unit
//!
//! DESIGN §22 gives a unit two routes, and they carry different contracts, so the unit side is a
//! [`Route`] parameter rather than one fixed path:
//!
//! - **Install** + `call.dyn` runs the unit in the caller's own frames. There it must answer exactly
//!   as the base module does — any difference is a gap.
//! - **Invoke** runs it as a seam-free leaf, where the whole `Instantiator` is unavailable by
//!   contract (#1578): a spawn would outlive the synchronous call over code nothing keeps. That one
//!   refusal — an `Instantiator` op that `CapFault`s — is a recorded decline, scored `Declines`;
//!   any other difference on this route is still a gap.
//!
//! The column first drove the invoke route alone, which is how it found the spawn family diverging
//! there and missed the install route diverging worse (#1726: each engine failed a same-module spawn
//! from an installed unit differently).
//!
//! ## Everything except the caller is held fixed
//!
//! The axis has exactly one variable, so the harness leaves nothing else free. Both sides run on
//! `bytecode::drive` (the only engine that runs units at all), over the same window, with the same
//! fuel — and, crucially, with the **same grant table**: the base side also grants the `Jit` and
//! compiles the very same unit, and simply never invokes it. Without that, the two sides' handle
//! numbering drifts by the grants the unit path needs, and every handle-minting op — `AddressSpace`
//! `sub`, `Budget` `split` — "diverges" by returning a different (equally correct) handle. Matching
//! the prefix is better than normalizing the answers: it keeps the comparison exact, so a real
//! difference in a minted handle would still be caught.
//!
//! ## Why refusal *kind* is not compared here, unlike on the concurrency column
//!
//! The two sides do not have the same refusal vocabulary available. A base module is handed to the
//! engine whole, so a call shape the bytecode subset has no arm for is refused at **compile** time
//! and the module never runs (`Verdict::Unsupported`, the embedder's cue to fall back to the
//! oracle). A unit is compiled behind `Jit.invoke`, so the identical refusal can only surface as a
//! `Trap::Malformed` out of the invoke. Those are one decision — "this engine has no arm for this
//! shape" — reported at the only point each path has, so they fold into one verdict. Which shapes
//! the bytecode subset covers is the *backend* axis's question (`OPS_PARITY.md`), and scoring it
//! again here would double-count it under a heading that is asking something else.
//!
//! What the column does compare is whether the guest gets an **answer**: a serviced value (a handle
//! or a `-errno`) on one side and a domain-killing trap on the other is the divergence this axis
//! exists to surface — the shape of what the column's first rendering found on the `Instantiator`
//! row.
//!
//! ## The audit rule
//!
//! As on the other driven columns, a row is scored only when the harness could mint its handle and
//! drive its ops. `Jit`, `JitCode`, `Offer` and `LiveImpl` need a live unit or a live peer, so they
//! stay `Unaudited` here exactly as they do on the debugger and concurrency columns.

mod support;
use std::sync::Arc;
use support::capability_probe::{
    probe_module_with, rows, unit_module_with, Row, MAX_ARGC, PROBE_BEYOND,
};
use temen_interp::{bytecode, Host, Trap, Value};
use temen_parity::frontier::{capability_axes, Axis, Capability};
use temen_parity::Status;

/// The trampoline the unit side runs: `f0(jit, code, cap)` invokes the installed unit, handing it
/// the capability handle. It is deliberately the *whole* base program on that side — the unit makes
/// the cap call, so anything the trampoline itself did would be noise in the comparison.
const INVOKE_TRAMPOLINE: &str = r#"memory 16
func (i32, i32, i32) -> (i64) {
block 0 (vjit: i32, vcode: i32, vcap: i32) {
  vc = i64.extend_i32_u vcode
  vh = i64.extend_i32_u vcap
  vr = call.cap 11 1 (i64, i64) -> (i64) vjit (vc, vh)
  return vr
  }
}
"#;

/// The `Jit.install` + `call.dyn` twin of [`INVOKE_TRAMPOLINE`]: install the unit, dispatch its slot
/// with the capability handle.
const INSTALL_TRAMPOLINE: &str = r#"memory 16
func (i32, i32, i32) -> (i64) {
block 0 (vjit: i32, vcode: i32, vcap: i32) {
  vc = i64.extend_i32_u vcode
  vslot = call.cap 11 3 (i64) -> (i64) vjit (vc)
  vs32 = i32.wrap_i64 vslot
  vh = i64.extend_i32_u vcap
  vr = call.dyn (i64) -> (i64) vs32 (vh)
  return vr
  }
}
"#;

/// Which DESIGN §22 route the unit side takes — see the module docs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Route {
    Invoke,
    Install,
}

/// What one side did with one call shape.
#[derive(Clone, PartialEq, Eq, Debug)]
enum Verdict {
    /// The bytecode engine has no arm for this call shape — refused at compile on the base side,
    /// reported as `Trap::Malformed` out of the invoke on the unit side. One decision, two places
    /// (see the module docs), so both fold here.
    Unsupported,
    /// The call reached no dispatch arm at all: the harness miscalled it. Skipped when it happens on
    /// both sides, kept when it happens on one.
    CapFault,
    /// The op ran and trapped, killing the domain.
    Trapped(String),
    /// The op ran and returned. A `-errno` counts: that is an answer.
    Answered(i64),
}

fn verdict(out: Option<Result<Vec<Value>, Trap>>) -> Verdict {
    match out {
        None | Some(Err(Trap::Malformed)) => Verdict::Unsupported,
        Some(Err(Trap::CapFault)) => Verdict::CapFault,
        Some(Err(t)) => Verdict::Trapped(format!("{t:?}")),
        Some(Ok(v)) => Verdict::Answered(match v.first() {
            Some(Value::I64(x)) => *x,
            other => panic!("the probe entry returns i64, got {other:?}"),
        }),
    }
}

/// The §22 `Jit` validator: a submitted unit is a real wire-encoded module, so decode and verify it
/// exactly as a host serving `vm_jit_compile` must (#922 — the host re-reads the blob for the unit's
/// type section, and a stub blob has none).
fn validate_unit(
    bytes: &[u8],
    _mode: Option<u8>,
    _sigs: &[u8],
) -> Result<Arc<[temen_ir::Func]>, i64> {
    let m = temen_encode::decode_module(bytes).map_err(|_| -22i64)?;
    temen_verify::verify_module(&m).map_err(|_| -22i64)?;
    Ok(m.funcs.into())
}

/// Grant the `Jit` and compile the probe's unit, returning `(jit handle, code handle)`. Run on
/// **both** sides so the two grant tables match byte for byte (see the module docs).
fn install_unit(
    host: &mut Host,
    iface: u32,
    op: u32,
    argc: usize,
    res: &str,
    real: &[(usize, i64)],
) -> Option<(i32, i32)> {
    let unit = unit_module_with(iface, op, argc, res, real);
    let blob = temen_encode::encode_module(&unit);
    // Reserve install slots (a `2^4` table) so the install route has somewhere to put the unit;
    // granted identically on both sides, so it moves no handle number.
    let jit = host.grant_jit_with_table(Some(16), 4);
    host.set_jit_validator(validate_unit);
    match host.jit_compile(jit, &blob) {
        Ok(Ok(code)) => Some((jit, code.handle)),
        // A unit the host itself refuses is not a code-origin answer; the sweep's shapes all
        // verify, so this is a harness bug rather than a finding.
        _ => None,
    }
}

/// The base module makes the cap call itself. The unit is still compiled (and never invoked) so the
/// handle numbering matches the other side.
fn from_base(
    mut host: Host,
    handle: i32,
    iface: u32,
    op: u32,
    argc: usize,
    res: &str,
    real: &[(usize, i64)],
) -> Verdict {
    install_unit(&mut host, iface, op, argc, res, real).expect("the probe unit compiles");
    let m = probe_module_with(iface, op, argc, res, real);
    let mut fuel = 1_000_000u64;
    verdict(bytecode::compile_and_run_with_host(
        &m,
        0,
        &[Value::I32(handle)],
        &mut fuel,
        &mut host,
    ))
}

/// The unit makes the cap call, reached by `route` from the matching trampoline.
#[allow(clippy::too_many_arguments)]
fn from_unit(
    route: Route,
    mut host: Host,
    handle: i32,
    iface: u32,
    op: u32,
    argc: usize,
    res: &str,
    real: &[(usize, i64)],
) -> Verdict {
    let (jit, code) =
        install_unit(&mut host, iface, op, argc, res, real).expect("the unit compiles");
    let tramp = match route {
        Route::Invoke => INVOKE_TRAMPOLINE,
        Route::Install => INSTALL_TRAMPOLINE,
    };
    let base = temen_text::parse_module(tramp).expect("the trampoline parses");
    temen_verify::verify_module(&base).expect("the trampoline verifies");
    let mut fuel = 1_000_000u64;
    verdict(bytecode::compile_and_run_with_host(
        &base,
        0,
        &[Value::I32(jit), Value::I32(code), Value::I32(handle)],
        &mut fuel,
        &mut host,
    ))
}

/// Every `(argc, result type)` shape the sweep tries.
fn shapes() -> impl Iterator<Item = (usize, &'static str)> {
    (0..MAX_ARGC).flat_map(|a| ["i64", "i32"].map(move |r| (a, r)))
}

/// The call shapes of `op` on which the base module and a unit reached by `route` disagree, as
/// `(shape, base, unit)`.
fn divergences(row: &Row, op: u32, route: Route) -> Vec<(String, Verdict, Verdict)> {
    let mut out = Vec::new();
    for (argc, res) in shapes() {
        let (h, x, real) = (row.mint)();
        let real: Vec<(usize, i64)> = real
            .iter()
            .filter(|(o, _, _)| *o == op)
            .map(|(_, idx, v)| (*idx, *v))
            .collect();
        let b = from_base(h, x, row.iface, op, argc, res, &real);
        let (h, x, _) = (row.mint)();
        let u = from_unit(route, h, x, row.iface, op, argc, res, &real);
        // Neither side found an arm: the harness miscalled this shape, so there is nothing to
        // compare. A cap fault on exactly one side *is* a divergence and is kept.
        if b == Verdict::CapFault && u == Verdict::CapFault {
            continue;
        }
        if b != u {
            out.push((format!("op {op}, argc {argc}, -> {res}"), b, u));
        }
    }
    out
}

fn col(a: Axis) -> usize {
    Axis::ALL.iter().position(|x| *x == a).expect("axis listed")
}

/// The §22 invoke contract (#1578): inside a `Jit.invoke` the `Instantiator` is unavailable, so its
/// ops `CapFault` there whatever the base module answers. The one invoke-route difference that is a
/// recorded decline rather than a gap.
fn invoke_contract(row: &Row, unit: &Verdict) -> bool {
    row.iface == temen_ir::cap_id::INSTANTIATOR && *unit == Verdict::CapFault
}

/// **The pin.** For every drivable capability, the manifest's `code origin` cell must match what a
/// unit actually gets out of that capability's ops, compared with the base module, on both routes:
/// `Full` when both agree everywhere, `Declines` when the only difference is the invoke contract,
/// `NotYet` for anything else.
#[test]
fn the_code_origin_column_matches_what_a_unit_and_the_base_module_actually_get() {
    let k = col(Axis::CodeOrigin);
    for row in rows() {
        let ops: Vec<u32> = if row.ops.is_empty() {
            // A capability with no callable ops: sweep a little way past its (empty) interface, so
            // "nothing to diverge about" is checked rather than assumed.
            (0..PROBE_BEYOND).collect()
        } else {
            row.ops.to_vec()
        };
        let mut found: Vec<(String, Verdict, Verdict)> = Vec::new();
        let mut declined = false;
        for op in &ops {
            found.extend(
                divergences(&row, *op, Route::Install)
                    .into_iter()
                    .map(|(s, b, u)| (format!("install: {s}"), b, u)),
            );
            for (s, b, u) in divergences(&row, *op, Route::Invoke) {
                if invoke_contract(&row, &u) {
                    declined = true;
                } else {
                    found.push((format!("invoke: {s}"), b, u));
                }
            }
        }
        let observed = if !found.is_empty() {
            Status::NotYet
        } else if declined {
            Status::Declines
        } else {
            Status::Full
        };
        let cell = capability_axes(row.cap)[k];
        assert_eq!(
            observed,
            cell.status,
            "{}: the frontier manifest says `{:?}` on the code-origin axis, but a unit and the base \
             module {}.\nDivergences found:\n{}",
            row.cap.name(),
            cell.status,
            if found.is_empty() {
                "agree on every call shape"
            } else {
                "disagree"
            },
            found
                .iter()
                .map(|(shape, b, u)| format!("  {shape}: base={b:?} unit={u:?}\n"))
                .collect::<String>(),
        );
    }
}

/// The `Instantiator` row's two routes, pinned as specific facts so a change names itself instead of
/// just moving a cell (#1578, #1726).
///
/// From the base module, on the same host with the same handle, `instantiate` (0),
/// `instantiate_module_named` (13) and `child_offer` (14) each answer `-EINVAL` probeably, and
/// `join` (1) gives the forgery trap for its forged child handle. An **installed** unit gets exactly
/// those answers — it runs in the caller's frames. An **invoked** unit gets a `CapFault` for every
/// one: the §22 seam-free leaf has no `Instantiator` (the cross-engine differential for a real spawn
/// is `crates/temen/tests/unit_instantiator.rs`).
#[test]
fn the_instantiator_row_by_route() {
    let row = rows()
        .into_iter()
        .find(|r| r.cap == Capability::Instantiator)
        .expect("the Instantiator row");
    let real = |op: u32| -> Vec<(usize, i64)> {
        (row.mint)()
            .2
            .iter()
            .filter(|(o, _, _)| *o == op)
            .map(|(_, i, v)| (*i, *v))
            .collect()
    };
    // (op, the arity that reaches its lowering arm, what the base module answers).
    for (op, argc, base_says) in [
        (0u32, 4usize, Verdict::Answered(-22)),
        (13, 7, Verdict::Answered(-22)),
        (14, 2, Verdict::Answered(-22)),
        (1, 1, Verdict::Trapped("ThreadFault".into())),
    ] {
        let args = real(op);
        let (h, x, _) = (row.mint)();
        assert_eq!(
            from_base(h, x, row.iface, op, argc, "i32", &args),
            base_says,
            "op {op} from the base module"
        );
        let (h, x, _) = (row.mint)();
        assert_eq!(
            from_unit(Route::Install, h, x, row.iface, op, argc, "i32", &args),
            base_says,
            "op {op} from an installed §22 unit — the caller's own frames"
        );
        let (h, x, _) = (row.mint)();
        assert_eq!(
            from_unit(Route::Invoke, h, x, row.iface, op, argc, "i32", &args),
            Verdict::CapFault,
            "op {op} from an invoked §22 unit — no Instantiator inside a seam-free leaf"
        );
    }
}
