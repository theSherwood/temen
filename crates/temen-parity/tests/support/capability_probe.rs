//! **The shared capability probe** — one fixture, two axis conformance tests (#1413).
//!
//! `debugger_conformance.rs` asks what the *debug tier* does with each powerbox capability's ops;
//! `concurrency_conformance.rs` asks what the **OS-thread parallel driver** does with the same ops.
//! Both questions need the identical scaffolding — how to mint a handle for each capability, which
//! ops its ABI declares, and how to build a one-call module around them — so that scaffolding lives
//! here once and each test supplies only its own driver and verdict vocabulary.
//!
//! That is INVARIANTS #15 applied to the matrix's own machinery: the row set is a *parameter* of one
//! structure, not a copy per column. A second copy would let the two columns drift apart on what
//! "the `Instantiator` row" even means, which is exactly the drift the matrix exists to catch.

#![allow(dead_code)] // each consumer uses a subset

use temen_interp::{Host, StreamRole};
use temen_parity::frontier::Capability;

pub fn a_module() -> temen_ir::Module {
    let m = temen_text::parse_module(
        "memory 15\nfunc (i64) -> (i64) {\nblock 0 (v0: i64) {\n  return v0\n  }\n}\n",
    )
    .expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// One drivable row: how to mint the handle, the interface id, and the ops the ABI declares for it
/// (`temen_ir::cap_id`'s docs are the ABI; the bytecode lowering's arms above the generic
/// fallthrough are the same list from the engine's side).
///
/// An **empty** `ops` means the capability declares no callable ops at all — the handle confers only
/// the authority to be *named* by another capability's op. Such a row is still swept (see
/// [`PROBE_BEYOND`]): the claim is that nothing reaches either gate, which is worth failing on
/// rather than assuming.
/// Real argument values a row's ops need, as `(op, argument index, value)` — see [`Row::mint`].
pub type RealArgs = Vec<(u32, usize, i64)>;

/// Mint a row's powerbox and handle, plus the real argument values its ops need.
pub type Mint = Box<dyn Fn() -> (Host, i32, RealArgs)>;

pub struct Row {
    pub cap: Capability,
    pub iface: u32,
    pub ops: &'static [u32],
    /// Mint the powerbox and this row's handle, **plus** any real argument values its ops need —
    /// `(op, argument index, value)`. Built by [`rows`]; most rows return an empty vec.
    ///
    /// Real values have to come from the mint rather than a constant, because a handle's numeric
    /// value is whatever the grant table hands out. Hardcoding one produces a *forged* handle that
    /// every driver rightly rejects — which looks like agreement while the interesting path goes
    /// untested. (#1570 was exactly that: with a forged module handle the two drivers agreed, and
    /// with a real one they did not.)
    ///
    /// The sweep otherwise passes **zeros**, which is deliberate: a `-EINVAL` or a trap is a
    /// serviced answer, so zeros are enough to ask "can this driver run this op at all". They are
    /// *not* enough to ask "do the drivers agree about a **valid** call", because a zero handle is
    /// forged and every driver rightly rejects it before reaching the interesting code. #1570 is
    /// exactly that trap: op 13's drivers agreed on a forged module handle while disagreeing about a
    /// real one, so a zeros-only probe would have scored the cell green over a live divergence.
    pub mint: Mint,
}

/// `instantiate_module_named`'s argument slots the probe fills with real values: the `Module` handle
/// (argument 0) and a `size_log2` matching that module's declared memory (argument 5). Arguments 1
/// and 2 are `grants_ptr`/`grants_n`, and 3/4/6 are entry/off/quota, which zeros suit.
///
/// With zeros in both of these the call dies at its handle resolve (`CapFault`), so the admission
/// path where the drivers differed (#1570) went untested. With them, the sweep reaches admission and
/// the two drivers' answers are comparable. It does not reach a *completed* spawn with a non-empty
/// grant list — that needs records planted in the parent window — so the column's claim is agreement
/// through admission, which is where the divergence was.
pub const OP13_MODULE_ARG: usize = 0;
pub const OP13_SIZE_LOG2_ARG: usize = 5;
/// The declared memory of [`a_module`], the module the probe grants.
pub const PROBE_MODULE_SIZE_LOG2: i64 = 15;

/// How far past a no-op capability's (empty) interface to probe when checking that nothing on it
/// reaches either gate.
pub const PROBE_BEYOND: u32 = 4;

pub fn rows() -> Vec<Row> {
    // Most rows need no real argument values: wrap a plain `(Host, handle)` mint.
    let row = |cap, iface, ops, mint: Box<dyn Fn() -> (Host, i32)>| -> Row {
        Row {
            cap,
            iface,
            ops,
            mint: Box::new(move || {
                let (h, x) = mint();
                (h, x, Vec::new())
            }),
        }
    };
    use temen_ir::cap_id as c;
    vec![
        row(
            Capability::Stream,
            c::STREAM,
            &[0, 1, 2],
            Box::new(|| {
                let mut h = Host::new();
                let x = h.grant_stream(StreamRole::Out);
                (h, x)
            }),
        ),
        // A pipe end is `Stream`-typed: the same three ops over a shared FIFO backing.
        row(
            Capability::PipeEnd,
            c::STREAM,
            &[0, 1, 2],
            Box::new(|| {
                let mut h = Host::new();
                let (r, _w) = h.grant_pipe();
                (h, r)
            }),
        ),
        row(
            Capability::Exit,
            c::EXIT,
            &[0],
            Box::new(|| {
                let mut h = Host::new();
                let x = h.grant_exit();
                (h, x)
            }),
        ),
        row(
            Capability::Clock,
            c::CLOCK,
            &[0],
            Box::new(|| {
                let mut h = Host::new();
                let x = h.grant_clock();
                (h, x)
            }),
        ),
        // op 4 is the guest-minted-region `create`/`grant` follow-up: wired in the oracle, vetoed by
        // name in the bytecode lowering.
        row(
            Capability::SharedRegion,
            c::SHARED_REGION,
            &[0, 1, 2, 3, 4],
            Box::new(|| {
                let mut h = Host::new();
                let x = h.grant_shared_region(4096);
                (h, x)
            }),
        ),
        row(
            Capability::AddressSpace,
            c::ADDRESS_SPACE,
            &[0, 1, 2, 3, 4],
            Box::new(|| {
                let mut h = Host::new();
                let x = h.grant_address_space(0, 1 << 16);
                (h, x)
            }),
        ),
        // The spawn family: instantiate/join (0/1), the module spawns (5/13), child_offer (14),
        // instantiate_detached (15), instantiate_rec (17). The coroutine variants (6/7) are the
        // legacy residue the lowering rejects wholesale.
        Row {
            cap: Capability::Instantiator,
            iface: c::INSTANTIATOR,
            ops: &[0, 1, 5, 6, 7, 13, 14, 15, 17],
            // op 13 (`instantiate_module_named`) takes a real `Module` handle in argument
            // `OP13_MODULE_ARG`; with the sweep's zero there it never gets past its handle resolve,
            // so the valid-handle path where the drivers differed (#1570) went untested.
            mint: Box::new(|| {
                let mut h = Host::new();
                let x = h.grant_instantiator(0, 1 << 16);
                let modh = h.grant_module(&a_module());
                (
                    h,
                    x,
                    vec![
                        (13, OP13_MODULE_ARG, modh as i64),
                        (13, OP13_SIZE_LOG2_ARG, PROBE_MODULE_SIZE_LOG2),
                    ],
                )
            }),
        },
        row(
            Capability::ModuleLoader,
            c::MODULE_LOADER,
            &[0],
            Box::new(|| {
                let mut h = Host::new();
                let x = h.grant_module_loader();
                (h, x)
            }),
        ),
        // No callable ops — a module handle confers only the authority to be named by the
        // `Instantiator`'s module spawns, so there is nothing for the debug tier to diverge about.
        // Swept anyway (see `Row`): the claim is that nothing on it reaches either gate.
        row(
            Capability::Module,
            c::MODULE,
            &[],
            Box::new(|| {
                let mut h = Host::new();
                let x = h.grant_module(&a_module());
                (h, x)
            }),
        ),
        row(
            Capability::Blocking,
            c::BLOCKING,
            &[0],
            Box::new(|| {
                let mut h = Host::new();
                let x = h.grant_blocking(std::time::Duration::ZERO, None);
                (h, x)
            }),
        ),
        row(
            Capability::HostProc,
            c::HOST_PROC,
            &[0],
            Box::new(|| {
                let mut h = Host::new();
                let x = h.grant_host_proc(Box::new(|_, _, _, _| Ok(vec![0])));
                (h, x)
            }),
        ),
        row(
            Capability::Budget,
            c::BUDGET,
            &[0, 1],
            Box::new(|| {
                let mut h = Host::new();
                let x = h.grant_budget(0, 1 << 16, 4);
                (h, x)
            }),
        ),
    ]
}

/// Ops are tried across this many arities: the lowering's arms are arity-guarded
/// (`(INSTANTIATOR, 5) if args.len() >= 5`), so calling an op with too few arguments makes it look
/// unsupported when it is merely miscalled. The widest arm takes 9.
pub const MAX_ARGC: usize = 10;

/// The one-call probe module: `f0(handle)` calls `call.cap iface op` with `argc` zero arguments and
/// returns its result. Argument *values* are all zero deliberately — a `-EINVAL` or a trap is a
/// serviced answer (the op reached its seam and a driver drove it), so only a driver's own gates
/// produce any other verdict.
pub fn probe_module(iface: u32, op: u32, argc: usize) -> temen_ir::Module {
    probe_module_typed(iface, op, argc, "i64")
}

/// [`probe_module_typed`] with `overrides` replacing individual zero arguments — `(index, value)`
/// pairs from a [`Row::real_args`] entry. See that field for why zeros alone are not enough.
pub fn probe_module_with(
    iface: u32,
    op: u32,
    argc: usize,
    result: &str,
    overrides: &[(usize, i64)],
) -> temen_ir::Module {
    build_probe(iface, op, argc, result, overrides)
}

/// [`probe_module`] with the call's declared **result type** chosen by the caller. The lowering's
/// arms are keyed by `(type_id, op)` *and* the call signature, so an op whose real result is `i32`
/// is not reached by an `i64`-returning call — it falls through to the generic dispatch, which for
/// some interfaces is rejected at compile time rather than at the call. A sweep that fixes the
/// result type therefore mistakes "miscalled" for "unsupported".
pub fn probe_module_typed(iface: u32, op: u32, argc: usize, result: &str) -> temen_ir::Module {
    build_probe(iface, op, argc, result, &[])
}

fn build_probe(
    iface: u32,
    op: u32,
    argc: usize,
    result: &str,
    overrides: &[(usize, i64)],
) -> temen_ir::Module {
    let mut body = String::new();
    let mut args = String::new();
    let mut params = String::new();
    for i in 0..argc {
        let v = overrides
            .iter()
            .find(|(idx, _)| *idx == i)
            .map_or(0, |(_, v)| *v);
        body.push_str(&format!("  va{i} = i64.const {v}\n"));
        if i > 0 {
            args.push_str(", ");
            params.push_str(", ");
        }
        args.push_str(&format!("va{i}"));
        params.push_str("i64");
    }
    // The entry always returns `i64`, so one harness reads every probe's answer; an `i32`-returning
    // op is widened at the return.
    let (call_res, tail) = match result {
        "i32" => ("i32", "  vw = i64.extend_i32_s vr\n  return vw"),
        _ => ("i64", "  return vr"),
    };
    let src = format!(
        "memory 16\nfunc (i32) -> (i64) {{\nblock 0 (vh: i32) {{\n{body}  \
         vr = call.cap {iface} {op} ({params}) -> ({call_res}) vh ({args})\n{tail}\n  }}\n}}\n"
    );
    let m = temen_text::parse_module(&src).expect("the generated module parses");
    temen_verify::verify_module(&m).expect("the generated module verifies");
    m
}
