//! **The shared capability probe** — one fixture, three axis conformance tests (#1413).
//!
//! `debugger_conformance.rs` asks what the *debug tier* does with each powerbox capability's ops;
//! `concurrency_conformance.rs` asks what the **OS-thread parallel driver** does with the same ops;
//! `code_origin_conformance.rs` asks whether the same ops answer the same way when the caller is a
//! §22 guest-JIT **unit** rather than the host-translated base module. All three need the identical
//! scaffolding — how to mint a handle for each capability, which ops its ABI declares, and how to
//! build a one-call module around them — so that scaffolding lives here once and each test supplies
//! only its own driver and verdict vocabulary.
//!
//! That is INVARIANTS #15 applied to the matrix's own machinery: the row set is a *parameter* of one
//! structure, not a copy per column. A second copy would let the columns drift apart on what "the
//! `Instantiator` row" even means, which is exactly the drift the matrix exists to catch. The same
//! rule is why the code-origin column's *unit* module is [`Entry`] — a parameter of the one probe
//! generator — rather than a second copy of it with a different signature line.

// This module is compiled into each test binary separately, and each uses a subset — so every
// helper is "never used" in the binaries that do not call it. The allow is unavoidable, and the
// cost is that a helper nothing calls *anywhere* is invisible too: `probe_module_typed` sat
// orphaned here after its last caller moved to `probe_module_with`. Check by hand when removing a
// call site; the compiler will not.
#![allow(dead_code)]

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
    /// exactly that trap: the retired op 13's drivers agreed on a forged module handle while
    /// disagreeing about a real one, so a zeros-only probe would have scored the cell green over a
    /// live divergence.
    pub mint: Mint,
}

/// `instantiate_detached`'s (op 15's) argument slot the probe fills with a real value: the `Module`
/// handle (argument 1). Argument 0 is the `Budget`, and the rest are grants/entry/size/quota, which
/// zeros suit.
///
/// With a zero there the call dies at its handle resolve (`CapFault`), so the admission path where the
/// drivers once differed (#1570, on the retired op 13) goes untested. With it, the sweep reaches
/// admission — refused for want of a `Budget` — and the two drivers' answers are comparable. It does
/// not reach a *completed* spawn, so the column's claim is agreement through admission, which is where
/// the divergence was. Op 17 runs the same admission once it has read its record.
pub const OP15_MODULE_ARG: usize = 1;

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
        // op 4 is #1511's runtime `grant`, unbuilt: every engine answers `-EINVAL`, and the bytecode
        // lowering vetoes it by name. op 5 is `close` (#2358).
        row(
            Capability::SharedRegion,
            c::SHARED_REGION,
            &[0, 1, 2, 3, 4, 5],
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
        // The spawn family: join (1), child_offer (14), instantiate_detached (15) and
        // instantiate_rec (17). The carve spawns (0/5/13) and the coroutine variants (6/7) are
        // retired: every engine traps them `CapFault` (#1867, `carve_spawn_refused.rs`).
        Row {
            cap: Capability::Instantiator,
            iface: c::INSTANTIATOR,
            ops: &[1, 14, 15, 17],
            // op 15 takes a real `Module` handle in argument `OP15_MODULE_ARG`; with the sweep's
            // zero there it never gets past its handle resolve, so admission would go untested.
            mint: Box::new(|| {
                let mut h = Host::new();
                let x = h.grant_instantiator(0, 1 << 16);
                let modh = h.grant_module(&a_module());
                (h, x, vec![(15, OP15_MODULE_ARG, modh as i64)])
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
                let x = h.grant_host_proc(
                    Box::new(|_, _, _, _| Ok(vec![0])),
                    temen_interp::CapState::Stateless,
                );
                (h, x)
            }),
        ),
        // op 3 is `close` (#2384).
        row(
            Capability::Budget,
            c::BUDGET,
            &[0, 1, 3],
            Box::new(|| {
                let mut h = Host::new();
                let x = h.grant_budget(-1, 1 << 16, 4);
                (h, x)
            }),
        ),
    ]
}

/// Ops are tried across this many arities: the lowering's arms are arity-guarded
/// (`(INSTANTIATOR, 15) if args.len() >= 7`), so calling an op with too few arguments makes it look
/// unsupported when it is merely miscalled. The widest arm takes 9.
pub const MAX_ARGC: usize = 10;

/// Which entry shape the generated probe wears — the parameter the **code-origin** column varies,
/// and the reason there is one probe generator rather than two.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Entry {
    /// `f0(handle: i32) -> i64` — a host-translated base module, run by a driver with the handle
    /// passed in as an argument.
    Base,
    /// `f0(handle: i64) -> i64` — a §22 unit, reached through `Jit.invoke`, whose marshalling ABI is
    /// i64-only. The handle therefore arrives widened and is narrowed back before the cap call; the
    /// call itself, argument for argument, is the one [`Entry::Base`] makes.
    Unit,
}

/// The one-call probe module: `f0(handle)` calls `call.cap iface op` with `argc` zero arguments and
/// returns its result. Argument *values* are all zero deliberately — a `-EINVAL` or a trap is a
/// serviced answer (the op reached its seam and a driver drove it), so only a driver's own gates
/// produce any other verdict.
pub fn probe_module(iface: u32, op: u32, argc: usize) -> temen_ir::Module {
    probe_module_with(iface, op, argc, "i64", &[])
}

/// **The base-module probe.** `result` picks the call's declared result type and `overrides` replaces
/// individual zero arguments — `(index, value)` pairs from a [`Row::mint`] entry; see that field for
/// why zeros alone are not enough.
///
/// `result` is a parameter rather than a constant because the lowering's arms are keyed by
/// `(type_id, op)` *and* the call signature, so an op whose real result is `i32` is not reached by an
/// `i64`-returning call — it falls through to the generic dispatch, which for some interfaces is
/// rejected at compile time rather than at the call. A sweep that fixes the result type therefore
/// mistakes "miscalled" for "unsupported".
pub fn probe_module_with(
    iface: u32,
    op: u32,
    argc: usize,
    result: &str,
    overrides: &[(usize, i64)],
) -> temen_ir::Module {
    build_probe(Entry::Base, iface, op, argc, result, overrides)
}

/// [`probe_module_with`]'s twin as a §22 **unit** — the identical call behind an
/// [`Entry::Unit`] signature, for `code_origin_conformance.rs` to compile and invoke.
pub fn unit_module_with(
    iface: u32,
    op: u32,
    argc: usize,
    result: &str,
    overrides: &[(usize, i64)],
) -> temen_ir::Module {
    build_probe(Entry::Unit, iface, op, argc, result, overrides)
}

fn build_probe(
    entry: Entry,
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
    // An Instantiator op answers the results its contract names (`temen_ir::instantiator_contract`),
    // and every engine traps a call declaring any other count `CapFault` before the op is reached. So
    // the probe declares that many, `result` the first's type and the rest `i64`, and returns the
    // first: `join` answers `(status, value)` (#2066).
    let extra = match iface {
        temen_ir::cap_id::INSTANTIATOR => {
            temen_ir::instantiator_contract(op).map_or(0, |(_, r)| r.len().saturating_sub(1))
        }
        _ => 0,
    };
    let names: String = (0..extra).map(|i| format!(", vx{i}")).collect();
    let call_res = format!("{call_res}{}", ", i64".repeat(extra));
    // The only difference between the two entry shapes: a unit is handed its handle widened (the
    // `Jit.invoke` marshalling ABI is i64-only), so it narrows it back before the call.
    let (sig, narrow) = match entry {
        Entry::Base => ("(i32) -> (i64) {\nblock 0 (vh: i32) {", String::new()),
        Entry::Unit => (
            "(i64) -> (i64) {\nblock 0 (vh64: i64) {",
            "  vh = i32.wrap_i64 vh64\n".to_string(),
        ),
    };
    let src = format!(
        "memory 16\nfunc {sig}\n{narrow}{body}  \
         vr{names} = call.cap {iface} {op} ({params}) -> ({call_res}) vh ({args})\n{tail}\n  }}\n}}\n"
    );
    let m = temen_text::parse_module(&src).expect("the generated module parses");
    temen_verify::verify_module(&m).expect("the generated module verifies");
    m
}
