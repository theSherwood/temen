//! **#1529 — a §22 guest-JIT unit that spawns detached finds the by-name spawn set.**
//!
//! The two reference powerboxes grant `"module"` (the running module, spawnable) and `"budget"`
//! (the detached-window allowance) only to a guest that can spawn detached — `spawns_detached`, a
//! static scan of the **root** module for `call.cap 6 15`. A guest whose only op 15 lives in a unit
//! it submits at runtime (`Jit.compile`) therefore resolved neither name and its spawn refused
//! `-EINVAL`: fail-closed and probeable, but a surprise, since the same op in the same domain works
//! when the root module happens to contain one too.
//!
//! `Host::jit_compile_linked` now grants the set on the first install of a unit that contains op 15
//! — the same least-authority rule, evaluated at install instead of at powerbox build. It is the
//! only point that sees both the validated unit and the host (the injected validator is a bare
//! `fn`), and it stays a strict subset of what the guest holds: a named `"instantiator"` must
//! already be granted (its window sizes the budget), `"module"` is the guest's own code, and a
//! durable domain is excluded (a `Module` grant is non-durable, and op 15 refuses there anyway).
//!
//! The blob validator is stubbed inline (the `invoke_fibers.rs` pattern), so this stays in the
//! temen-interp suite with no Cranelift dependency.

use std::sync::Arc;
use temen_interp::{bytecode, Host, Trap, Value};
use temen_text::parse_module;

/// What the detached child returns — read back through the unit's `join`.
const CHILD: i64 = 42;
/// The guest's declared window: the child's window too (op 15 admits only the module's own size).
const WIN_LOG2: u8 = 16;

/// Pack `name` as the little-endian `i64` words a guest stores before `self.resolve` (a unit keeps
/// no data segments — the validator hands back functions only — so it writes its own names).
fn packed(name: &str) -> Vec<i64> {
    name.as_bytes()
        .chunks(8)
        .map(|c| {
            let mut w = [0u8; 8];
            w[..c.len()].copy_from_slice(c);
            i64::from_le_bytes(w)
        })
        .collect()
}

/// Store `name` at `addr` and resolve it, yielding the IR for `dst = self.resolve(name)`.
fn resolve_by_name(dst: &str, addr: u64, name: &str) -> String {
    let mut s = String::new();
    for (i, w) in packed(name).iter().enumerate() {
        let at = addr + 8 * i as u64;
        s += &format!("  vp{dst}{i} = i64.const {at}\n  vw{dst}{i} = i64.const {w}\n  i64.store vp{dst}{i} vw{dst}{i}\n");
    }
    s += &format!("  vb{dst} = i64.const {addr}\n  vl{dst} = i64.const {}\n  {dst} = self.resolve vb{dst} vl{dst}\n", name.len());
    s
}

/// The **unit** the guest submits at runtime: it resolves `instantiator` / `module` / `budget` by
/// name and spawns the guest's own func 1 as a detached child (op 15, the 7-arg form), returning
/// the spawn's own result. A name it cannot resolve is `-1`, and op 15 through a `-1` handle is a
/// `CapFault` — so reaching a non-negative child handle at all is the proof the set was granted.
///
/// It does **not** `join` — [`joining_unit_src`] does.
fn unit_src() -> String {
    format!(
        r#"memory {WIN_LOG2}
func (i64) -> (i64) {{
block 0 (v0: i64) {{
{inst}{module}{budget}  vmh = i64.extend_i32_u vmod
  vmin = i64.extend_i32_u vbud
  vz = i64.const 0
  vent = i64.const 1
  vlog = i64.const {WIN_LOG2}
  vh = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) vinst (vmin, vmh, vz, vz, vent, vlog, vz)
  vr = i64.extend_i32_s vh
  return vr
  }}
}}
"#,
        inst = resolve_by_name("vinst", 20480, "instantiator"),
        module = resolve_by_name("vmod", 20512, "module"),
        budget = resolve_by_name("vbud", 20544, "budget"),
    )
}

/// The spawning unit plus a `join` of the child it just spawned: returns the child's result.
fn joining_unit_src() -> String {
    unit_src().replace(
        "  vr = i64.extend_i32_s vh\n  return vr",
        "  vr = call.cap 6 1 (i32) -> (i64) vinst (vh)\n  return vr",
    )
}

/// A unit that spawns nothing — the least-authority half: installing it grants no spawn set.
const QUIET_UNIT: &str = r#"memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  return v0
  }
}
"#;

/// How the guest reaches its unit — DESIGN §22's two routes, which carry different contracts: an
/// **installed** unit runs in the caller's frames and spawns like the base module; an **invoked**
/// one is a seam-free leaf with no `Instantiator` at all (#1578).
const INSTALL: &str = r#"  vc = i64.extend_i32_u v1
  vslot = call.cap 11 3 (i64) -> (i64) v0 (vc)
  vs32 = i32.wrap_i64 vslot
  vz = i64.const 0
  vr = call.dyn (i64) -> (i64) vs32 (vz)
  return vr"#;
const INVOKE: &str = r#"  vc = i64.extend_i32_u v1
  vz = i64.const 0
  vr = call.cap 11 1 (i64, i64) -> (i64) v0 (vc, vz)
  return vr"#;

/// The guest **program**: func 0 `(jit, code)` reaches the unit by `route`; func 1 is the detached
/// child entry the unit names. Crucially func 0 carries **no** `call.cap 6 15`, so the root-module
/// scan grants nothing — the whole point.
fn guest_by(route: &str) -> temen_ir::Module {
    let src = format!(
        r#"memory {WIN_LOG2}
func (i32, i32) -> (i64) {{
block 0 (v0: i32, v1: i32) {{
{route}
  }}
}}
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  va = i64.const 24576
  vk = i64.const {CHILD}
  i64.store va vk
  vw = i64.load va
  return vw
  }}
}}
"#
    );
    let m = parse_module(&src).expect("parse guest");
    temen_verify::verify_module(&m).expect("verify guest");
    m
}

/// The guest that installs its unit — the route the grant tests use.
fn guest() -> temen_ir::Module {
    guest_by(INSTALL)
}

/// The submitted blob: a real wire-encoded module. The host reads the unit's **type section** by
/// re-decoding these bytes (#922 interning), which a spawning unit needs — its `call.cap`s carry
/// interned signatures — so a stub blob would lose them.
fn unit_blob(src: &str) -> Vec<u8> {
    let m = parse_module(src).expect("parse unit");
    temen_verify::verify_module(&m).expect("verify unit");
    temen_encode::encode_module(&m)
}

/// The canonical validator shape: decode + verify the guest's blob, hand back its functions.
fn validator(
    bytes: &[u8],
    _mem_log2: Option<u8>,
    _symtab: &[u8],
) -> Result<Arc<[temen_ir::Func]>, i64> {
    let m = temen_encode::decode_module(bytes).map_err(|_| -22i64)?;
    temen_verify::verify_module(&m).map_err(|_| -22i64)?;
    Ok(m.funcs.into())
}

/// A reference-powerbox-shaped host: the running module registered and a named `"instantiator"`
/// over its window — and **no** `"module"` / `"budget"`, exactly what a guest whose root module
/// never spawns gets today.
fn powerbox(m: &temen_ir::Module) -> Host {
    let mut host = Host::new();
    host.set_self_module(&Arc::new(m.clone()));
    let inst = host.grant_instantiator(0, 1 << WIN_LOG2);
    host.register_cap_name("instantiator", inst);
    host
}

/// Grant a `Jit` domain (with install slots) and the validator, compile `src`'s unit, hand back the
/// entry args `(jit, code)`.
fn install(host: &mut Host, src: &str) -> Vec<Value> {
    let jit = host.grant_jit_with_table(Some(WIN_LOG2), 4);
    host.set_jit_validator(validator);
    let code = match host.jit_compile(jit, &unit_blob(src)) {
        Ok(Ok(c)) => c.handle,
        Ok(Err(e)) => panic!("compile refused: {e}"),
        Err(t) => panic!("compile trapped: {t:?}"),
    };
    vec![Value::I32(jit), Value::I32(code)]
}

#[test]
fn installing_a_spawning_unit_grants_the_by_name_spawn_set() {
    let m = guest();
    let mut host = powerbox(&m);
    assert!(
        host.resolve_cap_name("module").is_none(),
        "the root module spawns nothing, so the powerbox granted no spawn set"
    );
    install(&mut host, &unit_src());
    assert!(
        host.resolve_cap_name("module").is_some() && host.resolve_cap_name("budget").is_some(),
        "installing a unit that issues op 15 grants `module` + `budget`"
    );
}

/// Least authority: a unit that cannot spawn leaves the powerbox exactly as it found it (a `Module`
/// grant is non-durable — granting it to every guest-JIT user would cost them freezability).
#[test]
fn installing_a_unit_that_never_spawns_grants_nothing() {
    let m = guest();
    let mut host = powerbox(&m);
    install(&mut host, QUIET_UNIT);
    assert!(
        host.resolve_cap_name("module").is_none() && host.resolve_cap_name("budget").is_none(),
        "a spawn-free unit adds no authority"
    );
}

/// No named `"instantiator"` — the embedder never handed this guest §14 spawn authority — so the
/// install grants nothing either: the set is an extension of that decision, never a new frontier.
#[test]
fn without_a_granted_instantiator_the_install_grants_nothing() {
    let m = guest();
    let mut host = Host::new();
    host.set_self_module(&Arc::new(m.clone()));
    install(&mut host, &unit_src());
    assert!(
        host.resolve_cap_name("module").is_none(),
        "no instantiator, no spawn set"
    );
}

/// End to end, on the route §22 supports for spawning: the installed unit resolves the set the
/// install granted it, spawns the guest's own func 1 as a detached child over a fresh window, and
/// joins it — on both interpreters. Before #1529, `instantiator` resolved and `module` / `budget`
/// did not, so the spawn `CapFault`ed on a `-1` handle without ever reaching admission.
#[test]
fn an_installed_unit_spawns_and_joins_a_detached_child() {
    let m = guest();

    let mut host = powerbox(&m);
    let args = install(&mut host, &joining_unit_src());
    let mut fuel = u64::MAX;
    let tw = temen_interp::run_with_host(&m, 0, &args, &mut fuel, &mut host);
    assert_eq!(tw, Ok(vec![Value::I64(CHILD)]), "tree-walker");

    let mut host = powerbox(&m);
    let args = install(&mut host, &joining_unit_src());
    let mut fuel = u64::MAX;
    let bc = bytecode::compile_and_run_with_host(&m, 0, &args, &mut fuel, &mut host)
        .expect("bytecode supports the module");
    assert_eq!(bc, Ok(vec![Value::I64(CHILD)]), "bytecode");
}

/// The same unit **invoked** reaches no `Instantiator` (DESIGN §22, #1578): the spawn itself
/// `CapFault`s, on both engines, before any child exists. An invoked unit is never installed, so a
/// child would outlive the synchronous call over code nothing keeps, under a handle naming the
/// transient invoke vCPU's child — which nobody could join. The grant still happened (it is made at
/// install of the *compiled* unit, whatever route later runs it); the route is what refuses.
#[test]
fn an_invoked_unit_has_no_instantiator() {
    let m = guest_by(INVOKE);

    let mut host = powerbox(&m);
    let args = install(&mut host, &unit_src());
    let mut fuel = u64::MAX;
    let tw = temen_interp::run_with_host(&m, 0, &args, &mut fuel, &mut host);
    assert_eq!(tw, Err(Trap::CapFault), "tree-walker");

    let mut host = powerbox(&m);
    let args = install(&mut host, &unit_src());
    let mut fuel = u64::MAX;
    let bc = bytecode::compile_and_run_with_host(&m, 0, &args, &mut fuel, &mut host)
        .expect("bytecode supports the module");
    assert_eq!(bc, Err(Trap::CapFault), "bytecode");
}
