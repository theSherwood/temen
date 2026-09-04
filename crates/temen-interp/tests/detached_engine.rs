//! #1286 — **`instantiate_detached` (op 15) on the resumable `Vcpu` engine.** The tree-walker hosts
//! detached windows itself (`detached_windows.rs`); the resumable engine instead **surfaces** the spawn
//! as [`bytecode::VcpuEvent::InstantiateDetached`] — no carve, the host mints the window — after doing
//! the authority-bearing work in-engine: the `Instantiator` resolved, the module compiled + pushed to the
//! shared source, the `WindowMinter` quota taken (a miss lands `-EINVAL` probeably), the grant list
//! re-granted and stashed, and the optional spawn-time **args payload** read out of the parent's window.
//! The host here seeds the fresh window (data segments + payload at `module_args_base()`) and runs the
//! child with `new_confined_child_grow_over_host` — exactly what the browser's op-13 servicer does over a
//! detached child's own `WebAssembly.Memory`.

use std::sync::Arc;
use temen_interp::{bytecode, Host, Region, Trap, Value};

fn module(text: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(text).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// The detached child (a child-entry module, `memory 16`): loads the first 8 bytes of the first argv
/// string (the args blob is `{argc u32, envc u32}` then packed strings, at `module_args_base()` =
/// 16384 + 128), adds its `self.attest` report (1 = tier 1, `window_exposed = false`), returns the sum.
const CHILD: &str = r#"memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  vab = i64.const 16520
  va = i64.load vab
  vz = i32.const 0
  vat = call.cap 4294967295 4 () -> (i64) vz ()
  vs = i64.add va vat
  return vs
  }
}
"#;

/// "hello-de" as a little-endian i64 — the word the child reads at `args_base + 8`.
const ARGV_WORD: i64 = i64::from_le_bytes(*b"hello-de");

/// The parent: `v0` Instantiator, `v1` the child `Module`, `v2` the `WindowMinter`. Stores the args
/// blob (`argc = 1`, `"hello-detached\0"`) as three words at 18432, issues op 15 with the 9-arg form
/// (payload `(18432, 24)`) or the 7-arg form, then `join`s the child (or, in the refusal probe, returns
/// the spawn's own result).
fn parent(payload: bool, join: bool) -> String {
    let spawn = if payload {
        "vh = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vmin, vmh, vz, vz, ve, vlog, vq, vap, val)"
    } else {
        "vh = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vmin, vmh, vz, vz, ve, vlog, vq)"
    };
    let tail = if join {
        "vr = call.cap 6 1 (i32) -> (i64) v0 (vh)\n  return vr"
    } else {
        "vr = i64.extend_i32_s vh\n  return vr"
    };
    format!(
        r#"memory 17
func (i32, i32, i32) -> (i64) {{
block 0 (v0: i32, v1: i32, v2: i32) {{
  vb0 = i64.const 18432
  vw0 = i64.const 1
  i64.store vb0 vw0
  vb1 = i64.const 18440
  vw1 = i64.const {w1}
  i64.store vb1 vw1
  vb2 = i64.const 18448
  vw2 = i64.const {w2}
  i64.store vb2 vw2
  vmh = i64.extend_i32_u v1
  vmin = i64.extend_i32_u v2
  vz = i64.const 0
  ve = i64.const 0
  vlog = i64.const 16
  vq = i64.const 0
  vap = i64.const 18432
  val = i64.const 24
  {spawn}
  {tail}
  }}
}}
"#,
        w1 = ARGV_WORD,
        w2 = i64::from_le_bytes(*b"tached\0\0"),
    )
}

/// The host side of the protocol: mint a window of the child's declared size, seed the module's data
/// segments and the payload, run the child (granted powerbox if the engine stashed one), deliver the
/// join handle. Records the payload the event carried for the assertions.
fn drive(
    prog: &bytecode::VcpuProgram,
    child_mod: &temen_ir::Module,
    mut vcpu: bytecode::Vcpu<'_>,
    seen_payload: &mut Vec<Vec<u8>>,
) -> Result<Vec<Value>, Trap> {
    let mut children: Vec<Result<Vec<Value>, Trap>> = Vec::new();
    loop {
        match vcpu.run() {
            bytecode::VcpuEvent::Done(v) => return Ok(v),
            bytecode::VcpuEvent::Trapped(t) => return Err(t),
            bytecode::VcpuEvent::InstantiateDetached {
                module,
                entry,
                size_log2,
                fuel,
                args,
            } => {
                seen_payload.push(args.clone());
                // The fresh window: the host's to allocate — nothing of it in the parent's.
                let back = Arc::new(Region::new(1u64 << size_log2, 4096));
                for seg in &child_mod.data {
                    back.write_from(seg.offset, &seg.bytes);
                }
                back.write_from(temen_ir::module_args_base(), &args);
                // Committed window = the declared size; starter caps span the reservation (a root's
                // shape), so the child may `vm_map`-grow — the tree-walker's op-15 grants the same.
                let reserved = temen_ir::DEFAULT_RESERVED_LOG2;
                let child = match vcpu.take_granted_host() {
                    Some(host) => bytecode::Vcpu::new_confined_child_grow_over_host(
                        prog, module, entry, back, size_log2, reserved, fuel, host,
                    ),
                    None => bytecode::Vcpu::new_confined_child_grow(
                        prog, module, entry, back, size_log2, reserved, fuel,
                    ),
                }
                .expect("detached child builds");
                let r = drive(prog, child_mod, child, seen_payload);
                let handle = children.len() as i32;
                children.push(r);
                vcpu.deliver_handle(handle);
            }
            bytecode::VcpuEvent::Join { handle } => {
                vcpu.deliver_join(children[handle as usize].clone());
            }
            _ => panic!("unexpected event in the detached kernel"),
        }
    }
}

fn run(parent_src: &str, minter_quota: u64) -> (Result<Vec<Value>, Trap>, Vec<Vec<u8>>) {
    let parent = module(parent_src);
    let child = module(CHILD);
    let prog = bytecode::VcpuProgram::compile(&parent).expect("compile parent");
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, 1u64 << 17);
    let modh = host.grant_module(&child);
    let minter = host.grant_window_minter(minter_quota);
    let back = Arc::new(Region::new(1u64 << 17, 4096));
    let root = bytecode::Vcpu::new_root_with_powerbox(
        &prog,
        0,
        &[Value::I32(inst), Value::I32(modh), Value::I32(minter)],
        Arc::clone(&back),
        &[],
        host,
    )
    .expect("root vcpu");
    let mut seen = Vec::new();
    let r = drive(&prog, &child, root, &mut seen);
    (r, seen)
}

#[test]
fn op15_surfaces_a_detached_spawn_with_its_args_payload() {
    let (r, seen) = run(&parent(true, true), 1 << 16);
    assert_eq!(
        r,
        Ok(vec![Value::I64(ARGV_WORD + 1)]),
        "the child read argv from the host-seeded payload and attested tier 1 / unexposed"
    );
    assert_eq!(seen.len(), 1, "exactly one detached spawn surfaced");
    assert_eq!(seen[0].len(), 24);
    assert_eq!(&seen[0][..8], &1u64.to_le_bytes(), "argc = 1, envc = 0");
    assert_eq!(&seen[0][8..16], b"hello-de");
}

#[test]
fn the_seven_arg_form_seeds_no_payload() {
    let (r, seen) = run(&parent(false, true), 1 << 16);
    assert_eq!(
        r,
        Ok(vec![Value::I64(1)]),
        "no argv: the word reads 0, attest adds 1"
    );
    assert_eq!(seen, vec![Vec::<u8>::new()]);
}

#[test]
fn an_exhausted_minter_refuses_probeably_without_surfacing() {
    // Quota one byte short of the 64 KiB window: `-EINVAL` lands in place, no event, nothing charged.
    let (r, seen) = run(&parent(true, false), (1 << 16) - 1);
    assert_eq!(r, Ok(vec![Value::I64(-22)]), "EINVAL, not a trap");
    assert!(seen.is_empty(), "a refused spawn never reaches the host");
}
