//! Op-15 **pre-mapped region** — the 11-arg `instantiate_detached(…, region, child_off)` form: a
//! `SharedRegion` of the parent aliased whole, read-write, into the detached child's private window at
//! `child_off` before it starts. The child sees plain memory at a fixed offset — no `self.resolve`, no
//! `page_size` query, no `map` call — and the parent reads the child's output back through its own
//! mapping of the same region after `join`. This is the bulk data plane between a parent and a child
//! whose memory it cannot otherwise address (DESIGN.md §13 / DETACHED_JIT.md §3.4), minus the ceremony.
//!
//! Pinned on both interpreter engines (the tree-walk oracle and the resumable bytecode engine, driven
//! through its `VcpuEvent::InstantiateDetached` protocol): the alias round-trip, the probeable `-EINVAL`
//! refusals (an offset inside the NULL guard, a region overrunning the declared window — charging
//! nothing, no child spawned), and the `CapFault` a wrong-kind handle raises (invariant 5: forgery
//! traps, geometry refuses).

use std::sync::Arc;
use temen_interp::{bytecode, run_with_host, Host, Region, Trap, Value};

fn module(text: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(text).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// The child (`memory 17`, child entry): reads the parent's input word at the pre-mapped offset
/// (region byte 0), writes `2 × input` at region byte 8, returns `input + 1`. It holds no region
/// handle and calls no cap — the pre-mapped pages are just its memory.
const CHILD: &str = r#"memory 17
func (i64) -> (i64) {
block 0 (v0: i64) {
  va = i64.const 65536
  vin = i64.load va
  vtwo = i64.const 2
  vout = i64.mul vin vtwo
  vb = i64.const 65544
  i64.store vb vout
  vone = i64.const 1
  vr = i64.add vin vone
  return vr
  }
}
"#;

/// The offset the child's window aliases the region at: region-granularity aligned on every host
/// (64 KiB — Windows' allocation granularity), above the NULL guard, and `+ 64 KiB` exactly fills the
/// child's 128 KiB window.
const CHILD_OFF: u64 = 65536;

/// The parent: `v0` Instantiator, `v1` AddressSpace, `v2` the child `Module`, `v3` the detached-spawn
/// `Budget`. Mints a 64 KiB region, maps it at 65536 in its own window, stores the input word 41 at
/// region byte 0, spawns the child detached with the region pre-mapped at `off` (`reg` names the
/// handle expression — the region, or a wrong-kind handle for the forgery probe). `join` then also
/// joins the child and returns `1000 × result + the word the child wrote at region byte 8`; otherwise
/// it returns the spawn's own result (the refusal probes).
fn parent(off: u64, reg: &str, join: bool) -> String {
    let tail = if join {
        "vj = call.cap 6 1 (i32) -> (i64) v0 (vc)\n  vk = i64.const 1000\n  vm = i64.mul vj vk\n  vob = i64.const 65544\n  vo = i64.load vob\n  vr = i64.add vm vo\n  return vr"
    } else {
        "vr = i64.extend_i32_s vc\n  return vr"
    };
    format!(
        r#"memory 17
func (i32, i32, i32, i32) -> (i64) {{
block 0 (v0: i32, v1: i32, v2: i32, v3: i32) {{
  vlen = i64.const 65536
  vrh64 = call.cap 5 5 (i64) -> (i64) v1 (vlen)
  vrh = i32.wrap_i64 vrh64
  vwo = i64.const 65536
  vro = i64.const 0
  vprot = i32.const 3
  vm0 = call.cap 4 0 (i64, i64, i64, i32) -> (i64) vrh (vwo, vro, vlen, vprot)
  vin = i64.const 41
  i64.store vwo vin
  vmh = i64.extend_i32_u v2
  vb = i64.extend_i32_u v3
  vz = i64.const 0
  vlog = i64.const 17
  vreg = i64.extend_i32_u {reg}
  voff = i64.const {off}
  vc = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vb, vmh, vz, vz, vz, vlog, vz, vz, vz, vreg, voff)
  {tail}
  }}
}}
"#
    )
}

/// `1000 × (41 + 1) + 2 × 41`.
const ROUND_TRIP: i64 = 42_082;

fn host(child: &temen_ir::Module) -> (Host, [i32; 4]) {
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, 1u64 << 17);
    let aspace = host.grant_address_space(0, 1u64 << 17);
    let modh = host.grant_module(child);
    let budget = host.grant_budget(0, 1i64 << 17, 0); // exactly one 2^17 window
    (host, [inst, aspace, modh, budget])
}

/// The tree-walk oracle.
fn run_interp(parent_src: &str) -> Result<Vec<Value>, Trap> {
    let parent = module(parent_src);
    let child = module(CHILD);
    let (mut host, h) = host(&child);
    let mut fuel = 50_000_000u64;
    run_with_host(
        &parent,
        0,
        &[
            Value::I32(h[0]),
            Value::I32(h[1]),
            Value::I32(h[2]),
            Value::I32(h[3]),
        ],
        &mut fuel,
        &mut host,
    )
}

/// The resumable bytecode engine, driven through its detached-spawn protocol: the host mints the
/// child's window, seeds the segments + payload, and builds the child over the stashed powerbox —
/// which carries the pre-map, applied by the child constructor. Records how many spawns surfaced.
fn drive(
    prog: &bytecode::VcpuProgram,
    mut vcpu: bytecode::Vcpu<'_>,
    spawns: &mut usize,
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
                data,
            } => {
                *spawns += 1;
                let back = Arc::new(Region::new(1u64 << size_log2, 4096));
                for seg in data.iter() {
                    back.write_from(seg.offset, &seg.bytes);
                }
                back.write_from(temen_ir::module_args_base(), &args);
                let reserved = temen_ir::DEFAULT_RESERVED_LOG2;
                let host = vcpu
                    .take_granted_host()
                    .expect("a pre-mapped spawn always stashes the child powerbox");
                let child = bytecode::Vcpu::new_confined_child_grow_over_host(
                    prog, module, entry, back, size_log2, reserved, fuel, host,
                )
                .expect("detached child builds");
                let r = drive(prog, child, spawns);
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

fn run_bytecode(parent_src: &str) -> (Result<Vec<Value>, Trap>, usize) {
    let parent = module(parent_src);
    let child = module(CHILD);
    let prog = bytecode::VcpuProgram::compile(&parent).expect("compile parent");
    let (host, h) = host(&child);
    let back = Arc::new(Region::new(1u64 << 17, 4096));
    let root = bytecode::Vcpu::new_root_with_powerbox(
        &prog,
        0,
        &[
            Value::I32(h[0]),
            Value::I32(h[1]),
            Value::I32(h[2]),
            Value::I32(h[3]),
        ],
        Arc::clone(&back),
        &[],
        host,
    )
    .expect("root vcpu");
    let mut spawns = 0;
    let r = drive(&prog, root, &mut spawns);
    (r, spawns)
}

#[test]
fn a_pre_mapped_region_carries_bulk_data_both_ways_on_the_tree_walker() {
    assert_eq!(
        run_interp(&parent(CHILD_OFF, "vrh", true)),
        Ok(vec![Value::I64(ROUND_TRIP)]),
        "the child read the parent's word through the alias and the parent read the child's back"
    );
}

#[test]
fn a_pre_mapped_region_carries_bulk_data_both_ways_on_the_bytecode_engine() {
    let (r, spawns) = run_bytecode(&parent(CHILD_OFF, "vrh", true));
    assert_eq!(
        r,
        Ok(vec![Value::I64(ROUND_TRIP)]),
        "identical to the oracle"
    );
    assert_eq!(spawns, 1);
}

#[test]
fn an_offset_inside_the_null_guard_refuses_probeably_on_both_engines() {
    assert_eq!(
        run_interp(&parent(0, "vrh", false)),
        Ok(vec![Value::I64(-22)]),
        "EINVAL, not a trap"
    );
    let (r, spawns) = run_bytecode(&parent(0, "vrh", false));
    assert_eq!(r, Ok(vec![Value::I64(-22)]));
    assert_eq!(spawns, 0, "a refused spawn never reaches the host");
}

#[test]
fn a_region_overrunning_the_child_window_refuses_probeably_on_both_engines() {
    // 128 KiB + 64 KiB > the child's 128 KiB declared window.
    assert_eq!(
        run_interp(&parent(1 << 17, "vrh", false)),
        Ok(vec![Value::I64(-22)])
    );
    let (r, spawns) = run_bytecode(&parent(1 << 17, "vrh", false));
    assert_eq!(r, Ok(vec![Value::I64(-22)]));
    assert_eq!(spawns, 0);
}

#[test]
fn a_wrong_kind_handle_traps_on_both_engines() {
    // The AddressSpace handle where a SharedRegion is expected: a typing violation on a live handle.
    assert_eq!(
        run_interp(&parent(CHILD_OFF, "v1", false)),
        Err(Trap::CapFault)
    );
    let (r, spawns) = run_bytecode(&parent(CHILD_OFF, "v1", false));
    assert_eq!(r, Err(Trap::CapFault));
    assert_eq!(spawns, 0);
}
