//! **#1011 slice 3a — production wiring: a §14 op-13 grant list runs on the resumable engine.** The
//! sibling test `confined_child_grant.rs` drives the *constructor* (`new_confined_child_granted`) with a
//! hand-built installer closure. This proves the real path: a guest issues
//! `instantiate_module_named` (op 13) with a grant list, the **resumable `Vcpu` engine** re-grants the
//! named cap out of the *parent's own powerbox* (`read_grant_list` + `regrant_list_into_child`), stashes
//! the child powerbox, and the driver runs the granted child with `take_granted_host()` +
//! `new_confined_child_over_host` — the exact seam a JIT-tier nim phase child (a shared `fs`) uses. A
//! grant-less child would run with the plain `new_confined_child`; the driver picks by whether a stash
//! is present. Window confinement (§2) is untouched: the grant is authority (§3), a cross-tier
//! `cap.call`, not a window access.

use std::sync::{Arc, Mutex};
use temen_interp::{bytecode, ForkedProc, Host, HostProc, Region, Trap, Value};

// The granted child (a *separate* module): its `Instantiator` arrives as `v0` (unused); it seeds the
// name `"fs"` (`0x7366` little-endian = 'f','s') into its own window, resolves it to a handle, and
// calls the granted `HOST_PROC` counter (type 13, op 0) — a post-increment `1`. Identical child to
// `confined_child_grant.rs`.
const CHILD: &str = r#"memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  vname = i64.const 29542
  vzero = i64.const 0
  i64.store vzero vname
  vp0 = i64.const 0
  vl2 = i64.const 2
  vh = cap.self.resolve vp0 vl2
  vr = cap.call 13 0 (i64) -> (i64) vh (vp0)
  return vr
  }
}
"#;

// The parent (module 0). Entry args: `v0` = Instantiator, `v1` = the granted child `Module` handle,
// `v2` = the `"fs"` cap handle in the parent's powerbox. It seeds the name `"fs"` at window offset 2048,
// builds one 16-byte grant record at offset 1024 (`{name_off:u32=2048, name_len:u32=2, handle:i32=v2,
// flags:u32=0}`), then issues `instantiate_module_named` (op 13) — module `v1`, grant list at
// `(1024, 1)`, entry 0, carve `off=65536 size_log2=16 quota=0` — and joins the child (op 1), returning
// its result. A correct run returns the granted counter's `1`.
const PARENT: &str = r#"memory 17
func (i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32) {
  vname = i64.const 29542
  vnoff = i64.const 2048
  i64.store vnoff vname
  vrec0 = i64.const 8589936640
  vrecoff = i64.const 1024
  i64.store vrecoff vrec0
  vfsh = i64.extend_i32_u v2
  vrec1off = i64.const 1032
  i64.store vrec1off vfsh
  vmh = i64.extend_i32_u v1
  vgptr = i64.const 1024
  vgn = i64.const 1
  ventry = i64.const 0
  voff = i64.const 65536
  vsl = i64.const 16
  vq = i64.const 0
  vh = cap.call 6 13 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vmh, vgptr, vgn, ventry, voff, vsl, vq)
  vr = cap.call 6 1 (i32) -> (i64) v0 (vh)
  return vr
  }
}
"#;

fn module(text: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(text).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// The granted `"fs"` shape: a forkable host-proc counter (the re-grantable form a shared memfs takes),
/// sharing one `Arc` so a call from inside the confined child is observable here.
fn grant_fs(host: &mut Host, counter: &Arc<Mutex<i64>>) -> i32 {
    let c1 = Arc::clone(counter);
    let handler: HostProc = Box::new(move |_op, _args, _mem, _| {
        let mut c = c1.lock().unwrap();
        *c += 1;
        Ok(vec![*c])
    });
    let c2 = Arc::clone(counter);
    let fork = Arc::new(move |_pid: u64| {
        let c = Arc::clone(&c2);
        ForkedProc::shared(Box::new(move |_op, _args, _mem, _| {
            let mut c = c.lock().unwrap();
            *c += 1;
            Ok(vec![*c])
        }))
    });
    host.grant_host_proc_forkable(handler, fork)
}

/// A raw window base that crosses no thread here, but keeps derived provenance (offset into the one live
/// allocation). Sound: only ever offset into the allocation below and handed to `Region::shared`.
#[derive(Clone, Copy)]
struct WinPtr(*mut u8);

/// Drive one vCPU of the run to completion, servicing §14 instantiate events. On an `Instantiate` the
/// driver asks the engine for a stashed granted powerbox (`take_granted_host`): `Some` → run the child
/// over it (`new_confined_child_over_host` — the op-13 grant path); `None` → the plain confined child.
/// A leaf child runs synchronously (recursively drivable), its result delivered at the join.
fn drive(
    prog: &bytecode::VcpuProgram,
    base: WinPtr,
    mut vcpu: bytecode::Vcpu<'_>,
) -> Result<Vec<Value>, Trap> {
    let mut children: Vec<Result<Vec<Value>, Trap>> = Vec::new();
    loop {
        match vcpu.run() {
            bytecode::VcpuEvent::Done(v) => return Ok(v),
            bytecode::VcpuEvent::Trapped(t) => return Err(t),
            bytecode::VcpuEvent::Instantiate {
                module,
                entry,
                carve,
                size_log2,
                fuel,
            } => {
                let granted = vcpu.take_granted_host();
                // SAFETY: the engine validated the carve within this vCPU's window (which outlives the
                // child here); the child's region aliases that sub-window — the §14 shared data plane.
                let child_base = WinPtr(unsafe { base.0.add(carve as usize) });
                // SAFETY: as above — `2^size_log2` valid bytes at the validated carve.
                let back = Arc::new(unsafe { Region::shared(child_base.0, 1u64 << size_log2) });
                let child = match granted {
                    Some(host) => bytecode::Vcpu::new_confined_child_over_host(
                        prog, module, entry, back, size_log2, fuel, host,
                    ),
                    None => bytecode::Vcpu::new_confined_child(
                        prog, module, entry, back, size_log2, fuel,
                    ),
                }
                .expect("confined child builds");
                let r = drive(prog, child_base, child);
                let handle = children.len() as i32;
                children.push(r);
                vcpu.deliver_handle(handle);
            }
            bytecode::VcpuEvent::Join { handle } => {
                vcpu.deliver_join(children[handle as usize].clone());
            }
            _ => panic!("unexpected event in the op-13 grant kernel"),
        }
    }
}

#[test]
fn op13_grant_list_regrants_through_the_resumable_engine() {
    let parent = module(PARENT);
    let child = module(CHILD);
    let prog = bytecode::VcpuProgram::compile(&parent).expect("compile parent");

    let counter = Arc::new(Mutex::new(0i64));
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, 1u64 << 17);
    let modh = host.grant_module(&child);
    let fsh = grant_fs(&mut host, &counter);

    // A 2^17 backing (fits the child's 2^16 carve at offset 65536).
    let size = 1usize << 17;
    let layout = std::alloc::Layout::from_size_align(size, 8).unwrap();
    // SAFETY: non-zero layout; `size` valid 8-aligned bytes owned here until freed below.
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!base.is_null());
    // SAFETY: `base` addresses `size` valid bytes, exclusively this run's, freed only after the vCPUs.
    let back = Arc::new(unsafe { Region::shared(base, size as u64) });

    let root = bytecode::Vcpu::new_root_with_powerbox(
        &prog,
        0,
        &[Value::I32(inst), Value::I32(modh), Value::I32(fsh)],
        Arc::clone(&back),
        &[],
        host,
    )
    .expect("root vcpu");
    let r = drive(&prog, WinPtr(base), root);

    drop(back);
    // SAFETY: same layout; every vCPU and region view is dropped, so no borrow outlives this.
    unsafe { std::alloc::dealloc(base, layout) };

    assert_eq!(
        r,
        Ok(vec![Value::I64(1)]),
        "the granted child resolved 'fs' by name (re-granted through the engine's op-13 arm) and called it (counter -> 1)"
    );
    assert_eq!(
        *counter.lock().unwrap(),
        1,
        "the re-granted handler ran once inside the confined child, over the shared parent state"
    );
}
