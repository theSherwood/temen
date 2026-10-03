//! **#1011 slice 3a — production wiring: a spawn's grant list runs on the resumable engine.** A
//! guest spawns a separate module through a v1 record (op 17) carrying a grant list, and the
//! **resumable `Vcpu` engine** admits it through the one admission every driver uses
//! (`admit_detached_child`), re-granting the named cap out of the *parent's own powerbox* into the
//! child's. The driver only starts the admitted child over a fresh window (`take_child` +
//! `PendingChild::start`) — the seam a JIT-tier nim phase child (a shared `fs`) uses. The grant is
//! authority (§3), a cross-tier `call.cap`, not a window access.

#[path = "support/rec.rs"]
mod rec;

use std::sync::{Arc, Mutex};
use temen_interp::{bytecode, ForkedProc, Host, HostProc, Region, Trap, Value};
use temen_ir::SpawnRec;

// The granted child (a *separate* module): its `Instantiator` arrives as `v0` (unused); it resolves
// the name `"fs"` (a data segment) to a handle and calls the granted `HOST_PROC` counter (type 13,
// op 0) — a post-increment `1`.
const CHILD: &str = r#"memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  vp0 = i64.const 17408
  vl2 = i64.const 2
  vh = self.resolve vp0 vl2
  vr = call.cap 13 0 (i64) -> (i64) vh (vp0)
  return vr
  }
}
data 17408 "fs"
"#;

// The parent (module 0). Entry args: `v0` = Instantiator, `v1` = the granted child `Module` handle,
// `v2` = the `"fs"` cap handle in the parent's powerbox, `v3` = the `Budget` the child's window is
// paid from. It seeds the name `"fs"` at window offset 18432, builds one 16-byte grant record at
// offset 17408 (`{name_off:u32=18432, name_len:u32=2, handle:i32=v2, flags:u32=0}`), writes the
// module and budget into its v1 record at 17536 (grant list `(17408, 1)`, entry 0), spawns the
// child (op 17) and joins it (op 1), returning its result. A correct run returns the granted
// counter's `1`.
const PARENT: &str = r#"memory 17
func (i32, i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32, v3: i32) {
  vname = i64.const 29542
  vnoff = i64.const 18432
  i64.store vnoff vname
  vrec0 = i64.const 8589953024
  vrecoff = i64.const 17408
  i64.store vrecoff vrec0
  vfsh = i64.extend_i32_u v2
  vrec1off = i64.const 17416
  i64.store vrec1off vfsh
  vmod = i64.const 17560
  i32.store vmod v1
  vbud = i64.const 17564
  i32.store vbud v3
  vrec = i64.const 17536
  vh = call.cap 6 17 (i64) -> (i32) v0 (vrec)
  vr = call.cap 6 1 (i32) -> (i64) v0 (vh)
  return vr
  }
}
"#;

fn parent() -> String {
    let spawn = SpawnRec {
        grants_ptr: 17408,
        grants_n: 1,
        ..SpawnRec::v1(0)
    };
    format!("{PARENT}{}", rec::segment(17536, &spawn))
}

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
        ForkedProc::shared(
            Box::new(move |_op, _args, _mem, _| {
                let mut c = c.lock().unwrap();
                *c += 1;
                Ok(vec![*c])
            }),
            temen_interp::CapState::Stateless,
        )
    });
    host.grant_host_proc_forkable(handler, fork, temen_interp::CapState::Stateless)
}

/// A fresh reservation of `size` bytes: the root's window, and each child's own, as every driver
/// gives a detached child. The engine seeds it.
fn window(size: u64) -> Arc<Region> {
    Arc::new(Region::new(size, temen_interp::host_page_size()))
}

/// Drive one vCPU of the run to completion, servicing its spawns. On an `InstantiateDetached` the
/// driver starts the child the engine admitted (its powerbox already built, the grant included) over
/// a fresh window. A leaf child runs synchronously (recursively drivable), its result delivered at
/// the join.
fn drive(prog: &bytecode::VcpuProgram, mut vcpu: bytecode::Vcpu<'_>) -> Result<Vec<Value>, Trap> {
    let mut children: Vec<Result<Vec<Value>, Trap>> = Vec::new();
    loop {
        match vcpu.run() {
            bytecode::VcpuEvent::Done(v) => return Ok(v),
            bytecode::VcpuEvent::Trapped(t) => return Err(t),
            bytecode::VcpuEvent::InstantiateDetached { .. } => {
                let child = vcpu
                    .take_child()
                    .expect("an InstantiateDetached carries its admitted child")
                    .start(prog, window(1u64 << temen_ir::DEFAULT_RESERVED_LOG2), None)
                    .expect("the child builds");
                let r = drive(prog, child);
                let token = children.len() as u64;
                children.push(r);
                vcpu.deliver_child(token);
            }
            bytecode::VcpuEvent::Join { child } => {
                vcpu.deliver_join(children[child as usize].clone());
            }
            _ => panic!("unexpected event in the grant kernel"),
        }
    }
}

#[test]
fn a_grant_list_regrants_through_the_resumable_engine() {
    let parent = module(&parent());
    let child = module(CHILD);
    let prog = bytecode::VcpuProgram::compile(&parent).expect("compile parent");

    let counter = Arc::new(Mutex::new(0i64));
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, 1u64 << 17);
    let modh = host.grant_module(&child);
    let fsh = grant_fs(&mut host, &counter);
    let budget = host.grant_budget(-1, 1 << 20, -1);

    let root = bytecode::Vcpu::new_root_with_powerbox(
        &prog,
        0,
        &[
            Value::I32(inst),
            Value::I32(modh),
            Value::I32(fsh),
            Value::I32(budget),
        ],
        window(1 << 17),
        &[],
        host,
    )
    .expect("root vcpu");
    let r = drive(&prog, root);

    assert_eq!(
        r,
        Ok(vec![Value::I64(1)]),
        "the granted child resolved 'fs' by name (re-granted through the engine's spawn arm) and called it (counter -> 1)"
    );
    assert_eq!(
        *counter.lock().unwrap(),
        1,
        "the re-granted handler ran once inside the child, over the shared parent state"
    );
}
