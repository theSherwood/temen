//! THREADS.md 4c-domain §14-D2 — **Miri** verification of §14 children through the **resumable
//! `Vcpu`** API. The host starts each of [`miri_spawn`]'s two detached children on a thread of its own
//! over a fresh backing, as the browser starts each on a Worker; the engine seeds the child's window
//! and aliases its pre-mapped region in. This proves that machinery — the children's writes through
//! their views of the regions reaching the parent's views across the join — is **free of data races
//! / UB / provenance errors**. Miri's checker, not the iteration count, is the point, so the kernel (2
//! children) is small. The differential pins of the same protocol are `support/drivers.rs`'s `Vcpu`
//! driver and `detached_premap.rs`.
//!
//! Run: `cargo +nightly miri test -p temen-interp --test vcpu_instantiate_miri`

#[path = "support/miri_spawn.rs"]
mod miri_spawn;
#[path = "support/rec.rs"]
mod rec;

use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex};
use temen_interp::{bytecode, Region, Trap, Value};

struct Orch {
    next_id: Mutex<u64>,
    done: Mutex<HashMap<u64, Result<Vec<Value>, Trap>>>,
    cv: Condvar,
}

impl Orch {
    fn new() -> Orch {
        Orch {
            next_id: Mutex::new(0),
            done: Mutex::new(HashMap::new()),
            cv: Condvar::new(),
        }
    }
    fn fresh_id(&self) -> u64 {
        let mut n = self.next_id.lock().unwrap();
        let id = *n;
        *n += 1;
        id
    }
    fn publish(&self, id: u64, r: Result<Vec<Value>, Trap>) {
        self.done.lock().unwrap().insert(id, r);
        self.cv.notify_all();
    }
    fn join(&self, id: u64) -> Result<Vec<Value>, Trap> {
        let mut g = self.done.lock().unwrap();
        loop {
            if let Some(r) = g.remove(&id) {
                return r;
            }
            g = self.cv.wait(g).unwrap();
        }
    }
}

fn drive<'s, 'e>(
    scope: &'s std::thread::Scope<'s, 'e>,
    prog: &'e bytecode::VcpuProgram,
    orch: &'e Orch,
    mut vcpu: bytecode::Vcpu<'e>,
) -> Result<Vec<Value>, Trap> {
    loop {
        match vcpu.run() {
            bytecode::VcpuEvent::Done(vals) => return Ok(vals),
            bytecode::VcpuEvent::Trapped(t) => return Err(t),
            bytecode::VcpuEvent::InstantiateDetached { size_log2 } => {
                let id = orch.fresh_id();
                let back = Arc::new(Region::new(1u64 << size_log2, 4096));
                let child = vcpu
                    .take_child()
                    .expect("an InstantiateDetached carries its admitted child")
                    .start(prog, back)
                    .expect("detached child vcpu");
                scope.spawn(move || {
                    let r = drive(scope, prog, orch, child);
                    orch.publish(id, r);
                });
                vcpu.deliver_child(id);
            }
            bytecode::VcpuEvent::Join { child } => {
                vcpu.deliver_join(orch.join(child));
            }
            _ => panic!("unexpected event in the spawn Miri kernel"),
        }
    }
}

fn run_orchestrated() -> Result<Vec<Value>, Trap> {
    let m = miri_spawn::root();
    let prog = bytecode::VcpuProgram::compile(&m).expect("compile");
    let (host, args) = miri_spawn::host();
    let back = Arc::new(Region::new(1 << 18, 4096));
    let orch = Orch::new();
    let root = bytecode::Vcpu::new_root_with_powerbox(&prog, 0, &args, back, &[], host)
        .expect("root vcpu");
    std::thread::scope(|scope| drive(scope, &prog, &orch, root))
}

#[test]
fn vcpu_instantiate_race_free_under_miri() {
    assert_eq!(run_orchestrated(), Ok(vec![Value::I64(miri_spawn::RESULT)]));
}
