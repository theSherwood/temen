//! THREADS.md 4c-domain §14-D2 — **Miri** verification of §14 confined executor children through the
//! **resumable `Vcpu`** API. The differential test in `temen/tests/bytecode_vcpu_orchestration_instantiate.rs`
//! proves the path agrees with the cooperative oracle; this proves the genuinely-new machinery — a
//! confined child built over a **fresh `Region::shared` aliasing the parent window's carve** (per
//! DESIGN.md §14 a sub-window is indistinguishable from a top-level window), run on its own thread,
//! its writes handed back to the parent across the join — is **free of data races / UB / provenance
//! errors**. Miri's checker, not the iteration count, is the point, so the kernel (2 children) is small.
//!
//! Run: `cargo +nightly miri test -p temen-interp --test vcpu_instantiate_miri`

use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex};
use temen_interp::{bytecode, Host, Region, Trap, Value};
use temen_text::parse_module;

// Root (instantiator) instantiates 2 confined children (func 1) at 64 KiB / 68 KiB; each child writes
// the marker 21 at its own offset 0 (→ the shared backing, through its own carve region) and returns
// 5. The parent joins both (the happens-before that publishes the children's writes), reads both
// markers back through its own window, and returns 5 + 5 + 21 + 21 = 52 — real cross-thread
// non-atomic access through two aliasing region views of one allocation.
const SRC: &str = r#"memory 17
func (i32) -> (i64) {
block 0 (v0: i32) {
  ve = i64.const 1
  vsl = i64.const 12
  vq = i64.const 0
  ; spawn via record (op 17): entry=1 off=65536 sl=12 quota=0
  q0v0 = i64.const 4294967296
  q0v1 = i64.const 65536
  q0v2 = i64.const -4294967284
  q0v3 = i64.const 4294967295
  q0v4 = i64.const 0
  q0a0 = i64.const 17536
  i64.store q0a0 q0v0
  q0a1 = i64.const 17544
  i64.store q0a1 q0v1
  q0a2 = i64.const 17552
  i64.store q0a2 q0v2
  q0a3 = i64.const 17560
  i64.store q0a3 q0v3
  q0a4 = i64.const 17568
  i64.store q0a4 q0v4
  q0a5 = i64.const 17576
  i64.store q0a5 q0v4
  q0a6 = i64.const 17584
  i64.store q0a6 q0v4
  vh0 = cap.call 6 17 (i64) -> (i32) v0 (q0a0)
  ; spawn via record (op 17): entry=1 off=69632 sl=12 quota=0
  q1v0 = i64.const 4294967296
  q1v1 = i64.const 69632
  q1v2 = i64.const -4294967284
  q1v3 = i64.const 4294967295
  q1v4 = i64.const 0
  q1a0 = i64.const 17600
  i64.store q1a0 q1v0
  q1a1 = i64.const 17608
  i64.store q1a1 q1v1
  q1a2 = i64.const 17616
  i64.store q1a2 q1v2
  q1a3 = i64.const 17624
  i64.store q1a3 q1v3
  q1a4 = i64.const 17632
  i64.store q1a4 q1v4
  q1a5 = i64.const 17640
  i64.store q1a5 q1v4
  q1a6 = i64.const 17648
  i64.store q1a6 q1v4
  vh1 = cap.call 6 17 (i64) -> (i32) v0 (q1a0)
  vr0 = cap.call 6 1 (i32) -> (i64) v0 (vh0)
  vr1 = cap.call 6 1 (i32) -> (i64) v0 (vh1)
  vm0a = i64.const 65536
  vm0 = i32.load8_u vm0a
  vm0e = i64.extend_i32_u vm0
  vm1a = i64.const 69632
  vm1 = i32.load8_u vm1a
  vm1e = i64.extend_i32_u vm1
  vs1 = i64.add vr0 vr1
  vs2 = i64.add vs1 vm0e
  vs3 = i64.add vs2 vm1e
  return vs3
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vaddr = i64.const 0
  v21 = i32.const 21
  i32.store8 vaddr v21
  v5 = i64.const 5
  return v5
  }
}
"#;

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

/// A window base pointer that crosses the scoped-thread hand-off (raw pointers aren't `Send`) while
/// keeping **derived provenance** for Miri (no int→ptr round-trip). Sound: it is only ever offset
/// into the one live allocation and handed to `Region::shared`.
#[derive(Clone, Copy)]
struct WinPtr(*mut u8);
// SAFETY: the pointee is the shared window, whose cross-thread access discipline is exactly what
// this test exists to check; the pointer value itself is plain data.
unsafe impl Send for WinPtr {}

fn drive<'s, 'e>(
    scope: &'s std::thread::Scope<'s, 'e>,
    prog: &'e bytecode::VcpuProgram,
    win: WinPtr,
    orch: &'e Orch,
    mut vcpu: bytecode::Vcpu<'e>,
) -> Result<Vec<Value>, Trap> {
    let mut handles: Vec<u64> = Vec::new();
    loop {
        match vcpu.run() {
            bytecode::VcpuEvent::Done(vals) => return Ok(vals),
            bytecode::VcpuEvent::Trapped(t) => return Err(t),
            bytecode::VcpuEvent::Instantiate {
                module,
                entry,
                carve,
                size_log2,
                fuel,
            } => {
                let id = orch.fresh_id();
                // SAFETY: the engine validated the carve inside this vCPU's window, which outlives the
                // scope; the aliasing views are the §13 shared data plane, cross-thread access ordered
                // by the spawn hand-off / join rendezvous — exactly what Miri checks here.
                let child_win = WinPtr(unsafe { win.0.add(carve as usize) });
                // SAFETY: as above — `2^size_log2` valid bytes at the validated carve.
                let back = Arc::new(unsafe { Region::shared(child_win.0, 1u64 << size_log2) });
                let child =
                    bytecode::Vcpu::new_confined_child(prog, module, entry, back, size_log2, fuel)
                        .expect("confined child vcpu");
                scope.spawn(move || {
                    let r = drive(scope, prog, child_win, orch, child);
                    orch.publish(id, r);
                });
                let handle = handles.len() as i32;
                handles.push(id);
                vcpu.deliver_handle(handle);
            }
            bytecode::VcpuEvent::Join { handle } => {
                let id = handles[handle as usize];
                vcpu.deliver_join(orch.join(id));
            }
            _ => panic!("unexpected event in the instantiate Miri kernel"),
        }
    }
}

fn run_orchestrated() -> Result<Vec<Value>, Trap> {
    let m = parse_module(SRC).unwrap();
    let prog = bytecode::VcpuProgram::compile(&m).expect("compile");
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, 128 << 10);

    let size = 1usize << 17;
    let layout = std::alloc::Layout::from_size_align(size, 8).unwrap();
    // SAFETY: non-zero layout; `size` valid 8-aligned bytes owned here until freed below.
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!base.is_null());
    // SAFETY: `base` is `size` valid 8-aligned bytes, exclusively this run's, freed only after.
    let back = Arc::new(unsafe { Region::shared(base, size as u64) });

    let orch = Orch::new();
    let root = bytecode::Vcpu::new_root_with_powerbox(
        &prog,
        0,
        &[Value::I32(inst)],
        Arc::clone(&back),
        &[],
        host,
    )
    .expect("root vcpu");
    let r = std::thread::scope(|scope| drive(scope, &prog, WinPtr(base), &orch, root));

    drop(back);
    // SAFETY: same layout; the region (and all borrows) are gone (the scope joined every vCPU).
    unsafe { std::alloc::dealloc(base, layout) };
    r
}

#[test]
fn vcpu_instantiate_race_free_under_miri() {
    assert_eq!(run_orchestrated(), Ok(vec![Value::I64(52)])); // 5 + 5 + 21 + 21
}
