//! **#1011 slice 3a — a §14 confined child carries a re-granted powerbox.** The JIT-drivable nested
//! child path (`Vcpu::new_confined_child`) built children with only their attenuated
//! `Instantiator`+`AddressSpace` — "its capability set never includes the run's I/O grants". The nim
//! compiler driver needs to hand each phase child a **shared `fs`** so `nifler → nimsem → hexer` pass
//! `.nif` files; that is the op-13 grant list, which today only the *interpreter's* inline child run
//! honors. This proves the enabling primitive: `new_confined_child_granted` lets the caller install
//! named caps into the child's powerbox, and the confined child resolves one by name
//! (`cap.self.resolve`) and calls it — the shape a JIT'd phase child will use, its `cap.call` a
//! cross-tier bounce to the granted handler (window confinement, §2, is untouched — the grant is
//! authority, §3, not a window access).

use std::sync::{Arc, Mutex};
use svm_interp::{bytecode, ForkedProc, Host, HostProc, Region, Trap, Value};
use svm_text::parse_module;

// The child entry (its `Instantiator` handle arrives as `v0`, unused here) seeds the name `"fs"` into
// its window (`0x7366` little-endian = 'f','s'), resolves it to a handle, and calls the granted
// `HOST_PROC` cap (type 13, op 0) — returning the handler's result. A granted counter returns its
// post-increment value, so a correct run returns 1.
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

#[test]
fn confined_child_resolves_and_calls_a_granted_cap() {
    let m = parse_module(CHILD).expect("parse");
    let prog = bytecode::VcpuProgram::compile(&m).expect("compile");

    // The granted cap: a forkable host-proc counter (the re-grantable shape a shared `fs` uses),
    // sharing one `Arc` so a call from inside the child is observable here.
    let counter = Arc::new(Mutex::new(0i64));

    // A 64 KiB carve, exactly the child's declared window (`memory 16`).
    let size = 1usize << 16;
    let layout = std::alloc::Layout::from_size_align(size, 8).unwrap();
    // SAFETY: non-zero layout; `size` valid 8-aligned bytes owned here until freed below.
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!base.is_null());
    // SAFETY: `base` addresses `size` valid bytes, exclusively this run's, freed only after the vCPU.
    let back = Arc::new(unsafe { Region::shared(base, size as u64) });

    let r = {
        let counter = Arc::clone(&counter);
        // Install "fs" -> a forkable counter into the freshly-built child powerbox. In the real driver
        // this closure is the interpreter's op-13 `regrant_into_child` + `register_cap_name`; here it
        // mints the cap directly so the test needs no parent host.
        let mut install = move |child: &mut Host| {
            let c1 = Arc::clone(&counter);
            let handler: HostProc = Box::new(move |_op, _args, _mem, _| {
                let mut c = c1.lock().unwrap();
                *c += 1;
                Ok(vec![*c])
            });
            let c2 = Arc::clone(&counter);
            let fork = Arc::new(move |_pid: u64| {
                let c = Arc::clone(&c2);
                ForkedProc::shared(Box::new(move |_op, _args, _mem, _| {
                    let mut c = c.lock().unwrap();
                    *c += 1;
                    Ok(vec![*c])
                }))
            });
            let h = child.grant_host_proc_forkable(handler, fork);
            child.register_cap_name("fs", h);
        };
        let mut vcpu = bytecode::Vcpu::new_confined_child_granted(
            &prog,
            0,
            0,
            Arc::clone(&back),
            16,
            u64::MAX,
            &mut install,
        )
        .expect("granted confined child builds");
        // A leaf child (no sub-instantiate / join / tier-up) runs straight to completion in one `run()`.
        match vcpu.run() {
            bytecode::VcpuEvent::Done(v) => Ok::<_, Trap>(v),
            bytecode::VcpuEvent::Trapped(t) => Err(t),
            _ => panic!("unexpected event from a leaf child (expected Done/Trapped)"),
        }
    };

    drop(back);
    // SAFETY: same layout; the vCPU and its region view are dropped, so no borrow outlives this.
    unsafe { std::alloc::dealloc(base, layout) };

    assert_eq!(
        r,
        Ok(vec![Value::I64(1)]),
        "the child resolved 'fs' by name and called the granted cap (counter -> 1)"
    );
    assert_eq!(
        *counter.lock().unwrap(),
        1,
        "the granted handler ran inside the confined child, over the shared state"
    );
}
