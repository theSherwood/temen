//! §12 **parking-on-blocking, slice 1** (the plan that retires the ring — DESIGN.md §12): an
//! *offloadable* host-proc registration declares blockingness (`Host::grant_host_proc_offloadable`),
//! the handler decides per call (`Done` inline / `Offload` punt — the io_uring try-nonblocking
//! model), and a punting dispatch on the parking-aware face returns `Pending(completion_id)` so the
//! eval loop waits **without the host lock** (the single-fiber degenerate wait). Pins:
//!
//! - punt and inline produce identical results on every tier (tree-walk, bytecode, Cranelift JIT —
//!   the JIT stays on the sync face and runs punts inline: decline-never-diverge, INVARIANTS.md 9);
//! - the fast case (`Done`) never touches the parking machinery;
//! - **sync ops never pay**: a plain host proc / a `block_for = 0` `Blocking` call mints nothing;
//! - the shared-host parallel tier genuinely overlaps punted blocking ops (the §5b
//!   `max_active = 1` serialization fix), proven by rendezvous, not timing.

use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use temen_interp::{
    bytecode, run_with_host, Host, OffloadHostProc, OffloadOutcome, Region, Trap, Value,
};
use temen_jit::JitOutcome;
use temen_text::parse_module;
use temen_verify::verify_module;

/// The deterministic transform the punted jobs apply (mirrors `AsyncState::mix`).
fn mix(arg: i64) -> i64 {
    arg.wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}

fn checksum(n: u64) -> i64 {
    (0..n as i64).map(mix).fold(0i64, |a, b| a.wrapping_add(b))
}

/// `(i32 handle) -> (i64)`: `n` direct `cap.call 13 0` (HOST_PROC) in a loop, summing the results
/// (the `io_ring_bench` seq lane shape, pointed at the offloadable registration).
fn seq_src(n: u64) -> String {
    format!(
        "func (i32) -> (i64) {{
block 0 (v0: i32) {{
  v1 = i64.const 0
  v2 = i64.const 0
  br 1(v0, v1, v2)
}}
block 1 (v3: i32, v4: i64, v5: i64) {{
  v6 = i64.const {n}
  v7 = i64.lt_s v5 v6
  br_if v7 2(v3, v4, v5) 3(v4)
}}
block 2 (v8: i32, v9: i64, v10: i64) {{
  v11 = cap.call 13 0 (i64) -> (i64) v8(v10)
  v12 = i64.add v9 v11
  v13 = i64.const 1
  v14 = i64.add v10 v13
  br 1(v8, v12, v14)
}}
block 3 (v15: i64) {{
  return v15
  }}
}}
"
    )
}

fn parse(src: &str) -> temen_ir::Module {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    m
}

/// An offloadable handler that always punts: op 0 returns `mix(arg)` off the vCPU thread.
fn punting_handler() -> OffloadHostProc {
    Box::new(|_op, args| {
        let a = *args.first().unwrap_or(&0);
        OffloadOutcome::Offload(Box::new(move || mix(a)))
    })
}

/// An offloadable handler exercising the per-call decision: even args complete inline (`Done`),
/// odd args punt.
fn deciding_handler() -> OffloadHostProc {
    Box::new(|_op, args| {
        let a = *args.first().unwrap_or(&0);
        if a % 2 == 0 {
            OffloadOutcome::Done(Ok(vec![mix(a)]))
        } else {
            OffloadOutcome::Offload(Box::new(move || mix(a)))
        }
    })
}

/// Run `seq_src(n)` over `handler` on all three tiers; return each tier's result and the number of
/// completion ids its host minted (the parking-machinery odometer).
fn run_all_tiers(n: u64, handler: fn() -> OffloadHostProc) -> [(i64, u64); 3] {
    let m = parse(&seq_src(n));

    // Tree-walk oracle — parking-aware (the pending face; punts wait without the lock).
    let mut hi = Host::new();
    let h = hi.grant_host_proc_offloadable(handler());
    let comps_i = hi.completions();
    let mut fuel = 1_000_000u64;
    let iv = run_with_host(&m, 0, &[Value::I32(h)], &mut fuel, &mut hi).expect("interp ok");
    let iv = match iv.as_slice() {
        [Value::I64(x)] => *x,
        o => panic!("unexpected interp result {o:?}"),
    };
    assert_eq!(
        comps_i.outstanding(),
        0,
        "interp: all completions delivered"
    );

    // Bytecode interpreter — parking-aware.
    let prog = bytecode::VcpuProgram::compile(&m).expect("compile");
    let mut hb = Host::new();
    let hbh = hb.grant_host_proc_offloadable(handler());
    let comps_b = hb.completions();
    let mut vcpu =
        bytecode::Vcpu::new_root_reserved_with_powerbox(&prog, 0, &[Value::I32(hbh)], &[], hb, 20)
            .expect("vcpu");
    let bv = match vcpu.run() {
        bytecode::VcpuEvent::Done(vals) => match vals.as_slice() {
            [Value::I64(x)] => *x,
            o => panic!("unexpected bytecode result {o:?}"),
        },
        _ => panic!("unexpected bytecode event (not Done)"),
    };
    assert_eq!(
        comps_b.outstanding(),
        0,
        "bytecode: all completions delivered"
    );

    // Cranelift JIT — sync face only (generic `cap_thunk`): punts run inline, same results
    // (decline-never-diverge). Must mint nothing.
    let mut hj = Host::new();
    let hjh = hj.grant_host_proc_offloadable(handler());
    let comps_j = hj.completions();
    let jo = temen_jit::compile_and_run_with_host(
        &m,
        0,
        &[hjh as i64],
        temen_run::cap_thunk,
        &mut hj as *mut Host as *mut core::ffi::c_void,
    )
    .expect("jit");
    let jv = match jo {
        JitOutcome::Returned(vals) => vals[0],
        o => panic!("unexpected jit outcome {o:?}"),
    };

    [
        (iv, comps_i.minted()),
        (bv, comps_b.minted()),
        (jv, comps_j.minted()),
    ]
}

#[test]
fn offloadable_punt_matches_on_all_tiers() {
    let n = 8u64;
    let [(iv, im), (bv, bm), (jv, jm)] = run_all_tiers(n, punting_handler);
    let want = checksum(n);
    assert_eq!(iv, want, "tree-walk sum");
    assert_eq!(bv, want, "bytecode sum");
    assert_eq!(jv, want, "jit sum");
    // The two parking-aware tiers punted every call; the JIT's sync face ran them inline.
    assert_eq!(im, n, "tree-walk minted one completion per punt");
    assert_eq!(bm, n, "bytecode minted one completion per punt");
    assert_eq!(
        jm, 0,
        "jit sync face must never touch the parking machinery"
    );
}

#[test]
fn offloadable_fast_case_stays_inline() {
    let n = 8u64;
    let [(iv, im), (bv, bm), (jv, jm)] = run_all_tiers(n, deciding_handler);
    let want = checksum(n);
    assert_eq!(iv, want, "tree-walk sum");
    assert_eq!(bv, want, "bytecode sum");
    assert_eq!(jv, want, "jit sum");
    // Only the odd args (half of n) punted — `Done` never touches the parking machinery.
    assert_eq!(im, n / 2, "tree-walk: only the punts minted");
    assert_eq!(bm, n / 2, "bytecode: only the punts minted");
    assert_eq!(
        jm, 0,
        "jit sync face must never touch the parking machinery"
    );
}

#[test]
fn sync_ops_never_touch_parking_machinery() {
    let n = 16u64;
    let m = parse(&seq_src(n));

    // A plain (synchronous) host proc through the same guest loop: the parking odometer must
    // read zero — no completion id, no table touch (the §12 pin).
    let mut host = Host::new();
    let h = host.grant_host_proc(Box::new(|_op, args, _mem, _minter| {
        Ok(vec![mix(*args.first().unwrap_or(&0))])
    }));
    let comps = host.completions();
    let mut fuel = 1_000_000u64;
    let r = run_with_host(&m, 0, &[Value::I32(h)], &mut fuel, &mut host).expect("interp ok");
    assert_eq!(r, vec![Value::I64(checksum(n))]);
    assert_eq!(comps.minted(), 0, "plain host proc minted a completion id");

    // A `block_for = 0` Blocking call is the offloadable fast case — inline, nothing minted.
    let bsrc = seq_src(n).replace("cap.call 13 0", "cap.call 10 0");
    let bm = parse(&bsrc);
    let mut host = Host::new();
    let bh = host.grant_blocking(Duration::ZERO, None);
    let comps = host.completions();
    let mut fuel = 1_000_000u64;
    let r = run_with_host(&bm, 0, &[Value::I32(bh)], &mut fuel, &mut host).expect("interp ok");
    assert_eq!(r, vec![Value::I64(checksum(n))]);
    assert_eq!(comps.minted(), 0, "block=0 Blocking minted a completion id");
}

// ----- shared-host parallel tier: punted blocking ops genuinely overlap -----------------------

/// Main `(i32 blocking) -> (i64)`: spawn `n` workers, join, return the shared accumulator at
/// window offset 8. Worker: ONE `cap.call 10 0`, `atomic.rmw.add` the result into the cell (the
/// `io_ring_bench` threaded lane shape).
fn threaded_src(n: u64) -> String {
    format!(
        "memory 16
func (i32) -> (i64) {{
block 0 (v0: i32) {{
  vh = i64.extend_i32_u v0
  vi0 = i64.const 0
  br 1(vi0, vh)
}}
block 1 (vi: i64, vh1: i64) {{
  vn = i64.const {n}
  vlt = i64.lt_u vi vn
  br_if vlt 2(vi, vh1) 3()
}}
block 2 (vi2: i64, vh2: i64) {{
  vt = thread.spawn 1 vi2 vh2
  vm = i64.const 4
  voff = i64.mul vi2 vm
  vb = i64.const 16
  va = i64.add vb voff
  i32.store va vt
  vone = i64.const 1
  vnext = i64.add vi2 vone
  br 1(vnext, vh2)
}}
block 3 () {{
  vj0 = i64.const 0
  br 4(vj0)
}}
block 4 (vj: i64) {{
  vn2 = i64.const {n}
  vlt2 = i64.lt_u vj vn2
  br_if vlt2 5(vj) 6()
}}
block 5 (vj2: i64) {{
  vm2 = i64.const 4
  voff2 = i64.mul vj2 vm2
  vb2 = i64.const 16
  va2 = i64.add vb2 voff2
  vt2 = i32.load va2
  vjr = thread.join vt2
  vone2 = i64.const 1
  vnj = i64.add vj2 vone2
  br 4(vnj)
}}
block 6 () {{
  vz = i64.const 8
  vsum = i64.atomic.load vz
  return vsum
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (vsp: i64, vharg: i64) {{
  vhandle = i32.wrap_i64 vharg
  vr = cap.call 10 0 (i64) -> (i64) vhandle(vsp)
  vaddr = i64.const 8
  vold = i64.atomic.rmw.add vaddr vr
  vz = i64.const 0
  return vz
  }}
}}
"
    )
}

/// The `vcpu_shared_host_miri.rs` driver, duplicated verbatim (test targets cannot share code
/// without a fixture crate): spawn/join orchestration over the scoped-thread parallel tier.
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
    back: &'e Arc<Region>,
    pb: &'e Mutex<Host>,
    orch: &'e Orch,
    mut vcpu: bytecode::Vcpu<'e>,
) -> Result<Vec<Value>, Trap> {
    let mut handles: Vec<u64> = Vec::new();
    loop {
        match vcpu.run() {
            bytecode::VcpuEvent::Done(vals) => return Ok(vals),
            bytecode::VcpuEvent::Trapped(t) => return Err(t),
            bytecode::VcpuEvent::Spawn {
                func,
                sp,
                arg,
                module,
            } => {
                let id = orch.fresh_id();
                let child = bytecode::Vcpu::new_child_in(
                    prog,
                    module,
                    func,
                    &[Value::I64(sp), Value::I64(arg)],
                    Arc::clone(back),
                )
                .expect("child vcpu")
                .with_shared_host(pb);
                scope.spawn(move || {
                    let r = drive(scope, prog, back, pb, orch, child);
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
            _ => panic!("unexpected event in the parallel parking test"),
        }
    }
}

/// **The §5b serialization fix, pinned without timing.** Two parallel vCPUs each make one
/// blocking `cap.call` on a rendezvous-2 `Blocking` handle: each punts to the offload pool and
/// waits *outside* the shared host lock, so both jobs co-reside on the pool and meet the width-2
/// barrier. Under the old hold-the-lock-across-dispatch regime this deadlocks (the first caller
/// blocks on the barrier inside the lock; the second can never dispatch), so completing at all —
/// plus `max_active == 2` — proves the overlap.
#[test]
fn parallel_punted_blocking_ops_overlap_on_the_pool() {
    let m = parse_module(&threaded_src(2)).expect("parse threaded");
    let prog = bytecode::VcpuProgram::compile(&m).expect("compile threaded");

    let mut host = Host::new();
    let h = host.grant_blocking(Duration::ZERO, Some(2));
    let state = host.blocking_state(h).expect("blocking state");
    let comps = host.completions();
    let pb = Mutex::new(host);

    let size = 1usize << 16;
    let layout = std::alloc::Layout::from_size_align(size, 8).unwrap();
    // SAFETY: non-zero layout; `size` valid 8-aligned bytes owned here until freed below.
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!base.is_null());
    // SAFETY: `base` is `size` valid 8-aligned bytes, exclusively this run's, freed only after.
    let back = Arc::new(unsafe { Region::shared(base, size as u64) });

    let orch = Orch::new();
    let root = bytecode::Vcpu::new_root(&prog, 0, &[Value::I32(h)], Arc::clone(&back), &[])
        .expect("root vcpu")
        .with_shared_host(&pb);
    let r = std::thread::scope(|scope| drive(scope, &prog, &back, &pb, &orch, root));

    drop(back);
    // SAFETY: same layout; the region (and all borrows) are gone (the scope joined every vCPU).
    unsafe { std::alloc::dealloc(base, layout) };

    let got = match r.expect("parallel run ok").pop().expect("one result") {
        Value::I64(x) => x,
        o => panic!("unexpected result {o:?}"),
    };
    assert_eq!(got, mix(0).wrapping_add(mix(1)), "parallel checksum");
    assert_eq!(
        state.max_active(),
        2,
        "both punted ops co-resided on the pool"
    );
    assert_eq!(comps.minted(), 2, "one completion per punted call");
    assert_eq!(comps.outstanding(), 0, "all completions delivered");
}

/// **The JIT threaded tier's version of the same pin**: two OS threads call `cap_thunk_locked`
/// (the serialized per-domain `Mutex<Host>` thunk) concurrently against a rendezvous-2 `Blocking`
/// handle. Each dispatch punts, and the thunk releases the domain lock before waiting on the
/// completion — so both jobs co-reside on the offload pool and meet the barrier. Under the old
/// delegate-under-guard regime this deadlocks (the first caller sleeps on the barrier holding the
/// lock), so completion at all — plus `max_active == 2` — proves the release-before-wait.
#[test]
fn jit_locked_thunk_releases_lock_before_completion_wait() {
    let mut host = Host::new();
    let h = host.grant_blocking(Duration::ZERO, Some(2));
    let state = host.blocking_state(h).expect("blocking state");
    let comps = host.completions();
    let m = Mutex::new(host);
    let ctx = &m as *const Mutex<Host> as *mut core::ffi::c_void;
    // SAFETY: `ctx` is a live `*const Mutex<Host>` for the scope; args/results/trap_out are valid
    // per-thread locals; no window (`mem_base` null) — the `CapThunk` contract.
    let addr = ctx as usize;
    let results: Vec<i64> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..2i64)
            .map(|i| {
                s.spawn(move || {
                    let arg = [i];
                    let mut result = 0i64;
                    let mut trap = 0i64;
                    unsafe {
                        temen_run::cap_thunk_locked(
                            addr as *mut core::ffi::c_void,
                            std::ptr::null_mut(),
                            0,
                            0,
                            10, // cap_id::BLOCKING
                            0,
                            h,
                            arg.as_ptr(),
                            1,
                            &mut result,
                            1,
                            &mut trap,
                        );
                    }
                    assert_eq!(trap, 0, "worker {i}: clean dispatch");
                    result
                })
            })
            .collect();
        handles.into_iter().map(|t| t.join().unwrap()).collect()
    });
    let sum = results.iter().fold(0i64, |a, b| a.wrapping_add(*b));
    assert_eq!(sum, mix(0).wrapping_add(mix(1)), "locked-thunk checksum");
    assert_eq!(state.max_active(), 2, "both punts co-resided on the pool");
    assert_eq!(comps.minted(), 2, "one completion per punted call");
    assert_eq!(comps.outstanding(), 0, "all completions delivered");
}

/// **Slice 2 — parked vCPUs free their workers (the M:N win).** Four tree-walk vCPUs each punt
/// one call on a rendezvous-4 `Blocking` handle. A punting vCPU dispatches its pool job *before*
/// parking (`Blocked::CapPending`), so all four jobs reach the pool and meet the width-4 barrier
/// even when the scheduler has fewer worker threads than vCPUs — under a block-the-worker regime
/// this hangs whenever workers < 4. Completions re-queue the parked vCPUs in submission order
/// (`Pending::CapResult`).
#[test]
fn oracle_parked_vcpus_free_their_workers() {
    let m = parse(&threaded_src(4));
    let mut host = Host::new();
    let h = host.grant_blocking(Duration::ZERO, Some(4));
    let state = host.blocking_state(h).expect("blocking state");
    let comps = host.completions();
    let mut fuel = 50_000_000u64;
    let init = vec![0u8; 1 << 16];
    let (r, _mem) = temen_interp::run_capture_reserved_with_host(
        &m,
        0,
        &[Value::I32(h)],
        &mut fuel,
        &init,
        0,
        &mut host,
    );
    let got = match r.expect("oracle run ok").pop().expect("one result") {
        Value::I64(x) => x,
        o => panic!("unexpected result {o:?}"),
    };
    let want = (0..4).fold(0i64, |a, b| a.wrapping_add(mix(b)));
    assert_eq!(got, want, "oracle parked checksum");
    assert_eq!(
        state.max_active(),
        4,
        "all four punts co-resided on the pool"
    );
    assert_eq!(comps.minted(), 4, "one completion per punted call");
    assert_eq!(comps.outstanding(), 0, "all completions delivered");
}
