//! The **IoRing re-measure** (was CONSOLIDATION §5b; verdict + the parking plan that retires
//! the ring live in DESIGN.md §12 "Host-call ABI: async-first"). Run on demand, not in CI:
//!
//!   cargo test -p svm --release --test io_ring_bench -- --ignored --nocapture
//!
//! The ring's two original jobs were (1) boundary-crossing amortization (batching) and
//! (2) overlapping blocking host ops on the offload pool. §5b's premise: once a blocking host op
//! can simply *park the fiber*, job (2) is subsumed and only the batching question remains — so
//! measure ring-batched vs fast-path-sequential vs fiber/thread-overlapped and delete the ring on
//! measured redundancy (never on aesthetics, invariant 1).
//!
//! Three lanes, all computing the same checksum `Σ_{i=0}^{n-1} mix(i)` through the `Blocking`
//! fixture (iface 10 op 0 — the §5a offload-pool exerciser, granted directly by this harness):
//!
//! - **ring**: `n` 64-byte SQEs, ONE `cap.call` on the `IoRing` handle (`submit`, op 0), reap `n`
//!   CQEs — one crossing, pool-overlapped (`svm/tests/io_ring.rs` increment 2 shape, duplicated
//!   verbatim; test targets cannot share code without a fixture crate).
//! - **seq**: `n` direct `cap.call 10 0` in a loop — `n` crossings, inline on the vCPU thread
//!   (the `jit_diff::fast_cap` loop shape).
//! - **threaded**: the parallel bytecode tier (`thread.spawn`/`join`, THREADS.md), `n` worker
//!   vCPUs each issuing ONE direct blocking `cap.call` through the shared `Mutex<Host>` — the
//!   closest existing thing to "a blocking op parks the fiber". Whether this lane *actually*
//!   overlaps is itself a §5b finding: every cross-thread `cap.call` holds the one host lock for
//!   the duration of the dispatch, so a blocking op serializes the whole tier
//!   (`vcpu_shared_host_miri.rs` documents the lock).
//!
//! Two regimes:
//! - **overhead** (`block_for = 0`, large `n`, repeated): the pure ABI question — what does a
//!   ring crossing amortize vs `n` plain crossings when there is nothing to overlap.
//! - **blocking** (`block_for` real, small `n`): the overlap question — wall-clock per batch,
//!   plus each lane's realized pool concurrency (`AsyncState::max_active`) as proof of *why*.
//!
//! **§5b verdict (measured 2026-08-07, interp tier, pool K=4): the ring STAYS — the deletion
//! premise fails on measurement.** Numbers:
//! - *Batching is measurably negative*: Blocking SQEs at block=0, ring ~13.7 µs/op vs ~0.96 µs/op
//!   direct (14×); Clock SQEs (no pool, pure SQE/CQE ABI) ~10.0 µs/op vs ~0.75 µs/op direct
//!   (13×). The guest-side SQE build + host-side SQE parse + CQE writes swamp the one saved
//!   crossing — on this tier crossings are not the dominant cost the batching premise assumed.
//! - *But overlap is NOT subsumed*: at block=2 ms, n=8 — ring ~5.4 ms/batch (`max_active` = K=4,
//!   vs the 4-thread floor n/K·block = 4 ms), sequential ~17.1 ms (`max_active` 1, = n·block),
//!   and the threaded parallel tier ALSO ~18.4 ms (`max_active` 1): a cross-thread `cap.call`
//!   holds the one `Mutex<Host>` for the whole dispatch, so blocking ops serialize the tier.
//!   The §5b premise — "a blocking host op can simply park the fiber" — was never built; the
//!   ring is today the *only* mechanism that overlaps blocking host ops.
//!
//! Re-run this probe if parking-on-blocking ever lands: batching alone will not save the ring.

use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use svm_interp::{bytecode, run_with_host, Host, Region, Trap, Value, OFFLOAD_POOL_THREADS};
use svm_text::parse_module;
use svm_verify::verify_module;

/// The deterministic transform the mock `Blocking` op applies (mirrors `AsyncState::mix`).
fn mix(arg: i64) -> i64 {
    arg.wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}

fn checksum(n: u64) -> i64 {
    (0..n as i64).map(mix).fold(0i64, |a, b| a.wrapping_add(b))
}

// ----- lane 1: ring-batched (io_ring.rs increment-2 shape, duplicated verbatim) ----------------

/// Entry `(i32 ioring, i32 blocking) -> (i64)`: build `n` 64-byte SQEs at window offset 0 — each a
/// `Blocking.work` (`type_id 10, op 0, n_args 1, args[0] = i, user_data = i`) — `submit` the batch
/// on the IoRing handle, then sum the `n` CQE result fields (CQ at offset `sq_end`, past the SQ).
fn ring_src(n: u64) -> String {
    let cq = n * 64; // CQ starts just past the n·64-byte SQ
    format!(
        "memory 17
func (i32, i32) -> (i64) {{
block 0 (v0: i32, v1: i32) {{
  v2 = i64.const 0
  br 1(v0, v1, v2)
}}
block 1 (v3: i32, v4: i32, v5: i64) {{
  v6 = i64.const 64
  v7 = i64.mul v5 v6
  v8 = i32.const 10
  i32.store v7 v8
  v9 = i64.const 4
  v10 = i64.add v7 v9
  v11 = i32.const 0
  i32.store v10 v11
  v12 = i64.const 8
  v13 = i64.add v7 v12
  i32.store v13 v4
  v14 = i64.const 12
  v15 = i64.add v7 v14
  v16 = i32.const 1
  i32.store v15 v16
  v17 = i64.const 16
  v18 = i64.add v7 v17
  i64.store v18 v5
  v19 = i64.const 48
  v20 = i64.add v7 v19
  i64.store v20 v5
  v21 = i64.const 1
  v22 = i64.add v5 v21
  v23 = i64.const {n}
  v24 = i64.lt_u v22 v23
  br_if v24 1(v3, v4, v22) 2(v3)
}}
block 2 (v25: i32) {{
  v26 = i64.const 0
  v27 = i64.const {n}
  v28 = i64.const {cq}
  v29 = cap.call 9 0 (i64, i64, i64) -> (i64) v25 (v26, v27, v28)
  v30 = i64.const 0
  v31 = i64.const 0
  br 3(v30, v31)
}}
block 3 (v32: i64, v33: i64) {{
  v34 = i64.const 32
  v35 = i64.mul v32 v34
  v36 = i64.const {cq}
  v37 = i64.add v36 v35
  v38 = i64.const 8
  v39 = i64.add v37 v38
  v40 = i64.load v39
  v41 = i64.add v33 v40
  v42 = i64.const 1
  v43 = i64.add v32 v42
  v44 = i64.const {n}
  v45 = i64.lt_u v43 v44
  br_if v45 3(v43, v41) 4(v41)
}}
block 4 (v46: i64) {{
  return v46
  }}
}}
"
    )
}

/// `ring_src` with **Clock** SQEs (iface 2 op 0, zero args) instead of Blocking — the SQE kind
/// that is NOT offloaded: submit runs every deferred call inline on the submit thread, so this
/// lane prices the SQE/CQE ABI *alone* (batched crossings, no pool round-trip). Result: Σ 0..n-1
/// (the mock clock counts up per call).
fn ring_clock_src(n: u64) -> String {
    let cq = n * 64;
    format!(
        "memory 17
func (i32, i32) -> (i64) {{
block 0 (v0: i32, v1: i32) {{
  v2 = i64.const 0
  br 1(v0, v1, v2)
}}
block 1 (v3: i32, v4: i32, v5: i64) {{
  v6 = i64.const 64
  v7 = i64.mul v5 v6
  v8 = i32.const 2
  i32.store v7 v8
  v9 = i64.const 4
  v10 = i64.add v7 v9
  v11 = i32.const 0
  i32.store v10 v11
  v12 = i64.const 8
  v13 = i64.add v7 v12
  i32.store v13 v4
  v14 = i64.const 12
  v15 = i64.add v7 v14
  v16 = i32.const 0
  i32.store v15 v16
  v17 = i64.const 48
  v18 = i64.add v7 v17
  i64.store v18 v5
  v19 = i64.const 1
  v20 = i64.add v5 v19
  v21 = i64.const {n}
  v22 = i64.lt_u v20 v21
  br_if v22 1(v3, v4, v20) 2(v3)
}}
block 2 (v25: i32) {{
  v26 = i64.const 0
  v27 = i64.const {n}
  v28 = i64.const {cq}
  v29 = cap.call 9 0 (i64, i64, i64) -> (i64) v25 (v26, v27, v28)
  v30 = i64.const 0
  v31 = i64.const 0
  br 3(v30, v31)
}}
block 3 (v32: i64, v33: i64) {{
  v34 = i64.const 32
  v35 = i64.mul v32 v34
  v36 = i64.const {cq}
  v37 = i64.add v36 v35
  v38 = i64.const 8
  v39 = i64.add v37 v38
  v40 = i64.load v39
  v41 = i64.add v33 v40
  v42 = i64.const 1
  v43 = i64.add v32 v42
  v44 = i64.const {n}
  v45 = i64.lt_u v43 v44
  br_if v45 3(v43, v41) 4(v41)
}}
block 4 (v46: i64) {{
  return v46
  }}
}}
"
    )
}

/// Sequential direct `Clock.now` loop — the pair of [`ring_clock_src`] (n plain crossings).
fn seq_clock_src(n: u64) -> String {
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
  v11 = cap.call 2 0 () -> (i64) v8()
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

// ----- lane 2: fast-path-sequential (jit_diff::fast_cap loop shape) ----------------------------

/// Entry `(i32 blocking) -> (i64)`: `n` direct `cap.call 10 0` in a loop, summing the results.
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
  v11 = cap.call 10 0 (i64) -> (i64) v8(v10)
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

// ----- lane 3: threaded — n parallel vCPUs, one blocking cap.call each -------------------------

/// Main `(i32 blocking) -> (i64)`: spawn `n` workers (sp = index = the blocking arg, arg = the
/// handle), join all, return the shared accumulator at window offset 8. Worker: ONE
/// `cap.call 10 0` through the shared `Mutex<Host>`, `atomic.rmw.add` the result into the cell.
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

/// The `vcpu_shared_host_miri.rs` driver, duplicated verbatim (spawn/join orchestration over the
/// scoped-thread parallel tier; every child dispatches `cap.call` through the one `Mutex<Host>`).
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
            _ => panic!("unexpected event in the threaded §5b lane"),
        }
    }
}

/// One threaded-lane run: fresh shared window + fresh root, `n` workers, one blocking call each.
/// Returns `(checksum, realized pool concurrency)`.
fn run_threaded(prog: &bytecode::VcpuProgram, n: u64, block_for: Duration) -> (i64, usize) {
    let mut host = Host::new();
    let h = host.grant_blocking(block_for, None);
    let state = host.blocking_state(h).expect("blocking state");
    let pb = Mutex::new(host);

    let size = 1usize << 16;
    let layout = std::alloc::Layout::from_size_align(size, 8).unwrap();
    // SAFETY: non-zero layout; `size` valid 8-aligned bytes owned here until freed below.
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!base.is_null());
    // SAFETY: `base` is `size` valid 8-aligned bytes, exclusively this run's, freed only after.
    let back = Arc::new(unsafe { Region::shared(base, size as u64) });

    let orch = Orch::new();
    let root = bytecode::Vcpu::new_root(prog, 0, &[Value::I32(h)], Arc::clone(&back), &[])
        .expect("root vcpu")
        .with_shared_host(&pb);
    let r = std::thread::scope(|scope| drive(scope, prog, &back, &pb, &orch, root));

    drop(back);
    // SAFETY: same layout; the region (and all borrows) are gone (the scope joined every vCPU).
    unsafe { std::alloc::dealloc(base, layout) };
    let got = match r.expect("threaded run ok").pop().expect("one result") {
        Value::I64(x) => x,
        o => panic!("unexpected result {o:?}"),
    };
    assert_eq!(got, checksum(n), "threaded lane checksum");
    (got, state.max_active())
}

/// Run the ring lane once on `host` (window is re-zeroed per call), returning elapsed.
fn run_ring_once(
    m: &svm_ir::Module,
    ring: i32,
    blocking: i32,
    host: &mut Host,
    expected: i64,
) -> Duration {
    let init = vec![0u8; 128 << 10];
    let mut fuel = 500_000_000u64;
    let t = Instant::now();
    let (r, _mem) = svm_interp::run_capture_reserved_with_host(
        m,
        0,
        &[Value::I32(ring), Value::I32(blocking)],
        &mut fuel,
        &init,
        0,
        host,
    );
    let dt = t.elapsed();
    let got = match r.expect("ring run ok").pop().expect("one result") {
        Value::I64(x) => x,
        o => panic!("unexpected result {o:?}"),
    };
    assert_eq!(got, expected, "ring lane checksum");
    dt
}

/// Run the seq lane once on `host`, returning elapsed.
fn run_seq_once(m: &svm_ir::Module, blocking: i32, host: &mut Host, expected: i64) -> Duration {
    let mut fuel = 500_000_000u64;
    let t = Instant::now();
    let r = run_with_host(m, 0, &[Value::I32(blocking)], &mut fuel, host);
    let dt = t.elapsed();
    let got = match r.expect("seq run ok").pop().expect("one result") {
        Value::I64(x) => x,
        o => panic!("unexpected result {o:?}"),
    };
    assert_eq!(got, expected, "seq lane checksum");
    dt
}

fn parse(src: &str) -> svm_ir::Module {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    m
}

#[test]
#[ignore = "perf probe — run with --ignored --nocapture"]
fn ring_vs_seq_vs_threaded() {
    println!("offload pool width K = {OFFLOAD_POOL_THREADS}");

    // ---- regime A: pure ABI overhead (block_for = 0) — the batching question -----------------
    {
        let n = 64u64;
        let reps = 400u32;
        let ring_m = parse(&ring_src(n));
        let seq_m = parse(&seq_src(n));

        let mut hr = Host::new();
        let (ring_h, rb_h) = (hr.grant_io_ring(), hr.grant_blocking(Duration::ZERO, None));
        let mut hs = Host::new();
        let sb_h = hs.grant_blocking(Duration::ZERO, None);

        // warm
        for _ in 0..20 {
            run_ring_once(&ring_m, ring_h, rb_h, &mut hr, checksum(n));
            run_seq_once(&seq_m, sb_h, &mut hs, checksum(n));
        }
        let mut t_ring = Duration::ZERO;
        let mut t_seq = Duration::ZERO;
        for _ in 0..reps {
            t_ring += run_ring_once(&ring_m, ring_h, rb_h, &mut hr, checksum(n));
            t_seq += run_seq_once(&seq_m, sb_h, &mut hs, checksum(n));
        }
        let per = |t: Duration| t.as_nanos() as f64 / reps as f64 / n as f64;
        println!(
            "overhead (block=0, n={n}, reps={reps}): ring {:.0} ns/op  seq {:.0} ns/op  (ring/seq {:.2}x)",
            per(t_ring),
            per(t_seq),
            per(t_ring) / per(t_seq),
        );
    }

    // ---- regime A2: SQE/CQE ABI alone — Clock SQEs are NOT offloaded (inline deferred
    // dispatch on the submit thread), so this is batched crossings vs plain crossings with the
    // pool out of the picture entirely.
    {
        let n = 64u64;
        let reps = 400u32;
        let expected = (n * (n - 1) / 2) as i64; // mock clock counts up per call
        let ring_m = parse(&ring_clock_src(n));
        let seq_m = parse(&seq_clock_src(n));

        let mut hr = Host::new();
        let (ring_h, rc_h) = (hr.grant_io_ring(), hr.grant_clock());
        let mut hs = Host::new();
        let sc_h = hs.grant_clock();

        // The mock clock is monotonic across runs on one host: recompute expected per run.
        let mut clock_i = 0i64;
        let mut clock_s = 0i64;
        let next = |c: &mut i64| {
            let e: i64 = (0..n as i64).map(|k| *c + k).sum();
            *c += n as i64;
            e
        };
        let _ = expected;
        for _ in 0..20 {
            run_ring_once(&ring_m, ring_h, rc_h, &mut hr, next(&mut clock_i));
            run_seq_once(&seq_m, sc_h, &mut hs, next(&mut clock_s));
        }
        let mut t_ring = Duration::ZERO;
        let mut t_seq = Duration::ZERO;
        for _ in 0..reps {
            t_ring += run_ring_once(&ring_m, ring_h, rc_h, &mut hr, next(&mut clock_i));
            t_seq += run_seq_once(&seq_m, sc_h, &mut hs, next(&mut clock_s));
        }
        let per = |t: Duration| t.as_nanos() as f64 / reps as f64 / n as f64;
        println!(
            "abi-only (clock SQEs, no pool; n={n}, reps={reps}): ring {:.0} ns/op  seq {:.0} ns/op  (ring/seq {:.2}x)",
            per(t_ring),
            per(t_seq),
            per(t_ring) / per(t_seq),
        );
    }

    // ---- regime B: real blocking — the overlap question --------------------------------------
    {
        let n = 8u64;
        let block = Duration::from_millis(2);
        let reps = 5u32;
        let ring_m = parse(&ring_src(n));
        let seq_m = parse(&seq_src(n));
        let thr_m = parse_module(&threaded_src(n)).expect("parse threaded");
        let thr_prog = bytecode::VcpuProgram::compile(&thr_m).expect("compile threaded");

        let mut hr = Host::new();
        let (ring_h, rb_h) = (hr.grant_io_ring(), hr.grant_blocking(block, None));
        let rb_state = hr.blocking_state(rb_h).expect("state");
        let mut hs = Host::new();
        let sb_h = hs.grant_blocking(block, None);
        let sb_state = hs.blocking_state(sb_h).expect("state");

        let mut t_ring = Duration::ZERO;
        let mut t_seq = Duration::ZERO;
        let mut t_thr = Duration::ZERO;
        let mut thr_active = 0usize;
        for _ in 0..reps {
            t_ring += run_ring_once(&ring_m, ring_h, rb_h, &mut hr, checksum(n));
            t_seq += run_seq_once(&seq_m, sb_h, &mut hs, checksum(n));
            let t = Instant::now();
            let (_sum, active) = run_threaded(&thr_prog, n, block);
            t_thr += t.elapsed();
            thr_active = thr_active.max(active);
        }
        let per = |t: Duration| t.as_micros() as f64 / reps as f64;
        println!(
            "blocking (block={}us, n={n}, reps={reps}): ring {:.0} us/batch (pool max_active {})  \
             seq {:.0} us/batch (max_active {})  threaded {:.0} us/batch (max_active {})",
            block.as_micros(),
            per(t_ring),
            rb_state.max_active(),
            per(t_seq),
            sb_state.max_active(),
            per(t_thr),
            thr_active,
        );
        println!(
            "  ideal overlap floor = {} us; full serialization = {} us",
            block.as_micros(),
            block.as_micros() * n as u128,
        );
    }
}
