//! Shared fork **differential** harness (FORK.md §9 / OPS_PARITY.md `clone_caller`/`reap`).
//!
//! Native fork on the fast tiers lives on the bytecode cooperative `drive`; today it is pinned by a
//! single hand-written topology (`clone_caller.rs::bytecode_forks_the_twin_identically_to_the_oracle`).
//! This harness hardens that: it builds the same `SRC_TWIN` fork topology with **fuzz-chosen reply /
//! reaped-status values** and asserts the **bytecode** run matches the **tree-walk oracle** on the
//! observable result — the return value plus the *sorted* shared-stdout multiset. Order-independence
//! is deliberate: fork's post-return twin ordering is unspecified (the oracle's Real scheduler
//! interleaves the two copies; INVARIANTS.md #6 "post-teardown sibling effects are unspecified"), so
//! only the multiset is a semantic fact. A divergence here is a bytecode-fork miscompile.
//!
//! Driven by the `fork_diff` libFuzzer target and a stable seed-sweep test (`fork_fuzz.rs`).
#![allow(dead_code)]

use std::sync::Arc;
use svm_interp::{bytecode, run_with_host, Host, StreamRole, Trap, Value};

/// The `SRC_TWIN` topology with the two reply constants tokenized: a manager spawns a server + a
/// guest, the guest `fork()`s (`clone_caller(__RO__, __RT__)`), the twin runs over a private window +
/// duplicated powerbox, both copies write their reply to the shared stdout, and the parent `wait`s
/// the twin (`reap`). Returns the original's reply (`__RO__`). (Kept in lockstep with `SRC_TWIN` in
/// `crates/svm-interp/tests/clone_caller.rs`.)
const TEMPLATE: &str = r#"
memory 18
type 0 func (i64) -> (i64)
type 1 interface { fork: 0, wait: 0 }
export 0 interface "svc" 1 { fork: 2, wait: 3 }
data 300 "svc"
data 310 "o"
func (i32, i32) -> (i64) {
block 0 (v0: i32, vout: i32) {
  vlog = i64.const 12
  vq = i64.const 0
  q1v0 = i64.const 4294967296
  q1v1 = i64.const 131072
  q1v2 = i64.const -4294967284
  q1v3 = i64.const 4294967295
  q1v4 = i64.const 0
  q1a0 = i64.const 1216
  i64.store q1a0 q1v0
  q1a1 = i64.const 1224
  i64.store q1a1 q1v1
  q1a2 = i64.const 1232
  i64.store q1a2 q1v2
  q1a3 = i64.const 1240
  i64.store q1a3 q1v3
  q1a4 = i64.const 1248
  i64.store q1a4 q1v4
  q1a5 = i64.const 1256
  i64.store q1a5 q1v4
  q1a6 = i64.const 1264
  i64.store q1a6 q1v4
  vs = cap.call 6 17 (i64) -> (i32) v0 (q1a0)
  vz0 = i64.const 0
  vcap = cap.call 6 14 (i32, i64) -> (i32) v0 (vs, vz0)
  va0 = i64.const 256
  vnp = i32.const 300
  i32.store va0 vnp
  va1 = i64.const 260
  vnl = i32.const 3
  i32.store va1 vnl
  va2 = i64.const 264
  i32.store va2 vcap
  va3 = i64.const 272
  vnp2 = i32.const 310
  i32.store va3 vnp2
  va4 = i64.const 276
  vnl2 = i32.const 1
  i32.store va4 vnl2
  va5 = i64.const 280
  i32.store va5 vout
  q2v0 = i64.const 17179869184
  q2v1 = i64.const 135168
  q2v2 = i64.const -4294967284
  q2v3 = i64.const 4294967295
  q2v4 = i64.const 0
  q2v5 = i64.const 256
  q2v6 = i64.const 2
  q2a0 = i64.const 1280
  i64.store q2a0 q2v0
  q2a1 = i64.const 1288
  i64.store q2a1 q2v1
  q2a2 = i64.const 1296
  i64.store q2a2 q2v2
  q2a3 = i64.const 1304
  i64.store q2a3 q2v3
  q2a4 = i64.const 1312
  i64.store q2a4 q2v4
  q2a5 = i64.const 1320
  i64.store q2a5 q2v5
  q2a6 = i64.const 1328
  i64.store q2a6 q2v6
  vc = cap.call 6 17 (i64) -> (i32) v0 (q2a0)
  vjc = cap.call 6 1 (i32) -> (i64) v0 (vc)
  return vjc
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  br 1()
  }
block 1 () {
  vz = i32.const 0
  vn = cap.call 4294967295 10 () -> (i64) vz ()
  br 1()
  }
}
func (i64) -> (i64) {
block 0 (vx: i64) {
  vz = i32.const 0
  vro = i64.const __RO__
  vrt = i64.const __RT__
  vt = cap.call 4294967295 11 (i64, i64) -> (i64) vz (vro, vrt)
  return vt
  }
}
func (i64) -> (i64) {
block 0 (vpid: i64) {
  vz = i32.const 0
  vt = cap.call 4294967295 12 (i64) -> (i64) vz (vpid)
  return vt
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vsvc = i64.const 6518387
  vzero = i64.const 0
  i64.store vzero vsvc
  voname = i64.const 111
  va8 = i64.const 8
  i64.store va8 voname
  vp0 = i64.const 0
  vl3 = i64.const 3
  vhsvc = cap.self.resolve vp0 vl3
  vp8 = i64.const 8
  vl1 = i64.const 1
  vho = cap.self.resolve vp8 vl1
  br 1(vhsvc, vho)
  }
block 1 (vhsvc: i32, vho: i32) {
  varg = i64.const 7
  vr = cap.call 268435456 0 (i64) -> (i64) vhsvc (varg)
  v200 = i64.const __RT__
  vistwin = i64.eq vr v200
  br_if vistwin 4(vr, vho) 2(vr, vhsvc, vho)
  }
block 2 (vr: i64, vhsvc: i32, vho: i32) {
  vpid3 = i64.const 3
  vstatus = cap.call 268435456 1 (i64) -> (i64) vhsvc (vpid3)
  veagain = i64.const -11
  viseagain = i64.eq vstatus veagain
  br_if viseagain 2(vr, vhsvc, vho) 3(vr, vstatus, vhsvc, vho)
  }
block 3 (vr: i64, vstatus: i64, vhsvc: i32, vho: i32) {
  vechild = i64.const -10
  visechild = i64.eq vstatus vechild
  br_if visechild 1(vhsvc, vho) 4(vr, vho)
  }
block 4 (vr: i64, vho: i32) {
  vp16 = i64.const 16
  i64.store vp16 vr
  vlen = i64.const 8
  vw = cap.call 0 1 (i64, i64) -> (i64) vho (vp16, vlen)
  return vr
  }
}
"#;

/// Two **distinct** reply values, kept clear of the reap sentinels (`-EAGAIN` = -11, `-ECHILD` = -10)
/// the caller's retry/detection logic keys on — otherwise the *guest's* control flow (not the fork
/// substrate) would diverge, which is not what this differential is testing. The twin is detected by
/// `reply == __RT__`, so `__RO__ != __RT__` is required.
fn params(data: &[u8]) -> (i64, i64) {
    let rd = |o: usize| {
        let mut b = [0u8; 8];
        for (i, slot) in b.iter_mut().enumerate() {
            *slot = data.get(o + i).copied().unwrap_or(0);
        }
        i64::from_le_bytes(b)
    };
    let ro = rd(0);
    let mut rt = rd(8);
    while rt == ro || rt == -10 || rt == -11 {
        rt = rt.wrapping_add(1);
    }
    (ro, rt)
}

fn build(ro: i64, rt: i64) -> Option<Arc<svm_ir::Module>> {
    let src = TEMPLATE
        .replace("__RO__", &ro.to_string())
        .replace("__RT__", &rt.to_string());
    let m = svm_text::parse_module(&src).ok()?;
    svm_verify::verify_module(&m).ok()?;
    Some(Arc::new(m))
}

fn sorted_i64(bytes: &[u8]) -> Vec<i64> {
    let mut v: Vec<i64> = bytes
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    v.sort_unstable();
    v
}

/// A run's outcome (or `None` when the engine folds/declines the module).
type RunOut = Option<Result<Vec<Value>, Trap>>;
/// A run's observable: its result plus the sorted shared-stdout multiset.
type Observed = Option<(Result<Vec<Value>, Trap>, Vec<i64>)>;

fn results_eq(a: &Result<Vec<Value>, Trap>, b: &Result<Vec<Value>, Trap>) -> bool {
    match (a, b) {
        (Ok(x), Ok(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| p == q),
        (Err(x), Err(y)) => x == y,
        _ => false,
    }
}

fn observe(
    m: &Arc<svm_ir::Module>,
    run: impl FnOnce(&Arc<svm_ir::Module>, &[Value], &mut Host) -> RunOut,
) -> Observed {
    let mut host = Host::new();
    host.set_self_module(m);
    let inst = host.grant_instantiator(0, 1u64 << 18);
    let sink = host.shared_stdout();
    let out = host.grant_stream(StreamRole::Out);
    let r = run(m, &[Value::I32(inst), Value::I32(out)], &mut host)?;
    let bytes = sink.lock().unwrap_or_else(|e| e.into_inner()).clone();
    Some((r, sorted_i64(&bytes)))
}

/// One differential iteration: build the fork topology for the fuzz-chosen replies, run it on the
/// oracle and on the bytecode engine, and assert they agree (and match the expected observable).
pub fn fuzz_one(data: &[u8]) {
    let (ro, rt) = params(data);
    let Some(m) = build(ro, rt) else { return };

    let oracle = observe(&m, |m, args, host| {
        let mut fuel = 40_000_000u64;
        Some(run_with_host(m, 0, args, &mut fuel, host))
    });
    let Some((o_res, o_out)) = oracle else { return };

    // Fork modules run natively on the bytecode drive (the `bytecode_serves_fork` escape); a fold
    // (`None`) would be a routing regression, so surface it rather than skip.
    let bc = observe(&m, |m, args, host| {
        let mut fuel = 40_000_000u64;
        bytecode::compile_and_run_with_host(m, 0, args, &mut fuel, host)
    });
    let Some((b_res, b_out)) = bc else {
        panic!(
            "fork module folded on the bytecode engine (expected native fork); replies {ro}/{rt}"
        );
    };

    assert!(
        results_eq(&o_res, &b_res),
        "fork result diverged (replies {ro}/{rt}): oracle {o_res:?} vs bytecode {b_res:?}"
    );
    assert_eq!(o_out, b_out, "fork stdout diverged (replies {ro}/{rt})");

    // Sanity (catches a both-engines-wrong regression): the original resumed past `fork()` with
    // `reply_orig`, and both replies reached the shared sink — return-twice.
    assert_eq!(
        o_res.as_ref().ok().and_then(|v| v.first()),
        Some(&Value::I64(ro)),
        "the original resumes with reply_orig ({ro})"
    );
    let mut want = [ro, rt];
    want.sort_unstable();
    assert_eq!(
        o_out,
        want.to_vec(),
        "both copies wrote their reply ({ro}/{rt})"
    );
}
