//! **#1415 — a driver declines what it cannot service; it never traps.**
//!
//! `Instantiator.instantiate_detached` (op 15) is hosted by the tree-walker and surfaced as an event
//! by the resumable engine, but three run loops mint no windows at all: the cooperative driver, the
//! OS-thread parallel driver, and the debugger path. Before #1415 those three gave *three different
//! answers* to the same unsupported op — `-EINVAL` on the cooperative driver (whose comment stated
//! the intended policy: "refuse probeably, as the JIT tiers do, **never a trap**") and
//! `Trap::Malformed` on the other two.
//!
//! `Trap::Malformed` is wrong twice over. The module is not malformed — it parsed, it verified, and
//! the op is real; the driver simply does not implement it. And a trap is **terminal for the domain**
//! (INVARIANTS #6), so a guest that would have handled `-EINVAL` on its own error path is instead
//! killed for running on the wrong driver. INVARIANTS #5 reserves traps for forgery; #9 requires a
//! backend that cannot do something to "refuse probeably or fall back — it never runs wrong".
//!
//! This pins the behavioural half on the driver where it is publicly reachable and where the change
//! is largest: the parallel driver, which used to `return (Err(Trap::Malformed), mem)` — abandoning
//! the run. The guest below proves the opposite of a trap: it reads the errno back, *keeps running*,
//! and returns a sentinel it can only reach by surviving.

use std::sync::Arc;
use temen_interp::{bytecode, Host, Region, Value};

/// The guest: `v0` Instantiator, `v1` a granted `Module`, `v2` a detached-spawn `Budget`.
///
/// Issues op 15 (the 7-arg form), then — crucially — **does not stop**. It adds `1000` to whatever
/// the spawn returned and returns that. A driver that traps never reaches the `add`, so the sentinel
/// is the survival proof; the value carries the errno so the *kind* of refusal is pinned too.
///
/// `-EINVAL` is 22, so a declining driver yields `1000 + (-22) = 978`.
const PROBE: &str = r#"memory 16
func (i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32) {
  vmh = i64.extend_i32_u v1
  vmin = i64.extend_i32_u v2
  vz = i64.const 0
  ve = i64.const 0
  vlog = i64.const 15
  vq = i64.const 0
  vh = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vmin, vmh, vz, vz, ve, vlog, vq)
  vr = i64.extend_i32_s vh
  vk = i64.const 1000
  vs = i64.add vr vk
  return vs
  }
}
"#;

/// A minimal child-entry module for the `Module` grant — never actually instantiated here (every
/// driver under test declines before it would be), but the handle must resolve for the op to reach
/// the driver seam rather than failing earlier on a bad argument.
const CHILD: &str = r#"memory 15
func (i64) -> (i64) {
block 0 (v0: i64) {
  return v0
  }
}
"#;

fn module(text: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(text).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// Grant the three caps op 15 needs and hand back the argument vector.
fn powerbox(host: &mut Host, child: &temen_ir::Module, window: u64) -> Vec<Value> {
    let inst = host.grant_instantiator(0, window);
    let modh = host.grant_module(child);
    // A generous quota: the point is that the driver declines, not that admission fails. A quota
    // miss would also land `-EINVAL`, which would make the test pass for the wrong reason.
    let budget = host.grant_budget(0, 1 << 20, 0);
    vec![Value::I32(inst), Value::I32(modh), Value::I32(budget)]
}

/// `-EINVAL` (22) folded into the guest's `+1000` sentinel — see [`PROBE`].
const DECLINED: i64 = 1000 - 22;

/// A window the parallel driver can share across its vCPUs. The `unsafe` of borrowing host memory
/// lives here in the test embedder, exactly as in `bytecode_parallel.rs` — the engine stays
/// `#![forbid(unsafe_code)]` and just takes the `Arc<Region>`.
fn shared_window(size: u64) -> (Arc<Region>, *mut u8, std::alloc::Layout) {
    let layout = std::alloc::Layout::from_size_align(size as usize, 8).unwrap();
    // SAFETY: non-zero, 8-aligned layout; freed below once every borrow of `base` is gone.
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!base.is_null(), "window allocation");
    // SAFETY: `base` owns `size` zeroed bytes and outlives every vCPU (the run joins before dealloc).
    let back = Arc::new(unsafe { Region::shared(base, size) });
    (back, base, layout)
}

/// The OS-thread parallel driver mints no windows, so op 15 lands `-EINVAL` **and the run continues**
/// to completion. Before #1415 this arm returned `Err(Trap::Malformed)` and the domain died here.
#[test]
fn the_parallel_driver_declines_an_unsupported_op_instead_of_trapping() {
    let parent = module(PROBE);
    let child = module(CHILD);
    let mut host = Host::new();
    let args = powerbox(&mut host, &child, 1 << 16);
    let (back, base, layout) = shared_window(1 << 16);
    let mut fuel = u64::MAX;

    let (result, _image) = bytecode::compile_and_run_capture_over_parallel_with_host(
        &parent,
        0,
        &args,
        &mut fuel,
        &[],
        Arc::clone(&back),
        &mut host,
    )
    .expect("the parallel driver runs this module");

    drop(back);
    // SAFETY: same layout; the region and every borrow of `base` are gone (the run joined its vCPUs).
    unsafe { std::alloc::dealloc(base, layout) };

    assert_eq!(
        result,
        Ok(vec![Value::I64(DECLINED)]),
        "op 15 must land -EINVAL in the destination register and let the guest run on; a trap here \
         (the pre-#1415 `Trap::Malformed`) kills the domain for a module that verified"
    );
}

/// The cooperative driver's answer is the one the other drivers converged **onto**, so pinning it
/// here is what makes the parallel driver's assertion a *parity* claim rather than a lone number.
/// Same guest, same powerbox, same sentinel.
#[test]
fn the_cooperative_driver_declines_the_same_op_the_same_way() {
    let parent = module(PROBE);
    let child = module(CHILD);
    let mut host = Host::new();
    let args = powerbox(&mut host, &child, 1 << 16);
    let mut fuel = u64::MAX;

    let result = bytecode::compile_and_run_with_host(&parent, 0, &args, &mut fuel, &mut host)
        .expect("the cooperative driver runs this module");

    assert_eq!(
        result,
        Ok(vec![Value::I64(DECLINED)]),
        "the cooperative driver's incumbent answer — the parallel driver now matches it"
    );
}
