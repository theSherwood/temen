//! STAGE1.md item 6 — the **JIT pipeline**: two granted children running CONCURRENTLY on their own
//! OS threads, piped through a granted `SharedRegion` + canonical-key futex — the fast-backend twin
//! of `temen-interp/tests/concurrent_stages.rs`. The parent mints a region (the run's backing factory
//! is `temen_run::new_shared_region`, so it is a real OS shared-memory object), spawns producer and
//! consumer via `instantiate_named` (op 11, each re-granted the region by name), and joins both.
//!
//! What this pins, JIT-specifically:
//! - **op-11 children are async** (S1c): each runs on its own OS thread in its own guarded window.
//!   With a 1-slot ring and 4 items, run-to-completion order deadlocks — the producer MUST park
//!   mid-stream and be woken by the consumer, so the old synchronous spawn cannot pass this at all.
//! - **real aliasing into separate child windows**: each child `map`s the region into its OWN
//!   window (`MprotectWindow::map_region` — `mmap(MAP_SHARED|MAP_FIXED)` of the region's memfd on
//!   unix, placeholder + `MapViewOfFile3` on windows), so parent-minted bytes are the same physical
//!   pages in both children.
//! - **canonical futex keys across windows**: each child's first `call.cap` installs the region-canon
//!   hook over its own `mem_base`, so `atomic.wait`/`notify` in different windows key on the backing
//!   identity `(os_fd, offset)` and rendezvous. With per-window keys every wake misses — and the
//!   regression surfaces loudly, not as a hang: waits carry a 5 s timeout, each child folds its
//!   TIMED_OUT count into its result ×1000, and a child that times out more than 6 times bails.
//!
//! (Windows sizing: the §13 map granule is the 64 KiB allocation granularity there, so child windows
//! are 128 KiB — `memory 17` carves — and the children map `len = granule` queried at run time.)

#[path = "support/grant_hooks.rs"]
mod grant_hooks_mod;
use grant_hooks_mod::grant_hooks;

use std::sync::Arc;
use temen_interp::{run_with_host, Host, Value};
use temen_jit::{compile_and_run_capture_reserved_with_host_ex, JitOutcome};
use temen_text::parse_module;
use temen_verify::verify_module;

/// Byte-identical to the interpreter test's `PIPELINE` module, so the two backends run the same
/// program: func 0 (parent) mints a 64 KiB region, spawns producer (func 1) and consumer (func 2)
/// as 128 KiB carves granting `"ring"` → region, joins both → join(producer=4)*100 +
/// join(consumer=10) = 410. The stages resolve `"ring"`, query the map granule, map the region at
/// window offset 0, and move 1..=4 through the one-slot ring (flag at region byte 0, datum at 8).
const PIPELINE: &str = r#"
memory 19
data 16584 "ring"

func (i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32) {
  vlen = i64.const 65536
  vrh64 = call.cap 5 5 (i64) -> (i64) v1 (vlen)
  vrh = i32.wrap_i64 vrh64
  va1 = i64.const 16640
  vv1 = i32.const 16584
  i32.store va1 vv1
  va2 = i64.const 16644
  vv2 = i32.const 4
  i32.store va2 vv2
  va3 = i64.const 16648
  i32.store va3 vrh
  vgp = i64.const 16640
  vgn = i64.const 1
  vlog = i64.const 17
  vq = i64.const 0
  ; spawn via record (op 17): entry=1 off=131072 sl=17 quota=0
  q0v0 = i64.const 4294967296
  q0v1 = i64.const 131072
  q0v2 = i64.const -4294967279
  q0v3 = i64.const 4294967295
  q0v4 = i64.const 0
  q0v5 = i64.const 16640
  q0v6 = i64.const 1
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
  i64.store q0a5 q0v5
  q0a6 = i64.const 17584
  i64.store q0a6 q0v6
  vp = call.cap 6 17 (i64) -> (i32) v0 (q0a0)
  ; spawn via record (op 17): entry=2 off=262144 sl=17 quota=0
  q1v0 = i64.const 8589934592
  q1v1 = i64.const 262144
  q1v2 = i64.const -4294967279
  q1v3 = i64.const 4294967295
  q1v4 = i64.const 0
  q1v5 = i64.const 16640
  q1v6 = i64.const 1
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
  i64.store q1a5 q1v5
  q1a6 = i64.const 17648
  i64.store q1a6 q1v6
  vc = call.cap 6 17 (i64) -> (i32) v0 (q1a0)
  vjp = call.cap 6 1 (i32) -> (i64) v0 (vp)
  vjc = call.cap 6 1 (i32) -> (i64) v0 (vc)
  vk = i64.const 100
  vm = i64.mul vjp vk
  vs = i64.add vm vjc
  return vs
  }
}

func (i64) -> (i64) {
block 0 (v0: i64) {
  vnm = i64.const 1735289202
  vz = i64.const 16384
  i64.store vz vnm
  vp = i64.const 16384
  vl = i64.const 4
  vh = self.resolve vp vl
  vg = call.cap 4 3 () -> (i64) vh ()
  vroff = i64.const 0
  vwoff = i64.const 16384
  vprot = i32.const 3
  vm = call.cap 4 0 (i64, i64, i64, i32) -> (i64) vh (vwoff, vroff, vg, vprot)
  vone = i64.const 1
  br 1(vone, vroff)
  }
block 1 (vi: i64, vtos: i64) {
  vfour = i64.const 4
  vdone = i64.lt_s vfour vi
  br_if vdone 5(vtos) 2(vi, vtos)
  }
block 2 (vi: i64, vtos: i64) {
  vfa = i64.const 16384
  vf = i32.load vfa
  br_if vf 3(vi, vtos) 4(vi, vtos)
  }
block 3 (vi: i64, vtos: i64) {
  vfa = i64.const 16384
  vexp = i32.const 1
  vto = i64.const 5000000000
  vst = i32.atomic.wait vfa vexp vto
  vtwo = i32.const 2
  vis = i32.eq vst vtwo
  vis64 = i64.extend_i32_u vis
  vtos2 = i64.add vtos vis64
  vsix = i64.const 6
  vbail = i64.lt_s vsix vtos2
  br_if vbail 5(vtos2) 2(vi, vtos2)
  }
block 4 (vi: i64, vtos: i64) {
  vda = i64.const 16392
  i64.store vda vi
  vfa = i64.const 16384
  vfull = i32.const 1
  i32.store vfa vfull
  vcnt = i32.const 1
  vw = atomic.notify vfa vcnt
  vone = i64.const 1
  vni = i64.add vi vone
  br 1(vni, vtos)
  }
block 5 (vtos: i64) {
  vk = i64.const 1000
  vm = i64.mul vtos vk
  vfour = i64.const 4
  vr = i64.add vm vfour
  return vr
  }
}

func (i64) -> (i64) {
block 0 (v0: i64) {
  vnm = i64.const 1735289202
  vz = i64.const 16384
  i64.store vz vnm
  vp = i64.const 16384
  vl = i64.const 4
  vh = self.resolve vp vl
  vg = call.cap 4 3 () -> (i64) vh ()
  vroff = i64.const 0
  vwoff = i64.const 16384
  vprot = i32.const 3
  vm = call.cap 4 0 (i64, i64, i64, i32) -> (i64) vh (vwoff, vroff, vg, vprot)
  vone = i64.const 1
  br 1(vone, vroff, vroff)
  }
block 1 (vn: i64, vsum: i64, vtos: i64) {
  vfour = i64.const 4
  vdone = i64.lt_s vfour vn
  br_if vdone 5(vsum, vtos) 2(vn, vsum, vtos)
  }
block 2 (vn: i64, vsum: i64, vtos: i64) {
  vfa = i64.const 16384
  vf = i32.load vfa
  br_if vf 4(vn, vsum, vtos) 3(vn, vsum, vtos)
  }
block 3 (vn: i64, vsum: i64, vtos: i64) {
  vfa = i64.const 16384
  vexp = i32.const 0
  vto = i64.const 5000000000
  vst = i32.atomic.wait vfa vexp vto
  vtwo = i32.const 2
  vis = i32.eq vst vtwo
  vis64 = i64.extend_i32_u vis
  vtos2 = i64.add vtos vis64
  vsix = i64.const 6
  vbail = i64.lt_s vsix vtos2
  br_if vbail 5(vsum, vtos2) 2(vn, vsum, vtos2)
  }
block 4 (vn: i64, vsum: i64, vtos: i64) {
  vda = i64.const 16392
  vd = i64.load vda
  vsum2 = i64.add vsum vd
  vfa = i64.const 16384
  vempty = i32.const 0
  i32.store vfa vempty
  vcnt = i32.const 1
  vw = atomic.notify vfa vcnt
  vone = i64.const 1
  vnn = i64.add vn vone
  br 1(vnn, vsum2, vtos)
  }
block 5 (vsum: i64, vtos: i64) {
  vk = i64.const 1000
  vm = i64.mul vtos vk
  vr = i64.add vm vsum
  return vr
  }
}
"#;

/// The interpreter reference: the same source, same 410.
fn run_interp() -> Vec<Value> {
    let m = parse_module(PIPELINE).expect("parse");
    verify_module(&m).expect("verify");
    let m = Arc::new(m);
    let mut host = Host::new();
    host.set_self_module(&m);
    let hi = host.grant_instantiator(0, 1u64 << 19);
    let ha = host.grant_address_space(0, 1u64 << 19);
    let mut fuel = 50_000_000u64;
    run_with_host(
        &m,
        0,
        &[Value::I32(hi), Value::I32(ha)],
        &mut fuel,
        &mut host,
    )
    .expect("interp: no trap, no hang")
}

#[test]
fn two_concurrent_jit_children_pipe_through_a_shared_region_ring() {
    // Nesting requires the child runner (`fiber_rt`); where unsupported the JIT declines child
    // spawns, so there is nothing to pin — the interpreter remains the only backend there.
    if !temen_jit::fiber_supported() {
        return;
    }
    let ir = run_interp();
    assert_eq!(ir, vec![Value::I64(410)], "interp reference");

    let m = parse_module(PIPELINE).expect("parse");
    verify_module(&m).expect("verify");
    let mut host = Host::new();
    // Regions minted by this run are real OS shared-memory objects (memfd / section), so the JIT
    // children can `map` them for hardware aliasing.
    host.set_region_factory(temen_run::new_shared_region);
    let hi = host.grant_instantiator(0, 1u64 << 19);
    let ha = host.grant_address_space(0, 1u64 << 19);
    let (jo, _jmem) = compile_and_run_capture_reserved_with_host_ex(
        &m,
        0,
        &[hi as i64, ha as i64],
        &vec![0u8; 1 << 19],
        0,
        temen_run::cap_thunk,
        &mut host as *mut Host as *mut core::ffi::c_void,
        None,
        Some(grant_hooks()),
    )
    .expect("jit");
    assert!(
        matches!(jo, JitOutcome::Returned(ref s) if s == &[410]),
        "jit: producer published 4 (park while full), consumer summed 10 (park while empty), \
         zero timeouts — a 1-slot ring across two child OS threads, aliased into both windows; \
         got {jo:?}"
    );
}
