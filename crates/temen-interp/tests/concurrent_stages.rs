//! STAGE1.md item 6 / PROCESS.md §4 — **concurrent stages**: two children running
//! CONCURRENTLY, piped through a granted `SharedRegion` + canonical-key futex. The parent
//! creates a region, spawns a producer and a consumer (named-grant spawns, each re-granted
//! the region), and joins both; the children map the region into their own windows and move
//! four items through a **one-slot bounded ring** (flag at region byte 0, datum at byte 8),
//! parking on the flag and waking each other by `notify`.
//!
//! This is the shape sequential spawn/wait cannot run at all: with a 1-slot ring and 4
//! items, the producer MUST park mid-stream and be woken by the consumer (and vice versa) —
//! run-to-completion order deadlocks. It also pins the S1c futex-key residue closed: each
//! child maps the region in its OWN window (its own address space, its own per-window region
//! id), so wait/notify only rendezvous if the key is the **backing identity** — with
//! per-domain keys every wake misses. A regression surfaces loudly, not as a hang: waits
//! carry a 5 s timeout, each child folds its TIMED_OUT count into its result ×1000, and a
//! child that times out more than 6 times **bails** with that poisoned composite — so missed
//! wakes turn 410 into a big wrong number within seconds. (Windows sizing note: the §13 map
//! granule is the 64 KiB allocation granularity there, so child windows are 128 KiB —
//! `memory 17` — and the region is 64 KiB; the children map `len = granule` they query at
//! run time, portable across the 4 KiB/16 KiB/64 KiB granule platforms.)

use std::sync::Arc;
use temen_interp::{run_with_host, Host, Value};

/// func 0 — the parent: mint a region (AddressSpace op 5), build one named-grant record
/// (`"ring"` → the region handle, stored at runtime), spawn producer (entry 1) and consumer
/// (entry 2) as 128 KiB carves, join both. Composite: join(producer=4)*100 +
/// join(consumer=10) = 410.
///
/// funcs 1/2 — the stages: resolve `"ring"`, query the map granule, map the region at window
/// offset 0, then run the ring protocol. Producer publishes 1..=4 (park while full);
/// consumer sums them (park while empty) → 10.
const PIPELINE: &str = r#"
memory 19
data 16584 "ring"

func (i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32) {
  vlen = i64.const 65536
  vrh64 = cap.call 5 5 (i64) -> (i64) v1 (vlen)
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
  vp = cap.call 6 17 (i64) -> (i32) v0 (q0a0)
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
  vc = cap.call 6 17 (i64) -> (i32) v0 (q1a0)
  vjp = cap.call 6 1 (i32) -> (i64) v0 (vp)
  vjc = cap.call 6 1 (i32) -> (i64) v0 (vc)
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
  vh = cap.self.resolve vp vl
  vg = cap.call 4 3 () -> (i64) vh ()
  vroff = i64.const 0
  vwoff = i64.const 16384
  vprot = i32.const 3
  vm = cap.call 4 0 (i64, i64, i64, i32) -> (i64) vh (vwoff, vroff, vg, vprot)
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
  vh = cap.self.resolve vp vl
  vg = cap.call 4 3 () -> (i64) vh ()
  vroff = i64.const 0
  vwoff = i64.const 16384
  vprot = i32.const 3
  vm = cap.call 4 0 (i64, i64, i64, i32) -> (i64) vh (vwoff, vroff, vg, vprot)
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

/// The **detached** variant — the §5 model sentence as a test ("a shell would plausibly run
/// coreutils detached"): the same one-slot ring, but the two stages are spawned through a
/// `WindowMinter` (op 15) from their own module into windows the parent cannot read. The
/// region grant rides the same op-11-format named-grant records; the module's own data
/// segment carries the `"ring"` name into each private window. Private memory and an
/// explicit shared channel compose — exactly the separate-process discipline, in-process.
const DETACHED_STAGES: &str = r#"
memory 17
data 16384 "ring"

func (i64) -> (i64) {
block 0 (v0: i64) {
  vp = i64.const 16384
  vl = i64.const 4
  vh = cap.self.resolve vp vl
  vg = cap.call 4 3 () -> (i64) vh ()
  vroff = i64.const 0
  vwoff = i64.const 16384
  vprot = i32.const 3
  vm = cap.call 4 0 (i64, i64, i64, i32) -> (i64) vh (vwoff, vroff, vg, vprot)
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
  vp = i64.const 16384
  vl = i64.const 4
  vh = cap.self.resolve vp vl
  vg = cap.call 4 3 () -> (i64) vh ()
  vroff = i64.const 0
  vwoff = i64.const 16384
  vprot = i32.const 3
  vm = cap.call 4 0 (i64, i64, i64, i32) -> (i64) vh (vwoff, vroff, vg, vprot)
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

/// The detached-pipeline parent: mint the region, build the grant record, spawn producer
/// (entry 0) and consumer (entry 1) DETACHED from the granted module, join both → 410.
const DETACHED_PIPELINE_PARENT: &str = r#"
memory 17
data 16584 "ring"

func (i32, i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32, v3: i32) {
  vlen = i64.const 65536
  vrh64 = cap.call 5 5 (i64) -> (i64) v1 (vlen)
  vrh = i32.wrap_i64 vrh64
  va1 = i64.const 16640
  vv1 = i32.const 16584
  i32.store va1 vv1
  va2 = i64.const 16644
  vv2 = i32.const 4
  i32.store va2 vv2
  va3 = i64.const 16648
  i32.store va3 vrh
  vmh = i64.extend_i32_u v2
  vmin = i64.extend_i32_u v3
  vgp = i64.const 16640
  vgn = i64.const 1
  ve0 = i64.const 0
  ve1 = i64.const 1
  vlog = i64.const 17
  vq = i64.const 0
  vp = cap.call 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vmin, vmh, vgp, vgn, ve0, vlog, vq)
  vc = cap.call 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vmin, vmh, vgp, vgn, ve1, vlog, vq)
  vjp = cap.call 6 1 (i32) -> (i64) v0 (vp)
  vjc = cap.call 6 1 (i32) -> (i64) v0 (vc)
  vk = i64.const 100
  vm = i64.mul vjp vk
  vs = i64.add vm vjc
  return vs
  }
}
"#;

#[test]
fn two_detached_stages_pipe_through_a_shared_region_ring() {
    let a = temen_text::parse_module(DETACHED_PIPELINE_PARENT).expect("parse parent");
    temen_verify::verify_module(&a).expect("verify parent");
    let b = temen_text::parse_module(DETACHED_STAGES).expect("parse stages");
    temen_verify::verify_module(&b).expect("verify stages");
    let mut host = Host::new();
    let hi = host.grant_instantiator(0, 1u64 << 17);
    let ha = host.grant_address_space(0, 1u64 << 17);
    let hm = host.grant_module(&b);
    let hw = host.grant_window_minter(2 << 17); // exactly two 2^17 windows
    let mut fuel = 50_000_000u64;
    let r = run_with_host(
        &a,
        0,
        &[
            Value::I32(hi),
            Value::I32(ha),
            Value::I32(hm),
            Value::I32(hw),
        ],
        &mut fuel,
        &mut host,
    )
    .expect("no trap, no hang");
    assert_eq!(
        r,
        vec![Value::I64(410)],
        "the same ring, between two windows nobody's parent can read — private memory \
         and an explicit shared channel compose"
    );
}

/// The bytecode entry serves the pipeline identically via the standing oracle fallback
/// (Instantiator ops decline to compile → whole-module tree-walk).
#[test]
fn the_bytecode_entry_runs_the_pipeline_identically() {
    let m = temen_text::parse_module(PIPELINE).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let m = Arc::new(m);
    let mut host = Host::new();
    host.set_self_module(&m);
    let hi = host.grant_instantiator(0, 1u64 << 19);
    let ha = host.grant_address_space(0, 1u64 << 19);
    let mut fuel = 50_000_000u64;
    let r = temen_interp::run_with_host_fast(
        &m,
        0,
        &[Value::I32(hi), Value::I32(ha)],
        &mut fuel,
        &mut host,
    )
    .expect("no trap, no hang");
    assert_eq!(r, vec![Value::I64(410)], "identical to the tree-walk run");
}

#[test]
fn two_concurrent_children_pipe_through_a_shared_region_ring() {
    let m = temen_text::parse_module(PIPELINE).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let m = Arc::new(m);
    let mut host = Host::new();
    host.set_self_module(&m);
    let hi = host.grant_instantiator(0, 1u64 << 19);
    let ha = host.grant_address_space(0, 1u64 << 19);
    let mut fuel = 50_000_000u64;
    let r = run_with_host(
        &m,
        0,
        &[Value::I32(hi), Value::I32(ha)],
        &mut fuel,
        &mut host,
    )
    .expect("no trap, no hang");
    assert_eq!(
        r,
        vec![Value::I64(410)],
        "producer published 4 (park while full), consumer summed 10 (park while empty), \
         zero timeouts — a 1-slot ring across two live domains"
    );
}
