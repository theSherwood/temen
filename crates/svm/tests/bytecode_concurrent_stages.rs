//! STAGE1.md browser-parity slice 2 — **concurrent stages on the bytecode engine**. The
//! cooperative single-thread driver (`bytecode::compile_and_run_with_host`, the browser's wasm-safe
//! entry: no OS threads, no wall clock) now drives the full concurrent ring pipeline that
//! `svm-interp/tests/concurrent_stages.rs` pins on the tree-walk oracle — byte-identical module,
//! same 410 result.
//!
//! The one missing lowering was **Instantiator op 11 (`instantiate_named`)**: `instantiate` (op 0,
//! a same-module confined child) plus a by-name grant list re-granted into the child's powerbox —
//! the same-module counterpart of op 13 (`instantiate_module_named`, slice 1). Everything else was
//! already in place: the SharedRegion `map`/`page_size` and AddressSpace `create_region` ops ride
//! the generic `cap.call` dispatch on the bytecode engine (they service from `(Host, Mem)` alone),
//! and the cooperative `drive` scheduler already parks a task on `memory.wait` and wakes it on
//! `notify` — so two op-11 children moving four items through a **one-slot bounded ring** (flag at
//! region byte 0, datum at byte 8) interleave correctly. This is the shape sequential spawn/wait
//! cannot run at all: with a 1-slot ring and 4 items the producer MUST park mid-stream and be woken
//! by the consumer (and vice versa) — run-to-completion order deadlocks.
//!
//! It also pins the backing-identity futex key closed on the bytecode engine: each child maps the
//! region in its OWN window (its own address space, its own per-window region id), so wait/notify
//! only rendezvous if the key is the backing identity. A regression surfaces loudly, not as a hang:
//! waits carry a 5 s timeout, each child folds its TIMED_OUT count into its result ×1000, and a
//! child that times out more than 6 times bails — so a missed wake turns 410 into a big wrong
//! number within seconds. Differential: the tree-walk oracle and the bytecode engine agree on 410,
//! and the bytecode driver must actually drive it (return `Some`, not fall back to the oracle).

use std::sync::Arc;
use svm_interp::{bytecode, run_with_host, Host, Value};

/// func 0 — the parent: mint a region (AddressSpace op 5), build one named-grant record
/// (`"ring"` → the region handle, stored at runtime), spawn producer (entry 1) and consumer
/// (entry 2) as 128 KiB carves via `instantiate_named` (op 11), join both. Composite:
/// join(producer=4)*100 + join(consumer=10) = 410.
///
/// funcs 1/2 — the stages: resolve `"ring"`, query the map granule, map the region at window
/// offset 0, then run the ring protocol. Producer publishes 1..=4 (park while full); consumer
/// sums them (park while empty) → 10. Byte-identical to `svm-interp/tests/concurrent_stages.rs`.
const PIPELINE: &str = r#"
memory 19
data 200 "ring"

func (i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32) {
  vlen = i64.const 65536
  vrh64 = cap.call 5 5 (i64) -> (i64) v1 (vlen)
  vrh = i32.wrap_i64 vrh64
  va1 = i64.const 256
  vv1 = i32.const 200
  i32.store va1 vv1
  va2 = i64.const 260
  vv2 = i32.const 4
  i32.store va2 vv2
  va3 = i64.const 264
  i32.store va3 vrh
  vgp = i64.const 256
  vgn = i64.const 1
  vlog = i64.const 17
  vq = i64.const 0
  ; spawn via record (op 17): entry=1 off=131072 sl=17 quota=0
  q0v0 = i64.const 4294967296
  q0v1 = i64.const 131072
  q0v2 = i64.const -4294967279
  q0v3 = i64.const 4294967295
  q0v4 = i64.const 0
  q0v5 = i64.const 256
  q0v6 = i64.const 1
  q0a0 = i64.const 1152
  i64.store q0a0 q0v0
  q0a1 = i64.const 1160
  i64.store q0a1 q0v1
  q0a2 = i64.const 1168
  i64.store q0a2 q0v2
  q0a3 = i64.const 1176
  i64.store q0a3 q0v3
  q0a4 = i64.const 1184
  i64.store q0a4 q0v4
  q0a5 = i64.const 1192
  i64.store q0a5 q0v5
  q0a6 = i64.const 1200
  i64.store q0a6 q0v6
  vp = cap.call 6 17 (i64) -> (i32) v0 (q0a0)
  ; spawn via record (op 17): entry=2 off=262144 sl=17 quota=0
  q1v0 = i64.const 8589934592
  q1v1 = i64.const 262144
  q1v2 = i64.const -4294967279
  q1v3 = i64.const 4294967295
  q1v4 = i64.const 0
  q1v5 = i64.const 256
  q1v6 = i64.const 1
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
  i64.store q1a5 q1v5
  q1a6 = i64.const 1264
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
  vz = i64.const 0
  i64.store vz vnm
  vp = i64.const 0
  vl = i64.const 4
  vh = cap.self.resolve vp vl
  vg = cap.call 4 3 () -> (i64) vh ()
  vroff = i64.const 0
  vprot = i32.const 3
  vm = cap.call 4 0 (i64, i64, i64, i32) -> (i64) vh (vroff, vroff, vg, vprot)
  vone = i64.const 1
  br 1(vone, vroff)
  }
block 1 (vi: i64, vtos: i64) {
  vfour = i64.const 4
  vdone = i64.lt_s vfour vi
  br_if vdone 5(vtos) 2(vi, vtos)
  }
block 2 (vi: i64, vtos: i64) {
  vfa = i64.const 0
  vf = i32.load vfa
  br_if vf 3(vi, vtos) 4(vi, vtos)
  }
block 3 (vi: i64, vtos: i64) {
  vfa = i64.const 0
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
  vda = i64.const 8
  i64.store vda vi
  vfa = i64.const 0
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
  vz = i64.const 0
  i64.store vz vnm
  vp = i64.const 0
  vl = i64.const 4
  vh = cap.self.resolve vp vl
  vg = cap.call 4 3 () -> (i64) vh ()
  vroff = i64.const 0
  vprot = i32.const 3
  vm = cap.call 4 0 (i64, i64, i64, i32) -> (i64) vh (vroff, vroff, vg, vprot)
  vone = i64.const 1
  br 1(vone, vroff, vroff)
  }
block 1 (vn: i64, vsum: i64, vtos: i64) {
  vfour = i64.const 4
  vdone = i64.lt_s vfour vn
  br_if vdone 5(vsum, vtos) 2(vn, vsum, vtos)
  }
block 2 (vn: i64, vsum: i64, vtos: i64) {
  vfa = i64.const 0
  vf = i32.load vfa
  br_if vf 4(vn, vsum, vtos) 3(vn, vsum, vtos)
  }
block 3 (vn: i64, vsum: i64, vtos: i64) {
  vfa = i64.const 0
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
  vda = i64.const 8
  vd = i64.load vda
  vsum2 = i64.add vsum vd
  vfa = i64.const 0
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

/// Grant the parent an `Instantiator` + an `AddressSpace` over a 512 KiB window and `set_self_module`
/// (op 11 children run the parent's own program). Fuel is generous; a live run finishes in well
/// under a second (a regression trips the 5 s per-wait timeout, not a hang).
fn grants(m: &Arc<svm_ir::Module>) -> (Host, i32, i32) {
    let mut host = Host::new();
    host.set_self_module(m);
    let hi = host.grant_instantiator(0, 1u64 << 19);
    let ha = host.grant_address_space(0, 1u64 << 19);
    (host, hi, ha)
}

#[test]
fn bytecode_drives_two_concurrent_stages_through_a_shared_region_ring() {
    let m = Arc::new(svm_text::parse_module(PIPELINE).expect("parse"));
    svm_verify::verify_module(&m).expect("verify");

    // Tree-walk oracle.
    let (mut h_tw, hi_tw, ha_tw) = grants(&m);
    let mut f_tw = 50_000_000u64;
    let tw = run_with_host(
        &m,
        0,
        &[Value::I32(hi_tw), Value::I32(ha_tw)],
        &mut f_tw,
        &mut h_tw,
    );
    assert_eq!(
        tw,
        Ok(vec![Value::I64(410)]),
        "oracle: producer published 4, consumer summed 10, zero timeouts"
    );

    // Bytecode cooperative single-thread driver (the browser's wasm-safe entry). Must actually drive
    // op 11 + the ring (return `Some`, not fall back to the tree-walk oracle).
    let (mut h_bc, hi_bc, ha_bc) = grants(&m);
    let mut f_bc = 50_000_000u64;
    let bc = bytecode::compile_and_run_with_host(
        &m,
        0,
        &[Value::I32(hi_bc), Value::I32(ha_bc)],
        &mut f_bc,
        &mut h_bc,
    )
    .expect(
        "bytecode engine must drive the concurrent pipeline (op 11 + region ops), not fall back",
    );
    assert_eq!(
        bc,
        tw,
        "the bytecode engine and the tree-walk oracle agree on 410 — a 1-slot ring across two live \
         confined domains, parking and waking on the backing-identity futex key"
    );
}
