//! The detached ring pipeline: two stages spawned detached through op 15, each into a window the
//! parent cannot read, moving four items through a one-slot ring in a `SharedRegion` the parent
//! grants them by name. One program for every engine: temen-interp's `concurrent_stages` runs it on
//! the interpreters, temen's `bytecode_concurrent_stages` on the cooperative bytecode driver and
//! `jit_concurrent_stages` on the Cranelift JIT. Include with
//! `#[path = "support/pipeline.rs"] mod pipeline;`.
//!
//! With a 1-slot ring and 4 items, run-to-completion order deadlocks: the producer must park
//! mid-stream and be woken by the consumer. A missed wake shows up as a wrong number, not a hang:
//! waits carry a 5 s timeout, each stage folds its timeout count into its result ×1000, and a stage
//! that times out more than 6 times bails.
#![allow(dead_code)] // each test binary uses a different subset

use temen_interp::Host;
use temen_ir::Module;

/// The **detached** variant — the §5 model sentence as a test ("a shell would plausibly run
/// coreutils detached"): the same one-slot ring, but the two stages are spawned through a
/// detached-spawn `Budget` (op 15) from their own module into windows the parent cannot read. The
/// region grant rides the same op-11-format named-grant records; the module's own data
/// segment carries the `"ring"` name into each private window. Private memory and an
/// explicit shared channel compose — exactly the separate-process discipline, in-process.
pub const STAGES: &str = r#"
memory 17
data 16384 "ring"

func (i64) -> (i64) {
block 0 (v0: i64) {
  vp = i64.const 16384
  vl = i64.const 4
  vh = self.resolve vp vl
  vg = call.cap 4 3 () -> (i64) vh ()
  vroff = i64.const 0
  vwoff = i64.const 65536
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
  vfa = i64.const 65536
  vf = i32.load vfa
  br_if vf 3(vi, vtos) 4(vi, vtos)
  }
block 3 (vi: i64, vtos: i64) {
  vfa = i64.const 65536
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
  vda = i64.const 65544
  i64.store vda vi
  vfa = i64.const 65536
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
  vh = self.resolve vp vl
  vg = call.cap 4 3 () -> (i64) vh ()
  vroff = i64.const 0
  vwoff = i64.const 65536
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
  vfa = i64.const 65536
  vf = i32.load vfa
  br_if vf 4(vn, vsum, vtos) 3(vn, vsum, vtos)
  }
block 3 (vn: i64, vsum: i64, vtos: i64) {
  vfa = i64.const 65536
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
  vda = i64.const 65544
  vd = i64.load vda
  vsum2 = i64.add vsum vd
  vfa = i64.const 65536
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
pub const PARENT: &str = r#"
memory 17
data 16584 "ring"

func (i32, i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32, v3: i32) {
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
  vmh = i64.extend_i32_u v2
  vmin = i64.extend_i32_u v3
  vgp = i64.const 16640
  vgn = i64.const 1
  ve0 = i64.const 0
  ve1 = i64.const 1
  vlog = i64.const 17
  vq = i64.const 0
  vp = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vmin, vmh, vgp, vgn, ve0, vlog, vq)
  vc = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vmin, vmh, vgp, vgn, ve1, vlog, vq)
  vjp = call.cap 6 1 (i32) -> (i64) v0 (vp)
  vjc = call.cap 6 1 (i32) -> (i64) v0 (vc)
  vk = i64.const 100
  vm = i64.mul vjp vk
  vs = i64.add vm vjc
  return vs
  }
}
"#;

/// The parent and the stages' module, each verified.
pub fn modules() -> (Module, Module) {
    let load = |src: &str| {
        let m = temen_text::parse_module(src).expect("parse");
        temen_verify::verify_module(&m).expect("verify");
        m
    };
    (load(PARENT), load(STAGES))
}

/// The parent's four arguments in a fresh host: an `Instantiator` and an `AddressSpace` over its
/// window, the stages' `Module`, and a `Budget` of exactly the two stages' windows.
pub fn host(stages: &Module) -> (Host, [i32; 4]) {
    let mut host = Host::new();
    let hi = host.grant_instantiator(0, 1u64 << 17);
    let ha = host.grant_address_space(0, 1u64 << 17);
    let hm = host.grant_module(stages);
    let hw = host.grant_budget(-1, (2 << 17) as i64, -1);
    (host, [hi, ha, hm, hw])
}
