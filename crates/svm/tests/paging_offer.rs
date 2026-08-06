//! CONSOLIDATION.md §2.2 — **demand paging as a call on the pager provider**: `Instantiator`
//! op 16 (`spawn_process_demand`) spawns an op-11-shaped process child whose window starts
//! demand-paged and whose recoverable faults become `page(addr)` calls on the **spawner's own
//! impl export** (by index) — served by its live serve loop. The parent parks at
//! `svc.wait` as an ordinary provider; with direct handoff (`RunConfig::handoff`, on by
//! default) the fault's service runs inline on the child's thread — the §10.2 shape. The
//! substrate holds the pager binding on the child vCPU: the child can neither name nor call it
//! (userfaultfd-style; the child's authority is unchanged by being demand-paged).
//!
//! This is the offer-transport twin of the bespoke `spawn_demand_coroutine` shape pinned by
//! `coroutine.rs`/`bytecode_demand_coroutine.rs`, with the driving inverted: the child runs
//! concurrently and *calls* (via the fault seam); no resume loop exists. `paging_bench` carries
//! the latency comparison; this file pins the semantics.

use svm_run::{instantiate_with_imports, Backend, HostCap, Imports, Outcome, RunConfig};
use svm_text::parse_module;

/// One pager cycle: parent reifies its pager export, spawns a demand child over a carve of
/// `strides` 64 KiB strides, serves exactly `strides` faults from `svc.wait`, joins, and exits
/// with `child_sum + serves * 1000`. The handler stores `123` at the fault address (parent-window
/// coordinates — the same coordinates the bespoke `resume` hands the pager, pinned by
/// `demand_coroutine_reports_fault_address`).
fn pager_program(mem_log2: u8, carve_off: u64, carve_log2: u64, strides: u64) -> String {
    format!(
        "\
memory {mem_log2}
data 0 \"vm\"
type 0 func (i64) -> (i64)
type 1 interface {{ page: 0 }}
export 0 interface \"pager\" 1 {{ page: 2 }}
import 0 \"exit\" (i32) -> ()

func 0 () -> () {{
block 0 () {{
  vp = i64.const 0
  vl = i64.const 2
  vh = cap.self.resolve vp vl
  ; spawn a demand child via record (op 17): pager = impl export 0 (f16 hi), entry 1
  rrv0 = i64.const 4294967296
  rrv1 = i64.const {carve_off}
  rrv2 = i64.const {carve_log2}
  rrv3 = i64.const 4294967295
  rrvz = i64.const 0
  rra0 = i64.const 1152
  i64.store rra0 rrv0
  rra1 = i64.const 1160
  i64.store rra1 rrv1
  rra2 = i64.const 1168
  i64.store rra2 rrv2
  rra3 = i64.const 1176
  i64.store rra3 rrv3
  rra4 = i64.const 1184
  i64.store rra4 rrvz
  rra5 = i64.const 1192
  i64.store rra5 rrvz
  rra6 = i64.const 1200
  i64.store rra6 rrvz
  vch = cap.call 6 17 (i64) -> (i32) vh (rra0)
  vs0 = i64.const 0
  br 1(vh, vch, vs0)
}}
block 1 (vh1: i32, vch1: i32, vsrv: i64) {{
  vz = i32.const 0
  vs = svc.wait vz
  vsrv2 = i64.add vsrv vs
  vn = i64.const {strides}
  vcmp = i64.lt_s vsrv2 vn
  br_if vcmp 1(vh1, vch1, vsrv2) 2(vh1, vch1, vsrv2)
}}
block 2 (vh2: i32, vch2: i32, vsrvf: i64) {{
  vj = cap.call 6 1 (i32) -> (i64) vh2 (vch2)
  vk = i64.const 1000
  vm = i64.mul vsrvf vk
  vt = i64.add vj vm
  vc = i32.wrap_i64 vt
  call.import 0 (vc)
  unreachable
  }}
}}

func 1 (i64) -> (i64) {{
block 0 (v0: i64) {{
  vi0 = i64.const 0
  va0 = i64.const 0
  br 1(vi0, va0)
}}
block 1 (vi: i64, va: i64) {{
  vstride = i64.const 65536
  vaddr = i64.mul vi vstride
  vb = i32.load8_u vaddr
  vbw = i64.extend_i32_u vb
  va2 = i64.add va vbw
  vone = i64.const 1
  vi2 = i64.add vi vone
  vpn = i64.const {strides}
  vcmp = i64.lt_s vi2 vpn
  br_if vcmp 1(vi2, va2) 2(va2)
}}
block 2 (vaf: i64) {{
  return vaf
  }}
}}

func 2 (i64) -> (i64) {{
block 0 (vaddr: i64) {{
  vb = i32.const 123
  i32.store8 vaddr vb
  vzero = i64.const 0
  return vzero
  }}
}}
"
    )
}

fn run(backend: Backend, src: &str) -> i32 {
    let m = parse_module(src).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    let registry = Imports::new().provide("exit", HostCap::exit());
    let inst = instantiate_with_imports(m, registry).expect("instantiate");
    let r = inst
        .run_with_caps(
            backend,
            &RunConfig::default(),
            &[(
                "vm",
                HostCap::custom(6, 0, |h, win| h.grant_instantiator(0, win)),
            )],
        )
        .unwrap_or_else(|e| panic!("{backend:?}: {e}"));
    match r.outcome {
        Outcome::Exited(code) => code,
        other => panic!("{backend:?}: unexpected outcome {other:?}"),
    }
}

const BACKENDS: [Backend; 3] = [Backend::TreeWalk, Backend::Bytecode, Backend::Jit];

/// The single-fault vertical: spawn, one first-touch fault, the pager supplies `123`, the child's
/// rewound load reads it, join returns it. Exit = 123 + 1·1000.
#[test]
fn demand_child_fault_is_served_by_the_pager_provider() {
    let src = pager_program(17, 65536, 16, 1);
    for b in BACKENDS {
        assert_eq!(run(b, &src), 1123, "{b:?}: one fault, one serve, one page");
    }
}

/// Two 64 KiB strides ⇒ exactly two faults on any host page size ≤ 64 KiB: the serve loop, the
/// per-page supply (one `supply_page` maps only the faulted stride's host page), and the join
/// ordering (serve all faults *before* joining) all hold across a multi-fault run.
/// Exit = 2·123 + 2·1000.
#[test]
fn demand_child_faults_each_stride_once() {
    let src = pager_program(18, 131072, 17, 2);
    for b in BACKENDS {
        assert_eq!(run(b, &src), 2246, "{b:?}: two faults, two serves");
    }
}
