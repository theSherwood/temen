//! CONSOLIDATION.md §3d — **the config-record spawn (op 17) is native on the bytecode tier**.
//! Every test here runs the same program on the tree-walk oracle (`run_with_host`) and the
//! bytecode engine (`compile_and_run_with_host`), asserting identical results — and the
//! `.expect(...)` on the engine's `Option` is the non-vacuity gate: if the engine *declined* the
//! module (fell back), the test fails. That is what distinguishes these pins from the
//! `temen`-crate record differentials, which run through temen-run's `Backend::Bytecode`, where a
//! module the engine declines runs on the tree-walker without a word.
//!
//! Tier boundaries pinned here:
//! - a **pager-capable** module (op 17 + a pager-shaped export) runs on a driver that serves its
//!   children, which pages them (#2164), and declines to the oracle on one that does not
//!   (`compile_module_for`'s veto; temen-run's `module_demand_spawns` folds it off the Cranelift
//!   tier, which has no paging);
//! - a **budget** funds the child natively, and a dangling one refuses the spawn probeably. What a
//!   budget bounds is pinned on every driver by `spawn_every_driver.rs`.

#[path = "support/rec.rs"]
mod rec;

use temen_interp::{bytecode, run_with_host, Host, Trap, Value};
use temen_ir::SpawnRec;
use temen_text::parse_module;

use temen_ir::errno::EINVAL;

/// A record spawn (the spawn record [`rec_with_impl_export`] appends at 17536) plus a **pager-shaped
/// export** (a pager-capable module). The record's pager field is runtime data, so admission is by
/// the module's shape (`compile_module_for`).
const REC_WITH_IMPL_EXPORT: &str = r#"memory 17
type 0 func (i64) -> (i64)
type 1 interface { page: 0 }
export 0 interface "pager" 1 { page: 1 }
func (i32) -> (i64) {
block 0 (v0: i32) {
  qa0 = i64.const 17536
  vch = call.cap 6 17 (i64) -> (i32) v0 (qa0)
  vr_st, vr = call.cap 6 1 (i32) -> (i64, i64) v0 (vch)
  return vr
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vr = i64.const 42
  return vr
  }
}
"#;

/// [`REC_WITH_IMPL_EXPORT`] with its record. Admission is by the module's shape; the record is
/// never read here.
fn rec_with_impl_export() -> String {
    format!(
        "{REC_WITH_IMPL_EXPORT}{}",
        rec::segment(17536, &SpawnRec::default())
    )
}

/// A detached-record parent (`(i32 inst, i64 budget, i32 child) -> (i64)`): the budget and module
/// handles arrive at run time and are stored into the spawn record appended at 17536 (`child`, func 1's
/// child image, #2219, in its declared window). Spawn, join, return the child's 42 — or the spawn's
/// refusal.
fn budget_rec_src() -> String {
    format!(
        "memory 17
func (i32, i64, i32) -> (i64) {{
block 0 (v0: i32, v1: i64, v2: i32) {{
  qma = i64.const 17544
  i32.store qma v2
  qb = i32.wrap_i64 v1
  qba = i64.const 17548
  i32.store qba qb
  qa0 = i64.const 17536
  vch = call.cap 6 17 (i64) -> (i32) v0 (qa0)
  vz = i32.const 0
  vneg = i32.lt_s vch vz
  br_if vneg 1(vch) 2(v0, vch)
}}
block 1 (ve: i32) {{
  vr = i64.extend_i32_s ve
  return vr
}}
block 2 (vi2: i32, vch2: i32) {{
  vr_st, vr = call.cap 6 1 (i32) -> (i64, i64) vi2 (vch2)
  return vr
  }}
}}
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  vr = i64.const 42
  return vr
  }}
}}
{rec}",
        rec = rec::segment(17536, &SpawnRec::default())
    )
}

fn tw(src: &str, args: &[Value], host: &mut Host) -> Result<Vec<Value>, Trap> {
    let m = parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let mut fuel = 10_000_000u64;
    run_with_host(&m, 0, args, &mut fuel, host)
}

fn bc(src: &str, args: &[Value], host: &mut Host) -> Result<Vec<Value>, Trap> {
    let m = parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let mut fuel = 10_000_000u64;
    bytecode::compile_and_run_with_host(&m, 0, args, &mut fuel, host)
        .expect("bytecode engine must drive the record spawn natively (§3d)")
}

/// The vertical: a spawn record spawn + join is engine-native (compile succeeds — no oracle fallback)
/// and result-identical to the tree-walker, the child funded by the budget.
#[test]
fn record_spawn_is_native_and_matches_the_oracle() {
    let src = budget_rec_src();
    let m = parse_module(&src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    assert!(
        bytecode::VcpuProgram::compile(&m).is_some(),
        "op-17 module must compile natively (no impl exports)"
    );
    let image = temen_ir::child_image_at(&m, 1).expect("image");
    let powerbox = || {
        let mut h = Host::new();
        let i = h.grant_instantiator();
        let c = h.grant_module(&image);
        let b = h.grant_budget(5_000_000, -1, -1);
        (h, [Value::I32(i), Value::I64(b as i64), Value::I32(c)])
    };
    let (mut h, args) = powerbox();
    let r_tw = tw(&src, &args, &mut h);
    assert_eq!(r_tw, Ok(vec![Value::I64(42)]), "oracle");
    let (mut h, args) = powerbox();
    let r_bc = bc(&src, &args, &mut h);
    assert_eq!(r_bc, r_tw, "record spawn: bytecode != tree-walker");
}

/// #2164 — the pager admission: a driver that serves its children pages them, so it admits a
/// pager-capable module. `Vcpu` and the debug scheduler serve none, so they decline the module whole
/// to the oracle: `Vcpu` at its compile, the debug scheduler when a session opens. The same module
/// without its `pager` export opens a session. What the paging does is pinned on every driver by
/// `temen`'s `instantiate_record.rs`.
#[test]
fn a_pager_capable_record_module_runs_where_children_are_served() {
    let src = rec_with_impl_export();
    let m = parse_module(&src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    assert!(
        bytecode::admits(&m),
        "the cooperative driver pages its children itself"
    );
    assert!(
        bytecode::VcpuProgram::compile(&m).is_none(),
        "a `Vcpu` serves no children, so it declines a module that could page one"
    );
    let args = [Value::I32(0)];
    assert!(
        bytecode::ScheduledDebugRun::new(&m, 0, &args).is_none(),
        "the debug scheduler serves no children, so it declines a module that could page one"
    );
    let export = "export 0 interface \"pager\" 1 { page: 1 }\n";
    assert!(src.contains(export), "the rewrite must apply");
    let unpaged = parse_module(&src.replace(export, "")).expect("parse");
    assert!(
        bytecode::ScheduledDebugRun::new(&unpaged, 0, &args).is_some(),
        "without a `pager` export, the debug scheduler opens a session"
    );
}

/// A dangling budget handle refuses the detached spawn probeably (`-EINVAL`) on both engines,
/// natively: a spawn record's `budget` must name a `Budget` (anything else refuses). This test pinned
/// `CapFault` while its record spawned `module = -1` from a host that registered no program: the
/// module failed to resolve before the budget was looked at (#2219).
#[test]
fn dangling_budget_handle_refuses_the_detached_spawn_on_both_engines() {
    let src = budget_rec_src();
    let image = temen_ir::child_image_at(&parse_module(&src).expect("parse"), 1).expect("image");
    let mut h_tw = Host::new();
    let i_tw = h_tw.grant_instantiator();
    let c_tw = h_tw.grant_module(&image);
    let args_tw = [Value::I32(i_tw), Value::I64(99), Value::I32(c_tw)];
    let r_tw = tw(&src, &args_tw, &mut h_tw);
    assert_eq!(r_tw, Ok(vec![Value::I64(EINVAL)]), "oracle");

    let mut h_bc = Host::new();
    let i_bc = h_bc.grant_instantiator();
    let c_bc = h_bc.grant_module(&image);
    let r_bc = bc(
        &src,
        &[Value::I32(i_bc), Value::I64(99), Value::I32(c_bc)],
        &mut h_bc,
    );
    assert_eq!(r_bc, r_tw, "dangling budget: bytecode != tree-walker");
}

/// #2164 — a paged child's thread pages too: it runs over the child's window, whose pages the pager
/// supplies whichever vCPU touches one first. The parent spawns the child through a record naming its
/// `pager` export, serves in `svc.wait` until that answers 0 (its client has ended, #1217), joins,
/// and returns `child + serves * 1000`. Its pager stores 77 at `addr + 40000` and replies with that
/// address. The child (a module of its own, so the parent's serve escape admits the parent) joins a
/// thread that reads 32768, a page nothing has touched, then reads 49152 itself, and returns the
/// sum. The tree-walker paged only a child's first vCPU, so the thread's fault ended the child; and
/// only the child's end, not its thread's, may tell the pager its client is gone.
#[test]
fn a_paged_childs_thread_is_served_on_every_engine() {
    let parent = format!(
        r#"memory 17
type 0 func (i64) -> (i64)
type 1 interface {{ page: 0 }}
export 0 interface "pager" 1 {{ page: 1 }}
func (i32, i64, i32) -> (i64) {{
block 0 (v0: i32, v1: i64, v2: i32) {{
  qma = i64.const 17544
  i32.store qma v2
  qb = i32.wrap_i64 v1
  qba = i64.const 17548
  i32.store qba qb
  qa0 = i64.const 17536
  vch = call.cap 6 17 (i64) -> (i32) v0 (qa0)
  vz = i64.const 0
  br 1(v0, vch, vz)
}}
block 1 (vh: i32, vch1: i32, vt: i64) {{
  vzw = i32.const 0
  vs = svc.wait vzw
  vt2 = i64.add vt vs
  vz1 = i64.const 0
  vgone = i64.eq vs vz1
  br_if vgone 2(vh, vch1, vt2) 1(vh, vch1, vt2)
}}
block 2 (vh2: i32, vch2: i32, vt3: i64) {{
  vj_st, vj = call.cap 6 1 (i32) -> (i64, i64) vh2 (vch2)
  vk = i64.const 1000
  vsk = i64.mul vt3 vk
  vr = i64.add vj vsk
  return vr
  }}
}}
func (i64) -> (i64) {{
block 0 (vaddr: i64) {{
  voff = i64.const 40000
  vsrc = i64.add vaddr voff
  vb = i32.const 77
  i32.store8 vsrc vb
  return vsrc
  }}
}}
{rec}"#,
        rec = rec::segment(
            17536,
            &SpawnRec {
                pager: 0,
                ..SpawnRec::default()
            }
        )
    );
    let child = parse_module(
        "memory 17
func (i64) -> (i64) {
block 0 (v0: i64) {
  vsp = i64.const 0
  va = i64.const 32768
  vh = thread.spawn 1 vsp va
  vj = thread.join vh
  vb2 = i64.const 49152
  vc = i32.load8_u vb2
  vcw = i64.extend_i32_u vc
  vr = i64.add vj vcw
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, vaddr: i64) {
  vb = i32.load8_u vaddr
  vbw = i64.extend_i32_u vb
  return vbw
  }
}
",
    )
    .expect("parse");
    temen_verify::verify_module(&child).expect("verify");
    let m = std::sync::Arc::new(parse_module(&parent).expect("parse"));
    let powerbox = || {
        let mut h = Host::new();
        h.set_self_module(&m);
        let i = h.grant_instantiator();
        let c = h.grant_module(&child);
        let b = h.grant_budget(5_000_000, -1, -1);
        (h, [Value::I32(i), Value::I64(b as i64), Value::I32(c)])
    };
    let served = Ok(vec![Value::I64(2154)]);
    let (mut h, args) = powerbox();
    assert_eq!(tw(&parent, &args, &mut h), served, "oracle");
    let (mut h, args) = powerbox();
    assert_eq!(bc(&parent, &args, &mut h), served, "cooperative driver");
    let (mut h, args) = powerbox();
    let mut fuel = 10_000_000u64;
    let (r, _) = bytecode::compile_and_run_capture_over_parallel_with_host(
        &m,
        0,
        &args,
        &mut fuel,
        &[],
        None,
        &mut h,
    )
    .expect("the parallel driver runs it natively");
    assert_eq!(r, served, "parallel driver");
}

/// #2164 — a granted child pages its own grandchild, as a root pages a child. The child runs a
/// module of its own that declares a `pager` export. It spawns the grandchild through a record
/// naming that export, serves in `svc.wait` until it answers 0, joins, and returns
/// `grandchild + serves * 1000`. The grandchild returns the byte the pager supplied, 77. The root
/// declares no pager; it spawns the child, granting it the grandchild's module as `"grand"`, and
/// returns `status * 1_000_000 + value`.
///
/// The drivers that serve their children page the grandchild, as the oracle does.
#[test]
fn a_granted_child_pages_its_grandchild_on_every_engine() {
    let root = format!(
        r#"memory 17
data 17472 "grand"
func (i32, i32, i32, i32) -> (i64) {{
block 0 (vi: i32, vbud: i32, vchild: i32, vgrand: i32) {{
  vg = i64.const 17408
  vw = i64.const 21474853952
  i64.store vg vw
  vgh = i64.const 17416
  vgr = i64.extend_i32_u vgrand
  i64.store vgh vgr
  vrm = i64.const 17544
  i32.store vrm vchild
  vrb = i64.const 17548
  i32.store vrb vbud
  vrp = i64.const 17536
  vh = call.cap 6 17 (i64) -> (i32) vi (vrp)
  vj_st, vj = call.cap 6 1 (i32) -> (i64, i64) vi (vh)
  vm = i64.const 1000000
  vs = i64.mul vj_st vm
  vr = i64.add vs vj
  return vr
  }}
}}
{rec}"#,
        rec = rec::segment(
            17536,
            &SpawnRec {
                size_log2: 17,
                grants_ptr: 17408,
                grants_n: 1,
                ..SpawnRec::default()
            }
        )
    );
    let child = format!(
        r#"memory 17
type 0 func (i64) -> (i64)
type 1 interface {{ page: 0 }}
export 0 interface "pager" 1 {{ page: 1 }}
data 17472 "grand"
data 17480 "budget"
func (i64, i64) -> (i64) {{
block 0 (v0: i64, v1: i64) {{
  vbp = i64.const 17480
  vbl = i64.const 6
  vbud = self.resolve vbp vbl
  vgp = i64.const 17472
  vgl = i64.const 5
  vgrand = self.resolve vgp vgl
  vrm = i64.const 17640
  i32.store vrm vgrand
  vrb = i64.const 17644
  i32.store vrb vbud
  vrp = i64.const 17632
  vi = i32.wrap_i64 v0
  vh = call.cap 6 17 (i64) -> (i32) vi (vrp)
  vz = i64.const 0
  br 1(vi, vh, vz)
}}
block 1 (vi1: i32, vh1: i32, vt: i64) {{
  vzw = i32.const 0
  vs = svc.wait vzw
  vt2 = i64.add vt vs
  vz1 = i64.const 0
  vgone = i64.eq vs vz1
  br_if vgone 2(vi1, vh1, vt2) 1(vi1, vh1, vt2)
}}
block 2 (vi2: i32, vh2: i32, vt3: i64) {{
  vj_st, vj = call.cap 6 1 (i32) -> (i64, i64) vi2 (vh2)
  vk = i64.const 1000
  vsk = i64.mul vt3 vk
  vr = i64.add vj vsk
  return vr
  }}
}}
func (i64) -> (i64) {{
block 0 (vaddr: i64) {{
  voff = i64.const 40000
  vsrc = i64.add vaddr voff
  vb = i32.const 77
  i32.store8 vsrc vb
  return vsrc
  }}
}}
{rec}"#,
        rec = rec::segment(
            17632,
            &SpawnRec {
                pager: 0,
                ..SpawnRec::default()
            }
        )
    );
    let grand = parse_module(
        "memory 17
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  va = i64.const 49152
  vb = i32.load8_u va
  vw = i64.extend_i32_u vb
  return vw
  }
}
",
    )
    .expect("parse");
    let child = parse_module(&child).expect("parse");
    for m in [&child, &grand] {
        temen_verify::verify_module(m).expect("verify");
    }
    let m = std::sync::Arc::new(parse_module(&root).expect("parse"));
    let powerbox = || {
        let mut h = Host::new();
        h.set_self_module(&m);
        let i = h.grant_instantiator();
        let b = h.grant_budget(-1, -1, -1);
        let c = h.grant_module(&child);
        let g = h.grant_module(&grand);
        (h, [i, b, c, g].map(Value::I32))
    };
    let served = Ok(vec![Value::I64(77 + 1000)]);
    let (mut h, args) = powerbox();
    assert_eq!(tw(&root, &args, &mut h), served, "oracle");
    let (mut h, args) = powerbox();
    assert_eq!(bc(&root, &args, &mut h), served, "cooperative driver");
    let (mut h, args) = powerbox();
    let mut fuel = 10_000_000u64;
    let (r, _) = bytecode::compile_and_run_capture_over_parallel_with_host(
        &m,
        0,
        &args,
        &mut fuel,
        &[],
        None,
        &mut h,
    )
    .expect("the parallel driver runs it natively");
    assert_eq!(r, served, "parallel driver");
}
