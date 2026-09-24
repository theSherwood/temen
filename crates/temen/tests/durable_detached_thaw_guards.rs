//! #1733 — a thawed detached (op 15) child keeps the protections its window was built with: the
//! NULL guard and its `readonly` data segments. A thaw is indistinguishable from not freezing, so each
//! probe answers the same on a fresh run and after a freeze → codec → thaw, whichever engine froze
//! the tree and whichever thaws it.
#![cfg(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
))]

use temen_durable::{
    begin_thaw, init_durable_window, transform_module, transform_module_assume_confined,
    write_state, STATE_UNWINDING,
};
use temen_interp::{run_capture_reserved_with_host, FreezeScope, Host, Value};
use temen_ir::durable_abi::ShadowArena;
use temen_ir::errno::EINVAL;
use temen_jit::{JitError, JitOutcome};

const ARENA: ShadowArena = ShadowArena {
    base: 16448,
    end: 65536,
};
const PARENT_LOG2: u8 = 18;

/// Spawn the child detached (op 15: budget, module, no grants, entry 0, `size_log2` 17, no quota),
/// join it, return what it returns (a child trap propagates at the join).
const PARENT: &str = "memory 18 shadow 16448 65536
func (i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32) {
  vmh = i64.extend_i32_u v1
  vb = i64.extend_i32_u v2
  vz = i64.const 0
  vlog = i64.const 17
  vc = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vb, vmh, vz, vz, vz, vlog, vz)
  vr = call.cap 6 1 (i32) -> (i64) v0 (vc)
  return vr
  }
}
";

/// The child (128 KiB; a readonly segment at 81920, its own 16 KiB page clear of the shadow arena):
/// a zero-length `unmap` (refused, and there only to make the function may-suspend, so its loop header
/// polls), then a polled loop a freeze cuts, then `probe`, which may use `vsp` (its `AddressSpace`)
/// and binds `pr`, the value it returns.
fn child(probe: &str) -> String {
    format!(
        "memory 17 shadow 16448 65536
data ro 81920 \"abcd\"
func (i64, i64) -> (i64) {{
block 0 (v0: i64, v1: i64) {{
  vs0 = i32.wrap_i64 v1
  vz = i64.const 0
  vu = call.cap 5 1 (i64, i64) -> (i64) vs0 (vz, vz)
  br 1(vz, v1)
}}
block 1 (vi: i64, vs: i64) {{
  vn = i64.const 1000000
  vc = i64.lt_s vi vn
  br_if vc 2(vi, vs) 3(vs)
}}
block 2 (vj: i64, vt: i64) {{
  vo = i64.const 1
  vk = i64.add vj vo
  br 1(vk, vt)
}}
block 3 (vsp64: i64) {{
  vsp = i32.wrap_i64 vsp64
{probe}  return pr
  }}
}}
"
    )
}

/// `unmap` the NULL guard's first page through the child's own `AddressSpace`.
const UNMAP_GUARD: &str = "  vg0 = i64.const 0
  vg1 = i64.const 16384
  pr = call.cap 5 1 (i64, i64) -> (i64) vsp (vg0, vg1)
";
/// Load through NULL.
const NULL_LOAD: &str = "  vn0 = i64.const 0
  vnl = i32.load vn0
  pr = i64.extend_i32_u vnl
";
/// Store to the readonly segment.
const RO_STORE: &str = "  vra = i64.const 81920
  vrv = i32.const 7
  i32.store vra vrv
  vrl = i32.load vra
  pr = i64.extend_i32_u vrl
";
/// Control: read the readonly segment.
const RO_LOAD: &str = "  vra = i64.const 81920
  vrl = i32.load vra
  pr = i64.extend_i32_u vrl
";

/// What a run answers, engine-neutral: a returned value or a trap's name.
#[derive(Debug, PartialEq, Eq, Clone)]
enum Out {
    Ret(i64),
    Trap(String),
}

fn parse(src: &str) -> temen_ir::Module {
    temen_text::parse_module(src).expect("parse")
}

fn verified(m: temen_ir::Module) -> temen_ir::Module {
    temen_verify::verify_module(&m).expect("instrumented module verifies");
    m
}

fn parent() -> temen_ir::Module {
    verified(transform_module(&parse(PARENT)).expect("transform"))
}

/// The child touches memory, so it takes the confined transform: its data sits above the arena, and
/// its NULL probes are the point.
fn child_module(probe: &str) -> temen_ir::Module {
    verified(transform_module_assume_confined(&parse(&child(probe))).expect("transform"))
}

fn powerbox(child: &temen_ir::Module) -> (Host, Vec<i64>) {
    let mut host = Host::new();
    host.set_durable(true);
    let inst = host.grant_instantiator(0, 1 << PARENT_LOG2);
    let modh = host.grant_durable_module(child);
    let budget = host.grant_budget(0, 1 << 20, 0);
    host.grant_freeze_authority(FreezeScope::DetachedProgeny);
    (host, vec![inst as i64, modh as i64, budget as i64])
}

#[derive(Clone, Copy, Debug)]
enum Engine {
    Interp,
    Jit,
}

/// Run the parent over `win` on `engine`. `None` on a target without the JIT's child executor.
fn run(
    engine: Engine,
    parent: &temen_ir::Module,
    args: &[i64],
    win: &[u8],
    host: &mut Host,
) -> Option<(Out, Vec<u8>)> {
    match engine {
        Engine::Interp => {
            let iargs: Vec<Value> = args.iter().map(|&a| Value::I32(a as i32)).collect();
            let mut fuel = u64::MAX / 2;
            let (r, snap) = run_capture_reserved_with_host(
                parent,
                0,
                &iargs,
                &mut fuel,
                win,
                PARENT_LOG2,
                host,
            );
            let out = match r {
                Ok(v) => match v[..] {
                    [Value::I64(n)] => Out::Ret(n),
                    ref other => panic!("unexpected result {other:?}"),
                },
                Err(t) => Out::Trap(format!("{t:?}")),
            };
            Some((out, snap))
        }
        Engine::Jit => match temen_run::jit_cap_run(parent, 0, args, win, PARENT_LOG2, 0, host) {
            Ok((JitOutcome::Returned(v), snap)) => Some((Out::Ret(v[0]), snap)),
            Ok((JitOutcome::Trapped(t), snap)) => Some((Out::Trap(format!("{t:?}")), snap)),
            Ok((other, _)) => panic!("unexpected outcome {other:?}"),
            Err(JitError::Unsupported(_)) => None,
            Err(e) => panic!("JIT run failed: {e:?}"),
        },
    }
}

/// Fresh: the probe on `engine` with no freeze.
fn fresh(engine: Engine, probe: &str) -> Option<Out> {
    let child = child_module(probe);
    let (mut host, args) = powerbox(&child);
    let win = init_durable_window(1 << PARENT_LOG2, ARENA);
    run(engine, &parent(), &args, &win, &mut host).map(|(o, _)| o)
}

/// Freeze from the start on `froze` (the child is cut in its loop, before the probe), carry the tree
/// through the codec, and thaw it on `thaws`.
fn thawed(froze: Engine, thaws: Engine, probe: &str) -> Option<Out> {
    let parent = parent();
    let child = child_module(probe);
    let mut win = init_durable_window(1 << PARENT_LOG2, ARENA);
    write_state(&mut win, STATE_UNWINDING);
    // The JIT runs the child on its own thread, so on a loaded runner it can finish its loop (and
    // run the probe) before the freeze reaches it; that run is not the case under test — retry it
    // (#1760).
    let mut attempts = 0;
    let (fhost, args, fsnap) = loop {
        let (mut fhost, args) = powerbox(&child);
        let (_, fsnap) = run(froze, &parent, &args, &win, &mut fhost)?;
        if fhost.captured_detached().len() == 1 {
            break (fhost, args, fsnap);
        }
        attempts += 1;
        assert!(attempts < 50, "{froze:?} never froze the child live");
    };
    let art = temen_snapshot::freeze(&parent, &fsnap, &fhost).expect("serialize");
    let mut thost = Host::new();
    thost.set_durable(true);
    thost.grant_durable_module(&child);
    let mut twin = temen_snapshot::restore(&art, &parent, &mut thost).expect("restore");
    begin_thaw(&mut twin, ARENA, 0);
    run(thaws, &parent, &args, &twin, &mut thost).map(|(o, _)| o)
}

/// Every fresh and thawed run of `probe` answers `want`; reports every cell that does not.
fn check(probe: &str, want: Out) {
    use Engine::*;
    let mut wrong = Vec::new();
    for e in [Interp, Jit] {
        match fresh(e, probe) {
            Some(o) if o != want => wrong.push(format!("fresh on {e:?}: {o:?}")),
            _ => {}
        }
    }
    for froze in [Interp, Jit] {
        for thaws in [Interp, Jit] {
            match thawed(froze, thaws, probe) {
                Some(o) if o != want => {
                    wrong.push(format!("frozen on {froze:?}, thawed on {thaws:?}: {o:?}"))
                }
                _ => {}
            }
        }
    }
    assert!(wrong.is_empty(), "want {want:?}; got\n{}", wrong.join("\n"));
}

/// Control: the readonly segment reads back after a thaw.
#[test]
fn a_thawed_detached_child_reads_its_readonly_segment() {
    check(RO_LOAD, Out::Ret(0x6463_6261));
}

/// #1733: the NULL region stays reserved — `unmap` of it is refused, as before the freeze.
#[test]
fn a_thawed_detached_child_cannot_unmap_its_null_guard() {
    check(UNMAP_GUARD, Out::Ret(EINVAL));
}

/// #1733: a NULL load faults after a thaw as before it.
#[test]
fn a_thawed_detached_childs_null_load_faults() {
    check(NULL_LOAD, Out::Trap("MemoryFault".into()));
}

/// A store to the readonly segment faults after a thaw as before it.
#[test]
fn a_thawed_detached_childs_store_to_its_readonly_segment_faults() {
    check(RO_STORE, Out::Trap("MemoryFault".into()));
}
