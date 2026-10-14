//! The detached-child freeze and thaw the durable tests share: a parent that spawns a child detached
//! and joins it, the child's probe sources, and a freeze → codec → thaw on either engine. Included
//! by `durable_detached_thaw_guards.rs` (#1733) and `thaw_pool.rs` (#2318).
#![allow(dead_code)] // each including binary uses a different subset

use temen_durable::{
    begin_thaw, init_durable_window, transform_module, transform_module_assume_confined,
    write_state, STATE_UNWINDING,
};
use temen_interp::{run_capture_reserved_with_host, Host, MemLayout, Value};
use temen_ir::durable_abi::ShadowArena;
use temen_jit::{JitError, JitOutcome};

pub const ARENA: ShadowArena = ShadowArena::new(16448, 65536);
pub const PARENT_LOG2: u8 = 18;

/// Spawn the child detached (op 15: budget, module, no grants, entry 0, `size_log2` 17, no quota),
/// join it, return `status * 1000 + value`: what it returns, or its trap's wire code times 1000
/// (#2066).
pub const PARENT: &str = "memory 18 shadow 16448 65536
func (i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32) {
  vmh = i64.extend_i32_u v1
  vb = i64.extend_i32_u v2
  vz = i64.const 0
  vlog = i64.const 17
  vc = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vb, vmh, vz, vz, vz, vlog, vz)
  vr_st, vr = call.cap 6 1 (i32) -> (i64, i64) v0 (vc)
  vk = i64.const 1000
  vs = i64.mul vr_st vk
  vt = i64.add vs vr
  return vt
  }
}
";

/// The child (128 KiB; a readonly segment at 81920, its own 16 KiB page clear of the shadow arena):
/// a zero-length `unmap` (refused, and there only to make the function may-suspend, so its loop header
/// polls), then a polled loop a freeze cuts, then `probe`, which may use `vsp` (its `AddressSpace`)
/// and binds `pr`, the value it returns.
pub fn child(probe: &str) -> String {
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

/// What a run answers, engine-neutral: a returned value or a trap's name.
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Out {
    Ret(i64),
    Trap(String),
}

pub fn parse(src: &str) -> temen_ir::Module {
    temen_text::parse_module(src).expect("parse")
}

pub fn verified(m: temen_ir::Module) -> temen_ir::Module {
    temen_verify::verify_module(&m).expect("instrumented module verifies");
    m
}

pub fn parent() -> temen_ir::Module {
    parent_of(PARENT)
}

/// A parent from its source, instrumented as [`PARENT`] is.
pub fn parent_of(src: &str) -> temen_ir::Module {
    verified(transform_module(&parse(src)).expect("transform"))
}

/// The child touches memory, so it takes the confined transform: its data sits above the arena, and
/// its NULL probes are the point.
pub fn child_module(src: &str) -> temen_ir::Module {
    verified(transform_module_assume_confined(&parse(src)).expect("transform"))
}

/// The parent's powerbox: its `Instantiator`, `child` as a durable `Module`, and a `Budget` of 1 MiB
/// whose `spawn` ceiling is 4 ([`a_thawed_thread_is_charged_to_its_childs_budget_again`] reads it).
pub fn powerbox(child: &temen_ir::Module) -> (Host, Vec<i64>) {
    let mut host = Host::new();
    host.set_durable(true);
    let inst = host.grant_instantiator(0, 1 << PARENT_LOG2);
    let modh = host.grant_durable_module(child);
    let budget = host.grant_budget(-1, 1 << 20, 4);
    host.grant_freeze_authority();
    (host, vec![inst as i64, modh as i64, budget as i64])
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Engine {
    Interp,
    Jit,
}

/// Run the parent over `win` on `engine`. `None` on a target without the JIT's child executor.
pub fn run(
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
        Engine::Jit => match temen_run::jit_cap_run(
            parent,
            0,
            args,
            &MemLayout::image(win.to_vec()),
            PARENT_LOG2,
            0,
            host,
            None,
        ) {
            Ok((JitOutcome::Returned(v), snap)) => Some((Out::Ret(v[0]), snap.bytes().to_vec())),
            Ok((JitOutcome::Trapped(t), snap)) => {
                Some((Out::Trap(format!("{t:?}")), snap.bytes().to_vec()))
            }
            Ok((other, _)) => panic!("unexpected outcome {other:?}"),
            Err(JitError::Unsupported(_)) => None,
            Err(e) => panic!("JIT run failed: {e:?}"),
        },
    }
}

/// Fresh: `parent` over child `src` on `engine` with no freeze.
pub fn fresh(engine: Engine, parent: &temen_ir::Module, src: &str) -> Option<Out> {
    let child = child_module(src);
    let (mut host, args) = powerbox(&child);
    let win = init_durable_window(1 << PARENT_LOG2, ARENA);
    run(engine, parent, &args, &win, &mut host).map(|(o, _)| o)
}

/// Freeze `parent` over `child` from the start on `engine`, so the child is cut in its loop before
/// the probe: the frozen powerbox (the child captured live), the args, and the window image.
pub fn frozen(
    engine: Engine,
    parent: &temen_ir::Module,
    child: &temen_ir::Module,
) -> Option<(Host, Vec<i64>, Vec<u8>)> {
    let mut win = init_durable_window(1 << PARENT_LOG2, ARENA);
    write_state(&mut win, STATE_UNWINDING);
    // The JIT runs the child on its own thread, so on a loaded runner it can finish its loop (and
    // run the probe) before the freeze reaches it; that run is not the case under test — retry it
    // (#1760).
    let mut attempts = 0;
    loop {
        let (mut fhost, args) = powerbox(child);
        let (_, fsnap) = run(engine, parent, &args, &win, &mut fhost)?;
        if fhost.captured_detached().len() == 1 {
            return Some((fhost, args, fsnap));
        }
        attempts += 1;
        assert!(attempts < 50, "{engine:?} never froze the child live");
    }
}

/// Freeze `parent` on `froze` ([`frozen`]), carry the tree through the codec, and thaw it on `thaws`.
pub fn thawed(froze: Engine, thaws: Engine, parent: &temen_ir::Module, src: &str) -> Option<Out> {
    let child = child_module(src);
    let (art, args) = artifact(froze, parent, &child)?;
    thaw(&art, thaws, parent, &child, &args)
}

/// Freeze `parent` over `child` on `froze` ([`frozen`]) and carry the tree through the codec: the
/// artifact, and the args the parent runs on.
pub fn artifact(
    froze: Engine,
    parent: &temen_ir::Module,
    child: &temen_ir::Module,
) -> Option<(Vec<u8>, Vec<i64>)> {
    let (fhost, args, fsnap) = frozen(froze, parent, child)?;
    let art = temen_snapshot::freeze(parent, &fsnap, &fhost).expect("serialize");
    Some((art, args))
}

/// Thaw `art`, frozen over `child` ([`artifact`]), and run it on `thaws`.
pub fn thaw(
    art: &[u8],
    thaws: Engine,
    parent: &temen_ir::Module,
    child: &temen_ir::Module,
    args: &[i64],
) -> Option<Out> {
    let mut thost = Host::new();
    thost.set_durable(true);
    thost.grant_durable_module(child);
    let mut twin = temen_snapshot::restore(art, parent, &mut thost).expect("restore");
    begin_thaw(&mut twin, ARENA, 0);
    run(thaws, parent, args, &twin, &mut thost).map(|(o, _)| o)
}
