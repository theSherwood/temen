//! **§14 nested child on the *emitted* tier** (issue #1123) — the wasm-JIT twin of the native
//! `compile_child` (`temen-llvm/tests/rust_guest_op13.rs`). Where `instantiate_rec.rs` services the
//! spawn bounce by running the child on the tree-walk interpreter, this file services it by
//! **emitting the child module and running it as its own `wasmi` instance** — so both the parent and
//! the child execute on emitted wasm, exactly as the native Cranelift path runs a nimony phase child
//! on emitted code.
//!
//! The spawn is op 17 with a v1 record (#1865): the servicer decodes the record, mints the child a
//! memory of its own and emits the child over its declared window there. The carve spawns (ops 0
//! and 13) are retired (#1867) and no longer lowered. Also pinned: an emitted child's own
//! confinement — an out-of-window access faults, run directly or through the bounce — and its NULL
//! guard.
//!
//! Oracle (INVARIANTS.md #9): the child run on the interpreter gives the emitted child's answer.
//! `saw_emit`/`saw_module` are the non-vacuity guards (the handler really emitted + ran the child,
//! not a silent interpreter fallback).

mod support;

use temen_interp::{run, Value};
use temen_wasm_jit::compile_module_nested;
use wasmi::{Caller, Engine, Linker, Memory, MemoryType, Module as WModule, Store, Val};

const WIN_BASE: i32 = 0x1_0000; // parent window base (the env cell lives below it)
const ENV_PTR: i32 = 1024; // parent dispatcher-fuel cell
const CHILD_ENV_PTR: i32 = 512; // child dispatcher-fuel cell (below the parent's scratch)
const CARVE_OFF: i64 = 16384; // where a directly-run child's window sits past `WIN_BASE`: one #1094 NULL guard up, 1 KiB (child `memory 10`)-aligned

fn parse(src: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// Interpreter oracle for the child module's entry (func 0), run directly.
fn oracle_child(child: &temen_ir::Module) -> i64 {
    let mut fuel = u64::MAX;
    match run(child, 0, &[], &mut fuel) {
        Ok(v) => match v.first() {
            Some(Value::I64(x)) => *x,
            other => panic!("child oracle: {other:?}"),
        },
        other => panic!("child oracle run: {other:?}"),
    }
}

/// Host state threaded through the wasmi `Store`: the emitted child module and its IR, each spawned
/// child's result (indexed by the handle `env.instantiate_rec` returns), and the non-vacuity flags.
struct HostState {
    /// The emitted child module, instantiated afresh over each detached (op-17) child's own memory.
    child_module: Option<WModule>,
    child: Option<temen_ir::Module>,
    children: Vec<i64>,
    saw_emit: bool,
    saw_module: bool,
}

/// Run the child module's `f{entry}` **detached**: in a fresh memory of its own, over its declared
/// window at [`WIN_BASE`] (its emitted `"mapped"` self-inits to that window), with its own copy of the
/// nested imports. Bank its result under a dense handle and return that handle. Nothing of the
/// parent's memory is reachable from it.
fn spawn_detached_child(
    caller: &mut Caller<'_, HostState>,
    entry: u32,
) -> Result<i32, wasmi::Error> {
    let st = caller.data();
    let module = st.child_module.clone().expect("child module emitted");
    let declared = st
        .child
        .as_ref()
        .and_then(|c| c.memory.as_ref())
        .map_or(0, |mc| 1u64 << mc.size_log2);
    let pages = ((WIN_BASE as u64 + declared) / 65536 + 1) as u32;
    let own = Memory::new(&mut *caller, MemoryType::new(pages, None)).unwrap();
    own.write(
        &mut *caller,
        CHILD_ENV_PTR as usize,
        &i64::MAX.to_le_bytes(),
    )
    .unwrap();
    let mut linker: Linker<HostState> = Linker::new(caller.engine());
    wire_nested_imports(&mut linker, own);
    let inst = linker
        .instantiate(&mut *caller, &module)?
        .start(&mut *caller)?;
    let f = inst
        .get_func(&*caller, &format!("f{entry}"))
        .expect("child entry export");
    let mut r = [Val::I64(0)];
    f.call(
        &mut *caller,
        &[Val::I32(WIN_BASE), Val::I32(CHILD_ENV_PTR)],
        &mut r,
    )?;
    let st = caller.data_mut();
    st.children.push(r[0].i64().expect("child returns i64"));
    Ok((st.children.len() - 1) as i32)
}

/// Define the §14 nested imports on `linker`, sharing `memory`. `instantiate_rec` emits + runs the
/// child detached; `join` returns its banked result; the rest are unreachable here.
fn wire_nested_imports(linker: &mut Linker<HostState>, memory: Memory) {
    linker.define("env", "memory", memory).unwrap();
    linker
        .func_wrap("env", "trap", |_: Caller<'_, HostState>, _code: i32| {})
        .unwrap();
    linker
        .func_wrap::<_, ()>(
            "env",
            "call_interp",
            |_: Caller<'_, HostState>, _f: i32, _a: i32| {
                unreachable!("no interp leaf in this unit")
            },
        )
        .unwrap();
    // env.instantiate: the fixed carve-spawn slot (#1867). No emitted code calls it.
    linker
        .func_wrap(
            "env",
            "instantiate",
            |_: Caller<'_, HostState>,
             _win: i32,
             _inst: i32,
             _entry: i64,
             _off: i64,
             _slog: i64,
             _quota: i64|
             -> i32 { unreachable!("no carve spawn is emitted") },
        )
        .unwrap();
    // §3c.3 env.instantiate_rec (op 17): a detached spawn. Read the record out of the parent's
    // window, decode it as every tier does, and run the child in a memory of its own.
    linker
        .func_wrap(
            "env",
            "instantiate_rec",
            move |mut caller: Caller<'_, HostState>,
                  win: i32,
                  _inst: i32,
                  record_ptr: i64|
                  -> Result<i32, wasmi::Error> {
                let mut rec = [0u8; temen_ir::SPAWN_REC_LEN];
                memory
                    .read(&caller, (win as u64 + record_ptr as u64) as usize, &mut rec)
                    .expect("record in-window");
                let rec = temen_ir::SpawnRec::parse(&rec).expect("a well-formed record");
                assert_eq!(rec.modh, CHILD_MODULE_HANDLE, "the child module's handle");
                let st = caller.data_mut();
                st.saw_emit = true;
                st.saw_module = true;
                spawn_detached_child(&mut caller, rec.entry)
            },
        )
        .unwrap();
    linker
        .func_wrap(
            "env",
            "join",
            |caller: Caller<'_, HostState>, _inst: i32, child: i32| -> (i64, i64) {
                // A `-1` (or otherwise out-of-range) handle means the gate refused the spawn — map it
                // to a sentinel instead of indexing, so a fail-closed rejection surfaces as a value.
                // A child that ran returned at its spawn: `(status 0, its result)` (#2066).
                let st = caller.data();
                if child < 0 || child as usize >= st.children.len() {
                    return (0, i64::MIN);
                }
                (0, st.children[child as usize])
            },
        )
        .unwrap();
    linker
        .func_wrap(
            "env",
            "thread_spawn",
            |_: Caller<'_, HostState>, _f: i32, _sp: i64, _a: i64| -> i32 {
                unreachable!("no thread op in this unit")
            },
        )
        .unwrap();
    linker
        .func_wrap(
            "env",
            "thread_join",
            |_: Caller<'_, HostState>, _h: i32| -> i64 {
                unreachable!("no thread op in this unit")
            },
        )
        .unwrap();
    linker
        .func_wrap(
            "env",
            "mem_wait",
            |_: Caller<'_, HostState>, _w: i32, _a: i64, _e: i64, _t: i64, _is64: i32| -> i32 {
                unreachable!("no futex op in this unit")
            },
        )
        .unwrap();
    linker
        .func_wrap(
            "env",
            "mem_notify",
            |_: Caller<'_, HostState>, _w: i32, _a: i64, _c: i32| -> i32 {
                unreachable!("no futex op in this unit")
            },
        )
        .unwrap();
}

/// The result of running a parent over its child: `(parent_result, [i64@addr for addr in reads],
/// saw_module)`.
type ParentRun = (i64, Vec<i64>, bool);

/// Build `parent_src` + `child_src` (both `compile_module_nested`), run the parent's `f0(params)` on
/// wasmi, its op-17 spawn emitting the child into a memory of its own, and return the [`ParentRun`] —
/// or the wasmi error if the parent call traps (e.g. an escaping child fault propagated through the
/// bounce).
fn try_run_parent_over_child(
    parent_src: &str,
    child_src: &str,
    params: &[Val],
    reads: &[usize],
) -> Result<ParentRun, wasmi::Error> {
    let parent = parse(parent_src);
    let child = parse(child_src);
    let parent_wasm = compile_module_nested(&parent, false).expect("parent emits (nested)");
    let child_wasm = compile_module_nested(&child, false).expect("child module emits (nested)");

    let engine = Engine::default();
    let parent_module = WModule::new(&engine, &parent_wasm).expect("parent wasm validates");
    let child_module = WModule::new(&engine, &child_wasm).expect("child wasm validates");

    let mut store: Store<HostState> = Store::new(
        &engine,
        HostState {
            child_module: Some(child_module.clone()),
            child: Some(child),
            children: Vec::new(),
            saw_emit: false,
            saw_module: false,
        },
    );
    // Two pages (128 KiB) hold the parent window `[WIN_BASE, WIN_BASE + 64 KiB)`; the carve lives inside.
    let memory = Memory::new(&mut store, MemoryType::new(2, None)).unwrap();
    memory
        .write(&mut store, ENV_PTR as usize, &i64::MAX.to_le_bytes())
        .unwrap();
    memory
        .write(&mut store, CHILD_ENV_PTR as usize, &i64::MAX.to_le_bytes())
        .unwrap();

    let mut linker: Linker<HostState> = Linker::new(&engine);
    wire_nested_imports(&mut linker, memory);

    let parent_instance = linker
        .instantiate(&mut store, &parent_module)
        .unwrap()
        .start(&mut store)
        .unwrap();
    let f0 = parent_instance
        .get_func(&store, "f0")
        .expect("parent f0 export");

    let mut results = [Val::I64(0)];
    f0.call(&mut store, params, &mut results)?;

    assert!(
        store.data().saw_emit,
        "parent never bounced to the instantiate host import (silent fallback)"
    );
    let out = reads
        .iter()
        .map(|&a| {
            let mut b = [0u8; 8];
            memory.read(&store, a, &mut b).unwrap();
            i64::from_le_bytes(b)
        })
        .collect();
    Ok((
        results[0].i64().expect("parent returns i64"),
        out,
        store.data().saw_module,
    ))
}

/// The success-expecting front of [`try_run_parent_over_child`] — for tests where the parent must run
/// to completion. Returns `(parent_result, reads, saw_module)`.
fn run_parent_over_child(
    parent_src: &str,
    child_src: &str,
    params: &[Val],
    reads: &[usize],
) -> (i64, Vec<i64>, bool) {
    try_run_parent_over_child(parent_src, child_src, params, reads).expect("parent f0 runs")
}

/// A child that stores **out of its `memory 10` window** — at byte 1040 (> 1024). The wasm tier's
/// confinement (§4, D38) is the trap-confinement check `if eff > mapped - width { trap(MemoryFault) }`
/// with `mapped = 1 << size_log2 = 1024`, so the store `eff = 1040 > 1016` **faults** before touching
/// memory. Run over the carve at `win + off`, the fault means the child cannot reach `carve + 1040`
/// (past its 1-KiB carve, into the parent's window) — the confinement hinge for a nested child.
const CHILD_ESCAPE: &str = r#"memory 10
func () -> (i64) {
block 0 () {
  vaddr = i64.const 1040
  vval = i64.const 77
  i64.store vaddr vval
  vr = i64.const 0
  return vr
  }
}
"#;

#[test]
fn emitted_child_out_of_window_access_faults_and_stays_confined() {
    let child = parse(CHILD_ESCAPE);
    let child_wasm = compile_module_nested(&child, false).expect("escape child emits (nested)");

    let engine = Engine::default();
    let child_module = WModule::new(&engine, &child_wasm).expect("child wasm validates");
    let mut store: Store<HostState> = Store::new(
        &engine,
        HostState {
            child_module: None,
            child: None,
            children: Vec::new(),
            saw_emit: false,
            saw_module: false,
        },
    );
    let memory = Memory::new(&mut store, MemoryType::new(2, None)).unwrap();
    memory
        .write(&mut store, CHILD_ENV_PTR as usize, &i64::MAX.to_le_bytes())
        .unwrap();
    let mut linker: Linker<HostState> = Linker::new(&engine);
    wire_nested_imports(&mut linker, memory);
    let child_instance = linker
        .instantiate(&mut store, &child_module)
        .unwrap()
        .start(&mut store)
        .unwrap();
    let f0 = child_instance
        .get_func(&store, "f0")
        .expect("child emitted f0 export");

    // Run the child over the carve at `WIN_BASE + CARVE_OFF`. Its out-of-window store must fault
    // (trap-confinement), not silently land past the carve.
    let carve = WIN_BASE as i64 + CARVE_OFF;
    let mut r = [Val::I64(0)];
    let call = f0.call(
        &mut store,
        &[Val::I32(carve as i32), Val::I32(CHILD_ENV_PTR)],
        &mut r,
    );
    assert!(
        call.is_err(),
        "confinement (§4): an out-of-window child access must fault, not proceed"
    );

    // Nothing was written outside (or inside) the carve: the faulting store never reached memory.
    let read = |addr: i64| -> i64 {
        let mut b = [0u8; 8];
        memory.read(&store, addr as usize, &mut b).unwrap();
        i64::from_le_bytes(b)
    };
    assert_eq!(
        read(carve + 1040),
        0,
        "confinement: the child wrote nothing past its carve (into the parent window)"
    );
    assert_eq!(
        read(carve + 16),
        0,
        "the faulting store never executed (no masked write either)"
    );
}

/// The handle the op-17 test passes its parent as the child `Module` (the servicer checks the record
/// carries it).
const CHILD_MODULE_HANDLE: i32 = 99;

/// Where the op-17 parent builds its record: above the #1094 NULL guard.
const REC_AT: u64 = 18432;

/// The op-17 parent (#1865): `v0` is its `Instantiator`, `v1` the child `Module`. It stores `7` at its
/// own window's byte 16392, builds a v1 record for the child's func 0 (its declared window, no budget,
/// no grants) with the module handle filled in at run time, spawns it detached, joins it, and returns
/// the child's result plus its own byte 16392. [`CHILD_ABOVE_GUARD`] stores `55` at *its* 16392 and
/// returns it, so a child that ran in the parent's window would make this `110`, not `62`.
fn parent_op17() -> String {
    let stores = support::rec_stores(REC_AT, &temen_ir::SpawnRec::v1(0));
    let modh_at = REC_AT + 24;
    format!(
        r#"memory 16
func (i32, i32) -> (i64) {{
block 0 (v0: i32, v1: i32) {{
  vsa = i64.const 16392
  vsv = i64.const 7
  i64.store vsa vsv
{stores}  vma = i64.const {modh_at}
  i32.store vma v1
  vrp = i64.const {REC_AT}
  vh = call.cap 6 17 (i64) -> (i32) v0 (vrp)
  vr_st, vr = call.cap 6 1 (i32) -> (i64, i64) v0 (vh)
  vmine = i64.load vsa
  vs = i64.add vr vmine
  return vs
  }}
}}
"#
    )
}

#[test]
fn op17_detached_child_runs_on_the_emitted_tier_over_its_own_memory() {
    let want = oracle_child(&parse(CHILD_ABOVE_GUARD));
    assert_eq!(want, 55, "child oracle");
    let params = [
        Val::I32(WIN_BASE),
        Val::I32(ENV_PTR),
        Val::I32(7),
        Val::I32(CHILD_MODULE_HANDLE),
    ];
    let (r, reads, saw_module) = run_parent_over_child(
        &parent_op17(),
        CHILD_ABOVE_GUARD,
        &params,
        &[(WIN_BASE + 16392) as usize],
    );
    assert!(saw_module, "the parent bounced to env.instantiate_rec");
    assert_eq!(
        r,
        want + 7,
        "the child's result plus the parent's own untouched byte"
    );
    assert_eq!(reads[0], 7, "the child never wrote the parent's window");
}

// ---------------------------------------------------------------------------------------------------
// #1123 slice 2 — confinement: per-event window routing + the fail-closed carve gate.
// ---------------------------------------------------------------------------------------------------

/// Emit `child_src` (nested) and run its `f0` **directly** over a carve of `1 << carve_log2` bytes at
/// `WIN_BASE`, routing its live `"mapped"` global to the carve size. Returns the child's `i64` result, or
/// the wasmi error a confinement fault surfaces as. Probes a single child's emitted checks (the NULL
/// guard, the window bound) without a parent bounce.
fn run_child_direct(child_src: &str, carve_log2: u32) -> Result<i64, wasmi::Error> {
    let child = parse(child_src);
    let child_wasm = compile_module_nested(&child, false).expect("child emits (nested)");
    let engine = Engine::default();
    let child_module = WModule::new(&engine, &child_wasm).expect("child wasm validates");
    let mut store: Store<HostState> = Store::new(
        &engine,
        HostState {
            child_module: None,
            child: None,
            children: Vec::new(),
            saw_emit: false,
            saw_module: false,
        },
    );
    // Enough 64-KiB pages to back `[WIN_BASE, WIN_BASE + carve)`.
    let pages = ((WIN_BASE as u64 + (1u64 << carve_log2)) / 65536 + 1) as u32;
    let memory = Memory::new(&mut store, MemoryType::new(pages, None)).unwrap();
    memory
        .write(&mut store, CHILD_ENV_PTR as usize, &i64::MAX.to_le_bytes())
        .unwrap();
    let mut linker: Linker<HostState> = Linker::new(&engine);
    wire_nested_imports(&mut linker, memory);
    let inst = linker
        .instantiate(&mut store, &child_module)
        .unwrap()
        .start(&mut store)
        .unwrap();
    let f0 = inst.get_func(&store, "f0").expect("child f0 export");
    let mapped = inst
        .get_global(&store, "mapped")
        .expect("child mapped export");
    mapped
        .set(&mut store, Val::I64(1i64 << carve_log2))
        .unwrap();
    let mut r = [Val::I64(0)];
    f0.call(
        &mut store,
        &[Val::I32(WIN_BASE), Val::I32(CHILD_ENV_PTR)],
        &mut r,
    )?;
    Ok(r[0].i64().expect("child returns i64"))
}

#[test]
fn escape_through_the_op17_bounce_faults_and_stays_confined() {
    // The op-17 parent spawns `CHILD_ESCAPE` detached (it stores at 1040, past its `memory 10` window:
    // `mapped == 1024`). The out-of-window store must fault, and the fault must **propagate through the
    // bounce** to the parent's `f0` — the existing escape test runs the child directly; this one
    // exercises the confinement of a child driven by an emitted parent.
    let params = [
        Val::I32(WIN_BASE),
        Val::I32(ENV_PTR),
        Val::I32(7),
        Val::I32(CHILD_MODULE_HANDLE),
    ];
    let res = try_run_parent_over_child(&parent_op17(), CHILD_ESCAPE, &params, &[]);
    assert!(
        res.is_err(),
        "an out-of-window child access must fault the whole run through the op-17 bounce, not proceed"
    );
}

/// A child with a `>= 16 KiB` declared window (`memory 15` = 32 KiB) that stores at byte 8 — **inside**
/// the reserved NULL guard `[0, 16384)`. The emitted NULL-guard compare (#964/#1094) must trap it.
const CHILD_NULL_DEREF: &str = r#"memory 15
func () -> (i64) {
block 0 () {
  vaddr = i64.const 8
  vval = i64.const 77
  i64.store vaddr vval
  vr = i64.const 0
  return vr
  }
}
"#;

/// The same `memory 15` child storing **above** the guard (byte 16392) — a legal access that must run.
const CHILD_ABOVE_GUARD: &str = r#"memory 15
func () -> (i64) {
block 0 () {
  vaddr = i64.const 16392
  vsent = i64.const 55
  i64.store vaddr vsent
  vgot = i64.load vaddr
  return vgot
  }
}
"#;

#[test]
fn nested_child_gets_the_null_guard() {
    // A confined child with a `>= 16 KiB` window benefits from the trap-on-NULL guard (#964/#1094) just
    // like a root guest: the guard is an emitted compare installed on every child compile path, so a
    // low deref faults while an access above the guard runs. (A sub-16 KiB child declares no guard — the
    // `g <= win` gate — matching the interpreter, so the smaller children elsewhere in this file are
    // guard-free on both tiers.)
    assert!(
        run_child_direct(CHILD_NULL_DEREF, 15).is_err(),
        "a store inside `[0, 16384)` must trap the child's emitted NULL guard"
    );
    assert_eq!(
        run_child_direct(CHILD_ABOVE_GUARD, 15).expect("an access above the guard runs"),
        55,
        "the guard covers only `[0, 16384)`; `[16384, window)` is usable",
    );
}
