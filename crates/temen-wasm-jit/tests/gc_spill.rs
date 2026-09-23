//! **#1627 slice B — an emitted frame pushes its live values before a call that can reach
//! `gc.roots`.**
//!
//! #1546 vetoed every emit of a collecting module: an emitted frame's locals are invisible to the
//! op's scan, so a root held only there is missed. Spill mode (`gc_spill` on the B2 tier-up
//! entries) makes those locals visible instead. Around every call that can reach the op, the
//! emitted code pushes the values the call leaves live onto a spill stack named by the env cell,
//! and restores the cursor after it. The host then hands `[base, cursor)` to the bounce
//! (`CoopRun::bounce`'s `spill`, slice A).
//!
//! This test plays that host in wasmi. At each bounce it reads the spill region and checks it holds
//! **exactly** the words the frames beneath hold live: no fewer (that would be the #1546 hole) and
//! no dead ones, so the cost stays what #1627 measured.

use temen_wasm_jit::{
    compile_module_tierup_b2, ENV_SPILL_END_OFF, ENV_SPILL_SP_OFF, TRAP_SPILL_OVERFLOW,
};
use wasmi::{
    Caller, Engine, FuncRef, Linker, Memory, MemoryType, Module as WModule, Store, Table,
    TableType, Val,
};

const WIN: u32 = 0x1_0000;
const ENV_PTR: u32 = 1024;
/// The host's spill region, below the window and clear of the env cell.
const SPILL_BASE: u32 = 0x4000;
const SPILL_END: u32 = 0x8000;
const TABLE_LOG2: u32 = 4;

/// f0 holds `vx` across a call to the collector f1. f2 is interpreter-resident but cannot reach
/// the op; f3 holds `vy` across it, so it must push nothing. f4 holds `va` across an **emitted**
/// call into f0, and f5 holds `vb` across a `call.dyn` whose slot the test points at f0. Those two
/// are the nesting cases: at the collector's bounce, both frames' words must be present, in order.
const SRC: &str = r#"memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  vone = i64.const 1
  vx = i64.add v0 vone
  vr = call 1 (vx)
  vs = i64.add vr vx
  return vs
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vlo = i64.const 16384
  vhi = i64.const 32768
  vmask = i64.const -1
  vbuf = i64.const 20480
  vcap = i64.const 64
  vn = gc.roots vlo vhi vmask vbuf vcap
  return vn
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vas = i32.const 0
  voff = i64.const 65536
  vlen = i64.const 16384
  vprot = i32.const 3
  vr = call.cap 5 0 (i64, i64, i32) -> (i64) vas (voff, vlen, vprot)
  return vr
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vtwo = i64.const 2
  vy = i64.mul v0 vtwo
  vr = call 2 (vy)
  vs = i64.add vr vy
  return vs
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vk = i64.const 1000
  va = i64.add v0 vk
  vr = call 0 (v0)
  vs = i64.add vr va
  return vs
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vk = i64.const 2000
  vb = i64.add v0 vk
  vi = i32.const 0
  vr = call.dyn (i64) -> (i64) vi (v0)
  vs = i64.add vr vb
  return vs
  }
}
"#;

/// One bounce as the host saw it: the target, and the spill region's words at that moment.
type Bounce = (i32, Vec<u64>);

#[derive(Default)]
struct Host {
    mem: Option<Memory>,
    bounces: Vec<Bounce>,
    trap: Option<i32>,
}

fn module(text: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(text).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

fn read_u32(mem: &Memory, ctx: impl wasmi::AsContext, at: u32) -> u32 {
    let mut b = [0u8; 4];
    mem.read(ctx, at as usize, &mut b).unwrap();
    u32::from_le_bytes(b)
}

/// Run export `name` on the spill-mode emit of [`SRC`], with the spill region ending at `end`.
/// Returns the result (or the `env.trap` code) and every bounce, and checks the cursor is back at
/// the base after a clean return.
fn run(name: &str, arg: i64, end: u32) -> (Result<i64, i32>, Vec<Bounce>) {
    let (wasm, emitted) =
        compile_module_tierup_b2(&module(SRC), false, TABLE_LOG2, true).expect("emit");
    assert_eq!(
        emitted,
        [true, false, false, true, true, true],
        "spill mode emits every in-subset caller; the collector and the cap helper bounce"
    );
    let engine = Engine::default();
    let wmod = WModule::new(&engine, &wasm).expect("the spill-mode emit must validate");
    let mut store: Store<Host> = Store::new(&engine, Host::default());
    let mem = Memory::new(&mut store, MemoryType::new(3, None)).unwrap();
    store.data_mut().mem = Some(mem);
    let env = ENV_PTR as usize;
    mem.write(
        &mut store,
        env + ENV_SPILL_SP_OFF,
        &SPILL_BASE.to_le_bytes(),
    )
    .unwrap();
    mem.write(&mut store, env + ENV_SPILL_END_OFF, &end.to_le_bytes())
        .unwrap();

    let mut linker: Linker<Host> = Linker::new(&engine);
    linker.define("env", "memory", mem).unwrap();
    let tsize = 1u32 << TABLE_LOG2;
    let table = Table::new(
        &mut store,
        TableType::new(wasmi::core::ValType::FuncRef, tsize, Some(tsize)),
        Val::FuncRef(FuncRef::null()),
    )
    .unwrap();
    linker
        .define("env", "__indirect_function_table", table)
        .unwrap();
    linker
        .func_wrap("env", "trap", |mut c: Caller<'_, Host>, code: i32| {
            c.data_mut().trap = Some(code);
        })
        .unwrap();
    // The bounce: snapshot `[base, cursor)` as the slice-A servicer would pass it, and return 0.
    linker
        .func_wrap(
            "env",
            "call_interp",
            |mut c: Caller<'_, Host>, target: i32, args_ptr: i32| {
                let mem = c.data().mem.unwrap();
                let sp = read_u32(&mem, &c, ENV_PTR + ENV_SPILL_SP_OFF as u32);
                let mut bytes = vec![0u8; (sp - SPILL_BASE) as usize];
                mem.read(&c, SPILL_BASE as usize, &mut bytes).unwrap();
                let words = bytes
                    .chunks(8)
                    .map(|w| u64::from_le_bytes(w.try_into().unwrap()))
                    .collect();
                mem.write(&mut c, args_ptr as usize, &0i64.to_le_bytes())
                    .unwrap();
                c.data_mut().bounces.push((target, words));
            },
        )
        .unwrap();
    let inst = linker
        .instantiate(&mut store, &wmod)
        .unwrap()
        .start(&mut store)
        .unwrap();
    // Slot 0 → the emitted f0, as the host populates a B2 table with native funcrefs.
    let f0 = inst.get_func(&store, "f0").unwrap();
    table
        .set(&mut store, 0, Val::FuncRef(FuncRef::new(f0)))
        .unwrap();

    let f = inst.get_func(&store, name).expect("exported");
    let mut out = [Val::I64(0)];
    let res = f.call(
        &mut store,
        &[
            Val::I32(WIN as i32),
            Val::I32(ENV_PTR as i32),
            Val::I64(arg),
        ],
        &mut out,
    );
    let res = match res {
        Ok(()) => {
            assert_eq!(
                read_u32(&mem, &store, ENV_PTR + ENV_SPILL_SP_OFF as u32),
                SPILL_BASE,
                "{name}: every push is popped by the time the frame returns"
            );
            let Val::I64(v) = out[0] else {
                panic!("i64 result")
            };
            Ok(v)
        }
        Err(_) => Err(store.data().trap.expect("a trap names its code")),
    };
    (res, store.data().bounces.clone())
}

#[test]
fn a_frame_pushes_exactly_what_it_holds_across_the_collector() {
    let (res, bounces) = run("f0", 41, SPILL_END);
    // `v0` and `vone` die at the call, so only `vx` is pushed.
    assert_eq!(bounces, [(1, vec![42])]);
    assert_eq!(res, Ok(42), "the collector's 0 plus the restored `vx`");
}

#[test]
fn a_call_that_cannot_reach_the_collector_pushes_nothing() {
    let (res, bounces) = run("f3", 5, SPILL_END);
    assert_eq!(
        bounces,
        [(2, vec![])],
        "`vy` is live, but a bounce into f2 cannot run gc.roots"
    );
    assert_eq!(res, Ok(10));
}

#[test]
fn nested_emitted_frames_are_all_beneath_the_bounce() {
    let (res, bounces) = run("f4", 7, SPILL_END);
    assert_eq!(bounces, [(1, vec![1007, 8])], "f4's `va`, then f0's `vx`");
    assert_eq!(res, Ok(1007 + 8));
}

#[test]
fn an_indirect_call_pushes_before_dispatch() {
    let (res, bounces) = run("f5", 7, SPILL_END);
    assert_eq!(bounces, [(1, vec![2007, 8])], "f5's `vb`, then f0's `vx`");
    assert_eq!(res, Ok(2007 + 8));
}

#[test]
fn a_push_past_the_region_traps_instead_of_writing() {
    let (res, bounces) = run("f0", 41, SPILL_BASE + 4);
    assert_eq!(res, Err(TRAP_SPILL_OVERFLOW));
    assert!(bounces.is_empty(), "the trap precedes the call");
}

/// The opt-in is the only thing that lifts the veto, and it changes nothing for a module the veto
/// never applied to.
#[test]
fn spill_mode_is_opt_in_and_inert_without_gc_roots() {
    let m = module(SRC);
    let (_, vetoed) = compile_module_tierup_b2(&m, false, TABLE_LOG2, false).unwrap();
    assert!(vetoed.iter().all(|&e| !e), "#1546 still holds: {vetoed:?}");

    let plain = module(
        r#"memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  vi = i32.wrap_i64 v0
  vr = call.dyn (i64) -> (i64) vi (v0)
  vs = i64.add vr v0
  return vs
  }
}
"#,
    );
    assert_eq!(
        compile_module_tierup_b2(&plain, false, TABLE_LOG2, true).unwrap(),
        compile_module_tierup_b2(&plain, false, TABLE_LOG2, false).unwrap(),
        "no gc.roots, no spill code"
    );
}
