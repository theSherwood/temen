//! Slice 0 of the wasm-AOT plan (`WASM_AOT.md`) — the missing native pin for **mainline
//! per-function tier-up over a live window**, the shape the JACL consumer's
//! `TEMEN_BROWSER_TIERUP_FINDINGS.md` postmortem hit (emitted `f{func}` trapped `MemoryFault →
//! unreachable` inside its own body, before any `env.call_interp`).
//!
//! Every existing tier-up test (`tierup.rs`) runs each emitted `f{func}` **in isolation** under
//! wasmi over a hand-seeded window; none drives the real loop — the interpreter owns the top frame,
//! materializes the guest window (incl. `m.data`), and *tiers up* a mainline `Call` to emitted code
//! running over that **same live window**. That loop exists only in the browser (`temen_par_run` +
//! `worker.js`), which is exactly why the path went unpinned. [`bytecode::VcpuReactor`] is its
//! faithful native vehicle (BROWSER.md § "wasm-JIT tier"): it opens over a caller-owned window,
//! runs `_start` once (materializing data), keeps the window live, and hands each
//! [`VcpuEvent::TierUp`] to a `service` closure — here that closure runs the emitted `f{func}` over
//! a wasmi memory mirrored from the live window, then copies any writes back.
//!
//! The differential contract is INVARIANTS.md #9: a tiered-up frame must be observably identical to
//! the interpreter-only frame (same result, same trap). A divergence here is the JACL fault.

use std::sync::{Arc, Mutex};
use temen_interp::{bytecode, Host, Region, Trap, Value};
use temen_wasm_jit::{compile_module_tierup, TRAP_MEMORY_FAULT, TRAP_OUT_OF_FUEL};
use wasmi::{Caller, Engine, Linker, Memory, MemoryType, Module as WModule, Store, Val};

const WIN_BASE: u32 = 0x1_0000;
const ENV_PTR: u32 = 1024;
const FUEL: u64 = 100_000_000;

#[derive(Debug, PartialEq, Clone)]
enum Outcome {
    Vals(Vec<i64>),
    Trap(TrapKind),
}
#[derive(Debug, PartialEq, Clone, Copy)]
enum TrapKind {
    MemoryFault,
    OutOfFuel,
    Other,
}

fn build(src: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// An 8-aligned zeroed window buffer + a `Region::shared` over it. The buffer is `size` bytes; the
/// caller keeps `base`/`layout` to read the live window at tier-up and to free it afterwards.
fn shared_window(size: usize) -> (Arc<Region>, *mut u8, std::alloc::Layout) {
    let layout = std::alloc::Layout::from_size_align(size, 8).unwrap();
    // SAFETY: non-zero layout; `size` valid 8-aligned bytes owned here, used only as this window.
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!base.is_null());
    // SAFETY: `base` is `size` valid 8-aligned bytes, exclusively this window's, freed only after.
    let back = Arc::new(unsafe { Region::shared(base, size as u64) });
    (back, base, layout)
}

/// Run the emitted `f{func}(win, env, ...argv)` under wasmi over a memory **mirrored from the live
/// window** `[base, base+win_size)`, then copy any writes back into it — modelling the browser's one
/// shared linear memory, where emitted code and the interpreter address the same `win`. Returns the
/// region's outcome the way the browser host classifies it (`env.trap` code, else the wasm trap).
///
/// SAFETY: `base` is the live window buffer, read/written only here while the vCPU is paused inside
/// the `service` callback (single-threaded), so no aliasing with the interpreter's own access.
fn run_emitted_over_live_window(
    m: &temen_ir::Module,
    wasm: &[u8],
    func: u32,
    argv: &[i64],
    base: *mut u8,
    win_size: usize,
    mapped: u64,
) -> Outcome {
    let engine = Engine::default();
    let module = WModule::new(&engine, wasm).expect("emitted wasm must validate");
    let mut store: Store<i32> = Store::new(&engine, 0);
    // Enough pages to hold the window mirrored at WIN_BASE plus the env cell below it.
    let need = WIN_BASE as usize + win_size;
    let pages = (need as u32).div_ceil(1 << 16) + 1;
    let memory = Memory::new(&mut store, MemoryType::new(pages, None)).unwrap();
    memory
        .write(&mut store, ENV_PTR as usize, &(FUEL as i64).to_le_bytes())
        .unwrap();
    // Mirror the live window into wasmi at WIN_BASE.
    // SAFETY: see fn doc — `base` is `win_size` valid bytes, vCPU paused.
    let live = unsafe { std::slice::from_raw_parts(base, win_size) };
    memory.write(&mut store, WIN_BASE as usize, live).unwrap();

    let mut linker: Linker<i32> = Linker::new(&engine);
    linker.define("env", "memory", memory).unwrap();
    linker
        .func_wrap("env", "trap", |mut c: Caller<'_, i32>, code: i32| {
            *c.data_mut() = code;
        })
        .unwrap();
    linker
        .func_wrap::<_, ()>(
            "env",
            "call_interp",
            |_: Caller<'_, i32>, _f: i32, _a: i32| {
                unreachable!("no cross-tier call expected in this leaf");
            },
        )
        .unwrap();
    let instance = linker
        .instantiate(&mut store, &module)
        .unwrap()
        .start(&mut store)
        .unwrap();
    // #717 host sync: write the event's committed extent to the emitted `"mapped"` global before
    // the call (a fully-mapped window makes this equal the emit-time default — the sync is a no-op
    // here, but it is the contract every tier-up host follows).
    instance
        .get_global(&store, "mapped")
        .expect("emitted module exports the live-mapped global")
        .set(&mut store, Val::I64(mapped as i64))
        .unwrap();
    let f = instance
        .get_func(&store, &format!("f{func}"))
        .unwrap_or_else(|| panic!("f{func} not exported"));

    let sig = &m.funcs[func as usize];
    let mut params = vec![Val::I32(WIN_BASE as i32), Val::I32(ENV_PTR as i32)];
    // All-i64 leaves only (what the browser tier-up ABI marshals).
    for (t, a) in sig.params.iter().zip(argv) {
        assert_eq!(*t, temen_ir::ValType::I64, "tier-up ABI marshals i64 slots");
        params.push(Val::I64(*a));
    }
    let mut results: Vec<Val> = sig.results.iter().map(|_| Val::I64(0)).collect();

    let outcome = match f.call(&mut store, &params, &mut results) {
        Ok(()) => Outcome::Vals(
            results
                .iter()
                .map(|v| match v {
                    Val::I64(x) => *x,
                    Val::I32(x) => *x as i64,
                    _ => panic!("non-integer result"),
                })
                .collect(),
        ),
        Err(e) => {
            let host_code = *store.data();
            let kind = if host_code == TRAP_OUT_OF_FUEL {
                TrapKind::OutOfFuel
            } else if host_code == TRAP_MEMORY_FAULT {
                TrapKind::MemoryFault
            } else {
                match e.as_trap_code() {
                    Some(wasmi::core::TrapCode::UnreachableCodeReached) => TrapKind::MemoryFault,
                    _ => TrapKind::Other,
                }
            };
            Outcome::Trap(kind)
        }
    };
    // Copy any emitted writes back into the live window (shared-memory model): `memory` is a Copy
    // handle, still valid against `store` after being defined in the linker.
    // SAFETY: see fn doc.
    let back = unsafe { std::slice::from_raw_parts_mut(base, win_size) };
    let mut buf = vec![0u8; win_size];
    memory.read(&store, WIN_BASE as usize, &mut buf).unwrap();
    back.copy_from_slice(&buf);
    outcome
}

/// Drive `entry(arg)` on a live window through [`VcpuReactor`], tiering up eligible callees onto the
/// emitted wasm mirrored over that window. Returns the mainline frame's outcome.
fn tiered_frame(
    m: &temen_ir::Module,
    wasm: &[u8],
    eligible: &[bool],
    entry: u32,
    arg: i64,
) -> Outcome {
    let win_size = 1usize << m.memory.expect("module declares memory").size_log2;
    let (back, base, layout) = shared_window(win_size);
    let host = Mutex::new(Host::new());
    let mut reactor = bytecode::VcpuReactor::open(m, back, &host, &[])
        .expect("reactor opens (_start runs, window + data materialized)")
        .with_jit_eligible(Arc::from(eligible.to_vec().into_boxed_slice()));

    let out = reactor.frame(
        entry,
        &[Value::I64(arg)],
        &host,
        |func, argv, mapped, _info| match run_emitted_over_live_window(
            m, wasm, func, argv, base, win_size, mapped,
        ) {
            Outcome::Vals(v) => Ok(v),
            Outcome::Trap(TrapKind::OutOfFuel) => Err(Trap::OutOfFuel),
            Outcome::Trap(_) => Err(Trap::MemoryFault),
        },
    );

    // SAFETY: reactor (and its `Mem` aliasing `back`) dropped above; free the window buffer.
    unsafe { std::alloc::dealloc(base, layout) };

    match out {
        Ok(v) => Outcome::Vals(v.iter().map(as_i64).collect()),
        Err(Trap::OutOfFuel) => Outcome::Trap(TrapKind::OutOfFuel),
        Err(Trap::MemoryFault) => Outcome::Trap(TrapKind::MemoryFault),
        Err(_) => Outcome::Trap(TrapKind::Other),
    }
}

/// Pure-interpreter oracle: the same `entry(arg)` with **no** tier-up (INVARIANTS.md #9 ground truth).
fn oracle(m: &temen_ir::Module, entry: u32, arg: i64) -> Outcome {
    let mut fuel = FUEL;
    match temen_interp::run(m, entry, &[Value::I64(arg)], &mut fuel) {
        Ok(v) => Outcome::Vals(v.iter().map(as_i64).collect()),
        Err(Trap::MemoryFault) => Outcome::Trap(TrapKind::MemoryFault),
        Err(Trap::OutOfFuel) => Outcome::Trap(TrapKind::OutOfFuel),
        Err(_) => Outcome::Trap(TrapKind::Other),
    }
}

fn as_i64(v: &Value) -> i64 {
    match v {
        Value::I64(x) => *x,
        Value::I32(x) => *x as i64,
        _ => panic!("unexpected non-integer result"),
    }
}

/// Mainline func 0 (`_start`, interpreted driver) calls the eligible leaf func 1, which **reads a
/// data segment** and adds `arg`. This is the minimal "mainline tier-up over a live window with a
/// data segment" shape. The data (`0x0102030405060708` at offset 16384, above the #1094 NULL guard
/// `[0, 16384)`) is materialized by the interpreter at `open`; the tiered-up `f1` must see it over the
/// same window.
const DATA_LEAF: &str = r#"
memory 16
data 16384 "\08\07\06\05\04\03\02\01"
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = call 1 (v0)
  return v1
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 16384
  v2 = i64.load v1
  v3 = i64.add v2 v0
  return v3
  }
}
"#;

#[test]
fn mainline_tierup_reads_data_segment_over_live_window() {
    let m = build(DATA_LEAF);
    assert!(!m.data.is_empty(), "guest must carry a data segment");
    let (wasm, eligible) = compile_module_tierup(&m, false).expect("tier-up emit");
    // func 1 is a pure i64 leaf → eligible + emitted; func 0 drives it on the interpreter.
    assert!(
        eligible[1],
        "the data-reading leaf must be tier-up eligible"
    );
    for arg in [0i64, 1, 100, -7, 1_000_000] {
        let want = oracle(&m, 0, arg);
        let got = tiered_frame(&m, &wasm, &eligible, 0, arg);
        assert_eq!(want, got, "tier-up vs interpreter diverged at arg {arg}");
    }
}

/// The leaf writes through the window then reads it back, and also reads the data segment — proving
/// the mirrored-window round-trip (emitted writes are visible) matches the interpreter.
const DATA_RW_LEAF: &str = r#"
memory 16
data 16384 "\08\07\06\05\04\03\02\01"
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = call 1 (v0)
  return v1
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 20480
  i64.store v1 v0
  v2 = i64.load v1
  v3 = i64.const 16384
  v4 = i64.load v3
  v5 = i64.add v2 v4
  return v5
  }
}
"#;

#[test]
fn mainline_tierup_write_then_read_over_live_window() {
    let m = build(DATA_RW_LEAF);
    let (wasm, eligible) = compile_module_tierup(&m, false).expect("tier-up emit");
    assert!(eligible[1], "the rw leaf must be tier-up eligible");
    for arg in [0i64, 1, 42, -1000] {
        let want = oracle(&m, 0, arg);
        let got = tiered_frame(&m, &wasm, &eligible, 0, arg);
        assert_eq!(want, got, "tier-up vs interpreter diverged at arg {arg}");
    }
}
