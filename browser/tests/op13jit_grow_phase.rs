//! **#1253 — a phase-scale `vm_map`-growing §14 op-13 child on the single-shot wasm-JIT tier, byte-
//! identical to the interpreter oracle, with NO pre-sized carve.** The op-13 servicer used to back a
//! phase child with a fixed `alloc_zeroed` carve pre-sized 8× its declared window (`(decl + 3).max(24)`,
//! doubled into its buddy parent) — for a ~256 MiB phase, a 4 GiB block inside the browser's 1 GiB
//! memory, so the big nim phases could not tier up. Now the child is granted the policy carve
//! (`nimc::PHASE_WINDOW_LOG2`'s buddy half, 256 MiB) and **commits only its declared window**: `"mapped"`
//! starts at `1 << DECL` and each `vm_map` bounce (an outlined `env.call_interp` leaf) grows it into the
//! carve — the #1243 growable-child model, whose confinement `nested_grow_window` fuzzes.
//!
//! This pins the *scale* the pre-size blocked: a child declaring 1 MiB grows to the full 256 MiB carve
//! in four `vm_map`s and touches the deepest page of every grown chunk. The wasmi driver plays
//! `driveJitRun` (`env.call_interp` → `run_cross_tier`, re-pointing `"mapped"` after each bounce); the
//! oracle is `new_confined_child_grow_over_host` over the same carve — the exact constructor the op-13
//! step's interpreter fallback uses. Value parity, four bounces, and a `"mapped"` high-water at the whole
//! carve are the non-vacuity proof.

use std::sync::Arc;

use temen_browser::JitOnrampRun;
use temen_interp::{bytecode, Host, Region, Value};
use wasmi::{Caller, Engine, Linker, Memory, MemoryType, Module as WModule, Store, Val};

const ENV_PTR: u32 = 1024;
const WIN_BASE: u32 = 0x1_0000;
/// The child's declared window: `memory 20` = 1 MiB — the initial `"mapped"`.
const DECL: u8 = 20;
/// The parent-granted carve: the policy buddy half, 256 MiB — the child grows into `[1 MiB, 256 MiB)`.
const CARVE: u8 = 28;
const MIB: u64 = 1 << 20;

/// The child entry `(inst, as) -> i64`: four `vm_map`s (op 0 on its starter `AddressSpace`, `Rw`)
/// covering `[1 MiB, 64)`, `[64, 128)`, `[128, 192)`, `[192, 256) MiB`, then a sentinel store on the
/// first grown bytes (1 MiB + 8) and on the **last page of every chunk**, loaded back and summed with the
/// four map results (`0` each on success). Expected `1000 + 2000 + 3000 + 4000 + 5000 = 15000`.
fn child_src() -> String {
    let page = 4096u64;
    let maps = [
        (MIB, 63 * MIB),
        (64 * MIB, 64 * MIB),
        (128 * MIB, 64 * MIB),
        (192 * MIB, 64 * MIB),
    ];
    let sentinels = [
        (MIB + 8, 1000),
        (64 * MIB - page + 8, 2000),
        (128 * MIB - page + 8, 3000),
        (192 * MIB - page + 8, 4000),
        (256 * MIB - page + 8, 5000),
    ];
    let mut body = String::from("  vas = i32.wrap_i64 v1\n  vprot = i32.const 3\n");
    for (i, (off, len)) in maps.iter().enumerate() {
        body.push_str(&format!(
            "  vo{i} = i64.const {off}\n  vl{i} = i64.const {len}\n  vm{i} = call.cap 5 0 (i64, i64, i32) -> (i64) vas (vo{i}, vl{i}, vprot)\n"
        ));
    }
    for (i, (addr, val)) in sentinels.iter().enumerate() {
        body.push_str(&format!(
            "  va{i} = i64.const {addr}\n  vs{i} = i64.const {val}\n  i64.store va{i} vs{i}\n"
        ));
    }
    // Single-assignment IR: fold the five loads and the four map results into `vsum`.
    for i in 0..sentinels.len() {
        body.push_str(&format!("  vd{i} = i64.load va{i}\n"));
    }
    body.push_str(
        "  vp0 = i64.add vd0 vd1\n  vp1 = i64.add vp0 vd2\n  vp2 = i64.add vp1 vd3\n  vp3 = i64.add vp2 vd4\n",
    );
    body.push_str(
        "  vq0 = i64.add vm0 vm1\n  vq1 = i64.add vq0 vm2\n  vq2 = i64.add vq1 vm3\n  vsum = i64.add vp3 vq2\n",
    );
    format!("memory {DECL}\nfunc (i64, i64) -> (i64) {{\nblock 0 (v0: i64, v1: i64) {{\n{body}  return vsum\n  }}\n}}\n")
}

fn build() -> temen_ir::Module {
    let m = temen_text::parse_module(&child_src()).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    assert!(
        !temen_wasm_jit::module_uses_unmap_protect(&m),
        "a map-only child emits mask-only (not paged)"
    );
    m
}

/// The interpreter oracle: the child as a **growable** confined child over the carve — declared `DECL`
/// committed, `vm_map`-growing into `1 << CARVE` — via the constructor the op-13 step's fallback uses.
fn oracle(m: &temen_ir::Module) -> i64 {
    let prog = bytecode::VcpuProgram::compile(m).expect("compile");
    let carve = 1usize << CARVE;
    let layout = std::alloc::Layout::from_size_align(carve, 8).unwrap();
    // SAFETY: non-zero 8-aligned layout; owned here until freed after the vCPU (and its `Mem`) drop.
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!base.is_null());
    // SAFETY: `base` is `carve` valid bytes, exclusively this child's window, freed only after the vCPU.
    let back = Arc::new(unsafe { Region::shared(base, carve as u64) });
    let out = {
        let mut vcpu = bytecode::Vcpu::new_confined_child_grow_over_host(
            &prog,
            0,
            0,
            Arc::clone(&back),
            DECL,
            CARVE,
            u64::MAX,
            Host::new(),
        )
        .expect("growable confined child builds");
        match vcpu.run() {
            bytecode::VcpuEvent::Done(v) => match v.first() {
                Some(Value::I64(x)) => *x,
                other => panic!("child returns one i64, got {other:?}"),
            },
            bytecode::VcpuEvent::Trapped(t) => panic!("the child trapped: {t:?}"),
            _ => panic!("unexpected confined-child event (expected Done)"),
        }
    };
    drop(back);
    // SAFETY: same layout; the vCPU and its region view are dropped, so no borrow outlives this.
    unsafe { std::alloc::dealloc(base, layout) };
    out
}

struct Drv {
    run: Option<JitOnrampRun>,
    bounces: u32,
}

/// Drive the run's emitted `f0` on wasmi — `driveJitRun`'s role: `env.call_interp` runs the outlined
/// leaf on the interpreter (`run_cross_tier`) and re-points `"mapped"` at the grown extent after each
/// bounce (`syncGlobals()`). Returns the value, the bounce count, and the final `"mapped"` high-water.
fn drive(
    engine: &Engine,
    mut store: Store<Drv>,
    memory: Memory,
    slots: Vec<Val>,
) -> (i64, u32, u64) {
    let emitted_wasm = store.data().run.as_ref().unwrap().emitted_wasm().to_vec();
    let module = WModule::new(engine, &emitted_wasm).expect("emitted module validates");
    let mut linker: Linker<Drv> = Linker::new(engine);
    linker.define("env", "memory", memory).unwrap();
    linker
        .func_wrap("env", "trap", |_: Caller<'_, Drv>, _code: i32| {})
        .unwrap();
    linker
        .func_wrap(
            "env",
            "call_interp",
            move |mut caller: Caller<'_, Drv>,
                  func: i32,
                  args_ptr: i32|
                  -> Result<(), wasmi::Error> {
                caller.data_mut().bounces += 1;
                let (params, results) = {
                    let r = caller.data().run.as_ref().unwrap();
                    let (p, rs) = r.func_sig(func as u32);
                    (p.to_vec(), rs.to_vec())
                };
                let args: Vec<Value> = {
                    let data = memory.data(&caller);
                    params
                        .iter()
                        .enumerate()
                        .map(|(i, t)| {
                            let o = args_ptr as usize + i * 8;
                            let raw = u64::from_le_bytes(data[o..o + 8].try_into().unwrap());
                            match t {
                                temen_ir::ValType::I32 => Value::I32(raw as i32),
                                _ => Value::I64(raw as i64),
                            }
                        })
                        .collect()
                };
                let outcome = caller
                    .data_mut()
                    .run
                    .as_mut()
                    .unwrap()
                    .run_cross_tier(func as u32, &args);
                match outcome {
                    Ok(vals) => {
                        // The per-bounce refresh: a `vm_map` grow advanced the run's committed extent —
                        // re-point the emitted `"mapped"` so the grown region admits.
                        let mapped = caller.data().run.as_ref().unwrap().mapped();
                        if let Some(wasmi::Extern::Global(g)) = caller.get_export("mapped") {
                            g.set(&mut caller, Val::I64(mapped as i64)).unwrap();
                        }
                        let data = memory.data_mut(&mut caller);
                        for (i, v) in vals.iter().enumerate().take(results.len()) {
                            let raw = match v {
                                Value::I32(x) => *x as u32 as u64,
                                Value::I64(x) => *x as u64,
                                _ => 0,
                            };
                            let o = args_ptr as usize + i * 8;
                            data[o..o + 8].copy_from_slice(&raw.to_le_bytes());
                        }
                        Ok(())
                    }
                    Err(_) => Err(wasmi::Error::from(
                        wasmi::core::TrapCode::UnreachableCodeReached,
                    )),
                }
            },
        )
        .unwrap();
    let instance = linker
        .instantiate(&mut store, &module)
        .unwrap()
        .start(&mut store)
        .unwrap();
    let mg = instance
        .get_global(&store, "mapped")
        .expect("mapped global");
    let f0 = instance.get_func(&store, "f0").expect("emitted f0");
    // The initial sync (`syncGlobals()` before `f0`): `"mapped"` at the declared window.
    let initial = store.data().run.as_ref().unwrap().mapped();
    mg.set(&mut store, Val::I64(initial as i64)).unwrap();
    memory
        .write(&mut store, ENV_PTR as usize, &(1i64 << 60).to_le_bytes())
        .unwrap();
    let mut params = vec![Val::I32(WIN_BASE as i32), Val::I32(ENV_PTR as i32)];
    params.extend(slots);
    let mut results = [Val::I64(0)];
    f0.call(&mut store, &params, &mut results)
        .expect("the emitted phase child completes");
    let value = results[0].i64().expect("i64 result");
    let run = store.data().run.as_ref().unwrap();
    assert!(!run.exited(), "a return, not an exit");
    (value, store.data().bounces, run.mapped())
}

#[test]
fn phase_scale_child_grows_to_the_full_carve_matching_the_interpreter() {
    let m = build();
    let want = oracle(&m);
    assert_eq!(
        want, 15000,
        "the interpreter grew the child to 256 MiB: every sentinel past the declared window landed"
    );

    let mut host = Host::new();
    let carve = 1u64 << CARVE;
    // The op-13 step's setup: the starter caps span the parent-granted CARVE (authorizing the grow),
    // while the run below commits only the child's declared window.
    let cinst = host.grant_instantiator(0, carve);
    let cas = host.grant_address_space(0, carve);
    let engine = Engine::default();
    let mut store: Store<Drv> = Store::new(
        &engine,
        Drv {
            run: None,
            bounces: 0,
        },
    );
    // Physically cover `[WIN_BASE, WIN_BASE + carve)` — the reserved (demand-zero) carve the browser
    // arena provides — so a non-faulting grown access lands in real memory.
    let pages = ((WIN_BASE as u64 + carve) as u32).div_ceil(1 << 16) + 1;
    let memory = Memory::new(&mut store, MemoryType::new(pages, Some(pages))).unwrap();
    // SAFETY: the child's window (its carve) lives inside the fixed wasmi memory for the run's lifetime.
    let win_ptr = unsafe {
        memory
            .data_mut(&mut store)
            .as_mut_ptr()
            .add(WIN_BASE as usize)
    };
    let run = unsafe {
        JitOnrampRun::open_shared_run_over_host(
            &m,
            win_ptr,
            carve,
            CARVE,
            false,
            host,
            Vec::new(),
            None,
            cinst as u64,
            cas as u64,
            None,
        )
    }
    .expect("a map-only op-13 child opens as a single-shot JIT run");
    assert_eq!(
        run.mapped(),
        1u64 << DECL,
        "the run commits the DECLARED window, not the carve — no pre-size"
    );
    let slots: Vec<Val> = run
        .slots()
        .iter()
        .map(|v| match v {
            Value::I64(x) => Val::I64(*x),
            Value::I32(x) => Val::I32(*x),
            _ => Val::I64(0),
        })
        .collect();
    assert_eq!(slots.len(), 2, "(inst, as) starter handles");
    store.data_mut().run = Some(run);

    let (got, bounces, high_water) = drive(&engine, store, memory, slots);
    assert_eq!(
        got, want,
        "emitted phase child diverged from the interpreter"
    );
    assert_eq!(bounces, 4, "the four outlined `vm_map` leaves bounced");
    assert_eq!(
        high_water, carve,
        "\"mapped\" grew to the whole 256 MiB carve — the scale the 8× pre-size could not back"
    );
}
