//! **#1151 Slice 2c — a §14 granted unit that manages its own pages runs WasmDriven, paged, on the
//! browser's codegen path**, the Rust twin of `worker.js`'s confined `instCodegen` block over one wasmi
//! memory. The root `instantiate_module`s (op 5) the granted unit into a 128-KiB carve; the unit's
//! entry (emitted, **paged**) calls a helper that `unmap`s one carve page and `protect`s another
//! read-only (and, in one variant, `map`s the unmapped page back) — the helper is an out-of-subset
//! leaf, so it arrives as `env.call_interp` and runs on the **child's own vCPU over the carve**
//! (`temen_par_inst_call_interp` → `bounce_call`, the child's attenuated powerbox); the page-state
//! table is re-synced from that vCPU after the bounce, and the emitted accesses that follow honor it:
//! a read of the `Ro` page passes, a store to an `Rw` page passes, a store to the unmapped page traps,
//! a store to the re-`map`ped page passes. Differential against the cooperative interpreter running
//! the same root + unit (value or trap), with the table contents pinned as the non-vacuity proof.
//!
//! Twin-vs-browser difference: the browser's linear memory *is* the cdylib heap, so the table the
//! engine builds is addressed directly; here the table bytes are copied into the wasmi memory.

use std::sync::Mutex;

use temen_browser::{
    temen_par_child_confined, temen_par_compile, temen_par_deliver_handle, temen_par_deliver_join,
    temen_par_enable_inst_codegen, temen_par_ev_a, temen_par_ev_b, temen_par_ev_c, temen_par_ev_d,
    temen_par_free, temen_par_inst_call_interp, temen_par_inst_eligible, temen_par_inst_paged,
    temen_par_inst_pagestate_sync, temen_par_inst_unit_wasm_len, temen_par_inst_unit_wasm_ptr,
    temen_par_powerbox_inst, temen_par_root, temen_par_run, temen_par_tierup_argv_len,
    temen_par_tierup_argv_ptr, temen_par_tierup_pagestate_len, temen_par_tierup_pagestate_ptr,
    PAR_DONE, PAR_INSTANTIATE, PAR_JOIN, PAR_TRAP,
};
use temen_interp::{bytecode, host_page_size, Host, Value};
use wasmi::{Caller, Engine, Global, Linker, Memory, MemoryType, Module as WModule, Store, Val};

const FUEL: u64 = 10_000_000;
const ROOT_LOG2: u32 = 20;
const CARVE_OFF: u64 = 131072; // 128 KiB-aligned carve inside the 1 MiB root window
const CARVE_LOG2: u32 = 17;
/// The wasmi memory layout: the env cell below the window, the root window at `WIN_BASE`, the
/// page-state table copy right after it.
const ENV_PTR: u32 = 1024;
const WIN_BASE: u32 = 0x2_0000;
const TABLE_BASE: u32 = WIN_BASE + (1 << ROOT_LOG2);

// The codegen stash + its once-per-run memoization are process-global (see `par_tierup_driver.rs`).
static JIT_STATE_LOCK: Mutex<()> = Mutex::new(());

/// Root `(instantiator, module) -> i64`: `instantiate_module` the granted unit once into the carve,
/// join it, return its value.
fn root_src() -> String {
    format!(
        r#"memory {ROOT_LOG2}
func (i32, i32) -> (i64) {{
block 0 (vinst: i32, vmod: i32) {{
  vmod64 = i64.extend_i32_s vmod
  ventry = i64.const 0
  voff = i64.const {CARVE_OFF}
  vslog = i64.const {CARVE_LOG2}
  vquota = i64.const 0
  vh = call.cap 6 5 (i64, i64, i64, i64, i64) -> (i32) vinst (vmod64, ventry, voff, vslog, vquota)
  vr = call.cap 6 1 (i32) -> (i64) vinst (vh)
  return vr
  }}
}}
"#
    )
}

/// Which page the entry's store targets after the helper's page ops.
#[derive(Clone, Copy)]
enum Target {
    /// A page the helper left `Rw`: the store passes → `75 * 100 + 9`.
    Rw,
    /// The page the helper `unmap`ped: the store traps `MemoryFault`.
    Unmapped,
    /// The page the helper `unmap`ped then `map`ped back read-write: the store passes → `75 * 100 + 11`.
    Remapped,
}

/// The granted unit (`memory 17` — the carve). f0 `(inst, as) -> i64` (emitted, paged): calls the
/// helper, reads the byte at `Q` (the data segment "K" = 75, now on an `Ro` page), stores + loads a
/// marker at the target page, returns `75 * 100 + marker`. f1 `(as) -> i64` (the bounced leaf):
/// `unmap`s page `P`, `protect`s page `Q` read-only, and for `Remapped` `map`s `P` back.
fn unit_src(target: Target) -> String {
    let page = host_page_size();
    let (q, p, r) = (5 * page, 6 * page, 7 * page);
    let (t, marker) = match target {
        Target::Rw => (r, 9),
        Target::Unmapped => (p, 9),
        Target::Remapped => (p, 11),
    };
    let remap = if matches!(target, Target::Remapped) {
        "  vprot = i32.const 3\n  vr3 = call.cap 5 0 (i64, i64, i32) -> (i64) vas32 (vp, vpage, vprot)\n"
    } else {
        ""
    };
    format!(
        r#"memory {CARVE_LOG2}
data {q} "K"
func (i64, i64) -> (i64) {{
block 0 (vinst: i64, vas: i64) {{
  vr = call 1 (vas)
  vq = i64.const {q}
  vk = i64.load8_u vq
  vt = i64.const {t}
  vm = i64.const {marker}
  i64.store vt vm
  vld = i64.load vt
  vhundred = i64.const 100
  vkh = i64.mul vk vhundred
  vsum = i64.add vkh vld
  return vsum
  }}
}}
func (i64) -> (i64) {{
block 0 (vas: i64) {{
  vas32 = i32.wrap_i64 vas
  vpage = i64.const {page}
  vp = i64.const {p}
  vr1 = call.cap 5 1 (i64, i64) -> (i64) vas32 (vp, vpage)
  vq = i64.const {q}
  vro = i32.const 1
  vr2 = call.cap 5 2 (i64, i64, i32) -> (i64) vas32 (vq, vpage, vro)
{remap}  vsum = i64.add vr1 vr2
  return vsum
  }}
}}
"#
    )
}

fn build(src: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// The oracle: root + granted unit on the cooperative interpreter (in-engine op-5 spawn/join).
fn oracle(root: &temen_ir::Module, unit: &temen_ir::Module) -> Result<i64, ()> {
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, 1 << ROOT_LOG2);
    let modh = host.grant_module(unit);
    let mut run = bytecode::CoopRun::new(
        root,
        0,
        &[Value::I32(inst), Value::I32(modh)],
        FUEL,
        host,
        None,
    )
    .expect("supported")
    .expect("entry in range");
    match run.run() {
        bytecode::CoopEvent::Done(vals) => match vals.first() {
            Some(Value::I64(x)) => Ok(*x),
            other => panic!("non-i64 oracle result {other:?}"),
        },
        bytecode::CoopEvent::Trapped(_) => Err(()),
        other => panic!(
            "oracle did not run to completion: {:?}",
            core::mem::discriminant(&other)
        ),
    }
}

/// The wasmi driver's per-`Store` state for the emitted child run.
struct Drv {
    /// `*mut ParVcpu` as an address (the child vCPU, alive for the whole emitted run).
    child: usize,
    pagestate: Option<Global>,
    mapped: Option<Global>,
    trap: i32,
    bounces: u32,
    /// The page-state table as last synced (the non-vacuity pin).
    last_table: Vec<u8>,
}

/// The #750 driver refresh, twin-style: copy the engine-built table into the wasmi memory and
/// re-point both globals.
fn sync(mut ctx: impl wasmi::AsContextMut<Data = Drv>, memory: Memory) {
    let cx = ctx.as_context_mut();
    let child = cx.data().child as *mut temen_browser::ParVcpu;
    let (pg, mg) = (cx.data().pagestate.unwrap(), cx.data().mapped.unwrap());
    // SAFETY: the table lives in the child's `ParVcpu` until the next sync overwrites it.
    let table = unsafe {
        std::slice::from_raw_parts(
            temen_par_tierup_pagestate_ptr(child),
            temen_par_tierup_pagestate_len(child),
        )
    }
    .to_vec();
    let cover = temen_par_ev_b(child);
    memory.write(&mut ctx, TABLE_BASE as usize, &table).unwrap();
    pg.set(&mut ctx, Val::I32(TABLE_BASE as i32)).unwrap();
    mg.set(&mut ctx, Val::I64(cover)).unwrap();
    ctx.as_context_mut().data_mut().last_table = table;
}

struct Outcome {
    result: Result<i64, ()>,
    bounces: u32,
    table: Vec<u8>,
}

/// Drive root + unit the way `par.js` + `worker.js` do, single-threaded over one wasmi memory: the
/// root interprets; the confined child runs its emitted unit with `env.call_interp` serviced on the
/// child's own vCPU.
fn drive(root: &temen_ir::Module, unit: &temen_ir::Module) -> Outcome {
    let root_bytes = temen_encode::encode_module(root);
    let unit_bytes = temen_encode::encode_module(unit);
    assert_eq!(
        temen_par_powerbox_inst(1 << ROOT_LOG2, unit_bytes.as_ptr(), unit_bytes.len(), 0),
        1,
        "publish the §14 run recipe with the granted unit"
    );
    assert_eq!(temen_par_enable_inst_codegen(), 1, "the unit emits");
    assert_eq!(temen_par_inst_eligible(0), 1, "the entry is emitted");
    assert_eq!(
        temen_par_inst_eligible(1),
        0,
        "the page-op helper is a cross-tier leaf"
    );
    assert_eq!(temen_par_inst_paged(), 1, "a page-op unit emits paged");
    // The FFI emits the unit over the browser's SHARED linear memory, which wasmi 0.47 cannot
    // validate (no threads proposal), so run the same unit emitted non-shared.
    // SAFETY: the stash lives until the next run's emit.
    let stashed = unsafe {
        std::slice::from_raw_parts(
            temen_par_inst_unit_wasm_ptr(),
            temen_par_inst_unit_wasm_len(),
        )
    };
    let page_log2 = host_page_size().trailing_zeros() as u8;
    let temen_wasm_jit::Artifact {
        wasm: unit_wasm,
        emitted,
        drive,
    } = temen_wasm_jit::compile_nested_paged(unit, false, page_log2).expect("paged nested emit");
    assert!(matches!(
        drive,
        temen_wasm_jit::DriveMode::WasmDriven { entry: 0 }
    ));
    assert_eq!(emitted, vec![true, false]);
    // The shared import also carries a max limit (a few LEB bytes more); the emit is otherwise the
    // same code — the routing (`temen_par_inst_paged`, the eligibility split) is the FFI's, pinned above.
    assert!(
        stashed.len() > unit_wasm.len() && stashed.len() - unit_wasm.len() <= 8,
        "stash {} vs unshared {}",
        stashed.len(),
        unit_wasm.len()
    );
    let prog = temen_par_compile(root_bytes.as_ptr(), root_bytes.len());
    assert!(!prog.is_null(), "root program compiles");

    let engine = Engine::default();
    let module = WModule::new(&engine, &unit_wasm).expect("emitted unit validates");
    let mut store: Store<Drv> = Store::new(
        &engine,
        Drv {
            child: 0,
            pagestate: None,
            mapped: None,
            trap: 0,
            bounces: 0,
            last_table: Vec::new(),
        },
    );
    let total = TABLE_BASE as usize + (1usize << CARVE_LOG2) / host_page_size() as usize + 64;
    let pages = (total as u32).div_ceil(1 << 16) + 1;
    let memory = Memory::new(&mut store, MemoryType::new(pages, Some(pages))).unwrap();
    memory
        .write(&mut store, ENV_PTR as usize, &i64::MAX.to_le_bytes())
        .unwrap();
    // SAFETY: fixed-size memory ⇒ a stable data pointer; the root window `[WIN_BASE, +1 MiB)` lives
    // inside it and is used solely as this run's window (the browser's shared-linear-memory shape).
    let win_ptr = unsafe {
        memory
            .data_mut(&mut store)
            .as_mut_ptr()
            .add(WIN_BASE as usize)
    };
    let root_v = temen_par_root(prog, win_ptr, 1 << ROOT_LOG2, 0);
    assert!(!root_v.is_null(), "root vCPU builds");

    let mut linker: Linker<Drv> = Linker::new(&engine);
    linker.define("env", "memory", memory).unwrap();
    linker
        .func_wrap("env", "trap", |mut c: Caller<'_, Drv>, code: i32| {
            c.data_mut().trap = code;
        })
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
                let child = caller.data().child as *mut temen_browser::ParVcpu;
                // SAFETY: `args_ptr` is the env scratch inside the wasmi memory (stable pointer).
                let ap = unsafe {
                    memory
                        .data_mut(&mut caller)
                        .as_mut_ptr()
                        .add(args_ptr as usize)
                };
                if temen_par_inst_call_interp(child, func as u32, ap) != 0 {
                    return Err(wasmi::Error::from(
                        wasmi::core::TrapCode::UnreachableCodeReached,
                    ));
                }
                sync(&mut caller, memory);
                Ok(())
            },
        )
        .unwrap();
    linker
        .func_wrap(
            "env",
            "instantiate",
            |_: Caller<'_, Drv>, _w: i32, _i: i32, _e: i64, _o: i64, _s: i64, _q: i64| -> i32 {
                unreachable!("no nested instantiate in this unit")
            },
        )
        .unwrap();
    linker
        .func_wrap(
            "env",
            "join",
            |_: Caller<'_, Drv>, _i: i32, _c: i32| -> i64 { unreachable!("no join") },
        )
        .unwrap();
    linker
        .func_wrap(
            "env",
            "thread_spawn",
            |_: Caller<'_, Drv>, _f: i32, _sp: i64, _a: i64| -> i32 {
                unreachable!("no thread op")
            },
        )
        .unwrap();
    linker
        .func_wrap("env", "thread_join", |_: Caller<'_, Drv>, _h: i32| -> i64 {
            unreachable!("no thread op")
        })
        .unwrap();
    linker
        .func_wrap(
            "env",
            "mem_wait",
            |_: Caller<'_, Drv>, _w: i32, _a: i64, _e: i64, _t: i64, _is64: i32| -> i32 {
                unreachable!("no futex op")
            },
        )
        .unwrap();
    linker
        .func_wrap(
            "env",
            "mem_notify",
            |_: Caller<'_, Drv>, _w: i32, _a: i64, _c: i32| -> i32 { unreachable!("no futex op") },
        )
        .unwrap();

    let mut child_result: Option<Result<i64, ()>> = None;
    let result = loop {
        match temen_par_run(root_v) {
            PAR_DONE => break Ok(temen_par_ev_a(root_v)),
            PAR_TRAP => break Err(()),
            PAR_INSTANTIATE => {
                let am = temen_par_ev_a(root_v);
                let (smod, entry) = ((am >> 32) as u32, am as u32);
                let carve = temen_par_ev_b(root_v) as usize;
                let slog = temen_par_ev_c(root_v) as u32;
                assert_eq!((carve as u64, slog), (CARVE_OFF, CARVE_LOG2), "the carve");
                let cfuel = temen_par_ev_d(root_v);
                // SAFETY: the engine validated the carve lies inside the root window.
                let carve_ptr = unsafe { win_ptr.add(carve) };
                let child = temen_par_child_confined(prog, carve_ptr, slog, smod, entry, cfuel);
                assert!(!child.is_null(), "confined child vCPU builds");
                store.data_mut().child = child as usize;
                // The entry args: the child's starter cap handles, staged by the constructor.
                // SAFETY: the stash is stable until the child's next event (none on this path).
                let args = unsafe {
                    std::slice::from_raw_parts(
                        temen_par_tierup_argv_ptr(child),
                        temen_par_tierup_argv_len(child),
                    )
                }
                .to_vec();
                assert_eq!(args.len(), 2, "(inst, as) starter handles");

                let instance = linker
                    .instantiate(&mut store, &module)
                    .unwrap()
                    .start(&mut store)
                    .unwrap();
                store.data_mut().pagestate = Some(
                    instance
                        .get_global(&store, "pagestate")
                        .expect("paged unit"),
                );
                store.data_mut().mapped = Some(
                    instance
                        .get_global(&store, "mapped")
                        .expect("mapped global"),
                );
                temen_par_inst_pagestate_sync(child);
                sync(&mut store, memory);
                let f0 = instance.get_func(&store, "f0").expect("f0 exported");
                let params = [
                    Val::I32((WIN_BASE as usize + carve) as i32),
                    Val::I32(ENV_PTR as i32),
                    Val::I64(args[0]),
                    Val::I64(args[1]),
                ];
                let mut results = [Val::I64(0)];
                let r = match f0.call(&mut store, &params, &mut results) {
                    Ok(()) => Ok(results[0].i64().expect("i64")),
                    Err(_) => Err(()),
                };
                temen_par_free(child);
                child_result = Some(r);
                temen_par_deliver_handle(root_v, 0);
            }
            PAR_JOIN => {
                let r = child_result.expect("child ran before join");
                temen_par_deliver_join(root_v, r.unwrap_or(0), r.is_err() as i32);
            }
            ev => panic!("unexpected root event {ev}"),
        }
    };
    temen_par_free(root_v);
    Outcome {
        result,
        bounces: store.data().bounces,
        table: store.data().last_table.clone(),
    }
}

fn check(target: Target, want: Result<i64, ()>) {
    let _g = JIT_STATE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let root = build(&root_src());
    let unit = build(&unit_src(target));
    assert_eq!(oracle(&root, &unit), want, "oracle");
    let out = drive(&root, &unit);
    assert_eq!(
        out.result, want,
        "emitted paged child diverged from the interpreter oracle"
    );
    // Non-vacuity: the helper bounced exactly once, and the table the emitted accesses consulted
    // reflects its page ops (kinds: 0 = Unmapped, 1 = Rw, 2 = Ro).
    assert_eq!(out.bounces, 1, "the page-op helper bounces once");
    let (q, p) = (5usize, 6usize);
    assert_eq!(out.table[q], 2, "page Q protected read-only");
    let p_want = if matches!(target, Target::Remapped) {
        1
    } else {
        0
    };
    assert_eq!(out.table[p], p_want, "page P unmapped / re-mapped");
}

#[test]
fn paged_child_ro_read_and_rw_store_pass() {
    check(Target::Rw, Ok(75 * 100 + 9));
}

#[test]
fn paged_child_store_to_unmapped_page_traps() {
    check(Target::Unmapped, Err(()));
}

#[test]
fn paged_child_store_to_remapped_page_passes() {
    check(Target::Remapped, Ok(75 * 100 + 11));
}
