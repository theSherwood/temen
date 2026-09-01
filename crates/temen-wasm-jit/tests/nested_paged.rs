//! **§14 `unmap`/`protect` on the wasm-JIT nested tier under paged mode** (#1151) — a nested unit
//! that manages its own pages now *emits* instead of module-gating to emit-nothing. The page-op
//! `call.cap` rides the [`temen_wasm_jit::outline_nested_cap_calls`] outlined-leaf bounce (interp-
//! serviced through `env.call_interp`, exactly like `page_size`/`sub`), and the surrounding compute
//! emits **paged**: every emitted access consults the host-maintained byte-per-page state table (base
//! in the exported `"pagestate"` global) and traps `Ro`/`Unmapped` where the interpreter's
//! `check_prot` would.
//!
//! This closes the D40/§13 page-enforcement question on the *nested* axis (INVARIANTS #14); the paged
//! whole-program path already carries it (`module_uses_unmap_protect`, proven by
//! `crates/temen/tests/support/paged.rs`).
//!
//! **Scope of this file (Slice 1 — the emit change):** admission (a page-op nested unit emits under
//! the paged entry where the non-paged entry fails closed), the eligibility split (the page-op
//! wrapper stays an interp leaf, the compute entry emits), the `"pagestate"` export, and — through a
//! **real** `env.call_interp` bounce that remaps then refreshes the table from inside the handler
//! (the nested twin of `browser/tests/coop_tierup_driver.rs`'s bounce refresh) — the emitted nested
//! access honoring the fresh page state. The page map here is authored per the #750 driver contract
//! (as `paged.rs`'s `run_emitted` trusts `build_pagestate_table`); the emitted per-access check's
//! agreement with the interpreter oracle over that table is already fuzzed by `paged.rs` over the
//! identical emitted code. The full interp-oracle differential over a persistent shared-window driver
//! is Slice 2.

use temen_interp::bytecode::build_pagestate_table;
use temen_wasm_jit::{
    compile_module_nested, compile_module_nested_paged_with_eligibility, compile_nested_paged,
    outline_nested_cap_calls, DriveMode, TRAP_MEMORY_FAULT,
};
use wasmi::{Caller, Engine, Global, Linker, Memory, MemoryType, Module as WModule, Store, Val};

const WIN_BASE: i32 = 0x1_0000;
const ENV_PTR: i32 = 1024;
const PAGE: u64 = 4096;
const PAGE_LOG2: u8 = 12;
const WIN_LOG2: u32 = 17;
const WIN: u64 = 1 << WIN_LOG2; // `memory 17` — 128 KiB, 32 pages

/// A nested unit whose entry stores a sentinel at `ldaddr`, `unmap`s the page range
/// `[roff, roff+PAGE)` (ADDRESS_SPACE op 1 — an outlined leaf bounce), then loads `ldaddr`. When
/// `ldaddr`'s page was unmapped the emitted paged load must trap `MemoryFault`; otherwise it returns
/// the sentinel.
fn unmap_then_load_src(roff: u64, ldaddr: u64) -> String {
    format!(
        r#"memory 17
func (i32) -> (i64) {{
block 0 (vas: i32) {{
  vsent = i64.const 424242
  vsa = i64.const {ldaddr}
  i64.store vsa vsent
  voff = i64.const {roff}
  vlen = i64.const 4096
  vr = call.cap 5 1 (i64, i64) -> (i64) vas (voff, vlen)
  vld = i64.load vsa
  return vld
  }}
}}
"#
    )
}

/// A nested unit whose entry stores a sentinel at page 5, `protect`s that page read-only
/// (ADDRESS_SPACE op 2 — an outlined leaf bounce), then either writes it again (`write=true`, must
/// trap: a store to an `Ro` page) or reads it (`write=false`, must pass: a read of `Ro` is allowed).
fn protect_then_access_src(write: bool) -> String {
    let after = if write {
        "  vs2 = i64.const 777\n  i64.store vsa vs2\n  return vs2"
    } else {
        "  vld = i64.load vsa\n  return vld"
    };
    format!(
        r#"memory 17
func (i32) -> (i64) {{
block 0 (vas: i32) {{
  vsa = i64.const 20480
  vsent = i64.const 424242
  i64.store vsa vsent
  voff = i64.const 20480
  vlen = i64.const 4096
  vprot = i32.const 1
  vr = call.cap 5 2 (i64, i64, i32) -> (i64) vas (voff, vlen, vprot)
{after}
  }}
}}
"#
    )
}

fn parse(src: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// The wasmi driver's per-`Store` state: the outlined module (so `env.call_interp` can type the
/// wrapper's args), the live page-map model the servicer updates on an `unmap` bounce, the captured
/// `"pagestate"`/`"mapped"` globals (re-pointed from inside the handler, as `coop_tierup_driver`
/// does), and the emitted-trap sink.
struct DriverData {
    m: temen_ir::Module,
    mem: Option<Memory>,
    pagestate_global: Option<Global>,
    mapped_global: Option<Global>,
    /// Explicit page-state entries (`(page_base_off, kind)`; kind `2` = Unmapped) — the `Mem::map_info`
    /// encoding `build_pagestate_table` consumes. Grows as the guest unmaps.
    entries: Vec<(u64, u8)>,
    /// The emitted `env.trap(code)` sink (`0` = none yet).
    trap: i32,
    bounces: u32,
}

#[derive(Debug, PartialEq)]
enum Outcome {
    Val(i64),
    Trap(i32),
}

/// The page-state table lives immediately after the window in linear memory (the `paged.rs` convention).
fn table_base() -> usize {
    WIN_BASE as usize + WIN as usize
}

/// Rebuild the page-state table from `entries` and re-point `"pagestate"` + `"mapped"` via the
/// captured globals — the #750 driver refresh. Callable both from the pre-entry seed and from inside
/// the `env.call_interp` handler (which holds only a `Caller`, not the `Instance`, so the globals are
/// captured rather than looked up by name mid-call).
fn refresh_pagestate(mut ctx: impl wasmi::AsContextMut<Data = DriverData>) {
    let cx = ctx.as_context_mut();
    let entries = cx.data().entries.clone();
    let mem = cx.data().mem.unwrap();
    let pg = cx.data().pagestate_global.unwrap();
    let mg = cx.data().mapped_global.unwrap();
    let (table, cover) = build_pagestate_table(&(PAGE, WIN, WIN, entries));
    mem.write(&mut ctx, table_base(), &table).unwrap();
    pg.set(&mut ctx, Val::I32(table_base() as i32)).unwrap();
    mg.set(&mut ctx, Val::I64(cover as i64)).unwrap();
}

/// Compile `src` as a paged nested unit and run its entry `f0` under wasmi, servicing the outlined
/// `unmap` wrapper on the host (mark the range `Unmapped`, refresh the page-state table). Returns the
/// entry's outcome and the bounce count (non-vacuity).
fn run_paged_nested(src: &str) -> (Outcome, u32) {
    let mut m = parse(src);
    outline_nested_cap_calls(&mut m);
    let (wasm, eligible) = compile_module_nested_paged_with_eligibility(&m, false, PAGE_LOG2)
        .expect("paged nested emit");
    assert_eq!(
        eligible,
        vec![true, false],
        "paged nested: the entry emits, the unmap wrapper is a cross-tier leaf"
    );

    let engine = Engine::default();
    let module = WModule::new(&engine, &wasm).expect("paged nested wasm validates");
    let mut store: Store<DriverData> = Store::new(
        &engine,
        DriverData {
            m,
            mem: None,
            pagestate_global: None,
            mapped_global: None,
            entries: Vec::new(),
            trap: 0,
            bounces: 0,
        },
    );
    let need = table_base() + (WIN / PAGE) as usize;
    let pages = (need as u32).div_ceil(1 << 16) + 1;
    let memory = Memory::new(&mut store, MemoryType::new(pages, None)).unwrap();
    memory
        .write(&mut store, ENV_PTR as usize, &i64::MAX.to_le_bytes())
        .unwrap();
    store.data_mut().mem = Some(memory);

    let mut linker: Linker<DriverData> = Linker::new(&engine);
    linker.define("env", "memory", memory).unwrap();
    linker
        .func_wrap("env", "trap", |mut c: Caller<'_, DriverData>, code: i32| {
            c.data_mut().trap = code;
        })
        .unwrap();
    // The outlined `unmap` wrapper bounce: decode `(handle, off, len)` from the env scratch, mark the
    // page range Unmapped, write the wrapper's `0` result back, then refresh the page-state table so
    // the *following* emitted load reads the fresh state.
    let mem = memory;
    linker
        .func_wrap(
            "env",
            "call_interp",
            move |mut caller: Caller<'_, DriverData>, func: i32, args_ptr: i32| {
                // The wrapper's single instruction is the page-op cap-call; read its op to decide the
                // page kind it establishes (unmap ⇒ Unmapped, protect ⇒ Ro). `off`/`len` are the two
                // i64 args after the handle in every ADDRESS_SPACE page op.
                let op = match &caller.data().m.funcs[func as usize].blocks[0].insts[0] {
                    temen_ir::Inst::CapCall { op, .. } => *op,
                    other => panic!("wrapper body is a cap-call, got {other:?}"),
                };
                let data = mem.data(&caller);
                let slot = |i: usize| {
                    let o = args_ptr as usize + i * 8;
                    u64::from_le_bytes(data[o..o + 8].try_into().unwrap())
                };
                let off = slot(1);
                let len = slot(2);
                let kind = if op == 1 { 2 } else { 0 }; // 2 = Unmapped, 0 = Ro
                {
                    let st = caller.data_mut();
                    st.bounces += 1;
                    let mut p = off;
                    while p < off + len {
                        st.entries.push((p, kind));
                        p += PAGE;
                    }
                }
                // `unmap` returns 0 on success — write it into slot 0 (the result slot).
                let out = mem.data_mut(&mut caller);
                let o = args_ptr as usize;
                out[o..o + 8].copy_from_slice(&0u64.to_le_bytes());
                refresh_pagestate(&mut caller);
            },
        )
        .unwrap();
    // Unused nested imports (this unit spawns nothing / no threads / no futex).
    linker
        .func_wrap(
            "env",
            "instantiate",
            |_: Caller<'_, DriverData>,
             _w: i32,
             _i: i32,
             _e: i64,
             _o: i64,
             _s: i64,
             _q: i64|
             -> i32 { unreachable!("no instantiate in this unit") },
        )
        .unwrap();
    linker
        .func_wrap(
            "env",
            "join",
            |_: Caller<'_, DriverData>, _i: i32, _c: i32| -> i64 { unreachable!("no join") },
        )
        .unwrap();
    linker
        .func_wrap(
            "env",
            "thread_spawn",
            |_: Caller<'_, DriverData>, _f: i32, _sp: i64, _a: i64| -> i32 {
                unreachable!("no thread op")
            },
        )
        .unwrap();
    linker
        .func_wrap(
            "env",
            "thread_join",
            |_: Caller<'_, DriverData>, _h: i32| -> i64 { unreachable!("no thread op") },
        )
        .unwrap();
    linker
        .func_wrap(
            "env",
            "mem_wait",
            |_: Caller<'_, DriverData>, _w: i32, _a: i64, _e: i64, _t: i64, _is64: i32| -> i32 {
                unreachable!("no futex op")
            },
        )
        .unwrap();
    linker
        .func_wrap(
            "env",
            "mem_notify",
            |_: Caller<'_, DriverData>, _w: i32, _a: i64, _c: i32| -> i32 {
                unreachable!("no futex op")
            },
        )
        .unwrap();

    let instance = linker
        .instantiate(&mut store, &module)
        .unwrap()
        .start(&mut store)
        .unwrap();
    store.data_mut().pagestate_global = Some(
        instance
            .get_global(&store, "pagestate")
            .expect("paged nested module exports the page-state base global"),
    );
    store.data_mut().mapped_global = Some(
        instance
            .get_global(&store, "mapped")
            .expect("paged nested module exports the live-mapped global"),
    );
    // Seed the table for a fully-mapped window before entry (the per-call driver contract).
    refresh_pagestate(&mut store);

    let f0 = instance.get_func(&store, "f0").expect("f0 exported");
    let params = [Val::I32(WIN_BASE), Val::I32(ENV_PTR), Val::I32(1)];
    let mut results = [Val::I64(0)];
    let outcome = match f0.call(&mut store, &params, &mut results) {
        Ok(()) => Outcome::Val(results[0].i64().expect("i64")),
        Err(_) => Outcome::Trap(store.data().trap),
    };
    (outcome, store.data().bounces)
}

/// The trap case: unmap the page holding the load address, then load it — the emitted paged access
/// must fault exactly as the interpreter's `check_prot` would.
#[test]
fn nested_paged_unmap_then_load_traps() {
    // Load at 20480 (page 5, above the 16384 NULL guard); unmap that same page.
    let (out, bounces) = run_paged_nested(&unmap_then_load_src(20480, 20480));
    assert!(bounces >= 1, "the unmap wrapper must bounce, saw {bounces}");
    assert_eq!(
        out,
        Outcome::Trap(TRAP_MEMORY_FAULT),
        "an emitted load of an unmapped page must trap MemoryFault under paged nested mode"
    );
}

/// The control: unmap a *different* page, load a still-mapped one — the emitted access passes,
/// returning the sentinel. Proves the paged check is specific to the unmapped page, not a blanket
/// disable of the leaf.
#[test]
fn nested_paged_unmap_elsewhere_load_passes() {
    // Load at 20480 (page 5); unmap page 6 ([24576,28672)) — a different page.
    let (out, bounces) = run_paged_nested(&unmap_then_load_src(24576, 20480));
    assert!(bounces >= 1, "the unmap wrapper must bounce, saw {bounces}");
    assert_eq!(
        out,
        Outcome::Val(424242),
        "a load of a still-mapped page must return the stored sentinel"
    );
}

/// `protect` (op 2): a write to a page just protected read-only must trap under paged nested mode.
#[test]
fn nested_paged_protect_then_write_traps() {
    let (out, bounces) = run_paged_nested(&protect_then_access_src(true));
    assert!(
        bounces >= 1,
        "the protect wrapper must bounce, saw {bounces}"
    );
    assert_eq!(
        out,
        Outcome::Trap(TRAP_MEMORY_FAULT),
        "an emitted store to an Ro page must trap MemoryFault under paged nested mode"
    );
}

/// `protect` (op 2): a *read* of a read-only page must pass, returning the sentinel — the check is
/// write-vs-read aware, exactly as the interpreter's `check_prot`.
#[test]
fn nested_paged_protect_then_read_passes() {
    let (out, bounces) = run_paged_nested(&protect_then_access_src(false));
    assert!(
        bounces >= 1,
        "the protect wrapper must bounce, saw {bounces}"
    );
    assert_eq!(
        out,
        Outcome::Val(424242),
        "an emitted read of an Ro page must return the stored sentinel"
    );
}

/// The paged §14 front door ([`compile_nested_paged`]) emits a **WasmDriven** artifact for an
/// unmap/protect unit (where [`compile_nested`]'s mask-only path emits nothing), with the entry
/// eligible. This is the browser codegen path's routing target.
#[test]
fn compile_nested_paged_front_door_emits_wasm_driven() {
    let mut m = parse(&unmap_then_load_src(20480, 20480));
    outline_nested_cap_calls(&mut m);
    let a = compile_nested_paged(&m, false, PAGE_LOG2).expect("paged nested front door");
    assert_eq!(a.drive, DriveMode::WasmDriven { entry: 0 });
    assert_eq!(
        a.emitted,
        vec![true, false],
        "the entry emits paged; the outlined unmap wrapper is a cross-tier leaf"
    );
}

/// A `SharedRegion` aliasing op (iface 4) still fails closed to whole-interpreter even paged — a
/// `Backed` page's bytes live outside the window, which no trap check can honor.
#[test]
fn compile_nested_paged_shared_region_fails_closed() {
    // func 0: a SharedRegion `map` (iface 4 op 0) — the aliasing op paged mode cannot carry.
    let src = r#"memory 17
func (i32) -> (i64) {
block 0 (v0: i32) {
  vwo = i64.const 0
  vro = i64.const 0
  vlen = i64.const 4096
  vprot = i32.const 3
  vr = call.cap 4 0 (i64, i64, i64, i32) -> (i64) v0 (vwo, vro, vlen, vprot)
  return vr
  }
}
"#;
    let mut m = parse(src);
    outline_nested_cap_calls(&mut m);
    let a = compile_nested_paged(&m, false, PAGE_LOG2).expect("artifact");
    assert_eq!(a.drive, DriveMode::InterpDriven);
    assert!(
        a.emitted.iter().all(|&e| !e),
        "region-op unit emits nothing"
    );
}

/// The non-paged entry still fails closed on a page-op unit (the gate is only narrowed under paged).
#[test]
fn non_paged_nested_page_op_fails_closed() {
    let mut m = parse(&unmap_then_load_src(8192, 8192));
    outline_nested_cap_calls(&mut m);
    assert!(
        compile_module_nested(&m, false).is_err(),
        "the mask-only nested entry must still fail closed on unmap/protect"
    );
}
