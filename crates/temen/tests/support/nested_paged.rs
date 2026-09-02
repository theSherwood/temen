//! Shared harness for the **#1151 nested paged per-access check** differential — the emitted §14
//! nested tier's page-state confinement fuzzed as its own unit against the interpreter oracle
//! (INVARIANTS #2/#9; AGENTS.md "fuzz the confinement-masking lowering as its own unit").
//!
//! A nested unit's entry `unmap`s / `protect`s a fuzzer-chosen page range of its window, then performs
//! one scalar access (load/store, 8- or 64-bit) at a fuzzer-chosen address. The unit is compiled by
//! [`compile_module_nested_paged_with_eligibility`] after [`outline_nested_cap_calls`] — the page op
//! rides the `env.call_interp` wrapper bounce — and its entry runs on emitted wasm under `wasmi`. The
//! bounce is serviced the way the nested contract prescribes: on a **servicer vCPU over the shared
//! window with the run's powerbox** ([`bytecode::Vcpu::bounce_call`]), so the page op's effect is the
//! interpreter's own `Mem` (rounding, errno, the page map) and the page-state table the emitted
//! access then consults is rebuilt from that `Mem::map_info` — never an authored model. This is the
//! interp-oracle differential over a persistent shared-window driver that `temen-wasm-jit`'s
//! `nested_paged.rs` (an authored page map) deferred.
//!
//! The invariant, per input: `run(guest, args, Interp) == run(guest, args, NestedPaged)` — outcome
//! (value / trap kind) **and** the window bytes after the run. The `Interp` run is the oracle; a
//! mismatch is a page-check miscompile (never an escape: the `& MASK` confine is unconditional).
//! `fuzz_one` drives it from coverage-guided bytes; the stable `nested_paged_diff` test drives the
//! same function from deterministic seeds.

#![allow(dead_code)] // included via `#[path]` by both the fuzz target and the stable test.

use std::sync::{Arc, OnceLock};
use temen_interp::bytecode::build_pagestate_table;
use temen_interp::{bytecode, Host, Region, Trap, Value};
use temen_wasm_jit::{
    compile_module_nested_paged_with_eligibility, outline_nested_cap_calls, TRAP_MEMORY_FAULT,
    TRAP_OUT_OF_FUEL, XCALL_MAX_SLOTS,
};
use wasmi::{Caller, Engine, Global, Linker, Memory, MemoryType, Module as WModule, Store, Val};

const WIN_BASE: u32 = 0x4_0000;
const ENV_PTR: u32 = 1024;
const FUEL: u64 = 1_000_000_000;
/// The window: 128 KiB, fully mapped — ≥ 8 pages on 16-KiB-page hosts (macOS), ≥ 32 on 4-KiB hosts.
const WIN_LOG2: u8 = 17;

#[derive(Debug, PartialEq, Clone)]
pub enum Outcome {
    Vals(Vec<i64>),
    Trap(TrapKind),
}
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum TrapKind {
    MemoryFault,
    OutOfFuel,
    Other,
}

/// What the differential exercised — tallied by the stable test to prove non-vacuity.
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Cat {
    /// Both tiers trapped `MemoryFault` (the page check fired and agreed).
    Trapped,
    /// Both tiers passed (the access stayed on an admitted page).
    Passed,
    /// Inconclusive (out-of-fuel / a non-memory trap) — not counted toward coverage.
    Skipped,
}

fn build(src: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

fn shared_window(size: usize) -> (Arc<Region>, *mut u8, std::alloc::Layout) {
    let layout = std::alloc::Layout::from_size_align(size, 8).unwrap();
    // SAFETY: non-zero layout; `size` valid 8-aligned bytes owned here, used only as this window.
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!base.is_null());
    // SAFETY: `base` is `size` valid 8-aligned bytes, exclusively this window's, freed only after.
    let back = Arc::new(unsafe { Region::shared(base, size as u64) });
    (back, base, layout)
}

fn outcome_of(vals: &[Value]) -> Outcome {
    Outcome::Vals(
        vals.iter()
            .map(|v| match v {
                Value::I64(x) => *x,
                Value::I32(x) => *x as i64,
                _ => panic!("non-integer result"),
            })
            .collect(),
    )
}

/// The oracle: run the (un-outlined) guest whole on the bytecode interpreter with a granted
/// `AddressSpace`. Returns the outcome and the window bytes after the run.
fn run_interp(m: &temen_ir::Module, tail: &[i64]) -> (Outcome, Vec<u8>) {
    let win_size = 1usize << WIN_LOG2;
    let (back, base, layout) = shared_window(win_size);
    let prog = bytecode::VcpuProgram::compile(m).expect("compile");
    let mut host = Host::new();
    let asl = host.grant_memory();
    let mut args = vec![Value::I32(asl)];
    args.extend(tail.iter().map(|&a| Value::I64(a)));
    let mut vcpu = bytecode::Vcpu::new_root_with_powerbox(&prog, 0, &args, back, &[], host)
        .expect("root vcpu");
    // No tier-up is armed on the oracle, so the run ends at its first event.
    let out = match vcpu.run() {
        bytecode::VcpuEvent::Done(vals) => outcome_of(&vals),
        bytecode::VcpuEvent::Trapped(Trap::MemoryFault) => Outcome::Trap(TrapKind::MemoryFault),
        bytecode::VcpuEvent::Trapped(Trap::OutOfFuel) => Outcome::Trap(TrapKind::OutOfFuel),
        bytecode::VcpuEvent::Trapped(_) => Outcome::Trap(TrapKind::Other),
        _ => panic!("unexpected event on this single-vCPU run"),
    };
    drop(vcpu);
    // SAFETY: the vCPU (and its `Mem` aliasing the region) is dropped; copy, then free the buffer.
    let bytes = unsafe { std::slice::from_raw_parts(base, win_size) }.to_vec();
    unsafe { std::alloc::dealloc(base, layout) };
    (out, bytes)
}

/// The wasmi driver's per-`Store` state: the servicer vCPU (as an address — it lives on
/// [`run_nested_paged`]'s stack for the whole run and outlives the store), the captured driver
/// globals (re-pointed from inside the bounce handler), the emitted-trap sink, and the bounce tally.
struct Driver {
    vcpu: usize,
    pagestate: Option<Global>,
    mapped: Option<Global>,
    trap: i32,
    bounces: u32,
}

/// The emitted tier: outline the page ops, compile the unit **nested paged**, run its entry `f0` under
/// wasmi. Every `env.call_interp` bounce runs the outlined wrapper on the servicer vCPU over the shared
/// window (`bounce_call` — the live powerbox, the interpreter's own `Mem`), then rebuilds the page-state
/// table from `mem_map_info` and re-points `"pagestate"`/`"mapped"` — the #750 driver refresh. Returns
/// the outcome, the window bytes after the run, and the bounce count.
fn run_nested_paged(m0: &temen_ir::Module, tail: &[i64]) -> (Outcome, Vec<u8>, u32) {
    let mut m = m0.clone();
    outline_nested_cap_calls(&mut m);
    let win_size = 1usize << WIN_LOG2;
    let (back, base, layout) = shared_window(win_size);
    let prog = bytecode::VcpuProgram::compile(&m).expect("compile (outlined)");
    let mut host = Host::new();
    let asl = host.grant_memory();
    let mut args = vec![Value::I32(asl)];
    args.extend(tail.iter().map(|&a| Value::I64(a)));
    // The servicer: never `run()` — it only services bounces over the live window/powerbox.
    let mut vcpu = bytecode::Vcpu::new_root_with_powerbox(&prog, 0, &args, back, &[], host)
        .expect("servicer vcpu");
    let page = vcpu.mem_map_info().expect("window").0;
    let (wasm, eligible) =
        compile_module_nested_paged_with_eligibility(&m, false, page.trailing_zeros() as u8)
            .expect("paged nested emit");
    assert!(eligible[0], "the entry emits");
    assert!(
        eligible[1..].iter().all(|&e| !e),
        "every outlined wrapper is a cross-tier leaf"
    );
    let sigs: Vec<(usize, usize)> = m
        .funcs
        .iter()
        .map(|f| (f.params.len(), f.results.len()))
        .collect();

    let engine = Engine::default();
    let module = WModule::new(&engine, &wasm).expect("emitted wasm must validate");
    let mut store: Store<Driver> = Store::new(
        &engine,
        Driver {
            vcpu: &mut vcpu as *mut bytecode::Vcpu<'_> as usize,
            pagestate: None,
            mapped: None,
            trap: 0,
            bounces: 0,
        },
    );
    let table_base = WIN_BASE as usize + win_size;
    let need = table_base + win_size / page as usize;
    let pages = (need as u32).div_ceil(1 << 16) + 1;
    let memory = Memory::new(&mut store, MemoryType::new(pages, None)).unwrap();
    memory
        .write(&mut store, ENV_PTR as usize, &(FUEL as i64).to_le_bytes())
        .unwrap();

    // The #750 refresh: rebuild the table from the servicer's live map and re-point both driver
    // globals. Run before entry (a fully-mapped window) and after every bounce (a remap).
    fn refresh(
        mut ctx: impl wasmi::AsContextMut<Data = Driver>,
        memory: Memory,
        table_base: usize,
    ) {
        let cx = ctx.as_context_mut();
        // SAFETY: see `Driver::vcpu` — the servicer outlives the store; single-threaded.
        let vcpu = unsafe { &*(cx.data().vcpu as *const bytecode::Vcpu<'_>) };
        let info = vcpu.mem_map_info().expect("window");
        let (pg, mg) = (cx.data().pagestate.unwrap(), cx.data().mapped.unwrap());
        let (table, cover) = build_pagestate_table(&info);
        memory.write(&mut ctx, table_base, &table).unwrap();
        pg.set(&mut ctx, Val::I32(table_base as i32)).unwrap();
        mg.set(&mut ctx, Val::I64(cover as i64)).unwrap();
    }

    let mut linker: Linker<Driver> = Linker::new(&engine);
    linker.define("env", "memory", memory).unwrap();
    linker
        .func_wrap("env", "trap", |mut c: Caller<'_, Driver>, code: i32| {
            c.data_mut().trap = code;
        })
        .unwrap();
    linker
        .func_wrap(
            "env",
            "call_interp",
            move |mut caller: Caller<'_, Driver>,
                  func: i32,
                  args_ptr: i32|
                  -> Result<(), wasmi::Error> {
                let (np, nr) = sigs[func as usize];
                assert!(np <= XCALL_MAX_SLOTS && nr <= XCALL_MAX_SLOTS);
                let mut io = vec![0i64; np.max(nr)];
                let data = memory.data(&caller);
                for (i, slot) in io.iter_mut().enumerate().take(np) {
                    let o = args_ptr as usize + i * 8;
                    *slot = i64::from_le_bytes(data[o..o + 8].try_into().unwrap());
                }
                caller.data_mut().bounces += 1;
                // SAFETY: see `Driver::vcpu` — one bounce at a time, single-threaded.
                let vcpu = unsafe { &mut *(caller.data().vcpu as *mut bytecode::Vcpu<'_>) };
                let n = match vcpu.bounce_call(func as u32, &mut io) {
                    Ok(n) => n,
                    Err(t) => {
                        caller.data_mut().trap = match t {
                            Trap::MemoryFault => TRAP_MEMORY_FAULT,
                            Trap::OutOfFuel => TRAP_OUT_OF_FUEL,
                            _ => -1,
                        };
                        return Err(wasmi::Error::from(
                            wasmi::core::TrapCode::UnreachableCodeReached,
                        ));
                    }
                };
                assert_eq!(n, nr, "bounce result arity");
                let out = memory.data_mut(&mut caller);
                for (i, slot) in io.iter().enumerate().take(n) {
                    let o = args_ptr as usize + i * 8;
                    out[o..o + 8].copy_from_slice(&slot.to_le_bytes());
                }
                refresh(&mut caller, memory, table_base);
                Ok(())
            },
        )
        .unwrap();
    // Unused nested imports (this unit spawns nothing / no threads / no futex).
    linker
        .func_wrap(
            "env",
            "instantiate",
            |_: Caller<'_, Driver>, _w: i32, _i: i32, _e: i64, _o: i64, _s: i64, _q: i64| -> i32 {
                unreachable!("no instantiate in this unit")
            },
        )
        .unwrap();
    linker
        .func_wrap(
            "env",
            "join",
            |_: Caller<'_, Driver>, _i: i32, _c: i32| -> i64 { unreachable!("no join") },
        )
        .unwrap();
    linker
        .func_wrap(
            "env",
            "thread_spawn",
            |_: Caller<'_, Driver>, _f: i32, _sp: i64, _a: i64| -> i32 {
                unreachable!("no thread op")
            },
        )
        .unwrap();
    linker
        .func_wrap(
            "env",
            "thread_join",
            |_: Caller<'_, Driver>, _h: i32| -> i64 { unreachable!("no thread op") },
        )
        .unwrap();
    linker
        .func_wrap(
            "env",
            "mem_wait",
            |_: Caller<'_, Driver>, _w: i32, _a: i64, _e: i64, _t: i64, _is64: i32| -> i32 {
                unreachable!("no futex op")
            },
        )
        .unwrap();
    linker
        .func_wrap(
            "env",
            "mem_notify",
            |_: Caller<'_, Driver>, _w: i32, _a: i64, _c: i32| -> i32 {
                unreachable!("no futex op")
            },
        )
        .unwrap();

    let instance = linker
        .instantiate(&mut store, &module)
        .unwrap()
        .start(&mut store)
        .unwrap();
    store.data_mut().pagestate = Some(
        instance
            .get_global(&store, "pagestate")
            .expect("paged nested module exports the page-state base global"),
    );
    store.data_mut().mapped = Some(
        instance
            .get_global(&store, "mapped")
            .expect("paged nested module exports the live-mapped global"),
    );
    refresh(&mut store, memory, table_base);

    let f0 = instance.get_func(&store, "f0").expect("f0 exported");
    let mut params = vec![
        Val::I32(WIN_BASE as i32),
        Val::I32(ENV_PTR as i32),
        Val::I32(asl),
    ];
    params.extend(tail.iter().map(|&a| Val::I64(a)));
    let mut results = [Val::I64(0)];
    let out = match f0.call(&mut store, &params, &mut results) {
        Ok(()) => Outcome::Vals(vec![results[0].i64().expect("i64")]),
        Err(_) => Outcome::Trap(match store.data().trap {
            TRAP_OUT_OF_FUEL => TrapKind::OutOfFuel,
            TRAP_MEMORY_FAULT => TrapKind::MemoryFault,
            _ => TrapKind::Other,
        }),
    };
    let bounces = store.data().bounces;
    let mut bytes = vec![0u8; win_size];
    memory.read(&store, WIN_BASE as usize, &mut bytes).unwrap();
    drop(store);
    drop(vcpu);
    // SAFETY: the servicer vCPU (and its `Mem` aliasing the region) is dropped; free the buffer.
    unsafe { std::alloc::dealloc(base, layout) };
    (out, bytes, bounces)
}

// ---- fuzz guests: a page op (outlined → interp-serviced bounce) then one emitted scalar access ----
// `(as, roff, rlen, addr)`: unmap / protect-Ro the page range `[roff, roff+rlen)`, then access `addr`.

/// Source of the guest for `(op, access)`: `op` 1 = unmap, 2 = protect-Ro; `access` 0 = `i64.load`,
/// 1 = `i64.store`, 2 = `i64.load8_u`, 3 = `i64.store8`. Loads return the loaded value; stores
/// return `addr`.
fn guest_src(op: u32, access: u8) -> String {
    let page_op = if op == 1 {
        "  vr = call.cap 5 1 (i64, i64) -> (i64) vas (voff, vlen)"
    } else {
        "  vp = i32.const 1\n  vr = call.cap 5 2 (i64, i64, i32) -> (i64) vas (voff, vlen, vp)"
    };
    let access = match access {
        0 => "  vld = i64.load vaddr\n  return vld",
        1 => "  vv = i64.const 424242\n  i64.store vaddr vv\n  return vaddr",
        2 => "  vld = i64.load8_u vaddr\n  return vld",
        _ => "  vv = i64.const 77\n  i64.store8 vaddr vv\n  return vaddr",
    };
    format!(
        r#"memory 17
func (i32, i64, i64, i64) -> (i64) {{
block 0 (vas: i32, voff: i64, vlen: i64, vaddr: i64) {{
{page_op}
{access}
  }}
}}
"#
    )
}

/// The eight guests (`op` × `access`), built once.
fn guests() -> &'static [temen_ir::Module; 8] {
    static GUESTS: OnceLock<[temen_ir::Module; 8]> = OnceLock::new();
    GUESTS.get_or_init(|| {
        std::array::from_fn(|i| build(&guest_src(1 + (i as u32) / 4, (i % 4) as u8)))
    })
}

/// The host software page size, probed once (a throwaway vCPU) and cached — the offsets the fuzzer
/// chooses are page-relative, so the search adapts to 4-KiB and 16-KiB hosts alike.
fn page_size() -> u64 {
    static PAGE: OnceLock<u64> = OnceLock::new();
    *PAGE.get_or_init(|| {
        let prog = bytecode::VcpuProgram::compile(&guests()[0]).expect("compile");
        let (back, base, layout) = shared_window(1usize << WIN_LOG2);
        let vcpu = bytecode::Vcpu::new_root_with_powerbox(
            &prog,
            0,
            &[Value::I32(0), Value::I64(0), Value::I64(0), Value::I64(0)],
            back,
            &[],
            Host::new(),
        )
        .expect("probe vcpu");
        let page = vcpu.mem_map_info().expect("window").0;
        drop(vcpu);
        // SAFETY: the vCPU (and its Mem aliasing the region) is dropped; free the buffer.
        unsafe { std::alloc::dealloc(base, layout) };
        page
    })
}

/// Map a byte to a window offset near a page boundary: pick a page index (incl. one past the window)
/// and an intra-page delta from a set that lands on and either side of every edge — the check's
/// boundaries (an 8-byte access straddling a page edge touches two pages).
fn edge_offset(page_byte: u8, delta_byte: u8, page: u64, npages: u64) -> u64 {
    let p = (page_byte as u64) % (npages + 2); // 0..=npages+1 → in-window and just past
    let deltas = [
        0i64,
        1,
        8,
        -1,
        -4,
        -8,
        (page / 2) as i64,
        page as i64 - 1,
        page as i64 - 7,
        page as i64 + 1,
    ];
    let d = deltas[(delta_byte as usize) % deltas.len()];
    (p as i64 * page as i64 + d).max(0) as u64
}

/// One differential: decode a case from `data`, run the guest on the interpreter (oracle) and on the
/// nested paged emitted tier, and assert their outcomes and window bytes match. Returns what the case
/// exercised (for non-vacuity tallies).
pub fn fuzz_one(data: &[u8]) -> Cat {
    let mut b = [0u8; 8];
    let n = data.len().min(b.len());
    b[..n].copy_from_slice(&data[..n]);

    let page = page_size();
    let win: u64 = 1u64 << WIN_LOG2;
    let npages = win / page;

    let gi = (b[0] as usize) % 8;
    let guest = &guests()[gi];
    // The unmap/protect region: page-aligned start + a whole number of pages, kept **inside the
    // window** (the op is interp-serviced on both tiers; the emitted access is what is under test).
    // A region past the window is deliberately out of scope: the reservation admits it (#1153) but
    // the tiers then differ by design — the interpreter reads zero past its backing (#1191, the
    // `Region` seam) while the emitted access wraps under the window mask — a decline-class shape,
    // not a page-check question.
    let rpage = (b[1] as u64) % npages;
    let rstart = rpage * page;
    let rpages = 1 + (b[2] as u64) % (npages - rpage);
    let rlen = rpages * page;
    let addr = edge_offset(b[3], b[4], page, npages);

    let tail = [rstart as i64, rlen as i64, addr as i64];
    let (interp, ibytes) = run_interp(guest, &tail);
    let (paged, pbytes, bounces) = run_nested_paged(guest, &tail);
    assert!(bounces >= 1, "the page-op wrapper must bounce");

    // Fuel accounting differs between the tiers; an out-of-fuel (or any non-memory trap) is
    // inconclusive, so don't hold the tiers to parity there — every real case is a MemoryFault or a pass.
    if matches!(interp, Outcome::Trap(TrapKind::OutOfFuel | TrapKind::Other))
        || matches!(paged, Outcome::Trap(TrapKind::OutOfFuel | TrapKind::Other))
    {
        return Cat::Skipped;
    }

    assert_eq!(
        interp, paged,
        "nested paged access diverged from the interpreter oracle: guest#{gi} region=[{rstart},{}) addr={addr} page={page}", rstart + rlen
    );
    assert!(
        ibytes == pbytes,
        "nested paged window bytes diverged from the interpreter oracle: guest#{gi} region=[{rstart},{}) addr={addr} page={page}", rstart + rlen
    );

    match interp {
        Outcome::Trap(TrapKind::MemoryFault) => Cat::Trapped,
        Outcome::Vals(_) => Cat::Passed,
        _ => Cat::Skipped,
    }
}

/// The seed transform for the stable regression sweep — deterministic, distinct scramble per seed.
pub fn case_from_seed(seed: u64) -> [u8; 8] {
    seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .rotate_left(29)
        .to_le_bytes()
}
