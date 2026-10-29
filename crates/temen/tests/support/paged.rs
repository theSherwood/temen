//! Shared harness for the **#1081 paged bulk-memory per-page walk** differential — the confinement
//! hinge fuzzed as its own unit (INVARIANTS #2; AGENTS.md "fuzz the confinement-masking lowering as its
//! own unit"). A guest `unmap`s / `protect`s a page range of its window (interp-serviced, never emitted),
//! then a tiered-up leaf runs a `mem.fill` (write) or `mem.copy` (read src) over a fuzzer-chosen span.
//! The emitted per-page walk ([`temen_wasm_jit::emit_span_page_check`]) must trap `MemoryFault` at exactly
//! the pages the interpreter's `check_prot_span` does — an `Unmapped` page anywhere (read or write), a
//! non-`Rw` page for a write — for spans crossing page and window boundaries, at len 0/1/page±1.
//!
//! The invariant, per input: `run(guest, args, Interp) == run(guest, args, PagedSynced)`. The `Interp`
//! run is the oracle (INVARIANTS #9); a mismatch is a walk miscompile (a wrong page table is a trap-parity
//! divergence, never an escape — the `& MASK` confine is unconditional, #2). `fuzz_one` drives it from
//! coverage-guided bytes; the stable `paged_walk` test drives the same function from deterministic seeds.
//!
//! Every run is a pump run ([`CoopRun`]), the run every emitted tier-up goes through. The page-state
//! sweep (`fuzz_one_pagestate`) also runs each guest as a [`Mode::Leaf`]: emitted whole from its entry
//! as the run's root leaf, the way module cards, reactor ticks and warm evals run, its page ops bounced
//! out of the emitted code ([`CoopRun::bounce`]) and the table rebuilt after each bounce.

#![allow(dead_code)] // included via `#[path]` by both the fuzz target and the stable test.

use std::sync::Arc;
use temen_interp::bytecode::{CoopEvent, CoopRun, LeafOffer, ProgramEmit, TierUpConfig};
use temen_interp::{bytecode, host_page_size, Host, Region, Trap, Value};
use temen_wasm_jit::{
    compile_jit_page_checked, compile_module_tierup_paged, outline_cap_calls, DriveMode, Shape,
    TRAP_MEMORY_FAULT, TRAP_OUT_OF_FUEL,
};
use wasmi::core::ValType;
use wasmi::{
    AsContextMut, Caller, Engine, Global, Linker, Memory, MemoryType, Module as WModule, Store, Val,
};

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

/// How the guest runs.
#[derive(Clone, Copy, PartialEq)]
pub enum Mode {
    /// Pure interpreter: the oracle.
    Interp,
    /// The leaf tiers up, paged, with the table refreshed from the live map at each tier-up.
    PagedSynced,
    /// The whole guest runs emitted, paged, as the run's root leaf; its page ops bounce to the
    /// interpreter, and the table is refreshed after each bounce.
    Leaf,
}

/// What the differential exercised — tallied by the stable test to prove non-vacuity (the sweep must
/// actually reach tier-up and produce both trapping and passing spans, not silently skip everything).
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Cat {
    /// Both tiers trapped `MemoryFault` at the same span (the walk fired and agreed).
    Trapped,
    /// Both tiers passed (the span stayed on committed pages).
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

/// Run the emitted `f{func}(win, env, ...argv)` under wasmi over a memory mirrored from the live window,
/// with the page-state `table` placed after the window and both driver globals written. Copies emitted
/// writes back into the live buffer.
///
/// SAFETY: `base` is the live window buffer, touched only while the run is paused (single-threaded).
fn run_emitted(
    wasm: &[u8],
    func: u32,
    argv: &[i64],
    base: *mut u8,
    win_size: usize,
    table: &[u8],
    cover: u64,
) -> Outcome {
    let engine = Engine::default();
    let module = WModule::new(&engine, wasm).expect("emitted wasm must validate");
    let mut store: Store<i32> = Store::new(&engine, 0);
    let table_base = WIN_BASE as usize + win_size;
    let need = table_base + table.len();
    let pages = (need as u32).div_ceil(1 << 16) + 1;
    let memory = Memory::new(&mut store, MemoryType::new(pages, None)).unwrap();
    memory
        .write(&mut store, ENV_PTR as usize, &(FUEL as i64).to_le_bytes())
        .unwrap();
    // SAFETY: see fn doc.
    let live = unsafe { std::slice::from_raw_parts(base, win_size) };
    memory.write(&mut store, WIN_BASE as usize, live).unwrap();
    memory.write(&mut store, table_base, table).unwrap();

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
                unreachable!("no cross-tier call expected in these leaves");
            },
        )
        .unwrap();
    let instance = linker
        .instantiate(&mut store, &module)
        .unwrap()
        .start(&mut store)
        .unwrap();
    // The #750 driver contract: both globals written before the call.
    instance
        .get_global(&store, "mapped")
        .expect("paged module exports the live-mapped global")
        .set(&mut store, Val::I64(cover as i64))
        .unwrap();
    instance
        .get_global(&store, "pagestate")
        .expect("paged module exports the page-state base global")
        .set(&mut store, Val::I32(table_base as i32))
        .unwrap();
    let f = instance
        .get_func(&store, &format!("f{func}"))
        .unwrap_or_else(|| panic!("f{func} not exported"));

    let mut params = vec![Val::I32(WIN_BASE as i32), Val::I32(ENV_PTR as i32)];
    params.extend(argv.iter().map(|&a| Val::I64(a)));
    let mut results = vec![Val::I64(0)];

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
        Err(_) => Outcome::Trap(match *store.data() {
            TRAP_OUT_OF_FUEL => TrapKind::OutOfFuel,
            TRAP_MEMORY_FAULT => TrapKind::MemoryFault,
            _ => TrapKind::Other,
        }),
    };
    // SAFETY: see fn doc.
    let backs = unsafe { std::slice::from_raw_parts_mut(base, win_size) };
    let mut buf = vec![0u8; win_size];
    memory.read(&store, WIN_BASE as usize, &mut buf).unwrap();
    backs.copy_from_slice(&buf);
    outcome
}

/// Drive `guest(as, ..tail)` in `mode`, servicing a tier-up on emitted wasm with the page-check driver
/// contract (the table refreshed from the live map). `tail` is the guest's i64 arguments after the
/// implicit memory-cap handle. Returns the outcome.
fn run_guest_argv_bytes(guest_src: &str, tail: &[i64], mode: Mode) -> (Outcome, Vec<u8>) {
    let mut m = build(guest_src);
    let win_size = 1usize << WIN_LOG2;
    let (back, base, layout) = shared_window(win_size);
    let mut host = Host::new();
    let asl = host.grant_memory();
    let mut args = vec![Value::I32(asl)];
    args.extend(tail.iter().map(|&a| Value::I64(a)));

    let page_log2 = host_page_size().trailing_zeros() as u8;
    let (wasm, tierup) = match mode {
        Mode::Interp => (Vec::new(), None),
        Mode::PagedSynced => {
            let (wasm, eligible) = compile_module_tierup_paged(&m, false, page_log2).expect("emit");
            assert_eq!(
                eligible,
                vec![false, true],
                "paged mode: the page-op entry stays interpreted, the pure leaf emits"
            );
            let tierup = TierUpConfig {
                root: ProgramEmit {
                    eligible: Arc::from(eligible),
                    page_checked: true,
                    ..Default::default()
                },
                leaf: None,
                modules: None,
            };
            (wasm, Some(tierup))
        }
        Mode::Leaf => {
            // The page ops move into wrappers the emitted entry bounces to; the run is over the same
            // outlined module, so a bounce names a function the interpreter holds.
            outline_cap_calls(&mut m);
            let a = compile_jit_page_checked(&m, Shape::Batch { entry: 0 }, false, page_log2)
                .expect("emit");
            assert!(
                matches!(a.drive, DriveMode::WasmDriven { entry: 0 }),
                "leaf mode: the guest emits whole from its entry"
            );
            let tierup = TierUpConfig {
                root: ProgramEmit::default(),
                leaf: Some(Arc::new(|o: &LeafOffer| {
                    assert!(o.paged && !o.parks, "offered paged, and unable to park");
                    true
                })),
                modules: None,
            };
            (a.wasm, Some(tierup))
        }
    };
    let mut run = CoopRun::new_over(&m, 0, &args, FUEL, host, tierup, &[], WIN_LOG2, back)
        .expect("supported")
        .expect("entry in range");

    let mut tierups = 0;
    let out = loop {
        match run.run() {
            CoopEvent::Done(vals) => {
                break Outcome::Vals(
                    vals.iter()
                        .map(|v| match v {
                            Value::I64(x) => *x,
                            Value::I32(x) => *x as i64,
                            _ => panic!("non-integer result"),
                        })
                        .collect(),
                )
            }
            CoopEvent::Trapped(Trap::MemoryFault) => break Outcome::Trap(TrapKind::MemoryFault),
            CoopEvent::Trapped(Trap::OutOfFuel) => break Outcome::Trap(TrapKind::OutOfFuel),
            CoopEvent::Trapped(_) => break Outcome::Trap(TrapKind::Other),
            CoopEvent::TierUp {
                func, argv, mapped, ..
            } => {
                tierups += 1;
                let info = run.mem_map_info().expect("window");
                assert_eq!(mapped, info.2, "paged runs surface reserved");
                let (_, backed) = run.pending_win().expect("a flat window");
                let ended = match mode {
                    Mode::Leaf => {
                        assert_eq!(func, 0, "the root leaf tiers up at its entry");
                        run_leaf(&mut run, &m, &wasm, &argv, base, win_size)
                    }
                    _ => {
                        let (table, cover) = bytecode::build_pagestate_table(&info, backed);
                        run_emitted(&wasm, func, &argv, base, win_size, &table, cover)
                    }
                };
                match ended {
                    Outcome::Vals(v) => run.deliver_tierup(&v),
                    Outcome::Trap(TrapKind::OutOfFuel) => run.deliver_tierup_trap(Trap::OutOfFuel),
                    Outcome::Trap(_) => run.deliver_tierup_trap(Trap::MemoryFault),
                }
            }
            // One task over one window, serviced only for tier-up.
            CoopEvent::Paused
            | CoopEvent::Idle { .. }
            | CoopEvent::CapPark { .. }
            | CoopEvent::Resume { .. }
            | CoopEvent::JitInvoke { .. } => panic!("unexpected event on a one-task run"),
        }
    };
    drop(run);
    if mode == Mode::Leaf {
        assert_eq!(tierups, 1, "the guest ran as the root leaf");
    }
    // SAFETY: the run (and its `Mem` aliasing the region) is dropped; copy, then free the buffer.
    let bytes = unsafe { std::slice::from_raw_parts(base, win_size) }.to_vec();
    unsafe { std::alloc::dealloc(base, layout) };
    (out, bytes)
}

/// Per-`Store` state of a [`run_leaf`] call.
struct LeafCx {
    /// The run the emitted code bounces into, as an address (the store's data is `Send`). SAFETY:
    /// it is the `CoopRun` [`run_leaf`] borrows mutably for the whole call; the store lives inside
    /// that borrow, on one thread, and the run is reached only from one bounce at a time.
    run: usize,
    /// The live window's base address (the same discipline: the run is paused on this leaf).
    base: usize,
    /// Each function's `(params, results)` count, to marshal a bounce.
    sigs: Vec<(usize, usize)>,
    /// The emitted module's `"pagestate"` and `"mapped"` globals, refreshed after every bounce.
    pagestate: Option<Global>,
    mapped: Option<Global>,
    /// The code the emitted guard passed `env.trap` (or a bounce's trap), for the classification.
    trap: i32,
    bounces: u32,
}

/// The #750 refresh: rebuild the page-state table from the run's live map at `table_base` and
/// re-point both driver globals. Run before entry and after every bounce (a remap).
fn refresh_leaf(mut ctx: impl AsContextMut<Data = LeafCx>, memory: Memory, table_base: usize) {
    let cx = ctx.as_context_mut();
    // SAFETY: see `LeafCx::run`.
    let run = unsafe { &*(cx.data().run as *const CoopRun) };
    let info = run.mem_map_info().expect("window");
    let (_, backed) = run.pending_win().expect("a flat window");
    let (pg, mg) = (cx.data().pagestate.unwrap(), cx.data().mapped.unwrap());
    let (table, cover) = bytecode::build_pagestate_table(&info, backed);
    memory.write(&mut ctx, table_base, &table).unwrap();
    pg.set(&mut ctx, Val::I32(table_base as i32)).unwrap();
    mg.set(&mut ctx, Val::I64(cover as i64)).unwrap();
}

/// Run the emitted root leaf `f0(win, env, ...argv)` under wasmi over a memory mirrored from the live
/// window at `base`, serving each `env.call_interp` on `run` ([`CoopRun::bounce`]) as the browser host
/// does: the emitted writes reach the window before the bounce, the bounce's reach the mirror after
/// it, and the page-state table is rebuilt. Returns how the leaf ended, for the run's delivery.
fn run_leaf(
    run: &mut CoopRun,
    m: &temen_ir::Module,
    wasm: &[u8],
    argv: &[i64],
    base: *mut u8,
    win_size: usize,
) -> Outcome {
    let engine = Engine::default();
    let module = WModule::new(&engine, wasm).expect("emitted wasm must validate");
    let sigs = m
        .funcs
        .iter()
        .map(|f| (f.params.len(), f.results.len()))
        .collect();
    let cx = LeafCx {
        run: run as *mut CoopRun as usize,
        base: base as usize,
        sigs,
        pagestate: None,
        mapped: None,
        trap: 0,
        bounces: 0,
    };
    let mut store: Store<LeafCx> = Store::new(&engine, cx);
    // The table holds one byte per page, so a window's length bounds it.
    let table_base = WIN_BASE as usize + win_size;
    let pages = ((table_base + win_size) as u32).div_ceil(1 << 16) + 1;
    let memory = Memory::new(&mut store, MemoryType::new(pages, None)).unwrap();
    // SAFETY: `base` is the live window, touched here only while the run is paused on this leaf.
    let live = unsafe { std::slice::from_raw_parts(base, win_size) };
    memory.write(&mut store, WIN_BASE as usize, live).unwrap();

    let mut linker: Linker<LeafCx> = Linker::new(&engine);
    linker.define("env", "memory", memory).unwrap();
    linker
        .func_wrap("env", "trap", |mut c: Caller<'_, LeafCx>, code: i32| {
            c.data_mut().trap = code;
        })
        .unwrap();
    linker
        .func_wrap(
            "env",
            "call_interp",
            move |mut c: Caller<'_, LeafCx>,
                  func: i32,
                  args_ptr: i32|
                  -> Result<(), wasmi::Error> {
                c.data_mut().bounces += 1;
                let (np, nr) = c.data().sigs[func as usize];
                let mut io = vec![0i64; np.max(nr)];
                let data = memory.data(&c);
                for (i, slot) in io.iter_mut().enumerate().take(np) {
                    let o = args_ptr as usize + i * 8;
                    *slot = i64::from_le_bytes(data[o..o + 8].try_into().unwrap());
                }
                // SAFETY: see `LeafCx::base`.
                let window =
                    unsafe { std::slice::from_raw_parts_mut(c.data().base as *mut u8, win_size) };
                window.copy_from_slice(&data[WIN_BASE as usize..WIN_BASE as usize + win_size]);
                // SAFETY: see `LeafCx::run`.
                let run = unsafe { &mut *(c.data().run as *mut CoopRun) };
                let bounced = run.bounce(func as u32, &mut io, None);
                memory.write(&mut c, WIN_BASE as usize, window).unwrap();
                let n = match bounced {
                    Ok(Some(n)) => n,
                    Ok(None) => unreachable!("the guest cannot park"),
                    Err(t) => {
                        c.data_mut().trap = match t {
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
                let out = memory.data_mut(&mut c);
                for (i, slot) in io.iter().enumerate().take(n) {
                    let o = args_ptr as usize + i * 8;
                    out[o..o + 8].copy_from_slice(&slot.to_le_bytes());
                }
                refresh_leaf(&mut c, memory, table_base);
                Ok(())
            },
        )
        .unwrap();
    let instance = linker
        .instantiate(&mut store, &module)
        .unwrap()
        .start(&mut store)
        .unwrap();
    let global = |name: &str| {
        instance
            .get_global(&store, name)
            .unwrap_or_else(|| panic!("{name}"))
    };
    let [fuel, win, pagestate, mapped] = ["fuel", "win", "pagestate", "mapped"].map(global);
    store.data_mut().pagestate = Some(pagestate);
    store.data_mut().mapped = Some(mapped);
    fuel.set(&mut store, Val::I64(FUEL as i64)).unwrap();
    win.set(&mut store, Val::I32(WIN_BASE as i32)).unwrap();
    refresh_leaf(&mut store, memory, table_base);

    let f = instance.get_func(&store, "f0").expect("f0 exported");
    let ty = f.ty(&store);
    let mut params = vec![Val::I32(WIN_BASE as i32), Val::I32(ENV_PTR as i32)];
    for (t, &a) in ty.params()[2..].iter().zip(argv) {
        params.push(match t {
            ValType::I32 => Val::I32(a as i32),
            _ => Val::I64(a),
        });
    }
    let mut results: Vec<Val> = ty.results().iter().map(|_| Val::I64(0)).collect();
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
        Err(_) => Outcome::Trap(match store.data().trap {
            TRAP_OUT_OF_FUEL => TrapKind::OutOfFuel,
            TRAP_MEMORY_FAULT => TrapKind::MemoryFault,
            _ => TrapKind::Other,
        }),
    };
    // SAFETY: as above; the emitted code is done with the window.
    let window = unsafe { std::slice::from_raw_parts_mut(base, win_size) };
    memory.read(&store, WIN_BASE as usize, window).unwrap();
    assert!(store.data().bounces > 0, "the guest's page ops bounced");
    outcome
}

fn run_guest_argv(guest_src: &str, tail: &[i64], mode: Mode) -> Outcome {
    run_guest_argv_bytes(guest_src, tail, mode).0
}

// ---- fuzz guests: a page-op entry (interp-serviced) + a pure bulk-mem leaf (emitted under paged) ----
// `(as, roff, rlen, base, span)`: unmap/protect the page range `[roff, roff+rlen)`, then the leaf runs
// a `mem.fill` (write) or `mem.copy` (read the `base` span → a fixed Rw dst) of `span` bytes at `base`.

/// unmap `[roff, roff+rlen)`, then `mem.fill [base, base+span)` — the write walk over the unmap map.
const FILL_UNMAP: &str = r#"memory 17
func (i32, i64, i64, i64, i64) -> (i64) {
block 0 (vas: i32, voff: i64, vlen: i64, vbase: i64, vspan: i64) {
  vr = call.cap 5 1 (i64, i64) -> (i64) vas (voff, vlen)
  v1 = call 1 (vbase, vspan)
  return v1
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, vn: i64) {
  vval = i32.const 0
  mem.fill v0 vval vn
  return v0
  }
}
"#;

/// protect `[roff, roff+rlen)` read-only, then `mem.fill` — the write walk over the `Ro` map.
const FILL_PROTECT: &str = r#"memory 17
func (i32, i64, i64, i64, i64) -> (i64) {
block 0 (vas: i32, voff: i64, vlen: i64, vbase: i64, vspan: i64) {
  vp = i32.const 1
  vr = call.cap 5 2 (i64, i64, i32) -> (i64) vas (voff, vlen, vp)
  v1 = call 1 (vbase, vspan)
  return v1
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, vn: i64) {
  vval = i32.const 0
  mem.fill v0 vval vn
  return v0
  }
}
"#;

/// unmap `[roff, roff+rlen)`, then `mem.copy` reading `[base, base+span)` → a fixed Rw dst — the read walk.
const COPY_UNMAP: &str = r#"memory 17
func (i32, i64, i64, i64, i64) -> (i64) {
block 0 (vas: i32, voff: i64, vlen: i64, vbase: i64, vspan: i64) {
  vr = call.cap 5 1 (i64, i64) -> (i64) vas (voff, vlen)
  v1 = call 1 (vbase, vspan)
  return v1
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, vn: i64) {
  vdst = i64.const 1024
  mem.copy vdst v0 vn
  return v0
  }
}
"#;

/// protect `[roff, roff+rlen)` read-only, then `mem.copy` reading `base` (a read of an `Ro` page must
/// still succeed — only `Unmapped` traps a read; the dst write walk also runs, but parity is asserted).
const COPY_PROTECT: &str = r#"memory 17
func (i32, i64, i64, i64, i64) -> (i64) {
block 0 (vas: i32, voff: i64, vlen: i64, vbase: i64, vspan: i64) {
  vp = i32.const 1
  vr = call.cap 5 2 (i64, i64, i32) -> (i64) vas (voff, vlen, vp)
  v1 = call 1 (vbase, vspan)
  return v1
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, vn: i64) {
  vdst = i64.const 1024
  mem.copy vdst v0 vn
  return v0
  }
}
"#;

const GUESTS: [&str; 4] = [FILL_UNMAP, FILL_PROTECT, COPY_UNMAP, COPY_PROTECT];

/// Map a byte to a window offset near a page boundary: pick a page index (incl. one past the window) and
/// an intra-page delta from a set that lands on and either side of every edge — the walk's boundaries.
fn edge_offset(page_byte: u8, delta_byte: u8, page: u64, npages: u64) -> u64 {
    let p = (page_byte as u64) % (npages + 2); // 0..=npages+1 → in-window and just past
    let deltas = [
        0i64,
        1,
        8,
        -1,
        -8,
        (page / 2) as i64,
        page as i64 - 1,
        page as i64 + 1,
    ];
    let d = deltas[(delta_byte as usize) % deltas.len()];
    (p as i64 * page as i64 + d).max(0) as u64
}

/// Map a byte to a span length that stresses the walk: 0/1/8, page±1, and multi-page spans that cross
/// window boundaries — plus a raw value so the search is not confined to the structured set.
fn edge_span(sel: u8, raw: u8, page: u64) -> u64 {
    let set = [
        0,
        1,
        8,
        page - 1,
        page,
        page + 1,
        2 * page - 1,
        2 * page,
        2 * page + 1,
    ];
    if sel & 1 == 0 {
        set[(sel as usize >> 1) % set.len()]
    } else {
        // a raw length in [0, 3*page], reaching spans wider than several pages (and the window).
        (raw as u64 * page) / 64
    }
}

/// One differential: decode a case from `data`, run the leaf on the interpreter (oracle) and on the paged
/// emitted tier, and assert their outcomes match. Returns what the case exercised (for non-vacuity tallies).
pub fn fuzz_one(data: &[u8]) -> Cat {
    let mut b = [0u8; 8];
    let n = data.len().min(b.len());
    b[..n].copy_from_slice(&data[..n]);

    let page = host_page_size();
    let win: u64 = 1u64 << WIN_LOG2;
    let npages = win / page;

    let guest = GUESTS[(b[0] as usize) % GUESTS.len()];
    // The unmap/protect region: page-aligned start + a whole number of pages, so it carves clean
    // Unmapped/Ro page ranges the span can cross (the call.cap is interp-serviced on both tiers).
    let rstart = ((b[1] as u64) % (npages + 1)) * page;
    let rpages = (b[2] as u64) % (npages + 1);
    let rlen = rpages * page;
    let base = edge_offset(b[3], b[4], page, npages);
    let span = edge_span(b[5], b[6], page);

    let tail = [rstart as i64, rlen as i64, base as i64, span as i64];
    let interp = run_guest_argv(guest, &tail, Mode::Interp);
    let paged = run_guest_argv(guest, &tail, Mode::PagedSynced);

    // Fuel accounting differs between the tiers; an out-of-fuel (or any non-memory trap) is inconclusive
    // for the walk, so don't hold the tiers to parity there — every real case is a MemoryFault or a pass.
    if matches!(interp, Outcome::Trap(TrapKind::OutOfFuel | TrapKind::Other))
        || matches!(paged, Outcome::Trap(TrapKind::OutOfFuel | TrapKind::Other))
    {
        return Cat::Skipped;
    }

    assert_eq!(
        interp, paged,
        "paged bulk-mem walk diverged from the interpreter oracle: guest#{} region=[{rstart},{rlen}) base={base} span={span} page={page}",
        (b[0] as usize) % GUESTS.len()
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

// ---- #810: randomized page-state tables — three page ops, then one emitted access ----
// The `paged_walk` generator above carves ONE contiguous Unmapped/Ro range per case. Here the guest
// performs THREE fuzzer-chosen page ops (`unmap`, `protect` read-only, `protect` read-write, `map`
// read-write) over page-aligned in-window ranges, so the live map the driver's table is rebuilt from
// carries interleaved `Unmapped`/`Rw`/`Ro` runs, holes, and re-committed pages — and the leaf's
// fuzzer-placed access (scalar widths 1/2/4/8, a `v128`, or a bulk span) straddles every kind of state
// transition. Same invariant, same oracle, same harness; plus the window bytes must match.

/// The page-state leaf bodies over `(v0 = base, vn = span)`: scalar loads/stores per width, `v128`
/// load/store staged through an in-window scratch slot, the bulk write/read/move walks, and short
/// bulk ops of constant lengths (#2099: loads and stores, two page lookups a span) — a read, a write,
/// a fill, and a move whose spans overlap.
const PS_LEAVES: [&str; 17] = [
    "  vl = i64.load8_u v0\n  return vl",
    "  vl = i64.load16_u v0\n  return vl",
    "  vl = i64.load32_u v0\n  return vl",
    "  vl = i64.load v0\n  return vl",
    "  vv = i64.const 4242\n  i64.store8 v0 vv\n  vl = i64.load8_u v0\n  return vl",
    "  vv = i64.const 4242\n  i64.store16 v0 vv\n  vl = i64.load16_u v0\n  return vl",
    "  vv = i64.const 4242\n  i64.store32 v0 vv\n  vl = i64.load32_u v0\n  return vl",
    "  vv = i64.const 4242\n  i64.store v0 vv\n  vl = i64.load v0\n  return vl",
    "  vv = v128.load v0\n  vd = i64.const 20480\n  v128.store vd vv\n  vl = i64.load vd\n  return vl",
    "  vd = i64.const 20480\n  vv = v128.load vd\n  v128.store v0 vv\n  vl = i64.load v0\n  return vl",
    "  vval = i32.const 0\n  mem.fill v0 vval vn\n  return v0",
    "  vdst = i64.const 20480\n  mem.copy vdst v0 vn\n  return v0",
    "  vsrc = i64.const 20480\n  mem.move v0 vsrc vn\n  return v0",
    "  vdst = i64.const 20480\n  vn24 = i64.const 24\n  mem.copy vdst v0 vn24\n  return v0",
    "  vsrc = i64.const 20480\n  vn13 = i64.const 13\n  mem.move v0 vsrc vn13\n  return v0",
    "  vval = i32.const 421\n  vn32 = i64.const 32\n  mem.fill v0 vval vn32\n  return v0",
    "  vk = i64.const 5\n  vd = i64.add v0 vk\n  vn24 = i64.const 24\n  mem.move vd v0 vn24\n  return v0",
];

/// One page op over `[vo{i}, vo{i}+vl{i})`: `k % 4` — 0 `unmap`, 1 `protect` read-only, 2 `protect`
/// read-write (re-admits an `Ro` page), 3 `map` read-write (re-commits an `Unmapped` page).
fn ps_op(k: u8, i: usize) -> String {
    match k % 4 {
        0 => format!("  vr{i} = call.cap 5 1 (i64, i64) -> (i64) vas (vo{i}, vl{i})"),
        1 => format!(
            "  vp{i} = i32.const 1\n  vr{i} = call.cap 5 2 (i64, i64, i32) -> (i64) vas (vo{i}, vl{i}, vp{i})"
        ),
        2 => format!(
            "  vp{i} = i32.const 3\n  vr{i} = call.cap 5 2 (i64, i64, i32) -> (i64) vas (vo{i}, vl{i}, vp{i})"
        ),
        _ => format!(
            "  vp{i} = i32.const 3\n  vr{i} = call.cap 5 0 (i64, i64, i32) -> (i64) vas (vo{i}, vl{i}, vp{i})"
        ),
    }
}

/// The guest `(as, o0, l0, o1, l1, o2, l2, base, span)`: three page ops, then the leaf at `base`.
fn ps_guest_src(kinds: u8, leaf: usize) -> String {
    let ops = [ps_op(kinds, 0), ps_op(kinds >> 2, 1), ps_op(kinds >> 4, 2)].join("\n");
    let leaf = PS_LEAVES[leaf];
    format!(
        r#"memory 17
func (i32, i64, i64, i64, i64, i64, i64, i64, i64) -> (i64) {{
block 0 (vas: i32, vo0: i64, vl0: i64, vo1: i64, vl1: i64, vo2: i64, vl2: i64, vbase: i64, vspan: i64) {{
{ops}
  v1 = call 1 (vbase, vspan)
  return v1
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (v0: i64, vn: i64) {{
{leaf}
  }}
}}
"#
    )
}

/// One page-state differential: decode a case from `data` (three page ops + one access), run the guest
/// on the interpreter (oracle) and on the paged emitted tier, and assert outcomes and window bytes match.
pub fn fuzz_one_pagestate(data: &[u8]) -> Cat {
    let mut b = [0u8; 12];
    let n = data.len().min(b.len());
    b[..n].copy_from_slice(&data[..n]);

    let page = host_page_size();
    let win: u64 = 1u64 << WIN_LOG2;
    let npages = win / page;

    let leaf = (b[0] as usize) % PS_LEAVES.len();
    let kinds = b[1];
    // Three page-aligned ranges, each a non-empty whole number of pages kept **inside the window** (the
    // ops are interp-serviced on both tiers; the emitted access is what is under test — a range past
    // the window is the #1191 decline-class shape).
    let mut tail = Vec::with_capacity(8);
    for i in 0..3 {
        let rpage = (b[2 + 2 * i] as u64) % npages;
        let rpages = 1 + (b[3 + 2 * i] as u64) % (npages - rpage);
        tail.push((rpage * page) as i64);
        tail.push((rpages * page) as i64);
    }
    let base = edge_offset(b[8], b[9], page, npages);
    let span = edge_span(b[10], b[11], page);
    tail.push(base as i64);
    tail.push(span as i64);

    let src = ps_guest_src(kinds, leaf);
    let (interp, ibytes) = run_guest_argv_bytes(&src, &tail, Mode::Interp);
    if matches!(interp, Outcome::Trap(TrapKind::OutOfFuel | TrapKind::Other)) {
        return Cat::Skipped;
    }
    for mode in [Mode::PagedSynced, Mode::Leaf] {
        let (paged, pbytes) = run_guest_argv_bytes(&src, &tail, mode);
        if matches!(paged, Outcome::Trap(TrapKind::OutOfFuel | TrapKind::Other)) {
            return Cat::Skipped;
        }
        let tier = match mode {
            Mode::Leaf => "root leaf",
            _ => "leaf tier-up",
        };
        assert_eq!(
            interp, paged,
            "paged access ({tier}) over a randomized page map diverged from the interpreter oracle: leaf#{leaf} kinds={kinds:#x} ranges={:?} base={base} span={span} page={page}",
            &tail[..6]
        );
        assert!(
            ibytes == pbytes,
            "paged window bytes ({tier}) over a randomized page map diverged from the interpreter oracle: leaf#{leaf} kinds={kinds:#x} ranges={:?} base={base} span={span} page={page}",
            &tail[..6]
        );
    }

    match interp {
        Outcome::Trap(TrapKind::MemoryFault) => Cat::Trapped,
        Outcome::Vals(_) => Cat::Passed,
        _ => Cat::Skipped,
    }
}

/// The 12-byte seed transform for the page-state sweep — deterministic, distinct scramble per seed.
pub fn case_from_seed_pagestate(seed: u64) -> [u8; 12] {
    let a = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .rotate_left(29)
        .to_le_bytes();
    let c = seed
        .wrapping_mul(0xD6E8_FEB8_6659_FD93)
        .rotate_left(17)
        .to_le_bytes();
    let mut out = [0u8; 12];
    out[..8].copy_from_slice(&a);
    out[8..].copy_from_slice(&c[..4]);
    out
}
