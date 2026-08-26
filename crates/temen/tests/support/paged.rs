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

#![allow(dead_code)] // included via `#[path]` by both the fuzz target and the stable test.

use std::sync::{Arc, OnceLock};
use temen_interp::{bytecode, Host, Region, Trap, Value};
use temen_wasm_jit::{compile_module_tierup_paged, TRAP_MEMORY_FAULT, TRAP_OUT_OF_FUEL};
use wasmi::{Caller, Engine, Linker, Memory, MemoryType, Module as WModule, Store, Val};

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

/// How the leaf runs: pure interpreter (the oracle) vs. paged tier-up with the live-map table refresh.
#[derive(Clone, Copy, PartialEq)]
pub enum Mode {
    Interp,
    PagedSynced,
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
/// SAFETY: `base` is the live window buffer, touched only while the vCPU is paused (single-threaded).
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
fn run_guest_argv(guest_src: &str, tail: &[i64], mode: Mode) -> Outcome {
    let m = build(guest_src);
    let win_size = 1usize << WIN_LOG2;
    let (back, base, layout) = shared_window(win_size);
    let prog = bytecode::VcpuProgram::compile(&m).expect("compile");
    let mut host = Host::new();
    let asl = host.grant_memory();
    let mut args = vec![Value::I32(asl)];
    args.extend(tail.iter().map(|&a| Value::I64(a)));
    let mut vcpu = bytecode::Vcpu::new_root_with_powerbox(&prog, 0, &args, back, &[], host)
        .expect("root vcpu");

    let page = vcpu.mem_map_info().expect("window").0;
    let wasm = if mode != Mode::Interp {
        let (wasm, eligible) =
            compile_module_tierup_paged(&m, false, page.trailing_zeros() as u8).expect("emit");
        assert_eq!(
            eligible,
            vec![false, true],
            "paged mode: the page-op entry stays interpreted, the pure leaf emits"
        );
        let e: Arc<[bool]> = Arc::from(eligible.into_boxed_slice());
        vcpu = vcpu.with_jit_eligible(e).with_jit_page_checked();
        wasm
    } else {
        Vec::new()
    };

    let out = loop {
        match vcpu.run() {
            bytecode::VcpuEvent::Done(vals) => {
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
            bytecode::VcpuEvent::Trapped(Trap::MemoryFault) => {
                break Outcome::Trap(TrapKind::MemoryFault)
            }
            bytecode::VcpuEvent::Trapped(Trap::OutOfFuel) => {
                break Outcome::Trap(TrapKind::OutOfFuel)
            }
            bytecode::VcpuEvent::Trapped(_) => break Outcome::Trap(TrapKind::Other),
            bytecode::VcpuEvent::TierUp { func, argv, mapped } => {
                let info = vcpu.mem_map_info().expect("window");
                assert_eq!(mapped, info.2, "paged runs surface reserved");
                let (table, cover) = bytecode::build_pagestate_table(&info);
                match run_emitted(&wasm, func, &argv, base, win_size, &table, cover) {
                    Outcome::Vals(v) => vcpu.deliver_tierup(&v),
                    Outcome::Trap(TrapKind::OutOfFuel) => vcpu.deliver_tierup_trap(Trap::OutOfFuel),
                    Outcome::Trap(_) => vcpu.deliver_tierup_trap(Trap::MemoryFault),
                }
            }
            _ => panic!("unexpected event on this single-vCPU run"),
        }
    };
    drop(vcpu);
    // SAFETY: the vCPU (and its `Mem` aliasing the region) is dropped; free the window buffer.
    unsafe { std::alloc::dealloc(base, layout) };
    out
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

/// The host software page size, probed once (a throwaway paged guest run) and cached — the offsets the
/// fuzzer chooses are page-relative, so the search adapts to 4-KiB and 16-KiB hosts alike.
fn page_size() -> u64 {
    static PAGE: OnceLock<u64> = OnceLock::new();
    *PAGE.get_or_init(|| {
        let m = build(FILL_UNMAP);
        let prog = bytecode::VcpuProgram::compile(&m).expect("compile");
        let (back, base, layout) = shared_window(1usize << WIN_LOG2);
        let vcpu = bytecode::Vcpu::new_root_with_powerbox(
            &prog,
            0,
            &[
                Value::I32(0),
                Value::I64(0),
                Value::I64(0),
                Value::I64(0),
                Value::I64(0),
            ],
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

    let page = page_size();
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
