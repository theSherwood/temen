//! Shared harness for the **#810 live-`mapped` bound** differential — the mask-only wasm-JIT tier's
//! runtime-aware confinement fuzzed as its own unit against the interpreter oracle (INVARIANTS #2/#9;
//! AGENTS.md "fuzz the confinement-masking lowering as its own unit").
//!
//! Every emitted module bounds its accesses against the live `"mapped"` global (#717/#774/#784/#789 —
//! on by default in every emit; the drivers sync it from the tier-up event's committed-extent
//! snapshot). Until now the continuous fuzz targets exercised that check only at its emit-time default
//! (`wasm_diff` never grows), and `temen-wasm-jit/tests/live_mapped.rs` pins hand-picked extents. Here
//! the guest **shapes its own window** through its granted `AddressSpace` handle — two `vm_map` grows
//! into the reserved tail (contiguous, above a hole, or filling one; both interp-serviced on both tiers)
//! — and a pure leaf then performs one fuzzer-chosen access (scalar widths 1/2/4/8, a `v128`, an aligned atomic, or a bulk
//! span) at a fuzzer-chosen address near a page, window, or width boundary. The driver syncs `"mapped"`
//! from the [`bytecode::VcpuEvent::TierUp`] snapshot exactly as the browser Worker does.
//!
//! The invariant, per input: `run(guest, args, Interp) == run(guest, args, TierUp)` — outcome (value /
//! trap kind) **and** the window bytes after the run — and the emitted run never touches a byte past
//! the reservation (a canary placed there stays intact: the `& MASK` clamp is unconditional, so a wrong
//! bound is a trap-parity divergence, never an escape). A window state the single bound cannot
//! represent (a hole below the high-water) makes the vCPU **decline** tier-up (`Mem::scalar_extent`
//! has no answer) and interpret the leaf — tallied separately so the sweep proves that arm fires too.
//! Only grows: a module that `unmap`s/`protect`s is page-managing and the mask-only tier emits nothing
//! for it (#750's gate — the paged tier and the `pagestate` target own that surface), so a live bound
//! *below* the emit-time default is unreachable through the tier-up contract; the check's downward
//! arithmetic is pinned by hand in `temen-wasm-jit/tests/live_mapped.rs` (`shrink_faults_below_declared`).
//! `fuzz_one` drives it from coverage-guided bytes; the stable `live_mapped_diff` test drives the same
//! function from deterministic seeds.

#![allow(dead_code)] // included via `#[path]` by both the fuzz target and the stable test.

use std::sync::{Arc, OnceLock};
use temen_interp::{bytecode, Host, Region, Trap, Value};
use temen_wasm_jit::{compile_module_tierup, TRAP_MEMORY_FAULT, TRAP_OUT_OF_FUEL};
use wasmi::{Caller, Engine, Linker, Memory, MemoryType, Module as WModule, Store, Val};

const WIN_BASE: u32 = 0x4_0000;
const ENV_PTR: u32 = 1024;
const FUEL: u64 = 1_000_000_000;
/// The declared window (`memory 16`, 64 KiB) — the emit-time default of the live bound.
const DECLARED_LOG2: u8 = 16;
/// The reservation the guest may grow into: 256 KiB — 12 tail pages on 16-KiB hosts, 48 on 4-KiB.
const RESERVED_LOG2: u8 = 18;
/// Canary bytes placed right after the reservation in the emitted run's linear memory: the escape
/// property (INVARIANTS #2) is that no emitted access lands there, whatever the live bound says.
const CANARY: usize = 1 << 16;
const CANARY_BYTE: u8 = 0xA5;
/// An in-window scratch slot (above the #1094 NULL guard `[0, 16384)`) the `v128` leaves stage through.
const SCRATCH: u64 = 20480;

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
    /// Both tiers trapped `MemoryFault` (the live bound fired and agreed with the page map).
    Trapped,
    /// Both tiers passed (the access stayed within the committed extent).
    Passed,
    /// The vCPU declined tier-up (an unrepresentable window state) and interpreted the leaf; parity
    /// held trivially — counted so the sweep proves the fail-closed arm is reached.
    Declined,
    /// Inconclusive (out-of-fuel / a non-memory trap) — not counted toward coverage.
    Skipped,
}

/// The entry `(handle, goff, glen, goff2, glen2, a0, a1, a2)`: `vm_map` `[goff, goff+glen)` read-write,
/// then `vm_map` `[goff2, goff2+glen2)` (grows into the reserved tail — contiguous, above a hole, or
/// filling one; a zero length is a no-op / errno), then call the pure leaf `(a0, a1, a2)`. Both
/// capability calls are interp-serviced on both tiers; the leaf is what tiers up.
const ENTRY: &str = r#"memory 16
func (i64, i64, i64, i64, i64, i64, i64, i64) -> (i64) {
block 0 (vh: i64, vgoff: i64, vglen: i64, vgoff2: i64, vglen2: i64, va0: i64, va1: i64, va2: i64) {
  vas = i32.wrap_i64 vh
  vprot = i32.const 3
  vr0 = call.cap 5 0 (i64, i64, i32) -> (i64) vas (vgoff, vglen, vprot)
  vr1 = call.cap 5 0 (i64, i64, i32) -> (i64) vas (vgoff2, vglen2, vprot)
  v2 = call 1 (va0, va1, va2)
  return v2
  }
}
func (i64, i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64, v2: i64) {
"#;

/// The leaf bodies, over `(v0 = address, v1 = value | source, v2 = length | replacement)`. Stores read
/// their value back so a "passing" store is one that really reached the window (the bytes comparison
/// covers the rest). Grouped: scalar loads (widths 1/2/4/8), scalar stores, `v128` load/store staged
/// through [`SCRATCH`], aligned atomics, bulk spans.
const LEAVES: [&str; 18] = [
    "  vl = i64.load8_u v0\n  return vl",
    "  vl = i64.load16_u v0\n  return vl",
    "  vl = i64.load32_u v0\n  return vl",
    "  vl = i64.load v0\n  return vl",
    "  i64.store8 v0 v1\n  vl = i64.load8_u v0\n  return vl",
    "  i64.store16 v0 v1\n  vl = i64.load16_u v0\n  return vl",
    "  i64.store32 v0 v1\n  vl = i64.load32_u v0\n  return vl",
    "  i64.store v0 v1\n  vl = i64.load v0\n  return vl",
    "  vv = v128.load v0\n  vd = i64.const 20480\n  v128.store vd vv\n  vl = i64.load vd\n  return vl",
    "  vd = i64.const 20480\n  i64.store vd v1\n  vv = v128.load vd\n  v128.store v0 vv\n  vl = i64.load v0\n  return vl",
    "  vl = i64.atomic.load v0\n  return vl",
    "  i64.atomic.store v0 v1\n  vl = i64.atomic.load v0\n  return vl",
    "  vl = i64.atomic.rmw.add v0 v1\n  return vl",
    "  vl = i64.atomic.cmpxchg v0 v1 v2\n  return vl",
    "  vx = i32.wrap_i64 v1\n  vr = i32.atomic.rmw.add v0 vx\n  vl = i64.extend_i32_u vr\n  return vl",
    "  vv = i32.wrap_i64 v1\n  mem.fill v0 vv v2\n  return v0",
    "  mem.copy v0 v1 v2\n  return v0",
    "  mem.move v0 v1 v2\n  return v0",
];

/// The access width of leaf `gi` (for the atomics' alignment; bulk spans are width-free).
fn leaf_width(gi: usize) -> u64 {
    match gi {
        0 | 4 => 1,
        1 | 5 => 2,
        2 | 6 | 14 => 4,
        8 | 9 => 16,
        15..=17 => 0,
        _ => 8,
    }
}
fn is_atomic(gi: usize) -> bool {
    (10..=14).contains(&gi)
}
fn is_bulk(gi: usize) -> bool {
    gi >= 15
}

fn build(src: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// The emit input: the guest with `size_log2` bumped to the reservation, so the emitted mask covers
/// the guest's upper address range (the browser drivers' convention — `JitOnrampRun::open_owned_run`).
/// The emitted `"mapped"` global then *defaults* to the full reservation, which is exactly why the
/// per-call sync is mandatory over a reserved window.
fn bumped_for_emit(m: &temen_ir::Module) -> temen_ir::Module {
    let mut e = m.clone();
    if let Some(mc) = e.memory.as_mut() {
        mc.size_log2 = RESERVED_LOG2;
    }
    e
}

/// A guest built once: the module, its tier-up artifact, and the eligibility split.
struct Guest {
    m: temen_ir::Module,
    wasm: Vec<u8>,
    eligible: Arc<[bool]>,
}

fn guests() -> &'static Vec<Guest> {
    static GUESTS: OnceLock<Vec<Guest>> = OnceLock::new();
    GUESTS.get_or_init(|| {
        LEAVES
            .iter()
            .enumerate()
            .map(|(i, leaf)| {
                let m = build(&format!("{ENTRY}{leaf}\n  }}\n}}\n"));
                let (wasm, eligible) =
                    compile_module_tierup(&bumped_for_emit(&m), false).expect("emit");
                assert_eq!(
                    eligible,
                    vec![false, true],
                    "leaf#{i}: the window-shaping entry stays interpreted, the pure leaf emits"
                );
                Guest {
                    m,
                    wasm,
                    eligible: Arc::from(eligible.into_boxed_slice()),
                }
            })
            .collect()
    })
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
/// with the `"mapped"` global set to `sync` (the #717 driver contract) and a canary past the
/// reservation. Asserts the canary survived (the escape property) and copies emitted writes back.
///
/// SAFETY: `base` is the live window buffer, touched only while the vCPU is paused (single-threaded).
fn run_emitted(
    m: &temen_ir::Module,
    wasm: &[u8],
    func: u32,
    argv: &[i64],
    base: *mut u8,
    win_size: usize,
    sync: u64,
) -> Outcome {
    let engine = Engine::default();
    let module = WModule::new(&engine, wasm).expect("emitted wasm must validate");
    let mut store: Store<i32> = Store::new(&engine, 0);
    let canary_at = WIN_BASE as usize + win_size;
    let need = canary_at + CANARY;
    let pages = (need as u32).div_ceil(1 << 16) + 1;
    let memory = Memory::new(&mut store, MemoryType::new(pages, None)).unwrap();
    memory
        .write(&mut store, ENV_PTR as usize, &(FUEL as i64).to_le_bytes())
        .unwrap();
    // SAFETY: see fn doc.
    let live = unsafe { std::slice::from_raw_parts(base, win_size) };
    memory.write(&mut store, WIN_BASE as usize, live).unwrap();
    memory
        .write(&mut store, canary_at, &[CANARY_BYTE; CANARY])
        .unwrap();

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
    // The #717 host-sync contract: the event's committed-extent snapshot lands in the emitted
    // `"mapped"` global before the call.
    instance
        .get_global(&store, "mapped")
        .expect("emitted module exports the live-mapped global")
        .set(&mut store, Val::I64(sync as i64))
        .unwrap();
    let f = instance
        .get_func(&store, &format!("f{func}"))
        .unwrap_or_else(|| panic!("f{func} not exported"));

    let sig = &m.funcs[func as usize];
    let mut params = vec![Val::I32(WIN_BASE as i32), Val::I32(ENV_PTR as i32)];
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
        Err(_) => Outcome::Trap(match *store.data() {
            TRAP_OUT_OF_FUEL => TrapKind::OutOfFuel,
            TRAP_MEMORY_FAULT => TrapKind::MemoryFault,
            // wasmi's own traps (an unaligned atomic, a linear-memory overrun past the canary) are
            // not the guard's — the parity assertion classifies them.
            _ => TrapKind::Other,
        }),
    };
    // The escape property: whatever the live bound admitted, nothing landed past the reservation.
    let mut canary = vec![0u8; CANARY];
    memory.read(&store, canary_at, &mut canary).unwrap();
    assert!(
        canary.iter().all(|&b| b == CANARY_BYTE),
        "emitted leaf f{func} wrote past the reservation (a confinement escape): argv={argv:?} sync={sync}"
    );
    // SAFETY: see fn doc.
    let backs = unsafe { std::slice::from_raw_parts_mut(base, win_size) };
    let mut buf = vec![0u8; win_size];
    memory.read(&store, WIN_BASE as usize, &mut buf).unwrap();
    backs.copy_from_slice(&buf);
    outcome
}

/// Drive `entry(handle, ..tail)` over a reserved live window. With `tier`, the leaf's call surfaces as
/// [`bytecode::VcpuEvent::TierUp`] and is serviced on emitted wasm with the `"mapped"` sync; without,
/// this is the pure-interpreter oracle. Returns the outcome, the window bytes after the run, and how
/// many tier-ups fired (0 under `tier` means the vCPU declined — an unrepresentable window state).
fn frame(g: &Guest, tail: &[i64], tier: bool) -> (Outcome, Vec<u8>, u32) {
    let win_size = 1usize << RESERVED_LOG2;
    let (back, base, layout) = shared_window(win_size);
    let prog = bytecode::VcpuProgram::compile(&g.m).expect("compile");
    let mut host = Host::new();
    let handle = host.grant_memory();
    let mut args = vec![Value::I64(handle as i64)];
    args.extend(tail.iter().map(|&a| Value::I64(a)));
    let mut vcpu = bytecode::Vcpu::new_root_reserved_over_with_powerbox(
        &prog,
        0,
        &args,
        &[],
        host,
        RESERVED_LOG2,
        back,
    )
    .expect("root vcpu");
    if tier {
        vcpu = vcpu.with_jit_eligible(Arc::clone(&g.eligible));
    }
    let emit_m = bumped_for_emit(&g.m);

    let mut tierups = 0u32;
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
                tierups += 1;
                match run_emitted(&emit_m, &g.wasm, func, &argv, base, win_size, mapped) {
                    Outcome::Vals(v) => vcpu.deliver_tierup(&v),
                    Outcome::Trap(TrapKind::OutOfFuel) => vcpu.deliver_tierup_trap(Trap::OutOfFuel),
                    Outcome::Trap(_) => vcpu.deliver_tierup_trap(Trap::MemoryFault),
                }
            }
            _ => panic!("unexpected event on this single-vCPU run"),
        }
    };
    drop(vcpu);
    // SAFETY: the vCPU (and its `Mem` aliasing the region) is dropped; copy, then free the buffer.
    let bytes = unsafe { std::slice::from_raw_parts(base, win_size) }.to_vec();
    unsafe { std::alloc::dealloc(base, layout) };
    (out, bytes, tierups)
}

/// The host software page size, probed once (a throwaway vCPU over a reserved window) and cached —
/// the offsets the fuzzer chooses are page-relative, so the search adapts to 4-KiB and 16-KiB hosts.
fn page_size() -> u64 {
    static PAGE: OnceLock<u64> = OnceLock::new();
    *PAGE.get_or_init(|| {
        let g = &guests()[0];
        let prog = bytecode::VcpuProgram::compile(&g.m).expect("compile");
        let (back, base, layout) = shared_window(1usize << RESERVED_LOG2);
        let args: Vec<Value> = (0..8).map(|_| Value::I64(0)).collect();
        let vcpu = bytecode::Vcpu::new_root_reserved_over_with_powerbox(
            &prog,
            0,
            &args,
            &[],
            Host::new(),
            RESERVED_LOG2,
            back,
        )
        .expect("probe vcpu");
        let page = vcpu.mem_map_info().expect("window").0;
        drop(vcpu);
        // SAFETY: the vCPU (and its Mem aliasing the region) is dropped; free the buffer.
        unsafe { std::alloc::dealloc(base, layout) };
        page
    })
}

/// Map a byte to a window offset near a boundary: a page index (up to one past the reservation — the
/// declared edge, every grown high-water, and the reservation edge are all page multiples) plus an
/// intra-page delta that lands on and either side of the edge at every scalar width.
fn edge_offset(page_byte: u8, delta_byte: u8, page: u64, npages: u64) -> u64 {
    edge_at((page_byte as u64) % (npages + 2), delta_byte, page)
}

/// Page `p`'s base plus an intra-page delta from the edge set (see [`edge_offset`]).
fn edge_at(p: u64, delta_byte: u8, page: u64) -> u64 {
    let deltas = [
        0i64,
        1,
        -1,
        2,
        -2,
        4,
        -4,
        8,
        -8,
        15,
        -15,
        16,
        -16,
        (page / 2) as i64,
        page as i64 - 1,
        page as i64 + 1,
    ];
    let d = deltas[(delta_byte as usize) % deltas.len()];
    (p as i64 * page as i64 + d).max(0) as u64
}

/// Map a byte to a span length that stresses the bound: 0/1/8/16, page±1, and multi-page spans that
/// reach across the declared edge and the reservation — plus a raw value.
fn edge_span(sel: u8, raw: u8, page: u64) -> u64 {
    let set = [
        0,
        1,
        8,
        16,
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
        (raw as u64 * page) / 16
    }
}

/// One differential: decode a case from `data`, run the guest on the interpreter (oracle) and with the
/// leaf tiered up onto emitted wasm under the live-`mapped` sync, and assert their outcomes and window
/// bytes match. Returns what the case exercised (for non-vacuity tallies).
pub fn fuzz_one(data: &[u8]) -> Cat {
    let mut b = [0u8; 12];
    let n = data.len().min(b.len());
    b[..n].copy_from_slice(&data[..n]);

    let page = page_size();
    let declared = 1u64 << DECLARED_LOG2;
    let reserved = 1u64 << RESERVED_LOG2;
    let npages = reserved / page;
    let tail_pages = npages - declared / page;

    let gi = (b[0] as usize) % LEAVES.len();
    let g = &guests()[gi];
    // The grows: page-aligned ranges in the reserved tail (a zero length is a no-op / errno on both
    // tiers — the "unshaped" default the emit-time bound already covers). A grow starting at the
    // declared edge moves the single bound up; one starting higher leaves a hole the bound cannot
    // represent (the decline arm) — unless the second grow fills it, which makes the state
    // representable again (the sync must read the page map, not a high-water counter). Biased so the
    // representable shapes dominate (the emitted bound is what is under test): three grows in four
    // start at the declared edge, and the second grow — skipped one case in four — chains at the
    // first's end half the time, else lands anywhere in the tail.
    let goff = if b[1] & 3 != 0 {
        declared
    } else {
        declared + (((b[1] >> 2) as u64) % (tail_pages + 1)) * page
    };
    let glen = ((b[2] as u64) % (tail_pages + 1)) * page;
    let (goff2, glen2) = match b[4] & 3 {
        0 => (declared, 0),
        1 | 2 => {
            let end = goff + glen;
            let room = reserved.saturating_sub(end) / page; // grow 1 may already reach the top
            (end, (((b[4] >> 2) as u64) % (room + 1)) * page)
        }
        _ => (
            declared + ((b[3] as u64) % (tail_pages + 1)) * page,
            (((b[4] >> 2) as u64) % (tail_pages + 1)) * page,
        ),
    };
    // The committed high-water the sync should report for a contiguous shape (a hole is declined).
    let mut hi = declared;
    if goff == hi {
        hi += glen;
        if goff2 == hi {
            hi += glen2;
        }
    }
    // The access: around the declared edge, around the expected high-water (where an off-by-one in
    // the bound or the sync shows), or anywhere in the reservation (and one page past it).
    let dpages = declared / page;
    let hpage = hi / page;
    let p = match b[5] % 8 {
        0 => dpages - 1,
        1 => dpages,
        2 => dpages + 1,
        3 => hpage - 1,
        4 => hpage,
        5 => hpage + 1,
        _ => ((b[5] >> 3) as u64) % (npages + 2),
    };
    let mut addr = edge_at(p, b[6], page);
    if is_atomic(gi) {
        addr &= !(leaf_width(gi) - 1); // atomics are aligned by construction (the issue's scope)
    }
    let val = (b[7] as u64).wrapping_mul(0x0101_0101_0101_0101) as i64;
    let span = edge_span(b[8], b[9], page) as i64;
    let src = edge_offset(b[10], b[11], page, npages) as i64;
    let (a1, a2) = match gi {
        16 | 17 => (src, span),         // copy / move: (dst, src, len)
        15 => (val, span),              // fill: (dst, value, len)
        13 => (val, val ^ 0x5a5a_5a5a), // cmpxchg: (addr, expected, replacement)
        _ => (val, span),
    };

    let tail = [
        goff as i64,
        glen as i64,
        goff2 as i64,
        glen2 as i64,
        addr as i64,
        a1,
        a2,
    ];
    let (interp, ibytes, _) = frame(g, &tail, false);
    let (tier, tbytes, tierups) = frame(g, &tail, true);

    // Fuel accounting differs between the tiers; an out-of-fuel (or any non-memory trap) is
    // inconclusive, so don't hold the tiers to parity there — every real case is a MemoryFault or a pass.
    if matches!(interp, Outcome::Trap(TrapKind::OutOfFuel | TrapKind::Other))
        || matches!(tier, Outcome::Trap(TrapKind::OutOfFuel | TrapKind::Other))
    {
        return Cat::Skipped;
    }

    assert_eq!(
        interp, tier,
        "live-mapped access diverged from the interpreter oracle: leaf#{gi} grow=[{goff},{}) grow2=[{goff2},{}) addr={addr} a1={a1} a2={a2} page={page} tierups={tierups}",
        goff + glen, goff2 + glen2
    );
    assert!(
        ibytes == tbytes,
        "live-mapped window bytes diverged from the interpreter oracle: leaf#{gi} grow=[{goff},{}) grow2=[{goff2},{}) addr={addr} a1={a1} a2={a2} page={page} tierups={tierups}",
        goff + glen, goff2 + glen2
    );

    if tierups == 0 {
        return Cat::Declined;
    }
    match interp {
        Outcome::Trap(TrapKind::MemoryFault) => Cat::Trapped,
        Outcome::Vals(_) => Cat::Passed,
        _ => Cat::Skipped,
    }
}

/// The seed transform for the stable regression sweep — deterministic, distinct scramble per seed.
pub fn case_from_seed(seed: u64) -> [u8; 12] {
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
