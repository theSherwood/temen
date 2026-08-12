//! **The #750 gated software page-check** — the differential + boundary proof for the opt-in paged
//! tier ([`compile_module_tierup_paged`]), the escalation past emit-nothing for self-page-managing
//! guests. Per AGENTS.md this lowering is fuzz/boundary-tested as its own masking-hinge unit.
//!
//! A paged guest `unmap`s / `protect`s its own pages (`cap.call 5 1/2` — interp-serviced; such
//! functions are never emitted), then a tiered-up leaf accesses the window on emitted wasm. The
//! driver contract: before each emitted call, refresh the byte-per-page state table (`0 = Unmapped`,
//! `1 = Rw`, `2 = Ro`) from the live page map ([`bytecode::Vcpu::mem_map_info`]), write its base to
//! the `"pagestate"` global and its **coverage** to `"mapped"` (the bound check traps everything
//! above the table, where `check_prot` — no entries above — faults too). The emitted check traps
//! exactly where the interpreter's `check_prot` does (INVARIANTS #9):
//!  - a load of an `Unmapped` page and a store to an `Ro` page trap on both tiers;
//!  - a load of an `Ro` page succeeds on both (reading the same window bytes);
//!  - a **straddling** store whose *last* byte lands on an `Unmapped` page traps on both — the
//!    width>1 second page consultation, which a first-page-only check would miss;
//!  - an **unsynced** table (defaults only, ignoring the guest's remaps) diverges — the negative
//!    pin for why the per-call refresh is the contract.
//!
//! The trap decision runs strictly inside the always-emitted `& MASK` clamp, so a wrong table is a
//! trap-parity divergence, never an escape (INVARIANTS #2) — same argument as the #717 live bound.

use std::sync::Arc;
use svm_interp::{bytecode, Host, Region, Trap, Value};
use svm_wasm_jit::{
    compile_module_tierup, compile_module_tierup_paged, TRAP_MEMORY_FAULT, TRAP_OUT_OF_FUEL,
};
use wasmi::{Caller, Engine, Linker, Memory, MemoryType, Module as WModule, Store, Val};

const WIN_BASE: u32 = 0x4_0000;
const ENV_PTR: u32 = 1024;
const FUEL: u64 = 100_000_000;

/// The window: 128 KiB, fully mapped (`mapped == reserved`) — big enough for a whole page even on
/// 16-KiB-page hosts (macOS), leaving ≥ 8 pages to manage.
const WIN_LOG2: u8 = 17;

/// Guest `(as, off, len, probe)`: `unmap` `[off, off+len)`, then call the **load** leaf at `probe`.
const UNMAP_LOAD: &str = r#"memory 17
func (i32, i64, i64, i64) -> (i64) {
block 0 (vas: i32, voff: i64, vlen: i64, vprobe: i64) {
  vr = cap.call 5 1 (i64, i64) -> (i64) vas (voff, vlen)
  v1 = call 1 (vprobe)
  return v1
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vl = i64.load v0
  return vl
  }
}
"#;

/// Guest `(as, off, len, probe)`: `unmap` `[off, off+len)`, then call the **store** leaf at `probe`
/// (an 8-byte store — the straddle probe places its last byte on the unmapped page).
const UNMAP_STORE: &str = r#"memory 17
func (i32, i64, i64, i64) -> (i64) {
block 0 (vas: i32, voff: i64, vlen: i64, vprobe: i64) {
  vr = cap.call 5 1 (i64, i64) -> (i64) vas (voff, vlen)
  v1 = call 1 (vprobe)
  return v1
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  i64.store v0 v0
  vl = i64.load v0
  return vl
  }
}
"#;

/// Guest `(as, off, len, probe)`: store a marker at `probe`, `protect` `[off, off+len)` read-only,
/// then call the **load** leaf at `probe` — the load must still succeed on both tiers.
const PROTECT_LOAD: &str = r#"memory 17
func (i32, i64, i64, i64) -> (i64) {
block 0 (vas: i32, voff: i64, vlen: i64, vprobe: i64) {
  i64.store vprobe vprobe
  vp = i32.const 1
  vr = cap.call 5 2 (i64, i64, i32) -> (i64) vas (voff, vlen, vp)
  v1 = call 1 (vprobe)
  return v1
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vl = i64.load v0
  return vl
  }
}
"#;

/// As [`PROTECT_LOAD`] but the leaf **stores** — a write to the `Ro` page traps on both tiers.
const PROTECT_STORE: &str = r#"memory 17
func (i32, i64, i64, i64) -> (i64) {
block 0 (vas: i32, voff: i64, vlen: i64, vprobe: i64) {
  vp = i32.const 1
  vr = cap.call 5 2 (i64, i64, i32) -> (i64) vas (voff, vlen, vp)
  v1 = call 1 (vprobe)
  return v1
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  i64.store v0 v0
  vl = i64.load v0
  return vl
  }
}
"#;

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

/// How the run is driven.
#[derive(Clone, Copy, PartialEq)]
enum Mode {
    /// Pure interpreter — the oracle.
    Interp,
    /// Paged tier-up with the full driver contract (table refreshed from the live map per call).
    PagedSynced,
    /// Paged tier-up with a defaults-only table (the guest's remaps never applied) — the pre-fix /
    /// broken-driver divergence pin.
    PagedUnsynced,
}

fn build(src: &str) -> svm_ir::Module {
    let m = svm_text::parse_module(src).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
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

/// The driver-side table build: byte-per-page from [`bytecode::MemMapInfo`] — region default
/// (`Rw` below `mapped`, `Unmapped` above), overridden by the explicit entries (`map_info` kinds:
/// `0 = Ro`, `1 = Rw`, `2 = Unmapped`; `3 = Backed` never occurs — paged mode gates SharedRegion).
///
/// The table covers `[0, max(mapped, highest explicit entry end))` — NOT the reserved mask domain
/// (the default reservation is 2^40): the driver writes the coverage to the `"mapped"` global, so
/// the bound check traps everything above it, exactly where `check_prot` (no entries above
/// coverage) faults too. The two checks compose: bound = coverage, page state refines within.
fn build_table(info: &bytecode::MemMapInfo) -> Vec<u8> {
    let (page, mapped, _reserved, entries) = info;
    let top = entries
        .iter()
        .map(|(off, _)| off / page + 1)
        .max()
        .unwrap_or(0)
        .max(mapped / page);
    let mut t = vec![0u8; top as usize];
    for (i, b) in t.iter_mut().enumerate() {
        if (i as u64) * page < *mapped {
            *b = 1; // Rw default inside the mapped prefix
        }
    }
    for (off, kind) in entries {
        let state = match kind {
            0 => 2, // Ro
            1 => 1, // Rw
            2 => 0, // Unmapped
            k => panic!("Backed page ({k}) must never reach a paged run"),
        };
        t[(off / page) as usize] = state;
    }
    t
}

/// Run the emitted `f{func}(win, env, probe)` under wasmi over a memory mirrored from the live
/// window, with the page-state `table` placed after the window and both driver globals written
/// (`"mapped"` = reserved, `"pagestate"` = the table's address). Copies emitted writes back.
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

/// The host's software page size, probed from a throwaway run of `m` (the guests take page-aligned
/// offsets as arguments, so the test adapts to 4-KiB and 16-KiB hosts alike).
fn probe_page_size(m: &svm_ir::Module) -> u64 {
    let prog = bytecode::VcpuProgram::compile(m).expect("compile");
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
    let (page, ..) = vcpu.mem_map_info().expect("window");
    drop(vcpu);
    // SAFETY: the vCPU (and its `Mem` aliasing the region) is dropped; free the buffer.
    unsafe { std::alloc::dealloc(base, layout) };
    page
}

/// Drive `guest(as, off, len, probe)` in `mode`, servicing tier-ups on emitted wasm with the
/// page-check driver contract. Returns the outcome and how many tier-ups ran.
fn run_guest(guest_src: &str, off: u64, len: u64, probe: i64, mode: Mode) -> (Outcome, u32) {
    let m = build(guest_src);
    let win_size = 1usize << WIN_LOG2;
    let (back, base, layout) = shared_window(win_size);
    let prog = bytecode::VcpuProgram::compile(&m).expect("compile");
    let mut host = Host::new();
    let asl = host.grant_memory();
    let args = [
        Value::I32(asl),
        Value::I64(off as i64),
        Value::I64(len as i64),
        Value::I64(probe),
    ];
    let mut vcpu = bytecode::Vcpu::new_root_with_powerbox(&prog, 0, &args, back, &[], host)
        .expect("root vcpu");

    let page = vcpu.mem_map_info().expect("window").0;
    let (wasm, eligible) = if mode != Mode::Interp {
        let (wasm, eligible) =
            compile_module_tierup_paged(&m, false, page.trailing_zeros() as u8).expect("emit");
        assert_eq!(
            eligible,
            vec![false, true],
            "paged mode: the page-op entry stays interpreted, the pure leaf emits"
        );
        let e: Arc<[bool]> = Arc::from(eligible.into_boxed_slice());
        vcpu = vcpu
            .with_jit_eligible(Arc::clone(&e))
            .with_jit_page_checked();
        (wasm, e)
    } else {
        (Vec::new(), Arc::from(vec![].into_boxed_slice()))
    };
    let _ = &eligible;

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
                let info = vcpu.mem_map_info().expect("window");
                // #750: a page-checked run surfaces the RESERVED mask-domain size, never a decline
                // — the driver then narrows the global to its table coverage below.
                assert_eq!(mapped, info.2, "paged runs surface reserved");
                let table = match mode {
                    Mode::PagedSynced => build_table(&info),
                    // The broken driver: region defaults only, the guest's remaps ignored.
                    Mode::PagedUnsynced => build_table(&(info.0, info.1, info.2, Vec::new())),
                    Mode::Interp => unreachable!(),
                };
                let cover = table.len() as u64 * info.0;
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
    (out, tierups)
}

/// The managed page: the window's last page (`[win - page, win)`), on every host page size.
fn last_page(m_src: &str) -> (u64, u64) {
    let page = probe_page_size(&build(m_src));
    ((1u64 << WIN_LOG2) - page, page)
}

#[test]
fn unmapped_load_traps_on_both_tiers() {
    let (off, len) = last_page(UNMAP_LOAD);
    let probe = (off + 16) as i64; // inside the unmapped page
    let (want, _) = run_guest(UNMAP_LOAD, off, len, probe, Mode::Interp);
    assert_eq!(want, Outcome::Trap(TrapKind::MemoryFault), "oracle sanity");
    let (got, tierups) = run_guest(UNMAP_LOAD, off, len, probe, Mode::PagedSynced);
    assert_eq!(tierups, 1, "the leaf must actually tier up");
    assert_eq!(want, got, "paged tier diverged on an unmapped load");
}

#[test]
fn ro_load_succeeds_store_traps_on_both_tiers() {
    let (off, len) = last_page(PROTECT_LOAD);
    let probe = (off + 16) as i64; // inside the now-Ro page; the marker was stored pre-protect
    let (want, _) = run_guest(PROTECT_LOAD, off, len, probe, Mode::Interp);
    assert_eq!(
        want,
        Outcome::Vals(vec![probe]),
        "oracle sanity: Ro load reads the marker"
    );
    let (got, tierups) = run_guest(PROTECT_LOAD, off, len, probe, Mode::PagedSynced);
    assert_eq!(tierups, 1);
    assert_eq!(want, got, "paged tier diverged on an Ro load");

    let (want, _) = run_guest(PROTECT_STORE, off, len, probe, Mode::Interp);
    assert_eq!(
        want,
        Outcome::Trap(TrapKind::MemoryFault),
        "oracle sanity: Ro store faults"
    );
    let (got, tierups) = run_guest(PROTECT_STORE, off, len, probe, Mode::PagedSynced);
    assert_eq!(tierups, 1);
    assert_eq!(want, got, "paged tier diverged on an Ro store");
}

#[test]
fn straddling_store_traps_at_the_page_edge() {
    // The width>1 second consultation: an 8-byte store whose LAST byte lands on the unmapped page
    // ([off-4, off+4)) — a first-page-only check would admit it; the oracle walks both pages.
    let (off, len) = last_page(UNMAP_STORE);
    let straddle = (off - 4) as i64;
    let (want, _) = run_guest(UNMAP_STORE, off, len, straddle, Mode::Interp);
    assert_eq!(want, Outcome::Trap(TrapKind::MemoryFault), "oracle sanity");
    let (got, tierups) = run_guest(UNMAP_STORE, off, len, straddle, Mode::PagedSynced);
    assert_eq!(tierups, 1);
    assert_eq!(want, got, "paged tier diverged on a page-edge straddle");

    // Control: the same store fully inside the Rw neighbor ([off-8, off)) round-trips on both.
    let inside = (off - 8) as i64;
    let (want, _) = run_guest(UNMAP_STORE, off, len, inside, Mode::Interp);
    assert_eq!(want, Outcome::Vals(vec![inside]), "oracle sanity");
    let (got, tierups) = run_guest(UNMAP_STORE, off, len, inside, Mode::PagedSynced);
    assert_eq!(tierups, 1);
    assert_eq!(want, got, "paged tier diverged just inside the edge");
}

#[test]
fn unsynced_table_diverges() {
    // The negative pin: a driver that never applies the guest's remaps (defaults-only table)
    // admits the unmapped load the interpreter faults on. This divergence is why the per-call
    // table refresh is the contract, exactly like #717's unsynced-global pin.
    let (off, len) = last_page(UNMAP_LOAD);
    let probe = (off + 16) as i64;
    let (got, tierups) = run_guest(UNMAP_LOAD, off, len, probe, Mode::PagedUnsynced);
    assert_eq!(tierups, 1);
    assert_eq!(
        got,
        Outcome::Vals(vec![0]),
        "an unsynced table admits the unmapped load (reads the zeroed mirror) — the divergence"
    );
}

#[test]
fn unpaged_output_carries_no_pagestate() {
    // Lands-dark pin: the default (unpaged) entry emits no page-check machinery — no `"pagestate"`
    // export — and a page-op module still emits nothing there (the existing page_ops.rs contract),
    // while the paged entry lights the same module up.
    let m = build(UNMAP_LOAD);
    let (_, eligible) = compile_module_tierup(&m, false).expect("unpaged emit");
    assert_eq!(
        eligible,
        vec![false, false],
        "unpaged: page-op module emits nothing"
    );

    let plain = build(
        r#"memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  vl = i64.load v0
  return vl
  }
}
"#,
    );
    let (wasm, eligible) = compile_module_tierup(&plain, false).expect("unpaged emit");
    assert_eq!(eligible, vec![true]);
    let engine = Engine::default();
    let module = WModule::new(&engine, &wasm).expect("validates");
    let mut store: Store<i32> = Store::new(&engine, 0);
    let memory = Memory::new(&mut store, MemoryType::new(2, None)).unwrap();
    let mut linker: Linker<i32> = Linker::new(&engine);
    linker.define("env", "memory", memory).unwrap();
    linker
        .func_wrap("env", "trap", |_: Caller<'_, i32>, _c: i32| {})
        .unwrap();
    linker
        .func_wrap::<_, ()>(
            "env",
            "call_interp",
            |_: Caller<'_, i32>, _f: i32, _a: i32| {},
        )
        .unwrap();
    let instance = linker
        .instantiate(&mut store, &module)
        .unwrap()
        .start(&mut store)
        .unwrap();
    assert!(
        instance.get_global(&store, "pagestate").is_none(),
        "unpaged output must carry no pagestate global — the mode lands dark"
    );
    assert!(instance.get_global(&store, "mapped").is_some());
}
