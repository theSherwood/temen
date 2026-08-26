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
use temen_interp::{bytecode, Host, Region, Trap, Value};
use temen_wasm_jit::{
    compile_module_tierup, compile_module_tierup_b2_paged, compile_module_tierup_paged,
    TRAP_MEMORY_FAULT, TRAP_OUT_OF_FUEL,
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

/// #1081 — bulk-memory straddle guests (paged per-page walk). `(as, off, len, probe)`: `unmap`
/// `[off, off+len)`, then the leaf **`mem.fill`s** `[probe, probe+16)` — a write span the harness places
/// straddling from the page before into the unmapped page, so the per-page walk must trap on both tiers.
const UNMAP_FILL: &str = r#"memory 17
func (i32, i64, i64, i64) -> (i64) {
block 0 (vas: i32, voff: i64, vlen: i64, vprobe: i64) {
  vr = cap.call 5 1 (i64, i64) -> (i64) vas (voff, vlen)
  v1 = call 1 (vprobe)
  return v1
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vval = i32.const 0
  vn = i64.const 16
  mem.fill v0 vval vn
  return v0
  }
}
"#;

/// `(as, off, len, probe)`: `protect` `[off, off+len)` read-only, then the leaf **`mem.fill`s**
/// `[probe, probe+16)` — a write straddling into the `Ro` page must trap on both tiers.
const PROTECT_FILL: &str = r#"memory 17
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
  vval = i32.const 0
  vn = i64.const 16
  mem.fill v0 vval vn
  return v0
  }
}
"#;

/// `(as, off, len, probe)`: `unmap` `[off, off+len)`, then the leaf **`mem.copy`s** *from* the source
/// span `[probe, probe+16)` (straddling the unmapped page) to the plain-`Rw` dest `[17408, 17424)`
/// (above the #1094 NULL guard `[0, 16384)`) — the read side of the walk must trap on both tiers.
const UNMAP_COPY: &str = r#"memory 17
func (i32, i64, i64, i64) -> (i64) {
block 0 (vas: i32, voff: i64, vlen: i64, vprobe: i64) {
  vr = cap.call 5 1 (i64, i64) -> (i64) vas (voff, vlen)
  v1 = call 1 (vprobe)
  return v1
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vdst = i64.const 17408
  vn = i64.const 16
  mem.copy vdst v0 vn
  return v0
  }
}
"#;

/// As [`UNMAP_COPY`] but `protect` (`Ro`): a **read** whose source straddles the `Ro` page must
/// **succeed** on both tiers — the walk admits a load of an `Ro` page (only `Unmapped` traps a read).
const PROTECT_COPY: &str = r#"memory 17
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
  vdst = i64.const 17408
  vn = i64.const 16
  mem.copy vdst v0 vn
  return v0
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

// The driver-side table build is the engine-provided [`bytecode::build_pagestate_table`] — the
// per-emitted-call contract (table + the coverage to write to `"mapped"`), shared with the browser
// par flattening so no driver hand-rolls it.

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
fn probe_page_size(m: &temen_ir::Module) -> u64 {
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
                let (table, cover) = match mode {
                    Mode::PagedSynced => bytecode::build_pagestate_table(&info),
                    // The broken driver: region defaults only, the guest's remaps ignored.
                    Mode::PagedUnsynced => {
                        bytecode::build_pagestate_table(&(info.0, info.1, info.2, Vec::new()))
                    }
                    Mode::Interp => unreachable!(),
                };
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
fn paged_bulk_fill_straddling_unmapped_traps_on_both_tiers() {
    // #1081: the paged per-page walk for bulk memory. A `mem.fill [off-8, off+8)` straddles from the
    // Rw neighbor into the unmapped last page — a first-page-only check would admit the write; the walk
    // (like the oracle's `check_prot_span`) traps on the unmapped page. Both tiers must trap.
    let (off, len) = last_page(UNMAP_FILL);
    let straddle = (off - 8) as i64;
    let (want, _) = run_guest(UNMAP_FILL, off, len, straddle, Mode::Interp);
    assert_eq!(
        want,
        Outcome::Trap(TrapKind::MemoryFault),
        "oracle sanity: a fill straddling the unmapped page faults"
    );
    let (got, tierups) = run_guest(UNMAP_FILL, off, len, straddle, Mode::PagedSynced);
    assert_eq!(
        tierups, 1,
        "the bulk-mem leaf must tier up (no longer excluded from the paged subset)"
    );
    assert_eq!(
        want, got,
        "paged bulk walk diverged on a fill straddling an unmapped page"
    );

    // Control: the same fill fully inside the Rw neighbor round-trips on both tiers (no over-trap).
    let inside = (off - 64) as i64;
    let (want, _) = run_guest(UNMAP_FILL, off, len, inside, Mode::Interp);
    assert_eq!(
        want,
        Outcome::Vals(vec![inside]),
        "oracle sanity: an in-Rw fill succeeds"
    );
    let (got, _) = run_guest(UNMAP_FILL, off, len, inside, Mode::PagedSynced);
    assert_eq!(want, got, "paged bulk walk over-trapped an in-Rw fill");
}

#[test]
fn paged_bulk_fill_straddling_ro_traps_on_both_tiers() {
    // A write straddling into an `Ro` page faults on both tiers (a store admits only `Rw`).
    let (off, len) = last_page(PROTECT_FILL);
    let straddle = (off - 8) as i64;
    let (want, _) = run_guest(PROTECT_FILL, off, len, straddle, Mode::Interp);
    assert_eq!(
        want,
        Outcome::Trap(TrapKind::MemoryFault),
        "oracle sanity: a fill straddling an Ro page faults"
    );
    let (got, tierups) = run_guest(PROTECT_FILL, off, len, straddle, Mode::PagedSynced);
    assert_eq!(tierups, 1);
    assert_eq!(
        want, got,
        "paged bulk walk diverged on a fill straddling an Ro page"
    );
}

#[test]
fn paged_bulk_copy_read_ro_succeeds_unmapped_traps() {
    // The read side of the walk: a `mem.copy` whose SOURCE straddles the managed page. A load of an
    // `Ro` page is fine (succeeds on both tiers — the walk must NOT over-trap reads); a load of an
    // `Unmapped` page faults on both. This pins that the walk distinguishes read from write.
    let (off, len) = last_page(PROTECT_COPY);
    let straddle = (off - 8) as i64;
    let (want, _) = run_guest(PROTECT_COPY, off, len, straddle, Mode::Interp);
    assert_eq!(
        want,
        Outcome::Vals(vec![straddle]),
        "oracle sanity: a read whose source straddles an Ro page succeeds"
    );
    let (got, tierups) = run_guest(PROTECT_COPY, off, len, straddle, Mode::PagedSynced);
    assert_eq!(tierups, 1);
    assert_eq!(
        want, got,
        "paged bulk walk over-trapped a read straddling an Ro page"
    );

    let (off, len) = last_page(UNMAP_COPY);
    let straddle = (off - 8) as i64;
    let (want, _) = run_guest(UNMAP_COPY, off, len, straddle, Mode::Interp);
    assert_eq!(
        want,
        Outcome::Trap(TrapKind::MemoryFault),
        "oracle sanity: a read whose source straddles an unmapped page faults"
    );
    let (got, tierups) = run_guest(UNMAP_COPY, off, len, straddle, Mode::PagedSynced);
    assert_eq!(tierups, 1);
    assert_eq!(
        want, got,
        "paged bulk walk diverged on a read straddling an unmapped page"
    );
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

#[test]
fn b2_paged_composes_the_shared_table_import_with_the_pagestate_global() {
    // #1009 pump shape: a rodata-bearing guest whose dispatch leaf `call_indirect`s. The composed
    // entry must emit a module that BOTH imports the B2 shared table (so indirect dispatch routes
    // through the host-populated table) AND exports the `"pagestate"` global (so the paged driver can
    // point the emitted page check at the live table) — the two features are orthogonal.
    let m = build(
        r#"memory 17
data ro 16384 "readonlybytes"
func (i64) -> (i64) {
block 0 (v0: i64) {
  vs = i32.wrap_i64 v0
  vr = call_indirect (i64) -> (i64) vs (v0)
  return vr
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vl = i64.load v0
  return vl
  }
}
"#,
    );
    assert!(m.data.iter().any(|d| d.readonly), "guest carries rodata");
    let (wasm, emit) = compile_module_tierup_b2_paged(&m, false, 10, 16).expect("emit b2+paged");
    assert!(emit.iter().any(|&e| e), "some function tiers up");

    let engine = Engine::default();
    let module = WModule::new(&engine, &wasm[..]).expect("b2+paged wasm must validate");
    assert!(
        module
            .imports()
            .any(|i| i.module() == "env" && i.name() == "__indirect_function_table"),
        "B2 mode imports the shared reserved table"
    );
    assert!(
        module.exports().any(|e| e.name() == "pagestate"),
        "paged mode exports the page-state base global"
    );
    assert!(
        module.exports().any(|e| e.name() == "mapped"),
        "the live-mapped bound global is present"
    );
}

/// The reactor guest: func 0 `_start(as, off, len)` unmaps `[off, off+len)` at open; func 1
/// `tick(probe)` calls the load leaf (func 2) — the frame's tier-up. The reactor shape of the
/// paged driver seam ([`bytecode::VcpuReactor::with_jit_page_checked`]).
const REACTOR_UNMAP: &str = r#"memory 17
func (i32, i64, i64) -> () {
block 0 (vas: i32, voff: i64, vlen: i64) {
  vr = cap.call 5 1 (i64, i64) -> (i64) vas (voff, vlen)
  return
  }
}
func (i64) -> (i64) {
block 0 (vp: i64) {
  v1 = call 2 (vp)
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

/// Drive one `tick(probe)` frame through a **paged `VcpuReactor`** (the driver seam the browser
/// wiring mirrors): `_start` unmaps the window's last page at open, each tier-up hands the live
/// `map_info` to the service, which builds the table ([`bytecode::build_pagestate_table`]) and
/// runs the emitted leaf over the live window with the coverage as the bound.
fn reactor_frame(probe: i64, paged: bool) -> (Result<Vec<Value>, Trap>, u32) {
    let m = build(REACTOR_UNMAP);
    let win_size = 1usize << WIN_LOG2;
    let (back, base, layout) = shared_window(win_size);
    let page = probe_page_size_of(&m, Arc::clone(&back));
    let (off, len) = ((1u64 << WIN_LOG2) - page, page);

    let host = std::sync::Mutex::new({
        let mut h = Host::new();
        h.grant_memory();
        h
    });
    // The AddressSpace handle is powerbox slot-granted; re-grant inside for the arg value.
    let asl = {
        let mut g = host.lock().unwrap();
        *g = Host::new();
        g.grant_memory()
    };
    let start_args = [
        Value::I32(asl),
        Value::I64(off as i64),
        Value::I64(len as i64),
    ];
    let mut r = bytecode::VcpuReactor::open(&m, back, &host, &start_args).expect("open");
    let wasm = if paged {
        let (wasm, eligible) =
            compile_module_tierup_paged(&m, false, page.trailing_zeros() as u8).expect("emit");
        assert!(eligible[2], "the load leaf must be paged-eligible");
        r = r
            .with_jit_eligible(Arc::from(eligible.into_boxed_slice()))
            .with_jit_page_checked();
        wasm
    } else {
        Vec::new()
    };

    let mut tierups = 0u32;
    let out = r.frame(
        1,
        &[Value::I64(probe)],
        &host,
        |func, argv, _mapped, info| {
            tierups += 1;
            let info = info.expect("a paged reactor hands the live map to every tier-up");
            let (table, cover) = bytecode::build_pagestate_table(&info);
            match run_emitted(&wasm, func, argv, base, win_size, &table, cover) {
                Outcome::Vals(v) => Ok(v),
                Outcome::Trap(TrapKind::OutOfFuel) => Err(Trap::OutOfFuel),
                Outcome::Trap(_) => Err(Trap::MemoryFault),
            }
        },
    );
    drop(r);
    // SAFETY: the reactor (and its `Mem` aliasing the region) is dropped; free the window buffer.
    unsafe { std::alloc::dealloc(base, layout) };
    (out, tierups)
}

/// Page-size probe over an already-built module + backing (reactor-shape twin of
/// [`probe_page_size`], reusing the same region so no second buffer is needed).
fn probe_page_size_of(m: &temen_ir::Module, back: Arc<Region>) -> u64 {
    let prog = bytecode::VcpuProgram::compile(m).expect("compile");
    let vcpu = bytecode::Vcpu::new_root_with_powerbox(
        &prog,
        1, // a pure func as entry: constructing never runs it
        &[Value::I64(0)],
        back,
        &[],
        Host::new(),
    )
    .expect("probe vcpu");
    vcpu.mem_map_info().expect("window").0
}

#[test]
fn paged_reactor_frame_matches_interpreter() {
    let m = build(REACTOR_UNMAP);
    let win = 1i64 << WIN_LOG2;
    let (back, base, layout) = shared_window(1usize << WIN_LOG2);
    let page = probe_page_size_of(&m, back) as i64;
    // SAFETY: probe vcpu dropped inside probe_page_size_of; free the probe buffer.
    unsafe { std::alloc::dealloc(base, layout) };

    // Inside the unmapped last page: the oracle (unpaged reactor, interpreted leaf) faults, and
    // the paged reactor's emitted leaf must fault identically through the table.
    let probe_unmapped = win - page + 16;
    let (want, t0) = reactor_frame(probe_unmapped, false);
    assert_eq!(want, Err(Trap::MemoryFault), "oracle sanity");
    assert_eq!(t0, 0, "unpaged reactor never tiers up");
    let (got, t1) = reactor_frame(probe_unmapped, true);
    assert_eq!(t1, 1, "the leaf must tier up through the reactor seam");
    assert_eq!(
        got,
        Err(Trap::MemoryFault),
        "paged reactor diverged on the unmapped page"
    );

    // Inside the mapped prefix, above the #1094 NULL guard `[0, 16384)`: both succeed (zeroed window ⇒ 0).
    let probe_ok = 16384 + 8;
    let (want, _) = reactor_frame(probe_ok, false);
    assert_eq!(want, Ok(vec![Value::I64(0)]), "oracle sanity");
    let (got, t2) = reactor_frame(probe_ok, true);
    assert_eq!(t2, 1);
    assert_eq!(
        got.map(|v| v
            .iter()
            .map(|x| match x {
                Value::I64(i) => *i,
                _ => panic!(),
            })
            .collect::<Vec<_>>()),
        Ok(vec![0]),
        "paged reactor diverged inside the prefix"
    );
}

/// Guest `(as, off, len, probe)`: `unmap` `[off, off+len)`, then the leaf does an **aligned atomic
/// load** at `probe` — the `align=true` confine path, whose paged check is first-page-only (an
/// aligned access can never straddle; the second consultation is elided).
const UNMAP_ATOMIC_LOAD: &str = r#"memory 17
func (i32, i64, i64, i64) -> (i64) {
block 0 (vas: i32, voff: i64, vlen: i64, vprobe: i64) {
  vr = cap.call 5 1 (i64, i64) -> (i64) vas (voff, vlen)
  v1 = call 1 (vprobe)
  return v1
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vl = i64.atomic.load v0
  return vl
  }
}
"#;

#[test]
fn aligned_atomic_load_traps_on_unmapped_page() {
    // The align=true paged path: the (single) first-page consultation must still fire — an
    // aligned atomic load of the unmapped page traps on both tiers, and inside the prefix it
    // round-trips. Pins that eliding the redundant second consultation removed nothing needed.
    let (off, len) = last_page(UNMAP_ATOMIC_LOAD);
    let probe = (off + 16) as i64; // 8-aligned, inside the unmapped page
    let (want, _) = run_guest(UNMAP_ATOMIC_LOAD, off, len, probe, Mode::Interp);
    assert_eq!(want, Outcome::Trap(TrapKind::MemoryFault), "oracle sanity");
    let (got, tierups) = run_guest(UNMAP_ATOMIC_LOAD, off, len, probe, Mode::PagedSynced);
    assert_eq!(tierups, 1);
    assert_eq!(want, got, "paged tier diverged on an aligned atomic load");

    let inside = 16384 + 8; // aligned, inside the mapped prefix (above the #1094 NULL guard)
    let (want, _) = run_guest(UNMAP_ATOMIC_LOAD, off, len, inside, Mode::Interp);
    assert_eq!(want, Outcome::Vals(vec![0]), "oracle sanity");
    let (got, tierups) = run_guest(UNMAP_ATOMIC_LOAD, off, len, inside, Mode::PagedSynced);
    assert_eq!(tierups, 1);
    assert_eq!(
        want, got,
        "paged tier diverged on an in-prefix aligned atomic load"
    );
}
