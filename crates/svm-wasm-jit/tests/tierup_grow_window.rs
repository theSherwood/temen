//! **Grow-then-tier-up host sync** (#717) — the differential proof that a `vm_map`-growing guest
//! and the wasm-JIT tier agree once the driver syncs the live `"mapped"` global from the
//! [`VcpuEvent::TierUp`] entry snapshot.
//!
//! `live_mapped.rs` proves the emitted *check* reads the global (set by hand); `page_ops.rs` proves
//! a pure-grow module now keeps its leaves eligible. This test closes the loop end-to-end: the
//! interpreter services a real `ADDRESS_SPACE.map` (`cap.call 5 0`) growing a **reserved-tail**
//! window, then tiers a leaf up onto emitted wasm over that same live window — and the tiered frame
//! must be observably identical to the interpreter-only frame (INVARIANTS.md #9), in **both**
//! directions:
//!  - an access **inside** the grown region succeeds on both tiers (the #717 symptom was the
//!    emitted tier spuriously faulting here — pre-#774 the bound was baked at emit time);
//!  - an access **above** the committed high-water faults on both tiers (the emitted default after
//!    the reserved-mask bump is the full reservation, so an *unsynced* JIT would sail through —
//!    `unsynced_jit_diverges_without_host_sync` pins that divergence as the bug the sync fixes).
//!
//! The driver contract exercised here is exactly the browser Worker's (`worker.js` TIERUP): bump
//! the module's `size_log2` to the reservation before emitting (the mask must cover the guest's
//! upper address range), then write the event's `mapped` snapshot to the emitted global before
//! every `f{func}` call.

use std::sync::Arc;
use svm_interp::{bytecode, Host, Region, Trap, Value};
use svm_wasm_jit::{compile_module_tierup, TRAP_MEMORY_FAULT, TRAP_OUT_OF_FUEL};
use wasmi::{Caller, Engine, Linker, Memory, MemoryType, Module as WModule, Store, Val};

const WIN_BASE: u32 = 0x4_0000;
const ENV_PTR: u32 = 1024;
const FUEL: u64 = 100_000_000;

/// The reservation: 128 KiB of address space over a 64-KiB declared (`memory 16`) prefix.
const RESERVED_LOG2: u8 = 17;

/// The guest: func 0 (interp-driven — it remaps) `map`s `[64 KiB, 64 KiB + 16 KiB)` of the reserved
/// tail through its granted ADDRESS_SPACE handle, then calls the pure leaf (tier-up eligible) with
/// the probe address. The leaf stores the arg at `probe + 8`, loads it back, and returns it — a
/// round-trip through window bytes at the probe. 16 KiB is a whole page on every host page size we
/// run (4 KiB Linux, 16 KiB macOS), so the committed high-water is `80 KiB` on all of them.
const GROW_THEN_LEAF: &str = r#"memory 16
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  vas = i32.wrap_i64 v0
  voff = i64.const 65536
  vlen = i64.const 16384
  vprot = i32.const 3
  vr = cap.call 5 0 (i64, i64, i32) -> (i64) vas (voff, vlen, vprot)
  v2 = call 1 (v1)
  return v2
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vp = i64.const 8
  vaddr = i64.add v0 vp
  i64.store vaddr v0
  vld = i64.load vaddr
  return vld
  }
}
"#;

/// A probe inside the grown region `[64 KiB, 80 KiB)` — committed by the guest's `map`.
const PROBE_GROWN: i64 = 65536 + 16;
/// A probe above the committed high-water (and beyond macOS's 16-KiB page rounding of the grow),
/// still inside the reservation: `96 KiB + 16`. Uncommitted on every host page size.
const PROBE_UNCOMMITTED: i64 = 98304 + 16;

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

fn build(src: &str) -> svm_ir::Module {
    let m = svm_text::parse_module(src).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    m
}

/// The emit input: the guest module with `size_log2` bumped to the reservation, so the emitted mask
/// covers the guest's upper address range (the browser drivers' convention — see
/// `JitOnrampRun::open_owned_run`). The emitted `"mapped"` global then *defaults* to the full
/// reservation, which is exactly why the per-call sync is mandatory over a reserved window.
fn bumped_for_emit(m: &svm_ir::Module) -> svm_ir::Module {
    let mut e = m.clone();
    if let Some(mc) = e.memory.as_mut() {
        assert!(mc.size_log2 < RESERVED_LOG2, "test wants a real tail");
        mc.size_log2 = RESERVED_LOG2;
    }
    e
}

/// An 8-aligned zeroed buffer + `Region::shared` over it — the caller-owned live window.
fn shared_window(size: usize) -> (Arc<Region>, *mut u8, std::alloc::Layout) {
    let layout = std::alloc::Layout::from_size_align(size, 8).unwrap();
    // SAFETY: non-zero layout; `size` valid 8-aligned bytes owned here, used only as this window.
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!base.is_null());
    // SAFETY: `base` is `size` valid 8-aligned bytes, exclusively this window's, freed only after.
    let back = Arc::new(unsafe { Region::shared(base, size as u64) });
    (back, base, layout)
}

/// Run the emitted `f{func}(win, env, ...argv)` under wasmi over a memory mirrored from the live
/// window, with the `"mapped"` global set to `sync` when given (the #717 driver contract) or left
/// at its emit-time default when `None` (modelling a pre-fix driver). Copies emitted writes back.
///
/// SAFETY: `base` is the live window buffer, read/written only while the vCPU is paused inside the
/// tier-up service (single-threaded), so no aliasing with the interpreter's own access.
#[allow(clippy::too_many_arguments)]
fn run_emitted(
    m: &svm_ir::Module,
    wasm: &[u8],
    func: u32,
    argv: &[i64],
    base: *mut u8,
    win_size: usize,
    sync: Option<u64>,
) -> Outcome {
    let engine = Engine::default();
    let module = WModule::new(&engine, wasm).expect("emitted wasm must validate");
    let mut store: Store<i32> = Store::new(&engine, 0);
    let need = WIN_BASE as usize + win_size;
    let pages = (need as u32).div_ceil(1 << 16) + 1;
    let memory = Memory::new(&mut store, MemoryType::new(pages, None)).unwrap();
    memory
        .write(&mut store, ENV_PTR as usize, &(FUEL as i64).to_le_bytes())
        .unwrap();
    // SAFETY: see fn doc.
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
    if let Some(mapped) = sync {
        // The #717 host-sync contract: the event's committed-extent snapshot lands in the emitted
        // `"mapped"` global before the call.
        instance
            .get_global(&store, "mapped")
            .expect("emitted module exports the live-mapped global")
            .set(&mut store, Val::I64(mapped as i64))
            .unwrap();
    }
    let f = instance
        .get_func(&store, &format!("f{func}"))
        .unwrap_or_else(|| panic!("f{func} not exported"));

    let sig = &m.funcs[func as usize];
    let mut params = vec![Val::I32(WIN_BASE as i32), Val::I32(ENV_PTR as i32)];
    for (t, a) in sig.params.iter().zip(argv) {
        assert_eq!(*t, svm_ir::ValType::I64, "tier-up ABI marshals i64 slots");
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
    // SAFETY: see fn doc.
    let back = unsafe { std::slice::from_raw_parts_mut(base, win_size) };
    let mut buf = vec![0u8; win_size];
    memory.read(&store, WIN_BASE as usize, &mut buf).unwrap();
    back.copy_from_slice(&buf);
    outcome
}

/// The tier-up service configuration: the emitted wasm, its eligibility bitmap, and whether the
/// driver syncs the `"mapped"` global from the event (the fix vs the pre-fix driver).
type Emitted<'a> = (&'a [u8], &'a Arc<[bool]>, bool);

/// Drive `entry(handle, probe)` over a reserved live window. With `emitted`, the leaf's `Call`
/// surfaces as [`bytecode::VcpuEvent::TierUp`] and is serviced on emitted wasm. Without, this is
/// the pure-interpreter oracle. Returns the frame outcome and how many tier-ups fired.
fn reserved_frame(m: &svm_ir::Module, emitted: Option<Emitted>, probe: i64) -> (Outcome, u32) {
    let win_size = 1usize << RESERVED_LOG2;
    let (back, base, layout) = shared_window(win_size);
    let prog = bytecode::VcpuProgram::compile(m).expect("compile");
    let mut host = Host::new();
    let handle = host.grant_memory();
    let emit_m = bumped_for_emit(m);

    let mut vcpu = bytecode::Vcpu::new_root_reserved_over_with_powerbox(
        &prog,
        0,
        &[Value::I64(handle as i64), Value::I64(probe)],
        &[],
        host,
        RESERVED_LOG2,
        back,
    )
    .expect("root vcpu");
    if let Some((_, eligible, _)) = emitted {
        vcpu = vcpu.with_jit_eligible(Arc::clone(eligible));
    }

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
                let (wasm, _, sync) = emitted.expect("tier-up only fires with eligibility set");
                let sync = sync.then_some(mapped);
                match run_emitted(&emit_m, wasm, func, &argv, base, win_size, sync) {
                    Outcome::Vals(v) => vcpu.deliver_tierup(&v),
                    Outcome::Trap(TrapKind::OutOfFuel) => vcpu.deliver_tierup_trap(Trap::OutOfFuel),
                    Outcome::Trap(_) => vcpu.deliver_tierup_trap(Trap::MemoryFault),
                }
            }
            _ => panic!("unexpected event on the single-vCPU driver"),
        }
    };
    drop(vcpu);
    // SAFETY: the vCPU (and its `Mem` aliasing the region) is dropped; free the window buffer.
    unsafe { std::alloc::dealloc(base, layout) };
    (out, tierups)
}

/// Compile the tier-up artifact (over the reservation-bumped module) and assert the #717 gate
/// split routed it as intended: the mapping entry stays interpreted, the leaf is eligible.
fn emit(m: &svm_ir::Module) -> (Vec<u8>, Arc<[bool]>) {
    let (wasm, eligible) = compile_module_tierup(&bumped_for_emit(m), false).expect("emit");
    assert_eq!(
        eligible,
        vec![false, true],
        "grow module: entry interpreted, leaf eligible (page_ops.rs pins this)"
    );
    (wasm, Arc::from(eligible.into_boxed_slice()))
}

#[test]
fn grown_region_access_matches_interpreter() {
    // The #717 symptom, end-to-end: the tiered-up leaf round-trips a store/load through the
    // `vm_map`-grown region and must agree with the interpreter-only oracle (both succeed — the
    // synced `"mapped"` global admits the grown pages).
    let m = build(GROW_THEN_LEAF);
    let (wasm, eligible) = emit(&m);
    let (want, _) = reserved_frame(&m, None, PROBE_GROWN);
    assert_eq!(want, Outcome::Vals(vec![PROBE_GROWN]), "oracle sanity");
    let (got, tierups) = reserved_frame(&m, Some((&wasm, &eligible, true)), PROBE_GROWN);
    assert_eq!(tierups, 1, "the leaf must actually tier up");
    assert_eq!(
        want, got,
        "tier-up vs interpreter diverged in the grown region"
    );
}

#[test]
fn uncommitted_access_traps_on_both_tiers() {
    // Above the committed high-water both tiers must fault: the interpreter by its page map, the
    // emitted leaf by its live-`mapped` bound (synced to 80 KiB < probe). This is the direction an
    // unsynced JIT gets wrong over a reserved window (see the divergence pin below).
    let m = build(GROW_THEN_LEAF);
    let (wasm, eligible) = emit(&m);
    let (want, _) = reserved_frame(&m, None, PROBE_UNCOMMITTED);
    assert_eq!(want, Outcome::Trap(TrapKind::MemoryFault), "oracle sanity");
    let (got, tierups) = reserved_frame(&m, Some((&wasm, &eligible, true)), PROBE_UNCOMMITTED);
    assert_eq!(tierups, 1, "the leaf must actually tier up");
    assert_eq!(
        want, got,
        "tier-up vs interpreter diverged above the high-water"
    );
}

/// The same shape but the guest maps **above a hole** — `[96 KiB, 112 KiB)` with `[64 KiB, 96 KiB)`
/// left uncommitted — so the admitted set is not `[0, H)` for any `H` and `Mem::scalar_extent`
/// returns `None`.
const SPARSE_THEN_LEAF: &str = r#"memory 16
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  vas = i32.wrap_i64 v0
  voff = i64.const 98304
  vlen = i64.const 16384
  vprot = i32.const 3
  vr = cap.call 5 0 (i64, i64, i32) -> (i64) vas (voff, vlen, vprot)
  v2 = call 1 (v1)
  return v2
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vp = i64.const 8
  vaddr = i64.add v0 vp
  i64.store vaddr v0
  vld = i64.load vaddr
  return vld
  }
}
"#;

#[test]
fn sparse_map_declines_tierup_and_stays_correct() {
    // The fail-closed arm of the sync contract: a sparse grow (a hole below the mapped page) is
    // not representable by the single `"mapped"` bound, so the dispatch DECLINES tier-up for the
    // call (tierups == 0) and interprets it — which honors the full page map, keeping the frame
    // observably identical to the oracle. Nothing emitted ever runs over a window state the scalar
    // can't describe.
    let m = build(SPARSE_THEN_LEAF);
    let (wasm, eligible) = emit(&m);
    let probe = 98304 + 16; // inside the sparse-mapped island: interp admits it
    let (want, _) = reserved_frame(&m, None, probe);
    assert_eq!(want, Outcome::Vals(vec![probe]), "oracle sanity");
    let (got, tierups) = reserved_frame(&m, Some((&wasm, &eligible, true)), probe);
    assert_eq!(
        tierups, 0,
        "an unrepresentable window state must decline tier-up entirely"
    );
    assert_eq!(
        want, got,
        "the declined (interpreted) call must match the oracle"
    );
}

#[test]
fn unsynced_jit_diverges_without_host_sync() {
    // The negative pin — the pre-fix driver: same run, but the driver never writes the `"mapped"`
    // global, leaving the emit-time default (the full reservation). The emitted leaf then admits
    // the uncommitted probe the interpreter faults on. This divergence is the bug the #717 host
    // sync exists to close; if it stops reproducing, the emit default changed and the sync
    // contract needs re-deriving.
    let m = build(GROW_THEN_LEAF);
    let (wasm, eligible) = emit(&m);
    let (got, tierups) = reserved_frame(&m, Some((&wasm, &eligible, false)), PROBE_UNCOMMITTED);
    assert_eq!(tierups, 1, "the leaf must actually tier up");
    assert_eq!(
        got,
        Outcome::Vals(vec![PROBE_UNCOMMITTED]),
        "unsynced JIT admits the uncommitted probe (the divergence the sync fixes)"
    );
}
