//! **Grow-then-`Jit.invoke` host sync** (#717, the §22 arm) — the differential proof that a
//! `vm_map`-growing guest and a §22 unit run on the **wasm-JIT codegen tier** agree once the host
//! syncs the emitted unit's `"mapped"` global from the [`VcpuEvent::JitInvoke`] extent snapshot.
//!
//! The tier-up arm of the same contract is `temen-wasm-jit/tests/tierup_grow_window.rs`; this test
//! covers the other seam that runs emitted code over the guest window: the guest grows a
//! **reserved-tail** window through `ADDRESS_SPACE.map` (`cap.call 5 0`), then `Jit.invoke`s a
//! memory-touching unit. The codegen host (the browser Worker's JIT_INVOKE, stood in for by wasmi
//! here) must be observably identical to the interpreted invoke (INVARIANTS.md #9):
//!  - a probe **inside** the grown region round-trips on both tiers (the synced global admits it);
//!  - a probe **above** the committed high-water faults on both (the emitted default after the
//!    reserved-mask bump is the full reservation, so an *unsynced* host would sail through —
//!    pinned as the negative test);
//!  - a **sparse** grow is unrepresentable by the single bound: the event carries `mapped: None`
//!    and the host must decline codegen for that invoke (interpreted delivery), staying correct.

use std::sync::Arc;
use temen_interp::{bytecode, Host, Region, Trap, Value};
use temen_run::grant_jit;
use temen_text::parse_module;
use temen_verify::verify_module;
use wasmi::{Caller, Engine, Linker, Memory, MemoryType, Module as WModule, Store, Val};

const WIN_BASE: u32 = 0x4_0000;
const ENV_PTR: u32 = 1024;
const FUEL: u64 = 100_000_000;

/// The reservation: 128 KiB of address space over the 64-KiB declared (`memory 16`) prefix.
const RESERVED_LOG2: u8 = 17;

/// The invoked unit: `unit(probe)` stores `probe` at `probe + 8`, loads it back, and returns it — a
/// round-trip through window bytes at the probe. Declares the same `memory 16` as the guest (the
/// `jit_compile` memory-match precondition).
const UNIT: &str = r#"memory 16
func (i64) -> (i64) {
block 0 (vp: i64) {
  v8 = i64.const 8
  va = i64.add vp v8
  i64.store va vp
  vl = i64.load va
  return vl
  }
}
"#;

/// Guest `(jit, code, as, probe)`: `map` `[64 KiB, 80 KiB)` of the reserved tail (16 KiB — a whole
/// page on every host page size we run), then `invoke` the unit with the probe.
const GUEST_GROW: &str = r#"memory 16
func (i32, i32, i32, i64) -> (i64) {
block 0 (vjit: i32, vcode: i32, vas: i32, vprobe: i64) {
  voff = i64.const 65536
  vlen = i64.const 16384
  vprot = i32.const 3
  vg = cap.call 5 0 (i64, i64, i32) -> (i64) vas (voff, vlen, vprot)
  vc = i64.extend_i32_u vcode
  vr = cap.call 11 1 (i64, i64) -> (i64) vjit (vc, vprobe)
  return vr
  }
}
"#;

/// The same guest but mapping **above a hole** — `[96 KiB, 112 KiB)` with `[64 KiB, 96 KiB)` left
/// uncommitted — so the admitted set is not `[0, H)` for any `H` (`mapped: None` on the event).
const GUEST_SPARSE: &str = r#"memory 16
func (i32, i32, i32, i64) -> (i64) {
block 0 (vjit: i32, vcode: i32, vas: i32, vprobe: i64) {
  voff = i64.const 98304
  vlen = i64.const 16384
  vprot = i32.const 3
  vg = cap.call 5 0 (i64, i64, i32) -> (i64) vas (voff, vlen, vprot)
  vc = i64.extend_i32_u vcode
  vr = cap.call 11 1 (i64, i64) -> (i64) vjit (vc, vprobe)
  return vr
  }
}
"#;

/// A probe inside the grown region `[64 KiB, 80 KiB)`.
const PROBE_GROWN: i64 = 65536 + 16;
/// A probe above the committed high-water (beyond macOS's 16-KiB page rounding too), inside the
/// reservation — uncommitted on every host page size.
const PROBE_UNCOMMITTED: i64 = 98304 + 16;

/// How the one `JitInvoke` is serviced.
#[derive(Clone, Copy, PartialEq)]
enum Mode {
    /// Interpreted invoke (`deliver_jit_invoke`) — the oracle: the vCPU runs the unit over its own
    /// window, honoring the full page map.
    Interp,
    /// Emitted-wasm invoke with the #717 sync: write the event's `mapped` to the unit instance's
    /// `"mapped"` global, run `f0` over the live window, deliver the slots. `mapped: None` declines
    /// to the interpreted delivery (the browser flattening's `codegen` gate).
    WasmSynced,
    /// Emitted-wasm invoke WITHOUT the sync — the pre-fix host, kept as the divergence pin.
    WasmUnsynced,
}

fn build(src: &str) -> temen_ir::Module {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    m
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

/// Emit the unit for the codegen tier: `size_log2` bumped to the reservation first, so the emitted
/// mask covers the guest's upper address range (the driver convention everywhere) — which also makes
/// the emitted `"mapped"` default the full reservation, exactly why the sync is mandatory here.
fn emit_unit(unit: &temen_ir::Module) -> Vec<u8> {
    let mut e = unit.clone();
    if let Some(mc) = e.memory.as_mut() {
        assert!(mc.size_log2 < RESERVED_LOG2, "test wants a real tail");
        mc.size_log2 = RESERVED_LOG2;
    }
    let a = temen_wasm_jit::compile_jit(&e, temen_wasm_jit::Shape::Batch { entry: 0 }, false)
        .expect("unit emits");
    assert!(a.emitted[0], "the memory-touching unit must be in-subset");
    a.wasm
}

/// Run the emitted unit's `f0(win, env, probe)` under wasmi over a memory **mirrored from the live
/// window**, optionally syncing the `"mapped"` global first, then copy writes back. The native
/// stand-in for the browser Worker's JIT_INVOKE handler.
///
/// SAFETY: `base` is the live window buffer, read/written only while the vCPU is paused inside the
/// invoke service (single-threaded), so no aliasing with the interpreter's own access.
fn run_unit_over_live_window(
    wasm: &[u8],
    argv: &[i64],
    base: *mut u8,
    win_size: usize,
    sync: Option<u64>,
) -> Result<Vec<i64>, Trap> {
    let engine = Engine::default();
    let module = WModule::new(&engine, wasm).expect("emitted unit wasm must validate");
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
                unreachable!("no cross-tier call expected in this unit");
            },
        )
        .unwrap();
    let instance = linker
        .instantiate(&mut store, &module)
        .unwrap()
        .start(&mut store)
        .unwrap();
    if let Some(mapped) = sync {
        // The #717 host-sync contract (worker.js: `unit.mapped.value = ex.temen_par_ev_b(v)`).
        instance
            .get_global(&store, "mapped")
            .expect("emitted unit exports the live-mapped global")
            .set(&mut store, Val::I64(mapped as i64))
            .unwrap();
    }
    let f = instance.get_func(&store, "f0").expect("f0 exported");
    let mut params = vec![Val::I32(WIN_BASE as i32), Val::I32(ENV_PTR as i32)];
    params.extend(argv.iter().map(|&a| Val::I64(a)));
    let mut results = vec![Val::I64(0)];

    let out = match f.call(&mut store, &params, &mut results) {
        Ok(()) => Ok(results
            .iter()
            .map(|v| match v {
                Val::I64(x) => *x,
                Val::I32(x) => *x as i64,
                _ => panic!("non-integer result"),
            })
            .collect()),
        Err(_) => Err(match *store.data() {
            temen_wasm_jit::TRAP_OUT_OF_FUEL => Trap::OutOfFuel,
            _ => Trap::MemoryFault,
        }),
    };
    // SAFETY: see fn doc.
    let back = unsafe { std::slice::from_raw_parts_mut(base, win_size) };
    let mut buf = vec![0u8; win_size];
    memory.read(&store, WIN_BASE as usize, &mut buf).unwrap();
    back.copy_from_slice(&buf);
    out
}

/// Drive `guest(jit, code, as, probe)` over a reserved live window, servicing its one `JitInvoke`
/// in `mode`. Returns the frame outcome, how many invokes ran on emitted wasm, and how many were
/// declined to the interpreter by an unrepresentable extent.
fn run_guest(guest_src: &str, probe: i64, mode: Mode) -> (Result<Vec<Value>, Trap>, u32, u32) {
    let m = build(guest_src);
    let unit_m = build(UNIT);
    let unit_wasm = emit_unit(&unit_m);
    let win_size = 1usize << RESERVED_LOG2;
    let (back, base, layout) = shared_window(win_size);

    let mut host = Host::new();
    let jit = grant_jit(&mut host, &m, 4);
    let asl = host.grant_memory();
    let blob = temen_encode::encode_module(&unit_m);
    let code = host
        .jit_compile(jit, &blob)
        .expect("no trap")
        .expect("compile ok")
        .handle;

    let prog = bytecode::VcpuProgram::compile_with_jit_table(&m, 4).expect("compile guest");
    let args = [
        Value::I32(jit),
        Value::I32(code),
        Value::I32(asl),
        Value::I64(probe),
    ];
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

    let (mut emitted_ran, mut declined) = (0u32, 0u32);
    let out = loop {
        match vcpu.run() {
            bytecode::VcpuEvent::Done(vals) => break Ok(vals),
            bytecode::VcpuEvent::Trapped(t) => break Err(t),
            bytecode::VcpuEvent::JitInvoke {
                handle,
                code,
                argv,
                params: _,
                results: _,
                mapped,
            } => {
                let resolved = {
                    let h = vcpu.host_mut();
                    let domain = h.resolve_jit_domain(handle);
                    domain.and_then(|d| {
                        let (cd, cu) = h.resolve_jit_code(code)?;
                        if cd != d {
                            return Err(Trap::CapFault);
                        }
                        h.jit_unit_funcs(cd, cu).ok_or(Trap::CapFault)
                    })
                };
                // #922: the invoked unit's type section, resolved through the same authority.
                let resolved_types = {
                    let h = vcpu.host_mut();
                    h.resolve_jit_domain(handle)
                        .ok()
                        .and_then(|d| {
                            let (cd, cu) = h.resolve_jit_code(code).ok()?;
                            (cd == d).then_some(())?;
                            h.jit_unit_types(cd, cu)
                        })
                        .unwrap_or_else(|| std::sync::Arc::from(Vec::new()))
                };
                let sync = match mode {
                    Mode::Interp => None,
                    // The browser flattening's `codegen` gate: an unrepresentable extent declines
                    // emitted execution for this invoke (interpreted delivery below).
                    Mode::WasmSynced => mapped,
                    Mode::WasmUnsynced => mapped.map(|_| 1u64 << RESERVED_LOG2), // emit default
                };
                match (mode, sync) {
                    (Mode::Interp, _) | (_, None) => {
                        if mode != Mode::Interp {
                            declined += 1;
                        }
                        vcpu.deliver_jit_invoke(resolved, resolved_types);
                    }
                    (_, Some(s)) => match resolved {
                        Err(t) => vcpu.deliver_jit_invoke_trap(t),
                        Ok(_funcs) => {
                            emitted_ran += 1;
                            let sync = (mode == Mode::WasmSynced).then_some(s);
                            match run_unit_over_live_window(&unit_wasm, &argv, base, win_size, sync)
                            {
                                Ok(slots) => vcpu.deliver_jit_invoke_vals(&slots),
                                Err(t) => vcpu.deliver_jit_invoke_trap(t),
                            }
                        }
                    },
                }
            }
            _ => panic!("unexpected event (this guest only maps + invokes)"),
        }
    };
    drop(vcpu);
    // SAFETY: the vCPU (and its `Mem` aliasing the region) is dropped; free the window buffer.
    unsafe { std::alloc::dealloc(base, layout) };
    (out, emitted_ran, declined)
}

fn as_i64(vals: &[Value]) -> Vec<i64> {
    vals.iter()
        .map(|v| match v {
            Value::I64(x) => *x,
            Value::I32(x) => *x as i64,
            _ => panic!("non-integer result"),
        })
        .collect()
}

#[test]
fn grown_region_invoke_matches_interpreter() {
    // The §22 arm of the #717 symptom: the codegen-run unit round-trips a store/load through the
    // `vm_map`-grown region and must agree with the interpreted invoke (both succeed).
    let (want, _, _) = run_guest(GUEST_GROW, PROBE_GROWN, Mode::Interp);
    assert_eq!(as_i64(&want.expect("oracle ok")), vec![PROBE_GROWN]);
    let (got, emitted_ran, declined) = run_guest(GUEST_GROW, PROBE_GROWN, Mode::WasmSynced);
    assert_eq!(
        emitted_ran, 1,
        "the invoke must actually run on emitted wasm"
    );
    assert_eq!(declined, 0);
    assert_eq!(as_i64(&got.expect("codegen ok")), vec![PROBE_GROWN]);
}

#[test]
fn uncommitted_invoke_traps_on_both_tiers() {
    // Above the committed high-water both tiers fault: the interpreted unit by the page map, the
    // emitted unit by its synced live-`mapped` bound (80 KiB < probe).
    let (want, _, _) = run_guest(GUEST_GROW, PROBE_UNCOMMITTED, Mode::Interp);
    assert_eq!(want, Err(Trap::MemoryFault), "oracle sanity");
    let (got, emitted_ran, _) = run_guest(GUEST_GROW, PROBE_UNCOMMITTED, Mode::WasmSynced);
    assert_eq!(
        emitted_ran, 1,
        "the invoke must actually run on emitted wasm"
    );
    assert_eq!(
        got,
        Err(Trap::MemoryFault),
        "codegen must fault like the oracle"
    );
}

#[test]
fn unsynced_codegen_diverges_without_host_sync() {
    // The negative pin — the pre-fix host: the emitted unit's `"mapped"` stays at its emit-time
    // default (the full reservation), so it admits the uncommitted probe the interpreter faults
    // on. This divergence is what the JitInvoke `mapped` snapshot exists to close.
    let (got, emitted_ran, _) = run_guest(GUEST_GROW, PROBE_UNCOMMITTED, Mode::WasmUnsynced);
    assert_eq!(
        emitted_ran, 1,
        "the invoke must actually run on emitted wasm"
    );
    assert_eq!(
        as_i64(&got.expect("unsynced codegen sails through — the divergence the sync fixes")),
        vec![PROBE_UNCOMMITTED]
    );
}

#[test]
fn sparse_map_declines_codegen_and_stays_correct() {
    // The fail-closed arm: a sparse grow (a hole below the mapped island) makes the extent
    // unrepresentable — the event carries `mapped: None`, the host declines codegen for the
    // invoke (interpreted delivery), and the frame still matches the oracle exactly.
    let probe = 98304 + 16; // inside the sparse-mapped island: the interpreter admits it
    let (want, _, _) = run_guest(GUEST_SPARSE, probe, Mode::Interp);
    assert_eq!(as_i64(&want.expect("oracle ok")), vec![probe]);
    let (got, emitted_ran, declined) = run_guest(GUEST_SPARSE, probe, Mode::WasmSynced);
    assert_eq!(
        emitted_ran, 0,
        "an unrepresentable extent must never reach emitted wasm"
    );
    assert_eq!(declined, 1, "the decline path must actually fire");
    assert_eq!(as_i64(&got.expect("declined invoke ok")), vec![probe]);
}
