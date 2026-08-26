//! Generative interpreter-vs-**wasm-JIT** differential (DESIGN.md §18; issue #910). The wasm-JIT's
//! `emit_confine`/`emit_span_check` (`temen-wasm-jit`) is the tree's *third* confinement lowering, and
//! until now it had only hand-written kernels (`temen-wasm-jit/tests/differential.rs`) — no generative
//! coverage, unlike the Cranelift lowering which the `diff`/`jit_fuzz` generator drives. INVARIANTS #2
//! calls masking "the fuzzed hinge"; this closes the asymmetry for the wasm tier.
//!
//! It reuses the **shared irgen generator** (`irgen.rs`) and the **wasmi harness plumbing** (mirroring
//! `temen-wasm-jit/tests/differential.rs` + `entry_data.rs`): a generated verifier-valid module is
//! emitted with [`temen_wasm_jit::compile_module`] and run under `wasmi` against the tree-walk oracle
//! (INVARIANTS #9), comparing the result, that both terminate, and — for a float-free module with
//! memory — the **final window byte-for-byte** (the escape-oracle: a mis-masked access would perturb
//! it). Shared by the stable `wasm_diff` test and the libFuzzer `wasm_diff` target, exactly as
//! `irgen.rs` is shared by `jit_fuzz`/`diff`.
//!
//! Scope (kept deliberately narrow, matching the wasmi ABI the harness already proves correct):
//! - **entry signature ⊆ {i32, i64}** — the guest-arg / result marshalling the harness supports; a
//!   float/v128 *entry* type is skipped (its body ops are still covered by the Cranelift generator).
//!   Memory confinement lives in the body (loads/stores/bulk/atomics), independent of the entry types.
//! - modules the wasm tier **refuses** (`Unsupported`: any out-of-integer-subset op — call.cap, page
//!   ops, fiber/threads, fma, deferred SIMD) are skipped, exactly as the Cranelift path skips its
//!   `JitError::Unsupported`.
//! - a **read-only** data segment is skipped: the wasmi harness has no `PROT_NONE`/RO enforcement, so
//!   an interp RO-store fault has no wasm analogue (a false divergence, not a miscompile — the
//!   Cranelift path covers RO via real `mprotect`).
//!
//! Agreement policy mirrors the Cranelift `assert_outcomes_agree`: both-terminate agree (the trap
//! *kind* is not asserted — a block with several reachable traps lets eager-vs-emitted surface
//! different ones, and the IR pins neither order nor choice), while the strict checks stay strict —
//! a clean interp run must match the wasm value **and** window, and a *non-droppable* interp trap must
//! never let the wasm return a value.

#![allow(dead_code)] // each includer (test / fuzz target) uses a subset

#[path = "irgen.rs"]
mod irgen;

use irgen::{gen_args, values_equal};
pub use irgen::{gen_module, has_float, Gen};

use temen_interp::{run_capture_reserved_with_host, Host, Trap, Value};
use temen_ir::{Module, ValType};
use temen_verify::verify_module;
use wasmi::{Caller, Config, Engine, Linker, Memory, MemoryType, Module as WModule, Store, Val};

/// Where the guest window starts in the harness's linear memory (page 1 — page 0 holds the env cell,
/// so an emitter bug that misses the `win` rebase lands *outside* the window and diverges). Same
/// layout as `temen-wasm-jit/tests/differential.rs`.
const WIN_BASE: u32 = 0x1_0000;
/// The engine-side env cell (the i64 fuel counter), outside the window in "engine memory".
const ENV_PTR: u32 = 1024;
/// Ample fuel for a halt-by-construction generated module (never exhausts here; OutOfFuel parity is
/// the Cranelift path's `jit_matches_interp_under_tight_fuel` job).
const FUEL: u64 = 5_000_000;

/// A pure-op trap the optimizing lowering may drop when its result is dead (so the eager interpreter
/// takes it but the wasm need not) — the same latitude as the Cranelift `droppable_trap`.
fn droppable(t: &Trap) -> bool {
    matches!(t, Trap::DivByZero | Trap::IntOverflow | Trap::BadConversion)
}

/// Modeled traps, matching the Cranelift oracle's `interp_trap_kind` exactly — a trap outside this
/// set is *unmodeled* and not held against a wasm that returned. **`MemoryFault` is deliberately
/// excluded** (as it is from `interp_trap_kind`): a load/store whose result is dead can be DCE'd by
/// the optimizing lowering, so the eager interpreter faults where the wasm need not — and a *real*
/// confinement escape is caught by the final-window byte-compare (the `mem_oracle` branch below), not
/// by trap-kind agreement. Of the modeled set, only the non-droppable ones (`Unreachable`,
/// `IndirectCallType`) can actually reach a wasm-accepted module (cap/fuel traps do not occur here),
/// and those must never let the wasm return a value.
fn modeled(t: &Trap) -> bool {
    matches!(
        t,
        Trap::DivByZero
            | Trap::IntOverflow
            | Trap::BadConversion
            | Trap::Unreachable
            | Trap::IndirectCallType
            | Trap::CapFault
            | Trap::OutOfFuel
    )
}

/// What one engine produced under the wasm run: values, or that it trapped (the kind is unobservable
/// here and deliberately not distinguished — see the module-level agreement-policy note).
enum WasmOutcome {
    Vals(Vec<Value>),
    Trapped,
}

/// Every entry param/result is a type the wasmi arg/result marshalling supports (`i32`/`i64`).
fn int_entry_sig(m: &Module) -> bool {
    let int = |t: &ValType| matches!(t, ValType::I32 | ValType::I64);
    m.funcs[0].params.iter().all(int) && m.funcs[0].results.iter().all(int)
}

/// Run the emitted wasm's `f0` under `wasmi`: window seeded with `init` then the module's data
/// segments (what the interpreter's instantiation does, and what the browser/bench linkers do —
/// `entry_data.rs`), fuel in the `"fuel"` global. Returns the outcome and, on a clean return, the
/// final window bytes `[WIN_BASE, WIN_BASE+size)` for the escape-oracle byte-compare.
fn wasm_run(m: &Module, wasm: &[u8], args: &[Value], init: &[u8]) -> (WasmOutcome, Vec<u8>) {
    let size = m.memory.map_or(0, |mc| mc.size()) as usize;
    // wasmi gates the §17 v128 proposal behind the `simd` crate feature + this toggle; the emitter's
    // integer subset includes the core v128 lane ops, so a generated module can emit SIMD.
    let mut config = Config::default();
    config.wasm_simd(true);
    let engine = Engine::new(&config);
    let module = WModule::new(&engine, wasm).expect("emitted wasm must validate");
    let mut store: Store<i32> = Store::new(&engine, 0);
    // page 0 (env) + the window pages (window is `size` bytes at WIN_BASE = page 1).
    let pages = 2 + (size >> 16) as u32;
    let memory = Memory::new(&mut store, MemoryType::new(pages, None)).unwrap();
    memory
        .write(&mut store, ENV_PTR as usize, &(FUEL as i64).to_le_bytes())
        .unwrap();
    // Window init: the escape-oracle seed, then each data segment laid at WIN_BASE + offset (the
    // interpreter's `run_capture_*` applies both over the window before the run).
    if !init.is_empty() {
        memory.write(&mut store, WIN_BASE as usize, init).unwrap();
    }
    for seg in &m.data {
        memory
            .write(
                &mut store,
                WIN_BASE as usize + seg.offset as usize,
                &seg.bytes,
            )
            .unwrap();
    }
    let mut linker: Linker<i32> = Linker::new(&engine);
    linker.define("env", "memory", memory).unwrap();
    linker
        .func_wrap("env", "trap", |mut caller: Caller<'_, i32>, code: i32| {
            *caller.data_mut() = code;
        })
        .unwrap();
    // A fully in-subset module has no interp leaf, so `call_interp` is imported but never called.
    linker
        .func_wrap::<_, ()>(
            "env",
            "call_interp",
            |_: Caller<'_, i32>, _func: i32, _args: i32| {
                unreachable!("no cross-tier call in a fully in-subset module");
            },
        )
        .unwrap();
    let instance = linker
        .instantiate(&mut store, &module)
        .unwrap()
        .start(&mut store)
        .unwrap();
    if let Some(g) = instance.get_global(&store, "fuel") {
        g.set(&mut store, Val::I64(FUEL as i64)).unwrap();
    }
    let f = instance.get_func(&store, "f0").expect("f0 exported");

    let mut params = vec![Val::I32(WIN_BASE as i32), Val::I32(ENV_PTR as i32)];
    for (t, a) in m.funcs[0].params.iter().zip(args) {
        params.push(match (t, a) {
            (ValType::I32, Value::I32(v)) => Val::I32(*v),
            (ValType::I64, Value::I64(v)) => Val::I64(*v),
            _ => panic!("int_entry_sig should have filtered non-int entry args"),
        });
    }
    let mut results: Vec<Val> = m.funcs[0]
        .results
        .iter()
        .map(|t| match t {
            ValType::I32 => Val::I32(0),
            _ => Val::I64(0),
        })
        .collect();

    match f.call(&mut store, &params, &mut results) {
        Ok(()) => {
            let vals = results
                .iter()
                .map(|v| match v {
                    Val::I32(x) => Value::I32(*x),
                    Val::I64(x) => Value::I64(*x),
                    _ => panic!("non-integer result"),
                })
                .collect();
            // Read the final window back for the byte-compare (only used when mem_oracle).
            let mut win = vec![0u8; size];
            if size != 0 {
                memory.read(&store, WIN_BASE as usize, &mut win).unwrap();
            }
            (WasmOutcome::Vals(vals), win)
        }
        Err(_) => (WasmOutcome::Trapped, Vec::new()),
    }
}

/// Generate → verify → run the entry on the tree-walk oracle and the wasm-JIT under `wasmi`, and
/// assert they agree (see the module-level policy). Skips modules the wasm tier refuses or whose
/// entry/data shape the harness does not model. The single entry point shared by the stable seed
/// loop and the libFuzzer target.
pub fn fuzz_one_wasm(g: &mut Gen) {
    let m = gen_module(g);
    verify_module(&m).unwrap_or_else(|e| panic!("generator emitted invalid IR: {e:?}\n{m:#?}"));
    let params = m.funcs[0].params.clone();
    let args = gen_args(g, &params);
    run_differential_wasm(&m, &args);
}

/// The comparison itself, factored out so the stable regression pins can call it directly on a module.
pub fn run_differential_wasm(m: &Module, args: &[Value]) {
    // Only int-entry modules (the marshalled ABI), and only ones with no read-only data segment
    // (no RO enforcement in the wasmi harness), are in scope.
    if !int_entry_sig(m) || m.data.iter().any(|d| d.readonly) {
        return;
    }
    // Emit; the tier refusing a module (out-of-integer-subset op) is a skip, not a divergence —
    // exactly as the Cranelift path skips `JitError::Unsupported`.
    let wasm = match temen_wasm_jit::compile_module(m) {
        Ok(w) => w,
        Err(_) => return,
    };

    // Escape-oracle seed over the window, so a divergent or under-masked load/store shows up as a
    // final-memory mismatch (zero-init could hide a bad read that returns 0). Only for float-free
    // memory modules — the IR does not pin NaN bit-patterns across backends (`has_float`).
    let mem_oracle = !has_float(m) && m.memory.is_some();
    let init: Vec<u8> = match m.memory {
        Some(mc) if mem_oracle => {
            let size = mc.size() as usize;
            (0..size)
                .map(|i| (i as u8).wrapping_mul(31) ^ 0xa5)
                .collect()
        }
        _ => Vec::new(),
    };

    // Oracle: tree-walk interpreter, fully-mapped window (`reserved_log2 = 0`), no caps granted (an
    // accepted module has no call.cap). Returns the result and the final window.
    let mut fuel = FUEL;
    let mut host = Host::new();
    let (interp, imem) = run_capture_reserved_with_host(m, 0, args, &mut fuel, &init, 0, &mut host);

    let (wasm_out, wmem) = wasm_run(m, &wasm, args, &init);

    match (interp, wasm_out) {
        (Ok(want), WasmOutcome::Vals(got)) => {
            assert_eq!(want.len(), got.len(), "result arity\n{m:#?}");
            for (w, g) in want.iter().zip(&got) {
                assert!(
                    values_equal(w, g),
                    "interp/wasm-JIT disagree: {want:?} vs {got:?}\n{m:#?}"
                );
            }
            // Escape-oracle: both ran to completion, so the interpreter (the masking reference, §4)
            // confined every access to `[0, size)`; the wasm tier, lowering the same confinement on
            // the same inputs, must leave an identical window. A mismatch is an escaped / mis-masked
            // access.
            if mem_oracle {
                assert_eq!(imem.len(), wmem.len(), "window snapshot length\n{m:#?}");
                if let Some(i) = imem.iter().zip(&wmem).position(|(a, b)| a != b) {
                    panic!(
                        "escape-oracle: interp/wasm-JIT final memory differs at byte {i} \
                         (interp={:#04x} wasm={:#04x}) — an access not masked into [0,size)\n{m:#?}",
                        imem[i], wmem[i]
                    );
                }
            }
        }
        // Both terminate the guest — agreement at the level that matters (trap kind not asserted).
        (Err(_), WasmOutcome::Trapped) => {}
        // A droppable pure-op trap (div/rem-by-zero, int-overflow, bad convert) may be DCE'd, so the
        // wasm can return where the eager interpreter trapped; an *unmodeled* interp trap is likewise
        // not held against it. A non-droppable modeled trap must never let the wasm return a value.
        (Err(trap), WasmOutcome::Vals(_)) => assert!(
            droppable(&trap) || !modeled(&trap),
            "interp trapped {trap:?} but wasm-JIT returned\n{m:#?}"
        ),
        (Ok(_), WasmOutcome::Trapped) => {
            panic!("interp returned but wasm-JIT trapped\n{m:#?}")
        }
    }
}
