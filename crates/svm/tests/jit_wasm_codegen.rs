//! §22 **guest-compiled units on the wasm-JIT tier** — the native differential for the browser's
//! runtime-`Jit.compile` dynamic loop (BROWSER.md § "wasm-JIT tier", DESIGN.md §22).
//!
//! Today the browser only runs a *fixed* §22 unit host-compiled at setup on emitted wasm. The dynamic
//! loop is: a guest builds an IR blob **at runtime**, `Jit.compile`s it, and a later `Jit.invoke` runs
//! that guest's own unit on emitted wasm. This test proves that loop end-to-end **without a browser**:
//! a resumable [`Vcpu`] guest issues `cap.call 11 0` (compile) then `cap.call 11 1` (invoke); compile
//! runs the real [`Host::jit_compile`] gate (decode → verify → memory-match precondition) and the
//! host-injected wasm emitter ([`Host::set_jit_wasm_emitter`]) stashes the unit's wasm; the invoke is
//! serviced by running that emitted `f0(win, env, args…)` under `wasmi` and delivering the result
//! slots ([`Vcpu::deliver_jit_invoke_vals`]) — exactly the seam the browser Worker drives.
//!
//! The oracle (INVARIANTS.md #9) is the same guest with the invoke serviced by the **interpreter**
//! ([`Vcpu::deliver_jit_invoke`]). A codegen run must be byte-identical to it — value *and* trap. The
//! `emitted_ran` guard is the non-vacuity check (as the browser's tier-up counter is): a codegen run
//! that silently fell back to the interpreter fails the test.

use std::sync::Arc;
use svm_interp::{bytecode, Host, Region, Trap, Value};
use svm_ir::{Func, Module, ValType};
use svm_run::grant_jit;
use svm_text::parse_module;
use svm_verify::verify_module;
use wasmi::{Caller, Engine, Linker, Memory, MemoryType, Module as WModule, Store, Val};

/// Where the guest window starts in the wasmi harness's linear memory (page 1 — page 0 holds the env
/// cell, so an emitter bug that misses the `win` rebase lands outside the window and diverges).
const WIN_BASE: u32 = 0x1_0000;
/// The engine-side env cell (the i64 fuel counter) — outside the window, in "engine memory".
const ENV_PTR: u32 = 1024;
/// Where the harness stages the unit blob inside the **guest** window, so the guest's `compile` op
/// reads `(ptr, len)` = `(BLOB_OFF, blob.len())` there. Clear of the tiny guest's own memory use.
const BLOB_OFF: u64 = 0x2000;

// ---- the wasm-JIT emitter the browser tier installs (here: non-shared memory for native wasmi) ----

/// A [`svm_interp::JitWasmEmitter`]: emit the wasm that runs a validated closed unit's entry as
/// `f0(win, env, args…)`, or `None` if it is outside the emitter subset (then `invoke` falls back to
/// the interpreter). A bare `fn`, so the `svm-wasm-jit` dependency stays out of the core.
fn emit_unit(blob: &[u8]) -> Option<Vec<u8>> {
    let m = svm_encode::decode_module(blob).ok()?;
    match svm_wasm_jit::compile_jit(&m, svm_wasm_jit::Shape::Batch { entry: 0 }, false) {
        Ok(svm_wasm_jit::Artifact {
            wasm,
            drive: svm_wasm_jit::DriveMode::WasmDriven { .. },
            ..
        }) => Some(wasm),
        // Interp-driven (or unsupported) ⇒ there is nothing to run as `f0`; fail closed to the interp.
        _ => None,
    }
}

// ---- units the guest compiles at runtime ---------------------------------------------------------

/// `unit(a, b) = a*3 + b` — pure i64 compute (`unit(4, 5) = 17`).
const UNIT_I64: &str = r#"memory 16
func (i64, i64) -> (i64) {
block 0 (va: i64, vb: i64) {
  v3 = i64.const 3
  vm = i64.mul va v3
  vr = i64.add vm vb
  return vr
  }
}
"#;

/// The i32 twin (`unit(4, 5) = 17`) — pins i32 arg/result marshalling through the invoke seam.
const UNIT_I32: &str = r#"memory 16
func (i32, i32) -> (i32) {
block 0 (va: i32, vb: i32) {
  v3 = i32.const 3
  vm = i32.mul va v3
  vr = i32.add vm vb
  return vr
  }
}
"#;

/// `unit(x) = 100 / (x - 4)` — div-by-zero at `x == 4` (the invoke harness's first sample arg, so the
/// one-param invoke hits it). Proves a codegen-run trap surfaces exactly where the interpreter would.
const UNIT_TRAP: &str = r#"memory 16
func (i64) -> (i64) {
block 0 (vx: i64) {
  v4 = i64.const 4
  vd = i64.sub vx v4
  v100 = i64.const 100
  vr = i64.div_s v100 vd
  return vr
  }
}
"#;

/// A guest that `compile`s the blob staged at `(off, len)` then `invoke`s the resulting unit. `RET` is
/// the unit's result type spelling (`i64`/`i32`); `sig` the invoke cap.call signature after the leading
/// code handle; `args` the invoke argument instructions + operand names.
fn guest_src(ret: &str, sig: &str, arg_setup: &str, arg_ops: &str, off: u64, len: u64) -> String {
    format!(
        r#"memory 16
func (i32) -> ({ret}) {{
block 0 (vjit: i32) {{
  vptr = i64.const {off}
  vlen = i64.const {len}
  vcode = cap.call 11 0 (i64, i64) -> (i64) vjit (vptr, vlen)
{arg_setup}
  vr = cap.call 11 1 ({sig}) -> ({ret}) vjit (vcode, {arg_ops})
  return vr
  }}
}}
"#
    )
}

// ---- wasmi execution of an emitted unit (the native stand-in for the browser Worker) -------------

fn slot_to_val(t: ValType, s: i64) -> Val {
    match t {
        ValType::I32 => Val::I32(s as i32),
        ValType::I64 => Val::I64(s),
        _ => panic!("slice-1 units are integer-only"),
    }
}

fn val_to_slot(v: &Val) -> i64 {
    match v {
        Val::I32(x) => *x as i64,
        Val::I64(x) => *x,
        _ => panic!("non-integer unit result"),
    }
}

/// Map a wasmi trap back to the SVM trap taxonomy — the SVM-specific kinds arrive through `env.trap`
/// (fuel / memory fault), wasm's own kinds from the trap code.
fn map_trap(host_code: i32, e: &wasmi::Error) -> Trap {
    use svm_wasm_jit::{TRAP_MEMORY_FAULT, TRAP_OUT_OF_FUEL};
    if host_code == TRAP_OUT_OF_FUEL {
        return Trap::OutOfFuel;
    }
    if host_code == TRAP_MEMORY_FAULT {
        return Trap::MemoryFault;
    }
    use wasmi::core::TrapCode;
    match e.as_trap_code() {
        Some(TrapCode::IntegerDivisionByZero) => Trap::DivByZero,
        Some(TrapCode::IntegerOverflow) => Trap::IntOverflow,
        Some(TrapCode::BadConversionToInteger) => Trap::BadConversion,
        _ => Trap::Unreachable,
    }
}

/// Run the emitted unit's `f0(win, env, argv…)` under wasmi over a fresh window, returning its result
/// slots (raw i64) or the trap. The units here are pure compute, so the window is unused — a
/// memory-touching unit would share the guest window in (a later slice covers that).
fn run_unit_on_wasm(unit: &Module, wasm: &[u8], argv: &[i64]) -> Result<Vec<i64>, Trap> {
    let engine = Engine::default();
    let module = WModule::new(&engine, wasm).expect("emitted unit wasm must validate");
    let mut store: Store<i32> = Store::new(&engine, 0);
    let pages = 2 + unit.memory.map_or(0, |mc| (mc.size() >> 16) as u32);
    let memory = Memory::new(&mut store, MemoryType::new(pages, None)).unwrap();
    // Ample fuel — these kernels never exhaust it.
    memory
        .write(&mut store, ENV_PTR as usize, &i64::MAX.to_le_bytes())
        .unwrap();
    let mut linker: Linker<i32> = Linker::new(&engine);
    linker.define("env", "memory", memory).unwrap();
    linker
        .func_wrap("env", "trap", |mut caller: Caller<'_, i32>, code: i32| {
            *caller.data_mut() = code;
        })
        .unwrap();
    linker
        .func_wrap::<_, ()>(
            "env",
            "call_interp",
            |_: Caller<'_, i32>, _func: i32, _args: i32| {
                unreachable!("no cross-tier call in a pure-compute unit");
            },
        )
        .unwrap();
    let instance = linker
        .instantiate(&mut store, &module)
        .unwrap()
        .start(&mut store)
        .unwrap();
    let f = instance.get_func(&store, "f0").expect("f0 exported");

    let mut params = vec![Val::I32(WIN_BASE as i32), Val::I32(ENV_PTR as i32)];
    for (t, s) in unit.funcs[0].params.iter().zip(argv) {
        params.push(slot_to_val(*t, *s));
    }
    let mut results: Vec<Val> = unit.funcs[0]
        .results
        .iter()
        .map(|t| match t {
            ValType::I32 => Val::I32(0),
            _ => Val::I64(0),
        })
        .collect();

    match f.call(&mut store, &params, &mut results) {
        Ok(()) => Ok(results.iter().map(val_to_slot).collect()),
        Err(e) => Err(map_trap(*store.data(), &e)),
    }
}

// ---- shared plumbing -----------------------------------------------------------------------------

/// A fresh 64 KiB window (= `memory 16`) for a root vCPU. Leaked for the test's lifetime.
fn window() -> Arc<Region> {
    let size = 1usize << 16;
    let layout = std::alloc::Layout::from_size_align(size, 8).unwrap();
    // SAFETY: non-zero 8-aligned layout; leaked for the process — never freed, so no aliasing.
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!base.is_null());
    // SAFETY: `size` valid 8-aligned bytes, owned here and never freed.
    Arc::new(unsafe { Region::shared(base, size as u64) })
}

/// A resolved unit: its validated funcs and (when the wasm emitter fired) its emitted wasm.
type ResolvedUnit = (Arc<[Func]>, Option<Arc<[u8]>>);

/// Resolve a `JitInvoke` event's authority against the vCPU's own host (compile minted the handle
/// there) — mirroring the browser's `par_resolve_unit` — and hand back the unit's funcs + emitted wasm.
fn resolve(h: &Host, handle: i32, code: i32) -> Result<ResolvedUnit, Trap> {
    let domain = h.resolve_jit_domain(handle)?;
    let (cd, cu) = h.resolve_jit_code(code)?;
    if cd != domain {
        return Err(Trap::CapFault);
    }
    let funcs = h.jit_unit_funcs(cd, cu).ok_or(Trap::CapFault)?;
    Ok((funcs, h.jit_unit_wasm(cd, cu)))
}

/// How a `JitInvoke` is serviced.
#[derive(Clone, Copy, PartialEq)]
enum Mode {
    /// The engine runs the unit on the bytecode interpreter — the oracle.
    Interp,
    /// The host runs the unit on emitted wasm (`wasmi`) and delivers the result slots — the tier.
    Wasm,
}

/// Drive the single-vCPU guest to completion: it runtime-`compile`s `unit_src` (staged in its window),
/// then `invoke`s it once, serviced in `mode`.
fn run_guest(unit_src: &str, mode: Mode) -> Result<Vec<Value>, Trap> {
    let unit_m = parse_module(unit_src).unwrap();
    verify_module(&unit_m).expect("verify unit");
    let blob = svm_encode::encode_module(&unit_m);

    // Build the guest around the unit's shape (its entry sig is `funcs[0]`).
    let entry = &unit_m.funcs[0];
    let ret = tyname(entry.results[0]);
    let (arg_setup, arg_ops, invoke_sig) = invoke_shape(&entry.params);
    let guest_src = guest_src(
        ret,
        &invoke_sig,
        &arg_setup,
        &arg_ops,
        BLOB_OFF,
        blob.len() as u64,
    );
    let guest_m =
        parse_module(&guest_src).unwrap_or_else(|e| panic!("guest parse: {e}\n{guest_src}"));
    verify_module(&guest_m).expect("verify guest");

    let mut host = Host::new();
    let jit = grant_jit(&mut host, &guest_m, 4); // 2^4 = 16-slot table; mem_log2 from `memory 16`
    if mode == Mode::Wasm {
        host.set_jit_wasm_emitter(emit_unit);
    }

    // Stage the blob at `BLOB_OFF` in the window so the guest's `compile (ptr, len)` reads it there.
    let mut init_mem = vec![0u8; BLOB_OFF as usize];
    init_mem.extend_from_slice(&blob);

    let prog = bytecode::VcpuProgram::compile_with_jit_table(&guest_m, 4).unwrap();
    let back = window();
    let mut vcpu =
        bytecode::Vcpu::new_root_with_powerbox(&prog, 0, &[Value::I32(jit)], back, &init_mem, host)
            .expect("root");

    let mut emitted_ran = false;
    loop {
        match vcpu.run() {
            bytecode::VcpuEvent::Done(vals) => {
                if mode == Mode::Wasm {
                    assert!(
                        emitted_ran,
                        "codegen run never executed emitted wasm (silent interp fallback)"
                    );
                }
                return Ok(vals);
            }
            bytecode::VcpuEvent::Trapped(t) => return Err(t),
            bytecode::VcpuEvent::JitInvoke {
                handle,
                code,
                argv,
                params: _,
                results: _,
            } => {
                let resolved = resolve(vcpu.host_mut(), handle, code);
                match mode {
                    Mode::Interp => vcpu.deliver_jit_invoke(resolved.map(|(f, _)| f)),
                    Mode::Wasm => match resolved {
                        Err(t) => vcpu.deliver_jit_invoke_trap(t),
                        Ok((_funcs, wasm)) => {
                            let wasm =
                                wasm.expect("emitter must produce wasm for an in-subset unit");
                            emitted_ran = true;
                            match run_unit_on_wasm(&unit_m, &wasm, &argv) {
                                Ok(slots) => vcpu.deliver_jit_invoke_vals(&slots),
                                Err(t) => vcpu.deliver_jit_invoke_trap(t),
                            }
                        }
                    },
                }
            }
            _ => panic!("unexpected event (this guest only compiles + invokes)"),
        }
    }
}

fn tyname(t: ValType) -> &'static str {
    match t {
        ValType::I32 => "i32",
        ValType::I64 => "i64",
        _ => panic!("slice-1 units are integer-only"),
    }
}

/// Build the invoke-site pieces for a unit whose entry takes `params`: the const instructions that
/// materialize the arguments (`va = 4`, `vb = 5`, …), the operand list, and the cap.call signature
/// (a leading `i64` code handle, then the unit's params). Fixed sample arguments 4, 5, 6, … per param.
fn invoke_shape(params: &[ValType]) -> (String, String, String) {
    let mut setup = String::new();
    let mut ops = Vec::new();
    let mut sig = vec!["i64".to_string()]; // the code handle
    for (i, t) in params.iter().enumerate() {
        let name = format!("varg{i}");
        let val = 4 + i as i64;
        setup.push_str(&format!("  {name} = {}.const {val}\n", tyname(*t)));
        ops.push(name);
        sig.push(tyname(*t).to_string());
    }
    (setup, ops.join(", "), sig.join(", "))
}

// ---- the Host-level mechanism check (no vCPU) ----------------------------------------------------

/// The interpreter oracle for a unit run directly over `argv` slots.
fn oracle_unit(unit: &Module, argv: &[i64]) -> Result<Vec<i64>, Trap> {
    let args: Vec<Value> = unit.funcs[0]
        .params
        .iter()
        .zip(argv)
        .map(|(t, s)| match t {
            ValType::I32 => Value::I32(*s as i32),
            _ => Value::I64(*s),
        })
        .collect();
    let mut fuel = u64::MAX;
    svm_interp::run(unit, 0, &args, &mut fuel).map(|vals| {
        vals.iter()
            .map(|v| match v {
                Value::I32(x) => *x as i64,
                Value::I64(x) => *x,
                _ => panic!("non-integer unit result"),
            })
            .collect()
    })
}

/// `Host::jit_compile` + the emitter hook stash wasm that runs identically to the interpreter — the
/// mechanism in isolation, over the same compile gate a runtime guest reaches.
#[test]
fn compile_stashes_wasm_matching_the_interpreter() {
    for (name, src, argv) in [
        ("i64_muladd", UNIT_I64, vec![4i64, 5]),
        ("i32_muladd", UNIT_I32, vec![4i64, 5]),
    ] {
        let unit_m = parse_module(src).unwrap();
        verify_module(&unit_m).expect("verify unit");
        let blob = svm_encode::encode_module(&unit_m);

        let mut host = Host::new();
        let jit = grant_jit(&mut host, &unit_m, 4); // the unit *is* the parent here (memory-match)
        host.set_jit_wasm_emitter(emit_unit);
        let c = host
            .jit_compile(jit, &blob)
            .expect("no trap")
            .expect("compile ok");
        let wasm = host
            .jit_unit_wasm(c.domain, c.unit)
            .unwrap_or_else(|| panic!("{name}: emitter hook produced no wasm"));

        let want = oracle_unit(&unit_m, &argv);
        let got = run_unit_on_wasm(&unit_m, &wasm, &argv);
        assert_eq!(
            want, got,
            "{name}: emitted wasm diverged from the interpreter"
        );
    }
}

// ---- the end-to-end runtime-compile → invoke loop ------------------------------------------------

#[test]
fn runtime_compile_invoke_i64_matches_interp() {
    let want = run_guest(UNIT_I64, Mode::Interp);
    let got = run_guest(UNIT_I64, Mode::Wasm);
    assert_eq!(
        want,
        Ok(vec![Value::I64(17)]),
        "interp reference: unit(4,5)=17"
    );
    assert_eq!(
        got, want,
        "wasm-emitted runtime-compiled i64 unit diverged from interp"
    );
}

#[test]
fn runtime_compile_invoke_i32_matches_interp() {
    let want = run_guest(UNIT_I32, Mode::Interp);
    let got = run_guest(UNIT_I32, Mode::Wasm);
    assert_eq!(
        want,
        Ok(vec![Value::I32(17)]),
        "interp reference: unit(4,5)=17"
    );
    assert_eq!(
        got, want,
        "wasm-emitted runtime-compiled i32 unit diverged from interp"
    );
}

#[test]
fn runtime_compile_invoke_trap_parity() {
    // unit(7) = 100/(7-7) → div-by-zero; the codegen seam must trap exactly where the interp does.
    let want = run_guest(UNIT_TRAP, Mode::Interp);
    let got = run_guest(UNIT_TRAP, Mode::Wasm);
    assert!(want.is_err(), "interp invoke must trap at x=7");
    assert_eq!(
        got, want,
        "codegen trap diverged from interp (want {want:?})"
    );
}

// ---- the shared-`Mutex<Host>` runtime-compile flow (the browser's `ParJitCfg` path) --------------
// The browser's runtime-`Jit.compile` powerbox grants `Jit` (+ validator + emitter) on a shared
// `Mutex<Host>` that the root vCPU dispatches its `cap.call`s through (`with_shared_host`) — so the
// guest's runtime `compile` mints + emits into that shared host and a later `invoke` resolves the same
// host. `run_guest` above uses the vCPU's *own* host; this exercises the shared-host dispatch the
// browser actually uses (also the seam threaded compile will serialize on, DESIGN.md §22).

/// Like [`run_guest`], but the `Jit` grant + emitter live in a shared `Mutex<Host>` the vCPU
/// dispatches through (`with_shared_host`), and the invoke resolves against that shared host —
/// mirroring the browser's `svm_par_powerbox_jit_runtime` + `par_resolve_unit_rt`.
fn run_guest_shared(unit_src: &str, mode: Mode) -> Result<Vec<Value>, Trap> {
    let unit_m = parse_module(unit_src).unwrap();
    verify_module(&unit_m).expect("verify unit");
    let blob = svm_encode::encode_module(&unit_m);

    let entry = &unit_m.funcs[0];
    let ret = tyname(entry.results[0]);
    let (arg_setup, arg_ops, invoke_sig) = invoke_shape(&entry.params);
    let guest_src = guest_src(
        ret,
        &invoke_sig,
        &arg_setup,
        &arg_ops,
        BLOB_OFF,
        blob.len() as u64,
    );
    let guest_m = parse_module(&guest_src).unwrap_or_else(|e| panic!("guest parse: {e}"));
    verify_module(&guest_m).expect("verify guest");

    // The shared powerbox host (the browser's `ParJitCfg.host`): `Jit` granted, validator + emitter set.
    let mut host = Host::new();
    let jit = grant_jit(&mut host, &guest_m, 4);
    if mode == Mode::Wasm {
        host.set_jit_wasm_emitter(emit_unit);
    }
    let shared = std::sync::Mutex::new(host);

    let mut init_mem = vec![0u8; BLOB_OFF as usize];
    init_mem.extend_from_slice(&blob);

    let prog = bytecode::VcpuProgram::compile_with_jit_table(&guest_m, 4).unwrap();
    let back = window();
    // `new_root` builds an empty *own* host; the shared host services generic `cap.call`s — including
    // the guest's runtime `compile` (op 0), which mints + emits into it.
    let mut vcpu = bytecode::Vcpu::new_root(&prog, 0, &[Value::I32(jit)], back, &init_mem)
        .expect("root")
        .with_shared_host(&shared);

    let mut emitted_ran = false;
    loop {
        match vcpu.run() {
            bytecode::VcpuEvent::Done(vals) => {
                if mode == Mode::Wasm {
                    assert!(
                        emitted_ran,
                        "codegen run never executed emitted wasm (silent interp fallback)"
                    );
                }
                return Ok(vals);
            }
            bytecode::VcpuEvent::Trapped(t) => return Err(t),
            bytecode::VcpuEvent::JitInvoke {
                handle, code, argv, ..
            } => {
                // Resolve against the shared host (browser `par_resolve_unit_rt`); drop the lock before
                // delivering — the interpreter path re-locks it to run the unit.
                let resolved = {
                    let g = shared.lock().unwrap();
                    resolve(&g, handle, code)
                };
                match mode {
                    Mode::Interp => vcpu.deliver_jit_invoke(resolved.map(|(f, _)| f)),
                    Mode::Wasm => match resolved {
                        Err(t) => vcpu.deliver_jit_invoke_trap(t),
                        Ok((_funcs, wasm)) => {
                            let wasm =
                                wasm.expect("emitter must produce wasm for an in-subset unit");
                            emitted_ran = true;
                            match run_unit_on_wasm(&unit_m, &wasm, &argv) {
                                Ok(slots) => vcpu.deliver_jit_invoke_vals(&slots),
                                Err(t) => vcpu.deliver_jit_invoke_trap(t),
                            }
                        }
                    },
                }
            }
            _ => panic!("unexpected event (this guest only compiles + invokes)"),
        }
    }
}

#[test]
fn shared_host_runtime_compile_matches_interp() {
    let want = run_guest_shared(UNIT_I64, Mode::Interp);
    let got = run_guest_shared(UNIT_I64, Mode::Wasm);
    assert_eq!(
        want,
        Ok(vec![Value::I64(17)]),
        "interp reference: unit(4,5)=17"
    );
    assert_eq!(
        got, want,
        "shared-host wasm-emitted runtime-compiled unit diverged from interp"
    );
}

#[test]
fn shared_host_runtime_compile_trap_parity() {
    let want = run_guest_shared(UNIT_TRAP, Mode::Interp);
    let got = run_guest_shared(UNIT_TRAP, Mode::Wasm);
    assert!(want.is_err(), "interp invoke must trap");
    assert_eq!(
        got, want,
        "shared-host codegen trap diverged from interp (want {want:?})"
    );
}
