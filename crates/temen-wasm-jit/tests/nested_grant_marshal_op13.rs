//! **§14 op-13 grant *marshaling* across the emitted bounce** (#1025 slice 3a — the confinement core).
//! The sibling `nested_grant_op13_e2e.rs` proved a *granted* child runs on emitted wasm, but it delivered
//! the grant by **pre-wiring** the child host into `env.call_interp` — the op-13 bounce's `grants_ptr`/
//! `grants_n` were ignored. This file closes that gap: the `env.instantiate_module` servicer **reads the
//! guest's grant records out of the parent window** and re-grants them from the parent's powerbox via
//! [`Host::spawn_named_child_from_window`], so the child's `fs` authority arrives *through the bounce* —
//! exactly the marshaling the native Cranelift path (`grant_named_child_build`) does, now on the wasm tier.
//!
//! The child returns `40 + granted_counter()` = `41`; the shared counter ticks once — the observable proof
//! the *marshaled* (not pre-wired) authority ran inside the confined, emitted child. The servicer is the
//! **faithful production shape**: it composes the two independent fail-closed axes (INVARIANTS §2) — the
//! **carve gate** (`check_child_carve`: window confinement — aligned, in-parent, clears the NULL guard,
//! sized ≥ declared) and the **grant marshal** (authority). Refusals on either axis (an undersized or
//! misaligned carve; a forged handle or out-of-window record) run no child and leave the counter at 0.

use std::sync::{Arc, Mutex};

use temen_interp::{bytecode, ForkedProc, GrantMarshalError, Host, HostProc, Region, Value};
use temen_wasm_jit::{
    check_child_carve, compile_module_nested, compile_module_nested_with_eligibility,
};
use wasmi::{Caller, Engine, Func, Linker, Memory, MemoryType, Module as WModule, Store, Val};

const WIN_BASE: i32 = 0x1_0000; // parent window base (the env cell lives below it)
const WIN_SIZE: u64 = 1 << 16; // parent `memory 16` = 64 KiB
const ENV_PTR: i32 = 1024; // parent dispatcher-fuel cell
const CHILD_ENV_PTR: i32 = 512; // child dispatcher-fuel cell (below the parent's scratch)
const CARVE_OFF: i64 = 4096; // the child's carve, offset into the parent window
const CHILD_WIN_SIZE: u64 = 1 << 12; // 4 KiB — matches the child's `memory 12`

// Grant-record layout in the parent window (window-relative offsets — what a real guest would lay down):
const REC_OFF: u64 = 32; // the 16-byte record `{name_off, name_len, handle, flags}`
const NAME_OFF: u64 = 64; // the grant name bytes ("fs")

/// The granted child: `f0` (emitted) = `40 + f1()`; `f1` (a cross-tier `call.cap` leaf) seeds `"fs"`
/// (`0x7366` LE) into its window, resolves it, calls the granted `HOST_PROC` counter, returns its result.
const CHILD: &str = r#"memory 12
func () -> (i64) {
block 0 () {
  v40 = i64.const 40
  vcap = call 1 ()
  vr = i64.add v40 vcap
  return vr
  }
}
func () -> (i64) {
block 0 () {
  vname = i64.const 29542
  vzero = i64.const 0
  i64.store vzero vname
  vp0 = i64.const 0
  vl2 = i64.const 2
  vh = self.resolve vp0 vl2
  vr = call.cap 13 0 (i64) -> (i64) vh (vp0)
  return vr
  }
}
"#;

/// The op-13 parent: `v0` = `Instantiator` handle, `v1` = child `Module` handle. Spawns the module into
/// a carve at `carve_off` sized `1 << carve_slog` with a **one-entry grant list** (`vgptr = rec_ptr`,
/// `vgn = 1`) — the record the test seeds names `"fs"`. Emittable via the `env.instantiate_module` bounce.
fn parent_op13_src(rec_ptr: u64, carve_off: u64, carve_slog: u64) -> String {
    format!(
        r#"memory 16
func (i32, i32) -> (i64) {{
block 0 (v0: i32, v1: i32) {{
  vmh = i64.extend_i32_u v1
  vgptr = i64.const {rec_ptr}
  vgn = i64.const 1
  ventry = i64.const 0
  voff = i64.const {carve_off}
  vsl = i64.const {carve_slog}
  vq = i64.const 0
  vh = call.cap 6 13 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vmh, vgptr, vgn, ventry, voff, vsl, vq)
  vr = call.cap 6 1 (i32) -> (i64) v0 (vh)
  return vr
  }}
}}
"#
    )
}

fn parse(src: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// A fresh host holding the granted counter cap registered as `"fs"` (the re-grantable, forkable form a
/// shared memfs takes), plus the shared counter so a call from inside the confined child is observable
/// here. Returns the host, the counter, and the `"fs"` handle value (what a guest writes into its record).
fn granted_host() -> (Host, Arc<Mutex<i64>>, i32) {
    let counter = Arc::new(Mutex::new(0i64));
    let mut host = Host::new();
    let c1 = Arc::clone(&counter);
    let handler: HostProc = Box::new(move |_op, _args, _mem, _| {
        let mut c = c1.lock().unwrap();
        *c += 1;
        Ok(vec![*c])
    });
    let c2 = Arc::clone(&counter);
    let fork = Arc::new(move |_pid: u64| {
        let c = Arc::clone(&c2);
        ForkedProc::shared(Box::new(move |_op, _args, _mem, _| {
            let mut c = c.lock().unwrap();
            *c += 1;
            Ok(vec![*c])
        }))
    });
    let h = host.grant_host_proc_forkable(handler, fork);
    host.register_cap_name("fs", h);
    (host, counter, h)
}

/// Write a 16-byte grant record `{name_off, name_len, handle, flags}` and its name into the parent window
/// (window-relative offsets), so the emitted op-13 bounce marshals it exactly as a real guest's records.
fn seed_grant_record(memory: &Memory, store: &mut Store<HostState>, handle: i32, name: &[u8]) {
    let mut rec = [0u8; 16];
    rec[0..4].copy_from_slice(&(NAME_OFF as u32).to_le_bytes());
    rec[4..8].copy_from_slice(&(name.len() as u32).to_le_bytes());
    rec[8..12].copy_from_slice(&handle.to_le_bytes());
    // bytes 12..16 (flags) stay zero — reserved.
    memory
        .write(&mut *store, WIN_BASE as usize + REC_OFF as usize, &rec)
        .unwrap();
    memory
        .write(&mut *store, WIN_BASE as usize + NAME_OFF as usize, name)
        .unwrap();
}

/// Host state threaded through the wasmi `Store`: the pre-built child's emitted entry `f0`, the child host
/// the op-13 bounce **marshals** (set by the servicer, read by `env.call_interp`), and the banked child
/// results the parent joins. `carve_rejected` records a fail-closed marshal refusal.
struct HostState {
    child_entry: Option<Func>,
    marshaled_child: Arc<Mutex<Option<Host>>>,
    children: Vec<i64>,
    carve_rejected: bool,
}

/// Run one op-13 parent over the granted child. `rec_handle`/`rec_ptr` inject a forged handle or an
/// out-of-window record (authority / parse fail-closed); `carve_off`/`carve_slog` drive the carve gate
/// (window fail-closed). Returns `(parent_result, counter, rejected)`.
fn run_marshal(
    rec_handle_override: Option<i32>,
    rec_ptr: u64,
    carve_off: u64,
    carve_slog: u64,
) -> (i64, i64, bool) {
    let parent = parse(&parent_op13_src(rec_ptr, carve_off, carve_slog));
    let child = parse(CHILD);

    let (child_wasm, eligible) =
        compile_module_nested_with_eligibility(&child, false).expect("child emits (nested)");
    assert_eq!(
        eligible,
        vec![true, false],
        "child f0 emits; the call.cap leaf f1 is cross-tier"
    );
    let parent_wasm = compile_module_nested(&parent, false).expect("op-13 parent emits (nested)");

    let engine = Engine::default();
    let parent_module = WModule::new(&engine, &parent_wasm).expect("op-13 parent wasm validates");
    let child_module = WModule::new(&engine, &child_wasm).expect("child wasm validates");

    let marshaled_child: Arc<Mutex<Option<Host>>> = Arc::new(Mutex::new(None));
    let mut store: Store<HostState> = Store::new(
        &engine,
        HostState {
            child_entry: None,
            marshaled_child: Arc::clone(&marshaled_child),
            children: Vec::new(),
            carve_rejected: false,
        },
    );
    // Two pages (128 KiB) hold the parent window `[WIN_BASE, WIN_BASE + 64 KiB)`; the carve lives inside.
    let memory = Memory::new(&mut store, MemoryType::new(2, None)).unwrap();
    memory
        .write(&mut store, ENV_PTR as usize, &i64::MAX.to_le_bytes())
        .unwrap();
    memory
        .write(&mut store, CHILD_ENV_PTR as usize, &i64::MAX.to_le_bytes())
        .unwrap();

    // The parent's powerbox (holds the grantable `"fs"`) + the shared counter. The op-13 bounce re-grants
    // out of this host. `fs_handle` is what the guest writes into its record.
    let (parent_host, counter, fs_handle) = granted_host();
    let parent_host = Arc::new(Mutex::new(parent_host));
    seed_grant_record(
        &memory,
        &mut store,
        rec_handle_override.unwrap_or(fs_handle),
        b"fs",
    );

    let child_prog =
        Arc::new(bytecode::SharedProgram::compile(&child).expect("compile child prog"));
    let child_mod = Arc::new(child.clone());
    let carve_base = (WIN_BASE as i64 + CARVE_OFF) as usize;

    let mut linker: Linker<HostState> = Linker::new(&engine);
    linker.define("env", "memory", memory).unwrap();
    linker
        .func_wrap("env", "trap", |_: Caller<'_, HostState>, _code: i32| {})
        .unwrap();

    // env.call_interp: the child's `f1` leaf runs on the interpreter **over the child's carve** against the
    // **marshaled** child host — the one the op-13 bounce built from the window records, not a pre-wired one.
    {
        let prog_cb = Arc::clone(&child_prog);
        let mod_cb = Arc::clone(&child_mod);
        linker
            .func_wrap(
                "env",
                "call_interp",
                move |mut caller: Caller<'_, HostState>,
                      func: i32,
                      args_ptr: i32|
                      -> Result<(), wasmi::Error> {
                    let callee = &mod_cb.funcs[func as usize];
                    let args: Vec<Value> = {
                        let data = memory.data(&caller);
                        let mut off = args_ptr as usize;
                        callee
                            .params
                            .iter()
                            .map(|t| {
                                let o = off;
                                off += 8;
                                let raw = u64::from_le_bytes(data[o..o + 8].try_into().unwrap());
                                match t {
                                    temen_ir::ValType::I32 => Value::I32(raw as i32),
                                    _ => Value::I64(raw as i64),
                                }
                            })
                            .collect()
                    };
                    let base = memory.data_mut(&mut caller).as_mut_ptr();
                    // SAFETY: single-threaded; the 2-page wasm memory outlives the call and never grows.
                    // The child's window IS its carve, so the interp runs `f1` over `[carve, carve+size)`.
                    let back =
                        Arc::new(unsafe { Region::shared(base.add(carve_base), CHILD_WIN_SIZE) });
                    let mut fuel = u64::MAX;
                    let child_cell = caller.data().marshaled_child.clone();
                    let mut guard = child_cell.lock().unwrap();
                    let host = guard
                        .as_mut()
                        .expect("op-13 bounce marshaled the child host");
                    let r = prog_cb
                        .run_over(func as u32, &args, &mut fuel, back, host, false)
                        .map_err(|_| {
                            wasmi::Error::from(wasmi::core::TrapCode::UnreachableCodeReached)
                        })?;
                    let data = memory.data_mut(&mut caller);
                    let mut off = args_ptr as usize;
                    for v in &r {
                        let raw = match v {
                            Value::I32(x) => *x as u32 as u64,
                            Value::I64(x) => *x as u64,
                            _ => 0,
                        };
                        data[off..off + 8].copy_from_slice(&raw.to_le_bytes());
                        off += 8;
                    }
                    Ok(())
                },
            )
            .unwrap();
    }

    // env.instantiate_module (op 13): the **faithful production servicer** shape — two fail-closed axes,
    // in order. (1) Gate the carve through `check_child_carve` (window confinement: aligned, in-parent,
    // clears the NULL guard, sized ≥ the child's declared memory). (2) **Marshal** the grant list out of
    // the parent window and build the child powerbox (authority). Either refusal means no child runs and
    // `-1` → join → sentinel. Then emit + run the child over the carve.
    {
        let parent_cb = Arc::clone(&parent_host);
        let child_ir = Arc::clone(&child_mod);
        linker
            .func_wrap(
                "env",
                "instantiate_module",
                move |mut caller: Caller<'_, HostState>,
                      win: i32,
                      _inst: i32,
                      _module: i64,
                      grants_ptr: i64,
                      grants_n: i64,
                      _entry: i64,
                      off: i64,
                      slog: i64,
                      _quota: i64|
                      -> Result<i32, wasmi::Error> {
                    // (1) Carve gate — window confinement, fail-closed before any authority work.
                    if check_child_carve(
                        &child_ir,
                        off as u64,
                        slog as u32,
                        WIN_SIZE,
                        win as u64,
                        temen_ir::module_null_guard(),
                    )
                    .is_err()
                    {
                        caller.data_mut().carve_rejected = true;
                        return Ok(-1);
                    }
                    // Read the parent's window slice `[win, win+WIN_SIZE)` and marshal the grant records.
                    let window = {
                        let data = memory.data(&caller);
                        data[win as usize..win as usize + WIN_SIZE as usize].to_vec()
                    };
                    let built = parent_cb.lock().unwrap().spawn_named_child_from_window(
                        &window,
                        grants_ptr as u64,
                        grants_n as u64,
                        CHILD_WIN_SIZE,
                    );
                    let (child_host, _cinst, _cas) = match built {
                        Ok(t) => t,
                        Err(_e) => {
                            // Fail-closed: no child runs; `-1` propagates to `join` → sentinel.
                            caller.data_mut().carve_rejected = true;
                            return Ok(-1);
                        }
                    };
                    *caller.data().marshaled_child.lock().unwrap() = Some(child_host);
                    let child_entry = caller.data().child_entry.expect("child instance pre-built");
                    let carve = win + off as i32;
                    let mut r = [Val::I64(0)];
                    child_entry.call(
                        &mut caller,
                        &[Val::I32(carve), Val::I32(CHILD_ENV_PTR)],
                        &mut r,
                    )?;
                    let res = r[0].i64().expect("child returns i64");
                    let st = caller.data_mut();
                    st.children.push(res);
                    Ok((st.children.len() - 1) as i32)
                },
            )
            .unwrap();
    }
    linker
        .func_wrap(
            "env",
            "join",
            |caller: Caller<'_, HostState>, _inst: i32, child: i32| -> i64 {
                let st = caller.data();
                if child < 0 || child as usize >= st.children.len() {
                    return i64::MIN;
                }
                st.children[child as usize]
            },
        )
        .unwrap();
    // The remaining nested imports are present (uniform layout) but never fire in this unit.
    linker
        .func_wrap(
            "env",
            "instantiate",
            |_: Caller<'_, HostState>,
             _w: i32,
             _i: i32,
             _e: i64,
             _o: i64,
             _s: i64,
             _q: i64|
             -> i32 { unreachable!("no op-0 instantiate in this unit") },
        )
        .unwrap();
    linker
        .func_wrap(
            "env",
            "instantiate_rec",
            |_: Caller<'_, HostState>, _w: i32, _i: i32, _r: i64| -> i32 {
                unreachable!("no op-17 rec spawn in this unit")
            },
        )
        .unwrap();
    linker
        .func_wrap(
            "env",
            "thread_spawn",
            |_: Caller<'_, HostState>, _f: i32, _sp: i64, _a: i64| -> i32 { unreachable!() },
        )
        .unwrap();
    linker
        .func_wrap(
            "env",
            "thread_join",
            |_: Caller<'_, HostState>, _h: i32| -> i64 { unreachable!() },
        )
        .unwrap();
    linker
        .func_wrap(
            "env",
            "mem_wait",
            |_: Caller<'_, HostState>, _w: i32, _a: i64, _e: i64, _t: i64, _is64: i32| -> i32 {
                unreachable!()
            },
        )
        .unwrap();
    linker
        .func_wrap(
            "env",
            "mem_notify",
            |_: Caller<'_, HostState>, _w: i32, _a: i64, _c: i32| -> i32 { unreachable!() },
        )
        .unwrap();

    // Pre-build the child as its own wasmi instance over the shared memory; stash its emitted entry. (The
    // module handle resolves to this instance; only the *grant* is under test, so the code is pre-emitted.)
    let child_instance = linker
        .instantiate(&mut store, &child_module)
        .unwrap()
        .start(&mut store)
        .unwrap();
    let child_entry = child_instance
        .get_func(&store, "f0")
        .expect("child emitted f0 export");
    store.data_mut().child_entry = Some(child_entry);

    let parent_instance = linker
        .instantiate(&mut store, &parent_module)
        .unwrap()
        .start(&mut store)
        .unwrap();
    let f0 = parent_instance
        .get_func(&store, "f0")
        .expect("parent f0 export");

    // f0(win, env, inst_handle, module_handle).
    let params = [
        Val::I32(WIN_BASE),
        Val::I32(ENV_PTR),
        Val::I32(7),
        Val::I32(99),
    ];
    let mut results = [Val::I64(0)];
    f0.call(&mut store, &params, &mut results)
        .expect("op-13 parent f0 runs");
    let cval = *counter.lock().unwrap();
    (
        results[0].i64().expect("parent returns i64"),
        cval,
        store.data().carve_rejected,
    )
}

#[test]
fn op13_bounce_marshals_the_grant_from_the_window() {
    // The record names the real parent `"fs"` handle at the canonical offset. The bounce marshals it, the
    // child resolves `"fs"` through the *marshaled* powerbox, and its granted call ticks the shared counter.
    let (r, counter, rejected) = run_marshal(None, REC_OFF, CARVE_OFF as u64, 12);
    assert!(!rejected, "a valid grant list + valid carve is accepted");
    assert_eq!(
        r, 41,
        "emitted op-13 parent + marshaled granted child = 40 + counter(1)"
    );
    assert_eq!(
        counter, 1,
        "the *marshaled* (not pre-wired) `fs` ran exactly once inside the confined, emitted child"
    );
}

#[test]
fn forged_handle_in_the_record_is_refused_fail_closed() {
    // The record names a handle the parent never granted (`4242`). `spawn_named_child_from_window` refuses
    // it (`NotRegrantable`); the child never runs and the counter stays put.
    let (r, counter, rejected) = run_marshal(Some(4242), REC_OFF, CARVE_OFF as u64, 12);
    assert!(rejected, "a forged handle fails the marshal closed");
    assert_eq!(r, i64::MIN, "join mapped the `-1` refusal to the sentinel");
    assert_eq!(counter, 0, "no granted authority ran — nothing to tick");
}

#[test]
fn out_of_window_record_pointer_is_refused_fail_closed() {
    // `vgptr` points past the 64-KiB window (record read `[WIN_SIZE-8, WIN_SIZE+8)` overruns). The marshal
    // refuses (`OutOfWindow`) before touching the parent powerbox; the child never runs.
    let (r, counter, rejected) = run_marshal(None, WIN_SIZE - 8, CARVE_OFF as u64, 12);
    assert!(
        rejected,
        "an out-of-window record pointer fails the marshal closed"
    );
    assert_eq!(r, i64::MIN, "join mapped the `-1` refusal to the sentinel");
    assert_eq!(counter, 0, "no granted authority ran");
}

#[test]
fn undersized_carve_is_refused_even_with_a_valid_grant() {
    // The two fail-closed axes compose: the grant list is valid (names the real `"fs"`), but the carve
    // `slog = 11` (2 KiB) under-sizes the child's `memory 12` (4 KiB) window. `check_child_carve` refuses
    // it **before** the grant is marshaled — window confinement is independent of authority — so the child
    // never runs and the counter stays 0. (Native/interp gate this the same way; this is the emitted twin.)
    let (r, counter, rejected) = run_marshal(None, REC_OFF, CARVE_OFF as u64, 11);
    assert!(
        rejected,
        "an undersized carve is refused even with a valid grant"
    );
    assert_eq!(r, i64::MIN, "join mapped the carve refusal to the sentinel");
    assert_eq!(
        counter, 0,
        "fail-closed on the window axis: no granted authority ran"
    );
}

#[test]
fn misaligned_carve_is_refused_even_with_a_valid_grant() {
    // A carve offset that is not `1 << slog`-aligned (`CARVE_OFF + 8`, `slog = 12`) is refused by the gate,
    // independently of the (valid) grant — the alignment half of the buddy-carve invariant.
    let (_r, counter, rejected) = run_marshal(None, REC_OFF, CARVE_OFF as u64 + 8, 12);
    assert!(
        rejected,
        "a misaligned carve is refused even with a valid grant"
    );
    assert_eq!(counter, 0, "fail-closed on the window axis");
}

#[test]
fn marshal_error_maps_are_stable() {
    // Guard the public error surface the servicer switches on (MemoryFault vs CapFault trap selection).
    let mut host = Host::new();
    // An empty window with a non-zero record count is always out-of-window.
    assert!(matches!(
        host.spawn_named_child_from_window(&[], 0, 1, CHILD_WIN_SIZE),
        Err(GrantMarshalError::OutOfWindow)
    ));
    // Zero grants over an empty window is a valid (empty) child — no records to read.
    assert!(host
        .spawn_named_child_from_window(&[], 0, 0, CHILD_WIN_SIZE)
        .is_ok());
}
