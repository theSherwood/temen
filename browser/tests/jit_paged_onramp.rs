//! **#1201 — page-op guests on the single-shot wasm-JIT tier.** A guest that `unmap`s/`protect`s its
//! own pages used to decline this tier outright (`compile_jit` emits nothing for a page-op module);
//! now `JitOnrampRun` emits it **paged** (`compile_jit_paged` → `compile_module_reactor_paged`) and
//! rebuilds the #750 page-state table from each bounce's live map, the driver re-pointing the emitted
//! `"pagestate"`/`"mapped"` globals — the single-shot twin of the coop tier's `sync_pagestate`. This is
//! the wasmi driver playing `driveJitRun`'s role, for both cells of the frontier the tier serves:
//!
//! - a **root on-ramp guest** (`open_shared_run`, the paramless `_start` resolving `addrspace` by name):
//!   its helper `protect`s the page holding "K" = 75 read-only and `unmap`s another; `_start` reads K
//!   through the `Ro` page and stores a marker on an `Rw` page → 7509. The twins store on the
//!   `Ro` page / the unmapped page: the emitted access traps, as the interpreter's does.
//! - a **§14 op-13 child** (`open_shared_run_over_host`, the `[I64,I64]->[I64]` child-entry over a
//!   marshaled `fs` + its real starter `AddressSpace` handle — #1201 passes it instead of `0`): its
//!   leaf resolves `fs` (counter → 1) and `protect`s its "K" page; `f0` reads K back → 116, the
//!   store twin traps after the leaf ran.
//!
//! Each case is differential against the interpreter (`onramp_exec` for the root guest; the value /
//! trap the child's interpreter run gives), and the page-state table the emitted accesses consulted is
//! pinned (Q = `Ro`, P = `Unmapped`) as the non-vacuity proof.

use std::sync::{Arc, Mutex};

use temen_browser::{onramp_exec, JitOnrampRun, STATUS_TRAP};
use temen_interp::{bytecode, host_page_size, ForkedProc, Host, HostProc, Region, Value};
use wasmi::{Caller, Engine, Linker, Memory, MemoryType, Module as WModule, Store, Val};

const ENV_PTR: u32 = 1024;
const WIN_BASE: u32 = 0x1_0000;
/// Page-aligned on a 4 KiB or 16 KiB host: Q holds "K", P is unmapped, R stays `Rw`.
const Q: u64 = 49152;
const P: u64 = 32768;
const R: u64 = 16384 + 8;

/// Where the emitted store lands.
#[derive(Clone, Copy)]
enum Target {
    Rw,
    Ro,
    Unmapped,
}

impl Target {
    fn addr(self) -> u64 {
        match self {
            Target::Rw => R,
            Target::Ro => Q + 8,
            Target::Unmapped => P + 8,
        }
    }
}

/// The root on-ramp guest (`memory 16`): `_start()` = `K * 100 + marker + helper()`, the helper
/// resolving `addrspace` and doing `protect(Q, page, Ro)` + `unmap(P, page)` (both `0` on success).
fn root_src(target: Target) -> String {
    let page = host_page_size();
    let t = target.addr();
    format!(
        r#"memory 16
data {Q} "K"
data 16448 "addrspace"
func () -> (i64) {{
block 0 () {{
  vr = call 1 ()
  vq = i64.const {Q}
  vk = i64.load8_u vq
  vt = i64.const {t}
  vm = i64.const 9
  i64.store vt vm
  vld = i64.load vt
  vhundred = i64.const 100
  vkh = i64.mul vk vhundred
  vs = i64.add vkh vld
  vsum = i64.add vs vr
  return vsum
  }}
}}
func () -> (i64) {{
block 0 () {{
  vnp = i64.const 16448
  vnl = i64.const 9
  vas = self.resolve vnp vnl
  vlen = i64.const {page}
  vq = i64.const {Q}
  vro = i32.const 1
  vr2 = call.cap 5 2 (i64, i64, i32) -> (i64) vas (vq, vlen, vro)
  vp = i64.const {P}
  vr1 = call.cap 5 1 (i64, i64) -> (i64) vas (vp, vlen)
  vsum = i64.add vr1 vr2
  return vsum
  }}
}}
"#
    )
}

/// The op-13 child (`memory 15`): `f0(inst, as)` = `40 + f1(as) + K`; `f1` resolves the marshaled
/// `fs` (→ 1) then `protect`s the "K" page read-only. "K" sits at 16 KiB — just above the NULL guard
/// (`POWERBOX_NULL_GUARD`, seeded on every bounce), page-aligned on a 4 KiB or 16 KiB host; the twin
/// stores on that page after the protect.
fn child_src(store_to_ro: bool) -> String {
    let page = host_page_size();
    let store = if store_to_ro {
        "  vt = i64.const 16400\n  vseven = i64.const 7\n  i64.store vt vseven\n"
    } else {
        ""
    };
    format!(
        r#"memory 15
data 16384 "K"
func (i64, i64) -> (i64) {{
block 0 (vsp: i64, vas: i64) {{
  vc = call 1 (vas)
  vq = i64.const 16384
  vk = i64.load8_u vq
{store}  v40 = i64.const 40
  vs = i64.add v40 vc
  vr = i64.add vs vk
  return vr
  }}
}}
func (i64) -> (i64) {{
block 0 (vas: i64) {{
  vname = i64.const 29542
  vnp = i64.const 16392
  i64.store vnp vname
  vl2 = i64.const 2
  vh = self.resolve vnp vl2
  vc = call.cap 13 0 (i64) -> (i64) vh (vnp)
  vas32 = i32.wrap_i64 vas
  vq = i64.const 16384
  vlen = i64.const {page}
  vro = i32.const 1
  vpr = call.cap 5 2 (i64, i64, i32) -> (i64) vas32 (vq, vlen, vro)
  vsum = i64.add vc vpr
  return vsum
  }}
}}
"#
    )
}

fn build(src: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    assert!(
        temen_wasm_jit::module_uses_unmap_protect(&m),
        "the guest manages its own pages"
    );
    m
}

/// A host holding a forkable `"fs"` counter (the re-grantable shape a shared memfs takes).
fn fs_host() -> (Host, Arc<Mutex<i64>>) {
    let counter = Arc::new(Mutex::new(0i64));
    let mut host = Host::new();
    let c1 = Arc::clone(&counter);
    let handler: HostProc = Box::new(move |_op, _a, _m, _| {
        let mut c = c1.lock().unwrap();
        *c += 1;
        Ok(vec![*c])
    });
    let c2 = Arc::clone(&counter);
    let fork = Arc::new(move |_pid: u64| {
        let c = Arc::clone(&c2);
        ForkedProc::shared(Box::new(move |_op, _a, _m, _| {
            let mut c = c.lock().unwrap();
            *c += 1;
            Ok(vec![*c])
        }))
    });
    let h = host.grant_host_proc_forkable(handler, fork);
    host.register_cap_name("fs", h);
    (host, counter)
}

/// One emitted run's outcome: the value or a trap, plus the page-state table the last bounce left.
struct Outcome {
    result: Result<i64, ()>,
    table: Vec<u8>,
    bounces: u32,
}

struct Drv {
    run: Option<JitOnrampRun>,
    table_base: u32,
    last_table: Vec<u8>,
    bounces: u32,
}

/// Drive the run's emitted `f0` on wasmi — `driveJitRun`'s role, with the #1201 paged sync: copy the
/// run's table into the wasmi memory and re-point `"pagestate"`/`"mapped"` before `f0` and after each
/// bounce.
fn drive(engine: &Engine, mut store: Store<Drv>, memory: Memory, slots: Vec<Val>) -> Outcome {
    let (emitted_wasm, paged) = {
        let r = store.data().run.as_ref().unwrap();
        (r.emitted_wasm().to_vec(), r.paged())
    };
    assert!(paged, "a page-op guest emits paged on the single-shot tier");
    let module = WModule::new(engine, &emitted_wasm).expect("emitted module validates");
    let mut linker: Linker<Drv> = Linker::new(engine);
    linker.define("env", "memory", memory).unwrap();
    linker
        .func_wrap("env", "trap", |_: Caller<'_, Drv>, _code: i32| {})
        .unwrap();
    linker
        .func_wrap(
            "env",
            "call_interp",
            move |mut caller: Caller<'_, Drv>,
                  func: i32,
                  args_ptr: i32|
                  -> Result<(), wasmi::Error> {
                caller.data_mut().bounces += 1;
                let (params, results) = {
                    let r = caller.data().run.as_ref().unwrap();
                    let (p, rs) = r.func_sig(func as u32);
                    (p.to_vec(), rs.to_vec())
                };
                let args: Vec<Value> = {
                    let data = memory.data(&caller);
                    params
                        .iter()
                        .enumerate()
                        .map(|(i, t)| {
                            let o = args_ptr as usize + i * 8;
                            let raw = u64::from_le_bytes(data[o..o + 8].try_into().unwrap());
                            match t {
                                temen_ir::ValType::I32 => Value::I32(raw as i32),
                                _ => Value::I64(raw as i64),
                            }
                        })
                        .collect()
                };
                let outcome = caller
                    .data_mut()
                    .run
                    .as_mut()
                    .unwrap()
                    .run_cross_tier(func as u32, &args);
                match outcome {
                    Ok(vals) => {
                        // The per-bounce refresh (`driveJitRun`'s `syncGlobals()`): the run's rebuilt
                        // table into the wasmi memory, `"pagestate"`/`"mapped"` re-pointed at it.
                        let (table, mapped) = {
                            let r = caller.data().run.as_ref().unwrap();
                            (r.pagestate().to_vec(), r.mapped())
                        };
                        let base = caller.data().table_base;
                        memory.write(&mut caller, base as usize, &table).unwrap();
                        if let Some(wasmi::Extern::Global(g)) = caller.get_export("pagestate") {
                            g.set(&mut caller, Val::I32(base as i32)).unwrap();
                        }
                        if let Some(wasmi::Extern::Global(g)) = caller.get_export("mapped") {
                            g.set(&mut caller, Val::I64(mapped as i64)).unwrap();
                        }
                        caller.data_mut().last_table = table;
                        let data = memory.data_mut(&mut caller);
                        for (i, v) in vals.iter().enumerate().take(results.len()) {
                            let raw = match v {
                                Value::I32(x) => *x as u32 as u64,
                                Value::I64(x) => *x as u64,
                                _ => 0,
                            };
                            let o = args_ptr as usize + i * 8;
                            data[o..o + 8].copy_from_slice(&raw.to_le_bytes());
                        }
                        Ok(())
                    }
                    Err(_) => Err(wasmi::Error::from(
                        wasmi::core::TrapCode::UnreachableCodeReached,
                    )),
                }
            },
        )
        .unwrap();
    let instance = linker
        .instantiate(&mut store, &module)
        .unwrap()
        .start(&mut store)
        .unwrap();
    let pg = instance
        .get_global(&store, "pagestate")
        .expect("a paged emit exports pagestate");
    let mg = instance
        .get_global(&store, "mapped")
        .expect("mapped global");
    let f0 = instance.get_func(&store, "f0").expect("emitted f0");
    // The initial sync (the JS driver's `syncGlobals()` before `f0`).
    {
        let table = store.data().run.as_ref().unwrap().pagestate().to_vec();
        let mapped = store.data().run.as_ref().unwrap().mapped();
        let base = store.data().table_base;
        memory.write(&mut store, base as usize, &table).unwrap();
        pg.set(&mut store, Val::I32(base as i32)).unwrap();
        mg.set(&mut store, Val::I64(mapped as i64)).unwrap();
    }
    memory
        .write(&mut store, ENV_PTR as usize, &(1i64 << 60).to_le_bytes())
        .unwrap();
    let mut params = vec![Val::I32(WIN_BASE as i32), Val::I32(ENV_PTR as i32)];
    params.extend(slots);
    let mut results = [Val::I64(0)];
    let call = f0.call(&mut store, &params, &mut results);
    let run = store.data().run.as_ref().unwrap();
    let result = match call {
        Ok(()) => Ok(results[0].i64().expect("i64 result")),
        Err(_) => {
            assert!(!run.exited(), "a trap, not an exit");
            Err(())
        }
    };
    Outcome {
        result,
        table: store.data().last_table.clone(),
        bounces: store.data().bounces,
    }
}

fn new_memory(store: &mut Store<Drv>, win_size: u64) -> (Memory, u32) {
    let table_base = WIN_BASE + win_size as u32;
    let total = table_base as u64 + win_size / host_page_size() + 64;
    let pages = (total as u32).div_ceil(1 << 16) + 1;
    let memory = Memory::new(store, MemoryType::new(pages, Some(pages))).unwrap();
    (memory, table_base)
}

// ---- the root on-ramp guest -----------------------------------------------------------------------

fn root_case(target: Target) {
    let m = build(&root_src(target));
    let o = onramp_exec(&m, b"");
    let want: Result<i64, ()> = if o.status == STATUS_TRAP {
        Err(())
    } else {
        Ok(o.value)
    };

    let engine = Engine::default();
    let mut store: Store<Drv> = Store::new(
        &engine,
        Drv {
            run: None,
            table_base: 0,
            last_table: Vec::new(),
            bounces: 0,
        },
    );
    const WIN_LOG2: u8 = 16;
    let win_size = 1u64 << WIN_LOG2;
    let (memory, table_base) = new_memory(&mut store, win_size);
    store.data_mut().table_base = table_base;
    // SAFETY: fixed-size memory ⇒ a stable data pointer; the window lives inside it for the run.
    let win_ptr = unsafe {
        memory
            .data_mut(&mut store)
            .as_mut_ptr()
            .add(WIN_BASE as usize)
    };
    let run = unsafe {
        JitOnrampRun::open_shared_run(&m, win_ptr, win_size, WIN_LOG2, false, Vec::new())
    }
    .expect("a page-op on-ramp guest opens as a single-shot JIT run (paged)");
    store.data_mut().run = Some(run);
    let out = drive(&engine, store, memory, Vec::new());
    assert_eq!(
        out.result, want,
        "emitted paged run diverged from onramp_exec"
    );
    // Non-vacuity: the oracle itself ran the page ops (a value, or a trap only for the store twins),
    // and the emitted run bounced exactly the helper's three outlined cap wrappers (resolve, protect,
    // unmap) — the helper's own body ran emitted.
    assert_eq!(
        want,
        match target {
            Target::Rw => Ok(75 * 100 + 9),
            _ => Err(()),
        },
        "interpreter oracle"
    );
    assert_eq!(out.bounces, 3, "the three outlined cap wrappers bounced");
    let page = host_page_size();
    assert_eq!(out.table[(Q / page) as usize], 2, "Q protected read-only");
    assert_eq!(out.table[(P / page) as usize], 0, "P unmapped");
}

#[test]
fn onramp_guest_protect_unmap_then_rw_store_matches_interpreter() {
    root_case(Target::Rw);
}

#[test]
fn onramp_guest_store_to_protected_page_traps_on_both_tiers() {
    root_case(Target::Ro);
}

#[test]
fn onramp_guest_store_to_unmapped_page_traps_on_both_tiers() {
    root_case(Target::Unmapped);
}

// ---- the op-13 child over a marshaled host --------------------------------------------------------

/// The child's interpreter outcome over an identically-granted host.
fn child_oracle(m: &temen_ir::Module) -> (Result<i64, ()>, i64) {
    let (mut host, counter) = fs_host();
    const LOG2: u8 = 15;
    let size = 1u64 << LOG2;
    let cinst = host.grant_instantiator(0, size);
    let cas = host.grant_address_space(0, size);
    let prog = bytecode::SharedProgram::compile(m).expect("compile");
    let mut backing = vec![0u8; size as usize].into_boxed_slice();
    // SAFETY: `backing` outlives the run; the region is this call's exclusive window.
    let back = Arc::new(unsafe { Region::shared(backing.as_mut_ptr(), size) });
    let mut fuel = u64::MAX;
    let r = prog.run_over(
        0,
        &[Value::I64(cinst as i64), Value::I64(cas as i64)],
        &mut fuel,
        back,
        &mut host,
        true,
    );
    let out = match r {
        Ok(v) => match v.first() {
            Some(Value::I64(x)) => Ok(*x),
            other => panic!("oracle result {other:?}"),
        },
        Err(_) => Err(()),
    };
    let c = *counter.lock().unwrap();
    (out, c)
}

fn child_case(store_to_ro: bool) {
    let m = build(&child_src(store_to_ro));
    let (want, want_counter) = child_oracle(&m);

    let (mut host, counter) = fs_host();
    const LOG2: u8 = 15;
    let size = 1u64 << LOG2;
    let cinst = host.grant_instantiator(0, size);
    let cas = host.grant_address_space(0, size);
    let engine = Engine::default();
    let mut store: Store<Drv> = Store::new(
        &engine,
        Drv {
            run: None,
            table_base: 0,
            last_table: Vec::new(),
            bounces: 0,
        },
    );
    let (memory, table_base) = new_memory(&mut store, size);
    store.data_mut().table_base = table_base;
    // SAFETY: as above — the child's window (its carve) lives inside the fixed wasmi memory.
    let win_ptr = unsafe {
        memory
            .data_mut(&mut store)
            .as_mut_ptr()
            .add(WIN_BASE as usize)
    };
    let run = unsafe {
        JitOnrampRun::open_shared_run_over_host(
            &m,
            win_ptr,
            size,
            LOG2,
            false,
            host,
            Vec::new(),
            None,
            cinst as u64,
            cas as u64,
            None,
        )
    }
    .expect("a page-op op-13 child opens as a single-shot JIT run (paged)");
    let slots: Vec<Val> = run
        .slots()
        .iter()
        .map(|v| match v {
            Value::I64(x) => Val::I64(*x),
            Value::I32(x) => Val::I32(*x),
            _ => Val::I64(0),
        })
        .collect();
    assert_eq!(slots.len(), 2, "(inst, as) starter handles");
    assert_eq!(
        slots[1].i64(),
        Some(cas as i64),
        "the real AddressSpace handle, not 0"
    );
    store.data_mut().run = Some(run);
    let out = drive(&engine, store, memory, slots);
    assert_eq!(
        out.result, want,
        "emitted paged child diverged from the interpreter"
    );
    // Non-vacuity: the oracle ran the leaf (fs ticked; a trap only for the store twin), and the emitted
    // run bounced exactly the leaf's three outlined cap wrappers (resolve, fs, protect).
    assert_eq!(
        want,
        if store_to_ro {
            Err(())
        } else {
            Ok(40 + 1 + 75)
        },
        "interpreter oracle"
    );
    assert_eq!(want_counter, 1, "the oracle's leaf ran fs once");
    assert_eq!(out.bounces, 3, "the three outlined cap wrappers bounced");
    assert_eq!(
        *counter.lock().unwrap(),
        want_counter,
        "the marshaled fs ran inside the child"
    );
    assert_eq!(*counter.lock().unwrap(), 1);
    let page = host_page_size();
    assert_eq!(
        out.table[(16384 / page) as usize],
        2,
        "the K page protected read-only"
    );
}

#[test]
fn op13_child_protect_then_ro_read_matches_interpreter() {
    child_case(false);
}

#[test]
fn op13_child_store_to_protected_page_traps_on_both_tiers() {
    child_case(true);
}
