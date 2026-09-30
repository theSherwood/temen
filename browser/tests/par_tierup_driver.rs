//! A native harness for the `temen_par_*` FFI. It plays `par.js` + `worker.js` single-threaded, under
//! wasmi, over the real exports, so the parallel driver's event plumbing runs in `cargo test` and not
//! only in real Chromium. It pins the Worker's B2 table mirror: a runtime unit's `call.dyn` into the
//! program's natural prefix (#1347), and an installed unit surviving the release of its code handle
//! (#1339).

use temen_browser::{
    temen_par_enable_jit, temen_par_ev_a, temen_par_ev_b, temen_par_free, temen_par_root,
    temen_par_run, PAR_DONE,
};
use wasmi::{Caller, Engine, Linker, Memory, MemoryType, Module as WModule, Store, Val};

// #1182 — serialize the tests that drive the `temen_par_*` codegen path. The emitted-JIT stash
// (`WASMJIT`, `PAR_JIT_ELIGIBLE`, `PAR_JIT_PAGED`) and its once-per-run memoization
// (`PAR_RUN_GEN` → `TIERUP_DONE_GEN`) are process-GLOBAL and single-run by design: in production one
// page runs one program, and the page-side publisher bumps the generation serially before any Worker
// is alive (see `CodegenGuard` in `browser/src/lib.rs`). `cargo test` breaks that assumption by
// running the tests concurrently: their gen-bumps and `enable_jit` emits interleave. Holding this lock across each test body restores the serial single-run contract the globals assume.
// Poison-tolerant (`into_inner`) so one test's panic still lets the other run and report on its own.
static JIT_STATE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

// ---- #1347: a runtime unit's `call.dyn` into the dispatch table's NATURAL PREFIX on the par driver --
//
// A `compile_linked` unit's Slot import names a *program function* (the JACL macro-staging shape). On
// the interpreter and the coop tier that dispatches through the shared table's natural prefix; the
// parallel Worker's B2 table mirror used to fill only installed-unit slots, leaving every program slot a
// null funcref. This plays `worker.js` under wasmi over the real `temen_par_*` FFI: the root
// `compile_linked`s + `invoke`s a unit whose `call.dyn` lands on program function 1 — once
// interpreter-resident (a bounce shim → `temen_par_inst_call_interp` on the root vCPU), once emitted by
// the tier-up module (`emitted.f1`, natively) — differential against the interpreted service.

const LK_WIN_BASE: u32 = 0x4_0000;
const LK_ENV_PTR: u32 = 1024;
const LK_BLOB_OFF: usize = 0x6000;
const LK_SYMTAB_OFF: usize = 0x7000;
const LK_PROBE: i64 = 21;
const LK_K: i64 = 90909;
/// Where the emitted `F` stores + reloads its result (above the NULL guard, below the staged blob).
const LK_MARK: i64 = 0x5000;

/// `unit(x) = F(x) + LK_K`. `linked == false`: `F` is an unresolved `call.sym` the guest binds to
/// Slot 1 at link time (the blob it `compile_linked`s). `linked == true`: the same unit in its
/// post-link shape — the linker rewrites that `call.sym` into a `call.dyn` through the patched
/// placeholder (`Resolved::Slot`, IMPORTS.md §2.5) — the harness's source for the non-shared wasmi
/// twin of the FFI's emit (the linker itself is off limits outside the linker, `imports_gate.rs`).
fn lk_unit_module(linked: bool) -> temen_ir::Module {
    let call = if linked {
        "  v1 = i32.const 1\n  v2 = call.dyn (i64) -> (i64) v1 (v0)\n"
    } else {
        "  v1 = i32.const 0\n  v2 = call.sym \"F\" (i64) -> (i64) v1 (v0)\n"
    };
    let src = format!(
        r#"memory 16
func (i64) -> (i64) {{
block 0 (v0: i64) {{
{call}  vk = i64.const {LK_K}
  v3 = i64.add v2 vk
  return v3
  }}
}}
"#
    );
    temen_text::parse_module(&src).expect("unit parse")
}

/// The guest: f0 (root, arg = its `Jit` handle) `compile_linked`s the staged unit against the staged
/// symbol table (`"F"` → Slot 1) and `invoke`s it with `LK_PROBE`. f1 = `F(x) = 2x`: with
/// `emitted_callee` a pure leaf that stores + reloads through `[LK_MARK]` (so a mis-primed `"mapped"`
/// would fault it), else a `call.dyn` through slot 2 to f2 — off the tier-up emit, so the unit's
/// `call.dyn` to slot 1 must bounce. f2 = the helper `2x`.
fn lk_guest_module(blob_len: usize, st_len: usize, emitted_callee: bool) -> temen_ir::Module {
    let f1 = if emitted_callee {
        format!(
            "  v1 = i64.const 2\n  vm = i64.mul v0 v1\n  va = i64.const {LK_MARK}\n  i64.store va vm\n  vr = i64.load va\n  return vr\n"
        )
    } else {
        "  vs2 = i32.const 2\n  vr = call.dyn (i64) -> (i64) vs2 (v0)\n  return vr\n".to_string()
    };
    let src = format!(
        r#"memory 16
func (i32) -> (i64) {{
block 0 (v0: i32) {{
  vbp = i64.const {LK_BLOB_OFF}
  vbl = i64.const {blob_len}
  vsp = i64.const {LK_SYMTAB_OFF}
  vsn = i64.const {st_len}
  vcode = call.cap 11 5 (i64, i64, i64, i64) -> (i64) v0 (vbp, vbl, vsp, vsn)
  vprobe = i64.const {LK_PROBE}
  vres = call.cap 11 1 (i64, i64) -> (i64) v0 (vcode, vprobe)
  return vres
  }}
}}
func (i64) -> (i64) {{
block 0 (v0: i64) {{
{f1}  }}
}}
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  v1 = i64.const 2
  vr = i64.mul v0 v1
  return vr
  }}
}}
"#
    );
    let m = temen_text::parse_module(&src).expect("guest parse");
    temen_verify::verify_module(&m).expect("guest verify");
    m
}

struct LkDrv {
    v: usize,
    bounces: u32,
    mapped_globals: Vec<wasmi::Global>,
    fuel_globals: Vec<wasmi::Global>,
}

/// Instantiate an emitted module (unit / shim / tier-up module) against the harness's memory + table,
/// with `env.call_interp` routed to the root vCPU's live-state bounce — `worker.js`'s import objects.
fn lk_instantiate(
    store: &mut Store<LkDrv>,
    engine: &Engine,
    memory: Memory,
    table: wasmi::Table,
    wasm: &[u8],
) -> wasmi::Instance {
    use temen_browser::temen_par_inst_call_interp;
    let module = WModule::new(engine, wasm).expect("emitted wasm validates");
    let mut linker: Linker<LkDrv> = Linker::new(engine);
    linker.define("env", "memory", memory).unwrap();
    linker
        .define("env", "__indirect_function_table", table)
        .unwrap();
    linker
        .func_wrap("env", "trap", |_c: Caller<'_, LkDrv>, _code: i32| {})
        .unwrap();
    linker
        .func_wrap(
            "env",
            "call_interp",
            move |mut c: Caller<'_, LkDrv>,
                  target: i32,
                  args_ptr: i32|
                  -> Result<(), wasmi::Error> {
                c.data_mut().bounces += 1;
                let v = c.data().v as *mut temen_browser::ParVcpu;
                // SAFETY: `args_ptr` is the env scratch inside the fixed-size wasmi memory.
                let ap = unsafe { memory.data_mut(&mut c).as_mut_ptr().add(args_ptr as usize) };
                if temen_par_inst_call_interp(v, target as u32, ap) != 0 {
                    return Err(wasmi::Error::from(
                        wasmi::core::TrapCode::UnreachableCodeReached,
                    ));
                }
                Ok(())
            },
        )
        .unwrap();
    let inst = linker
        .instantiate(&mut *store, &module)
        .unwrap()
        .start(&mut *store)
        .unwrap();
    if let Some(g) = inst.get_global(&*store, "mapped") {
        store.data_mut().mapped_globals.push(g);
    }
    if let Some(g) = inst.get_global(&*store, "fuel") {
        store.data_mut().fuel_globals.push(g);
    }
    inst
}

/// One run: `(value, surfaced invokes, bounces)`. `codegen == false` is the oracle (the engine services
/// the invoke interpreted, no events); `codegen == true` plays the Worker.
fn lk_run(codegen: bool, emitted_callee: bool) -> (i64, u32, u32) {
    use temen_browser::{
        temen_par_compile_jit, temen_par_deliver_jit_invoke, temen_par_deliver_jit_invoke_trap,
        temen_par_jit_argv_len, temen_par_jit_argv_ptr, temen_par_jit_code,
        temen_par_jit_code_wasm_len, temen_par_jit_param_types_ptr, temen_par_jit_result_types_len,
        temen_par_jit_result_types_ptr, temen_par_jit_set_b2, temen_par_jit_set_codegen,
        temen_par_jit_slot_unit, temen_par_jit_table_log2, temen_par_nfuncs,
        temen_par_powerbox_jit_runtime, temen_par_shim_wasm_len, temen_wasmjit_len,
        temen_wasmjit_ptr, PAR_JIT_INVOKE,
    };
    let blob = temen_encode::encode_module(&lk_unit_module(false)); // unresolved — the guest links it
    let symtab: Vec<u8> = vec![1, 1, b'F', 0, 1]; // `"F"` → Slot(1) (canonical wire form)
    let guest = lk_guest_module(blob.len(), symtab.len(), emitted_callee);
    let guest_bytes = temen_encode::encode_module(&guest);

    assert_eq!(
        temen_par_powerbox_jit_runtime(guest_bytes.as_ptr(), guest_bytes.len()),
        1,
        "runtime-compile powerbox"
    );
    temen_par_jit_set_b2(1);
    temen_par_jit_set_codegen(if codegen { 1 } else { 0 });
    let prog = temen_par_compile_jit(guest_bytes.as_ptr(), guest_bytes.len());
    assert!(!prog.is_null(), "guest compiles");
    // The tier-up module: `F` (pure leaf, all-i64) emits + is eligible in the emitted-callee shape;
    // in the shim shape `F` calls f2 so it is interpreter-resident (f2 itself emits, unreached).
    let tiers = temen_par_enable_jit(guest_bytes.as_ptr(), guest_bytes.len());
    assert_eq!(tiers, 1, "some leaf emits in either shape");

    let engine = Engine::default();
    let mut store: Store<LkDrv> = Store::new(
        &engine,
        LkDrv {
            v: 0,
            bounces: 0,
            mapped_globals: Vec::new(),
            fuel_globals: Vec::new(),
        },
    );
    let pages = (LK_WIN_BASE + (1 << 16)).div_ceil(1 << 16) + 1;
    let memory = Memory::new(&mut store, MemoryType::new(pages, Some(pages))).unwrap();
    memory
        .write(&mut store, LK_ENV_PTR as usize, &i64::MAX.to_le_bytes())
        .unwrap();
    // Stage the unit blob + symbol table where the guest reads them (before the root seeds data).
    memory
        .write(&mut store, LK_WIN_BASE as usize + LK_BLOB_OFF, &blob)
        .unwrap();
    memory
        .write(&mut store, LK_WIN_BASE as usize + LK_SYMTAB_OFF, &symtab)
        .unwrap();
    // SAFETY: fixed-size memory ⇒ a stable data pointer; the window lives inside it (the browser's
    // shared-linear-memory shape) and is used solely as this run's window.
    let win_ptr = unsafe {
        memory
            .data_mut(&mut store)
            .as_mut_ptr()
            .add(LK_WIN_BASE as usize)
    };
    let v = temen_par_root(prog, win_ptr, 1 << 16, 0);
    assert!(!v.is_null(), "root vCPU builds");
    store.data_mut().v = v as usize;

    let tsize = 1u32 << temen_par_jit_table_log2();
    let table = wasmi::Table::new(
        &mut store,
        wasmi::TableType::new(wasmi::core::ValType::FuncRef, tsize, Some(tsize)),
        Val::FuncRef(wasmi::FuncRef::null()),
    )
    .unwrap();
    // wasmi validates no threads proposal: run non-shared twins of the FFI's shared-memory emits (the
    // shared import only adds a max limit — a few LEB bytes).
    let emitted = {
        let stashed =
            unsafe { std::slice::from_raw_parts(temen_wasmjit_ptr(), temen_wasmjit_len()) };
        let art = temen_wasm_jit::compile_jit(&guest, temen_wasm_jit::Shape::Threaded, false)
            .expect("tier-up emit");
        assert!(
            stashed.len() > art.wasm.len() && stashed.len() - art.wasm.len() <= 8,
            "tier-up stash {} vs unshared {}",
            stashed.len(),
            art.wasm.len()
        );
        assert_eq!(
            art.emitted[1], emitted_callee,
            "`F` emits iff it is the pure-leaf shape"
        );
        lk_instantiate(&mut store, &engine, memory, table, &art.wasm)
    };
    let linked = lk_unit_module(true);
    temen_verify::verify_module(&linked).expect("linked unit verifies");
    let unit_wasm = temen_wasm_jit::compile_module_b2(&linked, false, temen_par_jit_table_log2())
        .expect("B2 unit emit");
    let mut unit_inst: Option<wasmi::Instance> = None;
    let mut shims: Vec<Option<wasmi::Instance>> = vec![None; tsize as usize];

    let mut invokes = 0u32;
    let value = loop {
        match temen_par_run(v) {
            PAR_DONE => break temen_par_ev_a(v),
            PAR_JIT_INVOKE => {
                invokes += 1;
                assert!(invokes < 8, "runaway invokes");
                // `worker.js::jitSyncTable` (#1347): the natural prefix from the tier-up module or
                // a shim, installed slots from their units (none here), the rest null.
                let nfuncs = temen_par_nfuncs();
                assert_eq!(
                    nfuncs, 3,
                    "the natural prefix is the program's function count"
                );
                for slot in 0..tsize {
                    let entry = if (slot as usize) < nfuncs {
                        match emitted.get_func(&store, &format!("f{slot}")) {
                            Some(f) => Some(f),
                            None => {
                                if shims[slot as usize].is_none() {
                                    let ffi_len = temen_par_shim_wasm_len(slot);
                                    let (p, r) = (
                                        &guest.funcs[slot as usize].params,
                                        &guest.funcs[slot as usize].results,
                                    );
                                    let w = temen_wasm_jit::emit_slot_trampoline(p, r, slot, false)
                                        .expect("shim emit");
                                    assert!(
                                        ffi_len > w.len() && ffi_len - w.len() <= 8,
                                        "FFI shim {ffi_len} vs unshared {}",
                                        w.len()
                                    );
                                    shims[slot as usize] = Some(lk_instantiate(
                                        &mut store, &engine, memory, table, &w,
                                    ));
                                }
                                shims[slot as usize].unwrap().get_func(&store, "t")
                            }
                        }
                    } else {
                        assert_eq!(temen_par_jit_slot_unit(slot), -1, "nothing installed");
                        None
                    };
                    let fr = match entry {
                        Some(f) => wasmi::FuncRef::new(f),
                        None => wasmi::FuncRef::null(),
                    };
                    table
                        .set(&mut store, slot as u64, Val::FuncRef(fr))
                        .unwrap();
                }
                // The invoked unit (cached per code handle): the FFI emitted it (shared) on demand
                // from the guest-linked IR — the same code as the twin but for the shared-memory
                // import's max limit (a few LEB bytes).
                let ffi_len = temen_par_jit_code_wasm_len(v);
                assert!(
                    ffi_len > unit_wasm.len() && ffi_len - unit_wasm.len() <= 8,
                    "FFI linked-unit emit {ffi_len} vs unshared twin {}",
                    unit_wasm.len()
                );
                let _code = temen_par_jit_code(v);
                let inst = *unit_inst.get_or_insert_with(|| {
                    lk_instantiate(&mut store, &engine, memory, table, &unit_wasm)
                });
                // Prime every instance's `"mapped"` (the event's extent) + fuel, as the Worker does.
                let mapped = temen_par_ev_b(v);
                for g in store.data().mapped_globals.clone() {
                    g.set(&mut store, Val::I64(mapped)).unwrap();
                }
                for g in store.data().fuel_globals.clone() {
                    g.set(&mut store, Val::I64(1 << 61)).unwrap();
                }
                memory
                    .write(&mut store, LK_ENV_PTR as usize, &(1i64 << 61).to_le_bytes())
                    .unwrap();
                let n = temen_par_jit_argv_len(v);
                // SAFETY: pending-event operand stash, stable until the deliver.
                let argv = unsafe { std::slice::from_raw_parts(temen_par_jit_argv_ptr(v), n) };
                let ptypes =
                    unsafe { std::slice::from_raw_parts(temen_par_jit_param_types_ptr(v), n) };
                let mut params = vec![Val::I32(LK_WIN_BASE as i32), Val::I32(LK_ENV_PTR as i32)];
                for (a, tc) in argv.iter().zip(ptypes) {
                    params.push(match tc {
                        0 => Val::I32(*a as i32),
                        1 => Val::I64(*a),
                        _ => panic!("non-integer arg in this guest"),
                    });
                }
                let rn = temen_par_jit_result_types_len(v);
                let rtypes =
                    unsafe { std::slice::from_raw_parts(temen_par_jit_result_types_ptr(v), rn) };
                let mut results: Vec<Val> = rtypes
                    .iter()
                    .map(|tc| if *tc == 0 { Val::I32(0) } else { Val::I64(0) })
                    .collect();
                let f0 = inst.get_func(&store, "f0").expect("unit exports f0");
                match f0.call(&mut store, &params, &mut results) {
                    Ok(()) => {
                        let slots: Vec<i64> = results
                            .iter()
                            .map(|r| match r {
                                Val::I32(x) => *x as i64,
                                Val::I64(x) => *x,
                                _ => unreachable!(),
                            })
                            .collect();
                        temen_par_deliver_jit_invoke(v, slots.as_ptr(), slots.len());
                    }
                    Err(_) => temen_par_deliver_jit_invoke_trap(v),
                }
            }
            ev => panic!(
                "unexpected par event {ev} (codegen={codegen}, emitted_callee={emitted_callee})"
            ),
        }
    };
    let bounces = store.data().bounces;
    temen_par_free(v);
    (value, invokes, bounces)
}

#[test]
fn par_linked_unit_dispatches_program_functions_through_the_b2_mirror() {
    let _jit = JIT_STATE_LOCK.lock().unwrap_or_else(|e| e.into_inner()); // #1182 — serial single-run
    for emitted_callee in [false, true] {
        let (want, i0, b0) = lk_run(false, emitted_callee);
        assert_eq!(
            want,
            2 * LK_PROBE + LK_K,
            "oracle value (emitted_callee={emitted_callee})"
        );
        assert_eq!(
            (i0, b0),
            (0, 0),
            "the oracle services the invoke interpreted, no events"
        );
        let (got, invokes, bounces) = lk_run(true, emitted_callee);
        assert_eq!(
            got, want,
            "B2 codegen ≡ interpreter (emitted_callee={emitted_callee})"
        );
        assert_eq!(invokes, 1, "the linked unit ran on emitted wasm");
        // Non-vacuity: the unit's `call.dyn` reached program function 1 through the natural prefix —
        // via the bounce shim when interpreter-resident, natively (no bounce) when emitted.
        assert_eq!(
            bounces,
            u32::from(!emitted_callee),
            "bounce iff `F` is interpreter-resident"
        );
    }
}

// ---- #1339: the slot mirror survives the guest's `Jit.release` -----------------------------------

/// #1339 — **the parallel driver's slot mirror is keyed on the unit, not the code handle.**
///
/// `compile → install → release` is the ordinary §22 pattern: the unit stays installed in the shared
/// dispatch table, and the guest drops the code handle it no longer needs (Forth's outer interpreter
/// does exactly this for every colon definition). The mirror a Worker rebuilds its
/// `WebAssembly.Table` from used to record the **handle**, and the wasm fetch resolved through it —
/// so the moment the guest released, the fetch came back `0` for a live installed unit and the
/// Worker nulled its slot, trapping the next emitted `call.dyn` (`IndirectCallToNull`). The
/// cooperative driver had the same bug and was fixed in #1233; this is the parallel twin.
///
/// The guest compiles `f(x) = x + K` from the host-staged blob, installs it (taking slot 1 — the
/// first padding slot past its own single function), releases the handle, then `call.dyn`s the slot.
/// Asserts the dispatch still computes, and then — the regression proper — that **after the release**
/// the mirror still names the unit and the driver can still fetch its emitted wasm by slot: the two
/// reads `worker.js::jitSyncTable` makes to fill that slot.
#[test]
fn par_installed_unit_survives_the_release_of_its_code_handle() {
    // The par §22 statics (the powerbox, the slot mirror, the prefix shims) are process-global, so
    // this shares `JIT_STATE_LOCK` with the other single-run test in this binary (#1182).
    let _jit = JIT_STATE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    use temen_browser::{
        temen_par_compile_jit, temen_par_ev_a, temen_par_jit_slot_unit, temen_par_jit_table_gen,
        temen_par_jit_unit_wasm_by_slot_len, temen_par_powerbox_jit_runtime, temen_par_root,
        temen_par_run, PAR_DONE,
    };

    const K: i64 = 7;
    const X: i64 = 35;
    const BLOB_OFF: usize = 0x6000; // where the host stages the unit blob (as `lk_run` does)

    let unit = {
        let src = format!(
            "memory 16\nfunc (i64) -> (i64) {{\nblock 0 (v0: i64) {{\n  vk = i64.const {K}\n  vs = i64.add v0 vk\n  return vs\n  }}\n}}\n"
        );
        let m = temen_text::parse_module(&src).expect("unit parse");
        temen_verify::verify_module(&m).expect("unit verify");
        temen_encode::encode_module(&m)
    };
    // One function ⇒ the natural prefix is 1 ⇒ `install` takes slot 1 (the first padding slot). The
    // guest hard-codes that for its `call.dyn` and returns the slot so the test pins it too.
    let guest_src = format!(
        r#"memory 16
func (i32) -> (i64) {{
block 0 (v0: i32) {{
  vbp = i64.const {BLOB_OFF}
  vbl = i64.const {}
  vcode = call.cap 11 0 (i64, i64) -> (i64) v0 (vbp, vbl)
  vslot = call.cap 11 3 (i64) -> (i64) v0 (vcode)
  vrel = call.cap 11 2 (i64) -> (i64) v0 (vcode)
  vs1 = i32.const 1
  vx = i64.const {X}
  vres = call.dyn (i64) -> (i64) vs1 (vx)
  vchk = i64.add vslot vrel
  vsum = i64.add vres vchk
  return vsum
  }}
}}
"#,
        unit.len()
    );
    let guest = temen_text::parse_module(&guest_src).expect("guest parse");
    temen_verify::verify_module(&guest).expect("guest verify");
    let guest_bytes = temen_encode::encode_module(&guest);

    assert_eq!(
        temen_par_powerbox_jit_runtime(guest_bytes.as_ptr(), guest_bytes.len()),
        1,
        "runtime-compile powerbox"
    );
    let gen0 = temen_par_jit_table_gen();
    let prog = temen_par_compile_jit(guest_bytes.as_ptr(), guest_bytes.len());
    assert!(!prog.is_null(), "guest compiles");

    // Stage the unit blob where the guest reads it — the host seeds the window, as `lk_run` does.
    let mut win = vec![0u8; 1 << 16];
    win[BLOB_OFF..BLOB_OFF + unit.len()].copy_from_slice(&unit);
    let v = temen_par_root(prog, win.as_mut_ptr(), win.len(), 0);
    assert!(!v.is_null(), "root vCPU builds");
    assert_eq!(temen_par_run(v), PAR_DONE, "the guest runs to completion");
    // `vres` (= X + K) + `vslot` (1, the first padding slot) + `vrel` (0, release ok).
    assert_eq!(
        temen_par_ev_a(v),
        X + K + 1,
        "the installed unit dispatches through its slot after the handle was released"
    );

    assert!(
        temen_par_jit_slot_unit(1) >= 0,
        "the slot mirror must still name the installed unit after `release`"
    );
    assert!(
        temen_par_jit_unit_wasm_by_slot_len(1) > 0,
        "the driver must still fetch the installed unit's emitted wasm after `release`"
    );
    assert_ne!(
        temen_par_jit_table_gen(),
        gen0,
        "the install must advance the mirror generation so a Worker rebuilds"
    );
}
