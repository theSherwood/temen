//! #1825 — **shared code**: one compile, run as several instances (`CompiledModule::share`,
//! `SharedCode::instance`). A JIT process tree runs a fork twin on its parent's code and every
//! `execve` of a command on one compile of it; `temen-run`'s `jit_shared_code.rs` pins that end to
//! end. This file pins the contract underneath:
//!
//! * each instance dispatches its `call.cap`s into its **own** powerbox, even while another instance
//!   runs the same code on another thread;
//! * shared code cannot be **extended**: a unit defined into it would run in every instance;
//! * code that names an object one instance owns is **not shared** — but a §14 nursery stood up only
//!   for units the module may install (#1726) is not named by its code.

use core::ffi::c_void;
use temen_jit::{CompiledModule, JitError, JitOutcome, Quota, INERT_CAP_THUNK};
use temen_text::parse_module;

/// A `call.cap` whose answer is the powerbox's own: the thunk returns the `i64` its ctx points at.
unsafe extern "C" fn ctx_value_thunk(
    ctx: *mut c_void,
    _mem_base: *mut u8,
    _mem_size: u64,
    _mem_reserved: u64,
    _type_id: u32,
    _op: u32,
    _handle: i32,
    _args: *const i64,
    _n_args: u64,
    results: *mut i64,
    _n_results: u64,
    trap_out: *mut i64,
) {
    unsafe {
        *results = *(ctx as *const i64);
        *trap_out = 0;
    }
}

/// Returns what its one `call.cap` answers.
const ASKS_ITS_POWERBOX: &str = "memory 16
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = call.cap 2 0 () -> (i64) v0 ()
  return v1
  }
}
";

fn compile(
    src: &str,
    thunk: temen_jit::CapThunk,
    ctx: *mut c_void,
    table_log2: u8,
) -> CompiledModule {
    let m = parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    CompiledModule::compile(
        &m,
        0,
        thunk,
        ctx,
        temen_ir::DEFAULT_RESERVED_LOG2,
        None,
        None,
        None,
        None,
        None,
        Quota::default(),
        table_log2,
    )
    .expect("compile")
}

fn answer(cm: &mut CompiledModule) -> i64 {
    match cm.run(&[0], None, None).expect("run").0 {
        JitOutcome::Returned(r) => r[0],
        other => panic!("{other:?}"),
    }
}

#[test]
fn each_instance_answers_from_its_own_powerbox_while_the_others_run() {
    let mut powerboxes = [100i64, 200, 300];
    let [a, b, c] = powerboxes.each_mut().map(|p| p as *mut i64 as usize);
    let mut first = compile(ASKS_ITS_POWERBOX, ctx_value_thunk, a as *mut c_void, 0);
    let code = first.share().expect("only code");
    std::thread::scope(|s| {
        for ctx in [b, c] {
            let code = &code;
            s.spawn(move || {
                let mut cm = code.instance(ctx as *mut c_void);
                for _ in 0..200 {
                    assert_eq!(answer(&mut cm), unsafe { *(ctx as *const i64) });
                }
            });
        }
        for _ in 0..200 {
            assert_eq!(
                answer(&mut first),
                100,
                "the module shared from runs on as the first instance"
            );
        }
    });
}

#[test]
fn no_instance_of_shared_code_can_extend_it() {
    let unit = parse_module("func (i32) -> (i32) {\nblock 0 (v0: i32) {\n  return v0\n  }\n}\n")
        .expect("parse unit");
    let extends = |cm: &mut CompiledModule| cm.define_extra(&unit.funcs, &unit.types);
    let mut first = compile(ASKS_ITS_POWERBOX, INERT_CAP_THUNK, core::ptr::null_mut(), 4);
    let code = first.share().expect("only code");
    let mut other = code.instance(core::ptr::null_mut());
    for cm in [&mut first, &mut other] {
        assert!(
            matches!(extends(cm), Err(JitError::Unsupported(_))),
            "a unit defined into shared code would run in every instance"
        );
    }

    // Code something was defined into is more than the compile: it is not shared at all.
    let mut extended = compile(ASKS_ITS_POWERBOX, INERT_CAP_THUNK, core::ptr::null_mut(), 4);
    extends(&mut extended).expect("an unshared module defines a unit");
    assert!(extended.share().is_none());
}

#[test]
fn code_naming_an_object_one_instance_owns_is_not_shared() {
    // An `Instantiator` site bakes the address of the run's §14 nursery.
    let nests = "memory 16
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = call.cap 6 1 (i32) -> (i64) v0 (v0)
  return v1
  }
}
";
    let cm = compile(nests, INERT_CAP_THUNK, core::ptr::null_mut(), 0);
    // Only where the JIT nests at all (the fiber runtime's targets) does it stand up a nursery.
    assert_eq!(cm.share().is_none(), temen_fiber::supported());

    // Install room gets a nursery too, for the units the module may install (#1726) — but the
    // module's own code names none, and shared code takes no unit.
    let roomy = compile(ASKS_ITS_POWERBOX, INERT_CAP_THUNK, core::ptr::null_mut(), 4);
    assert!(roomy.share().is_some());
}
