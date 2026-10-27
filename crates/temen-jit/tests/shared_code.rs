//! #1825 — **shared code**: one compile, run as several instances (`CompiledModule::share`,
//! `SharedCode::instance`). A JIT process tree runs a fork twin on its parent's code and every
//! `execve` of a command on one compile of it; `temen-run`'s `jit_shared_code.rs` pins that end to
//! end. This file pins the contract underneath:
//!
//! * each instance dispatches its `call.cap`s into its **own** powerbox, even while another instance
//!   runs the same code on another thread;
//! * shared code cannot be **extended**: a unit defined into it would run in every instance;
//! * every runtime the code runs on, each instance stands up its own (temen#2389): the thread
//!   domain its waits, spawns and joins use, the fiber table and runtime its `cont.*` sites switch
//!   on, the §14 nursery its `Instantiator` sites spawn through, the `setjmp` table. The code loads
//!   each from its instance's context or finds it through a thread-local its entry sets, so it names
//!   none of them. (Before, the sites baked some, so code that spawned, ran fibers or used `setjmp`
//!   compiled once per process.)
//! * an instance polls the cells it is given, not the compile's: shared code outlives the run that
//!   compiled it (#2145).

use core::ffi::c_void;
use temen_jit::{CompiledModule, FuelCell, InstanceAddrs, JitError, JitOutcome, INERT_CAP_THUNK};
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
    _vm: &temen_jit::VmCtx,
) {
    unsafe {
        *results = *(ctx as *const i64);
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
        table_log2,
    )
    .expect("compile")
}

/// Answers 7, charging fuel when compiled with fuel checks.
const SEVEN: &str = "memory 16
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const 7
  return v1
  }
}
";

/// `src` compiled with fuel checks against `cell`.
fn compile_metered(src: &str, cell: *mut FuelCell) -> CompiledModule {
    let m = parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    CompiledModule::compile(
        &m,
        0,
        INERT_CAP_THUNK,
        core::ptr::null_mut(),
        temen_ir::DEFAULT_RESERVED_LOG2,
        None,
        None,
        Some(cell),
        None,
        None,
        0,
    )
    .expect("compile")
}

/// An instance's addresses with only its powerbox: these compiles arm no kill-path, fuel or signals.
fn powerbox(cap_ctx: *mut c_void) -> InstanceAddrs {
    InstanceAddrs {
        cap_ctx,
        ..InstanceAddrs::NONE
    }
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
                let mut cm = code.instance(powerbox(ctx as *mut c_void), None);
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
    let mut other = code.instance(powerbox(core::ptr::null_mut()), None);
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

/// Runs `first` and two more instances of the code it shares at once, each `n` times on a thread
/// of its own, and checks each answers `want` every time.
fn each_instance_answers(mut first: CompiledModule, n: usize, want: i64) {
    let code = first.share().expect("only code");
    std::thread::scope(|s| {
        for _ in 0..2 {
            let code = &code;
            s.spawn(move || {
                let mut cm = code.instance(powerbox(core::ptr::null_mut()), None);
                for _ in 0..n {
                    assert_eq!(answer(&mut cm), want);
                }
            });
        }
        for _ in 0..n {
            assert_eq!(
                answer(&mut first),
                want,
                "the module shared from runs on as the first instance"
            );
        }
    });
}

#[test]
fn code_that_spawns_children_is_shared_and_each_instance_spawns_through_its_own_nursery() {
    // The JIT nests only where its fiber runtime runs.
    if !temen_fiber::supported() {
        return;
    }
    // A join through an `Instantiator` the powerbox does not grant: each instance's nursery refuses
    // it, while the others run.
    let nests = "memory 16
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1_st, v1 = call.cap 6 1 (i32) -> (i64, i64) v0 (v0)
  return v1_st
  }
}
";
    let outcome = |cm: &mut CompiledModule| format!("{:?}", cm.run(&[0], None, None).map(|r| r.0));
    let mut first = compile(nests, INERT_CAP_THUNK, core::ptr::null_mut(), 0);
    let refused = outcome(&mut first);
    let code = first.share().expect("only code");
    std::thread::scope(|s| {
        for _ in 0..2 {
            let (code, refused) = (&code, &refused);
            s.spawn(move || {
                let mut cm = code.instance(powerbox(core::ptr::null_mut()), None);
                for _ in 0..50 {
                    assert_eq!(&outcome(&mut cm), refused);
                }
            });
        }
        for _ in 0..50 {
            assert_eq!(outcome(&mut first), refused);
        }
    });

    // Each instance's nursery is over its own thread domain.
    let [a, b] = [(), ()].map(|_| code.instance(powerbox(core::ptr::null_mut()), None));
    let hubs = [first.park_hub(), a.park_hub(), b.park_hub()];
    assert!(hubs.iter().all(|&h| h != 0), "{hubs:?}");
    assert!(hubs[0] != hubs[1] && hubs[1] != hubs[2] && hubs[0] != hubs[2]);

    // Install room gets a nursery too, for the units the module may install (#1726) — but the
    // module's own code names none, and shared code takes no unit.
    let roomy = compile(ASKS_ITS_POWERBOX, INERT_CAP_THUNK, core::ptr::null_mut(), 4);
    assert!(roomy.share().is_some());
}

#[test]
fn code_that_spawns_threads_is_shared_and_each_instance_runs_them_on_its_own_domain() {
    if !temen_fiber::supported() {
        return;
    }
    // Spawns a vCPU that answers its argument plus one, and joins it.
    let spawns = "func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const 0
  v2 = i64.const 41
  v3 = thread.spawn 1 v1 v2
  v4_st, v4 = thread.join v3
  return v4
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 1
  v3 = i64.add v1 v2
  return v3
  }
}
";
    each_instance_answers(
        compile(spawns, INERT_CAP_THUNK, core::ptr::null_mut(), 0),
        20,
        42,
    );
}

#[test]
fn code_that_runs_fibers_is_shared_and_each_instance_switches_on_its_own_runtime() {
    if !temen_fiber::supported() {
        return;
    }
    // A fiber over function 1 suspends its argument plus 777, then returns the next resume's
    // argument: the two resumes answer 782 and 5.
    let fibers = "memory 16
func (i32) -> (i64) {
block 0 (v0: i32) {
  vf = i32.const 1
  vsp = i64.const 0
  vk = cont.new vf vsp
  vx = i64.const 5
  vs1, vv1 = cont.resume vk vx
  vs2, vv2 = cont.resume vk vx
  vr = i64.add vv1 vv2
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 777
  v3 = i64.add v1 v2
  v4 = suspend v3
  return v4
  }
}
";
    each_instance_answers(
        compile(fibers, INERT_CAP_THUNK, core::ptr::null_mut(), 0),
        50,
        787,
    );
}

#[cfg(unix)]
#[test]
fn code_that_uses_setjmp_is_shared_and_each_instance_keeps_its_own_jmp_bufs() {
    if !temen_fiber::supported() {
        return;
    }
    each_instance_answers(
        compile(JUMPS, INERT_CAP_THUNK, core::ptr::null_mut(), 0),
        200,
        7,
    );
}

/// `setjmp` at 64, then a `longjmp` back to it with 7: the second return of `setjmp` answers.
#[cfg(unix)]
const JUMPS: &str = "memory 16
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const 64
  v2 = setjmp v1
  v3 = i32.const 0
  v4 = i32.eq v2 v3
  br_if v4 1() 2(v2)
  }
block 1 () {
  v5 = i64.const 64
  v6 = i32.const 7
  longjmp v5 v6
  unreachable
  }
block 2 (v7: i32) {
  v8 = i64.extend_i32_u v7
  return v8
  }
}
";

/// A unit's `setjmp` keeps its buffer in its instance's `setjmp` table, which only a module that
/// uses `setjmp` stands up: one defined into any other module would run over none (temen#2452).
#[cfg(unix)]
#[test]
fn a_unit_uses_setjmp_only_where_its_instance_has_a_setjmp_table() {
    if !temen_fiber::supported() {
        return;
    }
    let unit = parse_module(
        "func (i64) -> (i32) {\nblock 0 (v0: i64) {\n  v1 = setjmp v0\n  return v1\n  }\n}\n",
    )
    .expect("parse unit");
    let mut plain = compile(ASKS_ITS_POWERBOX, INERT_CAP_THUNK, core::ptr::null_mut(), 4);
    assert!(
        matches!(
            plain.define_extra(&unit.funcs, &unit.types),
            Err(JitError::Unsupported(_))
        ),
        "a module without a setjmp table took a unit that uses setjmp"
    );
    let mut jumps = compile(JUMPS, INERT_CAP_THUNK, core::ptr::null_mut(), 4);
    jumps
        .define_extra(&unit.funcs, &unit.types)
        .expect("a module that uses setjmp takes a unit that does");
}

#[test]
fn code_that_only_waits_is_shared_and_each_instance_waits_on_its_own_domain() {
    // The JIT waits only where its thread runtime runs (the fiber runtime's targets).
    if !temen_fiber::supported() {
        return;
    }
    // Waits 1 ms on a word nobody notifies, and answers the wait's status.
    let waits = "memory 16
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const 60000
  v2 = i32.const 0
  v3 = i64.const 1000000
  v4 = i32.atomic.wait v1 v2 v3
  v5 = i64.extend_i32_u v4
  return v5
  }
}
";
    const TIMED_OUT: i64 = 2;
    let mut first = compile(waits, INERT_CAP_THUNK, core::ptr::null_mut(), 0);
    let code = first
        .share()
        .expect("waiting needs no runtime but a domain");
    std::thread::scope(|s| {
        for _ in 0..2 {
            let code = &code;
            s.spawn(move || {
                let mut cm = code.instance(powerbox(core::ptr::null_mut()), None);
                for _ in 0..20 {
                    assert_eq!(answer(&mut cm), TIMED_OUT);
                }
            });
        }
        for _ in 0..20 {
            assert_eq!(
                answer(&mut first),
                TIMED_OUT,
                "the module shared from runs on as the first instance"
            );
        }
    });
}

#[test]
fn an_instance_charges_the_fuel_cell_it_is_given() {
    // #2145: a later run instances code an earlier run compiled, and charges its own cell.
    let mut compiled_with = FuelCell::fixed(100);
    let first = compile_metered(SEVEN, &mut *compiled_with);
    let code = first.share().expect("only code");
    let mut later = FuelCell::fixed(100);
    let mut cm = code.instance(
        InstanceAddrs {
            fuel: &mut *later,
            ..InstanceAddrs::NONE
        },
        None,
    );
    assert_eq!(answer(&mut cm), 7);
    assert!(later.left < 100, "the instance charged its own cell");
    assert_eq!(compiled_with.left, 100, "and not the compile's");
}

#[test]
#[should_panic(expected = "fuel cell")]
fn an_instance_of_fuel_checked_code_needs_a_fuel_cell() {
    let mut cell = FuelCell::fixed(100);
    let first = compile_metered(SEVEN, &mut *cell);
    let code = first.share().expect("only code");
    let _ = code.instance(powerbox(core::ptr::null_mut()), None);
}
