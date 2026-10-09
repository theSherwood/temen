//! Thread-var (`tvar`) lowering (NIM.md §3d).
//!
//! Leng marks a thread-local `tvar` (nimony's `__thread`; the allocator and exception state in the
//! real `system` module are thread-vars). How temen-leng lowers one depends on whether the program
//! can start a thread, which it can only through the compute shim's `pthread_create`:
//!
//! - **It cannot:** a `tvar` lowers **identically to a `gvar`**, one plain global at a fixed window
//!   offset. The program has one thread, so a thread-local has exactly one instance and a plain global
//!   *is* that instance. The first tests pin this: writes survive across calls, non-zero initializers
//!   seed it, and it links across modules the same as a `gvar`.
//! - **It can:** a `tvar` is a thread-local in the IR's sense (#1715), read at this thread's block
//!   plus its offset. The tests at the bottom pin that each thread reads its own copy, and that a
//!   sibling unit's reference reaches the same one.
//!
//! Every test runs on the tree-walker and the JIT.

use temen_interp::Value;
use temen_ir::LinkUnit;
use temen_leng::LengModule;

/// Run func `idx` on both engines; assert §9 parity; return the i64 result.
fn run(m: &temen_ir::Module, idx: u32, args: &[i64]) -> i64 {
    temen_verify::verify_module(m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    let ivals: Vec<Value> = args.iter().map(|&n| Value::I64(n)).collect();
    let mut fuel = u64::MAX;
    let interp = temen_interp::run(m, idx, &ivals, &mut fuel).expect("interp");
    let n = match interp.as_slice() {
        [Value::I64(n)] => *n,
        o => panic!("expected i64, got {o:?}"),
    };
    let jit = match temen_jit::compile_and_run(m, idx, args).expect("jit") {
        temen_jit::JitOutcome::Returned(v) => v,
        o => panic!("jit: {o:?}"),
    };
    assert_eq!(jit.as_slice(), &[n], "§9 interp/JIT parity");
    n
}

/// A `tvar` is real, persistent backing store: `bump` writes it, later calls read the accumulated
/// value back. Same shape as `whole_module::globals_const_and_intramodule_calls`, but the counter is
/// a **thread-var** — so this proves `tvar` lowers to a plain global that persists across calls.
#[test]
fn thread_var_persists_across_calls_like_a_global() {
    let leng = "\
(stmts
 (tvar :counter.0. . (i +64) .)
 (proc :bump.0. . (void) .
  (stmts .
   (asgn counter.0. (add (i +64) counter.0. 1))))
 (proc :bumpN.0. (params (param :n.0 . (i +64))) (void) .
  (stmts .
   (var :i.0 . (i +64) 0)
   (while (lt i.0 n.0)
    (stmts .
     (call bump.0.)
     (asgn i.0 (add (i +64) i.0 1))))))
 (proc :main.0. (params (param :n.0 . (i +64))) (i +64) .
  (stmts .
   (call bumpN.0. n.0)
   (ret counter.0.))))";
    let text = temen_leng::translate_to_text(leng).unwrap();
    assert!(
        text.starts_with("memory "),
        "a tvar needs a window:\n{text}"
    );
    let m = temen_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    // main is func 2; the thread-var counter starts 0 and is bumped n times → returns n.
    assert_eq!(run(&m, 2, &[0]), 0);
    assert_eq!(run(&m, 2, &[7]), 7);
    assert_eq!(run(&m, 2, &[100]), 100);
}

/// A `tvar` with a non-zero initializer seeds the window like a `gvar` initializer does.
#[test]
fn thread_var_nonzero_initializer_seeds_the_window() {
    let leng = "\
(stmts
 (tvar :g.0. . (i +64) 5)
 (tvar :h.0. . (i +32) 7)
 (proc :sum.0. . (i +64) . (stmts . (ret (add (i +64) g.0. (conv (i +64) h.0.))))))";
    let text = temen_leng::translate_to_text(leng).unwrap();
    assert!(
        text.contains("data "),
        "a non-zero tvar init emits a data segment:\n{text}"
    );
    let m = temen_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    assert_eq!(
        run(&m, 0, &[]),
        12,
        "5 + 7 read from the seeded thread-vars"
    );
}

/// A `tvar` links across modules exactly like a `gvar`: module `w`'s proc reads and writes a
/// thread-var `g` **defined in module `s`** (a `data.sym` the linker binds to `s`'s exported global).
/// This is the shape of the real allocator's thread-vars in `system`, referenced from user code —
/// write-then-read-back through the cross-module symbol must reach `s`'s storage, so `rw(v) = v`.
/// (Mirrors `link::cross_module_global_read_write`, with `g` a `tvar`.)
#[test]
fn thread_var_links_cross_module_like_a_global() {
    let mod_s = "\
(stmts
 (tvar :g.0. . (i +64) 0))";
    let mod_w = "\
(stmts
 (proc :rw.0. (params (param :v.0 . (i +64))) (i +64) .
  (stmts .
   (asgn g.0.mods v.0)
   (ret g.0.mods))))";
    let linked = temen_leng::link_units(&[
        LengModule {
            stem: "modw",
            src: mod_w,
            names: &["rw.0."],
        },
        LengModule {
            stem: "mods",
            src: mod_s,
            names: &[], // data-only unit: it just defines (and exports) the thread-var `g`
        },
    ])
    .unwrap_or_else(|e| panic!("link: {e}"));
    assert_eq!(
        run(&linked, 0, &[42]),
        42,
        "write then read the external tvar"
    );
    assert_eq!(run(&linked, 0, &[-5]), -5);
}

// ---------------------------------------------------------------------------
// A program that can start a thread (NIM.md §3d): its `tvar`s are thread-locals (#1715).
//
// Once a unit declares `pthread_create`, the import the compute shim starts a thread with, the link
// puts each `tvar` in its unit's thread-local template and reads it at this thread's block plus its
// offset. The block is the vCPU's `vcpu.tls` word, or the root block while that word is 0. These tests
// let a driver set the word, standing in for the shim's thread start.

/// The declaration that makes a program threaded: nimony's `std/rawthreads` spelling, uncalled.
const PTHREAD_CREATE: &str = " (proc :pthread_create.0. (params (param :t.0 . (i +64)) (param :a.0 . (i +64)) \
(param :f.0 . (i +64)) (param :x.0 . (i +64))) (i +32) (pragmas (importc \"pthread_create\")) (stmts .))";

/// `counter` and its accessors, `bump.0.mods` and `get.0.mods`, in a program that can start a thread.
fn tvar_accessors() -> String {
    format!(
        "\
(stmts
{PTHREAD_CREATE}
 (tvar :counter.0. . (i +64) .)
 (proc :bump.0. (params (param :n.0 . (i +64))) (void) .
  (stmts .
   (asgn counter.0. (add (i +64) counter.0. n.0))))
 (proc :get.0. . (i +64) . (stmts . (ret counter.0.))))"
    )
}

/// Link nimony-shaped units with a hand-written `driver`, which becomes the last unit.
fn link_with_driver(units: &[(&str, &str)], driver: &str) -> temen_ir::Module {
    let units: Vec<temen_leng::WholeModule> = units
        .iter()
        .map(|&(stem, src)| temen_leng::WholeModule { stem, src })
        .collect();
    let driver = temen_text::parse_module(driver).expect("driver parses");
    temen_leng::link_whole_with_runtime(
        &units,
        vec![LinkUnit {
            module: driver,
            exports: vec![("_start".into(), 0)],
            ..Default::default()
        }],
    )
    .unwrap_or_else(|e| panic!("link: {e}"))
}

/// The `_start` the driver exports, as a function index of the linked module.
fn start(m: &temen_ir::Module) -> u32 {
    m.exports
        .iter()
        .find(|e| e.name == "_start")
        .expect("_start")
        .func
}

#[test]
fn a_tvar_is_a_thread_local_only_once_the_program_can_start_a_thread() {
    // The switch is the `pthread_create` declaration: with it the counter is read through
    // `vcpu.tls`; without it the program has one thread and the counter is a plain global.
    let threaded = tvar_accessors();
    let single = threaded.replace(PTHREAD_CREATE, "");
    let text = |src: &str| {
        let m = temen_leng::link_whole_with_runtime(
            &[temen_leng::WholeModule { stem: "mods", src }],
            Vec::new(),
        )
        .unwrap_or_else(|e| panic!("link: {e}"));
        temen_text::print_module(&m)
    };
    assert!(
        text(&threaded).contains("vcpu.tls.get"),
        "a program that can start a thread reads its tvar through vcpu.tls"
    );
    assert!(
        !text(&single).contains("vcpu.tls"),
        "a program with one thread keeps its tvar a plain global"
    );
}

#[test]
fn each_thread_reads_its_own_copy_of_a_tvar() {
    // The root thread (word 0) bumps its copy by 1, block B0's by 3 and B1's by 5; each reads back
    // only its own: 1, 3 and 5 → 10305. One shared global would read 9 three times.
    // B0 = 18432 and B1 = 20480 are zeroed scratch above the NULL guard (#1094), below the globals.
    let src = tvar_accessors();
    let m = link_with_driver(
        &[("mods", &src)],
        "\
memory 16
import 0 \"bump.0.mods\" (i64) -> ()
import 1 \"get.0.mods\" () -> (i64)
func 0 () -> (i64) {
block 0 () {
  one = i64.const 1
  call.import 0 (one)
  b0 = i64.const 18432
  vcpu.tls.set b0
  three = i64.const 3
  call.import 0 (three)
  b1 = i64.const 20480
  vcpu.tls.set b1
  five = i64.const 5
  call.import 0 (five)
  zero = i64.const 0
  vcpu.tls.set zero
  g = call.import 1 ()
  vcpu.tls.set b0
  g0 = call.import 1 ()
  vcpu.tls.set b1
  g1 = call.import 1 ()
  k = i64.const 100
  hi = i64.mul g k
  hi2 = i64.mul hi k
  mid = i64.mul g0 k
  s = i64.add hi2 mid
  r = i64.add s g1
  return r
  }
}
",
    );
    assert_eq!(run(&m, start(&m), &[]), 10305);
}

#[test]
fn a_sibling_units_tvar_is_the_same_thread_local() {
    // `mods` defines `g` behind a filler `aa`, so `g` is not at offset 0, and reads it by its local
    // name; `modw` writes and reads it by its cross-module name `g.0.mods`. In block B0, `rw(42)`
    // and `peek()` must meet in one slot (42); in block B1 `g` is still 0. → 42*100 + 0 = 4200.
    let mod_s = format!(
        "\
(stmts
{PTHREAD_CREATE}
 (tvar :aa.0. . (i +64) .)
 (tvar :g.0. . (i +64) .)
 (proc :peek.0. . (i +64) . (stmts . (ret g.0.))))"
    );
    let mod_w = "\
(stmts
 (proc :rw.0. (params (param :v.0 . (i +64))) (i +64) .
  (stmts .
   (asgn g.0.mods v.0)
   (ret g.0.mods))))";
    let m = link_with_driver(
        &[("modw", mod_w), ("mods", &mod_s)],
        "\
memory 16
import 0 \"rw.0.modw\" (i64) -> (i64)
import 1 \"peek.0.mods\" () -> (i64)
func 0 () -> (i64) {
block 0 () {
  b0 = i64.const 18432
  vcpu.tls.set b0
  fortytwo = i64.const 42
  w = call.import 0 (fortytwo)
  p0 = call.import 1 ()
  b1 = i64.const 20480
  vcpu.tls.set b1
  p1 = call.import 1 ()
  k = i64.const 100
  hi = i64.mul p0 k
  r = i64.add hi p1
  return r
  }
}
",
    );
    assert_eq!(run(&m, start(&m), &[]), 4200);
}

#[test]
fn a_thread_local_with_an_initial_value_fails_closed() {
    // nimony rejects a `threadvar` with an initial value, and the template is zeros, so one that
    // reaches leng is malformed: a link error, not a thread that starts with the wrong value.
    let src = format!(
        "\
(stmts
{PTHREAD_CREATE}
 (tvar :g.0. . (i +64) 5)
 (proc :get.0. . (i +64) . (stmts . (ret g.0.))))"
    );
    let err = temen_leng::link_whole_with_runtime(
        &[temen_leng::WholeModule {
            stem: "mods",
            src: &src,
        }],
        Vec::new(),
    )
    .expect_err("a thread-local with an initial value must fail closed");
    assert!(
        err.to_string().contains("initial value"),
        "a specific error: {err}"
    );
}
