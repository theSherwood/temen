//! `demos/forth/forth.temt` — the sectorforth-class Forth whose kernel is hand-written Temen text
//! IR and whose every colon definition is JIT-compiled through the §22 `Jit` capability (issue
//! #1214). Each transcript runs under `run_diff`: the tree-walk interpreter and the Cranelift JIT
//! must agree byte-for-byte on stdout — the guest-emitted units are compiled by both engines.

#![cfg(all(unix, target_arch = "x86_64"))]

use temen_run::{instantiate, RunConfig};

fn forth(program: &str) -> String {
    let m = temen_text::parse_module(include_str!("../demos/forth/forth.temt"))
        .expect("forth.temt parses");
    let inst = instantiate(m).expect("forth.temt instantiates");
    let cfg = RunConfig {
        stdin: program.as_bytes().to_vec(),
        ..RunConfig::default()
    };
    let run = inst
        .run_diff(&cfg)
        .expect("interp == JIT differential over the Forth transcript");
    String::from_utf8(run.stdout).expect("utf-8 stdout")
}

#[test]
fn forth_arithmetic_and_repl_stack() {
    // The persistent REPL stack: values left on one line are consumed by the next.
    assert_eq!(forth("1 2 + .\n"), "3 ");
    assert_eq!(forth("1 2\n+ . cr\n"), "3 \n");
    assert_eq!(forth("7 -3 * . 17 5 mod . 17 5 / . cr\n"), "-21 2 3 \n");
}

#[test]
fn forth_colon_definitions_control_flow_and_recursion() {
    let out = forth(
        "\\ control flow and loops\n\
         : fact ( n -- n ) dup 1 > if dup 1- recurse * else drop 1 then ;\n\
         5 fact . 10 fact . cr\n\
         : fib ( n -- n ) dup 2 < if else 1- dup recurse swap 1- recurse + then ;\n\
         10 fib . cr\n\
         : countdown ( n -- ) begin dup . 1- dup 0= until drop cr ;\n\
         5 countdown\n\
         : sum-to ( n -- s ) 0 swap begin dup 0 > while tuck + swap 1- repeat drop ;\n\
         100 sum-to . cr\n\
         1 2 3 rot . . . cr\n\
         -7 abs . -7 . cr\n\
         3 4 min . 3 4 max . cr\n\
         2 3 over . . . cr\n",
    );
    assert_eq!(
        out,
        "120 3628800 \n55 \n5 4 3 2 1 \n5050 \n1 3 2 \n7 -7 \n3 4 \n2 3 2 \n"
    );
}

#[test]
fn forth_memory_words_and_strings() {
    let out = forth(
        "variable x\n\
         42 x !\n\
         x @ . cr\n\
         x @ 1+ x !  x @ . cr\n\
         .\" hello, forth\" cr\n\
         s\" abc\" type cr\n\
         create buf 10 allot\n\
         65 buf c! 66 buf 1+ c!\n\
         buf 2 type cr\n\
         here 8 allot here swap - . cr\n",
    );
    assert_eq!(out, "42 \n43 \nhello, forth\nabc\nAB\n8 \n");
}

#[test]
fn forth_errors_recover_per_line() {
    let out = forth(
        "1 2 + . cr\n\
         bogus\n\
         3 4 + . cr\n\
         : bad ( n -- n ) dup dup ;\n\
         5 . cr\n\
         : ok ( n -- n ) 1+ ;\n\
         5 ok . cr\n\
         if\n\
         7 . cr\n\
         drop\n\
         8 . cr\n",
    );
    assert_eq!(
        out,
        "3 \nunknown word near bogus\n7 \nstack effect mismatch near ;\n5 \n6 \n\
         stack underflow near if\n7 \nstack underflow near drop\n8 \n"
    );
}

/// Issue #1214 limitation 1, pinned: a fiber created by one REPL line is resumed by later lines.
/// Top-level lines run through `install` + `call.dyn` + `uninstall` (never `invoke`), so the fibers
/// live in the kernel vCPU's own registry — a fiber created under a nested `invoke` faulted on the
/// tree-walker once that invoke returned — and every fiber/thread body starts at the module-0
/// trampoline (`forth.temt` func 66), which is what the bytecode engine can resolve.
#[test]
fn forth_fibers_survive_across_repl_lines() {
    let out = forth(FIBERS);
    assert_eq!(out, FIBERS_OUT);
}

const FIBERS: &str = ": counter ( x -- y ) begin 1+ dup yield drop again ;\n\
    ' counter task\n\
    dup 0 resume . . cr\n\
    dup 10 resume . . cr\n\
    drop\n\
    : gen3 ( x -- y ) drop 1 yield drop 2 yield drop 3 yield drop 4 ;\n\
    ' gen3 task\n\
    dup 0 resume . . dup 0 resume . . dup 0 resume . . dup 0 resume . . cr\n\
    drop\n\
    ' counter task ' counter task\n\
    over 100 resume . . dup 200 resume . . cr\n\
    over 0 resume . . dup 0 resume . . cr\n\
    2drop\n";
const FIBERS_OUT: &str = "1 0 \n2 0 \n1 0 2 0 3 0 4 1 \n101 0 201 0 \n102 0 202 0 \n";

/// Threads (`spawn`/`join` over the kernel's `thread.spawn` trampoline) and the atomic templates.
#[test]
fn forth_threads_and_atomics() {
    let out = forth(THREADS);
    assert_eq!(out, THREADS_OUT);
}

const THREADS: &str = ": work ( x -- y ) 1000 * ;\n\
    ' work 7 spawn join . cr\n\
    variable c\n\
    : bump ( x -- y ) begin dup 0 > while 1 c atomic+! drop 1- repeat ;\n\
    ' bump 100 spawn ' bump 100 spawn join swap join + . c @ . cr\n\
    1 2 c cas . c atomic@ . 5 c atomic! c @ . cr\n";
const THREADS_OUT: &str = "7000 \n0 200 \n200 200 5 \n";

/// Typed `execute` (#1237): a runtime `call.dyn` dispatching on an xt (`' word`) at run time —
/// `execute0 ( xt -- )`, `execute1 ( x xt -- y )`, `execute2 ( a b xt -- y )`. Proves first-class,
/// runtime-dispatched execution: the last line stores an xt in a variable and calls it back
/// (a deferred word / dispatch-table cell), byte-identical interp == JIT.
#[test]
fn forth_typed_execute() {
    let out = forth(EXECUTE);
    assert_eq!(out, EXECUTE_OUT);
}

const EXECUTE: &str = ": sq ( n -- n ) dup * ;\n\
    6 ' sq execute1 . cr\n\
    : showit ( -- ) 42 . cr ;\n\
    ' showit execute0\n\
    : add3 ( a b -- n ) + 3 + ;\n\
    7 10 ' add3 execute2 . cr\n\
    variable xt\n\
    ' sq xt !\n\
    9 xt @ execute1 . cr\n";
const EXECUTE_OUT: &str = "36 \n42 \n20 \n81 \n";

/// `exit` (#1237): an early return from a colon definition. It terminates the current word body
/// with whatever the stack effect promises on every path, so a guarded `exit` and the fall-through
/// must agree — proving the compiler closes the live block at `exit` exactly as it does at `;`.
#[test]
fn forth_exit() {
    let out = forth(EXIT);
    assert_eq!(out, EXIT_OUT);
}

const EXIT: &str = ": g ( n -- n ) dup 10 < if exit then 99 + ;\n\
    5 g . 20 g . cr\n\
    : h ( n -- n ) dup 0 < if drop 0 exit then dup * ;\n\
    -4 h . 4 h . cr\n\
    : e ( -- ) 1 . exit 2 . ;\n\
    e cr\n";
const EXIT_OUT: &str = "5 119 \n0 16 \n1 \n";

/// The playground runs the kernel on the **bytecode** engine: the same transcripts must produce the
/// same bytes there (the card's own gate is `browser/tests/forth_asset.rs` over the built asset).
#[test]
fn forth_on_the_bytecode_engine() {
    for (program, expected) in [
        (FIBERS, FIBERS_OUT),
        (THREADS, THREADS_OUT),
        (EXECUTE, EXECUTE_OUT),
        (EXIT, EXIT_OUT),
    ] {
        let m = temen_text::parse_module(include_str!("../demos/forth/forth.temt")).unwrap();
        let inst = instantiate(m).unwrap();
        let cfg = RunConfig {
            stdin: program.as_bytes().to_vec(),
            ..RunConfig::default()
        };
        let run = inst
            .run(temen_run::Backend::Bytecode, &cfg)
            .expect("bytecode engine runs the Forth transcript");
        assert_eq!(String::from_utf8(run.stdout).unwrap(), expected);
    }
}
