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
        "3 \nline 2: unknown word near bogus\n7 \nline 4: stack effect mismatch near ;\n5 \n6 \n\
         line 8: stack underflow near if\n7 \nline 10: stack underflow near drop\n8 \n"
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

/// `constant` (#1237): `<value> constant name` fixes a value at definition time; each use loads it
/// back. Implemented as a read-only cell, so the value can come from any computation, and a constant
/// composes with colon definitions — byte-identical interp == JIT.
#[test]
fn forth_constant() {
    let out = forth(CONSTANT);
    assert_eq!(out, CONSTANT_OUT);
}

const CONSTANT: &str = ": sq ( n -- n ) dup * ;\n\
    10 constant ten\n\
    ten . cr\n\
    ten ten + . cr\n\
    ten sq . cr\n\
    : area ( r -- a ) sq 3 * ;\n\
    ten area . cr\n\
    2 3 + constant five\n\
    five . cr\n";
const CONSTANT_OUT: &str = "10 \n20 \n100 \n300 \n5 \n";

/// `defer`/`is` (#1237): a forward-declared word whose behavior is bound (and rebound) later. The
/// deferred word inlines a load-from-cell + `execute` at its declared effect, so it composes inside
/// colon definitions and `is` swaps the target at run time — byte-identical interp == JIT.
#[test]
fn forth_defer_is() {
    let out = forth(DEFER);
    assert_eq!(out, DEFER_OUT);
}

const DEFER: &str = ": sq ( n -- n ) dup * ;\n\
    : neg ( n -- n ) 0 swap - ;\n\
    defer ( n -- n ) op\n\
    ' sq is op\n\
    5 op . cr\n\
    ' neg is op\n\
    5 op . cr\n\
    : apply3 ( n -- n ) op op op ;\n\
    2 apply3 . cr\n";
const DEFER_OUT: &str = "25 \n-5 \n-2 \n";

/// Counted loops (#1237): `do`/`loop`/`+loop`/`i`/`j`/`leave`. The loop index and limit ride a
/// return stack (the RVS) carried through every branch as block params, so the data stack stays clean
/// — `sumn`/`mul` accumulate below the loop, `box` nests `i`/`j`, `firsthit` proves `leave`, and
/// `downby` a negative `+loop`. All byte-identical interp == JIT and on the bytecode engine.
#[test]
fn forth_do_loop() {
    let out = forth(DO_LOOP);
    assert_eq!(out, DO_LOOP_OUT);
}

const DO_LOOP: &str = ": sumn ( n -- s ) 0 swap 0 do i + loop ;\n\
    5 sumn . 10 sumn . cr\n\
    : mul ( a b -- p ) 0 swap 0 do over + loop nip ;\n\
    6 7 mul . cr\n\
    : box ( -- ) 2 0 do 2 0 do j . i . space loop loop cr ;\n\
    box\n\
    : firsthit ( n -- ) 0 do i dup 3 > if . leave then drop loop cr ;\n\
    10 firsthit\n\
    : downby ( -- ) 0 10 do i . -2 +loop cr ;\n\
    downby\n";
const DO_LOOP_OUT: &str = "10 45 \n42 \n0 0  0 1  1 0  1 1  \n4 \n10 8 6 4 2 0 \n";

/// Return-stack access and a few more primitives (#1237): `>r`/`r>`/`r@` move values to and from the
/// RVS the counted loops introduced, `char`/`[char]` push a character literal, and `2swap`/`2over`
/// (defined in the prelude over `>r`/`r>`) permute the top two cell-pairs. Byte-identical interp ==
/// JIT and on the bytecode engine.
#[test]
fn forth_return_stack_and_chars() {
    let out = forth(RSTACK);
    assert_eq!(out, RSTACK_OUT);
}

const RSTACK: &str = ": rot3 ( a b c -- b c a ) >r swap r> swap ;\n\
    1 2 3 rot3 . . . cr\n\
    : dupr ( a -- a a ) >r r@ r> ;\n\
    7 dupr . . cr\n\
    char A . [char] z . cr\n\
    1 2 3 4 2swap . . . . cr\n\
    1 2 3 4 2over . . . . . . cr\n\
    : rsum3 ( a b c -- s ) >r + r> + ;\n\
    10 20 30 rsum3 . cr\n";
const RSTACK_OUT: &str = "1 3 2 \n7 7 \n65 122 \n2 1 4 3 \n2 1 4 3 2 1 \n60 \n";

/// A whole program, not one feature: the sieve of Eratosthenes counts the primes below N. It leans on
/// everything at once — `variable`/`here`/`allot` for a byte array, a nested `do` loop whose inner
/// bound is `+loop`-stepped by the outer prime (`j`), `i`/`c@`/`c!` to mark multiples, and `if`/`0=`.
/// That it is byte-identical across interp == JIT (and the bytecode leg) is the language validated as
/// a whole, not just per-word.
#[test]
fn forth_sieve_of_eratosthenes() {
    let out = forth(SIEVE);
    assert_eq!(out, SIEVE_OUT);
}

const SIEVE: &str = "variable arr\n\
    variable lim\n\
    : primes ( n -- c )\n\
      lim !\n\
      here arr !\n\
      lim @ allot\n\
      0\n\
      lim @ 2 do\n\
        arr @ i + c@ 0= if\n\
          1+\n\
          i dup * lim @ < if\n\
            lim @ i dup * do\n\
              1 arr @ i + c!\n\
            j +loop\n\
          then\n\
        then\n\
      loop ;\n\
    10 primes . 30 primes . 100 primes . cr\n";
const SIEVE_OUT: &str = "4 10 25 \n";

/// The playground runs the kernel on the **bytecode** engine: the same transcripts must produce the
/// same bytes there (the card's own gate is `browser/tests/forth_asset.rs` over the built asset).
#[test]
fn forth_on_the_bytecode_engine() {
    for (program, expected) in [
        (FIBERS, FIBERS_OUT),
        (THREADS, THREADS_OUT),
        (EXECUTE, EXECUTE_OUT),
        (EXIT, EXIT_OUT),
        (CONSTANT, CONSTANT_OUT),
        (DEFER, DEFER_OUT),
        (DO_LOOP, DO_LOOP_OUT),
        (RSTACK, RSTACK_OUT),
        (SIEVE, SIEVE_OUT),
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
