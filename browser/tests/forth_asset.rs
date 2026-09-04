//! The **Forth playground card** (issue #1214): the committed `web/assets/forth.temen` — the
//! sectorforth-class Forth kernel hand-written in Temen text IR (`crates/temen-run/demos/forth/`),
//! built through `scripts/rebuild-assets.sh` (`ONLY=forth`) — runs the card's own default program
//! through the browser on-ramp (`onramp_exec`, the same entry `temen_run_onramp` wraps for the page)
//! on the bytecode engine, JIT-compiling every colon definition through the §22 `Jit` cap the
//! on-ramp grants a `vm_jit_*`-importing guest. The expected bytes are what the Cranelift JIT
//! prints for the same program under `temen-run` (and what `crates/temen-run/tests/forth.rs` pins
//! interp == JIT), so this is the asset gate: a wire-format change that invalidates the asset, or a
//! kernel change not rebuilt into it, turns this red.

use temen_browser::{onramp_exec, STATUS_OK};

const CARD_PROGRAM: &str = "\\ Forth on Temen: every word below is JIT-compiled to a verified IR unit.
: sq ( n -- n ) dup * ;
: fact ( n -- n ) dup 1 > if dup 1- recurse * else drop 1 then ;
5 sq . 10 fact . cr

\\ loops: begin/until, begin/while/repeat
: countdown ( n -- ) begin dup . 1- dup 0= until drop cr ;
5 countdown
: sum-to ( n -- s ) 0 swap begin dup 0 > while tuck + swap 1- repeat drop ;
100 sum-to . cr

\\ counted loops: do/loop, i is the index; the accumulator stays on the data stack
: sumsq ( n -- s ) 0 swap 0 do i i * + loop ;
5 sumsq . cr

\\ memory: variables, strings, the heap
variable x   42 x !   x @ 1+ x !   x @ . cr
.\" hello, forth\" cr

\\ fibers: a generator word is a task; resume it from any later line
: counter ( x -- y ) begin 1+ dup yield drop again ;
' counter task
dup 0 resume . . cr
dup 10 resume . . cr
drop

\\ threads: run a word on another vCPU, join its result; atomics on a shared cell
: work ( x -- y ) 1000 * ;
' work 7 spawn join . cr
variable hits
: bump ( n -- y ) begin dup 0 > while 1 hits atomic+! drop 1- repeat ;
' bump 100 spawn ' bump 100 spawn join swap join + . hits @ . cr
";

const EXPECTED: &str = "25 3628800 \n5 4 3 2 1 \n5050 \n30 \n43 \nhello, forth\n1 0 \n2 0 \n7000 \n0 200 \n";

#[test]
fn forth_card_program_runs_through_the_onramp() {
    let bytes = include_bytes!("../web/assets/forth.temen");
    let m = temen_encode::decode_module(bytes).expect("decode forth.temen (rebuild the asset?)");
    let out = onramp_exec(&m, CARD_PROGRAM.as_bytes());
    assert_eq!(
        out.status,
        STATUS_OK,
        "the Forth kernel should run cleanly; stdout so far: {}",
        String::from_utf8_lossy(&out.stdout)
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), EXPECTED);
}

/// A compile-time error in a definition is reported per line and the kernel keeps going.
#[test]
fn forth_card_reports_errors_and_recovers() {
    let bytes = include_bytes!("../web/assets/forth.temen");
    let m = temen_encode::decode_module(bytes).expect("decode forth.temen");
    let out = onramp_exec(&m, b"1 2 + . cr\nbogus\n: bad ( n -- n ) dup dup ;\n3 . cr\n");
    assert_eq!(out.status, STATUS_OK);
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "3 \nline 2: unknown word near bogus\nline 3: stack effect mismatch near ;\n3 \n"
    );
}
