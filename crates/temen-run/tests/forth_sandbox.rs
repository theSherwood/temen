//! #1234 — **the Forth `sandbox` word**: `s" …" sandbox` runs a program in a confined §14 child.
//!
//! The child is this same kernel, nested — a same-module child over a 2^19 sub-carve of the
//! parent's own window, entered at `child_start` (75). Same-module is what makes it cheap: no
//! `Module` grant and no `Budget`, only an `Instantiator`. It is born with exactly two capabilities,
//! re-granted by name in the spawn record's grant list: `stdout` (so its output joins ours) and
//! `jit` (so it can define words at all). Its memory is the carve and nothing else — the parent's
//! dictionary, REPL stack and heap live below it at addresses the child cannot name.

use temen_run::{Backend, RunConfig};

/// The two tiers that can run a nested copy of this kernel. The **Cranelift** tier cannot, and says
/// so: `compile_child` refuses a §14 child whose module uses §12 fibers or threads ("a §14 JIT child
/// using fibers/threads is not supported yet"), and the Forth kernel has `task`/`yield`/`resume` and
/// `spawn`/`join` — so it cannot be its own JIT child. That frontier is pinned by
/// [`the_jit_tier_declines_a_fiber_bearing_child`] rather than papered over.
const TIERS: [Backend; 2] = [Backend::TreeWalk, Backend::Bytecode];

fn kernel() -> temen_ir::Module {
    let m = temen_text::parse_module(include_str!("../demos/forth/forth.temt"))
        .expect("forth.temt parses");
    temen_verify::verify_module(&m).expect("forth.temt verifies");
    m
}

fn try_run(backend: Backend, program: &str) -> Result<(String, i64), String> {
    let inst = temen_run::instantiate(kernel()).expect("instance");
    let cfg = RunConfig {
        stdin: program.as_bytes().to_vec(),
        ..Default::default()
    };
    let r = inst.run(backend, &cfg)?;
    Ok((
        String::from_utf8_lossy(&r.stdout).into_owned(),
        match r.outcome {
            temen_run::Outcome::Exited(c) => c as i64,
            temen_run::Outcome::Returned(_) => 0,
        },
    ))
}

fn run(backend: Backend, program: &str) -> (String, i64) {
    let inst = temen_run::instantiate(kernel()).expect("instance");
    let cfg = RunConfig {
        stdin: program.as_bytes().to_vec(),
        ..Default::default()
    };
    let r = inst.run(backend, &cfg).expect("the kernel runs");
    (
        String::from_utf8_lossy(&r.stdout).into_owned(),
        match r.outcome {
            temen_run::Outcome::Exited(c) => c as i64,
            temen_run::Outcome::Returned(_) => 0,
        },
    )
}

/// The sandboxed program defines a word and prints, its output lands in *our* stream (§7c stdio
/// inheritance), and the parent keeps running afterwards with its own dictionary intact.
#[test]
fn sandbox_runs_a_program_in_a_confined_child() {
    for backend in TIERS {
        let (out, _) = run(
            backend,
            ": sq ( n -- n ) dup * ;\n\
             7 sq . cr\n\
             s\" : cube ( n -- n ) dup dup * * ; 3 cube . cr\" sandbox drop\n\
             9 sq . cr\n",
        );
        assert_eq!(out, "49 \n27 \n81 \n", "{backend:?}");
    }
}

/// A word defined **inside** the sandbox is not defined outside it: the child builds its own
/// dictionary in its own window, and the parent's tokenizer has never heard of `cube`.
#[test]
fn a_sandboxed_definition_does_not_escape() {
    let (out, _) = run(
        Backend::Bytecode,
        "s\" : cube ( n -- n ) dup dup * * ;\" sandbox drop\n\
         2 cube . cr\n",
    );
    assert!(
        out.contains("?") || out.contains("cube"),
        "the parent must not know `cube`: {out:?}"
    );
}

/// A sandbox cannot nest one: the child's own `Instantiator` arrives as its entry argument and is
/// never registered under a name, so the inner `sandbox` finds nothing and says so.
#[test]
fn a_sandbox_cannot_sandbox() {
    let (out, _) = run(Backend::Bytecode, "s\" 0 0 sandbox . cr\" sandbox drop\n");
    assert!(
        out.contains("no sandbox capability"),
        "inner sandbox must be refused: {out:?}"
    );
}

/// The declared frontier: the Cranelift tier refuses to compile a §14 child that uses fibers or
/// threads, and this kernel does — so `sandbox` there is a refusal, not a silent success. Loud by
/// design (`instantiator_rt`: "a child we cannot compile … is a CapFault, not a silent success").
/// If this starts passing, per-child fiber/thread runtimes landed: move `Backend::Jit` into `TIERS`.
///
/// It is also the pin for the **locked-ctx child hooks**: this kernel uses concurrency, so its JIT
/// run bakes a `*const Mutex<Host>` cap ctx, and the spawn gets far enough to build the child
/// powerbox and bind its manifest through that ctx before the compile refuses. Against the raw
/// (`*mut Host`) hooks that path read the mutex header as a `Host` — a SIGSEGV, not a refusal.
#[test]
fn the_jit_tier_declines_a_fiber_bearing_child() {
    let r = try_run(Backend::Jit, "s\" 1 2 + . cr\" sandbox drop\n");
    let e = r.expect_err("the JIT tier cannot nest this kernel");
    assert!(e.contains("CapFault"), "expected a refusal, got: {e}");
}

/// The issue's first gate: the child redefines a word the parent already has, prints with *its*
/// definition, and the parent's is untouched afterwards. Two dictionaries, two windows.
#[test]
fn a_child_redefinition_does_not_touch_the_parents_word() {
    for backend in TIERS {
        let (out, _) = run(
            backend,
            ": sq ( n -- n ) dup * ;\n\
             5 sq . cr\n\
             s\" : sq ( n -- n ) drop 999 ; 5 sq . cr\" sandbox drop\n\
             5 sq . cr\n",
        );
        assert_eq!(out, "25 \n999 \n25 \n", "{backend:?}");
    }
}

/// The issue's second gate, in the form invariant 2 actually guarantees: confinement is *masking*,
/// so a child reading an address the parent wrote does not trap — it reads its own window at that
/// offset, which the parent never wrote. The parent's cell survives, the child never saw it.
#[test]
fn a_child_cannot_read_the_parents_memory() {
    for backend in TIERS {
        let (out, _) = run(
            backend,
            "4242 200000 !\n\
             200000 @ . cr\n\
             s\" 200000 @ . cr\" sandbox drop\n\
             200000 @ . cr\n",
        );
        assert_eq!(out, "4242 \n0 \n4242 \n", "{backend:?}");
    }
}
