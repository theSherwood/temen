//! #2219 — a program's **child image** (`temen_ir::child_image`): the program with its root entry,
//! function 0, replaced by a bootstrap that runs the function it exports as `_child`. A host grants
//! the image to a program that spawns copies of itself, and the program spawns it at function 0.
//!
//! The image keeps every other function at its index, so a `call.dyn` slot number means the same in
//! the image as in the program (the Forth kernel's word table is such numbers); it keeps the program's
//! memory and data; and a run compiles it once however many children run it.
#![cfg(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
))]

use temen_interp::bytecode::{CoopEvent, CoopRun, Footprint};
use temen_interp::{bytecode, run_capture_reserved_with_host, Host, MemLayout, Value};
use temen_ir::{child_image, Module};
use temen_jit::JitOutcome;
use temen_text::parse_module;
use temen_verify::verify_module;

const PARENT_LOG2: u8 = 17;

/// A program whose child entry (function 1) reaches function 2 through `call.dyn` slot 2 and adds 2:
/// 42 in an image that kept function 2 at slot 2. Function 0 is its root entry.
const PROGRAM: &str = "memory 16 shadow 61440 65536
data 1024 \"kept\"
func () -> () {
block 0 () {
  return
  }
}
func (i64) -> (i64) {
block 0 (vi: i64) {
  vs = i32.const 2
  vr = call.dyn () -> (i64) vs ()
  vt = i64.const 2
  vo = i64.add vr vt
  return vo
  }
}
func () -> (i64) {
block 0 () {
  v = i64.const 40
  return v
  }
}
export 0 func \"_start\" 0
export 1 func \"_child\" 1
export 2 func \"forty\" 2
";

fn parse(src: &str) -> Module {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    m
}

fn image() -> Module {
    let m = child_image(&parse(PROGRAM))
        .expect("the program exports `_child`")
        .expect("an image");
    verify_module(&m).expect("the image verifies");
    m
}

#[test]
fn the_image_replaces_only_the_root_entry() {
    let (program, image) = (parse(PROGRAM), image());
    assert_eq!(image.funcs.len(), program.funcs.len(), "no function added");
    assert_eq!(
        image.funcs[1..],
        program.funcs[1..],
        "every other function kept, in place"
    );
    assert_eq!(
        (&image.funcs[0].params, &image.funcs[0].results),
        (&program.funcs[1].params, &program.funcs[1].results),
        "the bootstrap has the child entry's shape"
    );
    assert_eq!(
        image.memory, program.memory,
        "the window and its shadow arena"
    );
    assert_eq!(image.data, program.data);
    let names: Vec<&str> = image.exports.iter().map(|e| e.name.as_str()).collect();
    assert_eq!(names, ["forty"], "neither `_start` nor `_child`");
}

#[test]
fn a_program_without_a_child_entry_has_no_image() {
    let m = parse(&PROGRAM.replace("\"_child\"", "\"child\""));
    assert!(child_image(&m).is_none());
}

#[test]
fn an_unfit_child_entry_is_refused() {
    let at_root = parse(&PROGRAM.replace("\"_child\" 1", "\"_child\" 0"));
    let no_status = parse(&PROGRAM.replace("\"_child\" 1", "\"_child\" 2").replace(
        "func () -> (i64) {\nblock 0 () {\n  v = i64.const 40",
        "func (i32) -> (i64) {\nblock 0 (va: i32) {\n  v = i64.const 40",
    ));
    let calls_root =
        parse(&PROGRAM.replace("  vt = i64.const 2\n", "  call 0 ()\n  vt = i64.const 2\n"));
    for (m, why) in [
        (at_root, "the root entry itself"),
        (no_status, "not a child-entry shape"),
        (calls_root, "the program calls function 0"),
    ] {
        assert!(
            matches!(child_image(&m), Some(Err(_))),
            "{why} must be refused"
        );
    }
}

/// A parent `(inst, module, budget) -> i64` that spawns `module` detached `n` times at function 0,
/// joins each child, and returns the sum of their statuses.
fn parent(n: usize) -> Module {
    let mut body = String::new();
    for i in 0..n {
        body += &format!(
            "  c{i} = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) vinst (vb, me, gz, gz, gz, sl, gz)\n  \
             r{i} = call.cap 6 1 (i32) -> (i64) vinst (c{i})\n  \
             s{} = i64.add s{i} r{i}\n",
            i + 1
        );
    }
    parse(&format!(
        "memory {PARENT_LOG2}
func (i32, i32, i32) -> (i64) {{
block 0 (vinst: i32, vmod: i32, vbud: i32) {{
  me = i64.extend_i32_s vmod
  vb = i64.extend_i32_s vbud
  gz = i64.const 0
  sl = i64.const 16
  s0 = i64.const 0
{body}  return s{n}
  }}
}}
"
    ))
}

fn powerbox() -> (Host, [i32; 3]) {
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, 1 << PARENT_LOG2);
    let modh = host.grant_module(&image());
    let budget = host.grant_budget(-1, -1, -1);
    (host, [inst, modh, budget])
}

/// The image runs as a child on every engine, its `call.dyn` slot reaching the function it names.
#[test]
fn a_child_of_the_image_runs_on_every_engine() {
    let p = parent(1);
    let (mut host, h) = powerbox();
    let mut fuel = 50_000_000u64;
    let init = vec![0u8; 1 << PARENT_LOG2];
    let oracle =
        run_capture_reserved_with_host(&p, 0, &h.map(Value::I32), &mut fuel, &init, 0, &mut host).0;
    assert_eq!(oracle, Ok(vec![Value::I64(42)]), "oracle");

    let (mut host, h) = powerbox();
    let mut fuel = 50_000_000u64;
    let r = bytecode::compile_and_run_with_host(&p, 0, &h.map(Value::I32), &mut fuel, &mut host)
        .expect("the bytecode engine lowers the parent");
    assert_eq!(r, Ok(vec![Value::I64(42)]), "bytecode");

    let (mut host, h) = powerbox();
    let (o, _) = temen_run::jit_cap_run(
        &p,
        0,
        &h.map(i64::from),
        &MemLayout::image(init),
        PARENT_LOG2,
        0,
        &mut host,
        None,
    )
    .expect("Cranelift runs the parent");
    assert_eq!(o, JitOutcome::Returned(vec![42]), "Cranelift");
}

/// Three children of one granted module: the run compiles it once, so it holds two programs, the
/// root's and the module's (it held one per child before #2219).
#[test]
fn a_run_compiles_a_granted_module_once() {
    let (host, h) = powerbox();
    let mut run = CoopRun::new_reserved(
        &parent(3),
        0,
        &h.map(Value::I32),
        u64::MAX,
        host,
        None,
        &[],
        temen_ir::DEFAULT_RESERVED_LOG2,
    )
    .expect("the bytecode engine runs it")
    .expect("it starts");
    match run.run() {
        CoopEvent::Done(r) => assert_eq!(r, vec![Value::I64(126)]),
        _ => panic!("the run did not finish"),
    }
    assert_eq!(
        run.footprint(),
        Footprint {
            windows: 1,
            units: 2,
        },
        "the root's window, and the root's program and the image's"
    );
}
