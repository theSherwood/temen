//! #1830 — **a linked module keeps the funcref slots of its data image.** `link` bakes each unit's
//! `data.funcref` (a static initializer's function pointer: `var oomHandler = continueAfterOutOfMem`,
//! a vtable) into the data image as a function index, and records where it wrote one
//! (`Module::data_funcref_slots`). Without the record a linked module's bytes cannot say which of
//! them are function indices, and every analysis of the functions a `call.dyn` can reach read
//! `ref.func` alone: the JIT refused a fork beneath a call through such a pointer, and link-time DCE
//! had to skip any program whose units baked one.

use temen_interp::{run, Value};
use temen_ir::{
    data_funcref_targets, link, taken_funcs, Data, DataFuncref, LinkUnit, Module,
    POWERBOX_STACK_ALIGN,
};
use temen_verify::{verify_module, VerifyError};

fn unit(src: &str) -> LinkUnit {
    let module = temen_text::parse_module(src).expect("parse unit");
    LinkUnit {
        exports: module
            .exports
            .iter()
            .map(|e| (e.name.clone(), e.func))
            .collect(),
        data_exports: module.data_exports.clone(),
        module,
    }
}

/// Unit 0, a library: `twice(x) = 2x`, reached only through the program's function pointer.
const LIB: &str = r#"
memory 16
func (i64) -> (i64) {
block 0 (x: i64) {
  y = i64.add x x
  return y
  }
}
export 0 func "twice" 0
"#;

/// Unit 1, the program: a global function pointer at `16384` whose initializer is `twice`
/// (attached as a `data.funcref`, which has no text form), and `main`, which calls through it.
const PROG: &str = r#"
memory 16
data 16384 "\x00\x00\x00\x00"
func () -> (i64) {
block 0 () {
  p = data.self 16384
  f = i32.load p
  x = i64.const 21
  r = call.dyn (i64) -> (i64) f (x)
  return r
  }
}
export 0 func "main" 0
"#;

/// The two units, the program's pointer initialized to `twice`.
fn units() -> Vec<LinkUnit> {
    let mut prog = unit(PROG);
    prog.module.data_funcrefs = vec![DataFuncref {
        at: 16384,
        name: "twice".to_string(),
    }];
    vec![unit(LIB), prog]
}

/// The window offset the program unit's data starts at: unit 0 has no data, so the program's is
/// the first page-aligned region.
fn prog_dbase(linked: &Module) -> u64 {
    let slot = linked.data_funcref_slots[0];
    slot - 16384 - (slot - 16384) % POWERBOX_STACK_ALIGN
}

#[test]
fn link_records_every_funcref_it_bakes() {
    let linked = link(&units()).expect("link");
    verify_module(&linked).expect("the linked module verifies, its record included");
    assert_eq!(
        linked.data_funcref_slots.len(),
        1,
        "one baked funcref, one slot"
    );
    let at = linked.data_funcref_slots[0];
    assert_eq!(
        at,
        prog_dbase(&linked) + 16384,
        "where the program's data landed"
    );
    assert_eq!(
        data_funcref_targets(&linked),
        vec![Some(0)],
        "the slot holds `twice`, the library's function 0"
    );
    let mut fuel = 1_000_000;
    let r = run(&linked, 1, &[], &mut fuel).expect("run main");
    assert_eq!(
        r,
        vec![Value::I64(42)],
        "main calls `twice` through the pointer"
    );
}

/// A module linked before is a unit again: its slots name its own functions, so they shift with
/// them. The relinked program calls through the pointer as it did.
#[test]
fn a_relinked_module_shifts_the_slots_it_holds() {
    let first = link(&units()).expect("link");
    let pad = unit(
        "memory 16\nfunc () -> () {\nblock 0 () {\n  return\n  }\n}\nexport 0 func \"pad\" 0\n",
    );
    let again = unit_of(first);
    let relinked = link(&[pad, again]).expect("relink");
    verify_module(&relinked).expect("verifies");
    assert_eq!(
        data_funcref_targets(&relinked),
        vec![Some(1)],
        "`twice` moved up past `pad`, and the image's index moved with it"
    );
    let main = relinked.resolve_export("main").expect("main");
    let mut fuel = 1_000_000;
    let r = run(&relinked, main, &[], &mut fuel).expect("run main");
    assert_eq!(r, vec![Value::I64(42)]);
}

fn unit_of(module: Module) -> LinkUnit {
    LinkUnit {
        exports: module
            .exports
            .iter()
            .map(|e| (e.name.clone(), e.func))
            .collect(),
        data_exports: module.data_exports.clone(),
        module,
    }
}

/// The verifier holds the record to the image: a slot out of order or overlapping the one before,
/// one the data segments do not lay down, or one holding no function of the module, is refused —
/// the record is what an indirect-call analysis trusts to name every function the image hands out.
#[test]
fn the_verifier_holds_the_record_to_the_image() {
    let linked = link(&units()).expect("link");
    let at = linked.data_funcref_slots[0];
    let with = |slots: Vec<u64>, edit: &dyn Fn(&mut Module)| {
        let mut m = linked.clone();
        m.data_funcref_slots = slots;
        edit(&mut m);
        verify_module(&m)
    };
    let slot = |slot| Err(VerifyError::DataFuncrefSlot { slot });
    assert_eq!(with(vec![at], &|_| {}), Ok(()));
    // Not laid down: past the data, or straddling the end of its segment.
    assert_eq!(with(vec![at + 4096], &|_| {}), slot(0));
    assert_eq!(with(vec![at + 2], &|_| {}), slot(0));
    // No function of the module: the image's bytes say 9.
    assert_eq!(
        with(vec![at], &|m| {
            let d = m
                .data
                .iter_mut()
                .find(|d| d.offset == at)
                .expect("the segment");
            d.bytes[0] = 9;
        }),
        slot(0)
    );
    // Out of order, overlapping, repeated: a second slot in the same segment makes each shape.
    let two = |m: &mut Module| {
        m.data.push(Data {
            offset: at + 64,
            readonly: false,
            bytes: vec![0; 8],
        })
    };
    assert_eq!(with(vec![at, at + 64], &two), Ok(()));
    assert_eq!(with(vec![at + 64, at], &two), slot(1));
    assert_eq!(with(vec![at + 64, at + 66], &two), slot(1));
    assert_eq!(with(vec![at, at], &two), slot(1));
}

/// The record rides the wire and the text form: a linked module's encoding and its printed text both
/// come back with the same slots.
#[test]
fn the_record_rides_the_wire_and_the_text() {
    let linked = link(&units()).expect("link");
    let decoded =
        temen_encode::decode_module(&temen_encode::encode_module(&linked)).expect("decode");
    assert_eq!(decoded.data_funcref_slots, linked.data_funcref_slots);
    assert_eq!(decoded, linked, "the whole module round-trips");
    let text = temen_text::print_module(&linked);
    assert!(text.contains(&format!("data.funcref {}\n", linked.data_funcref_slots[0])));
    let parsed = temen_text::parse_module(&text).expect("reparse");
    assert_eq!(parsed.data_funcref_slots, linked.data_funcref_slots);
}

/// A function the image points at is *taken*, as a `ref.func` makes one — what the JIT's fork
/// instrumentation and link-time DCE read for the functions a `call.dyn` can select.
#[test]
fn a_function_the_image_points_at_is_taken() {
    let linked = link(&units()).expect("link");
    assert_eq!(
        taken_funcs(&linked),
        vec![true, false],
        "`twice` is taken (by the image); `main` is not"
    );
}
