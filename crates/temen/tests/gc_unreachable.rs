//! **Link-time dead-code elimination** (`temen_ir::gc_unreachable_funcs`, #1407).
//!
//! `link` merges whole modules, so a program that calls one library function carries the whole
//! library. This pass walks reachability from the module's addressable surface and drops the rest,
//! renumbering every funcidx that named a function — the cross-unit twin of chibicc's own `mark_live`.
//!
//! The risk it carries is that a *wrong* renumber is silent: the module still verifies, it just calls
//! the wrong function. So these tests pin the renumber by **behaviour** (the survivors still compute
//! what they computed) as well as by shape, and pin the two ways the pass is meant to decline rather
//! than guess.

use temen_ir::{Block, Export, Func, GcError, Inst, Memory, Module, Terminator, ValType};

/// A function returning `val`, with an optional tail of calls it makes (reachability edges).
fn leaf(val: i64) -> Func {
    Func {
        params: vec![ValType::I64],
        results: vec![ValType::I64],
        blocks: vec![Block {
            params: vec![ValType::I64],
            insts: vec![Inst::ConstI64(val)],
            term: Terminator::Return(vec![1]),
        }],
    }
}

/// A function that calls `callee` and returns its result.
fn caller(callee: u32) -> Func {
    Func {
        params: vec![ValType::I64],
        results: vec![ValType::I64],
        blocks: vec![Block {
            params: vec![ValType::I64],
            insts: vec![Inst::Call {
                func: callee,
                args: vec![0],
            }],
            term: Terminator::Return(vec![1]),
        }],
    }
}

fn module(funcs: Vec<Func>, exports: &[(&str, u32)]) -> Module {
    Module {
        memory: Some(Memory { size_log2: 16 }),
        funcs,
        exports: exports
            .iter()
            .map(|(n, f)| Export {
                name: (*n).to_string(),
                func: *f,
            })
            .collect(),
        ..Default::default()
    }
}

/// The shape the pass exists for: an exported entry reaches one helper; three library functions
/// nothing reaches are dropped, and the survivors' indices are rewritten to match.
#[test]
fn unreachable_functions_are_dropped_and_the_survivors_renumbered() {
    // 0: dead, 1: dead, 2: entry → 4, 3: dead, 4: helper
    let mut m = module(
        vec![leaf(10), leaf(11), caller(4), leaf(13), leaf(14)],
        &[("main", 2)],
    );
    let gc = temen_ir::gc_unreachable_funcs(&mut m, &[]).expect("gc");
    assert_eq!(gc.dropped, 3, "0, 1 and 3 are unreachable");
    assert_eq!(m.funcs.len(), 2);
    // Declaration order is preserved among survivors: the entry (old 2) is now 0, the helper 1.
    assert_eq!(gc.map[2], Some(0));
    assert_eq!(gc.map[4], Some(1));
    assert_eq!(gc.map[0], None);
    assert_eq!(gc.map[1], None);
    assert_eq!(gc.map[3], None);
    assert_eq!(m.exports[0].func, 0, "the export follows its function");
    // The entry's call was rewritten to the helper's new index — the renumber a wrong pass gets
    // silently wrong.
    let Inst::Call { func, .. } = m.funcs[0].blocks[0].insts[0] else {
        panic!("expected the entry's call")
    };
    assert_eq!(
        func, 1,
        "the call points at the helper, not at whatever is now index 4"
    );
    assert!(
        temen_verify::verify_module(&m).is_ok(),
        "and it still verifies"
    );
}

/// The renumber is right *behaviourally*, not just structurally: run the entry before and after and
/// get the same value. This is what catches a map that is self-consistent but off by one.
#[test]
fn the_survivors_still_compute_what_they_computed() {
    let mut m = module(
        vec![leaf(10), leaf(11), caller(4), leaf(13), leaf(99)],
        &[("main", 2)],
    );
    let run = |m: &Module, f: u32| {
        let mut fuel = 100_000u64;
        let out =
            temen_interp::run_fast(m, f, &[temen_interp::Value::I64(0)], &mut fuel).expect("runs");
        match out[0] {
            temen_interp::Value::I64(v) => v,
            other => panic!("expected i64, got {other:?}"),
        }
    };
    let before = run(&m, 2);
    let gc = temen_ir::gc_unreachable_funcs(&mut m, &[]).expect("gc");
    let after = run(&m, gc.map[2].unwrap());
    assert_eq!(before, 99, "the entry returns its helper's value");
    assert_eq!(after, before, "…and the same one after the collection");
}

/// `extra_roots` is how a caller keeps a funcidx it means to invoke directly but has not exported —
/// the entry it is about to hand `synth_manifest_start`, in the linker's case.
#[test]
fn an_extra_root_keeps_an_unexported_entry_alive() {
    let mut m = module(vec![leaf(10), caller(2), leaf(42)], &[]);
    // No exports at all: without a root, everything is garbage.
    let mut empty = m.clone();
    let gc = temen_ir::gc_unreachable_funcs(&mut empty, &[]).expect("gc");
    assert_eq!(gc.dropped, 3, "no roots ⇒ nothing survives");

    let gc = temen_ir::gc_unreachable_funcs(&mut m, &[1]).expect("gc");
    assert_eq!(gc.dropped, 1, "only func 0 is unreachable from the root");
    assert_eq!(gc.map[1], Some(0));
    assert_eq!(gc.map[2], Some(1));
    assert!(temen_verify::verify_module(&m).is_ok());
}

/// An **address-taken** function is reachable through `ref.func`, not a call — the case a
/// call-graph-only walk drops and a program then indirect-calls into nothing.
#[test]
fn a_function_whose_address_is_taken_survives() {
    let taker = Func {
        params: vec![ValType::I64],
        results: vec![ValType::Ref],
        blocks: vec![Block {
            params: vec![ValType::I64],
            insts: vec![Inst::RefFunc { func: 2 }],
            term: Terminator::Return(vec![1]),
        }],
    };
    let mut m = module(vec![leaf(10), taker, leaf(42)], &[("main", 1)]);
    let gc = temen_ir::gc_unreachable_funcs(&mut m, &[]).expect("gc");
    assert_eq!(gc.dropped, 1, "func 0 goes; the ref.func target stays");
    assert_eq!(gc.map[2], Some(1), "the address-taken function survived");
    let Inst::RefFunc { func } = m.funcs[0].blocks[0].insts[0] else {
        panic!("expected ref.func")
    };
    assert_eq!(func, 1, "and the reference was renumbered with it");
}

/// Debug info for a dropped function goes with it. A dangling `func` would make a stepper resolve a
/// stop to the wrong source line, and the DAP's own range checks reject it outright.
#[test]
fn debug_info_for_dropped_functions_is_dropped_with_them() {
    let text = r#"memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 10
  return v1
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 20
  return v1
  }
}
export 0 func "main" 1
debug.file 0 "dead.c"
debug.file 1 "live.c"
debug.fname 0 "dead"
debug.fname 1 "live"
debug.loc 0 0 0 0 5 1
debug.loc 1 1 0 0 9 1
debug.var 0 "gone" ssa 1 "int"
debug.var 1 "kept" ssa 1 "int"
debug.var global "g" fixed 0 "int"
"#;
    let mut m = temen_text::parse_module(text).expect("parses");
    let gc = temen_ir::gc_unreachable_funcs(&mut m, &[]).expect("gc");
    assert_eq!(gc.dropped, 1);
    let di = m.debug_info.as_ref().expect("debug info survives");
    assert_eq!(
        di.locs.len(),
        1,
        "the dead function's loc went: {:?}",
        di.locs
    );
    assert_eq!(di.locs[0].func, 0, "and the survivor's was renumbered");
    let names: Vec<&str> = di.vars.iter().map(|v| v.name.as_str()).collect();
    assert!(names.contains(&"kept"), "{names:?}");
    assert!(
        names.contains(&"g"),
        "a global's scope is not a funcidx: {names:?}"
    );
    assert!(!names.contains(&"gone"), "{names:?}");
    let fnames: Vec<&str> = di.func_names.iter().map(|n| n.name.as_str()).collect();
    assert_eq!(fnames, vec!["live"], "the dead function's name went too");
}

/// Nothing to drop ⇒ the module is untouched and the map is the identity, so a caller can remap
/// unconditionally.
#[test]
fn a_fully_reachable_module_is_left_alone() {
    let mut m = module(vec![caller(1), leaf(7)], &[("main", 0)]);
    let before = m.clone();
    let gc = temen_ir::gc_unreachable_funcs(&mut m, &[]).expect("gc");
    assert_eq!(gc.dropped, 0);
    assert_eq!(m, before, "byte-for-byte the same module");
    assert_eq!(gc.map, vec![Some(0), Some(1)], "the identity map");
}

/// The pass **declines** on a module whose data image has funcidxs baked into bytes: renumbering
/// would silently retarget them, and it cannot rewrite bytes it cannot locate. Changes nothing.
#[test]
fn a_module_with_baked_data_funcrefs_is_declined() {
    let mut m = module(vec![leaf(1), leaf(2)], &[("main", 0)]);
    m.data = vec![temen_ir::Data {
        offset: 0,
        readonly: false,
        bytes: vec![0; 8],
    }];
    m.data_funcrefs = vec![temen_ir::DataFuncref {
        at: 0,
        name: "main".to_string(),
    }];
    let before = m.clone();
    assert_eq!(
        temen_ir::gc_unreachable_funcs(&mut m, &[]),
        Err(GcError::DataFuncrefs)
    );
    assert_eq!(m, before, "a declined pass changes nothing");
}
