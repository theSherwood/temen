//! **Link-time dead-code elimination** (`temen_ir::stub_unreachable_funcs`, #1407).
//!
//! `link` merges whole modules, so a program that calls one library function carries the whole
//! library. This pass walks reachability from the module's addressable surface and empties the bodies
//! nothing reaches — the cross-unit twin of chibicc's own `mark_live`.
//!
//! It **stubs rather than removes**, and that is the load-bearing decision. A funcidx is observable:
//! `call.indirect` masks an `i32` into the domain dispatch table, whose slot `i` *is* funcidx `i`, and
//! a `funcref` is a deliberately forgeable integer (§3c). Removing functions would renumber that table
//! and shrink its mask, so an index that selected one function would select another — safe (the slot
//! signature check and trapping padding keep a forged index inert) but not *behaviour-preserving*.
//! Keeping the indices means a `ref.func`-derived call lands where it always did, a forged index that
//! selects an unreachable function traps instead of running dead code, and nothing is renumbered at
//! all — so there is no silent-retarget failure mode to test for in the first place.

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

/// The shape the pass exists for: an exported entry reaches one helper; the three library functions
/// nothing reaches become one-block traps, and **every index stays put**.
#[test]
fn unreachable_bodies_become_traps_and_every_index_stays_put() {
    // 0: dead, 1: dead, 2: entry → 4, 3: dead, 4: helper
    let mut m = module(
        vec![leaf(10), leaf(11), caller(4), leaf(13), leaf(14)],
        &[("main", 2)],
    );
    let gc = temen_ir::stub_unreachable_funcs(&mut m, &[]).expect("gc");
    assert_eq!(gc.stubbed, 3, "0, 1 and 3 are unreachable");
    assert_eq!(m.funcs.len(), 5, "the index space is unchanged");
    for dead in [0usize, 1, 3] {
        assert_eq!(m.funcs[dead].blocks.len(), 1, "one block");
        assert!(m.funcs[dead].blocks[0].insts.is_empty(), "no instructions");
        assert_eq!(
            m.funcs[dead].blocks[0].term,
            Terminator::Unreachable,
            "arriving by a forged index traps"
        );
        assert_eq!(
            m.funcs[dead].params,
            vec![ValType::I64],
            "the signature stays — the dispatch slot's check reads it"
        );
    }
    // Live bodies, and everything naming an index, are untouched.
    assert_eq!(m.exports[0].func, 2);
    let Inst::Call { func, .. } = m.funcs[2].blocks[0].insts[0] else {
        panic!("expected the entry's call")
    };
    assert_eq!(func, 4, "the call still names the helper's original index");
    assert_eq!(
        m.funcs[4].blocks[0].insts.len(),
        1,
        "the helper kept its body"
    );
    assert!(
        temen_verify::verify_module(&m).is_ok(),
        "and it still verifies"
    );
}

/// **The reason it stubs.** A `call.indirect` index is masked into the dispatch table, whose slot `i`
/// is funcidx `i`, and a funcref is a forgeable integer — so an index need not come from `ref.func`.
/// Here a live entry calls indirectly through an index it *computed*, selecting a function nothing
/// references. Before the pass it runs; after it traps. What it must never do is silently run a
/// **different** function, which is what removing-and-renumbering would have produced.
#[test]
fn a_computed_indirect_index_still_selects_the_same_slot() {
    let dialer = Func {
        params: vec![ValType::I64],
        results: vec![ValType::I64],
        blocks: vec![Block {
            params: vec![ValType::I64],
            insts: vec![
                Inst::ConstI32(3), // a *computed* index: no ref.func anywhere in the module
                Inst::CallIndirect {
                    ty: 0,
                    idx: 1,
                    args: vec![0],
                },
            ],
            term: Terminator::Return(vec![2]),
        }],
    };
    let mut m = module(vec![dialer, leaf(11), leaf(12), leaf(77)], &[("main", 0)]);
    m.types = vec![temen_ir::TypeEntry::Func(temen_ir::FuncType {
        params: vec![ValType::I64],
        results: vec![ValType::I64],
    })];
    let run = |m: &Module| {
        let mut fuel = 100_000u64;
        temen_interp::run_fast(m, 0, &[temen_interp::Value::I64(0)], &mut fuel)
    };
    // Slot 3 holds `leaf(77)`, and nothing in the module references it.
    assert_eq!(
        run(&m).expect("runs before")[0],
        temen_interp::Value::I64(77),
        "the computed index selects funcidx 3"
    );
    let gc = temen_ir::stub_unreachable_funcs(&mut m, &[]).expect("gc");
    assert_eq!(gc.stubbed, 3, "1, 2 and 3 are unreachable by the walk");
    assert!(
        run(&m).is_err(),
        "the emptied slot traps; it must not run some other function"
    );
    assert!(temen_verify::verify_module(&m).is_ok());
}

/// The reachable program is *behaviourally* untouched: run the entry before and after and get the
/// same value.
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
    temen_ir::stub_unreachable_funcs(&mut m, &[]).expect("gc");
    let after = run(&m, 2);
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
    let gc = temen_ir::stub_unreachable_funcs(&mut empty, &[]).expect("gc");
    assert_eq!(gc.stubbed, 3, "no roots ⇒ every body is emptied");

    let gc = temen_ir::stub_unreachable_funcs(&mut m, &[1]).expect("gc");
    assert_eq!(gc.stubbed, 1, "only func 0 is unreachable from the root");
    assert_eq!(
        m.funcs[1].blocks[0].insts.len(),
        1,
        "the root kept its body"
    );
    assert_eq!(
        m.funcs[2].blocks[0].insts.len(),
        1,
        "and so did what it calls"
    );
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
    let gc = temen_ir::stub_unreachable_funcs(&mut m, &[]).expect("gc");
    assert_eq!(
        gc.stubbed, 1,
        "func 0 is emptied; the ref.func target keeps its body"
    );
    assert_eq!(
        m.funcs[2].blocks[0].insts.len(),
        1,
        "the address-taken function survived"
    );
    let Inst::RefFunc { func } = m.funcs[1].blocks[0].insts[0] else {
        panic!("expected ref.func")
    };
    assert_eq!(func, 2, "and the reference still names it");
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
    let gc = temen_ir::stub_unreachable_funcs(&mut m, &[]).expect("gc");
    assert_eq!(gc.stubbed, 1);
    let di = m.debug_info.as_ref().expect("debug info survives");
    assert_eq!(
        di.locs.len(),
        1,
        "the dead function's loc went: {:?}",
        di.locs
    );
    assert_eq!(
        di.locs[0].func, 1,
        "the survivor's is unchanged (nothing is renumbered)"
    );
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

/// Nothing to stub ⇒ the module is untouched and the map is the identity, so a caller can remap
/// unconditionally.
#[test]
fn a_fully_reachable_module_is_left_alone() {
    let mut m = module(vec![caller(1), leaf(7)], &[("main", 0)]);
    let before = m.clone();
    let gc = temen_ir::stub_unreachable_funcs(&mut m, &[]).expect("gc");
    assert_eq!(gc.stubbed, 0);
    assert_eq!(m, before, "byte-for-byte the same module");
}

/// The pass **declines** on a module whose data image has funcidxs baked into bytes: the linker has
/// already resolved and cleared those, so a function reachable only from a static initializer would
/// look unreachable and be emptied out from under its caller. Changes nothing.
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
        temen_ir::stub_unreachable_funcs(&mut m, &[]),
        Err(GcError::DataFuncrefs)
    );
    assert_eq!(m, before, "a declined pass changes nothing");
}
