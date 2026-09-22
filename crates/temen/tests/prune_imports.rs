//! **Link-time manifest pruning** (`temen_ir::prune_unused_imports`, #1629).
//!
//! `link` publishes every unit's imports in the merged table whether or not the linked program can
//! reach them, and `stub_unreachable_funcs` (#1407) empties dead *bodies* without touching the rows
//! those bodies were the only users of. So a program that links a graphics library it never calls
//! still declares `fb_present`/`fb_poll` — asking the host for authority it has no path to, which is
//! the opposite of what the powerbox is for.
//!
//! Where the function DCE deliberately **keeps** dead indices (a funcidx is forgeable: it can reach
//! `call.indirect` from arithmetic, so renumbering would silently retarget a call), an import index
//! is an immediate — no instruction computes one, nothing dispatches indirectly through the manifest,
//! and nothing bakes one into the data image. The reference set is closed, so the table can actually
//! shrink. These tests pin both halves: the rows that go, and the renumbering of the rows that stay.

use temen_ir::{
    Block, Export, Func, FuncType, Import, ImportMode, ImportShape, Inst, Memory, Module,
    Terminator, TypeEntry,
};

/// One `() -> ()` signature at type index 0, which every import in these tests shares.
fn types() -> Vec<TypeEntry> {
    vec![TypeEntry::Func(FuncType {
        params: vec![],
        results: vec![],
    })]
}

fn import(name: &str) -> Import {
    Import {
        name: name.to_string(),
        shape: ImportShape::Func(0),
        mode: ImportMode::Required,
    }
}

/// A function whose body calls import slots `slots`, in order.
fn calls_imports(slots: &[u32]) -> Func {
    Func {
        params: vec![],
        results: vec![],
        blocks: vec![Block {
            params: vec![],
            insts: slots
                .iter()
                .map(|&import| Inst::CallImport {
                    import,
                    op: 0,
                    sig: 0,
                    args: vec![],
                })
                .collect(),
            term: Terminator::Return(vec![]),
        }],
    }
}

/// A one-block trap — what `stub_unreachable_funcs` leaves behind.
fn stub() -> Func {
    Func {
        params: vec![],
        results: vec![],
        blocks: vec![Block {
            params: vec![],
            insts: vec![],
            term: Terminator::Unreachable,
        }],
    }
}

fn module(funcs: Vec<Func>, imports: &[&str]) -> Module {
    Module {
        memory: Some(Memory {
            size_log2: 16,
            shadow: None,
        }),
        funcs,
        types: types(),
        imports: imports.iter().map(|n| import(n)).collect(),
        exports: vec![Export {
            name: "main".to_string(),
            func: 0,
        }],
        ..Default::default()
    }
}

fn names(m: &Module) -> Vec<&str> {
    m.imports.iter().map(|i| i.name.as_str()).collect()
}

/// Every `call.import` in the module, as `(func, slot)` — what the renumbering has to keep pointing
/// at the same *name* it pointed at before.
fn slots(m: &Module) -> Vec<(usize, u32)> {
    let mut out = Vec::new();
    for (f, func) in m.funcs.iter().enumerate() {
        for b in &func.blocks {
            for inst in &b.insts {
                if let Inst::CallImport { import, .. } = inst {
                    out.push((f, *import));
                }
            }
        }
    }
    out
}

/// The shape the pass exists for, in miniature: a live entry reaches two of five capabilities, the
/// library that declared the other three has been stubbed out, and the manifest is left describing
/// **the program**.
#[test]
fn rows_nothing_reaches_are_dropped_and_survivors_renumbered() {
    // Slots 0 `vm_fs`, 2 `exit` are called; 1 `fb_present`, 3 `fb_poll`, 4 `vm_map` are not.
    let mut m = module(
        vec![calls_imports(&[2, 0]), stub()],
        &["vm_fs", "fb_present", "exit", "fb_poll", "vm_map"],
    );
    let pruned = temen_ir::prune_unused_imports(&mut m);
    assert_eq!(pruned.pruned, 3, "fb_present, fb_poll and vm_map are dead");
    assert_eq!(
        names(&m),
        vec!["vm_fs", "exit"],
        "survivors keep their relative order"
    );
    assert_eq!(
        slots(&m),
        vec![(0, 1), (0, 0)],
        "`exit` moved 2 → 1 and `vm_fs` 0 → 0; both references followed their row"
    );
}

/// The renumbering, stated the way a mis-dispatch would break it: resolve every call site back through
/// the *new* table and it must name the same capability it named before the pass.
#[test]
fn every_reference_still_names_the_capability_it_named_before() {
    let before = ["a", "b", "c", "d", "e", "f"];
    // Call sites, in body order, naming slots out of order and with a repeat.
    let sites = [5u32, 0, 3, 0];
    let mut m = module(vec![calls_imports(&sites)], &before);
    temen_ir::prune_unused_imports(&mut m);
    let after: Vec<(usize, u32)> = slots(&m);
    assert_eq!(after.len(), sites.len(), "no call site was lost");
    for (site, (_, new)) in sites.iter().zip(after) {
        assert_eq!(
            m.imports[new as usize].name, before[*site as usize],
            "slot {site} must still resolve to `{}`",
            before[*site as usize]
        );
    }
    assert_eq!(names(&m), vec!["a", "d", "f"], "and only those three stay");
}

/// `import.attach` is a reference too — a rebindable slot the program attaches to but never calls
/// through must survive, or the attach would rebind some other capability.
#[test]
fn import_attach_keeps_a_slot_alive() {
    let mut m = module(vec![calls_imports(&[2])], &["gone", "attached", "called"]);
    m.imports[1].mode = ImportMode::Rebindable;
    // Prepend the attach to the body: a handle, then `import.attach 1, v0`.
    let insts = &mut m.funcs[0].blocks[0].insts;
    insts.insert(0, Inst::ConstI32(7));
    insts.insert(
        1,
        Inst::ImportAttach {
            import: 1,
            handle: 0,
        },
    );
    let pruned = temen_ir::prune_unused_imports(&mut m);
    assert_eq!(pruned.pruned, 1, "only `gone` is unreferenced");
    assert_eq!(names(&m), vec!["attached", "called"]);
    let Inst::ImportAttach { import, .. } = m.funcs[0].blocks[0].insts[1] else {
        panic!("expected the attach")
    };
    assert_eq!(import, 0, "the attach followed `attached` 1 → 0");
    assert_eq!(
        slots(&m),
        vec![(0, 1)],
        "and the call followed `called` 2 → 1"
    );
}

/// Nothing to do is not a rewrite: a module whose every row is reached comes back byte-identical, so
/// running the pass on an already-tight manifest cannot perturb indices.
#[test]
fn a_fully_reached_manifest_is_untouched() {
    let m = module(vec![calls_imports(&[0, 1])], &["a", "b"]);
    let mut pruned_m = m.clone();
    let pruned = temen_ir::prune_unused_imports(&mut pruned_m);
    assert_eq!(pruned.pruned, 0);
    assert_eq!(pruned_m, m, "unchanged");
}

/// An empty manifest is the common case (a module that inlines its capability calls) and must not be
/// a special case in the caller.
#[test]
fn an_empty_manifest_is_a_no_op() {
    let mut m = module(vec![calls_imports(&[])], &[]);
    assert_eq!(temen_ir::prune_unused_imports(&mut m).pruned, 0);
    assert!(m.imports.is_empty());
}

/// Fail closed on a broken reference: renumbering around an out-of-range index is the verifier's
/// problem to report, not this pass's to paper over — so it declines and changes **nothing**, leaving
/// the bad index intact for `verify_module` to reject.
#[test]
fn an_out_of_range_reference_declines_the_whole_pass() {
    let m = module(vec![calls_imports(&[0, 9])], &["a", "dead"]);
    let mut attempted = m.clone();
    let pruned = temen_ir::prune_unused_imports(&mut attempted);
    assert_eq!(pruned.pruned, 0, "declined");
    assert_eq!(
        attempted, m,
        "including `dead`, which it would otherwise have pruned"
    );
}
