//! **A linked program's manifest describes the program, not what it linked against** (#1629).
//!
//! `link` publishes every unit's imports in the merged table. Adding a library to the handle list
//! therefore used to add *its* capabilities to the program's manifest whether or not the program
//! could reach them — the prebuilt graphics unit going resident made every c_interpret lesson declare
//! `fb_present`/`fb_poll`, and since that host routes a program to its fast release runner only when
//! every declared cap is served, one unreachable row silently took the fast path away from
//! everything (theSherwood/c_interpret#35).
//!
//! The property this file pins is the one an embedder reasons about: **linking a library a program
//! never calls must not change its manifest at all.**

use temen_ir::{
    Block, Export, Func, FuncType, Import, ImportMode, ImportShape, Inst, LinkUnitRef, Memory,
    Module, Terminator, TypeEntry, ValType,
};

/// `(i64) -> (i32)` at type 0 — the powerbox entry shape, shared by every function here so one
/// type-section entry serves as both the import shape and each `call.import`'s self-describing sig.
fn types() -> Vec<TypeEntry> {
    vec![TypeEntry::Func(FuncType {
        params: vec![ValType::I64],
        results: vec![ValType::I32],
    })]
}

fn imports(names: &[&str]) -> Vec<Import> {
    names
        .iter()
        .map(|n| Import {
            name: (*n).to_string(),
            shape: ImportShape::Func(0),
            mode: ImportMode::Required,
        })
        .collect()
}

/// A `(i64) -> (i32)` function that calls import slot `slot` with its own argument and returns the
/// result — the smallest thing that makes a capability *reachable*.
fn calls_import(slot: u32) -> Func {
    Func {
        params: vec![ValType::I64],
        results: vec![ValType::I32],
        blocks: vec![Block {
            params: vec![ValType::I64],
            insts: vec![Inst::CallImport {
                import: slot,
                op: 0,
                sig: 0,
                args: vec![0],
            }],
            term: Terminator::Return(vec![1]),
        }],
    }
}

fn unit(funcs: Vec<Func>, import_names: &[&str], exports: &[(&str, u32)]) -> Module {
    Module {
        memory: Some(Memory {
            size_log2: 16,
            shadow: None,
        }),
        funcs,
        types: types(),
        imports: imports(import_names),
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

fn exports_of(m: &Module) -> Vec<(String, temen_ir::FuncIdx)> {
    m.exports.iter().map(|e| (e.name.clone(), e.func)).collect()
}

fn manifest(m: &Module) -> Vec<String> {
    m.imports.iter().map(|i| i.name.clone()).collect()
}

/// The issue's own repro, in miniature: a program that calls `stream_write` links against a graphics
/// library it never calls, and its manifest comes back **identical** to linking without it.
#[test]
fn linking_an_uncalled_library_leaves_the_manifest_unchanged() {
    // The graphics unit: one exported entry point, reaching two capabilities the program won't.
    let gfx = unit(
        vec![calls_import(0), calls_import(1)],
        &["fb_present", "fb_poll"],
        &[("fb_flush", 0), ("fb_key", 1)],
    );
    // The program: `main` calls `stream_write` and nothing else.
    let prog = unit(vec![calls_import(0)], &["stream_write"], &[("main", 0)]);

    let gfx_exports = exports_of(&gfx);
    let gfx_unit = LinkUnitRef {
        module: &gfx,
        exports: &gfx_exports,
        data_exports: &[],
    };

    let without = temen_browser::link_program_multi(&[], &prog, "main").expect("links without gfx");
    let with =
        temen_browser::link_program_multi(&[gfx_unit], &prog, "main").expect("links with gfx");

    assert_eq!(
        manifest(&without),
        vec!["stream_write".to_string()],
        "the program declares exactly what it calls"
    );
    assert_eq!(
        manifest(&with),
        manifest(&without),
        "adding a library the program never calls must not change its manifest"
    );
    assert!(
        !manifest(&with).iter().any(|n| n.starts_with("fb_")),
        "no framebuffer capability is requested by a program that cannot reach one"
    );
}

/// The other half: a capability the program *does* reach through the library survives, and the call
/// still dispatches to it. Pruning that got this wrong would be a silently mis-dispatched capability
/// call rather than a missing one — the failure worth testing for.
#[test]
fn a_capability_reached_through_a_library_survives_and_still_dispatches() {
    // `gfx::fb_flush` calls `fb_present`; `fb_key` (calling `fb_poll`) is never reached.
    let gfx = unit(
        vec![calls_import(0), calls_import(1)],
        &["fb_present", "fb_poll"],
        &[("fb_flush", 0), ("fb_key", 1)],
    );
    // `main` calls `gfx::fb_flush` — a cross-unit call the linker resolves to a direct `call`.
    let prog = Module {
        funcs: vec![Func {
            params: vec![ValType::I64],
            results: vec![ValType::I32],
            blocks: vec![Block {
                params: vec![ValType::I64],
                insts: vec![Inst::CallSym {
                    import: 0,
                    sig: 0,
                    handle: 0,
                    args: vec![0],
                }],
                term: Terminator::Return(vec![1]),
            }],
        }],
        ..unit(vec![], &["fb_flush"], &[("main", 0)])
    };

    let gfx_exports = exports_of(&gfx);
    let gfx_unit = LinkUnitRef {
        module: &gfx,
        exports: &gfx_exports,
        data_exports: &[],
    };
    let linked =
        temen_browser::link_program_multi(&[gfx_unit], &prog, "main").expect("links and verifies");

    assert_eq!(
        manifest(&linked),
        vec!["fb_present".to_string()],
        "the reached capability stays; the unreached sibling goes"
    );
    // And the surviving `call.import` names it through the *renumbered* table.
    let mut reached = Vec::new();
    for f in &linked.funcs {
        for b in &f.blocks {
            for inst in &b.insts {
                if let Inst::CallImport { import, .. } = inst {
                    reached.push(linked.imports[*import as usize].name.clone());
                }
            }
        }
    }
    assert_eq!(
        reached,
        vec!["fb_present".to_string()],
        "every surviving call.import resolves to the capability it always named"
    );
}
