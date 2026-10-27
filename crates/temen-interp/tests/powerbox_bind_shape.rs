//! #1524 — **the powerbox binder binds by name *and* shape.**
//!
//! `Host::bind_powerbox_manifest` resolves a manifest import's name through the one shared
//! name→capability table and binds the handle this host granted for it. Resolving the *name* is
//! not enough: a frontend emits two different things through the same `call.sym` syntax, and they
//! intern as the same [`ImportShape::Func`] —
//!
//! * a **capability** import — handle operand + the op's own args (chibicc `gen_builtin_import`,
//!   the `<temen.h>` §7 late-binding pattern, now spelled `TEMEN_CAP` in object mode);
//! * a **function-symbol** import — a cross-TU call to a declared-but-undefined function, whose
//!   args are the C list *led by the data stack pointer* (chibicc `--emit-object`).
//!
//! When no linked unit defines the latter, `link_with_manifest` retains it as a manifest import.
//! If its name then happens to be a powerbox row, binding by name alone dispatches the capability
//! with every argument shifted by one — the data-SP arrives where the first op argument belongs.
//!
//! These tests pin that a signature that is not the capability op's is **refused**, and that a
//! refused `required` import refuses the whole run before it starts (IMPORTS.md §2.1, #2579), as
//! one no grant answers to does, while the capability-shaped import for the same name binds.
//! `Stream` is used because its op signatures are pinned (`preseeded_iface_shapes`, #1515 slice 1)
//! — the check is only ever as strong as the seeded shapes, which is exactly what #1515's remaining
//! decisions widen.

use temen_interp::{cap_id, powerbox_grant, Host, HostProcGrant, ImportRefusal, ProcHandler};
use temen_ir::{FuncType, Import, ImportMode, ImportShape, PowerboxHandles, TypeEntry, ValType};

fn sig(params: Vec<ValType>, results: Vec<ValType>) -> FuncType {
    FuncType { params, results }
}

/// A host holding the §3e powerbox prefix, plus the `PowerboxHandles` naming those grants.
fn powerbox() -> (Host, PowerboxHandles) {
    let mut host = Host::new();
    let granted = PowerboxHandles::prefix(host.grant_powerbox_prefix(1 << 16));
    (host, granted)
}

/// Bind `imports` on `host` against its §3e grants alone.
fn bind(
    host: &mut Host,
    granted: &PowerboxHandles,
    imports: &[Import],
    types: &[TypeEntry],
) -> Result<(), Vec<ImportRefusal>> {
    host.bind_powerbox_manifest(imports, types, &powerbox_grant(granted, &[]))
}

fn import(name: &str, ty: u32, mode: ImportMode) -> Import {
    Import {
        name: name.into(),
        shape: ImportShape::Func(ty),
        mode,
    }
}

/// The capability form of `write` — `Stream.write` is the pinned `(i64, i64) -> (i64)`
/// (buf, len), with the stream handle riding the `call.sym` handle operand rather than the
/// argument list. It binds.
#[test]
fn capability_shaped_write_binds() {
    let (mut host, granted) = powerbox();
    let types = vec![TypeEntry::Func(sig(
        vec![ValType::I64, ValType::I64],
        vec![ValType::I64],
    ))];
    let imports = vec![Import {
        name: "write".into(),
        shape: ImportShape::Func(0),
        mode: ImportMode::Required,
    }];

    let bound = bind(&mut host, &granted, &imports, &types);
    assert_eq!(bound, Ok(()), "capability-shaped write binds");

    let b = host.import_binding(0).expect("write bound");
    assert_eq!(b.type_id, cap_id::STREAM);
    assert_eq!(b.op, 1, "write is Stream op 1");
    assert_eq!(b.handle, granted.stdout, "bound to the granted stdout");
}

/// The **function-symbol** form of the same name — what chibicc `--emit-object` emits for
/// `int write(int fd, void *buf, unsigned long n)` when no linked unit defines it: the C list led
/// by the data stack pointer, `(i64 SP, i32 fd, i64 buf, i64 n) -> (i64)`.
///
/// Binding it by name would dispatch `Stream.write` with the SP where `buf` belongs and `fd`
/// where `len` belongs — a write of `fd` bytes from a pointer the guest never chose. It is
/// refused instead, the refusal names both signatures, and the run is refused with it.
#[test]
fn function_symbol_shaped_write_is_refused() {
    let (mut host, granted) = powerbox();
    let c_abi = sig(
        vec![ValType::I64, ValType::I32, ValType::I64, ValType::I64],
        vec![ValType::I64],
    );
    let types = vec![TypeEntry::Func(c_abi.clone())];
    let imports = vec![Import {
        name: "write".into(),
        shape: ImportShape::Func(0),
        mode: ImportMode::Required,
    }];

    let refusals = bind(&mut host, &granted, &imports, &types).expect_err("refused");
    assert_eq!(refusals.len(), 1, "the mis-shaped import is refused");
    assert_eq!(refusals[0].import, 0);
    assert_eq!(refusals[0].name, "write");
    assert_eq!(
        refusals[0].shape,
        Some((
            c_abi,
            sig(vec![ValType::I64, ValType::I64], vec![ValType::I64])
        )),
        "reports what the module declared, and the capability op's pinned signature"
    );

    // Refused, not merely unreported: nothing is installed, and the note says why, since the
    // guest never runs to see it.
    assert!(host.import_binding(0).is_none(), "nothing is installed");
    let notes = host.take_notes();
    assert!(
        notes.len() == 1 && notes[0].contains("its import `write` is declared"),
        "the refusal is noted: {notes:?}"
    );
}

/// A `required` import no grant answers to refuses the run, named in the refusal and the note:
/// the browser on-ramp used to leave its slot empty and fault at the call, where every other host
/// refused the start (IMPORTS.md §2.1, #2579).
#[test]
fn an_import_no_grant_answers_to_refuses_the_run() {
    let (mut host, granted) = powerbox();
    let types = vec![TypeEntry::Func(sig(vec![ValType::I64], vec![ValType::I64]))];
    let imports = vec![import("frobnicate", 0, ImportMode::Required)];
    let refusals = bind(&mut host, &granted, &imports, &types).expect_err("refused");
    assert_eq!(
        refusals,
        vec![ImportRefusal {
            import: 0,
            name: "frobnicate".into(),
            shape: None
        }]
    );
    assert_eq!(
        host.take_notes(),
        vec!["refused the run: no grant satisfies its import `frobnicate`".to_string()]
    );
}

/// A `rebindable` import may start empty (IMPORTS.md §2.1): no grant answering to it refuses
/// nothing, and the slot waits for an `import.attach`. One a grant answers to binds and stays
/// attachable, as on `temen-run`'s named path.
#[test]
fn a_rebindable_import_starts_empty_or_bound_and_stays_attachable() {
    let (mut host, granted) = powerbox();
    let types = vec![TypeEntry::Func(sig(
        vec![ValType::I64, ValType::I64],
        vec![ValType::I64],
    ))];
    let imports = vec![
        import("frobnicate", 0, ImportMode::Rebindable),
        import("write", 0, ImportMode::Rebindable),
    ];
    assert_eq!(bind(&mut host, &granted, &imports, &types), Ok(()));
    assert!(
        host.import_binding(0).is_none(),
        "the empty slot waits for an attach"
    );
    let b = host.import_binding(1).expect("write bound");
    assert!(b.rebindable, "a rebindable import stays attachable");
    assert_eq!(b.handle, granted.stdout);
}

/// One refused import refuses the **whole run** (#2579), and the refusal names that import alone:
/// a manifest carrying a capability-shaped `write` and an SP-led `read` does not start, so a
/// program never runs with a slot it declared and could not have. (Under #1524 the good import
/// bound and the bad one faulted at its call: a run the native host refused started here.)
#[test]
fn one_refused_import_refuses_the_run_and_names_only_itself() {
    let (mut host, granted) = powerbox();
    let types = vec![
        TypeEntry::Func(sig(vec![ValType::I64, ValType::I64], vec![ValType::I64])),
        TypeEntry::Func(sig(
            vec![ValType::I64, ValType::I64, ValType::I64],
            vec![ValType::I64],
        )),
    ];
    let imports = vec![
        Import {
            name: "write".into(),
            shape: ImportShape::Func(0), // capability-shaped
            mode: ImportMode::Required,
        },
        Import {
            name: "read".into(),
            shape: ImportShape::Func(1), // SP-led C list
            mode: ImportMode::Required,
        },
    ];

    let refusals = bind(&mut host, &granted, &imports, &types).expect_err("refused");
    assert_eq!(refusals.len(), 1);
    assert_eq!(refusals[0].name, "read");
    assert!(
        host.import_binding(0).is_none() && host.import_binding(1).is_none(),
        "nothing is installed for a refused run"
    );
}

/// A built-in whose signature convention is **not pinned yet** binds unchecked — the honest
/// current limit of this check, and the reason #1524 is not closed by it alone. `AddressSpace`
/// has no seeded shape (#1515 decision 1 is the owner's to make), so a `vm_map` import carrying
/// the SP-led C list still binds today. This test records that gap deliberately: when #1515 seeds
/// `AddressSpace`, it flips to a refusal and this assertion is what says so.
#[test]
fn unpinned_builtin_binds_unchecked_for_now() {
    let (mut host, granted) = powerbox();
    // `extern long vm_map(long len, long prot)` under `--emit-object`: (SP, len, prot).
    let types = vec![TypeEntry::Func(sig(
        vec![ValType::I64, ValType::I64, ValType::I64],
        vec![ValType::I64],
    ))];
    let imports = vec![Import {
        name: "vm_map".into(),
        shape: ImportShape::Func(0),
        mode: ImportMode::Required,
    }];

    let bound = bind(&mut host, &granted, &imports, &types);
    assert!(
        bound.is_ok() && host.import_binding(0).is_some(),
        "AddressSpace has no pinned op signature yet, so the shape cannot be checked (#1515)"
    );
    assert!(
        temen_interp::builtin_iface_shape(cap_id::ADDRESS_SPACE).is_none(),
        "…and that is precisely because no shape is seeded for it — if this fails, \
         AddressSpace was seeded and the assertion above should become a refusal"
    );
}

/// A seam's name ([`powerbox_grant`]) overrides the powerbox table and is bound flat at op 0 — the
/// `vm_fs` memfs seam and the debugger's host-completed caps. Unchecked by design: a raw `HostProc` carries no
/// typed interface (its op rides in arg0), so there is no signature to compare against.
#[test]
fn host_proc_seam_overrides_the_table() {
    let (mut host, granted) = powerbox();
    let types = vec![TypeEntry::Func(sig(
        vec![ValType::I64, ValType::I64, ValType::I64],
        vec![ValType::I64],
    ))];
    let imports = vec![Import {
        name: "vm_fs".into(),
        shape: ImportShape::Func(0),
        mode: ImportMode::Required,
    }];
    let h = host.grant_host_proc(HostProcGrant::new(
        ProcHandler::Sync(Box::new(|_op, _args, _mem, _minter| Ok(vec![0i64]))),
        temen_interp::CapState::Stateless,
    ));

    let seams = [("vm_fs", h)];
    let bound = host.bind_powerbox_manifest(&imports, &types, &powerbox_grant(&granted, &seams));
    assert_eq!(bound, Ok(()));
    let b = host.import_binding(0).expect("vm_fs bound");
    assert_eq!(b.type_id, cap_id::HOST_PROC);
    assert_eq!(b.op, 0, "flat: the guest's fs op rides in arg0");
    assert_eq!(b.handle, h);
}

/// **A capability's scalar width may differ from the pinned one; its arity may not.**
///
/// The pinned `exit` is `(i32) -> ()`, but `temen-dap`'s `exit_code` manifest declares
/// `(i64) -> ()` and has always run correctly — the dispatcher reads the code the same way at
/// either width. An exact-`FuncType` check refused that well-formed import and took the DAP exit
/// path down on every platform; the discriminator is *arity*, because the data-SP the
/// function-symbol form prepends is what changes it. Both widths bind here; the SP-led two-arg
/// form does not.
#[test]
fn exit_binds_at_either_width_but_not_at_the_wrong_arity() {
    for code in [ValType::I32, ValType::I64] {
        let (mut host, granted) = powerbox();
        let imports = vec![Import {
            name: "exit".to_string(),
            shape: ImportShape::Func(0),
            mode: ImportMode::Required,
        }];
        let types = vec![TypeEntry::Func(sig(vec![code], vec![]))];
        let bound = bind(&mut host, &granted, &imports, &types);
        assert_eq!(
            bound,
            Ok(()),
            "exit declared as ({code:?}) -> () is a well-formed capability import"
        );
    }

    // The function-symbol form: `exit(SP, code)` — one parameter too many.
    let (mut host, granted) = powerbox();
    let imports = vec![Import {
        name: "exit".to_string(),
        shape: ImportShape::Func(0),
        mode: ImportMode::Required,
    }];
    let types = vec![TypeEntry::Func(sig(
        vec![ValType::I64, ValType::I32],
        vec![],
    ))];
    let refusals = bind(&mut host, &granted, &imports, &types).expect_err("refused");
    assert_eq!(refusals.len(), 1, "the SP-led form is refused");
    assert_eq!(refusals[0].name, "exit");
}
