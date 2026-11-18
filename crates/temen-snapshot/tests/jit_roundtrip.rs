//! §22 guest-JIT × the snapshot codec (DURABILITY.md §12.5 **Slice 2**): a durable domain's
//! compiled units survive freeze → serialize → restore, and the re-pinned `JitTable`/`JitCode`
//! handles resolve against the rebuilt `jit_tables`. Proves the two §12.6 invariants for the new
//! Section 5 (canonical byte-identical re-serialize; fail-closed on a tampered unit) plus the
//! handle-index bounds gate.

use temen_interp::{DurableJitTable, DurableJitUnit, Host, JitRestoreError, JitValidated};
use temen_ir::Module;
use temen_snapshot::{freeze, restore, RestoreError};

const SIZE_LOG2: u8 = 18;
const WINDOW: usize = 1 << SIZE_LOG2;

/// A minimal `Jit` validator (the embedder-injected decode+verify gate): decode the submitted
/// blob as a module, re-verify, hand back its funcs. Mirrors the real `jit_blob_validator` shape
/// without pulling in `temen-run` — the codec test only needs *some* validated unit in a domain.
fn validator(bytes: &[u8], _mem: Option<u8>, _symtab: &[u8]) -> Result<JitValidated, i64> {
    let m = temen_encode::decode_module(bytes).map_err(|_| -22i64)?;
    temen_verify::verify_module(&m).map_err(|_| -22i64)?;
    Ok(JitValidated {
        funcs: m.funcs.into(),
        types: m.types.into(),
        data: None,
    })
}

/// The module whose window/digest gates the artifact (a plain memory-18 module — the guest that
/// holds the `Jit` cap). The units are submitted separately to `compile`.
fn gate_module() -> Module {
    let m = temen_text::parse_module(
        "memory 18 shadow 16448 65536\nfunc () -> (i64) {\nblock 0 () {\n  v0 = i64.const 0\n  return v0\n  }\n}\n",
    )
    .expect("parse gate");
    temen_verify::verify_module(&m).expect("verify gate");
    m
}

/// A submitted unit blob: `() -> i64` returning `n` (declares memory 18 to match the domain).
fn unit_blob(n: i64) -> Vec<u8> {
    let src = format!(
        "memory 18 shadow 16448 65536\nfunc () -> (i64) {{\nblock 0 () {{\n  v0 = i64.const {n}\n  return v0\n  }}\n}}\n"
    );
    let m = temen_text::parse_module(&src).expect("parse unit");
    temen_verify::verify_module(&m).expect("verify unit");
    temen_encode::encode_module(&m)
}

#[test]
fn jit_domain_survives_freeze_serialize_restore() {
    let module = gate_module();

    // A domain with two compiled units (indices 0 and 1); keep both handles live.
    let mut host = Host::new();
    host.set_jit_validator(validator);
    let dom = host.grant_jit(Some(SIZE_LOG2));
    let c0 = match host.jit_compile(dom, &unit_blob(42)) {
        Ok(Ok(c)) => c,
        _ => panic!("compile 0 must succeed"),
    };
    let c1 = match host.jit_compile(dom, &unit_blob(99)) {
        Ok(Ok(c)) => c,
        _ => panic!("compile 1 must succeed"),
    };
    let funcs0 = host.jit_unit_funcs(c0.domain, c0.unit).expect("funcs 0");
    let funcs1 = host.jit_unit_funcs(c1.domain, c1.unit).expect("funcs 1");

    // Freeze → serialize (the JIT domain now rides Section 5 instead of blocking the freeze).
    let window = vec![0u8; WINDOW];
    let artifact = freeze(&module, &window, &host).expect("freeze admits a durable JIT domain");

    // Restore into a FRESH host (no validator installed — restore rebuilds + re-verifies the units
    // from the artifact itself, it never re-runs the validator).
    let mut thost = Host::new();
    let restored = restore(&artifact, &module, &mut thost).expect("restore");

    // §12.6 invariant 1 — canonical: re-serializing the restored domain is byte-identical.
    assert_eq!(
        freeze(&module, &restored, &thost).expect("re-freeze"),
        artifact,
        "re-serialize of a restored JIT domain is byte-identical"
    );

    // The re-pinned handles resolve against the rebuilt domain, and the units' funcs match.
    assert_eq!(
        thost.resolve_jit_domain(dom).expect("domain resolves"),
        c0.domain
    );
    assert_eq!(
        thost.resolve_jit_code(c0.handle).expect("code 0 resolves"),
        (c0.domain, c0.unit)
    );
    assert_eq!(
        thost.resolve_jit_code(c1.handle).expect("code 1 resolves"),
        (c1.domain, c1.unit)
    );
    assert_eq!(
        thost.jit_unit_funcs(c0.domain, c0.unit).expect("funcs 0"),
        funcs0,
        "restored unit 0 funcs match"
    );
    assert_eq!(
        thost.jit_unit_funcs(c1.domain, c1.unit).expect("funcs 1"),
        funcs1,
        "restored unit 1 funcs match"
    );
}

/// Install-durability (v17): a domain's B2 `install` occupancy + the `call.dyn` table
/// reservation ride Section 5 and re-grant on restore, byte-identically.
#[test]
fn jit_install_occupancy_round_trips() {
    let module = gate_module();
    let mut host = Host::new();
    host.set_jit_validator(validator);
    let dom = host.grant_jit_with_table(Some(SIZE_LOG2), 4); // reserve call.dyn padding
    let c0 = host.jit_compile(dom, &unit_blob(42)).unwrap().unwrap();
    // Record an install of the unit at a padding slot (what the eval-loop / native install arm does).
    host.jit_record_install(c0.domain, 3, c0.unit);

    let window = vec![0u8; WINDOW];
    let artifact = freeze(&module, &window, &host).expect("freeze");
    let mut thost = Host::new();
    let restored = restore(&artifact, &module, &mut thost).expect("restore");

    assert_eq!(
        thost.jit_installs(c0.domain),
        vec![(3, c0.unit)],
        "install occupancy re-grants (slot 3 → unit)"
    );
    assert_eq!(
        thost.jit_table_log2(),
        4,
        "the call.dyn table reservation is restored"
    );
    assert_eq!(
        freeze(&module, &restored, &thost).expect("re-freeze"),
        artifact,
        "re-serialize of a restored install-bearing domain is byte-identical"
    );
}

/// #3034 — a re-applied `install` is one the guest's own made: into a padding slot of the dispatch
/// table, past the module's functions, naming a unit its domain holds. On `b099bbb` an artifact could
/// place a unit over a module function's slot (slot 0 here): the next run's `call.dyn` of that slot
/// reached the unit's code instead of the module function the slot names. A slot past the table, or a
/// unit the domain does not hold, was skipped silently. Each is now refused.
#[test]
fn an_install_no_run_makes_is_refused() {
    let module = gate_module();
    let refused = |slot: u32, unit: Option<u32>| {
        let mut host = Host::new();
        host.set_jit_validator(validator);
        let dom = host.grant_jit_with_table(Some(SIZE_LOG2), 4); // 16 slots, function 0 in slot 0
        let c0 = host.jit_compile(dom, &unit_blob(42)).unwrap().unwrap();
        host.jit_record_install(c0.domain, slot, unit.unwrap_or(c0.unit));
        let artifact = freeze(&module, &vec![0u8; WINDOW], &host).expect("freeze");
        matches!(
            restore(&artifact, &module, &mut Host::new()),
            Err(RestoreError::Malformed)
        )
    };
    assert!(!refused(3, None), "a padding slot, as a run installs");
    assert!(refused(0, None), "a module function's slot");
    assert!(refused(16, None), "a slot past the table");
    assert!(refused(3, Some(5)), "a unit the domain does not hold");
}

/// #3102 — the dispatch table is the run's one, so an `uninstall` through any domain's handle
/// clears a slot another domain filled, and a later `install` through either takes it again. Each
/// slot has one record at the freeze: the restore admits the artifact, and only the new install is
/// re-applied. On `f2aff3d` the uninstall forgot the slot only in its own domain: the slot carried
/// two records, and the restore refused a real run's artifact as `Malformed`.
#[test]
fn a_slot_uninstalled_through_another_domain_and_reinstalled_has_one_record() {
    let module = gate_module();
    let mut host = Host::new();
    host.set_jit_validator(validator);
    let a = host.grant_jit_with_table(Some(SIZE_LOG2), 4);
    let b = host.grant_jit_with_table(Some(SIZE_LOG2), 4);
    let ca = host.jit_compile(a, &unit_blob(1)).unwrap().unwrap();
    let cb = host.jit_compile(b, &unit_blob(2)).unwrap().unwrap();
    // What the install and uninstall arms record: A installs at 3, B uninstalls 3 and installs there.
    host.jit_record_install(ca.domain, 3, ca.unit);
    host.jit_forget_install(3);
    host.jit_record_install(cb.domain, 3, cb.unit);
    assert_eq!(host.jit_installs(ca.domain), vec![], "A's install is gone");

    let artifact = freeze(&module, &vec![0u8; WINDOW], &host).expect("freeze");
    let mut thost = Host::new();
    restore(&artifact, &module, &mut thost).expect("a real run's artifact restores");
    assert_eq!(thost.jit_installs(ca.domain), vec![]);
    assert_eq!(thost.jit_installs(cb.domain), vec![(3, cb.unit)]);
}

/// A JIT-free durable domain's artifact is unaffected by Slice 2: Section 5 is elided, so it stays
/// byte-identical across a round-trip (the `!jit.is_empty()` guard).
#[test]
fn no_jit_domain_elides_section_5() {
    let module = gate_module();
    let host = Host::new();
    let window = vec![0u8; WINDOW];
    let artifact = freeze(&module, &window, &host).expect("freeze");
    let mut thost = Host::new();
    let restored = restore(&artifact, &module, &mut thost).expect("restore");
    assert_eq!(
        freeze(&module, &restored, &thost).expect("re-freeze"),
        artifact,
        "a JIT-free artifact round-trips byte-identically"
    );
}

/// Fail-closed at the reconstruction boundary: a `DurableJitTable` whose unit IR is garbage is
/// rejected (`Decode`) rather than admitting unverified funcs. The codec's `JitReconstruct` maps
/// this failure; here we exercise the `Host` primitive directly (deterministic, no byte-fiddling).
#[test]
fn restore_rejects_a_corrupt_unit() {
    let mut host = Host::new();
    let bad = vec![DurableJitTable {
        mem_log2: Some(SIZE_LOG2),
        units_left: 10,
        bytes_left: 1 << 20,
        units: vec![DurableJitUnit {
            unit_ir: vec![0xff, 0x00, 0x13, 0x37], // not a module image
            install_type_id: 0,
        }],
        installed: Vec::new(),
    }];
    assert_eq!(
        host.restore_durable_jit(&bad),
        Err(JitRestoreError::Decode),
        "a corrupt unit IR is rejected fail-closed"
    );
}

/// The codec's fail-closed path: tampering with the serialized unit IR makes `restore` refuse
/// (`JitReconstruct`) rather than mis-thaw. We corrupt the artifact's trailing unit-IR bytes —
/// Section 5 is last, and the unit IR is its tail — and confirm a clean error.
#[test]
fn codec_refuses_a_tampered_unit() {
    let module = gate_module();
    let mut host = Host::new();
    host.set_jit_validator(validator);
    let dom = host.grant_jit(Some(SIZE_LOG2));
    host.jit_compile(dom, &unit_blob(42)).unwrap().unwrap();

    let window = vec![0u8; WINDOW];
    let mut artifact = freeze(&module, &window, &host).expect("freeze");

    // Zero the last 8 bytes — inside the trailing unit-IR image — so `decode_module`/`verify_module`
    // fails on restore. (Non-canonical trailing bytes or a broken structure both fail-close.)
    let n = artifact.len();
    for b in &mut artifact[n - 8..] {
        *b = 0;
    }
    match restore(&artifact, &module, &mut Host::new()) {
        Err(RestoreError::JitReconstruct) | Err(RestoreError::Malformed) => {}
        other => panic!("expected a fail-closed refusal, got {other:?}"),
    }
}

/// #3034 — a `Jit` grant reserves at most `2^MAX_TABLE_LOG2` dispatch-table slots: every engine builds
/// the table eagerly. The ceiling itself is granted, and its artifact restores.
#[test]
fn a_grant_at_the_table_ceiling_is_granted_and_restores() {
    let module = gate_module();
    let mut host = Host::new();
    host.set_jit_validator(validator);
    host.grant_jit_with_table(Some(SIZE_LOG2), temen_ir::MAX_TABLE_LOG2);
    let artifact = freeze(&module, &vec![0u8; WINDOW], &host).expect("freeze");
    let mut thost = Host::new();
    restore(&artifact, &module, &mut thost).expect("restore");
    assert_eq!(thost.jit_table_log2(), temen_ir::MAX_TABLE_LOG2);
}

/// One past the ceiling is an embedder's precondition broken, as a full handle table is.
#[test]
#[should_panic(expected = "MAX_TABLE_LOG2")]
fn a_grant_past_the_table_ceiling_panics() {
    let mut host = Host::new();
    host.grant_jit_with_table(Some(SIZE_LOG2), temen_ir::MAX_TABLE_LOG2 + 1);
}

/// And so does a reservation set past it.
#[test]
#[should_panic(expected = "MAX_TABLE_LOG2")]
fn a_reservation_past_the_table_ceiling_panics() {
    Host::new().set_jit_table_log2(temen_ir::MAX_TABLE_LOG2 + 1);
}
