//! Shared fixtures for the frontier-matrix (#1413) and run-mode-matrix (#2545) conformance tests.
pub mod capability_probe;

/// Whether `note` carries a `#1234`-shaped issue reference: the check behind each matrix's
/// `every_known_gap_names_its_issue`, pinned by `frontier_conformance.rs`.
// Each test binary compiles this module, and only the two that check their gaps call it.
#[allow(dead_code)]
pub fn cites_an_issue(note: &str) -> bool {
    note.split('#')
        .skip(1)
        .any(|rest| rest.starts_with(|ch: char| ch.is_ascii_digit()))
}

/// The §22 `Jit` validator: a submitted unit is a real wire-encoded module, so decode and verify it
/// exactly as a host serving `vm_jit_compile` must (#922 — the host re-reads the blob for the unit's
/// type section, and a stub blob has none).
// Only the binaries that compile a unit call it.
#[allow(dead_code)]
pub fn validate_unit(
    bytes: &[u8],
    _mode: Option<u8>,
    _sigs: &[u8],
) -> Result<temen_interp::JitValidated, i64> {
    let m = temen_encode::decode_module(bytes).map_err(|_| -22i64)?;
    temen_verify::verify_module(&m).map_err(|_| -22i64)?;
    Ok(temen_interp::JitValidated {
        funcs: m.funcs.into(),
        types: m.types.into(),
        data: None,
    })
}
