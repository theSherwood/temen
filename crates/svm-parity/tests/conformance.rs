//! **The honesty pin.** For every catalogued op (that isn't explicitly `conf_skip`'d), build its
//! minimal module and check the real backends' `compile` results against the manifest. A `Full` the
//! backend can't compile — or a `Declines`/`NotYet` it *can* — is a hard failure, so the matrix can
//! never silently drift from what the backends actually do (INVARIANTS.md #9).
//!
//! Coverage boundaries, kept explicit:
//! * The **tree-walk** column is the oracle (Full by definition); it is not compile-checked here.
//! * **Conditional** cells (target-gated fiber/thread/`setjmp` ops) and ops needing host/import
//!   wiring are `conf_skip`'d — listed in the matrix, exercised by the differential harnesses, not
//!   re-pinned here. The test asserts the skip set stays small and reasoned.

use svm_parity::{catalog, Backend, Status};

/// Did a backend accept the module (emit code / a program), or decline it (Unsupported)?
#[derive(PartialEq, Debug)]
enum Got {
    Supported,
    Declined,
}

fn bytecode_support(m: &svm_ir::Module) -> Got {
    match svm_interp::bytecode::SharedProgram::compile(m) {
        Some(_) => Got::Supported,
        None => Got::Declined,
    }
}

fn cranelift_support(m: &svm_ir::Module, entry: u32) -> Got {
    match svm_jit::compile(m, entry) {
        Ok(_) => Got::Supported,
        Err(svm_jit::JitError::Unsupported(_)) => Got::Declined,
        Err(e) => panic!("cranelift returned an unexpected (non-Unsupported) error: {e:?}"),
    }
}

fn wasmjit_support(m: &svm_ir::Module) -> Got {
    match svm_wasm_jit::compile_module(m) {
        Ok(_) => Got::Supported,
        // `svm_wasm_jit::Error` is the single `Unsupported` variant — any `Err` is a decline.
        Err(svm_wasm_jit::Error::Unsupported(_)) => Got::Declined,
    }
}

/// What the manifest's [`Status`] predicts the backend does. `None` ⇒ the cell is target-conditional,
/// so we don't assert a fixed outcome.
fn expected(status: Status) -> Option<Got> {
    match status {
        Status::Full => Some(Got::Supported),
        Status::Declines | Status::NotYet => Some(Got::Declined),
        Status::Conditional => None,
    }
}

#[test]
fn manifest_matches_the_backends() {
    let mut mismatches: Vec<String> = Vec::new();

    for op in catalog() {
        if op.conf_skip.is_some() {
            continue;
        }

        // Every auto-checked module must verify — the matrix classifies *verified* IR.
        if let Err(e) = svm_verify::verify_module(&op.module) {
            mismatches.push(format!(
                "`{}`: fixture failed to verify: {e:?}",
                op.mnemonic
            ));
            continue;
        }

        let cells = op.cells();
        let checks = [
            (Backend::Bytecode, bytecode_support(&op.module)),
            (Backend::Cranelift, cranelift_support(&op.module, op.entry)),
            (Backend::WasmJit, wasmjit_support(&op.module)),
        ];
        for (backend, got) in checks {
            if let Some(want) = expected(cells[backend as usize].status) {
                if want != got {
                    mismatches.push(format!(
                        "`{}` on {}: manifest says {:?} (expect {:?}) but backend {:?}",
                        op.mnemonic,
                        backend.short(),
                        cells[backend as usize].status,
                        want,
                        got,
                    ));
                }
            }
        }
    }

    assert!(
        mismatches.is_empty(),
        "manifest disagrees with the backends ({} rows):\n{}",
        mismatches.len(),
        mismatches.join("\n"),
    );
}

/// The `conf_skip` set is a deliberate, auditable coverage gap — keep it from quietly growing. Every
/// skipped op must be target-conditional or host-wired; the count is asserted so a careless new skip
/// is noticed in review.
#[test]
fn skip_set_is_small_and_reasoned() {
    let skipped: Vec<_> = catalog()
        .into_iter()
        .filter(|o| o.conf_skip.is_some())
        .map(|o| o.mnemonic)
        .collect();
    // Fibers/threads/futex/setjmp/gc (target-conditional) + the cap/import/export host-wired ops +
    // the `process, serve & fork` sub-ops (scheduler/host-wired, exercised by the fork/serve
    // harnesses).
    assert!(
        skipped.len() <= 33,
        "conf_skip set grew to {} — audit these before raising the bound:\n{:?}",
        skipped.len(),
        skipped,
    );
}

/// The classifier duplicates the reserved cap-id / self-op numbers as bare data (to stay free of a
/// backend dep). Pin them equal to the interpreter's own constants so the two can never drift — a
/// renumbered op would otherwise silently mis-classify its whole matrix row.
#[test]
fn capcall_op_numbers_match_the_interpreter() {
    use svm_parity::capcall;
    assert_eq!(capcall::INSTANTIATOR, svm_interp::cap_id::INSTANTIATOR);
    assert_eq!(capcall::SVC_POLL, svm_interp::CAP_SELF_SVC_POLL);
    assert_eq!(capcall::SVC_WAIT, svm_interp::CAP_SELF_SVC_WAIT);
    assert_eq!(capcall::CLONE_CALLER, svm_interp::CAP_SELF_CLONE_CALLER);
    assert_eq!(capcall::REAP, svm_interp::CAP_SELF_REAP);
    assert_eq!(capcall::FUEL_REMAINING, svm_interp::CAP_SELF_FUEL_REMAINING);
    assert_eq!(capcall::EXEC, svm_interp::CAP_SELF_EXEC);
}
