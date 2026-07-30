//! SPEC.md slice 7 — completeness closure. For every structural op row (control / host
//! / concurrency / atomics / terminators): (1) the witness module verifies under both
//! `svm-verify` and the reference verifier — except `call_import`, the un-verifiable
//! pre-resolution form; (2) it round-trips `decode∘encode = id`; (3) its opcode byte is
//! pinned against the spec's restated map by first divergence against a baseline that
//! differs only in the target op. Together with the value/memory/SIMD suites this homes
//! every one of the 86 `Inst` variants and all 7 `Terminator`s in a spec row.

use svm::encode::{decode_module, decode_unit, encode_module, encode_unit};
use svm::verify::verify_module;
use svm_ir::{Inst, Module, Terminator};
use svm_spec::structural::struct_rows;
use svm_spec::Enc;

#[test]
fn structural_rows_verify_roundtrip_and_pin_encoding() {
    for row in struct_rows() {
        // The v9 link-form data addresses exist only in the **object dialect** — their rows
        // (and their baselines, so the byte-pin divergence isn't the header flag) ride
        // `encode_unit`/`decode_unit`; every other row rides the runnable path.
        let object = !row.is_term
            && matches!(
                row.module.funcs[0].blocks[0].insts[0],
                Inst::DataSelf { .. } | Inst::DataSym { .. } | Inst::DataTop
            );
        let enc = |m: &Module| if object { encode_unit(m) } else { encode_module(m) };
        let dec = |b: &[u8]| if object { decode_unit(b) } else { decode_module(b) };
        // (1) Typing witness: accepted by both verifiers (skip the un-verifiable
        // pre-resolution import form).
        if row.verifies {
            verify_module(&row.module)
                .unwrap_or_else(|e| panic!("[{}] production verifier rejected: {e:?}", row.id));
            svm_spec::verify::verify(&row.module)
                .unwrap_or_else(|e| panic!("[{}] reference verifier rejected: {e}", row.id));
        } else {
            assert!(
                verify_module(&row.module).is_err()
                    && svm_spec::verify::verify(&row.module).is_err(),
                "[{}] expected both verifiers to reject the pre-resolution form",
                row.id
            );
        }

        // (2) Round-trip identity.
        let bytes = enc(&row.module);
        let back = dec(&bytes).unwrap_or_else(|e| panic!("[{}] decode failed: {e:?}", row.id));
        assert_eq!(
            back, row.module,
            "[{}] decode∘encode changed the IR",
            row.id
        );

        // (3) Opcode pin: a baseline that differs only in the single target op (the
        // sole instruction, or the block-0 terminator) diverges from `bytes` exactly
        // at the op's opcode byte — the wire format has counts, not length prefixes.
        let mut base = row.module.clone();
        if row.is_term {
            base.funcs[0].blocks[0].term = if row.encoding == Enc::Byte(0x83) {
                Terminator::Unreachable // 0x8F, distinct from `return` (0x83)
            } else {
                Terminator::Return(vec![]) // 0x83, distinct from every other terminator
            };
        } else {
            base.funcs[0].blocks[0].insts[0] = Inst::ConstI32(0); // 0x10, distinct from all
        }
        let base_bytes = enc(&base);
        let i = bytes
            .iter()
            .zip(&base_bytes)
            .position(|(a, b)| a != b)
            .unwrap_or_else(|| panic!("[{}] no divergence vs baseline", row.id));
        match row.encoding {
            Enc::Byte(b) => assert_eq!(
                bytes[i], b,
                "[{}] opcode byte: spec says {b:#04x}, encoder wrote {:#04x}",
                row.id, bytes[i]
            ),
            Enc::Prefixed(p, s) => {
                assert_eq!(bytes[i], p, "[{}] escape prefix", row.id);
                assert_eq!(bytes[i + 1], s, "[{}] sub-opcode", row.id);
            }
        }
    }
}
