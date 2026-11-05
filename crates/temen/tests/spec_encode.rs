//! SPEC.md suite 3 — encoding conformance. For every op row in the executable spec,
//! (1) the single-op module round-trips `decode∘encode = id` (the directed, per-op
//! version of the `roundtrip` fuzz property), and (2) the instruction's **opcode byte
//! is pinned** against the spec's independently-restated byte map (`OpRow::encoding`),
//! so an `temen-encode` renumbering or family-base move is a red test, not a silent
//! format break.
//!
//! Opcode location: a baseline module differing *only* in the single instruction encodes
//! byte-identically up to the function's body, which follows its length (wire v17). Past the
//! length, the first byte at which the bodies differ IS the opcode ([`opcode_pin`]).

#[path = "support/opcode_pin.rs"]
mod opcode_pin;

use temen::encode::{decode_module, encode_module};
use temen_ir::{Inst, Module};
use temen_spec::{all_rows, mem_rows, module_for, module_for_mem, vectors_for, Enc, Shape};

/// Round-trip `m` and pin its single instruction's opcode byte to `encoding`.
#[track_caller]
fn check(id: &str, encoding: Enc, m: &Module) {
    // (1) Round-trip identity.
    let bytes = encode_module(m);
    let back = decode_module(&bytes).unwrap_or_else(|e| panic!("decode failed for {id}: {e:?}"));
    assert_eq!(back, *m, "decode∘encode changed the IR for {id}");

    // (2) Opcode pin by first divergence against a const baseline (a different const
    // for the const rows themselves).
    let mut base = m.clone();
    base.funcs[0].blocks[0].insts[0] = if encoding == Enc::Byte(0x10) {
        Inst::ConstI64(0)
    } else {
        Inst::ConstI32(0)
    };
    let base_bytes = encode_module(&base);
    let i = opcode_pin::first_divergence(m, &bytes, &base_bytes, encode_module)
        .unwrap_or_else(|| panic!("no divergence vs const baseline for {id}"));
    match encoding {
        Enc::Byte(b) => assert_eq!(
            bytes[i], b,
            "opcode byte for {id}: spec says {b:#04x}, encoder wrote {:#04x}",
            bytes[i]
        ),
        Enc::Prefixed(p, s) => {
            assert_eq!(
                bytes[i], p,
                "escape prefix for {id}: spec says {p:#04x}, encoder wrote {:#04x}",
                bytes[i]
            );
            assert_eq!(
                bytes[i + 1],
                s,
                "sub-opcode for {id}: spec says {s:#04x}, encoder wrote {:#04x}",
                bytes[i + 1]
            );
        }
    }
}

#[test]
fn spec_encoding_conformance() {
    for row in all_rows() {
        let m = match row.shape {
            Shape::Operands => module_for(&row, &[]),
            Shape::Immediate | Shape::OperandsImmediate(_) => {
                let sample = vectors_for(&row)
                    .into_iter()
                    .next()
                    .unwrap_or_else(|| panic!("no vectors for {}", row.id));
                module_for(&row, &sample)
            }
        };
        check(&row.id, row.encoding, &m);
    }
    // Memory rows, at a couple of immediate offsets (the offset is a uleb payload
    // after the opcode byte, so the pin is offset-independent — checked at both).
    for row in mem_rows() {
        for offset in [0u64, 0x123] {
            check(&row.id, row.encoding, &module_for_mem(&row, offset));
            if !row.has_offset {
                break;
            }
        }
    }
}
