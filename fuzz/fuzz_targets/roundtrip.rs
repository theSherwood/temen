//! libFuzzer target for format round-trip identity (`DESIGN.md` §3a).
//!
//! For any decodable module, re-encoding and decoding again must reproduce it
//! exactly (`decode ∘ encode = id` on the IR). For any *verified* module, the text
//! form must round-trip identically too (`parse ∘ print = id`). A crash here is an
//! encoder/decoder or printer/parser bug.
//!
//! Run: `cargo +nightly fuzz run roundtrip`
#![no_main]

use libfuzzer_sys::fuzz_target;
use svm_encode::{decode_module, decode_unit, encode_module, encode_unit};
use svm_text::{parse_module, print_module};
use svm_verify::verify_module;

fuzz_target!(|data: &[u8]| {
    // Object dialect (v9): for any decodable *unit*, `decode_unit ∘ encode_unit = id`, and the
    // runnable path must reject any actual object at the header (never decode link scaffolding).
    if let Ok(u) = decode_unit(data) {
        let re = decode_unit(&encode_unit(&u)).expect("re-decode of encoded unit failed");
        assert_eq!(u, re, "object round-trip changed the IR");
        if decode_module(data).is_ok() {
            // The bytes decoded on the runnable path too, so they carried flag 0 — the unit
            // must therefore be free of object-only payload.
            assert!(
                u.data_ptrs.is_empty() && u.data_exports.is_empty(),
                "runnable-dialect bytes decoded object-only sections"
            );
        }
    }

    let Ok(m) = decode_module(data) else { return };

    // Binary round-trip holds for every decodable module (not just verified ones).
    let re = decode_module(&encode_module(&m)).expect("re-decode of encoded IR failed");
    assert_eq!(m, re, "binary round-trip changed the IR");

    // Text round-trip is only well-defined for verified modules (the text form can
    // only name backward value references, valid call indices, etc.).
    if verify_module(&m).is_ok() {
        let parsed = parse_module(&print_module(&m)).expect("re-parse of printed IR failed");
        assert_eq!(m, parsed, "text round-trip changed the IR");
    }
});
