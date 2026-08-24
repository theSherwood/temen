//! libFuzzer target for the escape-TCB invariants (`DESIGN.md` §2a / §18).
//!
//! Same contract as the stable `fuzz_smoke` test, run under coverage-guided
//! fuzzing: on arbitrary bytes, `decode` must fail-closed (never panic/OOM/hang),
//! `verify` must never panic, and any *verified* module must be safe to interpret
//! (bounded by fuel). A crash here is a candidate fail-open / escape bug.
//!
//! Run: `cargo +nightly fuzz run decode_verify`
#![no_main]

use libfuzzer_sys::fuzz_target;

use temen::default_args;
use temen_encode::{decode_module, decode_unit, DecodeError};
use temen_interp::run;
use temen_verify::verify_module;

fuzz_target!(|data: &[u8]| {
    // The object dialect (v9) is tooling-facing but shares this decoder: `decode_unit` must
    // fail-closed on arbitrary bytes exactly like `decode_module`. And the header firewall —
    // any input the unit path decodes that the runnable path rejects must be rejected
    // *as an object* (flag bit 0), never for any other reason: link scaffolding stays
    // unreachable from the runtime load path.
    if decode_unit(data).is_ok() {
        match decode_module(data) {
            Ok(_) | Err(DecodeError::ObjectInput) => {}
            Err(e) => panic!("dialect divergence beyond the object flag: {e:?}"),
        }
    }
    if let Ok(m) = decode_module(data) {
        if verify_module(&m).is_ok() {
            for (fi, f) in m.funcs.iter().enumerate() {
                let args = default_args(&f.params);
                let mut fuel = 10_000u64;
                let _ = run(&m, fi as u32, &args, &mut fuel);
            }
        }
    }
});
