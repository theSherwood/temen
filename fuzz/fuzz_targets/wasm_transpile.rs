//! libFuzzer target for the **wasm frontend** (`temen-wasm`, `DESIGN.md` §3; the README §1a thesis).
//!
//! `temen-wasm` decodes *untrusted* core-wasm bytes and transpiles them to Temen IR. Like the binary
//! `decode` path it must **fail closed** on arbitrary input — never panic, OOM, or hang — and any IR
//! it does emit must clear the same escape-TCB gate every module passes: `verify`, then a
//! fuel-bounded interpret that cannot escape. (Robustness now rests on an up-front
//! `wasmparser::Validator` pass in `transpile`; this target is the standing net that keeps it honest.)
//!
//! Run: `cargo +nightly fuzz run wasm_transpile`
#![no_main]

use libfuzzer_sys::fuzz_target;

use temen::default_args;
use temen_interp::run;

fuzz_target!(|data: &[u8]| {
    // Transpiling arbitrary bytes must never panic/OOM/hang — it either errors or yields a Module.
    if let Ok(t) = temen_wasm::transpile(data) {
        // A transpiled module that verifies must be safe to interpret (bounded by fuel): the same
        // "verified ⇒ cannot escape" invariant the `decode_verify` target asserts for the binary path.
        if temen_verify::verify_module(&t.module).is_ok() {
            for (fi, f) in t.module.funcs.iter().enumerate() {
                let args = default_args(&f.params);
                let mut fuel = 10_000u64;
                let _ = run(&t.module, fi as u32, &args, &mut fuel);
            }
        }
    }
});
