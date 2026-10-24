// The threads build of the engine cdylib exports its function table, growable (#1414 5d), so
// `web/foreign-mem.js` can put each foreign memory's accessor functions in it and `Region::Foreign`
// call them there directly: one wasm call per access. The plain build has no foreign memories, so it
// links as before. A link argument here reaches every threads build of the cdylib, whatever else its
// `RUSTFLAGS` pass.
fn main() {
    let wasm = std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("wasm32");
    let atomics = std::env::var("CARGO_CFG_TARGET_FEATURE")
        .is_ok_and(|features| features.split(',').any(|f| f == "atomics"));
    if wasm && atomics {
        println!("cargo:rustc-link-arg-cdylib=--export-table");
        println!("cargo:rustc-link-arg-cdylib=--growable-table");
    }
}
