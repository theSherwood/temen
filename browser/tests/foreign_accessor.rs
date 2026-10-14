//! #2416: the foreign-memory word path's two wasm modules, the dispatcher an agent binds the engine's
//! `foreign_load`/`foreign_store` to and the accessor it instantiates over each registered memory, are
//! embedded in `web/foreign-mem.js` as bytes beside the WAT they are assembled from. This pins the
//! bytes to the WAT, so an edit to either alone fails here rather than in a browser.

/// The text of the `/* wasm:<tag>` block comment and the bytes of the `const <var> = new
/// Uint8Array([...])` after it.
fn embedded<'a>(js: &'a str, tag: &str, var: &str) -> (&'a str, Vec<u8>) {
    let open = format!("/* wasm:{tag}\n");
    let at = js.find(&open).expect("the WAT block") + open.len();
    let wat = &js[at..at + js[at..].find("*/").expect("the block's end")];
    let array = format!("const {var} = new Uint8Array([");
    let at = js.find(&array).expect("the byte array") + array.len();
    let bytes = js[at..at + js[at..].find("]);").expect("the array's end")]
        .split(',')
        .map(str::trim)
        .filter(|b| !b.is_empty())
        .map(|b| u8::from_str_radix(b.trim_start_matches("0x"), 16).expect("a hex byte"))
        .collect();
    (wat, bytes)
}

#[test]
fn the_embedded_foreign_memory_modules_are_their_wat() {
    let js = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/web/foreign-mem.js"))
        .expect("read web/foreign-mem.js");
    for (tag, var) in [("dispatch", "DISPATCH_WASM"), ("accessor", "ACCESSOR_WASM")] {
        let (wat, bytes) = embedded(&js, tag, var);
        let want = wat::parse_str(wat).unwrap_or_else(|e| panic!("{tag}'s WAT: {e}"));
        assert_eq!(bytes, want, "`{var}` is not `wasm:{tag}`'s WAT assembled");
    }
}
