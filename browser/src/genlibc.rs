//! **Build the prebuilt libc unit** (#1392) — `browser/web/assets/pg_libc.temeno`.
//!
//! The chibicc card's compile used to spend nearly all its time on the seeded `playground-include/`
//! libc, which is guest C and identical in every program (a `printf` program: 8.9 s; a graphics
//! lesson: 12.2 s). With the headers split three ways (`__pg_linkage.h`), the libc's bodies compile
//! **once** into their own linkable unit and a user's program is compiled decls-only against its
//! prototypes — ~12x cheaper. Compiling the unit itself costs ~13.6 s, which is exactly what must not
//! happen in the browser, so it is a committed asset built here.
//!
//! Self-hosted and toolchain-free: this runs the **committed** `chibicc.temen` over `__pg_libc.c`
//! through the same on-ramp powerbox the card uses, so it needs only cargo — no clang, no LLVM. It is
//! therefore wire-format coupled twice over (the chibicc asset it reads, and the unit it writes):
//! regenerate through `scripts/rebuild-assets.sh` on any IR / encoder / wire change, per AGENTS.md.
//! `browser/tests/pg_libc_asset.rs` is the gate that catches the drift.
//!
//! Usage: `cargo run --release --bin genlibc [-- <out-path>]` from `browser/`.

use temen_browser::{onramp_fs_exec, playground_include_files, playground_libc_tu};
use temen_browser::{STATUS_EXIT, STATUS_OK};

fn main() {
    let out = std::env::args().nth(1).unwrap_or_else(|| {
        concat!(env!("CARGO_MANIFEST_DIR"), "/web/assets/pg_libc.temeno").to_string()
    });
    let chibicc_path = concat!(env!("CARGO_MANIFEST_DIR"), "/web/assets/chibicc.temen");
    let bytes = match std::fs::read(chibicc_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("genlibc: SKIP — {chibicc_path}: {e}");
            std::process::exit(0);
        }
    };
    let chibicc = temen_encode::decode_module(&bytes).expect("decode chibicc.temen");

    // The card's own seeded image plus the one-line TU that instantiates the headers' bodies.
    let mut files = playground_include_files();
    files.push((
        "__pg_libc.c".to_string(),
        playground_libc_tu().as_bytes().to_vec(),
    ));
    let image = temen_fs::encode_image(&files, &["include".to_string()]);

    // `-g`: the unit carries its own debug info, which the linker merges into every program linked
    // against it — that is what lets a DAP session step *into* `printf` (`temen_link_text_lib`). A
    // release run ignores it.
    let t = std::time::Instant::now();
    let compiled = onramp_fs_exec(
        &chibicc,
        &image,
        &[
            b"chibicc",
            b"--emit-object",
            b"--data-page",
            b"65536",
            b"-Iinclude",
            b"-g",
            b"/__pg_libc.c",
        ],
        b"",
    );
    let ms = t.elapsed().as_millis();
    assert!(
        compiled.status == STATUS_OK || compiled.status == STATUS_EXIT,
        "genlibc: chibicc failed (status {}) — {}",
        compiled.status,
        String::from_utf8_lossy(&compiled.stderr)
    );
    let ir = String::from_utf8(compiled.stdout).expect("chibicc emits utf8 IR text");
    let unit = temen_text::parse_module(&ir).expect("the emitted unit parses");
    let encoded = temen_encode::encode_unit(&unit);

    // Re-decode what we are about to commit: an asset that cannot be loaded back is worse than none.
    let rt = temen_encode::decode_unit(&encoded).expect("the encoded unit decodes");
    assert!(
        rt.debug_info.is_some(),
        "the unit must carry debug info (built with -g) so a debug session can step into it"
    );
    assert!(
        rt.exports.iter().any(|e| e.name == "printf"),
        "the unit must export the libc bodies; got {:?}",
        rt.exports.iter().map(|e| &e.name).collect::<Vec<_>>()
    );

    std::fs::write(&out, &encoded).expect("write the unit");
    println!(
        "genlibc: {out} — {} B ({} funcs, {} exports, {} data exports) in {ms} ms",
        encoded.len(),
        rt.funcs.len(),
        rt.exports.len(),
        rt.data_exports.len(),
    );
}
