//! **Build the nimony stdlib image for the in-browser compiler card (#958).** Walk the nimony `lib/`
//! tree into a single `SVMFSIM1` blob with the exact key layout `nimc::compile_nim` mounts: every file
//! keyed `lib/<rel>`, and every `std/…` file *also* keyed flattened `lib/<rel-minus-std/>` — the same
//! two-view seed the headless `nim_selfdrive` driver and the `nimc` `io_hello` test build (nimony
//! resolves `std/syncio` and bare `syncio` alike, and its search path is `lib/`). `build_image` can't
//! produce this (it keys files relative to the dir with no `lib/` prefix and no flattened view), so the
//! browser asset gets its own builder.
//!
//!   cargo run --release -p svm-run --example build_nim_stdlib_image -- <nimony-lib-dir> <out.img>
//!
//! Ship it gzipped alongside the phase `.svmb.gz` (see `browser/web/assets/`); the card fetches +
//! inflates it and hands the bytes to `svm_compile_nim_fs`.

use std::path::{Path, PathBuf};
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let (Some(dir), Some(out)) = (args.next(), args.next()) else {
        eprintln!("usage: build_nim_stdlib_image <nimony-lib-dir> <out.img>");
        std::process::exit(2);
    };
    let lib = Path::new(&dir);
    let t = Instant::now();

    // Walk `lib/` into the two-view seed `compile_nim` expects (mirrors nimc.rs `tests::seed`).
    let mut files: Vec<(String, Vec<u8>)> = Vec::new();
    let mut stack: Vec<PathBuf> = vec![lib.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).expect("read_dir") {
            let p = e.expect("entry").path();
            if p.is_dir() {
                stack.push(p);
            } else if p.is_file() {
                let rel = p
                    .strip_prefix(lib)
                    .expect("strip lib prefix")
                    .to_string_lossy()
                    .replace('\\', "/");
                let bytes = std::fs::read(&p).expect("read file");
                if let Some(r) = rel.strip_prefix("std/") {
                    files.push((format!("lib/{r}"), bytes.clone()));
                }
                files.push((format!("lib/{rel}"), bytes));
            }
        }
    }
    files.sort_by(|a, b| a.0.cmp(&b.0));

    let img = svm_run::fs::encode_image(&files, &[]);
    std::fs::write(&out, &img).expect("write image");
    let bytes: usize = files.iter().map(|(_, d)| d.len()).sum();
    println!(
        "{out}: {} keys ({} KiB) -> {} KiB image in {:.1?}",
        files.len(),
        bytes / 1024,
        img.len() / 1024,
        t.elapsed(),
    );
    // Self-check: the image must round-trip (fail-closed before shipping).
    let _ = svm_run::fs::decode_image(&img).expect("image round-trips");
}
