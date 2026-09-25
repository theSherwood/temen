//! **The chibicc fork, built once per test *run* — not once per test process.**
//!
//! Fifteen test files each carried a byte-identical `chibicc()` that ran `make` in the shared
//! `frontend/chibicc` source tree behind a `OnceLock`. A `OnceLock` serialises threads inside one
//! process, and nextest gives every test its own process: a dozen of them race the same `make` in
//! the same directory. That has two failure windows, both of which have been seen on the macOS
//! runner (`a_compiled_c_parent_kills_its_forked_child_by_pid`, #1611) — two `make`s compiling the
//! same `.o` and linking a half-written one, and one process **rewriting `chibicc` while another is
//! executing it**, which surfaces as a bare non-zero exit from a compile that is otherwise fine.
//!
//! The tree's [`cache_lock`] closes both: the first holder builds the tree, every later holder finds
//! it up to date and writes nothing, so no execution ever overlaps a write.

#[path = "cache_lock.rs"]
mod cache_lock;

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

/// The built chibicc binary. Builds it on first use in this process, under the build lock.
pub fn chibicc() -> &'static Path {
    static CC: OnceLock<PathBuf> = OnceLock::new();
    CC.get_or_init(|| {
        // `../../frontend/chibicc` from any crate under `crates/` — the same anchor `repo_root()`
        // uses, resolved here so this module is self-contained across crates that `#[path]` it in.
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../frontend/chibicc")
            .canonicalize()
            .expect("frontend/chibicc");
        let _lock = cache_lock::lock(&dir);
        let status = Command::new("make")
            .arg("-s")
            .current_dir(&dir)
            .status()
            .expect("run `make` to build the chibicc fork");
        assert!(status.success(), "chibicc build failed");
        dir.join("chibicc")
    })
    .as_path()
}
