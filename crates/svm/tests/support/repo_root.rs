//! The repository root, resolved from this crate's manifest dir — the anchor the frontend/toolchain
//! integration suites (chibicc, stage1, POSIX) join their fixture and tool paths onto. Byte-identical
//! copies previously lived in fourteen test files (#923).

use std::path::PathBuf;

/// Absolute, canonicalized path to the repository root (`crates/svm/../..`).
pub fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}
