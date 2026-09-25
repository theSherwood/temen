//! **One writer per shared on-disk test cache.**
//!
//! Tests share caches on disk: the chibicc build tree, and the fetched and built sources under the
//! temp dir (openlibm, SQLite, Tcl, …). Tests run as parallel threads under `cargo test` and as
//! parallel processes under nextest, so a helper that checks whether its cache is populated and
//! then populates it races every other caller doing the same. The losers read half-written trees:
//! a `chibicc` rewritten while another process executes it (#1610), a Tcl module linked from 3 of
//! its 162 translation units while a second build deleted the rest. A `OnceLock` fixes neither,
//! because it serialises threads within one process only.
//!
//! [`lock`] is the fix. Every helper takes it before its check-then-populate and holds it for as
//! long as it reads what it populated. The lock is an OS file lock, so it also serialises threads,
//! which open the file separately. The OS releases it when the holder exits or is killed, so no
//! stale-lock timeout is needed.

use std::fs::File;
use std::path::Path;

/// Exclusive hold on the cache at `dir`, creating the directory if needed. Released when the
/// returned `File` drops. The lock file stays in place: deleting it would let a waiter that already
/// opened the old file and a newcomer that creates a new one both hold "the" lock.
pub fn lock(dir: &Path) -> File {
    std::fs::create_dir_all(dir).unwrap_or_else(|e| panic!("create {}: {e}", dir.display()));
    let path = dir.join(".cache.lock");
    let file = File::options()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&path)
        .unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    file.lock()
        .unwrap_or_else(|e| panic!("lock {}: {e}", path.display()));
    file
}
