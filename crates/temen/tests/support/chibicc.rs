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
//! One cross-process lock closes both: the first holder builds the tree, every later holder finds it
//! up to date and writes nothing, so no execution ever overlaps a write.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

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
        let _lock = BuildLock::acquire(&dir);
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

/// A cross-process mutex over the chibicc build tree, held for the whole of `make`.
///
/// `create_dir` is the portable atomic test-and-set: exactly one racer gets `Ok`, everyone else
/// gets `AlreadyExists` and spins. Released on `Drop`, so a panicking or asserting test hands it
/// back; a process *killed* mid-build cannot, so a lock nobody released within `BREAK_AFTER` is
/// broken rather than wedging every later test run.
struct BuildLock(PathBuf);

impl BuildLock {
    fn acquire(dir: &Path) -> Self {
        const BREAK_AFTER: Duration = Duration::from_secs(300);
        let path = dir.join(".build.lock");
        let waiting_since = Instant::now();
        loop {
            match std::fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    if waiting_since.elapsed() > BREAK_AFTER {
                        let _ = std::fs::remove_dir(&path);
                    }
                    std::thread::sleep(Duration::from_millis(50));
                }
                Err(e) => panic!("chibicc build lock {}: {e}", path.display()),
            }
        }
    }
}

impl Drop for BuildLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir(&self.0);
    }
}
