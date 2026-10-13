//! #2306 — the process's file descriptors, a pool every run's guests share, as they share its OS
//! mappings ([`temen_interp::maps_pool`]) and its address space ([`temen_interp::RESERVED_POOL`]).
//! One descriptor past the process's limit fails the next `open`, `memfd_create` or `pipe` anywhere
//! in it, the host's own included; so each descriptor a guest causes holds one of the pool's
//! ([`Descriptor`]) for as long as it is open, and one the pool has no room for is refused with
//! `-EMFILE`, as the OS itself would refuse it, before the host asks the OS for it.

use std::sync::OnceLock;
use temen_interp::Pool;
use temen_ir::errno::EMFILE;

/// What every run's guests hold of [`pool`] between them.
static IN_USE: Pool = Pool::new();

/// How many descriptors the guests of this process may hold between them: its soft
/// `RLIMIT_NOFILE`, read when first needed, less what the host keeps for itself, an eighth of it and
/// at least 64. A host with no such limit (Windows, whose handles have none this small): unbounded.
fn pool() -> u64 {
    static POOL: OnceLock<u64> = OnceLock::new();
    *POOL.get_or_init(|| match limits() {
        Some((soft, _)) => soft.saturating_sub((soft / 8).max(64)),
        None => u64::MAX,
    })
}

/// One descriptor a guest's use holds of the process's pool, handed back when dropped: kept beside
/// the descriptor it pays for, so they close together.
#[derive(Debug)]
pub(crate) struct Descriptor(());

impl Descriptor {
    /// One of the pool's, or `-EMFILE` when it has none left.
    pub(crate) fn take() -> Result<Descriptor, i64> {
        if IN_USE.take(1, pool(), true) {
            Ok(Descriptor(()))
        } else {
            Err(EMFILE)
        }
    }
}

impl Drop for Descriptor {
    fn drop(&mut self) {
        IN_USE.give(1);
    }
}

/// The process's `RLIMIT_NOFILE`, `(soft, hard)`, where it has one. (`rlim_t` is `u64` on the hosts
/// CI builds, narrower on some others: hence the casts.)
#[cfg(unix)]
#[allow(clippy::unnecessary_cast)]
fn limits() -> Option<(u64, u64)> {
    let mut lim = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: `getrlimit` writes the limit into `lim`, a valid `rlimit`.
    (unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut lim) } == 0)
        .then_some((lim.rlim_cur as u64, lim.rlim_max as u64))
}

#[cfg(not(unix))]
fn limits() -> Option<(u64, u64)> {
    None
}

/// Raise this process's soft descriptor limit to its hard one, so its guests' pool is as large as
/// the host allows. The binaries call it as they start, before the pool reads the limit; a library
/// embedder chooses its own. Where the hard limit is unlimited, macOS refuses it as a soft limit,
/// so its per-process default (10240) is tried instead.
#[allow(clippy::unnecessary_cast)]
pub fn raise_descriptor_limit() {
    #[cfg(unix)]
    if let Some((soft, hard)) = limits() {
        for want in [hard, hard.min(10240)] {
            if want <= soft {
                return;
            }
            let lim = libc::rlimit {
                rlim_cur: want as libc::rlim_t,
                rlim_max: hard as libc::rlim_t,
            };
            // SAFETY: `setrlimit` reads `lim`, a valid `rlimit` that raises only the soft limit.
            if unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &lim) } == 0 {
                return;
            }
        }
    }
}
