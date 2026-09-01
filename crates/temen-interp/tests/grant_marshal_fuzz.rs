//! **Generative fuzz of the §14 op-13 grant-marshaling bounce** (#1025 slice 3a.2) — the confinement
//! hinge (INVARIANTS §2: the marshaling that carries authority across the emitted `env.instantiate_module`
//! bounce is "suspect by default"). It drives [`Host::spawn_named_child_from_window`] — which reads
//! `grants_n × 16-byte` records `{name_off, name_len, handle, flags}` out of a confined window slice and
//! re-grants each named handle from the parent powerbox — with adversarial windows, pointers, and counts.
//!
//! Two properties, both fail-closed:
//!   1. **Memory safety.** Over fully arbitrary `(window bytes, grants_ptr, grants_n)` the parse never
//!      panics and never reads out of the slice (an out-of-window record/name yields `Err`, never UB;
//!      ASan/miri in CI catch a stray read that doesn't panic). The result is deterministic.
//!   2. **Authority soundness + completeness.** For deliberately-laid records, the child is built **iff**
//!      every record is in-window, its name is UTF-8, *and* its handle is one the parent can actually
//!      re-grant. A forged handle (one the parent never granted) is refused — `can_regrant` holds — and a
//!      wholly-valid list is never falsely refused.
//!
//! The window carve / no-widening property is `check_child_carve`'s (tested in `nested_grant_marshal_op13`
//! + the escape tests); this file fuzzes the record parse + the authority gate that `spawn_named_child_from_window` owns.

use temen_interp::{ForkedProc, GrantMarshalError, Host, HostProc};

const CHILD_SIZE: u64 = 1 << 12;

/// Reproducible xorshift PRNG (same family as `concurrent_fuzz`/`irgen`).
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed ^ 0x9e37_79b9_7f4a_7c15 | 1)
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn range(&mut self, lo: u64, hi: u64) -> u64 {
        lo + self.next() % (hi - lo + 1)
    }
    fn chance(&mut self, n: u64) -> bool {
        self.next().is_multiple_of(n)
    }
}

/// A fresh parent powerbox holding exactly one re-grantable cap — a forkable `HOST_PROC` (the shape a
/// shared `fs` takes) — under the name `"fs"`. Returns the host and the granted handle value; **only** this
/// handle is re-grantable, so any other handle in a record is a forgery the marshal must refuse.
fn parent_host() -> (Host, i32) {
    let mut host = Host::new();
    let handler: HostProc = Box::new(|_op, _args, _mem, _| Ok(vec![0]));
    let fork = std::sync::Arc::new(|_pid: u64| {
        ForkedProc::shared(Box::new(|_op, _args, _mem, _| Ok(vec![0])))
    });
    let h = host.grant_host_proc_forkable(handler, fork);
    (host, h)
}

/// Re-parse the window with the *same* record layout the function uses, to independently decide whether the
/// marshal should succeed. The structural half (in-window + UTF-8) mirrors the parse; the authority half
/// (`handle == grantable`) is independent ground truth — the test knows the one grantable handle by
/// construction, so "should succeed" is a real oracle, not a tautology.
fn should_succeed(window: &[u8], grants_ptr: u64, grants_n: u64, grantable: i32) -> bool {
    let read = |off: u64, len: u64| -> Option<&[u8]> {
        let end = off.checked_add(len)?;
        window.get(usize::try_from(off).ok()?..usize::try_from(end).ok()?)
    };
    for i in 0..grants_n {
        let Some(rec_off) = i.checked_mul(16).and_then(|d| grants_ptr.checked_add(d)) else {
            return false;
        };
        let Some(rec) = read(rec_off, 16) else {
            return false;
        };
        let name_off = u32::from_le_bytes([rec[0], rec[1], rec[2], rec[3]]) as u64;
        let name_len = u32::from_le_bytes([rec[4], rec[5], rec[6], rec[7]]) as u64;
        let handle = i32::from_le_bytes([rec[8], rec[9], rec[10], rec[11]]);
        let Some(name) = read(name_off, name_len) else {
            return false;
        };
        if std::str::from_utf8(name).is_err() {
            return false;
        }
        if handle != grantable {
            return false;
        }
    }
    true
}

#[test]
fn fuzz_grant_marshal_memory_safety_and_determinism() {
    // Fully arbitrary windows/pointers/counts: the parse must never panic and must be deterministic. Ok is
    // acceptable whenever the random bytes happen to decode to a valid, grantable record (that IS correct).
    for seed in 0..40_000u64 {
        let mut r = Rng::new(seed);
        let len = r.range(0, 96) as usize;
        let window: Vec<u8> = (0..len).map(|_| r.next() as u8).collect();
        // Pointers span in-window, just-past, and wildly out (including near u64::MAX to hit the overflow
        // guards); counts include 0 and large values so `grants_n * 16` overflow is exercised.
        let grants_ptr = if r.chance(3) {
            r.next() // anywhere in u64
        } else {
            r.range(0, len as u64 + 24)
        };
        let grants_n = if r.chance(4) {
            r.next() // anywhere in u64 — exercises the `grants_n * 16` overflow guard
        } else {
            r.range(0, 6)
        };

        let (mut host1, _) = parent_host();
        let (mut host2, _) = parent_host();
        let a = host1.spawn_named_child_from_window(&window, grants_ptr, grants_n, CHILD_SIZE);
        let b = host2.spawn_named_child_from_window(&window, grants_ptr, grants_n, CHILD_SIZE);
        assert_eq!(
            a.is_ok(),
            b.is_ok(),
            "marshal is deterministic (seed {seed}, ptr {grants_ptr}, n {grants_n})"
        );
    }
}

#[test]
fn fuzz_grant_marshal_authority_oracle() {
    // Deliberately-laid records with handles drawn from {grantable, forged} and names from {utf-8, non-
    // utf-8, out-of-window}: the marshal must succeed IFF the independent oracle says every record is
    // in-window, UTF-8, and grantable. This pins `can_regrant`-holds and no-false-refusal.
    for seed in 0..40_000u64 {
        let mut r = Rng::new(seed);
        let (mut host, grantable) = parent_host();

        let n = r.range(0, 5);
        // A window big enough to hold `n` records + a small name pool, sometimes deliberately truncated so
        // records/names fall out of window.
        let name_pool = 24u64;
        let full = n * 16 + name_pool;
        let len = if r.chance(4) {
            r.range(0, full) // truncated → some records/names out of window
        } else {
            full + r.range(0, 16)
        } as usize;
        let mut window = vec![0u8; len];
        let names_base = n * 16; // names live after the records

        for i in 0..n {
            let rec_off = (i * 16) as usize;
            // Handle: grantable, a clearly-forged large/negative value, or (rarely) 0.
            let handle: i32 = match r.range(0, 3) {
                0 => grantable,
                1 => 10_000 + r.range(0, 5_000) as i32,
                2 => -(1 + r.range(0, 5_000) as i32),
                _ => 0,
            };
            // Name: point into the pool with a small len (utf-8 ascii), a non-utf-8 byte, or out of window.
            let (name_off, name_len): (u64, u64) = match r.range(0, 3) {
                0 => (names_base + (i % 4) * 4, 2), // in-pool ascii
                1 => (names_base + (i % 4) * 4, 2), // in-pool but we'll poison a byte below
                _ => (len as u64 + 8, 2),           // out of window
            };
            if rec_off + 16 <= window.len() {
                window[rec_off..rec_off + 4].copy_from_slice(&(name_off as u32).to_le_bytes());
                window[rec_off + 4..rec_off + 8].copy_from_slice(&(name_len as u32).to_le_bytes());
                window[rec_off + 8..rec_off + 12].copy_from_slice(&handle.to_le_bytes());
                // flags left 0
            }
            // Lay the name bytes when in-window; poison one to non-utf-8 on the mode-1 branch.
            let no = name_off as usize;
            if no + name_len as usize <= window.len() {
                for k in 0..name_len as usize {
                    window[no + k] = b'x';
                }
                if r.chance(4) {
                    window[no] = 0xFF; // invalid utf-8 lead byte
                }
            }
        }

        let expect = should_succeed(&window, 0, n, grantable);
        let got = host.spawn_named_child_from_window(&window, 0, n, CHILD_SIZE);
        assert_eq!(
            got.is_ok(),
            expect,
            "authority oracle mismatch (seed {seed}, n {n}): got {:?}, expected ok={expect}",
            got.as_ref().map(|_| ()).map_err(|e| *e)
        );
        // A refusal must be one of the fail-closed kinds — never a silent partial child.
        if let Err(e) = got {
            assert!(matches!(
                e,
                GrantMarshalError::OutOfWindow
                    | GrantMarshalError::BadName
                    | GrantMarshalError::NotRegrantable
            ));
        }
    }
}
