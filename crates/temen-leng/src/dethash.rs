//! A **deterministic, seedless** hasher for temen-leng's maps.
//!
//! `std`'s default `HashMap`/`HashSet` seed their hasher from `RandomState`, which pulls OS entropy
//! (`getrandom`, a syscall). That is (a) a source of nondeterministic iteration order and (b)
//! **untranslatable** when temen-leng is compiled to run *as a guest on temen* (W5, NIM.md §3e): the
//! on-ramp synthesizes the allocator but cannot supply `getrandom`. FNV-1a is seedless and needs no
//! entropy. temen-leng's emitted output is already iteration-order-independent — its tests pass under
//! `std`'s *random* seed — so pinning the seed changes nothing observable but determinism and
//! guest-buildability. Same `HashMap`/`HashSet` types (still `std::collections`), only the hasher.
use core::hash::{BuildHasherDefault, Hasher};

/// FNV-1a over the hashed bytes (seedless).
#[derive(Default)]
pub(crate) struct FnvHasher(u64);

impl Hasher for FnvHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        // Start from the FNV offset basis on the first byte (state 0 → basis), then fold.
        let mut h = if self.0 == 0 {
            0xcbf2_9ce4_8422_2325
        } else {
            self.0
        };
        for &b in bytes {
            h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
        }
        self.0 = h;
    }
}

pub(crate) type HashMap<K, V> = std::collections::HashMap<K, V, BuildHasherDefault<FnvHasher>>;
pub(crate) type HashSet<T> = std::collections::HashSet<T, BuildHasherDefault<FnvHasher>>;
