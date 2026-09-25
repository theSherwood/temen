//! #1735 — `TrapKind` and the interpreters' `Trap` are one trap wire code (`temen_ir::trap_code`):
//! every JIT trap kind is a `Trap`, and every `Trap` but `Exit` (the trap cell's `EXIT_CODE`) is a
//! JIT trap kind. Hosts convert with `Trap::from_code(kind.code())` / `TrapKind::from_code(..)`
//! instead of hand-written tables, so the two must stay the same set.

use temen_interp::Trap;
use temen_jit::{TrapKind, EXIT_CODE, HOST_UNWIND_CODE};

#[test]
fn every_trap_kind_is_a_trap_and_every_trap_but_exit_is_a_kind() {
    let mut kinds = 0;
    for c in 0..64u32 {
        match TrapKind::from_code(c) {
            Some(k) => {
                kinds += 1;
                assert_eq!(k.code(), c as i64, "code {c} round-trips");
                let t = Trap::from_code(k.code()).expect("a trap kind is a trap");
                assert_eq!(t.code(), k.code());
            }
            None => {
                let t = Trap::from_code(c as i64);
                assert!(
                    t.is_none() || c == EXIT_CODE,
                    "code {c} is the trap {t:?} but no JIT trap kind"
                );
            }
        }
    }
    assert_eq!(kinds, 12);
    assert!(TrapKind::from_code(HOST_UNWIND_CODE).is_none());
    assert!(Trap::from_code(HOST_UNWIND_CODE as i64).is_none());
}
