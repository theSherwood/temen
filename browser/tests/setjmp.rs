//! #1062 regression on the **browser's own public path**. bash's error model is `setjmp`/`longjmp`,
//! and `COPY_PROCENV` memcpies the `jmp_buf` (`top_level`) on every `bash -c` — so the checkpoint
//! identity must ride in the buffer bytes, not its address. The tree-walk interp and the bytecode
//! engine both got the token-keyed fix; this pins it through `onramp_posix_exec`, the exact
//! POSIX-personality bytecode entry the playground runs, so a copied-buffer `longjmp` resolves to its
//! original checkpoint instead of trapping (which busy-looped `bash -c 'exit N'` before the fix).
//!
//! The committed fixture `fixtures/setjmp_copy.temen` is the same C witness the in-crate assertion
//! `crates/temen/tests/c_frontend.rs::c_longjmp_through_a_copied_jmp_buf` runs; regenerate it with
//! `cargo test -p temen --test c_frontend -- --ignored --exact gen_browser_setjmp_fixture`.

use temen_browser::{onramp_posix_exec, STATUS_OK};

#[test]
fn longjmp_through_a_copied_jmp_buf_on_the_browser_engine() {
    let bytes = include_bytes!("fixtures/setjmp_copy.temen");
    let m = temen_encode::decode_module(bytes).expect("decode setjmp_copy.temen");
    let out = onramp_posix_exec(&m, &[]);
    assert_eq!(
        out.status, STATUS_OK,
        "the copied-buffer longjmp must resolve, not trap (a trap here is the #1062 busy-loop root)",
    );
    // `main` returns 55: `setjmp(a)` → 0, memcpy `a`→`b`, `longjmp(b, 55)` re-enters and returns 55.
    assert_eq!(
        out.value, 55,
        "the guest returned the longjmp value through the copied buffer"
    );
}
