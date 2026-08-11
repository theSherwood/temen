//! Browser fork story (FORK.md §9 / OPS_PARITY.md `clone_caller`/`reap`): the `run_fork` entry runs
//! the `SRC_TWIN` fork topology on the bytecode cooperative `drive` — the exact path this crate runs
//! every multi-domain guest on, and the one the browser build uses. Fork rides portable primitives
//! (`Mem::fork_private` = `with_reservation` + `seed`; `Vm::clone`; `Host::fork_powerbox` — no native
//! APIs), so it works on wasm32 by construction; the wasm-JIT tier-up is orthogonal (cap/serve/fork
//! ops leaf-fold to the interp). This host-target test pins the entry's logic; the `#[no_mangle]`
//! export makes it callable from the wasm harness (`wasmtime --invoke run_fork`) too.

#[test]
fn run_fork_forks_returns_twice_through_the_bytecode_drive() {
    // `100` = the original resumed past `fork()` with reply_orig AND both replies (100 + 200) reached
    // the shared stdout — i.e. the twin genuinely ran over its private window + duplicated powerbox.
    assert_eq!(
        svm_browser::run_fork(),
        100,
        "the fork twin ran and returned twice on the browser's bytecode drive"
    );
}
