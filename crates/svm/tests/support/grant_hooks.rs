//! The §14 grant-child hook table that wires svm-run's production capability plumbing into the JIT
//! for the granted-spawn / applet / fork integration suites (#923). Byte-identical copies previously
//! lived in thirteen test files.

use svm_jit::GrantChildHooks;

/// The full production [`GrantChildHooks`] table — svm-run's child build / bind / release / mint /
/// thunk / serve entry points — as the granted-spawn tests install it on the JIT.
pub fn grant_hooks() -> GrantChildHooks {
    GrantChildHooks {
        build: svm_run::grant_child_build,
        build_named: svm_run::grant_named_child_build,
        bind_imports: svm_run::child_bind_imports,
        release: svm_run::grant_child_release,
        mint: svm_run::child_offer_mint,
        thunk: svm_run::cap_thunk_locked,
        register_serve: svm_run::child_register_serve,
    }
}
