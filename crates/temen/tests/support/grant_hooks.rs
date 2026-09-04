//! The §14 grant-child hook table that wires temen-run's production capability plumbing into the JIT
//! for the granted-spawn / applet / fork integration suites (#923). Byte-identical copies previously
//! lived in thirteen test files.

use temen_jit::GrantChildHooks;

/// The full production [`GrantChildHooks`] table — temen-run's child build / bind / release / mint /
/// thunk / serve entry points — as the granted-spawn tests install it on the JIT.
pub fn grant_hooks() -> GrantChildHooks {
    GrantChildHooks {
        build: temen_run::grant_child_build,
        build_named: temen_run::grant_named_child_build,
        build_detached: temen_run::grant_detached_child_build,
        minter_take: temen_run::minter_take,
        bind_imports: temen_run::child_bind_imports,
        release: temen_run::grant_child_release,
        mint: temen_run::child_offer_mint,
        thunk: temen_run::cap_thunk_locked,
        register_serve: temen_run::child_register_serve,
    }
}
