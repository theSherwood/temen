//! The §14 grant-child hook table that wires temen-run's production capability plumbing into the JIT
//! for the granted-spawn / applet / fork integration suites (#923). Byte-identical copies previously
//! lived in thirteen test files.

use temen_interp::Host;
use temen_jit::GrantChildHooks;
use temen_run::CapCtx;

/// The production [`GrantChildHooks`] table for a run that baked `host` as its raw `call.cap` ctx —
/// i.e. one compiled with `temen_run::cap_thunk`, which is what every caller here does.
///
/// #1234: this used to hand-roll the table, which meant it also hand-picked which *shape* of parent
/// pointer the hooks would decode — a choice that has to agree with the ctx the run baked, and that
/// nothing checked. It now delegates to [`temen_run::production_grant_hooks`], which derives the
/// family and the pointer from one [`CapCtx`], so the tests exercise the same table production does
/// (invariant 15) and cannot pair it with the wrong ctx.
pub fn grant_hooks(host: *mut Host) -> GrantChildHooks {
    temen_run::production_grant_hooks(CapCtx::Raw(host))
}
