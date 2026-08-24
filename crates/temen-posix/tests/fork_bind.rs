//! FORK.md PR 5 — the `fork` name-binding half of the personality endpoint. A guest module that
//! imports `fork` binds it to a **live fork offer** the personality-provider/parent wired (whose
//! handler runs pid-mode `clone_caller`), not to the shared libc `HOST_PROC` handle; with no offer
//! supplied the bind fails closed. The provider-side pid-mode contract itself is pinned in
//! `temen-interp/tests/clone_caller.rs` (`pid_mode_replies_the_twins_task_id_...`).

use temen_interp::Host;
use temen_posix::{bind, bind_with_fork};

fn module_with_fork_import() -> temen_ir::Module {
    temen_text::parse_module(
        "memory 16\n\
         import 0 \"write\" (i64, i64, i64) -> (i64)\n\
         import 1 \"fork\" () -> (i64)\n\
         func () -> (i64) {\n\
         block 0 () {\n\
           vr = call.import 1 ()\n\
           return vr\n\
           }\n\
         }\n",
    )
    .expect("parses")
}

#[test]
fn a_fork_import_binds_only_when_a_fork_offer_is_supplied() {
    let m = module_with_fork_import();
    let mut h = Host::new();
    let libc = h.grant_clock(); // stands in for the shared libc handle; bind only records it
    assert!(
        !bind(&m, &mut h, libc),
        "plain bind fails closed on a `fork` import — there is no offer to route it to"
    );
    assert!(
        !bind_with_fork(&m, &mut h, libc, None),
        "explicit None fails the same way"
    );
    // With a wired offer (type_id + handle as the parent supplies them), the bind succeeds; `fork`
    // routes to the offer, every other libc name to the shared handle.
    assert!(
        bind_with_fork(&m, &mut h, libc, Some((268435456, 7))),
        "a supplied fork offer binds the `fork` import"
    );
}

#[test]
fn modules_without_a_fork_import_bind_exactly_as_before() {
    let m = temen_text::parse_module(
        "memory 16\n\
         import 0 \"write\" (i64, i64, i64) -> (i64)\n\
         func () -> (i64) {\n\
         block 0 () {\n\
           vz = i64.const 0\n\
           return vz\n\
           }\n\
         }\n",
    )
    .expect("parses");
    let mut h = Host::new();
    let libc = h.grant_clock();
    assert!(bind(&m, &mut h, libc), "no fork import — plain bind works");
}
