//! #2318 — a store a thaw rebuilds from its artifact is held against the process's pool of mappings:
//! past any ceiling, since the artifact's use was granted, but never past the pool. One the pool has
//! no room for is not rebuilt, and nothing of it stays charged. In its own binary: the pool is the
//! process's, and the test fills it.

use temen_fs::MemFsHandle;
use temen_interp::{maps_pool, Host, HostCharge, MAPPED_BUFFER};

#[test]
fn a_store_the_pool_has_no_room_for_is_not_rebuilt_and_holds_nothing() {
    let pool = maps_pool();
    if pool > 1 << 22 {
        eprintln!("skipped: a pool of {pool} mappings is too large to fill");
        return;
    }
    // One file of `MAPPED_BUFFER` bytes: a mapping of its own.
    let big = vec![7u8; MAPPED_BUFFER as usize];
    let state = MemFsHandle::seeded(&[("big".to_string(), big)], &[]).capture_state();
    let mut payer = Host::new();
    payer.set_grant(-1, -1, -1, -1);
    // The pool, filled to its last mapping.
    let mut held = Vec::new();
    loop {
        let mut charge = HostCharge::new(Some(payer.own_node()));
        if !charge.resize(MAPPED_BUFFER) {
            break;
        }
        held.push(charge);
    }
    let before = payer.channel_used();
    assert!(
        MemFsHandle::rebuild(&state, &payer.own_node()).is_none(),
        "a full pool refuses the store"
    );
    assert_eq!(
        payer.channel_used(),
        before,
        "and nothing of it stays charged"
    );
    held.pop();
    let before = payer.channel_used();
    let fs = MemFsHandle::rebuild(&state, &payer.own_node()).expect("room for its one mapping");
    assert!(payer.channel_used() > before, "rebuilt, it is charged");
    drop(fs);
    assert_eq!(payer.channel_used(), before, "and dropped, it is not");
}
