//! STAGE1.md item 6 — the **JIT pipeline**: two children running CONCURRENTLY, piped through a
//! granted `SharedRegion` + canonical-key futex — the fast-backend twin of
//! `temen-interp/tests/concurrent_stages.rs`, running the same program (`support/pipeline.rs`). The
//! parent mints a region (the run's backing factory is `temen_run::new_shared_region`, so it is a
//! real OS shared-memory object), spawns producer and consumer detached (op 15, each granted the
//! region by name), and joins both.
//!
//! What this pins, JIT-specifically:
//! - **detached children are async** (D66): each runs as a task in its own window. With a 1-slot
//!   ring and 4 items, run-to-completion order deadlocks — the producer MUST park mid-stream and be
//!   woken by the consumer, so a synchronous spawn cannot pass this at all.
//! - **real aliasing into separate child windows**: each child `map`s the region into its OWN
//!   window (`MprotectWindow::map_region` — `mmap(MAP_SHARED|MAP_FIXED)` of the region's memfd on
//!   unix, placeholder + `MapViewOfFile3` on windows), so parent-minted bytes are the same physical
//!   pages in both children. They map at window offset 65536 (#1094: above the NULL guard AND
//!   aligned to the largest real granule — the Windows 64 KiB allocation granularity
//!   `MapViewOfFile3` requires for the placement address).
//! - **canonical futex keys across windows**: each child's first `call.cap` installs the region-canon
//!   hook over its own `mem_base`, so `atomic.wait`/`notify` in different windows key on the backing
//!   identity `(os_fd, offset)` and rendezvous. With per-window keys every wake misses — and the
//!   regression surfaces loudly, not as a hang (see `support/pipeline.rs`).

#[path = "../../temen-interp/tests/support/pipeline.rs"]
mod pipeline;

use temen_interp::{run_with_host, MemLayout, Value};
use temen_ir::DEFAULT_RESERVED_LOG2;
use temen_jit::JitOutcome;

#[test]
fn two_concurrent_jit_children_pipe_through_a_shared_region_ring() {
    // Nesting requires the child runner (`fiber_rt`); where unsupported the JIT declines child
    // spawns, so there is nothing to pin — the interpreter remains the only backend there.
    if !temen_jit::fiber_supported() {
        return;
    }
    let (parent, stages) = pipeline::modules();

    let (mut host, args) = pipeline::host(&stages);
    let mut fuel = 50_000_000u64;
    let ir = run_with_host(&parent, 0, &args.map(Value::I32), &mut fuel, &mut host)
        .expect("interp: no trap, no hang");
    assert_eq!(ir, vec![Value::I64(410)], "interp reference");

    let (mut host, args) = pipeline::host(&stages);
    // Regions minted by this run are real OS shared-memory objects (memfd / section), so the JIT
    // children can `map` them for hardware aliasing.
    host.set_region_factory(temen_run::new_shared_region);
    let (jo, _) = temen_run::jit_cap_run(
        &parent,
        0,
        &args.map(i64::from),
        &MemLayout::image(Vec::new()),
        DEFAULT_RESERVED_LOG2,
        0,
        &mut host,
        None,
    )
    .expect("jit");
    assert!(
        matches!(jo, JitOutcome::Returned(ref s) if s == &[410]),
        "jit: producer published 4 (park while full), consumer summed 10 (park while empty), \
         zero timeouts — a 1-slot ring across two child tasks, aliased into both windows; got {jo:?}"
    );
}
