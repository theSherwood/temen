//! STAGE1.md browser-parity slice 2 — **concurrent stages on the bytecode engine**. The
//! cooperative single-thread driver (`bytecode::compile_and_run_with_host`, the browser's wasm-safe
//! entry: no OS threads, no wall clock) drives the concurrent ring pipeline that
//! `temen-interp/tests/concurrent_stages.rs` pins on the tree-walk oracle — the same program
//! (`support/pipeline.rs`), the same 410.
//!
//! Two detached stages move four items through a **one-slot bounded ring** in a granted
//! `SharedRegion` (flag at region byte 0, datum at byte 8). The SharedRegion `map`/`page_size` and
//! AddressSpace `create_region` ops ride the generic `call.cap` dispatch on the bytecode engine, and
//! the cooperative `drive` scheduler parks a task on `memory.wait` and wakes it on `notify`. This is
//! the shape sequential spawn/wait cannot run at all: with a 1-slot ring and 4 items the producer
//! MUST park mid-stream and be woken by the consumer (and vice versa) — run-to-completion order
//! deadlocks.
//!
//! It also pins the backing-identity futex key closed on the bytecode engine: each stage maps the
//! region in its OWN window, so wait/notify only rendezvous if the key is the backing identity. A
//! regression surfaces loudly, not as a hang (see `support/pipeline.rs`). Differential: the
//! tree-walk oracle and the bytecode engine agree on 410, and the bytecode driver must actually
//! drive it (return `Some`, not fall back to the oracle).

#[path = "../../temen-interp/tests/support/pipeline.rs"]
mod pipeline;

use temen_interp::{bytecode, run_with_host, Value};

#[test]
fn bytecode_drives_two_concurrent_stages_through_a_shared_region_ring() {
    let (parent, stages) = pipeline::modules();

    // Tree-walk oracle.
    let (mut h_tw, args) = pipeline::host(&stages);
    let mut f_tw = 50_000_000u64;
    let tw = run_with_host(&parent, 0, &args.map(Value::I32), &mut f_tw, &mut h_tw);
    assert_eq!(
        tw,
        Ok(vec![Value::I64(410)]),
        "oracle: producer published 4, consumer summed 10, zero timeouts"
    );

    // Bytecode cooperative single-thread driver (the browser's wasm-safe entry). Must actually drive
    // the detached spawns + the ring (return `Some`, not fall back to the tree-walk oracle).
    let (mut h_bc, args) = pipeline::host(&stages);
    let mut f_bc = 50_000_000u64;
    let bc = bytecode::compile_and_run_with_host(
        &parent,
        0,
        &args.map(Value::I32),
        &mut f_bc,
        &mut h_bc,
    )
    .expect(
        "bytecode engine must drive the concurrent pipeline (op 15 + region ops), not fall back",
    );
    assert_eq!(
        bc, tw,
        "the bytecode engine and the tree-walk oracle agree on 410 — a 1-slot ring across two live \
         detached domains, parking and waking on the backing-identity futex key"
    );
}
