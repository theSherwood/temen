//! THREADS.md 4c-domain — **Miri** verification of §14 children under the **parallel** driver: each
//! of [`miri_spawn`]'s two detached children runs on a thread of its own over a fresh window, with
//! its region pre-mapped in, and the parent reads their writes through its own mappings after the
//! join. This proves that machinery is **free of data races / UB / provenance errors**. Miri's
//! checker, not the iteration count, is the point, so the kernel (2 children) is small. The
//! differential pins of the parallel driver's spawns are `spawn_every_driver.rs`.
//!
//! Run: `cargo +nightly miri test -p temen-interp --test parallel_instantiate_miri`

#[path = "support/miri_spawn.rs"]
mod miri_spawn;
#[path = "support/rec.rs"]
mod rec;

use std::sync::Arc;
use temen_interp::{bytecode, Region, Value};

#[test]
fn parallel_instantiate_race_free_under_miri() {
    let m = miri_spawn::root();
    let (mut host, args) = miri_spawn::host();
    let back = Arc::new(Region::new(1 << 18, 4096));
    let mut f = 50_000_000u64;
    let r = bytecode::compile_and_run_capture_over_parallel_with_host(
        &m,
        0,
        &args,
        &mut f,
        &[],
        Some(back),
        &mut host,
    )
    .expect("the parallel driver runs the spawns")
    .0;
    assert_eq!(r, Ok(vec![Value::I64(miri_spawn::RESULT)]));
}
