//! A cooperative run over the window the browser's coop driver gives a tier-up run: a flat buffer
//! of the module's declared size that grows in place (#1312), under the oracle's reservation.
//! Emitted code addresses its window as one span, so a tier-up over a window without one runs
//! interpreted (#1414 B6-3a). `CoopRun::new`'s own window has one only where it is an `mmap`
//! (unix); elsewhere it is the sparse table, and a test of tier-up would interpret there.
//! Include with `#[path = "support/coop.rs"] mod coop;`.

use std::sync::Arc;

use temen_interp::bytecode::{CoopRun, TierUpConfig};
use temen_interp::{Host, Region, Trap, Value};
use temen_ir::Module;

/// [`CoopRun::new`], over the coop driver's flat window. A module with no memory has no window.
pub fn coop_run(
    m: &Module,
    entry: u32,
    args: &[Value],
    fuel: u64,
    host: Host,
    tierup: Option<TierUpConfig>,
) -> Option<Result<CoopRun, Trap>> {
    let Some(mc) = m.memory else {
        return CoopRun::new(m, entry, args, fuel, host, tierup);
    };
    let back = Region::growable(1 << mc.size_log2, temen_interp::host_page_size())
        .expect("a flat window of the declared size");
    CoopRun::new_over(
        m,
        entry,
        args,
        fuel,
        host,
        tierup,
        &[],
        temen_ir::DEFAULT_RESERVED_LOG2,
        Arc::new(back),
    )
}
