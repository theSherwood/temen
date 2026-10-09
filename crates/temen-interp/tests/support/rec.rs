//! #1867 — a v1 (detached) spawn record in a test module. [`segment`] renders one as a text `data`
//! segment, which a module may carry after its funcs; a spawner that learns its module or `Budget`
//! handle at run time stores it at [`MODULE_AT`] or [`BUDGET_AT`] before the spawn. Include with
//! `#[path = "support/rec.rs"] mod rec;`.
//!
//! Place records above the durable control words, `[guard, guard + 32)`: the record's version word
//! is `1`, which at the guard base reads as a freeze in flight, even in a non-durable run (#2101).
//! These tests use 17408 and up.
#![allow(dead_code)] // each test binary uses a different subset

use temen_ir::SpawnRec;

/// Where a record's `module` handle sits, from the record's start.
pub const MODULE_AT: u64 = 24;
/// Where a record's `Budget` handle sits, from the record's start.
pub const BUDGET_AT: u64 = 28;

/// A text `data` segment holding `rec` at `at`.
pub fn segment(at: u64, rec: &SpawnRec) -> String {
    format!("data {at} \"{}\"\n", escape(&rec.encode()))
}

/// Text `data` segments for an **empty grant** (#2219): the name `name` at `name_at`, and at `at` the
/// 16-byte grant record naming it with the `GRANT_EMPTY` handle. The child's import `name` binds
/// empty: a child image keeps every import of its program, and binds them strictly, so a spawn
/// leaves out on purpose one its child never calls.
pub fn empty_grant(at: u64, name_at: u64, name: &str) -> String {
    let rec = [
        name_at as u32,
        name.len() as u32,
        temen_interp::GRANT_EMPTY,
        0,
    ]
    .map(u32::to_le_bytes)
    .concat();
    format!(
        "data {name_at} \"{name}\"\ndata {at} \"{}\"\n",
        escape(&rec)
    )
}

fn escape(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("\\x{b:02x}")).collect()
}
