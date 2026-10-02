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
    let esc: String = rec.encode().iter().map(|b| format!("\\x{b:02x}")).collect();
    format!("data {at} \"{esc}\"\n")
}
