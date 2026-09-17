//! **#1557 — the undo journal, level 1, against the replay path as oracle.**
//!
//! Time travel today restores the nearest checkpoint and replays forward. The journal records a
//! pre-image of every window range an op overwrites, so stepping backward is *undo* instead. The
//! correctness contract is therefore a differential, and the existing replay path is the oracle:
//!
//! > `undo_to(t)` leaves the window byte-identical to a fresh run ticked to `t`.
//!
//! These pin that at every turn of a write-heavy program, plus the inertness rule (INVARIANTS #9b:
//! journaling armed must not perturb the run), the bulk-op span coverage that `watch_accesses` buys,
//! and the level-2 coalescing bound (#1558) that makes aging the journal safe.

use temen_interp::bytecode::ScheduledDebugRun;

/// A loop that rewrites the same 8-byte cell many times, then writes a spread of distinct cells, then
/// a bulk `mem.fill` over a wide span. The three phases are the three cases the journal has to get
/// right: repeated writes to one address (what coalescing collapses), distinct addresses (what it
/// cannot), and a bulk op (whose span comes from `watch_accesses`, not a plain store width).
/// `iters` rewrites of one hot cell, then two distinct cells, then a 4 KiB `mem.fill`.
fn writer_src(iters: i32) -> String {
    format!(
        r#"memory 17
func () -> (i64) {{
block 0 () {{
  v0 = i32.const {iters}
  br 1(v0)
}}
block 1 (vk: i32) {{
  vhot = i64.const 16384
  vkx = i64.extend_i32_u vk
  i64.store vhot vkx
  vm1 = i32.const -1
  vnext = i32.add vk vm1
  br_if vnext 1(vnext) 2()
}}
block 2 () {{
  vspread = i64.const 32768
  vsv = i64.const 7
  i64.store vspread vsv
  vspread2 = i64.const 40960
  i64.store vspread2 vsv
  vfilld = i64.const 49152
  vfillb = i32.const 171
  vfilll = i64.const 4096
  mem.fill vfilld vfillb vfilll
  vend = i64.const 16384
  vr = i64.load vend
  return vr
  }}
}}
"#
    )
}

/// The differential fixture stays small: its test walks every turn and replays to each, so the cost
/// is quadratic in the turn count.
const DIFF_ITERS: i32 = 24;
/// The measurement fixture loops enough for the hot-cell collapse to show in *bytes*, not just in
/// entry count.
const REPORT_ITERS: i32 = 2000;

fn module_with(iters: i32) -> temen_ir::Module {
    let m = temen_text::parse_module(&writer_src(iters)).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

const FUEL: u64 = 5_000_000;

/// The whole touched window range: the hot cell, the spread cells, and the fill span.
const WIN_LEN: usize = 64 << 10;

fn run_with(iters: i32) -> ScheduledDebugRun {
    ScheduledDebugRun::new(&module_with(iters), 0, &[]).expect("in the bytecode debug subset")
}

fn run() -> ScheduledDebugRun {
    run_with(DIFF_ITERS)
}

/// The window bytes of a fresh run ticked to `t` — the oracle an undo must match.
fn window_at(t: u64) -> Vec<u8> {
    let mut r = run();
    let mut fuel = FUEL;
    while r.op_turn() < t && r.tick(&mut fuel) {}
    r.read_window(0, WIN_LEN).expect("readable window")
}

/// Drive a journaling run to completion, returning it and the turn it finished at.
fn armed_run_to_end() -> (ScheduledDebugRun, u64) {
    armed_run_of(DIFF_ITERS)
}

fn armed_run_of(iters: i32) -> (ScheduledDebugRun, u64) {
    let mut r = run_with(iters);
    r.set_journal_armed(true);
    let mut fuel = FUEL;
    while r.tick(&mut fuel) {}
    let end = r.op_turn();
    (r, end)
}

/// **The headline differential**: undoing to any turn reproduces the window a fresh run reaches by
/// ticking there. Walked backward from the end, so each step exercises undo from the state the
/// previous undo left, not from a fresh capture.
#[test]
fn undo_to_every_turn_matches_a_replayed_run() {
    let (mut r, end) = armed_run_to_end();
    assert!(
        end > 30,
        "the fixture should run a good number of turns, got {end}"
    );
    for t in (0..=end).rev() {
        r.undo_to(t);
        assert_eq!(
            r.read_window(0, WIN_LEN).expect("readable"),
            window_at(t),
            "undo_to({t}) must equal a fresh run ticked to {t}"
        );
    }
}

/// **Inertness** (INVARIANTS #9b): arming the journal changes nothing the guest or the engine can
/// see — same result, same turn count, same final window.
#[test]
fn journaling_is_inert() {
    let mut plain = run();
    let mut fuel = FUEL;
    while plain.tick(&mut fuel) {}

    let (armed, armed_end) = armed_run_to_end();
    assert_eq!(plain.op_turn(), armed_end, "same turn count");
    assert_eq!(plain.result(), armed.result(), "same result");
    assert_eq!(
        plain.read_window(0, WIN_LEN).expect("readable"),
        armed.read_window(0, WIN_LEN).expect("readable"),
        "same final window"
    );
}

/// A disarmed journal records nothing, so a run that never arms one holds no history at all.
#[test]
fn a_disarmed_journal_records_nothing() {
    let mut r = run();
    let mut fuel = FUEL;
    while r.tick(&mut fuel) {}
    let s = r.journal_stats();
    assert_eq!((s.entries, s.bytes, s.appended), (0, 0, 0));
}

/// **Bulk spans are journaled whole.** The fixture's `mem.fill` covers 4 KiB that no plain store
/// touches, so undoing across it can only restore those bytes if the span came from
/// `watch_accesses` rather than a store width. Pinned by undoing to before the fill and finding the
/// filled range back at zero.
#[test]
fn a_bulk_fill_is_journaled_by_its_whole_span() {
    let (mut r, _end) = armed_run_to_end();
    let filled = r.read_window(49152, 4096).expect("readable");
    assert!(
        filled.iter().all(|&b| b == 0xab),
        "the fixture's fill should have run"
    );
    // Undo the whole run: the fill's 4 KiB must come back as the zeros it overwrote.
    r.undo_to(0);
    let restored = r.read_window(49152, 4096).expect("readable");
    assert!(
        restored.iter().all(|&b| b == 0),
        "the fill's full span must be restored, not just a store-width prefix"
    );
}

/// **The level-2 bound** (#1558): coalescing keeps one pre-image per address, so the hot loop's
/// repeated writes to one cell collapse, the compacted journal is far smaller than what was
/// appended, and it never exceeds the window. Undo to the segment start still lands exactly.
#[test]
fn coalescing_collapses_repeated_writes_and_stays_under_the_window() {
    let (mut r, end) = armed_run_to_end();
    let before = r.journal_stats();
    r.coalesce_journal(end + 1); // compact the whole history into one segment
    let after = r.journal_stats();

    assert!(
        after.bytes < before.bytes,
        "coalescing must shrink a run that rewrites the same cell: {before:?} -> {after:?}"
    );
    assert!(
        after.bytes <= WIN_LEN,
        "a compacted segment must not exceed the window: {} > {WIN_LEN}",
        after.bytes
    );
    // And the compacted segment still undoes to its start exactly.
    r.undo_to(0);
    assert_eq!(
        r.read_window(0, WIN_LEN).expect("readable"),
        window_at(0),
        "a coalesced segment still undoes to the segment start"
    );
}

/// **The measurement** #1557 and #1558 ask to be recorded: how much level 1 appends, what level 2
/// keeps, and the resulting coalescing ratio. Printed rather than thresholded — the numbers are the
/// deliverable, and the bound itself is asserted above.
#[test]
fn report_journal_volume() {
    let (mut r, end) = armed_run_of(REPORT_ITERS);
    let l1 = r.journal_stats();
    r.coalesce_journal(end + 1);
    let l2 = r.journal_stats();
    println!(
        "turns={end} level1: {} entries / {} bytes | level2: {} entries / {} bytes | \
         coalescing ratio {:.1}x | window {WIN_LEN} bytes",
        l1.entries,
        l1.bytes,
        l2.entries,
        l2.bytes,
        l1.bytes as f64 / l2.bytes.max(1) as f64
    );
    assert!(l2.bytes <= WIN_LEN, "the bound holds");
}
