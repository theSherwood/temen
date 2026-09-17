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

/// `iters` rewrites of one hot cell, then two distinct cells, then a 4 KiB `mem.fill`. The three
/// phases are the three cases the journal has to get right: repeated writes to one address (what
/// coalescing collapses), distinct addresses (what it cannot), and a bulk op (whose span comes from
/// `watch_accesses`, not a plain store width).
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

/// The full observable state of a run at a stop: window, position, and call depth. Comparing all
/// three is what makes the differential cover the continuation as well as memory.
fn observe(r: &ScheduledDebugRun) -> (Vec<u8>, Option<temen_interp::IrPc>, usize, u64) {
    (
        r.read_window(0, WIN_LEN).expect("readable"),
        r.frame_pc(0),
        r.depth(),
        r.op_turn(),
    )
}

/// A fresh run ticked to `t` — the oracle an undo must match, observed in full.
fn state_at(t: u64) -> (Vec<u8>, Option<temen_interp::IrPc>, usize, u64) {
    let mut r = run();
    let mut fuel = FUEL;
    while r.op_turn() < t && r.tick(&mut fuel) {}
    observe(&r)
}

/// **The headline differential**: undoing to any turn reproduces what a fresh run reaches by ticking
/// there — not just the window, but the position and call depth too, now that the continuation and
/// host cursor rewind with it. Walked backward from the end, so each undo starts from the state the
/// previous one left rather than from a fresh capture.
#[test]
fn undo_to_every_turn_matches_a_replayed_run() {
    let (mut r, end) = armed_run_to_end();
    assert!(
        end > 30,
        "the fixture should run a good number of turns, got {end}"
    );
    for t in (0..=end).rev() {
        assert!(r.can_undo_to(t), "turn {t} should be undoable");
        assert!(r.undo_to(t), "undo_to({t}) should succeed");
        assert_eq!(
            observe(&r),
            state_at(t),
            "undo_to({t}) must equal a fresh run ticked to {t}"
        );
    }
}

/// **Undo then step forward re-executes identically.** This is what the continuation rewind buys: after
/// undoing, the run is genuinely back at that turn, so driving forward again reproduces the same
/// subsequent states rather than diverging.
#[test]
fn undo_then_replay_forward_reproduces_the_run() {
    let (mut r, end) = armed_run_to_end();
    let mid = end / 2;
    assert!(r.undo_to(mid), "undo to the midpoint");
    let mut fuel = FUEL;
    while r.op_turn() < end && r.tick(&mut fuel) {}
    assert_eq!(
        observe(&r),
        state_at(end),
        "stepping forward after an undo lands where the original run did"
    );
}

/// **Undo declines rather than half-rewinds.** A turn the journal never recorded — past the end, or
/// before the armed window — must be refused, leaving the run untouched for `seek` to serve.
#[test]
fn undo_declines_a_turn_it_does_not_hold() {
    let (mut r, end) = armed_run_to_end();
    let before = observe(&r);
    assert!(!r.can_undo_to(end + 100));
    assert!(!r.undo_to(end + 100), "a turn past the end is refused");
    assert_eq!(observe(&r), before, "a refused undo changes nothing");
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
    assert!(r.undo_to(0));
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
    assert!(r.undo_to(0), "the segment start survives coalescing");
    assert_eq!(
        observe(&r),
        state_at(0),
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

// ---- cap crossings: the tape rides the journal (#1557) -------------------------------------------

/// A guest that calls a `HOST_PROC` twice and sums the answers. The capability is **nondeterministic**
/// on purpose — it answers with an incrementing counter — so re-invoking it would give a different
/// answer than the first pass. That is exactly what must not happen after an undo.
const CAP_CALLER: &str = r#"memory 16
func (i32) -> (i64) {
block 0 (vh: i32) {
  vz = i64.const 0
  va = call.cap 13 0 (i64) -> (i64) vh (vz)
  vb = call.cap 13 0 (i64) -> (i64) vh (vz)
  vk = i64.const 1000
  vm = i64.mul va vk
  vsum = i64.add vm vb
  return vsum
  }
}
"#;

/// A host whose capability answers 1, 2, 3, … on successive calls, with recording armed.
fn counting_host() -> (temen_interp::Host, i32) {
    use std::sync::{Arc, Mutex};
    let mut host = temen_interp::Host::new();
    host.record_caps();
    let n = Arc::new(Mutex::new(0i64));
    let h = host.grant_host_proc(Box::new(move |_op, _args, _mem, _minter| {
        let mut g = n.lock().unwrap();
        *g += 1;
        Ok(vec![*g])
    }));
    (host, h)
}

fn cap_module() -> temen_ir::Module {
    let m = temen_text::parse_module(CAP_CALLER).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// **The tape rides the journal.** Run a guest across two nondeterministic cap crossings, undo back to
/// before the first, and step forward again: the answers must be the *recorded* 1 and 2, not a fresh 3
/// and 4 from the live counter. Nothing here rewinds a cursor by hand — the journal's host cursor
/// carries `cap_consumed`, so the tape comes back in step by construction.
#[test]
fn undoing_across_cap_calls_re_serves_the_recorded_inputs() {
    let m = cap_module();
    let (host, h) = counting_host();
    let mut r = ScheduledDebugRun::new_with_host(&m, 0, &[temen_interp::Value::I32(h)], host)
        .expect("in the debug subset");
    r.set_journal_armed(true);
    let mut fuel = FUEL;
    while r.tick(&mut fuel) {}
    // First pass: the capability answered 1 then 2, so 1*1000 + 2.
    assert_eq!(
        r.result().cloned(),
        Some(Ok(vec![temen_interp::Value::I64(1002)])),
        "the live capability answers 1 then 2 on the first pass"
    );
    let end = r.op_turn();

    assert!(
        r.undo_to(0),
        "undo the whole run, back across both crossings"
    );
    let mut fuel = FUEL;
    while r.op_turn() < end && r.tick(&mut fuel) {}
    assert_eq!(
        r.result().cloned(),
        Some(Ok(vec![temen_interp::Value::I64(1002)])),
        "re-execution must re-serve the taped 1 and 2 — a live re-invoke would give 3 and 4"
    );
}

/// **Fail-closed on state a cursor cannot invert.** A capability with opaque declared state
/// (`set_cap_state_capture`) has no inverse, so the journal records nothing for those turns and undo
/// declines, leaving the checkpoint-plus-replay path to serve. Refusing beats rewinding wrongly.
#[test]
fn a_stateful_capability_makes_the_run_decline_to_undo() {
    use std::sync::{Arc, Mutex};
    let m = cap_module();
    let mut host = temen_interp::Host::new();
    host.record_caps();
    let n = Arc::new(Mutex::new(0i64));
    let cap = Arc::clone(&n);
    let h = host.grant_host_proc(Box::new(move |_op, _args, _mem, _minter| {
        let mut g = cap.lock().unwrap();
        *g += 1;
        Ok(vec![*g])
    }));
    // Declare the capability's own state: now it is opaque-with-state, outside the invertible subset.
    let get = Arc::clone(&n);
    host.set_cap_state_capture(
        h,
        Box::new(move || get.lock().unwrap().to_le_bytes().to_vec()),
    );

    let mut r = ScheduledDebugRun::new_with_host(&m, 0, &[temen_interp::Value::I32(h)], host)
        .expect("in the debug subset");
    r.set_journal_armed(true);
    let mut fuel = FUEL;
    while r.tick(&mut fuel) {}
    let end = r.op_turn();
    assert!(
        !r.can_undo_to(end / 2),
        "a run holding a stateful capability must decline to undo, not rewind it wrongly"
    );
    assert!(!r.undo_to(end / 2));
}

// ---- the static retention policy (#1558) ----------------------------------------------------------

/// **The fine window is what it says.** With a narrow `fine_turns`, recent turns stay undoable to the
/// exact turn while older ones have been coalesced into a segment, so they are reachable only at the
/// segment boundary. That is the granularity trade the policy exists to make.
#[test]
fn the_policy_keeps_recent_history_fine_and_ages_the_rest() {
    let mut r = run_with(DIFF_ITERS);
    r.set_journal_armed(true);
    r.set_journal_policy(temen_interp::journal::JournalPolicy {
        fine_turns: 8,
        byte_budget: 0,
    });
    let mut fuel = FUEL;
    while r.tick(&mut fuel) {}
    let end = r.op_turn();

    // A turn inside the fine window undoes exactly, and still agrees with a replayed run.
    let recent = end - 3;
    assert!(
        r.can_undo_to(recent),
        "a turn inside the fine window is undoable"
    );
    assert!(r.undo_to(recent));
    assert_eq!(
        observe(&r),
        state_at(recent),
        "an undo inside the fine window still matches the replay oracle"
    );
}

/// **The byte budget bounds the journal, and does so fail-closed.** Past the ceiling the oldest history
/// is dropped: the journal stays under budget, and the turns that went away simply decline to be
/// undone rather than coming back wrong.
#[test]
fn the_byte_budget_bounds_the_journal_and_fails_closed() {
    let mut r = run_with(REPORT_ITERS);
    r.set_journal_armed(true);
    let budget = 2048;
    r.set_journal_policy(temen_interp::journal::JournalPolicy {
        fine_turns: 16,
        byte_budget: budget,
    });
    let mut fuel = FUEL;
    while r.tick(&mut fuel) {}

    let s = r.journal_stats();
    assert!(
        s.bytes <= budget,
        "the journal must respect its byte budget: {} > {budget}",
        s.bytes
    );
    // Turn 0's history is long gone, so undo declines it — and leaves the run untouched.
    let before = observe(&r);
    assert!(
        !r.can_undo_to(0),
        "dropped history declines rather than lies"
    );
    assert!(!r.undo_to(0));
    assert_eq!(observe(&r), before, "a declined undo changes nothing");
}
