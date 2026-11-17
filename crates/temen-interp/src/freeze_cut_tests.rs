//! #3034 R2.3 — the rule for whether a tree-walker run carries the cut its vCPUs unwound into.

use super::*;

/// #3034 R2.3 — a run carries the cut its vCPUs unwound into only when its root unwound and the run
/// did not fail: a root that returned while a thread unwound keeps its result, and nothing rides. No
/// deterministic guest reaches that on the tree-walker (the root cannot return without a poll after
/// the trigger), so the rule is shown directly.
#[test]
fn a_root_that_returned_carries_no_cut() {
    assert!(carries_cut(true, &Ok(vec![Value::I64(0)])));
    assert!(
        !carries_cut(false, &Ok(vec![Value::I64(3)])),
        "the root returned"
    );
    assert!(
        !carries_cut(true, &Err(Trap::Unreachable)),
        "the run failed"
    );
}
