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

/// #3034 round 5 — owner decision 25, default-deny: what the harvest does with a domain a trap of its
/// own ended in the cut ([`own_trap_fate`]), one line per shape. Only the proven shape completes.
#[test]
fn an_own_trap_completes_only_the_proven_shape() {
    use OwnTrapFate::*;
    let ok = Ok(vec![Value::I64(7)]);
    let err = Err(Trap::Unreachable);
    // (twin, withheld, has children, root unwound, stored result) → fate
    let table: [(
        (bool, bool, bool, bool, Option<&Result<Vec<Value>, Trap>>),
        OwnTrapFate,
        &str,
    ); 9] = [
        (
            (false, false, false, true, Some(&ok)),
            Trapped,
            "its root unwound, then its thread trapped",
        ),
        (
            (false, false, false, false, Some(&err)),
            Trapped,
            "its root ended with the trap",
        ),
        (
            (false, false, false, true, None),
            Trapped,
            "its root unwound, its outcome taken",
        ),
        (
            (false, false, false, false, Some(&ok)),
            Returned,
            "its root returned before the trap",
        ),
        (
            (true, false, false, true, Some(&ok)),
            Refused,
            "a fork twin",
        ),
        (
            (false, true, false, true, Some(&ok)),
            Refused,
            "withheld (demand-paged)",
        ),
        (
            (false, false, true, true, Some(&ok)),
            Refused,
            "with detached children of its own",
        ),
        (
            (false, false, true, false, Some(&ok)),
            Refused,
            "returned, with children of its own",
        ),
        (
            (false, false, false, false, None),
            Refused,
            "no outcome and no unwind",
        ),
    ];
    for ((twin, withheld, kids, unwound, stored), want, what) in table {
        assert_eq!(
            own_trap_fate(twin, withheld, kids, unwound, stored),
            want,
            "{what}"
        );
    }
}
