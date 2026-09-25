//! **Dead block-parameter pruning** (`temen_ir::prune_block_params`, #1831).
//!
//! Values are block-local, so a value is threaded through every block between its definition and a
//! use. The "locals as block parameters" frontends (temen-leng, chibicc's `codegen_ir`) thread every
//! local through every block regardless, and most of those parameters are dead where they sit. The
//! pass drops every parameter no path carries to a use — a backward fixpoint through the edges, since
//! a pass-through keeps itself referenced — and must change nothing a program or a debugger can see.

use temen_interp::Value;
use temen_ir::{DebugInfo, Module, SsaLoc, VarInfo, VarLoc};

fn run(m: &Module, n: i64) -> Result<Vec<Value>, temen_interp::Trap> {
    let mut fuel = 1_000_000u64;
    temen_interp::run(m, 0, &[Value::I64(n)], &mut fuel)
}

fn params(m: &Module) -> Vec<usize> {
    m.funcs[0].blocks.iter().map(|b| b.params.len()).collect()
}

// `sum(n)` = n + (n-1) + … + 1, in the φ model: every block carries every slot `(n, acc, junk)`,
// and `junk` — threaded header → body → header, and out to the exit — is never read.
const SUM: &str = "
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 0
  v2 = i64.const 777
  br 1(v0, v1, v2)
  }
block 1 (v0: i64, v1: i64, v2: i64) {
  v3 = i64.const 0
  v4 = i64.eq v0 v3
  br_if v4 3(v0, v1, v2) 2(v0, v1, v2)
  }
block 2 (v0: i64, v1: i64, v2: i64) {
  v3 = i64.add v1 v0
  v4 = i64.const -1
  v5 = i64.add v0 v4
  br 1(v5, v3, v2)
  }
block 3 (v0: i64, v1: i64, v2: i64) {
  return v1
  }
}
";

#[test]
fn a_value_threaded_around_a_loop_and_never_read_is_pruned() {
    let m = temen_text::parse_module(SUM).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let mut pruned = m.clone();
    temen_ir::prune_block_params(&mut pruned);
    temen_verify::verify_module(&pruned).expect("the pruned module verifies");
    // `junk` is gone everywhere; `n` is dead at the exit, where only `acc` is returned.
    assert_eq!(params(&pruned), vec![1, 2, 2, 1]);
    for n in [0, 1, 5, 30] {
        assert_eq!(run(&pruned, n), run(&m, n), "sum({n})");
    }
    assert_eq!(run(&pruned, 5), Ok(vec![Value::I64(15)]));
    // Pruning is idempotent.
    let mut again = pruned.clone();
    temen_ir::prune_block_params(&mut again);
    assert_eq!(again, pruned);
}

/// A debugger reads the same variable after pruning as before. The frontend named each slot by one
/// index in every block (`VarLoc::Ssa`); after pruning a slot sits at a different index in different
/// blocks, or nowhere, so it becomes a per-block list. A location on an instruction's result follows
/// the result down past the dropped parameters.
#[test]
fn value_keyed_debug_locations_follow_their_values() {
    let mut m = temen_text::parse_module(SUM).expect("parse");
    let var = |name: &str, loc| VarInfo {
        func: 0,
        name: name.to_string(),
        ty: "long".to_string(),
        loc,
        type_id: None,
        scope: None,
    };
    m.debug_info = Some(DebugInfo {
        vars: vec![
            var("acc", VarLoc::Ssa { value: 1 }),
            var("junk", VarLoc::Ssa { value: 2 }),
            var(
                "next",
                VarLoc::SsaList(vec![SsaLoc {
                    block: 2,
                    inst: 1,
                    value: 3,
                }]),
            ),
        ],
        ..Default::default()
    });
    temen_ir::prune_block_params(&mut m);
    let at = |block, value| SsaLoc {
        block,
        inst: 0,
        value,
    };
    let locs: Vec<VarLoc> = m
        .debug_info
        .unwrap()
        .vars
        .into_iter()
        .map(|v| v.loc)
        .collect();
    assert_eq!(
        locs,
        vec![
            // `acc`: the entry's constant, slot 1 in the loop, slot 0 at the exit (`n` dropped there).
            VarLoc::SsaList(vec![at(0, 1), at(1, 1), at(2, 1), at(3, 0)]),
            // `junk`: only where it is the entry's constant; dead everywhere else.
            VarLoc::SsaList(vec![at(0, 2)]),
            // `acc + n` in the body, one index down past the dropped `junk`.
            VarLoc::SsaList(vec![SsaLoc {
                block: 2,
                inst: 1,
                value: 2,
            }]),
        ]
    );
}
