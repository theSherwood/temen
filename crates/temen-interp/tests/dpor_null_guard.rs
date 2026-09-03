//! **#1183 — the NULL guard in the exhaustive schedule explorers.** `run_one_schedule` (shared by
//! the DPOR `explore_all` and the brute-force `explore_all_bruteforce`) once rebuilt its `Mem` from
//! bare funcs/data *without* seeding the guard, so the model-checker explored an **unguarded** address
//! space — a NULL dereference returned `0` there while every real tier trapped it (#1094). That was a
//! latent oracle/parity gap: the explorer could accept a schedule the guarded tiers reject. Now the
//! harness seeds the unconditional guard too, so a NULL dereference is a `MemoryFault` in exploration
//! exactly as on the interpreter/JIT tiers.

use temen_interp::{explore_all, explore_all_bruteforce, Trap};

/// A single guest that dereferences NULL: load from address 0. Under the #1094 guard `[0, guard)` is
/// `Unmapped`, so this must trap `MemoryFault` — not read back `0`.
const NULL_LOAD: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  v0 = i64.const 0
  vl = i64.load v0
  return vl
  }
}
"#;

fn module() -> temen_ir::Module {
    let m = temen_text::parse_module(NULL_LOAD).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// Both explorers seed the guard, so a NULL deref is a `MemoryFault` in every explored schedule —
/// never a silent `0` read. (Single-threaded guest → one schedule, but this exercises the shared
/// `run_one_schedule` Mem build both the DPOR and brute-force paths use.)
#[test]
fn null_deref_faults_in_the_schedule_explorers() {
    let m = module();
    for (label, ex) in [
        ("dpor", explore_all(&m, 0, &[], 1_000_000, 1_000)),
        (
            "brute",
            explore_all_bruteforce(&m, 0, &[], 1_000_000, 1_000),
        ),
    ] {
        assert!(
            !ex.outcomes.is_empty(),
            "{label}: at least one schedule explored"
        );
        for outcome in &ex.outcomes {
            assert_eq!(
                outcome,
                &Err(Trap::MemoryFault),
                "{label}: a NULL deref must trap in exploration, got {outcome:?}"
            );
        }
    }
}
