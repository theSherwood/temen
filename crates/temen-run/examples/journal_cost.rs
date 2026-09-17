//! **The undo journal's bail criteria, measured on real guests** (#1556 / #1557 / #1558).
//!
//! #1556 agreed the criteria up front, because the replay path stays as both oracle and fallback
//! throughout: back out if journal volume exceeds an agreed multiple of the window after compaction,
//! or if debug-engine slowdown exceeds an agreed factor. Until this existed the only numbers were
//! synthetic — `undo_journal.rs::report_journal_volume` hammers one cell 2000 times, which is the
//! coalescer's best case by construction and says nothing about a program that writes *widely*.
//!
//! What it reports, per guest:
//!
//! - **Inertness at scale** (INVARIANTS #9b). The armed and unarmed runs must agree on the result,
//!   the turn count and the final window hash. A disagreement is a hard failure, not a number —
//!   journaling that perturbed the run would invalidate everything below it.
//! - **Slowdown**: armed wall time over unarmed. The second bail criterion.
//! - **Volume**: level-1 bytes held, level-2 bytes after coalescing the whole history, each as a
//!   fraction of the window. The first bail criterion, and the bound the whole design rests on — a
//!   compacted segment must never exceed the window, or it is worse than the snapshot it replaces.
//! - **Coalescing ratio**: bytes ever appended over bytes retained after compaction. How much the
//!   level-2 rule actually buys on this program's write pattern.
//! - **Undo vs replay**: the cost of `undo_to(t)` against rebuilding and re-driving to `t`, which is
//!   what `seek` does today. This is the number that says whether journal-backed `step_back` is worth
//!   wiring at all.
//!
//! ```text
//! cargo run --release -p temen-run --example journal_cost
//! cargo run --release -p temen-run --example journal_cost -- gradient        # one guest
//! TEMEN_JOURNAL_TURNS=200000 cargo run --release -p temen-run --example journal_cost
//! ```
//!
//! Not a test: it is a cost study over committed playground assets, it takes minutes, and its output
//! is a table for a human. The correctness of undo is pinned by `undo_journal.rs`, whose oracle is the
//! replay path.

use std::time::{Duration, Instant};

use temen_interp::bytecode::{SchedBreak, SchedStop, ScheduledDebugRun};
use temen_interp::journal::JournalPolicy;
use temen_run::{Limits, RunConfig};

/// A guest to measure, and how to reach it.
struct Guest {
    name: &'static str,
    asset: &'static str,
    /// Why it is in the set — printed with the row, so the table says what each number is evidence of.
    why: &'static str,
    /// Files to seed an `fs` capability with, as `(guest path, host path)`. Empty ⇒ no fs cap.
    fs: &'static [(&'static str, &'static str)],
    /// argv, if the guest reads any.
    args: &'static [&'static str],
}

const GUESTS: &[Guest] = &[
    Guest {
        name: "gradient",
        asset: "browser/web/assets/gradient.temen",
        why: "bulk framebuffer writes — the bulk-write-heavy program #1556 names",
        fs: &[],
        args: &[],
    },
    Guest {
        name: "mandelzoom",
        asset: "browser/web/assets/mandelzoom.temen",
        why: "compute-heavy with a framebuffer — writes narrowly, computes widely",
        fs: &[],
        args: &[],
    },
    Guest {
        name: "forth",
        asset: "browser/web/assets/forth.temen",
        why: "a real interpreter guest — scattered writes over a dictionary + stacks",
        fs: &[],
        args: &[],
    },
    Guest {
        name: "chibicc",
        asset: "browser/web/assets/chibicc.temen",
        why: "the real compiler #1556 names — arena allocation, wide and scattered",
        fs: &[(
            "in.c",
            "crates/temen-run/demos/chibicc_selfhost/corpus/hash.c",
        )],
        args: &["chibicc", "-g0", "/in.c"],
    },
];

/// Window prefix hashed for the inertness comparison.
const HASH_PREFIX: usize = 256 * 1024;

/// What one run produced — the identity triple the inertness check compares.
struct RunOut {
    turns: u64,
    stop: String,
    window_hash: u64,
    elapsed: Duration,
}

/// FNV-1a over the window prefix. Cheap, and all the inertness check needs is "did these diverge".
/// A prefix rather than the whole window because these are 256 KiB–2 MiB windows hashed twice per
/// guest, and every guest here writes its live data low.
fn window_hash(run: &ScheduledDebugRun, len: usize) -> u64 {
    let Ok(bytes) = run.read_window(0, len) else {
        return 0;
    };
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in &bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

fn drive(run: &mut ScheduledDebugRun, turns: u64) -> (u64, String) {
    let mut fuel = u64::MAX;
    let mut stop = String::from("TurnLimit");
    while run.op_turn() < turns {
        match run.step(&mut fuel) {
            // An ordinary single step lands on its own target; anything else ends the drive.
            SchedStop::Break {
                reason: SchedBreak::Step,
                ..
            } => {}
            other => {
                stop = format!("{other:?}");
                break;
            }
        }
    }
    (run.op_turn(), stop)
}

/// `TEMEN_JOURNAL_STRIDE` overrides [`JournalPolicy::state_stride`] for the run — the dial the
/// continuation cost turned out to hang on. `0` means one continuation per op (the pre-#1556-design
/// behaviour), which is what the sweep compares against.
fn stride() -> u64 {
    std::env::var("TEMEN_JOURNAL_STRIDE")
        .ok()
        .and_then(|s| s.parse().ok())
        .map(|n: u64| n.max(1))
        .unwrap_or(temen_interp::journal::DEFAULT_STATE_STRIDE)
}

fn measure(g: &Guest, module: &temen_ir::Module, turns: u64, armed: bool) -> Option<RunOut> {
    let mut run = build(g, module)?;
    run.set_journal_armed(armed);
    if armed {
        // No ceiling and an effectively infinite fine window: this measures what the journal *would*
        // hold, which is the number the bail criterion is about. Policy trimming is measured
        // separately by coalescing the whole history below.
        run.set_journal_policy(JournalPolicy {
            fine_turns: u64::MAX,
            byte_budget: 0,
            state_stride: stride(),
        });
    }
    let t0 = Instant::now();
    let (turns, stop) = drive(&mut run, turns);
    let elapsed = t0.elapsed();
    let window_hash = window_hash(&run, HASH_PREFIX);
    if armed {
        LAST_ARMED.with(|c| c.set(Some(run.journal_stats())));
        LAST_RUN.with(|c| *c.borrow_mut() = Some(run));
    }
    Some(RunOut {
        turns,
        stop,
        window_hash,
        elapsed,
    })
}

thread_local! {
    static LAST_ARMED: std::cell::Cell<Option<temen_interp::journal::JournalStats>> =
        const { std::cell::Cell::new(None) };
    static LAST_RUN: std::cell::RefCell<Option<ScheduledDebugRun>> =
        const { std::cell::RefCell::new(None) };
}

fn build(g: &Guest, module: &temen_ir::Module) -> Option<ScheduledDebugRun> {
    let inst = temen_run::instantiate(module.clone()).expect("instantiate (verifies)");
    let cfg = RunConfig {
        limits: Limits {
            fuel: None,
            deadline: None,
            max_fibers: 0,
            max_vcpus: 0,
        },
        args: g.args.iter().map(|s| s.as_bytes().to_vec()).collect(),
        ..RunConfig::default()
    };
    let caps: Vec<(&str, temen_run::HostCap)> = if g.fs.is_empty() {
        vec![]
    } else {
        let files =
            g.fs.iter()
                .map(|(key, host)| {
                    let bytes = std::fs::read(host).unwrap_or_else(|e| panic!("read {host}: {e}"));
                    (key.to_string(), bytes)
                })
                .collect();
        vec![("fs", temen_run::fs::mem_fs_seeded(files, vec![]))]
    };
    inst.debug_run_with_caps(&cfg, &caps)
        .expect("build debug run")
}

fn pct(n: usize, of: u64) -> String {
    if of == 0 {
        return "—".into();
    }
    format!("{:.2}%", (n as f64 / of as f64) * 100.0)
}

fn kib(n: usize) -> String {
    format!("{:.1} KiB", n as f64 / 1024.0)
}

fn main() {
    let only: Vec<String> = std::env::args().skip(1).collect();
    let turns: u64 = std::env::var("TEMEN_JOURNAL_TURNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(300_000);

    println!(
        "# undo-journal cost, {turns} turns per guest, state_stride={} (#1556 bail criteria)\n",
        stride()
    );

    for g in GUESTS {
        if !only.is_empty() && !only.iter().any(|o| o == g.name) {
            continue;
        }
        let Ok(bytes) = std::fs::read(g.asset) else {
            println!("## {}\n  asset missing: {} — skipped\n", g.name, g.asset);
            continue;
        };
        let module = match temen_encode::decode_module(&bytes) {
            Ok(m) => m,
            Err(e) => {
                println!("## {}\n  decode failed: {e:?} — skipped\n", g.name);
                continue;
            }
        };
        let window = module.memory.map_or(0, |mc| 1u64 << mc.size_log2);

        println!("## {} — {}", g.name, g.why);
        let Some(cold) = measure(g, &module, turns, false) else {
            println!("  outside the bytecode debug engine's subset — the tree-walk Inspector serves it\n");
            continue;
        };
        let Some(hot) = measure(g, &module, turns, true) else {
            println!("  armed build declined (unexpected: the cold build succeeded)\n");
            continue;
        };
        let stats = LAST_ARMED.with(|c| c.get()).expect("armed stats");

        // Inertness first: every number below is meaningless if journaling changed the run.
        let inert =
            cold.turns == hot.turns && cold.stop == hot.stop && cold.window_hash == hot.window_hash;
        if !inert {
            println!(
                "  !! NOT INERT (INVARIANTS #9b): unarmed {} turns/{}/{:#x} vs armed {} turns/{}/{:#x}",
                cold.turns, cold.stop, cold.window_hash, hot.turns, hot.stop, hot.window_hash
            );
        }

        // Undo first: coalescing below collapses the fine tail too, and `step_back` asks about the
        // tail. Measuring after it would only ever report "declined".
        undo_vs_replay(g, &module, cold.turns);

        // Level 2 over the whole history: the bound the design rests on.
        let l1 = stats.bytes;
        let (l2, coalesce_time) = LAST_RUN.with(|c| {
            let mut b = c.borrow_mut();
            let run = b.as_mut().expect("armed run");
            let t0 = Instant::now();
            run.coalesce_journal(run.op_turn());
            (run.journal_stats().bytes, t0.elapsed())
        });

        let slowdown = hot.elapsed.as_secs_f64() / cold.elapsed.as_secs_f64().max(f64::EPSILON);
        println!(
            "  window            {} ({} bytes)",
            kib(window as usize),
            window
        );
        println!("  turns / stop      {} / {}", cold.turns, cold.stop);
        println!("  inert             {}", if inert { "yes" } else { "NO" });
        println!(
            "  time              {:.3}s unarmed → {:.3}s armed   ({slowdown:.2}× slowdown)",
            cold.elapsed.as_secs_f64(),
            hot.elapsed.as_secs_f64()
        );
        println!(
            "  writes journaled  {} entries, {} appended",
            stats.appended,
            kib(stats.appended_bytes)
        );
        println!(
            "  states journaled  {} continuations + host cursors ({} per turn)",
            stats.states,
            if cold.turns > 0 {
                format!("{:.2}", stats.states as f64 / cold.turns as f64)
            } else {
                "—".into()
            }
        );
        println!(
            "  level 1 held      {} = {} of window",
            kib(l1),
            pct(l1, window)
        );
        println!(
            "  level 2 held      {} = {} of window   (coalesced in {:.3}s)",
            kib(l2),
            pct(l2, window),
            coalesce_time.as_secs_f64()
        );
        if l2 > 0 {
            println!(
                "  coalescing ratio  {:.1}× ({} appended → {} retained)",
                stats.appended_bytes as f64 / l2 as f64,
                kib(stats.appended_bytes),
                kib(l2)
            );
        }
        println!(
            "  BOUND             level 2 {} window — {}",
            if (l2 as u64) <= window { "≤" } else { ">" },
            if (l2 as u64) <= window {
                "holds (a compacted segment is no worse than a snapshot)"
            } else {
                "VIOLATED — the journal is worse than the snapshot it replaces"
            }
        );

        println!();
    }
}

/// `undo_to(t)` against what `seek(t)` costs today: rebuild a fresh run and re-drive to `t`.
///
/// Measured at the *end* of the run and one step back, which is the `step_back` case — the one the DAP
/// backend would serve from the journal. The replay side is measured without a checkpoint ladder, so it
/// is the O(t) worst case; a ladder bounds it to the stride, and the stride is the honest comparison
/// for a long session. Both are printed so the ratio can be read either way.
fn undo_vs_replay(g: &Guest, module: &temen_ir::Module, turns: u64) {
    if turns < 2 {
        return;
    }
    let target = turns - 1;

    let undo = LAST_RUN.with(|c| {
        let mut b = c.borrow_mut();
        let run = b.as_mut().expect("armed run");
        if !run.can_undo_to(target) {
            return None;
        }
        let t0 = Instant::now();
        let ok = run.undo_to(target);
        Some((ok, t0.elapsed()))
    });

    let t0 = Instant::now();
    let replay = build(g, module).map(|mut run| {
        drive(&mut run, target);
        t0.elapsed()
    });

    match (undo, replay) {
        (Some((true, u)), Some(r)) => println!(
            "  step_back         undo {:.6}s vs replay-from-0 {:.3}s   ({:.0}× cheaper)",
            u.as_secs_f64(),
            r.as_secs_f64(),
            r.as_secs_f64() / u.as_secs_f64().max(f64::EPSILON)
        ),
        (Some((false, _)), _) | (None, _) => println!(
            "  step_back         journal declined this turn (coalesced past, or outside the \
             invertible subset) — seek serves it"
        ),
        (_, None) => println!("  step_back         replay build failed"),
    }
}
