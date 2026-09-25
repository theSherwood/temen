//! #1768 — how a transform reads a `call.dyn` ([`IndirectReach`]). A freeze must be able to unwind
//! wherever it lands, so any function of the call's signature is a target (R8). A fork is an unwind
//! the runtime may decline, so it reads only the functions the module takes the address of, and a
//! **barrier** fronts the slot of every other may-suspend function: an index that selects one was
//! forged, and the barrier holds the shadow stack occupied so the runtime declines an unwind beneath
//! the uninstrumented `call.dyn` that made it.
//!
//! On the interpreter, the transform's oracle; `temen-run`'s `jit_fork` runs the fork end to end.

use temen_durable::{init_durable_window, transform, IndirectReach, TransformOpts, BARRIER_SP};
use temen_interp::{run_capture_reserved_with_host, Host, Value};
use temen_ir::durable_abi::ShadowArena;
use temen_ir::{Inst, Memory, Module, Terminator};

const TEST_ARENA: ShadowArena = ShadowArena {
    base: 16448,
    end: 65536,
};
const SIZE_LOG2: u8 = 18;
const WINDOW: usize = 1 << SIZE_LOG2;

// `entry(clock, slot)` reaches the site (`Clock.now`, standing in for a fork) through `site`, a
// direct call, then makes a `call.dyn` of `site`'s signature through `slot`. `site` returns the SP
// word of the root context's shadow stack (at the arena base) — what the runtime reads before it
// starts an unwind. `pure` has the same signature and never suspends. No function's address is taken.
const SRC: &str = r#"
export 0 func "entry" 0
func (i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32) {
  v2 = call 1 (v0)
  v3 = call.dyn (i32) -> (i64) v1 (v0)
  v4 = i64.add v2 v3
  return v4
  }
}
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = call.cap 2 0 () -> (i64) v0 ()
  v2 = i64.const 16448
  v3 = i64.load v2
  return v3
  }
}
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const 7
  return v1
  }
}
"#;

fn module(src: &str) -> Module {
    let mut m = temen_text::parse_module(src).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
        shadow: Some(TEST_ARENA),
    });
    m
}

fn is_site(i: &Inst) -> bool {
    matches!(
        i,
        Inst::CapCall {
            type_id: temen_ir::cap_id::CLOCK,
            op: 0,
            ..
        }
    )
}

/// Whether `f` is a barrier over `body`: one block that calls `body` and returns its results.
fn is_barrier_over(f: &temen_ir::Func, body: u32) -> bool {
    f.blocks.len() == 1
        && f.blocks[0]
            .insts
            .iter()
            .any(|i| matches!(i, Inst::Call { func, .. } if *func == body))
        && matches!(f.blocks[0].term, Terminator::Return(_))
}

/// Run `m` at `func` with `(clock cap, slot)` over a fresh durable window; the result.
fn run(m: &Module, func: u32, slot: i32) -> Value {
    let mut host = Host::new();
    host.clock_ns = 1000;
    let clk = host.grant_clock();
    let mut fuel = 1_000_000u64;
    let (r, window) = run_capture_reserved_with_host(
        m,
        func,
        &[Value::I32(clk), Value::I32(slot)],
        &mut fuel,
        &init_durable_window(WINDOW, TEST_ARENA),
        SIZE_LOG2,
        &mut host,
    );
    let sp = TEST_ARENA.region_base(0) as usize;
    assert_eq!(
        window[sp..sp + 8],
        TEST_ARENA.frame_base(0).to_le_bytes(),
        "every barrier restored the shadow stack it found"
    );
    r.expect("runs").remove(0)
}

#[test]
fn a_fork_instruments_no_call_dyn_that_only_a_forged_index_takes_to_a_site() {
    let m = module(SRC);
    let fork = transform(&m, &TransformOpts::fork(&is_site)).expect("transform");
    temen_verify::verify_module(&fork.module).expect("verify");

    // `entry` and `site` may suspend, and no `call.dyn` can select either (no address is taken):
    // each slot holds a barrier, and the bodies moved to the end, where every static reference —
    // the export, `entry`'s call of `site` — now points.
    assert_eq!(fork.body, vec![3, 4, 2]);
    assert!(is_barrier_over(&fork.module.funcs[0], 3));
    assert!(is_barrier_over(&fork.module.funcs[1], 4));
    assert_eq!(fork.module.exports[0].func, 3, "a host enters the body");
    assert_eq!(fork.module.funcs[2], m.funcs[2], "`pure` never suspends");
    let calls_site_body = fork.module.funcs[3]
        .blocks
        .iter()
        .flat_map(|b| &b.insts)
        .any(|i| matches!(i, Inst::Call { func: 4, .. }));
    assert!(calls_site_body, "`entry`'s body calls `site`'s body");

    // A freeze reads the `call.dyn` as reaching `site` (same signature) — an extra suspend point —
    // and needs no barrier.
    let freeze_reading = TransformOpts {
        indirect: IndirectReach::Signature,
        ..TransformOpts::fork(&is_site)
    };
    let by_sig = transform(&m, &freeze_reading).expect("transform");
    assert_eq!(by_sig.body, vec![0, 1, 2]);
    assert_eq!(by_sig.module.funcs.len(), 3);
    assert!(
        by_sig.module.funcs[0].blocks.len() > fork.module.funcs[3].blocks.len(),
        "the by-signature reading instruments the `call.dyn` too"
    );
}

#[test]
fn a_barrier_computes_what_its_body_computes_and_holds_the_shadow_stack_occupied() {
    let fork = transform(&module(SRC), &TransformOpts::fork(&is_site)).expect("transform");
    let empty = TEST_ARENA.frame_base(0) as i64;
    let entry = fork.body[0];
    // Through `pure`'s slot: `site` (entered directly) saw an empty stack, `pure` returns 7.
    assert_eq!(run(&fork.module, entry, 2), Value::I64(empty + 7));
    // Through `site`'s slot — a forged index — the barrier holds the stack occupied while the
    // body runs: an unwind the runtime started there could not come back through this `call.dyn`.
    assert_eq!(
        run(&fork.module, entry, 1),
        Value::I64(empty + BARRIER_SP as i64)
    );
    // Entered at its slot rather than its body, `entry` itself runs behind a barrier.
    assert_eq!(run(&fork.module, 0, 2), Value::I64(BARRIER_SP as i64 + 7));
}

#[test]
fn a_taken_address_makes_its_signature_a_suspend_point_and_needs_no_barrier() {
    // As `SRC`, but `entry` calls through `site`'s own `ref.func`.
    let src = SRC.replace(
        "  v3 = call.dyn (i32) -> (i64) v1 (v0)",
        "  v5 = ref.func 1\n  v3 = call.dyn (i32) -> (i64) v5 (v0)",
    );
    let m = module(&src);
    let fork = transform(&m, &TransformOpts::fork(&is_site)).expect("transform");
    temen_verify::verify_module(&fork.module).expect("verify");
    // `site` is selectable, so its signature is tainted: `entry`'s `call.dyn` is a suspend point,
    // and `site` needs no barrier. `entry`'s own signature is still untainted.
    assert_eq!(fork.body, vec![3, 1, 2]);
    assert!(is_barrier_over(&fork.module.funcs[0], 3));
    let empty = TEST_ARENA.frame_base(0) as i64;
    assert_eq!(run(&fork.module, fork.body[0], 0), Value::I64(2 * empty));
}
