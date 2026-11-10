//! #2486 — what a guest does once per operation allocates nothing, on any engine: a capability
//! call, a call to a guest function, a fiber switch.
//!
//! A counting allocator stands in for the global one, and each path runs at two sizes. What the
//! larger run allocates beyond the smaller is what its extra operations allocated; compiling and
//! setting up cancel out. The count is the calling thread's own, so tests running alongside don't
//! add to it, and every path here runs on the thread that calls it.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::ffi::c_void;

use temen_interp::{put_results, CapState, Host, HostProcGrant, ProcHandler, Value};

struct Counting;

thread_local! {
    static ALLOCATIONS: Cell<u64> = const { Cell::new(0) };
}

fn count() {
    // `try_with`: the thread may be tearing its locals down.
    let _ = ALLOCATIONS.try_with(|n| n.set(n.get() + 1));
}

// SAFETY: every call goes straight to `System`. The count is a thread-local `Cell` with no
// destructor, which never allocates.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count();
        System.alloc(layout)
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count();
        System.alloc_zeroed(layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        count();
        System.realloc(ptr, layout, size)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

fn allocations() -> u64 {
    ALLOCATIONS.with(Cell::get)
}

const SMALL: i64 = 1_000;
const BIG: i64 = 11_000;

/// Allocations per operation of `run(n)`, which performs `n` of them.
fn per_op(mut run: impl FnMut(i64)) -> f64 {
    let start = allocations();
    run(SMALL);
    let small = allocations() - start;
    let start = allocations();
    run(BIG);
    let big = allocations() - start;
    (big as f64 - small as f64) / (BIG - SMALL) as f64
}

/// No path here allocates per operation. Compiling and setting up allocate about the same on
/// both runs, not exactly, so a run's count may differ by a few.
fn assert_free(what: &str, per_op: f64) {
    assert!(
        per_op.abs() < 0.01,
        "{what}: {per_op:.3} allocations per operation"
    );
}

fn module(src: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// A loop of `n` iterations taking a handle and `n`. Each iteration runs `body`, which makes `v13`
/// from the handle `v9` and the iteration's count `v12`. `rest` holds the functions it calls.
fn each(body: &str, rest: &str) -> temen_ir::Module {
    module(&format!(
        "memory 16
func (i32, i64) -> (i64) {{
block 0 (v0: i32, v1: i64) {{
  v2 = i64.const 0
  v3 = i64.const 0
  br 1(v0, v1, v2, v3)
}}
block 1 (v4: i32, v5: i64, v6: i64, v7: i64) {{
  v8 = i64.ge_s v7 v5
  br_if v8 3(v6) 2(v4, v5, v6, v7)
}}
block 2 (v9: i32, v10: i64, v11: i64, v12: i64) {{
{body}
  v14 = i64.add v11 v13
  v15 = i64.const 1
  v16 = i64.add v12 v15
  br 1(v9, v10, v14, v16)
}}
block 3 (v17: i64) {{
  return v17
}}
}}
{rest}"
    ))
}

/// `Clock.now`, on the handle the loop takes.
fn clock_calls() -> temen_ir::Module {
    each("  v13 = call.cap 2 0 () -> (i64) v9()", "")
}

/// A host procedure's op 0, on the handle the loop takes.
fn proc_calls() -> temen_ir::Module {
    each("  v13 = call.cap 13 0 (i64) -> (i64) v9(v12)", "")
}

/// A call to a guest leaf.
fn guest_calls() -> temen_ir::Module {
    each(
        "  v13 = call 1 (v12)",
        "func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 1
  v2 = i64.add v0 v1
  return v2
}
}",
    )
}

/// A fiber started and run to its return.
fn fiber_lives() -> temen_ir::Module {
    each(
        "  v20 = ref.func 1
  v21 = i64.const 0
  v22 = cont.new v20 v21
  v23, v13 = cont.resume v22 v12",
        "func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  return varg
}
}",
    )
}

/// A root that resumes one fiber `n` times; the fiber suspends forever.
fn fiber_switches() -> temen_ir::Module {
    module(
        "memory 16
func (i32, i64) -> (i64) {
block 0 (v0: i32, v1: i64) {
  v2 = ref.func 1
  v3 = i64.const 0
  v4 = cont.new v2 v3
  v5 = i64.const 0
  br 1(v4, v1, v5, v3)
}
block 1 (v6: i64, v7: i64, v8: i64, v9: i64) {
  v10 = i64.ge_s v8 v7
  br_if v10 3(v9) 2(v6, v7, v8, v9)
}
block 2 (v11: i64, v12: i64, v13: i64, v14: i64) {
  v15, v16 = cont.resume v11 v13
  v17 = i64.add v14 v16
  v18 = i64.const 1
  v19 = i64.add v13 v18
  br 1(v11, v12, v19, v17)
}
block 3 (v20: i64) {
  return v20
}
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  br 1(varg)
}
block 1 (v0: i64) {
  v1 = suspend v0
  br 1(v1)
}
}",
    )
}

/// A host granted the loop's capability: the clock, or a host procedure answering `x + 1`.
fn host(proc: bool) -> (Host, i32) {
    let mut host = Host::new();
    let handle = if proc {
        host.grant_host_proc(HostProcGrant::new(
            ProcHandler::Sync(Box::new(|_op, args, out, _mem, _minter| {
                put_results(out, &[args.first().copied().unwrap_or(0).wrapping_add(1)])
            })),
            CapState::Stateless,
        ))
    } else {
        host.grant_clock()
    };
    (host, handle)
}

/// The engines, each running module `m`'s loop `n` times over a fresh host.
#[derive(Clone, Copy, Debug)]
enum Engine {
    Jit,
    Bytecode,
    TreeWalker,
}

const ENGINES: [Engine; 3] = [Engine::Jit, Engine::Bytecode, Engine::TreeWalker];

impl Engine {
    fn run(self, m: &temen_ir::Module, proc: bool) -> impl FnMut(i64) + '_ {
        move |n| {
            let (mut host, handle) = host(proc);
            let mut fuel = u64::MAX;
            let args = [Value::I32(handle), Value::I64(n)];
            let ok = match self {
                Engine::Jit => {
                    let ctx = &mut host as *mut Host as *mut c_void;
                    let slots = [handle as i64, n];
                    temen_jit::compile_and_run_with_host(m, 0, &slots, temen_run::cap_thunk, ctx)
                        .is_ok()
                }
                Engine::Bytecode => temen_interp::bytecode::compile_and_run_with_host(
                    m, 0, &args, &mut fuel, &mut host,
                )
                .expect("the bytecode engine runs it, not the tree-walker")
                .is_ok(),
                Engine::TreeWalker => {
                    temen_interp::run_with_host(m, 0, &args, &mut fuel, &mut host).is_ok()
                }
            };
            assert!(ok, "{self:?} ran the loop to its end");
        }
    }
}

#[test]
fn a_capability_call_allocates_nothing() {
    let (clock, proc) = (clock_calls(), proc_calls());
    for e in ENGINES {
        assert_free(&format!("{e:?} Clock.now"), per_op(e.run(&clock, false)));
        assert_free(&format!("{e:?} host procedure"), per_op(e.run(&proc, true)));
    }
}

#[test]
fn a_guest_call_allocates_nothing() {
    let m = guest_calls();
    for e in ENGINES {
        assert_free(&format!("{e:?} guest call"), per_op(e.run(&m, false)));
    }
}

#[test]
fn a_fiber_switch_allocates_nothing() {
    let m = fiber_switches();
    for e in ENGINES {
        assert_free(&format!("{e:?} fiber switch"), per_op(e.run(&m, false)));
    }
}

/// What a fiber's life allocates (#2486), until it allocates nothing: four times on the JIT and
/// the tree-walker, twice on the bytecode engine. Any change fails here: one allocation more is a
/// regression, and one fewer brings this pin down with it.
#[test]
fn a_fiber_life_allocates_no_more_than_it_did() {
    let m = fiber_lives();
    for (e, was) in ENGINES.into_iter().zip([4.0, 2.0, 4.0]) {
        let now = per_op(e.run(&m, false));
        assert!(
            (now - was).abs() < 0.01,
            "{e:?}: a fiber's life allocates {now:.3} times, not {was}: update this when it falls"
        );
    }
}
