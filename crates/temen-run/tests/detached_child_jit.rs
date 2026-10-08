//! PROCESS.md §5 / #1287 — **`instantiate_detached` (op 15) on the native JIT**, differential against the
//! tree-walker. The parent (compiled by Cranelift) spawns a separate-module child DETACHED: the JIT's
//! op-15 thunk takes the detached-spawn `Budget` quota, builds the child powerbox through
//! `Host::spawn_detached_child` (attests `window_exposed = false`, starter caps over the reservation),
//! compiles the child over a **decoupled window** (its declared 64 KiB committed inside a root-sized lazy
//! reservation), seeds the module's data + the spawn-time argv payload into that window, and runs it on
//! its own OS thread — no carve, no copy-in, no copy-back. The child reads the argv word, `self.attest`s,
//! `vm_map`s past its declared window (committing a tail page of ITS reservation through its own
//! `AddressSpace`), stores/loads on the grown page and returns `word + attest`. Same result as the
//! interpreter's op-15 arm; an exhausted budget refuses `-EINVAL` on both.

use core::ffi::c_void;
use core::ptr::null_mut;
use std::sync::Mutex;
use temen_interp::{cap_id, run_with_host, Host, MemLayout, Trap, Value};
use temen_jit::{
    compile_and_run_capture_reserved_with_host_ex, GrantChild, GrantChildHooks, JitOutcome,
    TrapKind,
};

/// #1234 — the production table, derived from one [`temen_run::CapCtx`] so the hook family and
/// the parent pointer it decodes are chosen together (this used to hand-roll both, and nothing
/// checked that the pointer matched the ctx the run baked).
fn grant_hooks(host: *mut temen_interp::Host) -> GrantChildHooks {
    temen_run::production_grant_hooks(temen_run::CapCtx::Raw(host))
}

fn module(text: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(text).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// "hello-de" as a little-endian i64 — the word the child reads at `args_base + 8`.
const ARGV_WORD: i64 = i64::from_le_bytes(*b"hello-de");

/// The detached child (child-entry, `memory 16`, manifest `vm_map`): argv word at 16384+128+8, attest,
/// `vm_map [64 KiB, 80 KiB)`, store/load the word on the grown page, return `word + attest`.
const CHILD: &str = r#"memory 16
import 0 "vm_map" (i64, i64, i32) -> (i64)
func (i64) -> (i64) {
block 0 (v0: i64) {
  vab = i64.const 16520
  va = i64.load vab
  vz = i32.const 0
  vat = call.cap 4294967295 4 () -> (i64) vz ()
  voff = i64.const 65536
  vlen = i64.const 16384
  vprot = i32.const 3
  vg = call.import 0 (voff, vlen, vprot)
  vp = i64.const 65600
  i64.store vp va
  vld = i64.load vp
  vs = i64.add vld vat
  return vs
  }
}
"#;

/// The parent: `v0` Instantiator, `v1` the child `Module`, `v2` the detached-spawn `Budget`. Stores the args
/// blob (`argc 1`, `"hello-detached\0"`) as three words at 18432, spawns the child detached (9-arg op 15,
/// payload `(18432, 24)`, no grants, entry 0, window 2^16), then joins it — or, in the refusal probe,
/// returns the spawn's own result.
fn parent(join: bool) -> String {
    parent_then(if join {
        "vr = call.cap 6 1 (i32) -> (i64) v0 (vh)\n  return vr"
    } else {
        "vr = i64.extend_i32_s vh\n  return vr"
    })
}

/// [`parent`], ending in `tail` (which sees `v0` and the child handle `vh`).
fn parent_then(tail: &str) -> String {
    format!(
        r#"memory 17
func (i32, i32, i32) -> (i64) {{
block 0 (v0: i32, v1: i32, v2: i32) {{
  vb0 = i64.const 18432
  vw0 = i64.const 1
  i64.store vb0 vw0
  vb1 = i64.const 18440
  vw1 = i64.const {w1}
  i64.store vb1 vw1
  vb2 = i64.const 18448
  vw2 = i64.const {w2}
  i64.store vb2 vw2
  vmh = i64.extend_i32_u v1
  vmin = i64.extend_i32_u v2
  vz = i64.const 0
  ve = i64.const 0
  vlog = i64.const 16
  vq = i64.const 0
  vap = i64.const 18432
  val = i64.const 24
  vh = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vmin, vmh, vz, vz, ve, vlog, vq, vap, val)
  {tail}
  }}
}}
"#,
        w1 = ARGV_WORD,
        w2 = i64::from_le_bytes(*b"tached\0\0"),
    )
}

fn host(child: &temen_ir::Module, minter_quota: u64) -> (Host, [i32; 3]) {
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, 1u64 << 17);
    let modh = host.grant_module(child);
    let budget = host.grant_budget(-1, (minter_quota) as i64, -1);
    (host, [inst, modh, budget])
}

fn run_jit(parent: &temen_ir::Module, child: &temen_ir::Module, quota: u64) -> i64 {
    match jit_outcome(parent, host(child, quota)) {
        JitOutcome::Returned(ref v) => v.first().copied().unwrap_or(-1),
        ref o => panic!("jit ended abnormally: {o:?}"),
    }
}

/// Run `parent` on the JIT over `(host, handles)`, with no fuel armed for the root.
fn jit_outcome(parent: &temen_ir::Module, (mut host, h): (Host, [i32; 3])) -> JitOutcome {
    jit_on(parent, &mut host, h)
}

/// [`jit_outcome`] over a host the caller keeps.
fn jit_on(parent: &temen_ir::Module, host: &mut Host, h: [i32; 3]) -> JitOutcome {
    let args = [h[0] as i64, h[1] as i64, h[2] as i64];
    let (jo, _) = compile_and_run_capture_reserved_with_host_ex(
        parent,
        0,
        &args,
        &[],
        temen_ir::DEFAULT_RESERVED_LOG2,
        temen_run::cap_thunk,
        host as *mut Host as *mut c_void,
        Some(temen_run::module_resolver),
        Some(grant_hooks(host as *mut Host)),
    )
    .expect("jit run");
    jo
}

fn run_interp(parent: &temen_ir::Module, child: &temen_ir::Module, quota: u64) -> i64 {
    match interp_result(parent, host(child, quota))
        .expect("interp run")
        .first()
    {
        Some(Value::I64(x)) => *x,
        other => panic!("unexpected interp result {other:?}"),
    }
}

/// Run `parent` on the tree-walker over `(host, handles)`.
fn interp_result(
    parent: &temen_ir::Module,
    (mut host, h): (Host, [i32; 3]),
) -> Result<Vec<Value>, Trap> {
    interp_on(parent, &mut host, h)
}

/// [`interp_result`] over a host the caller keeps.
fn interp_on(parent: &temen_ir::Module, host: &mut Host, h: [i32; 3]) -> Result<Vec<Value>, Trap> {
    let mut fuel = 50_000_000u64;
    run_with_host(
        parent,
        0,
        &[Value::I32(h[0]), Value::I32(h[1]), Value::I32(h[2])],
        &mut fuel,
        host,
    )
}

#[test]
fn a_detached_child_on_the_jit_matches_the_interpreter() {
    let p = module(&parent(true));
    let c = module(CHILD);
    let want = ARGV_WORD + 1; // argv landed; attest = 1 (tier 1, window_exposed = false)
                              // The budget holds the child's window and the 16 KiB it grows (#1909: growth spends it too).
    let quota = (1 << 16) + (1 << 14);
    assert_eq!(run_interp(&p, &c, quota), want, "interpreter oracle");
    let before = temen_jit::child_compiles();
    assert_eq!(
        run_jit(&p, &c, quota),
        want,
        "the JIT-hosted detached child"
    );
    assert!(
        temen_jit::child_compiles() > before,
        "the child was JIT-compiled (not served by the interpreter)"
    );
}

/// A detached child that reads the room left in its own `"budget"`.
const CHILD_READS_ITS_BUDGET: &str = r#"memory 16
data 20000 "budget"
func (i64) -> (i64) {
block 0 (v0: i64) {
  vp = i64.const 20000
  vl = i64.const 6
  vb = self.resolve vp vl
  vf = i64.const 1
  vr = call.cap 14 1 (i64) -> (i64) vb (vf)
  return vr
  }
}
"#;

/// #1944 — the budget that pays for a detached child's window is the child's `"budget"`, on the JIT
/// as on the interpreter: the child reads its room with its own window already charged.
#[test]
fn a_detached_child_on_the_jit_holds_the_budget_that_paid_for_it() {
    let p = module(&parent(true));
    let c = module(CHILD_READS_ITS_BUDGET);
    let want = (1 << 20) - (1 << 16);
    assert_eq!(run_interp(&p, &c, 1 << 20), want, "interpreter oracle");
    assert_eq!(run_jit(&p, &c, 1 << 20), want, "the JIT");
}

/// A detached child (`memory 16`) that takes 1000 loop back-edges (one fuel each), then returns 7.
const CHILD_LOOPS: &str = r#"memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  vn = i32.const 1000
  br 1(vn)
}
block 1 (vi: i32) {
  one = i32.const 1
  vj = i32.sub vi one
  br_if vj 1(vj) 2()
}
block 2 () {
  v = i64.const 7
  return v
  }
}
"#;

/// [`host`] whose `Budget` is a node split from the granted one with a `fuel` ceiling.
fn fuel_host(child: &temen_ir::Module, fuel: i64) -> (Host, [i32; 3]) {
    let (mut host, mut h) = host(child, 1 << 20);
    h[2] = host
        .cap_dispatch_slots(cap_id::BUDGET, 0, h[2], &[fuel, -1, -1], None)
        .expect("split")[0] as i32;
    (host, h)
}

/// #1944 slice 3, #1705 — a detached child burns the fuel of the budget that paid for it, on the JIT
/// as on the interpreter, though the JIT's root runs with no fuel armed at all: a ceiling the child
/// loops past ends it (the join hands its trap to the parent), and one with room lets it finish.
#[test]
fn a_detached_childs_budget_bounds_its_fuel_under_an_unmetered_jit_root() {
    let p = module(&parent(true));
    let c = module(CHILD_LOOPS);
    for (fuel, want_interp, want_jit) in [
        (2000, Ok(vec![Value::I64(7)]), JitOutcome::Returned(vec![7])),
        (
            500,
            Err(Trap::OutOfFuel),
            JitOutcome::Trapped(TrapKind::OutOfFuel),
        ),
    ] {
        assert_eq!(
            interp_result(&p, fuel_host(&c, fuel)),
            want_interp,
            "interpreter, ceiling {fuel}"
        );
        assert_eq!(
            jit_outcome(&p, fuel_host(&c, fuel)),
            want_jit,
            "the JIT, ceiling {fuel}"
        );
    }
}

/// #2053 — `wait` (op 18) on the JIT answers as the interpreter does: a child that loops past its
/// node's fuel ceiling answers `OUT_OF_FUEL` and the parent runs on, unmetered as its root is, the
/// child reaped; one with room answers 0 and stays for its `join`.
#[test]
fn wait_reports_a_detached_childs_fuel_running_out_on_the_jit() {
    let wait = module(&parent_then(
        "vr = call.cap 6 18 (i32) -> (i64) v0 (vh)\n  return vr",
    ));
    let wait_join = module(&parent_then(
        "vw = call.cap 6 18 (i32) -> (i64) v0 (vh)\n  \
         vj = call.cap 6 1 (i32) -> (i64) v0 (vh)\n  \
         vk = i64.const 1000\n  \
         vs = i64.mul vw vk\n  \
         vr = i64.add vs vj\n  \
         return vr",
    ));
    let c = module(CHILD_LOOPS);
    for (p, fuel, want) in [
        (&wait_join, 2000, 7),
        (&wait, 500, temen_ir::trap_code::OUT_OF_FUEL),
    ] {
        assert_eq!(
            interp_result(p, fuel_host(&c, fuel)),
            Ok(vec![Value::I64(want)]),
            "interpreter, ceiling {fuel}"
        );
        assert_eq!(
            jit_outcome(p, fuel_host(&c, fuel)),
            JitOutcome::Returned(vec![want]),
            "the JIT, ceiling {fuel}"
        );
    }
    // The wait reaped the trapped child: joining it after is a spent handle.
    assert_eq!(
        interp_result(&wait_join, fuel_host(&c, 500)),
        Err(Trap::ThreadFault)
    );
    assert_eq!(
        jit_outcome(&wait_join, fuel_host(&c, 500)),
        JitOutcome::Trapped(TrapKind::ThreadFault)
    );
}

/// `v0` Instantiator, `v1` the child `Module`, `v2` the `Budget`: spawn the child detached (window
/// 2^16, no payload), join it, then spawn and join it again, returning the two results' sum. A refused
/// second spawn traps its join (a negative handle).
const SPAWN_JOIN_SPAWN: &str = r#"memory 17
func (i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32) {
  vmh = i64.extend_i32_u v1
  vb = i64.extend_i32_u v2
  vz = i64.const 0
  vlog = i64.const 16
  vh = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vb, vmh, vz, vz, vz, vlog, vz, vz, vz)
  vj = call.cap 6 1 (i32) -> (i64) v0 (vh)
  vh2 = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vb, vmh, vz, vz, vz, vlog, vz, vz, vz)
  vj2 = call.cap 6 1 (i32) -> (i64) v0 (vh2)
  vr = i64.add vj vj2
  return vr
  }
}
"#;

/// [`host`] whose `Budget` funds `spawn` live children (1 MiB of window).
fn spawn_host(child: &temen_ir::Module, spawn: i64) -> (Host, [i32; 3]) {
    let (mut host, mut h) = host(child, 1 << 20);
    h[2] = host.grant_budget(-1, 1 << 20, spawn);
    (host, h)
}

/// #1944 slice 3 — a detached child is one `spawn` of the budget that pays for it while it lives, on
/// the JIT as on the interpreter: a spawn-0 budget funds no child, and a one-child budget funds a
/// second once the first is joined (the JIT hands the child back through its lease hook).
#[test]
fn a_detached_child_is_one_spawn_of_its_budget_on_the_jit() {
    let c = module(CHILD_LOOPS);
    let refused = module(&parent(false));
    assert_eq!(
        interp_result(&refused, spawn_host(&c, 0)),
        Ok(vec![Value::I64(-22)]),
        "interpreter, a spawn-0 budget"
    );
    assert_eq!(
        jit_outcome(&refused, spawn_host(&c, 0)),
        JitOutcome::Returned(vec![-22]),
        "the JIT, a spawn-0 budget"
    );
    let twice = module(SPAWN_JOIN_SPAWN);
    assert_eq!(
        interp_result(&twice, spawn_host(&c, 1)),
        Ok(vec![Value::I64(14)]),
        "interpreter, a one-child budget in turn"
    );
    assert_eq!(
        jit_outcome(&twice, spawn_host(&c, 1)),
        JitOutcome::Returned(vec![14]),
        "the JIT, a one-child budget in turn"
    );
}

/// `v0` Instantiator, `v1` the child `Module`, `v2` the `Budget`: spawn the child detached (window
/// 2^16, no payload) and join it.
const SPAWN_JOIN: &str = r#"memory 17
func (i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32) {
  vmh = i64.extend_i32_u v1
  vb = i64.extend_i32_u v2
  vz = i64.const 0
  vlog = i64.const 16
  vh = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vb, vmh, vz, vz, vz, vlog, vz, vz, vz)
  vj = call.cap 6 1 (i32) -> (i64) v0 (vh)
  return vj
  }
}
"#;

/// A detached child (`memory 16`) that spawns a thread, joins it, then spawns another and joins it,
/// returning the sum of their results: each thread returns its `arg` plus one, 11 and 21.
const CHILD_SPAWNS_THREADS_IN_TURN: &str = r#"memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  vz = i64.const 0
  va = i64.const 10
  vt1 = thread.spawn 1 vz va
  vj1 = thread.join vt1
  vb = i64.const 20
  vt2 = thread.spawn 1 vz vb
  vj2 = thread.join vt2
  vr = i64.add vj1 vj2
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vone = i64.const 1
  vr = i64.add varg vone
  return vr
  }
}
"#;

/// #2001 — a detached child's threads are `spawn`s of the budget that pays for it while they live, on
/// the JIT as on the interpreter: a child whose ceiling is one vCPU fills it itself, so its first
/// `thread.spawn` traps, and one whose ceiling is two runs a thread, joins it and runs another in its
/// place (`Done::publish` hands the first one's charge back).
#[test]
fn a_detached_childs_threads_are_capped_by_its_budget_on_the_jit() {
    let p = module(SPAWN_JOIN);
    let c = module(CHILD_SPAWNS_THREADS_IN_TURN);
    for (spawn, want_interp, want_jit) in [
        (
            1,
            Err(Trap::ThreadFault),
            JitOutcome::Trapped(TrapKind::ThreadFault),
        ),
        (2, Ok(vec![Value::I64(32)]), JitOutcome::Returned(vec![32])),
    ] {
        assert_eq!(
            interp_result(&p, spawn_host(&c, spawn)),
            want_interp,
            "interpreter, a {spawn}-vCPU budget"
        );
        assert_eq!(
            jit_outcome(&p, spawn_host(&c, spawn)),
            want_jit,
            "the JIT, a {spawn}-vCPU budget"
        );
    }
}

/// A detached child (`memory 16`) importing `exit`, which no grant binds.
const CHILD_IMPORTS_EXIT: &str = r#"memory 16
import 0 "exit" (i32) -> ()
func (i64) -> (i64) {
block 0 (v0: i64) {
  vr = i64.const 42
  return vr
  }
}
"#;

/// `v0` Instantiator, `v1` the child `Module`, `v2` the `Budget`: spawn the child detached (window
/// 2^16, grants `(gp, gn)`), and if the spawn is refused return the room left in `v2`, else `-1`.
fn parent_room_after_refusal(gp: u64, gn: u64) -> String {
    format!(
        r#"memory 17
func (i32, i32, i32) -> (i64) {{
block 0 (v0: i32, v1: i32, v2: i32) {{
  vmh = i64.extend_i32_u v1
  vmin = i64.extend_i32_u v2
  vz = i64.const 0
  vgp = i64.const {gp}
  vgn = i64.const {gn}
  vlog = i64.const 16
  vh = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vmin, vmh, vgp, vgn, vz, vlog, vz, vz, vz)
  vz32 = i32.const 0
  vneg = i32.lt_s vh vz32
  br_if vneg 1(v2) 2()
}}
block 1 (vb: i32) {{
  vf = i64.const 1
  vr = call.cap 14 1 (i64) -> (i64) vb (vf)
  return vr
}}
block 2 () {{
  vm = i64.const -1
  return vm
  }}
}}
"#
    )
}

/// #1975 — a spawn refused after its admission charges nothing. The child's imports are bound after
/// the budget's take, and an unbound one refuses the spawn `-EINVAL`: the refusal hands the take
/// back, so the budget keeps all its room, on the JIT as on the interpreter.
#[test]
fn a_detached_spawn_refused_for_an_unbound_import_charges_nothing() {
    let p = module(&parent_room_after_refusal(0, 0));
    let c = module(CHILD_IMPORTS_EXIT);
    assert_eq!(run_interp(&p, &c, 1 << 20), 1 << 20, "interpreter oracle");
    assert_eq!(run_jit(&p, &c, 1 << 20), 1 << 20, "the JIT");
}

/// #1975 — on the JIT the builder reads the grant records after the budget's take, so a record out
/// of the window traps there: the trap hands the take back too. (The interpreter reads them before
/// its admission and traps the same way, having charged nothing.)
#[test]
fn a_detached_spawn_trapping_on_its_grant_records_charges_nothing() {
    let p = module(&parent_room_after_refusal(1 << 40, 1));
    let c = module(CHILD);
    let (mut host, h) = host(&c, 1 << 20);
    let args = [h[0] as i64, h[1] as i64, h[2] as i64];
    let (jo, _) = compile_and_run_capture_reserved_with_host_ex(
        &p,
        0,
        &args,
        &[],
        temen_ir::DEFAULT_RESERVED_LOG2,
        temen_run::cap_thunk,
        &mut host as *mut Host as *mut c_void,
        Some(temen_run::module_resolver),
        Some(grant_hooks(&mut host as *mut Host)),
    )
    .expect("jit run");
    assert!(
        matches!(jo, JitOutcome::Trapped(_)),
        "the record out of the window traps: {jo:?}"
    );
    let room = host
        .cap_dispatch_slots(cap_id::BUDGET, 1, h[2], &[1], None)
        .expect("read")[0];
    assert_eq!(room, 1 << 20, "the trap handed the window back");
}

/// #1972 — the JIT's op-15 thunk admits a child (`budget_mem_take`) and builds it
/// (`build_detached`) in two hook calls, and a concurrent run's hooks each take the parent's lock on
/// their own. So two vCPUs of one domain can admit, admit, build, build. Each build must still give
/// its child the lane and the `"budget"` that child's own admission charged. While the admission left
/// them on the parent for the next build to take, child A got B's and child B got neither.
#[test]
fn two_interleaved_detached_spawns_each_get_their_own_lane_and_budget() {
    const WINDOW: u64 = 1 << 16;
    let mut host = Host::new();
    let root = host.grant_budget(-1, -1, -1);
    let (a, b) = {
        let mut split = |mem: i64, lane: i64| {
            host.cap_dispatch_slots(cap_id::BUDGET, 0, root, &[-1, mem, -1, -1, lane], None)
                .expect("split")[0] as i32
        };
        (split(1 << 20, 1), split(1 << 19, 2))
    };
    let cell = Mutex::new(host);
    let hooks = temen_run::production_grant_hooks(temen_run::CapCtx::Locked(&cell));
    let ctx = &cell as *const Mutex<Host> as *mut c_void;
    let build = |budget: i32, lane: i64| -> GrantChild {
        let mut gc = GrantChild {
            ctx: null_mut(),
            retained_ctx: null_mut(),
            inst_handle: 0,
            as_handle: 0,
            grant_handle: 0,
            jit_table_log2: 0,
            domain: 0,
            lane_cap: -1,
            parent_domain: 0,
            parent_lane_cap: -1,
        };
        let mut trap = 0i64;
        let reservation = 1u64 << temen_ir::DEFAULT_RESERVED_LOG2;
        // SAFETY: `ctx` is the locked parent cell the hooks were made for; with no grant records,
        // nothing is read from the (null) window.
        let built = unsafe {
            (hooks.build_detached)(
                ctx,
                null_mut(),
                0,
                0,
                0,
                reservation,
                budget,
                lane,
                &mut gc,
                &mut trap,
            )
        };
        assert_eq!(built, 1, "the build trapped {trap}");
        gc
    };
    // SAFETY: as above.
    let lanes = unsafe {
        [
            (hooks.budget_mem_take)(ctx, a, WINDOW),
            (hooks.budget_mem_take)(ctx, b, WINDOW),
        ]
    };
    assert_eq!(
        lanes,
        [1, 2],
        "both admitted, with their lanes, before either is built"
    );
    let (ca, cb) = (build(a, lanes[0]), build(b, lanes[1]));
    let room = |gc: &GrantChild| -> i64 {
        // SAFETY: `gc.ctx` is the child powerbox cell the build filled, not yet released.
        let child = unsafe { &*(gc.ctx as *const Mutex<Host>) };
        let mut h = child.lock().unwrap();
        let budget = h
            .resolve_cap_name("budget")
            .expect("the child holds a budget");
        h.cap_dispatch_slots(cap_id::BUDGET, 1, budget, &[1], None)
            .expect("read")[0]
    };
    assert_eq!(
        (ca.lane_cap, room(&ca)),
        (1, (1 << 20) - WINDOW as i64),
        "child A: its own lane and budget"
    );
    assert_eq!(
        (cb.lane_cap, room(&cb)),
        (2, (1 << 19) - WINDOW as i64),
        "child B: its own lane and budget"
    );
    // Each child hands back the lane stamped on it when it is reaped, so the parent's Σ returns to 0.
    // SAFETY: as above; each child's two refs are released once each.
    unsafe {
        (hooks.lane_give)(ctx, ca.lane_cap);
        (hooks.lane_give)(ctx, cb.lane_cap);
        for c in [&ca, &cb] {
            (hooks.release)(c.ctx);
            (hooks.release)(c.retained_ctx);
        }
    }
    assert_eq!(
        cell.lock().unwrap().granted_lanes(),
        0,
        "every lane came back"
    );
}

#[test]
fn an_exhausted_minter_refuses_probeably_on_both_backends() {
    let p = module(&parent(false));
    let c = module(CHILD);
    assert_eq!(run_interp(&p, &c, (1 << 16) - 1), -22);
    assert_eq!(run_jit(&p, &c, (1 << 16) - 1), -22);
}

/// A parent that only issues the 7-arg op 15 and returns its result — no window stores, so it runs
/// unchanged under a **durable** host (whose shadow reserve spans `[0, 64 KiB)`).
const SPAWN_ONLY_PARENT: &str = r#"memory 17
func (i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32) {
  vmh = i64.extend_i32_u v1
  vmin = i64.extend_i32_u v2
  vz = i64.const 0
  ve = i64.const 0
  vlog = i64.const 16
  vq = i64.const 0
  vs = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vmin, vmh, vz, vz, ve, vlog, vq)
  vr = i64.extend_i32_s vs
  return vr
  }
}
"#;

/// #1412 / #1501 — **a durable detached spawn of an un-attested module declines identically on the
/// interpreter and the native JIT.**
///
/// INVARIANTS #9 has no transition state: the tree-walk interpreter defines guest-observable semantics
/// and the other engines match it or decline to it. Since #1361 step 4 a durable domain *does* spawn
/// detached children — when it holds freeze authority over its detached progeny (#1440) and the module
/// is attested freezable (#1501) — and its freeze captures them (`temen`'s `durable_detached_jit.rs`).
/// This pins the refusal half across the third engine: an **un-attested** module is refused on both,
/// probeably (`-EINVAL`), charging nothing. (The admission half, on the oracle and the resumable engine,
/// is `temen-interp`'s `durable_detached_parity.rs`.)
#[test]
fn a_durable_detached_spawn_of_an_unattested_module_declines_the_same_way_on_both_backends() {
    let p = module(SPAWN_ONLY_PARENT);
    let c = module(CHILD);

    // Native JIT tier, through the embedder's path (`jit_cap_run`, which installs the grant hooks): the
    // same `-EINVAL`, from the same rule — `mod_durable_ok` — before the budget is charged.
    {
        let (mut host, h) = host(&c, 1 << 16);
        host.set_durable(true);
        let args = [h[0] as i64, h[1] as i64, h[2] as i64];
        let (jo, _) = temen_run::jit_cap_run(
            &p,
            0,
            &args,
            &MemLayout::image(Vec::new()),
            temen_ir::DEFAULT_RESERVED_LOG2,
            0,
            &mut host,
            None,
        )
        .expect("jit run");
        let r = match jo {
            JitOutcome::Returned(ref v) => v.first().copied().unwrap_or(-1),
            ref o => panic!("jit ended abnormally: {o:?}"),
        };
        assert_eq!(r, -22, "native jit: EINVAL, not a trap");
        assert!(
            host.budget_mem_take(h[2], 1 << 16),
            "native jit: the refusal charged the budget nothing"
        );
    }

    // Interpreter tier: the same answer, byte for byte. `-22` is `-EINVAL`, matching the native thunk
    // above — and it is a *value*, so the run completes and the guest could have handled it.
    {
        let (mut host, h) = host(&c, 1 << 16);
        host.set_durable(true);
        let mut fuel = 50_000_000u64;
        let r = run_with_host(
            &p,
            0,
            &[Value::I32(h[0]), Value::I32(h[1]), Value::I32(h[2])],
            &mut fuel,
            &mut host,
        )
        .expect("interp run: the refusal is a value, so the run still completes");
        let slot = match r.first() {
            Some(Value::I64(x)) => *x,
            other => panic!("unexpected interp result {other:?}"),
        };
        assert_eq!(
            slot, -22,
            "interp must give the native JIT's answer for the same module (INVARIANTS #9): \
             -EINVAL, not an admission"
        );
        // And, like the native thunk, it charges nothing — the gate sits before the quota take, so the
        // guest keeps the budget for something it *is* allowed to do.
        assert!(
            host.budget_mem_take(h[2], 1 << 16),
            "interp: the refusal charged the budget nothing, exactly as the native thunk's does"
        );
    }
}

/// Op-15 **pre-mapped region** (the 11-arg form) on the native JIT, differential against the
/// tree-walker: the parent mints a 64 KiB `SharedRegion`, maps it at 65536 in its own window, stores
/// the word 41 at region byte 0 and spawns the child detached with the region pre-mapped at 65536 of
/// ITS window. The child reads the word through the alias, writes `2 × 41` at region byte 8 and
/// returns `41 + 1`; the parent joins and reads the child's word back through its own mapping:
/// `1000 × 42 + 82`. On the JIT both mappings are real `MAP_SHARED` views of one memfd/section
/// (`Host::apply_premap` over the child's `MprotectWindow`, before its first instruction); on the
/// interpreter they are `PageProt::Backed` aliases — same bytes, same result.
const PREMAP_CHILD: &str = r#"memory 17
func (i64) -> (i64) {
block 0 (v0: i64) {
  va = i64.const 65536
  vin = i64.load va
  vtwo = i64.const 2
  vout = i64.mul vin vtwo
  vb = i64.const 65544
  i64.store vb vout
  vone = i64.const 1
  vr = i64.add vin vone
  return vr
  }
}
"#;

/// `v0` Instantiator, `v1` AddressSpace, `v2` the child `Module`, `v3` the `Budget`; `off` is the
/// child-window offset the region is pre-mapped at (`65536` round-trips; `1 << 17` overruns the
/// child's window and refuses `-EINVAL`, which the non-joining form returns).
fn premap_parent(off: u64, join: bool) -> String {
    let tail = if join {
        "vj = call.cap 6 1 (i32) -> (i64) v0 (vc)\n  vk = i64.const 1000\n  vm = i64.mul vj vk\n  vob = i64.const 65544\n  vo = i64.load vob\n  vr = i64.add vm vo\n  return vr"
    } else {
        "vr = i64.extend_i32_s vc\n  return vr"
    };
    format!(
        r#"memory 17
func (i32, i32, i32, i32) -> (i64) {{
block 0 (v0: i32, v1: i32, v2: i32, v3: i32) {{
  vlen = i64.const 65536
  vrh64 = call.cap 5 5 (i64) -> (i64) v1 (vlen)
  vrh = i32.wrap_i64 vrh64
  vwo = i64.const 65536
  vro = i64.const 0
  vprot = i32.const 3
  vm0 = call.cap 4 0 (i64, i64, i64, i32) -> (i64) vrh (vwo, vro, vlen, vprot)
  vin = i64.const 41
  i64.store vwo vin
  vmh = i64.extend_i32_u v2
  vb = i64.extend_i32_u v3
  vz = i64.const 0
  vlog = i64.const 17
  vreg = i64.extend_i32_u vrh
  voff = i64.const {off}
  vc = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vb, vmh, vz, vz, vz, vlog, vz, vz, vz, vreg, voff)
  {tail}
  }}
}}
"#
    )
}

/// The pre-map parent's powerbox: the four handles, plus the OS shared-memory region factory the JIT's
/// real `MAP_SHARED` aliasing needs (a software `VecBacking` cannot be `mmap`ed).
fn premap_host(child: &temen_ir::Module) -> (Host, [i32; 4]) {
    let mut host = Host::new();
    host.set_region_factory(temen_run::new_shared_region);
    let inst = host.grant_instantiator(0, 1u64 << 17);
    let aspace = host.grant_address_space(0, 1u64 << 17);
    let modh = host.grant_module(child);
    let budget = host.grant_budget(-1, 1i64 << 17, -1);
    (host, [inst, aspace, modh, budget])
}

fn run_premap_jit(parent: &temen_ir::Module, child: &temen_ir::Module) -> i64 {
    let (mut host, h) = premap_host(child);
    let args = [h[0] as i64, h[1] as i64, h[2] as i64, h[3] as i64];
    let (jo, _) = compile_and_run_capture_reserved_with_host_ex(
        parent,
        0,
        &args,
        &[],
        temen_ir::DEFAULT_RESERVED_LOG2,
        temen_run::cap_thunk,
        &mut host as *mut Host as *mut c_void,
        Some(temen_run::module_resolver),
        Some(grant_hooks(&mut host as *mut Host)),
    )
    .expect("jit run");
    match jo {
        JitOutcome::Returned(ref v) => v.first().copied().unwrap_or(-1),
        ref o => panic!("jit ended abnormally: {o:?}"),
    }
}

fn run_premap_interp(parent: &temen_ir::Module, child: &temen_ir::Module) -> i64 {
    let (mut host, h) = premap_host(child);
    let mut fuel = 50_000_000u64;
    let r = run_with_host(
        parent,
        0,
        &[
            Value::I32(h[0]),
            Value::I32(h[1]),
            Value::I32(h[2]),
            Value::I32(h[3]),
        ],
        &mut fuel,
        &mut host,
    )
    .expect("interp run");
    match r.first() {
        Some(Value::I64(x)) => *x,
        other => panic!("unexpected interp result {other:?}"),
    }
}

#[test]
fn a_pre_mapped_region_round_trips_bulk_data_on_the_jit_as_on_the_interpreter() {
    let parent = module(&premap_parent(65536, true));
    let child = module(PREMAP_CHILD);
    let interp = run_premap_interp(&parent, &child);
    assert_eq!(interp, 42_082, "the oracle's round trip");
    assert_eq!(
        run_premap_jit(&parent, &child),
        interp,
        "the JIT matches the oracle"
    );
}

#[test]
fn a_pre_map_overrunning_the_child_window_refuses_probeably_on_both_backends() {
    let parent = module(&premap_parent(1 << 17, false));
    let child = module(PREMAP_CHILD);
    assert_eq!(run_premap_interp(&parent, &child), -22);
    assert_eq!(
        run_premap_jit(&parent, &child),
        -22,
        "EINVAL, not a trap, on the JIT too"
    );
}

/// #1956 — a detached child (`memory 16`) that reads a depth `d` from its payload. While `d > 0` it
/// spawns its own module detached from its own `"budget"`, with `d - 1` as the payload, joins it and
/// returns `10 * result + d`; at `d = 0` it returns 7. `orphan` makes the `d = 1` level return 5
/// without joining, and the `d = 0` level count down 200 000 first, so it outlives its parent.
fn nester(orphan: bool) -> String {
    let (join, leaf) = if orphan {
        (
            "vfive = i64.const 5\n  return vfive",
            "vn = i64.const 200000\n  br 3(vn)",
        )
    } else {
        (
            "vj = call.cap 6 1 (i32) -> (i64) vinst (vh)\n  vten = i64.const 10\n  vm = i64.mul vj vten\n  vr = i64.add vm vd1\n  return vr",
            "v7 = i64.const 7\n  return v7",
        )
    };
    format!(
        r#"memory 16
data 20000 "budget"
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  vab = i64.const {args}
  vd = i64.load vab
  vz = i64.const 0
  vleaf = i64.eq vd vz
  br_if vleaf 2() 1(v0, vd)
}}
block 1 (vi: i64, vd1: i64) {{
  vone = i64.const 1
  vnext = i64.sub vd1 vone
  vpp = i64.const 24576
  i64.store vpp vnext
  vnp = i64.const 20000
  vnl = i64.const 6
  vb = self.resolve vnp vnl
  vbw = i64.extend_i32_u vb
  vself = i64.const -1
  vzero = i64.const 0
  vlog = i64.const 16
  vpl = i64.const 8
  vinst = i32.wrap_i64 vi
  vh = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64, i64, i64) -> (i32) vinst (vbw, vself, vzero, vzero, vzero, vlog, vzero, vpp, vpl)
  {join}
}}
block 2 () {{
  {leaf}
}}
block 3 (vc: i64) {{
  vdec = i64.const 1
  vc1 = i64.sub vc vdec
  vdone = i64.eqz vc1
  br_if vdone 4() 3(vc1)
}}
block 4 () {{
  v7l = i64.const 7
  return v7l
  }}
}}
"#,
        args = temen_ir::module_args_base(),
    )
}

/// `v0` Instantiator, `v1` the [`nester`] module, `v2` the `Budget`: spawn it detached with depth `d`
/// as its payload, join it, and return its result.
fn nest_root(d: i64) -> String {
    format!(
        r#"memory 17
func (i32, i32, i32) -> (i64) {{
block 0 (v0: i32, v1: i32, v2: i32) {{
  vpp = i64.const 18432
  vd = i64.const {d}
  i64.store vpp vd
  vmh = i64.extend_i32_u v1
  vb = i64.extend_i32_u v2
  vz = i64.const 0
  vlog = i64.const 16
  vpl = i64.const 8
  vh = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vb, vmh, vz, vz, vz, vlog, vz, vpp, vpl)
  vj = call.cap 6 1 (i32) -> (i64) v0 (vh)
  return vj
  }}
}}
"#
    )
}

/// Run `f` on a thread of its own: a run that deadlocks fails here instead of hanging the suite.
fn within_a_minute<T: Send + 'static>(what: &str, f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.recv_timeout(std::time::Duration::from_secs(60))
        .unwrap_or_else(|_| panic!("{what}: no result within a minute (deadlocked?)"))
}

/// The `mem` room left along `budget`'s chain (Budget op 1, field 1).
fn mem_room(host: &mut Host, budget: i32) -> i64 {
    host.cap_dispatch_slots(cap_id::BUDGET, 1, budget, &[1], None)
        .expect("read")[0]
}

/// #1956 — a detached child spawns and joins detached children of its own on the JIT, as on the
/// interpreter, at every depth: each level has a nursery of its own, which spawns through the
/// child's own powerbox and pays from its own budget.
#[test]
fn detached_children_nest_three_deep_on_the_jit_as_on_the_interpreter() {
    let c = module(&nester(false));
    for (d, want) in [(0, 7), (1, 71), (2, 712), (3, 7123)] {
        let p = module(&nest_root(d));
        assert_eq!(run_interp(&p, &c, 1 << 20), want, "interpreter, depth {d}");
        assert_eq!(run_jit(&p, &c, 1 << 20), want, "the JIT, depth {d}");
    }
}

/// #1956 — a grandchild its parent never joins outlives it, on the JIT as on the interpreter: the
/// child returns while the grandchild still runs, and the grandchild's end still hands its window
/// back to the child's budget, so every byte of the run's budget is free again after the run. On the
/// JIT the grandchild's teardown reaches the ended child's host, which its nursery keeps alive.
#[test]
fn a_grandchild_outlives_its_parent_and_hands_its_window_back_on_the_jit() {
    let c = module(&nester(true));
    let p = module(&nest_root(1));
    let (mut host, h) = host(&c, 1 << 20);
    assert_eq!(
        interp_on(&p, &mut host, h),
        Ok(vec![Value::I64(5)]),
        "interpreter"
    );
    assert_eq!(
        mem_room(&mut host, h[2]),
        1 << 20,
        "interpreter: all returned"
    );
    let (mut host, h) = self::host(&c, 1 << 20);
    assert_eq!(
        jit_on(&p, &mut host, h),
        JitOutcome::Returned(vec![5]),
        "the JIT"
    );
    assert_eq!(mem_room(&mut host, h[2]), 1 << 20, "the JIT: all returned");
}

/// #1956 — under a run lane cap of 1, a child that joins its own child must step aside for it, on
/// the JIT as on the interpreter. The child runs as a task on the JIT's executor, so its `join`
/// parks the task (handing its worker and its lanes back) rather than blocking the worker, and the
/// grandchild, gated on every lane up to the root's, runs in its place.
#[test]
fn a_child_joining_its_child_under_a_lane_cap_of_one_steps_aside_on_the_jit() {
    let (interp, jit) = within_a_minute("depth 2 under a lane cap of 1", || {
        let c = module(&nester(false));
        let p = module(&nest_root(2));
        let capped = || {
            let (mut host, h) = host(&c, 1 << 20);
            host.set_lane_cap(1);
            (host, h)
        };
        (interp_result(&p, capped()), jit_outcome(&p, capped()))
    });
    assert_eq!(interp, Ok(vec![Value::I64(712)]), "interpreter");
    assert_eq!(jit, JitOutcome::Returned(vec![712]), "the JIT");
}

/// A detached child (`memory 16`) whose thread spawns a grandchild (func 2, which returns 5) from the
/// child's `"budget"` and joins it; the child joins the thread, which returns the result plus 100.
const CHILD_THREAD_SPAWNS: &str = r#"memory 16
data 20000 "budget"
func (i64) -> (i64) {
block 0 (v0: i64) {
  vz = i64.const 0
  vt = thread.spawn 1 vz v0
  vj = thread.join vt
  return vj
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, vi: i64) {
  vnp = i64.const 20000
  vnl = i64.const 6
  vb = self.resolve vnp vnl
  vbw = i64.extend_i32_u vb
  vself = i64.const -1
  vz = i64.const 0
  ve = i64.const 2
  vlog = i64.const 16
  vinst = i32.wrap_i64 vi
  vh = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64, i64, i64) -> (i32) vinst (vbw, vself, vz, vz, ve, vlog, vz, vz, vz)
  vj = call.cap 6 1 (i32) -> (i64) vinst (vh)
  vk = i64.const 100
  vr = i64.add vj vk
  return vr
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v = i64.const 5
  return v
  }
}
"#;

/// #1956 — a detached child's thread that joins a grandchild gives back its own lanes while it waits,
/// on the JIT as on the interpreter: the child's budget holds one lane, which the thread runs in and
/// the grandchild needs too.
#[test]
fn a_childs_thread_joining_a_grandchild_gives_back_the_childs_lane_on_the_jit() {
    let (interp, jit) = within_a_minute("a child's thread joining a grandchild", || {
        let c = module(CHILD_THREAD_SPAWNS);
        let p = module(SPAWN_JOIN);
        // The run's budget, narrowed to a node with one lane: the child's lane.
        let one_lane = || {
            let (mut host, mut h) = host(&c, 1 << 20);
            h[2] = host
                .cap_dispatch_slots(cap_id::BUDGET, 0, h[2], &[-1, 1 << 20, -1, -1, 1], None)
                .expect("split")[0] as i32;
            (host, h)
        };
        (interp_result(&p, one_lane()), jit_outcome(&p, one_lane()))
    });
    assert_eq!(interp, Ok(vec![Value::I64(105)]), "interpreter");
    assert_eq!(jit, JitOutcome::Returned(vec![105]), "the JIT");
}

/// A child that spins forever.
const CHILD_SPINS: &str = r#"memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  br 1()
}
block 1 () {
  br 1()
  }
}
"#;

/// A child parked forever on a word of its own window nobody notifies.
const CHILD_PARKS: &str = r#"memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  br 1()
}
block 1 () {
  va = i64.const 40000
  ve = i32.const 0
  vt = i64.const -1
  vs = i32.atomic.wait va ve vt
  br 1()
  }
}
"#;

/// A child whose thread is parked forever while its main joins the thread.
const CHILD_THREAD_PARKS: &str = r#"memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  vz = i64.const 0
  vt = thread.spawn 1 vz vz
  vr = thread.join vt
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  br 1()
}
block 1 () {
  va = i64.const 40000
  ve = i32.const 0
  vt = i64.const -1
  vs = i32.atomic.wait va ve vt
  br 1()
  }
}
"#;

/// #2074 — `kill` (op 12) ends a detached child however it is occupied, on the JIT as on the
/// interpreter: spinning, parked on a word of its own window, or with its thread parked while its main
/// joins it. The parent sleeps 50 ms so the child is spinning or parked first, kills it, and `wait`s:
/// each answers `THREAD_FAULT`. A kill that only flags a parked child leaves the `wait` hanging (or, on
/// the interpreter with nothing else alive, ends the run as a deadlock), so neither stands in for it.
#[test]
fn kill_ends_a_detached_child_spinning_or_parked() {
    let p = module(&parent_then(
        "vsa = i64.const 20000\n  \
         vse = i32.const 0\n  \
         vst = i64.const 50000000\n  \
         vsl = i32.atomic.wait vsa vse vst\n  \
         vk = call.cap 6 12 (i32) -> (i32) v0 (vh)\n  \
         vr = call.cap 6 18 (i32) -> (i64) v0 (vh)\n  \
         return vr",
    ));
    let fault = temen_ir::trap_code::THREAD_FAULT;
    let children = [
        ("spinning", CHILD_SPINS),
        ("parked", CHILD_PARKS),
        ("thread parked", CHILD_THREAD_PARKS),
    ];
    for (name, child) in children {
        let (p, c) = (p.clone(), module(child));
        assert_eq!(
            within_a_minute(&format!("interpreter, {name}"), move || interp_result(
                &p,
                host(&c, 1 << 20)
            )),
            Ok(vec![Value::I64(fault)]),
            "interpreter, {name}"
        );
    }
    for (name, child) in children {
        let (p, c) = (p.clone(), module(child));
        assert_eq!(
            within_a_minute(&format!("the JIT, {name}"), move || jit_outcome(
                &p,
                host(&c, 1 << 20)
            )),
            JitOutcome::Returned(vec![fault]),
            "the JIT, {name}"
        );
    }
}

/// A child that reads its attestation (a host call) and then spins.
const CHILD_CALLS_THEN_SPINS: &str = r#"memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  vz = i32.const 0
  vat = call.cap 4294967295 4 () -> (i64) vz ()
  br 1()
}
block 1 () {
  br 1()
  }
}
"#;

/// A child that makes a host call on every iteration of a loop that never ends.
const CHILD_SPINS_CALLING: &str = r#"memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  br 1()
}
block 1 () {
  vz = i32.const 0
  vat = call.cap 4294967295 4 () -> (i64) vz ()
  br 1()
  }
}
"#;

/// #2088 — a kill on the JIT survives the child's host calls: each successful `call.cap` resets the
/// trap cell, which used to erase a kill that landed during (or before) one, so the child ran on and
/// its parent's `wait` hung. The parent kills each child as soon as it is spawned; both answer
/// `THREAD_FAULT`, as on the interpreter.
#[test]
fn a_kill_survives_the_childs_host_calls_on_the_jit() {
    let p = module(&parent_then(
        "vk = call.cap 6 12 (i32) -> (i32) v0 (vh)\n  \
         vr = call.cap 6 18 (i32) -> (i64) v0 (vh)\n  \
         return vr",
    ));
    let fault = temen_ir::trap_code::THREAD_FAULT;
    for (name, child) in [
        ("a host call, then a spin", CHILD_CALLS_THEN_SPINS),
        ("a host call every iteration", CHILD_SPINS_CALLING),
    ] {
        let (pi, ci) = (p.clone(), module(child));
        assert_eq!(
            within_a_minute(&format!("interpreter, {name}"), move || interp_result(
                &pi,
                host(&ci, 1 << 20)
            )),
            Ok(vec![Value::I64(fault)]),
            "interpreter, {name}"
        );
        for round in 0..20 {
            let (pj, cj) = (p.clone(), module(child));
            assert_eq!(
                within_a_minute(&format!("the JIT, {name}"), move || jit_outcome(
                    &pj,
                    host(&cj, 1 << 20)
                )),
                JitOutcome::Returned(vec![fault]),
                "the JIT, {name}, round {round}"
            );
        }
    }
}

/// #1978 — a detached child that mints a 64 KiB `SharedRegion` through its own `AddressSpace`
/// (resolved by name), maps it at 65536 of its window, stores 41 through the mapping and loads it
/// back: it returns the map's status plus the word (41 when both work).
const MINTING_CHILD: &str = r#"memory 17
data 20000 "addrspace"
func (i64) -> (i64) {
block 0 (v0: i64) {
  vnp = i64.const 20000
  vnl = i64.const 9
  vas = self.resolve vnp vnl
  vlen = i64.const 65536
  vrh64 = call.cap 5 5 (i64) -> (i64) vas (vlen)
  vrh = i32.wrap_i64 vrh64
  vwo = i64.const 65536
  vro = i64.const 0
  vprot = i32.const 3
  vm = call.cap 4 0 (i64, i64, i64, i32) -> (i64) vrh (vwo, vro, vlen, vprot)
  vin = i64.const 41
  vz = i64.const 0
  vok = i64.eq vm vz
  br_if vok 1() 2(vm)
}
block 1 () {
  vat = i64.const 65536
  vword = i64.const 41
  i64.store vat vword
  vback = i64.load vat
  return vback
}
block 2 (verr: i64) {
  return verr
  }
}
"#;

/// `v0` Instantiator, `v1` the [`MINTING_CHILD`] module, `v2` the `Budget`: spawn it detached with a
/// window of `2^17`, join it, and return its result.
const MINTING_PARENT: &str = r#"memory 17
func (i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32) {
  vmh = i64.extend_i32_u v1
  vb = i64.extend_i32_u v2
  vz = i64.const 0
  vlog = i64.const 17
  vh = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vb, vmh, vz, vz, vz, vlog, vz, vz, vz)
  vj = call.cap 6 1 (i32) -> (i64) v0 (vh)
  return vj
  }
}
"#;

/// #1978 — a detached child maps a region it minted itself on the JIT, as on the interpreter. Its
/// host inherits the embedder's OS shared-memory region factory (`Host::child_host`); before, it
/// minted a software `VecBacking` the JIT cannot `mmap`, and the map answered `-EINVAL`.
#[test]
fn a_detached_child_maps_a_region_it_minted_on_the_jit_as_on_the_interpreter() {
    let parent = module(MINTING_PARENT);
    let child = module(MINTING_CHILD);
    let minting_host = || {
        let (mut host, h) = host(&child, 1 << 20);
        host.set_region_factory(temen_run::new_shared_region);
        (host, h)
    };
    let interp = match interp_result(&parent, minting_host())
        .expect("interp run")
        .first()
    {
        Some(Value::I64(x)) => *x,
        other => panic!("unexpected interp result {other:?}"),
    };
    assert_eq!(interp, 41, "the oracle maps the child's own region");
    let jit = match jit_outcome(&parent, minting_host()) {
        JitOutcome::Returned(ref v) => v.first().copied().unwrap_or(-1),
        ref o => panic!("jit ended abnormally: {o:?}"),
    };
    assert_eq!(jit, interp, "the JIT matches the oracle (was -22, #1978)");
}

/// #2219 — `v0` Instantiator, `v1` the child `Module`, `v2` the `Budget`: spawn the child's entry 0
/// detached by an op-17 record, with `grants` (each handle written as given) as its grant list. A
/// refused spawn returns its `-errno`; an admitted one is joined.
fn parent_granting(grants: &[(&str, u32)]) -> String {
    const REC: u64 = 17536;
    const GRANTS: u64 = 17664;
    const NAMES: u64 = 17920;
    let seg = |at: u64, bytes: &[u8]| {
        let esc: String = bytes.iter().map(|b| format!("\\x{b:02x}")).collect();
        format!("data {at} \"{esc}\"\n")
    };
    let rec = temen_ir::SpawnRec {
        grants_ptr: GRANTS,
        grants_n: grants.len() as u64,
        ..temen_ir::SpawnRec::v1(0)
    };
    let mut segments = seg(REC, &rec.encode());
    for (i, (name, handle)) in grants.iter().enumerate() {
        let name_at = NAMES + 32 * i as u64;
        let record = [
            (name_at as u32).to_le_bytes(),
            (name.len() as u32).to_le_bytes(),
            handle.to_le_bytes(),
            0u32.to_le_bytes(),
        ]
        .concat();
        segments += &seg(GRANTS + 16 * i as u64, &record);
        segments += &seg(name_at, name.as_bytes());
    }
    format!(
        r#"memory 17
func (i32, i32, i32) -> (i64) {{
block 0 (v0: i32, v1: i32, v2: i32) {{
  vma = i64.const {ma}
  i32.store vma v1
  vba = i64.const {ba}
  i32.store vba v2
  vrp = i64.const {REC}
  vh = call.cap 6 17 (i64) -> (i32) v0 (vrp)
  vz32 = i32.const 0
  vneg = i32.lt_s vh vz32
  br_if vneg 1(vh) 2(v0, vh)
}}
block 1 (ve: i32) {{
  vr = i64.extend_i32_s ve
  return vr
}}
block 2 (vi: i32, vc: i32) {{
  vr = call.cap 6 1 (i32) -> (i64) vi (vc)
  return vr
  }}
}}
{segments}"#,
        ma = REC + 24,
        ba = REC + 28,
    )
}

/// #2219 — the JIT honors an empty grant as the interpreter does: a child whose `exit` nothing grants
/// is refused, and noted by name, until the parent empties it, by name, by `*` or by a prefix.
#[test]
fn an_empty_grant_admits_a_child_on_the_jit_as_on_the_interpreter() {
    let c = module(CHILD_IMPORTS_EXIT);
    let empty = temen_interp::GRANT_EMPTY;
    for (grants, want) in [
        (vec![], -22),
        (vec![("exit", empty)], 42),
        (vec![("*", empty)], 42),
        (vec![("ex*", empty)], 42),
    ] {
        let p = module(&parent_granting(&grants));
        let (mut ih, h) = host(&c, 1 << 20);
        let interp = interp_on(&p, &mut ih, h).expect("interp run");
        assert_eq!(interp, vec![Value::I64(want)], "interpreter oracle, {grants:?}");
        let (mut jh, h) = host(&c, 1 << 20);
        let jit = jit_on(&p, &mut jh, h);
        assert!(
            matches!(jit, JitOutcome::Returned(ref v) if v == &[want]),
            "the JIT, {grants:?}: {jit:?}"
        );
        let noted = |h: &Host| h.take_notes().iter().any(|n| n.contains("`exit`"));
        assert_eq!(noted(&ih), want < 0, "the interpreter's note, {grants:?}");
        assert_eq!(noted(&jh), want < 0, "the JIT's note, {grants:?}");
    }
}
