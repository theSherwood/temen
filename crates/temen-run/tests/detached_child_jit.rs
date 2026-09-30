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
    let tail = if join {
        "vr = call.cap 6 1 (i32) -> (i64) v0 (vh)\n  return vr"
    } else {
        "vr = i64.extend_i32_s vh\n  return vr"
    };
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
    let args = [h[0] as i64, h[1] as i64, h[2] as i64];
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
    let mut fuel = 50_000_000u64;
    run_with_host(
        parent,
        0,
        &[Value::I32(h[0]), Value::I32(h[1]), Value::I32(h[2])],
        &mut fuel,
        &mut host,
    )
}

#[test]
fn a_detached_child_on_the_jit_matches_the_interpreter() {
    let p = module(&parent(true));
    let c = module(CHILD);
    let want = ARGV_WORD + 1; // argv landed; attest = 1 (tier 1, window_exposed = false)
    assert_eq!(run_interp(&p, &c, 1 << 16), want, "interpreter oracle");
    let before = temen_jit::child_compiles();
    assert_eq!(
        run_jit(&p, &c, 1 << 16),
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
