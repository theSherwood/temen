//! CALLS.md 4b.1 — **promotion**: an animated offer handler that parks mid-run.
//!
//! A `single` library provider's offer handler runs inline on the caller's vCPU as a reified
//! fiber over the provider's world (4a). When the handler hits a blocking point it must
//! **promote**: the provider `{mem, host, fuel}` are handed back to the instance (`busy` reopens,
//! §10.1 atomicity window closes) and the caller parks as the handler's resumer; the handler's
//! block-wake re-admits the caller, which re-acquires the world and resumes the handler to
//! completion. 4a stopped short of this (a mid-animation park was fail-closed until 4b), so these
//! tests exercise the piece the promotion machinery adds.

use std::time::{Duration, Instant};
use temen_interp::{run_with_host, Host, Value};

fn module(text: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(text).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// A provider whose single op timed-`atomic.wait`s on a zero cell (so it always parks — the
/// value matches the expected), then returns the wait status. Nobody notifies, so it resumes on
/// the **timeout**: a park that deterministically drives promotion end to end (park → timer wake
/// → resume → settle) with no cross-thread ordering to arrange.
fn timed_wait_provider() -> temen_ir::Module {
    module(
        "memory 16\n\
         func () -> (i64) {\n\
         block 0 () {\n\
           vaddr = i64.const 16384\n\
           vexp = i32.const 0\n\
           vto = i64.const 2000000\n\
           vst = i32.atomic.wait vaddr vexp vto\n\
           vst64 = i64.extend_i32_s vst\n\
           return vst64\n\
           }\n\
         }\n",
    )
}

#[test]
fn an_animated_handler_that_parks_promotes_and_resumes_on_its_timer() {
    let provider = timed_wait_provider();
    let mut h = Host::new();
    let offer = h.wire_offer_proc(&provider, &[0]).expect("instanced offer");
    let tid = h.resolve_offer(offer).unwrap().type_id;

    // A consumer that `cap.call`s the wired instanced offer once. The call animates the handler
    // (4a), the handler parks on its timed wait (promotion), and the 2 ms timer resumes it.
    let consumer_src = format!(
        "memory 16\n\
         func () -> (i64) {{\n\
         block 0 () {{\n\
           vh = i32.const {offer}\n\
           vr = cap.call {tid} 0 () -> (i64) vh ()\n\
           return vr\n\
           }}\n\
         }}\n"
    );
    let consumer = module(&consumer_src);
    let mut fuel = 100_000_000u64;
    let r = run_with_host(&consumer, 0, &[], &mut fuel, &mut h);
    // WAIT_TIMED_OUT == 2: the handler parked, its timer fired, and it resumed past the wait and
    // returned the status — proving the whole promotion round trip (a 4a mid-animation park would
    // have failed closed / corrupted the caller's stack instead).
    assert_eq!(
        r,
        Ok(vec![Value::I64(2)]),
        "the promoted handler resumed on its timer and returned WAIT_TIMED_OUT"
    );
}

/// The same provider called **twice in a row** by one caller: after the first dispatch promotes,
/// parks, and resumes to completion, `busy` must have reopened so the second dispatch is admitted
/// (not stranded `-EAGAIN`). Both return the timed-out status.
#[test]
fn a_promoted_dispatch_reopens_admission_for_the_next_call() {
    let provider = timed_wait_provider();
    let mut h = Host::new();
    let offer = h.wire_offer_proc(&provider, &[0]).expect("instanced offer");
    let tid = h.resolve_offer(offer).unwrap().type_id;

    let consumer_src = format!(
        "memory 16\n\
         func () -> (i64) {{\n\
         block 0 () {{\n\
           vh = i32.const {offer}\n\
           v1 = cap.call {tid} 0 () -> (i64) vh ()\n\
           v2 = cap.call {tid} 0 () -> (i64) vh ()\n\
           vsum = i64.add v1 v2\n\
           return vsum\n\
           }}\n\
         }}\n"
    );
    let consumer = module(&consumer_src);
    let mut fuel = 100_000_000u64;
    let r = run_with_host(&consumer, 0, &[], &mut fuel, &mut h);
    assert_eq!(
        r,
        Ok(vec![Value::I64(4)]),
        "both dispatches promoted, resumed, and returned 2 — admission reopened between them"
    );
}

/// A provider whose single op waits on a zero cell **forever** (timeout `-1`, never notified) —
/// its handler promotes and stays parked until the run tears down.
fn forever_wait_provider() -> temen_ir::Module {
    module(
        "memory 16\n\
         func () -> (i64) {\n\
         block 0 () {\n\
           vaddr = i64.const 16384\n\
           vexp = i32.const 0\n\
           vto = i64.const -1\n\
           vst = i32.atomic.wait vaddr vexp vto\n\
           vst64 = i64.extend_i32_s vst\n\
           return vst64\n\
           }\n\
         }\n",
    )
}

/// CALLS.md 4b.3 — **teardown abandons a promoted daemon.** The root spawns a daemon thread that
/// calls an offer whose handler waits forever; the daemon promotes and parks (its provider world
/// handed back, itself filed in `svc_waiters` under the provider domain). The root then returns,
/// which tears the whole run down — the parked daemon must be reaped and the run must end
/// **promptly**, never waiting out the daemon's (infinite) block. Exercises `reap` on a vCPU
/// carrying an `OfferParked` record and the run-teardown `svc_waiters` drain.
#[test]
fn teardown_abandons_a_promoted_daemon() {
    let provider = forever_wait_provider();
    let mut h = Host::new();
    let offer = h.wire_offer_proc(&provider, &[0]).expect("instanced offer");
    let tid = h.resolve_offer(offer).unwrap().type_id;

    // Root (func 0): spawn the daemon (func 1), briefly wait so it can promote+park, then return
    // 42 — the C rule (`main` returning is `exit`), ending the run with the daemon still parked.
    // Daemon (func 1): `cap.call` the forever-waiting offer, which animates and promotes.
    let src = format!(
        "memory 16\n\
         func () -> (i64) {{\n\
         block 0 () {{\n\
           v0 = i64.const 0\n\
           v1 = thread.spawn 1 v0 v0\n\
           v2 = i64.const 16392\n\
           v3 = i32.const 0\n\
           v4 = i64.const 5000000\n\
           v5 = i32.atomic.wait v2 v3 v4\n\
           v6 = i64.const 42\n\
           return v6\n\
           }}\n\
         }}\n\
         func (i64, i64) -> (i64) {{\n\
         block 0 (vsp: i64, varg: i64) {{\n\
           vh = i32.const {offer}\n\
           vr = cap.call {tid} 0 () -> (i64) vh ()\n\
           return vr\n\
           }}\n\
         }}\n"
    );
    let m = module(&src);
    let mut fuel = 100_000_000u64;
    let t0 = Instant::now();
    let r = run_with_host(&m, 0, &[], &mut fuel, &mut h);
    let took = t0.elapsed();
    assert_eq!(
        r,
        Ok(vec![Value::I64(42)]),
        "the root returns; the promoted daemon is abandoned, not awaited"
    );
    assert!(
        took < Duration::from_secs(5),
        "teardown must not wait out the promoted daemon's infinite block (took {took:?})"
    );
}

/// A provider whose single op just returns a constant — a handler that never parks.
fn const_provider(value: i64) -> temen_ir::Module {
    module(&format!(
        "memory 16\n\
         func () -> (i64) {{\n\
         block 0 () {{\n\
           vc = i64.const {value}\n\
           return vc\n\
           }}\n\
         }}\n"
    ))
}

/// A provider whose single op **calls another instanced offer** (granted into its own domain as
/// `inner`) and returns that result + 100 — an offer handler that is itself a caller, so the
/// animation nests over its own. `inner_tid` is the interned interface type id of the inner op.
fn outer_provider(inner_tid: u32) -> temen_ir::Module {
    module(&format!(
        "memory 16\n\
         data 16384 \"inner\"\n\
         func () -> (i64) {{\n\
         block 0 () {{\n\
           vp = i64.const 16384\n\
           vn = i64.const 5\n\
           vh = cap.self.resolve vp vn\n\
           vr = cap.call {inner_tid} 0 () -> (i64) vh ()\n\
           v100 = i64.const 100\n\
           vsum = i64.add vr v100\n\
           return vsum\n\
           }}\n\
         }}\n"
    ))
}

/// Wire `inner` and `outer` as instanced offers, grant `inner` into `outer`'s domain as `inner`,
/// and drive one consumer `cap.call` into `outer`. Returns the run result.
fn run_nested(
    inner: &temen_ir::Module,
    outer_of: impl Fn(u32) -> temen_ir::Module,
) -> Result<Vec<Value>, temen_interp::Trap> {
    let mut h = Host::new();
    let inner_offer = h.wire_offer_proc(inner, &[0]).expect("inner offer");
    let inner_tid = h.resolve_offer(inner_offer).unwrap().type_id;
    let outer = outer_of(inner_tid);
    let outer_offer = h.wire_offer_proc(&outer, &[0]).expect("outer offer");
    let outer_tid = h.resolve_offer(outer_offer).unwrap().type_id;
    h.grant_impl_cap(outer_offer, inner_offer, "inner")
        .expect("inner offer re-grants into the outer provider");

    let consumer_src = format!(
        "memory 16\n\
         func () -> (i64) {{\n\
         block 0 () {{\n\
           vh = i32.const {outer_offer}\n\
           vr = cap.call {outer_tid} 0 () -> (i64) vh ()\n\
           return vr\n\
           }}\n\
         }}\n"
    );
    let consumer = module(&consumer_src);
    let mut fuel = 100_000_000u64;
    run_with_host(&consumer, 0, &[], &mut fuel, &mut h)
}

/// **Nested run-to-completion**: an offer handler that calls another offer (neither parks). The
/// animation stack must nest — with the pre-4b.2 single-slot `offer_anim` the inner animation
/// clobbered the outer's, so the outer's return was mis-settled. Outer returns inner(7) + 100.
#[test]
fn a_handler_that_calls_another_offer_nests_the_animation() {
    let r = run_nested(&const_provider(7), outer_provider);
    assert_eq!(
        r,
        Ok(vec![Value::I64(107)]),
        "the nested (non-parking) inner offer settled under the outer animation"
    );
}

/// **Nested promotion** (the §1 cross-domain chain): the outer handler calls an inner offer whose
/// handler parks (timed wait) and promotes while the outer animation is still on the stack. The
/// inner resumes on its timer, returns, the outer continues and returns inner(2) + 100.
#[test]
fn a_nested_inner_handler_promotes_under_its_caller_handlers_animation() {
    let r = run_nested(&timed_wait_provider(), outer_provider);
    assert_eq!(
        r,
        Ok(vec![Value::I64(102)]),
        "the inner handler promoted+resumed while the outer animation stayed on the stack"
    );
}

// ---- CALLS.md 4c: queue-on-contention ----

/// A provider whose op `rec(depth)` **calls its own offer** (granted into itself as `me`) with
/// `depth - 1` until `depth == 0`, returning `depth` (each frame adds 1 to the base 0). A bounded
/// **self-re-entrant** handler: pre-4c the inner self-call found the instance busy and answered
/// `-EAGAIN`; with 4c.2 it runs as an ordinary recursive call over the provider's own world and
/// completes. `tid` is the interned `(i64) -> (i64)` interface id (embedded in the self-call).
fn self_recursive_provider(tid: u32) -> temen_ir::Module {
    module(&format!(
        "memory 16\n\
         data 16384 \"me\"\n\
         func (i64) -> (i64) {{\n\
         block 0 (vd: i64) {{\n\
           vzero = i64.const 0\n\
           vcond = i64.ne vd vzero\n\
           br_if vcond 1(vd) 2()\n\
         }}\n\
         block 1 (vd: i64) {{\n\
           vone = i64.const 1\n\
           vd1 = i64.sub vd vone\n\
           vp = i64.const 16384\n\
           vn = i64.const 2\n\
           vh = cap.self.resolve vp vn\n\
           vr = cap.call {tid} 0 (i64) -> (i64) vh (vd1)\n\
           vres = i64.add vr vone\n\
           return vres\n\
           }}\n\
         block 2 () {{\n\
           vzero = i64.const 0\n\
           return vzero\n\
           }}\n\
         }}\n"
    ))
}

/// **4c.2 — cyclic completion.** A guest `cap.call`s an offer whose handler re-enters its own
/// instance to a bounded depth. The direct self-call is no longer a `-EAGAIN` deadlock artifact:
/// it recurses over the provider's own world and completes, returning the depth (3).
#[test]
fn a_self_reentrant_offer_completes_instead_of_eagain() {
    // Wire once to learn the interned type id, embed it in the self-call, re-wire the real module.
    let mut h = Host::new();
    let probe = h
        .wire_offer_proc(&self_recursive_provider(0), &[0])
        .expect("probe offer");
    let tid = h.resolve_offer(probe).unwrap().type_id;

    let mut h = Host::new();
    let offer = h
        .wire_offer_proc(&self_recursive_provider(tid), &[0])
        .expect("self-recursive offer");
    let otid = h.resolve_offer(offer).unwrap().type_id;
    assert_eq!(
        otid, tid,
        "the interned (i64)->(i64) interface id is stable"
    );
    h.grant_impl_cap(offer, offer, "me")
        .expect("the offer re-grants into its own provider");

    let consumer_src = format!(
        "memory 16\n\
         func () -> (i64) {{\n\
         block 0 () {{\n\
           vh = i32.const {offer}\n\
           vd = i64.const 3\n\
           vr = cap.call {tid} 0 (i64) -> (i64) vh (vd)\n\
           return vr\n\
           }}\n\
         }}\n"
    );
    let consumer = module(&consumer_src);
    let mut fuel = 100_000_000u64;
    let r = run_with_host(&consumer, 0, &[], &mut fuel, &mut h);
    assert_eq!(
        r,
        Ok(vec![Value::I64(3)]),
        "the self-re-entrant offer recursed to depth 3 and completed (no -EAGAIN, no deadlock)"
    );
}

/// A provider whose op spins a bounded compute loop (holding the instance `busy` without parking),
/// then returns 1 — long enough that a concurrent second caller reliably contends.
fn busy_loop_provider() -> temen_ir::Module {
    module(
        "memory 16\n\
         func () -> (i64) {\n\
         block 0 () {\n\
           vk0 = i32.const 2000000\n\
           br 1(vk0)\n\
         }\n\
         block 1 (vk: i32) {\n\
           vone = i32.const 1\n\
           vk2 = i32.sub vk vone\n\
           br_if vk2 1(vk2) 2()\n\
         }\n\
         block 2 () {\n\
           vr = i64.const 1\n\
           return vr\n\
           }\n\
         }\n",
    )
}

/// **4c.1 — distinct-vCPU contention.** Two threads of one domain call the same `single` instance;
/// while one holds it `busy` (a compute loop), the other must **park + wake-retry** rather than
/// receive `-EAGAIN`. Each thread makes a *single* call (no `-EAGAIN` retry loop), so a `-11`
/// anywhere would surface as a wrong sum — both must observe the real result `1`. The root sums
/// its own call with the spawned thread's joined result: 1 + 1 = 2.
#[test]
fn concurrent_callers_park_and_retry_instead_of_eagain() {
    let provider = busy_loop_provider();
    let mut h = Host::new();
    let offer = h.wire_offer_proc(&provider, &[0]).expect("instanced offer");
    let tid = h.resolve_offer(offer).unwrap().type_id;

    // Root (func 0): spawn a sibling (func 1) that calls the offer, then call it itself, join the
    // sibling, and return the sum of both results. Sibling (func 1): one call, return its result.
    let src = format!(
        "memory 16\n\
         func () -> (i64) {{\n\
         block 0 () {{\n\
           v0 = i64.const 0\n\
           vt = thread.spawn 1 v0 v0\n\
           vh = i32.const {offer}\n\
           vmine = cap.call {tid} 0 () -> (i64) vh ()\n\
           vjoined = thread.join vt\n\
           vsum = i64.add vmine vjoined\n\
           return vsum\n\
           }}\n\
         }}\n\
         func (i64, i64) -> (i64) {{\n\
         block 0 (vsp: i64, varg: i64) {{\n\
           vh = i32.const {offer}\n\
           vr = cap.call {tid} 0 () -> (i64) vh ()\n\
           return vr\n\
           }}\n\
         }}\n"
    );
    let m = module(&src);
    let mut fuel = 1_000_000_000u64;
    let r = run_with_host(&m, 0, &[], &mut fuel, &mut h);
    assert_eq!(
        r,
        Ok(vec![Value::I64(2)]),
        "both concurrent callers completed with the real result (parked+retried, never -EAGAIN)"
    );
}

/// A provider whose handler (func 0) calls its own helper (func 1) **through `ref.func` +
/// `call_indirect`** — a unit-own-funcref handler. handler(x) = helper(x) + 10 = 2x + 10.
fn ref_func_provider() -> temen_ir::Module {
    module(
        "memory 16\n\
         func (i64) -> (i64) {\n\
         block 0 (va: i64) {\n\
           vf = ref.func 1\n\
           vr = call_indirect (i64) -> (i64) vf (va)\n\
           vten = i64.const 10\n\
           vs = i64.add vr vten\n\
           return vs\n\
           }\n\
         }\n\
         func (i64) -> (i64) {\n\
         block 0 (va: i64) {\n\
           vtwo = i64.const 2\n\
           vm = i64.mul va vtwo\n\
           return vm\n\
           }\n\
         }\n",
    )
}

/// CALLS.md 6a — a `ref.func`-taking offer handler **animates** (pre-6a it declined to the now-
/// retired `drive_arc` sub-run; pure conservatism, pinned by nothing until this test): the
/// animation installs the unit-own funcref remap (`install_unit_funcs`, cached per vCPU), so the
/// handler's `call_indirect` through `ref.func 1` resolves to its own helper — never the caller's
/// table. Called **twice**: the second call exercises the cache path (installs are run-permanent;
/// a re-install per call would leak table slots).
#[test]
fn a_ref_func_offer_handler_animates_with_the_unit_remap() {
    let provider = ref_func_provider();
    let mut host = Host::new();
    let offer = host.wire_offer_proc(&provider, &[0]).expect("offer");
    let tid = host.resolve_offer(offer).unwrap().type_id;

    let consumer_src = format!(
        "func () -> (i64) {{\n\
         block 0 () {{\n\
           vh = i32.const {offer}\n\
           va = i64.const 16\n\
           v1 = cap.call {tid} 0 (i64) -> (i64) vh (va)\n\
           v2 = cap.call {tid} 0 (i64) -> (i64) vh (va)\n\
           vs = i64.add v1 v2\n\
           return vs\n\
           }}\n\
         }}\n"
    );
    let consumer = module(&consumer_src);
    let mut fuel = 10_000_000u64;
    let r = run_with_host(&consumer, 0, &[], &mut fuel, &mut host);
    assert_eq!(
        r,
        Ok(vec![Value::I64(84)]),
        "handler(16) = 2*16+10 = 42, twice — the ref.func remap resolved both animated calls"
    );
}
