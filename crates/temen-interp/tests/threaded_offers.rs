//! CALLS.md increment 7 — the **`threaded`** concurrency policy (§5 axis 2, §10.1): a provider's
//! opt-in declaration that its handlers may run concurrently, with no admission gate. A `Threaded`
//! instanced offer animates every caller immediately over a `fork_for_thread` **view** of its
//! window (never taken, `busy` never set); the provider synchronizes its own state under the §12
//! defined-race regime.
//!
//! The observable that pins "no gate" is **contention**, not a direct self-call: a top-of-stack
//! self-re-entrant call recurses under *both* policies (4c.2), so the distinguishing shapes are a
//! **buried** re-entry (the instance on the animation stack but not the top — `single` answers
//! `-EAGAIN`, `threaded` admits) and genuinely **concurrent** callers on distinct vCPUs.

use temen_interp::{run_with_host, Host, OfferPolicy, Value};

fn module(t: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(t).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// A mutually-recursive forwarding handler, used for **both** providers X and Y: `f(other, self,
/// n)` returns `7` when `n == 0`, else `1 + other.op0(self, other, n − 1)`. Calling X with `n = 2`
/// drives `X → Y → X` — the inner X call finds X **buried** on the animation stack (`[X, Y]`, not
/// the top), the exact shape 4c.2's direct-recursion arm does not cover.
const FORWARDER: &str = "\
memory 16
func (cap, cap, i64) -> (i64) {
block 0 (vo: cap, vs: cap, vn: i64) {
  vn32 = i32.wrap_i64 vn
  br_if vn32 1(vo, vs, vn) 2()
}
block 1 (vo: cap, vs: cap, vn: i64) {
  vone = i64.const 1
  vn1 = i64.sub vn vone
  vr = cap.call 268435456 0 (cap, cap, i64) -> (i64) vo (vs, vo, vn1)
  vrr = i64.add vr vone
  return vrr
}
block 2 () {
  vseven = i64.const 7
  return vseven
  }
}
";

fn buried_cycle_consumer(x: i32, y: i32, tid: u32) -> temen_ir::Module {
    module(&format!(
        "func () -> (i64) {{\n\
         block 0 () {{\n\
           vhx = i32.const {x}\n\
           vhy = i32.const {y}\n\
           vn = i64.const 2\n\
           vr = cap.call {tid} 0 (cap, cap, i64) -> (i64) vhx (vhy, vhx, vn)\n\
           return vr\n\
           }}\n\
         }}\n"
    ))
}

/// **The 7.1 pin.** X wired `Threaded`, Y `Single`: the buried re-entry into X is admitted as a
/// second concurrent animation of X (`[X, Y, X]` on one vCPU, each over its own forked view), so
/// the chain completes: `1 + (1 + 7) = 9`. Under `single` this exact call answers `-EAGAIN` (the
/// control below) — the cleanest single-threaded observable of "no admission gate".
#[test]
fn a_threaded_provider_admits_a_buried_reentrant_call() {
    let forwarder = module(FORWARDER);
    let mut h = Host::new();
    let x = h
        .wire_offer_proc_with_policy(&forwarder, &[0], OfferPolicy::Threaded)
        .expect("threaded offer");
    let y = h.wire_offer_proc(&forwarder, &[0]).expect("single offer");
    let tid = h.resolve_offer(x).unwrap().type_id;
    assert_eq!(
        tid,
        h.resolve_offer(y).unwrap().type_id,
        "one shape, one id"
    );

    let consumer = buried_cycle_consumer(x, y, tid);
    let mut fuel = 100_000_000u64;
    let r = run_with_host(&consumer, 0, &[], &mut fuel, &mut h);
    assert_eq!(
        r,
        Ok(vec![Value::I64(9)]),
        "X(2) -> Y(1) -> X(0): the buried X call was admitted concurrently and the chain summed"
    );
}

/// **The control**: the identical cycle with X wired `Single` refuses the buried re-entry with a
/// probeable `-EAGAIN` (4c.2's recorded residue), which the handlers' `+1` arithmetic carries out
/// as `1 + (1 + (-11)) = -9`. Together with the pin above this isolates the policy as the one
/// variable.
#[test]
fn a_single_provider_still_refuses_the_buried_reentrant_call() {
    let forwarder = module(FORWARDER);
    let mut h = Host::new();
    let x = h.wire_offer_proc(&forwarder, &[0]).expect("single offer");
    let y = h.wire_offer_proc(&forwarder, &[0]).expect("single offer");
    let tid = h.resolve_offer(x).unwrap().type_id;

    let consumer = buried_cycle_consumer(x, y, tid);
    let mut fuel = 100_000_000u64;
    let r = run_with_host(&consumer, 0, &[], &mut fuel, &mut h);
    assert_eq!(
        r,
        Ok(vec![Value::I64(-9)]),
        "single X refuses the buried re-entry: 1 + (1 + EAGAIN) = -9"
    );
}

/// A stateful provider for the concurrent pin: op 0 `store(addr, val)` writes into the provider's
/// own window and returns `val`; op 1 `load(addr)` reads it back. Distinct callers write distinct
/// cells — no data race, but both animations must run over the **same** shared instance window.
const CELL_STORE: &str = "\
memory 16
func (i64, i64) -> (i64) {
block 0 (vaddr: i64, vval: i64) {
  i64.store vaddr vval
  return vval
  }
}
func (i64) -> (i64) {
block 0 (vaddr: i64) {
  vv = i64.load vaddr
  return vv
  }
}
";

/// **The 7.2 pin.** Two `thread.spawn`ed vCPUs each call one `Threaded` instance concurrently
/// (each animation over its own `fork_for_thread` view — the `thread.spawn` sibling shape); after
/// joining, the main thread reads both cells back through the same offer. Asserts concurrent
/// cross-vCPU admission lands both writes in the one shared provider window — safety and
/// shared-instance semantics, not timing (the pin passes whether or not the animations physically
/// overlap).
#[test]
fn concurrent_callers_share_one_threaded_instance() {
    let provider = module(CELL_STORE);
    let mut h = Host::new();
    let offer = h
        .wire_offer_proc_with_policy(&provider, &[0, 1], OfferPolicy::Threaded)
        .expect("threaded offer");
    let tid = h.resolve_offer(offer).unwrap().type_id;

    // Thread body (func 1): `arg i ∈ {0, 1}` picks cell `16 + 8i` and value `10 + i`; the store
    // returns the value, which the thread returns to its joiner.
    let consumer_src = format!(
        "memory 16\n\
         func () -> (i64) {{\n\
         block 0 () {{\n\
           vsp = i64.const 4096\n\
           va0 = i64.const 0\n\
           vt0 = thread.spawn 1 vsp va0\n\
           va1 = i64.const 1\n\
           vt1 = thread.spawn 1 vsp va1\n\
           vj0 = thread.join vt0\n\
           vj1 = thread.join vt1\n\
           vh = i32.const {offer}\n\
           vc0 = i64.const 16\n\
           vr0 = cap.call {tid} 1 (i64) -> (i64) vh (vc0)\n\
           vc1 = i64.const 24\n\
           vr1 = cap.call {tid} 1 (i64) -> (i64) vh (vc1)\n\
           vsum = i64.add vr0 vr1\n\
           return vsum\n\
           }}\n\
         }}\n\
         func (i64, i64) -> (i64) {{\n\
         block 0 (vsp: i64, vi: i64) {{\n\
           vten = i64.const 10\n\
           vval = i64.add vi vten\n\
           veight = i64.const 8\n\
           voff = i64.mul vi veight\n\
           vbase = i64.const 16\n\
           vaddr = i64.add vbase voff\n\
           vh = i32.const {offer}\n\
           vr = cap.call {tid} 0 (i64, i64) -> (i64) vh (vaddr, vval)\n\
           return vr\n\
           }}\n\
         }}\n"
    );
    let consumer = module(&consumer_src);
    let mut fuel = 100_000_000u64;
    let r = run_with_host(&consumer, 0, &[], &mut fuel, &mut h);
    assert_eq!(
        r,
        Ok(vec![Value::I64(21)]),
        "both spawned callers' writes (10 and 11) landed in the one shared instance window"
    );
}

/// **I69 regression.** Many concurrent callers on distinct vCPUs each store to their **own** cell
/// through one `Threaded` instance; after joining, the main thread sums every cell back. The pin is
/// that **no caller is silently dropped**: before the fix, two concurrent admissions could collide
/// on the provider's brief snapshot lock (`state_.try_lock()`), and the loser was handed a spurious
/// `-EAGAIN` — its `store` never ran, its cell stayed `0`, and the sum came up short (the observed
/// `left: 10, right: 21` on the two-caller pin). A `Threaded` provider promises **no admission
/// gate** (§10.1), so every one of the N stores must land. Ten callers widen the collision window so
/// a regression fails deterministically under the parallel suite rather than ~1-in-many.
#[test]
fn concurrent_callers_never_lose_an_admission() {
    const N: i64 = 10;
    let provider = module(CELL_STORE);
    let expected = N * (N + 1) / 2;

    // Build the consumer once the offer handle/type-id are known: root spawns N vCPUs (each `func 1`
    // stores `i + 1` into cell `16 + 8i`), joins them all, then sums the cells back.
    let build_consumer = |offer: i32, tid: u32| -> temen_ir::Module {
        let mut body = String::from(
            "memory 16\n\
             func () -> (i64) {\n\
             block 0 () {\n\
               vsp = i64.const 4096\n",
        );
        for i in 0..N {
            body.push_str(&format!(
                "               va{i} = i64.const {i}\n               vt{i} = thread.spawn 1 vsp va{i}\n"
            ));
        }
        for i in 0..N {
            body.push_str(&format!("               vj{i} = thread.join vt{i}\n"));
        }
        body.push_str(&format!("               vh = i32.const {offer}\n"));
        body.push_str("               vacc0 = i64.const 0\n");
        for i in 0..N {
            let addr = 16 + 8 * i;
            let next = i + 1;
            body.push_str(&format!(
                "               vc{i} = i64.const {addr}\n               vr{i} = cap.call {tid} 1 (i64) -> (i64) vh (vc{i})\n               vacc{next} = i64.add vacc{i} vr{i}\n"
            ));
        }
        body.push_str(&format!("               return vacc{N}\n"));
        body.push_str(
            "               }\n\
             }\n\
             func (i64, i64) -> (i64) {\n\
             block 0 (vsp: i64, vi: i64) {\n\
               vone = i64.const 1\n\
               vval = i64.add vi vone\n\
               veight = i64.const 8\n\
               voff = i64.mul vi veight\n\
               vbase = i64.const 16\n\
               vaddr = i64.add vbase voff\n",
        );
        body.push_str(&format!(
            "               vh = i32.const {offer}\n\
               vr = cap.call {tid} 0 (i64, i64) -> (i64) vh (vaddr, vval)\n\
               return vr\n\
               }}\n\
             }}\n"
        ));
        module(&body)
    };

    // Rerun with a fresh host to widen the odds of hitting the concurrent-admission window at least
    // once (`Host` is not `Clone`, so re-wire each iteration — the ids are deterministic).
    for _ in 0..50 {
        let mut h = Host::new();
        let offer = h
            .wire_offer_proc_with_policy(&provider, &[0, 1], OfferPolicy::Threaded)
            .expect("threaded offer");
        let tid = h.resolve_offer(offer).unwrap().type_id;
        let consumer = build_consumer(offer, tid);
        let mut fuel = 100_000_000u64;
        let r = run_with_host(&consumer, 0, &[], &mut fuel, &mut h);
        assert_eq!(
            r,
            Ok(vec![Value::I64(expected)]),
            "every concurrent threaded caller's write must land — none dropped to -EAGAIN"
        );
    }
}

/// Repeated sequential calls through a `Threaded` instance keep state across calls exactly as a
/// `single` instance does — the policy changes admission, never the shared-instance semantics
/// (§10.1: what is admitted is a dispatch; the world is the same one world).
#[test]
fn a_threaded_instance_keeps_state_across_calls() {
    let provider = module(CELL_STORE);
    let mut h = Host::new();
    let offer = h
        .wire_offer_proc_with_policy(&provider, &[0, 1], OfferPolicy::Threaded)
        .expect("threaded offer");
    let tid = h.resolve_offer(offer).unwrap().type_id;

    let consumer_src = format!(
        "func () -> (i64) {{\n\
         block 0 () {{\n\
           vh = i32.const {offer}\n\
           vaddr = i64.const 32\n\
           vval = i64.const 41\n\
           vs = cap.call {tid} 0 (i64, i64) -> (i64) vh (vaddr, vval)\n\
           vr = cap.call {tid} 1 (i64) -> (i64) vh (vaddr)\n\
           vone = i64.const 1\n\
           vrr = i64.add vr vone\n\
           return vrr\n\
           }}\n\
         }}\n"
    );
    let consumer = module(&consumer_src);
    let mut fuel = 100_000_000u64;
    let r = run_with_host(&consumer, 0, &[], &mut fuel, &mut h);
    assert_eq!(
        r,
        Ok(vec![Value::I64(42)]),
        "a later call reads what an earlier call wrote — one persistent instance"
    );
}

/// CALLS.md 7.3 — the **host-side tier** (a direct `cap_dispatch_slots`, the embedder/JIT/durable
/// route) admits a `Threaded` instance by running the handler in a sub-run over the instance's
/// live shared cell (`drive_arc_shared`) — under 7.1 this exact dispatch answered `-EAGAIN`.
/// Two sequential dispatches also pin that the host-side gate (one sub-run per instance at a
/// time) reopens after each settle.
#[test]
fn the_host_side_tier_admits_a_threaded_instance() {
    let provider = module(CELL_STORE);
    let mut h = Host::new();
    let offer = h
        .wire_offer_proc_with_policy(&provider, &[0, 1], OfferPolicy::Threaded)
        .expect("threaded offer");
    let tid = h.resolve_offer(offer).unwrap().type_id;

    assert_eq!(
        h.cap_dispatch_slots(tid, 0, offer, &[40, 5], None),
        Ok(vec![5]),
        "store(40, 5) ran host-side over the shared cell"
    );
    assert_eq!(
        h.cap_dispatch_slots(tid, 1, offer, &[40], None),
        Ok(vec![5]),
        "load(40) sees the store — one persistent instance, gate reopened"
    );
}

/// CALLS.md 7.4 — the **guest-facing declaration**: a module marks its own offer export
/// `threaded` (`export N interface "name" threaded T { … }`), and reifying that export
/// (`reify_export`, the `cap.self` route) mints the offer under the declared policy — §10.1's
/// "policy is a declaration by the provider" made literal. The undeclared control stays `Single`.
#[test]
fn a_module_declared_threaded_export_reifies_threaded() {
    let src = "\
memory 16
type 0 func (i64, i64) -> (i64)
type 1 interface { store: 0 }
export 0 interface \"cells\" threaded 1 { store: 0 }

func (i64, i64) -> (i64) {
block 0 (vaddr: i64, vval: i64) {
  i64.store vaddr vval
  return vval
  }
}
";
    let m = std::sync::Arc::new(module(src));
    let mut h = Host::new();
    h.set_self_module(&m);
    let handle = h.reify_export(0).expect("reify");
    assert_eq!(
        h.resolve_offer(handle).unwrap().policy,
        OfferPolicy::Threaded,
        "the module's own declaration is the minted policy"
    );

    let undeclared = std::sync::Arc::new(module(&src.replace(" threaded 1 ", " 1 ")));
    let mut h2 = Host::new();
    h2.set_self_module(&undeclared);
    let handle2 = h2.reify_export(0).expect("reify");
    assert_eq!(
        h2.resolve_offer(handle2).unwrap().policy,
        OfferPolicy::Single,
        "undeclared stays single"
    );
}
