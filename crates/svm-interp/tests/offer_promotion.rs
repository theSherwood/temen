//! CALLS.md 4b.1 — **promotion**: an animated offer handler that parks mid-run.
//!
//! A `single` library provider's offer handler runs inline on the caller's vCPU as a reified
//! fiber over the provider's world (4a). When the handler hits a blocking point it must
//! **promote**: the provider `{mem, host, fuel}` are handed back to the instance (`busy` reopens,
//! §10.1 atomicity window closes) and the caller parks as the handler's resumer; the handler's
//! block-wake re-admits the caller, which re-acquires the world and resumes the handler to
//! completion. 4a stopped short of this (a mid-animation park was fail-closed until 4b), so these
//! tests exercise the piece the promotion machinery adds.

use svm_interp::{run_with_host, Host, Value};

fn module(text: &str) -> svm_ir::Module {
    let m = svm_text::parse_module(text).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    m
}

/// A provider whose single op timed-`atomic.wait`s on a zero cell (so it always parks — the
/// value matches the expected), then returns the wait status. Nobody notifies, so it resumes on
/// the **timeout**: a park that deterministically drives promotion end to end (park → timer wake
/// → resume → settle) with no cross-thread ordering to arrange.
fn timed_wait_provider() -> svm_ir::Module {
    module(
        "memory 16\n\
         func () -> (i64) {\n\
         block 0 () {\n\
           vaddr = i64.const 0\n\
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
