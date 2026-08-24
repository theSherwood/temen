//! CALLS.md §10.5 / increment 5b — **caller-pays** fuel across a cross-domain crossing. The caller
//! owns *how much* runs (it made the call), so an animated offer handler drains the **caller's** own
//! fuel counter, not a provider reserve. Measured with `fuel.remaining` (5a): the caller reads its
//! fuel before and after a call whose handler does a known ~100-back-edge loop; the drop must reflect
//! that work. (Under the retired provider-pays model the handler ran on the provider's reserve and
//! the caller's fuel was saved/restored, so this drop would be ~0.)

use temen_interp::{run_with_host, Host, Value};

fn module(t: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(t).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// An offer handler that spins a fixed ~100-iteration back-edge loop (each taken back-edge charges
/// one fuel), then returns 0.
const LOOPING_PROVIDER: &str = "\
func () -> (i64) {
block 0 () {
  vk = i32.const 100
  br 1(vk)
}
block 1 (vk: i32) {
  vone = i32.const 1
  vk2 = i32.sub vk vone
  br_if vk2 1(vk2) 2()
}
block 2 () {
  vr = i64.const 0
  return vr
  }
}
";

#[test]
fn caller_pays_for_the_handlers_work() {
    let provider = module(LOOPING_PROVIDER);
    let mut host = Host::new();
    let offer = host.wire_offer_proc(&provider, &[0]).expect("offer");
    let tid = host.resolve_offer(offer).unwrap().type_id;

    // Consumer: fuel.remaining (op 13) before, call the offer, fuel.remaining after; return the drop.
    let consumer_src = format!(
        "func () -> (i64) {{\n\
         block 0 () {{\n\
           vz = i32.const 0\n\
           vh = i32.const {offer}\n\
           vbefore = cap.call 4294967295 13 () -> (i64) vz ()\n\
           vign = cap.call {tid} 0 () -> (i64) vh ()\n\
           vafter = cap.call 4294967295 13 () -> (i64) vz ()\n\
           vcost = i64.sub vbefore vafter\n\
           return vcost\n\
           }}\n\
         }}\n"
    );
    let consumer = module(&consumer_src);
    let mut fuel = 10_000_000u64;
    let r = run_with_host(&consumer, 0, &[], &mut fuel, &mut host);
    let cost = match r {
        Ok(v) => match v.as_slice() {
            [Value::I64(c)] => *c,
            other => panic!("unexpected result shape: {other:?}"),
        },
        Err(e) => panic!("run trapped: {e:?}"),
    };
    // The handler's ~100 back-edges + its function entry were charged to the CALLER's fuel.
    assert!(
        (95..=110).contains(&cost),
        "caller-pays: the caller's fuel dropped by the handler's ~100-iteration loop, got {cost}"
    );
}
