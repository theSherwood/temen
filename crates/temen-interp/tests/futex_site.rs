//! #1414 — a futex wait keys on the page it parked on. A waiter whose page is remapped under it (a §13
//! region `map`ped over it) stays keyed on the anonymous page it parked on, so a `notify` on the
//! remapped address does not wake it: the oracle forms the key at the park, as Linux does. The debug
//! scheduler used to key each waiter at the notify, through its window as it was then, and so woke it.

#[path = "support/drivers.rs"]
mod drivers;

use drivers::{agree_on, Ran, SCHEDULING};
use temen_interp::{Host, Value};

/// func 0 (the region handle `v0`): map the region at `65536` (above the NULL guard at any host
/// granule) and store `5` at its byte 0. Spawn func 1 on the next page, `A`, and spin until the thread
/// sets its flag at `A + 8`. Then map the same region over `A`'s page, so `A` reads `5`, and `notify`
/// one waiter at `A`. Returns `woken * 10 + (the thread's status == WAIT_WOKEN)`, plus 100 if either
/// `map` failed.
///
/// func 1: set the flag, then wait at `A` for `0`, for 20 ms.
///
/// The answer is 0 however the threads interleave. A thread that parked before the remap parked on
/// the anonymous page, so the notify on the region misses it and its wait times out. A thread that had
/// not yet waited finds `5` and returns not-equal, and the notify finds no waiter. On the two
/// cooperative drivers the thread always parks first (its flag store and its wait fall in one quantum),
/// so a driver that keys at the notify answers 11 there.
const SRC: &str = "memory 18
func (i32) -> (i64) {
block 0 (v0: i32) {
  vps = call.cap 4 3 () -> (i64) v0 ()
  vz = i64.const 0
  vr = i64.const 65536
  vprot = i32.const 3
  vm1 = call.cap 4 0 (i64, i64, i64, i32) -> (i64) v0 (vr, vz, vps, vprot)
  vfive = i32.const 5
  i32.atomic.store vr vfive
  va = i64.add vr vps
  vchild = thread.spawn 1 vz va
  br 1(v0, vps, va, vchild, vm1)
  }
block 1 (v0: i32, vps: i64, va: i64, vchild: i32, vm1: i64) {
  v8 = i64.const 8
  vfa = i64.add va v8
  vflag = i32.atomic.load vfa
  vone = i32.const 1
  vset = i32.eq vflag vone
  br_if vset 2(v0, vps, va, vchild, vm1) 1(v0, vps, va, vchild, vm1)
  }
block 2 (v0: i32, vps: i64, va: i64, vchild: i32, vm1: i64) {
  vz = i64.const 0
  vprot = i32.const 3
  vm2 = call.cap 4 0 (i64, i64, i64, i32) -> (i64) v0 (va, vz, vps, vprot)
  vone = i32.const 1
  vn = atomic.notify va vone
  vst = thread.join vchild
  vwoke = i64.eq vst vz
  vwoke64 = i64.extend_i32_u vwoke
  vn64 = i64.extend_i32_u vn
  vten = i64.const 10
  vhi = i64.mul vn64 vten
  vsum = i64.add vhi vwoke64
  vmaps = i64.or vm1 vm2
  vbad = i64.ne vmaps vz
  vbad64 = i64.extend_i32_u vbad
  vhundred = i64.const 100
  vpen = i64.mul vbad64 vhundred
  vres = i64.add vsum vpen
  return vres
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, va: i64) {
  v8 = i64.const 8
  vfa = i64.add va v8
  vone = i32.const 1
  i32.atomic.store vfa vone
  vexp = i32.const 0
  vto = i64.const 20000000
  vst = i32.atomic.wait va vexp vto
  vst64 = i64.extend_i32_u vst
  return vst64
  }
}
";

#[test]
fn a_waiter_whose_page_is_remapped_keeps_the_key_it_parked_on() {
    let m = temen_text::parse_module(SRC).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let setup = || {
        let mut host = Host::new();
        let h = host.grant_shared_region(1 << 16); // 64 KiB: at least one page at any host granule
        (host, vec![Value::I32(h)])
    };
    let want = Ran {
        result: Ok(vec![Value::I64(0)]),
        stdout: Vec::new(),
        stderr: Vec::new(),
    };
    agree_on(&SCHEDULING, "a remapped waiter", &m, &setup, &want);
}

/// func 0 (the region handle `v0`): map the region at `A = 65536` and again at `B`, the next page, and
/// spawn func 1. Then wake the thread at the anonymous word `W` (the page after `B`), and after that at
/// region byte 0 through `A`, each with a `notify` repeated until it wakes someone. Between tries the
/// root waits out a 1 µs timeout on `W + 8`, which nobody writes: on the logical clock of the cooperative
/// drivers that timeout fires only once every other task has parked, so the root never spins. Returns
/// the thread's result.
///
/// func 1: wait at `W` for `0`, then at `B` for `0`, with no timeout; returns `first * 10 + second`,
/// 0 when both were woken.
const PARKED_ACROSS_RESTORE: &str = "memory 18
func (i32) -> (i64) {
block 0 (v0: i32) {
  vps = call.cap 4 3 () -> (i64) v0 ()
  vz = i64.const 0
  va = i64.const 65536
  vb = i64.add va vps
  vprot = i32.const 3
  vm1 = call.cap 4 0 (i64, i64, i64, i32) -> (i64) v0 (va, vz, vps, vprot)
  vm2 = call.cap 4 0 (i64, i64, i64, i32) -> (i64) v0 (vb, vz, vps, vprot)
  vw = i64.add vb vps
  vh = i64.extend_i32_u v0
  vchild = thread.spawn 1 vz vh
  br 1(va, vw, vchild)
  }
block 1 (va: i64, vw: i64, vchild: i32) {
  v8 = i64.const 8
  vy = i64.add vw v8
  vexp = i32.const 0
  vto = i64.const 1000
  vyield = i32.atomic.wait vy vexp vto
  vone = i32.const 1
  vn = atomic.notify vw vone
  vgot = i32.lt_u vexp vn
  br_if vgot 2(va, vw, vchild) 1(va, vw, vchild)
  }
block 2 (va: i64, vw: i64, vchild: i32) {
  v8 = i64.const 8
  vy = i64.add vw v8
  vexp = i32.const 0
  vto = i64.const 1000
  vyield = i32.atomic.wait vy vexp vto
  vone = i32.const 1
  vn = atomic.notify va vone
  vgot = i32.lt_u vexp vn
  br_if vgot 3(vchild) 2(va, vw, vchild)
  }
block 3 (vchild: i32) {
  vr = thread.join vchild
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, vh64: i64) {
  vh = i32.wrap_i64 vh64
  vps = call.cap 4 3 () -> (i64) vh ()
  va = i64.const 65536
  vb = i64.add va vps
  vw = i64.add vb vps
  vexp = i32.const 0
  vto = i64.const -1
  vs1 = i32.atomic.wait vw vexp vto
  vs2 = i32.atomic.wait vb vexp vto
  vs1w = i64.extend_i32_u vs1
  vs2w = i64.extend_i32_u vs2
  vten = i64.const 10
  vhi = i64.mul vs1w vten
  vr = i64.add vhi vs2w
  return vr
  }
}
";

fn region_powerbox() -> (Host, Vec<Value>) {
    let mut host = Host::new();
    let h = host.grant_shared_region(1 << 16);
    (host, vec![Value::I32(h)])
}

/// The guest above wakes its thread at an anonymous word and at a region byte, on every driver.
#[test]
fn a_thread_parked_on_an_anonymous_word_and_a_region_byte_is_woken() {
    let m = temen_text::parse_module(PARKED_ACROSS_RESTORE).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let want = Ran {
        result: Ok(vec![Value::I64(0)]),
        stdout: Vec::new(),
        stderr: Vec::new(),
    };
    agree_on(&SCHEDULING, "parked waiters", &m, &region_powerbox, &want);
}

/// A debug checkpoint taken while the thread is parked, restored into a freshly built run, still
/// wakes it. The restore rebuilds the window and the region, so their backings sit at new addresses,
/// which is what a futex key names: a task parked on a key would never be woken again. It parks on
/// its site, which names the region by id, and the notify keys it through the rebuilt window.
#[test]
fn a_waiter_parked_at_a_checkpoint_wakes_after_a_restore() {
    use temen_interp::bytecode::{BlockedOn, ScheduledDebugRun};
    let m = temen_text::parse_module(PARKED_ACROSS_RESTORE).expect("parse");
    let session = || {
        let (host, args) = region_powerbox();
        ScheduledDebugRun::new_with_host(&m, 0, &args, host).expect("the debugger runs the module")
    };
    let mut reference = session();
    let mut fuel = drivers::FUEL;
    while reference.tick(&mut fuel) {}
    assert_eq!(reference.result(), Some(&Ok(vec![Value::I64(0)])));
    let total = reference.op_turn();

    // The futex words the thread (task 1) parks on, and how many checkpoints found it parked there.
    let mut parked_at = std::collections::BTreeMap::<u64, usize>::new();
    for c in 0..=total {
        let mut at_c = session();
        let mut fuel = drivers::FUEL;
        while at_c.op_turn() < c && at_c.tick(&mut fuel) {}
        let Ok(snap) = at_c.snapshot() else {
            continue;
        };
        for (task, on) in at_c.blocked_on() {
            if let (1, BlockedOn::Futex(addr)) = (task, on) {
                *parked_at.entry(addr).or_default() += 1;
            }
        }
        let mut warm = session();
        warm.restore(at_c.op_turn(), &snap);
        let mut fuel = drivers::FUEL;
        while warm.tick(&mut fuel) {}
        assert_eq!(
            warm.result(),
            Some(&Ok(vec![Value::I64(0)])),
            "restored at turn {c}"
        );
    }
    assert_eq!(
        parked_at.len(),
        2,
        "checkpoints caught the thread parked at both its anonymous word and its region byte: \
         {parked_at:?}"
    );
}
