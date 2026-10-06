//! #2113 — the resumable `Vcpu` engine meters the run's fuel as every other driver does.
//!
//! A `Vcpu` root opens its run's activation at its first `run`, which grants the powerbox's own node
//! [`DEFAULT_FUEL`], and draws from that node; a `thread.spawn`ed `Vcpu` draws from the node of the
//! powerbox it shares. What a run burns there is what the cooperative driver burns for the same
//! program, the threads' share included, and it is read back on the root powerbox once the run's
//! vCPUs have handed back what they did not burn.

#[path = "support/drivers.rs"]
mod drivers;

use drivers::{run_on_then, Driver, FUEL};
use temen_interp::{Host, Value, DEFAULT_FUEL};
use temen_ir::Module;

/// A loop of 1000 back-edges in the root, returning 42.
const LOOP: &str = r#"
func () -> (i64) {
block 0 () {
  v0 = i64.const 0
  br 1(v0)
}
block 1 (v1: i64) {
  v2 = i64.const 1000
  v3 = i64.lt_u v1 v2
  br_if v3 2(v1) 3()
}
block 2 (v4: i64) {
  v5 = i64.const 1
  v6 = i64.add v4 v5
  br 1(v6)
}
block 3 () {
  v7 = i64.const 42
  return v7
  }
}
"#;

/// The root spawns a vCPU that runs 1000 back-edges and returns 7, and joins it.
const THREADED: &str = r#"
memory 16
func () -> (i64) {
block 0 () {
  v0 = i64.const 0
  v1 = thread.spawn 1 v0 v0
  v2 = thread.join v1
  return v2
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  v0 = i64.const 0
  br 1(v0)
}
block 1 (v1: i64) {
  v2 = i64.const 1000
  v3 = i64.lt_u v1 v2
  br_if v3 2(v1) 3()
}
block 2 (v4: i64) {
  v5 = i64.const 1
  v6 = i64.add v4 v5
  br 1(v6)
}
block 3 () {
  v7 = i64.const 7
  return v7
  }
}
"#;

fn module(src: &str) -> Module {
    let m = temen_text::parse_module(src).unwrap_or_else(|e| panic!("parse: {e:?}"));
    temen_verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    m
}

/// Run `src` on `driver` and return its result and the fuel its run burned on the root powerbox's
/// node: what the activation granted less what is left once every vCPU has handed back its rest.
fn burned(driver: Driver, src: &str) -> (Vec<Value>, u64) {
    let m = module(src);
    let setup = || (Host::new(), Vec::new());
    let (ran, left) = run_on_then(driver, &m, &setup, &|h: &Host| h.fuel_left())
        .unwrap_or_else(|| panic!("{driver:?} declined the module"));
    let granted = if driver == Driver::Vcpu {
        DEFAULT_FUEL
    } else {
        FUEL
    };
    let result = ran
        .result
        .unwrap_or_else(|t| panic!("{driver:?} trapped: {t:?}"));
    (result, granted - left)
}

/// Every metered driver burns the same fuel for `src` and returns the same result: the oracle, the
/// cooperative and parallel drivers, and the resumable `Vcpu` engine. (The debug scheduler lends a
/// fixed allowance per resume instead of drawing from the run's node, so it has no readout here.)
fn burns_alike(src: &str) -> u64 {
    let (want, oracle) = burned(Driver::Oracle, src);
    assert!(oracle >= 1000, "the loop burns fuel: {oracle}");
    for d in [Driver::Coop, Driver::Parallel, Driver::Vcpu] {
        let (got, b) = burned(d, src);
        assert_eq!(got, want, "{d:?}: the result");
        assert_eq!(b, oracle, "{d:?}: the fuel the run burned on its node");
    }
    oracle
}

/// A `Vcpu` root draws from the node its run's activation opens, and pays the entry's one fuel as
/// every driver does.
#[test]
fn a_vcpu_root_burns_its_runs_fuel_as_every_driver_does() {
    burns_alike(LOOP);
}

/// A `thread.spawn`ed `Vcpu` draws from the same node as its root.
#[test]
fn a_vcpu_thread_burns_the_same_runs_fuel() {
    burns_alike(THREADED);
}
