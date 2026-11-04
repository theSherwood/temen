//! #2695 step 2 — a held domain ([`HeldDomain`]) keeps its window's share of the process's address
//! space (#2305) while its embedder holds it, as a process holds its own, and hands it back when it
//! ends, as an ordinary root does when its entry returns. One test in its own binary: the pool is the
//! process's, and tests in one binary share their process.

use std::sync::Arc;
use temen_interp::{
    maps_pool, reserved_pool, HeldDomain, Host, Reply, RootWindow, Value, MAPS_PER_DOMAIN,
    MAPS_PER_VCPU,
};

/// The address space a window of the default reservation holds of the pool.
const WINDOW: u64 = 1 << temen_ir::DEFAULT_RESERVED_LOG2;

/// A domain whose entry answers 1 and whose offer's one op, `twice(x)`, answers `2x`.
const DOMAIN: &str = r#"memory 16
type 0 func (i64) -> (i64)
type 1 interface { twice: 0 }
export 0 interface "svc" 1 { twice: 1 }
func () -> (i64) {
block 0 () {
  vr = i64.const 1
  return vr
  }
}
func (i64) -> (i64) {
block 0 (vx: i64) {
  vk = i64.const 2
  vr = i64.mul vx vk
  return vr
  }
}
"#;

/// An activation holding a window of the default reservation and nothing else, as another run of the
/// process would: `None` when the pool has no room for it.
fn hold() -> Option<Host> {
    let mut h = Host::new();
    let window = RootWindow {
        reserved: WINDOW,
        ..RootWindow::default()
    };
    h.begin_activation(0, window)?;
    Some(h)
}

#[test]
fn a_held_domain_holds_its_window_until_it_ends() {
    let pool = reserved_pool();
    let slots = pool / WINDOW;
    if pool == u64::MAX || maps_pool() < 2 * slots * (MAPS_PER_DOMAIN + MAPS_PER_VCPU) {
        eprintln!("skipped: no address-space pool, or a pool of mappings too small to fill it");
        return;
    }
    // Other runs of the process hold every window but one.
    let mut others: Vec<Host> = std::iter::from_fn(hold).collect();
    assert_eq!(others.len() as u64, slots, "the pool holds its windows");
    others.pop();
    let m = temen_text::parse_module(DOMAIN).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let m = Arc::new(m);
    let mut host = Host::new();
    host.set_self_module(&m);
    // On a thread of its own, so a run that hangs fails the test instead of stalling the suite.
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let (d, results) = HeldDomain::start(m, 0, &[], u64::MAX, host)
            .unwrap_or_else(|(t, _)| panic!("the entry ends in {t:?}"));
        assert_eq!(
            results,
            vec![Value::I64(1)],
            "the held domain takes the last window"
        );
        let mut fuel = 1_000_000u64;
        assert_eq!(d.call(0, 0, &[21], &mut fuel), Reply::Results(vec![42]));
        let while_held = hold();
        let (_host, trap) = d.end();
        let _ = tx.send((while_held.is_some(), trap, hold().is_some()));
    });
    let (while_held, trap, after) = rx
        .recv_timeout(std::time::Duration::from_secs(60))
        .expect("the domain ends");
    assert!(!while_held, "a held domain keeps its window");
    assert_eq!(trap, None);
    assert!(after, "its end hands the window back");
    drop(others);
}
