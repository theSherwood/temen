//! #1671 — [`freeze_census`] cause by cause, over seats built by hand: the causes a guest program can
//! reach cheaply are pinned end to end in `tests/freeze_declined.rs`; these are the rest, whose
//! reaching shapes (a parked serve handler, a detached child minted without a doorbell, a §13-mapped
//! detached window, a child holding a `Blocking`) are each a setup of their own.

use super::*;
use std::time::Duration;

struct Fixture {
    sched: SchedRef,
    root: Arc<Mutex<Host>>,
    child: Arc<Mutex<Host>>,
    registry: Arc<FiberRegistry>,
    threads: Vec<Option<TaskId>>,
    child_hosts: BTreeMap<usize, Arc<Mutex<Host>>>,
    child_freeze: BTreeMap<usize, (Arc<AtomicBool>, DetachedSpawn)>,
}

impl Fixture {
    fn new() -> Fixture {
        Fixture {
            sched: SchedRef::Real(Arc::new(Scheduler::new(8, 1))),
            root: Arc::new(Mutex::new(Host::new())),
            child: Arc::new(Mutex::new(Host::new())),
            registry: Arc::new(FiberRegistry::new()),
            threads: Vec::new(),
            child_hosts: BTreeMap::new(),
            child_freeze: BTreeMap::new(),
        }
    }

    /// The root's seat, with a detached child at slot 0 (task 7) when `threads` names one.
    fn root_seat(&self) -> Seat<'_> {
        Seat {
            id: 0,
            nested_child: false,
            threads: &self.threads,
            nested_children: &[],
            child_hosts: &self.child_hosts,
            child_freeze: &self.child_freeze,
            handler_parked: false,
            window_safe: true,
            host: &self.root,
            registry: &self.registry,
        }
    }

    fn census(&self, seat: &Seat) -> Option<DeclineCause> {
        freeze_census(seat, &self.sched, &self.root).map(|d| d.cause)
    }

    /// A live detached child at slot 0, task 7 — with a doorbell when `bell`.
    fn with_detached_child(mut self, bell: bool) -> Fixture {
        self.threads = vec![Some(7)];
        self.child_hosts.insert(0, Arc::clone(&self.child));
        if bell {
            let m = Arc::new(Module::default());
            self.child_freeze.insert(
                0,
                (
                    Arc::new(AtomicBool::new(false)),
                    DetachedSpawn {
                        entry: 0,
                        digest: module_digest(&m),
                        module: m,
                        max_vcpus: 1,
                        same_module: false,
                    },
                ),
            );
        }
        self
    }
}

#[test]
fn a_run_with_nothing_in_the_way_is_not_declined() {
    let f = Fixture::new().with_detached_child(true);
    assert_eq!(f.census(&f.root_seat()), None);
}

#[test]
fn a_root_holding_a_non_durable_handle_is_left_to_the_codec() {
    // The root's embedder can still drain it after the run, so the census does not decide for it.
    let f = Fixture::new();
    f.root
        .lock_unpoisoned()
        .grant_blocking(Duration::ZERO, None);
    assert_eq!(f.census(&f.root_seat()), None);
}

#[test]
fn a_parked_serve_handler_declines() {
    let f = Fixture::new();
    let seat = Seat {
        handler_parked: true,
        ..f.root_seat()
    };
    assert_eq!(f.census(&seat), Some(DeclineCause::ServeHandlerParked));
}

#[test]
fn a_live_detached_child_without_a_doorbell_declines() {
    let f = Fixture::new().with_detached_child(false);
    assert_eq!(
        f.census(&f.root_seat()),
        Some(DeclineCause::DetachedUnreachable)
    );
}

/// #1674: a trap now rides the artifact, so a child that finished with one is no reason to decline.
#[test]
fn a_detached_child_that_completed_with_a_trap_does_not_decline() {
    let f = Fixture::new().with_detached_child(true);
    if let SchedRef::Real(rs) = &f.sched {
        rs.lock().results.insert(
            7,
            Outcome {
                result: Err(Trap::Unreachable),
                mem: None,
                fuel: 0,
                trap_bt: Vec::new(),
                trap_fiber: None,
                trap_fault: None,
            },
        );
    }
    assert_eq!(f.census(&f.root_seat()), None);
}

#[test]
fn a_child_domain_holding_a_non_durable_handle_declines() {
    let f = Fixture::new();
    f.child
        .lock_unpoisoned()
        .grant_blocking(Duration::ZERO, None);
    let child_registry = Arc::new(FiberRegistry::new());
    let child_seat = Seat {
        id: 7,
        host: &f.child,
        registry: &child_registry,
        ..f.root_seat()
    };
    assert!(matches!(
        f.census(&child_seat),
        Some(DeclineCause::NonDurableHandle(NonDurableHandle {
            kind: NonDurableKind::Blocking,
            ..
        }))
    ));
}

#[test]
fn a_detached_window_with_a_shared_region_mapped_declines() {
    let f = Fixture::new();
    let child_registry = Arc::new(FiberRegistry::new());
    let child_seat = Seat {
        id: 7,
        window_safe: false,
        host: &f.child,
        registry: &child_registry,
        ..f.root_seat()
    };
    assert_eq!(
        f.census(&child_seat),
        Some(DeclineCause::SharedRegionWindow)
    );
    // The run root's own window is the embedder's to capture; the census leaves it alone.
    let root_seat = Seat {
        window_safe: false,
        ..f.root_seat()
    };
    assert_eq!(f.census(&root_seat), None);
}
