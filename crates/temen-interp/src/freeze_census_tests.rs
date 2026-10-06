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
            host: &self.root,
            registry: &self.registry,
        }
    }

    fn census(&self, seat: &Seat) -> Option<DeclineCause> {
        freeze_census(Some(seat), &self.sched, &self.root).map(|d| d.cause)
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

/// #1688 — an unreaped fork twin runs in a window and powerbox of its own that no artifact records.
#[test]
fn an_unreaped_fork_twin_declines() {
    let f = Fixture::new();
    let SchedRef::Real(rs) = &f.sched else {
        unreachable!()
    };
    rs.lock().forked_twins.insert(9, Twin { parent: 0 });
    assert_eq!(f.census(&f.root_seat()), Some(DeclineCause::ForkTwin));
    rs.lock().forked_twins.remove(&9); // reaped
    assert_eq!(f.census(&f.root_seat()), None);
}

/// A bare vCPU of the fixture's root domain, to park by hand.
fn parked_vcpu(f: &Fixture, id: TaskId) -> Box<VCpu> {
    let funcs: Arc<[Func]> = Arc::from(vec![Func {
        params: vec![],
        results: vec![],
        blocks: vec![temen_ir::Block {
            params: vec![],
            insts: vec![],
            term: Terminator::Return(vec![]),
        }],
    }]);
    let dt = Arc::new(DomainTable::new(&funcs, 0));
    Box::new(VCpu::new(
        funcs,
        Arc::from(Vec::new()),
        0,
        &[],
        None,
        Arc::clone(&f.root),
        Fuel::fixed(0),
        0,
        id,
        f.sched.clone(),
        dt,
    ))
}

/// #1898 — a vCPU parked where a freeze has no rule yet declines it up front, naming the site; one
/// parked where the rule is to re-issue does not.
#[test]
fn a_vcpu_parked_at_a_decline_site_declines() {
    let f = Fixture::new();
    let SchedRef::Real(rs) = &f.sched else {
        unreachable!()
    };
    rs.lock()
        .ticket_waiters
        .insert((1, 2), Waiter::VCpu(page_faulted(&f, 5)));
    assert_eq!(
        f.census(&f.root_seat()),
        Some(DeclineCause::Parked(ParkSite::PageFault))
    );
    let v = rs.lock().take(ParkSite::PageFault, false);
    assert_eq!(v.len(), 1);

    rs.lock()
        .cap_waiters
        .entry((1, 3))
        .or_default()
        .push(Waiter::VCpu(parked_vcpu(&f, 6)));
    assert_eq!(f.census(&f.root_seat()), None, "a stream read re-issues");
}

/// A vCPU parked on its pager's page (#1940), which shares `ticket_waiters` with a reply wait.
fn page_faulted(f: &Fixture, id: TaskId) -> Box<VCpu> {
    let mut v = parked_vcpu(f, id);
    v.page_fault = Some(0);
    v
}

/// #1901 — a caller parked on a live callee's reply does not decline: a freeze re-admits it to
/// abandon its wait, and records the ticket for its context on its powerbox, so the thaw's re-issued
/// call waits on it instead of enqueueing the call again.
#[test]
fn a_freeze_re_admits_a_reply_wait_and_records_its_ticket() {
    let f = Fixture::new();
    let SchedRef::Real(rs) = &f.sched else {
        unreachable!()
    };
    let mut v = parked_vcpu(&f, 5);
    v.reply_ticket = Some(2);
    v.durable_sp_ctx = 3;
    rs.lock().ticket_waiters.insert((1, 2), Waiter::VCpu(v));
    assert_eq!(f.census(&f.root_seat()), None, "a reply wait re-issues");

    let mut s = rs.lock();
    admit_parks_for_freeze(&mut s);
    assert!(s.ticket_waiters.is_empty());
    let v = s.runnable.pop_front().expect("re-admitted");
    assert_eq!(v.dstate, STATE_UNWINDING);
    assert!(matches!(v.pending, Some(Pending::Abandoned)));
    drop(s);
    assert_eq!(f.root.lock_unpoisoned().reply_waits(), vec![(3, 2)]);
}

/// #1901 — except in a nested carve, whose powerbox rides without the record: its thaw would issue
/// the call twice.
#[test]
fn a_reply_wait_in_a_nested_carve_declines() {
    let f = Fixture::new();
    let SchedRef::Real(rs) = &f.sched else {
        unreachable!()
    };
    let mut v = parked_vcpu(&f, 5);
    v.reply_ticket = Some(2);
    v.freeze_sink = Some(Arc::clone(&f.root));
    rs.lock().ticket_waiters.insert((1, 2), Waiter::VCpu(v));
    assert_eq!(
        f.census(&f.root_seat()),
        Some(DeclineCause::Parked(ParkSite::Reply))
    );
}

/// A vCPU parked in an indefinite `atomic.wait`: the park that lets freeze-on-quiesce fire.
fn park_on_futex(s: &mut Sched, f: &Fixture, id: TaskId) {
    let mut v = parked_vcpu(f, id);
    v.wait_indefinite = true;
    s.wait_waiters
        .entry(FutexKey::Anon(0, 64))
        .or_default()
        .push((0, Waiter::VCpu(v)));
}

/// #1918 — freeze-on-quiesce asks the census before it fires. With a page-fault park beside the futex
/// park, it declines: the decline lands on the run root's powerbox, the arm is spent, and nothing is
/// re-admitted or phased, so the run carries on as if it was never armed.
#[test]
fn a_freeze_on_quiesce_with_a_decline_site_park_declines() {
    let f = Fixture::new();
    let SchedRef::Real(rs) = &f.sched else {
        unreachable!()
    };
    {
        let mut s = rs.lock();
        park_on_futex(&mut s, &f, 4);
        s.ticket_waiters
            .insert((1, 2), Waiter::VCpu(page_faulted(&f, 5)));
        s.freeze_on_quiesce = Some(Arc::clone(&f.root));
    }
    let (s, acted) = freeze_step(rs, rs.lock());
    assert!(acted);
    assert!(s.freeze_on_quiesce.is_none(), "the one-shot arm is spent");
    assert!(s.runnable.is_empty(), "nothing re-admitted");
    assert!(
        s.parked(ParkSite::Futex)
            .iter()
            .all(|v| v.dstate != STATE_UNWINDING),
        "nothing phased"
    );
    drop(s);
    assert_eq!(
        f.root.lock_unpoisoned().take_freeze_declined(),
        Some(FreezeDeclined {
            cause: DeclineCause::Parked(ParkSite::PageFault),
            task: 5,
            slot: None
        })
    );
    // Declined, the run is left alone: the next idle look does not try again.
    let (s, acted) = freeze_step(rs, rs.lock());
    drop(s);
    assert!(!acted);
}

/// And with nothing in the way, freeze-on-quiesce fires as before: the futex waiter is re-admitted
/// under `UNWINDING`, and nothing is declined.
#[test]
fn a_freeze_on_quiesce_with_nothing_in_the_way_fires() {
    let f = Fixture::new();
    let SchedRef::Real(rs) = &f.sched else {
        unreachable!()
    };
    {
        let mut s = rs.lock();
        park_on_futex(&mut s, &f, 4);
        s.freeze_on_quiesce = Some(Arc::clone(&f.root));
    }
    let (s, acted) = freeze_step(rs, rs.lock());
    assert!(acted);
    assert!(s.freeze_on_quiesce.is_none());
    assert_eq!(s.runnable.len(), 1, "the futex waiter is re-admitted");
    assert!(s.runnable.iter().all(|v| v.dstate == STATE_UNWINDING));
    drop(s);
    assert_eq!(f.root.lock_unpoisoned().take_freeze_declined(), None);
}
