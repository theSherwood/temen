//! #1671 — [`freeze_census`] cause by cause, over seats built by hand. The causes a guest program
//! reached cheaply were a nested child's, pinned end to end until the carve spawn went (#1867); the
//! rest have reaching shapes (a parked serve handler, a detached child minted without a doorbell, a
//! §13-mapped detached window, a child holding a `Blocking`) that are each a setup of their own.

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
            threads: &self.threads,
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
                        digest: module_digest(&m),
                        module: m,
                        pager: None,
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

/// #1688 — a twin rides when its doorbell can reach it. One with none cannot ride: a
/// `posix_spawn`'s command image, which nothing can unwind, or a servicer's twin, whose reap does
/// not ride (#2571). A freeze declines while one lives; a personality's twin retires at its exit,
/// and holds nothing up.
#[test]
fn a_twin_no_freeze_can_carry_declines() {
    let f = Fixture::new();
    let SchedRef::Real(rs) = &f.sched else {
        unreachable!()
    };
    let twin = |reaper, bell| Twin {
        parent: 0,
        reaper,
        bell,
        entry: 0,
        ctx: 0,
        host: Arc::new(Mutex::new(Host::new())),
    };
    let bell = Some(Arc::new(AtomicBool::new(false)));
    rs.lock()
        .forked_twins
        .insert(9, twin(TwinReaper::Personality, bell));
    assert_eq!(f.census(&f.root_seat()), None, "a durable twin rides");
    rs.lock()
        .forked_twins
        .insert(10, twin(TwinReaper::Personality, None));
    assert_eq!(f.census(&f.root_seat()), Some(DeclineCause::ForkTwin));
    assert!(retire_personality_twin(&mut rs.lock(), 10), "it exited");
    assert_eq!(
        f.census(&f.root_seat()),
        None,
        "an ended one holds nothing up"
    );
    rs.lock()
        .forked_twins
        .insert(11, twin(TwinReaper::Servicer, None));
    assert_eq!(f.census(&f.root_seat()), Some(DeclineCause::ForkTwin));
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
    let dom = f.root.lock_unpoisoned().domain_id();
    rs.lock()
        .completion_waiters
        .insert((dom, 2), Waiter::VCpu(parked_vcpu(&f, 5)));
    assert_eq!(
        f.census(&f.root_seat()),
        Some(DeclineCause::Parked(ParkSite::Completion))
    );
    let v = rs.lock().take(ParkSite::Completion, false, &|_| true);
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

/// #1940, #2455 — a vCPU parked on its page does not decline: a freeze holds its pager to serve the
/// fault ([`hold_for_paged`]), so the park takes the freeze's phase and waits for its page, as a join
/// waits for its child, then unwinds at its next safepoint.
#[test]
fn a_page_fault_park_takes_the_freeze_phase_and_waits() {
    let f = Fixture::new();
    let SchedRef::Real(rs) = &f.sched else {
        unreachable!()
    };
    rs.lock()
        .ticket_waiters
        .insert((1, 2), Waiter::VCpu(page_faulted(&f, 5)));
    assert_eq!(f.census(&f.root_seat()), None);
    let mut s = rs.lock();
    admit_parks_for_freeze(&mut s);
    assert!(s.runnable.is_empty(), "nothing re-admitted");
    let parked = s.parked(ParkSite::PageFault);
    assert_eq!(parked.len(), 1, "still waiting for its page");
    assert_eq!(parked[0].dstate, STATE_UNWINDING);
}

/// #2455 — a freeze that reaches a domain paging a live detached child holds it: the child is rung,
/// the domain's parks stay where they are while the hold waits on the child, and once the child has
/// ended the domain is released and its parks are brought through the freeze.
#[test]
fn a_domain_paging_a_live_child_is_held_until_the_child_ends() {
    let mut f = Fixture::new().with_detached_child(true);
    f.child_freeze.get_mut(&0).expect("slot 0").1.pager = Some(0);
    let SchedRef::Real(rs) = &f.sched else {
        unreachable!()
    };
    let root = f.root.lock_unpoisoned().domain_id() as usize;
    let mut s = rs.lock();
    park_on_futex(&mut s, &f, 4);
    assert!(hold_for_paged(&mut s, root, Some(&f.root_seat())));
    assert!(
        f.child_freeze[&0].0.load(Ordering::Relaxed),
        "the child is rung"
    );
    assert_eq!(s.held[&root], BTreeSet::from([7]));

    admit_parks_for_freeze(&mut s);
    assert!(s.runnable.is_empty(), "a held domain's parks stay");
    assert!(
        !freeze_can_admit(&s),
        "so the worker loop does not spin on them"
    );

    end_hold_member(&mut s, 7);
    assert!(s.held.is_empty(), "released once the child has ended");
    let v = s
        .runnable
        .pop_front()
        .expect("its futex waiter is re-admitted");
    assert_eq!(v.dstate, STATE_UNWINDING);
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

/// A vCPU parked in an indefinite `atomic.wait`: the park that lets freeze-on-quiesce fire.
fn park_on_futex(s: &mut Sched, f: &Fixture, id: TaskId) {
    let mut v = parked_vcpu(f, id);
    v.wait_indefinite = true;
    s.wait_waiters
        .entry(FutexKey::Anon(0, 64))
        .or_default()
        .push((0, Waiter::VCpu(v)));
}

/// #1918 — freeze-on-quiesce asks the census before it fires. With a completion park beside the
/// futex park, it declines: the decline lands on the run root's powerbox, the arm is spent, and
/// nothing is re-admitted or phased, so the run carries on as if it was never armed.
#[test]
fn a_freeze_on_quiesce_with_a_decline_site_park_declines() {
    let f = Fixture::new();
    let SchedRef::Real(rs) = &f.sched else {
        unreachable!()
    };
    let dom = f.root.lock_unpoisoned().domain_id();
    {
        let mut s = rs.lock();
        park_on_futex(&mut s, &f, 4);
        s.completion_waiters
            .insert((dom, 2), Waiter::VCpu(parked_vcpu(&f, 5)));
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
            cause: DeclineCause::Parked(ParkSite::Completion),
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
