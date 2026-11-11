//! A personality reduced to its process doors, as temen-posix's are: op 0 `fork`s and op 1
//! `waitpid`s. Each asks the core through its process's own **park door**, which the driver installs
//! on the domain's signal source, so an engine that leaves a process's door unwired answers its
//! `fork()` and `waitpid()` `-ENOSYS` instead. A fork mints the new process a door of its own and a
//! `waitpid` slot of its own, and records its exit status in its parent's. Include with
//! `#[path = "support/proc_doors.rs"] mod proc_doors;`.
#![allow(dead_code)] // each test binary uses a different subset

use std::sync::{Arc, Mutex};
use temen_interp::put_results;
use temen_interp::{
    CapState, ForkedProc, Host, HostProc, HostProcFork, HostProcGrant, ParkEvent, ProcHandler,
    SignalSource,
};

const ENOSYS: i64 = -38;

/// Grant `host` the personality, its root process's door on `host`'s signal source, and return its
/// handle.
pub fn grant_proc_doors(host: &mut Host) -> i32 {
    let door = Arc::new(ParkDoor::default());
    let exited = Arc::default();
    host.set_signal_source(door.clone(), Arc::default());
    host.grant_host_proc(
        HostProcGrant::new(
            ProcHandler::Sync(doors(door, Arc::clone(&exited))),
            CapState::Stateless,
        )
        .forkable(forks(exited)),
    )
}

/// One process's doors: `fork` asks for the fork through `door`; `waitpid` answers the status its
/// child's exit recorded in `exited`, and until then asks to be re-run when the child ends.
fn doors(door: Arc<ParkDoor>, exited: Arc<Mutex<Option<i64>>>) -> HostProc {
    Box::new(move |op, args, out, _, _| {
        let (park, r) = match (op, *exited.lock().unwrap()) {
            (0, _) => (Some(ParkEvent::ForkSelf), ENOSYS),
            (_, Some(s)) => (None, s),
            (_, None) => (Some(ParkEvent::TaskExit(args[0] as u64)), ENOSYS),
        };
        if let (Some(ev), Some(req)) = (park, door.0.lock().unwrap().clone()) {
            req(ev);
        }
        put_results(out, &[r])
    })
}

/// The fork factory of a process whose children's exits land in `exited`: each new process gets its
/// own door, its own `waitpid` slot and this same factory for its own children.
fn forks(exited: Arc<Mutex<Option<i64>>>) -> HostProcFork {
    Arc::new(move |_pid| {
        let door = Arc::new(ParkDoor::default());
        let own: Arc<Mutex<Option<i64>>> = Arc::default();
        let parent = Arc::clone(&exited);
        Some(ForkedProc {
            signal: Some((door.clone(), Arc::default())),
            exit: Some(Arc::new(move |s| *parent.lock().unwrap() = Some(s))),
            refork: Some(forks(Arc::clone(&own))),
            ..ForkedProc::shared(ProcHandler::Sync(doors(door, own)), CapState::Stateless)
        })
    })
}

/// The park door the driver installs on a domain's signal source.
type ParkRequest = Arc<dyn Fn(ParkEvent) + Send + Sync>;

/// A signal source that holds only the park door the driver installs.
#[derive(Default)]
struct ParkDoor(Mutex<Option<ParkRequest>>);

impl SignalSource for ParkDoor {
    fn take_deliverable(&self) -> Option<(i32, i32, u64)> {
        None
    }
    fn set_park_request(&self, req: ParkRequest) {
        *self.0.lock().unwrap() = Some(req);
    }
}
