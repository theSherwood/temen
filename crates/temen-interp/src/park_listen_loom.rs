//! #2958 slice 5 — the two handshakes a parked task leans on to hear of a store outside the rules,
//! modelled under loom: `RUSTFLAGS="--cfg loom" cargo test -p temen-interp --lib loom`.
//!
//! A task that parks on a pipe or a punted call registers what can end its park, then looks once
//! more, then sleeps ([`crate::bytecode`]'s `ThreadRun::park`, #2757). The store that ends it lands
//! outside the rules: another vCPU's pipe write, a pool thread's answer. Each model runs a parker
//! against one store and asserts no wake is lost: the parker either sees the store in its look, or
//! the store finds its registration and fires it.
//!
//! **These model the order, not the code.** Each is a copy of the protocol, written in the orders
//! the code uses over the same kinds of locks and atomics, not `PipeWakers` or `Completions`
//! themselves (they use `std`'s types, which loom cannot step). A change to either's order must be
//! made here too; moving both behind a loom-swappable module so the model runs the code is
//! #3074. The models:
//!
//! - **A pipe** (`PipeProbe::listen` then `PipeProbe::ready`, against a write then
//!   `PipeWakers::fire`): the parker pushes its waker under the wakers' mutex and bumps the count
//!   (`SeqCst`), then reads the FIFO under its mutex; the writer pushes to the FIFO under that mutex,
//!   then reads the count, and takes the wakers only when it is not 0. Two locks and an atomic,
//!   not one lock: the FIFO's mutex orders the parker's look against the writer's store, and the
//!   look comes after the count's bump in the parker's program order.
//! - **A punted call's answer** (`Completions::set_notify` then `try_take`, against
//!   `Completions::complete`): the parker sets the store's hook under its own mutex, then takes the
//!   answer under the store's; the pool thread posts under the store's mutex, then reads the hook
//!   under its own. Two mutexes, in opposite orders on the two sides.
//!
//! The broken variant of each (the look before the registration) must find the lost wake.

use loom::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use loom::sync::{Arc, Mutex};

/// The pipe's side of the model: its FIFO, its wakers and their count, and whether a waker fired.
#[derive(Default)]
struct Pipe {
    fifo: Mutex<usize>,
    wakers: Mutex<usize>,
    count: AtomicUsize,
    fired: AtomicBool,
}

impl Pipe {
    /// `PipeProbe::listen`: register under the wakers' lock, then count in.
    fn listen(&self) {
        let mut w = self.wakers.lock().unwrap();
        *w += 1;
        self.count.fetch_add(1, Ordering::SeqCst);
    }

    /// `PipeProbe::ready` for a read.
    fn ready(&self) -> bool {
        *self.fifo.lock().unwrap() > 0
    }

    /// A write's store, then `PipeWakers::fire`.
    fn write(&self) {
        *self.fifo.lock().unwrap() += 1;
        if self.count.load(Ordering::SeqCst) == 0 {
            return;
        }
        let mut w = self.wakers.lock().unwrap();
        self.count.store(0, Ordering::SeqCst);
        if std::mem::take(&mut *w) > 0 {
            self.fired.store(true, Ordering::SeqCst);
        }
    }
}

fn pipe_model(listen_first: bool) {
    loom::model(move || {
        let p = Arc::new(Pipe::default());
        let q = Arc::clone(&p);
        let parker = loom::thread::spawn(move || {
            let saw = if listen_first {
                q.listen();
                q.ready()
            } else {
                let saw = q.ready();
                q.listen();
                saw
            };
            // It sleeps when its look saw nothing: then a waker must have fired, or fire later.
            !saw
        });
        p.write();
        let slept = parker.join().unwrap();
        assert!(
            !slept || p.fired.load(Ordering::SeqCst),
            "the parker slept and the store missed it"
        );
    });
}

#[test]
fn loom_a_pipe_parks_wake_is_never_lost() {
    pipe_model(true);
}

#[test]
#[should_panic(expected = "missed it")]
fn loom_a_pipe_park_that_looks_before_it_listens_loses_a_wake() {
    pipe_model(false);
}

/// The completion store's side: its answers, its hook, and whether the hook rang.
#[derive(Default)]
struct Store {
    ready: Mutex<bool>,
    hook: Mutex<bool>,
    rang: AtomicBool,
}

fn answer_model(listen_first: bool) {
    loom::model(move || {
        let s = Arc::new(Store::default());
        let t = Arc::clone(&s);
        let parker = loom::thread::spawn(move || {
            let set = |t: &Store| *t.hook.lock().unwrap() = true;
            let take = |t: &Store| std::mem::take(&mut *t.ready.lock().unwrap());
            let saw = if listen_first {
                set(&t);
                take(&t)
            } else {
                let saw = take(&t);
                set(&t);
                saw
            };
            !saw
        });
        // `Completions::complete`: post the answer, let go, then read the hook and ring it.
        *s.ready.lock().unwrap() = true;
        if *s.hook.lock().unwrap() {
            s.rang.store(true, Ordering::SeqCst);
        }
        let slept = parker.join().unwrap();
        assert!(
            !slept || s.rang.load(Ordering::SeqCst),
            "the parker slept and the answer missed it"
        );
    });
}

#[test]
fn loom_a_punted_calls_answer_is_never_lost() {
    answer_model(true);
}

#[test]
#[should_panic(expected = "missed it")]
fn loom_a_punted_call_that_looks_before_it_listens_loses_its_answer() {
    answer_model(false);
}
