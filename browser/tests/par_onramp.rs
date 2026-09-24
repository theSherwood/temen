//! #152 — the **on-ramp powerbox on the parallel driver** (`temen_par_powerbox_onramp`): a `.temen`
//! off the on-ramp toolchain whose runtime `thread.spawn`s runs each thread as its own vCPU over one
//! shared window, every vCPU dispatching its manifest imports through the one shared `Mutex<Host>`
//! the recipe publishes. This harness plays `par.js` + `worker.js` with **real OS threads**: one
//! thread per vCPU, `PAR_SPAWN` starts another, `PAR_JOIN` joins it, and `PAR_WAIT`/`PAR_NOTIFY`
//! are a futex over the shared window — so the guest's children genuinely run concurrently. The
//! single-threaded oracle is [`onramp_exec`], the playground's cooperative on-ramp run.

use std::sync::atomic::{AtomicI32, AtomicI64, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::Duration;
use temen_browser::{
    onramp_exec, temen_par_alloc, temen_par_child, temen_par_compile, temen_par_deliver_code,
    temen_par_deliver_handle, temen_par_deliver_join, temen_par_ev_a, temen_par_ev_b,
    temen_par_ev_c, temen_par_ev_d, temen_par_free, temen_par_powerbox_onramp, temen_par_root,
    temen_par_run, temen_par_stdout_len, temen_par_stdout_ptr, ParVcpu, PAR_DONE, PAR_JOIN,
    PAR_NOTIFY, PAR_SPAWN, PAR_TRAP, PAR_WAIT, STATUS_EXIT,
};

/// The `temen_par_*` recipes are process-global statics (one page runs one program); serialize the
/// tests that publish one.
static RECIPE_LOCK: Mutex<()> = Mutex::new(());

/// The native stand-in for `Atomics.wait`/`Atomics.notify`: a per-address FIFO of parked waiters.
/// A waiter checks the word and enqueues under the table lock, and a notifier dequeues under it, so
/// a wake between check and park is not lost; `notify` wakes at most `count` waiters on that
/// address, as `Atomics.notify` does.
struct Waiter {
    woken: Mutex<bool>,
    cv: Condvar,
}
static FUTEX: Mutex<Vec<(usize, std::sync::Arc<Waiter>)>> = Mutex::new(Vec::new());

/// Park on `addr` while its `width`-byte word holds `expected` (zero-extended, as the engine passes
/// it): `0` woken, `1` not-equal, `2` timed out.
fn futex_wait(addr: usize, width: i64, expected: i64, timeout_ns: i64) -> i32 {
    let w = std::sync::Arc::new(Waiter {
        woken: Mutex::new(false),
        cv: Condvar::new(),
    });
    {
        let mut q = FUTEX.lock().unwrap();
        // SAFETY: `addr` is inside the live window, naturally aligned by the guest's op.
        let cur = unsafe {
            if width == 8 {
                (*(addr as *const AtomicI64)).load(Ordering::SeqCst)
            } else {
                (*(addr as *const AtomicI32)).load(Ordering::SeqCst) as u32 as i64
            }
        };
        if cur != expected {
            return 1;
        }
        q.push((addr, w.clone()));
    }
    let mut woken = w.woken.lock().unwrap();
    let deadline = (timeout_ns >= 0)
        .then(|| std::time::Instant::now() + Duration::from_nanos(timeout_ns as u64));
    while !*woken {
        match deadline {
            None => woken = w.cv.wait(woken).unwrap(),
            Some(d) => {
                let now = std::time::Instant::now();
                if now >= d {
                    break;
                }
                woken = w.cv.wait_timeout(woken, d - now).unwrap().0;
            }
        }
    }
    if *woken {
        return 0;
    }
    drop(woken);
    // Timed out — unless a notifier dequeued us meanwhile, which counts as a wake.
    let mut q = FUTEX.lock().unwrap();
    match q.iter().position(|(_, x)| std::sync::Arc::ptr_eq(x, &w)) {
        Some(i) => {
            q.remove(i);
            2
        }
        None => 0,
    }
}

/// Wake up to `count` waiters parked on `addr`; returns how many were woken.
fn futex_notify(addr: usize, count: i64) -> i32 {
    let mut q = FUTEX.lock().unwrap();
    let mut n = 0;
    while n < count {
        let Some(i) = q.iter().position(|(a, _)| *a == addr) else {
            break;
        };
        let (_, w) = q.remove(i);
        *w.woken.lock().unwrap() = true;
        w.cv.notify_one();
        n += 1;
    }
    n as i32
}

/// The fixture's thread count and per-thread iterations.
const WORKERS: i64 = 4;
const ITERS: i64 = 2000;

/// The guest (see the fixture's header): 4 threads, each writing its letter through the **same**
/// `write` import bound in the shared host — the seam under test.
fn guest_src() -> String {
    include_str!("fixtures/threads_onramp.temt").to_string()
}

/// How a vCPU finished.
#[derive(Debug, PartialEq)]
enum End {
    Done(i64),
    /// `exit(code)` (`Some`) or another trap (`None`), with the trap's name.
    Trap(Option<i32>, String),
}

/// A run's shared state, as the Workers see it: the program and the one window.
#[derive(Clone, Copy)]
struct Run {
    prog: usize,
    win: usize,
    win_size: usize,
}

/// Drive one vCPU to its end on this thread — the `worker.js` event loop.
fn drive(run: Run, v: *mut ParVcpu) -> End {
    let mut children: Vec<std::thread::JoinHandle<End>> = Vec::new();
    loop {
        match temen_par_run(v) {
            PAR_DONE => {
                let r = temen_par_ev_a(v);
                temen_par_free(v);
                return End::Done(r);
            }
            PAR_TRAP => {
                let (code, exited) = (temen_par_ev_a(v), temen_par_ev_b(v));
                // SAFETY: `c`/`d` are a `&'static str`'s pointer and length.
                let name = unsafe {
                    std::str::from_utf8(std::slice::from_raw_parts(
                        temen_par_ev_c(v) as *const u8,
                        temen_par_ev_d(v) as usize,
                    ))
                    .unwrap()
                    .to_string()
                };
                temen_par_free(v);
                return End::Trap((exited == 1).then_some(code as i32), name);
            }
            PAR_SPAWN => {
                let am = temen_par_ev_a(v);
                let (module, func) = ((am >> 32) as u32, am as u32);
                let (sp, arg) = (temen_par_ev_b(v), temen_par_ev_c(v));
                let h = std::thread::spawn(move || {
                    let c = temen_par_child(
                        run.prog as *mut _,
                        run.win as *mut u8,
                        run.win_size,
                        module,
                        func,
                        sp,
                        arg,
                    );
                    assert!(!c.is_null(), "child vCPU build failed");
                    drive(run, c)
                });
                children.push(h);
                temen_par_deliver_handle(v, (children.len() - 1) as i32);
            }
            PAR_JOIN => {
                let i = temen_par_ev_a(v) as usize;
                // Each handle is joined once (the guest's `thread.join`); swap in a finished stub.
                let h = std::mem::replace(&mut children[i], std::thread::spawn(|| End::Done(0)));
                match h.join().expect("child thread panicked") {
                    End::Done(r) => temen_par_deliver_join(v, r, 0),
                    End::Trap(..) => temen_par_deliver_join(v, 0, 1),
                }
            }
            PAR_WAIT => {
                let addr = run.win + (temen_par_ev_a(v) as usize & (run.win_size - 1));
                let code = futex_wait(
                    addr,
                    temen_par_ev_c(v),
                    temen_par_ev_b(v),
                    temen_par_ev_d(v),
                );
                temen_par_deliver_code(v, code);
            }
            PAR_NOTIFY => {
                let addr = run.win + (temen_par_ev_a(v) as usize & (run.win_size - 1));
                temen_par_deliver_code(v, futex_notify(addr, temen_par_ev_b(v)));
            }
            e => panic!("unexpected par event {e}"),
        }
    }
}

/// Publish the on-ramp recipe for `m`, run its root over a fresh `win_size` window, and return how
/// the root ended plus the shared host's stdout.
fn par_onramp_run(m: &temen_ir::Module, win_size: usize) -> (End, Vec<u8>) {
    let bytes = temen_encode::encode_module(m);
    assert_eq!(
        temen_par_powerbox_onramp(bytes.as_ptr(), bytes.len(), core::ptr::null(), 0),
        1,
        "the on-ramp recipe must accept an on-ramp module"
    );
    let prog = temen_par_compile(bytes.as_ptr(), bytes.len());
    assert!(!prog.is_null(), "module unsupported on the parallel driver");
    let win = temen_par_alloc(win_size);
    let run = Run {
        prog: prog as usize,
        win: win as usize,
        win_size,
    };
    let root = temen_par_root(prog, win, win_size, 0);
    assert!(!root.is_null(), "root vCPU build failed");
    let end = drive(run, root);
    let n = temen_par_stdout_len();
    // SAFETY: the stash `temen_par_stdout_len` just filled.
    let out = if n == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(temen_par_stdout_ptr(), n) }.to_vec()
    };
    (end, out)
}

fn guest() -> temen_ir::Module {
    let m = temen_text::parse_module(&guest_src()).expect("guest parses");
    temen_verify::verify_module(&m).expect("guest verifies");
    m
}

/// The threads run concurrently on their own OS threads, every one's `write` lands in the one
/// shared host's stdout, and the root's `exit(7)` reaches the driver as an exit, not a crash — the
/// same observable result as the cooperative single-threaded on-ramp run.
#[test]
fn onramp_threads_run_in_parallel_through_one_shared_powerbox() {
    let _g = RECIPE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let m = guest();
    let total = WORKERS * ITERS + 10 * (0..WORKERS).sum::<i64>();

    let (end, out) = par_onramp_run(&m, 1 << 16);
    assert_eq!(end, End::Trap(Some(7), "Exit".into()), "stdout {out:?}");
    assert_eq!(out.len(), WORKERS as usize + 8, "stdout {out:?}");
    let mut letters = out[..WORKERS as usize].to_vec();
    letters.sort_unstable();
    assert_eq!(
        letters, b"abcd",
        "each thread's write reached the shared stdout"
    );
    assert_eq!(i64::from_le_bytes(out[4..12].try_into().unwrap()), total);

    // The oracle: the playground's single-threaded on-ramp run of the same bytes.
    let oracle = onramp_exec(&m, b"");
    assert_eq!(oracle.status, STATUS_EXIT);
    assert_eq!(oracle.exit_code, 7);
    let mut oletters = oracle.stdout[..WORKERS as usize].to_vec();
    oletters.sort_unstable();
    assert_eq!(oletters, letters);
    assert_eq!(oracle.stdout[4..], out[4..]);
}

/// The root reserves exactly the window, as every thread does: a `vm_map` past it is refused, rather
/// than succeeding into a tail whose writes the shared window silently drops.
#[test]
fn onramp_root_cannot_map_past_its_window() {
    let _g = RECIPE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let src = r#"memory 16
import 0 "vm_map" (i64, i64, i32) -> (i64)
func () -> (i64) {
block 0 () {
  voff = i64.const 65536
  vlen = i64.const 65536
  vprot = i32.const 3
  vr = call.import 0 (voff, vlen, vprot)
  return vr
  }
}
export 0 func "_start" 0
"#;
    let m = temen_text::parse_module(src).expect("parses");
    temen_verify::verify_module(&m).expect("verifies");
    let (end, _) = par_onramp_run(&m, 1 << 16);
    match end {
        End::Done(r) => assert!(r < 0, "vm_map past the window must fail, got {r}"),
        e => panic!("expected the root to return the vm_map result, got {e:?}"),
    }
    // Within a larger window the same map succeeds: the bound is the window, not the declaration.
    let (end, _) = par_onramp_run(&m, 1 << 18);
    assert_eq!(end, End::Done(0));
}

/// A module the on-ramp refuses (imports, but no `_start` entry to bind them to) publishes nothing.
#[test]
fn onramp_recipe_refuses_a_non_onramp_module() {
    let _g = RECIPE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let src = "memory 16\nimport 0 \"exit\" (i32) -> ()\nfunc (i64) -> (i64) {\nblock 0 (v0: i64) {\n  return v0\n  }\n}\n";
    let m = temen_text::parse_module(src).expect("parses");
    let bytes = temen_encode::encode_module(&m);
    assert_eq!(
        temen_par_powerbox_onramp(bytes.as_ptr(), bytes.len(), core::ptr::null(), 0),
        0
    );
    assert_eq!(
        temen_par_powerbox_onramp(b"junk".as_ptr(), 4, core::ptr::null(), 0),
        0
    );
}
