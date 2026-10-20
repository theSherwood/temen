//! #2305 — a window the address space has no room for is refused, never a host panic. Every window
//! reserves `2^DEFAULT_RESERVED_LOG2` bytes (128 GiB) of the process's address space. The process's
//! pool of it (`temen_interp::RESERVED_POOL`) refuses a window at its admission before the address
//! space runs out, unless other mappings fill it first, which is what this test does: it holds the
//! address space itself, leaving no gap a window fits in. A run on the JIT then ends with
//! `ThreadFault` before it starts, and a detached spawn answers `-EINVAL`, its budget charged
//! nothing. Once the address space is handed back, both go through. Its own binary: the address
//! space is the process's.
#![cfg(any(target_os = "linux", target_os = "android"))]

#[path = "support/detached.rs"]
mod detached;

use detached::{host, jit_on_reserved, module, parent_room_after_refusal};
use temen_jit::JitOutcome;
use temen_run::{instantiate, Backend, Outcome, RunConfig};

const SEVEN: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  v = i64.const 7
  return v
  }
}
"#;

/// A detached child with nothing to bind: its spawn is refused for its window or not at all.
const CHILD: &str = r#"memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  v = i64.const 42
  return v
  }
}
"#;

/// The reservation every window takes by default.
const WINDOW: usize = 1 << temen_ir::DEFAULT_RESERVED_LOG2;

/// What the test leaves free for this process's own threads, code and heap.
const HEADROOM: usize = 64 << 30;

/// An inaccessible, unbacked reservation of `len` bytes of the address space, if there is room.
fn reserve(len: usize) -> Option<usize> {
    // SAFETY: a fresh mapping of our own; nothing is read or written through it.
    let p = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_NONE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
            -1,
            0,
        )
    };
    (p != libc::MAP_FAILED).then_some(p as usize)
}

fn release(p: usize, len: usize) {
    // SAFETY: `[p, p + len)` is a reservation [`reserve`] made and nothing else uses.
    unsafe { libc::munmap(p as *mut libc::c_void, len) };
}

/// The address space held until no gap a [`WINDOW`] fits in is left, but one of [`HEADROOM`] is.
/// Dropped, it hands all of it back.
struct Exhausted(Vec<(usize, usize)>);

impl Exhausted {
    /// `None` when the address space is too large to fill.
    fn new() -> Option<Exhausted> {
        let mut held = Vec::new();
        // Every gap a window fits in, a window's worth at a time: 2^12 of them is 4 PiB, more than
        // any host maps without being asked to.
        while let Some(p) = reserve(WINDOW) {
            held.push((p, WINDOW));
            if held.len() == 1 << 12 {
                for (p, len) in held {
                    release(p, len);
                }
                return None;
            }
        }
        // Room for this process: a gap of HEADROOM, cut from the last window's worth if none is left.
        // Every other gap is smaller, and the top-down allocator takes the top of the one it cuts, so
        // what stays free there is HEADROOM and a neighbour smaller than it: no window fits.
        match reserve(HEADROOM) {
            Some(p) => release(p, HEADROOM),
            None => {
                let (p, len) = held.pop().expect("a window's worth was held");
                release(p, len);
                let p = reserve(WINDOW - HEADROOM).expect("the gap just freed");
                held.push((p, WINDOW - HEADROOM));
            }
        }
        Some(Exhausted(held))
    }
}

impl Drop for Exhausted {
    fn drop(&mut self) {
        for &(p, len) in &self.0 {
            release(p, len);
        }
    }
}

/// `m` run on the JIT, a root window of the default reservation.
fn run_jit(m: &temen_ir::Module) -> Result<Vec<temen_interp::Value>, String> {
    match instantiate(m.clone())?
        .run(Backend::Jit, &RunConfig::default())?
        .outcome
    {
        Outcome::Returned(v) => Ok(v),
        Outcome::Exited(code) => Err(format!("exited {code}")),
    }
}

/// The detached spawn of [`CHILD`] by a parent whose own window is small enough to fit: the room
/// left in the funding budget when the spawn is refused, else `-1`.
fn spawn_detached() -> i64 {
    let (p, c) = (module(&parent_room_after_refusal(0, 0)), module(CHILD));
    let (mut host, h) = host(&c, 1 << 20);
    match jit_on_reserved(&p, &mut host, h, 24) {
        JitOutcome::Returned(v) => v[0],
        o => panic!("the parent ends abnormally: {o:?}"),
    }
}

#[test]
fn a_window_the_address_space_cannot_hold_is_refused_not_a_panic() {
    if reserve(WINDOW).map(|p| release(p, WINDOW)).is_none() {
        eprintln!("skipped: this host cannot reserve a window of the default reservation");
        return;
    }
    let seven = temen_text::parse_module(SEVEN).expect("parse");
    let Some(exhausted) = Exhausted::new() else {
        eprintln!("skipped: the address space is too large to fill");
        return;
    };
    let ended = run_jit(&seven).expect_err("no window");
    assert!(
        ended.contains("ThreadFault"),
        "a run with no room for its window ends with ThreadFault: {ended}"
    );
    assert_eq!(
        spawn_detached(),
        1 << 20,
        "a detached spawn with no room for its window is refused, its budget charged nothing"
    );
    drop(exhausted);
    assert_eq!(
        run_jit(&seven),
        Ok(vec![temen_interp::Value::I64(7)]),
        "runs again"
    );
    assert_eq!(spawn_detached(), -1, "spawns again");
}
