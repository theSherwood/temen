//! **Blocking-stdin park**: the persistent-session seam. With [`Host::set_stdin_blocking`] on, a `read`
//! on an exhausted stdin buffer must *suspend* the run instead of returning EOF, and resume, re-issuing
//! the same read, once the host pushes more bytes: no bytes lost, the read re-executes, and the guest
//! never sees a spurious EOF. A run its embedder pumps, as the browser Postgres console's session
//! runs on the pump (#1414 S4-2), suspends this way on either executor (#1414 5b): it returns
//! [`CoopEvent::Idle`].

#[path = "support/drivers.rs"]
mod drivers;

use drivers::PUMPED;
use temen_interp::bytecode::{CoopEvent, CoopRun};
use temen_interp::{Host, StreamRole, Value, DEFAULT_FUEL};
use temen_text::parse_module;

/// Run `f` on its own thread, for at most 30 s: a run that never idles fails rather than hangs.
fn bounded<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.recv_timeout(std::time::Duration::from_secs(30))
        .expect("the run neither idled nor ended")
}

// A guest that forever: read(stdin, buf, 64) → write(stdout, buf, n) → repeat. `_start(vin, vout)`
// takes the two stream handles positionally. Reading an empty stdin under blocking mode parks it.
const ECHO: &str = r#"memory 16
func (i32, i32) -> (i64) {
block 0 (vin: i32, vout: i32) {
  br 1(vin, vout)
}
block 1 (vin1: i32, vout1: i32) {
  vptr = i64.const 17408
  vlen = i64.const 64
  vn = call.cap 0 0 (i64, i64) -> (i64) vin1 (vptr, vlen)
  vw = call.cap 0 1 (i64, i64) -> (i64) vout1 (vptr, vn)
  br 1(vin1, vout1)
  }
}
"#;

/// The echo guest's session, on each executor: every read on an empty stdin leaves nothing to run,
/// and the run returns `Idle`. Feeding stdin and pumping again re-admits the reader.
#[test]
fn stdin_read_idles_the_run_and_resumes() {
    for exec in PUMPED {
        bounded(move || echo_session(exec));
    }
}

fn echo_session(exec: drivers::Pumped) {
    let m = parse_module(ECHO).expect("parse echo guest");
    let mut host = Host::new();
    let vin = host.grant_stream(StreamRole::In);
    let vout = host.grant_stream(StreamRole::Out);
    host.set_stdin_blocking(true);
    host.push_stdin(b"abc"); // preload the first chunk

    let slots = [Value::I32(vin), Value::I32(vout)];
    // An engine-backed reservation, as the console's session has (small, for the test).
    let mut run = exec.open(&m, &slots, DEFAULT_FUEL, host, 20);

    // Echoes "abc", then the next read finds the buffer empty and parks: not EOF, not an exit.
    assert!(
        matches!(run.run(), CoopEvent::Idle),
        "{exec:?}: an exhausted blocking stdin must idle the run, not return EOF",
    );
    assert_eq!(
        run.host_mut().stdout,
        b"abc",
        "{exec:?}: the preloaded chunk was echoed"
    );
    assert!(
        matches!(run.run(), CoopEvent::Idle),
        "{exec:?}: pumped again with nothing fed, the run is still idle"
    );

    // Push more and pump: the parked read re-issues, echoes "de", and parks again.
    run.host_mut().push_stdin(b"de");
    assert!(matches!(run.run(), CoopEvent::Idle), "{exec:?}");
    assert_eq!(
        run.host_mut().stdout,
        b"abcde",
        "{exec:?}: the re-admitted read took the new bytes"
    );

    // And again: a REPL parks after every prompt.
    run.host_mut().push_stdin(b"f");
    assert!(matches!(run.run(), CoopEvent::Idle), "{exec:?}");
    assert_eq!(run.host_mut().stdout, b"abcdef", "{exec:?}");
}

// Sanity: with blocking OFF (the default — the one-shot `temen_run_pg` / oracle path), an exhausted read
// returns EOF and the guest runs to completion, unchanged. Here the guest reads once and returns `n`.
const READ_ONCE: &str = r#"memory 16
func (i32) -> (i64) {
block 0 (vin: i32) {
  vptr = i64.const 17408
  vlen = i64.const 64
  vn = call.cap 0 0 (i64, i64) -> (i64) vin (vptr, vlen)
  return vn
  }
}
"#;

#[test]
fn non_blocking_stdin_returns_eof_at_end() {
    for exec in PUMPED {
        bounded(move || {
            let m = parse_module(READ_ONCE).expect("parse read-once guest");
            let mut host = Host::new();
            let vin = host.grant_stream(StreamRole::In);
            // stdin_block defaults to false; buffer is empty → read returns 0 (EOF), the guest
            // returns 0.
            let mut run: CoopRun = exec.open(&m, &[Value::I32(vin)], DEFAULT_FUEL, host, 20);
            match run.run() {
                CoopEvent::Done(vals) => {
                    assert_eq!(vals.first(), Some(&Value::I64(0)), "{exec:?}: EOF = 0")
                }
                _ => panic!("{exec:?}: with blocking off an empty read is EOF, not a park"),
            }
        });
    }
}
