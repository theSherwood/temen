//! #1414 S5a — a run's stdout and stderr stream through its host's tee ([`Host::set_stream_tee`]) as
//! the guest writes them, in the order its buffers hold them, while the buffers keep every byte as
//! before. A browser run relays its output to the page through it, on either executor. A stream a host
//! shares (a §14 child's inherited stdout, a fork twin's) carries the tee in its buffer, so every
//! writer's bytes reach it.

use std::sync::{Arc, Mutex};
use temen_interp::{bytecode, cap_id, Host, StreamRole, Value, WindowMem};

/// Writes "a" to stdout, then "bb" to stderr, then "ccc" to stdout.
const WRITES: &str = r#"
memory 16
data 16384 "abbccc"
func (i32, i32) -> (i64) {
block 0 (vout: i32, verr: i32) {
  va = i64.const 16384
  v1 = i64.const 1
  w1 = call.cap 0 1 (i64, i64) -> (i64) vout(va, v1)
  vb = i64.const 16385
  v2 = i64.const 2
  w2 = call.cap 0 1 (i64, i64) -> (i64) verr(vb, v2)
  vc = i64.const 16387
  v3 = i64.const 3
  w3 = call.cap 0 1 (i64, i64) -> (i64) vout(vc, v3)
  vz = i64.const 0
  return vz
  }
}
"#;

type Seen = Arc<Mutex<Vec<(StreamRole, Vec<u8>)>>>;

/// A host with stdout and stderr granted, and a tee that records what it is handed.
fn host_with_tee() -> (Host, i32, i32, Seen) {
    let mut host = Host::new();
    let out = host.grant_stream(StreamRole::Out);
    let err = host.grant_stream(StreamRole::Err);
    let seen: Seen = Arc::default();
    let s = Arc::clone(&seen);
    host.set_stream_tee(Box::new(move |role, bytes| {
        s.lock().unwrap().push((role, bytes.to_vec()))
    }));
    (host, out, err, seen)
}

/// Run [`WRITES`] over `host` on the pump, or on executor 2.
fn run(host: &mut Host, out: i32, err: i32, parallel: bool) {
    let m = temen_text::parse_module(WRITES).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let args = [Value::I32(out), Value::I32(err)];
    let mut fuel = u64::MAX;
    let r = if parallel {
        bytecode::compile_and_run_capture_over_parallel_with_host(
            &m,
            0,
            &args,
            &mut fuel,
            &[],
            None,
            host,
        )
        .map(|(r, _image)| r)
    } else {
        bytecode::compile_and_run_with_host(&m, 0, &args, &mut fuel, host)
    };
    assert_eq!(r.expect("the engine runs it"), Ok(vec![Value::I64(0)]));
}

fn writes() -> Vec<(StreamRole, Vec<u8>)> {
    vec![
        (StreamRole::Out, b"a".to_vec()),
        (StreamRole::Err, b"bb".to_vec()),
        (StreamRole::Out, b"ccc".to_vec()),
    ]
}

#[test]
fn the_tee_sees_each_write_in_order_on_either_executor_and_the_buffers_keep_them() {
    for parallel in [false, true] {
        let (mut host, out, err, seen) = host_with_tee();
        run(&mut host, out, err, parallel);
        assert_eq!(*seen.lock().unwrap(), writes(), "parallel: {parallel}");
        assert_eq!(host.stdout_bytes(), b"accc", "parallel: {parallel}");
        assert_eq!(host.stderr_bytes(), b"bb", "parallel: {parallel}");
    }
}

/// A shared stream's writes reach the tee through the buffer they land in, whether the stream was
/// shared before the tee was set or after it.
#[test]
fn a_shared_stream_tees_what_lands_in_it() {
    // Shared after the tee: the buffer takes it as it is made.
    let (mut host, out, err, seen) = host_with_tee();
    host.shared_stdout();
    run(&mut host, out, err, false);
    assert_eq!(*seen.lock().unwrap(), writes());
    assert_eq!(host.stdout_bytes(), b"accc");

    // Shared before the tee: setting it reaches the buffer already made.
    let mut host = Host::new();
    let out = host.grant_stream(StreamRole::Out);
    let err = host.grant_stream(StreamRole::Err);
    host.shared_stdout();
    host.shared_stderr();
    let seen: Seen = Arc::default();
    let s = Arc::clone(&seen);
    host.set_stream_tee(Box::new(move |role, bytes| {
        s.lock().unwrap().push((role, bytes.to_vec()))
    }));
    run(&mut host, out, err, false);
    assert_eq!(*seen.lock().unwrap(), writes());
    assert_eq!(host.stderr_bytes(), b"bb");
}

/// A write the stream has no room for (`-ENOMEM`, #2306) lands nowhere, so the tee never hears of it:
/// what it was handed stays what the buffer holds, on a host's own stdout and on a shared one.
#[test]
fn a_write_with_no_room_is_not_teed() {
    let (mut host, out, _err, seen) = host_with_tee();
    host.set_grant(-1, 4096, -1, -1);
    let mut win = vec![7u8; 4096];
    let mut write = |host: &mut Host, len: i64| {
        let mut wm = WindowMem::new(&mut win, 4096);
        host.cap_dispatch_slots(cap_id::STREAM, 1, out, &[0, len], Some(&mut wm))
            .expect("dispatch")[0]
    };
    let teed = |seen: &Seen| {
        seen.lock()
            .unwrap()
            .iter()
            .map(|(_, b)| b.len())
            .sum::<usize>()
    };
    assert_eq!(write(&mut host, 3000), 3000);
    assert_eq!(write(&mut host, 2000), -12, "past the ceiling");
    assert_eq!(teed(&seen), 3000);
    assert_eq!(host.take_stdout().len(), 3000);

    host.shared_stdout();
    assert_eq!(write(&mut host, 4096), 4096);
    assert_eq!(
        write(&mut host, 1),
        -12,
        "the shared sink is charged as the buffer was"
    );
    assert_eq!(teed(&seen), 3000 + 4096);
    assert_eq!(host.stdout_bytes().len(), 4096);
}
