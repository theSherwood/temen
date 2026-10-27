//! #2360 — **a handle closed before a freeze stays closed after the thaw**, on both engines.
//!
//! A closed slot keeps its generation counter (D37): a later use of the handle answers `-EBADF`
//! (`handle_revoked`), and only a generation the slot never issued traps as a forgery. A freeze
//! carries every slot's counter, so the thawed table tells the two apart as the frozen one did.
//!
//! The guest closes its stdout, reads the clock (where the freeze lands), then writes through the
//! closed handle and returns what the write answered: `-EBADF`, uninterrupted or thawed.

use temen_durable::{
    begin_thaw, init_durable_window, read_state, transform_module_assume_confined, STATE_UNWINDING,
};
use temen_interp::{run_capture_reserved_with_host, Host, MemLayout, StreamRole, Value};
use temen_ir::Module;
use temen_jit::JitOutcome;

const SIZE_LOG2: u8 = 18;
const WINDOW: usize = 1 << SIZE_LOG2;
const TEST_ARENA: temen_ir::durable_abi::ShadowArena =
    temen_ir::durable_abi::ShadowArena::new(16448, 65536);
/// `-EBADF`, what a use of a closed handle answers.
const EBADF: i64 = -9;

/// func 0 `(stdout, Clock)`. With `FLIP` it stores `UNWINDING` into the freeze word just before the
/// clock read, so the freeze lands there.
const SRC: &str = "memory 18 shadow 16448 65536
func (i32, i32) -> (i64) {
block 0 (vout: i32, vclk: i32) {
  vcl = call.cap 0 2 () -> (i64) vout ()
FLIP
  vz = i32.const 0
  vt = call.cap 2 0 (i32) -> (i64) vclk (vz)
  vbuf = i64.const 70000
  vlen = i64.const 0
  vw = call.cap 0 1 (i64, i64) -> (i64) vout (vbuf, vlen)
  return vw
  }
}
";

/// The guest, instrumented; `flip` makes it request the freeze itself.
fn instrumented(flip: bool) -> Module {
    let flip = if flip {
        let word = TEST_ARENA.freeze_word().expect("the arena holds the word");
        format!("  vsa = i64.const {word}\n  vsu = i32.const 1\n  i32.store vsa vsu")
    } else {
        String::new()
    };
    let m = temen_text::parse_module(&SRC.replace("FLIP", &flip)).expect("parse");
    let inst = transform_module_assume_confined(&m).expect("transform");
    temen_verify::verify_module(&inst).expect("verify");
    inst
}

/// A durable host holding stdout and a clock, and the guest's arguments.
fn host() -> (Host, [Value; 2]) {
    let mut host = Host::new();
    host.set_durable(true);
    let out = host.grant_stream(StreamRole::Out);
    let clk = host.grant_clock();
    (host, [Value::I32(out), Value::I32(clk)])
}

/// The interpreter's run: what the guest returned (`None` when the freeze unwound it), and the
/// window it left.
fn interp(m: &Module, host: &mut Host, args: &[Value], win: &[u8]) -> (Option<i64>, Vec<u8>) {
    let mut fuel = 50_000_000u64;
    let (res, left) = run_capture_reserved_with_host(m, 0, args, &mut fuel, win, SIZE_LOG2, host);
    let res = res.expect("the guest neither traps nor faults");
    let out = (read_state(&left, TEST_ARENA) != STATE_UNWINDING).then(|| match res[..] {
        [Value::I64(v)] => v,
        _ => panic!("unexpected result {res:?}"),
    });
    (out, left)
}

/// The native JIT's run, as [`interp`].
fn jit(m: &Module, host: &mut Host, args: &[Value], win: &[u8]) -> (Option<i64>, Vec<u8>) {
    let args: Vec<i64> = args
        .iter()
        .map(|a| match a {
            Value::I32(x) => *x as i64,
            other => panic!("unexpected arg {other:?}"),
        })
        .collect();
    let layout = MemLayout::image(win.to_vec());
    let (out, left) = temen_run::jit_cap_run(m, 0, &args, &layout, SIZE_LOG2, 0, host, None)
        .expect("the JIT runs it");
    let out = match out {
        JitOutcome::Returned(v) if read_state(left.bytes(), TEST_ARENA) != STATE_UNWINDING => {
            Some(v[0])
        }
        JitOutcome::Returned(_) => None,
        other => panic!("the guest neither traps nor faults: {other:?}"),
    };
    (out, left.bytes().to_vec())
}

type Engine = fn(&Module, &mut Host, &[Value], &[u8]) -> (Option<i64>, Vec<u8>);

/// **Each engine's freeze thaws on each engine to `-EBADF`.** On main the thawed slot came back at
/// the restoring host's counter, `0`, so the write's generation was one it never issued and the thaw
/// trapped `CapFault`. The thawed table's counters are the frozen run's, so a grant after the thaw
/// issues the handle the uninterrupted run's would, and the artifact is canonical: a re-freeze of the
/// restored domain is byte-identical (§12.6), and both engines write the same artifact.
#[test]
fn a_handle_closed_before_the_freeze_answers_ebadf_after_the_thaw() {
    let engines: [(&str, Engine); 2] = [("interpreter", interp), ("JIT", jit)];
    let window = init_durable_window(WINDOW, TEST_ARENA);
    let mut artifacts = Vec::new();
    for (name, run) in engines {
        let (mut control, args) = host();
        let (base, _) = run(&instrumented(false), &mut control, &args, &window);
        assert_eq!(base, Some(EBADF), "{name}: uninterrupted run");
        let next = control.grant_clock();

        let m = instrumented(true);
        let (mut fhost, args) = host();
        let (res, snap) = run(&m, &mut fhost, &args, &window);
        assert_eq!(res, None, "{name}: frozen at the clock read");
        let artifact = temen_snapshot::freeze(&m, &snap, &fhost).expect("rides");

        for (thaw, run) in engines {
            let mut thost = Host::new();
            thost.set_durable(true);
            let rwin = temen_snapshot::restore(&artifact, &m, &mut thost).expect("restores");
            assert_eq!(
                temen_snapshot::freeze(&m, &rwin, &thost).expect("re-freeze"),
                artifact,
                "{name} → {thaw}: canonical re-freeze byte-identical"
            );
            assert_eq!(
                thost.slot_generations(),
                fhost.slot_generations(),
                "{name} → {thaw}: the table's counters ride, the closed slot's included"
            );
            let mut twin = rwin;
            begin_thaw(&mut twin, TEST_ARENA, 0);
            let (thawed, _) = run(&m, &mut thost, &args, &twin);
            assert_eq!(
                thawed,
                Some(EBADF),
                "{name} → {thaw}: the closed handle is still closed, not a forgery"
            );
            assert_eq!(
                thost.grant_clock(),
                next,
                "{name} → {thaw}: the next grant issues the uninterrupted run's handle"
            );
        }
        artifacts.push(artifact);
    }
    assert_eq!(
        artifacts[0], artifacts[1],
        "both engines write the same artifact"
    );
}
