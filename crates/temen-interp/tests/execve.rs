//! FORK.md §8.6 — **true cross-module `execve` (image-replace)**, the `exec_module` self-op
//! ([`temen_interp::CAP_SELF_EXEC`], op 14). A running guest replaces its **own image** with a granted
//! *separate command module*, in place, keeping its `TaskId` and fuel — so the command runs as the
//! caller's task, not as a spawned grandchild. This is the real capstone the compiled-C
//! `fork→exec→wait` demo stubbed with BusyBox-multicall applet dispatch.
//!
//! Topology (single top-level run): the host grants the guest an `Instantiator`, a `Stream`
//! (`"stdout"`), and the **command module** (as a resolvable handle). The guest builds a 1-entry grant
//! list `{"stdout" → stream}` and calls `exec_module(cmd, grants, 1, entry=0, size_log2=12)`. The eval
//! loop resolves the command, regrants `stdout` into a fresh powerbox, binds the command's imports, and
//! hands `dispatch` an image-replace: the guest vCPU **becomes** the command. The command resolves
//! `"stdout"` by name, writes `"EXEC"`, and returns `42`. The guest's post-`exec` `return 99` **never
//! runs** — proof the image was truly replaced — so the run returns `42` and the sink holds `"EXEC"`.

use std::sync::Arc;
use temen_interp::{run_with_host, Host, StreamRole, Value};

fn module(text: &str) -> Arc<temen_ir::Module> {
    let m = temen_text::parse_module(text).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    Arc::new(m)
}

/// The guest: `main(inst, cmd, stream)` builds `{"stdout" → stream}` at window offset 256 and calls
/// `exec_module`. On success it never returns; the trailing `return 99` is the did-not-exec sentinel.
const GUEST: &str = r#"
memory 12
data 300 "stdout"
func (i32, i64, i32) -> (i64) {
block 0 (vinst: i32, vcmd: i64, vstream: i32) {
  va0 = i64.const 256
  vnp = i32.const 300
  i32.store va0 vnp
  va1 = i64.const 260
  vlen = i32.const 6
  i32.store va1 vlen
  va2 = i64.const 264
  i32.store va2 vstream
  vz = i32.const 0
  vgp = i64.const 256
  vgn = i64.const 1
  ventry = i64.const 0
  vsl = i64.const 12
  vr = cap.call 4294967295 14 (i64, i64, i64, i64, i64) -> (i64) vz (vcmd, vgp, vgn, ventry, vsl)
  v99 = i64.const 99
  return v99
  }
}
"#;

/// The command: entry `(inst) -> status`. Resolves `"stdout"` by name (registered when `exec_module`
/// regranted it), writes its `"EXEC"` data segment to it, and exits `42`. Its data segments are
/// materialized into the caller's window by the image-replace, so `"EXEC"`/`"stdout"` are present.
const CMD: &str = r#"
memory 12
data 100 "EXEC"
data 200 "stdout"
func (i64) -> (i64) {
block 0 (vinst: i64) {
  vp = i64.const 200
  vl = i64.const 6
  vh = cap.self.resolve vp vl
  vwp = i64.const 100
  vwl = i64.const 4
  vw = cap.call 0 1 (i64, i64) -> (i64) vh (vwp, vwl)
  v42 = i64.const 42
  return v42
  }
}
"#;

#[test]
fn exec_module_replaces_the_image_with_a_separate_command_module() {
    let guest = module(GUEST);
    let cmd = module(CMD);

    let mut host = Host::new();
    host.set_self_module(&guest);
    let sink = host.shared_stdout();
    let inst = host.grant_instantiator(0, 1u64 << 12);
    let stream = host.grant_stream(StreamRole::Out);
    let cmd_h = host.grant_module(&cmd);

    let mut fuel = 40_000_000u64;
    let r = run_with_host(
        &guest,
        0,
        &[
            Value::I32(inst),
            Value::I64(cmd_h as i64),
            Value::I32(stream),
        ],
        &mut fuel,
        &mut host,
    )
    .expect("run");

    // The run returns the COMMAND's exit status (42), not the guest's did-not-exec 99 — the guest's
    // image was truly replaced, keeping its task.
    assert_eq!(
        r,
        vec![Value::I64(42)],
        "the command's exit status (42) is the task's result — the guest's `return 99` never ran"
    );

    // The command wrote "EXEC" to the inherited stdout — a *separate module* did real I/O as the
    // caller's task.
    let bytes = sink.lock().unwrap_or_else(|e| e.into_inner()).clone();
    assert_eq!(
        &bytes, b"EXEC",
        "the exec'd command wrote through the inherited stdout"
    );
}

/// OPS_PARITY.md (`exec_module`) — `exec_module` is eval-loop-only, so the fast tiers must **fold**
/// the module to the oracle, never run a generic `cap.call` thunk that answers `-EINVAL` where the
/// oracle image-replaces (a silent divergence, INVARIANTS.md #9). Pin the fail-closed fold: the same
/// guest run via `run_with_host_fast` (which tries the bytecode engine first) declines and folds, so
/// it image-replaces and returns the command's `42` — identical to the oracle run above.
#[test]
fn exec_module_folds_to_the_oracle_on_the_fast_path() {
    let guest = module(GUEST);
    let cmd = module(CMD);

    let mut host = Host::new();
    host.set_self_module(&guest);
    let sink = host.shared_stdout();
    let inst = host.grant_instantiator(0, 1u64 << 12);
    let stream = host.grant_stream(StreamRole::Out);
    let cmd_h = host.grant_module(&cmd);

    let mut fuel = 40_000_000u64;
    let r = temen_interp::run_with_host_fast(
        &guest,
        0,
        &[
            Value::I32(inst),
            Value::I64(cmd_h as i64),
            Value::I32(stream),
        ],
        &mut fuel,
        &mut host,
    )
    .expect("run");

    assert_eq!(
        r,
        vec![Value::I64(42)],
        "the fast path folds the exec module to the oracle and image-replaces (42), not -EINVAL"
    );
    let bytes = sink.lock().unwrap_or_else(|e| e.into_inner()).clone();
    assert_eq!(
        &bytes, b"EXEC",
        "the folded run did the real image-replace + I/O"
    );
}

/// The POSIX contract: **`execve` returns only on failure.** A bogus command handle (never granted)
/// is a probeable `-EINVAL` that leaves the caller running — so the guest falls through to its
/// `return 99` sentinel, and nothing is written to stdout. (Same guest as above; only the command
/// handle passed in is invalid.)
#[test]
fn a_failed_exec_module_returns_einval_and_leaves_the_caller_running() {
    let guest = module(GUEST);

    let mut host = Host::new();
    host.set_self_module(&guest);
    let sink = host.shared_stdout();
    let inst = host.grant_instantiator(0, 1u64 << 12);
    let stream = host.grant_stream(StreamRole::Out);
    // A module handle that was never granted — `resolve_module` fails, so the exec is refused.
    let bogus_cmd: i64 = 999;

    let mut fuel = 40_000_000u64;
    let r = run_with_host(
        &guest,
        0,
        &[Value::I32(inst), Value::I64(bogus_cmd), Value::I32(stream)],
        &mut fuel,
        &mut host,
    )
    .expect("run");

    assert_eq!(
        r,
        vec![Value::I64(99)],
        "a refused exec returns to the caller, which falls through to its sentinel (99)"
    );
    let bytes = sink.lock().unwrap_or_else(|e| e.into_inner()).clone();
    assert!(
        bytes.is_empty(),
        "a refused exec ran no command, so nothing was written"
    );
}
