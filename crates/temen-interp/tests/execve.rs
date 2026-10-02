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
use temen_interp::{run_with_host, Host, PreparedModule, StreamRole, Value};

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
  vr = call.cap 4294967295 14 (i64, i64, i64, i64, i64) -> (i64) vz (vcmd, vgp, vgn, ventry, vsl)
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
  vh = self.resolve vp vl
  vwp = i64.const 100
  vwl = i64.const 4
  vw = call.cap 0 1 (i64, i64) -> (i64) vh (vwp, vwl)
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

/// #2087 — a command **prepared once** ([`PreparedModule`]) execs from every host it is granted into,
/// as a plain grant does: a toolchain keeps its commands prepared across builds, each build's host a
/// fresh one.
#[test]
fn a_command_prepared_once_execs_from_every_host_it_is_granted_into() {
    let guest = module(GUEST);
    let cmd = PreparedModule::new(module(CMD));
    for _ in 0..2 {
        let mut host = Host::new();
        host.set_self_module(&guest);
        let sink = host.shared_stdout();
        let inst = host.grant_instantiator(0, 1u64 << 12);
        let stream = host.grant_stream(StreamRole::Out);
        let cmd_h = host.grant_prepared(&cmd);
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
        assert_eq!(
            r,
            vec![Value::I64(42)],
            "the command ran in the guest's place"
        );
        let bytes = sink.lock().unwrap_or_else(|e| e.into_inner()).clone();
        assert_eq!(&bytes, b"EXEC", "and wrote through the inherited stdout");
    }
}

/// #1080 — the **bytecode engine image-replaces natively** (no tree-walker fallback), the path the
/// browser playground runs on (`bash_exec` → `compile_and_run_capture_reserved_with_host`). Run the
/// same GUEST through the plain bytecode entry `compile_and_run_with_host`; it must resolve the
/// command, regrant `stdout`, materialize the command's image into the caller's window, swap the
/// activation, and return the command's `42` with `"EXEC"` in the sink — identical to the oracle.
/// (Before #1080 the engine *declined* an exec-bearing module, which is why bash — statically
/// carrying `exec_module` — could not run on the browser's bytecode tier.)
#[test]
fn exec_module_image_replaces_on_the_bytecode_engine() {
    let guest = module(GUEST);
    let cmd = module(CMD);

    let mut host = Host::new();
    host.set_self_module(&guest);
    let sink = host.shared_stdout();
    let inst = host.grant_instantiator(0, 1u64 << 12);
    let stream = host.grant_stream(StreamRole::Out);
    let cmd_h = host.grant_module(&cmd);

    let mut fuel = 40_000_000u64;
    let r = temen_interp::bytecode::compile_and_run_with_host(
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
    .expect("the bytecode engine compiles the exec-bearing module (no decline)")
    .expect("run");

    assert_eq!(
        r,
        vec![Value::I64(42)],
        "the bytecode engine image-replaced and returned the command's 42, not the guest's 99"
    );
    let bytes = sink.lock().unwrap_or_else(|e| e.into_inner()).clone();
    assert_eq!(
        &bytes, b"EXEC",
        "the exec'd command wrote through the inherited stdout on the bytecode engine"
    );
}

/// #1080 — `run_with_host_fast` tries the bytecode engine first; now that the engine image-replaces
/// natively (above), the same GUEST runs there directly (no fold to the oracle) and returns the
/// command's `42`. This keeps the fast-path outcome identical to the oracle run.
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

/// #1080 (TCB) — the refuse path on the **bytecode engine**: a bogus command handle must leave the
/// caller running (write `-EINVAL` to the result slot, fall through to `return 99`), never image-
/// replace or trap. The exec arm's every admissibility failure (bad handle, oversized command,
/// unregrantable grant, non-clean-root) takes this same path, so pinning the handle case pins the
/// contract that a refused exec is a survivable, probeable error on the tier the browser runs.
#[test]
fn a_failed_exec_module_on_the_bytecode_engine_leaves_the_caller_running() {
    let guest = module(GUEST);

    let mut host = Host::new();
    host.set_self_module(&guest);
    let sink = host.shared_stdout();
    let inst = host.grant_instantiator(0, 1u64 << 12);
    let stream = host.grant_stream(StreamRole::Out);
    let bogus_cmd: i64 = 999;

    let mut fuel = 40_000_000u64;
    let r = temen_interp::bytecode::compile_and_run_with_host(
        &guest,
        0,
        &[Value::I32(inst), Value::I64(bogus_cmd), Value::I32(stream)],
        &mut fuel,
        &mut host,
    )
    .expect("the exec-bearing module compiles")
    .expect("run");

    assert_eq!(
        r,
        vec![Value::I64(99)],
        "a refused exec on the bytecode engine returns to the caller's sentinel (99)"
    );
    let bytes = sink.lock().unwrap_or_else(|e| e.into_inner()).clone();
    assert!(
        bytes.is_empty(),
        "a refused exec ran no command, so nothing was written"
    );
}

/// A 64 KiB caller that leaves `90` at 48 KiB and `65` in the args region (the #801 exec ABI: a
/// caller packs argv there for the command), then execs the command with no grants.
const FRESH_GUEST: &str = r#"
memory 16
func (i32, i64) -> (i64) {
block 0 (vinst: i32, vcmd: i64) {
  vs = i64.const 49152
  v90 = i32.const 90
  i32.store8 vs v90
  va = i64.const 16520
  v65 = i32.const 65
  i32.store8 va v65
  vz = i32.const 0
  vgp = i64.const 0
  vgn = i64.const 0
  ventry = i64.const 0
  vsl = i64.const 16
  vr = call.cap 4294967295 14 (i64, i64, i64, i64, i64) -> (i64) vz (vcmd, vgp, vgn, ventry, vsl)
  v99 = i64.const 99
  return v99
  }
}
"#;

/// A 32 KiB command: returns what is at 48 KiB, past its own image, times 1000, plus the byte in
/// the args region.
const FRESH_CMD: &str = r#"
memory 15
func (i64) -> (i64) {
block 0 (vinst: i64) {
  vs = i64.const 49152
  vsb = i32.load8_u vs
  vs64 = i64.extend_i32_u vsb
  vk = i64.const 1000
  vhi = i64.mul vs64 vk
  va = i64.const 16520
  vab = i32.load8_u va
  va64 = i64.extend_i32_u vab
  vr = i64.add vhi va64
  return vr
  }
}
"#;

/// An exec replaces the address space: the command starts in a fresh window of the caller's
/// geometry, so the caller's bytes past the command's image are gone (the command reads `0` where
/// the caller left `90`), while the args region carries over (`65`). The tree-walker and the
/// bytecode engine agree, as the Cranelift JIT, whose exec starts the image in a fresh instance,
/// always has.
#[test]
fn exec_module_starts_the_command_in_a_fresh_window() {
    type Run = fn(&temen_ir::Module, &[Value], &mut u64, &mut Host) -> Vec<Value>;
    let engines: [(&str, Run); 2] = [
        ("tree-walker", |m, a, f, h| {
            run_with_host(m, 0, a, f, h).expect("run")
        }),
        ("bytecode", |m, a, f, h| {
            temen_interp::bytecode::compile_and_run_with_host(m, 0, a, f, h)
                .expect("the bytecode engine compiles it")
                .expect("run")
        }),
    ];
    for (engine, run) in engines {
        let guest = module(FRESH_GUEST);
        let cmd = module(FRESH_CMD);
        let mut host = Host::new();
        host.set_self_module(&guest);
        let inst = host.grant_instantiator(0, 1u64 << 16);
        let cmd_h = host.grant_module(&cmd);
        let mut fuel = 40_000_000u64;
        let args = [Value::I32(inst), Value::I64(cmd_h as i64)];
        assert_eq!(
            run(&guest, &args, &mut fuel, &mut host),
            vec![Value::I64(65)],
            "{engine}: the command reads zero past its image, and the args region the caller packed"
        );
    }
}
