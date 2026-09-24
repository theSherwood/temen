//! #1646 — a personality **caller request** must be answered identically however the guest reached
//! the op, and on whichever engine runs it.
//!
//! `fork`, `execve` and a blocking `wait4` are not ordinary host calls: the op fires a request into
//! the #799 door and returns a placeholder, and the *drive loop* acts on it. Which means every call
//! form and every engine needs the same answer — and twice now they have not had one:
//!
//! * **#1621** — `call.import`/`call.dyn` drained the request and dropped it, so an import-routed
//!   `fork`/`execve` could not work at all. Invisible to every chibicc guest (those arrive as
//!   `call.cap`/`call.sym`) and fatal to a no-C nim module, whose personality ops are imports.
//! * **#1635** — after #1621 the reap bench was *still* decided per arm: `call.cap`/`call.sym`
//!   parked, the import routes answered `-ECHILD` at once, so `execShellCmd` reported failure for
//!   a command that had in fact run.
//!
//! So the assertion here is not "each row is reasonable" but "**every row is the same**". A new
//! call form that quietly acquires its own answer fails this file.
//!
//! The grant axis is the other half of #1635: the same personality reached through **one** powerbox
//! entry (#1645, what temen-run's nim lane now builds) and through **an entry per op** (what it
//! built before). `Host::fork_powerbox` runs the fork factory once per entry, so the second shape
//! asks for one twin once per op — and every op must still land on one process with one working
//! door (the #1644 net).
//!
//! **Not covered here (1): the interrupt row.** #1647 made `-EINTR` on an interrupted park one
//! decision for all four forms, but exercising it needs a signal delivered *while* the caller is
//! parked. The deterministic idiom for that is `c_posix.rs`'s
//! `c_a_caught_signal_interrupts_a_blocked_capability_read_with_eintr`: a spawned guest thread
//! raises in a loop until the parked caller takes the interrupt, so it retries rather than races.
//! Reproducing it here means a hand-written IR guest with `thread.spawn`, atomics and the signal
//! setup; #1647 tracks it. A timing-based approximation would be a `kind:flaky-ci` issue waiting
//! to happen, so there isn't one.
//!
//! **Not covered here (2):** `call.cap` and `call.import.dyn`, whose handle operand is real rather than
//! vestigial — a `_start`-shaped powerbox run has no way to hand a guest a live `HOST_PROC` handle,
//! and building one by hand would bypass the scheduler these requests need. `call.cap` is covered
//! end-to-end instead by `crates/temen/tests/c_posix.rs`, where chibicc's real shell forks, execs
//! and reaps through it; `call.sym` here is its near neighbour (the same inline arm shape).
//! #1646 tracks closing the two remaining rows.

use std::sync::Arc;

use temen_run::{
    instantiate_with_imports, Backend, HostCap, Imports, Outcome, RunConfig, SharedHostProc, Value,
};
use temen_text::parse_module;

/// The command `/bin/c`: a child-entry `(i64) -> (i64)` returning 77. The point is *whether* it
/// runs, not what it does.
const COMMAND: &str = "memory 17\n\
func (i64) -> (i64) {\n\
block 0 (v0: i64) {\n\
  v1 = i64.const 77\n\
  return v1\n\
  }\n\
}\n";

/// #1768 — a command exec cannot start: its entry `(i32) -> (i32)` is no shape a §14 child enters by
/// (`child_entry_ok`), so the ENGINE refuses it after the personality has already resolved the path
/// and raised the request — the refusal only the engine can make.
const UNSTARTABLE: &str = "memory 17\n\
func (i32) -> (i32) {\n\
block 0 (v0: i32) {\n\
  return v0\n\
  }\n\
}\n";

/// #1768 — a command that reports the argv it was exec'd with: its entry returns the `argc` word at
/// `module_args_base` (`16512`), where the exec's commit puts the new image's args blob.
const ARGC_COMMAND: &str = "memory 17\n\
func (i64) -> (i64) {\n\
block 0 (v0: i64) {\n\
  v1 = i64.const 16512\n\
  v2 = i32.load v1\n\
  v3 = i64.extend_i32_u v2\n\
  return v3\n\
  }\n\
}\n";

/// How the guest spells a personality call. The two forms produce byte-identical modules apart from
/// the spelling — `vdummy` is emitted in both so even the value numbering matches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Form {
    /// Indexed `call.import <slot> (args)` — a no-C nim module's route.
    Import,
    /// `call.sym <slot> v<handle> (args)` — a chibicc/link-form guest's route. The handle operand
    /// is vestigial under the reference policy.
    Sym,
}

impl Form {
    fn call(self, slot: u32, args: &str) -> String {
        match self {
            Form::Import => format!("call.import {slot} ({args})"),
            Form::Sym => format!("call.sym {slot} vdummy ({args})"),
        }
    }
    fn all() -> [Form; 2] {
        [Form::Import, Form::Sym]
    }
}

/// How the personality occupies the powerbox.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Grant {
    /// One entry serving every op (#1645) — what `nim_posix_imports` builds.
    OneEntry,
    /// An entry per op over one shared fork factory — the pre-#1645 shape, kept as a row because
    /// the #1644 net is what still makes it behave.
    EntryPerOp,
}

impl Grant {
    fn all() -> [Grant; 2] {
        [Grant::OneEntry, Grant::EntryPerOp]
    }
}

/// What the guest does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Body {
    /// `execve("/bin/c", NULL, NULL)` then `exit(9)`. Reaching `exit` means the request did not
    /// replace the image.
    Exec,
    /// #1768 — `execve("/bin/c", ["x"], NULL)` against a command the engine refuses, then check the
    /// caller is untouched: the op answered `-EINVAL`, and neither the args region the new image would
    /// have read nor the personality's argv changed. `exit(9)` iff all three hold; `exit(1..3)` names
    /// the first that did not.
    ExecRefusedUntouched,
    /// #1768 — `execve("/bin/c", ["x", "x", "x"], NULL)` into [`ARGC_COMMAND`]: the image-replace must
    /// deliver the argv the op staged, so the command returns `3`.
    ExecDeliversArgv,
    /// `fork()`; the child `execve`s and falls back to `exit(9)`; the parent `wait4`s and exits
    /// with the reaped `WEXITSTATUS`. Nim's `execShellCmd` is exactly this shape.
    ForkExecReap,
}

/// The four personality slots every guest declares, in a fixed order so the call indices are
/// stable across bodies.
const IMPORTS: &str = "import 0 \"execve\" (i64, i64, i64) -> (i64)\n\
import 1 \"exit\" (i32) -> ()\n\
import 2 \"fork\" () -> (i64)\n\
import 3 \"wait4\" (i64, i64, i64, i64) -> (i64)\n\
import 4 \"argc\" () -> (i64)\n";

fn guest(form: Form, body: Body) -> String {
    let head = format!(
        "memory 17\n\n{IMPORTS}\n\
         data 40000 \"/bin/c\\x00\"\n\
         data 41000 \"\\xab\\xcd\\x00\\x00\"\n\n"
    );
    // `41000` starts as a marker, so "wait4 never wrote the status" is distinguishable from
    // "wait4 reported 0" — the difference between a dropped reap bench and a reaped exit.
    let exec = format!(
        "  vp = i64.const 40000\n\
         \x20 vz1 = i64.const 0\n\
         \x20 vr = {}\n\
         \x20 vnine = i32.const 9\n\
         \x20 {}\n\
         \x20 unreachable\n",
        form.call(0, "vp, vz1, vz1"),
        form.call(1, "vnine"),
    );
    match body {
        // `43000` is argv `["x", NULL]` (`42000` is "x"); `16512` is `module_args_base`, where the
        // blob an exec commits lands — its first word is the new image's argc.
        Body::ExecRefusedUntouched => format!(
            "{head}data 42000 \"x\\x00\"\n\
             data 43000 \"\\x10\\xa4\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\"\n\n\
             func () -> () {{\n\
             block 0 () {{\n\
             \x20 vdummy = i32.const 0\n\
             \x20 vab = i64.const 16512\n\
             \x20 va0 = i32.load vab\n\
             \x20 vn0 = {}\n\
             \x20 vp = i64.const 40000\n\
             \x20 vargv = i64.const 43000\n\
             \x20 vz = i64.const 0\n\
             \x20 vr = {}\n\
             \x20 va1 = i32.load vab\n\
             \x20 vn1 = {}\n\
             \x20 vinval = i64.const -22\n\
             \x20 vok = i64.eq vr vinval\n\
             \x20 br_if vok 1(va0, va1, vn0, vn1) 4()\n\
             \x20 }}\n\
             block 1 (wa0: i32, wa1: i32, wn0: i64, wn1: i64) {{\n\
             \x20 vdummy = i32.const 0\n\
             \x20 vok = i32.eq wa0 wa1\n\
             \x20 br_if vok 2(wn0, wn1) 5()\n\
             \x20 }}\n\
             block 2 (xn0: i64, xn1: i64) {{\n\
             \x20 vdummy = i32.const 0\n\
             \x20 vok = i64.eq xn0 xn1\n\
             \x20 br_if vok 3() 6()\n\
             \x20 }}\n\
             block 3 () {{\n\
             \x20 vdummy = i32.const 0\n\
             \x20 vc = i32.const 9\n\
             \x20 {}\n\
             \x20 unreachable\n\
             \x20 }}\n\
             block 4 () {{\n\
             \x20 vdummy = i32.const 0\n\
             \x20 vc = i32.const 1\n\
             \x20 {}\n\
             \x20 unreachable\n\
             \x20 }}\n\
             block 5 () {{\n\
             \x20 vdummy = i32.const 0\n\
             \x20 vc = i32.const 2\n\
             \x20 {}\n\
             \x20 unreachable\n\
             \x20 }}\n\
             block 6 () {{\n\
             \x20 vdummy = i32.const 0\n\
             \x20 vc = i32.const 3\n\
             \x20 {}\n\
             \x20 unreachable\n\
             \x20 }}\n\
             }}\n\
             export 0 func \"_start\" 0\n",
            form.call(4, ""),
            form.call(0, "vp, vargv, vz"),
            form.call(4, ""),
            form.call(1, "vc"),
            form.call(1, "vc"),
            form.call(1, "vc"),
            form.call(1, "vc"),
        ),
        // `43000` is argv `["x", "x", "x", NULL]` (`42000` is "x").
        Body::ExecDeliversArgv => format!(
            "{head}data 42000 \"x\\x00\"\n\
             data 43000 \"\\x10\\xa4\\x00\\x00\\x00\\x00\\x00\\x00\\x10\\xa4\\x00\\x00\\x00\\x00\\x00\\x00\\x10\\xa4\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\"\n\n\
             func () -> () {{\n\
             block 0 () {{\n\
             \x20 vdummy = i32.const 0\n\
             \x20 vp = i64.const 40000\n\
             \x20 vargv = i64.const 43000\n\
             \x20 vz = i64.const 0\n\
             \x20 vr = {}\n\
             \x20 vnine = i32.const 9\n\
             \x20 {}\n\
             \x20 unreachable\n\
             \x20 }}\n\
             }}\n\
             export 0 func \"_start\" 0\n",
            form.call(0, "vp, vargv, vz"),
            form.call(1, "vnine"),
        ),
        Body::Exec => format!(
            "{head}func () -> () {{\n\
             block 0 () {{\n\
             \x20 vdummy = i32.const 0\n\
             {exec}  }}\n\
             }}\n\
             export 0 func \"_start\" 0\n"
        ),
        Body::ForkExecReap => format!(
            "{head}func () -> () {{\n\
             block 0 () {{\n\
             \x20 vdummy = i32.const 0\n\
             \x20 vpid = {}\n\
             \x20 vz = i64.const 0\n\
             \x20 veq = i64.eq vpid vz\n\
             \x20 br_if veq 1() 2(vpid)\n\
             \x20 }}\n\
             block 1 () {{\n\
             \x20 vdummy = i32.const 0\n\
             {exec}  }}\n\
             block 2 (vkid: i64) {{\n\
             \x20 vdummy = i32.const 0\n\
             \x20 vst = i64.const 41000\n\
             \x20 vz2 = i64.const 0\n\
             \x20 vw = {}\n\
             \x20 vhi = i64.const 41001\n\
             \x20 vcode = i32.load8_u vhi\n\
             \x20 {}\n\
             \x20 unreachable\n\
             \x20 }}\n\
             }}\n\
             export 0 func \"_start\" 0\n",
            form.call(2, ""),
            form.call(3, "vkid, vst, vz2, vz2"),
            form.call(1, "vcode"),
        ),
    }
}

/// Run one cell of the table. `registered` decides whether `/bin/c` exists, which is the difference
/// between an exec that replaces the image and one that is refused.
fn run(form: Form, grant: Grant, body: Body, backend: Backend, registered: bool) -> Outcome {
    let caller = parse_module(&guest(form, body)).expect("parse caller");
    let command = parse_module(match body {
        Body::ExecRefusedUntouched => UNSTARTABLE,
        Body::ExecDeliversArgv => ARGC_COMMAND,
        _ => COMMAND,
    })
    .expect("parse command");
    let cmd_wl = command.memory.expect("command window").size_log2;

    let (posix, make) = temen_posix::cap(0, 0, Vec::new());
    let make: Arc<dyn Fn() -> temen_interp::HostProc + Send + Sync> = Arc::new(make);
    let slot = SharedHostProc::new(
        {
            let make = Arc::clone(&make);
            move || (*make)()
        },
        temen_posix::cap_fork_factory(&posix),
    );
    let per_op_fork = temen_posix::cap_fork_factory(&posix);
    let cap = |op: u32| match grant {
        Grant::OneEntry => HostCap::host_proc_shared(op, &slot),
        Grant::EntryPerOp => {
            let make = Arc::clone(&make);
            HostCap::host_proc_forkable(op, move || (*make)(), Arc::clone(&per_op_fork))
        }
    };
    let imports = Imports::new()
        .provide("execve", cap(temen_posix::OP_EXECVE))
        .provide("exit", HostCap::exit())
        .provide("fork", cap(temen_posix::OP_FORK))
        .provide("wait4", cap(temen_posix::OP_WAIT4))
        .provide("argc", cap(temen_posix::OP_ARGC));
    let inst = instantiate_with_imports(caller, imports).expect("instantiate");

    // The process doors, as every process-shaped lane installs them: without the signal source
    // there is no #799 caller-request door at all and the ops cannot even ask.
    let p = posix.clone();
    let slot_for_setup = slot.clone();
    let mut setup = move |host: &mut temen_interp::Host| {
        let handle = slot_for_setup.install(host);
        let (door, armed) = temen_posix::cap_signal_source(&p);
        host.set_signal_source(door, armed);
        host.push_exec_remap_hook(temen_posix::cap_exec_remap_hook(&p));
        let (names, sigs) = temen_posix::cap_vtable();
        host.set_host_proc_vtable(handle, names, sigs);
        if registered {
            let h = host.grant_module(&command);
            p.register_executable("/bin/c", h, cmd_wl);
        }
    };
    inst.run_with_caps_and_host(backend, &RunConfig::default(), &[], Some(&mut setup))
        .expect("run")
        .outcome
}

/// Every (form, grant, engine) cell of one behaviour must produce `want`. The message names the
/// cell, because "which row disagreed" is the whole diagnostic.
fn assert_parity(body: Body, registered: bool, want: Outcome) {
    // The JIT serves `execve` (#1768) but not yet `fork`, so the fork row runs on the interpreters.
    let backends: &[Backend] = match body {
        Body::ForkExecReap => &[Backend::TreeWalk, Backend::Bytecode],
        _ => &[Backend::TreeWalk, Backend::Bytecode, Backend::Jit],
    };
    for &backend in backends {
        for grant in Grant::all() {
            for form in Form::all() {
                let got = run(form, grant, body, backend, registered);
                assert_eq!(
                    got, want,
                    "{body:?} (registered={registered}) disagreed at {form:?} / {grant:?} / \
                     {backend:?}"
                );
            }
        }
    }
}

/// #1621 — `execve` replaces the image whichever way the op was reached. Before that fix the
/// import row exited 9 (request discarded) while the `call.sym` row returned 77.
#[test]
fn execve_replaces_the_image_identically_on_every_route() {
    assert_parity(Body::Exec, true, Outcome::Returned(vec![Value::I64(77)]));
}

/// POSIX: `execve` returns only on failure. An unregistered path must leave the caller running with
/// a probeable errno on every route — a route that *replaced* the image here would be handing out
/// an image nobody granted.
#[test]
fn a_refused_execve_leaves_the_caller_running_on_every_route() {
    assert_parity(Body::Exec, false, Outcome::Exited(9));
}

/// #1768 — the refusal only the engine can make (the command's entry is no shape exec can start),
/// reached after the personality resolved the path and raised the request. POSIX: a failed `execve`
/// returns to an unchanged caller — the argv the op staged must not have reached the caller's args
/// region or the personality's argv. Before the fix both were overwritten by the time the engine
/// refused (`exit(2)`: the args region's argc had become 1).
#[test]
fn an_execve_the_engine_refuses_leaves_the_caller_untouched_on_every_route() {
    assert_parity(Body::ExecRefusedUntouched, true, Outcome::Exited(9));
}

/// #1768 — the new image reads the argv its `execve` passed. On the JIT this is a different road
/// from the interpreters' (the image runs in a fresh window seeded from the commit, not the caller's
/// window reused in place), so it is a row of its own.
#[test]
fn an_execd_image_reads_the_argv_it_was_given_on_every_route() {
    assert_parity(
        Body::ExecDeliversArgv,
        true,
        Outcome::Returned(vec![Value::I64(3)]),
    );
}

/// #1635 — nim's `execShellCmd` shape. The twin must get its own process and its own door (so its
/// `execve` fires into its own cell rather than the parent's), and the parent's blocking `wait4`
/// must **bench** rather than answer `-ECHILD`. Before the fix this row was `Exited(205)` (the
/// untouched status marker — the reap was never serviced) on the import routes.
#[test]
fn a_fork_twin_execs_and_the_parent_reaps_identically_on_every_route() {
    assert_parity(Body::ForkExecReap, true, Outcome::Exited(77));
}
