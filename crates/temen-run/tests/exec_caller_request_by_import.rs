//! #1621 — a personality op reached through an **indexed `call.import`** must honor the caller
//! request it fires, exactly as one reached through `call.cap` or `call.sym` does.
//!
//! The three call forms had three answers. `call.cap` and `call.sym` each carried their own copy
//! of the fork/exec handling; `call.import` carried neither and *discarded* the request ("degrade
//! to the poll answer"). That is invisible to every consumer that already worked — a chibicc guest
//! reaches `__px_*` through `call.cap`, so bash and the shell demos fork and exec fine — and fatal
//! to a no-C nim module, whose personality ops arrive as imports: `fork` answered `-ENOSYS`, nim
//! read `pid < 0` as "fork failed", and `os.execShellCmd` reported failure having never called
//! `execve`.
//!
//! The guest here is the smallest thing that can tell the two apart: `_start` calls `execve`
//! through `call.import`, and either becomes the registered command (which exits 77) or falls
//! through to `exit(9)`. Before the fix it exits 9.

use std::sync::Arc;

use temen_run::{instantiate_with_imports, Backend, HostCap, Imports, Outcome, RunConfig, Value};
use temen_text::parse_module;

/// The command `/bin/c`: a child-entry `(i64) -> (i64)` that exits 77. Nothing else — the point is
/// *whether* it runs, not what it does.
const COMMAND: &str = "memory 17\n\
func (i64) -> (i64) {\n\
block 0 (v0: i64) {\n\
  v1 = i64.const 77\n\
  return v1\n\
  }\n\
}\n";

/// The caller: `execve(\"/bin/c\", NULL, NULL)` through an **indexed `call.import`** — the arm
/// under test. A NULL argv is C-legal (and this op's empty vector). Reaching `exit` means the
/// request was dropped and the op's `-ENOSYS` placeholder stood.
const CALLER: &str = "memory 17\n\
\n\
import 0 \"execve\" (i64, i64, i64) -> (i64)\n\
import 1 \"exit\" (i32) -> ()\n\
\n\
data 40000 \"/bin/c\\x00\"\n\
\n\
func () -> () {\n\
block 0 () {\n\
  vp = i64.const 40000\n\
  vz = i64.const 0\n\
  vr = call.import 0 (vp, vz, vz)\n\
  vcode = i32.const 9\n\
  call.import 1 (vcode)\n\
  unreachable\n\
  }\n\
}\n\
export 0 func \"_start\" 0\n";

#[test]
fn execve_through_call_import_replaces_the_image() {
    let caller = parse_module(CALLER).expect("parse caller");
    let command = parse_module(COMMAND).expect("parse command");
    let cmd_wl = command.memory.expect("command window").size_log2;

    let (posix, make) = temen_posix::cap(0, 0, Vec::new());
    let make = Arc::new(make);
    let imports = Imports::new()
        .provide("execve", {
            let make = Arc::clone(&make);
            HostCap::host_proc(temen_posix::OP_EXECVE, move || (*make)())
        })
        .provide("exit", HostCap::exit());
    let inst = instantiate_with_imports(caller, imports).expect("instantiate");

    // The personality's process doors, as `posix_cap` installs them on the powerbox path: without
    // the signal source there is no #799 caller-request door at all and the op cannot even ask.
    let fork = temen_posix::cap_fork_factory(&posix);
    let make_proc = Arc::clone(&make);
    let p = posix.clone();
    let mut setup = move |host: &mut temen_interp::Host| {
        let handle = host.grant_host_proc_forkable((*make_proc)(), Arc::clone(&fork));
        let (door, armed) = temen_posix::cap_signal_source(&p);
        host.set_signal_source(door, armed);
        host.push_exec_remap_hook(temen_posix::cap_exec_remap_hook(&p));
        let (names, sigs) = temen_posix::cap_vtable();
        host.set_host_proc_vtable(handle, names, sigs);
        let h = host.grant_module(&command);
        p.register_executable("/bin/c", h, cmd_wl);
    };
    let run = inst
        .run_with_caps_and_host(
            Backend::TreeWalk,
            &RunConfig::default(),
            &[],
            Some(&mut setup),
        )
        .expect("run");

    // The command's child-entry return — reached only by *becoming* it. A discarded request
    // leaves the caller running past the op and exiting 9 instead.
    assert_eq!(
        run.outcome,
        Outcome::Returned(vec![Value::I64(77)]),
        "the import-routed `execve` must replace the image (Exited(9) = request discarded, #1621)"
    );
}
