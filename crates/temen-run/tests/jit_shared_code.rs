//! #1825 — a JIT process tree compiles each program **once**. A fork twin runs its parent's code, and
//! every `execve` of a command instantiates the tree's one compile of it (`temen_jit::SharedCode`),
//! each process over its own powerbox, window and run state.
//!
//! The guest forks three children; child `i` execs `/bin/c` with `i + 1` arguments, and the command
//! returns the `argc` its own powerbox's personality answers, which is its exit status. The parent
//! reaps the three and exits with the sum of their statuses, `1 + 2 + 3 = 6`, on every engine. An
//! instance that dispatched into another process's powerbox would see another process's `argc`. On
//! the JIT the run compiles exactly two modules: the parent's program, which its twins run, and the
//! command, which all three execs run.
//!
//! The only test in its binary, so the process-wide compile counter moves only for this run.

use temen_run::{
    instantiate_with_imports, Backend, HostCap, Imports, Outcome, RunConfig, SharedHostProc,
};
use temen_text::parse_module;

/// The parent. `43000 + 32*i` is child `i`'s argv: `i + 1` pointers to `"x"` (at `42000`), then
/// `NULL`. A child whose `execve` fails exits with the errno. `41000` receives each reaped status;
/// its second byte is `WEXITSTATUS`.
const PARENT: &str = r#"memory 17 shadow 65536 69632
import 0 "execve" (i64, i64, i64) -> (i64)
import 1 "exit" (i32) -> ()
import 2 "fork" () -> (i64)
import 3 "wait4" (i64, i64, i64, i64) -> (i64)
data 40000 "/bin/c\x00"
data 42000 "x\x00"
data 43000 "\x10\xa4\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
data 43032 "\x10\xa4\x00\x00\x00\x00\x00\x00\x10\xa4\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
data 43064 "\x10\xa4\x00\x00\x00\x00\x00\x00\x10\xa4\x00\x00\x00\x00\x00\x00\x10\xa4\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
func () -> () {
block 0 () {
  vi = i64.const 0
  br 1(vi)
  }
block 1 (li: i64) {
  lpid = call.import 2 ()
  lz = i64.const 0
  lchild = i64.eq lpid lz
  br_if lchild 2(li) 3(li)
  }
block 2 (ci: i64) {
  cp = i64.const 40000
  c32 = i64.const 32
  coff = i64.mul ci c32
  cbase = i64.const 43000
  cargv = i64.add cbase coff
  cz = i64.const 0
  cr = call.import 0 (cp, cargv, cz)
  cneg = i64.sub cz cr
  cerr = i32.wrap_i64 cneg
  call.import 1 (cerr)
  unreachable
  }
block 3 (pi: i64) {
  pone = i64.const 1
  pn = i64.add pi pone
  pthree = i64.const 3
  pmore = i64.lt_s pn pthree
  pk = i64.const 0
  br_if pmore 1(pn) 4(pk, pk)
  }
block 4 (rk: i64, rsum: i64) {
  rany = i64.const -1
  rst = i64.const 41000
  rz = i64.const 0
  rw = call.import 3 (rany, rst, rz, rz)
  rhi = i64.const 41001
  rcode8 = i32.load8_u rhi
  rcode = i64.extend_i32_u rcode8
  rs = i64.add rsum rcode
  rone = i64.const 1
  rk1 = i64.add rk rone
  rthree = i64.const 3
  rmore = i64.lt_s rk1 rthree
  br_if rmore 4(rk1, rs) 5(rs)
  }
block 5 (xsum: i64) {
  xc = i32.wrap_i64 xsum
  call.import 1 (xc)
  unreachable
  }
}
export 0 func "_start" 0
"#;

/// The command: its status is the `argc` its powerbox's personality answers.
const COMMAND: &str = r#"memory 17
import 0 "__px_argc" () -> (i64)
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = call.import 0 ()
  return v1
  }
}
"#;

fn run_on(backend: Backend) -> Outcome {
    let parent = parse_module(PARENT).expect("parse parent");
    let command = parse_module(COMMAND).expect("parse command");
    let cmd_wl = command.memory.expect("command window").size_log2;
    let (posix, make) = temen_posix::cap(0, 0, Vec::new());
    let slot = SharedHostProc::new(make, temen_posix::cap_fork_factory(&posix));
    let cap = |op| HostCap::host_proc_shared(op, &slot);
    let imports = Imports::new()
        .provide("execve", cap(temen_posix::OP_EXECVE))
        .provide("exit", HostCap::exit())
        .provide("fork", cap(temen_posix::OP_FORK))
        .provide("wait4", cap(temen_posix::OP_WAIT4));
    let inst = instantiate_with_imports(parent, imports).expect("instantiate");
    let p = posix.clone();
    let mut setup = move |host: &mut temen_interp::Host| {
        let handle = slot.install(host);
        let (door, armed) = temen_posix::cap_signal_source(&p);
        host.set_signal_source(door, armed);
        host.push_exec_remap_hook(temen_posix::cap_exec_remap_hook(&p));
        let (names, sigs) = temen_posix::cap_vtable();
        host.set_host_proc_vtable(handle, names, sigs);
        let h = host.grant_module(&command);
        p.register_executable("/bin/c", h, cmd_wl);
    };
    inst.run_with_caps_and_host(backend, &RunConfig::default(), &[], Some(&mut setup))
        .unwrap_or_else(|e| panic!("{backend:?}: {e}"))
        .outcome
}

#[test]
fn a_tree_compiles_its_program_once_for_its_twins_and_a_command_once_for_its_execs() {
    for backend in [Backend::TreeWalk, Backend::Bytecode] {
        assert_eq!(run_on(backend), Outcome::Exited(6), "{backend:?}");
    }
    let before = temen_jit::module_compiles();
    assert_eq!(run_on(Backend::Jit), Outcome::Exited(6), "Jit");
    assert_eq!(
        temen_jit::module_compiles() - before,
        2,
        "the parent's program and the command, each compiled once"
    );
}
