//! A process tree on the cooperative bytecode driver holds what is running, not everything that
//! ran. A finished process gives its window back, and a command exec'd again runs the program its
//! first exec compiled. nimony's driver builds a program this way, forking and execing a compiler
//! per module, and the browser runs it on this driver.

use temen_interp::bytecode::{CoopEvent, CoopRun, Footprint};
use temen_interp::{BoundImport, Host, Trap};

/// `/bin/c`: a command that returns 7.
const COMMAND: &str = "memory 17\n\
func (i64) -> (i64) {\n\
block 0 (v0: i64) {\n\
  v1 = i64.const 7\n\
  return v1\n\
  }\n\
}\n";

/// Three times: `fork`; the child execs `/bin/c` (and exits 9 if it could not), the parent reaps
/// it. Then `exit(0)`.
const GUEST: &str = "memory 17\n\
import 0 \"__px_execve\" (i64, i64, i64) -> (i64)\n\
import 1 \"__px_exit\" (i64) -> ()\n\
import 2 \"__px_fork\" () -> (i64)\n\
import 3 \"__px_wait4\" (i64, i64, i64, i64) -> (i64)\n\
data 40000 \"/bin/c\\x00\"\n\
func () -> () {\n\
block 0 () {\n\
  vn = i64.const 3\n\
  br 1(vn)\n\
  }\n\
block 1 (wn: i64) {\n\
  vz = i64.const 0\n\
  vdone = i64.eq wn vz\n\
  br_if vdone 5() 2(wn)\n\
  }\n\
block 2 (xn: i64) {\n\
  vpid = call.import 2 ()\n\
  vz = i64.const 0\n\
  vchild = i64.eq vpid vz\n\
  br_if vchild 3() 4(xn, vpid)\n\
  }\n\
block 3 () {\n\
  vp = i64.const 40000\n\
  vz = i64.const 0\n\
  vr = call.import 0 (vp, vz, vz)\n\
  vnine = i64.const 9\n\
  call.import 1 (vnine)\n\
  unreachable\n\
  }\n\
block 4 (yn: i64, ypid: i64) {\n\
  vst = i64.const 41000\n\
  vz = i64.const 0\n\
  vw = call.import 3 (ypid, vst, vz, vz)\n\
  vone = i64.const 1\n\
  vn1 = i64.sub yn vone\n\
  br 1(vn1)\n\
  }\n\
block 5 () {\n\
  vz = i64.const 0\n\
  call.import 1 (vz)\n\
  unreachable\n\
  }\n\
}\n\
export 0 func \"_start\" 0\n";

fn module(text: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(text).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

#[test]
fn a_process_tree_holds_what_is_running() {
    let guest = module(GUEST);
    let mut host = Host::new();
    let (px, posix) = temen_posix::grant(&mut host, 0, 0, Vec::new());
    let binds = guest
        .imports
        .iter()
        .map(|i| {
            let c = temen_posix::resolve_import(&i.name).expect("a personality op");
            BoundImport::required(c.type_id, c.op, px)
        })
        .collect();
    host.set_import_bindings(binds);
    let command = host.grant_module(&module(COMMAND));
    posix.register_executable("/bin/c", command, 17);
    let mut run = CoopRun::new_reserved(
        &guest,
        0,
        &[],
        u64::MAX,
        host,
        None,
        &[],
        temen_ir::DEFAULT_RESERVED_LOG2,
    )
    .expect("the bytecode engine runs it")
    .expect("it starts");
    match run.run() {
        CoopEvent::Trapped(Trap::Exit(0)) | CoopEvent::Done(_) => {}
        CoopEvent::Trapped(t) => panic!("the run trapped: {t:?}"),
        _ => panic!("the run paused"),
    }
    assert_eq!(
        run.footprint(),
        Footprint {
            windows: 1,
            units: 2,
        },
        "the root's window and two programs, the root's and /bin/c's, after three children ran /bin/c"
    );
}
