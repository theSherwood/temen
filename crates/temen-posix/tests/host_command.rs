//! A command the embedder implements natively ([`temen_posix::Posix::register_host_command`]): a
//! process that execs it runs it over the files that process sees, and exits with its status.

use std::sync::Arc;

use temen_interp::bytecode::{CoopEvent, CoopRun};
use temen_interp::{BoundImport, Host, Trap};

/// `execve("/bin/h", ["/bin/h", "out.txt"], NULL)`, then `exit(9)` if that returned.
const GUEST: &str = "memory 17\n\
import 0 \"__px_execve\" (i64, i64, i64) -> (i64)\n\
import 1 \"__px_exit\" (i64) -> ()\n\
data 40000 \"/bin/h\\x00\"\n\
data 40100 \"out.txt\\x00\"\n\
data 43000 \"\\x40\\x9c\\x00\\x00\\x00\\x00\\x00\\x00\\xa4\\x9c\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\"\n\
func () -> () {\n\
block 0 () {\n\
  vp = i64.const 40000\n\
  vargv = i64.const 43000\n\
  vz = i64.const 0\n\
  vr = call.import 0 (vp, vargv, vz)\n\
  vnine = i64.const 9\n\
  call.import 1 (vnine)\n\
  unreachable\n\
  }\n\
}\n\
export 0 func \"_start\" 0\n";

#[test]
fn a_host_command_runs_over_the_process_files_and_ends_it() {
    let guest = temen_text::parse_module(GUEST).expect("parse");
    temen_verify::verify_module(&guest).expect("verify");
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
    posix.set_cwd("/w");
    posix.write_file("/w/in.txt", b"hi");
    // Upper-cases `in.txt` into the file its argument names, and says how it was called.
    posix.register_host_command(
        "/bin/h",
        Arc::new(|argv, files| {
            let Some(input) = files.read("in.txt") else {
                return 2;
            };
            let out = format!(
                "{} {}",
                argv.join(" "),
                input.to_ascii_uppercase().escape_ascii()
            );
            if files.write(&argv[1], out.as_bytes()) {
                7
            } else {
                3
            }
        }),
    );
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
    assert!(
        matches!(run.run(), CoopEvent::Trapped(Trap::Exit(7))),
        "the process exits with the command's status"
    );
    assert_eq!(
        posix.read_file("/w/out.txt").as_deref(),
        Some(&b"/bin/h out.txt HI"[..]),
        "it read and wrote relative to the process's directory, with the exec's arguments"
    );
}
