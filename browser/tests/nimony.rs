//! The browser's entry for nimony's own driver ([`temen_browser::nim_build`], #958), on a stand-in
//! driver: what the toolchain's processes rely on besides the engine. A command runs at every path it
//! is registered at (nimony finds a tool beside itself, the shell by `PATH`), the build runs in its
//! directory over the files it was given, and a program the build wrote is runnable, as a
//! compile-time evaluation's is. The real driver is measured by `src/nimbuild.rs`.

use temen_browser::{nim_build, STATUS_EXIT};

/// Returns 7: exec'd, the process exits 7.
const SEVEN: &str = "memory 17\n\
func (i64) -> (i64) {\n\
block 0 (v0: i64) {\n\
  v1 = i64.const 7\n\
  return v1\n\
  }\n\
}\n";

/// Another program that exits 7: the one the build "wrote".
const THREE_AND_FOUR: &str = "memory 17\n\
func (i64) -> (i64) {\n\
block 0 (v0: i64) {\n\
  v1 = i64.const 3\n\
  v2 = i64.const 4\n\
  v3 = i64.add v1 v2\n\
  return v3\n\
  }\n\
}\n";

/// Forks twice. The first child execs `/bin/c`, the second `./p`, and each exits 9 if its exec
/// failed. The driver reaps both and exits with the sum of their exit statuses.
const DRIVER: &str = "memory 17\n\
import 0 \"__px_execve\" (i64, i64, i64) -> (i64)\n\
import 1 \"__px_exit\" (i64) -> ()\n\
import 2 \"__px_fork\" () -> (i64)\n\
import 3 \"__px_wait4\" (i64, i64, i64, i64) -> (i64)\n\
data 40000 \"/bin/c\\x00\"\n\
data 40100 \"./p\\x00\"\n\
func () -> () {\n\
block 0 () {\n\
  vpid = call.import 2 ()\n\
  vz = i64.const 0\n\
  vchild = i64.eq vpid vz\n\
  br_if vchild 1() 2(vpid)\n\
  }\n\
block 1 () {\n\
  vp = i64.const 40000\n\
  vz = i64.const 0\n\
  vr = call.import 0 (vp, vz, vz)\n\
  vnine = i64.const 9\n\
  call.import 1 (vnine)\n\
  unreachable\n\
  }\n\
block 2 (xpid: i64) {\n\
  vst = i64.const 41000\n\
  vz = i64.const 0\n\
  vw = call.import 3 (xpid, vst, vz, vz)\n\
  vhi = i64.const 41001\n\
  vs1w = i32.load8_u vhi\n\
  vs1 = i64.extend_i32_u vs1w\n\
  vpid = call.import 2 ()\n\
  vchild = i64.eq vpid vz\n\
  br_if vchild 3() 4(vpid, vs1)\n\
  }\n\
block 3 () {\n\
  vp = i64.const 40100\n\
  vz = i64.const 0\n\
  vr = call.import 0 (vp, vz, vz)\n\
  vnine = i64.const 9\n\
  call.import 1 (vnine)\n\
  unreachable\n\
  }\n\
block 4 (ypid: i64, ys1: i64) {\n\
  vst = i64.const 41100\n\
  vz = i64.const 0\n\
  vw = call.import 3 (ypid, vst, vz, vz)\n\
  vhi = i64.const 41101\n\
  vs2w = i32.load8_u vhi\n\
  vs2 = i64.extend_i32_u vs2w\n\
  vsum = i64.add ys1 vs2\n\
  call.import 1 (vsum)\n\
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
fn the_toolchain_runs_at_its_paths_and_runs_what_it_built() {
    let driver = module(DRIVER);
    let tool = module(SEVEN);
    // A program the build "wrote": a module's encoding in a file of the directory it runs in.
    let built = temen_encode::encode_module(&module(THREE_AND_FOUR));
    let b = nim_build(
        &driver,
        &[(&tool, vec!["/w/bin/c", "/bin/c"])],
        &[("/w/p", &built)],
        &[b"bin/driver"],
        "/w",
    )
    .expect("the interpreter tier runs the driver");
    assert_eq!(
        (b.status, b.exit_code),
        (STATUS_EXIT, 14),
        "both children ran what they exec'd: the tool at its second path, the built program by a \
         path relative to the build's directory\n--- stderr ---\n{}",
        String::from_utf8_lossy(&b.stderr)
    );
    assert_eq!(
        (b.footprint.windows, b.footprint.units),
        (1, 3),
        "the driver's window, and three programs: the driver, the tool, and the built program"
    );
}
