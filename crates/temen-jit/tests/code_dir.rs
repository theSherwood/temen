//! temen#2452 — compiled code kept on disk ([`CodeDir`]). A later process's compile of a module
//! defines the code an earlier process kept, instead of compiling it, and the code runs the same.
//! Code that is not intact, or not its owner's alone, is compiled again and kept anew; and code is
//! kept only in a directory its owner alone can write.

#![cfg(unix)]

use core::ffi::c_void;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use temen_jit::{CodeDir, CompiledModule, JitOutcome};
use temen_text::parse_module;

/// Its `call.cap` answers 41, and function 1 adds one to that: its code calls the host and a
/// function of its own.
const PROGRAM: &str = "memory 16
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = call.cap 2 0 () -> (i64) v0 ()
  v2 = call 1 (v1)
  return v2
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 1
  v2 = i64.add v0 v1
  return v2
  }
}
";

/// A `call.cap` that answers 41.
unsafe extern "C" fn forty_one(
    _ctx: *mut c_void,
    _mem_base: *mut u8,
    _mem_size: u64,
    _mem_reserved: u64,
    _type_id: u32,
    _op: u32,
    _handle: i32,
    _args: *const i64,
    _n_args: u64,
    results: *mut i64,
    _n_results: u64,
    trap_out: *mut i64,
) {
    unsafe {
        *results = 41;
        *trap_out = 0;
    }
}

/// Where [`compile_and_run_here`] finds its code directory, and writes what it saw.
const DIR: &str = "TEMEN_JIT_CODE_DIR";
const OUT: &str = "TEMEN_JIT_CODE_DIR_OUT";

/// The half of each test that runs in a process of its own, so no other test's compile counts:
/// compile [`PROGRAM`] keeping its code in `$DIR`, run it, and write how many modules the compile
/// compiled and what the program answered.
#[test]
#[ignore = "run in a process of its own by the tests below"]
fn compile_and_run_here() {
    let (Some(dir), Some(out)) = (std::env::var_os(DIR), std::env::var_os(OUT)) else {
        return;
    };
    let dir = CodeDir::open(PathBuf::from(dir)).expect("open the code directory");
    let m = parse_module(PROGRAM).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let before = temen_jit::module_compiles();
    let mut cm = CompiledModule::compile_kept(
        Some((&dir, temen_encode::digest256(PROGRAM.as_bytes()))),
        &m,
        0,
        forty_one,
        core::ptr::null_mut(),
        temen_ir::DEFAULT_RESERVED_LOG2,
        None,
        None,
        None,
        None,
        None,
        0,
    )
    .expect("compile");
    let compiled = temen_jit::module_compiles() - before;
    let answer = match cm.run(&[0], None, None).expect("run").0 {
        JitOutcome::Returned(r) => r[0],
        other => panic!("{other:?}"),
    };
    std::fs::write(out, format!("{compiled} {answer}")).expect("write what this process saw");
}

/// Run [`compile_and_run_here`] in a process of its own over `dir`: how many modules it compiled,
/// and what the program answered.
fn in_a_process(dir: &Path) -> (u64, i64) {
    let out = dir.with_extension("out");
    let run = std::process::Command::new(std::env::current_exe().expect("this test binary"))
        .args(["--exact", "compile_and_run_here", "--ignored"])
        .env(DIR, dir)
        .env(OUT, &out)
        .output()
        .expect("run this test binary");
    assert!(
        run.status.success(),
        "{}\n{}{}",
        run.status,
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    let saw = std::fs::read_to_string(&out).expect("what the process saw");
    let _ = std::fs::remove_file(&out);
    let (compiled, answer) = saw.split_once(' ').expect("two numbers");
    (compiled.parse().unwrap(), answer.parse().unwrap())
}

/// A directory of this test's own, not yet made.
fn fresh(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("temen-jit-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

/// The one file the directory keeps code in.
fn kept_file(dir: &Path) -> PathBuf {
    let files: Vec<PathBuf> = std::fs::read_dir(dir)
        .expect("read the code directory")
        .map(|e| e.expect("an entry").path())
        .collect();
    assert_eq!(files.len(), 1, "{files:?}");
    files.into_iter().next().unwrap()
}

#[test]
fn a_later_process_defines_the_code_an_earlier_one_kept() {
    let dir = fresh("kept");
    assert_eq!(in_a_process(&dir), (1, 42), "the first process compiles");
    assert_eq!(
        in_a_process(&dir),
        (0, 42),
        "a later process defines the code the first kept"
    );

    // A file that is not intact is a miss: the process compiles, and keeps its code anew.
    let file = kept_file(&dir);
    let mut bytes = std::fs::read(&file).expect("read the kept code");
    let mid = bytes.len() / 2;
    bytes[mid] ^= 1;
    std::fs::write(&file, &bytes).expect("damage the kept code");
    assert_eq!(
        in_a_process(&dir),
        (1, 42),
        "damaged code is compiled again"
    );
    assert_eq!(in_a_process(&dir), (0, 42), "and kept anew");

    // So is a file others can write.
    let file = kept_file(&dir);
    std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o620)).expect("chmod");
    assert_eq!(
        in_a_process(&dir),
        (1, 42),
        "code others could have written is compiled again"
    );
    assert_eq!(
        in_a_process(&dir),
        (0, 42),
        "and kept anew, its owner's alone"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn code_is_kept_only_where_its_owner_alone_can_write() {
    let dir = fresh("perms");
    CodeDir::open(&dir).expect("a directory it makes is its owner's alone");
    let mode = std::fs::metadata(&dir).expect("made").permissions().mode();
    assert_eq!(mode & 0o077, 0, "made for its owner alone: {mode:o}");
    for others in [0o770, 0o707, 0o722] {
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(others)).expect("chmod");
        let refused = CodeDir::open(&dir).map(|_| ()).unwrap_err();
        assert_eq!(
            refused.kind(),
            std::io::ErrorKind::PermissionDenied,
            "{others:o}"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}
