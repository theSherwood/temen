//! The browser's entry for nimony's own driver ([`temen_browser::nim_build`], #958), on a stand-in
//! driver: what the toolchain's processes rely on besides the engine. A command runs at every path it
//! is registered at (nimony finds a tool beside itself, the shell by `PATH`), the build runs in its
//! directory over the files it was given, a program the build wrote is runnable, as a compile-time
//! evaluation's is, and `temen-link` is there without being given. The real driver is measured by
//! `src/nimbuild.rs`.

use temen_browser::{library_pack, nim_build, STATUS_EXIT};

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

/// Forks twice. The first child execs `a`, the second `b` (each an argv, whose first is the path
/// exec'd), and each exits 9 if its exec failed. The driver reaps both and exits with the sum of their
/// exit statuses.
fn driver(a: &[&str], b: &[&str]) -> String {
    // Each child's strings at 50000 + 2000·child + 200·arg, its argv array at 56000 + 100·child.
    let mut data = String::new();
    for (k, argv) in [a, b].iter().enumerate() {
        let mut ptrs = String::new();
        for (j, arg) in argv.iter().enumerate() {
            let at = 50000 + 2000 * k + 200 * j;
            data += &format!("data {at} \"{arg}\\x00\"\n");
            ptrs.extend(
                (at as u64)
                    .to_le_bytes()
                    .iter()
                    .map(|b| format!("\\x{b:02x}")),
            );
        }
        ptrs += &"\\x00".repeat(8);
        data += &format!("data {} \"{ptrs}\"\n", 56000 + 100 * k);
    }
    let exec = |k: usize| {
        format!(
            "  vp = i64.const {}\n  va = i64.const {}\n  vz = i64.const 0\n  vr = call.import 0 (vp, va, vz)\n",
            50000 + 2000 * k,
            56000 + 100 * k
        )
    };
    format!(
        "memory 17\n\
import 0 \"__px_execve\" (i64, i64, i64) -> (i64)\n\
import 1 \"__px_exit\" (i64) -> ()\n\
import 2 \"__px_fork\" () -> (i64)\n\
import 3 \"__px_wait4\" (i64, i64, i64, i64) -> (i64)\n\
{data}\
func () -> () {{\n\
block 0 () {{\n\
  vpid = call.import 2 ()\n\
  vz = i64.const 0\n\
  vchild = i64.eq vpid vz\n\
  br_if vchild 1() 2(vpid)\n\
  }}\n\
block 1 () {{\n\
{first}\
  vnine = i64.const 9\n\
  call.import 1 (vnine)\n\
  unreachable\n\
  }}\n\
block 2 (xpid: i64) {{\n\
  vst = i64.const 41000\n\
  vz = i64.const 0\n\
  vw = call.import 3 (xpid, vst, vz, vz)\n\
  vhi = i64.const 41001\n\
  vs1w = i32.load8_u vhi\n\
  vs1 = i64.extend_i32_u vs1w\n\
  vpid = call.import 2 ()\n\
  vchild = i64.eq vpid vz\n\
  br_if vchild 3() 4(vpid, vs1)\n\
  }}\n\
block 3 () {{\n\
{second}\
  vnine = i64.const 9\n\
  call.import 1 (vnine)\n\
  unreachable\n\
  }}\n\
block 4 (ypid: i64, ys1: i64) {{\n\
  vst = i64.const 41100\n\
  vz = i64.const 0\n\
  vw = call.import 3 (ypid, vst, vz, vz)\n\
  vhi = i64.const 41101\n\
  vs2w = i32.load8_u vhi\n\
  vs2 = i64.extend_i32_u vs2w\n\
  vsum = i64.add ys1 vs2\n\
  call.import 1 (vsum)\n\
  unreachable\n\
  }}\n\
}}\n\
export 0 func \"_start\" 0\n",
        first = exec(0),
        second = exec(1)
    )
}

fn module(text: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(text).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// Both ways a build runs its processes: interpreted, and with each leaf process — here the tool and
/// the built program, which import nothing and so cannot park — tiered up at its entry (#1896).
#[test]
fn the_toolchain_runs_at_its_paths_and_runs_what_it_built() {
    let driver = module(&driver(&["/bin/c"], &["./p"]));
    let tool = module(SEVEN);
    // A program the build "wrote": a module's encoding in a file of the directory it runs in.
    let built = temen_encode::encode_module(&module(THREE_AND_FOUR));
    for leaves in [false, true] {
        let b = nim_build(
            &driver,
            &[(&tool, vec!["/w/bin/c", "/bin/c"])],
            &[("/w/p", &built)],
            &[b"bin/driver"],
            "/w",
            leaves,
        )
        .expect("the interpreter tier runs the driver");
        assert_eq!(
            (b.status, b.exit_code),
            (STATUS_EXIT, 14),
            "both children ran what they exec'd: the tool at its second path, the built program by \
             a path relative to the build's directory (leaves: {leaves})\n--- stderr ---\n{}",
            String::from_utf8_lossy(&b.stderr)
        );
        assert_eq!(
            (b.footprint.windows, b.footprint.units),
            (1, 3),
            "the driver's window, and three programs: the driver, the tool, and the built program"
        );
        assert_eq!(
            b.leaves,
            if leaves { 2 } else { 0 },
            "the two children ran as leaves"
        );
    }
}

/// nimony's Temen backend links through `temen-link`, which the engine serves natively ([`nim_build`]
/// registers it): a build finds it where the driver looks for a tool and where the shell does, without
/// being given it, and it is [`temen_leng::link_command`] over the files of the process that exec'd
/// it, whose status the process exits with. The unit here is one module of a program, which the link
/// refuses (3: a program links with its `system`); an input that is not there is 2.
#[test]
fn temen_link_is_served_natively_at_its_paths() {
    let unit = include_bytes!("../../crates/temen-leng/tests/fixtures/real_module.leng.nif");
    let driver = module(&driver(
        &["/w/bin/temen-link", "-o:a.temen", "m.c.nif"],
        &["/bin/temen-link", "-o:b.temen", "gone.c.nif"],
    ));
    let b = nim_build(
        &driver,
        &[],
        &[("/w/m.c.nif", unit)],
        &[b"bin/driver"],
        "/w",
        false,
    )
    .expect("the interpreter tier runs the driver");
    let (names, sigs) = temen_posix::cap_vtable();
    let link = |input: &str| {
        temen_leng::link_command(
            &["-o:x.temen", input],
            (&names, &sigs),
            &mut |path| (path == "m.c.nif").then(|| unit.to_vec()),
            &mut |_, _| true,
        )
    };
    let (refused, missing) = (link("m.c.nif"), link("gone.c.nif"));
    assert_eq!((refused, missing), (3, 2));
    assert_eq!(
        (b.status, b.exit_code),
        (STATUS_EXIT, refused + missing),
        "each path ran the link over the process's files\n--- stderr ---\n{}",
        String::from_utf8_lossy(&b.stderr)
    );
}

/// A build's library pack is what it wrote under its `nimcache/` for library modules — each module
/// told by the source its `.p.nif` records — and nimony's options memo, in the order they were
/// written, which is the order that keeps them newer than their sources when seeded.
#[test]
fn a_library_pack_is_the_librarys_cache_in_write_order() {
    let mut host = temen_interp::Host::new();
    let (_px, posix) = temen_posix::grant(&mut host, 0, 0, Vec::new());
    let files: [(&str, &[u8]); 9] = [
        ("/w/lib/std/sys.nim", b"proc x() = discard"),
        ("/w/prog.nim", b"echo 1"),
        ("/w/nimcache/cachedconfigfile.txt", b" -d:temen"),
        (
            "/w/nimcache/sys1.p.nif",
            b"(.nif24)\n(stmts@,1,lib/std/sys.nim (proc))",
        ),
        (
            "/w/nimcache/pro2.p.nif",
            b"(.nif24)\n(stmts@,1,prog.nim (call))",
        ),
        ("/w/nimcache/sys1.s.nif", b"sem"),
        ("/w/nimcache/pro2.s.nif", b"sem"),
        ("/w/nimcache/sys1.temen/sys1.c.nif", b"lowered"),
        ("/w/nimcache/pro2.temen/sys1.c.nif", b"lowered for pro2"),
    ];
    for (path, bytes) in files {
        posix.write_file(path, bytes);
    }
    // Rewritten last: it goes where its latest write puts it.
    posix.write_file("/w/nimcache/sys1.p.nif", files[3].1);
    let pack = library_pack(&posix, "/w");
    let names: Vec<&str> = pack.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(
        names,
        [
            "/w/nimcache/cachedconfigfile.txt",
            "/w/nimcache/sys1.s.nif",
            "/w/nimcache/sys1.temen/sys1.c.nif",
            "/w/nimcache/sys1.p.nif",
        ]
    );
    assert!(pack
        .iter()
        .all(|(n, b)| posix.read_file(n).as_ref() == Some(b)));
}
