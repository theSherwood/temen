//! **#2294: a nim program links a C library unit beside the libc.** `temen_leng::link_nim`'s `libs`
//! are encoded `.temeno` units whose functions take the on-ramp's ABI, the data-stack pointer first,
//! at both ends of a call. This pins the seam **toolchain-free**, so it gates every PR and not only
//! the `nim-e2e` job, whose `a_nim_program_links_a_c_library` runs the same shape from real Nim.
//!
//! The library here is written the way the on-ramp translates C: every function takes `$sp` first
//! and hands it to the functions it calls; a capability it imports takes none.
//!
//! - `lib_twice(x)` returns `host_add(x, x)`, a callback the program serves with an `exportc` proc.
//!   Its body needs no frame, so without the on-ramp ABI it would take no `$sp` and the link would
//!   refuse the library's call.
//! - `lib_quit(code)` ends the program through `exit`, a capability: the program's manifest asks the
//!   host for it, and the host's powerbox serves it.

use temen_leng::{Edge, WholeModule};

/// The program: `main` quits with `lib_twice(21)`, and `host_add` is the library's callback. Leng as
/// nimony emits it: an `importc` proc has no body, and `main` is the C entry `_start` calls.
const PROG: &str = "\
(stmts
 (proc :lib_twice.0. (params (param :x.0 . (i +64))) (i +64) (pragmas (importc \"lib_twice\")) (stmts .))
 (proc :lib_quit.0. (params (param :code.0 . (i +64))) . (pragmas (importc \"lib_quit\")) (stmts .))
 (proc :host_add.0.
  (params (param :a.0 . (i +64)) (param :b.0 . (i +64)))
  (i +64)
  (pragmas (exportc \"host_add\"))
  (stmts . (ret (add (i +64) a.0 b.0))))
 (proc :main.0.
  (params
   (param :argc.0 . (i +32))
   (param :argv.0 . (ptr (ptr (c +8))))
   (param :envp.0 . (ptr (ptr (c +8)))))
  (i +32)
  (pragmas (exportc \"main\"))
  (stmts .
   (call lib_quit.0. (call lib_twice.0. 21))
   (ret 0))))";

/// `link_nim` finds the program's `system` module by its stem. This program uses nothing from it.
const SYSTEM: &str = "(stmts)";

/// The C library, as the on-ramp would translate it.
const LIB: &str = "\
import 0 \"host_add\" (i64, i64, i64) -> (i64)
import 1 \"exit\" (i32) -> ()
export 0 func \"lib_twice\" 0
export 1 func \"lib_quit\" 1

func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = call.import 0 (v0, v1, v1)
  return v2
  }
}

func (i64, i64) -> () {
block 0 (v0: i64, v1: i64) {
  v2 = i32.wrap_i64 v1
  call.import 1 (v2)
  return
  }
}";

/// The library as the `.temeno` its toolchain would write.
fn library() -> Vec<u8> {
    temen_encode::encode_unit(&temen_text::parse_module(LIB).expect("parse the library"))
}

fn link() -> temen_ir::Module {
    let units = [
        WholeModule {
            stem: "sysvstub",
            src: SYSTEM,
        },
        WholeModule {
            stem: "prog",
            src: PROG,
        },
    ];
    let m = temen_leng::link_nim(
        &units,
        Edge::Caps,
        None,
        &[&library()],
        temen_leng::Build::default(),
    )
    .unwrap_or_else(|e| panic!("link: {e}"));
    temen_verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    m
}

/// The library's functions and its callback are bound inside the program, and its capability is the
/// host's: `exit` is the only import the library adds to the manifest.
#[test]
fn a_c_library_binds_inside_the_program_and_asks_the_host_for_its_capability() {
    let m = link();
    let imports: Vec<&str> = m.imports.iter().map(|i| i.name.as_str()).collect();
    for bound in ["lib_twice", "lib_quit", "host_add"] {
        assert!(
            !imports.iter().any(|i| i.contains(bound)),
            "`{bound}` is bound inside the program, not imported: {imports:?}"
        );
    }
    assert!(
        imports.contains(&"exit"),
        "the library's capability is in the manifest: {imports:?}"
    );
}

/// The program calls into the library and the library calls back, on every engine: `$sp` crosses
/// the boundary both ways, and the host's `exit` ends the run with the callback's answer.
#[test]
fn a_c_library_and_its_callback_run_on_every_engine() {
    let m = link();
    for engine in [
        temen_run::Backend::TreeWalk,
        temen_run::Backend::Bytecode,
        temen_run::Backend::Jit,
    ] {
        let run = temen_run::instantiate(m.clone())
            .and_then(|i| i.run(engine, &temen_run::RunConfig::default()))
            .unwrap_or_else(|e| panic!("{engine:?}: {e}"));
        assert!(
            matches!(run.outcome, temen_run::Outcome::Exited(42)),
            "{engine:?}: lib_quit(lib_twice(21)) exits 42, got {:?}",
            run.outcome
        );
    }
}

/// `temen-link` takes a library as a `.temeno` input, which is how nimony passes the file a
/// `{.link.}` pragma names. The module it writes is the one `link_nim` links.
#[test]
fn temen_link_takes_a_library_as_a_temeno_input() {
    let lib = library();
    let files: std::collections::HashMap<&str, &[u8]> = [
        ("prog.c.nif", PROG.as_bytes()),
        ("sysvstub.c.nif", SYSTEM.as_bytes()),
        ("/opt/lib/c.temeno", lib.as_slice()),
    ]
    .into();
    let (names, sigs) = temen_posix::cap_vtable();
    let mut out = None;
    let status = temen_leng::link_command(
        &[
            "--edge:caps",
            "-o:prog.temen",
            "prog.c.nif",
            "sysvstub.c.nif",
            "/opt/lib/c.temeno",
        ],
        (&names, &sigs),
        &mut |path| files.get(path).map(|b| b.to_vec()),
        &mut |_, bytes| {
            out = Some(bytes.to_vec());
            true
        },
    );
    assert_eq!(status, 0, "temen-link links the program with its library");
    assert!(
        out.as_deref() == Some(temen_encode::encode_module(&link()).as_slice()),
        "temen-link wrote the module link_nim links"
    );
}
