//! #2186 — `temen-link` links a **release** build unless asked for a **debug** build with `-g`, as
//! `cc` does. A nim program's only debug info is the prebuilt guest libc's (nim code carries none),
//! and a module with any turns on the Cranelift JIT's trap capture in every function.
//!
//! Toolchain-free: a minimal Leng program and a committed `system` unit, linked against the committed
//! guest libc through `link_command`, the one function the in-guest `temen-link` and the browser's
//! native one both are.

use std::collections::HashMap;

/// `main` only: it needs nothing of `system` or the libc, which the link brings in all the same.
const PROG: &str = "\
(stmts
 (proc :main.0.
  (params
   (param :argc.0 . (i 32))
   (param :argv.0 . (ptr (ptr (c 8))))
   (param :envp.0 . (ptr (ptr (c 8)))))
  (i 32)
  (pragmas (exportc \"main\"))
  (stmts . (ret 0))))";

/// `temen-link <args>` over the program's two units and the libc at its default path; the module it
/// writes.
fn temen_link(args: &[&str], system: &[u8], libc: &[u8]) -> temen_ir::Module {
    let files: HashMap<&str, &[u8]> = [
        ("prog.c.nif", PROG.as_bytes()),
        ("sysvq0asl.c.nif", system),
        ("/lib/temen/libc.temeno", libc),
    ]
    .into();
    let (names, sigs) = temen_posix::cap_vtable();
    let mut out = None;
    let status = temen_leng::link_command(
        args,
        (&names, &sigs),
        &mut |path| files.get(path).map(|b| b.to_vec()),
        &mut |_, bytes| {
            out = Some(bytes.to_vec());
            true
        },
    );
    assert_eq!(status, 0, "temen-link {args:?}");
    temen_encode::decode_module(&out.expect("written")).expect("decodes")
}

#[test]
fn temen_link_links_a_release_build_unless_asked_for_debug() {
    let Ok(libc) = std::fs::read("../../browser/web/assets/pg_libc.temeno") else {
        eprintln!("SKIP: browser/web/assets/pg_libc.temeno absent");
        return;
    };
    let system = std::fs::read("tests/fixtures/real_system_arc.leng.nif").expect("the system unit");
    let units = ["prog.c.nif", "sysvq0asl.c.nif"];
    let release = temen_link(&[&["-o:prog.temen"], &units[..]].concat(), &system, &libc);
    let debug = temen_link(
        &[&["-g", "-o:prog.temen"], &units[..]].concat(),
        &system,
        &libc,
    );
    assert!(
        release.debug_info.is_none(),
        "a release build carries no debug info"
    );
    assert!(debug.debug_info.is_some(), "a debug build keeps the libc's");
    // The mode is the debug info alone: the code is the same either way.
    assert_eq!(
        release.funcs, debug.funcs,
        "the code differs between the modes"
    );
    temen_verify::verify_module(&release).expect("the release build verifies");
}
