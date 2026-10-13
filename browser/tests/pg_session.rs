//! **The Postgres console's session protocol, over a small C program** (#2288, #1414 S4-2).
//!
//! `temen_pg_open` boots a program to its first stdin read and leaves it parked there. Each
//! `temen_pg_query` feeds it a line and runs it to the next read. Postgres itself needs staged
//! artifacts (`pg_snapshot_roundtrip.rs`). Any program that reads lines on stdin drives the same
//! exports, so this one echoes them and exits on `quit`. Fail-soft: SKIPs if `chibicc.temen` isn't
//! built.

use temen_browser::{STATUS_EXIT, STATUS_OK, STATUS_UNSUPPORTED};

#[path = "support/pg_heap.rs"]
mod pg_heap;

/// Prompt, echo each line, and `exit(3)` on `quit`. Each prompt is flushed, so it is read before the
/// next stdin read parks the program.
const REPL: &str = r#"#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main() {
  char line[64];
  printf("ready> ");
  fflush(stdout);
  while (fgets(line, sizeof line, stdin)) {
    if (strncmp(line, "quit", 4) == 0) exit(3);
    printf("echo: %s", line);
    printf("ready> ");
    fflush(stdout);
  }
  return 0;
}
"#;

fn chibicc_temen() -> Option<temen_ir::Module> {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/web/assets/chibicc.temen");
    let bytes = std::fs::read(p).ok()?;
    Some(temen_encode::decode_module(&bytes).expect("decode chibicc.temen"))
}

/// The REPL compiled by the shipped chibicc and linked with the playground heap, encoded as
/// `temen_pg_open` takes a module.
fn repl_module(chibicc: &temen_ir::Module) -> Vec<u8> {
    let mut files: Vec<(String, Vec<u8>)> = temen_browser::playground_include_files();
    files.push(("in.c".to_string(), REPL.as_bytes().to_vec()));
    let image = temen_fs::encode_image(&files, &["include".to_string()]);
    let argv: [&[u8]; 5] = [
        b"chibicc",
        b"--data-page",
        b"65536",
        b"--emit-object",
        b"/in.c",
    ];
    let compiled = temen_browser::onramp_fs_exec(chibicc, &image, &argv, b"");
    assert!(
        compiled.status == STATUS_OK || compiled.status == STATUS_EXIT,
        "compile status {}: {}",
        compiled.status,
        String::from_utf8_lossy(&compiled.stderr)
    );
    let ir = String::from_utf8(compiled.stdout).expect("IR is utf8");
    let unit = temen_text::parse_module(&ir).unwrap_or_else(|e| panic!("parse IR: {e:?}"));
    temen_encode::encode_module(&pg_heap::link(&[], &unit))
}

/// The stdout the last `temen_pg_*` call handed back: the bytes the program wrote since the call
/// before.
fn engine_stdout() -> String {
    let ptr = temen_browser::temen_stdout_ptr();
    let len = temen_browser::temen_stdout_len();
    if ptr.is_null() || len == 0 {
        return String::new();
    }
    // SAFETY: `temen_stdout_ptr/_len` name a live cdylib allocation, valid until the next capturing call.
    String::from_utf8_lossy(unsafe { std::slice::from_raw_parts(ptr, len) }).into_owned()
}

fn query(sql: &str) -> (i32, String) {
    let rc = temen_browser::temen_pg_query(sql.as_ptr(), sql.len());
    (rc, engine_stdout())
}

#[test]
fn a_session_parks_between_queries_and_reports_its_exit() {
    let Some(chibicc) = chibicc_temen() else {
        eprintln!("SKIP: chibicc.temen absent");
        return;
    };
    let module = repl_module(&chibicc);
    let image = temen_fs::encode_image(&[], &[]);

    let rc =
        temen_browser::temen_pg_open(module.as_ptr(), module.len(), image.as_ptr(), image.len());
    assert_eq!(rc, 0, "the program boots to its first read and parks there");
    assert_eq!(engine_stdout(), "ready> ");

    assert_eq!(query("hello"), (0, "echo: hello\nready> ".to_string()));
    assert_eq!(
        query("again\n"),
        (0, "echo: again\nready> ".to_string()),
        "a line that already ends in a newline gets no second one"
    );

    // #2288: a program that calls `exit` has exited. It used to read as a trap.
    let (rc, _) = query("quit");
    assert_eq!(rc, -STATUS_EXIT, "`exit(3)` ends the session as an exit");
    assert_eq!(temen_browser::temen_status(), STATUS_EXIT);

    let (rc, _) = query("hello");
    assert_eq!(
        rc, -STATUS_UNSUPPORTED,
        "an ended session takes no more queries"
    );
    temen_browser::temen_pg_close();
}
