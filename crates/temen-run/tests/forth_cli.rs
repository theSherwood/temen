//! #1237 — **the interactive Forth REPL through the real CLI.** `temen-run forth.temt --interactive`
//! (the default on a terminal) wires the process's stdin to the guest's `read` a line at a time and
//! streams its stdout per write, and the kernel's `_start` interprets each line as it arrives. So the
//! session is *live*: every line is answered before the next is typed, a colon definition may span
//! lines, and a runtime `key` waits for the next line. Driven through pipes here — each expectation is
//! read back **before** the next line is written, which a slurp-then-run CLI could never satisfy.

#![cfg(all(unix, target_arch = "x86_64"))]

use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::Duration;

/// One byte at a time off the child's stdout until `want` has been seen (or the reader thread dies).
fn read_until(rx: &mpsc::Receiver<u8>, want: &str) -> String {
    let mut got = Vec::new();
    while !String::from_utf8_lossy(&got).contains(want) {
        match rx.recv_timeout(Duration::from_secs(30)) {
            Ok(b) => got.push(b),
            Err(e) => panic!(
                "waiting for {want:?}, got {:?}: {e}",
                String::from_utf8_lossy(&got)
            ),
        }
    }
    String::from_utf8(got).expect("utf-8 stdout")
}

#[test]
fn the_interactive_cli_answers_each_line_as_it_is_typed() {
    let kernel = concat!(env!("CARGO_MANIFEST_DIR"), "/demos/forth/forth.temt");
    let mut child = Command::new(env!("CARGO_BIN_EXE_temen-run"))
        .arg(kernel)
        .arg("--interactive")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn temen-run");
    let mut stdin = child.stdin.take().expect("stdin pipe");
    let mut stdout = child.stdout.take().expect("stdout pipe");
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut b = [0u8; 1];
        while let Ok(1) = stdout.read(&mut b) {
            if tx.send(b[0]).is_err() {
                break;
            }
        }
    });
    let mut say = |line: &str| {
        stdin.write_all(line.as_bytes()).expect("write line");
        stdin.flush().expect("flush");
    };

    // Answered before the next line exists.
    say("1 2 + . cr\n");
    assert_eq!(read_until(&rx, "\n"), "3 \n");
    // A definition across two lines: nothing is printed until it closes, then it is callable.
    say(": sq ( n -- n )\n");
    say("  dup * ;\n7 sq . cr\n");
    assert_eq!(read_until(&rx, "\n"), "49 \n");
    // The REPL stack persists between lines; an error is reported on its line and the session goes on.
    say("10 20\n");
    say("bogus\n");
    assert_eq!(read_until(&rx, "\n"), "line 6: unknown word near bogus\n");
    say("+ . cr\n");
    assert_eq!(read_until(&rx, "\n"), "30 \n");
    // A runtime `key` past the current line waits for the next one.
    say("key . key . cr\n");
    say("AB\n");
    assert_eq!(read_until(&rx, "\n"), "65 66 \n");
    // A fiber created on one line is resumed on a later one.
    say(": counter ( x -- y ) begin 1+ dup yield drop again ;\n");
    say("' counter task dup 0 resume . . cr\n");
    assert_eq!(read_until(&rx, "\n"), "1 0 \n");
    say("10 resume . . cr\n");
    assert_eq!(read_until(&rx, "\n"), "2 0 \n");

    drop(stdin);
    let status = child.wait().expect("wait temen-run");
    assert!(status.success(), "exit status {status:?}");
    let mut err = String::new();
    child
        .stderr
        .take()
        .expect("stderr pipe")
        .read_to_string(&mut err)
        .ok();
    assert!(err.is_empty(), "stderr: {err}");
}

/// Without `--interactive` (and with stdin not a terminal) the CLI is unchanged: the guest's stdin is
/// `--stdin FILE` or empty, and stdout is printed once at exit.
#[test]
fn without_interactive_the_cli_still_batches_a_stdin_file() {
    let kernel = concat!(env!("CARGO_MANIFEST_DIR"), "/demos/forth/forth.temt");
    let dir = std::env::temp_dir().join(format!("forth_cli_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("tmp dir");
    let program = dir.join("program.fs");
    std::fs::write(&program, ": sq ( n -- n ) dup * ;\n7 sq . cr\n").expect("write program");
    let out = Command::new(env!("CARGO_BIN_EXE_temen-run"))
        .arg(kernel)
        .arg("--stdin")
        .arg(&program)
        .stdin(Stdio::null())
        .output()
        .expect("run temen-run");
    let _ = std::fs::remove_dir_all(&dir);
    assert!(out.status.success(), "exit status {:?}", out.status);
    assert_eq!(String::from_utf8_lossy(&out.stdout), "49 \n");
}
