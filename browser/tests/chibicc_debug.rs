//! Source-level debugging of a chibicc-compiled C program (the playground's Debug button, end to end
//! in Rust): compile a compute-only C source **with `-g`** through `chibicc.svmb`, then drive the
//! `svm-dap` server (the bytecode backend the playground runs) over the emitted IR — set a breakpoint on
//! a **C source line**, run to it, and read the paused frame's **C locals by name**. This proves the
//! debug-info path the browser Debug button wires: chibicc's `-g` `debug.file`/`debug.loc`/`debug.var`
//! waist lets the DAP bind breakpoints to C lines and name C variables, on the compiled program.
//!
//! Two modes: a compute-only program debugs under the deny-all backend; a **capability-using** program
//! (a `printf` → a `write` cap) debugs under the on-ramp **I/O powerbox** (`launch` arg
//! `powerbox: "onramp"`), which runs it instead of `CapFault`ing, captures its output as DAP `output`
//! events, and — via the CapTape replay — **rewinds that output on reverse debugging**.
//!
//! Fail-soft on a missing `chibicc.svmb` (a fresh tree without the build), like `chibicc_printf.rs`.

use svm_browser::{onramp_fs_exec, playground_include_files, STATUS_EXIT, STATUS_OK};
use svm_dap::{DapServer, Json};

fn chibicc_svmb() -> Option<Vec<u8>> {
    std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/web/assets/chibicc.svmb"
    ))
    .ok()
}

/// Compile `src` with `-g` (debug info on) via the shipped compiler, returning the emitted SVM-IR text.
fn compile_g(chibicc: &svm_ir::Module, src: &str) -> String {
    let mut files = playground_include_files();
    files.push(("in.c".to_string(), src.as_bytes().to_vec()));
    let image = svm_fs::encode_image(&files, &vec!["include".to_string()]);
    let out = onramp_fs_exec(
        chibicc,
        &image,
        &[b"chibicc", b"--data-page", b"65536", b"-g", b"/in.c"],
        b"",
    );
    assert!(
        out.status == STATUS_OK || out.status == STATUS_EXIT,
        "compile status {}",
        out.status
    );
    String::from_utf8(out.stdout).expect("IR utf8")
}

fn req(seq: i64, command: &str, args: Json) -> Json {
    Json::obj(vec![
        ("seq", Json::i(seq)),
        ("type", Json::s("request")),
        ("command", Json::s(command)),
        ("arguments", args),
    ])
}
fn response(msgs: &[Json]) -> &Json {
    msgs.iter()
        .find(|m| m.get("type").and_then(|t| t.as_str()) == Some("response"))
        .expect("a response")
}
fn event<'a>(msgs: &'a [Json], name: &str) -> bool {
    msgs.iter().any(|m| {
        m.get("type").and_then(|t| t.as_str()) == Some("event")
            && m.get("event").and_then(|e| e.as_str()) == Some(name)
    })
}
/// The text of the last `output` event (category stdout) in a batch — the guest's captured stdout at
/// this stop (full, not a delta: it rewinds on a reverse `seek`). `None` if the batch carried none.
fn output_text(msgs: &[Json]) -> Option<String> {
    msgs.iter()
        .rev()
        .find(|m| m.get("event").and_then(|e| e.as_str()) == Some("output"))
        .and_then(|m| m.get("body"))
        .and_then(|b| b.get("output"))
        .and_then(|o| o.as_str())
        .map(|s| s.to_string())
}

// A compute-only C program (no libc / powerbox): sum 3+2+1 = 6. The `acc += i` line is the breakpoint
// target; `i` and `acc` are the C locals inspected there.
const SRC: &str = r#"int main(void) {
  int acc = 0;
  int i = 3;
  while (i > 0) {
    acc += i;
    i -= 1;
  }
  return acc;
}
"#;
const BP_LINE: i64 = 5; // the `acc += i;` line

#[test]
fn debug_a_chibicc_compiled_program_at_c_source_level() {
    let Some(bytes) = chibicc_svmb() else {
        eprintln!("SKIP: browser/web/assets/chibicc.svmb absent (run build-onramp-assets.mjs)");
        return;
    };
    let chibicc = svm_encode::decode_module(&bytes).expect("decode chibicc.svmb");
    let ir = compile_g(&chibicc, SRC);
    // The emitted IR carries chibicc's -g waist naming the C source and its locals.
    assert!(
        ir.contains(r#"debug.file 0 "/in.c""#),
        "-g IR names the C source /in.c"
    );
    assert!(
        ir.contains(r#""i""#) && ir.contains(r#""acc""#),
        "-g IR names the C locals"
    );

    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    let out = s.handle(&req(
        2,
        "launch",
        Json::obj(vec![
            ("programText", Json::s(&ir)),
            ("function", Json::i(0)),
            ("args", Json::Arr(vec![])),
            ("engine", Json::s("bytecode")),
        ]),
    ));
    assert_eq!(
        response(&out).get("success"),
        Some(&Json::Bool(true)),
        "launch ok"
    );

    // Breakpoint on the C `acc += i;` line — bound through chibicc's debug.loc (C line → IR pc).
    let out = s.handle(&req(
        3,
        "setBreakpoints",
        Json::obj(vec![
            ("source", Json::obj(vec![("path", Json::s("/in.c"))])),
            (
                "breakpoints",
                Json::Arr(vec![Json::obj(vec![("line", Json::i(BP_LINE))])]),
            ),
        ]),
    ));
    let bps = response(&out)
        .get("body")
        .unwrap()
        .get("breakpoints")
        .unwrap();
    let bp0 = &bps.as_array().unwrap()[0];
    assert_eq!(
        bp0.get("verified"),
        Some(&Json::Bool(true)),
        "the C-line breakpoint bound"
    );

    // Run to it — a `stopped` event (not run to completion) means the C-line breakpoint fired.
    let out = s.handle(&req(4, "configurationDone", Json::obj(vec![])));
    assert!(
        event(&out, "stopped"),
        "stopped at the C-line breakpoint (not run to completion)"
    );

    // The top frame is `main`, at the C line (source `/in.c`); its locals name the C variables. The DAP
    // formats a frame name as `#<n> <fn>`, so match on the function name within it.
    let out = s.handle(&req(
        5,
        "stackTrace",
        Json::obj(vec![("threadId", Json::i(1))]),
    ));
    let top = &response(&out)
        .get("body")
        .unwrap()
        .get("stackFrames")
        .unwrap()
        .as_array()
        .unwrap()[0];
    let fname = top.get("name").and_then(|n| n.as_str()).unwrap_or("");
    assert!(fname.contains("main"), "stopped in main (frame {fname:?})");
    assert_eq!(
        top.get("line").and_then(|l| l.as_i64()),
        Some(BP_LINE),
        "on the C `acc += i;` line"
    );
    assert_eq!(
        top.get("source")
            .and_then(|s| s.get("path"))
            .and_then(|p| p.as_str()),
        Some("/in.c"),
        "frame source is the C file"
    );
    let fid = top.get("id").unwrap().as_i64().unwrap();

    let out = s.handle(&req(
        6,
        "scopes",
        Json::obj(vec![("frameId", Json::i(fid))]),
    ));
    let vref = response(&out)
        .get("body")
        .unwrap()
        .get("scopes")
        .unwrap()
        .as_array()
        .unwrap()[0]
        .get("variablesReference")
        .unwrap()
        .as_i64()
        .unwrap();
    let out = s.handle(&req(
        7,
        "variables",
        Json::obj(vec![("variablesReference", Json::i(vref))]),
    ));
    let vars = response(&out)
        .get("body")
        .unwrap()
        .get("variables")
        .unwrap()
        .as_array()
        .unwrap();
    let names: std::collections::HashSet<&str> = vars
        .iter()
        .filter_map(|v| v.get("name").and_then(|n| n.as_str()))
        .collect();
    assert!(
        names.contains("i") && names.contains("acc"),
        "C locals resolve by name: {names:?}"
    );

    // Continue to termination — the compute program returns without needing a powerbox.
    let out = s.handle(&req(
        8,
        "continue",
        Json::obj(vec![("threadId", Json::i(1))]),
    ));
    // It may take several breakpoint hits (the loop); drive to the terminated event.
    let mut terminated = event(&out, "terminated");
    let mut seq = 9;
    while !terminated && seq < 40 {
        let out = s.handle(&req(
            seq,
            "continue",
            Json::obj(vec![("threadId", Json::i(1))]),
        ));
        terminated = event(&out, "terminated");
        seq += 1;
    }
    assert!(terminated, "the debugged compute program ran to completion");
}

// A capability-using C program: two `printf`s (each a `write` powerbox cap) then return. Under the
// on-ramp I/O powerbox the debugger runs it (no CapFault) and captures its output; a breakpoint on the
// second printf's line stops with only the first line printed.
const PRINTF_SRC: &str = r#"#include <stdio.h>
int main(void) {
  printf("A\n");
  printf("B\n");
  return 0;
}
"#;
const PRINTF_BP: i64 = 4; // the `printf("B\n");` line — stop here with only "A\n" printed so far

fn launch_printf(s: &mut DapServer, ir: &str) {
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    let out = s.handle(&req(
        2,
        "launch",
        Json::obj(vec![
            ("programText", Json::s(ir)),
            ("function", Json::i(0)),
            ("args", Json::Arr(vec![])),
            ("engine", Json::s("bytecode")),
            ("powerbox", Json::s("onramp")), // run under the on-ramp I/O powerbox (write/exit/…)
        ]),
    ));
    assert_eq!(response(&out).get("success"), Some(&Json::Bool(true)), "powerbox launch ok");
    s.handle(&req(
        3,
        "setBreakpoints",
        Json::obj(vec![
            ("source", Json::obj(vec![("path", Json::s("/in.c"))])),
            (
                "breakpoints",
                Json::Arr(vec![Json::obj(vec![("line", Json::i(PRINTF_BP))])]),
            ),
        ]),
    ));
}

/// **Capability-using C debugs under the powerbox**: a `printf` program runs (instead of `CapFault`ing),
/// its output is captured as DAP `output` events, and **reverse debugging rewinds the output** — the
/// CapTape/replay makes a `reverseContinue` reproduce the exact earlier stdout.
#[test]
fn debug_a_printf_program_with_captured_output_and_reverse() {
    let Some(bytes) = chibicc_svmb() else {
        eprintln!("SKIP: chibicc.svmb absent");
        return;
    };
    let chibicc = svm_encode::decode_module(&bytes).expect("decode");
    let ir = compile_g(&chibicc, PRINTF_SRC);

    let mut s = DapServer::new();
    launch_printf(&mut s, &ir);

    // Run to the second printf: it stops there (no CapFault at the first printf), and the captured
    // output so far is exactly "A\n" — the first printf ran, the second hasn't.
    let out = s.handle(&req(4, "configurationDone", Json::obj(vec![])));
    assert!(event(&out, "stopped"), "stopped at the C breakpoint (the write cap did not fault)");
    let at_bp = output_text(&out).expect("an output event carried the guest's stdout");
    assert_eq!(at_bp, "A\n", "only the first printf has run at the breakpoint");

    // Continue to completion: both printfs have now run, so the final output is "A\nB\n".
    let out = s.handle(&req(5, "continue", Json::obj(vec![("threadId", Json::i(1))])));
    assert!(event(&out, "terminated"), "ran to completion");
    assert_eq!(output_text(&out).as_deref(), Some("A\nB\n"), "both printfs ran by the end");

    // Reverse back to the breakpoint: the run is rebuilt + replayed to that op, so the captured output
    // **rewinds** to "A\n" — the CapTape replay reproduces the earlier stdout exactly.
    let out = s.handle(&req(6, "reverseContinue", Json::obj(vec![("threadId", Json::i(1))])));
    assert!(event(&out, "stopped"), "reverse stopped back at the breakpoint");
    assert_eq!(
        output_text(&out).as_deref(),
        Some("A\n"),
        "reverse debugging rewound the captured output to the earlier point"
    );
}
