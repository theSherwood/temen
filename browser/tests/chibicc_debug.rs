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
    assert_eq!(
        response(&out).get("success"),
        Some(&Json::Bool(true)),
        "powerbox launch ok"
    );
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
    assert!(
        event(&out, "stopped"),
        "stopped at the C breakpoint (the write cap did not fault)"
    );
    let at_bp = output_text(&out).expect("an output event carried the guest's stdout");
    assert_eq!(
        at_bp, "A\n",
        "only the first printf has run at the breakpoint"
    );

    // Continue to completion: both printfs have now run, so the final output is "A\nB\n".
    let out = s.handle(&req(
        5,
        "continue",
        Json::obj(vec![("threadId", Json::i(1))]),
    ));
    assert!(event(&out, "terminated"), "ran to completion");
    assert_eq!(
        output_text(&out).as_deref(),
        Some("A\nB\n"),
        "both printfs ran by the end"
    );

    // Reverse back to the breakpoint: the run is rebuilt + replayed to that op, so the captured output
    // **rewinds** to "A\n" — the CapTape replay reproduces the earlier stdout exactly.
    let out = s.handle(&req(
        6,
        "reverseContinue",
        Json::obj(vec![("threadId", Json::i(1))]),
    ));
    assert!(
        event(&out, "stopped"),
        "reverse stopped back at the breakpoint"
    );
    assert_eq!(
        output_text(&out).as_deref(),
        Some("A\n"),
        "reverse debugging rewound the captured output to the earlier point"
    );
}

/// **Step Back is depth-aware** — the reverse of *step over*, not *step in*. After a `printf` line has
/// run (the call descended into the guest libc and returned), `stepBack` rewinds **within `main`**, never
/// down into the libc internals (`__pf_flush` / `stdio.h`). Regression test for the bug where step-back
/// descended into the callee's last op instead of the caller's previous op.
#[test]
fn step_back_stays_in_the_users_frame_across_a_printf() {
    let Some(bytes) = chibicc_svmb() else {
        eprintln!("SKIP: chibicc.svmb absent");
        return;
    };
    let chibicc = svm_encode::decode_module(&bytes).expect("decode");
    let ir = compile_g(&chibicc, PRINTF_SRC);

    let mut s = DapServer::new();
    launch_printf(&mut s, &ir); // breakpoint on the second printf (line 4)
    let out = s.handle(&req(4, "configurationDone", Json::obj(vec![])));
    assert!(
        event(&out, "stopped"),
        "stopped at the second printf (first printf's call already returned)"
    );

    // Step back several times: every landing must stay in `main` at `/in.c` — never a libc frame
    // (`__pf_flush [/include/stdio.h]`, which is what the un-fixed op-granular step-back descended into).
    for i in 0..4 {
        let out = s.handle(&req(
            10 + i,
            "stepBack",
            Json::obj(vec![("threadId", Json::i(1))]),
        ));
        assert!(
            event(&out, "stopped"),
            "stepBack {i} stopped inside the guest"
        );
        let st = s.handle(&req(
            100 + i,
            "stackTrace",
            Json::obj(vec![("threadId", Json::i(1))]),
        ));
        let top = &response(&st)
            .get("body")
            .unwrap()
            .get("stackFrames")
            .unwrap()
            .as_array()
            .unwrap()[0];
        let src = top
            .get("source")
            .and_then(|s| s.get("path"))
            .and_then(|p| p.as_str())
            .unwrap_or("");
        let name = top.get("name").and_then(|n| n.as_str()).unwrap_or("");
        assert_eq!(
            src, "/in.c",
            "stepBack {i} stayed in the C source, not libc — frame {name:?} [{src}]"
        );
        assert!(
            name.contains("main"),
            "stepBack {i} stayed in main — got {name:?}"
        );
    }
}

// ---- Forward stepping coverage on the chibicc demo (next / stepIn / stepOut / oscillation) -----------
// The powerbox PR shipped with only `continue`/`reverseContinue` covered; these add the forward
// line-stepping the demo advertises (and that the Step-Back regression made clear was untested).

/// Launch `ir`'s `_start` (func 0) on the bytecode backend with a source breakpoint on `line`. `powerbox`
/// runs it under the on-ramp I/O powerbox (so a `printf` guest doesn't `CapFault`); off = deny-all.
fn launch_at(s: &mut DapServer, ir: &str, line: i64, powerbox: bool) {
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    let mut launch = vec![
        ("programText", Json::s(ir)),
        ("function", Json::i(0)),
        ("args", Json::Arr(vec![])),
        ("engine", Json::s("bytecode")),
    ];
    if powerbox {
        launch.push(("powerbox", Json::s("onramp")));
    }
    let out = s.handle(&req(2, "launch", Json::obj(launch)));
    assert_eq!(
        response(&out).get("success"),
        Some(&Json::Bool(true)),
        "launch ok"
    );
    s.handle(&req(
        3,
        "setBreakpoints",
        Json::obj(vec![
            ("source", Json::obj(vec![("path", Json::s("/in.c"))])),
            (
                "breakpoints",
                Json::Arr(vec![Json::obj(vec![("line", Json::i(line))])]),
            ),
        ]),
    ));
}

/// The paused top frame's `(line, function-name, source-path)` — `None` if the program finished.
fn top_frame(s: &mut DapServer, seq: i64) -> Option<(i64, String, String)> {
    let out = s.handle(&req(
        seq,
        "stackTrace",
        Json::obj(vec![("threadId", Json::i(1))]),
    ));
    let frames = response(&out).get("body")?.get("stackFrames")?.as_array()?;
    let f = frames.first()?;
    Some((
        f.get("line").and_then(|l| l.as_i64())?,
        f.get("name")
            .and_then(|n| n.as_str())
            .unwrap_or("")
            .to_string(),
        f.get("source")
            .and_then(|x| x.get("path"))
            .and_then(|p| p.as_str())
            .unwrap_or("")
            .to_string(),
    ))
}

const NEXT_SRC: &str = r#"#include <stdio.h>
int main(void) {
  printf("one\n");
  printf("two\n");
  printf("three\n");
  return 0;
}
"#;

/// **Forward `next` (Step Over) walks one C source line at a time, capturing each line's output.** Stepping
/// over a `printf` executes it (the output grows) and lands on the next source line in `main` — never
/// descending into the libc call. This is the forward counterpart the earlier tests lacked.
#[test]
fn forward_next_walks_source_lines_capturing_output() {
    let Some(bytes) = chibicc_svmb() else {
        eprintln!("SKIP: chibicc.svmb absent");
        return;
    };
    let chibicc = svm_encode::decode_module(&bytes).expect("decode");
    let ir = compile_g(&chibicc, NEXT_SRC);
    let mut s = DapServer::new();
    launch_at(&mut s, &ir, 3, true); // breakpoint on the first printf (line 3)
    assert!(
        event(
            &s.handle(&req(4, "configurationDone", Json::obj(vec![]))),
            "stopped"
        ),
        "stops at line 3"
    );
    assert_eq!(
        top_frame(&mut s, 5).map(|f| f.0),
        Some(3),
        "paused on the first printf, nothing printed yet"
    );

    // Each `next` steps over one printf: the line advances (staying in main) and that line's output appears.
    for (expect_line, expect_out) in [(4, "one\n"), (5, "one\ntwo\n"), (6, "one\ntwo\nthree\n")] {
        let out = s.handle(&req(
            10 + expect_line,
            "next",
            Json::obj(vec![("threadId", Json::i(1))]),
        ));
        assert!(
            event(&out, "stopped"),
            "next stopped in the guest (line {expect_line})"
        );
        let (line, name, src) =
            top_frame(&mut s, 100 + expect_line).expect("a live frame after next");
        assert_eq!(
            line, expect_line,
            "next advanced to C line {expect_line} (frame {name} [{src}])"
        );
        assert_eq!(
            name, "#0 main",
            "stepped over the printf — stayed in main, not the libc call"
        );
        assert_eq!(src, "/in.c", "frame source is the C file");
        assert_eq!(
            output_text(&out).as_deref(),
            Some(expect_out),
            "line {expect_line}'s output was captured"
        );
    }
}

const HELPER_SRC: &str = r#"int add(int a, int b) {
  return a + b;
}
int main(void) {
  int x = add(2, 3);
  int y = x + 1;
  return y;
}
"#;

/// **`stepIn` descends into a called function and `stepOut` returns to the caller** — the multi-frame case
/// no earlier test exercised. Stepping into `add` shows the callee frame with its parameters readable by
/// name (`a` = 2); stepping out lands back in `main` on the line after the call.
#[test]
fn step_in_and_out_across_a_helper() {
    let Some(bytes) = chibicc_svmb() else {
        eprintln!("SKIP: chibicc.svmb absent");
        return;
    };
    let chibicc = svm_encode::decode_module(&bytes).expect("decode");
    let ir = compile_g(&chibicc, HELPER_SRC);
    let mut s = DapServer::new();
    launch_at(&mut s, &ir, 5, false); // breakpoint on `int x = add(2, 3);` (line 5), no powerbox needed
    assert!(
        event(
            &s.handle(&req(4, "configurationDone", Json::obj(vec![]))),
            "stopped"
        ),
        "stops in main"
    );
    assert_eq!(
        top_frame(&mut s, 5).map(|f| f.1),
        Some("#0 main".to_string()),
        "paused in main"
    );

    // Step into `add` — the callee frame, at its body line, in the C source.
    let out = s.handle(&req(6, "stepIn", Json::obj(vec![("threadId", Json::i(1))])));
    assert!(event(&out, "stopped"), "stepIn stopped");
    let (line, name, src) = top_frame(&mut s, 7).expect("a frame inside the callee");
    assert_eq!(name, "#0 add", "stepped into add");
    assert_eq!(
        (line, src.as_str()),
        (2, "/in.c"),
        "at add's body line in the C source"
    );

    // The callee's parameters resolve by name at this frame: `add(2, 3)` ⇒ a = 2.
    let scopes = s.handle(&req(8, "scopes", Json::obj(vec![("frameId", Json::i(0))])));
    let vref = response(&scopes)
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
    let vars = s.handle(&req(
        9,
        "variables",
        Json::obj(vec![("variablesReference", Json::i(vref))]),
    ));
    let a = response(&vars)
        .get("body")
        .unwrap()
        .get("variables")
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .find(|v| v.get("name").and_then(|n| n.as_str()) == Some("a"))
        .and_then(|v| v.get("value"))
        .and_then(|x| x.as_str())
        .map(|s| s.to_string());
    assert_eq!(
        a.as_deref(),
        Some("2"),
        "the callee's parameter a = 2 reads by name"
    );

    // Step out — back in main, on the line after the call (`int y = x + 1;`, line 6).
    let out = s.handle(&req(
        10,
        "stepOut",
        Json::obj(vec![("threadId", Json::i(1))]),
    ));
    assert!(
        event(&out, "stopped"),
        "stepOut stopped back in the caller (didn't run to completion)"
    );
    let (line, name, _) = top_frame(&mut s, 11).expect("a frame back in main");
    assert_eq!(
        (name.as_str(), line),
        ("#0 main", 6),
        "stepOut returned to main at the line after the call"
    );
}

/// **Forward⇄back oscillation stays coherent**: stepping forward over the printfs grows the output and
/// advances the line; a run of `stepBack`s then stays in `main [/in.c]` with the line and captured output
/// both walking *backward* (never increasing, never descending into libc) — the property the Step-Back bug
/// violated. Then a `next` moves forward again.
#[test]
fn forward_then_back_oscillation_is_monotone_and_stays_in_main() {
    let Some(bytes) = chibicc_svmb() else {
        eprintln!("SKIP: chibicc.svmb absent");
        return;
    };
    let chibicc = svm_encode::decode_module(&bytes).expect("decode");
    let ir = compile_g(&chibicc, NEXT_SRC);
    let mut s = DapServer::new();
    launch_at(&mut s, &ir, 3, true);
    s.handle(&req(4, "configurationDone", Json::obj(vec![])));
    // Forward to the last printf line: output is fully "one\ntwo\nthree\n".
    for i in 0..3 {
        s.handle(&req(
            20 + i,
            "next",
            Json::obj(vec![("threadId", Json::i(1))]),
        ));
    }
    let (fwd_line, _, _) = top_frame(&mut s, 30).expect("a frame after forward stepping");
    let fwd_out_len = 14; // "one\ntwo\nthree\n"

    // Now step back several times: each landing stays in main [/in.c], and the line + output length are
    // both monotonically non-increasing (rewinding), never jumping into a libc frame.
    let mut prev_line = fwd_line;
    let mut prev_out = fwd_out_len;
    let mut saw_rewind = false;
    for i in 0..8 {
        let out = s.handle(&req(
            40 + i,
            "stepBack",
            Json::obj(vec![("threadId", Json::i(1))]),
        ));
        assert!(event(&out, "stopped"), "stepBack {i} stopped in the guest");
        let (line, name, src) = top_frame(&mut s, 140 + i).expect("a live frame after stepBack");
        assert_eq!(
            src, "/in.c",
            "stepBack {i} stayed in the C source (frame {name})"
        );
        assert!(
            name.contains("main"),
            "stepBack {i} stayed in main (got {name})"
        );
        assert!(
            line <= prev_line,
            "stepBack {i}: line went backward or held ({line} <= {prev_line})"
        );
        let out_len = output_text(&out).map(|t| t.len()).unwrap_or(prev_out);
        assert!(
            out_len <= prev_out,
            "stepBack {i}: output rewound or held ({out_len} <= {prev_out})"
        );
        if line < fwd_line || out_len < fwd_out_len {
            saw_rewind = true;
        }
        prev_line = line;
        prev_out = out_len;
    }
    assert!(
        saw_rewind,
        "stepping back visibly rewound the line and/or the captured output"
    );

    // And forward again advances: a `next` moves the line forward from where the rewind left off.
    let out = s.handle(&req(60, "next", Json::obj(vec![("threadId", Json::i(1))])));
    assert!(event(&out, "stopped"), "next after rewinding stopped");
    let (line, _, _) = top_frame(&mut s, 61).expect("a frame after resuming forward");
    assert!(
        line >= prev_line,
        "forward next advanced the line again ({line} >= {prev_line})"
    );
}
