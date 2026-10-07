//! Source-level debugging of a chibicc-compiled C program (the playground's Debug button, end to end
//! in Rust): compile a compute-only C source **with `-g`** through `chibicc.temen`, then drive the
//! `temen-dap` server (the bytecode backend the playground runs) over the emitted IR — set a breakpoint on
//! a **C source line**, run to it, and read the paused frame's **C locals by name**. This proves the
//! debug-info path the browser Debug button wires: chibicc's `-g` `debug.file`/`debug.loc`/`debug.var`
//! waist lets the DAP bind breakpoints to C lines and name C variables, on the compiled program.
//!
//! Two modes: a compute-only program debugs under the deny-all backend; a **capability-using** program
//! (a `printf` → a `write` cap) debugs under the on-ramp **I/O powerbox** (`launch` arg
//! `powerbox: "onramp"`), which runs it instead of `CapFault`ing, captures its output as DAP `output`
//! events, and — via the CapTape replay — **rewinds that output on reverse debugging**.
//!
//! Fail-soft on a missing `chibicc.temen` (a fresh tree without the build), like `chibicc_printf.rs`.

use temen_browser::{onramp_fs_exec, playground_include_files, STATUS_EXIT, STATUS_OK};
use temen_dap::{DapServer, Json};

#[path = "support/pg_heap.rs"]
mod pg_heap;

fn chibicc_temen() -> Option<Vec<u8>> {
    std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/web/assets/chibicc.temen"
    ))
    .ok()
}

/// Compile `src` with `-g` (debug info on) via the shipped compiler into a program unit and link it
/// against the heap unit, returning the linked program's TEMEN-IR text — what the card's debugger
/// launches, with the unit's debug info carried across the link.
fn compile_g(chibicc: &temen_ir::Module, src: &str) -> String {
    let mut files = playground_include_files();
    files.push(("in.c".to_string(), src.as_bytes().to_vec()));
    let image = temen_fs::encode_image(&files, &["include".to_string()]);
    let out = onramp_fs_exec(
        chibicc,
        &image,
        &[
            b"chibicc",
            b"--data-page",
            b"65536",
            b"--emit-object",
            b"-g",
            b"/in.c",
        ],
        b"",
    );
    assert!(
        out.status == STATUS_OK || out.status == STATUS_EXIT,
        "compile status {}",
        out.status
    );
    let unit = String::from_utf8(out.stdout).expect("IR utf8");
    let prog = temen_text::parse_module_debug(&unit).expect("parse the program unit");
    temen_text::print_module(&pg_heap::link(&[], &prog))
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
fn event(msgs: &[Json], name: &str) -> bool {
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
    let Some(bytes) = chibicc_temen() else {
        eprintln!("SKIP: browser/web/assets/chibicc.temen absent (run build-onramp-assets.mjs)");
        return;
    };
    let chibicc = temen_encode::decode_module(&bytes).expect("decode chibicc.temen");
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
    let Some(bytes) = chibicc_temen() else {
        eprintln!("SKIP: chibicc.temen absent");
        return;
    };
    let chibicc = temen_encode::decode_module(&bytes).expect("decode");
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
    let Some(bytes) = chibicc_temen() else {
        eprintln!("SKIP: chibicc.temen absent");
        return;
    };
    let chibicc = temen_encode::decode_module(&bytes).expect("decode");
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

/// **A debugged program's stderr is its own stream.** The seeded libc writes fd 2 through the
/// `"stderr"` capability, which the session grants when the program imports it, and the server
/// reports it as `output` events of category `stderr` — separate from stdout, so a client can show
/// diagnostics apart from (and not graded as) program output.
#[test]
fn stderr_is_reported_apart_from_stdout() {
    let Some(bytes) = chibicc_temen() else {
        eprintln!("SKIP: chibicc.temen absent");
        return;
    };
    let chibicc = temen_encode::decode_module(&bytes).expect("decode");
    let ir = compile_g(
        &chibicc,
        "#include <stdio.h>\nint main(void) {\n  printf(\"out\\n\");\n  fprintf(stderr, \"err\\n\");\n  return 0;\n}\n",
    );
    let mut s = DapServer::new();
    launch_at(&mut s, &ir, 2, true);
    let mut msgs = s.handle(&req(4, "configurationDone", Json::obj(vec![])));
    msgs.extend(s.handle(&req(
        5,
        "continue",
        Json::obj(vec![("threadId", Json::i(1))]),
    )));
    let last = |category: &str| {
        msgs.iter()
            .rev()
            .filter(|m| m.get("event").and_then(|e| e.as_str()) == Some("output"))
            .filter_map(|m| m.get("body"))
            .find(|b| b.get("category").and_then(|c| c.as_str()) == Some(category))
            .and_then(|b| b.get("output"))
            .and_then(|o| o.as_str())
            .map(str::to_string)
    };
    assert_eq!(last("stdout").as_deref(), Some("out\n"), "stdout: {msgs:?}");
    assert_eq!(last("stderr").as_deref(), Some("err\n"), "stderr: {msgs:?}");
}

const LOOPS_SRC: &str = r#"int main(void) {
  int a = 0;
  for (int i = 0; i < 2; i++)
    a += i;
  while (a < 3)
    a++;
  do
    a--;
  while (a > 1);
  return a;
}
"#;

/// **`next` visits a loop's test on every iteration**, as gdb does: the `for` line (increment +
/// condition) comes back between body runs, and so do a `while`'s and a `do`/`while`'s condition
/// lines. chibicc gave statements a line but not a loop's condition/increment *expressions*, so
/// those ops were unmapped and a line step from the body ran through the rest of the loop.
#[test]
fn next_stops_on_each_loop_test() {
    let Some(bytes) = chibicc_temen() else {
        eprintln!("SKIP: chibicc.temen absent");
        return;
    };
    let chibicc = temen_encode::decode_module(&bytes).expect("decode");
    let ir = compile_g(&chibicc, LOOPS_SRC);

    let mut s = DapServer::new();
    launch_at(&mut s, &ir, 2, false);
    s.handle(&req(4, "configurationDone", Json::obj(vec![])));
    let mut lines = vec![top_frame(&mut s, 5).expect("stopped at line 2").0];
    for i in 0..40 {
        s.handle(&req(
            10 + i,
            "next",
            Json::obj(vec![("threadId", Json::i(1))]),
        ));
        match top_frame(&mut s, 100 + i) {
            Some((line, name, _)) if name.contains("main") && line != 0 => lines.push(line),
            _ => break,
        }
        if lines.last() == Some(&10) {
            break;
        }
    }
    // Line 7, the `do` itself, is a stop too: chibicc emits the branch into the body there.
    assert_eq!(
        lines,
        [2, 3, 4, 3, 4, 3, 5, 6, 5, 6, 5, 7, 8, 9, 8, 9, 10],
        "the line sequence `next` walks"
    );
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
    let Some(bytes) = chibicc_temen() else {
        eprintln!("SKIP: chibicc.temen absent");
        return;
    };
    let chibicc = temen_encode::decode_module(&bytes).expect("decode");
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
    let Some(bytes) = chibicc_temen() else {
        eprintln!("SKIP: chibicc.temen absent");
        return;
    };
    let chibicc = temen_encode::decode_module(&bytes).expect("decode");
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
    let Some(bytes) = chibicc_temen() else {
        eprintln!("SKIP: chibicc.temen absent");
        return;
    };
    let chibicc = temen_encode::decode_module(&bytes).expect("decode");
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

// ---- Rich C inspection: struct/array/pointer locals + evaluate over chibicc-compiled C ------------
// The powerbox/stepping tests above use only scalar `int` locals. These lock in the Variables-pane
// aggregate expansion and the `evaluate` member/index/arrow paths — the debug info chibicc's `-g`
// emits (`debug.type agg`/`array`/`ptr`, `debug.field`, `win`/`ssalist` var locations) end to end.

const RICH_SRC: &str = r#"struct Point { int x; int y; };
int main(void) {
  int arr[3];
  arr[0] = 10; arr[1] = 20; arr[2] = 30;
  struct Point p;
  p.x = 7; p.y = 9;
  struct Point *pp = &p;
  int sum = arr[0] + p.x;
  return sum;
}
"#;
// Line 8 (`int sum = arr[0] + p.x;`) is the stop: arr, p, and pp are all live and assigned there.
const RICH_BP: i64 = 8;

/// The `variablesReference` of the top frame's locals scope, at the current stop.
fn locals_ref(s: &mut DapServer, seq: i64) -> i64 {
    let st = s.handle(&req(
        seq,
        "stackTrace",
        Json::obj(vec![("threadId", Json::i(1))]),
    ));
    let fid = response(&st)
        .get("body")
        .unwrap()
        .get("stackFrames")
        .unwrap()
        .as_array()
        .unwrap()[0]
        .get("id")
        .unwrap()
        .as_i64()
        .unwrap();
    let sc = s.handle(&req(
        seq + 1,
        "scopes",
        Json::obj(vec![("frameId", Json::i(fid))]),
    ));
    response(&sc)
        .get("body")
        .unwrap()
        .get("scopes")
        .unwrap()
        .as_array()
        .unwrap()[0]
        .get("variablesReference")
        .unwrap()
        .as_i64()
        .unwrap()
}

/// The `variables` list for a reference, as `(name, value, variablesReference)` triples.
fn vars_of(s: &mut DapServer, seq: i64, vref: i64) -> Vec<(String, String, i64)> {
    let out = s.handle(&req(
        seq,
        "variables",
        Json::obj(vec![("variablesReference", Json::i(vref))]),
    ));
    response(&out)
        .get("body")
        .unwrap()
        .get("variables")
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .map(|v| {
            (
                v.get("name")
                    .and_then(|n| n.as_str())
                    .unwrap_or("")
                    .to_string(),
                v.get("value")
                    .and_then(|n| n.as_str())
                    .unwrap_or("")
                    .to_string(),
                v.get("variablesReference")
                    .and_then(|r| r.as_i64())
                    .unwrap_or(0),
            )
        })
        .collect()
}

/// `evaluate` an expression in the top frame; `Some(result)` on success, `None` if the DAP rejected it.
fn eval_in_frame(s: &mut DapServer, seq: i64, expr: &str) -> Option<String> {
    let st = s.handle(&req(
        seq,
        "stackTrace",
        Json::obj(vec![("threadId", Json::i(1))]),
    ));
    let fid = response(&st)
        .get("body")?
        .get("stackFrames")?
        .as_array()?
        .first()?
        .get("id")?
        .as_i64()?;
    let out = s.handle(&req(
        seq + 1,
        "evaluate",
        Json::obj(vec![
            ("expression", Json::s(expr)),
            ("frameId", Json::i(fid)),
            ("context", Json::s("hover")),
        ]),
    ));
    let r = response(&out);
    (r.get("success") == Some(&Json::Bool(true)))
        .then(|| {
            r.get("body")
                .and_then(|b| b.get("result"))
                .and_then(|x| x.as_str())
                .map(|s| s.to_string())
        })
        .flatten()
}

/// **Struct, array, and pointer C locals inspect end to end.** At a stop where `arr`/`p`/`pp` are live:
/// the Variables pane expands the struct (`p.x`,`p.y`) and array (`arr[0..2]`); `evaluate` reads array
/// elements, struct members, arrow-through-pointer (`pp->x`), and arithmetic over them — the member/
/// index/arrow paths, including the promoted-SSA pointer chibicc emits for `struct Point *pp = &p;`.
#[test]
fn inspect_struct_array_and_pointer_locals() {
    let Some(bytes) = chibicc_temen() else {
        eprintln!("SKIP: chibicc.temen absent");
        return;
    };
    let chibicc = temen_encode::decode_module(&bytes).expect("decode");
    let ir = compile_g(&chibicc, RICH_SRC);
    // The -g IR carries the aggregate/array/pointer type info and fields.
    assert!(
        ir.contains(r#"debug.type 2 agg "struct Point""#) && ir.contains(r#"debug.field 2 "x""#),
        "-g IR describes the struct type and its fields"
    );

    let mut s = DapServer::new();
    launch_at(&mut s, &ir, RICH_BP, false); // compute-only: no powerbox needed
    assert!(
        event(
            &s.handle(&req(4, "configurationDone", Json::obj(vec![]))),
            "stopped"
        ),
        "stopped at line 8 with arr/p/pp live"
    );

    // Variables pane: the aggregates are expandable, the pointer shows a scalar value.
    let vref = locals_ref(&mut s, 5);
    let top = vars_of(&mut s, 7, vref);
    let find = |n: &str| top.iter().find(|(name, _, _)| name == n).cloned();
    let (_, _, p_ref) = find("p").expect("local p present");
    let (_, _, arr_ref) = find("arr").expect("local arr present");
    assert!(p_ref > 0, "the struct local p is expandable");
    assert!(arr_ref > 0, "the array local arr is expandable");
    assert!(find("pp").is_some(), "the pointer local pp is present");

    // Expand the struct: fields x=7, y=9.
    let pfields = vars_of(&mut s, 10, p_ref);
    assert_eq!(
        pfields
            .iter()
            .find(|(n, ..)| n == "x")
            .map(|(_, v, _)| v.as_str()),
        Some("7"),
        "p.x = 7 in the Variables pane"
    );
    assert_eq!(
        pfields
            .iter()
            .find(|(n, ..)| n == "y")
            .map(|(_, v, _)| v.as_str()),
        Some("9"),
        "p.y = 9 in the Variables pane"
    );

    // Expand the array: [0]=10, [1]=20, [2]=30.
    let elems = vars_of(&mut s, 12, arr_ref);
    let vals: Vec<&str> = elems.iter().map(|(_, v, _)| v.as_str()).collect();
    assert_eq!(vals, vec!["10", "20", "30"], "arr elements expand in order");

    // evaluate: array index, struct member, arrow-through-pointer, and arithmetic over them.
    assert_eq!(eval_in_frame(&mut s, 20, "arr[0]").as_deref(), Some("10"));
    assert_eq!(eval_in_frame(&mut s, 22, "arr[2]").as_deref(), Some("30"));
    assert_eq!(eval_in_frame(&mut s, 24, "p.x").as_deref(), Some("7"));
    assert_eq!(eval_in_frame(&mut s, 26, "p.y").as_deref(), Some("9"));
    assert_eq!(
        eval_in_frame(&mut s, 28, "pp->x").as_deref(),
        Some("7"),
        "arrow through the promoted-SSA pointer resolves"
    );
    assert_eq!(eval_in_frame(&mut s, 30, "pp->y").as_deref(), Some("9"));
    assert_eq!(
        eval_in_frame(&mut s, 32, "arr[0] + p.x").as_deref(),
        Some("17"),
        "arithmetic over member/index results"
    );
}

/// **A breakpoint on a bare `return x;` line binds and stops there** (#1713). chibicc emits nothing
/// for `return sum;` but the block's `return` terminator (`sum` is already in a register). Terminators
/// used to be skipped by both engines, so the DAP refused the line as unverified. A terminator is a
/// stop position now: the breakpoint binds, the run stops on line 9 with `sum` readable, and a `next`
/// from line 8 lands on line 9 rather than running off the end.
#[test]
fn breakpoint_on_bare_return_line_stops_there() {
    let Some(bytes) = chibicc_temen() else {
        eprintln!("SKIP: chibicc.temen absent");
        return;
    };
    let chibicc = temen_encode::decode_module(&bytes).expect("decode");
    let ir = compile_g(&chibicc, RICH_SRC);
    let launch = |s: &mut DapServer| {
        s.handle(&req(1, "initialize", Json::obj(vec![])));
        s.handle(&req(
            2,
            "launch",
            Json::obj(vec![
                ("programText", Json::s(&ir)),
                ("function", Json::i(0)),
                ("args", Json::Arr(vec![])),
                ("engine", Json::s("bytecode")),
            ]),
        ));
    };
    let set_bp = |s: &mut DapServer, line: i64| -> Json {
        let out = s.handle(&req(
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
        response(&out)
            .get("body")
            .and_then(|b| b.get("breakpoints"))
            .and_then(|b| b.as_array())
            .expect("breakpoints")[0]
            .clone()
    };

    // Line 9 is `return sum;` — only the return terminator.
    let mut s = DapServer::new();
    launch(&mut s);
    let bp = set_bp(&mut s, 9);
    assert_eq!(
        bp.get("verified"),
        Some(&Json::Bool(true)),
        "the bare-return line binds"
    );
    assert_eq!(bp.get("line").and_then(|l| l.as_i64()), Some(9));
    let out = s.handle(&req(4, "configurationDone", Json::obj(vec![])));
    assert_eq!(stopped_reason(&out).as_deref(), Some("breakpoint"));
    assert_eq!(
        top_frame(&mut s, 5).map(|f| f.0),
        Some(9),
        "stopped on the return line"
    );
    assert_eq!(eval_in_frame(&mut s, 6, "sum").as_deref(), Some("17"));

    // Stepping: from the line-8 breakpoint, one `next` lands on line 9.
    let mut s = DapServer::new();
    launch(&mut s);
    set_bp(&mut s, 8);
    s.handle(&req(4, "configurationDone", Json::obj(vec![])));
    assert_eq!(top_frame(&mut s, 5).map(|f| f.0), Some(8));
    let out = s.handle(&req(6, "next", Json::obj(vec![("threadId", Json::i(1))])));
    assert!(!event(&out, "terminated"), "the step ran off the end");
    assert_eq!(
        top_frame(&mut s, 7).map(|f| f.0),
        Some(9),
        "`next` lands on the return line"
    );
}

// ---- W4 blocking stdin: park on exhausted stdin, provideStdin resumes, replay is faithful ---------
// The INTERACTIVE_EMBEDDING.md W4 acceptance: a prompt-loop C guest round-trips two provided inputs;
// rewinding and re-running replays both byte-identically from the CapTape with no new suspensions.

const ECHO_SRC: &str = r#"int read(int fd, char *buf, long n);
int write(int fd, char *buf, long n);
int main(void) {
  char buf[16];
  int n = read(0, buf, 16);
  write(1, buf, n);
  n = read(0, buf, 16);
  write(1, buf, n);
  return 0;
}
"#;

/// The reason of the batch's `stopped` event, if any.
fn stopped_reason(msgs: &[Json]) -> Option<String> {
    msgs.iter()
        .find(|m| m.get("event").and_then(|e| e.as_str()) == Some("stopped"))
        .and_then(|m| m.get("body"))
        .and_then(|b| b.get("reason"))
        .and_then(|r| r.as_str())
        .map(|s| s.to_string())
}

fn launch_echo(s: &mut DapServer, ir: &str, block_stdin: bool) {
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    let mut launch = vec![
        ("programText", Json::s(ir)),
        ("function", Json::i(0)),
        ("args", Json::Arr(vec![])),
        ("engine", Json::s("bytecode")),
        ("powerbox", Json::s("onramp")),
    ];
    if block_stdin {
        launch.push(("blockStdin", Json::Bool(true)));
    }
    let out = s.handle(&req(2, "launch", Json::obj(launch)));
    assert_eq!(
        response(&out).get("success"),
        Some(&Json::Bool(true)),
        "echo launch ok"
    );
}

/// **A blocking-stdin session parks at each exhausted `read`, resumes on `provideStdin`, and its
/// reverse replay reproduces the provided inputs with no new suspensions** — the W4 acceptance.
#[test]
fn blocking_stdin_round_trips_and_replays() {
    let Some(bytes) = chibicc_temen() else {
        eprintln!("SKIP: chibicc.temen absent");
        return;
    };
    let chibicc = temen_encode::decode_module(&bytes).expect("decode");
    let ir = compile_g(&chibicc, ECHO_SRC);

    let mut s = DapServer::new();
    launch_echo(&mut s, &ir, true);

    // No breakpoints: the run parks at the first read (reason "stdin") instead of EOF-completing.
    let out = s.handle(&req(3, "configurationDone", Json::obj(vec![])));
    assert_eq!(
        stopped_reason(&out).as_deref(),
        Some("stdin"),
        "parked awaiting input at the first read (not terminated: {out:?})"
    );

    // Provide the first input and resume: it echoes, then parks at the second read.
    let out = s.handle(&req(
        4,
        "provideStdin",
        Json::obj(vec![("data", Json::s("A\n"))]),
    ));
    assert_eq!(
        response(&out).get("success"),
        Some(&Json::Bool(true)),
        "provideStdin accepted"
    );
    let out = s.handle(&req(
        5,
        "continue",
        Json::obj(vec![("threadId", Json::i(1))]),
    ));
    assert_eq!(
        stopped_reason(&out).as_deref(),
        Some("stdin"),
        "parked at the second read"
    );
    assert_eq!(
        output_text(&out).as_deref(),
        Some("A\n"),
        "the first provided input was echoed"
    );

    // Provide the second input and resume to completion.
    s.handle(&req(
        6,
        "provideStdin",
        Json::obj(vec![("data", Json::s("B!"))]),
    ));
    let out = s.handle(&req(
        7,
        "continue",
        Json::obj(vec![("threadId", Json::i(1))]),
    ));
    assert!(event(&out, "terminated"), "ran to completion");
    assert_eq!(
        output_text(&out).as_deref(),
        Some("A\nB!"),
        "both provided inputs were echoed"
    );

    // Rewind to the start: the captured output rewinds with the program.
    let out = s.handle(&req(
        8,
        "reverseContinue",
        Json::obj(vec![("threadId", Json::i(1))]),
    ));
    assert!(event(&out, "stopped"), "rewound to the start");
    assert_eq!(
        output_text(&out).as_deref().unwrap_or(""),
        "",
        "output rewound to empty at the start"
    );

    // Forward again: the CapTape replays both provided reads byte-identically — the run completes
    // with the same output and **no new stdin suspension** (the W4 replay acceptance).
    let out = s.handle(&req(
        9,
        "continue",
        Json::obj(vec![("threadId", Json::i(1))]),
    ));
    assert!(
        stopped_reason(&out).is_none(),
        "replay served the provided inputs from the tape — no re-park"
    );
    assert!(event(&out, "terminated"), "replay ran to completion");
    assert_eq!(
        output_text(&out).as_deref(),
        Some("A\nB!"),
        "replay reproduced the provided inputs byte-identically"
    );
}

/// **Inertness pin** (invariant 9b): without `blockStdin`, the same guest keeps plain EOF semantics —
/// exhausted reads return 0 and the run completes with no stop. The mode is armed, never ambient.
#[test]
fn without_block_stdin_exhausted_reads_stay_eof() {
    let Some(bytes) = chibicc_temen() else {
        eprintln!("SKIP: chibicc.temen absent");
        return;
    };
    let chibicc = temen_encode::decode_module(&bytes).expect("decode");
    let ir = compile_g(&chibicc, ECHO_SRC);

    let mut s = DapServer::new();
    launch_echo(&mut s, &ir, false);
    let out = s.handle(&req(3, "configurationDone", Json::obj(vec![])));
    assert!(
        stopped_reason(&out).is_none(),
        "no stdin stop without the mode"
    );
    assert!(
        event(&out, "terminated"),
        "EOF reads (0 bytes) let the guest run to completion unchanged"
    );
}

// Two threads that each write their own global, for the split-view stepping test below. Line 13 is
// `main`'s first statement after the spawn; `count_a`'s body is lines 5–7.
const TWO_THREADS_SRC: &str = r#"#include <pthread.h>
int a = 0;
int b = 0;
void* count_a(void* arg) {
  a = 1;
  a = 2;
  a = 3;
  return 0;
}
int main(void) {
  pthread_t t;
  pthread_create(&t, 0, count_a, 0);
  b = 10;
  b = 20;
  b = 30;
  pthread_join(t, 0);
  return a + b;
}
"#;

/// The `(line, bare function name)` of DAP thread `tid`'s top frame.
fn thread_top(s: &mut DapServer, seq: i64, tid: i64) -> Option<(i64, String)> {
    let out = s.handle(&req(
        seq,
        "stackTrace",
        Json::obj(vec![("threadId", Json::i(tid))]),
    ));
    let f = response(&out)
        .get("body")?
        .get("stackFrames")?
        .as_array()?
        .first()?
        .clone();
    let name = f.get("name")?.as_str()?;
    Some((
        f.get("line")?.as_i64()?,
        name.split_once(' ').map_or(name, |(_, n)| n).to_string(),
    ))
}

/// **Split-view stepping on a chosen thread, and a Step Back that undoes the last step.** A step
/// names its thread (`threadId`) and, with `singleThread`, moves only that thread; a `stepBack` then undoes the most recent
/// step, whichever thread took it, and leaves the other thread where it was. Stepping again after a
/// step back rewrites the future, and the rewritten future is what later replays see.
///
/// Before the fix a step ignored `threadId` (it always drove the thread that last stopped), and a
/// step back replayed the default schedule instead of the one the steps took, landing somewhere else
/// entirely.
#[test]
fn a_step_moves_the_thread_it_names_and_step_back_undoes_the_last_step() {
    let Some(bytes) = chibicc_temen() else {
        eprintln!("SKIP: chibicc.temen absent");
        return;
    };
    let chibicc = temen_encode::decode_module(&bytes).expect("decode");
    let ir = compile_g(&chibicc, TWO_THREADS_SRC);

    let mut s = DapServer::new();
    launch_at(&mut s, &ir, 13, true);
    let out = s.handle(&req(4, "configurationDone", Json::obj(vec![])));
    assert_eq!(stopped_reason(&out).as_deref(), Some("breakpoint"));
    let mut seq = 10;
    // Split view steps one thread at a time: `singleThread` keeps the other frozen.
    let mut go = |s: &mut DapServer, cmd: &str, tid: i64| {
        seq += 1;
        let args = vec![
            ("threadId", Json::i(tid)),
            ("singleThread", Json::Bool(true)),
        ];
        s.handle(&req(seq, cmd, Json::obj(args)))
    };
    let at = |s: &mut DapServer, tid: i64| thread_top(s, 900 + tid, tid);
    let val = |s: &mut DapServer, name: &str| eval_in_frame(s, 950, name);

    // Thread 2 starts in the spawn trampoline; step it into `count_a`. Thread 1 stays put.
    for _ in 0..20 {
        if at(&mut s, 2).is_some_and(|(_, f)| f == "count_a") {
            break;
        }
        go(&mut s, "stepIn", 2);
    }
    assert_eq!(
        at(&mut s, 2),
        Some((5, "count_a".into())),
        "thread 2 reached count_a"
    );
    assert_eq!(
        at(&mut s, 1),
        Some((13, "main".into())),
        "thread 1 did not move"
    );
    assert_eq!(val(&mut s, "b").as_deref(), Some("0"));

    go(&mut s, "next", 2);
    go(&mut s, "next", 2);
    assert_eq!(at(&mut s, 2), Some((7, "count_a".into())));
    assert_eq!(val(&mut s, "a").as_deref(), Some("2"));
    go(&mut s, "next", 1);
    assert_eq!(at(&mut s, 1), Some((14, "main".into())));
    assert_eq!(val(&mut s, "b").as_deref(), Some("10"));
    assert_eq!(
        at(&mut s, 2),
        Some((7, "count_a".into())),
        "thread 2 did not move"
    );

    // Step Back undoes thread 1's step, then thread 2's, each leaving the other thread alone.
    go(&mut s, "stepBack", 1);
    assert_eq!(
        at(&mut s, 1),
        Some((13, "main".into())),
        "thread 1's step undone"
    );
    assert_eq!(val(&mut s, "b").as_deref(), Some("0"));
    assert_eq!(
        at(&mut s, 2),
        Some((7, "count_a".into())),
        "thread 2 untouched"
    );
    assert_eq!(val(&mut s, "a").as_deref(), Some("2"));
    go(&mut s, "stepBack", 1);
    assert_eq!(
        at(&mut s, 2),
        Some((6, "count_a".into())),
        "thread 2's step undone"
    );
    assert_eq!(val(&mut s, "a").as_deref(), Some("1"));
    assert_eq!(
        at(&mut s, 1),
        Some((13, "main".into())),
        "thread 1 untouched"
    );

    // A new step from here rewrites the future; stepping back over it replays the new one.
    go(&mut s, "next", 1);
    assert_eq!(at(&mut s, 1), Some((14, "main".into())));
    assert_eq!(val(&mut s, "b").as_deref(), Some("10"));
    assert_eq!(at(&mut s, 2), Some((6, "count_a".into())));
    go(&mut s, "stepBack", 1);
    assert_eq!(at(&mut s, 1), Some((13, "main".into())));
    assert_eq!(at(&mut s, 2), Some((6, "count_a".into())));
    assert_eq!(val(&mut s, "a").as_deref(), Some("1"));

    // And the program still runs to its answer.
    let out = go(&mut s, "continue", 1);
    let code = out
        .iter()
        .find(|m| m.get("event").and_then(|e| e.as_str()) == Some("exited"))
        .and_then(|m| m.get("body")?.get("exitCode")?.as_i64());
    assert_eq!(code, Some(33), "a + b");
}

/// `main` spins on a flag only the worker sets.
const SPIN_WAIT_SRC: &str = r#"#include <pthread.h>
int ready = 0;
void* worker(void* arg) {
  ready = 1;
  return 0;
}
int main(void) {
  pthread_t t;
  pthread_create(&t, 0, worker, 0);
  while (ready == 0) {}
  int after = 1;
  pthread_join(t, 0);
  return after;
}
"#;

/// **A step over a spin-wait lands, and so does a run.** A step kept to its thread and the default
/// schedule ran the lowest-index thread, so while `main` spun on `ready` the worker that sets it never
/// ran: the step never ended, and neither did `continue`. A step without `singleThread` now shares the
/// turns with the other runnable threads, as the DAP spec has it, and a run rotates between runnable
/// threads each quantum, as the release engine's preemption does.
#[test]
fn a_step_over_a_spin_wait_lands_and_so_does_a_run() {
    let Some(bytes) = chibicc_temen() else {
        eprintln!("SKIP: chibicc.temen absent");
        return;
    };
    let chibicc = temen_encode::decode_module(&bytes).expect("decode");
    let ir = compile_g(&chibicc, SPIN_WAIT_SRC);

    // Step over each line from the spawn on, as a learner does: the steps reach the line after the
    // loop.
    let mut s = DapServer::new();
    launch_at(&mut s, &ir, 10, true);
    let out = s.handle(&req(4, "configurationDone", Json::obj(vec![])));
    assert_eq!(stopped_reason(&out).as_deref(), Some("breakpoint"));
    let mut lines = vec![];
    for seq in 10..30 {
        let top = thread_top(&mut s, 100 + seq, 1);
        lines.push(top.as_ref().map_or(-1, |t| t.0));
        if top == Some((12, "main".into())) {
            break;
        }
        s.handle(&req(seq, "next", Json::obj(vec![("threadId", Json::i(1))])));
    }
    assert_eq!(
        lines.last(),
        Some(&12),
        "the steps got past the spin loop: {lines:?}"
    );

    // A plain run of the same program ends.
    let mut s = DapServer::new();
    launch_at(&mut s, &ir, 1, true);
    s.handle(&req(4, "configurationDone", Json::obj(vec![])));
    let out = s.handle(&req(5, "continue", Json::obj(vec![])));
    assert!(
        out.iter()
            .any(|m| m.get("event").and_then(|e| e.as_str()) == Some("terminated")),
        "the run ended"
    );
}

/// **Code a macro expanded to sits on the macro's invocation line.** `atomic_fetch_add` is a macro in
/// <stdatomic.h>, and its expansion used to carry that header's line, so a step onto it stopped "in"
/// the header and a thread running it showed no line of the program. It is `main`'s line 5 now, as a C
/// debugger shows it and as `__LINE__` reads it.
#[test]
fn a_macro_expansion_steps_as_the_line_that_invoked_it() {
    let Some(bytes) = chibicc_temen() else {
        eprintln!("SKIP: chibicc.temen absent");
        return;
    };
    let chibicc = temen_encode::decode_module(&bytes).expect("decode");
    let src = "#include <stdatomic.h>\natomic_int n;\nint main(void) {\n  int a = 1;\n  atomic_fetch_add(&n, 1);\n  a = 2;\n  return a;\n}\n";
    let ir = compile_g(&chibicc, src);
    let mut s = DapServer::new();
    launch_at(&mut s, &ir, 4, true);
    let out = s.handle(&req(4, "configurationDone", Json::obj(vec![])));
    assert_eq!(stopped_reason(&out).as_deref(), Some("breakpoint"));
    s.handle(&req(5, "next", Json::obj(vec![])));
    assert_eq!(
        top_frame(&mut s, 6).map(|(l, _, p)| (l, p)),
        Some((5, "/in.c".into())),
        "the step lands on the invocation, in the program"
    );
}

/// **The scheduler trace records calls and returns (#1981).** A worker thread that lives and dies
/// inside one `continue` shows on the tape as a `call` into `worker` on its own task, followed by the
/// matching `return` — what a flame chart builds that thread's span from, where stack samples taken
/// at stops never see it. Each task's calls and returns pair up, never returning past its entry.
#[test]
fn the_scheduler_trace_records_calls_and_returns() {
    let Some(bytes) = chibicc_temen() else {
        eprintln!("SKIP: chibicc.temen absent");
        return;
    };
    let chibicc = temen_encode::decode_module(&bytes).expect("decode");
    let src = "#include <pthread.h>\nint x;\nvoid *worker(void *a) { x = 1; return 0; }\nint main(void) {\n  pthread_t t;\n  pthread_create(&t, 0, worker, 0);\n  pthread_join(t, 0);\n  return x;\n}\n";
    let ir = compile_g(&chibicc, src);
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    let launch = vec![
        ("programText", Json::s(&ir)),
        ("function", Json::i(0)),
        ("args", Json::Arr(vec![])),
        ("engine", Json::s("bytecode")),
        ("powerbox", Json::s("onramp")),
        ("schedTrace", Json::Bool(true)),
    ];
    let out = s.handle(&req(2, "launch", Json::obj(launch)));
    assert_eq!(response(&out).get("success"), Some(&Json::Bool(true)));
    s.handle(&req(3, "configurationDone", Json::obj(vec![])));
    s.handle(&req(4, "continue", Json::obj(vec![])));
    let out = s.handle(&req(5, "schedTrace", Json::obj(vec![])));
    let tape = response(&out)
        .get("body")
        .and_then(|b| b.as_array())
        .map(<[Json]>::to_vec)
        .expect("a trace tape");
    let field = |e: &Json, k: &str| e.get(k).cloned();
    let kind = |e: &Json| {
        e.get("kind")
            .and_then(|k| k.as_str())
            .unwrap_or("")
            .to_string()
    };
    let task = |e: &Json| e.get("task").and_then(|t| t.as_i64()).unwrap_or(-1);
    let call_worker = tape
        .iter()
        .position(|e| {
            kind(e) == "call" && task(e) == 1 && field(e, "func") == Some(Json::s("worker"))
        })
        .expect("the worker's call into `worker` is on the tape");
    assert!(
        tape[call_worker..]
            .iter()
            .any(|e| kind(e) == "return" && task(e) == 1),
        "and so is its return"
    );
    for t in 0..2 {
        let mut depth = 0i64;
        for e in tape.iter().filter(|e| task(e) == t) {
            match kind(e).as_str() {
                "call" => depth += 1,
                "return" => depth -= 1,
                _ => {}
            }
            assert!(depth >= 0, "task {t} returned past its entry");
        }
    }
}

/// Two threads that each hold one mutex and wait for the other's.
const DEADLOCK_SRC: &str = r#"#include <pthread.h>
pthread_mutex_t m1, m2;
volatile int ready1 = 0, ready2 = 0;
void *worker(void *arg) {
  pthread_mutex_lock(&m2);
  ready2 = 1;
  while (!ready1) {}
  pthread_mutex_lock(&m1);
  return 0;
}
int main(void) {
  pthread_t t;
  pthread_create(&t, 0, worker, 0);
  pthread_mutex_lock(&m1);
  ready1 = 1;
  while (!ready2) {}
  pthread_mutex_lock(&m2);
  return 0;
}
"#;

/// **A deadlock names its cycle (#1986).** The run stops with reason `deadlock` (not `pause`, which a
/// client running in budgeted slices would take for a spent budget and resume forever), and
/// `blockedThreads` lists each thread's futex word with its value. A locked mutex holds its holder's
/// thread id + 1 (`pthread.h`), so the two waits read as "thread 1 waits on m2, held by thread 2" and
/// "thread 2 waits on m1, held by thread 1".
#[test]
fn a_deadlock_names_its_cycle() {
    let Some(bytes) = chibicc_temen() else {
        eprintln!("SKIP: chibicc.temen absent");
        return;
    };
    let chibicc = temen_encode::decode_module(&bytes).expect("decode");
    let ir = compile_g(&chibicc, DEADLOCK_SRC);
    let mut s = DapServer::new();
    launch_at(&mut s, &ir, 1, true);
    s.handle(&req(4, "configurationDone", Json::obj(vec![])));
    let out = s.handle(&req(5, "continue", Json::obj(vec![])));
    assert_eq!(stopped_reason(&out).as_deref(), Some("deadlock"));
    let out = s.handle(&req(6, "blockedThreads", Json::obj(vec![])));
    let threads = response(&out)
        .get("body")
        .and_then(|b| b.get("threads"))
        .and_then(|t| t.as_array())
        .map(<[Json]>::to_vec)
        .expect("the blocked threads");
    let wait = |id: i64| {
        let t = threads
            .iter()
            .find(|t| t.get("id").and_then(|v| v.as_i64()) == Some(id))
            .unwrap_or_else(|| panic!("thread {id} is blocked: {threads:?}"));
        let n = |k: &str| t.get(k).and_then(|v| v.as_i64()).expect(k);
        (n("futex"), n("value"))
    };
    let ((m2, held_by_2), (m1, held_by_1)) = (wait(1), wait(2));
    assert_ne!(m1, m2, "each waits on a different mutex");
    // Ids are pthread's (0 = main = DAP thread 1); the word holds id + 1.
    assert_eq!(
        held_by_2, 2,
        "thread 1 waits on m2, held by the worker (id 1)"
    );
    assert_eq!(held_by_1, 1, "thread 2 waits on m1, held by main (id 0)");
}

/// Both threads bump `counter`; with `LOCK` defined, under a mutex.
const RACE_SRC: &str = r#"#include <pthread.h>
int counter = 0;
pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
void *worker(void *arg) {
#ifdef LOCK
  pthread_mutex_lock(&m);
#endif
  counter++;
#ifdef LOCK
  pthread_mutex_unlock(&m);
#endif
  return 0;
}
int main(void) {
  pthread_t t;
  pthread_create(&t, 0, worker, 0);
#ifdef LOCK
  pthread_mutex_lock(&m);
#endif
  counter++;
#ifdef LOCK
  pthread_mutex_unlock(&m);
#endif
  pthread_join(t, 0);
  return counter;
}
"#;

/// The `races` request's list after running `src` to the end under `raceDetect`.
fn races_of(chibicc: &temen_ir::Module, src: &str) -> Vec<Json> {
    let ir = compile_g(chibicc, src);
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    let launch = vec![
        ("programText", Json::s(&ir)),
        ("function", Json::i(0)),
        ("args", Json::Arr(vec![])),
        ("engine", Json::s("bytecode")),
        ("powerbox", Json::s("onramp")),
        ("raceDetect", Json::Bool(true)),
    ];
    let out = s.handle(&req(2, "launch", Json::obj(launch)));
    assert_eq!(response(&out).get("success"), Some(&Json::Bool(true)));
    s.handle(&req(3, "configurationDone", Json::obj(vec![])));
    s.handle(&req(4, "continue", Json::obj(vec![])));
    let out = s.handle(&req(5, "races", Json::obj(vec![])));
    response(&out)
        .get("body")
        .and_then(|b| b.get("races"))
        .and_then(|r| r.as_array())
        .map(<[Json]>::to_vec)
        .expect("a races list")
}

/// **The race detector (#1987).** Two threads bumping a counter with nothing between them race: the
/// run reports a race on one word between thread 1 and thread 2, at least one side a write. Under a
/// mutex the same increments are ordered by the lock word (its CAS acquires, its unlock's store
/// releases) and nothing is reported; nor are the spawn's and the join's own orderings.
#[test]
fn the_race_detector_finds_an_unprotected_counter_and_not_a_locked_one() {
    let Some(bytes) = chibicc_temen() else {
        eprintln!("SKIP: chibicc.temen absent");
        return;
    };
    let chibicc = temen_encode::decode_module(&bytes).expect("decode");
    let racy = races_of(&chibicc, RACE_SRC);
    let thread = |r: &Json, side: &str| {
        r.get(side)
            .and_then(|s| s.get("thread"))
            .and_then(|t| t.as_i64())
    };
    let write =
        |r: &Json, side: &str| r.get(side).and_then(|s| s.get("write")) == Some(&Json::Bool(true));
    assert!(
        racy.iter().any(|r| {
            let mut pair = [thread(r, "first"), thread(r, "second")];
            pair.sort();
            pair == [Some(1), Some(2)] && (write(r, "first") || write(r, "second"))
        }),
        "a race between the two threads: {racy:?}"
    );
    let locked = races_of(&chibicc, &format!("#define LOCK\n{RACE_SRC}"));
    assert!(locked.is_empty(), "no race under the mutex: {locked:?}");
}

/// A program that calls two **declared host-completed caps** (#1953): `ping(x)` 300 times in a loop,
/// then `show(p, n)` over a global buffer the host reads back out of the window. Every call is
/// answered by the embedder: `ping` with `2x + 1`, `show` with the sum of the `n` bytes at `p`.
const HOST_CAPS_SRC: &str = r#"#include <stdio.h>
long __vm_resolve(const char *name, long len);
__attribute__((temen_cap)) extern long ping(int h, long x);
__attribute__((temen_cap)) extern long show(int h, unsigned char *p, long n);
unsigned char buf[8] = {1, 2, 3, 4, 5, 6, 7, 8};
int main(void) {
  int hp = (int)__vm_resolve("ping", 4);
  int hs = (int)__vm_resolve("show", 4);
  long s = 0;
  for (long i = 0; i < 300; i++) {
    s += ping(hp, i);
    if (i % 100 == 0) printf("i %ld s %ld\n", i, s);
  }
  long t = show(hs, buf, 8);
  printf("sum %ld show %ld\n", s, t);
  return (int)(s % 97);
}
"#;

/// The embedder's answer to a declared-cap call, given a window reader.
fn answer_cap(name: &str, args: &[i64], read: &mut dyn FnMut(u64, usize) -> Vec<u8>) -> i64 {
    match name {
        "ping" => 2 * args[0] + 1,
        "show" => read(args[0] as u64, args[1] as usize)
            .iter()
            .map(|&b| b as i64)
            .sum(),
        other => panic!("unexpected cap {other}"),
    }
}

/// Decode standard base64 (the DAP `readMemory` body).
fn b64_decode(s: &str) -> Vec<u8> {
    let val = |c: u8| match c {
        b'A'..=b'Z' => c - b'A',
        b'a'..=b'z' => c - b'a' + 26,
        b'0'..=b'9' => c - b'0' + 52,
        b'+' => 62,
        _ => 63,
    };
    let bytes: Vec<u8> = s.bytes().filter(|&c| c != b'=').map(val).collect();
    let mut out = Vec::new();
    for chunk in bytes.chunks(4) {
        let n = chunk
            .iter()
            .enumerate()
            .fold(0u32, |a, (i, &v)| a | (v as u32) << (18 - 6 * i));
        out.extend_from_slice(&n.to_be_bytes()[1..chunk.len()]);
    }
    out
}

/// The debug session's run of `ir` with `hostCaps` `ping`/`show`, answering each `stopped{cap}` with
/// `provideCap`: `(stdout, exit code, parks)`.
fn debug_host_caps_run(ir: &str) -> (String, Option<i64>, usize) {
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    let out = s.handle(&req(
        2,
        "launch",
        Json::obj(vec![
            ("programText", Json::s(ir)),
            ("function", Json::i(0)),
            ("args", Json::Arr(vec![])),
            ("engine", Json::s("bytecode")),
            ("powerbox", Json::s("onramp")),
            (
                "hostCaps",
                Json::Arr(vec![Json::s("ping"), Json::s("show")]),
            ),
        ]),
    ));
    assert_eq!(response(&out).get("success"), Some(&Json::Bool(true)));
    s.handle(&req(3, "configurationDone", Json::obj(vec![])));
    let mut seq = 10;
    let mut parks = 0;
    let mut out = s.handle(&req(seq, "continue", Json::obj(vec![])));
    loop {
        if stopped_reason(&out).as_deref() != Some("cap") {
            break;
        }
        parks += 1;
        let body = out
            .iter()
            .find(|m| m.get("event").and_then(|e| e.as_str()) == Some("stopped"))
            .and_then(|m| m.get("body"))
            .expect("stopped body")
            .clone();
        let id = body.get("capId").and_then(|v| v.as_i64()).expect("capId");
        let name = body
            .get("capName")
            .and_then(|v| v.as_str())
            .expect("capName")
            .to_string();
        let args: Vec<i64> = match body.get("args") {
            Some(Json::Arr(a)) => a.iter().filter_map(|v| v.as_i64()).collect(),
            _ => Vec::new(),
        };
        let mut read = |addr: u64, len: usize| {
            seq += 1;
            let r = s.handle(&req(
                seq,
                "readMemory",
                Json::obj(vec![
                    ("memoryReference", Json::s(addr.to_string())),
                    ("count", Json::i(len as i64)),
                ]),
            ));
            let data = response(&r)
                .get("body")
                .and_then(|b| b.get("data"))
                .and_then(|d| d.as_str())
                .unwrap_or("")
                .to_string();
            b64_decode(&data)
        };
        let value = answer_cap(&name, &args, &mut read);
        seq += 1;
        let r = s.handle(&req(
            seq,
            "provideCap",
            Json::obj(vec![("id", Json::i(id)), ("value", Json::i(value))]),
        ));
        assert_eq!(response(&r).get("success"), Some(&Json::Bool(true)));
        seq += 1;
        out = s.handle(&req(seq, "continue", Json::obj(vec![])));
    }
    let stdout = output_text(&out).unwrap_or_default();
    let code = out
        .iter()
        .find(|m| m.get("event").and_then(|e| e.as_str()) == Some("exited"))
        .and_then(|m| m.get("body")?.get("exitCode")?.as_i64());
    (stdout, code, parks)
}

/// The release run of `module` — the tier-up session with no regions — with the same declared caps,
/// pumped in `budget`-op slices: `(stdout, exit code, parks)`.
fn release_host_caps_run(module: &temen_ir::Module, budget: u64) -> (String, Option<i64>, usize) {
    use temen_browser::{
        temen_alloc, temen_coop_cap_len, temen_coop_cap_ptr, temen_coop_close,
        temen_coop_deliver_cap, temen_coop_open, temen_coop_read, temen_coop_read_ptr,
        temen_coop_run_for, temen_coop_value, temen_exit_code, temen_status, temen_stdout_len,
        temen_stdout_ptr, COOP_NO_REGIONS, COOP_RUN_CAP_PARK, COOP_RUN_DONE, COOP_RUN_PAUSED,
        STATUS_EXIT, STATUS_OK,
    };
    let bytes = temen_encode::encode_module(module);
    let p = temen_alloc(bytes.len());
    // SAFETY: `temen_alloc` returned a live allocation of that length.
    unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), p, bytes.len()) };
    let caps = b"ping\nshow";
    let cp = temen_alloc(caps.len());
    unsafe { core::ptr::copy_nonoverlapping(caps.as_ptr(), cp, caps.len()) };
    let opened = temen_coop_open(
        p,
        bytes.len(),
        core::ptr::null(),
        0,
        0,
        cp,
        caps.len(),
        COOP_NO_REGIONS,
    );
    assert_eq!(opened, temen_browser::STATUS_OK);
    let names = ["ping", "show"];
    let mut out = Vec::new();
    let mut parks = 0;
    loop {
        let r = temen_coop_run_for(budget);
        let (sp, sn) = (temen_stdout_ptr(), temen_stdout_len());
        if !sp.is_null() && sn > 0 {
            // SAFETY: the stash stays live until the next call that replaces it.
            out.extend_from_slice(unsafe { core::slice::from_raw_parts(sp, sn) });
        }
        match r {
            COOP_RUN_DONE => break,
            COOP_RUN_PAUSED => {}
            COOP_RUN_CAP_PARK => {
                parks += 1;
                // SAFETY: the request words stay live until the next deliver/run/close.
                let words = unsafe {
                    core::slice::from_raw_parts(temen_coop_cap_ptr(), temen_coop_cap_len())
                }
                .to_vec();
                let (id, name, args) = (words[0] as u64, names[words[1] as usize], &words[2..]);
                let mut read = |addr: u64, len: usize| {
                    let n = temen_coop_read(addr, len);
                    // SAFETY: as above, until the next read.
                    unsafe { core::slice::from_raw_parts(temen_coop_read_ptr(), n) }.to_vec()
                };
                let value = answer_cap(name, args, &mut read);
                assert_eq!(temen_coop_deliver_cap(id, value), 1);
                assert_eq!(temen_coop_deliver_cap(id, value), 0, "answered once");
            }
            other => panic!("unexpected release event {other}"),
        }
    }
    // `main`'s return: an `exit` status, or (a C entry that returns) the run's value — the DAP
    // session reports either as its exit code.
    let code = match temen_status() {
        STATUS_EXIT => Some(temen_exit_code() as i64),
        STATUS_OK => Some(temen_coop_value()),
        _ => None,
    };
    temen_coop_close();
    (String::from_utf8_lossy(&out).into_owned(), code, parks)
}

/// **The release session serves declared host-completed caps as the debug session does** (#1953):
/// the same program, its calls answered the same way, gives the same output, exit code and number
/// of parks on both — the release run unsliced and in 1000-op slices. `show` proves the release
/// session's bounded window read sees what the debug session's `readMemory` sees.
#[test]
fn the_release_session_serves_declared_caps_as_the_debug_session_does() {
    let Some(bytes) = chibicc_temen() else {
        eprintln!("SKIP: browser/web/assets/chibicc.temen absent");
        return;
    };
    let chibicc = temen_encode::decode_module(&bytes).expect("decode chibicc.temen");
    let ir = compile_g(&chibicc, HOST_CAPS_SRC);
    let module = temen_text::parse_module(&ir).expect("parse IR");
    let want = (
        "i 0 s 1\ni 100 s 10201\ni 200 s 40401\nsum 90000 show 36\n".to_string(),
        Some(90000 % 97),
        301,
    );
    assert_eq!(debug_host_caps_run(&ir), want, "the debug session");
    assert_eq!(
        release_host_caps_run(&module, u64::MAX),
        want,
        "release, unsliced"
    );
    assert_eq!(
        release_host_caps_run(&module, 1_000),
        want,
        "release, sliced"
    );
}
