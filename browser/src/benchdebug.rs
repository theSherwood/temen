//! **Where does the debug session's slowdown come from?** — attribution for the two-run-driver
//! question (c_interpret#26's "remove the fork instead").
//!
//! c_interpret runs a lesson two ways: `temen_run_onramp` (run to completion, one FFI call) for
//! ordinary lessons, and the DAP session's `continue` for anything that needs a capability park —
//! because `OffloadOutcome::Host` is *"admitted only under a driver that surfaces the park… any other
//! driver has no way to complete it and declines with `CapFault`"*. That fork is why an interactive
//! lesson runs its whole life on the debug driver, quoted at 8.6x the plain engine (12.8x with the
//! mem model armed).
//!
//! Two designs follow from that number, and they differ by a lot of surface area:
//!
//!  - if the gap is **intrinsic** to the session, the answer is to make the release runner resumable
//!    so it can park too — a second stateful driver at the FFI boundary;
//!  - if the gap is **per-op debug bookkeeping**, the answer is to stop doing that work when nothing
//!    is armed, and there is one driver instead of two.
//!
//! `drive()` maps `cur_ir_pc` and probes `breakpoints.contains(&pc)` on every op. Watchpoints get an
//! `is_empty()` guard; breakpoints do not. So this measures the same guest four ways to separate the
//! session's floor from what the arming costs:
//!
//!   1. `onramp_exec`                     — the release runner (the baseline)
//!   2. DAP `continue`, nothing armed     — the session's floor
//!   3. DAP `continue`, mem model on      — what c_interpret actually launches with
//!   4. DAP `continue`, one breakpoint    — armed but never hit
//!
//! Run: `cargo run --release --bin benchdebug` from `browser/`.

use std::time::Instant;

/// A compute-only guest: sum `0..n`, all in registers, no memory traffic and no capability calls — so
/// what is timed is interpretation and the driver's per-op work, not the powerbox. Import-free, so
/// `onramp_check` passes it as-is and func 0 runs with no args.
fn guest(n: i64) -> String {
    format!(
        "memory 17
func () -> (i64) {{
block 0 () {{
  vn = i64.const {n}
  vi = i64.const 0
  vacc = i64.const 0
  br 1(vi, vacc, vn)
  }}
block 1 (vi: i64, vacc: i64, vn: i64) {{
  vdone = i64.ge_s vi vn
  br_if vdone 3(vacc) 2(vi, vacc, vn)
  }}
block 2 (vi: i64, vacc: i64, vn: i64) {{
  vacc2 = i64.add vacc vi
  vone = i64.const 1
  vi2 = i64.add vi vone
  br 1(vi2, vacc2, vn)
  }}
block 3 (vacc: i64) {{
  return vacc
  }}
}}
"
    )
}

/// A memory-touching guest: the same loop, but each iteration stores to and reloads from the window.
/// The register-only guest above cannot say anything about the **mem model**, which tracks *memory*
/// accesses — with no accesses to track its cost is necessarily zero, which would be a confounded
/// measurement rather than a cheap mem model. This one gives it something to do, and is also closer
/// to real compiled C (which spills, and calls through libc).
fn guest_mem(n: i64) -> String {
    format!(
        "memory 17
func () -> (i64) {{
block 0 () {{
  vn = i64.const {n}
  vi = i64.const 0
  vacc = i64.const 0
  br 1(vi, vacc, vn)
  }}
block 1 (vi: i64, vacc: i64, vn: i64) {{
  vdone = i64.ge_s vi vn
  br_if vdone 3(vacc) 2(vi, vacc, vn)
  }}
block 2 (vi: i64, vacc: i64, vn: i64) {{
  vaddr = i64.const 65536
  i64.store vaddr vi
  vld = i64.load vaddr
  vacc2 = i64.add vacc vld
  vone = i64.const 1
  vi2 = i64.add vi vone
  br 1(vi2, vacc2, vn)
  }}
block 3 (vacc: i64) {{
  return vacc
  }}
}}
"
    )
}

/// The release path: `onramp_exec`, exactly what `temen_run_onramp` calls.
fn time_release(m: &temen_ir::Module) -> (f64, i64) {
    let t = Instant::now();
    let out = temen_browser::onramp_exec(m, &[]);
    (t.elapsed().as_secs_f64() * 1e3, out.value)
}

/// One DAP session driven to completion, as `worker-temen.ts` drives it: `initialize`, `launch`
/// (bytecode engine, on-ramp powerbox), optional `setBreakpoints`, then `continue`.
fn time_debug(ir: &str, mem_model: bool, breakpoint: Option<i64>) -> (f64, String) {
    let mut server = temen_dap::DapServer::new();
    let send = |server: &mut temen_dap::DapServer, body: String| -> Vec<String> {
        let req = temen_dap::parse(&body).expect("request parses");
        server
            .handle(&req)
            .iter()
            .map(|j| j.to_string())
            .collect::<Vec<_>>()
    };
    send(
        &mut server,
        r#"{"seq":1,"type":"request","command":"initialize","arguments":{}}"#.to_string(),
    );
    let launch = format!(
        r#"{{"seq":2,"type":"request","command":"launch","arguments":{{"programText":{},"function":0,"args":[],"engine":"bytecode","powerbox":"onramp","blockStdin":true,"memModel":{}}}}}"#,
        json_string(ir),
        mem_model
    );
    let out = send(&mut server, launch);
    if !out.iter().any(|m| m.contains("\"success\":true")) {
        return (f64::NAN, format!("launch failed: {out:?}"));
    }
    if let Some(line) = breakpoint {
        send(
            &mut server,
            format!(
                r#"{{"seq":3,"type":"request","command":"setBreakpoints","arguments":{{"source":{{"path":"/in.c"}},"breakpoints":[{{"line":{line}}}]}}}}"#
            ),
        );
    }
    // Time only the run, not the launch — the launch parses/verifies/compiles, which the release path
    // also does inside `onramp_exec`, but comparing the runs keeps the attribution clean.
    let t = Instant::now();
    let msgs = send(
        &mut server,
        r#"{"seq":4,"type":"request","command":"continue","arguments":{"threadId":1}}"#.to_string(),
    );
    let ms = t.elapsed().as_secs_f64() * 1e3;
    let ended = msgs
        .iter()
        .any(|m| m.contains("\"terminated\"") || m.contains("\"exited\""));
    (
        ms,
        if ended {
            "ran to completion".to_string()
        } else {
            format!("did NOT finish: {msgs:?}")
        },
    )
}

/// Minimal JSON string escaping — the IR is multi-line, so it cannot go in raw.
fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn main() {
    // Big enough that per-op costs dominate fixed setup, small enough to stay well under a minute
    // even on the slowest row.
    const N: i64 = 2_000_000;
    let expect = (N - 1) * N / 2;

    let guests = [
        ("register-only loop (no memory traffic)", guest(N)),
        ("same loop, one store + one load per iteration", guest_mem(N)),
    ];

    for (what, ir) in &guests {
        let m = temen_text::parse_module(ir).expect("the guest parses");

        // Warm both paths so neither pays a cold cache in the numbers below.
        let _ = time_release(&m);
        let _ = time_debug(ir, false, None);

        let (release_ms, value) = time_release(&m);
        assert_eq!(value, expect, "{what}: the release run computed the wrong sum");

        println!("\n=== {what} — sum 0..{N} ===");
        println!("{:<46} {:>10}  {:>10}", "driver", "ms", "vs release");
        println!(
            "{:<46} {:>10.1}  {:>10}",
            "onramp_exec (release runner)", release_ms, "1.0x"
        );
        for (label, mem_model, bp) in [
            ("DAP continue, nothing armed", false, None),
            ("DAP continue, mem model on", true, None),
            ("DAP continue, 1 breakpoint (never hit)", false, Some(1)),
            ("DAP continue, mem model + breakpoint", true, Some(1)),
        ] {
            let (ms, note) = time_debug(ir, mem_model, bp);
            if ms.is_nan() || !note.starts_with("ran") {
                println!("{label:<46} {:>10}  {note}", "FAILED");
                continue;
            }
            println!("{:<46} {:>10.1}  {:>9.1}x", label, ms, ms / release_ms);
        }
    }

    println!(
        "\nHow to read it:
  * `nothing armed` vs release is the session's **floor** — a `continue` with nothing to check.
  * `1 breakpoint` minus `nothing armed` is what **arming** costs. `drive()` probes
    `breakpoints.contains(&pc)` every op with no `is_empty()` guard, so a small delta means the
    probe is not the cost and the unconditional per-op `cur_ir_pc` mapping is.
  * `mem model` minus `nothing armed` is what c_interpret arms on every launch. Read it off the
    **second** guest; the first has no memory accesses for it to track."
    );
}
