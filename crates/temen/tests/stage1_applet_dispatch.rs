//! Stage 1 (STAGE1.md) slice 3 — **exit-status fidelity across a multi-applet binary**: one module
//! carries several "external commands" as applets (`true` → 0, `false` → 1, `echo` → writes its
//! seeded argv and returns the byte count), and a parent "shell" spawns a chosen applet, inherits
//! stdout into it, `join`s, and returns its status. Spawning different applets yields different
//! `(stdout, status)` pairs — the guarantee the shell's command dispatch rests on: look a command up,
//! spawn the matching applet, thread its exit code into `$?`.
//!
//! The binary exports each applet by name, and the host grants each one's child image (#2219) under
//! that name, so the shell looks a command up by name and spawns what it finds. BusyBox-multicall
//! shape (a detached v1 record spawn with one named grant, + `join`), differential interp==JIT.
//!
//! Gated `#![cfg(unix)]` like the other JIT differential suites (temen-jit's guard page is unix-only).
#![cfg(unix)]

#[path = "../../temen-interp/tests/support/rec.rs"]
mod rec;

use temen_interp::{run_capture_reserved_with_host, Host, MemLayout, StreamRole, Trap, Value};
use temen_ir::{module_args_base, Module, SpawnRec};
use temen_jit::JitOutcome;
use temen_text::parse_module;
use temen_verify::verify_module;

const WIN: usize = 128 << 10;
/// Where the parent stores the applet's argv before the spawn, in its own window.
const TOKEN_AT: u64 = 16700;

/// One module: parent (func 0) plus three applets — func 1 `true` (→0), func 2 `false` (→1), func 3
/// `echo` (resolve `stdout`, write its 3 argv bytes, →3). The parent stores `token`, lays a `stdout`
/// grant record, resolves `applet`, spawns it detached through a v1 record paid from its `Budget`
/// that carries the token as the args payload (it lands at the applet's `module_args_base`), joins,
/// and returns its status.
fn src(applet: &str, token: &[u8; 3]) -> String {
    let seed: String = token
        .iter()
        .enumerate()
        .map(|(i, &b)| {
            let addr = TOKEN_AT + i as u64;
            format!("  q{i} = i64.const {addr}\n  c{i} = i32.const {b}\n  i32.store8 q{i} c{i}\n")
        })
        .collect();
    let spawn = SpawnRec {
        grants_ptr: 16384,
        grants_n: 1,
        args: (TOKEN_AT, token.len() as u64),
        ..SpawnRec::v1(0)
    };
    let args = module_args_base();
    format!(
        r#"memory 17
data 16484 "stdout"
data 16620 "{applet}"
{rec}export 0 func "true" 1
export 1 func "false" 2
export 2 func "echo" 3
func (i32, i32, i32) -> (i64) {{
block 0 (vinst: i32, vout: i32, vbud: i32) {{
{stdout_grant}{seed}  vap = i64.const 16620
  val = i64.const {applet_len}
  vapp = self.resolve vap val
  rrm = i64.const 17560
  i32.store rrm vapp
  rrb = i64.const 17564
  i32.store rrb vbud
  rra0 = i64.const 17536
  vch = call.cap 6 17 (i64) -> (i32) vinst (rra0)
  r = call.cap 6 1 (i32) -> (i64) vinst (vch)
  return r
  }}
}}
func (i64) -> (i64) {{
block 0 (vt: i64) {{
  z = i64.const 0
  return z
  }}
}}
func (i64) -> (i64) {{
block 0 (vf: i64) {{
  o = i64.const 1
  return o
  }}
}}
func (i64) -> (i64) {{
block 0 (vci: i64) {{
  ; `"stdout"` is a data segment, so the applet finds the name in its own window too
  np = i64.const 16484
  len6 = i64.const 6
  hout = self.resolve np len6
  a0 = i64.const {args}
  len3 = i64.const 3
  w = call.cap 0 1 (i64, i64) -> (i64) hout (a0, len3)
  return w
  }}
}}
"#,
        rec = rec::segment(17536, &spawn),
        stdout_grant = rec::grant("g", 16384, 16484, 6, "vout"),
        applet_len = applet.len(),
    )
}

/// The module, and a host for it with the parent's three args: an `Instantiator`, the `stdout` it
/// grants the applet, and the `Budget` that pays for the applet's window. The host grants each
/// applet the binary exports by its name.
fn setup(applet: &str, token: &[u8; 3]) -> (Module, Host, [i32; 3]) {
    let m = parse_module(&src(applet, token)).expect("parse");
    verify_module(&m).expect("verify");
    let mut host = Host::new();
    let ih = host.grant_instantiator(0, WIN as u64);
    let oh = host.grant_stream(StreamRole::Out);
    let bh = host.grant_budget(-1, 1 << 20, -1);
    for e in &m.exports {
        let h = host.grant_module(&temen_ir::child_image_at(&m, e.func).expect("applet image"));
        host.register_cap_name(&e.name, h);
    }
    (m, host, [ih, oh, bh])
}

fn run_interp(applet: &str, token: &[u8; 3]) -> (Result<Vec<Value>, Trap>, Vec<u8>) {
    let (m, mut host, args) = setup(applet, token);
    let mut fuel = 5_000_000u64;
    let (res, _snap) = run_capture_reserved_with_host(
        &m,
        0,
        &args.map(Value::I32),
        &mut fuel,
        &[0u8; WIN],
        0,
        &mut host,
    );
    (res, host.stdout_bytes())
}

fn run_jit(applet: &str, token: &[u8; 3]) -> (JitOutcome, Vec<u8>) {
    let (m, mut host, args) = setup(applet, token);
    let (jo, _) = temen_run::jit_cap_run(
        &m,
        0,
        &args.map(i64::from),
        &MemLayout::image(vec![0u8; WIN]),
        0,
        0,
        &mut host,
        None,
    )
    .expect("jit");
    (jo, host.stdout_bytes())
}

/// Spawning each applet yields its own `(status, stdout)`: `true`→(0,""), `false`→(1,""),
/// `echo`→(3,"hey"). Both backends agree — the shell's dispatch can thread any command's exit code
/// into `$?` and see its output on the inherited stream.
#[test]
fn dispatch_selects_applet_and_threads_its_status() {
    // (applet, expected status, expected stdout)
    let cases: &[(&str, i64, &[u8])] = &[("true", 0, b""), ("false", 1, b""), ("echo", 3, b"hey")];
    for &(applet, status, out) in cases {
        let token = b"hey";
        let (ir, iout) = run_interp(applet, token);
        let (jo, jout) = run_jit(applet, token);
        assert_eq!(
            ir.expect("interp run ok"),
            vec![Value::I64(status)],
            "interp: applet {applet} status"
        );
        assert_eq!(iout, out, "interp: applet {applet} stdout");
        assert!(
            matches!(jo, JitOutcome::Returned(ref s) if s == &[status]),
            "jit: applet {applet} status must be {status}, got {jo:?}"
        );
        assert_eq!(jout, iout, "jit: applet {applet} stdout must match interp");
    }
}
