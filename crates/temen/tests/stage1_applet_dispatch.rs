//! Stage 1 (STAGE1.md) slice 3 — **exit-status fidelity across a multi-applet binary**: one module
//! carries several "external commands" as applet entries (`true` → 0, `false` → 1, `echo` → writes
//! its seeded argv and returns the byte count), and a parent "shell" spawns a chosen applet, inherits
//! stdout into it, `join`s, and returns its status. Spawning different applets yields different
//! `(stdout, status)` pairs — the guarantee the shell's command dispatch rests on: look a command up,
//! spawn the matching applet, thread its exit code into `$?`.
//!
//! The name→entry lookup itself is trivial personality glue (a map) and lives above this; here the
//! entry index is chosen per case, exactly as the shell will compute it, and the host grants the
//! parent that applet's child image (#2219). BusyBox-multicall shape (a detached v1 record spawn
//! with one named grant, + `join`), differential interp==JIT.
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
/// grant record, spawns the applet image it is handed detached through a v1 record paid from its
/// `Budget` that carries the token as the args payload (it lands at the applet's
/// `module_args_base`), joins, and returns its status.
fn src(token: &[u8; 3]) -> String {
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
{rec}func (i32, i32, i32, i32) -> (i64) {{
block 0 (vinst: i32, vout: i32, vbud: i32, vapp: i32) {{
  a0 = i64.const 16384
  n100 = i32.const 16484
  i32.store a0 n100
  a4 = i64.const 16388
  n6 = i32.const 6
  i32.store a4 n6
  a8 = i64.const 16392
  i32.store a8 vout
  a12 = i64.const 16396
  z0 = i32.const 0
  i32.store a12 z0
  cs = i32.const 115
  ct = i32.const 116
  cd = i32.const 100
  co = i32.const 111
  cu = i32.const 117
  p100 = i64.const 16484
  i32.store8 p100 cs
  p101 = i64.const 16485
  i32.store8 p101 ct
  p102 = i64.const 16486
  i32.store8 p102 cd
  p103 = i64.const 16487
  i32.store8 p103 co
  p104 = i64.const 16488
  i32.store8 p104 cu
  p105 = i64.const 16489
  i32.store8 p105 ct
{seed}  rrm = i64.const 17560
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
  cs = i32.const 115
  ct = i32.const 116
  cd = i32.const 100
  co = i32.const 111
  cu = i32.const 117
  a200 = i64.const 16584
  i32.store8 a200 cs
  a201 = i64.const 16585
  i32.store8 a201 ct
  a202 = i64.const 16586
  i32.store8 a202 cd
  a203 = i64.const 16587
  i32.store8 a203 co
  a204 = i64.const 16588
  i32.store8 a204 cu
  a205 = i64.const 16589
  i32.store8 a205 ct
  len6 = i64.const 6
  hout = self.resolve a200 len6
  a0 = i64.const {args}
  len3 = i64.const 3
  w = call.cap 0 1 (i64, i64) -> (i64) hout (a0, len3)
  return w
  }}
}}
"#,
        rec = rec::segment(17536, &spawn),
    )
}

/// The module, and a host for it with the parent's four args: an `Instantiator`, the `stdout` it
/// grants the applet, the `Budget` that pays for the applet's window, and applet `entry`'s child
/// image.
fn setup(entry: u32, token: &[u8; 3]) -> (Module, Host, [i32; 4]) {
    let m = parse_module(&src(token)).expect("parse");
    verify_module(&m).expect("verify");
    let mut host = Host::new();
    let ih = host.grant_instantiator(0, WIN as u64);
    let oh = host.grant_stream(StreamRole::Out);
    let bh = host.grant_budget(-1, 1 << 20, -1);
    let ah = host.grant_module(&temen_ir::child_image_at(&m, entry).expect("applet image"));
    (m, host, [ih, oh, bh, ah])
}

fn run_interp(entry: u32, token: &[u8; 3]) -> (Result<Vec<Value>, Trap>, Vec<u8>) {
    let (m, mut host, args) = setup(entry, token);
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

fn run_jit(entry: u32, token: &[u8; 3]) -> (JitOutcome, Vec<u8>) {
    let (m, mut host, args) = setup(entry, token);
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
    // (entry, expected status, expected stdout)
    let cases: &[(u32, i64, &[u8])] = &[(1, 0, b""), (2, 1, b""), (3, 3, b"hey")];
    for &(entry, status, out) in cases {
        let token = b"hey";
        let (ir, iout) = run_interp(entry, token);
        let (jo, jout) = run_jit(entry, token);
        assert_eq!(
            ir.expect("interp run ok"),
            vec![Value::I64(status)],
            "interp: applet {entry} status"
        );
        assert_eq!(iout, out, "interp: applet {entry} stdout");
        assert!(
            matches!(jo, JitOutcome::Returned(ref s) if s == &[status]),
            "jit: applet {entry} status must be {status}, got {jo:?}"
        );
        assert_eq!(jout, iout, "jit: applet {entry} stdout must match interp");
    }
}
