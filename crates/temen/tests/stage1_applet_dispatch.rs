//! Stage 1 (STAGE1.md) slice 3 — **exit-status fidelity across a multi-applet binary**: one module
//! carries several "external commands" as applet entries (`true` → 0, `false` → 1, `echo` → writes
//! its seeded argv and returns the byte count), and a parent "shell" spawns a chosen applet, inherits
//! stdout into it, `join`s, and returns its status. Spawning different applets yields different
//! `(stdout, status)` pairs — the guarantee the shell's command dispatch rests on: look a command up,
//! spawn the matching entry, thread its exit code into `$?`.
//!
//! The name→entry lookup itself is trivial personality glue (a map) and lives above this; here the
//! entry index is chosen per case, exactly as the shell will compute it. BusyBox-multicall shape
//! (`instantiate_named`, op 11 + `join`, op 1), differential interp==JIT.
//!
//! Gated `#![cfg(unix)]` like the other JIT differential suites (temen-jit's guard page is unix-only).
#![cfg(unix)]

#[path = "support/grant_hooks.rs"]
mod grant_hooks_mod;
use grant_hooks_mod::grant_hooks;

use temen_interp::{run_capture_reserved_with_host, Host, StreamRole, Trap, Value};
use temen_jit::{compile_and_run_capture_reserved_with_host_ex, JitOutcome};
use temen_text::parse_module;
use temen_verify::verify_module;

const WIN: usize = 128 << 10;
const CARVE: u64 = 64 << 10;

/// One module: parent (func 0) plus three applets — func 1 `true` (→0), func 2 `false` (→1), func 3
/// `echo` (resolve `stdout`, write 3 seeded bytes, →3). The parent seeds `token` into the applet's
/// carve, lays a `stdout` grant record, spawns applet `entry`, joins, and returns its status.
fn src(entry: u64, token: &[u8; 3]) -> String {
    let f0 = (entry << 32) as i64;
    let seed: String = token
        .iter()
        .enumerate()
        .map(|(i, &b)| {
            let addr = CARVE + 16384 + i as u64;
            format!("  q{i} = i64.const {addr}\n  c{i} = i32.const {b}\n  i32.store8 q{i} c{i}\n")
        })
        .collect();
    format!(
        r#"memory 17
func (i32, i32) -> (i64) {{
block 0 (vinst: i32, vout: i32) {{
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
{seed}  ; spawn via record (op 17): off=CARVE sl=16 quota=0, one named grant record at 16384
  rrv0 = i64.const {f0}
  rrv1 = i64.const {CARVE}
  rrv2 = i64.const -4294967280
  rrv3 = i64.const 4294967295
  rrvz = i64.const 0
  rrgp = i64.const 16384
  rrv1n = i64.const 1
  rra0 = i64.const 17536
  i64.store rra0 rrv0
  rra1 = i64.const 17544
  i64.store rra1 rrv1
  rra2 = i64.const 17552
  i64.store rra2 rrv2
  rra3 = i64.const 17560
  i64.store rra3 rrv3
  rra4 = i64.const 17568
  i64.store rra4 rrvz
  rra5 = i64.const 17576
  i64.store rra5 rrgp
  rra6 = i64.const 17584
  i64.store rra6 rrv1n
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
  a0 = i64.const 16384
  len3 = i64.const 3
  w = call.cap 0 1 (i64, i64) -> (i64) hout (a0, len3)
  return w
  }}
}}
"#
    )
}

fn run_interp(entry: u64, token: &[u8; 3]) -> (Result<Vec<Value>, Trap>, Vec<u8>) {
    let m = parse_module(&src(entry, token)).expect("parse");
    verify_module(&m).expect("verify");
    let mut host = Host::new();
    let ih = host.grant_instantiator(0, WIN as u64);
    let oh = host.grant_stream(StreamRole::Out);
    let mut fuel = 5_000_000u64;
    let (res, _snap) = run_capture_reserved_with_host(
        &m,
        0,
        &[Value::I32(ih), Value::I32(oh)],
        &mut fuel,
        &[0u8; WIN],
        0,
        &mut host,
    );
    (res, host.stdout_bytes())
}

fn run_jit(entry: u64, token: &[u8; 3]) -> (JitOutcome, Vec<u8>) {
    let m = parse_module(&src(entry, token)).expect("parse");
    verify_module(&m).expect("verify");
    let mut host = Host::new();
    let ih = host.grant_instantiator(0, WIN as u64);
    let oh = host.grant_stream(StreamRole::Out);
    let (jo, _jmem) = compile_and_run_capture_reserved_with_host_ex(
        &m,
        0,
        &[ih as i64, oh as i64],
        &[0u8; WIN],
        0,
        temen_run::cap_thunk,
        &mut host as *mut Host as *mut core::ffi::c_void,
        None,
        Some(grant_hooks()),
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
    let cases: &[(u64, i64, &[u8])] = &[(1, 0, b""), (2, 1, b""), (3, 3, b"hey")];
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
