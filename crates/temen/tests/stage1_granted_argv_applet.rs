//! Stage 1 (STAGE1.md) — an applet echoes its parent-**seeded** `argv` through a re-granted
//! `stdout`, differentially on both backends (the output tracks the seed, so it's a real argv).
//!
//! **§3d note:** this file originally pinned the positional handle-as-3rd-entry-arg ABI
//! (`instantiate_granted`, op 8). That op is deleted — no committed asset used it, and the real
//! compiled-C path gets its handles through the **import-manifest binding**
//! (`bind_child_manifest`: a chibicc child's generic `write`/`read`/`exit` imports resolve against
//! its granted powerbox — the op-13/17 named-grant path the STAGE1 shell actually uses; see
//! `stage1_stdio_child.rs`). The spawn here is now the op-17 record with a one-entry named-grant
//! list, and the applet resolves `"g"` by name — the argv-seed + granted-I/O coverage is
//! unchanged.
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

/// Parent (`(Instantiator, stdout)`) seeds `token` into the applet's carve, spawns the applet
/// (func 1) via the record re-granting its stdout under `"g"` (name at 4096, grant record at 4104,
/// spawn record at 4160), joins, and returns the applet's status. The applet
/// (`(Instantiator, AddressSpace)`) resolves `"g"` and writes its seeded 3-byte argv through it,
/// returning the byte count.
fn src(token: &[u8; 3]) -> String {
    let seed: String = token
        .iter()
        .enumerate()
        .map(|(i, &b)| {
            let addr = CARVE + i as u64;
            format!("  q{i} = i64.const {addr}\n  c{i} = i32.const {b}\n  i32.store8 q{i} c{i}\n")
        })
        .collect();
    format!(
        "memory 17
func (i32, i32) -> (i64) {{
block 0 (vinst: i32, vout: i32) {{
{seed}  vg = i32.const 103
  vnp1 = i64.const 4096
  i32.store8 vnp1 vg
  vgr0 = i64.const 4104
  vno = i32.const 4096
  i32.store vgr0 vno
  vgr1 = i64.const 4108
  vnl1 = i32.const 1
  i32.store vgr1 vnl1
  vgr2 = i64.const 4112
  i32.store vgr2 vout
  vgr3 = i64.const 4116
  vz32 = i32.const 0
  i32.store vgr3 vz32
  rrv0 = i64.const 4294967296
  rrv1 = i64.const {CARVE}
  rrv2 = i64.const -4294967280
  rrv3 = i64.const 4294967295
  rrvz = i64.const 0
  rrgp = i64.const 4104
  rrgn = i64.const 1
  rra0 = i64.const 4160
  i64.store rra0 rrv0
  rra1 = i64.const 4168
  i64.store rra1 rrv1
  rra2 = i64.const 4176
  i64.store rra2 rrv2
  rra3 = i64.const 4184
  i64.store rra3 rrv3
  rra4 = i64.const 4192
  i64.store rra4 rrvz
  rra5 = i64.const 4200
  i64.store rra5 rrgp
  rra6 = i64.const 4208
  i64.store rra6 rrgn
  vch = cap.call 6 17 (i64) -> (i32) vinst (rra0)
  vres = cap.call 6 1 (i32) -> (i64) vinst (vch)
  return vres
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (vcinst: i64, vcas: i64) {{
  vg = i32.const 103
  vnp = i64.const 512
  i32.store8 vnp vg
  vnl = i64.const 1
  vsh = cap.self.resolve vnp vnl
  vptr = i64.const 0
  vlen = i64.const 3
  vw = cap.call 0 1 (i64, i64) -> (i64) vsh (vptr, vlen)
  return vw
  }}
}}
"
    )
}

fn run_interp(token: &[u8; 3]) -> (Result<Vec<Value>, Trap>, Vec<u8>) {
    let m = parse_module(&src(token)).expect("parse");
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

fn run_jit(token: &[u8; 3]) -> (JitOutcome, Vec<u8>) {
    let m = parse_module(&src(token)).expect("parse");
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

/// The applet writes its seeded argv through its named-grant stdout, returning the byte count —
/// identically on both backends, output tracking the seed. (The compiled-C handle plumbing lives
/// in the import-manifest binding — see the header.)
#[test]
fn granted_applet_echoes_argv_through_handle_arg() {
    for token in [b"hey", b"yo!"] {
        let (ir, iout) = run_interp(token);
        let (jo, jout) = run_jit(token);
        assert_eq!(
            ir.expect("interp run ok"),
            vec![Value::I64(3)],
            "interp: applet status = bytes written for {token:?}"
        );
        assert_eq!(
            iout, token,
            "interp: applet echoed seeded argv via the handle arg"
        );
        assert!(
            matches!(jo, JitOutcome::Returned(ref s) if s == &[3]),
            "jit: applet status must be 3 for {token:?}, got {jo:?}"
        );
        assert_eq!(jout, iout, "jit: handle-arg echo must match interp");
    }
}
