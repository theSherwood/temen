//! #2962 — what the message-region bounce costs a call nobody interposes. A guest loop `run`s a
//! `cat`-shaped job with `N` bytes of stdin and reads its `N` bytes of output back, through the
//! `exec.msg` region, in two shapes:
//!
//! - **bounce**: the caller's library copies its stdin into the region before `run` and the output
//!   out of it after `read_out` (`mem.copy`, as a libc would);
//! - **native**: a program that keeps its data in the region already, which copies nothing.
//!
//! Against the scripted backend (no process: the protocol and copies are all there is) and a real
//! `host_exec(["cat"])` (a process per call: where I/O dominates). Per-call time is
//! `(t(K) − t(0)) / K`, so instantiation and the grant drop out. Run with
//! `cargo run --release -p temen-run --example exec_bounce`. main's pointer path is measured by the
//! same loop over the old protocol (see #2962).

use std::time::Instant;
use temen_run::exec::{host_exec, scripted_exec, ScriptedEntry};
use temen_run::{instantiate_with_imports, Backend, HostCap, Imports, Limits, RunConfig};

/// The loop: `k` calls, each `run(cat, stdin n)` + one `read_out` of up to `n` + `close`.
fn guest(k: u64, n: usize, bounce: bool) -> String {
    let copy_in = if bounce {
        "  mem.copy vstdin vsrc vn\n"
    } else {
        ""
    };
    let copy_out = if bounce {
        "  mem.copy vdst vout vr\n"
    } else {
        ""
    };
    format!(
        "memory 18
data 16384 \"exec.msg\"
data 16392 \"cat\"
func 0 () -> () {{
block 0 () {{
  vp = i64.const 16384
  vl = i64.const 4
  vh = self.resolve vp vl
  vml = i64.const 8
  vmsg = self.resolve vp vml
  vwin = i64.const 131072
  vz = i64.const 0
  vlen = i64.const 65536
  vprot = i32.const 3
  vm = call.cap 4 0 (i64, i64, i64, i32) -> (i64) vmsg (vwin, vz, vlen, vprot)
  vname = i64.const 16392
  vthree = i64.const 3
  mem.copy vwin vname vthree
  vk = i64.const {k}
  br 1(vh, vk)
}}
block 1 (h: i32, left: i64) {{
  z = i64.const 0
  done = i64.eq left z
  br_if done 3() 2(h, left)
}}
block 2 (h: i32, left: i64) {{
  vn = i64.const {n}
  vstdin = i64.const 131075
  vsrc = i64.const 32768
{copy_in}  vrun = i64.const 0
  voff = i64.const 0
  val = i64.const 3
  vjob = call.cap 13 0 (i64, i64, i64, i64) -> (i64) h (vrun, voff, val, vn)
  vread = i64.const 1
  vo = i64.const 4096
  vr = call.cap 13 0 (i64, i64, i64, i64) -> (i64) h (vread, vjob, vo, vn)
  vout = i64.const 135168
  vdst = i64.const 98304
{copy_out}  vclose = i64.const 4
  vc = call.cap 13 0 (i64, i64, i64, i64) -> (i64) h (vclose, vjob, voff, voff)
  one = i64.const 1
  left2 = i64.sub left one
  br 1(h, left2)
}}
block 3 () {{
  return
  }}
}}
export 0 func \"_start\" 0
"
    )
}

fn time(cap: &HostCap, backend: Backend, k: u64, n: usize, bounce: bool) -> f64 {
    let m = temen_text::parse_module(&guest(k, n, bounce)).expect("parse");
    let inst = instantiate_with_imports(m, Imports::new()).expect("instantiate");
    let cfg = RunConfig {
        limits: Limits {
            fuel: Some(u64::MAX),
            ..Limits::default()
        },
        ..RunConfig::default()
    };
    let t = Instant::now();
    inst.run_with_caps(backend, &cfg, &[("exec", cap.clone())])
        .expect("run");
    t.elapsed().as_secs_f64()
}

/// The median per-call time over `reps` runs of `k` calls, in ns.
fn per_call(cap: &HostCap, backend: Backend, k: u64, n: usize, bounce: bool, reps: usize) -> f64 {
    let mut v: Vec<f64> = (0..reps)
        .map(|_| (time(cap, backend, k, n, bounce) - time(cap, backend, 0, n, bounce)) / k as f64)
        .collect();
    v.sort_by(f64::total_cmp);
    v[reps / 2] * 1e9
}

fn main() {
    for n in [64usize, 4096, 32768] {
        let scripted = scripted_exec(vec![ScriptedEntry {
            argv_prefix: vec!["cat".into()],
            stdout: vec![b'x'; n],
            stderr: Vec::new(),
            exit: 0,
        }]);
        for backend in [Backend::Jit, Backend::Bytecode] {
            let native = per_call(&scripted, backend, 20_000, n, false, 7);
            let bounce = per_call(&scripted, backend, 20_000, n, true, 7);
            println!("scripted {backend:?} n={n}: native {native:.0} ns, bounce {bounce:.0} ns");
        }
    }
    let cat = host_exec(&["cat"]);
    for n in [64usize, 32768] {
        for backend in [Backend::Jit, Backend::Bytecode] {
            let native = per_call(&cat, backend, 300, n, false, 5);
            let bounce = per_call(&cat, backend, 300, n, true, 5);
            println!("host cat {backend:?} n={n}: native {native:.0} ns, bounce {bounce:.0} ns");
        }
    }
}
