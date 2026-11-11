//! #2962 (universal B, slice 1) — **one compiled `exec` client, served by the host or by a guest,
//! none the wiser** (EXEC.md's one-interface promise, with buffers).
//!
//! The client is ordinary C compiled by chibicc. It resolves the `"exec.msg"` message region the
//! wirer granted beside its `exec` slot, maps it, and drives `exec` through one flat import,
//! `exec(op, a, b, c)`: its small library bounces argv and stdin into the region before `run` and
//! each output chunk out of it after `read_out`. It runs `echo hi` and `cat` with stdin `yo`, checks
//! every byte and status it gets back, and returns `1111` iff all of it is right.
//!
//! The same compiled module is spawned, detached, by two parents:
//!
//! - **host-served:** the parent holds a real `host_exec(["echo", "cat"])` and re-grants it, and the
//!   region the grant brought, to the child by name. The host servicer reads and writes the region's
//!   backing and never touches the child's window.
//! - **guest-served:** the parent mints a region, maps it into its own window, and grants it to the
//!   child with a live grant of its own `exec` export. Its handler (IR, func 2) reads argv and stdin
//!   out of the region at the offsets the call names and writes output back there; its serve loop
//!   answers until the child is gone.
//!
//! Each runs on every driver: the tree-walker, the bytecode engine, the Cranelift JIT and the
//! parallel driver (INVARIANTS 14). The JIT runs a serving parent through the bytecode engine, as it
//! does every serve loop it folds.
#![cfg(unix)]

#[path = "support/chibicc.rs"]
mod chibicc_mod;
use chibicc_mod::chibicc;

#[path = "../../temen-interp/tests/support/rec.rs"]
mod rec;

use std::process::Command;
use std::sync::Arc;

use temen_ir::{Module, SpawnRec};
use temen_run::{instantiate_with_imports, Backend, HostCap, Imports, Outcome, RunConfig};
use temen_text::parse_module;
use temen_verify::verify_module;

/// Every driver a detached run takes: each backend, then the parallel driver (`None`).
const DRIVERS: [Option<Backend>; 4] = [
    Some(Backend::TreeWalk),
    Some(Backend::Bytecode),
    Some(Backend::Jit),
    None,
];

/// The client: a two-function `exec` library over the message region, and a `main` that checks
/// what two runs give back. Scored so a wrong answer says which part was wrong.
const CLIENT: &str = r#"
long exec(int h, long op, long a, long b, long c);
long __vm_region_map(int r, long win_off, long region_off, long len, int prot);
long __vm_resolve(const char *name, long len);

enum { RUN = 0, READ_OUT = 1, STATUS = 3, CLOSE = 4 };

_Alignas(65536) static char msg[65536];

static void copy(char *d, const char *s, long n) {
  for (long i = 0; i < n; i++)
    d[i] = s[i];
}

static int same(const char *a, const char *b, long n) {
  for (long i = 0; i < n; i++)
    if (a[i] != b[i])
      return 0;
  return 1;
}

/* The library: bounce argv and stdin into the region, then run. */
static long xrun(const char *argv, long argv_len, const char *in, long in_len) {
  copy(msg, argv, argv_len);
  copy(msg + argv_len, in, in_len);
  return exec(0, RUN, 0, argv_len, in_len);
}

/* Read a chunk of output into the region, then bounce it out. */
static long xread(long job, char *buf, long cap) {
  long n = exec(0, READ_OUT, job, 0, cap);
  if (n > 0)
    copy(buf, msg, n);
  return n;
}

int main(void) {
  long r = __vm_resolve("exec.msg", 8);
  if (r < 0)
    return 1;
  long mapped = __vm_region_map(r, (long)msg, 0, 65536, 3);
  if (mapped < 0)
    return 2;
  char out[32];
  int score = 0;
  long job = xrun("echo\0hi", 7, "", 0);
  if (job < 0)
    return 3;
  long n = xread(job, out, 32);
  if (n == 3 && same(out, "hi\n", 3))
    score += 1;
  if (exec(0, STATUS, job, 0, 0) == 0)
    score += 10;
  exec(0, CLOSE, job, 0, 0);
  job = xrun("cat", 3, "yo", 2);
  if (job < 0)
    return 4;
  n = xread(job, out, 32);
  if (n == 2 && same(out, "yo", 2))
    score += 100;
  if (xread(job, out, 32) == 0)
    score += 1000;
  exec(0, CLOSE, job, 0, 0);
  return score;
}
"#;

/// The client compiled by chibicc with the §14 spawnable `--child-entry` ABI, parsed and verified.
fn client() -> Module {
    let base = std::env::temp_dir().join(format!("temen_exec_region_{}", std::process::id()));
    let (cfile, irfile) = (base.with_extension("c"), base.with_extension("temen"));
    std::fs::write(&cfile, CLIENT).unwrap();
    let (cin, cout) = (cfile.to_str().unwrap(), irfile.to_str().unwrap());
    let status = Command::new(chibicc())
        .args([
            "-cc1",
            "--emit-ir",
            "--child-entry",
            "-cc1-input",
            cin,
            "-cc1-output",
            cout,
            cin,
        ])
        .status()
        .expect("run chibicc");
    assert!(status.success(), "chibicc failed on the client");
    let m = parse_module(&std::fs::read_to_string(&irfile).unwrap()).expect("parse the client");
    verify_module(&m).expect("verify the client");
    m
}

/// A parent that spawns the `"client"` module detached, granting it `exec` and `exec.msg` (the
/// grant list at 16640, the handles in `vex` and `vmsg`, which `setup` defines), then runs `serve`
/// and returns what the client returned. `"exec.msg"` is the data at 16684, whose first four bytes
/// spell `"exec"`. `servicer` is the guest servicer's types and export, then its functions from
/// `func 1` on; empty for none.
fn parent(setup: &str, serve: &str, servicer: &str) -> String {
    let (head, funcs) = servicer
        .split_once("func 1")
        .map_or(("", ""), |(h, f)| (h, f));
    let funcs = if funcs.is_empty() {
        String::new()
    } else {
        format!("func 1{funcs}")
    };
    let rec = SpawnRec {
        grants_ptr: 16640,
        grants_n: 2,
        ..SpawnRec::default()
    };
    format!(
        "\
memory 17
data 16384 \"vm\"
data 16400 \"budget\"
data 16448 \"client\"
data 16684 \"exec.msg\"
{rec}{head}func 0 () -> (i32) {{
block 0 () {{
  vp = i64.const 16384
  vl = i64.const 2
  vh = self.resolve vp vl
  vbp = i64.const 16400
  vbl = i64.const 6
  vb = self.resolve vbp vbl
  vba = i64.const {budget_at}
  i32.store vba vb
  vcp = i64.const 16448
  vcl = i64.const 6
  vchild = self.resolve vcp vcl
  vma = i64.const {module_at}
  i32.store vma vchild
{setup}{g_exec}{g_msg}  vrp = i64.const 17408
  vch = call.cap 6 17 (i64) -> (i32) vh (vrp)
{serve}  vj_st, vj = call.cap 6 1 (i32) -> (i64, i64) vh (vch)
  vr = i32.wrap_i64 vj
  return vr
  }}
}}
{funcs}",
        rec = rec::segment(17408, &rec),
        budget_at = 17408 + SpawnRec::BUDGET_AT,
        module_at = 17408 + SpawnRec::MODULE_AT,
        g_exec = rec::grant("ge", 16640, 16684, 4, "vex"),
        g_msg = rec::grant("gm", 16656, 16684, 8, "vmsg"),
    )
}

/// The host-served parent: it re-grants the `exec` it holds and the region its grant brought.
fn host_served() -> String {
    parent(
        "  vep = i64.const 16684
  vel = i64.const 4
  vex = self.resolve vep vel
  vml = i64.const 8
  vmsg = self.resolve vep vml
",
        "",
        "",
    )
}

/// The guest-served parent. It mints a 64 KiB region (`AddressSpace` op 5), maps it at 65536 of its
/// own window, grants it with a live grant of its export 0, and serves until `svc.wait` finds the
/// client gone. Its handler (func 2) keeps a job table in its own window: per job `j` a length at
/// `24576 + 16j`, a read cursor 8 on, and the output at `28672 + 512j`; the next job at 24568.
/// `run` is `echo` (output: `argv[1]` and a newline) or `cat` (output: the stdin), anything else
/// `-EPERM`; `read_out` drains the output into the region; `status` is 0; `read_err` and `close`
/// answer 0.
fn guest_served() -> String {
    parent(
        &format!(
            "  vap = i64.const 16696
  vas = self.resolve vap vl
  vrlen = i64.const 65536
  vrh = call.cap 5 5 (i64) -> (i64) vas (vrlen)
  vmsg = i32.wrap_i64 vrh
  vwin = i64.const 65536
  vz = i64.const 0
  vprot = i32.const 3
  vmapped = call.cap 4 0 (i64, i64, i64, i32) -> (i64) vmsg (vwin, vz, vrlen, vprot)
  vex = i32.const {tag}
",
            tag = temen_interp::GRANT_SERVE_LIVE_TAG as i32
        ),
        "  br 1(vh, vch)
}
block 1 (vh: i32, vch: i32) {
  vz1 = i32.const 0
  vn = svc.wait vz1
  vzero = i64.const 0
  vgone = i64.eq vn vzero
  br_if vgone 2(vh, vch) 1(vh, vch)
}
block 2 (vh: i32, vch: i32) {
",
        GUEST_SERVICER,
    )
    .replace(
        "data 16684 \"exec.msg\"\n",
        "data 16684 \"exec.msg\"\ndata 16696 \"as\"\n",
    )
}

/// The guest servicer's types, export and handler (func 2); func 1 is unused, so the parent's entry
/// stays function 0.
const GUEST_SERVICER: &str = "\
type 0 func (i64, i64, i64, i64) -> (i64)
type 1 interface { exec: 0 }
export 0 interface \"exec\" 1 { exec: 2 }
func 1 () -> () {
block 0 () {
  return
  }
}
func 2 (i64, i64, i64, i64) -> (i64) {
block 0 (op: i64, a: i64, b: i64, c: i64) {
  z = i64.const 0
  isrun = i64.eq op z
  br_if isrun 1(a, b, c) 6(op, a, b, c)
}
block 1 (off: i64, al: i64, sl: i64) {
  vnj = i64.const 24568
  j = i64.load vnj
  one = i64.const 1
  j1 = i64.add j one
  i64.store vnj j1
  k16 = i64.const 16
  mj = i64.mul j k16
  mb = i64.const 24576
  meta = i64.add mb mj
  k512 = i64.const 512
  oj = i64.mul j k512
  ob = i64.const 28672
  out = i64.add ob oj
  zero = i64.const 0
  i64.store meta zero offset=8
  w = i64.const 65536
  argp = i64.add w off
  c0 = i32.load8_u argp
  ke = i32.const 101
  isecho = i32.eq c0 ke
  br_if isecho 2(j, meta, out, argp, al) 3(j, meta, out, argp, al, sl, c0)
}
block 2 (j: i64, meta: i64, out: i64, argp: i64, al: i64) {
  five = i64.const 5
  src = i64.add argp five
  n = i64.sub al five
  mem.copy out src n
  nlat = i64.add out n
  nl = i32.const 10
  i32.store8 nlat nl
  one = i64.const 1
  len = i64.add n one
  i64.store meta len
  return j
}
block 3 (j: i64, meta: i64, out: i64, argp: i64, al: i64, sl: i64, c0: i32) {
  kc = i32.const 99
  iscat = i32.eq c0 kc
  br_if iscat 4(j, meta, out, argp, al, sl) 5()
}
block 4 (j: i64, meta: i64, out: i64, argp: i64, al: i64, sl: i64) {
  src = i64.add argp al
  mem.copy out src sl
  i64.store meta sl
  return j
}
block 5 () {
  eperm = i64.const -1
  return eperm
}
block 6 (op: i64, a: i64, b: i64, c: i64) {
  one = i64.const 1
  isread = i64.eq op one
  br_if isread 8(a, b, c) 7()
}
block 7 () {
  z = i64.const 0
  return z
}
block 8 (j: i64, off: i64, cap: i64) {
  k16 = i64.const 16
  mj = i64.mul j k16
  mb = i64.const 24576
  meta = i64.add mb mj
  k512 = i64.const 512
  oj = i64.mul j k512
  ob = i64.const 28672
  out = i64.add ob oj
  len = i64.load meta
  pos = i64.load meta offset=8
  rem = i64.sub len pos
  short = i64.lt_s rem cap
  n = select short rem cap
  w = i64.const 65536
  dst = i64.add w off
  src = i64.add out pos
  mem.copy dst src n
  pos2 = i64.add pos n
  i64.store meta pos2 offset=8
  return n
  }
}
";

/// Run `src` on `driver` with `extra` granted beside the spawn set (`vm`, `budget`, `as`, and the
/// compiled `client`), and answer the root's result. On its own thread with a deadline, so a hang
/// fails the test.
fn run(
    driver: Option<Backend>,
    src: &str,
    extra: Vec<(&'static str, HostCap)>,
) -> Result<i32, String> {
    let m = parse_module(src).unwrap_or_else(|e| panic!("parse: {e:?}\n{src}"));
    verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}\n{src}"));
    let client = Arc::new(client());
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let inst = instantiate_with_imports(m, Imports::new()).expect("instantiate");
        let mut caps = vec![
            ("vm", HostCap::custom(6, 0, |h, _| h.grant_instantiator())),
            ("budget", HostCap::detached_budget(64 << 20)),
            (
                "as",
                HostCap::custom(5, 0, |h, win| h.grant_address_space(0, win)),
            ),
            (
                "client",
                HostCap::custom(temen_ir::cap_id::MODULE, 0, move |h, _| {
                    h.grant_module(&client)
                }),
            ),
        ];
        caps.extend(extra);
        let r = match driver {
            Some(b) => inst.run_with_caps(b, &RunConfig::default(), &caps),
            None => inst.run_with_caps_parallel(&RunConfig::default(), &caps),
        };
        let _ = tx.send(r.map(|r| r.outcome).map_err(|e| e.to_string()));
    });
    let outcome = rx
        .recv_timeout(std::time::Duration::from_secs(60))
        .unwrap_or_else(|e| panic!("{driver:?}: the run did not finish: {e}"))?;
    match outcome {
        Outcome::Returned(v) => match v.as_slice() {
            [temen_interp::Value::I32(x)] => Ok(*x),
            other => Err(format!("unexpected result {other:?}")),
        },
        Outcome::Exited(c) => Err(format!("exited {c}")),
    }
}

/// The client gets every byte and status right from a real `echo` and `cat` the host runs.
#[test]
fn a_compiled_client_runs_against_the_host_served_exec() {
    for d in DRIVERS {
        let host = temen_run::exec::host_exec(&["echo", "cat"]);
        assert_eq!(
            run(d, &host_served(), vec![("exec", host)]),
            Ok(1111),
            "{d:?}: echo hi → hi\\n, exit 0; cat ← yo → yo, then EOF"
        );
    }
}

/// The same compiled client, unchanged, gets the same answers from its parent's own code.
#[test]
fn the_same_client_runs_against_a_guest_served_exec() {
    for d in DRIVERS {
        assert_eq!(
            run(d, &guest_served(), Vec::new()),
            Ok(1111),
            "{d:?}: the parent's handler read argv and stdin from the region and wrote the output there"
        );
    }
}
