//! #2962 (universal B) — **one compiled `exec` client, served by the host or by a guest, none the
//! wiser** (EXEC.md's one-interface promise, with buffers).
//!
//! The client is ordinary C built by clang, through #2961's surface: it declares the `exec`
//! interface (`__temen_iface__exec`) and its five ops, which import it as one grouped slot bound by
//! op name. It maps the `exec.msg` message region the wirer granted beside the slot, and its small
//! library bounces argv and stdin into the region before `run` and each output chunk out of it
//! after `read_out`. It runs `echo hi` and `cat` with stdin `yo`, checks every byte and status it
//! gets back, and returns `1111` iff all of it is right.
//!
//! The same compiled module is spawned, detached, by two parents (hand-written IR):
//!
//! - **host-served:** the parent holds a real `host_exec(["echo", "cat"])` and re-grants it, with
//!   the `exec.msg` region its grant brought, to the child by name. The child's grouped import
//!   binds the host handler by the op names it publishes. The host servicer reads and writes the
//!   region's backing and never touches the child's window.
//! - **guest-served:** the parent mints a region, maps it into its own window, and grants it to
//!   the child with a live grant of its own `exec` export, whose five ops carry the same names. Its
//!   `run` copies the call's range out of the region once and parses its copy (the caller can
//!   still write the region meanwhile); `read_out` writes output back to the region. Its serve loop
//!   answers until the child is gone.
//!
//! Each runs on every driver: the tree-walker, the bytecode engine, the Cranelift JIT and the
//! parallel driver (INVARIANTS 14).

#![cfg(target_os = "linux")]

#[path = "../../temen-interp/tests/support/rec.rs"]
mod rec;

use std::path::Path;
use std::process::Command;
use std::sync::Arc;

use temen_ir::{Module, SpawnRec};
use temen_run::{instantiate, Backend, HostCap, Outcome, RunConfig, Value};

/// `exec`'s descriptor, as `temen_exec::EXEC_IFACE` states it.
fn descriptor() -> String {
    let ops: Vec<String> = temen_run::exec::EXEC_IFACE
        .iter()
        .map(|(name, sig)| format!("{name}: {sig}"))
        .collect();
    format!("exec {{ {} }}", ops.join("; "))
}

/// The client: a two-function `exec` library over the message region, and a `main` that checks
/// what two runs give back. Scored so a wrong answer says which part was wrong.
fn client() -> String {
    format!(
        r#"
#define TEMEN_INTERFACE(name, ops) const char __temen_iface__##name[] = ops
#define TEMEN_OP(iface, op) __temen_op__##iface##__##op
int __vm_cap_resolve(const char *name, long len);
long __vm_region_map(int r, long win_off, long region_off, long len, int prot);

TEMEN_INTERFACE(exec, "{desc}");
long TEMEN_OP(exec, run)(long off, long argv_len, long stdin_len);
long TEMEN_OP(exec, read_out)(long job, long off, long cap);
long TEMEN_OP(exec, read_err)(long job, long off, long cap);
long TEMEN_OP(exec, status)(long job);
long TEMEN_OP(exec, close)(long job);

/* A 64 KiB-aligned window range the region maps over. */
static char pages[1 << 17];
static volatile char *msg;

static void put(volatile char *d, const char *s, long n) {{
  for (long i = 0; i < n; i++)
    d[i] = s[i];
}}

static int same(const char *a, const char *b, long n) {{
  for (long i = 0; i < n; i++)
    if (a[i] != b[i])
      return 0;
  return 1;
}}

/* The library: bounce argv and stdin into the region, then run. */
static long xrun(const char *argv, long argv_len, const char *in, long in_len) {{
  put(msg, argv, argv_len);
  put(msg + argv_len, in, in_len);
  return TEMEN_OP(exec, run)(0, argv_len, in_len);
}}

/* Read a chunk of output into the region, then bounce it out. */
static long xread(long job, char *buf, long cap) {{
  long n = TEMEN_OP(exec, read_out)(job, 0, cap);
  for (long i = 0; i < n; i++)
    buf[i] = msg[i];
  return n;
}}

int main(void) {{
  int r = __vm_cap_resolve("exec.msg", 8);
  if (r < 0)
    return 1;
  long at = ((long)pages + 65535) & ~65535L;
  if (__vm_region_map(r, at, 0, 65536, 3) < 0)
    return 2;
  msg = (volatile char *)at;
  char out[32];
  int score = 0;
  long job = xrun("echo\0hi", 7, "", 0);
  if (job < 0)
    return 3;
  long n = xread(job, out, 32);
  if (n == 3 && same(out, "hi\n", 3))
    score += 1;
  if (TEMEN_OP(exec, status)(job) == 0)
    score += 10;
  TEMEN_OP(exec, close)(job);
  job = xrun("cat", 3, "yo", 2);
  if (job < 0)
    return 4;
  n = xread(job, out, 32);
  if (n == 2 && same(out, "yo", 2))
    score += 100;
  if (xread(job, out, 32) == 0)
    score += 1000;
  TEMEN_OP(exec, close)(job);
  return score;
}}
"#,
        desc = descriptor()
    )
}

/// `clang -O2` the client to `.ll` and translate it with the §14 child entry; `None` if clang is
/// unavailable.
fn compiled_client() -> Option<Module> {
    let dir = std::env::temp_dir().join(format!("exec_region_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("work dir");
    let (c, ll) = (dir.join("client.c"), dir.join("client.ll"));
    std::fs::write(&c, client()).expect("write the client");
    let ok = Command::new("clang")
        .args(["-O2", "-emit-llvm", "-S"])
        .arg(&c)
        .arg("-o")
        .arg(&ll)
        .status()
        .is_ok_and(|s| s.success());
    ok.then(|| translate(&ll))
}

fn translate(ll: &Path) -> Module {
    let opts = temen_llvm::TranslateOptions {
        child_entry: true,
        ..Default::default()
    };
    let t = temen_llvm::translate_ll_path_with_options(ll, opts).expect("translate");
    temen_verify::verify_module(&t.module).expect("verify");
    t.module
}

/// A parent that spawns the `"client"` module detached, granting it `exec` and `exec.msg` (the
/// grant list at 16640, the handles in `vex` and `vmsg`, which `setup` defines), then runs `serve`
/// and returns what the client returned. `"exec.msg"` is the data at 16684, whose first four bytes
/// spell `"exec"`. Its window is 8 MiB: the run's own `mem` ceiling follows it, and the 2 MiB
/// client is charged there too. `head` is the servicer's types and export, `funcs` its functions from `func 1`.
fn parent(setup: &str, serve: &str, head: &str, funcs: &str) -> String {
    let rec = SpawnRec {
        grants_ptr: 16640,
        grants_n: 2,
        ..SpawnRec::default()
    };
    format!(
        "\
memory 23
data 16384 \"vm\"
data 16400 \"budget\"
data 16448 \"client\"
data 16684 \"exec.msg\"
data 16696 \"as\"
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
        "",
    )
}

/// The guest-served parent. It mints a 64 KiB region (`AddressSpace` op 5), maps it at 65536 of its
/// own window, grants it with a live grant of its export 0, and serves until `svc.wait` finds the
/// client gone.
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
        SERVICER_HEAD,
        SERVICER_FUNCS,
    )
}

/// The guest servicer's types and export: `exec`'s five ops by name, as the client imports them.
const SERVICER_HEAD: &str = "\
type 0 func (i64, i64, i64) -> (i64)
type 1 func (i64) -> (i64)
type 2 interface { run: 0, read_out: 0, read_err: 0, status: 1, close: 1 }
export 0 interface \"exec\" 2 { run: 2, read_out: 3, read_err: 4, status: 5, close: 5 }
";

/// The guest servicer's functions (func 1 is unused, so the parent's entry stays function 0). Its
/// job table is in its own window: per job `j` a length at `24576 + 16j`, a read cursor 8 on, and
/// the output at `28672 + 512j`; the next job at 24568. `run` (func 2) copies the call's whole
/// range, argv then stdin, out of the region to 20480 once — at most 4000 bytes, `-E2BIG` past
/// that — and parses that copy: `echo` (output: `argv[1]` and a newline) or `cat` (output: the
/// stdin), anything else `-EPERM`. `read_out` (func 3) drains the output into the region;
/// `read_err` (func 4) has nothing; `status` and `close` (func 5) answer 0.
const SERVICER_FUNCS: &str = "\
func 1 () -> () {
block 0 () {
  return
  }
}
func 2 (i64, i64, i64) -> (i64) {
block 0 (off: i64, al: i64, sl: i64) {
  n = i64.add al sl
  cap = i64.const 4000
  big = i64.lt_s cap n
  br_if big 5() 1(off, al, sl, n)
}
block 1 (off: i64, al: i64, sl: i64, n: i64) {
  w = i64.const 65536
  src = i64.add w off
  cp = i64.const 20480
  mem.copy cp src n
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
  c0 = i32.load8_u cp
  ke = i32.const 101
  isecho = i32.eq c0 ke
  br_if isecho 2(j, meta, out, cp, al) 3(j, meta, out, cp, al, sl, c0)
}
block 2 (j: i64, meta: i64, out: i64, cp: i64, al: i64) {
  five = i64.const 5
  src = i64.add cp five
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
block 3 (j: i64, meta: i64, out: i64, cp: i64, al: i64, sl: i64, c0: i32) {
  kc = i32.const 99
  iscat = i32.eq c0 kc
  br_if iscat 4(j, meta, out, cp, al, sl) 6()
}
block 4 (j: i64, meta: i64, out: i64, cp: i64, al: i64, sl: i64) {
  src = i64.add cp al
  mem.copy out src sl
  i64.store meta sl
  return j
}
block 5 () {
  e2big = i64.const -7
  return e2big
}
block 6 () {
  eperm = i64.const -1
  return eperm
  }
}
func 3 (i64, i64, i64) -> (i64) {
block 0 (j: i64, off: i64, cap: i64) {
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
func 4 (i64, i64, i64) -> (i64) {
block 0 (j: i64, off: i64, cap: i64) {
  z = i64.const 0
  return z
  }
}
func 5 (i64) -> (i64) {
block 0 (j: i64) {
  z = i64.const 0
  return z
  }
}
";

const DRIVERS: [Option<Backend>; 4] = [
    Some(Backend::TreeWalk),
    Some(Backend::Bytecode),
    Some(Backend::Jit),
    None,
];

/// Run `src` on `driver` with `extra` granted beside the spawn set (`vm`, `budget`, `as`, and the
/// compiled `client`), and answer the root's result. On its own thread with a deadline, so a hang
/// fails the test.
fn run(
    driver: Option<Backend>,
    src: &str,
    client: &Arc<Module>,
    extra: Vec<(&'static str, HostCap)>,
) -> Result<i32, String> {
    let m = temen_text::parse_module(src).unwrap_or_else(|e| panic!("parse: {e:?}\n{src}"));
    temen_verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}\n{src}"));
    let client = Arc::clone(client);
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let inst = instantiate(m).expect("instantiate");
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
        let _ = tx.send(r.map_err(|e| e.to_string()));
    });
    let r = rx
        .recv_timeout(std::time::Duration::from_secs(60))
        .unwrap_or_else(|e| panic!("{driver:?}: the run did not finish: {e}"))?;
    match r.outcome {
        Outcome::Returned(v) => match v[..] {
            [Value::I32(x)] => Ok(x),
            _ => Err(format!("result {v:?}")),
        },
        o => Err(format!("outcome {o:?}, notes {:?}", r.notes)),
    }
}

/// The compiled client, or `None` (the test skips) without clang.
fn the_client() -> Option<Arc<Module>> {
    let m = compiled_client()?;
    assert!(
        m.imports
            .iter()
            .any(|i| i.name == "exec" && matches!(i.shape, temen_ir::ImportShape::Interface(_))),
        "the client imports `exec` as one grouped slot"
    );
    Some(Arc::new(m))
}

/// The client gets every byte and status right from a real `echo` and `cat` the host runs.
#[test]
fn a_compiled_client_runs_against_the_host_served_exec() {
    let Some(client) = the_client() else {
        eprintln!("skipped: no clang");
        return;
    };
    for d in DRIVERS {
        let host = temen_run::exec::host_exec(&["echo", "cat"]);
        assert_eq!(
            run(d, &host_served(), &client, vec![("exec", host)]),
            Ok(1111),
            "{d:?}: echo hi → hi\\n, exit 0; cat ← yo → yo, then EOF"
        );
    }
}

/// The same compiled client, unchanged, gets the same answers from its parent's own code.
#[test]
fn the_same_client_runs_against_a_guest_served_exec() {
    let Some(client) = the_client() else {
        eprintln!("skipped: no clang");
        return;
    };
    for d in DRIVERS {
        assert_eq!(
            run(d, &guest_served(), &client, Vec::new()),
            Ok(1111),
            "{d:?}: the parent's handlers read argv and stdin from the region and wrote the output there"
        );
    }
}
