//! #2961 — **a compiled guest serves an interface its compiled child calls by name.** A parent built
//! by clang (and by rustc) serves `exec`; its child, built the same way, imports `exec` and calls
//! each op through the slot. The wiring is the substrate's own: the parent's ops are an impl-export,
//! the spawn grants it as a live offer its serve loop answers (`GRANT_SERVE_LIVE_TAG | export`), the
//! child's import binds to that grant by name, and each call is a `call.import` the parent's
//! `__vm_svc_wait` loop serves until the child is gone, before it joins.
//!
//! A handle crosses both ways. `take` answers the parent's own `SharedRegion` in a result (beside an
//! integer: a two-result op), and the child maps it and writes a byte into it. `give` takes the
//! child's handle back in an argument; the parent maps what it received and reads the byte, so the
//! handle named the same region in all three tables. A correct run returns `join * 10 + served`
//! with the child's `42 * 10000 + 7 * 1000 + 'C'` and three calls served: 4_270_673, on the
//! tree-walker, the bytecode engine, the Cranelift JIT, and the bytecode engine's parallel driver.

#![cfg(target_os = "linux")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use temen_run::{Backend, HostCap, Outcome, RunConfig, Value};

/// What the parent answers on every driver.
const ANSWER: i32 = (42 * 10_000 + 7 * 1000 + b'C' as i32) * 10 + 3;

/// The interface's ops, as both sides declare them.
const OPS: &str = "run: (i64, i64) -> (i64); take: (i64) -> (i64, cap); give: (cap) -> (i64)";

/// The descriptor naming the interface `name`.
fn descriptor(name: &str) -> String {
    format!("{name} {{ {OPS} }}")
}

/// The source surface (#2961), as a C header would carry it.
const C_HEADER: &str = r#"
#define TEMEN_INTERFACE(name, ops) const char __temen_iface__##name[] = ops
#define TEMEN_OP(iface, op) __temen_op__##iface##__##op
#define TEMEN_SERVE_GRANT(name) __vm_serve_grant(__temen_iface__##name)
long __vm_svc_wait(void);
int __vm_serve_grant(const char *descriptor);
int __vm_cap_resolve(const char *name, long len);
long __vm_instantiate_rec(int inst, long rec);
long __vm_join(int inst, long child, long *status);
long __vm_region_create(long len);
long __vm_region_map(int r, long win_off, long region_off, long len, int prot);
long __vm_region_unmap(int r, long win_off, long len);
long __vm_region_page_size(int r);
struct took { long v; int h; };
/* A page-aligned window range a region maps over (the region's page is at most 64 KiB). */
static char pages[2][1 << 17];
static long page_of(int i, long g) { return ((long)pages[i] + g - 1) & ~(g - 1); }
"#;

fn c_parent() -> String {
    format!(
        r#"{C_HEADER}
TEMEN_INTERFACE(exec, "{desc}");
static int region;

long TEMEN_OP(exec, run)(long a, long b) {{ return a + b; }}

struct took TEMEN_OP(exec, take)(long x) {{
  struct took t = {{ x + 1, region }};
  return t;
}}

long TEMEN_OP(exec, give)(int h) {{
  long g = __vm_region_page_size(h);
  long at = page_of(1, g);
  if (g <= 0 || __vm_region_map(h, at, 0, g, 3) < 0) return -1;
  long c = *(volatile char *)at;
  __vm_region_unmap(h, at, g);
  return c;
}}

/* The op-17 spawn record (`temen_ir::SpawnRec`, u32 words) and one grant record. */
static unsigned rec[16] __attribute__((aligned(8)));
static unsigned grant[4];

int main(void) {{
  int inst = __vm_cap_resolve("vm", 2);
  int budget = __vm_cap_resolve("budget", 6);
  int child = __vm_cap_resolve("child", 5);
  region = (int)__vm_region_create(1 << 16);
  if (inst < 0 || budget < 0 || child < 0 || region < 0) return -1;
  grant[0] = (unsigned)(long)"exec";
  grant[1] = 4;
  grant[2] = (unsigned)TEMEN_SERVE_GRANT(exec);
  grant[3] = 0;
  unsigned words[16] = {{0, ~0u, (unsigned)child, (unsigned)budget, (unsigned)(long)grant, 0, 1, 0,
                        0, 0, 0, 0, ~0u, 0, 0, 0}};
  for (int i = 0; i < 16; i++) rec[i] = words[i];
  long h = __vm_instantiate_rec(inst, (long)rec);
  if (h < 0) return (int)h;
  long served = 0, n;
  while ((n = __vm_svc_wait()) > 0) served += n;
  long st = 0;
  long r = __vm_join(inst, h, &st);
  if (st != 0) return (int)(-1000 - st);
  return (int)(r * 10 + served);
}}
"#,
        desc = descriptor("exec")
    )
}

fn c_child() -> String {
    format!(
        r#"{C_HEADER}
TEMEN_INTERFACE(exec, "{desc}");
long TEMEN_OP(exec, run)(long a, long b);
struct took TEMEN_OP(exec, take)(long x);
long TEMEN_OP(exec, give)(int h);

int main(void) {{
  long r = TEMEN_OP(exec, run)(40, 2);
  struct took t = TEMEN_OP(exec, take)(6);
  long g = __vm_region_page_size(t.h);
  long at = page_of(0, g);
  if (g <= 0 || __vm_region_map(t.h, at, 0, g, 3) < 0) return -2;
  *(volatile char *)at = 'C';
  long c = TEMEN_OP(exec, give)(t.h);
  return (int)(r * 10000 + t.v * 1000 + c);
}}
"#,
        desc = descriptor("exec")
    )
}

/// The Rust guests' shared prelude: `no_std`, the descriptor under the tag `exec`, naming the
/// interface `exec.host` (a dotted name, which only the descriptor carries), and the builtins both
/// sides call.
fn rust_prelude() -> String {
    let desc = descriptor(RUST_NAME);
    format!(
        r##"#![no_std]
#![allow(internal_features, non_upper_case_globals, static_mut_refs)]
#[panic_handler]
fn ph(_: &core::panic::PanicInfo) -> ! {{ loop {{}} }}
#[no_mangle]
pub extern "C" fn rust_eh_personality() {{}}

#[no_mangle]
pub static __temen_iface__exec: [u8; {n}] = *b"{desc}";

#[repr(C)]
pub struct Took {{ v: i64, h: i32 }}

extern "C" {{
    fn __vm_svc_wait() -> i64;
    fn __vm_serve_grant(descriptor: *const u8) -> i32;
    fn __vm_cap_resolve(name: *const u8, len: i64) -> i32;
    fn __vm_join(inst: i32, child: i64, status: *mut i64) -> i64;
    fn __vm_region_create(len: i64) -> i64;
    fn __vm_region_map(r: i32, win_off: i64, region_off: i64, len: i64, prot: i32) -> i64;
    fn __vm_region_unmap(r: i32, win_off: i64, len: i64) -> i64;
    fn __vm_region_page_size(r: i32) -> i64;
}}

#[repr(C, align(65536))]
struct Pages([[u8; 1 << 16]; 2]);
static mut PAGES: Pages = Pages([[0; 1 << 16]; 2]);
unsafe fn page(i: usize) -> i64 {{ core::ptr::addr_of_mut!(PAGES.0[i]) as i64 }}
"##,
        n = desc.len()
    )
}

const RUST_PARENT: &str = r##"
static mut REGION: i32 = -1;

#[export_name = "__temen_op__exec__run"]
pub extern "C" fn run(a: i64, b: i64) -> i64 { a + b }

#[export_name = "__temen_op__exec__take"]
pub extern "C" fn take(x: i64) -> Took { Took { v: x + 1, h: unsafe { REGION } } }

#[export_name = "__temen_op__exec__give"]
pub extern "C" fn give(h: i32) -> i64 {
    unsafe {
        let g = __vm_region_page_size(h);
        let at = page(1);
        if g <= 0 || __vm_region_map(h, at, 0, g, 3) < 0 {
            return -1;
        }
        let c = core::ptr::read_volatile(at as *const u8) as i64;
        __vm_region_unmap(h, at, g);
        c
    }
}

#[no_mangle]
pub extern "C" fn main() -> i32 {
    unsafe {
        let inst = __vm_cap_resolve(b"vm".as_ptr(), 2);
        let budget = __vm_cap_resolve(b"budget".as_ptr(), 6);
        let child = __vm_cap_resolve(b"child".as_ptr(), 5);
        REGION = __vm_region_create(1 << 16) as i32;
        if inst < 0 || budget < 0 || child < 0 || REGION < 0 {
            return -1;
        }
        let exec = __vm_serve_grant(__temen_iface__exec.as_ptr());
        let h = vm_spawn(inst, budget, child, &[(b"exec.host", exec)], &[]);
        if h < 0 {
            return h as i32;
        }
        let mut served = 0;
        loop {
            let n = __vm_svc_wait();
            if n <= 0 {
                break;
            }
            served += n;
        }
        let mut st = 0i64;
        let r = __vm_join(inst, h, &mut st);
        if st != 0 {
            return (-1000 - st) as i32;
        }
        (r * 10 + served) as i32
    }
}
"##;

const RUST_CHILD: &str = r##"
extern "C" {
    #[link_name = "__temen_op__exec__run"]
    fn exec_run(a: i64, b: i64) -> i64;
    #[link_name = "__temen_op__exec__take"]
    fn exec_take(x: i64) -> Took;
    #[link_name = "__temen_op__exec__give"]
    fn exec_give(h: i32) -> i64;
}

#[no_mangle]
pub extern "C" fn main() -> i32 {
    unsafe {
        let r = exec_run(40, 2);
        let t = exec_take(6);
        let g = __vm_region_page_size(t.h);
        let at = page(0);
        if g <= 0 || __vm_region_map(t.h, at, 0, g, 3) < 0 {
            return -2;
        }
        core::ptr::write_volatile(at as *mut u8, b'C');
        let c = exec_give(t.h);
        (r * 10000 + t.v * 1000 + c) as i32
    }
}
"##;

/// The shared guest-side spawn (`support/guest_vm_spawn.rs`).
const VM_SPAWN: &str = include_str!("support/guest_vm_spawn.rs");

/// `rustc -O --emit=llvm-ir` a `no_std` cdylib to textual `.ll`; `None` if rustc is unavailable.
fn rustc_ll(dir: &Path, name: &str, src: &str) -> Option<PathBuf> {
    let rs = dir.join(format!("{name}.rs"));
    let ll = dir.join(format!("{name}.ll"));
    std::fs::write(&rs, src).expect("write Rust source");
    let ok = Command::new("rustc")
        .args([
            "--edition",
            "2021",
            "-O",
            "-Cpanic=abort",
            "--emit=llvm-ir",
            "--crate-type=cdylib",
        ])
        .arg(&rs)
        .arg("-o")
        .arg(&ll)
        .status()
        .is_ok_and(|s| s.success());
    ok.then_some(ll)
}

fn work_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("serve_iface_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&d).expect("work dir");
    d
}

/// `clang -O2` a C source to textual `.ll`; `None` if clang is unavailable.
fn clang_ll(dir: &Path, name: &str, src: &str) -> Option<PathBuf> {
    let c = dir.join(format!("{name}.c"));
    let ll = dir.join(format!("{name}.ll"));
    std::fs::write(&c, src).expect("write C source");
    let ok = Command::new("clang")
        .args(["-O2", "-emit-llvm", "-S"])
        .arg(&c)
        .arg("-o")
        .arg(&ll)
        .status()
        .is_ok_and(|s| s.success());
    ok.then_some(ll)
}

fn translate(ll: &Path, child_entry: bool) -> temen_ir::Module {
    let opts = temen_llvm::TranslateOptions {
        child_entry,
        ..Default::default()
    };
    let t = temen_llvm::translate_ll_path_with_options(ll, opts).expect("translate");
    temen_verify::verify_module(&t.module).expect("verify");
    t.module
}

/// Run `parent` with `child` granted as `"child"`, on `backend` (`None`: the parallel driver).
/// Run `parent` with each `children` module granted under its name, on `backend` (`None`: the
/// parallel driver): its result and its stdout.
fn run(
    parent: &temen_ir::Module,
    children: &[(&'static str, Arc<temen_ir::Module>)],
    backend: Option<Backend>,
) -> Result<(i32, Vec<u8>), String> {
    let inst = temen_run::instantiate(parent.clone()).map_err(|e| e.to_string())?;
    let mut caps = vec![
        ("vm", HostCap::custom(6, 0, |h, _| h.grant_instantiator())),
        ("budget", HostCap::detached_budget(64 << 20)),
    ];
    for (name, child) in children {
        let child = Arc::clone(child);
        caps.push((
            name,
            HostCap::custom(temen_ir::cap_id::MODULE, 0, move |h, _| {
                h.grant_module(&child)
            }),
        ));
    }
    let cfg = RunConfig::default();
    let r = match backend {
        Some(b) => inst.run_with_caps(b, &cfg, &caps),
        None => inst.run_with_caps_parallel(&cfg, &caps),
    }
    .map_err(|e| e.to_string())?;
    match r.outcome {
        Outcome::Returned(v) => match v[..] {
            [Value::I32(x)] => Ok((x, r.stdout)),
            _ => Err(format!("result {v:?}")),
        },
        o => Err(format!("outcome {o:?}, notes {:?}", r.notes)),
    }
}

const DRIVERS: [Option<Backend>; 4] = [
    Some(Backend::TreeWalk),
    Some(Backend::Bytecode),
    Some(Backend::Jit),
    None,
];

/// The Rust lane's interface name.
const RUST_NAME: &str = "exec.host";

fn check(name: &str, parent: temen_ir::Module, child: temen_ir::Module) {
    // The parent serves `exec` and imports nothing by it; the child imports it whole.
    assert_eq!(
        parent.impl_exports.len(),
        1,
        "the parent serves one interface"
    );
    assert_eq!(parent.impl_exports[0].name, name);
    assert!(
        child
            .imports
            .iter()
            .any(|i| i.name == name && matches!(i.shape, temen_ir::ImportShape::Interface(_))),
        "the child imports `exec` as one grouped slot"
    );
    assert!(
        temen_interp::bytecode::admits(&parent),
        "the bytecode engine serves the parent itself: no handler can park"
    );
    let child = Arc::new(child);
    for b in DRIVERS {
        let r = run(&parent, &[("child", Arc::clone(&child))], b).map(|(x, _)| x);
        assert_eq!(r, Ok(ANSWER), "{b:?}");
    }
}

#[test]
fn a_clang_parent_serves_its_clang_childs_calls() {
    let dir = work_dir("c");
    let (Some(p), Some(c)) = (
        clang_ll(&dir, "parent", &c_parent()),
        clang_ll(&dir, "child", &c_child()),
    ) else {
        eprintln!("note: skipping (clang unavailable)");
        return;
    };
    check("exec", translate(&p, false), translate(&c, true));
}

#[test]
fn a_rustc_parent_serves_its_rustc_childs_calls() {
    let dir = work_dir("rust");
    let prelude = rust_prelude();
    let (Some(p), Some(c)) = (
        rustc_ll(&dir, "parent", &format!("{prelude}{RUST_PARENT}{VM_SPAWN}")),
        rustc_ll(&dir, "child", &format!("{prelude}{RUST_CHILD}")),
    ) else {
        eprintln!("note: skipping (rustc unavailable)");
        return;
    };
    check(RUST_NAME, translate(&p, false), translate(&c, true));
}

/// What the translator refuses, each with the reason in its error: a descriptor's ops half defined,
/// an op symbol no descriptor names, an op whose source signature is not its interface's, and a
/// grant of an interface the module does not serve.
#[test]
fn the_translator_refuses_what_it_cannot_serve_or_import() {
    const DESC: &str = "calc { add: (i64, i64) -> (i64); neg: (i64) -> (i64) }";
    let desc = format!(
        "@__temen_iface__calc = constant [{} x i8] c\"{DESC}\\00\"",
        DESC.len() + 1
    );
    let refused =
        |body: &str, why: &str| match temen_llvm::translate_ll_str(&format!("{desc}\n{body}")) {
            Err(temen_llvm::Error::Unsupported(e)) => {
                assert!(e.contains(why), "refused for another reason: {e}")
            }
            other => panic!("expected a refusal naming `{why}`, got {other:?}"),
        };
    let main = "define i32 @main() {\n  ret i32 0\n}\n";
    refused(
        &format!(
            "define i64 @__temen_op__calc__add(i64 %a, i64 %b) {{\n  %s = add i64 %a, %b\n  ret i64 %s\n}}\n\
             declare i64 @__temen_op__calc__neg(i64)\n{main}"
        ),
        "1 of its 2 ops are defined here",
    );
    refused(
        &format!("declare i64 @__temen_op__calc__mul(i64, i64)\n{main}"),
        "names no op of any interface descriptor",
    );
    refused(
        &format!("declare i32 @__temen_op__calc__neg(i32)\n{main}"),
        "its interface says",
    );
    refused(
        "declare i32 @__vm_serve_grant(ptr)\n\
         define i32 @main() {\n  %g = call i32 @__vm_serve_grant(ptr @__temen_iface__calc)\n  ret i32 %g\n}\n",
        "is not an interface this module serves",
    );
}

// ---- #2964: served ops that park, allocate and do I/O ------------------------------------------

/// The `jobs` interface: `log` allocates and prints; `wait` parks until `open` releases it, holding
/// live locals on its data stack across the park; `ready` says whether a `wait` is parked.
const JOBS: &str = "jobs { log: (i64) -> (i64); wait: (i64) -> (i64); open: (i64) -> (i64); \
                    ready: () -> (i64) }";

const JOBS_HEADER: &str = r#"
#define TEMEN_INTERFACE(name, ops) const char __temen_iface__##name[] = ops
#define TEMEN_OP(iface, op) __temen_op__##iface##__##op
#define TEMEN_SERVE_GRANT(name) __vm_serve_grant(__temen_iface__##name)
long __vm_svc_wait(void);
int __vm_serve_grant(const char *descriptor);
int __vm_cap_resolve(const char *name, long len);
long __vm_instantiate_rec(int inst, long rec);
long __vm_join(int inst, long child, long *status);
"#;

fn c_jobs_parent() -> String {
    format!(
        r#"{JOBS_HEADER}
#include <stdio.h>
#include <stdlib.h>
int __vm_wait32(int *addr, int expected, long timeout_ns);
int __vm_notify(int *addr, int count);
int __vm_atomic_load32(int *addr);
void __vm_atomic_store32(int *addr, int v);
TEMEN_INTERFACE(jobs, "{JOBS}");

static int gate, waiting;

long TEMEN_OP(jobs, log)(long x) {{
  long *cell = malloc(sizeof(long) * 4);
  if (!cell) return -1;
  cell[2] = x * 2;
  printf("log %ld\n", x);
  long r = cell[2];
  free(cell);
  return r;
}}

/* Fill a frame of locals, park until `open`, then sum them: a handler run meanwhile on the same
   stack would have overwritten them. */
long TEMEN_OP(jobs, wait)(long x) {{
  volatile long a[64];
  for (int i = 0; i < 64; i++) a[i] = x * 1000 + i;
  __vm_atomic_store32(&waiting, 1);
  while (__vm_atomic_load32(&gate) == 0) __vm_wait32(&gate, 0, -1);
  long sum = 0;
  for (int i = 0; i < 64; i++) sum += a[i];
  return sum + __vm_atomic_load32(&gate);
}}

long TEMEN_OP(jobs, open)(long x) {{
  volatile long b[64];
  for (int i = 0; i < 64; i++) b[i] = -1;
  long s = 0;
  for (int i = 0; i < 64; i++) s += b[i];
  __vm_atomic_store32(&gate, (int)x);
  __vm_notify(&gate, 1);
  return x + s + 64;
}}

long TEMEN_OP(jobs, ready)(void) {{ return __vm_atomic_load32(&waiting); }}

static unsigned rec[2][16] __attribute__((aligned(8)));
static unsigned grant[4];

static long spawn(int inst, int budget, int module, int k) {{
  unsigned words[16] = {{0, ~0u, (unsigned)module, (unsigned)budget, (unsigned)(long)grant, 0, 1,
                        0, 0, 0, 0, 0, ~0u, 0, 0, 0}};
  for (int i = 0; i < 16; i++) rec[k][i] = words[i];
  return __vm_instantiate_rec(inst, (long)rec[k]);
}}

int main(void) {{
  int inst = __vm_cap_resolve("vm", 2);
  int budget = __vm_cap_resolve("budget", 6);
  int a = __vm_cap_resolve("waiter", 6);
  int b = __vm_cap_resolve("opener", 6);
  if (inst < 0 || budget < 0 || a < 0 || b < 0) return -1;
  grant[0] = (unsigned)(long)"jobs";
  grant[1] = 4;
  grant[2] = (unsigned)TEMEN_SERVE_GRANT(jobs);
  grant[3] = 0;
  long ha = spawn(inst, budget, a, 0);
  long hb = spawn(inst, budget, b, 1);
  if (ha < 0 || hb < 0) return -2;
  while (__vm_svc_wait() > 0) {{}}
  long st = 0;
  long ra = __vm_join(inst, ha, &st);
  if (st != 0) return (int)(-1000 - st);
  long rb = __vm_join(inst, hb, &st);
  if (st != 0) return (int)(-2000 - st);
  return (int)(ra * 10000 + rb);
}}
"#
    )
}

fn c_jobs_child(body: &str) -> String {
    format!(
        r#"{JOBS_HEADER}
TEMEN_INTERFACE(jobs, "{JOBS}");
long TEMEN_OP(jobs, log)(long x);
long TEMEN_OP(jobs, wait)(long x);
long TEMEN_OP(jobs, open)(long x);
long TEMEN_OP(jobs, ready)(void);
int main(void) {{ {body} }}
"#
    )
}

/// The waiter's `wait(1)`: its 64 locals `1000 + i` summed, plus the gate `open` set (7).
const WAITED: i32 = 64 * 1000 + 63 * 64 / 2 + 7;
/// The opener's `log(21) * 100 + open(7)`: the waiter's answer, then the opener's.
const OPENED: i32 = 42 * 100 + 7;

#[test]
fn a_clang_parent_serves_ops_that_park_allocate_and_print() {
    let dir = work_dir("jobs");
    let waiter = c_jobs_child("return (int)TEMEN_OP(jobs, wait)(1);");
    let opener = c_jobs_child(
        "long l = TEMEN_OP(jobs, log)(21);\n  while (TEMEN_OP(jobs, ready)() == 0) {}\n  \
         return (int)(l * 100 + TEMEN_OP(jobs, open)(7));",
    );
    let (Some(p), Some(w), Some(o)) = (
        clang_ll(&dir, "parent", &c_jobs_parent()),
        clang_ll(&dir, "waiter", &waiter),
        clang_ll(&dir, "opener", &opener),
    ) else {
        eprintln!("note: skipping (clang unavailable)");
        return;
    };
    let parent = translate(&p, false);
    // `wait` parks, so every op takes a data stack of its own from the heap.
    assert!(
        !temen_interp::bytecode::admits(&parent),
        "a server whose handlers park runs on the tree-walker, as hand-written IR does"
    );
    let children = [
        ("waiter", Arc::new(translate(&w, true))),
        ("opener", Arc::new(translate(&o, true))),
    ];
    // The parallel driver's entry pins the bytecode engine, so it declines the module outright where
    // a cooperative bytecode run declines it to the oracle (INVARIANTS 9).
    let r = run(&parent, &children, None);
    assert!(
        matches!(&r, Err(e) if e.contains("outside the parallel engine's subset")),
        "the parallel driver declines it: {r:?}"
    );
    for b in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        let r = run(&parent, &children, Some(b));
        assert_eq!(
            r,
            Ok((WAITED * 10_000 + OPENED, b"log 21\n".to_vec())),
            "{b:?}"
        );
    }
}

/// [`c_jobs_parent`] in Rust: the descriptor, then the same four ops and the same spawn-and-serve.
fn rust_jobs_parent() -> String {
    format!(
        r##"#![no_std]
#![allow(internal_features, non_upper_case_globals, static_mut_refs)]
#[panic_handler]
fn ph(_: &core::panic::PanicInfo) -> ! {{ loop {{}} }}
#[no_mangle]
pub extern "C" fn rust_eh_personality() {{}}

#[no_mangle]
pub static __temen_iface__jobs: [u8; {n}] = *b"{JOBS}";

extern "C" {{
    fn __vm_svc_wait() -> i64;
    fn __vm_serve_grant(descriptor: *const u8) -> i32;
    fn __vm_cap_resolve(name: *const u8, len: i64) -> i32;
    fn __vm_join(inst: i32, child: i64, status: *mut i64) -> i64;
    fn __vm_wait32(addr: *mut i32, expected: i32, timeout_ns: i64) -> i32;
    fn __vm_notify(addr: *mut i32, count: i32) -> i32;
    fn __vm_atomic_load32(addr: *mut i32) -> i32;
    fn __vm_atomic_store32(addr: *mut i32, v: i32);
    fn malloc(n: usize) -> *mut i64;
    fn free(p: *mut i64);
    fn printf(fmt: *const u8, ...) -> i32;
}}

static mut GATE: i32 = 0;
static mut WAITING: i32 = 0;

#[export_name = "__temen_op__jobs__log"]
pub extern "C" fn log(x: i64) -> i64 {{
    unsafe {{
        let cell = malloc(32);
        if cell.is_null() {{
            return -1;
        }}
        core::ptr::write_volatile(cell.add(2), x * 2);
        printf(b"log %ld\n\0".as_ptr(), x);
        let r = core::ptr::read_volatile(cell.add(2));
        free(cell);
        r
    }}
}}

#[export_name = "__temen_op__jobs__wait"]
pub extern "C" fn wait(x: i64) -> i64 {{
    unsafe {{
        let mut a = [0i64; 64];
        for i in 0..64 {{
            core::ptr::write_volatile(&mut a[i], x * 1000 + i as i64);
        }}
        __vm_atomic_store32(core::ptr::addr_of_mut!(WAITING), 1);
        while __vm_atomic_load32(core::ptr::addr_of_mut!(GATE)) == 0 {{
            __vm_wait32(core::ptr::addr_of_mut!(GATE), 0, -1);
        }}
        let mut sum = 0;
        for i in 0..64 {{
            sum += core::ptr::read_volatile(&a[i]);
        }}
        sum + __vm_atomic_load32(core::ptr::addr_of_mut!(GATE)) as i64
    }}
}}

#[export_name = "__temen_op__jobs__open"]
pub extern "C" fn open(x: i64) -> i64 {{
    unsafe {{
        let mut b = [0i64; 64];
        for i in 0..64 {{
            core::ptr::write_volatile(&mut b[i], -1);
        }}
        let mut s = 0;
        for i in 0..64 {{
            s += core::ptr::read_volatile(&b[i]);
        }}
        __vm_atomic_store32(core::ptr::addr_of_mut!(GATE), x as i32);
        __vm_notify(core::ptr::addr_of_mut!(GATE), 1);
        x + s + 64
    }}
}}

#[export_name = "__temen_op__jobs__ready"]
pub extern "C" fn ready() -> i64 {{
    unsafe {{ __vm_atomic_load32(core::ptr::addr_of_mut!(WAITING)) as i64 }}
}}

#[no_mangle]
pub extern "C" fn main() -> i32 {{
    unsafe {{
        let inst = __vm_cap_resolve(b"vm".as_ptr(), 2);
        let budget = __vm_cap_resolve(b"budget".as_ptr(), 6);
        let a = __vm_cap_resolve(b"waiter".as_ptr(), 6);
        let b = __vm_cap_resolve(b"opener".as_ptr(), 6);
        if inst < 0 || budget < 0 || a < 0 || b < 0 {{
            return -1;
        }}
        let jobs = __vm_serve_grant(__temen_iface__jobs.as_ptr());
        let ha = vm_spawn(inst, budget, a, &[(b"jobs", jobs)], &[]);
        let hb = vm_spawn(inst, budget, b, &[(b"jobs", jobs)], &[]);
        if ha < 0 || hb < 0 {{
            return -2;
        }}
        while __vm_svc_wait() > 0 {{}}
        let mut st = 0i64;
        let ra = __vm_join(inst, ha, &mut st);
        if st != 0 {{
            return (-1000 - st) as i32;
        }}
        let rb = __vm_join(inst, hb, &mut st);
        if st != 0 {{
            return (-2000 - st) as i32;
        }}
        (ra * 10000 + rb) as i32
    }}
}}
{VM_SPAWN}
"##,
        n = JOBS.len()
    )
}

/// A Rust child of [`rust_jobs_parent`] whose `main` runs `body`.
fn rust_jobs_child(body: &str) -> String {
    format!(
        r##"#![no_std]
#![allow(internal_features)]
#[panic_handler]
fn ph(_: &core::panic::PanicInfo) -> ! {{ loop {{}} }}
#[no_mangle]
pub extern "C" fn rust_eh_personality() {{}}

#[no_mangle]
pub static __temen_iface__jobs: [u8; {n}] = *b"{JOBS}";

extern "C" {{
    #[link_name = "__temen_op__jobs__log"]
    fn log(x: i64) -> i64;
    #[link_name = "__temen_op__jobs__wait"]
    fn wait(x: i64) -> i64;
    #[link_name = "__temen_op__jobs__open"]
    fn open(x: i64) -> i64;
    #[link_name = "__temen_op__jobs__ready"]
    fn ready() -> i64;
}}

#[no_mangle]
pub extern "C" fn main() -> i32 {{
    unsafe {{ {body} }}
}}
"##,
        n = JOBS.len()
    )
}

#[test]
fn a_rustc_parent_serves_ops_that_park_allocate_and_print() {
    let dir = work_dir("rust_jobs");
    let waiter = rust_jobs_child("wait(1) as i32");
    let opener =
        rust_jobs_child("let l = log(21); while ready() == 0 {} (l * 100 + open(7)) as i32");
    let (Some(p), Some(w), Some(o)) = (
        rustc_ll(&dir, "parent", &rust_jobs_parent()),
        rustc_ll(&dir, "waiter", &waiter),
        rustc_ll(&dir, "opener", &opener),
    ) else {
        eprintln!("note: skipping (rustc unavailable)");
        return;
    };
    let parent = translate(&p, false);
    let children = [
        ("waiter", Arc::new(translate(&w, true))),
        ("opener", Arc::new(translate(&o, true))),
    ];
    for b in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        let r = run(&parent, &children, Some(b));
        assert_eq!(
            r,
            Ok((WAITED * 10_000 + OPENED, b"log 21\n".to_vec())),
            "{b:?}"
        );
    }
}

/// #2964 — a dispatch that gets no data stack is answered, and the servicer lives. The parent
/// defines its own `malloc` (a bump arena the translator's trampolines use like any other), which
/// `deny` turns off; the child's next calls then get INVARIANTS 8's transport failure through
/// `svc.fail`: `twice` its `-ENOMEM`, and `grab`, which answers a handle, `(-ENOMEM, 0)` with no
/// handle of the servicer's granted. `nap` can park, so every op takes a stack of its own.
#[test]
fn a_dispatch_without_a_stack_is_answered_enomem() {
    const MEM: &str = "mem { deny: () -> (i64); twice: (i64) -> (i64); nap: () -> (i64); \
                       grab: () -> (i64, cap) }";
    let parent = format!(
        r#"{JOBS_HEADER}
int __vm_wait32(int *addr, int expected, long timeout_ns);
TEMEN_INTERFACE(mem, "{MEM}");
static char arena[4 << 20] __attribute__((aligned(16)));
static long used;
static int denied;
void *malloc(unsigned long n) {{
  if (denied || used + (long)n > (long)sizeof arena) return 0;
  void *p = arena + used;
  used += (n + 15) & ~15L;
  return p;
}}
void free(void *p) {{ (void)p; }}
static int never = 1;
long TEMEN_OP(mem, deny)(void) {{ denied = 1; return 0; }}
long TEMEN_OP(mem, twice)(long x) {{ return 2 * x; }}
long TEMEN_OP(mem, nap)(void) {{ return __vm_wait32(&never, 0, 0); }}
struct grabbed {{ long v; int h; }};
struct grabbed TEMEN_OP(mem, grab)(void) {{ struct grabbed g = {{ 7, 0 }}; return g; }}
static unsigned rec[16] __attribute__((aligned(8)));
static unsigned grant[4];
int main(void) {{
  int inst = __vm_cap_resolve("vm", 2);
  int budget = __vm_cap_resolve("budget", 6);
  int child = __vm_cap_resolve("child", 5);
  grant[0] = (unsigned)(long)"mem";
  grant[1] = 3;
  grant[2] = (unsigned)TEMEN_SERVE_GRANT(mem);
  unsigned words[16] = {{0, ~0u, (unsigned)child, (unsigned)budget, (unsigned)(long)grant, 0, 1, 0,
                        0, 0, 0, 0, ~0u, 0, 0, 0}};
  for (int i = 0; i < 16; i++) rec[i] = words[i];
  long h = __vm_instantiate_rec(inst, (long)rec);
  if (h < 0) return -1;
  while (__vm_svc_wait() > 0) {{}}
  long st = 0;
  long r = __vm_join(inst, h, &st);
  return st != 0 ? (int)(-1000 - st) : (int)r;
}}
"#
    );
    let child = format!(
        r#"{JOBS_HEADER}
TEMEN_INTERFACE(mem, "{MEM}");
long TEMEN_OP(mem, deny)(void);
long TEMEN_OP(mem, twice)(long x);
long TEMEN_OP(mem, nap)(void);
struct grabbed {{ long v; int h; }};
struct grabbed TEMEN_OP(mem, grab)(void);
int main(void) {{
  long a = TEMEN_OP(mem, twice)(21);
  TEMEN_OP(mem, deny)();
  long b = TEMEN_OP(mem, twice)(21);
  struct grabbed g = TEMEN_OP(mem, grab)();
  return (int)(a * 1000 + b + g.v * 100000 + g.h);
}}
"#
    );
    let dir = work_dir("enomem");
    let (Some(p), Some(c)) = (
        clang_ll(&dir, "parent", &parent),
        clang_ll(&dir, "child", &child),
    ) else {
        eprintln!("note: skipping (clang unavailable)");
        return;
    };
    let parent = translate(&p, false);
    let children = [("child", Arc::new(translate(&c, true)))];
    for b in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        let r = run(&parent, &children, Some(b)).map(|(x, _)| x);
        assert_eq!(
            r,
            Ok(42 * 1000 - 12 - 12 * 100_000),
            "{b:?}: twice(21), then -ENOMEM, and grab's (-ENOMEM, 0) though it answers a handle"
        );
    }
}
