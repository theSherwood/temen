//! **#1025 — run-in-guest: a Rust guest mints a module from its own bytes, then op-13-spawns it.**
//!
//! The endgame of moving the nim compiler driver *into* the sandbox is that the driver runs the program
//! it produced. `rust_guest_op13` proved a Rust-on-Temen guest can op-13-spawn a child — but the child
//! there is a module the **host** granted (`grant_module` + `__vm_cap_resolve("child")`). A driver that
//! *linked* its program (slice 3c step 12, in-guest) holds it as **bytes in its own memory**, not a
//! host grant. This proves the primitive that bridges the two: `__vm_module_from_bytes` (the
//! `ModuleLoader` capability, iface 7) has the host decode+verify the guest's bytes into a `Module`
//! handle the guest then spawns exactly like a host-granted one.
//!
//! It is `rust_guest_op13` with one line changed in spirit: instead of resolving a host-granted child
//! module by name, the guest calls `from_bytes` over the child's encoded module (embedded in the guest's
//! own data segment — where a linker's output would live) and spawns *that* handle. Same counter child,
//! same op-13 grant list, same join, and — crucially — run on **both engines** (invariant 9/14: the
//! primitive lands identically on the tree-walker and the Cranelift JIT). A correct run returns `1` and
//! the re-granted `fs` counter ticks once, on each engine.
//!
//! Gated to Linux + a present `rustc` (like the other on-ramp guest tests); skips cleanly otherwise.

#![cfg(target_os = "linux")]

use core::ffi::c_void;
use std::sync::{Arc, Mutex};
use temen_interp::{run_capture_reserved_with_host, ForkedProc, Host, HostProc, Value};
use temen_jit::{compile_and_run_capture_reserved_with_host_ex, GrantChildHooks, JitOutcome};

// The child (identical to `rust_guest_op13`'s): its `Instantiator` arrives as `v0` (unused). It seeds
// the name `"fs"` (`0x7366` little-endian) into its own window, resolves it, and calls the granted
// `HOST_PROC` counter (type 13, op 0) — post-increment `1`. `memory 17` matches the 128 KiB carve.
const CHILD: &str = r#"memory 17
func (i64) -> (i64) {
block 0 (v0: i64) {
  vname = i64.const 29542
  vzero = i64.const 16384
  i64.store vzero vname
  vp0 = i64.const 16384
  vl2 = i64.const 2
  vh = self.resolve vp0 vl2
  vr = call.cap 13 0 (i64) -> (i64) vh (vp0)
  return vr
  }
}
"#;

/// Build the Rust guest source with `child_bytes` embedded as a `static` array — the child's encoded
/// module, where an in-guest linker would leave its output. The guest resolves `inst`/`loader`/`fs` by
/// name, mints the child module via `__vm_module_from_bytes` over that static, lays a one-entry grant
/// record for `fs`, spawns the child into a 128 KiB-aligned 128 KiB carve inside `POOL`, and returns
/// `join(child)`. `POOL` (384 KiB) forces a window big enough for the carve and holds the record + name.
fn guest_src(child_bytes: &[u8]) -> String {
    let mut bytes_lit = String::new();
    for (i, b) in child_bytes.iter().enumerate() {
        if i % 16 == 0 {
            bytes_lit.push_str("\n    ");
        }
        bytes_lit.push_str(&format!("{b},"));
    }
    format!(
        r##"
#![no_std]
#![allow(internal_features)]

#[panic_handler]
fn ph(_: &core::panic::PanicInfo) -> ! {{
    loop {{}}
}}
#[no_mangle]
pub extern "C" fn rust_eh_personality() {{}}

#[repr(C, align(8))]
struct Pool([u8; 393216]);
static mut POOL: Pool = Pool([0; 393216]);

// The child module's encoded bytes — the linker's output a real driver would hold in memory.
static CHILD_BYTES: [u8; {n}] = [{bytes_lit}
];

extern "C" {{
    fn __vm_cap_resolve(name: *const u8, len: i64) -> i32;
    fn __vm_module_from_bytes(loader: i32, ptr: i64, len: i64) -> i64;
    fn __vm_instantiate(
        inst: i32,
        module: i64,
        grants_ptr: i64,
        grants_n: i64,
        entry: i64,
        off: i64,
        size_log2: i64,
        quota: i64,
    ) -> i64;
    fn __vm_join(inst: i32, child: i64) -> i64;
}}

#[no_mangle]
pub extern "C" fn run() -> i64 {{
    unsafe {{
        let inst = __vm_cap_resolve(b"inst".as_ptr(), 4);
        let loader = __vm_cap_resolve(b"loader".as_ptr(), 6);
        let fs = __vm_cap_resolve(b"fs".as_ptr(), 2);
        if inst < 0 || loader < 0 || fs < 0 {{
            return -1;
        }}
        // Promote the guest's own bytes to a spawnable module handle — the run-in-guest primitive.
        let mh = __vm_module_from_bytes(
            loader,
            CHILD_BYTES.as_ptr() as i64,
            CHILD_BYTES.len() as i64,
        );
        if mh < 0 {{
            return -2;
        }}
        let base = core::ptr::addr_of_mut!(POOL) as i64;
        // grant record at base: {{name_off:u32, name_len:u32, handle:i32, flags:u32}}
        let rec = base as *mut u32;
        rec.add(0).write((base + 16) as u32); // name_off
        rec.add(1).write(2); // name_len ("fs")
        rec.add(2).write(fs as u32); // handle
        rec.add(3).write(0); // flags
        let nm = (base + 16) as *mut u8;
        nm.add(0).write(b'f');
        nm.add(1).write(b's');
        // a 128 KiB-aligned 128 KiB carve above the record
        let carve = (base + 131071) & !131071;
        let child = __vm_instantiate(inst, mh, base, 1, 0, carve, 17, 0);
        __vm_join(inst, child)
    }}
}}
"##,
        n = child_bytes.len(),
        bytes_lit = bytes_lit,
    )
}

/// The granted `"fs"` shape: a forkable host-proc counter (the re-grantable form a shared memfs takes),
/// one shared `Arc` so a call from inside the confined child is observable here.
fn grant_fs(host: &mut Host, counter: &Arc<Mutex<i64>>) -> i32 {
    let c1 = Arc::clone(counter);
    let handler: HostProc = Box::new(move |_op, _args, _mem, _| {
        let mut c = c1.lock().unwrap();
        *c += 1;
        Ok(vec![*c])
    });
    let c2 = Arc::clone(counter);
    let fork = Arc::new(move |_pid: u64| {
        let c = Arc::clone(&c2);
        ForkedProc::shared(Box::new(move |_op, _args, _mem, _| {
            let mut c = c.lock().unwrap();
            *c += 1;
            Ok(vec![*c])
        }))
    });
    host.grant_host_proc_forkable(handler, fork)
}

/// `rustc --emit=llvm-ir` the guest to a textual `.ll` (single-crate `no_std`). False if `rustc` is
/// absent or codegen fails.
fn rustc_emit_ll(src_path: &std::path::Path, ll_path: &std::path::Path) -> bool {
    std::process::Command::new("rustc")
        .args([
            "--edition",
            "2021",
            "-O",
            "-Cpanic=abort",
            "--emit=llvm-ir",
            "--crate-type=cdylib",
        ])
        .arg(src_path)
        .arg("-o")
        .arg(ll_path)
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn parse_child() -> temen_ir::Module {
    let m = temen_text::parse_module(CHILD).expect("parse child");
    temen_verify::verify_module(&m).expect("verify child");
    m
}

/// The production [`GrantChildHooks`] table as the granted-spawn suites install it on the JIT.
fn grant_hooks() -> GrantChildHooks {
    GrantChildHooks {
        build: temen_run::grant_child_build,
        build_named: temen_run::grant_named_child_build,
        build_detached: temen_run::grant_detached_child_build,
        minter_take: temen_run::minter_take,
        bind_imports: temen_run::child_bind_imports,
        release: temen_run::grant_child_release,
        mint: temen_run::child_offer_mint,
        thunk: temen_run::cap_thunk_locked,
        register_serve: temen_run::child_register_serve,
    }
}

/// A host granting `inst` (Instantiator), `loader` (ModuleLoader — the run-in-guest cap), and `fs` by
/// name over a fresh counter. **No** `child` module is host-granted — the guest mints it from bytes.
fn granted_host(win: u64) -> (Host, Arc<Mutex<i64>>) {
    let counter = Arc::new(Mutex::new(0i64));
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, win);
    let loader = temen_run::grant_module_loader(&mut host);
    let fsh = grant_fs(&mut host, &counter);
    host.register_cap_name("inst", inst);
    host.register_cap_name("loader", loader);
    host.register_cap_name("fs", fsh);
    (host, counter)
}

/// Interpreter (cooperative engine — honors the op-13 grant list inline): returns `(result, counter)`.
fn run_interp(m: &temen_ir::Module, entry: u32, sp: i64, win: u64) -> (i64, i64) {
    let (mut host, counter) = granted_host(win);
    let mut fuel = 200_000_000u64;
    let (r, _) =
        run_capture_reserved_with_host(m, entry, &[Value::I64(sp)], &mut fuel, &[], 0, &mut host);
    let out = match r.expect("interp run").as_slice() {
        [Value::I64(x)] => *x,
        [Value::I32(x)] => *x as i64,
        other => panic!("interp result: {other:?}"),
    };
    let cval = *counter.lock().unwrap();
    (out, cval)
}

/// JIT (given the module resolver + named-grant hooks op-13 needs): returns `(result, counter)`.
fn run_jit(m: &temen_ir::Module, entry: u32, sp: i64, win: u64) -> (i64, i64) {
    let (mut host, counter) = granted_host(win);
    let (jo, _) = compile_and_run_capture_reserved_with_host_ex(
        m,
        entry,
        &[sp],
        &[],
        0,
        temen_run::cap_thunk,
        &mut host as *mut Host as *mut c_void,
        Some(temen_run::module_resolver),
        Some(grant_hooks()),
    )
    .expect("jit run");
    let out = match jo {
        JitOutcome::Returned(ref v) => v.first().copied().unwrap_or(0),
        JitOutcome::Exited(c) => c as i64,
        ref o => panic!("jit ended abnormally: {o:?}"),
    };
    let cval = *counter.lock().unwrap();
    (out, cval)
}

#[test]
fn rust_guest_mints_module_from_bytes_then_spawns_it() {
    // The child encoded exactly as an in-guest linker would emit it — the bytes the guest holds.
    let child = parse_child();
    let child_bytes = temen_encode::encode_module(&child);

    let dir = std::env::temp_dir().join(format!("rust_guest_mfb_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create work dir");
    let src = dir.join("guest.rs");
    let ll = dir.join("guest.ll");
    std::fs::write(&src, guest_src(&child_bytes)).expect("write guest source");

    if !rustc_emit_ll(&src, &ll) {
        eprintln!("note: skipping (rustc --emit=llvm-ir unavailable or failed)");
        return;
    }

    let t = temen_llvm::translate_ll_path(&ll).expect("temen-llvm translates the Rust guest");
    temen_verify::verify_module(&t.module).expect("the translated guest verifies");
    let entry = t
        .exports
        .iter()
        .find(|(n, _)| n == "run")
        .expect("guest exports `run`")
        .1;
    let sp = t.entry_sp as i64;
    let win = 1u64 << t.module.memory.expect("guest window").size_log2;

    let (io, ic) = run_interp(&t.module, entry, sp, win);
    let (jo, jc) = run_jit(&t.module, entry, sp, win);

    assert_eq!(
        io, 1,
        "interp: the guest minted the child from its bytes, op-13-spawned it, and joined its result (1)"
    );
    assert_eq!(
        jo, 1,
        "jit: same from_bytes -> op-13 spawn, same joined result (1)"
    );
    assert_eq!(
        io, jo,
        "§9 the guest's from_bytes+op-13 run agrees on both engines"
    );
    assert_eq!(
        (ic, jc),
        (1, 1),
        "the re-granted `fs` ran once inside the child minted from bytes, on each engine"
    );
}
