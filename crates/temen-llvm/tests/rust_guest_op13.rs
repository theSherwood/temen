//! **#1011 slice 3c — a Rust guest drives a §14 op-13 spawn.** The nim-compiler driver's endgame is to
//! move phase orchestration *into* the sandbox: a Rust-on-Temen guest that spawns each nimony phase child
//! via `instantiate_module_named` (op 13) over a shared `fs`, instead of the native exec cap — so the
//! phase children run on the tier-up-capable engine (slice 3a) rather than the tree-walker. This is the
//! enabling first light: the on-ramp now lowers `__vm_instantiate`/`__vm_join` (§14 Instantiator ops 13
//! and 1) to a `call.cap` on a name-resolved handle, so a real Rust guest — not a hand-written shell —
//! can issue the spawn. The C precedent is `temen/tests/c_shell_exec.rs`; this is its Rust counterpart via
//! the new builtins (the Rust on-ramp can't do chibicc's named-import passthrough, hence the builtins).
//!
//! The guest resolves the `Instantiator`, the child `Module`, and the shared `fs` by name
//! (`__vm_cap_resolve`), writes one 16-byte grant record for `fs` into a global workspace, spawns the
//! child into a 128 KiB carve, and joins it. The child (a separate module) resolves `fs` by name and
//! calls it — a granted counter returning `1`. So a correct run returns `1` and the shared counter
//! ticks once. Window confinement (§2) is untouched: the `fs` grant is authority (§3), a cross-tier
//! `call.cap`, not a window access.
//!
//! Gated to Linux + a present `rustc` (like the other on-ramp guest tests); skips cleanly otherwise.

#![cfg(target_os = "linux")]

use core::ffi::c_void;
use std::sync::{Arc, Mutex};
use temen_interp::{run_capture_reserved_with_host, ForkedProc, Host, HostProc, Value};
use temen_jit::{compile_and_run_capture_reserved_with_host_ex, GrantChildHooks, JitOutcome};

// The child: its `Instantiator` arrives as `v0` (unused). It seeds the name `"fs"` (`0x7366`
// little-endian = 'f','s') into its own window, resolves it, and calls the granted `HOST_PROC` counter
// (type 13, op 0) — post-increment `1`. `memory 17` matches the 128 KiB carve exactly (both engines).
const CHILD: &str = r#"memory 17
func (i64) -> (i64) {
block 0 (v0: i64) {
  vname = i64.const 29542
  vzero = i64.const 0
  i64.store vzero vname
  vp0 = i64.const 0
  vl2 = i64.const 2
  vh = self.resolve vp0 vl2
  vr = call.cap 13 0 (i64) -> (i64) vh (vp0)
  return vr
  }
}
"#;

// The Rust guest driver. No `std`, no allocator (it allocates nothing) — just the new §14 builtins over
// a global workspace. It resolves `inst`/`child`/`fs` by name, lays a grant record for `fs`, spawns the
// child into a 128 KiB-aligned 128 KiB carve inside `POOL`, and returns `join(child)`. `POOL` (384 KiB)
// both forces a window big enough for the carve and holds the record + name + carve — the C shell's
// `pool[]` trick, in Rust.
const GUEST_SRC: &str = r##"
#![no_std]
#![allow(internal_features)]

#[panic_handler]
fn ph(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
// alloc/unwind reference the personality even under panic=abort; never called here.
#[no_mangle]
pub extern "C" fn rust_eh_personality() {}

#[repr(C, align(8))]
struct Pool([u8; 393216]);
static mut POOL: Pool = Pool([0; 393216]);

extern "C" {
    fn __vm_cap_resolve(name: *const u8, len: i64) -> i32;
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
}

#[no_mangle]
pub extern "C" fn run() -> i64 {
    unsafe {
        let inst = __vm_cap_resolve(b"inst".as_ptr(), 4);
        let child_mod = __vm_cap_resolve(b"child".as_ptr(), 5);
        let fs = __vm_cap_resolve(b"fs".as_ptr(), 2);
        if inst < 0 || child_mod < 0 || fs < 0 {
            return -1;
        }
        let base = core::ptr::addr_of_mut!(POOL) as i64;
        // grant record at base: {name_off:u32, name_len:u32, handle:i32, flags:u32}
        let rec = base as *mut u32;
        rec.add(0).write((base + 16) as u32); // name_off
        rec.add(1).write(2); // name_len ("fs")
        rec.add(2).write(fs as u32); // handle
        rec.add(3).write(0); // flags
        // the name "fs" at base+16
        let nm = (base + 16) as *mut u8;
        nm.add(0).write(b'f');
        nm.add(1).write(b's');
        // a 128 KiB-aligned 128 KiB carve above the record
        let carve = (base + 131071) & !131071;
        let child = __vm_instantiate(inst, child_mod as i64, base, 1, 0, carve, 17, 0);
        __vm_join(inst, child)
    }
}
"##;

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

/// `rustc --emit=llvm-ir` the guest to a textual `.ll` (single-crate `no_std`, no `llvm-link`/`opt`).
/// Returns false if `rustc` is absent or codegen fails.
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

/// The production [`GrantChildHooks`] table (temen-run's child build/bind/release/mint/thunk/serve) as the
/// granted-spawn suites install it on the JIT — the same table `temen/tests/c_shell_exec.rs` uses.
fn grant_hooks() -> GrantChildHooks {
    GrantChildHooks {
        build: temen_run::grant_child_build,
        build_named: temen_run::grant_named_child_build,
        bind_imports: temen_run::child_bind_imports,
        release: temen_run::grant_child_release,
        mint: temen_run::child_offer_mint,
        thunk: temen_run::cap_thunk_locked,
        register_serve: temen_run::child_register_serve,
    }
}

/// Build a host granting `inst`/`child`/`fs` by name over a fresh counter; returns `(host, counter)`.
fn granted_host(child: &temen_ir::Module, win: u64) -> (Host, Arc<Mutex<i64>>) {
    let counter = Arc::new(Mutex::new(0i64));
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, win);
    let modh = host.grant_module(child);
    let fsh = grant_fs(&mut host, &counter);
    host.register_cap_name("inst", inst);
    host.register_cap_name("child", modh);
    host.register_cap_name("fs", fsh);
    (host, counter)
}

/// Interpreter (cooperative engine — honors the op-13 grant list inline): returns `(result, counter)`.
fn run_interp(
    m: &temen_ir::Module,
    entry: u32,
    sp: i64,
    child: &temen_ir::Module,
    win: u64,
) -> (i64, i64) {
    let (mut host, counter) = granted_host(child, win);
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
fn run_jit(
    m: &temen_ir::Module,
    entry: u32,
    sp: i64,
    child: &temen_ir::Module,
    win: u64,
) -> (i64, i64) {
    let (mut host, counter) = granted_host(child, win);
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
fn rust_guest_spawns_child_via_op13() {
    let dir = std::env::temp_dir().join(format!("rust_guest_op13_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create work dir");
    let src = dir.join("guest.rs");
    let ll = dir.join("guest.ll");
    std::fs::write(&src, GUEST_SRC).expect("write guest source");

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

    let child = parse_child();

    let (io, ic) = run_interp(&t.module, entry, sp, &child, win);
    let (jo, jc) = run_jit(&t.module, entry, sp, &child, win);

    assert_eq!(
        io, 1,
        "interp: the guest spawned the child via op-13 and joined its result (1)"
    );
    assert_eq!(jo, 1, "jit: same op-13 spawn, same joined result (1)");
    assert_eq!(io, jo, "§9 the guest's op-13 spawn agrees on both engines");
    assert_eq!(
        (ic, jc),
        (1, 1),
        "the re-granted `fs` ran once inside the confined child on each engine"
    );
}
