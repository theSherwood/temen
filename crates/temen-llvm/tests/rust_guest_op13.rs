//! **#1011 slice 3c — a Rust guest drives a §14 spawn.** The nim-compiler driver's endgame is to move
//! phase orchestration *into* the sandbox: a Rust-on-Temen guest that spawns each nimony phase child
//! over a shared `fs`, instead of the native exec cap. This is the enabling first light: the on-ramp
//! lowers `__vm_instantiate_rec`/`__vm_join` (§14 Instantiator ops 17 and 1) to a `call.cap` on a
//! name-resolved handle, so a real Rust guest — not a hand-written shell — can issue the spawn, through
//! the one guest-side helper every driver guest shares (`support/guest_vm_spawn.rs`: an op-17 v1
//! record, the child in a window of its own). The C precedent is `temen/tests/c_shell_exec.rs`.
//!
//! The guest resolves the `Instantiator`, the child `Module`, its `Budget` and the shared `fs` by name
//! (`__vm_cap_resolve`) and spawns the child with `{"fs"}` re-granted, then joins it. The child (a
//! separate module) resolves `fs` by name and calls it — a granted counter returning `1`. So a correct
//! run returns `1` and the shared counter ticks once. Window confinement (§2) is untouched: the `fs`
//! grant is authority (§3), a cross-tier `call.cap`, not a window access.
//!
//! Gated to Linux + a present `rustc` (like the other on-ramp guest tests); skips cleanly otherwise.

#![cfg(target_os = "linux")]

use core::ffi::c_void;
use std::sync::{Arc, Mutex};
use temen_interp::{run_capture_reserved_with_host, ForkedProc, Host, HostProc, Value};
use temen_jit::{compile_and_run_capture_reserved_with_host_ex, GrantChildHooks, JitOutcome};

// The child: its `Instantiator` arrives as `v0` (unused). It seeds the name `"fs"` (`0x7366`
// little-endian = 'f','s') into its own window, resolves it, and calls the granted `HOST_PROC` counter
// (type 13, op 0) — post-increment `1`. Its window is its declared `memory 17`.
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

// The Rust guest driver. No `std`, no allocator — the §14 builtins through the shared `vm_spawn`
// (`support/guest_vm_spawn.rs`, appended at build). It resolves `inst`/`child`/`budget`/`fs` by name,
// spawns the child with `{"fs"}` re-granted, and returns `join(child)`.
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

extern "C" {
    fn __vm_cap_resolve(name: *const u8, len: i64) -> i32;
    fn __vm_join(inst: i32, child: i64) -> i64;
}

#[no_mangle]
pub extern "C" fn run() -> i64 {
    unsafe {
        let inst = __vm_cap_resolve(b"inst".as_ptr(), 4);
        let child_mod = __vm_cap_resolve(b"child".as_ptr(), 5);
        let budget = __vm_cap_resolve(b"budget".as_ptr(), 6);
        let fs = __vm_cap_resolve(b"fs".as_ptr(), 2);
        if inst < 0 || child_mod < 0 || budget < 0 || fs < 0 {
            return -1;
        }
        let child = vm_spawn(inst, budget, child_mod, &[(b"fs", fs)], &[]);
        __vm_join(inst, child)
    }
}
"##;

/// The shared guest-side spawn, appended to every driver guest's source.
const VM_SPAWN: &str = include_str!("support/guest_vm_spawn.rs");

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
        ForkedProc::shared(
            Box::new(move |_op, _args, _mem, _| {
                let mut c = c.lock().unwrap();
                *c += 1;
                Ok(vec![*c])
            }),
            temen_interp::CapState::Stateless,
        )
    });
    host.grant_host_proc_forkable(handler, fork, temen_interp::CapState::Stateless)
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
fn grant_hooks(host: *mut temen_interp::Host) -> GrantChildHooks {
    temen_run::production_grant_hooks(temen_run::CapCtx::Raw(host))
}

/// Build a host granting `inst`/`child`/`fs` by name over a fresh counter; returns `(host, counter)`.
fn granted_host(child: &temen_ir::Module, win: u64) -> (Host, Arc<Mutex<i64>>) {
    let counter = Arc::new(Mutex::new(0i64));
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, win);
    let modh = host.grant_module(child);
    let budget = host.grant_budget(0, 1 << 17, 0); // the child's window
    let fsh = grant_fs(&mut host, &counter);
    host.register_cap_name("inst", inst);
    host.register_cap_name("child", modh);
    host.register_cap_name("budget", budget);
    host.register_cap_name("fs", fsh);
    (host, counter)
}

/// Interpreter (cooperative engine — honors the grant list inline): returns `(result, counter)`.
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

/// JIT (given the module resolver + named-grant hooks the spawn needs): returns `(result, counter)`.
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
        Some(grant_hooks(&mut host as *mut Host)),
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
fn rust_guest_spawns_a_child() {
    let dir = std::env::temp_dir().join(format!("rust_guest_op13_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create work dir");
    let src = dir.join("guest.rs");
    let ll = dir.join("guest.ll");
    std::fs::write(&src, format!("{GUEST_SRC}\n{VM_SPAWN}")).expect("write guest source");

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
        "interp: the guest spawned the child and joined its result (1)"
    );
    assert_eq!(jo, 1, "jit: same spawn, same joined result (1)");
    assert_eq!(io, jo, "§9 the guest's spawn agrees on both engines");
    assert_eq!(
        (ic, jc),
        (1, 1),
        "the re-granted `fs` ran once inside the confined child on each engine"
    );
}
