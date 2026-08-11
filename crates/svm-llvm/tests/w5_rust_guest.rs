//! **W5 first light** (NIM.md §3e, Path L): a real Rust program runs **as a guest on svm**.
//!
//! W5 self-hosts the toolchain — the payoff is running the Rust `svm-leng` translator on svm. The
//! measured recommendation (§3e) is to reach first light through the **LLVM on-ramp** (already the
//! proven Rust→svm primitive: `rustbench`, the `peval` fixtures), before the wasm on-ramp. This is
//! that first light on **`svm-leng`-shaped work**: a `no_std + alloc` Rust program that *emits IR-ish
//! text and re-parses/folds it* — the exact surface the real translator leans on (`String`/`Vec`
//! building, byte scanning, integer format/parse) — compiled `rustc --emit=llvm-ir` → `svm-llvm`, and
//! run on svm. The Rust analog of chibicc's C self-host, one rung down.
//!
//! Oracle discipline: the guest result must equal (a) itself across **interp and JIT** (§9) and (b)
//! the **identical algorithm run natively** in this test (§18). Any divergence is a real on-ramp bug.
//!
//! Gated to Linux + a present `rustc` (like the C on-ramp tests); skips cleanly otherwise.

#![cfg(target_os = "linux")]

use svm_interp::Value;

/// The guest: the `rustbench` `no_std` prelude (bump `#[global_allocator]`, panic handler, the
/// `alloc`-personality stub) + a `run(n) -> i64` doing the leng-shaped text work. `svm-llvm` prepends
/// the data-stack pointer, so the translated entry is `run(sp, n)`.
const GUEST_SRC: &str = r##"
#![no_std]
#![allow(internal_features)]
extern crate alloc;
use alloc::string::String;
use alloc::vec::Vec;
use core::alloc::{GlobalAlloc, Layout};

const ARENA: usize = 1 << 20;
static mut HEAP: [u8; ARENA] = [0; ARENA];
static mut OFF: usize = 0;

struct Bump;
unsafe impl GlobalAlloc for Bump {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let a = l.align();
        let s = (OFF + a - 1) & !(a - 1);
        if s + l.size() > ARENA {
            return core::ptr::null_mut();
        }
        OFF = s + l.size();
        (core::ptr::addr_of_mut!(HEAP) as *mut u8).add(s)
    }
    unsafe fn dealloc(&self, _: *mut u8, _: Layout) {}
}
#[global_allocator]
static GA: Bump = Bump;

#[panic_handler]
fn ph(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

// alloc is built with unwind and references the personality even under panic=abort; never called.
#[no_mangle]
pub extern "C" fn rust_eh_personality() {}

fn push_i64(s: &mut String, mut v: i64) {
    if v == 0 {
        s.push('0');
        return;
    }
    let mut buf = [0u8; 20];
    let mut i = 20;
    while v > 0 {
        i -= 1;
        buf[i] = b'0' + (v % 10) as u8;
        v /= 10;
    }
    let mut j = i;
    while j < 20 {
        s.push(buf[j] as char);
        j += 1;
    }
}

fn parse_i64(bytes: &[u8]) -> i64 {
    let mut v = 0i64;
    for &b in bytes {
        if b.is_ascii_digit() {
            v = v * 10 + (b - b'0') as i64;
        }
    }
    v
}

#[no_mangle]
pub extern "C" fn run(n: i64) -> i64 {
    unsafe {
        OFF = 0;
    }
    // Emit a little "program": k lines `add <i*3 + (n&7)>` — a lowering pass building output text.
    let k = (n & 63) + 1;
    let mut prog = String::new();
    let mut i = 0i64;
    while i < k {
        prog.push_str("add ");
        push_i64(&mut prog, i * 3 + (n & 7));
        prog.push('\n');
        i += 1;
    }
    // Re-read the emitted text (like re-parsing emitted IR) and fold it.
    let mut acc = 0i64;
    let mut toks: Vec<i64> = Vec::new();
    for line in prog.as_bytes().split(|&b| b == b'\n') {
        if line.is_empty() {
            continue;
        }
        let num = &line[4..]; // past "add "
        acc = acc.wrapping_add(parse_i64(num));
        toks.push(acc);
    }
    // Checksum over the collected Vec (exercise indexed iteration).
    let mut sum = 0i64;
    let mut idx = 0i64;
    for &t in toks.iter() {
        sum = sum.wrapping_add(t ^ idx);
        idx += 1;
    }
    sum
}
"##;

/// The native oracle — the **identical algorithm**, run in-process. If this ever disagrees with the
/// svm-run guest, `svm-llvm` mistranslated real Rust.
fn oracle(n: i64) -> i64 {
    let k = (n & 63) + 1;
    let mut prog = String::new();
    let mut i = 0i64;
    while i < k {
        prog.push_str("add ");
        prog.push_str(&(i * 3 + (n & 7)).to_string());
        prog.push('\n');
        i += 1;
    }
    let mut acc = 0i64;
    let mut toks: Vec<i64> = Vec::new();
    for line in prog.as_bytes().split(|&b| b == b'\n') {
        if line.is_empty() {
            continue;
        }
        let num = &line[4..];
        let mut v = 0i64;
        for &b in num {
            if b.is_ascii_digit() {
                v = v * 10 + (b - b'0') as i64;
            }
        }
        acc = acc.wrapping_add(v);
        toks.push(acc);
    }
    let mut sum = 0i64;
    for (idx, &t) in toks.iter().enumerate() {
        sum = sum.wrapping_add(t ^ idx as i64);
    }
    sum
}

/// `rustc --emit=llvm-ir` the guest to a textual `.ll` (the `rustbench` single-file recipe: no
/// `llvm-link`/`opt` needed for one crate). Host target (Linux x86_64 under the cfg gate). Returns
/// false if `rustc` is absent or codegen fails.
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

#[test]
fn rust_guest_runs_leng_shaped_work_on_svm() {
    let dir = std::env::temp_dir().join(format!("w5_rust_guest_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create work dir");
    let src = dir.join("guest.rs");
    let ll = dir.join("guest.ll");
    std::fs::write(&src, GUEST_SRC).expect("write guest source");

    if !rustc_emit_ll(&src, &ll) {
        eprintln!("note: skipping W5 first-light (rustc --emit=llvm-ir unavailable or failed)");
        return;
    }

    let t = svm_llvm::translate_ll_path(&ll).expect("svm-llvm translates the Rust guest's LLVM IR");
    svm_verify::verify_module(&t.module).expect("the translated Rust guest verifies");
    let entry = t
        .exports
        .iter()
        .find(|(n, _)| n == "run")
        .expect("guest exports `run`")
        .1;
    let sp = t.entry_sp as i64;

    for n in [0i64, 1, 2, 7, 8, 63, 64, 255, 1000, -1] {
        let mut fuel = u64::MAX;
        let interp = match svm_interp::run(
            &t.module,
            entry,
            &[Value::I64(sp), Value::I64(n)],
            &mut fuel,
        )
        .expect("interp run")
        .as_slice()
        {
            [Value::I64(v)] => *v,
            other => panic!("interp: expected i64, got {other:?}"),
        };
        let jit = match svm_jit::compile_and_run(&t.module, entry, &[sp, n]).expect("jit run") {
            svm_jit::JitOutcome::Returned(v) => v[0],
            other => panic!("jit did not return: {other:?}"),
        };
        assert_eq!(interp, jit, "§9 interp/JIT parity for n={n}");
        assert_eq!(
            interp,
            oracle(n),
            "§18 svm-run == native oracle for n={n} (Rust guest ran correctly on svm)"
        );
    }
}
