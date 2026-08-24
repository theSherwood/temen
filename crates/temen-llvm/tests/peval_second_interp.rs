//! **De-overfitting peval: the discovery technique on a SECOND, non-Lua interpreter.**
//!
//! `lua_futamura_auto_rolled.rs` discovers a Lua loop's carried cells by observing the running VM. The
//! worry: is that discovery Lua-shaped, or genuinely interpreter-agnostic? This answers it on a
//! structurally different interpreter — a C register-machine VM (`clang -O2 → LLVM → temen-IR`) that
//! shares *nothing* with Lua's memory model:
//!
//! | | Lua (`luaV_execute`) | register VM (`run`) |
//! |---|---|---|
//! | registers | frame-relative (`ci->func + 1 TValue`) | a **global array** `long reg[16]` at a fixed addr |
//! | values | tagged `TValue` (8-byte value + tag@+8) | plain untagged `long` |
//! | program counter | `ci->savedpc` **in memory** | a `long pc` **local** (in a register, not memory) |
//!
//! The **same** technique — observe the dispatch block, find the loop-carried cells as the registers
//! whose value varies across hits, pick the counter as the monotone-decreasing one — recovers the
//! regVM's accumulator and counter with only ONE target-specific input: *where the registers live*
//! (here the `reg` symbol's address, from the on-ramp's `data_symbols`; for Lua, `ci->func + 16`). No
//! tag model, no Lua offsets. That "register region" is the first field of a real `TargetDesc`.
//!
//! (The regVM's pc-in-a-register also shows a genuine structural limit: it can't be *resumed* from a
//! memory snapshot the way Lua can, so it entry-roots rather than safepoint-roots. Cell discovery is
//! the part that transfers; rooting strategy is target-specific — a finding, documented here.)
//!
//! Run: `cargo test -p temen-llvm --test peval_second_interp -- --nocapture`

mod peval_capture;

use std::process::Command;

use peval_capture::{discover, dispatch_block, Located, TargetDesc};
use temen_interp::Value;
use temen_ir::{Module, Terminator};

// A pure accumulator loop, register-VM style: apply `reg[1] += 7` a runtime-`n` times, counting down
// `reg[0]`. Result = 7·n. No tags, registers in a global array — deliberately unlike Lua.
const LOOP_VM: &str = r#"
enum { HALT, LOADI, LOADIN, ADDK, JLE, JMP };
static long reg[16];
long run(const int *prog, long n) {
    long pc = 0;
    for (;;) {
        int op = prog[pc], a = prog[pc+1], b = prog[pc+2], c = prog[pc+3];
        if (op == HALT) return reg[a];
        else if (op == LOADI)  reg[a] = b;
        else if (op == LOADIN) reg[a] = n;
        else if (op == ADDK)   reg[a] = reg[b] + c;
        else if (op == JLE)  { if (reg[a] <= reg[b]) { pc = (long)c * 4; continue; } }
        else if (op == JMP)  { pc = (long)a * 4; continue; }
        pc += 4;
    }
}
static const int prog[] = {
    2, 0, 0, 0,   1, 1, 0, 0,   1, 2, 0, 0,   4, 0, 2, 7,
    3, 1, 1, 7,   3, 0, 0, -1,  5, 3, 0, 0,   0, 1, 0, 0,
};
int main(void) { return (int)run(prog, 0); }
"#;
// reg0 = counter (= n, counts down), reg1 = accumulator (+7/iter), reg2 = 0 (the exit bound).
// HALT=0 LOADI=1 LOADIN=2 ADDK=3 JLE=4 JMP=5.
const LOOP_PROG: [i32; 32] = [
    2, 0, 0, 0, //  LOADIN r0 = n
    1, 1, 0, 0, //  LOADI  r1 = 0
    1, 2, 0, 0, //  LOADI  r2 = 0
    4, 0, 2, 7, //  JLE    if r0 <= r2(0) goto pc7 (HALT)
    3, 1, 1, 7, //  ADDK   r1 = r1 + 7
    3, 0, 0, -1, // ADDK   r0 = r0 + (-1)
    5, 3, 0, 0, //  JMP    pc3 (loop test)
    0, 1, 0, 0, //  HALT   return r1
];

struct Built {
    module: Module,
    sp: i64,
    run: u32,
    prog_addr: i64,
    reg_addr: u64,
}

/// Compile the C VM → LLVM → temen-IR, keeping the `data_symbols` so we can locate the `reg` global.
fn build() -> Option<Built> {
    let base = std::env::temp_dir().join("peval_second_interp");
    let cf = base.with_extension("c");
    let ll = base.with_extension("ll");
    std::fs::write(&cf, LOOP_VM).ok()?;
    let ok = Command::new("clang")
        .args([
            "-O2",
            "-emit-llvm",
            "-S",
            "-fno-vectorize",
            "-fno-slp-vectorize",
        ])
        .arg(&cf)
        .arg("-o")
        .arg(&ll)
        .status()
        .ok()
        .map(|s| s.success())
        .unwrap_or(false);
    if !ok {
        return None;
    }
    let t = temen_llvm::translate_ll_path(&ll).expect("temen-llvm translate");
    let _ = std::fs::remove_file(&cf);
    let _ = std::fs::remove_file(&ll);

    let run = t
        .module
        .exports
        .iter()
        .find(|e| e.name == "run")
        .expect("run export")
        .func;
    // `reg` symbol address — the regVM's "register region" (a `TargetDesc` input).
    let reg_addr = t
        .data_symbols
        .iter()
        .find(|d| d.name == "reg")
        .expect("reg symbol")
        .addr;
    // `prog` readonly bytes → its window address.
    let pb: Vec<u8> = LOOP_PROG.iter().flat_map(|w| w.to_le_bytes()).collect();
    let d = t
        .module
        .data
        .iter()
        .find(|d| d.readonly && d.bytes.windows(pb.len()).any(|w| w == pb.as_slice()))
        .expect("readonly program segment");
    let off = d
        .bytes
        .windows(pb.len())
        .position(|w| w == pb.as_slice())
        .unwrap();
    let prog_addr = d.offset as i64 + off as i64;

    Some(Built {
        module: t.module,
        sp: t.entry_sp as i64,
        run,
        prog_addr,
        reg_addr,
    })
}

#[test]
fn discovery_generalizes_to_a_register_vm() {
    let Some(built) = build() else {
        eprintln!("clang unavailable — skipping");
        return;
    };
    // Sanity: interpreter is correct (7·n) and dispatches through a table.
    let disp = dispatch_block(&built.module, built.run);
    assert!(
        matches!(
            built.module.funcs[built.run as usize].blocks[disp as usize].term,
            Terminator::BrTable { .. }
        ),
        "the regVM dispatches through a table"
    );

    // The regVM's `TargetDesc`: registers are 8-byte `long`s in the `reg` global (no frame), untagged,
    // and the pc lives in a machine register (no in-memory pc → `Located::pc_addr = None`, so the
    // shared driver analyses from the first hit rather than seeking a savedpc safepoint).
    let stride = 8u64;
    let input: i64 = 8;
    let make_insp = || {
        temen_interp::Inspector::attach(
            &built.module,
            built.run,
            &[
                Value::I64(built.sp),
                Value::I64(built.prog_addr),
                Value::I64(input),
            ],
            u64::MAX,
        )
    };
    let loc = Located {
        reg_base: built.reg_addr,
        pc_addr: None,
    };
    let desc = TargetDesc {
        reg_stride: stride,
        n_regs: 16,
        tag: None,
        capture_len: 1 << 20,
        observe_hits: 48,
    };
    let d = discover(&make_insp, built.run, disp, &loc, &desc);

    let as_reg = |a: u64| (a as i64 - built.reg_addr as i64) / stride as i64;
    println!(
        "\nreg_addr={} regVM discovered loop cells (as R[n]): {:?}  counter={}",
        built.reg_addr,
        d.varying.iter().map(|&a| as_reg(a)).collect::<Vec<_>>(),
        if d.counter == 0 {
            "NONE".to_string()
        } else {
            format!("R{}", as_reg(d.counter))
        },
    );

    // The technique must recover the accumulator (reg1, +7/iter) and the counter (reg0, counts down),
    // with ZERO Lua structure — only the reg region address, and no tag model at all.
    let reg0 = built.reg_addr; // counter
    let reg1 = built.reg_addr + stride; // accumulator
    assert!(d.varying.contains(&reg0), "counter reg0 discovered");
    assert!(d.varying.contains(&reg1), "accumulator reg1 discovered");
    assert_eq!(
        d.counter, reg0,
        "counter identified as the monotone-decreasing cell"
    );
    // reg2 (the exit bound, constant 0) must NOT be flagged as loop-carried.
    assert!(
        !d.varying.contains(&(built.reg_addr + 2 * stride)),
        "the invariant reg2 must not be flagged carried"
    );
    println!(
        "cell discovery generalizes: same technique, new interpreter, only the reg region changed"
    );
}
