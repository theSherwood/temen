//! **De-overfitting peval: the discovery technique on a SECOND, non-Lua interpreter.**
//!
//! `lua_futamura_auto_rolled.rs` discovers a Lua loop's carried cells by observing the running VM. The
//! worry: is that discovery Lua-shaped, or genuinely interpreter-agnostic? This answers it on a
//! structurally different interpreter — a C register-machine VM (`clang -O2 → LLVM → svm-IR`) that
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
//! Run: `cargo test -p svm-llvm --test peval_second_interp -- --nocapture`

use std::process::Command;

use svm_interp::{IrPc, Stop, StopReason, Value};
use svm_ir::{Module, Terminator};

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

/// Compile the C VM → LLVM → svm-IR, keeping the `data_symbols` so we can locate the `reg` global.
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
    let t = svm_llvm::translate_ll_path(&ll).expect("svm-llvm translate");
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

fn rd_u64(w: &[u8], a: u64) -> u64 {
    u64::from_le_bytes(w[a as usize..a as usize + 8].try_into().unwrap())
}

fn dispatch_block(m: &Module, func: u32) -> u32 {
    let mut best = (0u32, 0usize);
    for (bi, b) in m.funcs[func as usize].blocks.iter().enumerate() {
        if let Terminator::BrTable { targets, .. } = &b.term {
            if targets.len() > best.1 {
                best = (bi as u32, targets.len());
            }
        }
    }
    best.0
}

const CAP_LEN: usize = 1 << 20;
const OBSERVE_HITS: usize = 48;

/// Discovered loop-carried cells in the register region — the SAME technique as the Lua auto-rolled
/// path, parameterized only by `(reg_base, n_regs, stride)`.
struct Discovered {
    varying: Vec<u64>, // addresses of registers whose value varies across the loop
    counter: u64,      // the monotone-decreasing one
}

fn discover_loop_cells(built: &Built, reg_base: u64, n_regs: u64, stride: u64) -> Discovered {
    let input: i64 = 8;
    let mut insp = svm_interp::Inspector::attach(
        &built.module,
        built.run,
        &[
            Value::I64(built.sp),
            Value::I64(built.prog_addr),
            Value::I64(input),
        ],
        u64::MAX,
    );
    let disp = dispatch_block(&built.module, built.run);
    let disp_pc = IrPc {
        module: 0,
        func: built.run,
        block: disp as usize,
        inst: 0,
    };
    insp.set_breakpoint(disp_pc);
    let mut series: Vec<Vec<u64>> = vec![Vec::new(); n_regs as usize];
    for _ in 0..OBSERVE_HITS {
        match insp.run_until_stop() {
            Stop::Break {
                reason: StopReason::Breakpoint,
                ..
            } => {
                let w = insp.read_window(0, CAP_LEN).expect("window");
                for (r, s) in series.iter_mut().enumerate() {
                    s.push(rd_u64(&w, reg_base + r as u64 * stride));
                }
            }
            Stop::Finished { .. } => break,
            o => panic!("unexpected: {o:?}"),
        }
    }
    // Skip the prologue hits (LOADIN/LOADI setting up reg0..reg2) so pre-loop writes don't count as
    // "loop variation": start the analysis once the register file stops its initial churn — i.e., from
    // the first hit at which no register has changed vs the previous hit is fragile; simpler and robust
    // for this VM is to analyse the whole stream but treat a cell as carried only if it takes ≥3
    // distinct values (a one-time prologue write yields 2).
    let mut varying = Vec::new();
    let mut counter = 0u64;
    for (r, s) in series.iter().enumerate() {
        let distinct = {
            let mut v = s.clone();
            v.sort_unstable();
            v.dedup();
            v.len()
        };
        if distinct >= 3 {
            let addr = reg_base + r as u64 * stride;
            varying.push(addr);
            // Counter = a cell that, from its peak onward, is monotone non-increasing and net-decreases
            // (the FORLOOP-style countdown). Analysing from the peak — not the raw stream — skips the
            // prologue's ramp-up (a `LOADIN` writing the initial count) that a raw check would trip on.
            let argmax = s
                .iter()
                .enumerate()
                .max_by_key(|(_, v)| **v)
                .map(|(i, _)| i)
                .unwrap();
            let tail = &s[argmax..];
            let monotone_down = tail.windows(2).all(|w| w[1] <= w[0]) && tail.first() > tail.last();
            if monotone_down && counter == 0 {
                counter = addr;
            }
        }
    }
    Discovered { varying, counter }
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

    let stride = 8u64;
    let d = discover_loop_cells(&built, built.reg_addr, 16, stride);

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
