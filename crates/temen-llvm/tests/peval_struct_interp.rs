//! **De-overfitting, the strong test: the SAFEPOINT roll on a non-Lua interpreter.**
//!
//! `peval_second_interp.rs` proved *cell discovery* generalizes, but its register-VM keeps its pc in a
//! machine register, so it can only be entry-rooted — it never exercises the hard Lua machinery
//! (mid-loop safepoint capture + rolling with dynamic cells). This does: a third interpreter shaped
//! like Lua's resume model but with a completely different memory layout.
//!
//! A C VM whose state is a **struct** `VM { long pc; long r[16]; }` at a fixed global address:
//! `resume(prog)` is a pure dispatch loop that **resumes from `vm.pc`** (an in-memory pc, like Lua's
//! `ci->savedpc`), and `run(prog, n)` sets the struct up and calls it. The shared [`discover`] driver —
//! unchanged — finds the loop-body safepoint via the in-memory pc and the carried cells (`r0` counter,
//! `r1` accumulator) in the struct. We then specialize `resume` rooted at that safepoint with the
//! discovered dynamic cells, and the real dispatch loop **rolls**: dispatch folds, the loop stays, the
//! residual takes the carried cells as parameters — exactly the Lua rolled-residual shape, on a struct
//! layout that shares nothing with Lua.
//!
//! Run: `cargo test -p temen-llvm --test peval_struct_interp -- --nocapture`

mod peval_capture;

use std::process::Command;

use peval_capture::{discover, dispatch_block, rd_u64, Located, TargetDesc};
use temen_interp::{Inspector, Value};
use temen_ir::{Func, Module, Terminator};
use temen_peval::{specialize_with_config, SpecArg, SpecConfig};

// State is a struct at a fixed address: pc at vm+0, r[16] at vm+8. `resume` resumes from vm.pc.
// Program: `for n iterations: r1 += 7` counting down r0. resume returns r1. Result = 7·n.
// HALT=0 ADDK=1 JLE=2 JMP=3.
const STRUCT_VM: &str = r#"
enum { HALT, ADDK, JLE, JMP };
typedef struct { long pc; long r[16]; } VM;
static VM vm;
__attribute__((noinline)) long resume(const int *prog) {
    for (;;) {
        int op = prog[vm.pc], a = prog[vm.pc+1], b = prog[vm.pc+2], c = prog[vm.pc+3];
        if (op == HALT) return vm.r[a];
        else if (op == ADDK) vm.r[a] = vm.r[b] + c;
        else if (op == JLE) { if (vm.r[a] <= vm.r[b]) { vm.pc = (long)c * 4; continue; } }
        else if (op == JMP) { vm.pc = (long)a * 4; continue; }
        vm.pc += 4;
    }
}
long run(const int *prog, long n) {
    vm.pc = 0;
    for (int i = 0; i < 16; i++) vm.r[i] = 0;
    vm.r[0] = n;
    return resume(prog);
}
static const int prog[] = {
    2, 0, 2, 4,   1, 1, 1, 7,   1, 0, 0, -1,   3, 0, 0, 0,   0, 1, 0, 0,
};
int main(void) { return (int)run(prog, 0); }
"#;
// pc0: JLE r0<=r2(0) -> pc4 (HALT)   pc1: ADDK r1=r1+7   pc2: ADDK r0=r0-1   pc3: JMP pc0   pc4: HALT r1
const PROG: [i32; 20] = [2, 0, 2, 4, 1, 1, 1, 7, 1, 0, 0, -1, 3, 0, 0, 0, 0, 1, 0, 0];

const VM_SIZE: usize = 8 + 16 * 8; // pc + r[16]
const REG_OFF: u64 = 8; // vm.r starts at vm+8
const PC_OFF: u64 = 0; // vm.pc at vm+0

struct Built {
    module: Module,
    sp: i64,
    resume: u32,
    run: u32,
    prog_addr: i64,
    vm_addr: u64,
}

fn build() -> Option<Built> {
    let base = std::env::temp_dir().join("peval_struct_interp");
    let cf = base.with_extension("c");
    let ll = base.with_extension("ll");
    std::fs::write(&cf, STRUCT_VM).ok()?;
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
    let exp = |name: &str| {
        t.module
            .exports
            .iter()
            .find(|e| e.name == name)
            .map(|e| e.func)
    };
    let resume = exp("resume").expect("resume export");
    let run = exp("run").expect("run export");
    let vm_addr = t
        .data_symbols
        .iter()
        .find(|d| d.name == "vm")
        .expect("vm symbol")
        .addr;
    let pb: Vec<u8> = PROG.iter().flat_map(|w| w.to_le_bytes()).collect();
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
        resume,
        run,
        prog_addr,
        vm_addr,
    })
}

fn has_br_table(f: &Func) -> bool {
    f.blocks
        .iter()
        .any(|b| matches!(b.term, Terminator::BrTable { .. }))
}
fn tw(m: &Module, e: u32, a: &[i64]) -> i64 {
    let mut fuel = u64::MAX;
    let vs: Vec<Value> = a.iter().map(|&x| Value::I64(x)).collect();
    match temen_interp::run(m, e, &vs, &mut fuel)
        .expect("no trap")
        .as_slice()
    {
        [Value::I64(x)] => *x,
        o => panic!("bad {o:?}"),
    }
}
fn jit(m: &Module, e: u32, a: &[i64]) -> i64 {
    match temen_jit::compile_and_run(m, e, a) {
        Ok(temen_jit::JitOutcome::Returned(v)) => v[0],
        o => panic!("jit {o:?}"),
    }
}

#[test]
fn struct_vm_rolls_via_safepoint_capture() {
    let Some(built) = build() else {
        eprintln!("clang unavailable — skipping");
        return;
    };
    let n: i64 = 8;
    // Sanity: interpreter runs the loop and dispatches through a table.
    assert_eq!(
        tw(&built.module, built.run, &[built.sp, built.prog_addr, n]),
        7 * n
    );
    let disp = dispatch_block(&built.module, built.resume);
    assert!(
        has_br_table(&built.module.funcs[built.resume as usize]),
        "resume dispatches via table"
    );

    // Capture with the SHARED driver — the only new inputs are this VM's TargetDesc / Located. The pc
    // lives in memory (vm.pc), so `pc_addr = Some` and the driver uses the safepoint path (unlike the
    // register-pc regVM). Observe `resume`'s dispatch while entering through `run`.
    let make_insp = || {
        Inspector::attach(
            &built.module,
            built.run,
            &[
                Value::I64(built.sp),
                Value::I64(built.prog_addr),
                Value::I64(n),
            ],
            u64::MAX,
        )
    };
    let loc = Located {
        reg_base: built.vm_addr + REG_OFF,
        pc_addr: Some(built.vm_addr + PC_OFF),
    };
    let desc = TargetDesc {
        reg_stride: 8,
        n_regs: 16,
        tag: None,
        capture_len: 1 << 20,
        observe_hits: 48,
    };
    let d = discover(&make_insp, built.resume, disp, &loc, &desc);

    let as_reg = |a: u64| (a - loc.reg_base) / 8;
    println!(
        "\nstruct-VM safepoint: dyn cells (as r[n]): {:?}  counter=r{}",
        d.varying.iter().map(|&a| as_reg(a)).collect::<Vec<_>>(),
        as_reg(d.counter),
    );
    // The carried cells are the counter r0 and accumulator r1; the counter is r0.
    assert!(d.varying.contains(&loc.reg_base), "counter r0 discovered");
    assert!(
        d.varying.contains(&(loc.reg_base + 8)),
        "accumulator r1 discovered"
    );
    assert_eq!(d.counter, loc.reg_base, "counter is r0");
    assert!(d.counter != 0, "a counter was found");

    // ---- Specialize `resume` rooted at the safepoint, rolling on the dynamic cells. ----
    let w = &d.window;
    let slice = |a: u64, len: usize| w[a as usize..a as usize + len].to_vec();
    let pb: Vec<u8> = PROG.iter().flat_map(|x| x.to_le_bytes()).collect();
    let cfg = SpecConfig {
        const_overlays: vec![
            (built.vm_addr, slice(built.vm_addr, VM_SIZE)),
            (built.prog_addr as u64, pb.clone()),
        ],
        rename: Some((built.vm_addr, built.vm_addr + VM_SIZE as u64)),
        rename_is_private: true,
        rename_seed_from_image: true,
        dynamic_cells: d.varying.iter().map(|&a| (a, 8)).collect(),
        indirect_targets_cap: Some(16),
        ..SpecConfig::default()
    };
    let args = [
        SpecArg::ConstI64(built.sp),
        SpecArg::ConstI64(built.prog_addr),
    ];
    let r = specialize_with_config(&built.module, built.resume, &args, &cfg)
        .expect("struct-VM resume rolls");
    temen_verify::verify_module(&r).expect("residual verifies");

    let f = &r.funcs[0];
    let nparams = d.varying.len();
    println!(
        "STRUCT-VM ROLLED: {} blocks -> {} blocks, {} params, br_table={}",
        built.module.funcs[built.resume as usize].blocks.len(),
        f.blocks.len(),
        f.params.len(),
        has_br_table(f),
    );
    assert!(!has_br_table(f), "dispatch folded");
    assert_eq!(
        f.params.len(),
        nparams,
        "residual takes the discovered dynamic cells"
    );
    assert!(
        f.blocks.len() <= 80,
        "the loop rolled ({} blocks)",
        f.blocks.len()
    );

    // `resume` returns the accumulator directly (no write-back wrapper needed). The residual's params
    // are the dynamic cells ascending = [r0 (counter), r1 (accumulator)].
    let counter_ix = d.varying.iter().position(|&a| a == d.counter).unwrap();
    let acc_ix = d
        .varying
        .iter()
        .position(|&a| a == loc.reg_base + 8)
        .unwrap();
    let call = |c: i64, a: i64| {
        let mut args = vec![0i64; nparams];
        args[counter_ix] = c;
        args[acc_ix] = a;
        (tw(&r, 0, &args), jit(&r, 0, &args))
    };

    // The safepoint is body-entered (the loop test already passed once this iteration — the same
    // "in-flight iteration" the Lua rolled residual has), so seeding the counter with N and the
    // accumulator with 0 reproduces a full run of trip count N (for N ≥ 1). Anchor correctness to the
    // interpreter itself, not a hand formula: residual(N, 0) must equal `run(prog, N)` on both backends.
    for nn in [1i64, 5, 50, 500] {
        let want = tw(&built.module, built.run, &[built.sp, built.prog_addr, nn]);
        assert_eq!(
            call(nn, 0),
            (want, want),
            "rolled residual == interpreter @ N={nn}"
        );
    }
    // And the captured seed reproduces the original run (n = 8 ⇒ 7·8).
    let cap_args: Vec<i64> = d.varying.iter().map(|&a| rd_u64(w, a) as i64).collect();
    assert_eq!(tw(&r, 0, &cap_args), 7 * n, "captured tree-walk");
    assert_eq!(jit(&r, 0, &cap_args), 7 * n, "captured jit");
    println!("struct-VM rolled + matches the interpreter across a trip sweep — SAFEPOINT path generalizes");
}
