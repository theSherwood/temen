//! **Futamura projection of a register-machine bytecode VM** — the Lua / QuickJS execution model
//! (a register file + a constant program + a decode-and-dispatch loop), a step up from the
//! tape-machine Brainfuck demo (`crates/temen-llvm/tests/peval_demo.rs`).
//!
//! A real register VM written in C is compiled `clang -O2 → LLVM → temen-llvm → temen-IR`, then
//! partial-evaluated by `temen-peval` against a fixed bytecode program (the `prog` pointer is opaque to
//! clang, but we declare it + its readonly bytes constant — the §20c caller contract). The residual
//! is the *compiled* program: the opcode decode and the `br_table` dispatch fold away, leaving the
//! data-dependent loop over the register file. We measure the residual against the un-specialized
//! interpreter across **all four backends** (tree-walk, bytecode, temen-jit, temen-wasm-jit on Wasmtime).
//!
//! Run: `cargo run --release --manifest-path bench/Cargo.toml --bin peval_regvm`
//!
//! **Why a register VM, not real Lua.** `luaV_execute` in the lua on-ramp is an 841-block function
//! whose bytecode is reached through the runtime heap (`L → ci → closure → Proto → code`), not a
//! pointer argument, and which calls host functions / the GC. Folding it needs partial-execution to
//! materialize the heap bytecode as a constant region plus a much larger supported op subset — a
//! separate, larger harness. This VM captures the *execution model* (register dispatch) that makes
//! the specialization win real, on a program the current engine handles end to end.

use std::hint::black_box;
use std::process::Command;
use std::time::{Duration, Instant};

use temen_ir::{Module, ValType, DEFAULT_RESERVED_LOG2};
use temen_jit::{CompiledModule, JitOutcome, Quota, INERT_CAP_THUNK};
use temen_peval::{optimize_module, specialize, SpecArg};
use temen_verify::verify_module;

const WIN_BASE: u32 = 0x1_0000;
const ENV_PTR: u32 = 1024;

/// A register-machine bytecode interpreter. Fixed-width instructions `(op, a, b, c)` in a flat
/// `int` array; a 16-slot `long` register file; `prog` is an **opaque pointer** clang cannot fold.
/// The bundled program computes `3 * input` via a runtime-count loop, so the residual is a loop
/// (not fully unrolled): r0 = counter (input), r1 = accumulator, r2 = 0.
const VM_C: &str = r#"
enum { HALT, LOADI, LOADIN, ADD, SUB, ADDK, JLE, JMP };
static long reg[16];   /* zero-initialized at load; the program sets each register before reading it */
long run(const int *prog, long input) {
    long pc = 0;
    for (;;) {
        int op = prog[pc], a = prog[pc+1], b = prog[pc+2], c = prog[pc+3];
        if (op == HALT) return reg[a];
        else if (op == LOADI)  reg[a] = b;
        else if (op == LOADIN) reg[a] = input;
        else if (op == ADD)    reg[a] = reg[b] + reg[c];
        else if (op == SUB)    reg[a] = reg[b] - reg[c];
        else if (op == ADDK)   reg[a] = reg[b] + c;
        else if (op == JLE)  { if (reg[a] <= reg[b]) { pc = (long)c * 4; continue; } }
        else if (op == JMP)  { pc = (long)a * 4; continue; }
        pc += 4;
    }
}
/* r0=counter r1=acc r2=zero.  acc += 3 while counter-- > 0  =>  acc = 3*input. */
static const int prog[] = {
    LOADIN, 0, 0,  0,   /* 0: r0 = input          */
    LOADI,  1, 0,  0,   /* 1: r1 = 0              */
    LOADI,  2, 0,  0,   /* 2: r2 = 0              */
    JLE,    0, 2,  7,   /* 3: if r0<=0 goto 7      (loop header) */
    ADDK,   1, 1,  3,   /* 4: r1 += 3             */
    ADDK,   0, 0, -1,   /* 5: r0 -= 1             */
    JMP,    3, 0,  0,   /* 6: goto 3              */
    HALT,   1, 0,  0,   /* 7: return r1           */
};
int main(void) { return (int)run(prog, 0); }
"#;

/// The program bytes as they appear in the readonly segment (little-endian `int`s).
fn prog_bytes() -> Vec<u8> {
    let words: [i32; 32] = [
        2, 0, 0, 0, 1, 1, 0, 0, 1, 2, 0, 0, 6, 0, 2, 7, 5, 1, 1, 3, 5, 0, 0, -1, 7, 3, 0, 0, 0, 1,
        0, 0,
    ];
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn compile_c_to_temen(src: &str) -> (Module, i64) {
    let base = std::env::temp_dir().join("peval_regvm");
    let cf = base.with_extension("c");
    let ll = base.with_extension("ll");
    std::fs::write(&cf, src).unwrap();
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
        .unwrap()
        .success();
    assert!(ok, "clang failed (need clang on PATH)");
    let t = temen_llvm::translate_ll_path(&ll).expect("temen-llvm translate");
    let _ = std::fs::remove_file(&cf);
    let _ = std::fs::remove_file(&ll);
    (t.module, t.entry_sp as i64)
}

fn run_entry(m: &Module) -> u32 {
    // temen-llvm prepends the data-stack pointer, so `run(prog, input)` is `run(sp, prog, input)`.
    m.exports
        .iter()
        .find(|e| e.name == "run")
        .map(|e| e.func)
        .or_else(|| {
            m.funcs
                .iter()
                .position(|f| {
                    f.params == [ValType::I64, ValType::I64, ValType::I64]
                        && f.results == [ValType::I64]
                })
                .map(|i| i as u32)
        })
        .expect("run(i64,i64,i64)->i64 entry")
}

/// The window address of the readonly segment holding the program.
fn prog_addr(m: &Module) -> i64 {
    let pb = prog_bytes();
    m.data
        .iter()
        .find(|d| d.readonly && d.bytes.windows(pb.len()).any(|w| w == pb.as_slice()))
        .map(|d| {
            let off = d
                .bytes
                .windows(pb.len())
                .position(|w| w == pb.as_slice())
                .unwrap();
            d.offset as i64 + off as i64
        })
        .expect("readonly program segment")
}

// ---- backend runners ----

fn tree_walk(m: &Module, entry: u32, args: &[i64]) -> i64 {
    let mut fuel = u64::MAX;
    let vs: Vec<temen_interp::Value> = args.iter().map(|&a| temen_interp::Value::I64(a)).collect();
    match temen_interp::run(m, entry, &vs, &mut fuel) {
        Ok(v) => match v.as_slice() {
            [temen_interp::Value::I64(x)] => *x,
            o => panic!("bad tree-walk result {o:?}"),
        },
        Err(t) => panic!("tree-walk trapped: {t:?}"),
    }
}

fn bytecode(m: &Module, entry: u32, args: &[i64]) -> i64 {
    let mut fuel = u64::MAX;
    let vs: Vec<temen_interp::Value> = args.iter().map(|&a| temen_interp::Value::I64(a)).collect();
    match temen_interp::bytecode::compile_and_run(m, entry, &vs, &mut fuel) {
        Some(Ok(v)) => match v.as_slice() {
            [temen_interp::Value::I64(x)] => *x,
            o => panic!("bad bytecode result {o:?}"),
        },
        Some(Err(t)) => panic!("bytecode trapped: {t:?}"),
        None => panic!("bytecode declined the module"),
    }
}

fn jit_run(m: &Module, entry: u32, args: &[i64]) -> i64 {
    match temen_jit::compile_and_run(m, entry, args) {
        Ok(JitOutcome::Returned(v)) => v[0],
        o => panic!("jit outcome {o:?}"),
    }
}

fn jit_compile(m: &Module, entry: u32) -> CompiledModule {
    CompiledModule::compile(
        m,
        entry,
        INERT_CAP_THUNK,
        std::ptr::null_mut(),
        DEFAULT_RESERVED_LOG2,
        None,
        None,
        None,
        None,
        None,
        Quota::default(),
        0,
    )
    .expect("jit compile")
}

fn jit_call(cm: &mut CompiledModule, args: &[i64]) -> i64 {
    match cm.run(args, None, None) {
        Ok((JitOutcome::Returned(v), _)) => v[0],
        o => panic!("jit outcome {o:?}"),
    }
}

/// Compile with temen-wasm-jit and run `f{entry}` on Wasmtime with the confined-window ABI.
fn wasmjit_prepare(m: &Module, entry: u32) -> (wasmtime::Store<i32>, wasmtime::Instance, String) {
    use wasmtime::{Caller, Engine, Linker, Memory, MemoryType, Module as WModule, Store};
    let wasm = temen_wasm_jit::compile_module(m).expect("wasm-jit emit");
    let engine = Engine::default();
    let module = WModule::new(&engine, &wasm).expect("emitted wasm validates");
    let mut store: Store<i32> = Store::new(&engine, 0);
    let win_pages = m.memory.map_or(1, |mc| (mc.size() >> 16) as u32 + 1);
    let memory = Memory::new(&mut store, MemoryType::new(2 + win_pages, None)).unwrap();
    memory
        .write(&mut store, ENV_PTR as usize, &(1i64 << 52).to_le_bytes())
        .unwrap();
    // Seed the window from the module's data segments (each at WIN_BASE + offset) so the interpreter
    // reads its readonly program — what the browser / bench linkers do (temen-wasm-jit entry_data test).
    for seg in &m.data {
        memory
            .write(
                &mut store,
                WIN_BASE as usize + seg.offset as usize,
                &seg.bytes,
            )
            .expect("window init: data segment fits");
    }
    let mut linker: Linker<i32> = Linker::new(&engine);
    linker.define(&store, "env", "memory", memory).unwrap();
    linker
        .func_wrap("env", "trap", |mut c: Caller<'_, i32>, code: i32| {
            *c.data_mut() = code
        })
        .unwrap();
    linker
        .func_wrap::<_, ()>(
            "env",
            "call_interp",
            |_: Caller<'_, i32>, _f: i32, _a: i32| {
                unreachable!("no cross-tier call in this kernel")
            },
        )
        .unwrap();
    let instance = linker
        .instantiate(&mut store, &module)
        .expect("instantiate");
    (store, instance, format!("f{entry}"))
}

fn wasmjit_run(
    store: &mut wasmtime::Store<i32>,
    inst: &wasmtime::Instance,
    fname: &str,
    args: &[i64],
) -> i64 {
    use wasmtime::Val;
    let f = inst.get_func(&mut *store, fname).expect("entry exported");
    let mut params = vec![Val::I32(WIN_BASE as i32), Val::I32(ENV_PTR as i32)];
    params.extend(args.iter().map(|&a| Val::I64(a)));
    let mut results = [Val::I64(0)];
    f.call(&mut *store, &params, &mut results).expect("call");
    match results[0] {
        Val::I64(x) => x,
        o => panic!("bad wasm result {o:?}"),
    }
}

fn best_of(reps: usize, mut f: impl FnMut() -> i64) -> Duration {
    f();
    let mut best = Duration::MAX;
    for _ in 0..reps {
        let t = Instant::now();
        black_box(f());
        best = best.min(t.elapsed());
    }
    best
}

fn blocks(m: &Module, entry: u32) -> usize {
    m.funcs[entry as usize].blocks.len()
}

fn main() {
    let (m, sp) = compile_c_to_temen(VM_C);
    verify_module(&m).expect("interpreter verifies");
    let entry = run_entry(&m);
    let pa = prog_addr(&m);

    let residual = optimize_module(
        &specialize(
            &m,
            entry,
            &[
                SpecArg::ConstI64(sp),
                SpecArg::ConstI64(pa),
                SpecArg::Dynamic,
            ],
        )
        .expect("specializes the register VM"),
    );
    verify_module(&residual).expect("residual re-verifies");

    // Correctness across all four backends before timing (small inputs).
    for input in [0i64, 1, 2, 7, 100, 1000] {
        let want = 3 * input;
        assert_eq!(
            tree_walk(&m, entry, &[sp, pa, input]),
            want,
            "interpreter wrong @ {input}"
        );
        assert_eq!(
            tree_walk(&residual, 0, &[input]),
            want,
            "treewalk residual @ {input}"
        );
        assert_eq!(
            bytecode(&residual, 0, &[input]),
            want,
            "bytecode residual @ {input}"
        );
        assert_eq!(
            jit_run(&residual, 0, &[input]),
            want,
            "temen-jit residual @ {input}"
        );
        let (mut si, ii, fi) = wasmjit_prepare(&m, entry);
        assert_eq!(
            wasmjit_run(&mut si, &ii, &fi, &[sp, pa, input]),
            want,
            "wasm-jit interp @ {input}"
        );
        let (mut s, i, fname) = wasmjit_prepare(&residual, 0);
        assert_eq!(
            wasmjit_run(&mut s, &i, &fname, &[input]),
            want,
            "wasm-jit residual @ {input}"
        );
    }

    let n: i64 = 2_000_000;
    let reps = 9;

    let mut cm_i = jit_compile(&m, entry);
    let mut cm_r = jit_compile(&residual, 0);
    let (mut wi_s, wi_i, wi_f) = wasmjit_prepare(&m, entry);
    let (mut wr_s, wr_i, wr_f) = wasmjit_prepare(&residual, 0);

    let rows = [
        (
            "tree-walk   ",
            best_of(reps, || tree_walk(&m, entry, &[sp, pa, n])),
            best_of(reps, || tree_walk(&residual, 0, &[n])),
        ),
        (
            "bytecode    ",
            best_of(reps, || bytecode(&m, entry, &[sp, pa, n])),
            best_of(reps, || bytecode(&residual, 0, &[n])),
        ),
        (
            "temen-jit     ",
            best_of(reps, || jit_call(&mut cm_i, &[sp, pa, n])),
            best_of(reps, || jit_call(&mut cm_r, &[n])),
        ),
        (
            "temen-wasm-jit",
            best_of(reps, || wasmjit_run(&mut wi_s, &wi_i, &wi_f, &[sp, pa, n])),
            best_of(reps, || wasmjit_run(&mut wr_s, &wr_i, &wr_f, &[n])),
        ),
    ];

    println!(
        "\n=== register-VM Futamura ROI across all backends (3*n loop, n={n}, best of {reps}) ==="
    );
    println!(
        "interpreter run(): {} blocks   residual: {} blocks\n",
        blocks(&m, entry),
        blocks(&residual, 0)
    );
    println!(
        "{:<14} {:>12} {:>12} {:>9}",
        "backend", "interpreter", "residual", "speedup"
    );
    println!("{}", "-".repeat(50));
    for (name, interp, res) in rows {
        println!(
            "{name} {interp:>12.3?} {res:>12.3?} {:>8.2}x",
            interp.as_secs_f64() / res.as_secs_f64()
        );
    }
}
