//! #1110 emit-split prototype — **synthetic-module emitter** for the Node/V8 measurement.
//!
//! V8's Liftoff→TurboFan tier-up is the phenomenon under study, and it is only observable under V8
//! (wasmi/Cranelift are single-tier). So this tool emits self-contained wasm blobs that a Node harness
//! (`split_proto.mjs`) instantiates and times. It builds one synthetic guest:
//!
//! - `f0` — a hot loop `for i in n..0 { acc = fold(acc, helper(i)) }`, whose body is spread across
//!   `hot_blocks` chained blocks so `f0` itself can be made a *large* wasm function (the QuickJS
//!   `JS_CallInternal` shape). This is the "hot path" we want TurboFanned.
//! - `f1` — `helper(x) = 3x + 1`, called every iteration (the intra/cross-module call under test).
//! - `f2..fK` — cold filler functions (never called) that inflate the *module* to multiple MB.
//!
//! It emits four configurations, all sharing the reserved-table ABI so one Node harness drives them:
//!
//! - `single`     — one whole-program B2 module (status quo: `f0` sits in the multi-MB module).
//! - `split_good` — A={f0,f1} | B={filler}: hot path is a *tiny* module, helper intra-module.
//! - `split_bad`  — A={f0} | B={f1,filler}: helper stranded in the cold module (bad partition).
//! - `split_xmod` — A={f0} | B={f1} | C={filler}: f0 and f1 each in their own tiny module — isolates
//!   the pure cross-module `call_indirect` cost.
//!
//! Usage: `cargo run -p temen-wasm-jit --example split_proto -- <out_dir> [n_filler] [filler_len] [hot_blocks] [hot_block_len]`.

use std::path::{Path, PathBuf};
use temen_ir::{BinOp, Block, CmpOp, Func, Inst, IntTy, Memory, Module, Terminator, ValType};
use temen_wasm_jit::{
    compile_module_b2, compile_module_split, compile_module_with, compile_split_fn,
    est_emitted_size,
};

fn i64f() -> Func {
    Func {
        params: vec![ValType::I64],
        results: vec![ValType::I64],
        blocks: vec![],
    }
}

/// `f0(n)`: `acc = 0; for i = n; i != 0; i -= 1 { acc = fold(acc, helper(i)) }; return acc`, where the
/// per-iteration body is spread across `hot_blocks` chained blocks of `hot_block_len` arithmetic ops each.
///
/// Spreading the body across many blocks (rather than one giant block) is what makes `f0` a *large* wasm
/// function while staying under V8's per-function local cap — the shape of QuickJS's `JS_CallInternal`
/// (~1800 blocks, one giant function whose own TurboFan compile is slow). The module-size story does not
/// capture this; the function-size story does. Value numbering is block-local (params first, then each
/// inst result in append order) — the emitter's own convention.
fn hot_loop(hot_blocks: usize, hot_block_len: usize) -> Func {
    let k_chain = hot_blocks.max(1);
    let ops = [BinOp::Add, BinOp::Xor, BinOp::Mul, BinOp::Sub, BinOp::Or];
    let mut f = i64f();
    // block 0 (v0=n): v1 = 0; br 1(n, 0)
    f.blocks.push(Block {
        params: vec![ValType::I64],
        insts: vec![Inst::ConstI64(0)],
        term: Terminator::Br {
            target: 1,
            args: vec![0, 1],
        },
    });
    // block 1 (v0=i, v1=acc): v2 = 0; v3 = (i == 0); br_if v3 -> 2(acc) : 3(i, acc)  [chain starts at 3]
    f.blocks.push(Block {
        params: vec![ValType::I64, ValType::I64],
        insts: vec![
            Inst::ConstI64(0),
            Inst::IntCmp {
                ty: IntTy::I64,
                op: CmpOp::Eq,
                a: 0,
                b: 2,
            },
        ],
        term: Terminator::BrIf {
            cond: 3,
            then_blk: 2,
            then_args: vec![1],
            else_blk: 3,
            else_args: vec![0, 1],
        },
    });
    // block 2 (v0=r): return r
    f.blocks.push(Block {
        params: vec![ValType::I64],
        insts: vec![],
        term: Terminator::Return(vec![0]),
    });
    // Chain blocks 3 .. 3+k_chain-1. Each carries (i, acc). The first calls helper(i) once; every block
    // folds `hot_block_len` ops into acc; the last decrements i and loops back to block 1.
    for j in 0..k_chain {
        let this_blk = 3 + j;
        let mut insts = Vec::new();
        let mut acc: u32 = 1; // v1 = acc
        let mut next: u32 = 2;
        if j == 0 {
            insts.push(Inst::Call {
                func: 1,
                args: vec![0],
            }); // v2 = helper(i)
            insts.push(Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Add,
                a: acc,
                b: next,
            }); // acc + helper
            acc = next + 1;
            next += 2;
        }
        for s in 0..hot_block_len {
            let c = ((this_blk * 131 + s) as i64 * 2654435761).rem_euclid(0xffff) + 1;
            insts.push(Inst::ConstI64(c));
            let cval = next;
            next += 1;
            insts.push(Inst::IntBin {
                ty: IntTy::I64,
                op: ops[(this_blk + s) % ops.len()],
                a: acc,
                b: cval,
            });
            acc = next;
            next += 1;
        }
        let term = if j + 1 < k_chain {
            // Pass (i unchanged = v0, acc') to the next chain block.
            Terminator::Br {
                target: (this_blk + 1) as u32,
                args: vec![0, acc],
            }
        } else {
            // Last chain block: i' = i - 1; br 1(i', acc').
            insts.push(Inst::ConstI64(1));
            let one = next;
            next += 1;
            insts.push(Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Sub,
                a: 0,
                b: one,
            });
            let dec = next;
            Terminator::Br {
                target: 1,
                args: vec![dec, acc],
            }
        };
        f.blocks.push(Block {
            params: vec![ValType::I64, ValType::I64],
            insts,
            term,
        });
    }
    f
}

/// A **call-free** hot loop `f0(n)`, same shape as [`hot_loop`] but the per-iteration body folds inline
/// (no helper call), so the whole hot path is one self-contained function — the input to `compile_split_fn`
/// (the real intra-function splitter), which requires a call-free function.
fn callfree_loop(hot_blocks: usize, hot_block_len: usize) -> Func {
    let k_chain = hot_blocks.max(1);
    let ops = [BinOp::Add, BinOp::Xor, BinOp::Mul, BinOp::Sub, BinOp::Or];
    let mut f = i64f();
    f.blocks.push(Block {
        params: vec![ValType::I64],
        insts: vec![Inst::ConstI64(0)],
        term: Terminator::Br {
            target: 1,
            args: vec![0, 1],
        },
    });
    f.blocks.push(Block {
        params: vec![ValType::I64, ValType::I64],
        insts: vec![
            Inst::ConstI64(0),
            Inst::IntCmp {
                ty: IntTy::I64,
                op: CmpOp::Eq,
                a: 0,
                b: 2,
            },
        ],
        term: Terminator::BrIf {
            cond: 3,
            then_blk: 2,
            then_args: vec![1],
            else_blk: 3,
            else_args: vec![0, 1],
        },
    });
    f.blocks.push(Block {
        params: vec![ValType::I64],
        insts: vec![],
        term: Terminator::Return(vec![0]),
    });
    for j in 0..k_chain {
        let this_blk = 3 + j;
        let mut insts = Vec::new();
        let mut acc: u32 = 1;
        let mut next: u32 = 2;
        for s in 0..hot_block_len {
            let c = ((this_blk * 131 + s) as i64 * 2654435761).rem_euclid(0xffff) + 1;
            insts.push(Inst::ConstI64(c));
            let cval = next;
            next += 1;
            insts.push(Inst::IntBin {
                ty: IntTy::I64,
                op: ops[(this_blk + s) % ops.len()],
                a: acc,
                b: cval,
            });
            acc = next;
            next += 1;
        }
        let term = if j + 1 < k_chain {
            Terminator::Br {
                target: (this_blk + 1) as u32,
                args: vec![0, acc],
            }
        } else {
            insts.push(Inst::ConstI64(1));
            let one = next;
            next += 1;
            insts.push(Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Sub,
                a: 0,
                b: one,
            });
            let dec = next;
            Terminator::Br {
                target: 1,
                args: vec![dec, acc],
            }
        };
        f.blocks.push(Block {
            params: vec![ValType::I64, ValType::I64],
            insts,
            term,
        });
    }
    f
}

/// `helper(x) = 3*x + 1`.
fn helper() -> Func {
    let mut f = i64f();
    f.blocks.push(Block {
        params: vec![ValType::I64],
        insts: vec![
            Inst::ConstI64(3),
            Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Mul,
                a: 0,
                b: 1,
            },
            Inst::ConstI64(1),
            Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Add,
                a: 2,
                b: 3,
            },
        ],
        term: Terminator::Return(vec![4]),
    });
    f
}

/// A cold filler `(i64)->(i64)`: a chain of `len` mixed arithmetic ops over the incoming value. Never
/// called — pure module bloat, so V8 keeps the whole thing on Liftoff longer. `seed` varies the constants
/// so the bodies differ (defeats any dedup and mimics distinct real functions).
fn filler(len: usize, seed: u64) -> Func {
    let mut f = i64f();
    let ops = [BinOp::Add, BinOp::Xor, BinOp::Mul, BinOp::Sub, BinOp::Or];
    let mut insts = Vec::with_capacity(len * 2);
    // Value 0 = param x. Each step appends a const (new value) then a binop folding it into the running
    // accumulator (the most-recent value). Track the accumulator's value index.
    let mut acc: u32 = 0;
    let mut next: u32 = 1;
    for k in 0..len {
        let c = (seed.wrapping_mul(2654435761).wrapping_add(k as u64) & 0xffff) as i64 + 1;
        insts.push(Inst::ConstI64(c)); // value `next`
        let cval = next;
        next += 1;
        insts.push(Inst::IntBin {
            ty: IntTy::I64,
            op: ops[k % ops.len()],
            a: acc,
            b: cval,
        }); // value `next`
        acc = next;
        next += 1;
    }
    f.blocks.push(Block {
        params: vec![ValType::I64],
        insts,
        term: Terminator::Return(vec![acc]),
    });
    f
}

fn build(n_filler: usize, filler_len: usize, hot_blocks: usize, hot_block_len: usize) -> Module {
    let mut funcs = vec![hot_loop(hot_blocks, hot_block_len), helper()];
    for i in 0..n_filler {
        funcs.push(filler(filler_len, i as u64 + 1));
    }
    Module {
        funcs,
        memory: Some(Memory { size_log2: 16 }),
        ..Default::default()
    }
}

/// A **handler** `(i64)->(i64)`: `blocks` chained blocks of `block_len` folds over the running value —
/// the same body shape as a chunk of `hot_loop`, but as its own function. `blocks/block_len` are sized so
/// K handlers together equal one monolithic `hot_loop` body: this is the *outlined* form of the hot path.
fn chain(blocks: usize, block_len: usize, seed: u64) -> Func {
    let b = blocks.max(1);
    let ops = [BinOp::Add, BinOp::Xor, BinOp::Mul, BinOp::Sub, BinOp::Or];
    let mut f = i64f();
    for j in 0..b {
        let mut insts = Vec::new();
        let mut acc: u32 = 0; // v0 = incoming value
        let mut next: u32 = 1;
        for s in 0..block_len {
            let c = ((seed as usize * 131 + j * 17 + s) as i64 * 2654435761).rem_euclid(0xffff) + 1;
            insts.push(Inst::ConstI64(c));
            let cval = next;
            next += 1;
            insts.push(Inst::IntBin {
                ty: IntTy::I64,
                op: ops[(j + s) % ops.len()],
                a: acc,
                b: cval,
            });
            acc = next;
            next += 1;
        }
        let term = if j + 1 < b {
            Terminator::Br {
                target: (j + 1) as u32,
                args: vec![acc],
            }
        } else {
            Terminator::Return(vec![acc])
        };
        f.blocks.push(Block {
            params: vec![ValType::I64],
            insts,
            term,
        });
    }
    f
}

/// The **outlined** hot path: a small dispatcher `f0(n)` whose per-iteration body calls each of `k`
/// handler functions (`h_1..h_k`) in sequence, threading `acc` through. The k handlers hold the bulk of
/// the compute — each ≈ `total_blocks/k` blocks — so the hot code is spread across k smaller functions
/// instead of one giant `f0`. This is the transform under evaluation: does V8 tier the (now smaller)
/// hot functions up sooner in aggregate than the one monolithic function of the same total size?
fn build_outlined(
    n_filler: usize,
    filler_len: usize,
    total_blocks: usize,
    block_len: usize,
    k: usize,
) -> Module {
    let k = k.max(2);
    let per = total_blocks.div_ceil(k);
    // func 0 = dispatcher; funcs 1..=k = handlers; then filler.
    let mut funcs = vec![outlined_dispatcher(k)];
    for j in 0..k {
        funcs.push(chain(per, block_len, j as u64 + 1));
    }
    for i in 0..n_filler {
        funcs.push(filler(filler_len, i as u64 + 1));
    }
    Module {
        funcs,
        memory: Some(Memory { size_log2: 16 }),
        ..Default::default()
    }
}

/// `f0(n)` dispatcher: `acc=0; for i=n; i!=0; i-=1 { acc = h_k(...h_2(h_1(acc))...) }; return acc`.
fn outlined_dispatcher(k: usize) -> Func {
    let mut f = i64f();
    // block 0 (v0=n): acc=0; br 1(n, 0)
    f.blocks.push(Block {
        params: vec![ValType::I64],
        insts: vec![Inst::ConstI64(0)],
        term: Terminator::Br {
            target: 1,
            args: vec![0, 1],
        },
    });
    // block 1 (v0=i, v1=acc): i==0 ? return acc : body
    f.blocks.push(Block {
        params: vec![ValType::I64, ValType::I64],
        insts: vec![
            Inst::ConstI64(0),
            Inst::IntCmp {
                ty: IntTy::I64,
                op: CmpOp::Eq,
                a: 0,
                b: 2,
            },
        ],
        term: Terminator::BrIf {
            cond: 3,
            then_blk: 2,
            then_args: vec![1],
            else_blk: 3,
            else_args: vec![0, 1],
        },
    });
    // block 2 (v0=r): return r
    f.blocks.push(Block {
        params: vec![ValType::I64],
        insts: vec![],
        term: Terminator::Return(vec![0]),
    });
    // block 3 (v0=i, v1=acc): acc = h_k(...h_1(acc)...); i-=1; br 1(i-1, acc)
    let mut insts = Vec::new();
    let mut acc: u32 = 1; // start from incoming acc (v1)
    let mut next: u32 = 2;
    for j in 0..k {
        insts.push(Inst::Call {
            func: (1 + j) as u32, // handler h_{j+1}
            args: vec![acc],
        });
        acc = next;
        next += 1;
    }
    insts.push(Inst::ConstI64(1));
    let one = next;
    next += 1;
    insts.push(Inst::IntBin {
        ty: IntTy::I64,
        op: BinOp::Sub,
        a: 0,
        b: one,
    });
    let dec = next;
    f.blocks.push(Block {
        params: vec![ValType::I64, ValType::I64],
        insts,
        term: Terminator::Br {
            target: 1,
            args: vec![dec, acc],
        },
    });
    f
}

fn table_log2_for(n: usize) -> u32 {
    let mut log2 = 1u32;
    while (1usize << log2) < n {
        log2 += 1;
    }
    log2
}

fn write(dir: &Path, name: &str, bytes: &[u8]) {
    let p = dir.join(name);
    std::fs::write(&p, bytes).unwrap_or_else(|e| panic!("write {}: {e}", p.display()));
    eprintln!("  {:<22} {:>10} bytes", name, bytes.len());
}

fn main() {
    let mut args = std::env::args().skip(1);
    let out_dir = PathBuf::from(args.next().unwrap_or_else(|| ".".to_string()));
    let n_filler: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(400);
    let filler_len: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(1500);
    let hot_blocks: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(1);
    let hot_block_len: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    let outline: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(1);
    let splitfn: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    std::fs::create_dir_all(&out_dir).unwrap();

    // splitfn mode (`splitfn >= 2`): the REAL intra-function splitter. Build ONE call-free hot-loop
    // function, emit it monolithic (`single`) and split into `splitfn` block-groups (`splitfn`), and
    // measure both under V8 — confirming `compile_split_fn` inherits the tier-up win.
    if splitfn >= 2 {
        let f = callfree_loop(hot_blocks, hot_block_len);
        let m = Module {
            funcs: vec![f],
            memory: Some(Memory { size_log2: 16 }),
            ..Default::default()
        };
        let mono = compile_module_with(&m, false).expect("monolithic emits");
        let split = compile_split_fn(&m, 0, splitfn, false).expect("split-fn emits");
        eprintln!(
            "splitfn guest: 1 call-free function, {hot_blocks} blocks × {hot_block_len} ops; \
             monolithic {} B, split into {splitfn} groups {} B",
            mono.len(),
            split.len()
        );
        write(&out_dir, "single.wasm", &mono);
        write(&out_dir, "splitfn.wasm", &split);
        let manifest = concat!(
            "{\n  \"table_log2\": 1,\n  \"n_funcs\": 1,\n  \"entry\": 0,\n  \"configs\": {\n",
            "    \"single\":  [{\"wasm\": \"single.wasm\",  \"funcs\": [0]}],\n",
            "    \"splitfn\": [{\"wasm\": \"splitfn.wasm\", \"funcs\": [0]}]\n",
            "  }\n}\n"
        );
        write(&out_dir, "manifest.json", manifest.as_bytes());
        eprintln!("done → {}", out_dir.display());
        return;
    }

    // Outlined mode (`outline >= 2`): emit just the `single` config, but with the hot path spread across
    // `outline` handler functions instead of one monolithic `f0` — the function-outlining experiment.
    // Compare its tier-up run (via the harness `single` config) against a monolithic run of the same
    // total hot size. Same manifest shape, one config.
    if outline >= 2 {
        let m = build_outlined(n_filler, filler_len, hot_blocks, hot_block_len, outline);
        let n = m.funcs.len();
        let log2 = table_log2_for(n);
        let est: usize = m.funcs.iter().map(est_emitted_size).sum();
        let per_handler = est_emitted_size(&m.funcs[1]);
        eprintln!(
            "outlined guest: dispatcher + {outline} handlers ({} blocks each) + {n_filler} filler; per-handler est {:.2} MB, est_emitted {:.2} MB, table_log2 {log2}",
            hot_blocks.div_ceil(outline),
            per_handler as f64 / (1 << 20) as f64,
            est as f64 / (1 << 20) as f64
        );
        // `single` (direct): dispatcher + handlers in one module — TurboFan can inline the handler calls.
        let single = compile_module_b2(&m, false, log2).expect("outlined single emits");
        write(&out_dir, "single.wasm", &single);
        // `indirect`: dispatcher alone | all handlers | filler. The dispatcher→handler calls become
        // cross-module `call_indirect` through the shared table — the real per-opcode dispatch shape,
        // which TurboFan cannot inline. Isolates the indirect-dispatch tax vs `single` (Slice 0).
        let disp_mask: Vec<bool> = (0..n).map(|i| i == 0).collect(); // {dispatcher}
        let hand_mask: Vec<bool> = (0..n).map(|i| i >= 1 && i <= outline).collect(); // {h_1..h_K}
        let cold_mask: Vec<bool> = (0..n).map(|i| i > outline).collect(); // {filler}
        let disp = compile_module_split(&m, false, log2, &disp_mask).expect("disp emits");
        let hand = compile_module_split(&m, false, log2, &hand_mask).expect("handlers emit");
        let cold = compile_module_split(&m, false, log2, &cold_mask).expect("cold emits");
        write(&out_dir, "disp.wasm", &disp);
        write(&out_dir, "handlers.wasm", &hand);
        write(&out_dir, "ocold.wasm", &cold);
        let idxs = |mask: &[bool]| -> String {
            (0..n)
                .filter(|&i| mask[i])
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(",")
        };
        let all: Vec<String> = (0..n).map(|i| i.to_string()).collect();
        let manifest = format!(
            concat!(
                "{{\n  \"table_log2\": {log2},\n  \"n_funcs\": {n},\n  \"entry\": 0,\n  \"configs\": {{\n",
                "    \"single\":   [{{\"wasm\": \"single.wasm\", \"funcs\": [{all}]}}],\n",
                "    \"indirect\": [{{\"wasm\": \"disp.wasm\", \"funcs\": [{d}]}}, {{\"wasm\": \"handlers.wasm\", \"funcs\": [{h}]}}, {{\"wasm\": \"ocold.wasm\", \"funcs\": [{c}]}}]\n",
                "  }}\n}}\n"
            ),
            log2 = log2,
            n = n,
            all = all.join(","),
            d = idxs(&disp_mask),
            h = idxs(&hand_mask),
            c = idxs(&cold_mask),
        );
        write(&out_dir, "manifest.json", manifest.as_bytes());
        eprintln!("done → {}", out_dir.display());
        return;
    }

    let m = build(n_filler, filler_len, hot_blocks, hot_block_len);
    let n = m.funcs.len();
    let log2 = table_log2_for(n);
    let est: usize = m.funcs.iter().map(est_emitted_size).sum();
    let hot_est = est_emitted_size(&m.funcs[0]);
    eprintln!(
        "synthetic guest: {n} funcs ({n_filler} filler × {filler_len} ops), hot f0: {hot_blocks} blocks × {hot_block_len} ops (est {:.2} MB), est_emitted {:.2} MB, table_log2 {log2}",
        hot_est as f64 / (1 << 20) as f64,
        est as f64 / (1 << 20) as f64
    );

    // Partitions. A slot's owner is whichever partition has `true` at that index.
    let mut a_good = vec![false; n]; // {f0, f1}
    a_good[0] = true;
    a_good[1] = true;
    let b_good: Vec<bool> = a_good.iter().map(|&x| !x).collect();

    let mut a_bad = vec![false; n]; // {f0}
    a_bad[0] = true;
    let b_bad: Vec<bool> = a_bad.iter().map(|&x| !x).collect();

    eprintln!("emitting single (whole-program B2):");
    let single = compile_module_b2(&m, false, log2).expect("single emits");
    write(&out_dir, "single.wasm", &single);

    eprintln!("emitting split_good (A={{f0,f1}} | B={{filler}}):");
    let ga = compile_module_split(&m, false, log2, &a_good).expect("good A emits");
    let gb = compile_module_split(&m, false, log2, &b_good).expect("good B emits");
    write(&out_dir, "split_good_hot.wasm", &ga);
    write(&out_dir, "split_good_cold.wasm", &gb);

    eprintln!("emitting split_bad (A={{f0}} | B={{f1,filler}}):");
    let ba = compile_module_split(&m, false, log2, &a_bad).expect("bad A emits");
    let bb = compile_module_split(&m, false, log2, &b_bad).expect("bad B emits");
    write(&out_dir, "split_bad_hot.wasm", &ba);
    write(&out_dir, "split_bad_cold.wasm", &bb);

    // 3-way: f0 | f1 | filler. f0 and f1 each land in their own tiny module (both TurboFan fast), so the
    // only difference from split_good is that f0→f1 is a **cross-module** call — isolates that call's cost
    // from the "hot callee stranded in the cold module" penalty that split_bad also carries.
    eprintln!("emitting split_xmod (A={{f0}} | B={{f1}} | C={{filler}}):");
    let mut x_f0 = vec![false; n];
    x_f0[0] = true;
    let mut x_f1 = vec![false; n];
    x_f1[1] = true;
    let x_fill: Vec<bool> = (0..n).map(|i| i >= 2).collect();
    let xa = compile_module_split(&m, false, log2, &x_f0).expect("xmod f0 emits");
    let xb = compile_module_split(&m, false, log2, &x_f1).expect("xmod f1 emits");
    let xc = compile_module_split(&m, false, log2, &x_fill).expect("xmod filler emits");
    write(&out_dir, "split_xmod_f0.wasm", &xa);
    write(&out_dir, "split_xmod_f1.wasm", &xb);
    write(&out_dir, "split_xmod_cold.wasm", &xc);

    // Manifest for the Node harness: which temen indices each blob emits (→ table slots), plus the ABI.
    let idxs = |mask: &[bool]| -> String {
        (0..n)
            .filter(|&i| mask[i])
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(",")
    };
    let all: Vec<bool> = vec![true; n];
    let manifest = format!(
        concat!(
            "{{\n",
            "  \"table_log2\": {log2},\n",
            "  \"n_funcs\": {n},\n",
            "  \"entry\": 0,\n",
            "  \"configs\": {{\n",
            "    \"single\":     [{{\"wasm\": \"single.wasm\",         \"funcs\": [{single_f}]}}],\n",
            "    \"split_good\": [{{\"wasm\": \"split_good_hot.wasm\",  \"funcs\": [{ga_f}]}}, {{\"wasm\": \"split_good_cold.wasm\", \"funcs\": [{gb_f}]}}],\n",
            "    \"split_bad\":  [{{\"wasm\": \"split_bad_hot.wasm\",   \"funcs\": [{ba_f}]}}, {{\"wasm\": \"split_bad_cold.wasm\",  \"funcs\": [{bb_f}]}}],\n",
            "    \"split_xmod\": [{{\"wasm\": \"split_xmod_f0.wasm\",   \"funcs\": [{xa_f}]}}, {{\"wasm\": \"split_xmod_f1.wasm\",   \"funcs\": [{xb_f}]}}, {{\"wasm\": \"split_xmod_cold.wasm\", \"funcs\": [{xc_f}]}}]\n",
            "  }}\n",
            "}}\n"
        ),
        log2 = log2,
        n = n,
        single_f = idxs(&all),
        ga_f = idxs(&a_good),
        gb_f = idxs(&b_good),
        ba_f = idxs(&a_bad),
        bb_f = idxs(&b_bad),
        xa_f = idxs(&x_f0),
        xb_f = idxs(&x_f1),
        xc_f = idxs(&x_fill),
    );
    write(&out_dir, "manifest.json", manifest.as_bytes());
    eprintln!("done → {}", out_dir.display());
}
