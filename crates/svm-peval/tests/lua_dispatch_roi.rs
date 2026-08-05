//! **Measuring the ROI of the Futamura projection's core win: decode + dispatch elimination.**
//!
//! Projecting the *whole* `luaV_execute` is blocked before its loop starts by Lua's stack-management
//! substrate (see `svm-llvm/tests/lua_futamura_specialize.rs`). But the win a Futamura projection is
//! *for* — collapsing the per-opcode fetch + decode + N-way dispatch so only the opcode bodies remain
//! — is separable, and this test measures it on the **real captured Lua 5.4.7 bytecode** for
//!
//!     local x = 0; for i = 1, 50 do x = x + 3 end; return x
//!
//! We build a **faithful dispatch skeleton**: a byte-for-byte model of `luaV_execute`'s inner loop —
//! fetch the 32-bit instruction from a (readonly, so foldable) bytecode array, decode the 7-bit
//! opcode, and dispatch through an **83-way `br_table`** (Lua 5.4 has 83 opcodes) to per-opcode
//! handlers. The handlers decode operands exactly as Lua does (A = bits 7..14, sBx = Bx − 65535, ADDI
//! immediate = C − 127) and `ADDI` does the real `pc += 2` that skips the trailing `MMBIN`. The
//! opcode stream is the real one: `VARARGPREP, LOADI×4, FORPREP, ADDI, (MMBIN skipped), FORLOOP,
//! RETURN`.
//!
//! Two deliberate, documented simplifications keep the measurement *about dispatch* and nothing else:
//!   • Registers are plain `i64` (no `TValue` tag byte) — tag-checking is the rename-region concern,
//!     orthogonal to dispatch.
//!   • The accumulator's initial value is a **runtime input** (seeded at the loop preheader, in the
//!     `FORPREP` handler) rather than the baked `0`. This keeps the loop body's arithmetic in the
//!     residual — otherwise the whole constant program would fold to `return 150`, conflating dispatch
//!     elimination with plain constant-folding.
//!
//! The result is the ideal projection shape: the specializer residualizes the loop counter and emits
//! a **clean dispatch-free loop** — `pc` folds to a constant at every point, so the fetch, the opcode
//! mask, and the 83-way `br_table` all vanish, leaving only the opcode bodies (load/add/store the
//! accumulator; decrement/test the counter) wrapped in the surviving back-edge.
//!
//! The ROI is then the gap between running the **same skeleton** two ways with the same input:
//!   • **baseline** — through the interpreter, paying fetch/decode/dispatch every opcode;
//!   • **specialized** — with the bytecode folded constant, so the specializer collapses all of it.
//! We assert both compute the same answer, then report the per-iteration instruction drop and that the
//! fetch + `br_table` are gone entirely.
//!
//! Run: `cargo test -p svm-peval --test lua_dispatch_roi -- --nocapture`

use svm_interp::Value;
use svm_ir::{Data, Inst, LoadOp, Module, Terminator};
use svm_peval::{specialize_with_config, SpecArg, SpecConfig};
use svm_text::parse_module;
use svm_verify::verify_module;

/// The real Lua 5.4.7 bytecode for the script (captured at `luaV_execute` entry in
/// `svm-llvm/tests/lua_futamura_specialize.rs`; opcodes verified there).
const BYTECODE: [u32; 11] = [
    0x0000_0051, // [0] VARARGPREP
    0x7fff_8001, // [1] LOADI  A=0 sBx=0    (x = 0; overridden by the runtime seed at FORPREP)
    0x8000_0081, // [2] LOADI  A=1 sBx=1    (for init)
    0x8018_8101, // [3] LOADI  A=2 sBx=50   (for limit)
    0x8000_0181, // [4] LOADI  A=3 sBx=1    (for step)
    0x0001_00ca, // [5] FORPREP A=1 Bx=2
    0x8200_0015, // [6] ADDI   A=0 C=130 → imm = 130 − 127 = 3
    0x0682_002f, // [7] MMBINI (never dispatched — ADDI does pc += 2)
    0x0001_80c9, // [8] FORLOOP A=1 Bx=3
    0x0102_0046, // [9] RETURN  A=0
    0x0101_00c6, // [10] RETURN A=1
];

const BYTECODE_BASE: u64 = 4096;
const LOOP_TRIPS: i64 = 50;

// Lua 5.4 opcode numbers we route to real handlers; every other index → the trap default.
const OP_LOADI: usize = 1;
const OP_ADDI: usize = 21;
const OP_RETURN: usize = 70;
const OP_FORLOOP: usize = 73;
const OP_FORPREP: usize = 74;
const OP_VARARGPREP: usize = 81;

// Handler block numbers in the skeleton (see `skeleton_src`).
const B_VARARGPREP: u32 = 2;
const B_LOADI: u32 = 3;
const B_FORPREP: u32 = 4;
const B_ADDI: u32 = 5;
const B_FORLOOP: u32 = 6;
const B_RETURN: u32 = 7;
const B_DEFAULT: u32 = 8;

/// The 83-entry dispatch table as an edge list for the `br_table`: opcode index → handler block,
/// each edge forwarding `(vpc, vinstr)`.
fn dispatch_edges() -> String {
    let mut edges = Vec::with_capacity(83);
    for op in 0..83usize {
        let blk = match op {
            OP_VARARGPREP => B_VARARGPREP,
            OP_LOADI => B_LOADI,
            OP_FORPREP => B_FORPREP,
            OP_ADDI => B_ADDI,
            OP_FORLOOP => B_FORLOOP,
            OP_RETURN => B_RETURN,
            _ => B_DEFAULT,
        };
        edges.push(format!("{blk}(vpc, vinstr)"));
    }
    edges.join(", ")
}

/// The dispatch skeleton in textual IR. `run(x0)` interprets `BYTECODE`, returning `x0 + 3*50`.
fn skeleton_src() -> String {
    let edges = dispatch_edges();
    format!(
        r#"
memory 16
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  v64 = i64.const 64
  i64.store v64 v0
  vpc0 = i64.const 0
  br 1(vpc0)
}}
block 1 (vpc: i64) {{
  v4 = i64.const 4
  voff = i64.mul vpc v4
  vcbase = i64.const {base}
  vaddr = i64.add vcbase voff
  vinstr = i32.load vaddr
  v7f = i32.const 127
  vop = i32.and vinstr v7f
  br_table vop [ {edges} ] {default}(vpc, vinstr)
}}
block 2 (vpc: i64, vinstr: i32) {{
  v1 = i64.const 1
  vnpc = i64.add vpc v1
  br 1(vnpc)
}}
block 3 (vpc: i64, vinstr: i32) {{
  v7 = i32.const 7
  va32 = i32.shr_u vinstr v7
  vff = i32.const 255
  va32m = i32.and va32 vff
  va = i64.extend_i32_u va32m
  v8 = i64.const 8
  vraddr = i64.mul va v8
  v15 = i32.const 15
  vbx32 = i32.shr_u vinstr v15
  v1ffff = i32.const 131071
  vbx32m = i32.and vbx32 v1ffff
  vbx = i64.extend_i32_u vbx32m
  v65535 = i64.const 65535
  vsbx = i64.sub vbx v65535
  i64.store vraddr vsbx
  v1 = i64.const 1
  vnpc = i64.add vpc v1
  br 1(vnpc)
}}
block 4 (vpc: i64, vinstr: i32) {{
  v8 = i64.const 8
  vinit = i64.load v8
  v16 = i64.const 16
  vlim = i64.load v16
  v24 = i64.const 24
  vstep = i64.load v24
  vdiff = i64.sub vlim vinit
  vq = i64.div_s vdiff vstep
  v1 = i64.const 1
  vcount = i64.add vq v1
  i64.store v8 vcount
  v64 = i64.const 64
  vx0 = i64.load v64
  v0a = i64.const 0
  i64.store v0a vx0
  vnpc = i64.add vpc v1
  br 1(vnpc)
}}
block 5 (vpc: i64, vinstr: i32) {{
  v7 = i32.const 7
  va32 = i32.shr_u vinstr v7
  vff = i32.const 255
  va32m = i32.and va32 vff
  va = i64.extend_i32_u va32m
  v8 = i64.const 8
  vraddr = i64.mul va v8
  v24s = i32.const 24
  vc32 = i32.shr_u vinstr v24s
  vc32m = i32.and vc32 vff
  vc = i64.extend_i32_u vc32m
  v127 = i64.const 127
  vimm = i64.sub vc v127
  vold = i64.load vraddr
  vnew = i64.add vold vimm
  i64.store vraddr vnew
  v2 = i64.const 2
  vnpc = i64.add vpc v2
  br 1(vnpc)
}}
block 6 (vpc: i64, vinstr: i32) {{
  v8 = i64.const 8
  vc = i64.load v8
  v1 = i64.const 1
  vc2 = i64.sub vc v1
  i64.store v8 vc2
  v15 = i32.const 15
  vbx32 = i32.shr_u vinstr v15
  v1ffff = i32.const 131071
  vbx32m = i32.and vbx32 v1ffff
  vbx = i64.extend_i32_u vbx32m
  vcont = i64.sub vpc vbx
  vcont2 = i64.add vcont v1
  vexit = i64.add vpc v1
  vzero = i64.const 0
  vcond = i64.ne vc2 vzero
  br_if vcond 1(vcont2) 1(vexit)
}}
block 7 (vpc: i64, vinstr: i32) {{
  v7 = i32.const 7
  va32 = i32.shr_u vinstr v7
  vff = i32.const 255
  va32m = i32.and va32 vff
  va = i64.extend_i32_u va32m
  v8 = i64.const 8
  vraddr = i64.mul va v8
  vr = i64.load vraddr
  return vr
}}
block 8 (vpc: i64, vinstr: i32) {{
  unreachable
}}
}}
"#,
        base = BYTECODE_BASE,
        edges = edges,
        default = B_DEFAULT,
    )
}

/// Parse the skeleton and attach the real bytecode as a **readonly** data segment (so the specializer
/// folds the fetch, while an unspecialized run reads it from memory like a real interpreter).
fn skeleton_module() -> Module {
    let mut m = parse_module(&skeleton_src()).expect("skeleton parses");
    let mut bytes = Vec::with_capacity(BYTECODE.len() * 4);
    for w in BYTECODE {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    m.data.push(Data {
        offset: BYTECODE_BASE,
        readonly: true,
        bytes,
    });
    m
}

fn run_fuel(m: &Module, x0: i64) -> (Result<Vec<Value>, svm_interp::Trap>, u64) {
    let start = 50_000_000u64;
    let mut fuel = start;
    let r = svm_interp::run(m, 0, &[Value::I64(x0)], &mut fuel);
    (r, start - fuel)
}

fn inst_count(m: &Module) -> usize {
    m.funcs
        .iter()
        .flat_map(|f| &f.blocks)
        .map(|b| b.insts.len() + 1) // +1 for the terminator
        .sum()
}

/// Count the "interpreter tax" instructions — the fetch/dispatch machinery a projection is meant to
/// erase. The instruction fetch is the only **`i32.load`** in the skeleton (register loads are
/// `i64.load` — the real work, which must survive); the `br_table` is the 83-way dispatch itself.
fn dispatch_insts(m: &Module) -> usize {
    let mut n = 0;
    for f in &m.funcs {
        for b in &f.blocks {
            for i in &b.insts {
                if matches!(
                    i,
                    Inst::Load {
                        op: LoadOp::I32,
                        ..
                    }
                ) {
                    n += 1;
                }
            }
            if matches!(b.term, Terminator::BrTable { .. }) {
                n += 1;
            }
        }
    }
    n
}

#[test]
fn dispatch_fold_roi_on_real_lua_bytecode() {
    let m = skeleton_module();
    verify_module(&m).expect("skeleton verifies");

    // Sanity: the skeleton interprets the real bytecode correctly.
    let expected = 3 * LOOP_TRIPS; // x0 + 3*50, measured at x0 = 0
    let (base_res, base_fuel) = run_fuel(&m, 0);
    assert_eq!(
        base_res,
        Ok(vec![Value::I64(expected)]),
        "baseline skeleton miscomputed"
    );

    // Specialize: bytecode folded constant (readonly data), accumulator seed x0 dynamic.
    let residual = specialize_with_config(&m, 0, &[SpecArg::Dynamic], &SpecConfig::default())
        .expect("dispatch skeleton specializes");
    verify_module(&residual).expect("residual verifies");

    // The projection must erase the dispatch machinery entirely.
    let resid_dispatch = dispatch_insts(&residual);
    assert_eq!(
        resid_dispatch, 0,
        "residual still contains {resid_dispatch} fetch/dispatch instruction(s) — dispatch not folded"
    );
    assert!(
        !residual
            .funcs
            .iter()
            .flat_map(|f| &f.blocks)
            .any(|b| matches!(b.term, Terminator::BrTable { .. })),
        "residual still has a br_table — the 83-way dispatch survived"
    );

    // Differential correctness across several inputs, and the ROI numbers at x0 = 0.
    for x0 in [0i64, 1, -7, 1000, i64::MAX - 200] {
        let (rb, _) = run_fuel(&m, x0);
        let (rr, _) = run_fuel(&residual, x0);
        assert_eq!(rb, rr, "baseline vs residual diverged at x0={x0}");
        assert_eq!(rr, Ok(vec![Value::I64(x0.wrapping_add(expected))]));
    }
    let (_, resid_fuel) = run_fuel(&residual, 0);

    // The most representative figure is the **per-loop-iteration** instruction cost — it captures the
    // decode tax (shifts/masks/fetch) that the coarse, backedge-anchored fuel counter misses.
    //
    // Baseline per iteration executes two full dispatch cycles — DISPATCH→ADDI, then DISPATCH→FORLOOP
    // (block 1 is DISPATCH; 5 is ADDI; 6 is FORLOOP in `skeleton_src`).
    let blk = |mm: &Module, b: usize| mm.funcs[0].blocks[b].insts.len() + 1; // +terminator
    let base_iter = 2 * blk(&m, 1) + blk(&m, 5) + blk(&m, 6);
    // Residual per iteration is just its loop body: the block range spanned by the back-edge.
    let (lo, hi) = loop_body_range(&residual);
    let resid_iter: usize = (lo..=hi)
        .map(|b| residual.funcs[0].blocks[b].insts.len() + 1)
        .sum();

    let base_static = inst_count(&m);
    let resid_static = inst_count(&residual);
    let base_dispatch = dispatch_insts(&m);
    let dispatched_opcodes = base_fuel - 1; // fuel = 1 entry charge + one per opcode dispatched
    println!("\n=== Futamura dispatch-fold ROI on real Lua 5.4.7 bytecode ===");
    println!(
        "program: local x=0; for i=1,{LOOP_TRIPS} do x=x+3 end; return x  (x0 = runtime input)"
    );
    println!(
        "residual shape: a clean dispatch-free loop (blocks b{lo}..b{hi}), no br_table, no fetch"
    );
    println!("opcode dispatches:      interpreter {dispatched_opcodes}  →  residual 0");
    println!(
        "fetch/dispatch insts:   skeleton {base_dispatch} (fetch + 83-way br_table)  →  residual 0"
    );
    println!("per-iteration insts:    interpreter {base_iter}  →  residual {resid_iter}   ({:.1}× fewer)", base_iter as f64 / resid_iter.max(1) as f64);
    println!("whole-program static:   skeleton {base_static}  →  residual {resid_static}");
    println!("backedge fuel (x0=0):   {base_fuel}  →  {resid_fuel}");

    // The core claims, asserted.
    assert_eq!(
        resid_dispatch, 0,
        "the residual must carry zero fetch/dispatch instructions"
    );
    assert!(
        resid_iter * 2 <= base_iter,
        "expected the residual loop body ({resid_iter}) to at least halve the interpreter's \
         per-iteration cost ({base_iter}) once fetch/decode/dispatch is removed"
    );
    assert!(
        base_dispatch >= 2,
        "the skeleton should carry the fetch + the 83-way br_table it eliminates"
    );
}

/// The residual's innermost loop body: `[then_target ..= br_if_block]` for the back-edge `br_if`
/// (the one whose then-target precedes it). Used to size the per-iteration residual cost.
fn loop_body_range(m: &Module) -> (usize, usize) {
    for (bi, b) in m.funcs[0].blocks.iter().enumerate() {
        if let Terminator::BrIf { then_blk, .. } = &b.term {
            if (*then_blk as usize) < bi {
                return (*then_blk as usize, bi);
            }
        }
    }
    panic!("no back-edge loop found in the residual — expected a dispatch-free residual loop");
}
