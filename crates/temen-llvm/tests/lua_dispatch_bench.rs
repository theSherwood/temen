//! **Numeric-kernel dispatch benchmark.** The merged real-Lua work proved the Futamura projection
//! collapses `luaV_execute`'s dispatch, and that a rolled, dispatch-free residual is out of reach for
//! the *real* interpreter (its loop-exit continuation drags in the GC/error machinery — see #613). But
//! the dispatch-fold *itself* is measurable on real Lua bytecode via a **faithful dispatch skeleton**:
//! a byte-accurate model of `luaV_execute`'s inner loop (constant-bytecode fetch → 7-bit opcode decode
//! → 83-way `br_table` → per-opcode handlers with Lua-exact operand decode), but with plain integer
//! registers in memory (no `TValue` tags), so it rolls cleanly where the real interpreter can't.
//!
//! For a set of call-free numeric kernels, this captures each one's **real** Lua 5.4.7 bytecode, runs
//! it through the skeleton two ways — **baseline** (interpret, paying fetch/decode/dispatch every
//! opcode) and **specialized** (bytecode folded constant, so dispatch collapses to a rolled loop) —
//! and reports the ROI: dynamic instructions executed (speed) and residual size, with a differential
//! correctness check on every kernel.
//!
//! Scope: this measures the dispatch-fold on a faithful *model*, not the real `luaV_execute` raced
//! wall-clock vs Wasmtime (which needs a real compiled artifact — the projection can't produce one for
//! input-taking programs). It quantifies exactly the overhead a working projection removes per opcode.
//!
//! Run: `cargo test -p temen-llvm --test lua_dispatch_bench -- --nocapture`

use temen_interp::{IrPc, Stop, StopReason, Value};
use temen_ir::{Data, Inst, Module, Terminator};
use temen_peval::{specialize_with_config, SpecConfig};
use temen_text::parse_module;
use temen_verify::verify_module;

const BYTECODE_BASE: u64 = 65536; // 64 KiB-aligned: the RO data page never overlaps the registers

// Lua 5.4 opcodes we model (the union over the kernels below); everything else → the trap default.
const OP_MOVE: usize = 0;
const OP_LOADI: usize = 1;
const OP_ADDI: usize = 21;
const OP_ADD: usize = 34;
const OP_SUB: usize = 35;
const OP_MUL: usize = 36;
const OP_RETURN: usize = 70;
const OP_FORLOOP: usize = 73;
const OP_FORPREP: usize = 74;
const OP_VARARGPREP: usize = 81;

// Handler block numbers.
const B_VARARGPREP: u32 = 2;
const B_LOADI: u32 = 3;
const B_FORPREP: u32 = 4;
const B_ADDI: u32 = 5;
const B_FORLOOP: u32 = 6;
const B_RETURN: u32 = 7;
const B_DEFAULT: u32 = 8;
const B_ADD: u32 = 9;
const B_SUB: u32 = 10;
const B_MUL: u32 = 11;
const B_MOVE: u32 = 12;

fn dispatch_edges() -> String {
    (0..83usize)
        .map(|op| {
            let b = match op {
                OP_VARARGPREP => B_VARARGPREP,
                OP_LOADI => B_LOADI,
                OP_FORPREP => B_FORPREP,
                OP_ADDI => B_ADDI,
                OP_FORLOOP => B_FORLOOP,
                OP_RETURN => B_RETURN,
                OP_ADD => B_ADD,
                OP_SUB => B_SUB,
                OP_MUL => B_MUL,
                OP_MOVE => B_MOVE,
                _ => B_DEFAULT,
            };
            format!("{b}(vpc, vinstr)")
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// The faithful dispatch skeleton, in textual IR. `run() -> i64` interprets the given bytecode and
/// returns `R[A]` of its RETURN. Registers are 8-byte integer cells at `R[i] = 16384 + i*8` (no tags;
/// the register file sits just above the #1094 NULL guard so R[0] clears the unmapped `[0,16384)`).
fn skeleton_src() -> String {
    let edges = dispatch_edges();
    // A three-op operand decode used all over: `<reg>addr = ((instr >> shift) & 0xff) * 8`.
    let reg = |name: &str, shift: u32| {
        format!(
            "  {name}_o = i32.shr_u vinstr vsh{shift}\n  {name}_m = i32.and {name}_o vff\n  \
             {name}_e = i64.extend_i32_u {name}_m\n  {name}_p = i64.mul {name}_e v8\n  \
             {name}_b = i64.const 16384\n  {name}_a = i64.add {name}_p {name}_b"
        )
    };
    format!(
        r#"
memory 17
func () -> (i64) {{
block 0 () {{
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
  vn = i64.add vpc v1
  br 1(vn)
}}
block 3 (vpc: i64, vinstr: i32) {{
  vsh7 = i32.const 7
  vff = i32.const 255
  v8 = i64.const 8
{ra}
  vsh15 = i32.const 15
  vbx0 = i32.shr_u vinstr vsh15
  v1ffff = i32.const 131071
  vbx1 = i32.and vbx0 v1ffff
  vbx = i64.extend_i32_u vbx1
  v65535 = i64.const 65535
  vsbx = i64.sub vbx v65535
  i64.store ra_a vsbx
  v1 = i64.const 1
  vn = i64.add vpc v1
  br 1(vn)
}}
block 4 (vpc: i64, vinstr: i32) {{
  vsh7 = i32.const 7
  vff = i32.const 255
  v8 = i64.const 8
{ra}
  v8c = i64.const 8
  v16c = i64.const 16
  v24c = i64.const 24
  ra1 = i64.add ra_a v8c
  ra2 = i64.add ra_a v16c
  ra3 = i64.add ra_a v24c
  vinit = i64.load ra_a
  vlim = i64.load ra1
  vstep = i64.load ra2
  vdiff = i64.sub vlim vinit
  vq = i64.div_s vdiff vstep
  v1 = i64.const 1
  vcount = i64.add vq v1
  i64.store ra_a vcount
  i64.store ra3 vinit
  vn = i64.add vpc v1
  br 1(vn)
}}
block 5 (vpc: i64, vinstr: i32) {{
  vsh7 = i32.const 7
  vsh16 = i32.const 16
  vsh24 = i32.const 24
  vff = i32.const 255
  v8 = i64.const 8
{ra}
{rb}
  vc0 = i32.shr_u vinstr vsh24
  vc1 = i32.and vc0 vff
  vc = i64.extend_i32_u vc1
  v127 = i64.const 127
  vimm = i64.sub vc v127
  vb = i64.load rb_a
  vr = i64.add vb vimm
  i64.store ra_a vr
  v2 = i64.const 2
  vn = i64.add vpc v2
  br 1(vn)
}}
block 6 (vpc: i64, vinstr: i32) {{
  vsh7 = i32.const 7
  vff = i32.const 255
  v8 = i64.const 8
{ra}
  v16c = i64.const 16
  v24c = i64.const 24
  ra2 = i64.add ra_a v16c
  ra3 = i64.add ra_a v24c
  vcount = i64.load ra_a
  v1 = i64.const 1
  vc2 = i64.sub vcount v1
  i64.store ra_a vc2
  vstep = i64.load ra2
  vi = i64.load ra3
  vi2 = i64.add vi vstep
  i64.store ra3 vi2
  vsh15 = i32.const 15
  vbx0 = i32.shr_u vinstr vsh15
  v1ffff = i32.const 131071
  vbx1 = i32.and vbx0 v1ffff
  vbx = i64.extend_i32_u vbx1
  vcont = i64.sub vpc vbx
  vcont2 = i64.add vcont v1
  vexit = i64.add vpc v1
  vzero = i64.const 0
  vcond = i64.ne vc2 vzero
  br_if vcond 1(vcont2) 1(vexit)
}}
block 7 (vpc: i64, vinstr: i32) {{
  vsh7 = i32.const 7
  vff = i32.const 255
  v8 = i64.const 8
{ra}
  vr = i64.load ra_a
  return vr
}}
block 8 (vpc: i64, vinstr: i32) {{
  unreachable
}}
block 9 (vpc: i64, vinstr: i32) {{
  vsh7 = i32.const 7
  vsh16 = i32.const 16
  vsh24 = i32.const 24
  vff = i32.const 255
  v8 = i64.const 8
{ra}
{rb}
{rc}
  vb = i64.load rb_a
  vc = i64.load rc_a
  vr = i64.add vb vc
  i64.store ra_a vr
  v2 = i64.const 2
  vn = i64.add vpc v2
  br 1(vn)
}}
block 10 (vpc: i64, vinstr: i32) {{
  vsh7 = i32.const 7
  vsh16 = i32.const 16
  vsh24 = i32.const 24
  vff = i32.const 255
  v8 = i64.const 8
{ra}
{rb}
{rc}
  vb = i64.load rb_a
  vc = i64.load rc_a
  vr = i64.sub vb vc
  i64.store ra_a vr
  v2 = i64.const 2
  vn = i64.add vpc v2
  br 1(vn)
}}
block 11 (vpc: i64, vinstr: i32) {{
  vsh7 = i32.const 7
  vsh16 = i32.const 16
  vsh24 = i32.const 24
  vff = i32.const 255
  v8 = i64.const 8
{ra}
{rb}
{rc}
  vb = i64.load rb_a
  vc = i64.load rc_a
  vr = i64.mul vb vc
  i64.store ra_a vr
  v2 = i64.const 2
  vn = i64.add vpc v2
  br 1(vn)
}}
block 12 (vpc: i64, vinstr: i32) {{
  vsh7 = i32.const 7
  vsh16 = i32.const 16
  vff = i32.const 255
  v8 = i64.const 8
{ra}
{rb}
  vb = i64.load rb_a
  i64.store ra_a vb
  v1 = i64.const 1
  vn = i64.add vpc v1
  br 1(vn)
}}
}}
"#,
        base = BYTECODE_BASE,
        edges = edges,
        default = B_DEFAULT,
        ra = reg("ra", 7),
        rb = reg("rb", 16),
        rc = reg("rc", 24),
    )
}

fn skeleton_module(bytecode: &[u32]) -> Module {
    let mut m = parse_module(&skeleton_src()).expect("skeleton parses");
    let mut bytes = Vec::with_capacity(bytecode.len() * 4);
    for w in bytecode {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    m.data.push(Data {
        offset: BYTECODE_BASE,
        readonly: true,
        bytes,
    });
    m
}

// --- capture real Lua bytecode for a script ---------------------------------------------------

fn lua_module() -> Module {
    let path = format!(
        "{}/tests/fixtures/lua/lua_eval.ll",
        env!("CARGO_MANIFEST_DIR")
    );
    temen_llvm::translate_ll_path(&path)
        .expect("translate")
        .module
}

fn capture_bytecode(script: &str) -> Vec<u32> {
    let m = lua_module();
    let luav = m
        .exports
        .iter()
        .find(|e| e.name == "luaV_execute")
        .unwrap()
        .func;
    let inst = temen_run::instantiate(m.clone()).unwrap();
    let win = m.memory.map_or(0, |mc| 1u64 << mc.size_log2);
    let mut insp = inst.debug_attach(script.as_bytes().to_vec(), u64::MAX);
    insp.set_breakpoint(IrPc {
        module: 0,
        func: luav,
        block: 0,
        inst: 0,
    });
    let (ci, w) = match insp.run_until_stop() {
        Stop::Break {
            reason: StopReason::Breakpoint,
            ..
        } => {
            let ci = match insp.read_ir_value(0, 2) {
                Some(Value::I64(v)) => v as u64,
                o => panic!("{o:?}"),
            };
            (
                ci,
                insp.read_window(0, (win as usize).min(8 << 20)).unwrap(),
            )
        }
        o => panic!("{o:?}"),
    };
    let rd = |a: u64| u64::from_le_bytes(w[a as usize..a as usize + 8].try_into().unwrap());
    let rdi = |a: u64| i32::from_le_bytes(w[a as usize..a as usize + 4].try_into().unwrap());
    let ru = |a: u64| u32::from_le_bytes(w[a as usize..a as usize + 4].try_into().unwrap());
    let func = rd(ci);
    let closure = rd(func);
    let proto = rd(closure + 24);
    let code = rd(proto + 64);
    let n = rdi(proto + 24) as u64;
    (0..n).map(|i| ru(code + 4 * i)).collect()
}

// --- measurement ------------------------------------------------------------------------------

fn run(m: &Module) -> Result<Vec<Value>, temen_interp::Trap> {
    let mut fuel = u64::MAX;
    temen_interp::run(m, 0, &[], &mut fuel)
}

/// Total IR instructions executed to run `m` to completion (the debug clock is a per-op counter).
fn dynamic_ops(m: &Module) -> u64 {
    let inst = temen_run::instantiate(m.clone()).expect("instantiate");
    let mut insp = inst.debug_attach(Vec::new(), u64::MAX);
    match insp.run_until_stop() {
        Stop::Finished(_) => insp.clock(),
        o => panic!("did not finish: {o:?}"),
    }
}

fn static_insts(m: &Module) -> usize {
    m.funcs
        .iter()
        .flat_map(|f| &f.blocks)
        .map(|b| b.insts.len() + 1)
        .sum()
}

fn has_dispatch(m: &Module) -> bool {
    m.funcs
        .iter()
        .flat_map(|f| &f.blocks)
        .any(|b| matches!(b.term, Terminator::BrTable { .. }))
}
fn n_i32_loads(m: &Module) -> usize {
    m.funcs
        .iter()
        .flat_map(|f| &f.blocks)
        .flat_map(|b| &b.insts)
        .filter(|i| {
            matches!(
                i,
                Inst::Load {
                    op: temen_ir::LoadOp::I32,
                    ..
                }
            )
        })
        .count()
}

struct Kernel {
    name: &'static str,
    script: &'static str,
    expect: i64,
}

#[test]
fn dispatch_fold_roi_across_numeric_kernels() {
    // Call-free numeric kernels; `expect` is the value each returns (also the differential oracle).
    let kernels = [
        Kernel {
            name: "add-const",
            script: "local x=0\nfor i=1,50 do x=x+3 end\nreturn x\n",
            expect: 150,
        },
        Kernel {
            name: "sum-i",
            script: "local s=0\nfor i=1,50 do s=s+i end\nreturn s\n",
            expect: 1275,
        },
        Kernel {
            name: "sum-sq",
            script: "local s=0\nfor i=1,50 do s=s+i*i end\nreturn s\n",
            expect: 42925,
        },
        Kernel {
            name: "double",
            script: "local x=1\nfor i=1,30 do x=x+x end\nreturn x\n",
            expect: 1 << 30,
        },
        Kernel {
            name: "countdown",
            script: "local x=1000\nfor i=1,50 do x=x-i end\nreturn x\n",
            expect: 1000 - 1275,
        },
        Kernel {
            name: "nested",
            script: "local s=0\nfor i=1,20 do for j=1,20 do s=s+1 end end\nreturn s\n",
            expect: 400,
        },
        Kernel {
            name: "fib-iter",
            script: "local a=0\nlocal b=1\nfor i=1,40 do a,b=b,a+b end\nreturn a\n",
            expect: 102334155,
        },
    ];

    println!(
        "\n{:<11} {:>8} {:>10} {:>10} {:>8}   {:>10} {:>10} {:>7}",
        "kernel", "bc.ops", "interp.ops", "resid.ops", "speedup", "skel.size", "res.size", "shrink"
    );
    println!("{}", "-".repeat(88));

    for k in kernels {
        let bc = capture_bytecode(k.script);
        let skel = skeleton_module(&bc);
        verify_module(&skel).expect("skeleton verifies");

        // Baseline sanity: the skeleton interprets the real bytecode to the right answer.
        assert_eq!(
            run(&skel),
            Ok(vec![Value::I64(k.expect)]),
            "{}: skeleton miscomputed",
            k.name
        );

        // Specialize with the bytecode folded constant (registers stay residual memory → the loop
        // rolls; the dispatch collapses).
        let resid = specialize_with_config(&skel, 0, &[], &SpecConfig::default())
            .unwrap_or_else(|e| panic!("{}: specialize failed: {e:?}", k.name));
        verify_module(&resid).expect("residual verifies");

        // Correctness + the dispatch is gone.
        assert_eq!(run(&resid), run(&skel), "{}: residual diverged", k.name);
        assert!(has_dispatch(&skel), "{}: skeleton should dispatch", k.name);
        assert!(
            !has_dispatch(&resid),
            "{}: residual still has a br_table",
            k.name
        );
        assert_eq!(
            n_i32_loads(&resid),
            0,
            "{}: residual still fetches bytecode",
            k.name
        );

        let interp = dynamic_ops(&skel);
        let res = dynamic_ops(&resid);
        let skel_sz = static_insts(&skel);
        let res_sz = static_insts(&resid);
        println!(
            "{:<11} {:>8} {:>10} {:>10} {:>7.1}x   {:>10} {:>10} {:>6.1}x",
            k.name,
            bc.len(),
            interp,
            res,
            interp as f64 / res.max(1) as f64,
            skel_sz,
            res_sz,
            skel_sz as f64 / res_sz.max(1) as f64,
        );
    }
    println!(
        "\ninterp.ops/resid.ops = dynamic IR instructions executed (dispatch + decode eliminated);"
    );
    println!("skel.size = the 'interpreter' (all 83 handlers); res.size = the compiled kernel.");
}
