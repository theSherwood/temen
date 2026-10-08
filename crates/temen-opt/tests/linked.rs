//! Spec for the whole-program passes a linker runs (#2147). [`order_blocks`] numbers each function's
//! blocks in reverse postorder, so the only branch to an equal-or-earlier block is a loop's back
//! edge — the branch every engine charges fuel on (INVARIANTS #9). The inliner's one-pass driver
//! inlines what its splices bring in, and stops at its size limit. [`optimize_linked`] runs the two
//! with a cleanup between them that folds no floats. Every transform is checked against the reference
//! interpreter: same results, output that re-verifies.

use temen_interp::Value;
use temen_ir::{
    BinOp, Block, CmpOp, DebugInfo, FBinOp, FloatTy, Func, FuncName, Inst, IntTy, Loc, Module,
    SsaLoc, Terminator, ValType, VarInfo, VarLoc,
};
use temen_opt::cfg::{successors, Cfg};
use temen_opt::interproc::{inline_calls, inline_calls_with, InlineLimits};
use temen_opt::{optimize_linked, order_blocks};
use temen_verify::verify_module;

/// Run `func` on the reference interpreter: its results and the fuel it burned.
fn run(m: &Module, func: u32, args: &[Value]) -> (Vec<Value>, u64) {
    let start = 1_000_000u64;
    let mut fuel = start;
    let out = temen_interp::run(m, func, args, &mut fuel).expect("runs");
    (out, start - fuel)
}

/// The branches in `f` to an equal-or-earlier block, `(from, to)`: where INVARIANTS #9 charges fuel.
fn index_back_edges(f: &Func) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    for (b, blk) in f.blocks.iter().enumerate() {
        for t in successors(&blk.term) {
            if t <= b as u32 {
                out.push((b as u32, t));
            }
        }
    }
    out
}

/// Whether block `a` dominates block `b` in `f`.
fn dominates(f: &Func, a: u32, b: u32) -> bool {
    let idom = Cfg::new(&f.blocks).dominators();
    let mut at = Some(b);
    while let Some(x) = at {
        if x == a {
            return true;
        }
        at = idom[x as usize];
    }
    false
}

fn n_calls(m: &Module) -> usize {
    m.funcs
        .iter()
        .flat_map(|f| &f.blocks)
        .flat_map(|b| &b.insts)
        .filter(|i| matches!(i, Inst::Call { .. }))
        .count()
}

fn add(a: u32, b: u32) -> Inst {
    Inst::IntBin {
        ty: IntTy::I32,
        op: BinOp::Add,
        a,
        b,
    }
}

fn sub(a: u32, b: u32) -> Inst {
    Inst::IntBin {
        ty: IntTy::I32,
        op: BinOp::Sub,
        a,
        b,
    }
}

fn cmp(op: CmpOp, a: u32, b: u32) -> Inst {
    Inst::IntCmp {
        ty: IntTy::I32,
        op,
        a,
        b,
    }
}

fn call(func: u32, args: Vec<u32>) -> Inst {
    Inst::Call { func, args }
}

fn func(params: usize, blocks: Vec<Block>) -> Func {
    Func {
        params: vec![ValType::I32; params],
        results: vec![ValType::I32],
        blocks,
    }
}

fn block(params: usize, insts: Vec<Inst>, term: Terminator) -> Block {
    Block {
        params: vec![ValType::I32; params],
        insts,
        term,
    }
}

fn br(target: u32, args: Vec<u32>) -> Terminator {
    Terminator::Br { target, args }
}

fn br_if(cond: u32, then: (u32, Vec<u32>), els: (u32, Vec<u32>)) -> Terminator {
    Terminator::BrIf {
        cond,
        then_blk: then.0,
        then_args: then.1,
        else_blk: els.0,
        else_args: els.1,
    }
}

fn module(funcs: Vec<Func>) -> Module {
    let m = Module {
        funcs,
        ..Default::default()
    };
    verify_module(&m).expect("input verifies");
    m
}

/// `pick(a) = a != 0 ? 10 : 20`, laid out the way leng lays out an `if`: the join block (1) before
/// the arms (2, 3) that branch to it.
fn early_join() -> Func {
    func(
        1,
        vec![
            // b0(a): a != 0 ? b2 : b3
            block(
                1,
                vec![Inst::ConstI32(0), cmp(CmpOp::Ne, 0, 1)],
                br_if(2, (2, vec![]), (3, vec![])),
            ),
            // b1(x): the join
            block(1, vec![], Terminator::Return(vec![0])),
            block(0, vec![Inst::ConstI32(10)], br(1, vec![0])),
            block(0, vec![Inst::ConstI32(20)], br(1, vec![0])),
        ],
    )
}

#[test]
fn an_early_join_moves_after_its_arms_and_stops_charging_fuel() {
    let m = module(vec![early_join()]);
    assert_eq!(
        index_back_edges(&m.funcs[0]),
        vec![(2, 1), (3, 1)],
        "both arms branch back to the early join"
    );
    let mut o = m.clone();
    order_blocks(&mut o);
    verify_module(&o).expect("ordered module verifies");
    assert!(
        index_back_edges(&o.funcs[0]).is_empty(),
        "an `if` has no loop, so no back edge"
    );
    assert_eq!(
        o.funcs[0].blocks[0].insts, m.funcs[0].blocks[0].insts,
        "the entry stays block 0"
    );
    for a in [0, 1, -5] {
        let (want, fuel_before) = run(&m, 0, &[Value::I32(a)]);
        let (got, fuel_after) = run(&o, 0, &[Value::I32(a)]);
        assert_eq!(got, want, "a={a}");
        assert_eq!(
            fuel_before - fuel_after,
            1,
            "a={a}: the arm's branch to the join no longer charges a fuel"
        );
    }
}

/// `sum(n) = n + (n-1) + ... + 1`, with the loop's exit (1) placed before its header (2):
/// `b0(n): br b2(n, 0)` / `b1(acc): return acc` / `b2(i, acc): i != 0 ? b3(i, acc) : b1(acc)` /
/// `b3(i, acc): br b2(i-1, acc+i)`.
fn loop_with_early_exit() -> Func {
    func(
        1,
        vec![
            block(1, vec![Inst::ConstI32(0)], br(2, vec![0, 1])),
            block(1, vec![], Terminator::Return(vec![0])),
            block(
                2,
                vec![Inst::ConstI32(0), cmp(CmpOp::Ne, 0, 2)],
                br_if(3, (3, vec![0, 1]), (1, vec![1])),
            ),
            block(
                2,
                vec![Inst::ConstI32(1), sub(0, 2), add(1, 0)],
                br(2, vec![3, 4]),
            ),
        ],
    )
}

#[test]
fn a_loop_keeps_exactly_its_back_edge() {
    let m = module(vec![loop_with_early_exit()]);
    assert_eq!(
        index_back_edges(&m.funcs[0]),
        vec![(2, 1), (3, 2)],
        "the exit edge counts as a back edge, besides the latch"
    );
    let mut o = m.clone();
    order_blocks(&mut o);
    verify_module(&o).expect("ordered module verifies");
    let back = index_back_edges(&o.funcs[0]);
    assert_eq!(back.len(), 1, "only the latch is left: {back:?}");
    let (from, to) = back[0];
    assert!(
        dominates(&o.funcs[0], to, from),
        "the one back edge goes to the loop header"
    );
    for n in 0..6 {
        let (want, fuel_before) = run(&m, 0, &[Value::I32(n)]);
        let (got, fuel_after) = run(&o, 0, &[Value::I32(n)]);
        assert_eq!(got, want, "n={n}");
        assert_eq!(got, vec![Value::I32((1..=n).sum())], "n={n}");
        assert_eq!(
            fuel_before - fuel_after,
            1,
            "n={n}: leaving the loop no longer charges; every iteration still does"
        );
    }
}

#[test]
fn unreachable_blocks_go_last_with_their_targets_remapped() {
    // b0(a): br b2(a) / b1: unreachable from the entry, br b2(7) / b2(x): return x
    let m = module(vec![func(
        1,
        vec![
            block(1, vec![], br(2, vec![0])),
            block(0, vec![Inst::ConstI32(7)], br(2, vec![0])),
            block(1, vec![], Terminator::Return(vec![0])),
        ],
    )]);
    let mut o = m.clone();
    order_blocks(&mut o);
    verify_module(&o).expect("ordered module verifies");
    let f = &o.funcs[0];
    assert_eq!(f.blocks.len(), 3, "no block is dropped");
    assert_eq!(
        f.blocks[1].term,
        Terminator::Return(vec![0]),
        "reachable blocks first"
    );
    assert_eq!(
        f.blocks[2],
        block(0, vec![Inst::ConstI32(7)], br(1, vec![0])),
        "the unreachable block is last, still branching to the return block"
    );
    assert_eq!(
        run(&o, 0, &[Value::I32(3)]).0,
        run(&m, 0, &[Value::I32(3)]).0
    );
}

#[test]
fn debug_positions_move_with_their_blocks() {
    let mut m = module(vec![early_join()]);
    let loc = |block: u32, line: u32| Loc {
        func: 0,
        block,
        inst: 0,
        file: 0,
        line,
        col: 1,
    };
    m.debug_info = Some(DebugInfo {
        files: vec!["pick.nim".into()],
        locs: vec![loc(2, 10), loc(3, 20), loc(1, 30)],
        vars: vec![VarInfo {
            func: 0,
            name: "x".into(),
            ty: "int".into(),
            loc: VarLoc::SsaList(vec![SsaLoc {
                block: 1,
                inst: 0,
                value: 0,
            }]),
            type_id: None,
            scope: None,
        }],
        ..Default::default()
    });
    let mut o = m.clone();
    order_blocks(&mut o);
    let f = &o.funcs[0];
    // Find each block by what it holds: the `10` arm, the `20` arm, the join.
    let at = |want: &Block| f.blocks.iter().position(|b| b == want).unwrap() as u32;
    let ten = at(&block(0, vec![Inst::ConstI32(10)], br(at_join(f), vec![0])));
    let twenty = at(&block(0, vec![Inst::ConstI32(20)], br(at_join(f), vec![0])));
    let join = at_join(f);
    let d = o.debug_info.as_ref().expect("debug info kept");
    let lines: Vec<(u32, u32)> = d.locs.iter().map(|l| (l.block, l.line)).collect();
    assert_eq!(lines, vec![(ten, 10), (twenty, 20), (join, 30)]);
    assert_eq!(
        d.vars[0].loc,
        VarLoc::SsaList(vec![SsaLoc {
            block: join,
            inst: 0,
            value: 0,
        }])
    );
}

/// The index of `f`'s join block (the one `return`).
fn at_join(f: &Func) -> u32 {
    f.blocks
        .iter()
        .position(|b| matches!(b.term, Terminator::Return(_)))
        .unwrap() as u32
}

/// `leaf(a) = a + 1 + 1 + …`, a one-block leaf of exactly `n` instructions.
fn leaf_of(n: usize) -> Func {
    let mut insts = vec![Inst::ConstI32(1)];
    for k in 1..n as u32 {
        // v1 = 1, v2 = v0 + v1, v3 = v2 + v1, …
        insts.push(add(if k == 1 { 0 } else { k }, 1));
    }
    let last = n as u32;
    func(1, vec![block(1, insts, Terminator::Return(vec![last]))])
}

fn entry_calling(callee: u32) -> Func {
    func(
        1,
        vec![block(
            1,
            vec![call(callee, vec![0])],
            Terminator::Return(vec![1]),
        )],
    )
}

#[test]
fn the_inliner_stops_at_its_size_limit() {
    let m = module(vec![entry_calling(1), leaf_of(7)]);
    let six = InlineLimits {
        max_callee: 6,
        budget: 1000,
    };
    let seven = InlineLimits {
        max_callee: 7,
        budget: 1000,
    };
    assert_eq!(
        n_calls(&inline_calls_with(&m, six)),
        1,
        "7 > 6: not inlined"
    );
    let inl = inline_calls_with(&m, seven);
    verify_module(&inl).expect("inlined module verifies");
    assert_eq!(n_calls(&inl), 0, "7 <= 7: inlined");
    assert_eq!(
        run(&inl, 0, &[Value::I32(5)]).0,
        run(&m, 0, &[Value::I32(5)]).0
    );
}

#[test]
fn a_splice_brings_its_calls_into_the_pass() {
    // entry(a) = f(a); f(a) = g(a) + 1; g(a) = a + 1. One pass inlines f, then the call to g that
    // f's body brought along.
    let f = func(
        1,
        vec![block(
            1,
            vec![call(2, vec![0]), Inst::ConstI32(1), add(1, 2)],
            Terminator::Return(vec![3]),
        )],
    );
    let m = module(vec![entry_calling(1), f, leaf_of(2)]);
    let inl = inline_calls(&m);
    verify_module(&inl).expect("inlined module verifies");
    assert_eq!(n_calls(&inl), 0);
    assert_eq!(
        run(&inl, 0, &[Value::I32(5)]).0,
        vec![Value::I32(7)],
        "g(5) + 1"
    );

    // entry(a) = add1(abs(a)): the call after the multi-block `abs` lands in the appended
    // continuation, which the pass reaches too.
    let abs = func(
        1,
        vec![
            block(
                1,
                vec![Inst::ConstI32(0), cmp(CmpOp::LtS, 0, 1)],
                br_if(2, (1, vec![0]), (2, vec![0])),
            ),
            block(
                1,
                vec![Inst::ConstI32(0), sub(1, 0)],
                Terminator::Return(vec![2]),
            ),
            block(1, vec![], Terminator::Return(vec![0])),
        ],
    );
    let entry = func(
        1,
        vec![block(
            1,
            vec![call(1, vec![0]), call(2, vec![1])],
            Terminator::Return(vec![2]),
        )],
    );
    let m = module(vec![entry, abs, leaf_of(2)]);
    let inl = inline_calls(&m);
    verify_module(&inl).expect("inlined module verifies");
    assert_eq!(n_calls(&inl), 0);
    for a in [-3, 0, 4] {
        assert_eq!(
            run(&inl, 0, &[Value::I32(a)]).0,
            vec![Value::I32(a.abs() + 1)]
        );
    }
}

#[test]
fn inlining_keeps_the_debug_info_that_still_holds() {
    let mut m = module(vec![entry_calling(1), leaf_of(2), leaf_of(3)]);
    let loc = |func: u32, line: u32| Loc {
        func,
        block: 0,
        inst: 0,
        file: 0,
        line,
        col: 1,
    };
    let name = |func: u32, name: &str| FuncName {
        func,
        name: name.into(),
    };
    m.debug_info = Some(DebugInfo {
        files: vec!["p.nim".into()],
        locs: vec![loc(0, 1), loc(1, 2), loc(2, 3)],
        func_names: vec![name(0, "main"), name(1, "inc"), name(2, "unused")],
        ..Default::default()
    });
    let inl = inline_calls(&m);
    assert_eq!(n_calls(&inl), 0);
    let d = inl.debug_info.as_ref().expect("debug info kept");
    assert_eq!(d.func_names.len(), 3, "no function was renumbered");
    let lines: Vec<u32> = d.locs.iter().map(|l| l.line).collect();
    assert_eq!(
        lines,
        vec![2, 3],
        "only the caller's positions went stale: the callee and the uncalled function keep theirs"
    );
}

#[test]
fn optimize_linked_inlines_tiny_callees_and_orders_blocks() {
    // `early_join` whose arms each call a 2-instruction leaf: both calls are inlined, and the join
    // moves after the arms that branch to it.
    let mut entry = early_join();
    for arm in [2usize, 3] {
        entry.blocks[arm].insts.push(call(1, vec![0]));
        entry.blocks[arm].term = br(1, vec![1]);
    }
    let m = module(vec![entry, leaf_of(2)]);
    let o = optimize_linked(&m);
    verify_module(&o).expect("linked output verifies");
    assert_eq!(n_calls(&o), 0, "the tiny leaf is inlined at both sites");
    assert!(
        index_back_edges(&o.funcs[0]).is_empty(),
        "no loop, no back edge"
    );
    for a in [0, 1] {
        let (want, fuel_before) = run(&m, 0, &[Value::I32(a)]);
        let (got, fuel_after) = run(&o, 0, &[Value::I32(a)]);
        assert_eq!(got, want, "a={a}");
        assert_eq!(
            fuel_before - fuel_after,
            2,
            "a={a}: neither the call's entry nor the branch to the join charges"
        );
    }
}

#[test]
fn optimize_linked_cleans_up_after_its_splices() {
    // entry(a) = abs(-5) + a. Inlining threads -5 into `abs`'s branch; the cleanup resolves the
    // branch, prunes the arm it never takes and merges what is left into one block.
    let abs = func(
        1,
        vec![
            block(
                1,
                vec![Inst::ConstI32(0), cmp(CmpOp::LtS, 0, 1)],
                br_if(2, (1, vec![0]), (2, vec![0])),
            ),
            block(
                1,
                vec![Inst::ConstI32(0), sub(1, 0)],
                Terminator::Return(vec![2]),
            ),
            block(1, vec![], Terminator::Return(vec![0])),
        ],
    );
    let entry = func(
        1,
        vec![block(
            1,
            vec![Inst::ConstI32(-5), call(1, vec![1]), add(2, 0)],
            Terminator::Return(vec![3]),
        )],
    );
    // The uncalled leaf makes the program big enough for the inline budget, half the program, to
    // cover the splice.
    let m = module(vec![entry, abs, leaf_of(8)]);
    let o = optimize_linked(&m);
    verify_module(&o).expect("linked output verifies");
    assert_eq!(n_calls(&o), 0, "abs is inlined");
    assert_eq!(
        o.funcs[0].blocks.len(),
        1,
        "the branch on a constant is gone"
    );
    for a in [-3, 0, 4] {
        assert_eq!(
            run(&o, 0, &[Value::I32(a)]).0,
            run(&m, 0, &[Value::I32(a)]).0,
            "a={a}"
        );
    }
}

#[test]
fn optimize_linked_folds_no_floats() {
    // f() = (1.5 + 2.25, 2 + 3): the link folds the integer add and leaves the float add, because a
    // host linker and an in-guest one must produce the same module and a float fold need not.
    let f = Func {
        params: vec![],
        results: vec![ValType::F64, ValType::I32],
        blocks: vec![Block {
            params: vec![],
            insts: vec![
                Inst::ConstF64(1.5f64.to_bits()),
                Inst::ConstF64(2.25f64.to_bits()),
                Inst::FBin {
                    ty: FloatTy::F64,
                    op: FBinOp::Add,
                    a: 0,
                    b: 1,
                },
                Inst::ConstI32(2),
                Inst::ConstI32(3),
                add(3, 4),
            ],
            term: Terminator::Return(vec![2, 5]),
        }],
    };
    let m = module(vec![f]);
    let o = optimize_linked(&m);
    verify_module(&o).expect("linked output verifies");
    let insts = &o.funcs[0].blocks[0].insts;
    assert!(
        insts.iter().any(|i| matches!(i, Inst::FBin { .. })),
        "the float add stays: {insts:?}"
    );
    assert!(
        !insts.iter().any(|i| matches!(i, Inst::IntBin { .. })),
        "the integer add folds: {insts:?}"
    );
    assert_eq!(run(&o, 0, &[]).0, run(&m, 0, &[]).0);
}

#[test]
fn the_cleanup_drops_only_the_positions_it_made_stale() {
    // f0 folds `2 + 3`, so its positions go stale; nothing in f1 changes, so its stay.
    let folds = func(
        1,
        vec![block(
            1,
            vec![Inst::ConstI32(2), Inst::ConstI32(3), add(1, 2)],
            Terminator::Return(vec![3]),
        )],
    );
    let mut m = module(vec![folds, leaf_of(3)]);
    let loc = |func: u32, line: u32| Loc {
        func,
        block: 0,
        inst: 0,
        file: 0,
        line,
        col: 1,
    };
    m.debug_info = Some(DebugInfo {
        files: vec!["p.nim".into()],
        locs: vec![loc(0, 1), loc(1, 2)],
        ..Default::default()
    });
    let o = optimize_linked(&m);
    let d = o.debug_info.as_ref().expect("debug info kept");
    let lines: Vec<u32> = d.locs.iter().map(|l| l.line).collect();
    assert_eq!(lines, vec![2], "only f0's position went stale");
}
