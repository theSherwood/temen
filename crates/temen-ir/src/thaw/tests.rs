use super::*;
use crate::durable_abi::ShadowArena;
use crate::{Func, Memory};
use alloc::vec;

/// The arena the durable tests declare: 11 contexts of 4 KiB from 16448.
const A: ShadowArena = ShadowArena::new(16448, 65536);

/// `func 0 () -> (i64)` (a root entry) and `func 1 (i64, i64) -> (i64)` (a spawn entry).
fn module(arena: Option<ShadowArena>) -> Module {
    let f = |params: Vec<ValType>| Func {
        params,
        results: vec![ValType::I64],
        blocks: Vec::new(),
    };
    Module {
        funcs: vec![f(vec![]), f(vec![ValType::I64, ValType::I64])],
        memory: Some(Memory {
            size_log2: 17,
            shadow: arena,
        }),
        ..Module::default()
    }
}

type Raw = RawResidue<(), ()>;

/// [`validate`] for a domain that holds a `Jit` grant (its code may make every record kind), in context 0.
fn admit(m: &Module, r: Raw) -> Result<Thaw<(), ()>, Malformed> {
    validate(m, true, 0, r).map_err(Malformed::from)
}

/// A residue a run makes: two fibers (one free), a live vCPU in the top context, a completed one, a
/// completed detached child, a live detached child, a live twin and an exited one.
fn real() -> Raw {
    RawResidue {
        fibers: vec![
            FrozenFiber {
                slot: 0,
                func: 7,
                arg0: 3,
                shadow_sp: A.frame_base(1) + 64,
                generation: u64::MAX,
                consumed: true,
            },
            FrozenFiber::free(1, 5),
        ],
        vcpus: vec![
            FrozenVCpu {
                task: 1,
                parent_task: 0,
                slot: 0,
                joinable: true,
                func: 0,
                args: vec![],
                shadow_sp: 0,
                completed_result: Some(9),
            },
            FrozenVCpu {
                task: 3,
                parent_task: 2, // its spawner finished and rides no record (#3034 D3)
                slot: 1,
                joinable: true,
                func: 1,
                args: vec![4, 5],
                shadow_sp: A.frame_base(10) + 8,
                completed_result: None,
            },
        ],
        root_sp: Some(A.frame_base(0) + 16),
        children: vec![1, 1, 2, 3],
        detached: vec![FrozenDetached {
            parent_task: 0,
            slot: 2,
            completed_result: Err(crate::trap_code::THREAD_FAULT),
        }],
        live_detached: vec![LiveDetached {
            parent_task: 0,
            slot: 3,
            joinable: true,
            task: 4,
            lane: -1,
            payload: (),
        }],
        twins: vec![LiveTwin {
            pid: 6,
            entry: 0,
            ctx: 10,
            payload: (),
        }],
        exited_twins: vec![FrozenTwin {
            pid: 7,
            outcome: Ok(0),
        }],
    }
}

#[test]
fn a_residue_a_run_makes_is_admitted_whole() {
    let t = admit(&module(Some(A)), real()).expect("admitted");
    assert_eq!(t.into_parts(), real());
    assert!(admit(&module(Some(A)), Raw::default())
        .expect("empty")
        .is_empty());
    // A module without an arena admits only the empty residue (INVARIANTS #16).
    assert!(admit(&module(None), Raw::default()).is_ok());
}

/// A context whose unwound chain fills its region exactly: the extent is the region's end, which the
/// transform's push check admits (`newsp > region_end` traps). Before #3034's fix `vcpu_of_sp` floored
/// it into the next context and the pass refused a real freeze.
#[test]
fn an_extent_at_its_regions_end_is_admitted() {
    let mut r = real();
    r.vcpus[1].shadow_sp = A.region_base(10) + A.stride;
    r.root_sp = Some(A.region_base(0) + A.stride);
    r.fibers[0].shadow_sp = A.region_base(1) + A.stride;
    assert!(admit(&module(Some(A)), r).is_ok());
    // One past it is the next context's SP word, and refused.
    let mut r = real();
    r.vcpus[1].shadow_sp = A.region_base(10) + A.stride + 1;
    assert_eq!(admit(&module(Some(A)), r).map(|_| ()), Err(Malformed));
}

#[test]
fn each_value_no_run_makes_is_refused() {
    type Bend = fn(&mut Raw);
    let cases: &[(&str, Bend)] = &[
        // vCPUs
        ("task past TASK_MAX", |r| r.vcpus[1].task = usize::MAX),
        ("two vCPUs with one task", |r| r.vcpus[1].task = 1),
        ("a parent at or above its child", |r| {
            r.vcpus[1].parent_task = 3
        }),
        ("shadow-SP past the arena", |r| {
            r.vcpus[1].shadow_sp = A.end + 64
        }),
        ("shadow-SP below the arena", |r| r.vcpus[1].shadow_sp = 16),
        ("shadow-SP in the root's region", |r| {
            r.vcpus[1].shadow_sp = A.base + 8
        }),
        ("shadow-SP in a context's header", |r| {
            r.vcpus[1].shadow_sp = A.region_base(10) + 8
        }),
        ("an entry past the module", |r| r.vcpus[1].func = 99),
        ("an entry of another signature", |r| r.vcpus[1].func = 0),
        ("three spawn args", |r| r.vcpus[1].args.push(6)),
        ("a join slot past the table", |r| {
            r.vcpus[1].slot = handles::SLOTS
        }),
        ("a join slot never issued", |r| {
            r.children.push(1);
            r.vcpus[1].slot = 9;
        }),
        ("two records at one slot", |r| r.vcpus[1].slot = 0),
        ("an unjoinable vCPU at a slot", |r| {
            r.vcpus[1].joinable = false
        }),
        ("a completed vCPU with an entry", |r| r.vcpus[0].func = 1),
        ("two vCPUs in one context", |r| {
            let mut v = r.vcpus[1].clone();
            v.task = 8;
            v.slot = 0;
            v.joinable = false;
            r.vcpus.push(v);
        }),
        ("no root extent", |r| r.root_sp = None),
        ("a root extent past its region", |r| {
            r.root_sp = Some(A.region_base(1) + 8)
        }),
        // the child table
        ("a counter Table::issue skips", |r| {
            r.children[0] = 1 << handles::GEN_BITS
        }),
        ("a trailing zero", |r| r.children.push(0)),
        ("more counters than slots", |r| {
            r.children = vec![1; handles::SLOTS + 1]
        }),
        // fibers
        ("a fiber slot gap", |r| r.fibers[1].slot = 2),
        ("a fiber in a vCPU's context", |r| {
            r.fibers = (0..10).map(|s| FrozenFiber::free(s, 0)).collect()
        }),
        ("a fiber past the arena", |r| {
            r.vcpus.truncate(1);
            r.children = vec![1, 0, 2, 3];
            r.fibers = (0..11).map(|s| FrozenFiber::free(s, 0)).collect();
        }),
        ("a fiber extent in another region", |r| {
            r.fibers[0].shadow_sp = A.frame_base(2)
        }),
        ("a free slot with an entry", |r| r.fibers[1].func = 3),
        // detached children and twins
        ("a live detached task past TASK_MAX", |r| {
            r.live_detached[0].task = u64::MAX
        }),
        ("a lane below -1", |r| r.live_detached[0].lane = -2),
        ("a twin entry past the module", |r| r.twins[0].entry = 2),
        ("a twin context past the arena", |r| r.twins[0].ctx = 11),
        ("a twin pid past TASK_MAX", |r| r.twins[0].pid = u64::MAX),
        ("an exited twin pid of 0", |r| r.exited_twins[0].pid = 0),
        // trap codes a join outcome carries
        ("a detached trap code of 0", |r| {
            r.detached[0].completed_result = Err(0)
        }),
        ("a JIT-internal trap code", |r| {
            r.detached[0].completed_result = Err(crate::trap_code::DOMAIN_DONE)
        }),
        ("a status on a trap that is not an exit", |r| {
            r.detached[0].completed_result = Err(crate::trap_code::THREAD_FAULT | 1 << 32)
        }),
        ("an exited twin's trap code of 0", |r| {
            r.exited_twins[0].outcome = Err(0)
        }),
    ];
    for (name, bend) in cases {
        let mut r = real();
        bend(&mut r);
        assert_eq!(
            admit(&module(Some(A)), r).map(|_| ()),
            Err(Malformed),
            "{name}"
        );
    }
    // Any residue at all on a module with no arena.
    let mut r = Raw::default();
    r.fibers.push(FrozenFiber::free(0, 0));
    assert_eq!(admit(&module(None), r).map(|_| ()), Err(Malformed));
}

/// Each record kind is admitted only where the domain's code can make it (#3034): a fiber
/// from a `cont.new`, a vCPU from a `thread.spawn`, a detached child from an `Instantiator` call, or
/// any of them from code installed through a `Jit` grant.
#[test]
fn a_record_kind_the_domains_code_cannot_make_is_refused() {
    let site = |m: &mut Module, inst: Inst| {
        m.funcs[1].blocks = vec![crate::Block {
            parent: 0,
            params: vec![],
            captures: vec![],
            insts: vec![inst],
            term: crate::Terminator::Unreachable,
        }]
    };
    type Kind = (&'static str, fn(&mut Raw), Inst);
    let kinds: &[Kind] = &[
        (
            "fibers",
            |r| r.fibers.clear(),
            Inst::ContNew { func: 0, arg0: 0 },
        ),
        (
            "vCPUs",
            |r| {
                r.vcpus.clear();
                r.root_sp = None;
            },
            Inst::ThreadSpawn {
                func: 1,
                arg0: 0,
                arg1: 0,
            },
        ),
        (
            "detached children",
            |r| {
                r.detached.clear();
                r.live_detached.clear();
            },
            Inst::CapCall(Box::new(crate::CapCall {
                type_id: crate::cap_id::INSTANTIATOR,
                op: 0,
                sig: 0,
                handle: 0,
                args: Box::new([]),
            })),
        ),
    ];
    // `real()` with only `kind` (all other kinds cleared).
    let only = |keep: usize| {
        let mut r = real();
        r.twins.clear();
        r.exited_twins.clear();
        for (i, (_, clear, _)) in kinds.iter().enumerate() {
            if i != keep {
                clear(&mut r);
            }
        }
        r
    };
    for (k, (name, _, inst)) in kinds.iter().enumerate() {
        let bare = module(Some(A));
        assert!(
            validate(&bare, false, 0, only(k)).is_err(),
            "{name}: no site, no Jit grant"
        );
        assert!(
            validate(&bare, true, 0, only(k)).is_ok(),
            "{name}: under a Jit grant"
        );
        let mut m = module(Some(A));
        site(&mut m, inst.clone());
        assert!(
            validate(&m, false, 0, only(k)).is_ok(),
            "{name}: at its site"
        );
    }
    // A serve op's handlers are fibers (#2742): its site admits them too.
    let mut m = module(Some(A));
    site(
        &mut m,
        Inst::CapCall(Box::new(crate::CapCall {
            type_id: crate::CAP_SELF_TYPE_ID,
            op: crate::durable_abi::SVC_WAIT_OP,
            sig: 0,
            handle: 0,
            args: Box::new([]),
        })),
    );
    assert!(
        validate(&m, false, 0, only(0)).is_ok(),
        "fibers: at a serve op"
    );
}

/// A refusal hands the residue back whole.
#[test]
fn a_refused_residue_is_handed_back() {
    let mut r = real();
    r.vcpus[1].func = 99;
    let want = r.clone();
    match validate(&module(Some(A)), true, 0, r) {
        Err(r) => assert_eq!(r.residue.vcpus, want.vcpus),
        Ok(_) => panic!("admitted"),
    }
}

/// A freeze hands its residue over in memory in the order its records unwound; the pass admits it in
/// the order every engine seeds in, fibers by slot and vCPUs by task.
#[test]
fn an_unsorted_residue_is_admitted_in_seeding_order() {
    let mut r = real();
    r.fibers.reverse();
    r.vcpus.reverse();
    assert_eq!(
        admit(&module(Some(A)), r).expect("admitted").into_parts(),
        real()
    );
}

/// #3034 — the one rule over a guest-written extent ([`extent_fits`]), as each engine's freeze applies
/// it where it reads one: the root's context (0), a vCPU's (the top of the pool), a fiber's (one past
/// its slot). Each region admits its empty frame base and its end, and nothing outside: not zero, not
/// a header word, not a neighbour's region, not past the arena.
#[test]
fn an_extent_fits_only_inside_its_own_region() {
    for ctx in [0, 1, MAX_SHADOW_CONTEXTS - 1] {
        if !A.region_fits(ctx) {
            continue;
        }
        let (base, end) = (A.frame_base(ctx), A.region_base(ctx) + A.stride);
        assert!(extent_fits(A, ctx, base), "ctx {ctx}: its empty frame base");
        assert!(extent_fits(A, ctx, end), "ctx {ctx}: a region exactly full");
        assert!(!extent_fits(A, ctx, 0), "ctx {ctx}: a zero word");
        assert!(!extent_fits(A, ctx, base - 1), "ctx {ctx}: its header");
        assert!(!extent_fits(A, ctx, end + 1), "ctx {ctx}: past its end");
    }
    assert!(
        !extent_fits(A, 0, A.frame_base(1)),
        "the root's word in a fiber's region"
    );
    assert!(
        !extent_fits(ShadowArena::EMPTY, 0, 0),
        "no arena: no extent"
    );
}
