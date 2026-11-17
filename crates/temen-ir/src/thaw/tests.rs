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
    let t = validate(&module(Some(A)), 0, real()).expect("admitted");
    assert_eq!(t.into_parts(), real());
    assert!(validate(&module(Some(A)), 0, Raw::default())
        .expect("empty")
        .is_empty());
    // A module without an arena admits only the empty residue (INVARIANTS #16).
    assert!(validate(&module(None), 0, Raw::default()).is_ok());
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
    ];
    for (name, bend) in cases {
        let mut r = real();
        bend(&mut r);
        assert_eq!(
            validate(&module(Some(A)), 0, r).map(|_| ()),
            Err(Malformed),
            "{name}"
        );
    }
    // Any residue at all on a module with no arena.
    let mut r = Raw::default();
    r.fibers.push(FrozenFiber::free(0, 0));
    assert_eq!(validate(&module(None), 0, r).map(|_| ()), Err(Malformed));
}
