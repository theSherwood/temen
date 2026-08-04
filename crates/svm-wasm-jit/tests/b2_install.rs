//! **§22 Model B2 on the wasm-JIT tier** — an `install`ed unit becomes a funcref that *another*
//! instance's `call_indirect` reaches, through one **shared reserved funcref table** (`DESIGN.md`
//! §22, BROWSER.md "wasm-JIT tier"). This is the old→new cross-instance direction the local
//! per-module table cannot express.
//!
//! The mechanism under test is [`svm_wasm_jit::compile_module_b2`]: a module *imports*
//! `env.__indirect_function_table` (sized to the domain reservation `1<<log2`) instead of declaring
//! its own, and populates nothing — the **host** writes every slot via `table.set`, exactly as the
//! interpreter's `DomainTable` is host-populated (`DomainTable::new` + `install`). So here the host
//! plays the domain: it emits a *caller* (whose `call_indirect` masks into the shared table) and a
//! *callee unit*, `table.set`s the callee's exported func into a reserved slot, then invokes the
//! caller — whose indirect call must land in the callee.
//!
//! The oracle (INVARIANTS.md #9) is the tree-walk interpreter running the callee unit directly: a B2
//! dispatch must return exactly what the interpreter returns for that unit (value), and must fail
//! **closed** on an empty slot or a signature mismatch (as `dispatch_indirect` traps on `TABLE_EMPTY`
//! / a type-id mismatch). The reserved-size mask (`idx & (1<<log2 - 1)`) is pinned by a wrap case.

use svm_interp::{run, Value};
use svm_wasm_jit::{compile_module_b2, compile_module_with};
use wasmi::core::ValType;
use wasmi::{
    Engine, FuncRef, Linker, Memory, MemoryType, Module as WModule, Store, Table, TableType, Val,
};

/// The guest window base in the wasmi harness (page 1; page 0 holds the env cell).
const WIN_BASE: i32 = 0x1_0000;
/// The env cell (i64 fuel counter) — outside the window.
const ENV_PTR: i32 = 1024;
/// `2^LOG2` = the reserved shared-table size, matching a `grant_jit_with_table(_, LOG2)` domain.
const LOG2: u32 = 4; // 16 slots

fn parse(src: &str) -> svm_ir::Module {
    let m = svm_text::parse_module(src).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    m
}

/// The installed unit: `() -> (i32)` returning a constant. Its emitted `f0` has the env-prepended
/// wasm signature `(win:i32, env:i32) -> (i32)` — identical to what the caller's `call_indirect`
/// `() -> (i32)` declares, so wasm's signature check (= the §3c type-id check) accepts it.
const CALLEE_I32: &str = r#"memory 16
func () -> (i32) {
block 0 () {
  vr = i32.const 42
  return vr
  }
}
"#;

/// A differently-typed unit (`() -> (i64)`) — installed to prove a signature mismatch fails closed.
const CALLEE_I64: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  vr = i64.const 42
  return vr
  }
}
"#;

/// The caller (old code): takes a slot index and `call_indirect`s it with the callee's `() -> (i32)`
/// signature. The emitter masks the index `& (1<<LOG2 - 1)` into the shared table.
const CALLER: &str = r#"memory 16
func (i32) -> (i32) {
block 0 (vslot: i32) {
  vr = call_indirect () -> (i32) vslot ()
  return vr
  }
}
"#;

/// A **middle** installed unit that is *itself* a `call_indirect` caller: `() -> (i32)` that dispatches
/// the constant inner slot 5. Installed at another slot, it proves a new→new chain — an installed unit
/// reaching a second installed unit through the same shared table. It imports the shared table (B2).
const MIDDLE_TO_SLOT5: &str = r#"memory 16
func () -> (i32) {
block 0 () {
  vinner = i32.const 5
  vr = call_indirect () -> (i32) vinner ()
  return vr
  }
}
"#;

/// One unit the host installs into the shared table before the top-level dispatch: its source and the
/// reserved slot it goes into. A unit with its own `call_indirect` must be emitted in B2 mode (so it
/// imports the shared table); a leaf (no indirect call) is a plain whole-module emit.
struct Install<'a> {
    src: &'a str,
    slot: u32,
    b2: bool,
}

/// Emit + wire a shared-table domain in one wasmi `Store`: install each unit's `f0` into its slot,
/// then `uninstall` (null out) every slot in `clear`, then call the B2 [`CALLER`] with `slot_arg`.
/// Returns the `i32` result, or `Err` on a trap. This is the host playing the domain — `table.set`
/// per install is the JS `install`/grant analog; nulling a slot is `uninstall` (reusable; stale
/// calls trap).
fn b2_run(installs: &[Install], clear: &[u32], slot_arg: i32) -> Result<i32, ()> {
    let engine = Engine::default();
    let mut store: Store<i32> = Store::new(&engine, 0);
    let memory = Memory::new(&mut store, MemoryType::new(2, None)).unwrap();
    memory
        .write(&mut store, ENV_PTR as usize, &i64::MAX.to_le_bytes())
        .unwrap();

    // The one shared reserved table (all null → every slot traps until the host writes it).
    let table = Table::new(
        &mut store,
        TableType::new(ValType::FuncRef, 1 << LOG2, Some(1 << LOG2)),
        Val::FuncRef(FuncRef::null()),
    )
    .unwrap();

    let mut linker: Linker<i32> = Linker::new(&engine);
    linker.define("env", "memory", memory).unwrap();
    linker
        .func_wrap(
            "env",
            "trap",
            |mut caller: wasmi::Caller<'_, i32>, code: i32| {
                *caller.data_mut() = code;
            },
        )
        .unwrap();
    linker
        .func_wrap::<_, ()>(
            "env",
            "call_interp",
            |_: wasmi::Caller<'_, i32>, _f: i32, _a: i32| {
                unreachable!("no cross-tier call in these pure units");
            },
        )
        .unwrap();
    linker
        .define("env", "__indirect_function_table", table)
        .unwrap();

    // Install each unit: instantiate it, grab its `f0`, write it into its slot (the host owning slots).
    for u in installs {
        let m = parse(u.src);
        let wasm = if u.b2 {
            compile_module_b2(&m, false, LOG2).expect("unit emits (B2)")
        } else {
            compile_module_with(&m, false).expect("unit emits")
        };
        let module = WModule::new(&engine, &wasm).expect("unit wasm validates");
        let inst = linker
            .instantiate(&mut store, &module)
            .unwrap()
            .start(&mut store)
            .unwrap();
        let f0 = inst.get_func(&store, "f0").expect("unit f0");
        table
            .set(&mut store, u.slot as u64, Val::FuncRef(FuncRef::new(f0)))
            .unwrap();
    }

    // `uninstall`: the host nulls the slot back to trapping padding (a later `install` may reuse it;
    // a stale `call_indirect` to it now traps — DESIGN.md §22).
    for &slot in clear {
        table
            .set(&mut store, slot as u64, Val::FuncRef(FuncRef::null()))
            .unwrap();
    }

    // Instantiate the top caller (imports the same shared table) and dispatch.
    let caller = parse(CALLER);
    let caller_wasm = compile_module_b2(&caller, false, LOG2).expect("caller emits (B2)");
    let caller_mod = WModule::new(&engine, &caller_wasm).expect("caller wasm validates");
    let caller_inst = linker
        .instantiate(&mut store, &caller_mod)
        .unwrap()
        .start(&mut store)
        .unwrap();
    let caller_f0 = caller_inst.get_func(&store, "f0").expect("caller f0");

    let params = [Val::I32(WIN_BASE), Val::I32(ENV_PTR), Val::I32(slot_arg)];
    let mut results = [Val::I32(0)];
    match caller_f0.call(&mut store, &params, &mut results) {
        Ok(()) => Ok(results[0].i32().expect("i32 result")),
        Err(_) => Err(()),
    }
}

/// Install one leaf unit at `slot` and dispatch `slot_arg` through the top caller.
fn b2_dispatch(callee_src: &str, slot: u32, slot_arg: i32) -> Result<i32, ()> {
    // A leaf callee (no `call_indirect`) needs no table import.
    b2_run(
        &[Install {
            src: callee_src,
            slot,
            b2: false,
        }],
        &[],
        slot_arg,
    )
}

/// The interpreter oracle: the callee unit run directly (INVARIANTS.md #9).
fn oracle_i32(src: &str) -> i32 {
    let m = parse(src);
    let mut fuel = u64::MAX;
    match run(&m, 0, &[], &mut fuel) {
        Ok(v) => match v.first() {
            Some(Value::I32(x)) => *x,
            other => panic!("oracle result: {other:?}"),
        },
        other => panic!("oracle: {other:?}"),
    }
}

/// old→new: a `call_indirect` from one instance dispatches to an `install`ed unit in the shared
/// table, returning exactly what the interpreter returns for that unit.
#[test]
fn b2_dispatch_matches_interp() {
    let want = oracle_i32(CALLEE_I32);
    assert_eq!(want, 42, "oracle: callee() = 42");
    // Slot 5 is a reserved padding slot (≥ the caller's own func count) — where `install` lands.
    assert_eq!(
        b2_dispatch(CALLEE_I32, 5, 5),
        Ok(want),
        "B2 dispatch != interp"
    );
}

/// The confinement mask is the *reserved* size: an index past the table wraps `& (1<<LOG2 - 1)` to
/// the same installed slot (never out of bounds — invariant I2).
#[test]
fn b2_dispatch_masks_to_reserved_size() {
    let want = oracle_i32(CALLEE_I32);
    // 5 + 16 wraps to 5 under the 16-slot mask; 5 + 32 too. Both must still hit the installed unit.
    assert_eq!(
        b2_dispatch(CALLEE_I32, 5, 5 + 16),
        Ok(want),
        "mask wrap +16"
    );
    assert_eq!(
        b2_dispatch(CALLEE_I32, 5, 5 + 32),
        Ok(want),
        "mask wrap +32"
    );
}

/// new→new: an installed unit `call_indirect`s a *second* installed unit through the same shared
/// table (a 2-hop chain top → middle@6 → callee@5), landing at the interpreter's value. This is the
/// transitivity B2 rests on — every hop dispatches through one host-populated table.
#[test]
fn b2_chained_install_dispatch_matches_interp() {
    let want = oracle_i32(CALLEE_I32);
    let installs = [
        Install {
            src: CALLEE_I32,
            slot: 5,
            b2: false,
        },
        Install {
            src: MIDDLE_TO_SLOT5,
            slot: 6,
            b2: true,
        },
    ];
    // Dispatch the middle unit at slot 6; it re-dispatches slot 5 to the callee.
    assert_eq!(
        b2_run(&installs, &[], 6),
        Ok(want),
        "chained B2 dispatch != interp"
    );
}

/// Fail-closed: a `call_indirect` to an empty (never-installed) reserved slot traps, as
/// `dispatch_indirect` traps on `TABLE_EMPTY`.
#[test]
fn b2_empty_slot_traps() {
    // Install at slot 5 but call slot 7 (empty) — masks in-bounds, but the slot is null.
    assert_eq!(
        b2_dispatch(CALLEE_I32, 5, 7),
        Err(()),
        "empty slot must trap"
    );
}

/// Fail-closed: dispatching a slot whose installed unit has the *wrong* signature traps — wasm's
/// call_indirect signature check is the §3c type-id check.
#[test]
fn b2_type_mismatch_traps() {
    // Install an `() -> (i64)` unit at slot 5; the caller declares `() -> (i32)`.
    assert_eq!(
        b2_dispatch(CALLEE_I64, 5, 5),
        Err(()),
        "signature mismatch must trap"
    );
}

/// `uninstall`: after a unit is installed and dispatchable, nulling its slot makes a stale
/// `call_indirect` to it trap (DESIGN.md §22 — "clear an installed slot; stale calls trap"), while a
/// *different* still-installed slot keeps dispatching. Proves the slot is reusable/clearable, not
/// merely append-only.
#[test]
fn b2_uninstall_makes_stale_call_trap() {
    let want = oracle_i32(CALLEE_I32);
    let installs = [Install {
        src: CALLEE_I32,
        slot: 5,
        b2: false,
    }];
    // Sanity: installed → dispatches.
    assert_eq!(
        b2_run(&installs, &[], 5),
        Ok(want),
        "installed must dispatch"
    );
    // Uninstall slot 5, then dispatch it → trap (the slot is null padding again).
    assert_eq!(
        b2_run(&installs, &[5], 5),
        Err(()),
        "stale call after uninstall must trap"
    );
}

/// **The confinement mask as its own unit** (INVARIANTS.md §4). Install a distinct unit in *every*
/// reserved slot — each returns its own slot index — then sweep indices across (and past) the table:
/// a `call_indirect idx` must land in slot `idx & (size - 1)` for every index, i.e. the emitted mask
/// is exactly `dispatch_indirect`'s and can never address outside `[0, size)`. Over-range indices
/// (`idx ≥ size`, up to `4·size`) prove the wrap; this is the security hinge the whole tier rests on.
#[test]
fn b2_mask_confines_every_index_to_reserved_table() {
    let size = 1u32 << LOG2; // 16
                             // A unit per slot: `() -> (i32)` returning the slot's own index (owned Strings kept alive below).
    let srcs: Vec<String> = (0..size)
        .map(|s| format!("memory 16\nfunc () -> (i32) {{\nblock 0 () {{\n  vr = i32.const {s}\n  return vr\n  }}\n}}\n"))
        .collect();
    let installs: Vec<Install> = (0..size)
        .map(|s| Install {
            src: &srcs[s as usize],
            slot: s,
            b2: false,
        })
        .collect();
    // Sweep well past the table (0 .. 4·size) — every index must resolve to `idx & (size-1)`.
    for idx in 0..(4 * size) {
        let want = (idx & (size - 1)) as i32;
        assert_eq!(
            b2_run(&installs, &[], idx as i32),
            Ok(want),
            "call_indirect {idx} must land in slot {want} (mask idx & {})",
            size - 1
        );
    }
}

/// **The per-Worker table-mirror design** (de-risks cross-Worker B2). wasm funcrefs can't cross
/// Workers, so cross-Worker B2 gives each Worker its *own* `WebAssembly.Table` materialized from the
/// shared slot→unit map: each Worker instantiates an installed unit locally and `table.set`s its own
/// funcref at the shared slot index. This models that — 8 threads each build an **independent** domain
/// (own engine/store/table) from the *same* install map and dispatch, and every one agrees with the
/// interpreter oracle. So a domain replicated per Worker dispatches consistently, no funcref sharing
/// required. (The remaining browser work is wiring this into the par `Jit.invoke`/`install` path.)
#[test]
fn b2_per_worker_mirror_is_consistent() {
    let want = oracle_i32(CALLEE_I32);
    let installs = [Install {
        src: CALLEE_I32,
        slot: 5,
        b2: false,
    }];
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..8)
            .map(|_| scope.spawn(|| b2_run(&installs, &[], 5)))
            .collect();
        for h in handles {
            assert_eq!(
                h.join().unwrap(),
                Ok(want),
                "a per-Worker mirror diverged from the oracle"
            );
        }
    });
}
