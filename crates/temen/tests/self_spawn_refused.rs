//! #2219 — **a child runs a module the spawner holds, from its function 0, on every engine.** A spawn
//! that names `-1`, which once named the spawner's own program, faults: `-1` is a forged `Module`
//! handle like any other. So does an entry other than 0 in a record's word at 4: a child starts at
//! its module's child bootstrap. And `Module` op 0, which looked an export up by name to pick an
//! entry, is gone: a `Module` handle has no callable op. #2067 — so is op 15, the positional spawn:
//! the v1 record (op 17) is the one spawn form. Each traps `CapFault` on the tree-walker, each
//! bytecode driver and the Cranelift JIT alike, so no engine runs a spawn another refuses
//! (INVARIANTS #9).
//!
//! Each form is otherwise well-formed — a budget that pays, a child module whose function 0 is a
//! child entry, an export the lookup would find — so only the retired part is refused, and the same
//! harness runs the record to its join with the granted module at entry 0.

#[path = "../../temen-interp/tests/support/drivers.rs"]
mod drivers;
#[path = "../../temen-interp/tests/support/rec.rs"]
mod rec;

use temen_interp::{Host, MemLayout, Trap, Value};
use temen_ir::{Module, SpawnRec};
use temen_jit::{JitOutcome, TrapKind};

fn module(src: &str) -> Module {
    let m = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// The granted child: a 4 KiB window, its function 0 a child entry that returns 42, exported as
/// `"go"`.
const CHILD: &str = "memory 12
export 0 func \"go\" 0
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 42
  return v1
  }
}
";

/// Where every parent keeps its spawn record.
const REC: u64 = 17408;
/// Where the parent keeps the export name `"go"`.
const NAME: u64 = 17536;

/// A v1 record at [`REC`] whose word at 4, the retired entry, is `entry`.
fn record(entry: u32) -> String {
    let mut b = SpawnRec::v1().encode();
    b[4..8].copy_from_slice(&entry.to_le_bytes());
    let bytes: String = b.iter().map(|x| format!("\\x{x:02x}")).collect();
    format!("data {REC} \"{bytes}\"\ndata {NAME} \"go\"\n")
}

/// A parent `(inst, module, budget) -> i64` that runs `body` and returns its `vr`, then func 1, an
/// entry that returns 7, which a spawn of the parent's own program would reach. `body` sees `vi`
/// (the `Instantiator`), `vmod` and `vb` (the module and the `Budget`), `vm` and `vb64` (both as
/// `i64`s), `vz` (0) and `vp` (the record at [`REC`]), whose `budget` is already stored.
fn parent(body: &str, data: &str) -> Module {
    module(&format!(
        "memory 17
func (i32, i32, i32) -> (i64) {{
block 0 (vi: i32, vmod: i32, vb: i32) {{
  vm = i64.extend_i32_s vmod
  vb64 = i64.extend_i32_s vb
  vz = i64.const 0
  vp = i64.const {REC}
  vba = i64.const {budget_at}
  i32.store vba vb
{body}
  return vr
  }}
}}
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  v1 = i64.const 7
  return v1
  }}
}}
{data}",
        budget_at = REC + rec::BUDGET_AT,
    ))
}

/// Text IR that stores the `i32` in `handle` as the record's module.
fn module_is(handle: &str) -> String {
    format!(
        "  vma = i64.const {}\n  i32.store vma {handle}\n",
        REC + rec::MODULE_AT
    )
}

/// Spawn through the record, join the child, and return its result.
const SPAWN_REC: &str = "  vh = call.cap 6 17 (i64) -> (i32) vi (vp)
  vst, vr = call.cap 6 1 (i32) -> (i64, i64) vi (vh)";

/// The retired op 15's 7-arg form `(budget, module, grants_ptr, grants_n, entry, size_log2, quota)`,
/// naming the granted module at entry 0 — the spawn the record makes; join the child and return its
/// result.
const OP15: &str = "  vh = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) vi (vb64, vm, vz, vz, vz, vz, vz)
  vst, vr = call.cap 6 1 (i32) -> (i64, i64) vi (vh)";

fn forms() -> Vec<(&'static str, Module)> {
    vec![
        (
            "a record naming module -1",
            parent(
                &format!("  vself = i32.const -1\n{}{SPAWN_REC}", module_is("vself")),
                &record(0),
            ),
        ),
        (
            "a record with entry 1",
            parent(&format!("{}{SPAWN_REC}", module_is("vmod")), &record(1)),
        ),
        ("op 15, the retired positional spawn", parent(OP15, &record(0))),
        (
            "Module op 0",
            parent(
                &format!(
                    "  vn = i64.const {NAME}\n  vl = i64.const 2\n  vr = call.cap 8 0 (i64, i64) -> (i64) vmod (vn, vl)"
                ),
                &record(0),
            ),
        ),
    ]
}

/// The parent's powerbox: an `Instantiator` over its window, the granted child module, and a
/// `Budget` the child's window spends, with their handles.
fn powerbox() -> (Host, [i32; 3]) {
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, 1 << 17);
    let child = host.grant_module(&module(CHILD));
    let budget = host.grant_budget(-1, 1 << 20, -1);
    (host, [inst, child, budget])
}

fn setup() -> (Host, Vec<Value>) {
    let (host, h) = powerbox();
    (host, h.map(Value::I32).to_vec())
}

/// The Cranelift JIT as an embedder drives it (`temen_run::jit_cap_run`): the reference host and
/// the production hooks a spawn's admission needs, without which a valid spawn faults too.
fn cranelift(m: &Module) -> JitOutcome {
    let (mut host, h) = powerbox();
    temen_run::jit_cap_run(
        m,
        0,
        &h.map(i64::from),
        &MemLayout::image(Vec::new()),
        temen_ir::DEFAULT_RESERVED_LOG2,
        0,
        &mut host,
        None,
    )
    .expect("the JIT compiles the module")
    .0
}

#[test]
fn every_engine_traps_a_self_spawn_or_an_entry_with_capfault() {
    for (form, m) in forms() {
        for driver in drivers::ALL {
            let ran = drivers::run_on(driver, &m, &setup)
                .unwrap_or_else(|| panic!("{form}: {driver:?} declined the module"));
            assert_eq!(ran.result, Err(Trap::CapFault), "{form} on {driver:?}");
        }
        assert_eq!(
            cranelift(&m),
            JitOutcome::Trapped(TrapKind::CapFault),
            "{form} on the Cranelift JIT"
        );
    }
}

/// The control: the record, naming the granted module at entry 0, spawns and joins to its 42 on
/// every engine — so the harness can spawn, and each refusal above is the retired part's.
#[test]
fn the_same_harness_runs_a_granted_module_from_function_0() {
    for (form, m) in [(
        "the record",
        parent(&format!("{}{SPAWN_REC}", module_is("vmod")), &record(0)),
    )] {
        for driver in drivers::ALL {
            let ran = drivers::run_on(driver, &m, &setup)
                .unwrap_or_else(|| panic!("{form}: {driver:?} declined the module"));
            assert_eq!(ran.result, Ok(vec![Value::I64(42)]), "{form} on {driver:?}");
        }
        assert_eq!(
            cranelift(&m),
            JitOutcome::Returned(vec![42]),
            "{form} on the Cranelift JIT"
        );
    }
}
