//! **#964 / #1094 oracle enforcement** — the NULL guard on the interpreter tiers. Every module's
//! window seeds `[0, POWERBOX_NULL_GUARD)` `Unmapped` at init (the guard is **unconditional** now —
//! #1094, the one canonical layout), so a NULL dereference traps `MemoryFault` on the tree-walker
//! AND the bytecode engine (the tier differential). The reserved region is also **refused** to the
//! page ops (`unmap` below the guard returns a negative errno — the `mmap_min_addr` analogue that
//! keeps the JIT tiers' baked guard constant sound), and legal page ops above the guard still work.

use std::sync::Arc;

use temen_interp::{bytecode, Host, Region, Trap, Value};

const GUARD: i64 = temen_ir::POWERBOX_NULL_GUARD as i64;

/// f0: probe load at arg. f1: probe store at arg. f2 `(as, off, len)`: `unmap` via the AddressSpace
/// cap, returning its errno. `memory 17` = 128 KiB, fully mapped.
const BODY: &str = r#"
func (i64) -> (i64) {
block 0 (v0: i64) {
  vl = i64.load v0
  return vl
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  i64.store v0 v0
  return v0
  }
}
func (i32, i64, i64) -> (i64) {
block 0 (vas: i32, voff: i64, vlen: i64) {
  vr = call.cap 5 1 (i64, i64) -> (i64) vas (voff, vlen)
  return vr
  }
}
"#;

/// A plain powerbox module — no marker needed (#1094: the guard is unconditional).
fn module() -> temen_ir::Module {
    let src = format!("memory 17\nexport 0 func \"_start\" 0\n{BODY}");
    let m = temen_text::parse_module(&src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

fn tree(m: &temen_ir::Module, func: u32, arg: i64) -> Result<i64, Trap> {
    let mut fuel = 1_000_000u64;
    temen_interp::run(m, func, &[Value::I64(arg)], &mut fuel).map(|v| match v.first() {
        Some(Value::I64(x)) => *x,
        other => panic!("result {other:?}"),
    })
}

fn byte(m: &temen_ir::Module, func: u32, arg: i64) -> Result<i64, Trap> {
    let mut fuel = 1_000_000u64;
    bytecode::compile_and_run(m, func, &[Value::I64(arg)], &mut fuel)
        .expect("in bytecode subset")
        .map(|v| match v.first() {
            Some(Value::I64(x)) => *x,
            other => panic!("result {other:?}"),
        })
}

/// NULL loads and stores trap on every module — on both engines — while the first mapped address
/// above the guard works on both. #1094: no marker, the guard is unconditional.
#[test]
fn null_access_traps_on_both_engines() {
    let m = module();
    for (func, what) in [(0u32, "load"), (1u32, "store")] {
        for probe in [0i64, 8, GUARD - 8, GUARD - 1] {
            assert_eq!(
                tree(&m, func, probe),
                Err(Trap::MemoryFault),
                "tree-walk {what}({probe}) must trap under the guard"
            );
            assert_eq!(
                byte(&m, func, probe),
                Err(Trap::MemoryFault),
                "bytecode {what}({probe}) must trap under the guard"
            );
        }
        // At and above the guard: legal on every tier.
        for probe in [GUARD, (1 << 17) - 8] {
            assert!(tree(&m, func, probe).is_ok(), "tree {what}({probe}) admits");
            assert!(byte(&m, func, probe).is_ok(), "byte {what}({probe}) admits");
        }
    }
}

/// The reserved region is refused to the page ops: `unmap` below the guard returns a negative
/// errno; an `unmap` at/above the guard stays legal. Runs on the bytecode powerbox harness (the cap
/// needs a granted handle).
#[test]
fn page_ops_below_the_guard_are_refused() {
    let win = 1usize << 17;
    let unmap = |m: &temen_ir::Module, off: i64, len: i64| -> i64 {
        let prog = bytecode::VcpuProgram::compile(m).expect("compile");
        let layout = std::alloc::Layout::from_size_align(win, 8).unwrap();
        // SAFETY: a fresh zeroed window buffer, exclusively this run's, freed after the vCPU drops.
        let base = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!base.is_null());
        // SAFETY: `base` is `win` valid bytes owned here; the Region aliases it only for this run.
        let back = Arc::new(unsafe { Region::shared(base, win as u64) });
        let mut host = Host::new();
        let asl = host.grant_memory(); // the AddressSpace handle the guest's call.cap names
        let mut vcpu = bytecode::Vcpu::new_root_with_powerbox(
            &prog,
            2,
            &[Value::I32(asl), Value::I64(off), Value::I64(len)],
            back,
            &[],
            host,
        )
        .expect("vcpu");
        let r = match vcpu.run() {
            bytecode::VcpuEvent::Done(vals) => match vals.first() {
                Some(Value::I64(x)) => *x,
                other => panic!("result {other:?}"),
            },
            bytecode::VcpuEvent::Trapped(t) => panic!("unmap probe trapped: {t:?}"),
            _ => panic!("unmap probe did not finish"),
        };
        drop(vcpu);
        // SAFETY: the vCPU (and its Mem aliasing the region) is dropped; free the buffer.
        unsafe { std::alloc::dealloc(base, layout) };
        r
    };

    let m = module();
    let page = temen_interp::host_page_size() as i64;
    assert!(
        unmap(&m, 0, GUARD) < 0,
        "unmap of the reserved NULL region is refused"
    );
    assert!(
        unmap(&m, GUARD - page, page) < 0,
        "unmap of the guard's last page is refused"
    );
    assert_eq!(
        unmap(&m, GUARD, page),
        0,
        "unmap at the guard boundary is legal"
    );
}
