//! **The experimental NULL-page guard** (`compile_module_tierup_nullguard`) — the differential for
//! the trap-on-NULL measurement mode: every confined access traps when its first byte lands below
//! the guard, exactly where an interpreter whose page map seeds `[0, guard)` `Unmapped` traps
//! (`run_capture_reserved_with_host_prots` is the seeding seam). A *bottom* guard needs no
//! last-byte consultation — an access starting at or above the guard cannot reach down — but an
//! access *starting* below it traps even when its span crosses into mapped pages (the straddle
//! probe). Like the paged check, the guard is **never elided**: a constant-address access the
//! bound-check elision proves in-window still consults it (elision bounds from above, not below).
//!
//! The trap decision runs strictly inside the always-emitted `& MASK` clamp, so a wrong guard is a
//! trap-parity divergence, never an escape — the same INVARIANTS #2 argument as the #750 page
//! check this mode's lowering is weighed against.

use temen_interp::{run_capture_reserved_with_host_prots, CapturedProt, Host, Value};
use temen_wasm_jit::{compile_module_tierup_nullguard, TRAP_MEMORY_FAULT};
use wasmi::{Caller, Engine, Linker, Memory, MemoryType, Module as WModule, Store, Val};

const WIN_LOG2: u8 = 17; // 128 KiB window = 8 guard-sized (16 KiB) pages
const WIN_SIZE: u64 = 1 << WIN_LOG2;
/// 16 KiB — the **max host page** (macOS), so the seeded interp region is host-page-exact on every
/// CI host: the interpreter's page map coalesces to host-page granularity, and a 4 KiB guard on a
/// 16 KiB-page host would make the interp trap `[4096, 16384)` where the emitted guard admits
/// (exactly the macOS divergence this constant fixes). Also the #964 recommendation (the wider
/// NULL net that catches big-struct field derefs).
const GUARD: u64 = 16384;
const WIN_BASE: u32 = 0x2_0000;
const ENV_PTR: u32 = 1024;
const FUEL: u64 = 1_000_000;

/// f0: 8-byte load at the probe address. f1: 8-byte store at the probe. f2: a **constant**-address
/// load below the guard — the elision pin (the bound check is provably in-window and elidable; the
/// guard must fire anyway).
const SRC: &str = r#"memory 17
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
func () -> (i64) {
block 0 () {
  va = i64.const 8
  vl = i64.load va
  return vl
  }
}
"#;

fn build() -> temen_ir::Module {
    let m = temen_text::parse_module(SRC).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// Run the emitted `f{func}` under wasmi with the live-`mapped` global set to the window size.
/// `Ok(v)` on a clean return, `Err(trap_code)` on an emitted trap.
fn run_emitted(wasm: &[u8], func: u32, argv: &[i64]) -> Result<i64, i32> {
    let engine = Engine::default();
    let module = WModule::new(&engine, wasm).expect("emitted wasm validates");
    let mut store: Store<i32> = Store::new(&engine, 0);
    let pages = WIN_BASE.div_ceil(1 << 16) + (WIN_SIZE as u32).div_ceil(1 << 16) + 1;
    let memory = Memory::new(&mut store, MemoryType::new(pages, None)).unwrap();
    memory
        .write(&mut store, ENV_PTR as usize, &(FUEL as i64).to_le_bytes())
        .unwrap();
    let mut linker: Linker<i32> = Linker::new(&engine);
    linker.define("env", "memory", memory).unwrap();
    linker
        .func_wrap("env", "trap", |mut c: Caller<'_, i32>, code: i32| {
            *c.data_mut() = code;
        })
        .unwrap();
    linker
        .func_wrap::<_, ()>(
            "env",
            "call_interp",
            |_: Caller<'_, i32>, _: i32, _: i32| {
                unreachable!("pure leaves");
            },
        )
        .unwrap();
    let instance = linker
        .instantiate(&mut store, &module)
        .unwrap()
        .start(&mut store)
        .unwrap();
    instance
        .get_global(&store, "mapped")
        .expect("live-mapped global")
        .set(&mut store, Val::I64(WIN_SIZE as i64))
        .unwrap();
    let f = instance
        .get_func(&store, &format!("f{func}"))
        .unwrap_or_else(|| panic!("f{func} exported"));
    let mut params = vec![Val::I32(WIN_BASE as i32), Val::I32(ENV_PTR as i32)];
    params.extend(argv.iter().map(|&a| Val::I64(a)));
    let mut results = [Val::I64(0)];
    match f.call(&mut store, &params, &mut results) {
        Ok(()) => match results[0] {
            Val::I64(x) => Ok(x),
            _ => panic!("result type"),
        },
        Err(_) => Err(*store.data()),
    }
}

/// The oracle: the same probe on the interpreter over a page map seeding `[0, GUARD)` `Unmapped`
/// (everything else `Rw`) — the interp half of the trap-on-NULL design, available today through the
/// durable prot-seeding seam. `Ok(v)` / `Err(())` on a trap.
fn run_oracle(m: &temen_ir::Module, func: u32, argv: &[i64]) -> Result<i64, ()> {
    let npages = (WIN_SIZE / 4096) as usize; // CapturedProt granularity (DURABLE_SNAPSHOT_PAGE)
    let mut prots = vec![CapturedProt::Rw; npages];
    for slot in prots.iter_mut().take((GUARD / 4096) as usize) {
        *slot = CapturedProt::Unmapped;
    }
    let args: Vec<Value> = argv.iter().map(|&a| Value::I64(a)).collect();
    let mut fuel = FUEL;
    let mut host = Host::new();
    let init_mem = vec![0u8; WIN_SIZE as usize];
    let (res, _, _) = run_capture_reserved_with_host_prots(
        m,
        func,
        &args,
        &mut fuel,
        &init_mem,
        Some(&prots),
        WIN_LOG2,
        &mut host,
    );
    match res {
        Ok(vals) => match vals.first() {
            Some(Value::I64(x)) => Ok(*x),
            _ => panic!("oracle result"),
        },
        Err(_) => Err(()),
    }
}

/// Trap-parity across the guard boundary: loads and stores below the guard (including a straddle
/// whose first byte is below it) trap on BOTH tiers; at and above the guard both admit.
#[test]
fn guard_traps_match_the_seeded_interpreter() {
    let m = build();
    let (wasm, emitted) = compile_module_tierup_nullguard(&m, false, GUARD).expect("emits");
    assert_eq!(emitted, vec![true, true, true], "all three leaves emit");

    // (probe, expect_trap): below / straddling-from-below / at / inside the window.
    let probes: [(i64, bool); 6] = [
        (0, true),
        (8, true),
        (GUARD as i64 - 8, true), // 8-byte span entirely in the NULL region
        (GUARD as i64 - 1, true), // first byte below the guard, span crosses past it
        (GUARD as i64, false),
        (WIN_SIZE as i64 - 8, false),
    ];
    for func in [0u32, 1] {
        for (probe, expect_trap) in probes {
            let e = run_emitted(&wasm, func, &[probe]);
            let o = run_oracle(&m, func, &[probe]);
            assert_eq!(
                e.is_err(),
                o.is_err(),
                "tier divergence at f{func}({probe}): emitted {e:?} vs interp {o:?}"
            );
            assert_eq!(e.is_err(), expect_trap, "f{func}({probe})");
            if let Err(code) = e {
                assert_eq!(code, TRAP_MEMORY_FAULT, "f{func}({probe}) trap kind");
            }
        }
    }
}

/// The elision pin: a **constant** address below the guard — whose bound check the in-window
/// elision may drop — still traps on both tiers. A guard folded into the (elidable) bound check
/// would miss exactly this case; the dedicated check must not.
#[test]
fn guard_is_never_elided() {
    let m = build();
    let (wasm, _) = compile_module_tierup_nullguard(&m, false, GUARD).expect("emits");
    let e = run_emitted(&wasm, 2, &[]);
    let o = run_oracle(&m, 2, &[]);
    assert!(e.is_err(), "constant addr 8 must trap under the guard");
    assert!(o.is_err(), "oracle traps the same access");
    assert_eq!(e.unwrap_err(), TRAP_MEMORY_FAULT);
}

/// #1094: the guard is **unconditional**, so the plain tier-up entry now derives it from
/// `module_null_guard` (always `Some(GUARD)`) on every standard entry — its emit is byte-identical to
/// the explicit-`GUARD` measurement emit, and it traps a NULL access (where the pre-#1094 unguarded
/// plain emit admitted) while admitting at the guard boundary. No `__null_guard` marker is involved.
#[test]
fn plain_emit_carries_the_unconditional_guard() {
    let m = build();
    assert_eq!(
        temen_ir::module_null_guard(),
        GUARD,
        "the guard is unconditional (#1094) — no marker needed"
    );

    let (plain, e1) = temen_wasm_jit::compile_module_tierup(&m, false).expect("plain emits");
    let (explicit, e2) =
        compile_module_tierup_nullguard(&m, false, temen_ir::POWERBOX_NULL_GUARD).expect("emits");
    assert_eq!(
        plain, explicit,
        "the plain entry derives the same guard as the explicit measurement entry"
    );
    assert_eq!(e1, e2);
    assert_eq!(
        run_emitted(&plain, 0, &[8]),
        Err(TRAP_MEMORY_FAULT),
        "plain emit traps a NULL load"
    );
    assert_eq!(
        run_emitted(&plain, 0, &[GUARD as i64]),
        Ok(0),
        "plain emit admits at the guard boundary"
    );
}

/// #1004: the **bulk-memory** span check carries the same guard low bound — a `mem.fill`/`mem.copy`
/// whose span dips into `[0, guard)` traps on BOTH tiers, and one at or above the guard admits on
/// both, exactly where the interpreter's `check_prot_span` faults on the seeded `Unmapped` guard
/// pages. The copy exercises two spans (dst and src), each guard-checked; a zero-length op is a
/// no-op that never faults even at base 0 (the `if len != 0` short-circuit). The functions emitting
/// at all (`emitted == [true, true]`) is half the pin — before #1004 a marked module's bulk-mem
/// functions left the subset and never reached the wasm tier.
#[test]
fn bulk_guard_traps_match_the_seeded_interpreter() {
    // f0(base, len): fill `[base, base+len)` with a constant byte, return base.
    // f1(dst, src, len): copy `len` bytes src→dst, return dst.
    let src = r#"memory 17
func (i64, i64) -> (i64) {
block 0 (vbase: i64, vlen: i64) {
  vbyte = i32.const 171
  mem.fill vbase vbyte vlen
  return vbase
  }
}
func (i64, i64, i64) -> (i64) {
block 0 (vdst: i64, vsrc: i64, vlen: i64) {
  mem.copy vdst vsrc vlen
  return vdst
  }
}
"#;
    let m = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let (wasm, emitted) = compile_module_tierup_nullguard(&m, false, GUARD).expect("emits");
    assert_eq!(
        emitted,
        vec![true, true],
        "#1004: bulk-mem functions now emit under the guard"
    );

    // (base, len, expect_trap): the span's lowest byte vs the guard.
    let cases: [(i64, i64, bool); 6] = [
        (0, 8, true),                      // wholly below the guard
        (GUARD as i64 - 8, 8, true),       // span ends at the guard, base below
        (GUARD as i64 - 1, 32, true),      // straddles: base below, span crosses past
        (GUARD as i64, 64, false),         // exactly at the guard
        (GUARD as i64 + 4096, 128, false), // well inside the window
        (0, 0, false),                     // zero length: no-op, never faults even at base 0
    ];
    // fill (f0): args (base, len).
    for (base, len, expect_trap) in cases {
        let e = run_emitted(&wasm, 0, &[base, len]);
        let o = run_oracle(&m, 0, &[base, len]);
        assert_eq!(
            e.is_err(),
            o.is_err(),
            "fill divergence at ({base}, {len}): emitted {e:?} vs interp {o:?}"
        );
        assert_eq!(e.is_err(), expect_trap, "fill({base}, {len})");
        if let Err(code) = e {
            assert_eq!(code, TRAP_MEMORY_FAULT, "fill({base}, {len}) trap kind");
        }
    }
    // copy (f1): dst a safe high region, src the probed span — the src span is guard-checked too.
    let dst = WIN_SIZE as i64 - 4096;
    for (base, len, expect_trap) in cases {
        let e = run_emitted(&wasm, 1, &[dst, base, len]);
        let o = run_oracle(&m, 1, &[dst, base, len]);
        assert_eq!(
            e.is_err(),
            o.is_err(),
            "copy(src) divergence at ({base}, {len}): emitted {e:?} vs interp {o:?}"
        );
        assert_eq!(e.is_err(), expect_trap, "copy src span ({base}, {len})");
    }
}
