//! #717 — the wasm-JIT bounds-trap must confine against the **live** window `mapped` size, not the
//! emit-time constant. The emitter now reads the live size from an exported mutable `"mapped"` global
//! (self-initialized to `1 << size_log2`), so an access into memory a guest grew at runtime via
//! `vm_map` no longer faults on the JIT where the interpreter — which admits the freshly-committed
//! pages — allows it.
//!
//! These tests drive the emit change **directly and as its own unit** (AGENTS.md: "fuzz the
//! confinement-masking lowering as its own unit"): compile a kernel, set the `"mapped"` global to a
//! chosen live extent `M`, and assert the emitted trap boundary tracks `M`. The analytical oracle is
//! exactly the interpreter's net `confine_span`/`check_prot` behavior over a window committed to `M`
//! (a byte `< M` is backed → admitted; `>= M` is uncommitted → faults). The default-`M` case
//! reproduces the old baked-constant behavior, pinned already by the full `differential.rs` suite.
//!
//! The check is the whole of escape safety here: the emitted module's memory is the embedder's linear
//! memory and `& MASK` clamps only to the 2^40 reservation, so whatever `M` the host writes, the check
//! must admit no more than `[0, M)` — including an `M` smaller than the access (the tests below
//! drive `M` down to 0).

use temen_wasm_jit::{compile_module, TRAP_MEMORY_FAULT};
use wasmi::{Caller, Engine, Linker, Memory, MemoryType, Module as WModule, Store, Val};

const WIN_BASE: u32 = 0x1_0000;
const ENV_PTR: u32 = 1024;
const FUEL: i64 = 1 << 40;
/// The largest live extent any test drives the `"mapped"` global to; the harness sizes its wasmi
/// linear memory to physically cover `WIN_BASE + MAX_EXTENT` so a *non*-faulting grown access lands in
/// real memory (a bounds bug then diverges instead of reading adjacent bytes).
const MAX_EXTENT: u32 = 1 << 20;

#[derive(Debug, PartialEq)]
enum R {
    Val(i64),
    MemFault,
    OtherTrap,
}

/// A kernel compiled from TEMEN-text and instantiated once under wasmi, run at chosen live extents.
struct Kernel {
    store: Store<i32>,
    instance: wasmi::Instance,
    results: usize,
}

impl Kernel {
    fn new(src: &str) -> Kernel {
        let m = temen_text::parse_module(src).expect("parse");
        temen_verify::verify_module(&m).expect("verify");
        let wasm = compile_module(&m).expect("emit");

        let engine = Engine::default();
        let module = WModule::new(&engine, &wasm).expect("emitted wasm must validate");
        let mut store: Store<i32> = Store::new(&engine, 0);
        let pages = (WIN_BASE + MAX_EXTENT) / 0x1_0000 + 2;
        let memory = Memory::new(&mut store, MemoryType::new(pages, None)).unwrap();
        let mut linker: Linker<i32> = Linker::new(&engine);
        linker.define("env", "memory", memory).unwrap();
        linker
            .func_wrap("env", "trap", |mut caller: Caller<'_, i32>, code: i32| {
                *caller.data_mut() = code;
            })
            .unwrap();
        linker
            .func_wrap::<_, ()>(
                "env",
                "call_interp",
                |_: Caller<'_, i32>, _f: i32, _a: i32| unreachable!("in-subset kernel"),
            )
            .unwrap();
        let instance = linker
            .instantiate(&mut store, &module)
            .unwrap()
            .start(&mut store)
            .unwrap();
        let results = m.funcs[0].results.len();
        Kernel {
            store,
            instance,
            results,
        }
    }

    /// Run `f0(win, env, ...args)` with the `"mapped"` global forced to `live_mapped`. Returns the
    /// single result, or the trap class.
    fn run(&mut self, args: &[i64], live_mapped: u64) -> R {
        let store = &mut self.store;
        *store.data_mut() = 0;
        if let Some(g) = self.instance.get_global(&*store, "fuel") {
            g.set(&mut *store, Val::I64(FUEL)).unwrap();
        }
        // The change under test: override the live window size the bounds check reads.
        self.instance
            .get_global(&*store, "mapped")
            .expect("mapped global exported")
            .set(&mut *store, Val::I64(live_mapped as i64))
            .unwrap();

        let f = self.instance.get_func(&*store, "f0").expect("f0 exported");
        let mut params = vec![Val::I32(WIN_BASE as i32), Val::I32(ENV_PTR as i32)];
        for a in args {
            params.push(Val::I64(*a));
        }
        let mut results = vec![Val::I64(0); self.results];
        match f.call(&mut *store, &params, &mut results) {
            Ok(()) => R::Val(match results.first() {
                Some(Val::I64(x)) => *x,
                Some(Val::I32(x)) => *x as i64,
                _ => 0,
            }),
            Err(_) => {
                if *store.data() == TRAP_MEMORY_FAULT {
                    R::MemFault
                } else {
                    R::OtherTrap
                }
            }
        }
    }
}

/// Compile `src` (TEMEN-text), instantiate under wasmi with the `"mapped"` global forced to
/// `live_mapped`, and run `f0(win, env, ...args)`. Returns the single i64 result, or the trap class.
fn run_at_mapped(src: &str, args: &[i64], live_mapped: u64) -> R {
    Kernel::new(src).run(args, live_mapped)
}

/// `memory 16` ⇒ emit-time mapped = 65536. A load of 8 bytes from a param address.
const LOAD8: &str = r#"
memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  vr = i64.load v0
  return vr
  }
}
"#;

/// Store a value at a param address then read it back — proves a grown access truly reads/writes the
/// grown region (not merely "does not trap").
const STORE_LOAD8: &str = r#"
memory 16
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  i64.store v0 v1
  vr = i64.load v0
  return vr
  }
}
"#;

/// Bulk `mem.fill dst val len` — exercises `emit_span_check` (the whole-span analogue of the scalar
/// confine), which also confined against the emit-time constant before #717.
const FILL: &str = r#"
memory 16
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  vv = i32.const 171
  mem.fill v0 vv v1
  vr = i64.const 0
  return vr
  }
}
"#;

const DECLARED: u64 = 1 << 16; // 65536, the LOAD8/STORE_LOAD8/FILL declared window
const W: u64 = 8; // i64 access width

#[test]
fn default_mapped_matches_declared_boundary() {
    // With the global left at its instantiation default (= the emit-time size), the boundary is the
    // declared window — i.e. behavior identical to the old baked constant. `run_at_mapped` sets the
    // global explicitly to DECLARED here, which is exactly that default value.
    // An in-window byte above the #1094 NULL guard `[0, 16384)` is backed ⇒ admitted (address 0
    // itself now traps on the unconditional guard, so probe the first post-guard word instead).
    assert_eq!(run_at_mapped(LOAD8, &[16384], DECLARED), R::Val(0));
    assert_eq!(
        run_at_mapped(LOAD8, &[(DECLARED - W) as i64], DECLARED),
        R::Val(0)
    );
    // One past the last in-window byte faults.
    assert_eq!(
        run_at_mapped(LOAD8, &[(DECLARED - W + 1) as i64], DECLARED),
        R::MemFault
    );
    assert_eq!(
        run_at_mapped(LOAD8, &[DECLARED as i64], DECLARED),
        R::MemFault
    );
}

#[test]
fn grow_admits_previously_faulting_access() {
    // The #717 reproducer, distilled: an access at the byte just past the declared window.
    let addr = DECLARED as i64; // 65536 — the open_memstream allocation-header address shape
                                // At the declared (default) size it faults on the JIT — the pre-fix behavior.
    assert_eq!(run_at_mapped(LOAD8, &[addr], DECLARED), R::MemFault);
    // After a grow (the guest committed [declared, 2*declared) via vm_map, and the host raised the
    // `mapped` global), the identical access now succeeds — matching the interpreter.
    assert_eq!(run_at_mapped(LOAD8, &[addr], 2 * DECLARED), R::Val(0));
    // And a store/load round-trips through the grown region: real backing, not a masked alias.
    assert_eq!(
        run_at_mapped(STORE_LOAD8, &[addr, 0x0123_4567_89ab_cdef], 2 * DECLARED),
        R::Val(0x0123_4567_89ab_cdef)
    );
    // The new frontier still fault-closes just past the grown extent.
    assert_eq!(
        run_at_mapped(LOAD8, &[(2 * DECLARED) as i64], 2 * DECLARED),
        R::MemFault
    );
}

#[test]
fn shrink_faults_below_declared() {
    // A live `mapped` *smaller* than the declared size (a host that reports a shrunk committed extent)
    // fault-closes accesses in [shrunk, declared) — the check follows the global downward too, so this
    // stays fail-closed (never fail-open) even though the address is within the declared window.
    let shrunk = DECLARED / 2; // 32768
    assert_eq!(
        run_at_mapped(LOAD8, &[(shrunk - W) as i64], shrunk),
        R::Val(0)
    );
    assert_eq!(run_at_mapped(LOAD8, &[shrunk as i64], shrunk), R::MemFault);
    assert_eq!(
        run_at_mapped(LOAD8, &[(DECLARED - W) as i64], shrunk),
        R::MemFault
    );
}

#[test]
fn boundary_sweep_scalar() {
    // Masking-hinge fuzz: for several live extents (shrunk, declared, and grown multiples), sweep the
    // access address across the window edge and assert the emitted trap fires iff the access is not
    // wholly within [0, M) — the interpreter oracle for a window committed to M. Deterministic (no
    // RNG): a dense band around each boundary, which is where an off-by-one in the lowering shows.
    for &m in &[
        DECLARED / 2,
        DECLARED,
        2 * DECLARED,
        3 * DECLARED,
        7 * DECLARED,
    ] {
        for d in -4i64..=4 {
            let addr = (m as i64) + d;
            if addr < 0 {
                continue;
            }
            let want_ok = (addr as u64) + W <= m; // wholly in-window ⇒ admitted
            let got = run_at_mapped(LOAD8, &[addr], m);
            let ok = matches!(got, R::Val(_));
            assert_eq!(
                ok, want_ok,
                "scalar boundary: M={m} addr={addr} want_ok={want_ok} got={got:?}"
            );
        }
    }
}

#[test]
fn boundary_sweep_bulk_span() {
    // The same sweep for the bulk-span check (`emit_span_check`): a `mem.fill` of `len` bytes at `dst`
    // is admitted iff `[dst, dst+len)` is wholly within [0, M). Fill 16 bytes and slide the *end* of
    // the span across the live boundary.
    let len = 16u64;
    for &m in &[DECLARED, 2 * DECLARED, 5 * DECLARED] {
        for d in -4i64..=4 {
            // Place the span so its end sits at M + d.
            let end = (m as i64) + d;
            let dst = end - len as i64;
            if dst < 0 {
                continue;
            }
            let want_ok = (dst as u64) + len <= m;
            let got = run_at_mapped(FILL, &[dst, len as i64], m);
            let ok = matches!(got, R::Val(_)); // fill returns i64 0 on success
            assert_eq!(
                ok, want_ok,
                "bulk boundary: M={m} dst={dst} len={len} want_ok={want_ok} got={got:?}"
            );
        }
    }
}

/// A load at `addr + 16408`: an offset past the NULL guard, so an address that wraps `addr + offset`
/// around to a small value lands on backed window bytes rather than on the guard.
const LOAD8_OFF: &str = r#"
memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  vr = i64.load v0 offset=16400
  return vr
  }
}
"#;

#[test]
fn a_live_extent_below_the_access_width_admits_nothing() {
    // `M < width` leaves no `width`-byte access inside `[0, M)`. The check once computed
    // `M - width`, which wrapped to 2^64 - (width - M) and admitted every address — the whole of
    // the embedder's linear memory, since the clamp is to the reservation, not the window. The
    // probes sit past the NULL guard, where only the bounds check stands between them and a load.
    for m in [0, 1, W - 1] {
        for addr in [16384, (DECLARED - W) as i64, 1 << 19] {
            assert_eq!(
                run_at_mapped(LOAD8, &[addr], m),
                R::MemFault,
                "M={m} addr={addr}: nothing fits"
            );
        }
    }
    // At `M = width` the one word `[0, 8)` fits the bound, and the NULL guard faults it.
    assert_eq!(run_at_mapped(LOAD8, &[0], W), R::MemFault);
    assert_eq!(run_at_mapped(LOAD8, &[16384], W), R::MemFault);
}

#[test]
fn an_address_whose_offset_overflows_faults() {
    // `Window::checked` faults when `addr + offset` overflows. The emitted `eff = addr + offset`
    // once wrapped instead, here to 16392, a backed word past the NULL guard, and loaded it.
    let addr = -8i64; // 2^64 - 8
    assert_eq!(run_at_mapped(LOAD8_OFF, &[addr], DECLARED), R::MemFault);
    // The same offset from a real address is admitted up to the window's end, and no further.
    assert_eq!(run_at_mapped(LOAD8_OFF, &[0], DECLARED), R::Val(0));
    let last = (DECLARED - W - 16400) as i64;
    assert_eq!(run_at_mapped(LOAD8_OFF, &[last], DECLARED), R::Val(0));
    assert_eq!(run_at_mapped(LOAD8_OFF, &[last + 1], DECLARED), R::MemFault);
}

#[test]
fn the_check_is_window_checked_at_every_live_extent() {
    // The emitted check against `temen_mask::Window::checked` itself, the one definition: over
    // scalar widths, offsets (none, small, past the NULL guard, past 32 bits), and every edge of
    // the live extent — below the widest access, around the guard and the declared window — and of
    // the address, where `addr + offset` and `addr + offset + width` overflow. Admitted means the
    // bytes are inside `[0, M)` and past the NULL guard, which faults every access below it.
    let guard = 16384u64;
    let loads = [
        (1u64, "i32.load8_u", "i32"),
        (2, "i32.load16_u", "i32"),
        (4, "i32.load", "i32"),
        (8, "i64.load", "i64"),
    ];
    let extents: Vec<u64> = (0..=17)
        .chain([
            guard - 1,
            guard,
            guard + 1,
            DECLARED - 1,
            DECLARED,
            DECLARED + 1,
        ])
        .chain([u64::from(MAX_EXTENT)])
        .collect();
    for (width, op, ty) in loads {
        for offset in [0u64, 3, 16400, (1 << 32) + 7] {
            let src = format!(
                "memory 16\nfunc (i64) -> ({ty}) {{\nblock 0 (v0: i64) {{\n  \
                 vr = {op} v0 offset={offset}\n  return vr\n  }}\n}}\n"
            );
            let mut kernel = Kernel::new(&src);
            for &m in &extents {
                let fit = m.wrapping_sub(offset).wrapping_sub(width);
                let addrs = [
                    0,
                    1,
                    guard - 1,
                    guard,
                    fit.wrapping_sub(1),
                    fit,
                    fit.wrapping_add(1),
                    m,
                    0u64.wrapping_sub(offset).wrapping_sub(width),
                    0u64.wrapping_sub(offset),
                    u64::MAX - 7,
                    u64::MAX,
                    1 << 63,
                ];
                for addr in addrs {
                    let window = temen_mask::Window::with_mapped(40, m);
                    let admitted = window.checked(addr, offset, width as u32).is_some()
                        && addr + offset >= guard;
                    let want = if admitted { R::Val(0) } else { R::MemFault };
                    assert_eq!(
                        kernel.run(&[addr as i64], m),
                        want,
                        "{op} offset={offset} at {addr:#x}, M={m}"
                    );
                }
            }
        }
    }
}
