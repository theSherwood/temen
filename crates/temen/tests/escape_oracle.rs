//! Escape-oracle plumbing tests (`DESIGN.md` §4/§18): the generative differential
//! (`jit_fuzz`/`diff`) also byte-compares the final guest window across the interpreter
//! and JIT for float-free modules — realizing the §18 move *"verified ⇒ cannot escape"* at
//! the module level (the `fuzz/mask` unit proves the confinement arithmetic in isolation).
//! The broad coverage is the seed loop; these hand-written cases pin the mechanism down under
//! **trap-confinement**: that an **out-of-window access faults at the offending access** —
//! identically on both backends, with no aliasing back into the window — and that the capture
//! path reflects guest stores.

use temen_interp::{run_capture, run_capture_reserved, Value};
use temen_jit::{compile, compile_and_run_capture, compile_and_run_capture_reserved, JitOutcome};

#[path = "support/detached_probe.rs"]
mod detached_probe;
#[path = "../../temen-interp/tests/support/drivers.rs"]
mod drivers;

use detached_probe::Report;

/// True on a 4 KiB-page host. A couple of `reserved_*` cases below hardcode the mapped/tail
/// boundary at address 4096 (`memory 12`); on a 16 KiB-page host (macOS ARM) the JIT rounds the
/// 4 KiB `mapped` up to one 16 KiB host page, so that address is still backed and the "tail
/// access faults" premise no longer holds. Those cases skip there; the guard itself is covered on
/// 16 KiB by the `temen-jit` PAL conformance test (64 KiB / 1 MiB windows, host-page-aligned).
#[cfg(unix)]
fn host_4k() -> bool {
    // SAFETY: sysconf is always safe; _SC_PAGESIZE is positive.
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) == 4096 }
}

/// Parse + verify a module, then run it on both backends with `init` seeding the window; return
/// both final-window snapshots (asserting both ran to completion and agree on the result). Used by
/// the success cases (an in-window access, or a no-op) where both backends complete.
fn both_windows(src: &str, init: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let m = temen::text::parse_module(src).expect("parse");
    temen::verify::verify_module(&m).expect("verify");
    let mut fuel = 1_000_000u64;
    let (ir, imem) = run_capture_reserved(&m, 0, &[Value::I32(0)], &mut fuel, init, 0);
    let (jo, jmem) = compile_and_run_capture_reserved(&m, 0, &[0i64], init, 0).expect("jit");
    assert!(ir.is_ok(), "interp trapped: {ir:?}");
    assert!(
        matches!(jo, JitOutcome::Returned(_)),
        "jit did not return: {jo:?}"
    );
    (imem, jmem)
}

/// Like [`both_windows`], but returns each backend's **trap disposition** (interpreter-trapped,
/// JIT outcome) alongside the two snapshots — the harness for the trap-confinement cases, which
/// assert both backends fault on the same out-of-window access. (Memory is not compared on a trap:
/// a faulted store never writes, and capture-on-trap is not a guaranteed byte-for-byte snapshot.)
fn both_windows_disposition(src: &str, init: &[u8]) -> (bool, JitOutcome, Vec<u8>, Vec<u8>) {
    let m = temen::text::parse_module(src).expect("parse");
    temen::verify::verify_module(&m).expect("verify");
    let mut fuel = 1_000_000u64;
    let (ir, imem) = run_capture_reserved(&m, 0, &[Value::I32(0)], &mut fuel, init, 0);
    let (jo, jmem) = compile_and_run_capture_reserved(&m, 0, &[0i64], init, 0).expect("jit");
    (ir.is_err(), jo, imem, jmem)
}

#[test]
fn out_of_window_store_faults_identically() {
    // Window = 2^8 = 256 bytes. A store to 261 is out of the window; under trap-confinement it
    // raises `MemoryFault` *at the offending access* on both backends (no aliasing to 261 & 255 = 5),
    // leaving the window untouched.
    let src = "\
memory 8
func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = i64.const 261
  v2 = i32.const 171
  i32.store8 v1 v2
  v3 = i32.const 0
  return v3
  }
}
";
    let (it, jo, imem, _jmem) = both_windows_disposition(src, &[0u8; 256]);
    assert!(it, "interp did not fault on the out-of-window store");
    assert!(
        matches!(jo, JitOutcome::Trapped(temen_jit::TrapKind::MemoryFault)),
        "jit did not detect-and-kill the out-of-window store: {jo:?}"
    );
    assert_eq!(
        imem.iter().filter(|&&b| b != 0).count(),
        0,
        "a faulted store must not write the window"
    );
}

#[test]
fn far_address_and_offset_fault_identically() {
    // A huge base plus a folded immediate offset is out of the window; under trap-confinement it
    // faults on both backends (the old model masked it to an in-window byte). base = i64::MAX,
    // offset 8.
    let src = "\
memory 8
func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = i64.const 9223372036854775807
  v2 = i32.const 200
  i32.store8 v1 v2 offset=8
  v3 = i32.const 0
  return v3
  }
}
";
    let (it, jo, imem, _jmem) = both_windows_disposition(src, &[0u8; 256]);
    assert!(it, "interp did not fault on the far out-of-window store");
    assert!(
        matches!(jo, JitOutcome::Trapped(temen_jit::TrapKind::MemoryFault)),
        "jit did not detect-and-kill the far store: {jo:?}"
    );
    assert_eq!(
        imem.iter().filter(|&&b| b != 0).count(),
        0,
        "a faulted store must not write the window"
    );
}

#[test]
fn seed_survives_when_untouched() {
    // A no-op body must leave the seeded window exactly as provided, on both backends — so
    // a real divergence later can't hide behind a zeroed window.
    let init: Vec<u8> = (0..256)
        .map(|i| (i as u8).wrapping_mul(31) ^ 0xa5)
        .collect();
    let src = "\
memory 8
func (i32) -> (i32) {
block 0 (v0: i32) {
  return v0
  }
}
";
    let (imem, jmem) = both_windows(src, &init);
    assert_eq!(imem, init, "interp did not preserve the seeded window");
    assert_eq!(jmem, init, "jit did not preserve the seeded window");
}

/// #2176: a JIT run copies its window back only when asked. Asked for nothing, it returns nothing:
/// copying a large window costs a page fault per page, for bytes nobody reads. Asked for the backed
/// prefix (`Some(0)`, what the capture entries above ask for), it returns the whole window, the
/// guest's store included.
#[test]
fn a_jit_run_copies_its_window_back_only_when_asked() {
    let src = "\
memory 8
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 171
  i64.store v0 v1
  return v1
  }
}
";
    let m = temen::text::parse_module(src).expect("parse");
    temen::verify::verify_module(&m).expect("verify");
    let mut cm = compile(&m, 0).expect("compile");
    let init = [0u8; 256];
    let returned = |out: &JitOutcome| matches!(out, JitOutcome::Returned(s) if s == &[171]);
    let (out, mem) = cm.run(&[8], Some(&init), None).expect("run");
    assert!(returned(&out), "{out:?}");
    assert!(
        mem.is_empty(),
        "asked for nothing, copied {} bytes",
        mem.len()
    );
    let (out, mem) = cm.run(&[8], Some(&init), Some(0)).expect("run");
    assert!(returned(&out), "{out:?}");
    assert_eq!(mem.len(), init.len(), "the backed prefix");
    assert_eq!(mem[8], 171, "the guest's store");
}

/// The JIT elides the bounds check when the address is *provably* in-window (the §1a
/// "check-when-not" path). This pins that the elided path stays confined: `addr = (n & 7)*8`
/// is provably ≤ 56 in a 256-byte window, so the check is dropped — and for adversarial `n`
/// (incl. negative / i64::MAX, whose low bits still confine via `& 7`) the interpreter and the
/// JIT must leave an identical window and land at the same slot (never faulting, never escaping).
#[test]
fn elided_bounded_address_confines() {
    let src = "\
memory 8
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 7
  v2 = i64.and v0 v1
  v3 = i64.const 8
  v4 = i64.mul v2 v3
  v5 = i64.const 171
  i64.store v4 v5
  v6 = i64.load v4
  return v6
  }
}
";
    let m = temen::text::parse_module(src).expect("parse");
    temen::verify::verify_module(&m).expect("verify");
    for n in [0i64, 5, 0x12345, -1, i64::MAX, 9_999_999] {
        let init = [0u8; 256];
        let mut fuel = 1_000_000u64;
        let (ir, imem) = run_capture(&m, 0, &[Value::I64(n)], &mut fuel, &init);
        let (jo, jmem) = compile_and_run_capture(&m, 0, &[n], &init).expect("jit");
        assert_eq!(ir.ok(), Some(vec![Value::I64(171)]), "interp result n={n}");
        assert!(
            matches!(jo, JitOutcome::Returned(ref s) if s == &[171]),
            "jit {jo:?} n={n}"
        );
        assert_eq!(imem, jmem, "elided-address windows diverge at n={n}");
        let slot = ((n as u64 & 7) * 8) as usize;
        assert_eq!(imem[slot], 171, "store landed at wrong slot for n={n}");
    }
}

/// Run both backends with a host **reservation** (`reserved_log2`): only the declared `memory`
/// bytes are backed inside the larger reserved range. Returns whether each backend trapped plus the
/// two window snapshots, so a test can assert they agree on the trap disposition *and* the final
/// memory (the escape-oracle under the decoupled `reserved`/`mapped` model, §4). `n` is the entry
/// arg (an `i64`); `init` must be `1 << size_log2` bytes (the JIT snapshots the whole backed window,
/// so the seed length sets the compared extent).
#[cfg(unix)] // only the `cfg(unix)` reserved-tail tests use it (windows runs them once temen-run's
             // Memory-cap path is ported — Phase 3.5); avoids a dead-code warning on windows.
fn both_reserved(
    src: &str,
    init: &[u8],
    reserved_log2: u8,
    n: i64,
) -> (bool, JitOutcome, Vec<u8>, Vec<u8>) {
    let m = temen::text::parse_module(src).expect("parse");
    temen::verify::verify_module(&m).expect("verify");
    let mut fuel = 1_000_000u64;
    let (ir, imem) = run_capture_reserved(&m, 0, &[Value::I64(n)], &mut fuel, init, reserved_log2);
    let (jo, jmem) =
        compile_and_run_capture_reserved(&m, 0, &[n], init, reserved_log2).expect("jit");
    (ir.is_err(), jo, imem, jmem)
}

/// Trap-confinement (§4): an access past the backed `mapped` prefix **faults** on both backends —
/// whether it lands in a decoupled reserved-but-unmapped tail (`reserved > mapped`) or one byte past
/// a fully-mapped window (`reserved == mapped`). Trap-confinement bounds the guest to `[0, mapped)`
/// in *both* configs, so the reservation is now purely internal defense-in-depth (it no longer
/// changes the guest-visible bound). This pins that the same out-of-`mapped` store faults either way.
#[cfg(unix)]
#[test]
fn reserved_tail_access_faults_identically() {
    if !host_4k() {
        return; // hardcodes the 4 KiB mapped/tail boundary (address 4096); see `host_4k`.
    }
    // memory 12 = 4 KiB backed (exactly one page). Store one byte at address 4096 — the first
    // byte *past* the mapped window.
    let src = "\
memory 12
func (i64) -> (i32) {
block 0 (v0: i64) {
  v1 = i64.const 4096
  v2 = i32.const 7
  i32.store8 v1 v2
  v3 = i32.const 0
  return v3
  }
}
";
    // Fully mapped (reserved == mapped == 4 KiB): 4096 is one past the top ⇒ fault on both backends
    // (trap-confinement — the old model wrapped 4096 & 4095 = 0 in).
    let (it0, jo0, _im0, _jm0) = both_reserved(src, &[0u8; 4096], 0, 0);
    assert!(
        it0,
        "interp should fault one byte past a fully-mapped window"
    );
    assert!(
        matches!(jo0, JitOutcome::Trapped(temen_jit::TrapKind::MemoryFault)),
        "jit did not detect-and-kill the past-the-top access (fully mapped): {jo0:?}"
    );

    // Reserved (2^24) > mapped (2^12): 4096 is in the unmapped tail ⇒ fault on both backends.
    let (it1, jo1, _im1, _jm1) = both_reserved(src, &[0u8; 4096], 24, 0);
    assert!(
        it1,
        "interp did not fault on the out-of-mapped (tail) access"
    );
    assert!(
        matches!(jo1, JitOutcome::Trapped(temen_jit::TrapKind::MemoryFault)),
        "jit did not detect-and-kill the tail access: {jo1:?}"
    );
}

/// Under `reserved > mapped`, an access that stays within the backed `mapped` prefix still
/// succeeds and leaves byte-identical windows on both backends — the reservation only changes
/// what happens *outside* `mapped`, not in-window behaviour.
#[cfg(unix)]
#[test]
fn reserved_in_mapped_access_matches() {
    // Address = (n & 511) * 8, provably ≤ 4088 < mapped (4 KiB) — always in the backed prefix.
    let src = "\
memory 12
func (i64) -> (i32) {
block 0 (v0: i64) {
  v1 = i64.const 511
  v2 = i64.and v0 v1
  v3 = i64.const 8
  v4 = i64.mul v2 v3
  v5 = i32.const 99
  i32.store8 v4 v5
  v6 = i32.const 0
  return v6
  }
}
";
    for n in [0i64, 1, 7, 511, 512, i64::MAX, -1] {
        let (it, jo, imem, jmem) = both_reserved(src, &[0u8; 4096], 24, n);
        assert!(!it, "interp faulted on an in-mapped access, n={n}");
        assert!(matches!(jo, JitOutcome::Returned(_)), "jit n={n}: {jo:?}");
        assert_eq!(imem, jmem, "in-mapped windows diverge at n={n}");
        let slot = ((n as u64 & 511) * 8) as usize;
        assert_eq!(imem[slot], 99, "store landed at wrong slot for n={n}");
    }
}

/// #1867 decision 3 — a child whose access is `access`, run as a **detached child** in a window of its
/// own (`memory 16`, base 0): the escape-oracle's placement for a spawned guest, where a carve child
/// once sat inside its parent's window.
fn edge_child(access: &str) -> temen_ir::Module {
    let src = format!(
        "memory 16
func (i64) -> (i64) {{
block 0 (vs: i64) {{
{access}
  }}
}}
"
    );
    let m = temen::text::parse_module(&src).expect("parse");
    temen::verify::verify_module(&m).expect("verify");
    m
}

/// The probe root's report for `child` on every engine: the tree-walk oracle, each bytecode driver
/// ([`drivers::ALL`]) and the Cranelift JIT.
fn every_engine(child: &temen_ir::Module) -> Vec<(String, Report)> {
    let root = detached_probe::root(0);
    let powerbox = || {
        let (host, args) = detached_probe::powerbox(child);
        (host, args.map(Value::I32).to_vec())
    };
    let mut reports: Vec<(String, Report)> = drivers::ALL
        .into_iter()
        .map(|d| {
            let ran = drivers::run_on(d, &root, &powerbox)
                .unwrap_or_else(|| panic!("{d:?} declined the probe root"));
            let r = ran
                .result
                .unwrap_or_else(|t| panic!("{d:?}: the probe root trapped: {t:?}"));
            (format!("{d:?}"), Report::of_values(&r))
        })
        .collect();
    let (jo, _) = detached_probe::run_jit(&root, child, &[]).expect("jit");
    match jo {
        JitOutcome::Returned(v) => reports.push(("Jit".into(), Report::of(&v))),
        other => panic!("Jit: the probe root did not return: {other:?}"),
    }
    reports
}

/// A detached child stores at its window's top byte and loads it back: both land, identically on every
/// engine, and the root's window is untouched across the child's life. Pairs with
/// [`a_detached_child_past_its_window_faults_on_every_engine`].
#[test]
fn a_detached_childs_window_edge_holds_on_every_engine() {
    let child = edge_child(
        "  va = i64.const 65535
  vv = i32.const 99
  i32.store8 va vv
  vl = i32.load8_u va
  vr = i64.extend_i32_u vl
  return vr",
    );
    for (engine, report) in every_engine(&child) {
        assert_eq!(
            report,
            Report {
                child: Ok(99),
                canary: 0
            },
            "{engine}: the edge store and load land in the child's own window"
        );
    }
}

/// The confinement half: a store or a load one byte past the top of a detached child's window
/// **faults** at the access on every engine — it is not aliased back into the window — and the root's
/// window is untouched.
#[test]
fn a_detached_child_past_its_window_faults_on_every_engine() {
    let fault = temen_interp::Trap::MemoryFault.code();
    for access in [
        "  va = i64.const 65536
  vv = i32.const 200
  i32.store8 va vv
  vz = i64.const 0
  return vz",
        "  va = i64.const 65536
  vl = i32.load8_u va
  vr = i64.extend_i32_u vl
  return vr",
    ] {
        for (engine, report) in every_engine(&edge_child(access)) {
            assert_eq!(
                report,
                Report {
                    child: Err(fault),
                    canary: 0
                },
                "{engine}: an access past the child's window faults, the root untouched:\n{access}"
            );
        }
    }
}

/// Detect-and-kill (§4/§5): a store that overruns the top of the window is caught as a clean
/// `MemoryFault` — the host survives, no crash. Under trap-confinement the bounds check fires
/// *before* the access (an 8-byte store at 65532 needs `[65532, 65540) ⊆ [0, 65536)`, which fails);
/// the hardware guard page behind `mapped` remains as defense-in-depth for any elided path.
#[cfg(unix)]
#[test]
fn guard_page_fault_is_detect_and_kill() {
    // memory 16 = 64 KiB. An 8-byte store at 65532 writes [65532,65540), overrunning the 64 KiB top.
    let src = "\
memory 16
func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = i64.const 65532
  v2 = i64.const 0
  i64.store v1 v2
  v3 = i32.const 0
  return v3
  }
}
";
    let m = temen::text::parse_module(src).expect("parse");
    temen::verify::verify_module(&m).expect("verify");
    let out = temen_jit::compile_and_run(&m, 0, &[0]).expect("jit");
    assert!(
        matches!(out, JitOutcome::Trapped(temen_jit::TrapKind::MemoryFault)),
        "expected a caught MemoryFault, got {out:?}"
    );
}

/// #2147 — loads and stores through one base share a bounds check (`temen-jit`'s `CheckedBases`):
/// the first access through `v0` tests its own address, and later ones at a constant non-negative
/// distance from it, within the guard page past the window's end, reuse that base. Sweep the base `n`
/// so each access in turn is the first to cross the end, and below zero so the adds wrap: the JIT must
/// fault where the interpreter does, after the same stores, and return the same result.
#[cfg(unix)]
#[test]
fn accesses_sharing_a_checked_base_fault_where_the_interpreter_does() {
    // Access order, by address: `n+16` (8-byte store: the check point), `n+28` (8-byte load,
    // `n+24` plus offset 4: shares it), `n+40` (1-byte store, `n` plus offset 40: before the check
    // point's root offset, so it checks again and becomes the check point), `n` (8-byte load:
    // shares that one).
    let shared = "\
memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 16
  v2 = i64.add v0 v1
  v3 = i64.const 255
  i64.store v2 v3
  v4 = i64.const 24
  v5 = i64.add v0 v4
  v6 = i64.load v5 offset=4
  v7 = i32.const 9
  i32.store8 v0 v7 offset=40
  v8 = i64.load v0
  v9 = i64.add v6 v8
  return v9
  }
}
";
    // Neither later access may share the check point at `n+20000`: `n+28192` lies more than a page
    // past it (past the guard page, for `n` near the top), and `n+8` lies before it (for `n` just
    // below zero, `n+20000` wraps into the window while `n+8` does not).
    let unshared = "\
memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 20000
  v2 = i64.add v0 v1
  v3 = i64.load v2
  v4 = i64.const 28192
  v5 = i64.add v0 v4
  v6 = i64.load v5
  v7 = i64.load v0 offset=8
  v8 = i64.add v3 v6
  v9 = i64.add v8 v7
  return v9
  }
}
";
    let top: i64 = 1 << 16;
    let mut ns: Vec<i64> = vec![0, 8, 4096, i64::MAX, i64::MIN, top, top + 4096];
    ns.extend((0..=48).map(|k| top - k));
    ns.extend((0..=48).map(|k| -k)); // the adds wrap around zero
    ns.extend([
        top - 20008,
        top - 28200,
        top - 28192,
        -20000 + 16384,
        -16,
        -3616,
    ]);
    let init: Vec<u8> = (0..top).map(|i| (i % 251) as u8).collect();
    for src in [shared, unshared] {
        let m = temen::text::parse_module(src).expect("parse");
        temen::verify::verify_module(&m).expect("verify");
        for &n in &ns {
            let mut fuel = 1_000_000u64;
            let (ir, imem) = run_capture_reserved(&m, 0, &[Value::I64(n)], &mut fuel, &init, 0);
            let (jo, jmem) = compile_and_run_capture_reserved(&m, 0, &[n], &init, 0).expect("jit");
            match (&ir, &jo) {
                (Ok(v), JitOutcome::Returned(r)) => {
                    let r: Vec<Value> = r.iter().map(|&x| Value::I64(x)).collect();
                    assert_eq!(v, &r, "n={n}: the results differ\n{src}");
                }
                (Err(_), JitOutcome::Trapped(temen_jit::TrapKind::MemoryFault)) => {}
                _ => panic!("n={n}: the interpreter gave {ir:?}, the JIT {jo:?}\n{src}"),
            }
            assert!(imem == jmem, "n={n}: the windows differ\n{src}");
        }
    }
}
