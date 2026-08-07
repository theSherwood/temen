//! Thread-var (`tvar`) lowering — the single-threaded TLS model (NIM.md §3d).
//!
//! Leng marks a thread-local `tvar` (nimony's `__thread`; the allocator and exception state in the
//! real `system` module are thread-vars). svm-leng lowers a `tvar` **identically to a `gvar`**: one
//! plain, zero-initialized global at a fixed window offset. This is sound because every guest we
//! target — each nimony compiler phase, each svm domain — runs single-threaded, so a thread-local has
//! exactly one instance and a plain global *is* that instance. It mirrors the C on-ramp, which strips
//! `__thread` before clang (`demos/nimony/build_nimony.sh`). These tests pin that model: a `tvar`
//! must behave as a persistent global — writes survive across calls, non-zero initializers seed it,
//! and it links across modules the same as a `gvar` — on both engines. The real multi-threaded
//! `__thread` lowering over `vcpu.tls` (NIM.md §3d Tier 2) is implemented behind `tls_mode` and
//! covered by the Tier-2 tests at the bottom of this file.

use svm_interp::Value;
use svm_ir::LinkUnit;
use svm_leng::LengModule;

/// Run func `idx` on both engines; assert §9 parity; return the i64 result.
fn run(m: &svm_ir::Module, idx: u32, args: &[i64]) -> i64 {
    svm_verify::verify_module(m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    let ivals: Vec<Value> = args.iter().map(|&n| Value::I64(n)).collect();
    let mut fuel = u64::MAX;
    let interp = svm_interp::run(m, idx, &ivals, &mut fuel).expect("interp");
    let n = match interp.as_slice() {
        [Value::I64(n)] => *n,
        o => panic!("expected i64, got {o:?}"),
    };
    let jit = match svm_jit::compile_and_run(m, idx, args).expect("jit") {
        svm_jit::JitOutcome::Returned(v) => v,
        o => panic!("jit: {o:?}"),
    };
    assert_eq!(jit.as_slice(), &[n], "§9 interp/JIT parity");
    n
}

/// A `tvar` is real, persistent backing store: `bump` writes it, later calls read the accumulated
/// value back. Same shape as `whole_module::globals_const_and_intramodule_calls`, but the counter is
/// a **thread-var** — so this proves `tvar` lowers to a plain global that persists across calls.
#[test]
fn thread_var_persists_across_calls_like_a_global() {
    let leng = "\
(stmts
 (tvar :counter.0. . (i +64) .)
 (proc :bump.0. . (void) .
  (stmts .
   (asgn counter.0. (add (i +64) counter.0. 1))))
 (proc :bumpN.0. (params (param :n.0 . (i +64))) (void) .
  (stmts .
   (var :i.0 . (i +64) 0)
   (while (lt i.0 n.0)
    (stmts .
     (call bump.0.)
     (asgn i.0 (add (i +64) i.0 1))))))
 (proc :main.0. (params (param :n.0 . (i +64))) (i +64) .
  (stmts .
   (call bumpN.0. n.0)
   (ret counter.0.))))";
    let text = svm_leng::translate_to_text(leng).unwrap();
    assert!(
        text.starts_with("memory "),
        "a tvar needs a window:\n{text}"
    );
    let m = svm_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    // main is func 2; the thread-var counter starts 0 and is bumped n times → returns n.
    assert_eq!(run(&m, 2, &[0]), 0);
    assert_eq!(run(&m, 2, &[7]), 7);
    assert_eq!(run(&m, 2, &[100]), 100);
}

/// A `tvar` with a non-zero initializer seeds the window like a `gvar` initializer does.
#[test]
fn thread_var_nonzero_initializer_seeds_the_window() {
    let leng = "\
(stmts
 (tvar :g.0. . (i +64) 5)
 (tvar :h.0. . (i +32) 7)
 (proc :sum.0. . (i +64) . (stmts . (ret (add (i +64) g.0. (conv (i +64) h.0.))))))";
    let text = svm_leng::translate_to_text(leng).unwrap();
    assert!(
        text.contains("data "),
        "a non-zero tvar init emits a data segment:\n{text}"
    );
    let m = svm_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    assert_eq!(
        run(&m, 0, &[]),
        12,
        "5 + 7 read from the seeded thread-vars"
    );
}

/// A `tvar` links across modules exactly like a `gvar`: module `w`'s proc reads and writes a
/// thread-var `g` **defined in module `s`** (a `data.sym` the linker binds to `s`'s exported global).
/// This is the shape of the real allocator's thread-vars in `system`, referenced from user code —
/// write-then-read-back through the cross-module symbol must reach `s`'s storage, so `rw(v) = v`.
/// (Mirrors `link::cross_module_global_read_write`, with `g` a `tvar`.)
#[test]
fn thread_var_links_cross_module_like_a_global() {
    let mod_s = "\
(stmts
 (tvar :g.0. . (i +64) 0))";
    let mod_w = "\
(stmts
 (proc :rw.0. (params (param :v.0 . (i +64))) (i +64) .
  (stmts .
   (asgn g.0.mods v.0)
   (ret g.0.mods))))";
    let linked = svm_leng::link_units(&[
        LengModule {
            stem: "modw",
            src: mod_w,
            names: &["rw.0."],
        },
        LengModule {
            stem: "mods",
            src: mod_s,
            names: &[], // data-only unit: it just defines (and exports) the thread-var `g`
        },
    ])
    .unwrap_or_else(|e| panic!("link: {e}"));
    assert_eq!(
        run(&linked, 0, &[42]),
        42,
        "write then read the external tvar"
    );
    assert_eq!(run(&linked, 0, &[-5]), -5);
}

// ---------------------------------------------------------------------------
// Tier 2 (NIM.md §3d): the real multi-threaded `__thread` lowering over `vcpu.tls`.
//
// In `translate_tls`, a `tvar` no longer collapses to one plain global — it gets an offset in the
// per-vCPU TLS block, and each access lowers to `vcpu.tls.get() + off` (the fs/gs-base recipe). svm
// supplies the per-vCPU base register (§12, seeded to the vCPU id, guest-overwritable); the runtime
// establishes each thread's block base with `vcpu.tls.set` at thread entry. These tests exercise
// svm-leng's half — the access lowering — with a hand-written driver standing in for that runtime.

/// A tvar module for Tier-2 lowering: `bump(n)` adds to the thread-var `counter`, `get()` reads it.
/// `bump` is func 0, `get` is func 1.
const TVAR_ACCESSORS: &str = "\
(stmts
 (tvar :counter.0. . (i +64) .)
 (proc :bump.0. (params (param :n.0 . (i +64))) (void) .
  (stmts .
   (asgn counter.0. (add (i +64) counter.0. n.0))))
 (proc :get.0. . (i +64) . (stmts . (ret counter.0.))))";

#[test]
fn tls_mode_lowers_a_tvar_through_vcpu_tls() {
    // Tier 2 reads/writes the tvar via the per-vCPU register; Tier 1 (the default) keeps it a plain
    // global. The presence/absence of `vcpu.tls` in the emitted text is the switch.
    let tls = svm_leng::translate_tls_to_text(TVAR_ACCESSORS).unwrap();
    assert!(
        tls.contains("vcpu.tls.get"),
        "TLS mode reaches the tvar through the per-vCPU base:\n{tls}"
    );
    let plain = svm_leng::translate_to_text(TVAR_ACCESSORS).unwrap();
    assert!(
        !plain.contains("vcpu.tls"),
        "Tier 1 keeps the tvar a plain global (no vcpu.tls):\n{plain}"
    );
}

#[test]
fn tvar_is_isolated_per_tls_base() {
    // The core Tier-2 property: the tvar follows `vcpu.tls`, so two different bases are two
    // independent instances — exactly the per-thread isolation a spawned thread gets when the
    // runtime hands it its own block. A single vCPU proves the mechanism deterministically by
    // switching the base (what a real per-thread `vcpu.tls.set` does): set base B0, bump +3; set
    // base B1, bump +5; read B0 back (3) and B1 back (5) → 3*100 + 5 = 305. If the tvar were a
    // shared global, both reads would see 8.
    let user =
        svm_leng::translate_tls(TVAR_ACCESSORS).unwrap_or_else(|e| panic!("translate_tls: {e}"));

    // The driver stands in for the thread runtime: it owns the `vcpu.tls.set`s and calls the
    // translated accessors (imported by name, bound to the user unit's exports by the linker).
    // B0 = 2048, B1 = 4096 — two zeroed 8-byte blocks in the low scratch page.
    let driver = svm_text::parse_module(
        "\
memory 16
import 0 \"bump\" (i64) -> ()
import 1 \"get\" () -> (i64)
func 0 () -> (i64) {
block 0 () {
  b0 = i64.const 2048
  vcpu.tls.set b0
  three = i64.const 3
  call.import 0 (three)
  b1 = i64.const 4096
  vcpu.tls.set b1
  five = i64.const 5
  call.import 0 (five)
  vcpu.tls.set b0
  g0 = call.import 1 ()
  vcpu.tls.set b1
  g1 = call.import 1 ()
  hund = i64.const 100
  m = i64.mul g0 hund
  r = i64.add m g1
  return r
  }
}
export 0 func \"_start\" 0
",
    )
    .expect("driver parses");

    let linked = svm_ir::link(&[
        LinkUnit {
            module: driver,
            exports: vec![],
            ..Default::default()
        },
        LinkUnit {
            module: user,
            exports: vec![("bump".into(), 0), ("get".into(), 1)],
            ..Default::default()
        },
    ])
    .unwrap_or_else(|e| panic!("link: {e:?}"));
    // _start is the driver's func 0 → func 0 of the linked module.
    assert_eq!(
        run(&linked, 0, &[]),
        305,
        "B0's counter is 3 and B1's is 5 — the tvar is per-tls-base, not shared"
    );
}

#[test]
fn non_zero_tvar_initializer_fails_closed_in_tls_mode() {
    // Tier 2 zeroes each per-thread block; a non-zero initializer would need per-thread seeding by
    // the runtime (a bounded follow-up), so it's a clean error rather than a silently-wrong global.
    let leng = "\
(stmts
 (tvar :g.0. . (i +64) 5)
 (proc :get.0. . (i +64) . (stmts . (ret g.0.))))";
    let err =
        svm_leng::translate_tls_to_text(leng).expect_err("non-zero tvar init must fail-closed");
    assert!(
        format!("{err:?}").contains("TLS mode"),
        "clean, specific error: {err:?}"
    );
}

#[test]
fn tvar_block_offset_agrees_across_modules() {
    // The cross-module Tier-2 property (NIM.md §3d): a `tvar` defined in one unit and referenced from
    // another must resolve to the *same* per-vCPU block offset — the TLS analog of a relocated
    // cross-module global. The linker pools every unit's tvars into one shared layout and hands it to
    // all, so both sides bake the same `vcpu.tls.get()+off`.
    //
    // `mods` defines thread-var `g` and a `peek()` that reads it by its *local* name. `modw`'s `rw(v)`
    // writes and reads `g` by its *cross-module* name `g.0.mods`. A driver sets a TLS base, then
    // `rw(42)` writes 42 through modw's offset and `peek()` reads through mods's offset: if the two
    // offsets agree, `peek` sees 42 (a disagreement would read a different, zeroed slot → 0).
    // Result = rw*100 + peek = 42*100 + 42 = 4242.
    //
    // The filler thread-var `aa` (sorted before `g`) pushes `g`'s shared offset to 8, so the test
    // proves the *accumulated* layout propagates cross-module — not a coincidental offset-0 match.
    let mod_s = "\
(stmts
 (tvar :aa.0. . (i +64) .)
 (tvar :g.0. . (i +64) .)
 (proc :peek.0. . (i +64) . (stmts . (ret g.0.))))";
    let mod_w = "\
(stmts
 (proc :rw.0. (params (param :v.0 . (i +64))) (i +64) .
  (stmts .
   (asgn g.0.mods v.0)
   (ret g.0.mods))))";

    // The driver stands in for the thread runtime: `vcpu.tls.set` a block base, then call the
    // translated procs (imported under their stem-suffixed export names).
    let driver = svm_text::parse_module(
        "\
memory 16
import 0 \"rw.0.modw\" (i64) -> (i64)
import 1 \"peek.0.mods\" () -> (i64)
func 0 () -> (i64) {
block 0 () {
  b = i64.const 2048
  vcpu.tls.set b
  fortytwo = i64.const 42
  r1 = call.import 0 (fortytwo)
  r2 = call.import 1 ()
  hund = i64.const 100
  m = i64.mul r1 hund
  res = i64.add m r2
  return res
  }
}
export 0 func \"_start\" 0
",
    )
    .expect("driver parses");

    let linked = svm_leng::link_units_tls_with_runtime(
        &[
            LengModule {
                stem: "modw",
                src: mod_w,
                names: &["rw.0."],
            },
            LengModule {
                stem: "mods",
                src: mod_s,
                names: &["peek.0."],
            },
        ],
        vec![LinkUnit {
            module: driver,
            exports: vec![],
            ..Default::default()
        }],
    )
    .unwrap_or_else(|e| panic!("link_units_tls: {e:?}"));
    // Funcs: modw.rw=0, mods.peek=1, driver._start=2.
    assert_eq!(
        run(&linked, 2, &[]),
        4242,
        "cross-module write (modw) and local read (mods) hit the same tvar slot"
    );
}
