//! **#964/#1094 — the NULL guard reaches entry-less kernels too.** A reactor kernel (a `tick`-only
//! module: no `main`, so temen-llvm synthesizes no powerbox `_start`) used to be a legacy carve-out:
//! the `__null_guard` marker rode the synthesized `_start`, and an entry-less module's globals sat at
//! `DATA_BASE` (16) — *inside* the would-be guard region — so it could never be seeded `Unmapped`.
//!
//! This pins the fix: under the guard an entry-less module (a) bases its globals one guard up
//! (`scratch + DATA_BASE`), leaving `[0, guard)` empty, and (b) exports the `__null_guard` marker
//! (aliasing its first function), so a marker-aware host seeds the reserved region `Unmapped` and a
//! NULL dereference traps — exactly like the `_start` path. Legacy (flag off) stays byte-identical, and
//! the guarded kernel computes the same result as the legacy one. No clang: the fixture is inline
//! textual LLVM IR, so this runs in every job.

use temen_interp::Value;

/// An **entry-less** kernel: a global and a `tick(x) = g + x` that reads it. No `main` / imports /
/// `malloc`, so temen-llvm synthesizes no `_start` — the exact reactor-kernel shape.
const KERNEL: &str = r#"
@g = global i64 100

define i64 @tick(i64 %x) {
entry:
  %v = load i64, ptr @g
  %r = add i64 %v, %x
  ret i64 %r
}
"#;

/// Translate the kernel, returning the module and the data-stack base (`$sp`) temen-llvm prepends to
/// every on-ramp function — the leading argument `tick` expects.
fn translate(null_guard: bool) -> (temen_ir::Module, i64) {
    let opts = temen_llvm::TranslateOptions {
        null_guard,
        ..Default::default()
    };
    let t = temen_llvm::translate_ll_str_with_options(KERNEL, opts).expect("translate kernel");
    temen_verify::verify_module(&t.module).expect("verify kernel");
    (t.module, t.entry_sp as i64)
}

/// Guarded entry-less kernel: marked, globals shifted above the guard so `[0, guard)` is empty; legacy
/// stays unmarked with globals at `DATA_BASE`. Both compute `tick(x) = 100 + x` identically.
#[test]
fn entryless_kernel_is_guarded_and_keeps_the_null_region_empty() {
    let guard = temen_ir::POWERBOX_NULL_GUARD;
    let (legacy, legacy_sp) = translate(false);
    let (guarded, guarded_sp) = translate(true);

    // The marker: absent on legacy, present on the guarded kernel even though it has no `_start`.
    assert_eq!(
        temen_ir::module_null_guard(&legacy),
        None,
        "legacy: no marker"
    );
    assert_eq!(
        temen_ir::module_null_guard(&guarded),
        Some(guard),
        "guarded entry-less kernel carries the `__null_guard` marker"
    );
    assert!(
        !temen_run::is_named_powerbox_entry(&guarded),
        "still entry-less — the marker does not turn it into a powerbox `_start`"
    );

    // The reserved NULL region is empty: every data segment (the global `g`) starts at or above the
    // guard on the guarded kernel, and below it on legacy (globals at `DATA_BASE`).
    assert!(
        guarded.data.iter().all(|d| d.offset >= guard),
        "guarded: no data segment intrudes on [0, {guard})"
    );
    assert!(
        legacy.data.iter().any(|d| d.offset < guard),
        "legacy: globals sit low (at DATA_BASE), the carve-out this fixes"
    );

    // Behavior parity: the shift is pure relocation — `tick(x) = g + x = 100 + x` on both. The
    // interpreter sets up the module's own window and applies its baked data (the `g = 100`
    // initializer); `tick` takes the prepended `$sp` then `x`.
    let tick = |m: &temen_ir::Module, sp: i64, x: i64| -> i64 {
        let idx = m
            .exports
            .iter()
            .find(|e| e.name == "tick")
            .expect("tick export")
            .func;
        let mut fuel = 1_000_000u64;
        match temen_interp::run(m, idx, &[Value::I64(sp), Value::I64(x)], &mut fuel)
            .expect("run tick")
            .as_slice()
        {
            [Value::I64(v)] => *v,
            o => panic!("unexpected result {o:?}"),
        }
    };
    for x in [0i64, 5, 42] {
        assert_eq!(tick(&legacy, legacy_sp, x), 100 + x, "legacy tick({x})");
        assert_eq!(
            tick(&guarded, guarded_sp, x),
            100 + x,
            "guarded tick({x}) — same value, shifted layout"
        );
    }
}
