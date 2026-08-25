//! **#964/#1094 — the NULL guard reaches entry-less kernels too.** A reactor kernel (a `tick`-only
//! module: no `main`, so temen-llvm synthesizes no powerbox `_start`) used to be a legacy carve-out:
//! an entry-less module's globals sat at `DATA_BASE` (16) — *inside* the would-be guard region — so it
//! could never be seeded `Unmapped`.
//!
//! This pins the fix: an entry-less module bases its globals one guard up (`scratch + DATA_BASE`),
//! leaving `[0, guard)` empty, so a host seeds the reserved region `Unmapped` and a NULL dereference
//! traps — exactly like the `_start` path, and **unconditionally** (#1094 — the one canonical layout;
//! no `__null_guard` marker export needed). No clang: the fixture is inline textual LLVM IR, so this
//! runs in every job.

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
fn translate() -> (temen_ir::Module, i64) {
    let t = temen_llvm::translate_ll_str_with_options(
        KERNEL,
        temen_llvm::TranslateOptions::default(),
    )
    .expect("translate kernel");
    temen_verify::verify_module(&t.module).expect("verify kernel");
    (t.module, t.entry_sp as i64)
}

/// The guarded entry-less kernel: globals shifted above the guard so `[0, guard)` is empty, still
/// entry-less (no synthesized `_start`), and `tick(x) = 100 + x`.
#[test]
fn entryless_kernel_is_guarded_and_keeps_the_null_region_empty() {
    let guard = temen_ir::POWERBOX_NULL_GUARD;
    let (kernel, sp) = translate();

    // The guard is unconditional (#1094) even though the kernel has no `_start`.
    assert_eq!(
        temen_ir::module_null_guard(&kernel),
        Some(guard),
        "the guard is unconditional for an entry-less kernel too"
    );
    assert!(
        !temen_run::is_named_powerbox_entry(&kernel),
        "still entry-less — no synthesized powerbox `_start`"
    );
    // No stale marker export is emitted any more.
    assert_eq!(
        kernel.resolve_export("__null_guard"),
        None,
        "the retired `__null_guard` marker export is not emitted (#1094)"
    );

    // The reserved NULL region is empty: every data segment (the global `g`) starts at or above the
    // guard, so a host can seed `[0, guard)` `Unmapped` without clobbering a live byte.
    assert!(
        kernel.data.iter().all(|d| d.offset >= guard),
        "no data segment intrudes on [0, {guard})"
    );

    // Behavior: the shift is pure relocation — `tick(x) = g + x = 100 + x`. The interpreter sets up
    // the module's own window and applies its baked data (the `g = 100` initializer); `tick` takes the
    // prepended `$sp` then `x`.
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
        assert_eq!(
            tick(&kernel, sp, x),
            100 + x,
            "guarded tick({x}) — shifted layout, same value"
        );
    }
}
