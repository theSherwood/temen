//! **#964/#1094 — the NULL guard reaches entry-less kernels too.** A reactor kernel (a `tick`-only
//! module: no `main`, so temen-llvm synthesizes no powerbox `_start`) used to be a legacy carve-out:
//! an entry-less module's globals sat at `DATA_BASE` (16) — *inside* the would-be guard region — so it
//! could never be seeded `Unmapped`.
//!
//! This pins the fix: an entry-less module leaves `[0, guard)` empty, so a host seeds the reserved
//! region `Unmapped` and a NULL dereference traps — exactly like the `_start` path, and
//! **unconditionally** (#1094 — the one canonical layout; no `__null_guard` marker export needed).
//!
//! #1777 took the same step for the rest of the low scratch: an entry-less unit's globals start at
//! the globals base every module uses (`stack_page`), not one guard up. An entry-less *library* (a
//! language runtime) is linked with a program and wrapped by `synth_manifest_start`, whose `_start`
//! seeds the heap words at `guard + 32/40` and whose host seeds the §3e args blob at `guard + 128` —
//! both used to land on the library's globals. No clang: the fixture is inline textual LLVM IR, so
//! this runs in every job.

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
    let t =
        temen_llvm::translate_ll_str_with_options(KERNEL, temen_llvm::TranslateOptions::default())
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
        temen_ir::module_null_guard(),
        guard,
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
    // ...and so is the low scratch above it (#1777): the heap words, the durable control words and
    // the args blob a host seeds there never land on a global.
    let scratch_end = guard + temen_ir::POWERBOX_ARGS_END;
    assert!(
        kernel.data.iter().all(|d| d.offset >= scratch_end),
        "no data segment intrudes on the low scratch [{guard}, {scratch_end})"
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

/// An entry-less **library** in the shape of a language runtime: a zero-initialized table (BSS — no
/// data segment, so the layout assertion above cannot see it) and a function that reads it back.
const LIBRARY: &str = r#"
@table = global [256 x i64] zeroinitializer

define i64 @table_or() {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %i1, %loop ]
  %acc = phi i64 [ 0, %entry ], [ %a1, %loop ]
  %p = getelementptr [256 x i64], ptr @table, i64 0, i64 %i
  %v = load i64, ptr %p
  %a1 = or i64 %acc, %v
  %i1 = add i64 %i, 1
  %c = icmp ult i64 %i1, 256
  br i1 %c, label %loop, label %done
done:
  ret i64 %a1
}
"#;

/// #1777: wrapped as a powerbox program by `synth_manifest_start` — the way a language runtime is run
/// once linked with its program — and given argv + env, the library's zeroed globals stay zero. The
/// host seeds the §3e args blob at `guard + 128`; with the old entry-less base (`guard + 16`) the
/// 2 KiB table sat under it, and the run read the blob's bytes back as the table's contents.
#[test]
fn an_entryless_librarys_globals_survive_a_seeded_args_blob() {
    let t =
        temen_llvm::translate_ll_str_with_options(LIBRARY, temen_llvm::TranslateOptions::default())
            .expect("translate library");
    let entry = t
        .module
        .resolve_export("table_or")
        .expect("table_or export");
    let program = temen_ir::synth_manifest_start(t.module, entry, false).expect("powerbox wrap");
    let cfg = temen_run::RunConfig {
        args: vec![b"prog".to_vec(), b"an-argument".to_vec()],
        env: vec![b"KEY=a-value-long-enough-to-matter".to_vec()],
        ..Default::default()
    };
    let run = temen_run::instantiate(program)
        .expect("instantiate")
        .run_diff(&cfg)
        .expect("run (interp == JIT)");
    assert_eq!(
        run.outcome,
        temen_run::Outcome::Returned(vec![Value::I64(0)]),
        "the library's zeroed table must read back zero: seeding argv/env may not reach a global"
    );
}
