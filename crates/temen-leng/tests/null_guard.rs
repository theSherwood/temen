//! **#964/#1094 — the NULL guard is on for every temen-leng powerbox program.** The guarded
//! synth-`_start` link path ([`temen_leng::link_whole_powerbox_manifest`], and the `link_nim_powerbox`
//! bridge built on it) shifts Leng's whole low layout one guard up, so `[0, POWERBOX_NULL_GUARD)` holds
//! *nothing* and a host (`run_powerbox`, the DAP, the browser on-ramp) seeds it `Unmapped`
//! unconditionally (#1094 — the one canonical layout) — a NULL dereference then traps on every engine
//! instead of reading zeros. This test pins the layout contract **toolchain-free** (no nimony needed),
//! so it gates every PR, not only the `nim-e2e` job: a regression that let a segment slip back into the
//! reserved region would go red here immediately.
//!
//! The end-to-end trap parity itself is the oracle's job (`temen-interp`'s `null_guard_oracle` proves a
//! module traps identically on the tree-walker and the bytecode engine); real Nim allocating and running
//! under the shifted layout is `nim_e2e`'s (toolchain-gated). Here we prove the **producer** — that
//! Leng's linker keeps the reserved region clear.

use temen_leng::WholeModule;

/// A minimal C-shaped program: `main($sp, argc, argv, envp) -> cint` (the entry `synth_start` binds),
/// made **frame-needing** by taking the address of a local (`(addr x.0)`) — so the translator prepends
/// the `$sp` param and its signature matches `synth_start`'s 4-arg `call.import "main"`. It calls a
/// void `sink` with that address and returns 0. No allocation, no `system` edges: it links whole with
/// an empty runtime, so this is toolchain-free.
const PROG: &str = "\
(stmts
 (proc :main.0.
  (params
   (param :argc.0 . (i 32))
   (param :argv.0 . (ptr (ptr (c 8))))
   (param :envp.0 . (ptr (ptr (c 8)))))
  (i 32)
  (pragmas (exportc \"main\"))
  (stmts .
   (var :x.0 . (i 64) 0)
   (call sink.0. (addr x.0))
   (ret 0)))
 (proc :sink.0.
  (params (param :p.0 . (ptr (i 64)))) . .
  (stmts . (ret .))))";

/// Linking a whole program through the guarded powerbox manifest path leaves `[0, POWERBOX_NULL_GUARD)`
/// completely empty and bakes the heap bump-pointer words one guard up (in the guard's scratch page) so
/// the compute-shim `mmap` reads them without faulting.
#[test]
fn guarded_powerbox_link_marks_and_clears_the_null_region() {
    let guard = temen_ir::POWERBOX_NULL_GUARD;
    let m = temen_leng::link_whole_powerbox_manifest(
        &[WholeModule {
            stem: "prog",
            src: PROG,
        }],
        vec![],
    )
    .unwrap_or_else(|e| panic!("guarded powerbox link: {e}"));
    temen_verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));

    // (1) The module runs under the (unconditional, #1094) guard extent.
    assert_eq!(
        temen_ir::module_null_guard(&m),
        Some(guard),
        "the powerbox link runs under the unconditional NULL guard"
    );

    // (2) Nothing lives in the reserved NULL region — every data segment starts at or above the guard,
    // so the host can seed `[0, guard)` `Unmapped` without clobbering a real byte.
    for d in &m.data {
        assert!(
            d.offset >= guard,
            "data segment at window offset {} intrudes on the reserved NULL region [0, {guard})",
            d.offset
        );
    }

    // (3) The heap bump-pointer words are baked one guard up, in the guard's scratch page — at
    // `guard + POWERBOX_HEAP_BRK`/`TOP`, exactly where the compute shim reads them and the DAP heap
    // view keys off (`scratch == module_null_guard`). A word left at the legacy 32/40 would sit in the
    // now-`Unmapped` region and fault the shim's `mmap`.
    let scratch = temen_ir::module_null_guard(&m).unwrap();
    let read_word = |off: u64| -> u64 {
        let seg = m
            .data
            .iter()
            .find(|d| d.offset <= off && off + 8 <= d.offset + d.bytes.len() as u64)
            .unwrap_or_else(|| panic!("no data segment covers window offset {off:#x}"));
        let lo = (off - seg.offset) as usize;
        u64::from_le_bytes(seg.bytes[lo..lo + 8].try_into().unwrap())
    };
    let win = 1u64 << m.memory.expect("powerbox module has a window").size_log2;
    let brk = read_word(scratch + temen_ir::POWERBOX_HEAP_BRK);
    let top = read_word(scratch + temen_ir::POWERBOX_HEAP_TOP);
    assert_eq!(
        brk,
        temen_ir::powerbox_entry_sp(&m) + temen_ir::POWERBOX_STACK_RESERVE,
        "heap break seeded just above the data stack"
    );
    assert_eq!(top, win, "heap ceiling seeded to the mapped window top");
    assert!(brk >= guard, "the heap break itself is above the guard");
}

/// The shifted layout is internally consistent end-to-end: running the same guarded program through
/// the **real** `run_powerbox` host — which seeds `[0, guard)` `Unmapped` unconditionally (#1094) —
/// completes cleanly. A regression that left a live store/frame in the reserved region
/// (rather than shifting it up) would fault here instead of returning. The trap side of the guard (a
/// deliberate NULL access faulting) is pinned generically by `temen-interp`'s `null_guard_oracle`.
#[test]
fn guarded_program_runs_clean_under_the_powerbox_host() {
    let m = temen_leng::link_whole_powerbox_manifest(
        &[WholeModule {
            stem: "prog",
            src: PROG,
        }],
        vec![],
    )
    .unwrap_or_else(|e| panic!("guarded powerbox link: {e}"));
    assert!(
        temen_run::is_named_powerbox_entry(&m),
        "the guarded link is a powerbox entry (paramless func-0 `_start`)"
    );
    let run = temen_run::run_powerbox(&m, &[]).unwrap_or_else(|e| {
        panic!("guarded program must run clean under the guard-seeding host, got: {e}")
    });
    match run.outcome {
        temen_run::Outcome::Returned(_) | temen_run::Outcome::Exited(0) => {}
        other => panic!("guarded program ended abnormally: {other:?}"),
    }
}
