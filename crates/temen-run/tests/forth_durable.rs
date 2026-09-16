//! #1236 — **freeze/thaw a live Forth session**, step 1: *is the Forth kernel a durable domain?*
//!
//! Yes, on both halves. Its **code** is durable-ready: the durable transform accepts all 77 functions
//! through `transform_module_assume_confined`, and the instrumented module verifies. Its **memory map**
//! declares where the durable runtime may keep its per-context shadow regions — `memory 20 shadow
//! 475136 524288`, i.e. `[0x74000, 0x80000)`, inside the `sandbox` spawn-scratch page — and the verifier
//! holds that declaration to the geometry every backend assumes, including that no data segment
//! overlaps it (the R9 "guest bytes never alias the arena" contract, checked statically).
//!
//! History: before #1503 the arena was a substrate constant under `0x10000`, and the kernel's 24 data
//! segments at `0x8000` sat inside it — a live collision once a session had a root plus a few `task`
//! fibers. The fix was never to move the data: placement is the guest's (INVARIANTS.md #16), so the
//! kernel now says where its arena goes and the collision cannot exist. `transform_module` (the strict
//! path for an *untrusted* module) still fails closed with `GuestUsesMemory` because the kernel does
//! guest loads/stores at all; a cooperating toolchain's module — this one — uses `assume_confined`.
#![cfg(all(unix, target_arch = "x86_64"))]

fn kernel() -> temen_ir::Module {
    let m = temen_text::parse_module(include_str!("../demos/forth/forth.temt"))
        .expect("forth.temt parses");
    temen_verify::verify_module(&m).expect("forth.temt verifies");
    m
}

/// The kernel's **code** is durable-ready: every function passes the durable transform's shape rules
/// through the confined path, and the instrumented module verifies. This is the half of #1236 that
/// needed no work — if it ever regresses, the memory-map work below is pointless.
#[test]
fn the_durable_transform_accepts_the_whole_kernel() {
    let m = kernel();
    let t = temen_durable::transform_module_assume_confined(&m)
        .expect("the durable transform accepts the kernel through the confined path");
    assert_eq!(
        t.funcs.len(),
        m.funcs.len(),
        "the transform instruments in place; it must not add or drop functions"
    );
    temen_verify::verify_module(&t).expect("the instrumented kernel verifies");
}

/// …and the **strict** path refuses it, because the kernel does guest memory ops that could alias
/// the reserved region. That refusal is correct and is what `assume_confined` exists to let a
/// cooperating guest opt out of — once it genuinely reserves the region (see the module docs).
#[test]
fn the_strict_transform_refuses_the_kernel_for_aliasing_the_durable_reserve() {
    let m = kernel();
    assert_eq!(
        temen_durable::transform_module(&m),
        Err(temen_durable::TransformError::GuestUsesMemory),
        "the strict path must fail closed for a guest whose memory ops could alias the durable control words or its arena"
    );
}

/// The kernel **declares** its shadow arena, the verifier accepts it, and it is clear of every data
/// segment — the precondition #1236's freeze/thaw needs, now a property of the module rather than a
/// gap to work around. It also holds enough contexts for a REPL with a root and a handful of `task`
/// fibers.
#[test]
fn the_kernel_declares_an_arena_clear_of_its_data() {
    use temen_ir::durable_abi::{ShadowArena, SHADOW_STRIDE};
    let m = kernel(); // `kernel()` already ran the verifier, which rejects a data/arena overlap
    let arena = m
        .memory
        .and_then(|x| x.shadow)
        .expect("forth.temt declares a shadow arena");
    assert_eq!(
        arena,
        ShadowArena {
            base: 0x74000,
            end: 0x80000
        },
        "the arena lives in the sandbox spawn-scratch page, below the child carve"
    );
    for (i, d) in m.data.iter().enumerate() {
        let end = d.offset + d.bytes.len() as u64;
        assert!(
            end <= arena.base || d.offset >= arena.end,
            "data segment {i} [{:#x}, {end:#x}) overlaps the arena [{:#x}, {:#x})",
            d.offset,
            arena.base,
            arena.end
        );
    }
    assert!(
        arena.contexts() >= 8,
        "a REPL with a root and a few task fibers needs several contexts; got {} (stride {SHADOW_STRIDE})",
        arena.contexts()
    );
}
