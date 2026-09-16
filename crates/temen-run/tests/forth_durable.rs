//! #1236 — **freeze/thaw a live Forth session.** The issue says "pin it or record what refuses", so
//! this file records the answer to step 1: *what stops the Forth kernel being a durable domain today.*
//!
//! The answer is narrow and good news. The kernel's **code** is already durable-ready: the durable
//! transform accepts all 77 of its functions through `transform_module_assume_confined`, and the
//! instrumented module verifies. Nothing about the tokenizer, the compiler, the fibers or the §22
//! install path is outside the transform's shape.
//!
//! What refuses is the **memory map**, and only one region of it. `transform_module` — the strict
//! path for an untrusted module — fails closed with `GuestUsesMemory` because the kernel does guest
//! loads/stores that could alias the reserved durable region `[0, `ShadowArena::LEGACY.end`)` (R9). Its own
//! doc names the way out: a guest from a cooperating toolchain that *reserves* that region (basing
//! its data and heap at `ShadowArena::LEGACY.end`) uses `transform_module_assume_confined` instead. The
//! Forth kernel almost does reserve it — everything from the globals up already starts at exactly
//! `ShadowArena::LEGACY.end` — **except its 24 data segments, which sit at `0x8000`**, inside the region the
//! durable runtime puts the state word, the shadow-SP and the per-context shadow stacks.
//!
//! That is not a latent collision, it is a live one for exactly the session #1236 wants to freeze:
//! shadow context `i` occupies `[ShadowArena::region_base(i), +STRIDE)`, so a session with a
//! handful of contexts (the root plus a few `task` fibers) reaches `0x8000` and overwrites the
//! prelude. Using `assume_confined` as the map stands would be lying to the transform.
//!
//! So step 1's finding: **a Forth-side shape change, not a durability-axis gap** — relocate the data
//! block above `ShadowArena::LEGACY.end`. The map below the session-snapshot line (`0x40000`, #1235) is
//! fully packed, so that means reclaiming ~6.5 KiB from a region that has slack; it is a self-
//! contained change and the rest of #1236 (freeze mid-session, thaw, compare transcripts) unblocks
//! behind it. These tests pin the finding so the next person does not re-derive it.
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
        "the strict path must fail closed for a guest whose memory ops could reach [0, ShadowArena::LEGACY.end)"
    );
}

/// The one thing standing between the kernel and a durable domain: its data segments live inside the
/// region the durable runtime owns. Pinned as a **known gap**, so the day the map moves this test
/// flips and says so rather than quietly passing.
#[test]
fn the_kernels_data_block_still_sits_inside_the_durable_reserve() {
    let m = kernel();
    let reserve = temen_interp::ShadowArena::LEGACY.end;
    let inside: Vec<_> = m.data.iter().filter(|d| d.offset < reserve).collect();
    let lo = inside.iter().map(|d| d.offset).min();
    let hi = inside.iter().map(|d| d.offset + d.bytes.len() as u64).max();

    // Everything that is NOT a data segment already clears the reserve — the globals base at exactly
    // `ShadowArena::LEGACY.end`. So the relocation is one contiguous block, not a re-lay of the whole map.
    assert!(
        !inside.is_empty(),
        "the data block has moved above ShadowArena::LEGACY.end ({reserve:#x}) — #1236's precondition is met. \
         Delete this test, switch the kernel to `transform_module_assume_confined`, and carry on with \
         freeze/thaw (step 2: freeze mid-session with a suspended `task`, thaw, and compare the two \
         halves' stdout against the uninterrupted run)."
    );
    let (lo, hi) = (lo.unwrap(), hi.unwrap());
    let bytes: usize = inside.iter().map(|d| d.bytes.len()).sum();
    assert!(
        lo >= temen_ir::POWERBOX_NULL_GUARD,
        "the data block must at least clear the NULL guard: starts at {lo:#x}"
    );
    // The numbers the relocation has to satisfy: this much payload has to find a home in
    // [`ShadowArena::LEGACY.end`, SESSION_SNAP) — above the durable region, below the line a `JitSession`
    // carries across prompts (#1235), because the prelude and the error messages must survive one.
    println!(
        "#1236 blocker: {} data segment(s), {bytes} bytes spanning [{lo:#x}, {hi:#x}), \
         inside the durable reserve [0, {reserve:#x})",
        inside.len()
    );
}
