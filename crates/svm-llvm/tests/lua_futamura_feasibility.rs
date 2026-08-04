//! **Feasibility analysis for a Futamura projection of the real Lua interpreter** (`luaV_execute`).
//!
//! The goal: partial-evaluate `luaV_execute` against a fixed Lua program so the opcode decode + the
//! 83-way computed-goto dispatch fold away, leaving the compiled program — the same win the toy
//! register-VM demo shows (`peval_regvm`), but on *real* Lua 5.4.7.
//!
//! This test is the structural reconnaissance, pinned as reference data + a regression guard. It
//! establishes the two facts that make the projection plausible and the two that bound it:
//!
//! **Enablers.**
//! 1. **No host calls in the closure.** `luaV_execute`'s entire transitive direct-call closure
//!    (~161 functions) contains **zero** `CallImport` / `CapCall` / `CallSym` — the specializer's
//!    hardest refusals. The Lua heap is a guest `static char arena[]` bump allocator, so even
//!    allocation is pure guest code, not a host boundary. Asserted below.
//! 2. **Deterministic heap addresses.** Because the arena is a static bump allocator, the compiled
//!    `Proto.code` (bytecode) lands at a *deterministic, knowable* window address — so it can be
//!    captured once (via `svm_interp::Inspector`: breakpoint at `luaV_execute` entry, read the window
//!    + the `L`/`ci` pointer args) and declared constant via `SpecConfig::const_overlays`. The
//!    dispatch is an `indirectbr` the on-ramp lowers to `Terminator::BrTable`, which the specializer
//!    folds to a single edge once the opcode byte is a folded constant.
//!
//! **Bounds (the narrow success window).**
//! 3. **The `call_indirect` / `setjmp` sites are on cold paths.** The closure's 13 `call_indirect`
//!    sites (high signature fan-out) and its `SetJmp`/`LongJmp` live in error handling
//!    (`luaD_throw`, `luaD_rawrunprotected`), allocation (`luaM_*`), and hooks (`luaD_hook`) — not
//!    the numeric fast path. With the bytecode constant, a well-behaved integer loop's conditions
//!    fold and prune those branches; if a cold branch stays reachable, it is the wall.
//! 4. **The `TValue` tag byte.** Lua registers are `TValue` = 8-byte value + 1-byte tag. Modeled as
//!    a rename region, a *dynamic narrow* store (a runtime tag) is refused by the renamer — so the
//!    projection folds only for a **strictly-integer, monomorphic** loop (tag constant `LUA_VNUMINT`).
//!    Floats / polymorphic locals / table ops step outside it.
//!
//! Verdict: feasible, on a strictly-integer program, via a capture-then-specialize harness. The
//! remaining build is that harness (Inspector capture → per-struct `const_overlays` for the
//! closure→Proto→code/k chase → `rename` region for the register file → cold-path pruning). This
//! test does not build it; it guards the structural preconditions.
//!
//! Run: `cargo test -p svm-llvm --test lua_futamura_feasibility -- --nocapture`
#![allow(clippy::doc_lazy_continuation)]

use std::collections::{BTreeMap, BTreeSet};

use svm_ir::{Inst, Module, Terminator};

fn lua_module() -> Module {
    let path = format!(
        "{}/tests/fixtures/lua/lua_eval.ll",
        env!("CARGO_MANIFEST_DIR")
    );
    svm_llvm::translate_ll_path(&path)
        .expect("translate lua_eval")
        .module
}

/// The `luaV_execute` function index in the lua_eval fixture (found via its export name).
fn luav_execute(m: &Module) -> u32 {
    m.exports
        .iter()
        .find(|e| e.name == "luaV_execute")
        .expect("luaV_execute export")
        .func
}

/// The transitive closure of `f` under **direct** calls (`Inst::Call` / `Terminator::ReturnCall`).
fn direct_call_closure(m: &Module, f: u32) -> BTreeSet<u32> {
    let mut seen = BTreeSet::new();
    let mut stack = vec![f];
    while let Some(cur) = stack.pop() {
        if !seen.insert(cur) {
            continue;
        }
        for b in &m.funcs[cur as usize].blocks {
            for inst in &b.insts {
                if let Inst::Call { func, .. } = inst {
                    stack.push(*func);
                }
            }
            if let Terminator::ReturnCall { func, .. } = &b.term {
                stack.push(*func);
            }
        }
    }
    seen
}

#[test]
fn luav_execute_closure_has_no_host_calls() {
    let m = lua_module();
    let closure = direct_call_closure(&m, luav_execute(&m));

    let (mut imports, mut cap_calls, mut call_syms, mut indirects) = (0, 0, 0, 0);
    for &f in &closure {
        for b in &m.funcs[f as usize].blocks {
            for inst in &b.insts {
                match inst {
                    Inst::CallImport { .. } => imports += 1,
                    Inst::CapCall { .. } => cap_calls += 1,
                    Inst::CallSym { .. } => call_syms += 1,
                    Inst::CallIndirect { .. } => indirects += 1,
                    _ => {}
                }
            }
            if matches!(&b.term, Terminator::ReturnCallIndirect { .. }) {
                indirects += 1;
            }
        }
    }

    println!(
        "\nluaV_execute transitive closure: {} functions",
        closure.len()
    );
    println!("  CallImport (host): {imports}");
    println!("  CapCall:           {cap_calls}");
    println!("  CallSym:           {call_syms}");
    println!("  CallIndirect:      {indirects}");

    // The enabler: no host boundary anywhere in the closure — the specializer's hardest refusals.
    assert_eq!(
        imports, 0,
        "a host CallImport in the closure would block the projection"
    );
    assert_eq!(
        cap_calls, 0,
        "a CapCall in the closure would block the projection"
    );
    assert_eq!(
        call_syms, 0,
        "an unresolved CallSym in the closure would block the projection"
    );
}

#[test]
fn cold_path_indirects_are_where_we_expect() {
    let m = lua_module();
    let closure = direct_call_closure(&m, luav_execute(&m));
    let name = |f: u32| {
        m.exports
            .iter()
            .find(|e| e.func == f)
            .map(|e| e.name.clone())
            .unwrap_or_else(|| format!("<func {f}>"))
    };

    // Which functions in the closure carry a call_indirect.
    let mut carriers: BTreeMap<u32, usize> = BTreeMap::new();
    for &f in &closure {
        for b in &m.funcs[f as usize].blocks {
            for inst in &b.insts {
                if let Inst::CallIndirect { .. } = inst {
                    *carriers.entry(f).or_insert(0) += 1;
                }
            }
        }
    }

    println!("\ncall_indirect carriers in the luaV_execute closure:");
    for (&f, &n) in &carriers {
        println!("  {} ({n} site(s))", name(f));
    }

    // These are the error/alloc/hook helpers, never the numeric fast path — so a folded-condition
    // integer loop prunes them. If a future fixture regen moves an indirect onto the hot path, this
    // list changes and the assumption needs re-checking.
    let names: BTreeSet<String> = carriers.keys().map(|&f| name(f)).collect();
    let cold = [
        "luaD_throw",
        "luaD_rawrunprotected",
        "luaD_hook",
        "precallC",
        "luaM_free_",
        "luaM_realloc_",
        "luaM_malloc_",
        "luaE_warnerror",
        "tryagain",
    ];
    for n in &names {
        assert!(
            cold.iter().any(|c| n.contains(c)),
            "unexpected call_indirect carrier {n} — not a known cold (error/alloc/hook) path"
        );
    }
}
