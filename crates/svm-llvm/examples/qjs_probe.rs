//! Scratch: translate QuickJS and analyze `JS_CallInternal`'s dispatch structure — the static
//! generality check for peval (does the Lua dispatch-fold approach apply to a stack VM with JSValues?).
//! Usage: cargo run --release --example qjs_probe -- /tmp/qjs_full/qjs_full.ll

use svm_ir::{Inst, Terminator};

fn main() {
    let p = std::env::args().nth(1).expect("usage: qjs_probe <ll>");
    let opts = svm_llvm::TranslateOptions {
        stub_unresolved_externs: true,
        ..Default::default()
    };
    let t = match svm_llvm::translate_ll_path_with_options(&p, opts) {
        Ok(t) => t,
        Err(e) => {
            println!("TRANSLATE ERR: {e:?}");
            std::process::exit(1);
        }
    };
    let m = &t.module;
    println!("TRANSLATED: {} funcs", m.funcs.len());

    // Find JS_CallInternal (the bytecode VM) + a few key runtime functions.
    let named = |needle: &str| -> Vec<(u32, String)> {
        m.exports
            .iter()
            .filter(|e| e.name.contains(needle))
            .map(|e| (e.func, e.name.clone()))
            .collect()
    };
    for n in ["JS_CallInternal", "js_call_c_function", "JS_NewFloat64", "js_binary_arith_slow"] {
        println!("  {n}: {:?}", named(n));
    }

    let Some((jci, _)) = named("JS_CallInternal").into_iter().next() else {
        println!("JS_CallInternal not found (not exported?) — scanning for the biggest br_table func");
        // Fallback: the dispatch loop is the function with the largest br_table.
        let mut best = (0u32, 0usize);
        for (fi, f) in m.funcs.iter().enumerate() {
            let maxbt = f
                .blocks
                .iter()
                .filter_map(|b| match &b.term {
                    Terminator::BrTable { targets, .. } => Some(targets.len()),
                    _ => None,
                })
                .max()
                .unwrap_or(0);
            if maxbt > best.1 {
                best = (fi as u32, maxbt);
            }
        }
        println!("  biggest br_table: func {} with {} targets", best.0, best.1);
        return;
    };

    let f = &m.funcs[jci as usize];
    let (mut brtabs, mut max_targets, mut loads, mut stores, mut calls, mut indirect) =
        (0, 0, 0, 0, 0, 0);
    for b in &f.blocks {
        for i in &b.insts {
            match i {
                Inst::Load { .. } => loads += 1,
                Inst::Store { .. } => stores += 1,
                Inst::Call { .. } => calls += 1,
                Inst::CallIndirect { .. } => indirect += 1,
                _ => {}
            }
        }
        match &b.term {
            Terminator::BrTable { targets, .. } => {
                brtabs += 1;
                max_targets = max_targets.max(targets.len());
            }
            Terminator::ReturnCallIndirect { .. } => indirect += 1,
            _ => {}
        }
    }
    println!("\nJS_CallInternal (func {jci}):");
    println!("  {} blocks, params={:?} results={:?}", f.blocks.len(), f.params.len(), f.results.len());
    println!("  br_table terminators: {brtabs}  (max targets = {max_targets})  <- the opcode dispatch");
    println!("  loads={loads} stores={stores} calls={calls} call_indirect={indirect}");
    println!("  {} functions in its direct-call closure", closure_size(m, jci));
}

fn closure_size(m: &svm_ir::Module, f: u32) -> usize {
    let mut seen = std::collections::BTreeSet::new();
    let mut stack = vec![f];
    while let Some(c) = stack.pop() {
        if !seen.insert(c) {
            continue;
        }
        for b in &m.funcs[c as usize].blocks {
            for i in &b.insts {
                if let Inst::Call { func, .. } = i {
                    stack.push(*func);
                }
            }
            if let Terminator::ReturnCall { func, .. } = &b.term {
                stack.push(*func);
            }
        }
    }
    seen.len()
}
