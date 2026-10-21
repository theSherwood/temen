//! The close builtins: each lowers to its interface's `close` op on the handle in its one argument,
//! as chibicc's builtin of that name does. `__vm_region_close(region)` is `call.cap SHARED_REGION 5`
//! (#2358), beside the region's `map`/`unmap`/`page_size` (ops 0, 1 and 3); `__vm_budget_close(budget)`
//! is `call.cap BUDGET 3` (#2384), beside `split`/`read` (ops 0 and 1).

use temen_ir::{cap_id, Inst};

/// The `(interface, op, argument count)` of every capability call `name(handle)` lowers to.
fn lowered(name: &str) -> Vec<(u32, u32, usize)> {
    let ll = format!(
        "declare i64 @{name}(i32)\n\ndefine i64 @give_up(i32 %h) {{\n  %r = call i64 @{name}(i32 %h)\n  ret i64 %r\n}}\n"
    );
    let t = temen_llvm::translate_ll_str(&ll).expect("translate");
    temen_verify::verify_module(&t.module).expect("verify");
    t.module
        .funcs
        .iter()
        .flat_map(|f| f.blocks.iter().flat_map(|b| b.insts.iter()))
        .filter_map(|i| match i {
            Inst::CapCall {
                type_id, op, args, ..
            } => Some((*type_id, *op, args.len())),
            _ => None,
        })
        .collect()
}

#[test]
fn region_close_lowers_to_shared_region_op_5_on_its_argument() {
    assert_eq!(
        lowered("__vm_region_close"),
        vec![(cap_id::SHARED_REGION, 5, 0)],
        "one close, no arguments"
    );
}

#[test]
fn budget_close_lowers_to_budget_op_3_on_its_argument() {
    assert_eq!(
        lowered("__vm_budget_close"),
        vec![(cap_id::BUDGET, 3, 0)],
        "one close, no arguments"
    );
}
