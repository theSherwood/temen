//! #1936 — the notification builtins, as chibicc's of those names: `__vm_post(region, off, badge)`
//! is `call.cap SHARED_REGION 6` on the region handle in its first argument, with the offset and the
//! badge, and `__vm_signal(post)` is `call.cap SHARED_REGION 7` on the post in its one argument.

use temen_ir::{cap_id, Inst};

/// The `(interface, op, argument count)` of every capability call in `ll`'s functions.
fn lowered(ll: &str) -> Vec<(u32, u32, usize)> {
    let t = temen_llvm::translate_ll_str(ll).expect("translate");
    temen_verify::verify_module(&t.module).expect("verify");
    t.module
        .funcs
        .iter()
        .flat_map(|f| f.blocks.iter().flat_map(|b| b.insts.iter()))
        .filter_map(|i| match i {
            Inst::CapCall(c) => {
                let temen_ir::CapCall {
                    type_id, op, args, ..
                } = &**c;
                Some((*type_id, *op, args.len()))
            }
            _ => None,
        })
        .collect()
}

#[test]
fn post_lowers_to_shared_region_op_6_with_its_offset_and_badge() {
    let ll = "declare i64 @__vm_post(i32, i64, i64)\n\n\
              define i64 @mint(i32 %r) {\n  %p = call i64 @__vm_post(i32 %r, i64 8, i64 4)\n  ret i64 %p\n}\n";
    assert_eq!(lowered(ll), vec![(cap_id::SHARED_REGION, 6, 2)]);
}

#[test]
fn signal_lowers_to_shared_region_op_7_on_its_argument() {
    let ll = "declare i64 @__vm_signal(i32)\n\n\
              define i64 @ring(i32 %p) {\n  %r = call i64 @__vm_signal(i32 %p)\n  ret i64 %r\n}\n";
    assert_eq!(lowered(ll), vec![(cap_id::SHARED_REGION, 7, 0)]);
}
