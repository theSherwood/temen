//! #2358 — `__vm_region_close(region)`: the on-ramp lowers it to `call.cap SHARED_REGION 5` on the
//! region handle in its one argument, as chibicc's builtin of that name does, beside the region's
//! `map`/`unmap`/`page_size` (ops 0, 1 and 3).

use temen_ir::Inst;

const LL: &str = r#"
declare i64 @__vm_region_close(i32)

define i64 @give_up(i32 %region) {
  %r = call i64 @__vm_region_close(i32 %region)
  ret i64 %r
}
"#;

#[test]
fn close_lowers_to_shared_region_op_5_on_its_argument() {
    let t = temen_llvm::translate_ll_str(LL).expect("translate");
    temen_verify::verify_module(&t.module).expect("verify");
    let caps: Vec<(u32, u32, usize)> = t
        .module
        .funcs
        .iter()
        .flat_map(|f| f.blocks.iter().flat_map(|b| b.insts.iter()))
        .filter_map(|i| match i {
            Inst::CapCall {
                type_id, op, args, ..
            } => Some((*type_id, *op, args.len())),
            _ => None,
        })
        .collect();
    assert_eq!(
        caps,
        vec![(temen_ir::cap_id::SHARED_REGION, 5, 0)],
        "one close, no arguments"
    );
}
