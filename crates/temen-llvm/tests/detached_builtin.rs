//! `__vm_instantiate_detached`: the on-ramp lowers it to `call.cap INSTANTIATOR 15` on the handle in its
//! first argument, passing the other nine (`budget, module, grants_ptr, grants_n, entry, size_log2,
//! quota, args_ptr, args_len`) through in order — the detached spawn (PROCESS.md §5) reachable from C
//! and Rust, beside `__vm_instantiate_rec` (op 17).

use temen_ir::Inst;

const LL: &str = r#"
declare i64 @__vm_instantiate_detached(i32, i64, i64, i64, i64, i64, i64, i64, i64, i64)
declare i64 @__vm_join(i32, i64)

define i64 @spawn(i32 %inst, i64 %budget, i64 %module) {
  %h = call i64 @__vm_instantiate_detached(i32 %inst, i64 %budget, i64 %module, i64 0, i64 0, i64 0, i64 21, i64 1000, i64 0, i64 0)
  %r = call i64 @__vm_join(i32 %inst, i64 %h)
  ret i64 %r
}
"#;

#[test]
fn it_lowers_to_instantiator_op_15_with_nine_arguments() {
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
        vec![(6, 15, 9), (6, 1, 1)],
        "spawn detached, then join"
    );
}

const LL_REC: &str = r#"
declare i64 @__vm_instantiate_rec(i32, i64)
declare i64 @__vm_join(i32, i64)

define i64 @spawn(i32 %inst, i64 %rec) {
  %h = call i64 @__vm_instantiate_rec(i32 %inst, i64 %rec)
  %r = call i64 @__vm_join(i32 %inst, i64 %h)
  ret i64 %r
}
"#;

/// `__vm_instantiate_rec` — the one spawn form (the op-17 v1 record) — lowers to `call.cap
/// INSTANTIATOR 17` with the record pointer as its one argument, as chibicc's builtin of that name.
#[test]
fn the_record_spawn_lowers_to_instantiator_op_17() {
    let t = temen_llvm::translate_ll_str(LL_REC).expect("translate");
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
    assert_eq!(caps, vec![(6, 17, 1), (6, 1, 1)], "spawn, then join");
}
