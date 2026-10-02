//! `__vm_instantiate_detached`: the on-ramp lowers it to `call.cap INSTANTIATOR 15` on the handle in its
//! first argument, passing the other eight (`budget, module, grants_ptr, grants_n, entry, size_log2,
//! args_ptr, args_len`) through in order, with op 15's retired `quota` slot 0 (#1944 slice 3) — the
//! detached spawn (PROCESS.md §5) reachable from C and Rust, beside `__vm_instantiate_rec` (op 17).

use temen_ir::Inst;

const LL: &str = r#"
declare i64 @__vm_instantiate_detached(i32, i64, i64, i64, i64, i64, i64, i64, i64)
declare i64 @__vm_join(i32, i64)

define i64 @spawn(i32 %inst, i64 %budget, i64 %module) {
  %h = call i64 @__vm_instantiate_detached(i32 %inst, i64 %budget, i64 %module, i64 0, i64 0, i64 0, i64 21, i64 0, i64 0)
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

const LL_WAIT_BUDGET: &str = r#"
declare i64 @__vm_budget_split(i32, i64, i64, i64)
declare i64 @__vm_budget_read(i32, i64)
declare i64 @__vm_instantiate_detached(i32, i64, i64, i64, i64, i64, i64, i64, i64)
declare i64 @__vm_wait(i32, i64)
declare i64 @__vm_join(i32, i64)

define i64 @spawn(i32 %inst, i32 %budget, i64 %module) {
  %node = call i64 @__vm_budget_split(i32 %budget, i64 1000, i64 -1, i64 -1)
  %room = call i64 @__vm_budget_read(i32 %budget, i64 0)
  %h = call i64 @__vm_instantiate_detached(i32 %inst, i64 %node, i64 %module, i64 0, i64 0, i64 0, i64 21, i64 0, i64 0)
  %w = call i64 @__vm_wait(i32 %inst, i64 %h)
  %r = call i64 @__vm_join(i32 %inst, i64 %h)
  %s = add i64 %w, %r
  %t = add i64 %s, %room
  ret i64 %t
}
"#;

/// #2053 — a spawner bounds a child's fuel and outlives its running out: `__vm_budget_split` lowers
/// to `call.cap BUDGET 0` (three ceilings), `__vm_budget_read` to `call.cap BUDGET 1` (the field), and
/// `__vm_wait` to `call.cap INSTANTIATOR 18` with the child handle, as `__vm_join` is op 1.
#[test]
fn budget_split_read_and_wait_lower_to_their_ops() {
    let t = temen_llvm::translate_ll_str(LL_WAIT_BUDGET).expect("translate");
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
        vec![(14, 0, 3), (14, 1, 1), (6, 15, 9), (6, 18, 1), (6, 1, 1)],
        "split, read, spawn detached, wait, join"
    );
}
