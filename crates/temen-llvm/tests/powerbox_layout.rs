//! A library (no `main`) that becomes a powerbox program only after translation — a separately
//! linked runtime given an entry by `synth_manifest_child_start` — keeps its globals clear of the
//! argument area, so a detached spawn's args payload never lands on them. Every translation lays
//! its globals out from `stack_page` (#1777); #1776 first made this an opt-in
//! `TranslateOptions::powerbox_layout`, and these were its tests. The translator cannot know a unit
//! will be wrapped, so the opt-in's off state was only ever the bug.

use temen_interp::{run_capture_reserved_with_host, Host, Value};

/// The child: an initialized global array and an entry returning the payload's first word plus
/// `g[15]` (7 unless something overwrote it).
const CHILD: &str = r#"
@g = global [32 x i64] [i64 7, i64 7, i64 7, i64 7, i64 7, i64 7, i64 7, i64 7, i64 7, i64 7, i64 7, i64 7, i64 7, i64 7, i64 7, i64 7, i64 7, i64 7, i64 7, i64 7, i64 7, i64 7, i64 7, i64 7, i64 7, i64 7, i64 7, i64 7, i64 7, i64 7, i64 7, i64 7]

define i64 @entry() {
  %a = load volatile i64, ptr inttoptr (i64 16512 to ptr)
  %p = getelementptr [32 x i64], ptr @g, i64 0, i64 15
  %c = load volatile i64, ptr %p
  %r = add i64 %a, %c
  ret i64 %r
}
"#;

/// The parent: spawns the child detached with a 16-byte payload `{35, 0}` and joins it. The spawn
/// record (`temen_ir::SpawnRec`) is laid on its stack as eleven words: version 1, the child's window
/// and no pager, the module and the budget, the payload, no region; every other word is 0, none.
fn parent(child_log2: u8) -> String {
    let window_no_pager = i64::from(child_log2) | (i64::from(u32::MAX) << 32);
    format!(
        r#"
declare i64 @__vm_instantiate_rec(i32, i64)
declare i64 @__vm_join(i32, i64, ptr)

define i64 @spawn(i32 %inst, i64 %budget, i64 %module) {{
  %buf = alloca [2 x i64]
  %rec = alloca [11 x i64]
  %st = alloca i64
  store i64 35, ptr %buf
  %hi = getelementptr i64, ptr %buf, i64 1
  store i64 0, ptr %hi
  %p = ptrtoint ptr %buf to i64
  store i64 1, ptr %rec
  %w1 = getelementptr i64, ptr %rec, i64 1
  store i64 0, ptr %w1
  %w2 = getelementptr i64, ptr %rec, i64 2
  store i64 {window_no_pager}, ptr %w2
  %m = and i64 %module, 4294967295
  %b = shl i64 %budget, 32
  %mb = or i64 %m, %b
  %w3 = getelementptr i64, ptr %rec, i64 3
  store i64 %mb, ptr %w3
  %w4 = getelementptr i64, ptr %rec, i64 4
  store i64 0, ptr %w4
  %w5 = getelementptr i64, ptr %rec, i64 5
  store i64 0, ptr %w5
  %w6 = getelementptr i64, ptr %rec, i64 6
  store i64 0, ptr %w6
  %w7 = getelementptr i64, ptr %rec, i64 7
  store i64 %p, ptr %w7
  %w8 = getelementptr i64, ptr %rec, i64 8
  store i64 16, ptr %w8
  %w9 = getelementptr i64, ptr %rec, i64 9
  store i64 4294967295, ptr %w9
  %w10 = getelementptr i64, ptr %rec, i64 10
  store i64 0, ptr %w10
  %ri = ptrtoint ptr %rec to i64
  %h = call i64 @__vm_instantiate_rec(i32 %inst, i64 %ri)
  %r = call i64 @__vm_join(i32 %inst, i64 %h, ptr %st)
  ret i64 %r
}}
"#
    )
}

fn child() -> temen_llvm::Translated {
    temen_llvm::translate_ll_str(CHILD).expect("translate the child")
}

fn g_addr(t: &temen_llvm::Translated) -> u64 {
    t.data_symbols
        .iter()
        .find(|d| d.name == "g")
        .expect("@g")
        .addr
}

/// Spawns `t` (as a synthesized child `_start`) from a translated parent on the tree-walk oracle.
fn spawn(t: temen_llvm::Translated) -> i64 {
    let entry = t
        .exports
        .iter()
        .find(|(n, _)| n == "entry")
        .expect("entry")
        .1;
    let image = temen_ir::synth_manifest_child_start(t.module, entry, false).expect("synth");
    temen_verify::verify_module(&image).expect("verify the child");
    let log2 = image.memory.expect("a window").size_log2;
    let p = temen_llvm::translate_ll_str(&parent(log2)).expect("translate the parent");
    temen_verify::verify_module(&p.module).expect("verify the parent");
    let func = p
        .exports
        .iter()
        .find(|(n, _)| n == "spawn")
        .expect("spawn")
        .1;
    let plog2 = p.module.memory.expect("a window").size_log2;
    let sp = temen_ir::powerbox_entry_sp(&p.module);
    let mut host = Host::new();
    let inst = host.grant_instantiator();
    let module = host.grant_module(&image);
    // The child's fuel is capped by the budget it is spawned with (#1944 slice 3).
    let budget = host.grant_budget(1_000_000, -1, -1);
    let mut fuel = 10_000_000u64;
    let init = vec![0u8; 1 << plog2];
    let args = [
        Value::I64(sp as i64),
        Value::I32(inst),
        Value::I64(budget as i64),
        Value::I64(module as i64),
    ];
    let out =
        run_capture_reserved_with_host(&p.module, func, &args, &mut fuel, &init, 0, &mut host)
            .0
            .expect("run the parent");
    match out.as_slice() {
        [Value::I64(v)] => *v,
        other => panic!("unexpected result {other:?}"),
    }
}

#[test]
fn a_librarys_globals_sit_above_the_argument_area() {
    assert!(g_addr(&child()) >= temen_ir::module_args_end());
}

#[test]
fn a_detached_payload_leaves_the_globals_intact() {
    assert_eq!(spawn(child()), 35 + 7, "the payload must not land on @g");
}
