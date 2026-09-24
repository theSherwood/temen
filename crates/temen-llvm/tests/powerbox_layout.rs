//! `TranslateOptions::powerbox_layout`: a library (no `main`) that becomes a powerbox program only
//! after translation — a separately linked runtime given an entry by `synth_manifest_child_start` —
//! keeps its globals clear of the argument area. Without it they start at the guarded `DATA_BASE`,
//! inside `[module_args_base(), module_args_end())`, and a detached spawn's payload (op 15) lands on
//! them.

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

/// The parent: spawns the child detached with a 16-byte payload `{35, 0}` and joins it.
fn parent(child_log2: u8) -> String {
    format!(
        r#"
declare i64 @__vm_instantiate_detached(i32, i64, i64, i64, i64, i64, i64, i64, i64, i64)
declare i64 @__vm_join(i32, i64)

define i64 @spawn(i32 %inst, i64 %budget, i64 %module) {{
  %buf = alloca [2 x i64]
  store i64 35, ptr %buf
  %hi = getelementptr i64, ptr %buf, i64 1
  store i64 0, ptr %hi
  %p = ptrtoint ptr %buf to i64
  %h = call i64 @__vm_instantiate_detached(i32 %inst, i64 %budget, i64 %module, i64 0, i64 0, i64 0, i64 {child_log2}, i64 1000000, i64 %p, i64 16)
  %r = call i64 @__vm_join(i32 %inst, i64 %h)
  ret i64 %r
}}
"#
    )
}

fn child(powerbox_layout: bool) -> temen_llvm::Translated {
    let opts = temen_llvm::TranslateOptions {
        powerbox_layout,
        ..Default::default()
    };
    temen_llvm::translate_ll_str_with_options(CHILD, opts).expect("translate the child")
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
    let inst = host.grant_instantiator(0, 1 << plog2);
    let module = host.grant_module(&image);
    let budget = host.grant_budget(-1, -1, -1);
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
fn it_moves_a_librarys_globals_above_the_argument_area() {
    assert!(
        g_addr(&child(false)) < temen_ir::module_args_end(),
        "the default layout this option exists for"
    );
    assert!(g_addr(&child(true)) >= temen_ir::module_args_end());
}

#[test]
fn a_detached_payload_leaves_the_globals_intact() {
    assert_eq!(spawn(child(true)), 35 + 7);
    assert_ne!(
        spawn(child(false)),
        35 + 7,
        "without it, the payload lands on @g"
    );
}
