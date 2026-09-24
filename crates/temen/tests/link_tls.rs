//! **Thread-local templates through the linker** (#1715): each unit's `data tls` template is stacked
//! into one per-thread block, placed twice in the window (a read-only pristine image and the root
//! thread's writable block), and every `data.self tls` / `data.sym tls` reference resolves to an
//! offset within that block. A thread that copies the image into its own block and installs it in
//! `vcpu.tls` sees every unit's initial values, never another thread's writes.

use temen_interp::{
    bytecode, run_with_host, run_with_host_fast, Host, Inspector, IrPc, Stop, Value, VarValue,
};
use temen_ir::{
    link, LinkError, LinkUnit, Module, POWERBOX_STACK_ALIGN, TLS_END_SYM, TLS_IMAGE_SYM,
    TLS_ROOT_SYM,
};

fn unit(src: &str) -> LinkUnit {
    let module = temen_text::parse_module(src).expect("parse unit");
    LinkUnit {
        exports: module
            .exports
            .iter()
            .map(|e| (e.name.clone(), e.func))
            .collect(),
        data_exports: module.data_exports.clone(),
        module,
    }
}

/// Unit 0, a library: one exported thread-local, `t_lib = 5`, and no plain data.
const LIB: &str = r#"
memory 16
data tls 0 "\x05\x00\x00\x00\x00\x00\x00\x00"
export 0 data "t_lib" tls 0
"#;

/// Unit 1, the program: its own thread-locals `t_own = 7` (offset 0) and `t_ptr` (offset 8, whose
/// initializer is `&g`, patched by `data.ptr tls`), plus a plain global `g = 42`. `main` writes 99
/// into the root thread's `t_own`, then runs a child thread that builds its block from the pristine
/// image and reads all three through `vcpu.tls`: `7*10000 + 5*100 + 42`. `main` returns the child's
/// result times 1000 plus its own `t_own`.
const PROG: &str = r#"
memory 16
data 16384 "\x2a\x00\x00\x00\x00\x00\x00\x00"
data tls 0 "\x07\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
data.ptr tls 8 self 16384
func () -> (i64) {
block 0 () {
  root = data.sym "__tls_root" 0
  own = data.self tls 0
  a = i64.add root own
  v99 = i64.const 99
  i64.store a v99
  blk = data.top
  k4096 = i64.const 4096
  sp = i64.add blk k4096
  t = thread.spawn 1 sp blk
  jr = thread.join t
  mine = i64.load a
  k1000 = i64.const 1000
  hi = i64.mul jr k1000
  r = i64.add hi mine
  return r
  }
}
func (i64, i64) -> (i64) {
block 0 (sp: i64, blk: i64) {
  img = data.sym "__tls_image" 0
  end = data.sym "__tls_end" 0
  n = i64.sub end img
  mem.copy blk img n
  vcpu.tls.set blk
  b = vcpu.tls.get
  own = data.self tls 0
  ao = i64.add b own
  x = i64.load ao
  lib = data.sym tls "t_lib" 0
  al = i64.add b lib
  y = i64.load al
  pt = data.self tls 8
  ap = i64.add b pt
  p = i64.load ap
  z = i64.load p
  k10000 = i64.const 10000
  k100 = i64.const 100
  xs = i64.mul x k10000
  ys = i64.mul y k100
  s = i64.add xs ys
  r = i64.add s z
  return r
  }
}
debug.file 0 "prog.c"
debug.var global "t_own" tls 0 "long"
"#;

fn linked() -> Module {
    let m = link(&[unit(LIB), unit(PROG)]).expect("link");
    temen_verify::verify_module(&m).expect("verify the linked program");
    m
}

#[test]
fn each_thread_sees_every_units_initial_values_not_another_threads_writes() {
    let m = linked();
    let want = Ok(vec![Value::I64(70542 * 1000 + 99)]);
    let mut fuel = u64::MAX;
    assert_eq!(
        run_with_host(&m, 0, &[], &mut fuel, &mut Host::new()),
        want,
        "tree-walker"
    );
    let mut fuel = u64::MAX;
    assert_eq!(
        run_with_host_fast(&m, 0, &[], &mut fuel, &mut Host::new()),
        want,
        "bytecode engine"
    );
}

/// The window address a `data.sym "<name>" 0` resolved to: the linker rewrites each to an
/// `i64.const`, so find the program's `main` (function 0 of unit 1 is function 0 here — the library
/// has no functions) and read the constants in order.
fn main_consts(m: &Module) -> Vec<i64> {
    m.funcs[0].blocks[0]
        .insts
        .iter()
        .filter_map(|i| match i {
            temen_ir::Inst::ConstI64(c) => Some(*c),
            _ => None,
        })
        .collect()
}

#[test]
fn the_templates_stack_into_one_block_placed_as_a_read_only_image_and_a_root_block() {
    let m = linked();
    // Unit 0's template is 8 bytes at block offset 0; unit 1's starts at the next 16-byte boundary.
    let (tls_lib, tls_prog, size) = (0u64, 16u64, 32u64);
    // `main`'s first constants: `__tls_root`, then `data.self tls 0` (unit 1's base + 0).
    let consts = main_consts(&m);
    let root = consts[0] as u64;
    assert_eq!(
        consts[1] as u64, tls_prog,
        "unit 1's own thread-local offset"
    );
    let seg = |addr: u64| m.data.iter().find(|d| d.offset == addr);
    // Each copy starts on its own host page, the image read-only and the root block writable.
    assert_eq!(root % POWERBOX_STACK_ALIGN, 0);
    let image = m
        .data
        .iter()
        .filter(|d| d.readonly)
        .map(|d| d.offset)
        .min()
        .expect("a read-only image");
    assert_eq!(image % POWERBOX_STACK_ALIGN, 0);
    assert!(
        image + size <= root,
        "the image and the root block do not overlap"
    );
    for base in [image, root] {
        let lib = seg(base + tls_lib).expect("unit 0's template");
        assert_eq!(lib.bytes, 5u64.to_le_bytes(), "t_lib's initializer");
        let prog = seg(base + tls_prog).expect("unit 1's template");
        assert_eq!(prog.bytes[..8], 7u64.to_le_bytes(), "t_own's initializer");
        // `data.ptr tls 8 self 16384`: unit 1's data base is window 0 (unit 0 has no data).
        assert_eq!(
            prog.bytes[8..],
            16384u64.to_le_bytes(),
            "t_ptr = &g, in both copies"
        );
        assert_eq!(lib.readonly, base == image);
    }
    assert!(m.tls.is_empty(), "a linked module carries no template");
    // Only plain data is exported from the linked module: a thread-local has no one address.
    assert!(m.data_exports.iter().all(|e| !e.tls && e.name != "t_lib"));
}

#[test]
fn a_program_with_no_thread_locals_links_exactly_as_before() {
    const PLAIN: &str = r#"
memory 16
data 16384 "\x01\x00\x00\x00\x00\x00\x00\x00"
func () -> (i64) {
block 0 () {
  a = data.sym "__tls_image" 0
  b = data.sym "__tls_end" 0
  c = data.sym "__tls_root" 0
  n = i64.sub b a
  return n
  }
}
"#;
    let m = link(&[unit(PLAIN)]).expect("link");
    assert_eq!(m.data.len(), 1, "no template copies are placed");
    let consts = main_consts(&m);
    assert_eq!(consts[0], consts[1], "an empty image");
    assert_eq!(consts[0], consts[2], "the three symbols name one address");
    let mut fuel = u64::MAX;
    assert_eq!(
        run_with_host(&m, 0, &[], &mut fuel, &mut Host::new()),
        Ok(vec![Value::I64(0)]),
        "a block size of 0"
    );
}

#[test]
fn a_thread_local_and_a_plain_reference_to_the_same_name_do_not_link() {
    // A plain `data.sym` naming a thread-local export…
    let plain_ref = unit(
        r#"
func () -> (i64) {
block 0 () {
  a = data.sym "t_lib" 0
  return a
  }
}
"#,
    );
    assert_eq!(
        link(&[unit(LIB), plain_ref]).err(),
        Some(LinkError::TlsMismatch("t_lib".into()))
    );
    // …and a `data.sym tls` naming a plain export.
    let lib = unit(
        r#"
memory 16
data 16384 "\x01\x00\x00\x00"
export 0 data "g" 16384
"#,
    );
    let tls_ref = unit(
        r#"
func () -> (i64) {
block 0 () {
  a = data.sym tls "g" 0
  return a
  }
}
"#,
    );
    assert_eq!(
        link(&[lib, tls_ref]).err(),
        Some(LinkError::TlsMismatch("g".into()))
    );
}

#[test]
fn the_linker_owns_the_three_thread_local_symbol_names() {
    for name in [TLS_IMAGE_SYM, TLS_END_SYM, TLS_ROOT_SYM] {
        let src = format!("memory 16\ndata 16384 \"\\x01\"\nexport 0 data \"{name}\" 16384\n");
        assert_eq!(
            link(&[unit(&src)]).err(),
            Some(LinkError::DuplicateSymbol(name.into())),
            "a unit may not define {name}"
        );
    }
    // Two units exporting the same thread-local collide like any other symbol.
    assert_eq!(
        link(&[unit(LIB), unit(LIB)]).err(),
        Some(LinkError::DuplicateSymbol("t_lib".into()))
    );
}

#[test]
fn an_unlinked_template_does_not_verify() {
    let m = temen_text::parse_module("memory 16\ndata tls 0 \"\\x01\"\n").expect("parse");
    assert_eq!(
        temen_verify::verify_module(&m),
        Err(temen_verify::VerifyError::UnlinkedTls)
    );
}

#[test]
fn the_thread_local_forms_survive_the_text_and_binary_waists() {
    let m = temen_text::parse_module(PROG).expect("parse");
    let lib = temen_text::parse_module(LIB).expect("parse");
    for u in [m, lib] {
        let text = temen_text::print_module(&u);
        assert_eq!(
            temen_text::parse_module(&text).expect("reparse"),
            u,
            "{text}"
        );
        let bytes = temen_encode::encode_unit(&u);
        assert_eq!(temen_encode::decode_unit(&bytes).expect("decode"), u);
    }
}

/// The debugger reads a thread-local **per thread**: stopped in the child just after it installs its
/// block, `t_own` is the child's 7; switched to the root thread (parked in `thread.join`), it is the
/// 99 `main` wrote into the root block. Both engines, through the linked module's `tls_root`.
#[test]
fn the_debugger_shows_each_threads_own_copy() {
    let m = linked();
    let di = m.debug_info.as_ref().expect("merged debug info");
    assert!(di.tls_root.is_some(), "the linker records the root block");
    // Child (func 1), block 0, inst 5: `b = vcpu.tls.get`, just after `vcpu.tls.set blk`.
    let bp = IrPc {
        module: 0,
        func: 1,
        block: 0,
        inst: 5,
    };
    let val = |v: Option<VarValue>| match v {
        Some(VarValue::Bytes(b)) => i64::from_le_bytes(b.try_into().expect("8 bytes")),
        other => panic!("expected window bytes, got {other:?}"),
    };

    let mut insp = Inspector::attach_scheduled(&m, 0, &[], u64::MAX, vec![]);
    insp.set_breakpoint(bp);
    assert!(matches!(insp.run_until_stop(), Stop::Break { pc, .. } if pc == bp));
    assert_eq!(
        val(insp.read_var(0, "t_own", 8)),
        7,
        "tree-walker: the child's copy"
    );
    let child = insp.stopped_task();
    let root = insp.threads().into_iter().find(|&t| Some(t) != child);
    assert!(insp.select_task(root.expect("the root thread is live")));
    assert_eq!(
        val(insp.read_var(0, "t_own", 8)),
        99,
        "tree-walker: the root thread's copy"
    );

    let mut dbg = bytecode::ScheduledDebugRun::new(&m, 0, &[]).expect("bytecode debug session");
    dbg.set_breakpoints(vec![bp]);
    let mut fuel = u64::MAX;
    assert!(matches!(
        dbg.run_until_stop(&mut fuel),
        bytecode::SchedStop::Break { pc, .. } if pc == bp
    ));
    assert_eq!(
        val(dbg.read_var(0, "t_own", 8)),
        7,
        "bytecode: the child's copy"
    );
    let child = dbg.stopped_task();
    let root = dbg.threads().into_iter().find(|&t| Some(t) != child);
    assert!(dbg.select_task(root.expect("the root thread is live")));
    assert_eq!(
        val(dbg.read_var(0, "t_own", 8)),
        99,
        "bytecode: the root thread's copy"
    );
}
