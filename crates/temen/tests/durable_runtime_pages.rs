//! #2426 — a durable module's runtime keeps words in its window that the host reads and writes
//! outside the guest's guarded code, all in the shadow arena the module declared (#2518): its control
//! words (a freeze request rings the freeze word) and its regions (a freeze flattens each parked fiber
//! into its region). A guest that re-protected one of their pages made the host fault: a SIGSEGV in
//! the embedder. So every engine refuses a page op over those pages, as it refuses the NULL guard
//! (`temen_ir::durable_abi::ShadowArena::fences`).
#![cfg(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
))]

use std::time::Duration;
use temen_durable::{init_durable_window, transform_module};
use temen_interp::{bytecode, host_page_size, Host, MemLayout, Value};
use temen_ir::durable_abi::ShadowArena;
use temen_ir::errno::EINVAL;
use temen_jit::JitOutcome;

/// The arena the guests here declare: page-aligned on every host page size, past a gap above the
/// NULL guard.
const ARENA: ShadowArena = ShadowArena::new(1 << 17, 1 << 18);

/// `memory 19` with `ARENA` declared, or none.
fn memory(arena: bool) -> String {
    match arena {
        true => format!("memory 19 shadow {} {}\n", ARENA.base, ARENA.end),
        false => "memory 19\n".to_string(),
    }
}

/// The page ops below, as `(op, offset, len, prot)`, on a host of `page` bytes: three over the
/// arena's first page, which holds its control words; one over the first page above the NULL guard,
/// where the control words sat until #2518 and which is the guest's now; two more over the arena (its
/// last page, and one straddling into it from below); and one just past it.
fn ops(page: u64) -> [(u32, u64, u64, i64); 7] {
    let words = ARENA.freeze_word().expect("the arena's control words") / page * page;
    let above_guard = temen_ir::module_null_guard().next_multiple_of(page);
    [
        (2, words, page, 1),
        (0, words, page, 3),
        (1, words, page, 0),
        (2, above_guard, page, 1),
        (2, ARENA.end - page, page, 1),
        (2, ARENA.base - page, 2 * page, 1),
        (2, ARENA.end, page, 1),
    ]
}

/// A guest that makes each of [`ops`] through its `AddressSpace` (`v0`) and returns a mask: bit `i`
/// set when op `i` was refused with `-EINVAL`, and `1 << 8` when another op answered anything but 0.
fn page_ops(arena: bool) -> temen_ir::Module {
    let mut src =
        memory(arena) + "func (i32) -> (i64) {\nblock 0 (v0: i32) {\n  vm0 = i64.const 0\n";
    for (i, (op, off, len, prot)) in ops(host_page_size()).into_iter().enumerate() {
        let sig = match op {
            1 => format!("(i64, i64) -> (i64) v0 (vo{i}, vl{i})"),
            _ => format!("(i64, i64, i64) -> (i64) v0 (vo{i}, vl{i}, vp{i})"),
        };
        let bit = 1i64 << i;
        src += &format!(
            "  vo{i} = i64.const {off}\n  vl{i} = i64.const {len}\n  vp{i} = i64.const {prot}\n  \
             vr{i} = call.cap 5 {op} {sig}\n  ve{i} = i64.const {EINVAL}\n  \
             vq{i} = i64.eq vr{i} ve{i}\n  vz{i} = i64.eqz vr{i}\n  vok{i} = i32.or vq{i} vz{i}\n  \
             vb{i} = i64.const {bit}\n  vn{i} = i64.const 0\n  vbe{i} = i64.const 256\n  \
             vs{i} = select vq{i} vb{i} vn{i}\n  vbad{i} = select vok{i} vn{i} vbe{i}\n  \
             vmi{i} = i64.or vm{i} vs{i}\n  vm{j} = i64.or vmi{i} vbad{i}\n",
            j = i + 1,
        );
    }
    src += &format!("  return vm{}\n  }}\n}}\n", ops(1).len());
    let m = temen_text::parse_module(&src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// `m`'s entry on the tree-walker, the bytecode engine and the Cranelift JIT, given its
/// `AddressSpace`: what it returned on each.
fn on_every_engine(m: &temen_ir::Module) -> [Option<i64>; 3] {
    let one = |r: Result<Vec<Value>, temen_interp::Trap>| match r.as_deref() {
        Ok([Value::I64(x)]) => *x,
        other => panic!("the guest ended {other:?}"),
    };
    let fresh = || {
        let mut host = Host::new();
        let space = host.grant_memory();
        (host, space)
    };
    let (mut h, space) = fresh();
    let mut fuel = u64::MAX / 2;
    let tree_walker = one(temen_interp::run_with_host(
        m,
        0,
        &[Value::I32(space)],
        &mut fuel,
        &mut h,
    ));
    let (mut h, space) = fresh();
    let mut fuel = u64::MAX / 2;
    let bytecode =
        bytecode::compile_and_run_with_host(m, 0, &[Value::I32(space)], &mut fuel, &mut h)
            .map(one)
            .expect("the bytecode engine runs it");
    let (mut h, space) = fresh();
    let window = MemLayout::image(vec![0; 1 << 19]);
    let reserved = temen_ir::DEFAULT_RESERVED_LOG2;
    let jit =
        match temen_run::jit_cap_run(m, 0, &[space as i64], &window, reserved, 0, &mut h, None)
            .map(|(outcome, _)| outcome)
        {
            Ok(JitOutcome::Returned(v)) => Some(v[0]),
            Err(temen_jit::JitError::Unsupported(_)) => None,
            other => panic!("the JIT run ended {other:?}"),
        };
    [Some(tree_walker), Some(bytecode), jit]
}

/// A module that declares an arena has its arena fenced, the page holding its control words with
/// the rest, on every engine; a page below it, or past it, takes page ops as before, the one above
/// the NULL guard included (#2518). One that declares none fences nothing past the NULL guard.
#[test]
fn every_engine_refuses_page_ops_over_a_durable_modules_runtime_pages() {
    let fenced = 0b0011_0111;
    for (arena, want) in [(true, fenced), (false, 0)] {
        for (engine, got) in ["tree-walker", "bytecode", "jit"]
            .into_iter()
            .zip(on_every_engine(&page_ops(arena)))
        {
            if let Some(got) = got {
                assert_eq!(got, want, "{engine}, arena declared: {arena}");
            }
        }
    }
}

/// A durable guest that, through its `AddressSpace` (`v0`), protects `[at, at + page)` to `prot`
/// once its fiber (context 1, whose region starts at `ARENA.region_base(1)`) is parked, then polls
/// in a long loop a freeze cuts.
fn guest(at: u64, prot: i64) -> temen_ir::Module {
    let src = format!(
        "{}func (i32) -> (i64) {{
block 0 (v0: i32) {{
  vf = ref.func 1
  vsp = i64.const 400000
  vk = cont.new vf vsp
  vz = i64.const 0
  vs, vx = cont.resume vk vz
  va = i64.const {at}
  vl = i64.const {page}
  vp = i64.const {prot}
  vr = call.cap 5 2 (i64, i64, i64) -> (i64) v0 (va, vl, vp)
  br 1(vz, vr)
}}
block 1 (vi: i64, vr1: i64) {{
  vn = i64.const 3000000000
  vc = i64.lt_s vi vn
  br_if vc 2(vi, vr1) 3(vr1)
}}
block 2 (vj: i64, vr2: i64) {{
  vo = i64.const 1
  vk2 = i64.add vj vo
  br 1(vk2, vr2)
}}
block 3 (vx3: i64) {{
  return vx3
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (v0: i64, v1: i64) {{
  br 1(v1)
}}
block 1 (va: i64) {{
  vn = suspend va
  br 1(vn)
  }}
}}
",
        memory(true),
        page = host_page_size(),
    );
    let m = temen_text::parse_module(&src).expect("parse");
    let m = transform_module(&m).expect("transform");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// Run `m` durably on the JIT while the embedder requests a freeze from another thread, and return
/// how it ended: before #2426 the host faulted on the page the guest re-protected, killing this
/// process with SIGSEGV.
fn freeze_on_the_jit(m: &temen_ir::Module) -> Option<JitOutcome> {
    let mut host = Host::new();
    host.set_durable(true);
    host.grant_freeze_authority();
    let space = host.grant_memory();
    let win = init_durable_window(1 << 19, ARENA);
    let fc = temen_jit::FreezeController::new();
    let ctl = {
        let fc = fc.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            fc.request_freeze();
        })
    };
    let r = temen_run::jit_cap_run(
        m,
        0,
        &[space as i64],
        &MemLayout::image(win),
        19,
        0,
        &mut host,
        Some(fc),
    );
    ctl.join().expect("controller");
    match r {
        Ok((outcome, _)) => Some(outcome),
        Err(temen_jit::JitError::Unsupported(_)) => None,
        Err(e) => panic!("the JIT run failed: {e:?}"),
    }
}

/// The freeze request rings the freeze word: it reaches a guest that tried to make its page
/// read-only, which the guest's own run freezes under.
#[test]
fn a_freeze_rings_the_freeze_word_of_a_guest_that_tried_to_protect_it() {
    let page = host_page_size();
    let words = ARENA.freeze_word().expect("the arena's control words") / page * page;
    let outcome = freeze_on_the_jit(&guest(words, 1));
    assert!(
        matches!(outcome, None | Some(JitOutcome::Returned(_))),
        "{outcome:?}"
    );
}

/// The freeze flattens the parked fiber into its region: it reaches a guest that tried to make that
/// region inaccessible.
#[test]
fn a_freeze_flattens_a_fiber_whose_region_the_guest_tried_to_close() {
    let outcome = freeze_on_the_jit(&guest(ARENA.region_base(1), 0));
    assert!(
        matches!(outcome, None | Some(JitOutcome::Returned(_))),
        "{outcome:?}"
    );
}
