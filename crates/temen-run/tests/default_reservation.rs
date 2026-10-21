//! #2517 — the embedder chooses the reservation its process's windows take when what builds them
//! names none (`temen_ir::set_default_reserved_log2`), and every engine builds the detached windows
//! it chose: a child's starter `AddressSpace` maps up to that reservation, raised to the child's own
//! size, and refuses a map past it, on the JIT and on every interpreter driver alike. Under the
//! default 128 GiB, both maps past it would fit. Its own binary: the choice is the process's, and
//! its first read fixes it.

#[path = "support/detached.rs"]
mod detached;
#[path = "../../temen-interp/tests/support/drivers.rs"]
mod drivers;

use detached::{jit_on_reserved, module};
use drivers::{agree_on_every_driver, Ran};
use temen_interp::{Host, Value};
use temen_ir::{SpawnRec, DEFAULT_RESERVED_LOG2};
use temen_jit::JitOutcome;

/// The reservation chosen: 1 MiB.
const CHOSEN: u8 = 20;

/// The bytes each map commits.
const LEN: u64 = 1 << 14;

/// A root with a window of `2^size_log2` bytes that spawns its func 1's child image, whose window is
/// the same size, joins it and returns what it returned: 1 if a map of [`LEN`] bytes ending at
/// `2^limit` fits, plus 2 if one starting there is refused.
fn src(size_log2: u8, limit: u8) -> String {
    let end = 1u64 << limit;
    format!(
        "memory {size_log2}
func (i32, i32, i32) -> (i64) {{
block 0 (vinst: i32, vbud: i32, vmod: i32) {{
  rm = i64.const 17416
  i32.store rm vmod
  rb = i64.const 17420
  i32.store rb vbud
  rp = i64.const 17408
  vch = call.cap 6 17 (i64) -> (i32) vinst (rp)
  vr_st, vr = call.cap 6 1 (i32) -> (i64, i64) vinst (vch)
  return vr
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (vinst: i64, vas64: i64) {{
  vas = i32.wrap_i64 vas64
  under = i64.const {under}
  past = i64.const {end}
  n = i64.const {LEN}
  rw = i64.const 3
  z = i64.const 0
  r1 = call.cap 5 0 (i64, i64, i64) -> (i64) vas (under, n, rw)
  r2 = call.cap 5 0 (i64, i64, i64) -> (i64) vas (past, n, rw)
  b1 = i64.eq r1 z
  b2 = i64.lt_s r2 z
  w1 = i64.extend_i32_u b1
  w2 = i64.extend_i32_u b2
  two = i64.const 2
  x2 = i64.mul w2 two
  s = i64.add w1 x2
  return s
  }}
}}
{rec}",
        under = end - LEN,
        rec = detached::rec::segment(17408, &SpawnRec::default()),
    )
}

/// The root's powerbox: an `Instantiator`, an unbounded `Budget` and the child image of its func 1.
fn setup(m: &temen_ir::Module) -> impl Fn() -> (Host, Vec<Value>) + '_ {
    move || {
        let mut h = Host::new();
        let i = h.grant_instantiator();
        let b = h.grant_budget(-1, -1, -1);
        let c = h.grant_module(&temen_ir::child_image_at(m, 1).expect("child image"));
        (h, [i, b, c].map(Value::I32).to_vec())
    }
}

#[test]
fn every_engine_reserves_what_the_embedder_chose() {
    assert!(
        !temen_ir::set_default_reserved_log2(DEFAULT_RESERVED_LOG2 + 1),
        "the choice only lowers the default"
    );
    assert!(
        temen_ir::set_default_reserved_log2(CHOSEN),
        "chosen before any window"
    );
    assert!(
        !temen_ir::set_default_reserved_log2(CHOSEN - 1),
        "and only once"
    );
    assert_eq!(temen_ir::default_reserved_log2(), CHOSEN);
    let both = Ran {
        result: Ok(vec![Value::I64(3)]),
        stdout: Vec::new(),
        stderr: Vec::new(),
    };
    // A window smaller than the choice reserves the choice; a larger one reserves itself.
    for (size_log2, limit) in [(16, CHOSEN), (22, 22)] {
        let m = module(&src(size_log2, limit));
        let what = format!("a 2^{size_log2} child");
        agree_on_every_driver(&what, &m, &setup(&m), &both);
        let (mut host, args) = setup(&m)();
        let h = [0, 1, 2].map(|k| match args[k] {
            Value::I32(h) => h,
            other => panic!("a handle, not {other:?}"),
        });
        match jit_on_reserved(&m, &mut host, h, temen_ir::default_reserved_log2()) {
            JitOutcome::Returned(v) => assert_eq!(v, [3], "{what} on the JIT"),
            other => panic!("{what} on the JIT ended abnormally: {other:?}"),
        }
    }
}
