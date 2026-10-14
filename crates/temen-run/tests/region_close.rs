//! **#2358 — `SharedRegion.close`** (op 5): a domain gives a region handle up, and with it its hold
//! on the region, which goes when no domain holds it. The slot closes as a `Stream`'s does, so the
//! handle answers `-EBADF` from then on, and a region the window still maps refuses with `-EBUSY`.
//!
//! Every case runs on the tree-walk oracle, every bytecode driver ([`drivers::ALL`]) and the
//! Cranelift JIT, and each must give the oracle's answer. The op is one arm of the shared dispatch;
//! what differs by engine is the window that answers whether it still maps the region.

use std::sync::Arc;

use temen_interp::{Host, MemLayout, Trap, Value};
use temen_ir::Module;
use temen_jit::JitOutcome;

#[path = "../../temen-interp/tests/support/drivers.rs"]
mod drivers;

use drivers::{agree_on_every_driver, Ran};

/// The window: 256 KiB, the region mapped at 64 KiB.
const WINDOW_LOG2: u32 = 18;

fn module(src: &str) -> Module {
    let m = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// The powerbox: an `AddressSpace` over the whole window, which mints the regions. Regions are OS
/// shared memory, so the JIT aliases them for real.
fn setup() -> (Host, Vec<Value>) {
    let mut h = Host::new();
    h.set_region_factory(temen_run::new_shared_region);
    let a = h.grant_address_space(0, 1 << WINDOW_LOG2);
    (h, vec![Value::I32(a)])
}

/// `m` on the Cranelift JIT, over the powerbox [`setup`] builds, on a thread of its own: a run that
/// hangs fails here instead of stalling the suite.
fn on_the_jit(m: &Module) -> Result<Vec<Value>, Trap> {
    let m = Arc::new(m.clone());
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let (mut host, args) = setup();
        let args: Vec<i64> = args
            .iter()
            .map(|v| match v {
                Value::I32(x) => i64::from(*x),
                other => panic!("unexpected arg {other:?}"),
            })
            .collect();
        let window = MemLayout::image(vec![0u8; 1 << WINDOW_LOG2]);
        let (jo, _) = temen_run::jit_cap_run(&m, 0, &args, &window, 0, 0, &mut host, None)
            .expect("the JIT compiles it");
        let _ = tx.send(jo);
    });
    match rx
        .recv_timeout(std::time::Duration::from_secs(60))
        .expect("the JIT run hung")
    {
        JitOutcome::Returned(v) => Ok(v.into_iter().map(Value::I64).collect()),
        JitOutcome::Trapped(k) => Err(Trap::from_code(k as i64).expect("a trap code")),
        other => panic!("the JIT run ended as {other:?}"),
    }
}

/// `m` returns `want` on the oracle, every bytecode driver and the Cranelift JIT.
fn agree_everywhere(what: &str, m: &Module, want: i64) {
    let want = Ran {
        result: Ok(vec![Value::I64(want)]),
        stdout: Vec::new(),
        stderr: Vec::new(),
    };
    agree_on_every_driver(what, m, &setup, &want);
    assert_eq!(on_the_jit(m), want.result, "{what}: the Cranelift JIT");
}

/// Text IR that folds `vals` (each in `-100..900`) into `dst` as three decimal digits apiece, each
/// value plus 100, so one returned `i64` carries every answer and a wrong one shows where.
fn pack(dst: &str, vals: &[&str]) -> String {
    let mut s = String::from("  pk = i64.const 1000\n  pc = i64.const 100\n  acc0 = i64.const 0\n");
    for (i, v) in vals.iter().enumerate() {
        let next = match i + 1 == vals.len() {
            true => dst.to_string(),
            false => format!("acc{}", i + 1),
        };
        s += &format!(
            "  sh{i} = i64.mul acc{i} pk\n  of{i} = i64.add {v} pc\n  {next} = i64.add sh{i} of{i}\n"
        );
    }
    s
}

/// The answers [`pack`] folds `vals` into.
fn packed(vals: &[i64]) -> i64 {
    vals.iter().fold(0, |acc, v| acc * 1000 + v + 100)
}

/// **A domain that closes what it mints mints past its table.** 300 times it mints a 64 KiB region,
/// maps it, stores the iteration's number through it and loads it back, unmaps it and closes it.
/// Without the close, the table's 256 slots run out partway (`-EMFILE`). It returns
/// `iterations * 1_000_000 + Σ loaded`, the sum taking every map, unmap and close result too (each 0),
/// or `errno * 1000 - iteration` at the first mint refused.
#[test]
fn a_domain_that_closes_what_it_mints_mints_past_its_table() {
    let m = module(
        "memory 18
func (i32) -> (i64) {
block 0 (vas: i32) {
  vn0 = i64.const 0
  vs0 = i64.const 0
  br 1(vas, vn0, vs0)
}
block 1 (va: i32, vn: i64, vs: i64) {
  vlen = i64.const 65536
  vh = call.cap 5 5 (i64) -> (i64) va (vlen)
  vz = i64.const 0
  vneg = i64.lt_s vh vz
  br_if vneg 3(vh, vn) 2(va, vh, vn, vs)
}
block 2 (va2: i32, vh2: i64, vn2: i64, vs2: i64) {
  vr = i32.wrap_i64 vh2
  vat = i64.const 65536
  vroff = i64.const 0
  vlen2 = i64.const 65536
  vprot = i32.const 3
  vm = call.cap 4 0 (i64, i64, i64, i32) -> (i64) vr (vat, vroff, vlen2, vprot)
  i64.store vat vn2
  vld = i64.load vat
  vu = call.cap 4 1 (i64, i64) -> (i64) vr (vat, vlen2)
  vc = call.cap 4 5 () -> (i64) vr ()
  vs3 = i64.add vs2 vld
  vs4 = i64.add vs3 vm
  vs5 = i64.add vs4 vu
  vs6 = i64.add vs5 vc
  vone = i64.const 1
  vn3 = i64.add vn2 vone
  vmax = i64.const 300
  vdone = i64.ge_s vn3 vmax
  br_if vdone 4(vn3, vs6) 1(va2, vn3, vs6)
}
block 3 (verr: i64, vn4: i64) {
  vk = i64.const 1000
  vx = i64.mul verr vk
  vr2 = i64.sub vx vn4
  return vr2
}
block 4 (vn5: i64, vs7: i64) {
  vmil = i64.const 1000000
  vx2 = i64.mul vn5 vmil
  vr3 = i64.add vx2 vs7
  return vr3
}
}
",
    );
    agree_everywhere(
        "300 regions minted, used and closed",
        &m,
        300 * 1_000_000 + (0..300).sum::<i64>(),
    );
}

/// **A mapped region refuses to close; a closed one answers `-EBADF`.** The guest mints a region and
/// maps it, then closes it (`-EBUSY`: still mapped), unmaps it through the same handle (`0`: the
/// refused close changed nothing), closes it (`0`), closes it again (`-EBADF`) and maps it again
/// (`-EBADF`).
#[test]
fn a_mapped_region_refuses_to_close_and_a_closed_one_answers_ebadf() {
    let m = module(&format!(
        "memory 18
func (i32) -> (i64) {{
block 0 (va: i32) {{
  vlen = i64.const 65536
  vh = call.cap 5 5 (i64) -> (i64) va (vlen)
  vr = i32.wrap_i64 vh
  vat = i64.const 65536
  vroff = i64.const 0
  vprot = i32.const 3
  vm = call.cap 4 0 (i64, i64, i64, i32) -> (i64) vr (vat, vroff, vlen, vprot)
  vbusy = call.cap 4 5 () -> (i64) vr ()
  vu = call.cap 4 1 (i64, i64) -> (i64) vr (vat, vlen)
  vc = call.cap 4 5 () -> (i64) vr ()
  vagain = call.cap 4 5 () -> (i64) vr ()
  vm2 = call.cap 4 0 (i64, i64, i64, i32) -> (i64) vr (vat, vroff, vlen, vprot)
{}  return vp
}}
}}
",
        pack("vp", &["vm", "vbusy", "vu", "vc", "vagain", "vm2"])
    ));
    let (ebusy, ebadf) = (temen_ir::errno::EBUSY, temen_ir::errno::EBADF);
    agree_everywhere(
        "close while mapped, then unmapped, then again",
        &m,
        packed(&[0, ebusy, 0, 0, ebadf, ebadf]),
    );
}
