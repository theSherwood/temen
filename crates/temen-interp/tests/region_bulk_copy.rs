//! #2962 — `mem.copy` and `mem.fill` over a window with a §13 region mapped copy a page at a time on
//! the bytecode engine, to the region for an aliased page and to the window for the rest. A message
//! region makes that the common case for every caller of a buffer-taking interface; before it each
//! byte of such a copy, aliased or not, took the address-space lock and a page lookup. The tree-walk
//! oracle keeps its byte loop, so the drivers are compared against it.
//!
//! The root maps an 8 KiB region at 65536 and, across the plain pages on either side, fills, fills
//! again and copies spans that cross both boundaries, then returns a weighted checksum of
//! `[56000, 80000)`.

#[path = "support/drivers.rs"]
mod drivers;

use drivers::{agree_on_every_driver, Ran};
use temen_interp::{Host, Value};
use temen_ir::Module;

/// The spans the root writes: `(dst, val, len)` fills, then a `(dst, src, len)` copy.
const FILLS: [(usize, u8, usize); 2] = [(60000, 7, 20000), (64000, 9, 2000)];
const COPY: (usize, usize, usize) = (72000, 63000, 4000);
const SUM: (usize, usize) = (56000, 80000);

fn src() -> String {
    let fills: String = FILLS
        .iter()
        .enumerate()
        .map(|(i, (d, v, n))| {
            format!(
                "  fd{i} = i64.const {d}\n  fv{i} = i32.const {v}\n  fn{i} = i64.const {n}\n  mem.fill fd{i} fv{i} fn{i}\n"
            )
        })
        .collect();
    let (cd, cs, cn) = COPY;
    let (lo, hi) = SUM;
    format!(
        "memory 17
func (i32) -> (i64) {{
block 0 (vas: i32) {{
  vlen = i64.const 8192
  vrh = call.cap 5 5 (i64) -> (i64) vas (vlen)
  vr = i32.wrap_i64 vrh
  vwin = i64.const 65536
  vz = i64.const 0
  vprot = i32.const 3
  vm = call.cap 4 0 (i64, i64, i64, i32) -> (i64) vr (vwin, vz, vlen, vprot)
{fills}  cd = i64.const {cd}
  cs = i64.const {cs}
  cn = i64.const {cn}
  mem.copy cd cs cn
  vlo = i64.const {lo}
  br 1(vlo, vz, vm)
}}
block 1 (at: i64, sum: i64, vm: i64) {{
  vhi = i64.const {hi}
  done = i64.eq at vhi
  br_if done 3(sum, vm) 2(at, sum, vm)
}}
block 2 (at: i64, sum: i64, vm: i64) {{
  b = i64.load8_u at
  vone = i64.const 1
  w = i64.add at vone
  bw = i64.mul b w
  sum2 = i64.add sum bw
  at2 = i64.add at vone
  br 1(at2, sum2, vm)
}}
block 3 (sum: i64, vm: i64) {{
  k = i64.const 1000000000000
  t = i64.mul vm k
  r = i64.add sum t
  return r
  }}
}}
"
    )
}

/// The checksum a correct run returns (the map answering 0).
fn want() -> i64 {
    let mut mem = vec![0u8; 1 << 17];
    for (d, v, n) in FILLS {
        mem[d..d + n].fill(v);
    }
    let (cd, cs, cn) = COPY;
    mem.copy_within(cs..cs + cn, cd);
    (SUM.0..SUM.1).fold(0i64, |s, i| s.wrapping_add(mem[i] as i64 * (i as i64 + 1)))
}

#[test]
fn bulk_copies_across_an_aliased_page_agree_on_every_driver() {
    let m: Module = temen_text::parse_module(&src()).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let setup = || {
        let mut h = Host::new();
        let a = h.grant_address_space(0, 1 << 17);
        (h, vec![Value::I32(a)])
    };
    let want = Ran {
        result: Ok(vec![Value::I64(want())]),
        stdout: Vec::new(),
        stderr: Vec::new(),
    };
    agree_on_every_driver("region bulk copy", &m, &setup, &want);
}
