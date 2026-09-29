//! **#1855 — a §14 confined spawn answers the same on every driver.** The spawn family (op 0, op 5,
//! op 13 and the §3d record op 17) had one admission per driver, and three of them still refused
//! what the cooperative driver serves: a budget-funded or grant-carrying same-module op-17 spawn
//! trapped `Malformed` on the parallel driver and the debug scheduler (the grant list on `Vcpu` too),
//! and every op-13 spawn trapped under the debugger. The debug scheduler also checked a nested
//! holder's carve against the root's NULL guard instead of its own.
//!
//! Each case runs on the oracle, the cooperative executor, the parallel driver, the debug scheduler
//! and an orchestrated `Vcpu`, and every one must give the oracle's result and stream bytes.

#[path = "support/drivers.rs"]
mod drivers;

use drivers::{agree_on_every_driver, Ran};
use temen_interp::{Host, StreamRole, Value};
use temen_ir::Module;
use temen_text::parse_module;

fn module(src: &str) -> Module {
    let m = parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

fn ok(v: i64) -> Ran {
    Ran {
        result: Ok(vec![Value::I64(v)]),
        stdout: Vec::new(),
        stderr: Vec::new(),
    }
}

/// Text-IR stores of the seven `i64` record fields `vf0..vf48` at `at` (the §3d 56-byte layout).
fn store_record(at: u64) -> String {
    ["vf0", "vf8", "vf16", "vf24", "vf32", "vf40", "vf48"]
        .iter()
        .enumerate()
        .map(|(i, f)| {
            format!(
                "  vra{i} = i64.const {}\n  i64.store vra{i} {f}\n",
                at + 8 * i as u64
            )
        })
        .collect()
}

/// Text-IR byte stores spelling `name` at `at`, with registers prefixed `p`.
fn store_name(p: &str, at: u64, name: &str) -> String {
    name.bytes()
        .enumerate()
        .map(|(i, b)| {
            let a = at + i as u64;
            format!("  {p}a{i} = i64.const {a}\n  {p}b{i} = i32.const {b}\n  i32.store8 {p}a{i} {p}b{i}\n")
        })
        .collect()
}

/// A 16-byte grant record `{name_off, name_len, handle, flags}` at `at`, naming the handle in
/// register `handle` by the name stored at `name_at`.
fn store_grant(p: &str, at: u64, name_at: u64, name: &str, handle: &str) -> String {
    format!(
        "  {p}r0 = i64.const {at}\n  {p}n0 = i32.const {name_at}\n  i32.store {p}r0 {p}n0\n\
         \x20 {p}r1 = i64.const {}\n  {p}n1 = i32.const {}\n  i32.store {p}r1 {p}n1\n\
         \x20 {p}r2 = i64.const {}\n  i32.store {p}r2 {handle}\n\
         \x20 {p}r3 = i64.const {}\n  {p}n3 = i32.const 0\n  i32.store {p}r3 {p}n3\n{}",
        at + 4,
        name.len(),
        at + 8,
        at + 12,
        store_name(p, name_at, name),
    )
}

/// The join tail every parent shares: a refused spawn (negative handle) returns its `-errno`, an
/// admitted one is joined and its result returned. Expects `vinst` and `vch` in block 0.
const JOIN_OR_ERRNO: &str = "\
  vz = i32.const 0
  vneg = i32.lt_s vch vz
  br_if vneg 1(vch) 2(vinst, vch)
}
block 1 (ve: i32) {
  vr = i64.extend_i32_s ve
  return vr
}
block 2 (vi2: i32, vch2: i32) {
  vr = call.cap 6 1 (i32) -> (i64) vi2 (vch2)
  return vr
  }
}
";

/// A same-module op-17 parent `(i32 inst, i64 budget) -> i64`: a record for func 1 in a 4 KiB carve
/// at 64 KiB, funded by `budget` (0 = none). Func 1 returns 42.
fn op17_same_module() -> String {
    let f0 = 1i64 << 32; // version 0 | entry 1
    let f16 = (12u64 | (0xFFFF_FFFFu64 << 32)) as i64; // size_log2 12 | no pager
    format!(
        "memory 17
func (i32, i64) -> (i64) {{
block 0 (vinst: i32, vbud: i64) {{
  vf0 = i64.const {f0}
  vf8 = i64.const 65536
  vf16 = i64.const {f16}
  vsh = i64.const 32
  vbs = i64.shl vbud vsh
  vself = i64.const 4294967295
  vf24 = i64.or vself vbs
  vf32 = i64.const 0
  vf40 = i64.const 0
  vf48 = i64.const 0
{rec}  vrp = i64.const 17536
  vch = call.cap 6 17 (i64) -> (i32) vinst (vrp)
{JOIN_OR_ERRNO}func (i64) -> (i64) {{
block 0 (v0: i64) {{
  vr = i64.const 42
  return vr
  }}
}}
",
        rec = store_record(17536),
    )
}

/// A same-module op-17 parent `(i32 inst, i32 out, i32 err) -> i64` that re-grants `stdout` and
/// `stderr` by name to func 1, which writes `O` and `E` through them and returns 7.
fn op17_same_module_granted() -> String {
    let f0 = 1i64 << 32;
    let f16 = (16u64 | (0xFFFF_FFFFu64 << 32)) as i64;
    format!(
        "memory 17
func (i32, i32, i32) -> (i64) {{
block 0 (vinst: i32, vout: i32, verr: i32) {{
{g0}{g1}  vf0 = i64.const {f0}
  vf8 = i64.const 65536
  vf16 = i64.const {f16}
  vf24 = i64.const 4294967295
  vf32 = i64.const 0
  vf40 = i64.const 16384
  vf48 = i64.const 2
{rec}  vrp = i64.const 17408
  vch = call.cap 6 17 (i64) -> (i32) vinst (vrp)
{JOIN_OR_ERRNO}func (i64) -> (i64) {{
block 0 (v0: i64) {{
{n0}  l6 = i64.const 6
  ho = i64.const 16384
  hout = self.resolve ho l6
{n1}  he = i64.const 16416
  herr = self.resolve he l6
  bo = i64.const 16400
  co = i32.const 79
  i32.store8 bo co
  be = i64.const 16424
  ce = i32.const 69
  i32.store8 be ce
  one = i64.const 1
  wo = call.cap 0 1 (i64, i64) -> (i64) hout (bo, one)
  we = call.cap 0 1 (i64, i64) -> (i64) herr (be, one)
  v7 = i64.const 7
  return v7
  }}
}}
",
        g0 = store_grant("g0", 16384, 16484, "stdout", "vout"),
        g1 = store_grant("g1", 16400, 16494, "stderr", "verr"),
        rec = store_record(17408),
        n0 = store_name("c0", 16384, "stdout"),
        n1 = store_name("c1", 16416, "stderr"),
    )
}

/// A separate child module (4 KiB window) whose entry returns 42.
const CHILD_42: &str = "memory 12
func (i64) -> (i64) {
block 0 (v0: i64) {
  vr = i64.const 42
  return vr
  }
}
";

/// A separate child module whose entry resolves `stdout` by name, writes `M` and returns 42.
fn child_writes_m() -> String {
    format!(
        "memory 12
func (i64) -> (i64) {{
block 0 (v0: i64) {{
{n}  vl = i64.const 6
  vp = i64.const 100
  vh = self.resolve vp vl
  vb = i64.const 200
  vc = i32.const 77
  i32.store8 vb vc
  one = i64.const 1
  vw = call.cap 0 1 (i64, i64) -> (i64) vh (vb, one)
  vr = i64.const 42
  return vr
  }}
}}
",
        n = store_name("c", 100, "stdout"),
    )
}

/// An op-13 parent `(i32 inst, i32 module, i32 out) -> i64` spawning `module`'s entry 0 into a 4 KiB
/// carve at 64 KiB, re-granting `stdout` by name iff `grant`. Op 13 always carries a grant list; with
/// `grant` false it is empty.
fn op13(grant: bool) -> String {
    format!(
        "memory 17
func (i32, i32, i32) -> (i64) {{
block 0 (vinst: i32, vmod: i32, vout: i32) {{
{g}  vm = i64.extend_i32_s vmod
  gp = i64.const 16384
  gn = i64.const {n}
  en = i64.const 0
  off = i64.const 65536
  sl = i64.const 12
  q = i64.const 0
  vch = call.cap 6 13 (i64, i64, i64, i64, i64, i64, i64) -> (i32) vinst (vm, gp, gn, en, off, sl, q)
{JOIN_OR_ERRNO}",
        g = store_grant("g", 16384, 16484, "stdout", "vout"),
        n = u8::from(grant),
    )
}

/// A module-form op-17 parent `(i32 inst, i32 module, i64 budget) -> i64`: `module`'s entry 0 in a
/// 4 KiB carve at 64 KiB, funded by `budget`.
fn op17_module_funded() -> String {
    let f16 = (12u64 | (0xFFFF_FFFFu64 << 32)) as i64;
    format!(
        "memory 17
func (i32, i32, i64) -> (i64) {{
block 0 (vinst: i32, vmod: i32, vbud: i64) {{
  vf0 = i64.const 0
  vf8 = i64.const 65536
  vf16 = i64.const {f16}
  vm = i64.extend_i32_u vmod
  vsh = i64.const 32
  vbs = i64.shl vbud vsh
  vf24 = i64.or vm vbs
  vf32 = i64.const 0
  vf40 = i64.const 0
  vf48 = i64.const 0
{rec}  vrp = i64.const 17536
  vch = call.cap 6 17 (i64) -> (i32) vinst (vrp)
{JOIN_OR_ERRNO}",
        rec = store_record(17536),
    )
}

/// Depth 2: the root spawns func 1 into an 8 KiB carve, and func 1 spawns func 2 at offset 0 of its
/// own window. An 8 KiB window is below the 16 KiB NULL guard, so it is unguarded and offset 0 is
/// a legal carve for it — the holder's guard decides, not the root's (#1094). Func 2 returns 42.
const NESTED_LOW_CARVE: &str = "memory 17
func (i32) -> (i64) {
block 0 (vinst: i32) {
  ve = i64.const 1
  vo = i64.const 65536
  vs = i64.const 13
  vq = i64.const 0
  vh = call.cap 6 0 (i64, i64, i64, i64) -> (i32) vinst (ve, vo, vs, vq)
  vr = call.cap 6 1 (i32) -> (i64) vinst (vh)
  return vr
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vinst = i32.wrap_i64 v0
  ve = i64.const 2
  vo = i64.const 0
  vs = i64.const 12
  vq = i64.const 0
  vch = call.cap 6 0 (i64, i64, i64, i64) -> (i32) vinst (ve, vo, vs, vq)
  vz = i32.const 0
  vneg = i32.lt_s vch vz
  br_if vneg 1(vch) 2(vinst, vch)
}
block 1 (ve2: i32) {
  vr = i64.extend_i32_s ve2
  return vr
}
block 2 (vi2: i32, vch2: i32) {
  vr = call.cap 6 1 (i32) -> (i64) vi2 (vch2)
  return vr
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vr = i64.const 42
  return vr
  }
}
";

#[test]
fn a_plain_same_module_record_spawn() {
    let m = module(&op17_same_module());
    let setup = || {
        let mut h = Host::new();
        let i = h.grant_instantiator(0, 1 << 17);
        (h, vec![Value::I32(i), Value::I64(0)])
    };
    agree_on_every_driver("op 17, plain", &m, &setup, &ok(42));
}

#[test]
fn a_budget_funded_same_module_record_spawn() {
    let m = module(&op17_same_module());
    let setup = || {
        let mut h = Host::new();
        let i = h.grant_instantiator(0, 1 << 17);
        let b = h.grant_budget(5_000_000, -1, -1);
        (h, vec![Value::I32(i), Value::I64(b as i64)])
    };
    agree_on_every_driver("op 17 + Budget", &m, &setup, &ok(42));
}

#[test]
fn a_grant_carrying_same_module_record_spawn() {
    let m = module(&op17_same_module_granted());
    let setup = || {
        let mut h = Host::new();
        let i = h.grant_instantiator(0, 1 << 17);
        let o = h.grant_stream(StreamRole::Out);
        let e = h.grant_stream(StreamRole::Err);
        (h, vec![Value::I32(i), Value::I32(o), Value::I32(e)])
    };
    let want = Ran {
        result: Ok(vec![Value::I64(7)]),
        stdout: b"O".to_vec(),
        stderr: b"E".to_vec(),
    };
    agree_on_every_driver("op 17 + named grants", &m, &setup, &want);
}

#[test]
fn a_grant_less_module_spawn() {
    let m = module(&op13(false));
    let child = module(CHILD_42);
    let setup = || {
        let mut h = Host::new();
        let i = h.grant_instantiator(0, 1 << 17);
        let c = h.grant_module(&child);
        let o = h.grant_stream(StreamRole::Out);
        (h, vec![Value::I32(i), Value::I32(c), Value::I32(o)])
    };
    agree_on_every_driver("op 13, no grants", &m, &setup, &ok(42));
}

#[test]
fn a_grant_carrying_module_spawn() {
    let m = module(&op13(true));
    let child = module(&child_writes_m());
    let setup = || {
        let mut h = Host::new();
        let i = h.grant_instantiator(0, 1 << 17);
        let c = h.grant_module(&child);
        let o = h.grant_stream(StreamRole::Out);
        (h, vec![Value::I32(i), Value::I32(c), Value::I32(o)])
    };
    let want = Ran {
        result: Ok(vec![Value::I64(42)]),
        stdout: b"M".to_vec(),
        stderr: Vec::new(),
    };
    agree_on_every_driver("op 13 + a named grant", &m, &setup, &want);
}

#[test]
fn a_budget_funded_module_record_spawn() {
    let m = module(&op17_module_funded());
    let child = module(CHILD_42);
    let setup = || {
        let mut h = Host::new();
        let i = h.grant_instantiator(0, 1 << 17);
        let c = h.grant_module(&child);
        let b = h.grant_budget(5_000_000, -1, -1);
        (h, vec![Value::I32(i), Value::I32(c), Value::I64(b as i64)])
    };
    agree_on_every_driver("op 17 module form + Budget", &m, &setup, &ok(42));
}

#[test]
fn a_nested_holder_carves_below_the_roots_guard() {
    let m = module(NESTED_LOW_CARVE);
    let setup = || {
        let mut h = Host::new();
        let i = h.grant_instantiator(0, 1 << 17);
        (h, vec![Value::I32(i)])
    };
    agree_on_every_driver("depth-2 carve at offset 0", &m, &setup, &ok(42));
}
