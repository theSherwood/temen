//! **#1855 — a spawn answers the same on every driver.** The §14 confined family (op 0, op 5, op 13
//! and the §3d record op 17) had one admission per driver, and three of them still refused what the
//! cooperative driver serves: a budget-funded or grant-carrying same-module op-17 spawn trapped
//! `Malformed` on the parallel driver and the debug scheduler (the grant list on `Vcpu` too), and
//! every op-13 spawn trapped under the debugger. The debug scheduler also checked a nested holder's
//! carve against the root's NULL guard instead of its own. The §5 detached spawn (op 15) had a
//! second admission on `Vcpu`, whose host built the child's powerbox and seeded its window (#1414).
//!
//! Each case runs on the oracle, the cooperative executor, the parallel driver, the debug scheduler
//! and an orchestrated `Vcpu`, and every one must give the oracle's result and stream bytes.

#[path = "support/drivers.rs"]
mod drivers;
#[path = "support/rec.rs"]
mod rec;

use drivers::{agree_on, agree_on_every_driver, run_on, run_on_then, Driver, Ran, SCHEDULING};
use temen_interp::{cap_id, Attestation, Host, StreamRole, Trap, Value};
use temen_ir::{Module, SpawnRec};
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

/// A same-module op-17 parent `(i32 inst, i32 out, i32 err, i32 budget) -> i64` that re-grants
/// `stdout` and `stderr` by name to func 1, spawned detached by the v1 record at 17408 and paid from
/// `budget`. Func 1 resolves the two names from the module's data, writes `O` and `E` through them and
/// returns 7.
fn op17_same_module_granted() -> String {
    let child = SpawnRec {
        grants_ptr: 17600,
        grants_n: 2,
        ..SpawnRec::v1(1)
    };
    format!(
        "memory 17
data 17700 \"stdout\"
data 17710 \"stderr\"
func (i32, i32, i32, i32) -> (i64) {{
block 0 (vinst: i32, vout: i32, verr: i32, vbud: i32) {{
{g0}{g1}  vbf = i64.const 17436
  i32.store vbf vbud
  vrp = i64.const 17408
  vch = call.cap 6 17 (i64) -> (i32) vinst (vrp)
{JOIN_OR_ERRNO}func (i64) -> (i64) {{
block 0 (v0: i64) {{
  l6 = i64.const 6
  ho = i64.const 17700
  hout = self.resolve ho l6
  he = i64.const 17710
  herr = self.resolve he l6
  bo = i64.const 17720
  co = i32.const 79
  i32.store8 bo co
  be = i64.const 17721
  ce = i32.const 69
  i32.store8 be ce
  one = i64.const 1
  wo = call.cap 0 1 (i64, i64) -> (i64) hout (bo, one)
  we = call.cap 0 1 (i64, i64) -> (i64) herr (be, one)
  v7 = i64.const 7
  return v7
  }}
}}
{rec}",
        g0 = store_grant("g0", 17600, 17700, "stdout", "vout"),
        g1 = store_grant("g1", 17616, 17710, "stderr", "verr"),
        rec = rec::segment(17408, &child),
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

/// A separate child module that imports `exit` — a required slot the spawn must bind or refuse.
const CHILD_IMPORTS_EXIT: &str = "memory 12
import 0 \"exit\" (i32) -> ()
func (i64) -> (i64) {
block 0 (v0: i64) {
  vr = i64.const 42
  return vr
  }
}
";

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

/// A module-form op-17 parent `(i32 inst, i32 module, i64 budget) -> i64`: `module`'s entry 0 in its
/// declared window, spawned detached by the v1 record at 17536 and paid from `budget`.
fn op17_module_funded() -> String {
    format!(
        "memory 17
func (i32, i32, i64) -> (i64) {{
block 0 (vinst: i32, vmod: i32, vbud: i64) {{
  vma = i64.const 17560
  i32.store vma vmod
  vb = i32.wrap_i64 vbud
  vba = i64.const 17564
  i32.store vba vb
  vrp = i64.const 17536
  vch = call.cap 6 17 (i64) -> (i32) vinst (vrp)
{JOIN_OR_ERRNO}{rec}",
        rec = rec::segment(17536, &SpawnRec::v1(0)),
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
        h.set_self_module(&std::sync::Arc::new(m.clone()));
        let i = h.grant_instantiator(0, 1 << 17);
        let o = h.grant_stream(StreamRole::Out);
        let e = h.grant_stream(StreamRole::Err);
        let b = h.grant_budget(-1, 1 << 20, -1);
        (
            h,
            vec![Value::I32(i), Value::I32(o), Value::I32(e), Value::I32(b)],
        )
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

/// A module child whose required import nothing binds is refused `-EINVAL` before any child code runs
/// (IMPORTS.md §3.3 withhold) — on every driver. The resumable `Vcpu` used to surface the spawn and
/// trap `Malformed` when its host built the child; it now admits through the same function as the rest.
#[test]
fn a_module_child_whose_import_is_unbound_is_refused() {
    let m = module(&op13(false));
    let child = module(CHILD_IMPORTS_EXIT);
    let setup = || {
        let mut h = Host::new();
        let i = h.grant_instantiator(0, 1 << 17);
        let c = h.grant_module(&child);
        let o = h.grant_stream(StreamRole::Out);
        (h, vec![Value::I32(i), Value::I32(c), Value::I32(o)])
    };
    agree_on_every_driver("op 13, an unbound required import", &m, &setup, &ok(-22));
}

// ---- §5 detached spawns (op 15): a fresh window of the child's own, never a carve ----

/// The payload word an op-15 parent hands its child at `module_args_base()`.
const PAYLOAD: i64 = 1000;

/// An op-15 parent `(i32 inst, i32 module, i32 budget, i32 out) -> i64`: `module`'s entry 0 in a fresh
/// 32 KiB window, funded by `budget`, handed `PAYLOAD` as its 8-byte args payload and, iff `grant`,
/// `stdout` by name.
fn op15(grant: bool) -> String {
    op15_then(grant, 0, JOIN_OR_ERRNO)
}

/// #1975 — the tail of a parent that reports what a refused spawn left in its budget: the `mem` room,
/// or `-1` if the spawn was admitted after all.
const ROOM_AFTER_REFUSAL: &str = "\
  vz = i32.const 0
  vneg = i32.lt_s vch vz
  br_if vneg 1(vbud) 2()
}
block 1 (vb1: i32) {
  vf = i64.const 1
  vroom = call.cap 14 1 (i64) -> (i64) vb1 (vf)
  return vroom
}
block 2 () {
  vm = i64.const -1
  return vm
  }
}
";

/// [`op15`] passing the fuel `quota` operand `quota` and ending in `tail` instead of
/// [`JOIN_OR_ERRNO`].
fn op15_then(grant: bool, quota: i64, tail: &str) -> String {
    format!(
        "memory 17
func (i32, i32, i32, i32) -> (i64) {{
block 0 (vinst: i32, vmod: i32, vbud: i32, vout: i32) {{
{g}  pa = i64.const 20480
  pw = i64.const {PAYLOAD}
  i64.store pa pw
  vm = i64.extend_i32_s vmod
  vb = i64.extend_i32_s vbud
  gp = i64.const 16384
  gn = i64.const {n}
  en = i64.const 0
  sl = i64.const 15
  q = i64.const {quota}
  al = i64.const 8
  vch = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64, i64, i64) -> (i32) vinst (vb, vm, gp, gn, en, sl, q, pa, al)
{tail}",
        g = store_grant("g", 16384, 16484, "stdout", "vout"),
        n = u8::from(grant),
    )
}

/// The powerbox an op-15 parent runs over: its `Instantiator`, `child` as a `Module`, a `Budget` whose
/// `mem` quota is `mem` bytes, and `stdout`.
fn op15_setup(child: &Module, mem: i64) -> impl Fn() -> (Host, Vec<Value>) + '_ {
    move || {
        let mut h = Host::new();
        let i = h.grant_instantiator(0, 1 << 17);
        let c = h.grant_module(child);
        let b = h.grant_budget(-1, mem, -1);
        let o = h.grant_stream(StreamRole::Out);
        let args = vec![Value::I32(i), Value::I32(c), Value::I32(b), Value::I32(o)];
        (h, args)
    }
}

/// A detached child that returns its payload word plus the word of its data segment.
const CHILD_READS: &str = "memory 15
data 24576 \"ABCDEFGH\"
func (i64) -> (i64) {
block 0 (v0: i64) {
  va = i64.const 16512
  vp = i64.load va
  vd = i64.const 24576
  vw = i64.load vd
  vr = i64.add vp vw
  return vr
  }
}
";

/// A detached child that returns its `self.attest` report: `tier | window_exposed << 8 |
/// freeze_exposed << 9`.
const CHILD_ATTESTS: &str = "memory 15
func (i64) -> (i64) {
block 0 (v0: i64) {
  vz = i32.const 0
  vr = call.cap 4294967295 4 () -> (i64) vz ()
  return vr
  }
}
";

/// A detached child that imports `exit` — a required slot the spawn must bind or refuse.
const DETACHED_IMPORTS_EXIT: &str = "memory 15
import 0 \"exit\" (i32) -> ()
func (i64) -> (i64) {
block 0 (v0: i64) {
  vr = i64.const 42
  return vr
  }
}
";

/// A detached child that writes to its own read-only data segment.
const CHILD_WRITES_RO: &str = "memory 15
data ro 24576 \"ABCDEFGH\"
func (i64) -> (i64) {
block 0 (v0: i64) {
  va = i64.const 24576
  vw = i64.const 7
  i64.store va vw
  vr = i64.const 42
  return vr
  }
}
";

/// A detached child whose entry resolves `stdout` by name, writes `M` and returns 42 (its scratch above
/// the NULL guard its 32 KiB window has).
fn detached_writes_m() -> String {
    format!(
        "memory 15
func (i64) -> (i64) {{
block 0 (v0: i64) {{
{n}  vl = i64.const 6
  vp = i64.const 20000
  vh = self.resolve vp vl
  vb = i64.const 20100
  vc = i32.const 77
  i32.store8 vb vc
  one = i64.const 1
  vw = call.cap 0 1 (i64, i64) -> (i64) vh (vb, one)
  vr = i64.const 42
  return vr
  }}
}}
",
        n = store_name("c", 20000, "stdout"),
    )
}

#[test]
fn a_detached_child_starts_with_its_data_and_payload() {
    let m = module(&op15(false));
    let child = module(CHILD_READS);
    let want = ok(PAYLOAD.wrapping_add(i64::from_le_bytes(*b"ABCDEFGH")));
    agree_on_every_driver(
        "op 15, data + payload",
        &m,
        &op15_setup(&child, 1 << 20),
        &want,
    );
}

/// A detached child attests its spawner's isolation tier, with a window no ancestor reads (PROCESS.md
/// §5). `Vcpu`'s host used to build the child's powerbox and gave it the default report, tier 1.
#[test]
fn a_detached_child_attests_its_spawners_tier() {
    let m = module(&op15(false));
    let child = module(CHILD_ATTESTS);
    let base = op15_setup(&child, 1 << 20);
    let setup = || {
        let (mut h, args) = base();
        h.set_attestation(Attestation {
            tier: 0,
            window_exposed: false,
            freeze_exposed: false,
        });
        (h, args)
    };
    agree_on_every_driver("op 15, a tier-0 spawner", &m, &setup, &ok(0));
}

/// An unbound required import refuses the spawn `-EINVAL` (IMPORTS.md §3.3 withhold). `Vcpu`'s host
/// used to bind the manifest when it built the child and trapped `Malformed`.
#[test]
fn a_detached_child_whose_import_is_unbound_is_refused() {
    let m = module(&op15(false));
    let child = module(DETACHED_IMPORTS_EXIT);
    let setup = op15_setup(&child, 1 << 20);
    agree_on_every_driver("op 15, an unbound import", &m, &setup, &ok(-22));
    // #1975 — and it charged nothing: the import is bound after the admission, whose take the refusal
    // hands back, so the budget still has all its room.
    let m = module(&op15_then(false, 0, ROOM_AFTER_REFUSAL));
    agree_on_every_driver(
        "op 15, an unbound import charges nothing",
        &m,
        &setup,
        &ok(1 << 20),
    );
}

#[test]
fn a_detached_child_gets_the_caps_granted_it_by_name() {
    let m = module(&op15(true));
    let child = module(&detached_writes_m());
    let want = Ran {
        result: Ok(vec![Value::I64(42)]),
        stdout: b"M".to_vec(),
        stderr: Vec::new(),
    };
    agree_on_every_driver(
        "op 15 + a named grant",
        &m,
        &op15_setup(&child, 1 << 20),
        &want,
    );
}

/// A `ro` data segment is read-only in the child's fresh window. `Vcpu`'s hosts used to seed the
/// window themselves, writably.
#[test]
fn a_detached_childs_read_only_data_stays_read_only() {
    let m = module(&op15(false));
    let child = module(CHILD_WRITES_RO);
    let want = Ran {
        result: Err(Trap::MemoryFault),
        stdout: Vec::new(),
        stderr: Vec::new(),
    };
    agree_on_every_driver(
        "op 15, a write to ro data",
        &m,
        &op15_setup(&child, 1 << 20),
        &want,
    );
}

#[test]
fn a_detached_spawn_beyond_the_budget_is_refused() {
    let m = module(&op15(false));
    let child = module(CHILD_READS);
    let setup = op15_setup(&child, (1 << 15) - 1);
    agree_on_every_driver("op 15, a budget one byte short", &m, &setup, &ok(-22));
}

/// [`op15_setup`] over a budget tree: a node split from the root with `node_mem` (`-1` = unbounded on
/// its own), and the root already charged `root_used`. The guest pays with the node, or with the root
/// when `pay_root`.
fn op15_tree_setup(
    child: &Module,
    root_mem: i64,
    node_mem: i64,
    root_used: u64,
    pay_root: bool,
) -> impl Fn() -> (Host, Vec<Value>) + '_ {
    let base = op15_setup(child, root_mem);
    move || {
        let (mut h, mut args) = base();
        let Value::I32(root) = args[2] else {
            unreachable!("op15_setup hands the budget third")
        };
        let node = h
            .cap_dispatch_slots(cap_id::BUDGET, 0, root, &[-1, node_mem, -1], None)
            .expect("split")[0] as i32;
        assert!(h.budget_mem_take(root, root_used), "the root's own use");
        if !pay_root {
            args[2] = Value::I32(node);
        }
        (h, args)
    }
}

/// #1944 — on every driver, a detached window is charged to the node that pays and to every
/// ancestor, and a `split` reserves nothing: the root's own use leaves a 32 KiB node no room for a
/// 32 KiB window; a node's own ceiling refuses what its root could hold; and the root can still pay
/// for the window its 32 KiB child node could have used.
#[test]
fn a_detached_window_is_charged_to_every_level_of_the_chain() {
    let m = module(&op15(false));
    let child = module(CHILD_READS);
    let ran = ok(PAYLOAD.wrapping_add(i64::from_le_bytes(*b"ABCDEFGH")));
    agree_on_every_driver(
        "op 15, the root's use caps its node",
        &m,
        &op15_tree_setup(&child, 1 << 16, 1 << 15, 3 << 14, false),
        &ok(-22),
    );
    agree_on_every_driver(
        "op 15, room at every level",
        &m,
        &op15_tree_setup(&child, 1 << 16, 1 << 15, 1 << 14, false),
        &ran,
    );
    agree_on_every_driver(
        "op 15, the node one byte short",
        &m,
        &op15_tree_setup(&child, 1 << 20, (1 << 15) - 1, 0, false),
        &ok(-22),
    );
    agree_on_every_driver(
        "op 15, a split reserves nothing",
        &m,
        &op15_tree_setup(&child, 1 << 15, 1 << 15, 0, true),
        &ran,
    );
}

// ---- #1944 slice 3: a detached child burns the fuel of the budget that paid for it ----

/// A detached child that takes its payload word's worth of loop back-edges (one fuel each), then
/// returns 7.
const CHILD_LOOPS: &str = "memory 15
func (i64) -> (i64) {
block 0 (v0: i64) {
  va = i64.const 16512
  vp = i64.load va
  vn = i32.wrap_i64 vp
  br 1(vn)
}
block 1 (vi: i32) {
  one = i32.const 1
  vj = i32.sub vi one
  br_if vj 1(vj) 2()
}
block 2 () {
  v = i64.const 7
  return v
  }
}
";

/// [`op15_setup`] paying with a node split from the root with a `fuel` ceiling (`-1` = unbounded on
/// its own, so only the run's fuel caps it).
fn op15_fuel_setup(child: &Module, fuel: i64) -> impl Fn() -> (Host, Vec<Value>) + '_ {
    let base = op15_setup(child, 1 << 20);
    move || {
        let (mut h, mut args) = base();
        let Value::I32(root) = args[2] else {
            unreachable!("op15_setup hands the budget third")
        };
        let node = h
            .cap_dispatch_slots(cap_id::BUDGET, 0, root, &[fuel, -1, -1], None)
            .expect("split")[0];
        args[2] = Value::I32(node as i32);
        (h, args)
    }
}

fn trapped(t: Trap) -> Ran {
    Ran {
        result: Err(t),
        stdout: Vec::new(),
        stderr: Vec::new(),
    }
}

/// #1944 slice 3 — on every driver, a detached child burns the fuel of the budget that paid for it:
/// a node's fuel ceiling ends a child that loops past it, though its parent has fuel to spare (the
/// join hands the child's trap to the parent), and a node with room lets it finish.
#[test]
fn a_detached_childs_fuel_is_capped_by_its_budget() {
    let m = module(&op15(false));
    let child = module(CHILD_LOOPS);
    agree_on_every_driver(
        "op 15, a fuel ceiling with room",
        &m,
        &op15_fuel_setup(&child, 2 * PAYLOAD),
        &ok(7),
    );
    agree_on_every_driver(
        "op 15, a node unbounded on its own",
        &m,
        &op15_fuel_setup(&child, -1),
        &ok(7),
    );
    agree_on_every_driver(
        "op 15, a fuel ceiling the child loops past",
        &m,
        &op15_fuel_setup(&child, PAYLOAD / 2),
        &trapped(Trap::OutOfFuel),
    );
}

/// #1944 slice 3 — the per-spawn fuel `quota` is retired (the budget is the one fuel limit): a
/// nonzero one traps `CapFault` on every driver, as op 15's operand and as a v1 record's field.
#[test]
fn a_detached_spawn_with_a_fuel_quota_traps() {
    let child = module(CHILD_LOOPS);
    let m = module(&op15_then(false, 5, JOIN_OR_ERRNO));
    agree_on_every_driver(
        "op 15, a nonzero quota",
        &m,
        &op15_setup(&child, 1 << 20),
        &trapped(Trap::CapFault),
    );
    let mut rec = SpawnRec::v1(1);
    rec.quota = 5;
    let m = module(&record_spawn(&rec));
    agree_on_every_driver(
        "op 17, a v1 record with a nonzero quota",
        &m,
        &generations_setup(&m),
        &trapped(Trap::CapFault),
    );
    let m = module(&record_spawn(&SpawnRec::v1(1)));
    agree_on_every_driver(
        "op 17, a v1 record with no quota",
        &m,
        &generations_setup(&m),
        &ok(3),
    );
}

/// A detached child that returns its payload word.
const CHILD_RETURNS_PAYLOAD: &str = "memory 15
func (i64) -> (i64) {
block 0 (v0: i64) {
  va = i64.const 16512
  vp = i64.load va
  return vp
  }
}
";

/// An op-15 parent `(i32 inst, i32 module, i32 budget) -> i64`: spawns three detached children
/// `vh0..vh2`, child `k` handed payload word `100 + k`, then runs `joins`, which must leave the result
/// in `vr`.
fn three_children_then(joins: &str) -> String {
    let spawns: String = (0..3)
        .map(|k| {
            format!(
                "  pa{k} = i64.const {at}\n  pw{k} = i64.const {w}\n  i64.store pa{k} pw{k}\n\
                 \x20 vh{k} = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64, i64, i64) -> (i32) vinst (vb, vm, gz, gz, gz, sl, gz, pa{k}, al)\n",
                at = 20480 + 8 * k,
                w = 100 + k,
            )
        })
        .collect();
    format!(
        "memory 17
func (i32, i32, i32) -> (i64) {{
block 0 (vinst: i32, vmod: i32, vbud: i32) {{
  vm = i64.extend_i32_s vmod
  vb = i64.extend_i32_s vbud
  gz = i64.const 0
  sl = i64.const 15
  al = i64.const 8
{spawns}{joins}
  return vr
  }}
}}
"
    )
}

/// Join handle `h` (a text-IR operand) into `dst`.
fn join(dst: &str, h: &str) -> String {
    format!("  {dst} = call.cap 6 1 (i32) -> (i64) vinst ({h})\n")
}

/// Run `joins` after three detached spawns on every driver. A join's handle is resolved by the
/// oracle's rule (`resolve_thread`): negative traps, any other is masked to the table's power-of-two
/// span, and a spent or never-issued slot traps. The `Vcpu`'s hosts each kept their own table and
/// rule until the engine took it over: the test orchestrator waited forever on a re-join.
fn joins_agree(what: &str, joins: &str, want: &Ran) {
    let m = module(&three_children_then(joins));
    let child = module(CHILD_RETURNS_PAYLOAD);
    let setup = |h: &mut Host| {
        let i = h.grant_instantiator(0, 1 << 17);
        let c = h.grant_module(&child);
        let b = h.grant_budget(-1, 1 << 20, -1);
        vec![Value::I32(i), Value::I32(c), Value::I32(b)]
    };
    let setup = || {
        let mut h = Host::new();
        let args = setup(&mut h);
        (h, args)
    };
    agree_on_every_driver(what, &m, &setup, want);
}

#[test]
fn each_child_joins_once_by_its_handle() {
    let joins = format!(
        "{}{}{}  vs = i64.add vr0 vr1\n  vr = i64.add vs vr2\n",
        join("vr0", "vh0"),
        join("vr1", "vh1"),
        join("vr2", "vh2")
    );
    joins_agree("three joins", &joins, &ok(303));
}

#[test]
fn a_second_join_of_a_child_traps() {
    let joins = format!("{}{}", join("vr0", "vh0"), join("vr", "vh0"));
    joins_agree("a re-join", &joins, &trapped(Trap::ThreadFault));
}

#[test]
fn a_negative_handle_traps() {
    joins_agree(
        "join(-1)",
        &format!("  vn = i32.const -1\n{}", join("vr", "vn")),
        &trapped(Trap::ThreadFault),
    );
}

#[test]
fn a_handle_past_the_table_traps() {
    // Three children: the span is 4, and slot 3 was never issued.
    joins_agree(
        "join(3)",
        &format!("  vn = i32.const 3\n{}", join("vr", "vn")),
        &trapped(Trap::ThreadFault),
    );
}

#[test]
fn a_handle_masks_onto_the_table() {
    // 5 & (4 - 1) is slot 1: the second child.
    joins_agree(
        "join(5)",
        &format!("  vn = i32.const 5\n{}", join("vr", "vn")),
        &ok(101),
    );
}

// ---- #1944: a detached child pays for its own detached child ------------------------------------

/// Three generations of one `memory 16` module, each window 64 KiB. The root resolves its `"budget"`,
/// splits a node with a `mem` ceiling of `mem` and a `spawn` ceiling of `spawn` (`-1` = unbounded),
/// and spawns func 1 detached, paid from it.
/// Func 1 resolves its own `"budget"` — the node that paid for it — and spawns func 2 from it: a
/// refused spawn returns its `-errno`, an admitted one `join + 20`. Func 2 returns 3. The root returns
/// what its child did plus 100.
fn three_generations(mem: i64, spawn: i64) -> String {
    format!(
        "memory 16
data 16384 \"budget\"
{r1}{r2}func (i32) -> (i64) {{
block 0 (vinst: i32) {{
  np = i64.const 16384
  nl = i64.const 6
  vroot = self.resolve np nl
  all = i64.const -1
  cap = i64.const {mem}
  sp = i64.const {spawn}
  vsub = call.cap 14 0 (i64, i64, i64) -> (i32) vroot (all, cap, sp)
  bf = i64.const 17436
  i32.store bf vsub
  rp = i64.const 17408
  vh = call.cap 6 17 (i64) -> (i32) vinst (rp)
  vj = call.cap 6 1 (i32) -> (i64) vinst (vh)
  k = i64.const 100
  vr = i64.add vj k
  return vr
  }}
}}
func (i64) -> (i64) {{
block 0 (va: i64) {{
  vinst = i32.wrap_i64 va
  np = i64.const 16384
  nl = i64.const 6
  vb = self.resolve np nl
  bf = i64.const 17532
  i32.store bf vb
  rp = i64.const 17504
  vch = call.cap 6 17 (i64) -> (i32) vinst (rp)
  vz = i32.const 0
  vneg = i32.lt_s vch vz
  br_if vneg 1(vch) 2(vinst, vch)
  }}
block 1 (ve: i32) {{
  vr = i64.extend_i32_s ve
  return vr
  }}
block 2 (vi: i32, vh: i32) {{
  vj = call.cap 6 1 (i32) -> (i64) vi (vh)
  k = i64.const 20
  vr = i64.add vj k
  return vr
  }}
}}
func (i64) -> (i64) {{
block 0 (va: i64) {{
  v = i64.const 3
  return v
  }}
}}
",
        r1 = rec::segment(17408, &SpawnRec::v1(1)),
        r2 = rec::segment(17504, &SpawnRec::v1(2)),
    )
}

/// The root's powerbox: its `Instantiator`, its running module, and a 1 MiB `"budget"`.
fn generations_setup(m: &Module) -> impl Fn() -> (Host, Vec<Value>) + '_ {
    move || {
        let mut h = Host::new();
        h.set_self_module(&std::sync::Arc::new(m.clone()));
        let i = h.grant_instantiator(0, 1 << 16);
        let b = h.grant_budget(-1, 1 << 20, -1);
        h.register_cap_name("budget", b);
        (h, vec![Value::I32(i)])
    }
}

/// A `memory 16` root that spawns its func 1 by the v1 record `rec`, paid from its `"budget"`, and
/// returns the join. Func 1 returns 3.
fn record_spawn(rec: &SpawnRec) -> String {
    format!(
        "memory 16
data 16384 \"budget\"
{r}func (i32) -> (i64) {{
block 0 (vinst: i32) {{
  np = i64.const 16384
  nl = i64.const 6
  vb = self.resolve np nl
  bf = i64.const 17436
  i32.store bf vb
  rp = i64.const 17408
  vh = call.cap 6 17 (i64) -> (i32) vinst (rp)
  vj = call.cap 6 1 (i32) -> (i64) vinst (vh)
  return vj
  }}
}}
func (i64) -> (i64) {{
block 0 (va: i64) {{
  v = i64.const 3
  return v
  }}
}}
",
        r = rec::segment(17408, rec),
    )
}

#[test]
fn a_detached_child_spawns_a_grandchild_from_its_own_budget() {
    let m = module(&three_generations(1 << 17, -1));
    agree_on_every_driver(
        "a child whose 128 KiB ceiling holds its window and its child's",
        &m,
        &generations_setup(&m),
        &ok(3 + 20 + 100),
    );
}

#[test]
fn a_childs_ceiling_caps_its_subtree_while_its_parent_has_room() {
    let m = module(&three_generations(1 << 16, -1));
    agree_on_every_driver(
        "a child whose 64 KiB ceiling its own window fills",
        &m,
        &generations_setup(&m),
        &ok(-22 + 100),
    );
}

// ---- #1944 slice 3: a detached child is one `spawn` of the budget that pays for it while it lives ----

/// An op-15 spawn of `module` into `dst` (a text-IR operand), funded by `vb`, whose payload word is
/// `word` at `at`. Expects `vinst`, `vb`, `vm`, `gz`, `sl` and `al` in scope.
fn spawn_into(dst: &str, at: u64, word: i64) -> String {
    format!(
        "  pa{dst} = i64.const {at}\n  pw{dst} = i64.const {word}\n  i64.store pa{dst} pw{dst}\n\
         \x20 {dst} = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64, i64, i64) -> (i32) vinst (vb, vm, gz, gz, gz, sl, gz, pa{dst}, al)\n"
    )
}

/// An op-15 parent `(i32 inst, i32 module, i32 budget) -> i64` that spawns child A, joins it, then spawns
/// child C and joins it, returning `join(A) + join(C)`: each child returns its payload word, A's 100
/// and C's 102. A refused C traps its join (a negative handle).
fn spawn_after_join() -> String {
    format!(
        "memory 17
func (i32, i32, i32) -> (i64) {{
block 0 (vinst: i32, vmod: i32, vbud: i32) {{
  vm = i64.extend_i32_s vmod
  vb = i64.extend_i32_s vbud
  gz = i64.const 0
  sl = i64.const 15
  al = i64.const 8
{a}{ja}{c}{jc}  vr = i64.add ja jc
  return vr
  }}
}}
",
        a = spawn_into("vha", 20480, 100),
        ja = join("ja", "vha"),
        c = spawn_into("vhc", 20496, 102),
        jc = join("jc", "vhc"),
    )
}

/// The powerbox a [`spawn_after_join`] parent runs over: its `Instantiator`, `child` as a `Module`,
/// and a `Budget` of 1 MiB whose `spawn` ceiling is `spawn`.
fn spawn_setup(child: &Module, spawn: i64) -> impl Fn() -> (Host, Vec<Value>) + '_ {
    move || {
        let mut h = Host::new();
        let i = h.grant_instantiator(0, 1 << 17);
        let c = h.grant_module(child);
        let b = h.grant_budget(-1, 1 << 20, spawn);
        (h, vec![Value::I32(i), Value::I32(c), Value::I32(b)])
    }
}

/// On every driver, a detached child hands its `spawn` back when it ends: a one-child budget funds C
/// once A is joined. (That a live child holds its charge is
/// [`a_childs_spawn_ceiling_counts_itself_and_its_children`]'s: a child is live while it spawns.)
#[test]
fn a_joined_detached_child_hands_its_spawn_back() {
    let m = module(&spawn_after_join());
    let child = module(CHILD_RETURNS_PAYLOAD);
    agree_on_every_driver(
        "op 15, a one-child budget spawning twice in turn",
        &m,
        &spawn_setup(&child, 1),
        &ok(100 + 102),
    );
}

/// On every driver, a budget whose `spawn` ceiling is 0 funds no child, and its refusal charges
/// nothing: the budget's `mem` room is whole after it.
#[test]
fn a_spawn_0_budget_funds_no_child() {
    let m = module(&op15_then(false, 0, ROOM_AFTER_REFUSAL));
    let child = module(CHILD_READS);
    let setup = || {
        let (mut h, mut args) = op15_setup(&child, 1 << 20)();
        args[2] = Value::I32(h.grant_budget(-1, 1 << 20, 0));
        (h, args)
    };
    agree_on_every_driver("op 15, a spawn-0 budget", &m, &setup, &ok(1 << 20));
}

/// On every driver, a child's `spawn` ceiling counts its own first vCPU and its children: a child
/// whose ceiling is one live vCPU fills it itself and is refused a grandchild, and one whose ceiling
/// is two is not.
#[test]
fn a_childs_spawn_ceiling_counts_itself_and_its_children() {
    let m = module(&three_generations(1 << 17, 1));
    agree_on_every_driver(
        "a child whose one-vCPU ceiling it fills",
        &m,
        &generations_setup(&m),
        &ok(-22 + 100),
    );
    let m = module(&three_generations(1 << 17, 2));
    agree_on_every_driver(
        "a child whose two-vCPU ceiling holds its child",
        &m,
        &generations_setup(&m),
        &ok(3 + 20 + 100),
    );
}

// ---- #1944 slice 3: a detached child's pipes are charged to the budget that pays for it ----

/// A pipe's worst-case FIFO, the `channel` memory each mint charges (`temen_interp`'s `PIPE_CAP`).
const PIPE_CAP: i64 = 64 * 1024;

/// A detached child that mints pipes (the `pipe()` self-op, 16) until one is refused, and returns how
/// many it minted.
const CHILD_MINTS_PIPES: &str = "memory 15
func (i64) -> (i64) {
block 0 (v0: i64) {
  vn0 = i64.const 0
  br 1(vn0)
}
block 1 (vn: i64) {
  vz = i32.const 0
  vfds = i64.const 20480
  vr = call.cap 4294967295 16 (i64) -> (i32) vz (vfds)
  vrz = i32.const 0
  vfail = i32.lt_s vr vrz
  br_if vfail 2(vn) 3(vn)
}
block 2 (vnf: i64) {
  return vnf
}
block 3 (vnok: i64) {
  vone = i64.const 1
  vn2 = i64.add vnok vone
  br 1(vn2)
  }
}
";

/// On every driver, a detached child's pipes are charged to the budget that paid for it: a child
/// whose budget holds one pipe's channel memory mints exactly one.
#[test]
fn a_detached_childs_pipes_are_capped_by_its_budget() {
    let m = module(&op15(false));
    let child = module(CHILD_MINTS_PIPES);
    let base = op15_setup(&child, 1 << 20);
    let setup = || {
        let (mut h, mut args) = base();
        let Value::I32(root) = args[2] else {
            unreachable!("op15_setup hands the budget third")
        };
        let node = h
            .cap_dispatch_slots(cap_id::BUDGET, 0, root, &[-1, -1, -1, PIPE_CAP], None)
            .expect("split")[0];
        args[2] = Value::I32(node as i32);
        (h, args)
    };
    agree_on_every_driver("op 15, a one-pipe channel ceiling", &m, &setup, &ok(1));
}

// ---- #2001: a domain's threads are `spawn`s of its node while they live ----

/// A detached child that spawns a thread, joins it, then spawns another and joins it, returning the
/// sum of their results: each thread returns its `arg` plus one, 11 and 21.
const CHILD_SPAWNS_THREADS_IN_TURN: &str = "memory 15
func (i64) -> (i64) {
block 0 (v0: i64) {
  vz = i64.const 0
  va = i64.const 10
  vt1 = thread.spawn 1 vz va
  vj1 = thread.join vt1
  vb = i64.const 20
  vt2 = thread.spawn 1 vz vb
  vj2 = thread.join vt2
  vr = i64.add vj1 vj2
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vone = i64.const 1
  vr = i64.add varg vone
  return vr
  }
}
";

/// On every driver, a detached child's threads are `spawn`s of the budget that pays for it while they
/// live: a child whose ceiling is one vCPU fills it itself, so its first `thread.spawn` traps, and one
/// whose ceiling is two runs a thread, joins it, and runs another in its place.
#[test]
fn a_detached_childs_threads_are_capped_by_its_budget() {
    let m = module(&op15(false));
    let child = module(CHILD_SPAWNS_THREADS_IN_TURN);
    for (spawn, want) in [(1, trapped(Trap::ThreadFault)), (2, ok(11 + 21))] {
        let setup = || {
            let (mut h, mut args) = op15_setup(&child, 1 << 20)();
            args[2] = Value::I32(h.grant_budget(-1, 1 << 20, spawn));
            (h, args)
        };
        agree_on_every_driver(
            &format!("op 15, a child whose budget holds {spawn} vCPUs spawning threads in turn"),
            &m,
            &setup,
            &want,
        );
    }
}

// ---- #2053: `wait` (op 18) — how a child ended, without inheriting its trap ----

/// The tail of a parent that `wait`s for its child and returns the answer.
const WAIT: &str = "\
  w1 = call.cap 6 18 (i32) -> (i64) vinst (vch)
  return w1
  }
}
";

/// The tail of a parent that `wait`s twice, then `join`s: `(first * 1000 + second) * 1000 + value`.
const WAIT_WAIT_JOIN: &str = "\
  w1 = call.cap 6 18 (i32) -> (i64) vinst (vch)
  w2 = call.cap 6 18 (i32) -> (i64) vinst (vch)
  jr = call.cap 6 1 (i32) -> (i64) vinst (vch)
  k = i64.const 1000
  s1 = i64.mul w1 k
  s2 = i64.add s1 w2
  s3 = i64.mul s2 k
  s4 = i64.add s3 jr
  return s4
  }
}
";

/// The tail of a parent that `wait`s, then `join`s the same handle.
const WAIT_JOIN: &str = "\
  w1 = call.cap 6 18 (i32) -> (i64) vinst (vch)
  jr = call.cap 6 1 (i32) -> (i64) vinst (vch)
  return jr
  }
}
";

/// A detached child that reaches `unreachable`.
const CHILD_UNREACHABLE: &str = "memory 15
func (i64) -> (i64) {
block 0 (v0: i64) {
  unreachable
  }
}
";

/// #2053 — on every driver, `wait` parks until the child ends and answers how: 0 for a child that
/// returned, which stays for its `join` (a second `wait` answers 0 again, and the `join` still gets its
/// value); else the child's trap's wire code, and the parent runs on — a child that loops past its
/// node's fuel ceiling answers `OUT_OF_FUEL`, one that reaches `unreachable` answers `UNREACHABLE`. A
/// trapped child has nothing left to join: the wait reaps it, so a `join` after it is a spent handle.
#[test]
fn wait_answers_how_a_detached_child_ended() {
    use temen_ir::trap_code;
    let loops = module(CHILD_LOOPS);
    let crashes = module(CHILD_UNREACHABLE);
    agree_on_every_driver(
        "wait, wait, join on a child that returned",
        &module(&op15_then(false, 0, WAIT_WAIT_JOIN)),
        &op15_fuel_setup(&loops, 2 * PAYLOAD),
        &ok(7),
    );
    agree_on_every_driver(
        "wait on a child past its fuel ceiling",
        &module(&op15_then(false, 0, WAIT)),
        &op15_fuel_setup(&loops, PAYLOAD / 2),
        &ok(trap_code::OUT_OF_FUEL),
    );
    agree_on_every_driver(
        "wait on a child that reaches unreachable",
        &module(&op15_then(false, 0, WAIT)),
        &op15_setup(&crashes, 1 << 20),
        &ok(trap_code::UNREACHABLE),
    );
    agree_on_every_driver(
        "join after a wait reaped a trapped child",
        &module(&op15_then(false, 0, WAIT_JOIN)),
        &op15_fuel_setup(&loops, PAYLOAD / 2),
        &trapped(Trap::ThreadFault),
    );
}

// ---- #2068: poll, detach and kill (Instantiator ops 9/10/12) on the bytecode engine ----

/// A detached child that spins forever.
const CHILD_SPINS: &str = "memory 15
func (i64) -> (i64) {
block 0 (v0: i64) {
  br 1()
}
block 1 () {
  br 1()
  }
}
";

/// The tail of a parent that `kill`s the child it just spawned, then `wait`s:
/// `kill * 1000 + wait`.
const KILL_WAIT: &str = "\
  k1 = call.cap 6 12 (i32) -> (i32) vinst (vch)
  w1 = call.cap 6 18 (i32) -> (i64) vinst (vch)
  k2 = i64.extend_i32_s k1
  k = i64.const 1000
  s1 = i64.mul k2 k
  s2 = i64.add s1 w1
  return s2
  }
}
";

/// The tail of a parent that `wait`s, `poll`s, then `join`s: `(wait * 1000 + poll) * 1000 + value`.
const WAIT_POLL_JOIN: &str = "\
  w1 = call.cap 6 18 (i32) -> (i64) vinst (vch)
  p1 = call.cap 6 9 (i32) -> (i32) vinst (vch)
  jr = call.cap 6 1 (i32) -> (i64) vinst (vch)
  p2 = i64.extend_i32_s p1
  k = i64.const 1000
  s1 = i64.mul w1 k
  s2 = i64.add s1 p2
  s3 = i64.mul s2 k
  s4 = i64.add s3 jr
  return s4
  }
}
";

/// The tail of a parent that `wait`s, then `detach`es, then `join`s the spent handle.
const WAIT_DETACH_JOIN: &str = "\
  w1 = call.cap 6 18 (i32) -> (i64) vinst (vch)
  d1 = call.cap 6 10 (i32) -> (i32) vinst (vch)
  jr = call.cap 6 1 (i32) -> (i64) vinst (vch)
  return jr
  }
}
";

/// The tail of a parent that `wait`s, then `kill`s the child that has already ended, then `join`s.
const WAIT_KILL_JOIN: &str = "\
  w1 = call.cap 6 18 (i32) -> (i64) vinst (vch)
  k1 = call.cap 6 12 (i32) -> (i32) vinst (vch)
  jr = call.cap 6 1 (i32) -> (i64) vinst (vch)
  return jr
  }
}
";

/// #2068, #2074 — `poll`, `detach` and `kill` answer as the tree-walker's do on every driver that
/// schedules its own children:
/// - a kill ends a child that would spin forever, and `wait` answers `THREAD_FAULT` (a kill that
///   did nothing would run the child into its node's fuel ceiling: `OUT_OF_FUEL`);
/// - `poll` on a child that returned answers 1, and leaves it for its `join`;
/// - `detach` spends the handle, so a `join` after it is a `ThreadFault`;
/// - a kill of a child that has ended does nothing: its `join` still gets its value.
///
/// The `Vcpu`'s host runs its children and has no surface to answer these yet (#2083), so each
/// one traps `ThreadFault` there when it runs.
#[test]
fn poll_detach_and_kill_answer_on_every_scheduling_driver() {
    use temen_ir::trap_code;
    let spins = module(CHILD_SPINS);
    let loops = module(CHILD_LOOPS);
    // The spinning child's node holds 10M fuel, so a kill that did nothing fails in seconds, not
    // at the run's ceiling; the kill reaches it long before.
    let cases: [(&str, &str, &Module, i64, Ran); 4] = [
        (
            "kill, then wait",
            KILL_WAIT,
            &spins,
            10_000_000,
            ok(trap_code::THREAD_FAULT),
        ),
        ("wait, poll, join", WAIT_POLL_JOIN, &loops, -1, ok(1007)),
        (
            "wait, detach, join",
            WAIT_DETACH_JOIN,
            &loops,
            -1,
            trapped(Trap::ThreadFault),
        ),
        ("wait, kill, join", WAIT_KILL_JOIN, &loops, -1, ok(7)),
    ];
    for (what, tail, child, fuel, want) in cases {
        let m = module(&op15_then(false, 0, tail));
        let setup = op15_fuel_setup(child, fuel);
        agree_on(&SCHEDULING, what, &m, &setup, &want);
        assert_eq!(
            run_on(Driver::Vcpu, &m, &setup),
            Some(trapped(Trap::ThreadFault)),
            "{what}: the Vcpu fails closed"
        );
    }
}

/// A parent that `join`s, with a `kill` on a branch it never takes.
const JOIN_THEN_DEAD_KILL: &str = "\
  jr = call.cap 6 1 (i32) -> (i64) vinst (vch)
  z = i32.const 0
  br_if z 1(vinst, vch) 2(jr)
}
block 1 (ki: i32, kc: i32) {
  k1 = call.cap 6 12 (i32) -> (i32) ki (kc)
  r = i64.extend_i32_s k1
  return r
}
block 2 (rv: i64) {
  return rv
  }
}
";

/// #2083 — a module that contains `kill` but never runs it runs on every driver, the `Vcpu`
/// included: every JACL program links `unir_kill`, and the browser's Worker driver is a `Vcpu`.
#[test]
fn a_kill_that_never_runs_does_not_stop_a_module_on_any_driver() {
    let loops = module(CHILD_LOOPS);
    let m = module(&op15_then(false, 0, JOIN_THEN_DEAD_KILL));
    agree_on_every_driver("dead kill", &m, &op15_fuel_setup(&loops, -1), &ok(7));
}

/// A detached child parked on a 1 ms timed wait, which returns its wait status.
const CHILD_WAITS: &str = "memory 15
func (i64) -> (i64) {
block 0 (v0: i64) {
  va = i64.const 24576
  vz = i32.const 0
  vto = i64.const 1000000
  vst = i32.atomic.wait va vz vto
  vr = i64.extend_i32_s vst
  return vr
  }
}
";

/// A parent that returns its child's handle without joining it.
const NO_JOIN: &str = "\
  vr = i64.extend_i32_s vch
  return vr
  }
}
";

/// The `mem` and `spawn` the budget a setup granted still holds.
fn budget_held(h: &Host) -> (i64, i64) {
    let node = h
        .capture_durable_budgets()
        .into_iter()
        .find(|n| n.parent.is_none())
        .expect("the granted budget");
    (node.used.mem, node.used.spawn)
}

/// #2006 — a detached child still live when its run ends ends with it, as the oracle's teardown reaps
/// it: its window and first vCPU go back to the budget that paid for them. The parent returns without
/// joining a child parked on a timed wait; once the run is over, the budget holds nothing. (The
/// `Vcpu`'s host runs its children, and the engine hands a child's charge back only at its join:
/// #2119.)
#[test]
fn a_child_live_at_the_runs_end_hands_its_window_back() {
    let m = module(&op15_then(false, 0, NO_JOIN));
    let child = module(CHILD_WAITS);
    let setup = op15_setup(&child, 1 << 15);
    for d in SCHEDULING {
        let (ran, held) = run_on_then(d, &m, &setup, &budget_held).expect("runs");
        assert_eq!(ran, ok(0), "{d:?}");
        assert_eq!(held, (0, 0), "{d:?}: the run's end left the child charged");
    }
}

/// #2196: the root maps 64 KiB past its declared 64 KiB window, stores 77 there and loads it back,
/// then spawns a thread that loads the same word and joins it. Returns `own * 1000 + thread`, plus a
/// million if the `map` failed. The parallel driver ran over a backing the size of the declared
/// window, so the store was dropped and both loads read 0. The `Vcpu`'s host builds a thread's window
/// from its own page map, so the thread faults there: #2196's other half, B6 of #1414.
const MAP_PAST_THE_WINDOW: &str = "memory 16
func (i32) -> (i64) {
block 0 (vas: i32) {
  voff = i64.const 65536
  vlen = i64.const 65536
  vprot = i32.const 3
  vr = call.cap 5 0 (i64, i64, i32) -> (i64) vas (voff, vlen, vprot)
  vsev = i64.const 77
  i64.store voff vsev
  vown = i64.load voff
  vz = i64.const 0
  vt = thread.spawn 1 vz voff
  vth = thread.join vt
  vk = i64.const 1000
  vhi = i64.mul vown vk
  vsum = i64.add vhi vth
  vbad = i64.ne vr vz
  vbad64 = i64.extend_i32_u vbad
  vm = i64.const 1000000
  vpen = i64.mul vbad64 vm
  vres = i64.add vsum vpen
  return vres
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, vaddr: i64) {
  vx = i64.load vaddr
  return vx
  }
}
";

#[test]
fn a_thread_reads_a_page_its_spawner_mapped_past_the_declared_window() {
    let m = module(MAP_PAST_THE_WINDOW);
    let setup = || {
        let mut host = Host::new();
        let asl = host.grant_memory();
        (host, vec![Value::I32(asl)])
    };
    agree_on(
        &SCHEDULING,
        "a page mapped past the window",
        &m,
        &setup,
        &ok(77_077),
    );
}
