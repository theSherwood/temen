//! CONSOLIDATION.md §3a — **the config-record spawn** (`Instantiator` op 17,
//! `instantiate_rec(record_ptr)`): one 56-byte record subsumes every spawn variant as data —
//! module (`-1` = self), entry, carve `(off, size_log2)`, pager export (`u32::MAX` = none),
//! quota (raw scalar until §3b's `Budget` handle), and the op-11 named-grant list. These tests
//! are differentials: each legacy spawn shape (op 0 plain, op 16 demand-pager, op 5 module
//! child) is re-spelled as a record and must produce the identical result, plus the fail-closed
//! record validations (nonzero version / reserved budget field). The record ABI is the §3 end
//! state; the legacy ops stay until §3d migrates their callers and deletes them.
//!
//! Record layout (little-endian, window-relative pointer):
//! `{ version: u32 (0), entry: u32, off: u64, size_log2: u32, pager: u32 (MAX = none),
//!    module: i32 (-1 = self), budget: i32 (reserved 0), quota: i64, grants_ptr: u64,
//!    grants_n: u64 }`

use temen_interp::{run_capture_reserved_with_host, Host, StreamRole, Value};
use temen_run::{instantiate_with_imports, Backend, HostCap, Imports, Outcome, RunConfig};
use temen_text::parse_module;
use temen_verify::verify_module;

/// Emit text-IR stores building a record at window offset `at` (7 aligned i64 stores). The
/// caller provides each packed field as an i64 expression already in scope.
fn store_record(at: u64) -> String {
    format!(
        "\
  vra = i64.const {at}
  i64.store vra vf0
  vra1 = i64.const {a1}
  i64.store vra1 vf8
  vra2 = i64.const {a2}
  i64.store vra2 vf16
  vra3 = i64.const {a3}
  i64.store vra3 vf24
  vra4 = i64.const {a4}
  i64.store vra4 vf32
  vra5 = i64.const {a5}
  i64.store vra5 vf40
  vra6 = i64.const {a6}
  i64.store vra6 vf48
",
        at = at,
        a1 = at + 8,
        a2 = at + 16,
        a3 = at + 24,
        a4 = at + 32,
        a5 = at + 40,
        a6 = at + 48,
    )
}

/// A parent that spawns its func-1 child (returns 42) via the record op and exits with the
/// join result. `version`/`budget` parameterized for the fail-closed cases.
fn record_program(version: u64, budget: u64) -> String {
    // Fields: f0 = version | entry(1)<<32; f8 = off 65536; f16 = size_log2 16 | pager MAX<<32;
    // f24 = module -1 (u32 MAX) | budget<<32; f32 = quota 0; f40/f48 = no grants.
    let f0 = (version | (1u64 << 32)) as i64;
    let f16 = (16u64 | (0xFFFF_FFFFu64 << 32)) as i64;
    let f24 = (0xFFFF_FFFFu64 | (budget << 32)) as i64;
    format!(
        "\
memory 17
data 16384 \"vm\"
import 0 \"exit\" (i32) -> ()

func 0 () -> () {{
block 0 () {{
  vp = i64.const 16384
  vl = i64.const 2
  vh = self.resolve vp vl
  vf0 = i64.const {f0}
  vf8 = i64.const 65536
  vf16 = i64.const {f16}
  vf24 = i64.const {f24}
  vf32 = i64.const 0
  vf40 = i64.const 0
  vf48 = i64.const 0
{stores}
  vrp = i64.const 17408
  vch = call.cap 6 17 (i64) -> (i32) vh (vrp)
  vj = call.cap 6 1 (i32) -> (i64) vh (vch)
  vc = i32.wrap_i64 vj
  call.import 0 (vc)
  unreachable
  }}
}}

func 1 (i64) -> (i64) {{
block 0 (v0: i64) {{
  vr = i64.const 42
  return vr
  }}
}}
",
        f0 = f0,
        f16 = f16,
        f24 = f24,
        stores = store_record(17408),
    )
}

/// The op-0 legacy spelling of the same spawn, for the differential.
const LEGACY_PLAIN: &str = "\
memory 17
data 16384 \"vm\"
import 0 \"exit\" (i32) -> ()

func 0 () -> () {
block 0 () {
  vp = i64.const 16384
  vl = i64.const 2
  vh = self.resolve vp vl
  ventry = i64.const 1
  voff = i64.const 65536
  vsl = i64.const 16
  vq = i64.const 0
  vch = call.cap 6 0 (i64, i64, i64, i64) -> (i32) vh (ventry, voff, vsl, vq)
  vj = call.cap 6 1 (i32) -> (i64) vh (vch)
  vc = i32.wrap_i64 vj
  call.import 0 (vc)
  unreachable
  }
}

func 1 (i64) -> (i64) {
block 0 (v0: i64) {
  vr = i64.const 42
  return vr
  }
}
";

/// The §2.2 pager shape spelled as a record: pager export 0, one 64 KiB stride, handler
/// supplies 123 — the record twin of `paging_offer.rs`'s single-fault vertical (exit 1123). The child
/// faults at 16 KiB: its carve reserves the NULL guard below that like any window (#1206), and a guard
/// fault is fatal, never the recoverable kind a pager services.
fn record_pager_program() -> String {
    record_pager_program_at(16384)
}

/// [`record_pager_program`] with the child's faulting address chosen by the caller.
fn record_pager_program_at(fault: u64) -> String {
    let f0 = (1u64 << 32) as i64; // version 0, entry 1
    let f16 = 16i64; // size_log2 16, pager export 0
    let f24 = 0xFFFF_FFFFi64; // module -1, budget 0
    format!(
        "\
memory 17
data 16384 \"vm\"
type 0 func (i64) -> (i64)
type 1 interface {{ page: 0 }}
export 0 interface \"pager\" 1 {{ page: 2 }}
import 0 \"exit\" (i32) -> ()

func 0 () -> () {{
block 0 () {{
  vp = i64.const 16384
  vl = i64.const 2
  vh = self.resolve vp vl
  vf0 = i64.const {f0}
  vf8 = i64.const 65536
  vf16 = i64.const {f16}
  vf24 = i64.const {f24}
  vf32 = i64.const 0
  vf40 = i64.const 0
  vf48 = i64.const 0
{stores}
  vrp = i64.const 17408
  vch = call.cap 6 17 (i64) -> (i32) vh (vrp)
  vz = i32.const 0
  vs = svc.wait vz
  vj = call.cap 6 1 (i32) -> (i64) vh (vch)
  vk = i64.const 1000
  vm2 = i64.mul vs vk
  vt = i64.add vj vm2
  vc = i32.wrap_i64 vt
  call.import 0 (vc)
  unreachable
  }}
}}

func 1 (i64) -> (i64) {{
block 0 (v0: i64) {{
  vaddr = i64.const {fault}
  vb = i32.load8_u vaddr
  vbw = i64.extend_i32_u vb
  return vbw
  }}
}}

func 2 (i64) -> (i64) {{
block 0 (vaddr: i64) {{
  vb = i32.const 123
  i32.store8 vaddr vb
  vzero = i64.const 0
  return vzero
  }}
}}
",
        f0 = f0,
        f16 = f16,
        f24 = f24,
        fault = fault,
        stores = store_record(17408),
    )
}

fn run(backend: Backend, src: &str) -> Result<i32, String> {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let registry = Imports::new().provide("exit", HostCap::exit());
    let inst = instantiate_with_imports(m, registry).expect("instantiate");
    let r = inst
        .run_with_caps(
            backend,
            &RunConfig::default(),
            &[(
                "vm",
                HostCap::custom(6, 0, |h, win| h.grant_instantiator(0, win)),
            )],
        )
        .map_err(|e| e.to_string())?;
    match r.outcome {
        Outcome::Exited(code) => Ok(code),
        other => Err(format!("unexpected outcome {other:?}")),
    }
}

const BACKENDS: [Backend; 3] = [Backend::TreeWalk, Backend::Bytecode, Backend::Jit];

/// [`run`] on its own thread with a deadline: the #1217 pins guard against a *hang* (a pager
/// parked forever), so a regression must fail the test, not stall the binary until CI's timeout.
fn run_bounded(backend: Backend, src: &str) -> Result<i32, String> {
    let src = src.to_string();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(run(backend, &src));
    });
    rx.recv_timeout(std::time::Duration::from_secs(60))
        .unwrap_or_else(|_| panic!("{backend:?}: the run hung (pager parked with a dead client?)"))
}

/// The record spelling of a plain op-0 spawn produces the identical result.
#[test]
fn record_spawn_matches_legacy_plain_spawn() {
    let rec = record_program(0, 0);
    for b in BACKENDS {
        let legacy = run(b, LEGACY_PLAIN).expect("legacy");
        let record = run(b, &rec).expect("record");
        assert_eq!(record, legacy, "{b:?}: record ≡ op 0");
        assert_eq!(record, 42, "{b:?}: the child's result");
    }
}

/// The record spelling of the §2.2 demand-pager spawn: one fault, one serve, page supplied.
#[test]
fn record_spawn_carries_the_pager_binding() {
    let src = record_pager_program();
    for b in BACKENDS {
        assert_eq!(run(b, &src).expect("run"), 1123, "{b:?}: record ≡ op 16");
    }
}

/// #1206: a pager-bound child's fault **below its NULL guard** is fatal on every backend — never the
/// recoverable kind the pager services (the reserved region cannot be mapped). The parent is parked
/// in `svc.wait` for a request that never comes; #1217 releases it (its `svc.wait` returns `0`, the
/// no-progress answer) so its `join` surfaces the trap — the run traps instead of hanging.
#[test]
fn pager_child_guard_fault_is_fatal() {
    let src = record_pager_program_at(0);
    for b in BACKENDS {
        assert!(
            run_bounded(b, &src).is_err(),
            "{b:?}: a NULL fault in a paged child is fatal"
        );
    }
}

/// #1217: a demand child that **never faults** — it returns `5` without touching its window — must
/// not strand the pager parked in `svc.wait`. The wait returns `0` once the child is gone; the
/// `join` delivers `5`; the run exits `0 * 1000 + 5`.
#[test]
fn pager_child_that_never_faults_releases_the_parked_pager() {
    let src = record_pager_program_at(0).replace("vb = i32.load8_u vaddr", "vb = i32.const 5");
    for b in BACKENDS {
        assert_eq!(run_bounded(b, &src).expect("run"), 5, "{b:?}");
    }
}

/// #1217, the other order: the child is already gone (joined) when the parent reaches `svc.wait`,
/// so nothing could ever wake it — the wait returns `0` immediately rather than parking.
#[test]
fn pager_svc_wait_after_the_child_is_joined_returns_zero() {
    let src = record_pager_program_at(0)
        .replace("vb = i32.load8_u vaddr", "vb = i32.const 5")
        .replace(
            "  vs = svc.wait vz\n  vj = call.cap 6 1 (i32) -> (i64) vh (vch)\n",
            "  vj = call.cap 6 1 (i32) -> (i64) vh (vch)\n  vs = svc.wait vz\n",
        );
    assert!(
        src.contains("(vch)\n  vs = svc.wait"),
        "the swap must apply"
    );
    for b in BACKENDS {
        assert_eq!(run_bounded(b, &src).expect("run"), 5, "{b:?}");
    }
}

/// A record with a nonzero version — or a nonzero reserved budget field (§3b's slot) — fails
/// the spawn closed before any child state exists.
#[test]
fn malformed_records_fail_closed() {
    for (version, budget) in [(1u64, 0u64), (0, 7)] {
        let src = record_program(version, budget);
        for b in BACKENDS {
            let e = run(b, &src).expect_err("must trap");
            assert!(
                e.contains("trap") || e.contains("Cap"),
                "{b:?}: v{version}/b{budget}: {e}"
            );
        }
    }
}

/// The record spelling of an op-5 module-child spawn (host-granted separate module) matches
/// the legacy op. Host-level harness: entry receives `(Instantiator, Module)` handles.
#[test]
fn record_spawn_matches_legacy_module_spawn() {
    let child_src = "\
memory 16
data 16484 \"K\"

func 0 (i64) -> (i64) {
block 0 (v0: i64) {
  va = i64.const 16484
  vb = i32.load8_u va
  vw = i64.extend_i32_u vb
  return vw
  }
}
";
    // Parent: build the record with the module handle packed at offset 24 (low half of vf24).
    // The module handle arrives as the second entry arg; carve = 64 KiB at 64 KiB (the child's
    // declared memory). f24 = (module as u32) | budget(0)<<32 — assembled at runtime.
    let rec_parent = format!(
        "\
memory 17

func 0 (i32, i32) -> (i64) {{
block 0 (vh: i32, vmod: i32) {{
  vf0 = i64.const {f0}
  vf8 = i64.const 65536
  vf16 = i64.const {f16}
  vmask = i64.const 4294967295
  vm64 = i64.extend_i32_s vmod
  vf24 = i64.and vm64 vmask
  vf32 = i64.const 0
  vf40 = i64.const 0
  vf48 = i64.const 0
{stores}
  vrp = i64.const 17408
  vch = call.cap 6 17 (i64) -> (i32) vh (vrp)
  vj = call.cap 6 1 (i32) -> (i64) vh (vch)
  return vj
  }}
}}
",
        f0 = 0i64, // version 0, entry 0 (the child module's func 0)
        f16 = (16u64 | (0xFFFF_FFFFu64 << 32)) as i64,
        stores = store_record(17408),
    );
    let legacy_parent = "\
memory 17

func 0 (i32, i32) -> (i64) {
block 0 (vh: i32, vmod: i32) {
  vm64 = i64.extend_i32_s vmod
  ventry = i64.const 0
  voff = i64.const 65536
  vsl = i64.const 16
  vq = i64.const 0
  vch = call.cap 6 5 (i64, i64, i64, i64, i64) -> (i32) vh (vm64, ventry, voff, vsl, vq)
  vj = call.cap 6 1 (i32) -> (i64) vh (vch)
  return vj
  }
}
";
    let child = parse_module(child_src).expect("parse child");
    verify_module(&child).expect("verify child");
    let mut results: Vec<i64> = Vec::new();
    for parent_src in [rec_parent.as_str(), legacy_parent] {
        let parent = parse_module(parent_src).expect("parse parent");
        verify_module(&parent).expect("verify parent");
        let mut host = Host::new();
        let ih = host.grant_instantiator(0, 128 << 10);
        let mh = host.grant_module(&child);
        let mut fuel = 5_000_000u64;
        let (r, _) = run_capture_reserved_with_host(
            &parent,
            0,
            &[Value::I32(ih), Value::I32(mh)],
            &mut fuel,
            &[],
            0,
            &mut host,
        );
        match r {
            Ok(vals) => match vals.first() {
                Some(Value::I64(x)) => results.push(*x),
                other => panic!("unexpected result {other:?}"),
            },
            Err(t) => panic!("trapped: {t:?}"),
        }
    }
    assert_eq!(results[0], results[1], "record ≡ op 5");
    assert_eq!(results[0], b'K' as i64, "the module child's data byte");
}

/// The record spelling of an op-11 named-grant spawn: the parent lays the same two grant
/// records (`"stdout"`@100, `"stderr"`@110) and the child resolves each by name and writes one
/// byte — `instantiate_named.rs`'s shape with the spawn spelled as data. Host-level harness.
#[test]
fn record_spawn_carries_named_grants() {
    let src = format!(
        r#"memory 17
func (i32, i32, i32) -> (i64) {{
block 0 (vinst: i32, vout: i32, verr: i32) {{
  a0 = i64.const 16384
  n100 = i32.const 16484
  i32.store a0 n100
  a4 = i64.const 16388
  n6 = i32.const 6
  i32.store a4 n6
  a8 = i64.const 16392
  i32.store a8 vout
  a12 = i64.const 16396
  z0 = i32.const 0
  i32.store a12 z0
  a16 = i64.const 16400
  n110 = i32.const 16494
  i32.store a16 n110
  a20 = i64.const 16404
  i32.store a20 n6
  a24 = i64.const 16408
  i32.store a24 verr
  a28 = i64.const 16412
  i32.store a28 z0
  cs = i32.const 115
  ct = i32.const 116
  cd = i32.const 100
  co = i32.const 111
  cu = i32.const 117
  ce = i32.const 101
  cr = i32.const 114
  p100 = i64.const 16484
  i32.store8 p100 cs
  p101 = i64.const 16485
  i32.store8 p101 ct
  p102 = i64.const 16486
  i32.store8 p102 cd
  p103 = i64.const 16487
  i32.store8 p103 co
  p104 = i64.const 16488
  i32.store8 p104 cu
  p105 = i64.const 16489
  i32.store8 p105 ct
  p110 = i64.const 16494
  i32.store8 p110 cs
  p111 = i64.const 16495
  i32.store8 p111 ct
  p112 = i64.const 16496
  i32.store8 p112 cd
  p113 = i64.const 16497
  i32.store8 p113 ce
  p114 = i64.const 16498
  i32.store8 p114 cr
  p115 = i64.const 16499
  i32.store8 p115 cr
  vf0 = i64.const {f0}
  vf8 = i64.const 65536
  vf16 = i64.const {f16}
  vf24 = i64.const {f24}
  vf32 = i64.const 0
  vf40 = i64.const 16384
  vf48 = i64.const 2
{stores}
  vrp = i64.const 17408
  vch = call.cap 6 17 (i64) -> (i32) vinst (vrp)
  r = call.cap 6 1 (i32) -> (i64) vinst (vch)
  return r
  }}
}}
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  cs = i32.const 115
  ct = i32.const 116
  cd = i32.const 100
  co = i32.const 111
  cu = i32.const 117
  ce = i32.const 101
  cr = i32.const 114
  a0 = i64.const 16384
  i32.store8 a0 cs
  a1 = i64.const 16385
  i32.store8 a1 ct
  a2 = i64.const 16386
  i32.store8 a2 cd
  a3 = i64.const 16387
  i32.store8 a3 co
  a4 = i64.const 16388
  i32.store8 a4 cu
  a5 = i64.const 16389
  i32.store8 a5 ct
  len6 = i64.const 6
  hout = self.resolve a0 len6
  a16 = i64.const 16400
  cO = i32.const 79
  i32.store8 a16 cO
  one = i64.const 1
  wo = call.cap 0 1 (i64, i64) -> (i64) hout (a16, one)
  a32 = i64.const 16416
  i32.store8 a32 cs
  a33 = i64.const 16417
  i32.store8 a33 ct
  a34 = i64.const 16418
  i32.store8 a34 cd
  a35 = i64.const 16419
  i32.store8 a35 ce
  a36 = i64.const 16420
  i32.store8 a36 cr
  a37 = i64.const 16421
  i32.store8 a37 cr
  herr = self.resolve a32 len6
  a40 = i64.const 16424
  cE = i32.const 69
  i32.store8 a40 cE
  we = call.cap 0 1 (i64, i64) -> (i64) herr (a40, one)
  v7 = i64.const 7
  return v7
  }}
}}
"#,
        f0 = (1u64 << 32) as i64,                      // version 0, entry 1
        f16 = (16u64 | (0xFFFF_FFFFu64 << 32)) as i64, // size_log2 16, no pager
        f24 = 0xFFFF_FFFFi64,                          // module -1, budget 0
        stores = store_record(17408),
    );
    let m = parse_module(&src).expect("parse");
    verify_module(&m).expect("verify");
    let mut host = Host::new();
    let ih = host.grant_instantiator(0, 128 << 10);
    let oh = host.grant_stream(StreamRole::Out);
    let eh = host.grant_stream(StreamRole::Err);
    let mut fuel = 5_000_000u64;
    let (res, _) = run_capture_reserved_with_host(
        &m,
        0,
        &[Value::I32(ih), Value::I32(oh), Value::I32(eh)],
        &mut fuel,
        &[],
        0,
        &mut host,
    );
    assert_eq!(res, Ok(vec![Value::I64(7)]), "child ran and joined");
    assert_eq!(host.stdout_bytes(), b"O", "name-resolved stdout grant");
    assert_eq!(host.stderr_bytes(), b"E", "name-resolved stderr grant");
}

// ===== §3b — the record's Budget field ==========================================================

/// Host-level harness: parent `(Instantiator, Budget)` builds a record with the budget handle
/// and spawns func 1 (loops `iters`, returns 7). On spawn success the parent joins and returns
/// `join*1000 + fuel_read + mem_read` (both post-consumption reads — 0 each after a funded
/// spawn); on a refused spawn (`-EINVAL`) it returns `1 + fuel_read*100000 + mem_read` so the
/// intact budget is observable in-guest.
fn run_budgeted(
    budget: (i64, i64, i64),
    quota: i64,
    iters: i64,
) -> Result<Vec<Value>, temen_interp::Trap> {
    let src = format!(
        r#"memory 17
func (i32, i32) -> (i64) {{
block 0 (vinst: i32, vbud: i32) {{
  vf0 = i64.const {f0}
  vf8 = i64.const 65536
  vf16 = i64.const {f16}
  vmask = i64.const 4294967295
  vb64 = i64.extend_i32_s vbud
  vbm = i64.and vb64 vmask
  vsh = i64.const 32
  vbs = i64.shl vbm vsh
  vmod = i64.const 4294967295
  vf24 = i64.or vmod vbs
  vf32 = i64.const {quota}
  vf40 = i64.const 0
  vf48 = i64.const 0
{stores}
  vrp = i64.const 17408
  vch = call.cap 6 17 (i64) -> (i32) vinst (vrp)
  vzero = i32.const 0
  visneg = i32.lt_s vch vzero
  br_if visneg 2(vbud) 1(vinst, vbud, vch)
}}
block 1 (vinst1: i32, vbud1: i32, vch1: i32) {{
  vj = call.cap 6 1 (i32) -> (i64) vinst1 (vch1)
  fld0 = i64.const 0
  fld1 = i64.const 1
  vfr = call.cap 14 1 (i64) -> (i64) vbud1 (fld0)
  vmr = call.cap 14 1 (i64) -> (i64) vbud1 (fld1)
  k = i64.const 1000
  t0 = i64.mul vj k
  t1 = i64.add t0 vfr
  t2 = i64.add t1 vmr
  return t2
}}
block 2 (vbud2: i32) {{
  fld0 = i64.const 0
  fld1 = i64.const 1
  vfr = call.cap 14 1 (i64) -> (i64) vbud2 (fld0)
  vmr = call.cap 14 1 (i64) -> (i64) vbud2 (fld1)
  k = i64.const 100000
  t0 = i64.mul vfr k
  t1 = i64.add t0 vmr
  vone = i64.const 1
  t2 = i64.add t1 vone
  return t2
  }}
}}
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  vi0 = i64.const 0
  br 1(vi0)
}}
block 1 (vi: i64) {{
  vone = i64.const 1
  vi2 = i64.add vi vone
  vn = i64.const {iters}
  vcmp = i64.lt_s vi2 vn
  br_if vcmp 1(vi2) 2(vi2)
}}
block 2 (vf: i64) {{
  v7 = i64.const 7
  return v7
  }}
}}
"#,
        f0 = (1u64 << 32) as i64,
        f16 = (16u64 | (0xFFFF_FFFFu64 << 32)) as i64,
        quota = quota,
        iters = iters,
        stores = store_record(17408),
    );
    let m = parse_module(&src).expect("parse");
    verify_module(&m).expect("verify");
    let mut host = Host::new();
    let ih = host.grant_instantiator(0, 128 << 10);
    let bh = host.grant_budget(budget.0, budget.1, budget.2);
    let mut fuel = 50_000_000u64;
    let (res, _) = run_capture_reserved_with_host(
        &m,
        0,
        &[Value::I32(ih), Value::I32(bh)],
        &mut fuel,
        &[],
        0,
        &mut host,
    );
    res
}

/// A budget-funded child draws its fuel from the budget: a tight bounded budget starves the
/// loop (`OutOfFuel` propagates through `join` as the parent's trap); an unbounded one
/// completes — and the post-spawn budget reads report 0 (consumed at commit).
#[test]
fn budget_funds_the_child_and_is_consumed_at_spawn() {
    let res = run_budgeted((-1, -1, -1), 0, 200_000);
    assert_eq!(
        res,
        Ok(vec![Value::I64(7_000)]),
        "unbounded budget completes; post-spawn reads are 0"
    );

    let res = run_budgeted((5_000, -1, -1), 0, 200_000);
    assert!(
        matches!(res, Err(temen_interp::Trap::OutOfFuel)),
        "bounded 5k fuel starves a 200k-iteration child: {res:?}"
    );
}

/// The budget's mem quota gates the carve: a 64 KiB carve over a 32 KiB budget is a probeable
/// `-EINVAL` — and the refused spawn leaves the budget INTACT (peek-then-drain discipline):
/// `1 + 1000*100000 + 32768`.
#[test]
fn budget_mem_quota_gates_the_carve_and_survives_refusal() {
    let res = run_budgeted((1_000, 32 << 10, -1), 0, 10);
    assert_eq!(
        res,
        Ok(vec![Value::I64(1 + 1_000 * 100_000 + (32 << 10))]),
        "spawn refused with -EINVAL; budget intact"
    );
}

/// A record carrying both a Budget handle and a raw quota scalar is ambiguous — fail closed.
#[test]
fn budget_plus_raw_quota_fails_closed() {
    let res = run_budgeted((-1, -1, -1), 12345, 10);
    assert!(
        matches!(res, Err(temen_interp::Trap::CapFault)),
        "budget XOR quota: {res:?}"
    );
}

/// §3c.2 — the budget-record program: a Budget-funded record spawn that the
/// §3c.2 hook funds on every tier (the pre-3c.2 form of this test asserted the `-EINVAL`
/// gap). The shared program: resolve "vm" + "bgt", build a record carrying the budget handle,
/// spawn func 1 (returns 42); exit = join on success, else `100000 - errno` (so `-EINVAL`
/// reads as 100022).
fn budget_record_src() -> String {
    format!(
        r#"memory 17
data 16384 "vm"
data 16392 "bgt"
import 0 "exit" (i32) -> ()

func 0 () -> () {{
block 0 () {{
  vp = i64.const 16384
  vl = i64.const 2
  vh = self.resolve vp vl
  vbp = i64.const 16392
  vbl = i64.const 3
  vbud = self.resolve vbp vbl
  vf0 = i64.const {f0}
  vf8 = i64.const 65536
  vf16 = i64.const {f16}
  vmask = i64.const 4294967295
  vb64 = i64.extend_i32_s vbud
  vbm = i64.and vb64 vmask
  vsh = i64.const 32
  vbs = i64.shl vbm vsh
  vmod = i64.const 4294967295
  vf24 = i64.or vmod vbs
  vf32 = i64.const 0
  vf40 = i64.const 0
  vf48 = i64.const 0
{stores}
  vrp = i64.const 17408
  vch = call.cap 6 17 (i64) -> (i32) vh (vrp)
  vzero = i32.const 0
  visneg = i32.lt_s vch vzero
  br_if visneg 2(vch) 1(vh, vch)
}}
block 1 (vh1: i32, vch1: i32) {{
  vj = call.cap 6 1 (i32) -> (i64) vh1 (vch1)
  vc = i32.wrap_i64 vj
  call.import 0 (vc)
  unreachable
}}
block 2 (verr: i32) {{
  vk = i32.const 100000
  vn = i32.sub vk verr
  call.import 0 (vn)
  unreachable
  }}
}}

func 1 (i64) -> (i64) {{
block 0 (v0: i64) {{
  vr = i64.const 42
  return vr
  }}
}}
"#,
        f0 = (1u64 << 32) as i64,
        f16 = (16u64 | (0xFFFF_FFFFu64 << 32)) as i64,
        stores = store_record(17408),
    )
}

/// §3c.2 — the **native-JIT budget record works**: the `BudgetTaker` host hook (peek at
/// validation, drain at commit) funds the child exactly as the interpreter does.
#[test]
fn budget_record_funds_the_child_on_every_tier() {
    let run_b = |backend: Backend| -> i32 {
        let m = parse_module(&budget_record_src()).expect("parse");
        verify_module(&m).expect("verify");
        let registry = Imports::new().provide("exit", HostCap::exit());
        let inst = instantiate_with_imports(m, registry).expect("instantiate");
        let r = inst
            .run_with_caps(
                backend,
                &RunConfig::default(),
                &[
                    (
                        "vm",
                        HostCap::custom(6, 0, |h, win| h.grant_instantiator(0, win)),
                    ),
                    (
                        "bgt",
                        HostCap::custom(14, 0, |h, _| h.grant_budget(-1, -1, -1)),
                    ),
                ],
            )
            .unwrap_or_else(|e| panic!("{backend:?}: {e}"));
        match r.outcome {
            Outcome::Exited(code) => code,
            other => panic!("{backend:?}: unexpected outcome {other:?}"),
        }
    };
    assert_eq!(
        run_b(Backend::TreeWalk),
        42,
        "interp: budget funds the child"
    );
    assert_eq!(run_b(Backend::Bytecode), 42, "bytecode (vetoes to oracle)");
    assert_eq!(
        run_b(Backend::Jit),
        42,
        "native JIT: the BudgetTaker funds the child (was the §3c -EINVAL gap)"
    );
}

/// §3c.2's **narrowed** gap, pinned: a budget with a bounded `spawn` ceiling (child-quota
/// threading the JIT tier doesn't have yet) stays a probeable `-EINVAL` on the native lane
/// while the interpreter honors it — budget intact either way. Bounded-zero fuel is the same
/// class. When child-quota threading lands, flip this to 42 like its sibling above.
#[test]
fn spawn_bounded_budget_record_is_the_narrowed_jit_gap() {
    let run_b = |backend: Backend| -> i32 {
        let m = parse_module(&budget_record_src()).expect("parse");
        verify_module(&m).expect("verify");
        let registry = Imports::new().provide("exit", HostCap::exit());
        let inst = instantiate_with_imports(m, registry).expect("instantiate");
        let r = inst
            .run_with_caps(
                backend,
                &RunConfig::default(),
                &[
                    (
                        "vm",
                        HostCap::custom(6, 0, |h, win| h.grant_instantiator(0, win)),
                    ),
                    (
                        "bgt",
                        HostCap::custom(14, 0, |h, _| h.grant_budget(-1, -1, 5)),
                    ),
                ],
            )
            .unwrap_or_else(|e| panic!("{backend:?}: {e}"));
        match r.outcome {
            Outcome::Exited(code) => code,
            other => panic!("{backend:?}: unexpected outcome {other:?}"),
        }
    };
    assert_eq!(
        run_b(Backend::TreeWalk),
        42,
        "interp honors a spawn ceiling"
    );
    assert_eq!(
        run_b(Backend::Jit),
        100_022,
        "native JIT: bounded spawn stays -EINVAL until child-quota threading"
    );
}
