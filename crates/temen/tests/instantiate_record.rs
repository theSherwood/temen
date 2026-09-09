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

/// #1217, the granted-offer shape: the root spawns a **server** child (func 1: one `svc.wait`,
/// returning its count), mints a `child_offer` over its export, and spawns a **guest** child
/// (func 3, `guest`) with that offer re-granted by name as `"fork"`; then `join`s the server and
/// exits its count. The guest's 4-KiB carve is below the guard size, so it stages the name at 0.
fn granted_client_program(guest: &str) -> String {
    format!(
        "\
memory 18
data 16384 \"vm\"
data 16684 \"fork\"
type 0 func (i64) -> (i64)
type 1 interface {{ op: 0 }}
export 0 interface \"fork\" 1 {{ op: 2 }}
import 0 \"exit\" (i32) -> ()
func 0 () -> () {{
block 0 () {{
  vp = i64.const 16384
  vl = i64.const 2
  v0 = self.resolve vp vl
  vf0 = i64.const 4294967296
  vf8 = i64.const 65536
  vf16 = i64.const -4294967284
  vf24 = i64.const 4294967295
  vf32 = i64.const 0
  vf40 = i64.const 0
  vf48 = i64.const 0
{server_rec}
  vsp = i64.const 17536
  vs = call.cap 6 17 (i64) -> (i32) v0 (vsp)
  vz0 = i64.const 0
  voff = call.cap 6 14 (i32, i64) -> (i32) v0 (vs, vz0)
  va0 = i64.const 16640
  vnp0 = i32.const 16684
  i32.store va0 vnp0
  va1 = i64.const 16644
  vfour = i32.const 4
  i32.store va1 vfour
  va2 = i64.const 16648
  i32.store va2 voff
  vg0 = i64.const 12884901888
  vg8 = i64.const 131072
  vg40 = i64.const 16640
  vg48 = i64.const 1
{guest_rec}
  vgp = i64.const 17600
  vg = call.cap 6 17 (i64) -> (i32) v0 (vgp)
  vjs = call.cap 6 1 (i32) -> (i64) v0 (vs)
  vc = i32.wrap_i64 vjs
  call.import 0 (vc)
  unreachable
  }}
}}
func 1 (i64) -> (i64) {{
block 0 (v0: i64) {{
  vz = i32.const 0
  vs = svc.wait vz
  return vs
  }}
}}
func 2 (i64) -> (i64) {{
block 0 (vx: i64) {{
  vseven = i64.const 7
  return vseven
  }}
}}
func 3 (i64) -> (i64) {{
block 0 (v0: i64) {{
{guest}
  }}
}}
",
        server_rec = store_record(17536),
        guest_rec = store_record(17600)
            .replace("vra", "vgra")
            .replace("vf0", "vg0")
            .replace("vf8", "vg8")
            .replace("vf40", "vg40")
            .replace("vf48", "vg48"),
        guest = guest,
    )
}

/// The healthy exchange, pinning the program shape: the guest resolves `"fork"` and calls it once
/// (the handler returns 7), the server's `svc.wait` counts 1, the root exits 1.
#[test]
fn granted_client_that_calls_once_is_served() {
    let src = granted_client_program(
        "  vfn = i64.const 1802661734
  vz8 = i64.const 0
  i64.store vz8 vfn
  vl4 = i64.const 4
  vfork = self.resolve vz8 vl4
  vr = call.cap 268435456 0 (i64) -> (i64) vfork (vz8)
  return vr",
    );
    for b in BACKENDS {
        assert_eq!(run_bounded(b, &src).expect("run"), 1, "{b:?}");
    }
}

/// #1217: the guest traps before calling. The server's only client is gone, so its `svc.wait`
/// returns `0` (never parks forever), it returns, and the root's `join` delivers `0`.
#[test]
fn granted_client_that_dies_before_calling_releases_the_server() {
    let src = granted_client_program("  unreachable");
    for b in BACKENDS {
        assert_eq!(run_bounded(b, &src).expect("run"), 0, "{b:?}");
    }
}

/// #1228: the **daemon** shape #1217 cannot cover. The server is written as an unconditional
/// `loop { svc.wait }` (the fork-manager server shape), not a wait-once-and-return, so the
/// client-death token (#1217) gives it one spurious `0` and the loop re-parks it, untimed,
/// forever. With its only client gone and the root `join`ing it, the run is a genuine
/// join-deadlock: the root parked in `join`, the daemon parked in `svc.wait`, nothing else live
/// and no external wake source. INVARIANTS.md #9 forbids a hang, so the run must **trap**
/// `ThreadFault`. The cooperative bytecode driver already does; this pins the tree-walk executor
/// (and the JIT, which declines these ops to the oracle) to the identical outcome — the #1228 fix.
#[test]
fn joined_daemon_whose_client_is_gone_deadlocks_to_threadfault() {
    let base = granted_client_program("  unreachable");
    // Re-spell the wait-once server (func 1) as an unconditional `loop { svc.wait }` daemon.
    let src = base.replace(
        "block 0 (v0: i64) {\n  vz = i32.const 0\n  vs = svc.wait vz\n  return vs\n  }",
        "block 0 (v0: i64) {\n  br 1()\n  }\nblock 1 () {\n  vz = i32.const 0\n  vs = svc.wait vz\n  br 1()\n  }",
    );
    assert!(
        src.contains("block 1 () {\n  vz = i32.const 0\n  vs = svc.wait vz\n  br 1()"),
        "the daemon rewrite must apply"
    );
    for b in BACKENDS {
        let e = run_bounded(b, &src).expect_err("the join-deadlock must trap, not hang");
        assert!(
            e.contains("ThreadFault"),
            "{b:?}: the all-parked join-deadlock traps ThreadFault, got: {e}"
        );
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

// ===== #989 slice 1b — the record's Budget `channel` bounds a spawned child ======================

/// The worst-case per-pipe host-served channel charge (`PIPE_CAP`, a full 64 KiB FIFO). Mirror of
/// `temen_interp`'s internal constant; kept in sync by the `channel_budget_bounds_...` assertions.
const PIPE_CAP: i64 = 64 << 10;

/// Host-level harness for the §14 channel bound: a parent `(Instantiator, Budget-with-channel)`
/// spawns func 1 through an op-17 record; the child loops minting host-served pipes
/// (`self.pipe`, op 16) into its OWN powerbox and returns how many it minted before the mint was
/// refused. When the funding budget's `channel` field bounded the child (slice 1b stamps it onto the
/// child's `channel_cap`), the mint fails closed at exactly `channel / PIPE_CAP` pipes; an unbounded
/// channel only stops when the handle table fills (many more). fuel/mem/spawn are all unbounded
/// (`-1`) so the ONLY thing constraining the child is its channel ceiling. The parent joins and
/// returns the child's count (post-spawn budget reads are 0 — consumed at commit — so the
/// `run_budgeted`-shaped `join*1000 + fuel + mem` collapses to `count*1000`).
fn run_channel_bounded_child(channel: i64) -> Result<Vec<Value>, temen_interp::Trap> {
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
  vf32 = i64.const 0
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
  vneg = i64.const -1
  return vneg
  }}
}}
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  vn0 = i64.const 0
  br 1(vn0)
}}
block 1 (vn: i64) {{
  vz = i32.const 0
  vfds = i64.const 20480
  vr = call.cap 4294967295 16 (i64) -> (i32) vz (vfds)
  vrz = i32.const 0
  vfail = i32.lt_s vr vrz
  br_if vfail 2(vn) 3(vn)
}}
block 2 (vnf: i64) {{
  return vnf
}}
block 3 (vnok: i64) {{
  vone = i64.const 1
  vn2 = i64.add vnok vone
  br 1(vn2)
  }}
}}
"#,
        f0 = (1u64 << 32) as i64,
        f16 = (16u64 | (0xFFFF_FFFFu64 << 32)) as i64,
        stores = store_record(17408),
    );
    let m = parse_module(&src).expect("parse");
    verify_module(&m).expect("verify");
    let mut host = Host::new();
    let ih = host.grant_instantiator(0, 128 << 10);
    // fuel/mem/spawn unbounded — only `channel` constrains the child.
    let bh = host.grant_budget_channel(-1, -1, -1, channel);
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

/// #989 slice 1b — a `Budget` split's `channel` field bounds a §14 child's host-served pipe memory:
/// a channel of exactly 2×`PIPE_CAP` lets the child mint exactly two pipes before the third fails
/// closed (`-EMFILE`), so the parent's join reports `2` (× 1000). An **unbounded** channel (`-1`,
/// the default) mints far more (until the handle table fills), so the exact `2` is proof the
/// funding budget's ceiling propagated onto the child's `channel_cap` at spawn.
#[test]
fn channel_budget_bounds_a_spawned_childs_pipe_mints() {
    assert_eq!(
        run_channel_bounded_child(2 * PIPE_CAP),
        Ok(vec![Value::I64(2_000)]),
        "a 2×PIPE_CAP channel budget bounds the child to exactly two pipe mints"
    );
    // A one-pipe ceiling bounds it to a single mint — the ceiling scales with the budget.
    assert_eq!(
        run_channel_bounded_child(PIPE_CAP),
        Ok(vec![Value::I64(1_000)]),
        "a 1×PIPE_CAP channel budget bounds the child to exactly one pipe mint"
    );
    // Unbounded channel: the child mints well past two (handle-table bound, not channel) — so the
    // bounded cases above are enforcing the cap, not hitting some incidental limit at two.
    let unbounded = run_channel_bounded_child(-1).expect("unbounded spawn runs");
    let n = match unbounded.as_slice() {
        [Value::I64(v)] => v / 1000,
        other => panic!("unexpected unbounded result: {other:?}"),
    };
    assert!(
        n > 2,
        "an unbounded channel mints more than two pipes (got {n}) — the bound above is real"
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

// ===== #744 (EXEC.md row 4) — guest-served exec: a parent serves its child's "exec" ==============

/// #744 — the **mediation-consistent guest-served exec** (EXEC.md row 4): the parent grants its child a
/// `"exec"` capability backed by **its own code**, and the child is none-the-wiser. The grant is a
/// named-grant record whose `handle` carries `GRANT_SERVE_LIVE_TAG` over the parent's impl-export
/// index (export 0 here, interface `"exec"` with one op `run`) — the record's reserved `flags` word
/// stays 0 and ignored, as the ABI promises: the spawn installs into the CHILD a
/// live-callee offer whose callee is the parent's running powerbox. The child resolves `"exec"` by name
/// and calls `run(40, 2)` — which parks it until the PARENT's `svc.wait` serve loop runs handler func 2
/// over the parent's live world and replies `42`. The parent joins the child and returns
/// `join*100 + served` = `42*100 + 1` = `4201`. The parent never holds a self-referential cap (the
/// offer exists only in the child's table), so there is no reference cycle and nothing to mis-call.
///
/// `handle` is a parameter so the fail-closed edge is pinned too: a tag over an impl-export the
/// parent does not have is refused at spawn (`CapFault`) before any child state is built.
fn serve_live_src(handle: i32) -> String {
    format!(
        r#"memory 17
type 0 func (i64, i64) -> (i64)
type 1 interface {{ run: 0 }}
export 0 interface "exec" 1 {{ run: 2 }}

func (i32) -> (i64) {{
block 0 (vinst: i32) {{
  ; the grant name "exec" at 16484 (parent window, below the carve)
  ce = i32.const 101
  cx = i32.const 120
  cc = i32.const 99
  p0 = i64.const 16484
  i32.store8 p0 ce
  p1 = i64.const 16485
  i32.store8 p1 cx
  p2 = i64.const 16486
  i32.store8 p2 ce
  p3 = i64.const 16487
  i32.store8 p3 cc
  ; one grant record at 16384: {{name_off=16484, name_len=4, handle=TAG|export, flags=0 (reserved, ignored)}}
  g0 = i64.const 16384
  n16484 = i32.const 16484
  i32.store g0 n16484
  g4 = i64.const 16388
  n4 = i32.const 4
  i32.store g4 n4
  g8 = i64.const 16392
  vhandle = i32.const {handle}
  i32.store g8 vhandle
  g12 = i64.const 16396
  z0 = i32.const 0
  i32.store g12 z0
  ; op-17 record at 17408: entry 1, carve off=65536 size_log2=16, no pager, module=-1 (self), no budget,
  ; quota 0, grants=(16384, 1)
  vf0 = i64.const {f0}
  vf8 = i64.const 65536
  vf16 = i64.const {f16}
  vf24 = i64.const 4294967295
  vf32 = i64.const 0
  vf40 = i64.const 16384
  vf48 = i64.const 1
{stores}
  vrp = i64.const 17408
  vch = call.cap 6 17 (i64) -> (i32) vinst (vrp)
  vzero = i32.const 0
  visneg = i32.lt_s vch vzero
  br_if visneg 2(vch) 1(vinst, vch)
}}
block 1 (vinst1: i32, vch1: i32) {{
  ; serve the child's one `exec.run` call: OUR handler (func 2) answers over OUR live world
  vz = i32.const 0
  vn = call.cap 4294967295 10 () -> (i64) vz ()
  vj = call.cap 6 1 (i32) -> (i64) vinst1 (vch1)
  k = i64.const 100
  t0 = i64.mul vj k
  t1 = i64.add t0 vn
  return t1
}}
block 2 (vbad: i32) {{
  vb64 = i64.extend_i32_s vbad
  return vb64
  }}
}}
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  ; re-materialize "exec" in OUR window and resolve the live self-serve grant by name
  ce = i32.const 101
  cx = i32.const 120
  cc = i32.const 99
  a0 = i64.const 16484
  i32.store8 a0 ce
  a1 = i64.const 16485
  i32.store8 a1 cx
  a2 = i64.const 16486
  i32.store8 a2 ce
  a3 = i64.const 16487
  i32.store8 a3 cc
  len4 = i64.const 4
  hexec = self.resolve a0 len4
  ; `run(40, 2)` through it: parks us until the PARENT's serve loop answers with its own code
  va = i64.const 40
  vb = i64.const 2
  vr = call.cap 268435456 0 (i64, i64) -> (i64) hexec (va, vb)
  return vr
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (va: i64, vb: i64) {{
  s = i64.add va vb
  return s
  }}
}}
"#,
        handle = handle,
        f0 = (1u64 << 32) as i64,
        f16 = (16u64 | (0xFFFF_FFFFu64 << 32)) as i64,
        stores = store_record(17408),
    )
}

fn run_serve_live(handle: i32) -> Result<Vec<Value>, temen_interp::Trap> {
    let src = serve_live_src(handle);
    let m = parse_module(&src).expect("parse");
    verify_module(&m).expect("verify");
    let am = std::sync::Arc::new(m);
    let mut host = Host::new();
    // The parent's impl-exports (what `GRANT_SERVE_LIVE` names and `offer_shape` resolves) live on
    // its registered self module — the same seeding the serve-loop and `child_offer` harnesses do.
    host.set_self_module(&am);
    let ih = host.grant_instantiator(0, 128 << 10);
    let mut fuel = 50_000_000u64;
    let (res, _) =
        run_capture_reserved_with_host(&am, 0, &[Value::I32(ih)], &mut fuel, &[], 0, &mut host);
    res
}

/// **The #744 pin**: the child's `exec.run(40, 2)` is answered by the PARENT's own handler over the
/// parent's live world — `join*100 + served` = `4201`. This is the guest-served exec backend
/// (EXEC.md row 4) end to end on the tree-walker oracle: mint-at-spawn into the child only, caller
/// parks, parent serves, reply, join.
#[test]
fn a_parent_serves_its_childs_exec_with_its_own_code() {
    assert_eq!(
        run_serve_live(temen_interp::GRANT_SERVE_LIVE_TAG as i32),
        Ok(vec![Value::I64(4_201)]),
        "child's exec.run(40,2) → parent's handler replies 42; parent served 1 and joined 42"
    );
}

/// The fail-closed edge: a tagged handle naming an impl-export the parent does NOT have (export 7;
/// only export 0 exists) is refused at spawn (`CapFault`) — the shape must resolve before any child
/// state is built, never a dangling live offer.
#[test]
fn a_live_self_serve_grant_of_a_missing_export_refuses_the_spawn() {
    assert!(
        matches!(
            run_serve_live((temen_interp::GRANT_SERVE_LIVE_TAG | 7) as i32),
            Err(temen_interp::Trap::CapFault)
        ),
        "a live self-serve grant of a nonexistent export fails the spawn closed"
    );
}
