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

#[path = "../../temen-interp/tests/support/rec.rs"]
mod rec;

use temen_interp::{run_capture_reserved_with_host, Host, StreamRole, Value};
use temen_ir::SpawnRec;
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
/// supplies 123 — the carve twin of `a_detached_demand_child_is_served_by_a_pager_that_cannot_address_it` (exit 1123). The child
/// faults at 16 KiB: its carve reserves the NULL guard below that like any window (#1206), and a guard
/// fault is fatal, never the recoverable kind a pager services.
/// The pager (func 2) follows the #1862 contract: it gets the child's fault address **in the child's
/// own coordinates**, fills its own buffer (here `addr + 16384`, clear of the carve at 64 KiB), and
/// replies with that address. It stores `123 + (addr >> 16)`: the child's fault at 16 KiB reads
/// `123`, where a parent-window address (64 KiB higher) would make it `124`. The runtime copies the page around it into the child. It never writes
/// the child's memory, so the same pager serves a child it cannot address.
fn record_pager_program() -> String {
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
  vaddr = i64.const 16384
  vb = i32.load8_u vaddr
  vbw = i64.extend_i32_u vb
  return vbw
  }}
}}

func 2 (i64) -> (i64) {{
block 0 (vaddr: i64) {{
  voff = i64.const 16384
  vsrc = i64.add vaddr voff
  vsh = i64.const 16
  vhi = i64.shr_u vaddr vsh
  vhi32 = i32.wrap_i64 vhi
  vbase = i32.const 123
  vb = i32.add vbase vhi32
  i32.store8 vsrc vb
  return vsrc
  }}
}}
",
        f0 = f0,
        f16 = f16,
        f24 = f24,
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

/// #1217, the granted-offer shape: the root spawns a **server** child (func 1: one `svc.wait`,
/// returning its count), mints a `child_offer` over its export, and spawns a **guest** child
/// (func 3, `guest`) with that offer re-granted by name as `"fork"`; then `join`s the server and
/// exits its count. Both children are detached, paid from the root's `"budget"`; the `"fork"` name is
/// a data segment, so the guest finds it in its own window too.
fn granted_client_program(guest: &str) -> String {
    let guest_rec = SpawnRec {
        grants_ptr: 16640,
        grants_n: 1,
        ..SpawnRec::v1(3)
    };
    format!(
        "\
memory 17
data 16384 \"vm\"
data 16400 \"budget\"
data 16684 \"fork\"
{server_rec}{guest_rec}type 0 func (i64) -> (i64)
type 1 interface {{ op: 0 }}
export 0 interface \"fork\" 1 {{ op: 2 }}
import 0 \"exit\" (i32) -> ()
func 0 () -> () {{
block 0 () {{
  vp = i64.const 16384
  vl = i64.const 2
  v0 = self.resolve vp vl
  vbp = i64.const 16400
  vbl = i64.const 6
  vb = self.resolve vbp vbl
  vsb = i64.const 17436
  i32.store vsb vb
  vsp = i64.const 17408
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
  vgb = i64.const 17532
  i32.store vgb vb
  vgp = i64.const 17504
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
        server_rec = rec::segment(17408, &SpawnRec::v1(1)),
        guest_rec = rec::segment(17504, &guest_rec),
    )
}

/// [`granted_client_program`] in both spawn spellings: the op-17 records, and op 15's positional
/// form (retiring, #2067), which every engine runs natively today — the Cranelift JIT included, where
/// the record spelling of a module with impl-exports still folds to the tree-walker (#744).
fn granted_client_programs(guest: &str) -> [String; 2] {
    let rec = granted_client_program(guest);
    // `dst = instantiate_detached(budget, self, grants_ptr, grants_n, entry, 17, 0)`, its operands
    // named with `x` so the two spawns' names stay distinct.
    let op15 = |dst: &str, x: &str, entry: u32, grants_ptr: u64, grants_n: u64| {
        format!(
            "vb{x} = i64.extend_i32_u vb\n  vm{x} = i64.const -1\n  vgp{x} = i64.const {grants_ptr}\n  \
             vgn{x} = i64.const {grants_n}\n  ve{x} = i64.const {entry}\n  vsl{x} = i64.const 17\n  \
             vq{x} = i64.const 0\n  {dst} = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 \
             (vb{x}, vm{x}, vgp{x}, vgn{x}, ve{x}, vsl{x}, vq{x})"
        )
    };
    let positional = rec
        .replace(
            "vs = call.cap 6 17 (i64) -> (i32) v0 (vsp)",
            &op15("vs", "s15", 1, 0, 0),
        )
        .replace(
            "vg = call.cap 6 17 (i64) -> (i32) v0 (vgp)",
            &op15("vg", "g15", 3, 16640, 1),
        );
    assert_eq!(
        positional.matches("call.cap 6 15").count(),
        2,
        "both spawns respelled"
    );
    [rec, positional]
}

/// The healthy exchange, pinning the program shape: the guest resolves `"fork"` and calls it once
/// (the handler returns 7), the server's `svc.wait` counts 1, the root exits 1.
#[test]
fn granted_client_that_calls_once_is_served() {
    for src in granted_client_programs(
        "  vnp = i64.const 16684
  vl4 = i64.const 4
  vfork = self.resolve vnp vl4
  vz8 = i64.const 0
  vr = call.cap 268435456 0 (i64) -> (i64) vfork (vz8)
  return vr",
    ) {
        for b in BACKENDS {
            assert_eq!(run_detached(b, &src).expect("run"), 1, "{b:?}");
        }
    }
}

/// #1217: the guest traps before calling. The server's only client is gone, so its `svc.wait`
/// returns `0` (never parks forever), it returns, and the root's `join` delivers `0`.
#[test]
fn granted_client_that_dies_before_calling_releases_the_server() {
    for src in granted_client_programs("  unreachable") {
        for b in BACKENDS {
            assert_eq!(run_detached(b, &src).expect("run"), 0, "{b:?}");
        }
    }
}

/// #1228: the **daemon** shape #1217 cannot cover. The server is written as an unconditional
/// `loop { svc.wait }` (the fork-manager server shape), not a wait-once-and-return, so the
/// client-death token (#1217) gives it one spurious `0` and the loop re-parks it, untimed,
/// forever. With its only client gone and the root `join`ing it, the run is a genuine
/// join-deadlock: the root parked in `join`, the daemon parked in `svc.wait`, nothing else live
/// and no external wake source. INVARIANTS.md #9 forbids a hang, so the run must **trap**
/// `ThreadFault`. The cooperative bytecode driver already does; this pins the tree-walk executor
/// (#1228) and the Cranelift JIT, whose `svc.wait` park and `join` count for its deadlock verdict
/// (#2173, #1820), to the identical outcome.
#[test]
fn joined_daemon_whose_client_is_gone_deadlocks_to_threadfault() {
    for base in granted_client_programs("  unreachable") {
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
            let e = run_detached(b, &src).expect_err("the join-deadlock must trap, not hang");
            assert!(
                e.contains("ThreadFault"),
                "{b:?}: the all-parked join-deadlock traps ThreadFault, got: {e}"
            );
        }
    }
}

/// #2173: a call nothing will ever serve. The guest calls the server's `"fork"` offer, but the
/// server never reaches a service point — it waits forever on a word nothing stores — while the root
/// `join`s it. Every vCPU is parked: the caller on its reply, the server in its wait, the root in its
/// `join`. So the run traps `ThreadFault` on every engine. The JIT's caller wait used to sit outside
/// its deadlock count, and the run hung.
#[test]
fn a_call_nothing_will_serve_deadlocks_to_threadfault() {
    let guest = "  vnp = i64.const 16684
  vl4 = i64.const 4
  vfork = self.resolve vnp vl4
  vz8 = i64.const 0
  vr = call.cap 268435456 0 (i64) -> (i64) vfork (vz8)
  return vr";
    for base in granted_client_programs(guest) {
        let server =
            "block 0 (v0: i64) {\n  vz = i32.const 0\n  vs = svc.wait vz\n  return vs\n  }";
        assert!(
            base.contains(server),
            "the server's body is the rewrite target"
        );
        let src = base.replace(
            server,
            "block 0 (v0: i64) {\n  va = i64.const 16392\n  vexp = i32.const 0\n  vinf = i64.const -1\n  \
             vst = i32.atomic.wait va vexp vinf\n  vst64 = i64.extend_i32_u vst\n  return vst64\n  }",
        );
        for b in BACKENDS {
            let e = run_detached(b, &src).expect_err("the deadlock must trap, not hang");
            assert!(
                e.contains("ThreadFault"),
                "{b:?}: the all-parked deadlock traps ThreadFault, got: {e}"
            );
        }
    }
}

/// #1820: a root `join`ing a detached child that waits forever on a word nothing stores is a
/// deadlock — the root parked in its `join`, the child in its wait, nothing else live — so the run
/// traps `ThreadFault` on every engine. The JIT's `join` used to stay out of its deadlock count, and
/// nothing decided the verdict: the run hung.
#[test]
fn a_root_joining_a_child_that_waits_forever_deadlocks_to_threadfault() {
    let base = detached_record_program();
    let body = format!(
        "  va = i64.const {}\n  vb = i32.load8_u va\n  vbw = i64.extend_i32_u vb\n  vk = i64.const 100\n  \
         vr = i64.add vbw vk\n  return vr\n",
        temen_ir::module_args_base()
    );
    assert!(
        base.contains(&body),
        "the child's body is the rewrite target"
    );
    let src = base.replace(
        &body,
        "  vaddr = i64.const 16392\n  vexp = i32.const 0\n  vinf = i64.const -1\n  \
         vst = i32.atomic.wait vaddr vexp vinf\n  vst64 = i64.extend_i32_u vst\n  return vst64\n",
    );
    for b in BACKENDS {
        let e = run_detached(b, &src).expect_err("the join-deadlock must trap, not hang");
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

/// §3c.2's **narrowed** gap, pinned: a budget with a bounded `spawn` ceiling stays a probeable
/// `-EINVAL` on the native lane while the interpreter funds the carve child — budget intact either
/// way. Bounded-zero fuel is the same class. The gap goes with the carve (#1867): a detached child
/// is charged its budget's `spawn` on every engine.
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
/// stays 0 and ignored, as the ABI promises. The parent spawns its own module's func 1 detached
/// (`positional`: through op 15's argument form instead of a v1 record); the spawn installs into the
/// CHILD a live-callee offer whose callee is the parent's running powerbox. The child resolves
/// `"exec"` by name and calls `run(40, 2)` — which parks it until the PARENT's `svc.wait` serve loop
/// runs handler func 2 over the parent's live world and replies `42`. The parent joins the child and
/// exits `join*100 + served` = `42*100 + 1` = `4201`. The parent never holds a self-referential cap
/// (the offer exists only in the child's table), so there is no reference cycle and nothing to
/// mis-call.
///
/// `handle` is a parameter so the fail-closed edge is pinned too: a tag over an impl-export the
/// parent does not have is refused at spawn (`CapFault`) before any child state is built.
fn serve_live_program(handle: i32, positional: bool) -> String {
    let child = SpawnRec {
        grants_ptr: 16640,
        grants_n: 1,
        ..SpawnRec::v1(1)
    };
    let spawn = if positional {
        "vb64 = i64.extend_i32_s vb
  vself = i64.const -1
  vgp = i64.const 16640
  vg1 = i64.const 1
  vz64 = i64.const 0
  vch = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) vh (vb64, vself, vgp, vg1, vg1, vz64, vz64)"
    } else {
        "vrp = i64.const 17408
  vch = call.cap 6 17 (i64) -> (i32) vh (vrp)"
    };
    format!(
        "\
memory 17
data 16384 \"vm\"
data 16400 \"budget\"
data 16684 \"exec\"
{rec}type 0 func (i64, i64) -> (i64)
type 1 interface {{ run: 0 }}
export 0 interface \"exec\" 1 {{ run: 2 }}
import 0 \"exit\" (i32) -> ()
func 0 () -> () {{
block 0 () {{
  vp = i64.const 16384
  vl = i64.const 2
  vh = self.resolve vp vl
  vbp = i64.const 16400
  vbl = i64.const 6
  vb = self.resolve vbp vbl
  vba = i64.const {budget_at}
  i32.store vba vb
  ; the one grant record at 16640: {{name_off, name_len, handle = TAG | export, flags = 0}}
  vg = i64.const 16640
  vgn = i32.const 16684
  i32.store vg vgn
  vgl = i32.const 4
  i32.store vg vgl offset=4
  vgh = i32.const {handle}
  i32.store vg vgh offset=8
  {spawn}
  vz = i32.const 0
  vneg = i32.lt_s vch vz
  br_if vneg 2(vch) 1(vh, vch)
}}
block 1 (vh1: i32, vch1: i32) {{
  ; serve the child's one `exec.run` call: OUR handler (func 2) answers over OUR live world
  vz1 = i32.const 0
  vn = svc.wait vz1
  vj = call.cap 6 1 (i32) -> (i64) vh1 (vch1)
  vk = i64.const 100
  vt = i64.mul vj vk
  vs = i64.add vt vn
  vc = i32.wrap_i64 vs
  call.import 0 (vc)
  unreachable
}}
block 2 (verr: i32) {{
  call.import 0 (verr)
  unreachable
  }}
}}
func 1 (i64) -> (i64) {{
block 0 (v0: i64) {{
  ; resolve the live self-serve grant by name (`\"exec\"` is a data segment, so it is in our window
  ; too) and call `run(40, 2)` through it: parks us until the PARENT's serve loop answers
  vnp = i64.const 16684
  vl4 = i64.const 4
  vexec = self.resolve vnp vl4
  va = i64.const 40
  vb = i64.const 2
  vr = call.cap 268435456 0 (i64, i64) -> (i64) vexec (va, vb)
  return vr
  }}
}}
func 2 (i64, i64) -> (i64) {{
block 0 (va: i64, vb: i64) {{
  s = i64.add va vb
  return s
  }}
}}
",
        rec = rec::segment(17408, &child),
        budget_at = 17408 + rec::BUDGET_AT,
    )
}

/// **The #744 pin**: the child's `exec.run(40, 2)` is answered by the PARENT's own handler over the
/// parent's live world — `join*100 + served` = `4201` — the guest-served exec backend (EXEC.md row 4)
/// end to end: mint-at-spawn into the child only, caller parks, parent serves, reply, join. Every
/// backend agrees: the Cranelift JIT runs the parent itself, serving the child's call over the root's
/// shared cell; the bytecode engine still declines it to the tree-walker.
#[test]
fn a_parent_serves_its_childs_exec_with_its_own_code() {
    let src = serve_live_program(temen_interp::GRANT_SERVE_LIVE_TAG as i32, false);
    for b in BACKENDS {
        // Both transports: the child's call queued for the parent's serve loop, and (handoff on,
        // the default) run inline on the child's thread when it finds the parent parked.
        for handoff in [true, false] {
            let cfg = RunConfig {
                handoff,
                ..RunConfig::default()
            };
            assert_eq!(
                run_detached_with(b, &src, 1 << 20, cfg),
                Ok(4_201),
                "{b:?}, handoff {handoff}: the child's exec.run(40,2) → the parent's handler \
                 replies 42; the parent served 1 and joined 42"
            );
        }
    }
}

/// #1217 on the self-serve grant: a child that holds the parent's live offer but finishes without
/// calling it releases the parent's `svc.wait`, which answers `0` instead of waiting on a caller that
/// is gone; the parent then joins the child's `7` — `join*100 + served` = `700` on every backend,
/// whichever of the child's end and the parent's wait comes first.
#[test]
fn a_child_that_never_calls_releases_its_parents_wait() {
    let src = serve_live_program(temen_interp::GRANT_SERVE_LIVE_TAG as i32, false).replace(
        "  vr = call.cap 268435456 0 (i64, i64) -> (i64) vexec (va, vb)\n  return vr\n",
        "  v7 = i64.const 7\n  return v7\n",
    );
    assert!(src.contains("return v7"), "the rewrite must apply");
    for b in BACKENDS {
        assert_eq!(
            run_detached(b, &src),
            Ok(700),
            "{b:?}: the wait answers 0 once the child is gone; the join gets 7"
        );
    }
}

/// The fail-closed edge: a tagged handle naming an impl-export the parent does NOT have (export 7;
/// only export 0 exists) is refused at spawn (`CapFault`) — the shape must resolve before any child
/// state is built, never a dangling live offer.
#[test]
fn a_live_self_serve_grant_of_a_missing_export_refuses_the_spawn() {
    let src = serve_live_program((temen_interp::GRANT_SERVE_LIVE_TAG | 7) as i32, false);
    for b in BACKENDS {
        let r = run_detached(b, &src);
        assert!(
            matches!(&r, Err(e) if e.contains("CapFault")),
            "{b:?}: a live self-serve grant of a nonexistent export fails the spawn closed: {r:?}"
        );
    }
}

/// Op 15's positional form (retiring, #2067) carries the tag too: every detached spawn follows one
/// rule. The Cranelift JIT runs this serving parent natively over a shared cell it can be called
/// back through; the bytecode engine declines it to the tree-walker.
#[test]
fn op_15_carries_a_live_self_serve_grant_too() {
    let src = serve_live_program(temen_interp::GRANT_SERVE_LIVE_TAG as i32, true);
    for b in BACKENDS {
        assert_eq!(
            run_detached(b, &src).expect("run"),
            4201,
            "{b:?}: our handler answered the child's exec over op 15 too"
        );
    }
}

/// #1815: a client's death must reach the server's `svc.wait` even when another vCPU of the server
/// domain is parked **under the same key** as a non-consumer. Here that vCPU is a server thread
/// idling in `cont.resume.block` on a futex-parked fiber (keyed on its own domain, I48). The client
/// finishes while that resumer is parked and the serve loop is not yet at `svc.wait` (it is joining
/// the thread). The death used to spend its one-shot wake on the resumer — which re-parks, its
/// fiber still blocked — and leave no token, so the serve loop's later `svc.wait` parked forever.
///
/// The ordering is forced, not hoped for: the fiber raises a flag and futex-parks, the root waits
/// for the flag (plus a settle for the resumer to file its park), and only then spawns the client,
/// which returns at once. After joining the client the root releases the fiber, the thread
/// returns, and the serve loop reaches `svc.wait` with its only client long gone: it must return
/// `0`, so the root exits `0`. The flag (byte 0) and the fiber's go-cell (byte 8) are in a
/// `SharedRegion` the root mints and maps at 65536, and pre-maps into the server's window at the
/// same offset.
#[test]
fn client_death_releases_the_server_while_a_resumer_is_parked_under_its_key() {
    let src = granted_client_program("  vz9 = i64.const 0\n  return vz9")
        .replace(
            "data 16400 \"budget\"\n",
            "data 16400 \"budget\"\ndata 16416 \"as\"\n",
        )
        .replace(
            "block 0 (v0: i64) {\n  vz = i32.const 0\n  vs = svc.wait vz\n  return vs\n  }",
            "block 0 (v0: i64) {
  vz = i64.const 0
  vt = thread.spawn 4 vz vz
  vj = thread.join vt
  vz32 = i32.const 0
  vs = svc.wait vz32
  return vs
  }",
        )
        .replace(
            "  vsp = i64.const 17408\n",
            "  vap = i64.const 16416
  val = i64.const 2
  vas = self.resolve vap val
  vlen = i64.const 65536
  vrh64 = call.cap 5 5 (i64) -> (i64) vas (vlen)
  vrh = i32.wrap_i64 vrh64
  vwo = i64.const 65536
  vro = i64.const 0
  vprot = i32.const 3
  vmap = call.cap 4 0 (i64, i64, i64, i32) -> (i64) vrh (vwo, vro, vlen, vprot)
  vreg = i64.const 17480
  i32.store vreg vrh
  vcoff = i64.const 17488
  i64.store vcoff vwo
  vsp = i64.const 17408
",
        )
        .replace(
            "  vs = call.cap 6 17 (i64) -> (i32) v0 (vsp)\n",
            "  vs = call.cap 6 17 (i64) -> (i32) v0 (vsp)
  vfl = i64.const 65536
  vfe = i32.const 0
  vinf = i64.const -1
  vfw = i32.atomic.wait vfl vfe vinf
  vdead = i64.const 65552
  vms = i64.const 50000000
  vsettle = i32.atomic.wait vdead vfe vms
",
        )
        .replace(
            "  vg = call.cap 6 17 (i64) -> (i32) v0 (vgp)\n",
            "  vg = call.cap 6 17 (i64) -> (i32) v0 (vgp)
  vjg = call.cap 6 1 (i32) -> (i64) v0 (vg)
  vgo = i64.const 65544
  vgone = i32.const 1
  i32.store vgo vgone
  vgw = atomic.notify vgo vgone
",
        )
        + "func 4 (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vf = ref.func 5
  vz = i64.const 0
  vk = cont.new vf vz
  vs, vv = cont.resume.block vk vz
  return vv
  }
}
func 5 (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vflag = i64.const 65536
  vone = i32.const 1
  i32.store vflag vone
  vw = atomic.notify vflag vone
  vgo = i64.const 65544
  vexp = i32.const 0
  vto = i64.const -1
  vst = i32.atomic.wait vgo vexp vto
  vst64 = i64.extend_i32_s vst
  return vst64
  }
}
";
    assert!(
        src.contains("data 16416 \"as\"")
            && src.contains("thread.spawn 4")
            && src.contains("vcoff")
            && src.contains("vsettle")
            && src.contains("vjg"),
        "the rewrites must apply"
    );
    for b in BACKENDS {
        assert_eq!(run_detached(b, &src).expect("run"), 0, "{b:?}");
    }
}

// ---- #1863: the v1 (detached) record — op 15 as data ----

/// Stores building an 88-byte v1 record at `at` from the i64 values `vr0`..`vr80` in scope (one per
/// 8-byte word, named by offset).
fn store_record_v1(at: u64) -> String {
    (0..11)
        .map(|k| {
            let o = k * 8;
            format!(
                "  vra{o} = i64.const {a}\n  i64.store vra{o} vr{o}\n",
                a = at + o
            )
        })
        .collect()
}

/// A root that spawns **its own module** (`module = -1`) detached through a v1 record: the child's
/// window is its own (`size_log2` 17, the module's declared memory), funded by the `"budget"` grant,
/// with a one-byte args payload (`*` = 42) the child reads back at `module_args_base`. The child
/// (func 1) returns `payload + 100`; the root exits with the join result. `pager` is `u32::MAX` here;
/// `extra` rewrites let a test vary the record.
fn detached_record_program() -> String {
    let args_base = temen_ir::module_args_base();
    format!(
        "\
memory 17
data 16384 \"vm\"
data 16400 \"budget\"
data 16416 \"*\"
import 0 \"exit\" (i32) -> ()

func 0 () -> () {{
block 0 () {{
  vp = i64.const 16384
  vl = i64.const 2
  vh = self.resolve vp vl
  vbp = i64.const 16400
  vbl = i64.const 6
  vb = self.resolve vbp vbl
  vb64 = i64.extend_i32_u vb
  v32 = i64.const 32
  vbsh = i64.shl vb64 v32
  vself = i64.const 4294967295
  vr0 = i64.const 4294967297
  vr8 = i64.const 0
  vr16 = i64.const -4294967279
  vr24 = i64.or vbsh vself
  vr32 = i64.const 0
  vr40 = i64.const 0
  vr48 = i64.const 0
  vr56 = i64.const 16416
  vr64 = i64.const 1
  vr72 = i64.const 4294967295
  vr80 = i64.const 0
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
  va = i64.const {args_base}
  vb = i32.load8_u va
  vbw = i64.extend_i32_u vb
  vk = i64.const 100
  vr = i64.add vbw vk
  return vr
  }}
}}
",
        stores = store_record_v1(17408),
    )
}

/// Run with the Instantiator (`"vm"`), an AddressSpace (`"as"`) and a detached-window budget
/// (`"budget"`, 1 MiB of `Budget.mem`).
fn run_detached(backend: Backend, src: &str) -> Result<i32, String> {
    run_detached_with(backend, src, 1 << 20, RunConfig::default())
}

/// [`run_detached`] with a `mem`-byte budget, under `cfg`. The run gets its own thread and a
/// deadline: the #1217 pins guard against a *hang* (a pager parked for a child that is gone), so a
/// regression must fail the test, not stall the binary until CI's timeout.
fn run_detached_with(backend: Backend, src: &str, mem: u64, cfg: RunConfig) -> Result<i32, String> {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let registry = Imports::new().provide("exit", HostCap::exit());
        let inst = instantiate_with_imports(m, registry).expect("instantiate");
        let r = inst.run_with_caps(
            backend,
            &cfg,
            &[
                (
                    "vm",
                    HostCap::custom(6, 0, |h, win| h.grant_instantiator(0, win)),
                ),
                ("budget", HostCap::detached_budget(mem)),
                (
                    "as",
                    HostCap::custom(5, 0, |h, win| h.grant_address_space(0, win)),
                ),
            ],
        );
        let _ = tx.send(r.map(|r| r.outcome).map_err(|e| e.to_string()));
    });
    let outcome = rx
        .recv_timeout(std::time::Duration::from_secs(60))
        .unwrap_or_else(|e| panic!("{backend:?}: the run did not finish: {e}"))?;
    match outcome {
        Outcome::Exited(code) => Ok(code),
        other => Err(format!("unexpected outcome {other:?}")),
    }
}

/// #1863: a v1 record spawns the parent's own module into a window of its own, and the args
/// payload arrives at the child's `module_args_base` — identically on every tier.
#[test]
fn a_v1_record_spawns_self_detached_with_an_args_payload() {
    let src = detached_record_program();
    for b in BACKENDS {
        assert_eq!(run_detached(b, &src).expect("run"), 142, "{b:?}");
    }
}

/// A v1 record whose `size_log2` is `0` asks for the module's declared window — a spawner holding a
/// module it did not build need not know its size. A nonzero size that is not the declared memory is
/// still refused, probeably: the root exits 7 when the spawn returns a negative handle.
#[test]
fn a_v1_record_with_size_zero_gets_the_declared_window() {
    let base = detached_record_program();
    // size_log2 0 | pager u32::MAX
    let src = base.replace(
        "  vr16 = i64.const -4294967279\n",
        "  vr16 = i64.const -4294967296\n",
    );
    assert!(src.contains("-4294967296"), "the rewrite must apply");
    for b in BACKENDS {
        assert_eq!(run_detached(b, &src).expect("run"), 142, "{b:?}");
    }
    // size_log2 16 against a declared 17: refused before any join.
    let wrong = base
        .replace(
            "  vr16 = i64.const -4294967279\n",
            "  vr16 = i64.const -4294967280\n",
        )
        .replace(
            "  vj = call.cap 6 1 (i32) -> (i64) vh (vch)\n  vc = i32.wrap_i64 vj\n",
            "  vz0 = i32.const 0\n  vneg = i32.lt_s vch vz0\n  vseven = i32.const 7\n  vc = select vneg vseven vz0\n",
        );
    assert!(wrong.contains("vneg"), "the rewrite must apply");
    for b in BACKENDS {
        assert_eq!(run_detached(b, &wrong).expect("run"), 7, "{b:?}");
    }
}

/// #1863: v1's `off` is reserved — a record naming a carve offset is malformed, not a carve spawn.
#[test]
fn a_v1_record_with_an_offset_fails_closed() {
    let src =
        detached_record_program().replace("  vr8 = i64.const 0\n", "  vr8 = i64.const 65536\n");
    assert!(
        src.contains("vr8 = i64.const 65536"),
        "the rewrite must apply"
    );
    for b in BACKENDS {
        let e = run_detached(b, &src).expect_err("a v1 offset is refused");
        assert!(e.contains("CapFault"), "{b:?}: {e}");
    }
}

/// A root that serves its own detached demand child: the v1 record names impl export 0 as the pager,
/// and the root waits in `svc.wait` before it joins the child, then exits `byte + serves * 1000`. The
/// child (func 1) returns the byte at `fault`, its first touch of that page. The pager (func 2) gets
/// the fault address in the child's coordinates, stores 77 at `addr + 40000` in its own window, and
/// replies with that address (#1862).
fn detached_pager_program_at(fault: u64) -> String {
    let src = detached_record_program()
        .replace(
            "import 0 \"exit\" (i32) -> ()\n",
            "type 0 func (i64) -> (i64)\ntype 1 interface { page: 0 }\nexport 0 interface \"pager\" 1 { page: 2 }\nimport 0 \"exit\" (i32) -> ()\n",
        )
        // pager = impl export 0
        .replace("  vr16 = i64.const -4294967279\n", "  vr16 = i64.const 17\n")
        .replace(
            "  vj = call.cap 6 1 (i32) -> (i64) vh (vch)\n  vc = i32.wrap_i64 vj\n",
            "  vz = i32.const 0\n  vs = svc.wait vz\n  vj = call.cap 6 1 (i32) -> (i64) vh (vch)\n  vk1 = i64.const 1000\n  vsk = i64.mul vs vk1\n  vt = i64.add vj vsk\n  vc = i32.wrap_i64 vt\n",
        )
        .replace(
            &format!("  va = i64.const {}\n  vb = i32.load8_u va\n  vbw = i64.extend_i32_u vb\n  vk = i64.const 100\n  vr = i64.add vbw vk\n  return vr\n", temen_ir::module_args_base()),
            &format!("  va = i64.const {fault}\n  vb = i32.load8_u va\n  vbw = i64.extend_i32_u vb\n  return vbw\n"),
        )
        + "
func 2 (i64) -> (i64) {
block 0 (vaddr: i64) {
  voff = i64.const 40000
  vsrc = i64.add vaddr voff
  vb = i32.const 77
  i32.store8 vsrc vb
  return vsrc
  }
}
";
    assert!(
        src.contains("interface \"pager\"")
            && src.contains("vr16 = i64.const 17\n")
            && src.contains("svc.wait")
            && src.contains(&format!("va = i64.const {fault}\n")),
        "the rewrites must apply"
    );
    src
}

/// #1862 + #1863: a **detached** demand child — a window the pager cannot address — is served by a
/// pager that fills its own buffer and replies with its address. The child's first touch (at 20000)
/// faults into a `page(addr)` the root serves from `svc.wait`; the child reads 77; the root exits
/// `77 + 1 serve * 1000`.
#[test]
fn a_detached_demand_child_is_served_by_a_pager_that_cannot_address_it() {
    let src = detached_pager_program_at(20000);
    for b in BACKENDS {
        assert_eq!(run_detached(b, &src).expect("run"), 1077, "{b:?}");
    }
}

/// #744, the typed pager: a record's pager must be an export of exactly `{ page: (i64) -> (i64) }`,
/// the one shape a page fault's dispatch serves. Naming any other export — here one whose op has
/// another name, and one with a second op — fails the spawn closed on every backend.
#[test]
fn a_pager_that_is_not_a_page_export_refuses_the_spawn() {
    let ok = detached_pager_program_at(20000);
    let renamed = ok
        .replace("interface { page: 0 }", "interface { fetch: 0 }")
        .replace("\"pager\" 1 { page: 2 }", "\"pager\" 1 { fetch: 2 }");
    let two_ops = ok
        .replace("interface { page: 0 }", "interface { page: 0, peek: 0 }")
        .replace(
            "\"pager\" 1 { page: 2 }",
            "\"pager\" 1 { page: 2, peek: 2 }",
        );
    for src in [renamed, two_ops] {
        assert!(
            !src.contains("interface { page: 0 }"),
            "the rewrite must apply"
        );
        for b in BACKENDS {
            let r = run_detached(b, &src);
            assert!(
                matches!(&r, Err(e) if e.contains("CapFault")),
                "{b:?}: a non-pager export as the pager fails the spawn closed: {r:?}"
            );
        }
    }
}

/// #1815: a demand child's page fault must **wake** a pager parked at `svc.wait` when the fault is
/// not served by direct handoff. The fault arm enqueued the request and parked the child on the
/// reply ticket, but only the handoff path ever reached the pager — with handoff off (or a handoff
/// that found no serve loop to take) the parked pager slept through the request and the run
/// deadlocked. The fault is the same request a `call.cap` makes, so it takes the same
/// enqueue + `svc_wake` + park. Pinned handoff-off; handoff-on is the test above.
#[test]
fn a_page_fault_wakes_a_parked_pager_without_handoff() {
    let src = detached_pager_program_at(20000);
    let cfg = || RunConfig {
        handoff: false,
        ..RunConfig::default()
    };
    for b in BACKENDS {
        assert_eq!(
            run_detached_with(b, &src, 1 << 20, cfg()).expect("run"),
            1077,
            "{b:?}: the pager serves the fault"
        );
    }
}

/// #1206: a pager-bound child's fault **below its NULL guard** is fatal on every backend — never the
/// recoverable kind the pager services (the reserved region cannot be mapped). The parent is parked
/// in `svc.wait` for a request that never comes; #1217 releases it (its `svc.wait` returns `0`, the
/// no-progress answer) so its `join` surfaces the trap — the run traps instead of hanging.
#[test]
fn pager_child_guard_fault_is_fatal() {
    let src = detached_pager_program_at(0);
    for b in BACKENDS {
        assert!(
            run_detached(b, &src).is_err(),
            "{b:?}: a NULL fault in a paged child is fatal"
        );
    }
}

/// #1217: a demand child that **never faults** — it returns `5` without touching its window — must
/// not strand the pager parked in `svc.wait`. The wait returns `0` once the child is gone; the
/// `join` delivers `5`; the run exits `0 * 1000 + 5`.
#[test]
fn pager_child_that_never_faults_releases_the_parked_pager() {
    let src =
        detached_pager_program_at(0).replace("  vb = i32.load8_u va\n", "  vb = i32.const 5\n");
    assert!(src.contains("vb = i32.const 5\n"), "the rewrite must apply");
    for b in BACKENDS {
        assert_eq!(run_detached(b, &src).expect("run"), 5, "{b:?}");
    }
}

/// #1217, the other order: the child is already gone (joined) when the parent reaches `svc.wait`,
/// so nothing could ever wake it — the wait returns `0` immediately rather than parking.
#[test]
fn pager_svc_wait_after_the_child_is_joined_returns_zero() {
    let src = detached_pager_program_at(0)
        .replace("  vb = i32.load8_u va\n", "  vb = i32.const 5\n")
        .replace(
            "  vs = svc.wait vz\n  vj = call.cap 6 1 (i32) -> (i64) vh (vch)\n",
            "  vj = call.cap 6 1 (i32) -> (i64) vh (vch)\n  vs = svc.wait vz\n",
        );
    assert!(
        src.contains("vb = i32.const 5\n") && src.contains("(vch)\n  vs = svc.wait"),
        "the rewrites must apply"
    );
    for b in BACKENDS {
        assert_eq!(run_detached(b, &src).expect("run"), 5, "{b:?}");
    }
}

/// #1862: a pager reply names a source the pager's window can't supply — here 1 MiB past a buffer
/// it did fill, beyond its 128 KiB window. The fault is unserviceable, so the child dies as a
/// pagerless fault would (detect-and-kill), and the parent's `join` raises it: the run traps instead
/// of the child reading unsupplied bytes.
#[test]
fn a_pager_reply_outside_its_window_is_a_fatal_fault() {
    let src = detached_pager_program_at(20000).replace(
        "  return vsrc\n",
        "  vpast = i64.const 1048576\n  vbad = i64.add vsrc vpast\n  return vbad\n",
    );
    assert!(src.contains("return vbad"), "the rewrite must apply");
    for b in BACKENDS {
        let e = run_detached(b, &src).expect_err("an unsuppliable page is fatal");
        assert!(e.contains("MemoryFault"), "{b:?}: {e}");
    }
}

/// Owner, 2026-09-29: `Budget.mem` accounts **live** windows — a detached child's window goes back to
/// the budget that paid for it when the child ends. With a budget of exactly one window (2^17), the
/// program spawns itself, joins, and spawns itself again: both run (`142 + 142`). Before, the first
/// spawn spent the budget for good and the second was refused — a program could copy itself once,
/// ever.
#[test]
fn a_joined_detached_childs_window_returns_to_its_budget() {
    let src = detached_record_program().replace(
        "  vj = call.cap 6 1 (i32) -> (i64) vh (vch)\n  vc = i32.wrap_i64 vj\n",
        "  vj = call.cap 6 1 (i32) -> (i64) vh (vch)\n  vch2 = call.cap 6 17 (i64) -> (i32) vh (vrp)\n  vj2 = call.cap 6 1 (i32) -> (i64) vh (vch2)\n  vjs = i64.add vj vj2\n  vc = i32.wrap_i64 vjs\n",
    );
    assert!(src.contains("vch2"), "the rewrite must apply");
    for b in BACKENDS {
        assert_eq!(
            run_detached_with(b, &src, 1 << 17, RunConfig::default()).expect("run"),
            284,
            "{b:?}: the second child fits the window the first gave back"
        );
    }
}

/// #1944 — three generations of one `memory 17` module: the root splits its budget into
/// a node with a `mem` ceiling of `child_ceiling` and spawns func 1 detached, paid from it; func 1
/// spawns func 2 from its own `"budget"` — the node that paid for it — and returns `join + 20`, or the
/// refusal's `-errno`. Func 2 returns 3. The root exits with its child's result plus 100.
fn three_generations(child_ceiling: i64) -> String {
    format!(
        "memory 17
data 16384 \"vm\"
data 16400 \"budget\"
{r1}{r2}import 0 \"exit\" (i32) -> ()

func 0 () -> () {{
block 0 () {{
  vp = i64.const 16384
  vl = i64.const 2
  vh = self.resolve vp vl
  np = i64.const 16400
  nl = i64.const 6
  vroot = self.resolve np nl
  all = i64.const -1
  cap = i64.const {child_ceiling}
  vsub = call.cap 14 0 (i64, i64, i64) -> (i32) vroot (all, cap, all)
  bf = i64.const 17436
  i32.store bf vsub
  rp = i64.const 17408
  vch = call.cap 6 17 (i64) -> (i32) vh (rp)
  vj = call.cap 6 1 (i32) -> (i64) vh (vch)
  k = i64.const 100
  vr = i64.add vj k
  vc = i32.wrap_i64 vr
  call.import 0 (vc)
  unreachable
  }}
}}
func 1 (i64) -> (i64) {{
block 0 (va: i64) {{
  vinst = i32.wrap_i64 va
  np = i64.const 16400
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
block 2 (vi: i32, vc: i32) {{
  vj = call.cap 6 1 (i32) -> (i64) vi (vc)
  k = i64.const 20
  vr = i64.add vj k
  return vr
  }}
}}
func 2 (i64) -> (i64) {{
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

/// #1944: a detached child spawns and joins a detached grandchild paid from its own budget — the one
/// that paid for its window.
#[test]
fn a_detached_child_spawns_a_grandchild_from_its_own_budget() {
    let src = three_generations(1 << 18);
    for b in BACKENDS {
        assert_eq!(run_detached(b, &src).expect("run"), 123, "{b:?}");
    }
}

/// #1909, INVARIANTS #3 R2 — a detached child's growth past its declared window spends the budget
/// that paid for the window, on every tier. The root pays for func 1 from a node capped at the child's
/// 64 KiB window plus 64 KiB; the child grows 64 KiB (it fits), asks for 64 KiB more (`-ENOMEM`), gives
/// the first back (refunded at once), grows the second (it fits again), and returns a bit per answer.
/// The root exits `child * 10 + 1` when its node's room is the whole ceiling again after the join: the
/// child's end handed back the growth it still held.
fn growth_program() -> String {
    format!(
        "memory 16
data 17920 \"vm\"
data 17928 \"budget\"
{rec}import 0 \"exit\" (i32) -> ()

func 0 () -> () {{
block 0 () {{
  vp = i64.const 17920
  vl = i64.const 2
  vinst = self.resolve vp vl
  bp = i64.const 17928
  bl = i64.const 6
  vbud = self.resolve bp bl
  f = i64.const -1
  m = i64.const 131072
  vsub = call.cap 14 0 (i64, i64, i64) -> (i32) vbud (f, m, f)
  rb = i64.const 17436
  i32.store rb vsub
  rp = i64.const 17408
  vch = call.cap 6 17 (i64) -> (i32) vinst (rp)
  vr = call.cap 6 1 (i32) -> (i64) vinst (vch)
  one = i64.const 1
  vroom = call.cap 14 1 (i64) -> (i64) vsub (one)
  vfull = i64.eq vroom m
  vfw = i64.extend_i32_u vfull
  ten = i64.const 10
  vhi = i64.mul vr ten
  vsum = i64.add vhi vfw
  vc = i32.wrap_i64 vsum
  call.import 0 (vc)
  unreachable
  }}
}}
func 1 (i64, i64) -> (i64) {{
block 0 (vinst: i64, vas64: i64) {{
  vas = i32.wrap_i64 vas64
  a1 = i64.const 65536
  a2 = i64.const 131072
  n = i64.const 65536
  rw = i64.const 3
  z = i64.const 0
  enomem = i64.const -12
  r1 = call.cap 5 0 (i64, i64, i64) -> (i64) vas (a1, n, rw)
  r2 = call.cap 5 0 (i64, i64, i64) -> (i64) vas (a2, n, rw)
  r3 = call.cap 5 1 (i64, i64) -> (i64) vas (a1, n)
  r4 = call.cap 5 0 (i64, i64, i64) -> (i64) vas (a2, n, rw)
  b1 = i64.eq r1 z
  b2 = i64.eq r2 enomem
  b3 = i64.eq r3 z
  b4 = i64.eq r4 z
  w1 = i64.extend_i32_u b1
  w2 = i64.extend_i32_u b2
  w3 = i64.extend_i32_u b3
  w4 = i64.extend_i32_u b4
  two = i64.const 2
  four = i64.const 4
  eight = i64.const 8
  x2 = i64.mul w2 two
  x3 = i64.mul w3 four
  x4 = i64.mul w4 eight
  s1 = i64.add w1 x2
  s2 = i64.add s1 x3
  s3 = i64.add s2 x4
  return s3
  }}
}}
",
        rec = rec::segment(17408, &SpawnRec::v1(1)),
    )
}

#[test]
fn a_detached_childs_growth_spends_its_budget_on_every_tier() {
    let src = growth_program();
    for b in BACKENDS {
        assert_eq!(run_detached(b, &src).expect("run"), 151, "{b:?}");
    }
}

/// #2111 — a region a detached child mints spends the `channel` of the budget that paid for its
/// window, on every tier. The root pays for func 1 from a node whose `channel` ceiling is one 64 KiB
/// region; the child mints one (it fits) and asks for a second (`-ENOMEM`), returning a bit per answer.
/// The root exits `child * 10 + 1` when its node's `channel` room is the whole ceiling again after the
/// join: the child's end let go of the region it still held.
fn region_program() -> String {
    format!(
        "memory 16
data 17920 \"vm\"
data 17928 \"budget\"
{rec}import 0 \"exit\" (i32) -> ()

func 0 () -> () {{
block 0 () {{
  vp = i64.const 17920
  vl = i64.const 2
  vinst = self.resolve vp vl
  bp = i64.const 17928
  bl = i64.const 6
  vbud = self.resolve bp bl
  f = i64.const -1
  c = i64.const 65536
  vsub = call.cap 14 0 (i64, i64, i64, i64) -> (i32) vbud (f, f, f, c)
  rb = i64.const 17436
  i32.store rb vsub
  rp = i64.const 17408
  vch = call.cap 6 17 (i64) -> (i32) vinst (rp)
  vr = call.cap 6 1 (i32) -> (i64) vinst (vch)
  three = i64.const 3
  vroom = call.cap 14 1 (i64) -> (i64) vsub (three)
  vfull = i64.eq vroom c
  vfw = i64.extend_i32_u vfull
  ten = i64.const 10
  vhi = i64.mul vr ten
  vsum = i64.add vhi vfw
  vc = i32.wrap_i64 vsum
  call.import 0 (vc)
  unreachable
  }}
}}
func 1 (i64, i64) -> (i64) {{
block 0 (vinst: i64, vas64: i64) {{
  vas = i32.wrap_i64 vas64
  n = i64.const 65536
  z = i64.const 0
  enomem = i64.const -12
  r1 = call.cap 5 5 (i64) -> (i64) vas (n)
  r2 = call.cap 5 5 (i64) -> (i64) vas (n)
  b1 = i64.ge_s r1 z
  b2 = i64.eq r2 enomem
  w1 = i64.extend_i32_u b1
  w2 = i64.extend_i32_u b2
  two = i64.const 2
  x2 = i64.mul w2 two
  s = i64.add w1 x2
  return s
  }}
}}
",
        rec = rec::segment(17408, &SpawnRec::v1(1)),
    )
}

#[test]
fn a_detached_childs_region_spends_its_budget_on_every_tier() {
    let src = region_program();
    for b in BACKENDS {
        assert_eq!(run_detached(b, &src).expect("run"), 31, "{b:?}");
    }
}

/// #1944 (the owner's A/B/C example): a child's ceiling caps its whole subtree even while its parent
/// has room. The child's 128 KiB ceiling holds its own window, so its child's spawn is refused
/// (`-22 + 100`), under a root budget of 1 MiB.
#[test]
fn a_childs_ceiling_caps_its_subtree_while_its_parent_has_room() {
    let src = three_generations(1 << 17);
    for b in BACKENDS {
        assert_eq!(run_detached(b, &src).expect("run"), 78, "{b:?}");
    }
}
