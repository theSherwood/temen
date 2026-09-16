//! PROCESS.md §5 / #1287 — **`instantiate_detached` (op 15) on the native JIT**, differential against the
//! tree-walker. The parent (compiled by Cranelift) spawns a separate-module child DETACHED: the JIT's
//! op-15 thunk takes the detached-spawn `Budget` quota, builds the child powerbox through
//! `Host::spawn_detached_child` (attests `window_exposed = false`, starter caps over the reservation),
//! compiles the child over a **decoupled window** (its declared 64 KiB committed inside a root-sized lazy
//! reservation), seeds the module's data + the spawn-time argv payload into that window, and runs it on
//! its own OS thread — no carve, no copy-in, no copy-back. The child reads the argv word, `self.attest`s,
//! `vm_map`s past its declared window (committing a tail page of ITS reservation through its own
//! `AddressSpace`), stores/loads on the grown page and returns `word + attest`. Same result as the
//! interpreter's op-15 arm; an exhausted budget refuses `-EINVAL` on both.

use core::ffi::c_void;
use temen_interp::{run_with_host, Host, Value};
use temen_jit::{
    compile_and_run_capture_reserved_with_host_durable,
    compile_and_run_capture_reserved_with_host_ex, GrantChildHooks, JitOutcome,
};

/// #1234 — the production table, derived from one [`temen_run::CapCtx`] so the hook family and
/// the parent pointer it decodes are chosen together (this used to hand-roll both, and nothing
/// checked that the pointer matched the ctx the run baked).
fn grant_hooks(host: *mut temen_interp::Host) -> GrantChildHooks {
    temen_run::production_grant_hooks(temen_run::CapCtx::Raw(host))
}

fn module(text: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(text).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// "hello-de" as a little-endian i64 — the word the child reads at `args_base + 8`.
const ARGV_WORD: i64 = i64::from_le_bytes(*b"hello-de");

/// The detached child (child-entry, `memory 16`, manifest `vm_map`): argv word at 16384+128+8, attest,
/// `vm_map [64 KiB, 80 KiB)`, store/load the word on the grown page, return `word + attest`.
const CHILD: &str = r#"memory 16
import 0 "vm_map" (i64, i64, i32) -> (i64)
func (i64) -> (i64) {
block 0 (v0: i64) {
  vab = i64.const 16520
  va = i64.load vab
  vz = i32.const 0
  vat = call.cap 4294967295 4 () -> (i64) vz ()
  voff = i64.const 65536
  vlen = i64.const 16384
  vprot = i32.const 3
  vg = call.import 0 (voff, vlen, vprot)
  vp = i64.const 65600
  i64.store vp va
  vld = i64.load vp
  vs = i64.add vld vat
  return vs
  }
}
"#;

/// The parent: `v0` Instantiator, `v1` the child `Module`, `v2` the detached-spawn `Budget`. Stores the args
/// blob (`argc 1`, `"hello-detached\0"`) as three words at 18432, spawns the child detached (9-arg op 15,
/// payload `(18432, 24)`, no grants, entry 0, window 2^16), then joins it — or, in the refusal probe,
/// returns the spawn's own result.
fn parent(join: bool) -> String {
    let tail = if join {
        "vr = call.cap 6 1 (i32) -> (i64) v0 (vh)\n  return vr"
    } else {
        "vr = i64.extend_i32_s vh\n  return vr"
    };
    format!(
        r#"memory 17
func (i32, i32, i32) -> (i64) {{
block 0 (v0: i32, v1: i32, v2: i32) {{
  vb0 = i64.const 18432
  vw0 = i64.const 1
  i64.store vb0 vw0
  vb1 = i64.const 18440
  vw1 = i64.const {w1}
  i64.store vb1 vw1
  vb2 = i64.const 18448
  vw2 = i64.const {w2}
  i64.store vb2 vw2
  vmh = i64.extend_i32_u v1
  vmin = i64.extend_i32_u v2
  vz = i64.const 0
  ve = i64.const 0
  vlog = i64.const 16
  vq = i64.const 0
  vap = i64.const 18432
  val = i64.const 24
  vh = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vmin, vmh, vz, vz, ve, vlog, vq, vap, val)
  {tail}
  }}
}}
"#,
        w1 = ARGV_WORD,
        w2 = i64::from_le_bytes(*b"tached\0\0"),
    )
}

fn host(child: &temen_ir::Module, minter_quota: u64) -> (Host, [i32; 3]) {
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, 1u64 << 17);
    let modh = host.grant_module(child);
    let budget = host.grant_budget(0, (minter_quota) as i64, 0);
    (host, [inst, modh, budget])
}

fn run_jit(parent: &temen_ir::Module, child: &temen_ir::Module, quota: u64) -> i64 {
    let (mut host, h) = host(child, quota);
    let args = [h[0] as i64, h[1] as i64, h[2] as i64];
    let (jo, _) = compile_and_run_capture_reserved_with_host_ex(
        parent,
        0,
        &args,
        &[],
        temen_ir::DEFAULT_RESERVED_LOG2,
        temen_run::cap_thunk,
        &mut host as *mut Host as *mut c_void,
        Some(temen_run::module_resolver),
        Some(grant_hooks(&mut host as *mut Host)),
    )
    .expect("jit run");
    match jo {
        JitOutcome::Returned(ref v) => v.first().copied().unwrap_or(-1),
        ref o => panic!("jit ended abnormally: {o:?}"),
    }
}

fn run_interp(parent: &temen_ir::Module, child: &temen_ir::Module, quota: u64) -> i64 {
    let (mut host, h) = host(child, quota);
    let mut fuel = 50_000_000u64;
    let r = run_with_host(
        parent,
        0,
        &[Value::I32(h[0]), Value::I32(h[1]), Value::I32(h[2])],
        &mut fuel,
        &mut host,
    )
    .expect("interp run");
    match r.first() {
        Some(Value::I64(x)) => *x,
        other => panic!("unexpected interp result {other:?}"),
    }
}

#[test]
fn a_detached_child_on_the_jit_matches_the_interpreter() {
    let p = module(&parent(true));
    let c = module(CHILD);
    let want = ARGV_WORD + 1; // argv landed; attest = 1 (tier 1, window_exposed = false)
    assert_eq!(run_interp(&p, &c, 1 << 16), want, "interpreter oracle");
    let before = temen_jit::child_compiles();
    assert_eq!(
        run_jit(&p, &c, 1 << 16),
        want,
        "the JIT-hosted detached child"
    );
    assert!(
        temen_jit::child_compiles() > before,
        "the child was JIT-compiled (not served by the interpreter)"
    );
}

#[test]
fn an_exhausted_minter_refuses_probeably_on_both_backends() {
    let p = module(&parent(false));
    let c = module(CHILD);
    assert_eq!(run_interp(&p, &c, (1 << 16) - 1), -22);
    assert_eq!(run_jit(&p, &c, (1 << 16) - 1), -22);
}

/// A parent that only issues the 7-arg op 15 and returns its result — no window stores, so it runs
/// unchanged under a **durable** host (whose shadow reserve spans `[0, 64 KiB)`).
const SPAWN_ONLY_PARENT: &str = r#"memory 17
func (i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32) {
  vmh = i64.extend_i32_u v1
  vmin = i64.extend_i32_u v2
  vz = i64.const 0
  ve = i64.const 0
  vlog = i64.const 16
  vq = i64.const 0
  vs = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vmin, vmh, vz, vz, ve, vlog, vq)
  vr = i64.extend_i32_s vs
  return vr
  }
}
"#;

/// #1412 — **a durable detached spawn declines identically on the interpreter and the native JIT.**
///
/// This test previously asserted the opposite, and its old name said so:
/// `a_durable_detached_spawn_admits_on_the_interpreter_but_still_declines_on_the_native_jit`. It was
/// added with #1289 R1 to pin what that ruling left behind — R1 lifted the tree-walker's `!durable`
/// op-15 gate and only the tree-walker's, so the two backends diverged, and this test recorded the
/// divergence as the intended "transition state".
///
/// INVARIANTS #9 does not have a transition state. The tree-walk interpreter defines guest-observable
/// semantics and the other engines match it or decline to it; two engines giving a verified module two
/// different answers is the thing the invariant exists to forbid. A test asserting that they disagree
/// cannot go red when the disagreement is wrong — which is how it stood for six days.
///
/// Owner decision 2026-09-14 (#1412): R1's end state stands (freeze authority is a per-grant
/// capability; a durable parent will spawn detached children whose freeze *captures* them), but its
/// spawn lift is deferred until freeze authority exists in code (#1440) and the per-child capture
/// lands (#1361). Until then both backends refuse, probeably, charging nothing.
///
/// So the assertion inverts: not "they differ, as planned", but **"they agree"**. That is what makes
/// this a pin rather than a record. It extends `temen-interp`'s `durable_detached_parity.rs`
/// (oracle ↔ resumable engine) to the third engine.
#[test]
fn a_durable_detached_spawn_declines_the_same_way_on_the_interpreter_and_the_native_jit() {
    let p = module(SPAWN_ONLY_PARENT);
    let c = module(CHILD);

    // Native JIT tier: still declines. It learns durability from the run entry (`cm.durable` → the
    // nursery's flag), not from the host, so the durable run entry is the one to use. It installs no
    // grant hooks: a thunk that reached the hook lookup would trap `CapFault`, so a `-22` here can only
    // come from the durable gate that precedes it.
    {
        let (mut host, h) = host(&c, 1 << 16);
        host.set_durable(true);
        let args = [h[0] as i64, h[1] as i64, h[2] as i64];
        let (jo, _, _) = compile_and_run_capture_reserved_with_host_durable(
            &p,
            0,
            &args,
            &[],
            &[],
            &[],
            temen_ir::DEFAULT_RESERVED_LOG2,
            temen_run::cap_thunk,
            &mut host as *mut Host as *mut c_void,
        )
        .expect("jit run");
        let r = match jo {
            JitOutcome::Returned(ref v) => v.first().copied().unwrap_or(-1),
            ref o => panic!("jit ended abnormally: {o:?}"),
        };
        assert_eq!(r, -22, "native jit: EINVAL, not a trap");
        assert!(
            host.budget_mem_take(h[2], 1 << 16),
            "native jit: the refusal charged the budget nothing"
        );
    }

    // Interpreter tier: the same answer, byte for byte. `-22` is `-EINVAL`, matching the native thunk
    // above — and it is a *value*, so the run completes and the guest could have handled it.
    {
        let (mut host, h) = host(&c, 1 << 16);
        host.set_durable(true);
        let mut fuel = 50_000_000u64;
        let r = run_with_host(
            &p,
            0,
            &[Value::I32(h[0]), Value::I32(h[1]), Value::I32(h[2])],
            &mut fuel,
            &mut host,
        )
        .expect("interp run: the refusal is a value, so the run still completes");
        let slot = match r.first() {
            Some(Value::I64(x)) => *x,
            other => panic!("unexpected interp result {other:?}"),
        };
        assert_eq!(
            slot, -22,
            "interp must give the native JIT's answer for the same module (INVARIANTS #9): \
             -EINVAL, not an admission"
        );
        // And, like the native thunk, it charges nothing — the gate sits before the quota take, so the
        // guest keeps the budget for something it *is* allowed to do.
        assert!(
            host.budget_mem_take(h[2], 1 << 16),
            "interp: the refusal charged the budget nothing, exactly as the native thunk's does"
        );
    }
}

/// Op-15 **pre-mapped region** (the 11-arg form) on the native JIT, differential against the
/// tree-walker: the parent mints a 64 KiB `SharedRegion`, maps it at 65536 in its own window, stores
/// the word 41 at region byte 0 and spawns the child detached with the region pre-mapped at 65536 of
/// ITS window. The child reads the word through the alias, writes `2 × 41` at region byte 8 and
/// returns `41 + 1`; the parent joins and reads the child's word back through its own mapping:
/// `1000 × 42 + 82`. On the JIT both mappings are real `MAP_SHARED` views of one memfd/section
/// (`Host::apply_premap` over the child's `MprotectWindow`, before its first instruction); on the
/// interpreter they are `PageProt::Backed` aliases — same bytes, same result.
const PREMAP_CHILD: &str = r#"memory 17
func (i64) -> (i64) {
block 0 (v0: i64) {
  va = i64.const 65536
  vin = i64.load va
  vtwo = i64.const 2
  vout = i64.mul vin vtwo
  vb = i64.const 65544
  i64.store vb vout
  vone = i64.const 1
  vr = i64.add vin vone
  return vr
  }
}
"#;

/// `v0` Instantiator, `v1` AddressSpace, `v2` the child `Module`, `v3` the `Budget`; `off` is the
/// child-window offset the region is pre-mapped at (`65536` round-trips; `1 << 17` overruns the
/// child's window and refuses `-EINVAL`, which the non-joining form returns).
fn premap_parent(off: u64, join: bool) -> String {
    let tail = if join {
        "vj = call.cap 6 1 (i32) -> (i64) v0 (vc)\n  vk = i64.const 1000\n  vm = i64.mul vj vk\n  vob = i64.const 65544\n  vo = i64.load vob\n  vr = i64.add vm vo\n  return vr"
    } else {
        "vr = i64.extend_i32_s vc\n  return vr"
    };
    format!(
        r#"memory 17
func (i32, i32, i32, i32) -> (i64) {{
block 0 (v0: i32, v1: i32, v2: i32, v3: i32) {{
  vlen = i64.const 65536
  vrh64 = call.cap 5 5 (i64) -> (i64) v1 (vlen)
  vrh = i32.wrap_i64 vrh64
  vwo = i64.const 65536
  vro = i64.const 0
  vprot = i32.const 3
  vm0 = call.cap 4 0 (i64, i64, i64, i32) -> (i64) vrh (vwo, vro, vlen, vprot)
  vin = i64.const 41
  i64.store vwo vin
  vmh = i64.extend_i32_u v2
  vb = i64.extend_i32_u v3
  vz = i64.const 0
  vlog = i64.const 17
  vreg = i64.extend_i32_u vrh
  voff = i64.const {off}
  vc = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vb, vmh, vz, vz, vz, vlog, vz, vz, vz, vreg, voff)
  {tail}
  }}
}}
"#
    )
}

/// The pre-map parent's powerbox: the four handles, plus the OS shared-memory region factory the JIT's
/// real `MAP_SHARED` aliasing needs (a software `VecBacking` cannot be `mmap`ed).
fn premap_host(child: &temen_ir::Module) -> (Host, [i32; 4]) {
    let mut host = Host::new();
    host.set_region_factory(temen_run::new_shared_region);
    let inst = host.grant_instantiator(0, 1u64 << 17);
    let aspace = host.grant_address_space(0, 1u64 << 17);
    let modh = host.grant_module(child);
    let budget = host.grant_budget(0, 1i64 << 17, 0);
    (host, [inst, aspace, modh, budget])
}

fn run_premap_jit(parent: &temen_ir::Module, child: &temen_ir::Module) -> i64 {
    let (mut host, h) = premap_host(child);
    let args = [h[0] as i64, h[1] as i64, h[2] as i64, h[3] as i64];
    let (jo, _) = compile_and_run_capture_reserved_with_host_ex(
        parent,
        0,
        &args,
        &[],
        temen_ir::DEFAULT_RESERVED_LOG2,
        temen_run::cap_thunk,
        &mut host as *mut Host as *mut c_void,
        Some(temen_run::module_resolver),
        Some(grant_hooks(&mut host as *mut Host)),
    )
    .expect("jit run");
    match jo {
        JitOutcome::Returned(ref v) => v.first().copied().unwrap_or(-1),
        ref o => panic!("jit ended abnormally: {o:?}"),
    }
}

fn run_premap_interp(parent: &temen_ir::Module, child: &temen_ir::Module) -> i64 {
    let (mut host, h) = premap_host(child);
    let mut fuel = 50_000_000u64;
    let r = run_with_host(
        parent,
        0,
        &[
            Value::I32(h[0]),
            Value::I32(h[1]),
            Value::I32(h[2]),
            Value::I32(h[3]),
        ],
        &mut fuel,
        &mut host,
    )
    .expect("interp run");
    match r.first() {
        Some(Value::I64(x)) => *x,
        other => panic!("unexpected interp result {other:?}"),
    }
}

#[test]
fn a_pre_mapped_region_round_trips_bulk_data_on_the_jit_as_on_the_interpreter() {
    let parent = module(&premap_parent(65536, true));
    let child = module(PREMAP_CHILD);
    let interp = run_premap_interp(&parent, &child);
    assert_eq!(interp, 42_082, "the oracle's round trip");
    assert_eq!(
        run_premap_jit(&parent, &child),
        interp,
        "the JIT matches the oracle"
    );
}

#[test]
fn a_pre_map_overrunning_the_child_window_refuses_probeably_on_both_backends() {
    let parent = module(&premap_parent(1 << 17, false));
    let child = module(PREMAP_CHILD);
    assert_eq!(run_premap_interp(&parent, &child), -22);
    assert_eq!(
        run_premap_jit(&parent, &child),
        -22,
        "EINVAL, not a trap, on the JIT too"
    );
}
