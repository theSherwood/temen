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

fn grant_hooks() -> GrantChildHooks {
    GrantChildHooks {
        build: temen_run::grant_child_build,
        build_named: temen_run::grant_named_child_build,
        build_detached: temen_run::grant_detached_child_build,
        budget_mem_take: temen_run::budget_mem_take,
        bind_imports: temen_run::child_bind_imports,
        release: temen_run::grant_child_release,
        mint: temen_run::child_offer_mint,
        thunk: temen_run::cap_thunk_locked,
        register_serve: temen_run::child_register_serve,
    }
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
        Some(grant_hooks()),
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

/// PROCESS.md §5 / #1289 R1 — the tree-walker's `!durable` op-15 gate is lifted (freeze authority is a
/// per-grant capability, not a placement rule), so the two backends now **diverge** on a durable detached
/// spawn: the interpreter admits it and charges the window to `Budget.mem`, while the native thunk still
/// declines `-EINVAL` with the budget intact — a tracked in-flight decline until the native freeze path is
/// wired (the later detached-durable freeze slices). This pins that transition state.
#[test]
fn a_durable_detached_spawn_admits_on_the_interpreter_but_still_declines_on_the_native_jit() {
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

    // Interpreter tier: R1 admits the durable detached spawn (returns a non-negative slot) and charges
    // the child's 2^16 window to `Budget.mem`, exactly as a non-durable spawn does.
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
        .expect("interp run");
        let slot = match r.first() {
            Some(Value::I64(x)) => *x,
            other => panic!("unexpected interp result {other:?}"),
        };
        assert!(
            slot >= 0,
            "interp: durable detached spawn admits, slot {slot}"
        );
        assert!(
            !host.budget_mem_take(h[2], 1 << 16),
            "interp: the admitted spawn charged the child's window — the quota is now exhausted"
        );
    }
}
