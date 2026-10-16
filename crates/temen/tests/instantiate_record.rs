//! CONSOLIDATION.md §3a — **the config-record spawn** (`Instantiator` op 17,
//! `instantiate_rec(record_ptr)`): one 64-byte spawn record (`SpawnRec`) names a detached spawn as
//! data — the child module, its window (`size_log2`, `0` = the module's declared memory), a pager
//! export (`u32::MAX` = none), the `Budget` that funds it, a named-grant list, an args payload and a
//! pre-mapped `SharedRegion`. The carve spawn is refused (#1867, `carve_spawn_refused.rs`).
//!
//! What a record carries, on every tier: granted offers and the deadlocks around them (#1217), a
//! budget's `channel` bounding the child (#989), a parent serving its child's `exec` (#744), and the
//! spawn record's payload, window, pager, budget and region rules (#1863).

#[path = "../../temen-interp/tests/support/rec.rs"]
mod rec;

use std::sync::Arc;

use temen_interp::{run_capture_reserved_with_host, Host, Value};
use temen_ir::{Module, SpawnRec};
use temen_run::{instantiate_with_imports, Backend, HostCap, Imports, Outcome, RunConfig};
use temen_text::parse_module;
use temen_verify::verify_module;

const BACKENDS: [Backend; 3] = [Backend::TreeWalk, Backend::Bytecode, Backend::Jit];

/// Every driver a detached run takes: each backend, then the bytecode engine's parallel driver
/// (`None`).
const DRIVERS: [Option<Backend>; 4] = [
    Some(Backend::TreeWalk),
    Some(Backend::Bytecode),
    Some(Backend::Jit),
    None,
];

/// #744 — whether the cooperative bytecode driver runs `src` itself, serving its children, rather
/// than declining it to the tree-walker: a `Backend::Bytecode` run of it then pins that driver.
fn runs_on_bytecode(src: &str) -> bool {
    temen_interp::bytecode::admits(&parse_module(src).expect("parse"))
}

/// #1217, the granted-offer shape: the root spawns a **server** child (`"server"`, one `svc.wait`,
/// returning its count), mints a `child_offer` over the server's export, and spawns a **guest**
/// child (`"guest"`) with that offer re-granted by name as `"fork"`; then `join`s the server and
/// returns its count. Both children are detached, paid from the root's `"budget"`. The `"fork"`
/// name is a data segment, so the guest finds it in its own window too.
fn granted_client_program(guest: &str) -> String {
    let guest_rec = SpawnRec {
        grants_ptr: 16640,
        grants_n: 1,
        ..SpawnRec::default()
    };
    format!(
        "\
memory 17
data 16384 \"vm\"
data 16400 \"budget\"
data 16448 \"server\"
data 16456 \"guest\"
data 16684 \"fork\"
{server_rec}{guest_rec}type 0 func (i64) -> (i64)
type 1 interface {{ op: 0 }}
export 0 interface \"fork\" 1 {{ op: 2 }}
export 0 func \"server\" 1
export 1 func \"guest\" 3
func 0 () -> (i32) {{
block 0 () {{
  vp = i64.const 16384
  vl = i64.const 2
  v0 = self.resolve vp vl
  vbp = i64.const 16400
  vbl = i64.const 6
  vb = self.resolve vbp vbl
  vsb = i64.const 17420
  i32.store vsb vb
  vsrvp = i64.const 16448
  vsrvl = i64.const 6
  vserver = self.resolve vsrvp vsrvl
  vsrvm = i64.const 17416
  i32.store vsrvm vserver
  vgstp = i64.const 16456
  vgstl = i64.const 5
  vguest = self.resolve vgstp vgstl
  vgstm = i64.const 17512
  i32.store vgstm vguest
  vsp = i64.const 17408
  vs = call.cap 6 17 (i64) -> (i32) v0 (vsp)
  vz0 = i64.const 0
  voff = call.cap 6 14 (i32, i64) -> (i32) v0 (vs, vz0)
{fork_grant}  vgb = i64.const 17516
  i32.store vgb vb
  vgp = i64.const 17504
  vg = call.cap 6 17 (i64) -> (i32) v0 (vgp)
  vjs_st, vjs = call.cap 6 1 (i32) -> (i64, i64) v0 (vs)
  vc = i32.wrap_i64 vjs
  return vc
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
        server_rec = rec::segment(17408, &SpawnRec::default()),
        guest_rec = rec::segment(17504, &guest_rec),
        fork_grant = rec::grant("gf", 16640, 16684, 4, "voff"),
    )
}

/// The healthy exchange, pinning the program shape: the guest resolves `"fork"` and calls it once
/// (the handler returns 7), the server's `svc.wait` counts 1, the root returns 1.
#[test]
fn granted_client_that_calls_once_is_served() {
    let src = granted_client_program(
        "  vnp = i64.const 16684
  vl4 = i64.const 4
  vfork = self.resolve vnp vl4
  vz8 = i64.const 0
  vr = call.cap 268435456 0 (i64) -> (i64) vfork (vz8)
  return vr",
    );
    assert!(runs_on_bytecode(&src), "the cooperative driver serves it");
    for b in BACKENDS {
        assert_eq!(run_detached(b, &src).expect("run"), 1, "{b:?}");
    }
}

/// #1217: the guest traps before calling. The server's only client is gone, so its `svc.wait`
/// returns `0` (never parks forever), it returns, and the root's `join` delivers `0`.
#[test]
fn granted_client_that_dies_before_calling_releases_the_server() {
    let src = granted_client_program("  unreachable");
    assert!(runs_on_bytecode(&src), "the cooperative driver serves it");
    for b in BACKENDS {
        assert_eq!(run_detached(b, &src).expect("run"), 0, "{b:?}");
    }
}

/// #1228: the **daemon** shape #1217 cannot cover. The server is written as an unconditional
/// `loop { svc.wait }` (the fork-manager server shape), not a wait-once-and-return, so the
/// client-death token (#1217) gives it one spurious `0` and the loop re-parks it, untimed,
/// forever. With its only client gone and the root `join`ing it, the run is a genuine
/// join-deadlock: the root parked in `join`, the daemon parked in `svc.wait`, nothing else live
/// and no external wake source. INVARIANTS.md #9 forbids a hang, so the run must **trap**
/// `ThreadFault`. This pins the tree-walk executor (#1228), the cooperative bytecode driver, which
/// serves the shape itself (#744), and the Cranelift JIT, whose `svc.wait` park and `join` count
/// for its deadlock verdict (#2173, #1820), to the identical outcome.
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
    assert!(runs_on_bytecode(&src), "the cooperative driver serves it");
    for b in BACKENDS {
        let e = run_detached(b, &src).expect_err("the join-deadlock must trap, not hang");
        assert!(
            e.contains("ThreadFault"),
            "{b:?}: the all-parked join-deadlock traps ThreadFault, got: {e}"
        );
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
    let base = granted_client_program(guest);
    let server = "block 0 (v0: i64) {\n  vz = i32.const 0\n  vs = svc.wait vz\n  return vs\n  }";
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

// ===== #989 slice 1b — the record's Budget `channel` bounds a spawned child ======================

/// The worst-case per-pipe host-served channel charge (`PIPE_CAP`, a full 64 KiB FIFO). Mirror of
/// `temen_interp`'s internal constant; kept in sync by the `channel_budget_bounds_...` assertions.
const PIPE_CAP: i64 = 64 << 10;

/// Host-level harness for the §14 channel bound: a parent `(Instantiator, Budget-with-channel,
/// child Module)` spawns its child image (#2219) through a spawn record the budget funds; the child loops
/// minting host-served pipes (`self.pipe`, op 16) into its OWN powerbox and returns how many it minted
/// before the mint was refused. A child's pipes are charged to the budget that paid for it (#1944), so
/// a bounded `channel` fails the mint closed at exactly `channel / PIPE_CAP` pipes; an unbounded
/// channel only stops when the handle table fills (many more). fuel/mem/spawn are all unbounded
/// (`-1`), so the ONLY thing constraining the child is its channel ceiling. The parent joins and
/// returns the child's count × 1000 (`-1` if the spawn was refused).
fn run_channel_bounded_child(channel: i64) -> Result<Vec<Value>, temen_interp::Trap> {
    let src = format!(
        r#"memory 17
func (i32, i32, i32) -> (i64) {{
block 0 (vinst: i32, vbud: i32, vmod: i32) {{
  vrp = i64.const 17408
  i32.store vrp vmod offset={module_at}
  i32.store vrp vbud offset={budget_at}
  vch = call.cap 6 17 (i64) -> (i32) vinst (vrp)
  vzero = i32.const 0
  visneg = i32.lt_s vch vzero
  br_if visneg 2() 1(vinst, vch)
}}
block 1 (vinst1: i32, vch1: i32) {{
  vj_st, vj = call.cap 6 1 (i32) -> (i64, i64) vinst1 (vch1)
  k = i64.const 1000
  t0 = i64.mul vj k
  return t0
}}
block 2 () {{
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
{record}"#,
        module_at = SpawnRec::MODULE_AT,
        budget_at = SpawnRec::BUDGET_AT,
        record = rec::segment(17408, &SpawnRec::default()),
    );
    let m = parse_module(&src).expect("parse");
    verify_module(&m).expect("verify");
    let mut host = Host::new();
    let ih = host.grant_instantiator();
    // fuel/mem/spawn unbounded — only `channel` constrains the child.
    let bh = host.grant_budget_channel(-1, -1, -1, channel);
    let mh = host.grant_module(&temen_ir::child_image_at(&m, 1).expect("child image"));
    let mut fuel = 50_000_000u64;
    let (res, _) = run_capture_reserved_with_host(
        &m,
        0,
        &[Value::I32(ih), Value::I32(bh), Value::I32(mh)],
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
/// child's mints charge the budget that paid for it.
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

// ===== #744 (EXEC.md row 4) — guest-served exec: a parent serves its child's "exec" ==============

/// #744 — the **mediation-consistent guest-served exec** (EXEC.md row 4): the parent grants its
/// child a `"exec"` capability backed by **its own code**, and the child is none-the-wiser. The
/// grant is a named-grant record whose `handle` carries `GRANT_SERVE_LIVE_TAG` over the parent's
/// impl-export index (export 0 here, interface `"exec"` with one op `run`) — the record's reserved
/// `flags` word stays 0 and ignored, as the ABI promises. The parent spawns `"child"` detached
/// through a spawn record; the spawn installs into the CHILD a live-callee offer whose callee is the parent's running powerbox. The child resolves
/// `"exec"` by name and calls `run(40, 2)` — which parks it until the PARENT's `svc.wait` serve
/// loop runs handler func 2 over the parent's live world and replies `42`. The parent joins the
/// child and returns `join*100 + served` = `42*100 + 1` = `4201`. The parent never holds a
/// self-referential cap (the offer exists only in the child's table), so there is no reference
/// cycle and nothing to mis-call.
///
/// `handle` is a parameter so the fail-closed edge is pinned too: a tag over an impl-export the
/// parent does not have is refused at spawn (`CapFault`) before any child state is built.
fn serve_live_program(handle: i32) -> String {
    let child = SpawnRec {
        grants_ptr: 16640,
        grants_n: 1,
        ..SpawnRec::default()
    };
    format!(
        "\
memory 17
data 16384 \"vm\"
data 16400 \"budget\"
data 16448 \"child\"
data 16684 \"exec\"
{rec}type 0 func (i64, i64) -> (i64)
type 1 interface {{ run: 0 }}
export 0 interface \"exec\" 1 {{ run: 2 }}
export 0 func \"child\" 1
func 0 () -> (i32) {{
block 0 () {{
  vp = i64.const 16384
  vl = i64.const 2
  vh = self.resolve vp vl
  vbp = i64.const 16400
  vbl = i64.const 6
  vb = self.resolve vbp vbl
  vba = i64.const {budget_at}
  i32.store vba vb
  vcp = i64.const 16448
  vcl = i64.const 5
  vchild = self.resolve vcp vcl
  vma = i64.const {module_at}
  i32.store vma vchild
  ; the one grant record at 16640, its handle `TAG | export`
  vgh = i32.const {handle}
{exec_grant}  vrp = i64.const 17408
  vch = call.cap 6 17 (i64) -> (i32) vh (vrp)
  vz = i32.const 0
  vneg = i32.lt_s vch vz
  br_if vneg 2(vch) 1(vh, vch)
}}
block 1 (vh1: i32, vch1: i32) {{
  ; serve the child's one `exec.run` call: OUR handler (func 2) answers over OUR live world
  vz1 = i32.const 0
  vn = svc.wait vz1
  vj_st, vj = call.cap 6 1 (i32) -> (i64, i64) vh1 (vch1)
  vk = i64.const 100
  vt = i64.mul vj vk
  vs = i64.add vt vn
  vc = i32.wrap_i64 vs
  return vc
}}
block 2 (verr: i32) {{
  return verr
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
        exec_grant = rec::grant("ge", 16640, 16684, 4, "vgh"),
        budget_at = 17408 + SpawnRec::BUDGET_AT,
        module_at = 17408 + SpawnRec::MODULE_AT,
    )
}

/// **The #744 pin**: the child's `exec.run(40, 2)` is answered by the PARENT's own handler over the
/// parent's live world — `join*100 + served` = `4201` — the guest-served exec backend (EXEC.md row 4)
/// end to end: mint-at-spawn into the child only, caller parks, parent serves, reply, join. Every
/// backend agrees, and each runs the parent itself, serving the child's call over the root's shared
/// cell: the Cranelift JIT, the cooperative bytecode driver and the parallel driver (`None`).
#[test]
fn a_parent_serves_its_childs_exec_with_its_own_code() {
    let src = serve_live_program(temen_interp::GRANT_SERVE_LIVE_TAG as i32);
    assert!(runs_on_bytecode(&src), "the cooperative driver serves it");
    for b in DRIVERS {
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

/// #2284 — the same exec, with the parent's spawn, serve loop and join in a helper (func 3) its entry
/// calls. A child image keeps every function of its program, so the child's carries that helper,
/// unreached: a module that, scanned whole, serves and spawns. A driver that serves its children
/// admits the image as it admits the program, so the child runs there instead of the spawn trapping
/// `Malformed` while the oracle answers.
#[test]
fn a_child_image_that_carries_its_programs_serve_loop_runs_where_the_program_does() {
    let src = serve_live_program(temen_interp::GRANT_SERVE_LIVE_TAG as i32);
    let (head, rest) = src.split_once("func 0 () -> (i32) {\n").expect("the entry");
    let (serve, funcs) = rest.split_once("func 1 ").expect("the child entry");
    let src = format!(
        "{head}func 0 () -> (i32) {{\nblock 0 () {{\n  vr = call 3 ()\n  return vr\n  }}\n}}\n\
         func 1 {funcs}func 3 () -> (i32) {{\n{serve}"
    );
    assert!(runs_on_bytecode(&src), "the cooperative driver serves it");
    for b in DRIVERS {
        assert_eq!(
            run_detached(b, &src),
            Ok(4_201),
            "{b:?}: the child runs its image, and the parent serves its exec.run(40, 2)"
        );
    }
}

/// #1217 on the self-serve grant: a child that holds the parent's live offer but finishes without
/// calling it releases the parent's `svc.wait`, which answers `0` instead of waiting on a caller that
/// is gone; the parent then joins the child's `7` — `join*100 + served` = `700` on every backend,
/// whichever of the child's end and the parent's wait comes first.
#[test]
fn a_child_that_never_calls_releases_its_parents_wait() {
    let src = serve_live_program(temen_interp::GRANT_SERVE_LIVE_TAG as i32).replace(
        "  vr = call.cap 268435456 0 (i64, i64) -> (i64) vexec (va, vb)\n  return vr\n",
        "  v7 = i64.const 7\n  return v7\n",
    );
    assert!(src.contains("return v7"), "the rewrite must apply");
    assert!(runs_on_bytecode(&src), "the cooperative driver serves it");
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
    let src = serve_live_program((temen_interp::GRANT_SERVE_LIVE_TAG | 7) as i32);
    assert!(
        runs_on_bytecode(&src),
        "the cooperative driver refuses it itself"
    );
    for b in BACKENDS {
        let r = run_detached(b, &src);
        assert!(
            matches!(&r, Err(e) if e.contains("CapFault")),
            "{b:?}: a live self-serve grant of a nonexistent export fails the spawn closed: {r:?}"
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
/// which returns at once. After joining the client the root releases the fiber, the thread returns,
/// and the serve loop reaches `svc.wait` with its only client long gone: it must return `0`, so the
/// root returns `0`. The flag (byte 0) and the fiber's go-cell (byte 8) are in a `SharedRegion` the
/// root mints and maps at 65536, and pre-maps into the server's window at the same offset.
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
  vreg = i64.const 17456
  i32.store vreg vrh
  vcoff = i64.const 17464
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
  vjg_st, vjg = call.cap 6 1 (i32) -> (i64, i64) v0 (vg)
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

// ---- #1863: the (detached) spawn record ----

/// Stores building a spawn record at `at` from the i64 values `vr0`..`vr56` in scope (one per 8-byte
/// word, named by offset).
fn store_record(at: u64) -> String {
    (0..8)
        .map(|k| {
            let o = k * 8;
            format!(
                "  vra{o} = i64.const {a}\n  i64.store vra{o} vr{o}\n",
                a = at + o
            )
        })
        .collect()
}

/// A root that spawns `"child"` detached through a spawn record: the child's window is its own
/// (`size_log2` 17, the module's declared memory), funded by the `"budget"` grant, with a one-byte
/// args payload (`*` = 42) the child reads back at `module_args_base`. The child returns
/// `payload + 100`; the root returns the join result. `pager` is `u32::MAX` here; `extra` rewrites
/// let a test vary the record.
fn detached_record_program() -> String {
    let args_base = temen_ir::module_args_base();
    format!(
        "\
memory 17
data 16384 \"vm\"
data 16400 \"budget\"
data 16416 \"*\"
data 16448 \"child\"
export 0 func \"child\" 1

func 0 () -> (i32) {{
block 0 () {{
  vp = i64.const 16384
  vl = i64.const 2
  vh = self.resolve vp vl
  vbp = i64.const 16400
  vbl = i64.const 6
  vb = self.resolve vbp vbl
  vcp = i64.const 16448
  vcl = i64.const 5
  vchild = self.resolve vcp vcl
  vb64 = i64.extend_i32_u vb
  v32 = i64.const 32
  vbsh = i64.shl vb64 v32
  vmod = i64.extend_i32_u vchild
  vr0 = i64.const -4294967279
  vr8 = i64.or vbsh vmod
  vr16 = i64.const 0
  vr24 = i64.const 0
  vr32 = i64.const 16416
  vr40 = i64.const 1
  vr48 = i64.const 4294967295
  vr56 = i64.const 0
{stores}
  vrp = i64.const 17408
  vch = call.cap 6 17 (i64) -> (i32) vh (vrp)
  vj_st, vj = call.cap 6 1 (i32) -> (i64, i64) vh (vch)
  vc = i32.wrap_i64 vj
  return vc
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
        stores = store_record(17408),
    )
}

/// Run with the Instantiator (`"vm"`), an AddressSpace (`"as"`), a detached-window budget
/// (`"budget"`, 1 MiB of `Budget.mem`), and, under its export name, the child image (#2219) of each
/// function the program exports: a program exports the functions it spawns. The root returns its
/// answer. On `backend`, or on the parallel driver for `None`.
fn run_detached(backend: impl Into<Option<Backend>>, src: &str) -> Result<i32, String> {
    run_detached_with(backend, src, 1 << 20, RunConfig::default())
}

/// [`run_detached`] with a `mem`-byte budget, under `cfg`. The run gets its own thread and a
/// deadline: the #1217 pins guard against a *hang* (a pager parked for a child that is gone), so a
/// regression must fail the test, not stall the binary until CI's timeout.
fn run_detached_with(
    backend: impl Into<Option<Backend>>,
    src: &str,
    mem: u64,
    cfg: RunConfig,
) -> Result<i32, String> {
    let backend = backend.into();
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let images: Vec<(String, Arc<Module>)> = m
        .exports
        .iter()
        .map(|e| {
            let image = temen_ir::child_image_at(&m, e.func).expect("child image");
            (e.name.clone(), Arc::new(image))
        })
        .collect();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let inst = instantiate_with_imports(m, Imports::new()).expect("instantiate");
        let mut caps = vec![
            ("vm", HostCap::custom(6, 0, |h, _| h.grant_instantiator())),
            ("budget", HostCap::detached_budget(mem)),
            (
                "as",
                HostCap::custom(5, 0, |h, win| h.grant_address_space(0, win)),
            ),
        ];
        for (name, image) in &images {
            let image = Arc::clone(image);
            caps.push((
                name.as_str(),
                HostCap::custom(temen_ir::cap_id::MODULE, 0, move |h, _| {
                    h.grant_module(&image)
                }),
            ));
        }
        let r = match backend {
            Some(b) => inst.run_with_caps(b, &cfg, &caps),
            None => inst.run_with_caps_parallel(&cfg, &caps),
        };
        let _ = tx.send(r.map(|r| r.outcome).map_err(|e| e.to_string()));
    });
    let outcome = rx
        .recv_timeout(std::time::Duration::from_secs(60))
        .unwrap_or_else(|e| panic!("{backend:?}: the run did not finish: {e}"))?;
    match outcome {
        Outcome::Returned(v) => match v[..] {
            [Value::I32(answer)] => Ok(answer),
            _ => Err(format!("unexpected result {v:?}")),
        },
        other => Err(format!("unexpected outcome {other:?}")),
    }
}

/// #1863: a spawn record spawns a child image into a window of its own, and the args payload arrives
/// at the child's `module_args_base` — identically on every tier.
#[test]
fn a_record_spawns_a_child_detached_with_an_args_payload() {
    let src = detached_record_program();
    for b in BACKENDS {
        assert_eq!(run_detached(b, &src).expect("run"), 142, "{b:?}");
    }
}

/// A spawn record whose `size_log2` is `0` asks for the module's declared window — a spawner holding a
/// module it did not build need not know its size. A nonzero size that is not the declared memory
/// is still refused, probeably: the root returns 7 when the spawn returns a negative handle.
#[test]
fn a_record_with_size_zero_gets_the_declared_window() {
    let base = detached_record_program();
    // size_log2 0 | pager u32::MAX
    let src = base.replace(
        "  vr0 = i64.const -4294967279\n",
        "  vr0 = i64.const -4294967296\n",
    );
    assert!(src.contains("-4294967296"), "the rewrite must apply");
    for b in BACKENDS {
        assert_eq!(run_detached(b, &src).expect("run"), 142, "{b:?}");
    }
    // size_log2 16 against a declared 17: refused before any join.
    let wrong = base
        .replace(
            "  vr0 = i64.const -4294967279\n",
            "  vr0 = i64.const -4294967280\n",
        )
        .replace(
            "  vj_st, vj = call.cap 6 1 (i32) -> (i64, i64) vh (vch)\n  vc = i32.wrap_i64 vj\n",
            "  vz0 = i32.const 0\n  vneg = i32.lt_s vch vz0\n  vseven = i32.const 7\n  vc = select vneg vseven vz0\n",
        );
    assert!(wrong.contains("vneg"), "the rewrite must apply");
    for b in BACKENDS {
        assert_eq!(run_detached(b, &wrong).expect("run"), 7, "{b:?}");
    }
}

/// A record's padding word, after the region handle, is zero (#2091). Anything else is malformed,
/// not some other spawn, and traps on every tier.
#[test]
fn a_record_with_nonzero_padding_fails_closed() {
    let (from, to) = ("  vr48 = i64.const 4294967295\n", "  vr48 = i64.const -1\n");
    let src = detached_record_program().replace(from, to);
    assert!(src.contains(to), "the rewrite must apply");
    for b in BACKENDS {
        let e = run_detached(b, &src).expect_err("a malformed record is refused");
        assert!(e.contains("CapFault"), "{b:?}: {e}");
    }
}

/// A root that serves its own detached demand child: the spawn record names impl export 0 as the
/// pager, and the root waits in `svc.wait` before it joins the child, then returns `byte + serves *
/// 1000 + status * 1_000_000`, the join's status the wire code of a trap the child ended in
/// (#2066). The child (func 1) returns the byte at `fault`, its first touch of that page. The pager
/// (func 2) gets the fault address in the child's coordinates, stores 77 at `addr + 40000` in its
/// own window, and replies with that address (#1862).
fn detached_pager_program_at(fault: u64) -> String {
    let src = detached_record_program()
        .replace(
            "export 0 func \"child\" 1\n",
            "export 0 func \"child\" 1\ntype 0 func (i64) -> (i64)\ntype 1 interface { page: 0 }\n\
             export 0 interface \"pager\" 1 { page: 2 }\n",
        )
        // pager = impl export 0
        .replace("  vr0 = i64.const -4294967279\n", "  vr0 = i64.const 17\n")
        .replace(
            "  vj_st, vj = call.cap 6 1 (i32) -> (i64, i64) vh (vch)\n  vc = i32.wrap_i64 vj\n",
            "  vz = i32.const 0\n  vs = svc.wait vz\n  vj_st, vj = call.cap 6 1 (i32) -> (i64, i64) vh (vch)\n  vk1 = i64.const 1000\n  vsk = i64.mul vs vk1\n  vt = i64.add vj vsk\n  vk2 = i64.const 1000000\n  vstk = i64.mul vj_st vk2\n  vt2 = i64.add vt vstk\n  vc = i32.wrap_i64 vt2\n",
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
            && src.contains("vr0 = i64.const 17\n")
            && src.contains("svc.wait")
            && src.contains(&format!("va = i64.const {fault}\n")),
        "the rewrites must apply"
    );
    assert!(
        runs_on_bytecode(&src),
        "the cooperative driver pages its child itself (#2164)"
    );
    src
}

/// #1862 + #1863: a **detached** demand child — a window the pager cannot address — is served by a
/// pager that fills its own buffer and replies with its address. The child's first touch (at 20000)
/// faults into a `page(addr)` the root serves from `svc.wait`; the child reads 77; the root returns
/// `77 + 1 serve * 1000`.
#[test]
fn a_detached_demand_child_is_served_by_a_pager_that_cannot_address_it() {
    let src = detached_pager_program_at(20000);
    for b in DRIVERS {
        assert_eq!(run_detached(b, &src).expect("run"), 1077, "{b:?}");
    }
}

/// #2095: a paged child attests `window_exposed`. Its spawner is its pager, which chooses each page's
/// contents at first touch and sees every fault address in the child's coordinates. The child returns
/// `self.attest` after its first touch: tier 1 with the exposed bit (256), plus 1 serve * 1000.
#[test]
fn a_paged_child_attests_its_window_exposed() {
    let src = detached_pager_program_at(20000).replace(
        "  return vbw\n",
        "  vz = i32.const 0\n  vat = call.cap 4294967295 4 () -> (i64) vz ()\n  return vat\n",
    );
    assert!(src.contains("return vat\n"), "the rewrite must apply");
    for b in DRIVERS {
        assert_eq!(run_detached(b, &src).expect("run"), 1257, "{b:?}");
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
        for b in DRIVERS {
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
    for b in DRIVERS {
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
/// no-progress answer) so its `join` answers the child's `MEMORY_FAULT` — the run ends instead of
/// hanging.
///
/// #2469 — a bulk op on the guard is fatal too: a `mem.copy` from it, a `mem.fill` of it. Their span
/// check had no guard arm, so it recorded a recoverable fault, the pager was asked for the guard
/// page, and its supply mapped the guard read-write.
#[test]
fn pager_child_guard_fault_is_fatal() {
    let faulted = temen_ir::trap_code::MEMORY_FAULT as i32 * 1_000_000;
    for (what, access) in [
        ("a load", None),
        (
            "a copy's source",
            Some("  vd = i64.const 32768\n  vn = i64.const 4\n  mem.copy vd va vn\n  vb = i32.load8_u vd\n"),
        ),
        (
            "a fill",
            Some("  vf = i32.const 5\n  vn = i64.const 4\n  mem.fill va vf vn\n  vb = i32.load8_u va\n"),
        ),
    ] {
        let mut src = detached_pager_program_at(0);
        if let Some(access) = access {
            src = src.replace("  vb = i32.load8_u va\n", access);
            assert!(src.contains(access), "the rewrite must apply");
        }
        for b in DRIVERS {
            assert_eq!(
                run_detached(b, &src).expect("run"),
                faulted,
                "{b:?}: {what} of the NULL guard in a paged child is fatal to it"
            );
        }
    }
}

/// #2469 — an access that crosses from a supplied page into one not yet supplied faults on the page
/// it needs. The child touches 32765, which brings in the page below 32768 (a 16 KiB boundary, so a
/// page boundary on any host), then reads a word across it: a scalar load, a `mem.copy` whose source
/// crosses, a `mem.fill` whose destination does. The fault used to be recorded where the access
/// starts, in the page already supplied: the pager supplied that page again, and the child faulted
/// again. The root serves both faults and returns `child + 2 * 1000`. The child returns the byte it
/// touched, 77, plus the word: the pager's 77 lands at 32768, the word's third byte.
#[test]
fn a_paged_access_that_crosses_into_an_unsupplied_page_faults_on_that_page() {
    let read = "  vx = i64.const 32766\n  vw = i32.load vx\n";
    let copy = "  vd = i64.const 32700\n  vsrc = i64.const 32760\n  vn = i64.const 16\n  \
                mem.copy vd vsrc vn\n  vx = i64.const 32706\n  vw = i32.load vx\n";
    let fill = "  vd = i64.const 32760\n  vf = i32.const 5\n  vn = i64.const 16\n  \
                mem.fill vd vf vn\n  vx = i64.const 32766\n  vw = i32.load vx\n";
    for (what, access, word) in [
        ("a load", read, 77 << 16),
        ("a copy's source", copy, 77 << 16),
        ("a fill", fill, 0x0505_0505),
    ] {
        let src = detached_pager_program_at(32765)
            .replace(
                "  vs = svc.wait vz\n",
                "  vs1 = svc.wait vz\n  vs2 = svc.wait vz\n  vs = i64.add vs1 vs2\n",
            )
            .replace(
                "  return vbw\n",
                &format!(
                    "{access}  vw64 = i64.extend_i32_u vw\n  vr = i64.add vbw vw64\n  return vr\n"
                ),
            );
        assert!(
            src.contains("vs2 = svc.wait") && src.contains(access),
            "the rewrites must apply"
        );
        for b in DRIVERS {
            assert_eq!(
                run_detached(b, &src).expect("run"),
                77 + word + 2000,
                "{what} on {b:?}"
            );
        }
    }
}

/// #1217: a demand child that **never faults** — it returns `5` without touching its window — must
/// not strand the pager parked in `svc.wait`. The wait returns `0` once the child is gone; the
/// `join` delivers `5`; the root returns `0 * 1000 + 5`.
#[test]
fn pager_child_that_never_faults_releases_the_parked_pager() {
    let src =
        detached_pager_program_at(0).replace("  vb = i32.load8_u va\n", "  vb = i32.const 5\n");
    assert!(src.contains("vb = i32.const 5\n"), "the rewrite must apply");
    for b in DRIVERS {
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
            "  vs = svc.wait vz\n  vj_st, vj = call.cap 6 1 (i32) -> (i64, i64) vh (vch)\n",
            "  vj_st, vj = call.cap 6 1 (i32) -> (i64, i64) vh (vch)\n  vs = svc.wait vz\n",
        );
    assert!(
        src.contains("vb = i32.const 5\n") && src.contains("(vch)\n  vs = svc.wait"),
        "the rewrites must apply"
    );
    for b in DRIVERS {
        assert_eq!(run_detached(b, &src).expect("run"), 5, "{b:?}");
    }
}

/// #1862: a pager reply names a source the pager's window can't supply — here 1 MiB past a buffer
/// it did fill, beyond its 128 KiB window. The fault is unserviceable, so the child dies as a
/// pagerless fault would (detect-and-kill), after the one serve, and the parent's `join` answers its
/// `MEMORY_FAULT`: the child never reads unsupplied bytes.
#[test]
fn a_pager_reply_outside_its_window_is_a_fatal_fault() {
    let src = detached_pager_program_at(20000).replace(
        "  return vsrc\n",
        "  vpast = i64.const 1048576\n  vbad = i64.add vsrc vpast\n  return vbad\n",
    );
    assert!(src.contains("return vbad"), "the rewrite must apply");
    let faulted = temen_ir::trap_code::MEMORY_FAULT as i32 * 1_000_000 + 1000;
    for b in DRIVERS {
        assert_eq!(
            run_detached(b, &src).expect("run"),
            faulted,
            "{b:?}: an unsuppliable page is fatal to the child"
        );
    }
}

/// Owner, 2026-09-29: `Budget.mem` accounts **live** windows — a detached child's window goes back to
/// the budget that paid for it when the child ends. With a budget of exactly one window (2^17), the
/// program spawns its child, joins, and spawns it again: both run (`142 + 142`). Before, the first
/// spawn spent the budget for good and the second was refused — a program could spawn one child,
/// ever.
#[test]
fn a_joined_detached_childs_window_returns_to_its_budget() {
    let src = detached_record_program().replace(
        "  vj_st, vj = call.cap 6 1 (i32) -> (i64, i64) vh (vch)\n  vc = i32.wrap_i64 vj\n",
        "  vj_st, vj = call.cap 6 1 (i32) -> (i64, i64) vh (vch)\n  vch2 = call.cap 6 17 (i64) -> (i32) vh (vrp)\n  vj2_st, vj2 = call.cap 6 1 (i32) -> (i64, i64) vh (vch2)\n  vjs = i64.add vj vj2\n  vc = i32.wrap_i64 vjs\n",
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

/// #1944 — three generations of one `memory 17` module, each a child image (#2219): the root splits
/// its budget into a node with a `mem` ceiling of `child_ceiling` and spawns `"child"` detached,
/// paid from it and granted `"grandchild"`; the child spawns `"grandchild"` from its own
/// `"budget"` — the node that paid for it — and returns `join + 20`, or the refusal's `-errno`. The
/// grandchild returns 3, and the root returns its child's result plus 100.
fn three_generations(child_ceiling: i64) -> String {
    let r1 = SpawnRec {
        grants_ptr: 16640,
        grants_n: 1,
        ..SpawnRec::default()
    };
    format!(
        "memory 17
data 16384 \"vm\"
data 16400 \"budget\"
data 16448 \"child\"
data 16456 \"grandchild\"
{r1}{r2}export 0 func \"child\" 1
export 1 func \"grandchild\" 2

func 0 () -> (i32) {{
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
  bf = i64.const 17420
  i32.store bf vsub
  cp = i64.const 16448
  cl = i64.const 5
  vc1 = self.resolve cp cl
  mf = i64.const 17416
  i32.store mf vc1
  gp = i64.const 16456
  gl = i64.const 10
  vc2 = self.resolve gp gl
{grandchild_grant}  rp = i64.const 17408
  vch = call.cap 6 17 (i64) -> (i32) vh (rp)
  vj_st, vj = call.cap 6 1 (i32) -> (i64, i64) vh (vch)
  k = i64.const 100
  vr = i64.add vj k
  vc = i32.wrap_i64 vr
  return vc
  }}
}}
func 1 (i64) -> (i64) {{
block 0 (va: i64) {{
  vinst = i32.wrap_i64 va
  np = i64.const 16400
  nl = i64.const 6
  vb = self.resolve np nl
  bf = i64.const 17516
  i32.store bf vb
  gp = i64.const 16456
  gl = i64.const 10
  vg = self.resolve gp gl
  mf = i64.const 17512
  i32.store mf vg
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
  vj_st, vj = call.cap 6 1 (i32) -> (i64, i64) vi (vc)
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
        r1 = rec::segment(17408, &r1),
        r2 = rec::segment(17504, &SpawnRec::default()),
        grandchild_grant = rec::grant("g", 16640, 16456, 10, "vc2"),
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
/// that paid for the window, on every tier. The root pays for `"child"` from a node capped at the
/// child's 64 KiB window plus 64 KiB; the child grows 64 KiB (it fits), asks for 64 KiB more
/// (`-ENOMEM`), gives the first back (refunded at once), grows the second (it fits again), and
/// returns a bit per answer. The root returns `child * 10 + 1` when its node's room is the whole
/// ceiling again after the join: the child's end handed back the growth it still held.
fn growth_program() -> String {
    format!(
        "memory 16
data 17920 \"vm\"
data 17928 \"budget\"
data 16448 \"child\"
{rec}export 0 func \"child\" 1

func 0 () -> (i32) {{
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
  rb = i64.const 17420
  i32.store rb vsub
  mp = i64.const 16448
  ml = i64.const 5
  vmod = self.resolve mp ml
  mf = i64.const 17416
  i32.store mf vmod
  rp = i64.const 17408
  vch = call.cap 6 17 (i64) -> (i32) vinst (rp)
  vr_st, vr = call.cap 6 1 (i32) -> (i64, i64) vinst (vch)
  one = i64.const 1
  vroom = call.cap 14 1 (i64) -> (i64) vsub (one)
  vfull = i64.eq vroom m
  vfw = i64.extend_i32_u vfull
  ten = i64.const 10
  vhi = i64.mul vr ten
  vsum = i64.add vhi vfw
  vc = i32.wrap_i64 vsum
  return vc
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
        rec = rec::segment(17408, &SpawnRec::default()),
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
/// window, on every tier. The root pays for `"child"` from a node whose `channel` ceiling is one 64
/// KiB region; the child mints one (it fits) and asks for a second (`-ENOMEM`), returning a bit per
/// answer. The root returns `child * 10 + 1` when its node's `channel` room is the whole ceiling
/// again after the join: the child's end let go of the region it still held.
fn region_program() -> String {
    format!(
        "memory 16
data 17920 \"vm\"
data 17928 \"budget\"
data 16448 \"child\"
{rec}export 0 func \"child\" 1

func 0 () -> (i32) {{
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
  rb = i64.const 17420
  i32.store rb vsub
  mp = i64.const 16448
  ml = i64.const 5
  vmod = self.resolve mp ml
  mf = i64.const 17416
  i32.store mf vmod
  rp = i64.const 17408
  vch = call.cap 6 17 (i64) -> (i32) vinst (rp)
  vr_st, vr = call.cap 6 1 (i32) -> (i64, i64) vinst (vch)
  three = i64.const 3
  vroom = call.cap 14 1 (i64) -> (i64) vsub (three)
  vfull = i64.eq vroom c
  vfw = i64.extend_i32_u vfull
  ten = i64.const 10
  vhi = i64.mul vr ten
  vsum = i64.add vhi vfw
  vc = i32.wrap_i64 vsum
  return vc
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
        rec = rec::segment(17408, &SpawnRec::default()),
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
