//! #1296 slice 1 — a §14 child holds a **`Jit` capability of its own**. A parent re-grants its `jit`
//! handle by name (op 11's grant list); the child gets a **fresh, empty unit table** — not an alias of
//! the parent's (units, installs and B2 dispatch slots are per-table artifacts) — with a compile quota
//! attenuated to at most the parent's remaining, and compiles + invokes a unit over its own window.
//! The memory-match precondition resolves against the child's *own* module at compile time (here the
//! same module as the parent, `memory 17`). Both interpreter engines must agree.
//!
//! Also pinned: the quota attenuates (a parent with one unit left hands the child one unit; the
//! child's second `compile` is `-ENOMEM`), and a child spawned **without** the grant has no `jit`
//! name to resolve (the resolve returns a negative errno, never a handle into the parent's table).
//!
//! #1296 slice 2 — the **native JIT** runs the same program (`run_jit`): a §14 child compiles to the
//! root's shape (`CompiledModule`), so its re-granted `Jit` table compiles units into the child's own
//! module and installs into the child's own dispatch table; the four outcomes must match the
//! interpreter's byte-for-byte.

#[path = "../../temen-interp/tests/support/rec.rs"]
mod rec;

use temen_encode::encode_module;
use temen_interp::{bytecode, run_with_host, Host, MemLayout, Trap, Value};
use temen_ir::{SpawnRec, DEFAULT_RESERVED_LOG2};
use temen_jit::JitOutcome;
use temen_run::grant_jit;
use temen_text::parse_module;
use temen_verify::verify_module;

/// Where the unit blob sits, a data segment above the NULL guard and the grant scratch: the child
/// runs the parent's child image, which keeps its data, so its own window holds the blob there too.
const BLOB_OFF: usize = 20480;

/// The unit the child compiles, declaring the module's memory (`memory 17`): `(a, b) -> a + b`, or for
/// the install probe a 0-arg `() -> 42` (an installed slot is `call.dyn`ed with no args).
fn blob(zero_arg: bool) -> Vec<u8> {
    let src = if zero_arg {
        "memory 17\nfunc () -> (i32) {\nblock 0 () {\n  v0 = i32.const 42\n  return v0\n  }\n}\n"
    } else {
        "memory 17\nfunc (i32, i32) -> (i32) {\nblock 0 (v0: i32, v1: i32) {\n  v2 = i32.add v0 v1\n  return v2\n  }\n}\n"
    };
    let m = parse_module(src).expect("parse blob");
    verify_module(&m).expect("verify blob");
    encode_module(&m)
}

/// func 0 (parent, `(Instantiator, Jit, Budget, Module)`): one grant record at 16384 naming the `Jit`
/// handle `"jit"` (the name a data segment at 16484 holds), spawn the child detached through a v1
/// record at 17536 (the `Module`, func 1's child image, #2219, paid from the `Budget`, grant list
/// `(16384, grants_n)`), join, return `status * 1000 + value` (#2066): its result, or its trap's wire
/// code times 1000.
/// func 1 (child, `(Instantiator)`): resolve `"jit"` by name (the same data in its own window),
/// compile the blob at `BLOB_OFF` `times` times (the last compile's code handle is what it invokes),
/// invoke `(3, 4)`, return the sum. `times == 2` is the quota probe: it returns the **second compile's**
/// result instead (`-ENOMEM` when the child's table has one unit). Modes 5/6: the child `install`s
/// its unit and returns the slot; in mode 5 the parent `call.dyn`s that slot (in its own table), in
/// mode 6 it returns the slot as is. Mode 7: the parent installs a unit of its own first; the child
/// installs one into its own table, at the same index, then uninstalls it; the parent returns
/// `its slot << 16` plus the child's uninstall result.
fn src(grants_n: u32, blob: &[u8], mode: u32) -> String {
    let blob_len = blob.len();
    let blob_data: String = blob.iter().map(|b| format!("\\x{b:02x}")).collect();
    let spawn = SpawnRec {
        grants_ptr: 16384,
        grants_n: u64::from(grants_n),
        ..SpawnRec::default()
    };
    let child_pre = format!(
        "  n0 = i64.const 16484\n  len3 = i64.const 3\n  hj = self.resolve n0 len3\n  vb = i64.const {}\n  vl = i64.const {}\n  vc = call.cap 11 0 (i64, i64) -> (i64) hj (vb, vl)\n",
        BLOB_OFF, blob_len
    );
    let child_tail = match mode {
        2 => "  vc2 = call.cap 11 0 (i64, i64) -> (i64) hj (vb, vl)\n  return vc2\n".to_string(),
        5 | 6 => "  vslot = call.cap 11 3 (i64) -> (i64) hj (vc)\n  return vslot\n".to_string(),
        7 => "  vslot = call.cap 11 3 (i64) -> (i64) hj (vc)\n  vu = call.cap 11 4 (i64) -> (i64) hj (vslot)\n  return vu\n".to_string(),
        _ => "  va = i32.const 3\n  vb4 = i32.const 4\n  vr = call.cap 11 1 (i64, i32, i32) -> (i32) hj (vc, va, vb4)\n  vr64 = i64.extend_i32_s vr\n  return vr64\n".to_string(),
    };
    // Mode 5: the parent `call.dyn`s the slot the child installed — in the PARENT's table, which the
    // child's install never touched, so it must trap (the child's units are its own).
    let parent_pre = if mode == 7 {
        format!(
            "  pb = i64.const {BLOB_OFF}\n  pl = i64.const {blob_len}\n  pc = call.cap 11 0 (i64, i64) -> (i64) vjit (pb, pl)\n  pslot = call.cap 11 3 (i64) -> (i64) vjit (pc)\n"
        )
    } else {
        String::new()
    };
    let parent_tail = match mode {
        5 => "  rs = i32.wrap_i64 r\n  rd = call.dyn () -> (i32) rs ()\n  rd64 = i64.extend_i32_s rd\n  return rd64\n",
        7 => "  k16 = i64.const 65536\n  ps = i64.mul pslot k16\n  out = i64.add ps r\n  return out\n",
        _ => "  k1000 = i64.const 1000\n  rs1000 = i64.mul r_st k1000\n  out = i64.add rs1000 r\n  return out\n",
    };
    format!(
        r#"memory 17
func (i32, i32, i32, i32) -> (i64) {{
block 0 (vinst: i32, vjit: i32, vbud: i32, vmod: i32) {{
{parent_pre}  a0 = i64.const 16384
  n100 = i32.const 16484
  i32.store a0 n100
  a4 = i64.const 16388
  n3 = i32.const 3
  i32.store a4 n3
  a8 = i64.const 16392
  i32.store a8 vjit
  a12 = i64.const 16396
  z0 = i32.const 0
  i32.store a12 z0
  qm = i64.const 17544
  i32.store qm vmod
  qb = i64.const 17548
  i32.store qb vbud
  q0a0 = i64.const 17536
  vch = call.cap 6 17 (i64) -> (i32) vinst (q0a0)
  r_st, r = call.cap 6 1 (i32) -> (i64, i64) vinst (vch)
{parent_tail}  }}
}}
func (i64) -> (i64) {{
block 0 (vci: i64) {{
{child_pre}{child_tail}  }}
}}
data 16484 "jit"
data {BLOB_OFF} "{blob_data}"
{rec}"#,
        rec = rec::segment(17536, &spawn),
    )
}

/// The `Jit` table reservation: 16 install slots for the install probes (modes 5–7).
fn table_log2(mode: u32) -> u8 {
    if mode >= 5 {
        4
    } else {
        0
    }
}

/// The parent's host for one run: an `Instantiator`, a `Jit` table (16 install slots for the install
/// probe) with a `units` compile quota, a `Budget` that pays for the child, and the child.
fn setup(m: &temen_ir::Module, mode: u32, units: u32) -> (Host, [i32; 4]) {
    let mut host = Host::new();
    let ih = host.grant_instantiator();
    let jh = grant_jit(&mut host, m, table_log2(mode));
    host.set_jit_quota(units, 1 << 20);
    let bh = host.grant_budget(-1, 1 << 20, -1);
    let ch = host.grant_module(&temen_ir::child_image_at(m, 1).expect("child image"));
    (host, [ih, jh, bh, ch])
}

/// Run on the tree-walker and on the bytecode engine with the same host setup (a root `Jit` grant of
/// `units` units); return both outcomes for the caller's assertions.
fn run_both(grants_n: u32, mode: u32, units: u32) -> [Result<Vec<Value>, Trap>; 2] {
    let b = blob(mode >= 5);
    let m = parse_module(&src(grants_n, &b, mode)).expect("parse");
    verify_module(&m).expect("verify");
    let mut out = Vec::new();
    for engine in 0..2 {
        let (mut host, handles) = setup(&m, mode, units);
        let args = handles.map(Value::I32);
        let mut fuel = 5_000_000u64;
        let r = if engine == 0 {
            run_with_host(&m, 0, &args, &mut fuel, &mut host)
        } else {
            match bytecode::compile_and_run_with_host(&m, 0, &args, &mut fuel, &mut host) {
                Some(r) => r,
                None => panic!(
                    "the bytecode engine declined; tree-walker said {:?}",
                    out[0]
                ),
            }
        };
        out.push(r);
    }
    [out.remove(0), out.remove(0)]
}

/// The same program on the native JIT, with temen-run's production child hooks (the child is built
/// host-side by the detached builder, which reports the `Jit` grant's table reservation).
fn run_jit(grants_n: u32, mode: u32, units: u32) -> JitOutcome {
    let b = blob(mode >= 5);
    let m = parse_module(&src(grants_n, &b, mode)).expect("parse");
    verify_module(&m).expect("verify");
    let (mut host, handles) = setup(&m, mode, units);
    temen_run::jit_cap_run(
        &m,
        0,
        &handles.map(i64::from),
        &MemLayout::image(Vec::new()),
        DEFAULT_RESERVED_LOG2,
        table_log2(mode), // the root's own table, as the grant reserves it
        &mut host,
        None,
    )
    .expect("jit")
    .0
}

/// Interp `Ok([I64(x)])` ≡ JIT `Returned([x])`; any interp trap ≡ any JIT trap (the kind is not part
/// of the pinned contract here — the interp's `IndirectCallType` vs the JIT's `Trapped(..)`).
fn agrees(interp: &Result<Vec<Value>, Trap>, jit: &JitOutcome) -> bool {
    match (interp, jit) {
        (Ok(v), JitOutcome::Returned(r)) => {
            matches!((v.first(), r.first()), (Some(Value::I64(a)), Some(b)) if a == b)
        }
        (Err(_), JitOutcome::Trapped(_)) => true,
        _ => false,
    }
}

#[test]
fn a_child_compiles_and_invokes_a_unit_on_its_own_jit_table() {
    let [tw, bc] = run_both(1, 1, 4096);
    assert_eq!(
        tw,
        Ok(vec![Value::I64(7)]),
        "tree-walker: child invoked 3 + 4"
    );
    assert_eq!(bc, tw, "bytecode engine agrees");
    let jo = run_jit(1, 1, 4096);
    assert!(agrees(&tw, &jo), "native JIT agrees: {jo:?}");
}

#[test]
fn the_child_table_quota_is_at_most_the_parents_remaining() {
    // The parent has one unit left; the child inherits one, so its second compile is -ENOMEM.
    let [tw, bc] = run_both(1, 2, 1);
    assert_eq!(
        tw,
        Ok(vec![Value::I64(-12)]),
        "tree-walker: ENOMEM on the second compile"
    );
    assert_eq!(bc, tw, "bytecode engine agrees");
    let jo = run_jit(1, 2, 1);
    assert!(agrees(&tw, &jo), "native JIT agrees: {jo:?}");
}

#[test]
fn a_child_without_the_grant_cannot_reach_the_parents_table() {
    // No `jit` name resolves, so the child's compile goes through a negative handle and traps
    // `CapFault`, which its parent's join answers (#2066).
    let want = temen_ir::trap_code::CAP_FAULT * 1000;
    let [tw, bc] = run_both(0, 1, 4096);
    assert_eq!(tw, Ok(vec![Value::I64(want)]), "tree-walker");
    assert_eq!(bc, tw, "both engines refuse the same way");
    let jo = run_jit(0, 1, 4096);
    assert!(agrees(&tw, &jo), "native JIT agrees: {jo:?}");
}

#[test]
fn a_childs_install_is_invisible_to_the_parents_dispatch_table() {
    // The child installs its unit and reports the slot; the parent's `call.dyn` on that slot finds
    // nothing (its own table was never written) and traps — on both engines.
    let [tw, bc] = run_both(1, 5, 4096);
    assert!(
        tw.is_err(),
        "tree-walker: the parent's call.dyn must trap, got {tw:?}"
    );
    assert!(
        bc.is_err(),
        "bytecode: the parent's call.dyn must trap, got {bc:?}"
    );
    let jo = run_jit(1, 5, 4096);
    assert!(
        matches!(jo, JitOutcome::Trapped(_)),
        "native JIT: the parent's call.dyn must trap, got {jo:?}"
    );
}

#[test]
fn a_child_installs_into_its_own_reserved_table() {
    // The `Jit` grant carries the parent's table reservation (16 slots) into the child, so the
    // child's `install` lands in a padding slot of the CHILD's dispatch table: a non-negative slot
    // (never `-ENOSPC` from a natural, padding-free table), identical on every engine.
    let [tw, bc] = run_both(1, 6, 4096);
    assert!(
        matches!(tw.as_deref(), Ok([Value::I64(s)]) if *s >= 0),
        "tree-walker: the child's install must return a slot, got {tw:?}"
    );
    assert_eq!(bc, tw, "bytecode engine agrees on the slot");
    let jo = run_jit(1, 6, 4096);
    assert!(agrees(&tw, &jo), "native JIT agrees: {jo:?}");
}

#[test]
fn a_childs_uninstall_leaves_the_parents_slot_mirror_alone() {
    // Both tables fill slot 2 first (two natural functions); the child's uninstall clears its own.
    let [tw, bc] = run_both(1, 7, 4096);
    assert_eq!(
        tw,
        Ok(vec![Value::I64(2 << 16)]),
        "tree-walker: the parent installed at slot 2, the child's uninstall answered 0"
    );
    assert_eq!(bc, tw, "bytecode engine agrees");
    let jo = run_jit(1, 7, 4096);
    assert!(agrees(&tw, &jo), "native JIT agrees: {jo:?}");

    // The cooperative driver's B2 mirror is the root's emitted dispatch table: the child's
    // uninstall of its own slot 2 must not empty the root's.
    let b = blob(true);
    let m = parse_module(&src(1, &b, 7)).expect("parse");
    verify_module(&m).expect("verify");
    let (host, handles) = setup(&m, 7, 4096);
    let mut run = bytecode::CoopRun::new(&m, 0, &handles.map(Value::I32), 5_000_000, host, None)
        .expect("bytecode coverage")
        .expect("entry in range");
    let bytecode::CoopEvent::Done(v) = run.run() else {
        panic!("the run ends");
    };
    assert_eq!(v, vec![Value::I64(2 << 16)]);
    assert!(
        run.slot_unit(2).is_some(),
        "the root's mirror still names its own unit at slot 2"
    );
}

/// #3158 — a whole-vCPU park made inside a `Jit.invoke`, on a pool task. The child [`joining_src`]
/// spawns a grandchild (function 4's image), then compiles a unit and invokes it. The unit
/// `call.dyn`s the child's function 2 (new→old), which makes a fiber over function 3 and resumes it,
/// and that fiber joins the grandchild. A join blocks the child's vCPU on every engine. The
/// grandchild waits with no deadline on a word nothing notifies. On the JIT a task cannot decide the
/// deadlock verdict: an OS vCPU's park decides it once the run is quiescent
/// (`ChildExec::wake_vcpus_if_quiescent`), and the run is not while the child runs to its join. So
/// the grandchild is live at the join, however slow the run, with no clock and no fuel in it. On
/// the oracle it reaches its verdict, which the oracle's answer does not read: the oracle's invoke
/// child refuses the join itself.
fn joining_src(blob: &[u8]) -> String {
    let blob_data: String = blob.iter().map(|b| format!("\\x{b:02x}")).collect();
    let spawn = SpawnRec {
        grants_ptr: 16384,
        grants_n: 2,
        ..SpawnRec::default()
    };
    let grand = SpawnRec::default();
    // One grant record: the name at `at`'s slot `name`, `len` bytes, and the handle `h`.
    let grant = |i: u32, name: u32, len: u32, h: &str| {
        let a = 16384 + 16 * i;
        format!("  g{i}a = i64.const {a}\n  g{i}n = i32.const {name}\n  i32.store g{i}a g{i}n\n  g{i}b = i64.const {}\n  g{i}l = i32.const {len}\n  i32.store g{i}b g{i}l\n  g{i}c = i64.const {}\n  i32.store g{i}c {h}\n  g{i}d = i64.const {}\n  g{i}z = i32.const 0\n  i32.store g{i}d g{i}z\n", a + 4, a + 8, a + 12)
    };
    format!(
        r#"memory 17
func (i32, i32, i32, i32, i32) -> (i64) {{
block 0 (vinst: i32, vjit: i32, vbud: i32, vmod: i32, vgmod: i32) {{
{g0}{g1}  qm = i64.const 17544
  i32.store qm vmod
  qb = i64.const 17548
  i32.store qb vbud
  q0a0 = i64.const 17536
  vch = call.cap 6 17 (i64) -> (i32) vinst (q0a0)
  r_st, r = call.cap 6 1 (i32) -> (i64, i64) vinst (vch)
  k1000 = i64.const 1000
  rs1000 = i64.mul r_st k1000
  out = i64.add rs1000 r
  return out
  }}
}}
func (i64) -> (i64) {{
block 0 (vci: i64) {{
  n0 = i64.const 16484
  len3 = i64.const 3
  hj = self.resolve n0 len3
  n1 = i64.const 16488
  hg = self.resolve n1 len3
  gm = i64.const 17672
  i32.store gm hg
  nb = i64.const 16496
  len6 = i64.const 6
  hb = self.resolve nb len6
  gb = i64.const 17676
  i32.store gb hb
  vinst = i32.wrap_i64 vci
  gr = i64.const 17664
  vg = call.cap 6 17 (i64) -> (i32) vinst (gr)
  vga = i64.const 70008
  i32.store vga vg
  vgi = i64.const 70012
  i32.store vgi vinst
  vb = i64.const {BLOB_OFF}
  vl = i64.const {len}
  vc = call.cap 11 0 (i64, i64) -> (i64) hj (vb, vl)
  vh = call.cap 11 1 (i64) -> (i64) hj (vc)
  v100 = i64.const 100
  vr = i64.add vh v100
  return vr
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (vsp: i64, varg: i64) {{
  vf = i32.const 3
  vfsp = i64.const 32768
  vk = cont.new vf vfsp
  vz = i64.const 0
  vs, vv = cont.resume vk vz
  return vv
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (vsp: i64, varg: i64) {{
  vga = i64.const 70008
  vg = i32.load vga
  vgi = i64.const 70012
  vinst = i32.load vgi
  vj_st, vj = call.cap 6 1 (i32) -> (i64, i64) vinst (vg)
  return vj
  }}
}}
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  vp = i64.const 70000
  vw0 = i32.const 0
  vto = i64.const -1
  vw = i32.atomic.wait vp vw0 vto
  v7 = i64.const 7
  return v7
  }}
}}
data 16484 "jit"
data 16488 "gcm"
data 16496 "budget"
data {BLOB_OFF} "{blob_data}"
{rec}
{grec}"#,
        g0 = grant(0, 16484, 3, "vjit"),
        g1 = grant(1, 16488, 3, "vgmod"),
        len = blob.len(),
        rec = rec::segment(17536, &spawn),
        grec = rec::segment(17664, &grand),
    )
}

/// #3158 — the join of [`joining_src`]. On the JIT the child is a task and the invoke a guarded call
/// nested in its frames, so the task cannot park there whole: the join fails closed at its own door
/// (`ThreadFault`) before it parks, and the parent's join reads that trap. Before the door the join
/// parked the fiber, and the park's hand-up found the task under the invoke's floor and left the
/// fiber parked mid-join (the hand-up's `debug_assert`, which a debug build fails on). The
/// tree-walker's invoke child refuses the join (`CapFault`): the invoke divergence #3154 records,
/// which #3155 removes.
#[test]
fn a_join_inside_an_invoke_on_a_pool_task_fails_closed_at_its_door() {
    let unit = "memory 17\nfunc () -> (i64) {\nblock 0 () {\n  vf = i32.const 2\n  vz = i64.const 0\n  vr = call.dyn (i64, i64) -> (i64) vf (vz, vz)\n  return vr\n  }\n}\n";
    let b = encode_module(&parse_module(unit).expect("parse unit"));
    let m = parse_module(&joining_src(&b)).expect("parse");
    verify_module(&m).expect("verify");
    fn setup(m: &temen_ir::Module) -> (Host, [i32; 5]) {
        let mut host = Host::new();
        let ih = host.grant_instantiator();
        let jh = grant_jit(&mut host, m, 0);
        let bh = host.grant_budget(-1, 1 << 20, -1);
        let ch = host.grant_module(&temen_ir::child_image_at(m, 1).expect("child image"));
        let gh = host.grant_module(&temen_ir::child_image_at(m, 4).expect("grandchild image"));
        (host, [ih, jh, bh, ch, gh])
    }
    // The parent answers `status * 1000 + value` of its join of the child.
    let trapped = |code: i64| code * 1000;
    // On its own thread with a bound in time, so a run that does not end fails rather than hangs.
    let r = {
        let m = m.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let (mut host, h) = setup(&m);
            let mut fuel = 50_000_000u64;
            let _ = tx.send(run_with_host(
                &m,
                0,
                &h.map(Value::I32),
                &mut fuel,
                &mut host,
            ));
        });
        rx.recv_timeout(std::time::Duration::from_secs(60))
            .expect("the oracle's run ends")
    };
    assert_eq!(
        r,
        Ok(vec![Value::I64(trapped(temen_ir::trap_code::CAP_FAULT))]),
        "oracle"
    );
    let (mut host, h) = setup(&m);
    let (out, _) = temen_run::jit_cap_run(
        &m,
        0,
        &h.map(i64::from),
        &MemLayout::image(Vec::new()),
        DEFAULT_RESERVED_LOG2,
        0,
        &mut host,
        None,
    )
    .expect("jit");
    assert!(
        matches!(out, JitOutcome::Returned(ref v) if v == &[trapped(temen_ir::trap_code::THREAD_FAULT)]),
        "jit: the child's join fails closed at its door: {out:?}"
    );
}

/// #3158 review B1 — the parent runs under a lane cap of 1. It spawns the serving child
/// [`lane_server`] with its `Jit` and a punting host procedure granted by name, mints an
/// offer over the child's export, calls `run(5)` through it, joins the child, and returns the reply
/// plus 1000 times the join's status.
fn lane_parent() -> String {
    let spawn = SpawnRec {
        grants_ptr: 16384,
        grants_n: 2,
        ..SpawnRec::default()
    };
    format!(
        r#"memory 17
type 0 func (i64) -> (i64)
type 1 interface {{ run: 0 }}
func (i32, i32, i32, i32, i32) -> (i64) {{
block 0 (vinst: i32, vjit: i32, vbud: i32, vmod: i32, vproc: i32) {{
  a0 = i64.const 16384
  n0 = i32.const 16484
  i32.store a0 n0
  a4 = i64.const 16388
  l3 = i32.const 3
  i32.store a4 l3
  a8 = i64.const 16392
  i32.store a8 vjit
  a12 = i64.const 16396
  z0 = i32.const 0
  i32.store a12 z0
  b0 = i64.const 16400
  n1 = i32.const 16488
  i32.store b0 n1
  b4 = i64.const 16404
  i32.store b4 l3
  b8 = i64.const 16408
  i32.store b8 vproc
  b12 = i64.const 16412
  i32.store b12 z0
  qm = i64.const 17544
  i32.store qm vmod
  qb = i64.const 17548
  i32.store qb vbud
  q0a0 = i64.const 17536
  vch = call.cap 6 17 (i64) -> (i32) vinst (q0a0)
  vz = i64.const 0
  voff = call.cap 6 14 (i32, i64) -> (i32) vinst (vch, vz)
  v5 = i64.const 5
  vr = call.cap 268435456 0 (i64) -> (i64) voff (v5)
  r_st, r = call.cap 6 1 (i32) -> (i64, i64) vinst (vch)
  k1000 = i64.const 1000
  rs1000 = i64.mul r_st k1000
  out = i64.add rs1000 vr
  return out
  }}
}}
data 16484 "jit"
data 16488 "prc"
{rec}"#,
        rec = rec::segment(17536, &spawn),
    )
}

/// The unit [`lane_server`] compiles: it calls the punting procedure whose handle the child stored
/// at 70000 with 5, answering the job's 105.
const LANE_UNIT: &str = "memory 17\nfunc () -> (i64) {\nblock 0 () {\n  va = i64.const 70000\n  vh = i32.load va\n  v5 = i64.const 5\n  vr = call.cap 13 0 (i64) -> (i64) vh (v5)\n  return vr\n  }\n}\n";

/// #3158 review B1 — the serving child. Its entry resolves `"jit"` and `"prc"`, compiles
/// [`LANE_UNIT`] (the blob at [`BLOB_OFF`]), spawns a notifier vCPU (function 2), and serves one
/// call. The handler, `run(a)`, waits with no deadline on a word the notifier notifies until a notify
/// wakes a waiter, so the handler's fiber parks once, however late its wait begins, and a pool
/// worker resumes it. The notifier sleeps a millisecond on a word of its own between tries: that
/// only paces it. What makes the park certain is the wait's lack of a deadline and the notifier
/// trying until one of its notifies wakes a waiter. Then the handler invokes the unit, whose call
/// the host punts, and answers the unit's 105 plus `a`.
fn lane_server(blob: &[u8]) -> String {
    let blob_data: String = blob.iter().map(|b| format!("\\x{b:02x}")).collect();
    format!(
        r#"memory 17
type 0 func (i64) -> (i64)
type 1 interface {{ run: 0 }}
export 0 interface "svc" 1 {{ run: 1 }}
func (i64) -> (i64) {{
block 0 (vci: i64) {{
  n0 = i64.const 16484
  l3 = i64.const 3
  hj = self.resolve n0 l3
  n1 = i64.const 16488
  hp = self.resolve n1 l3
  ap = i64.const 70000
  i32.store ap hp
  aj = i64.const 70004
  i32.store aj hj
  vb = i64.const {BLOB_OFF}
  vl = i64.const {len}
  vc = call.cap 11 0 (i64, i64) -> (i64) hj (vb, vl)
  ac = i64.const 70008
  i64.store ac vc
  vnz = i64.const 0
  vnt = thread.spawn 2 vnz vnz
  vz = i32.const 0
  vn = svc.wait vz
  return vn
  }}
}}
func (i64) -> (i64) {{
block 0 (va: i64) {{
  vp = i64.const 70100
  vw0 = i32.const 0
  vto = i64.const -1
  vw = i32.atomic.wait vp vw0 vto
  aj = i64.const 70004
  hj = i32.load aj
  ac = i64.const 70008
  vc = i64.load ac
  vh = call.cap 11 1 (i64) -> (i64) hj (vc)
  vr = i64.add vh va
  return vr
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (vsp: i64, varg: i64) {{
  br 1()
}}
block 1 () {{
  vp = i64.const 70100
  vone = i32.const 1
  vn = atomic.notify vp vone
  vz = i32.const 0
  vnone = i32.eq vn vz
  br_if vnone 2() 3(vn)
}}
block 2 () {{
  vq = i64.const 70200
  vq0 = i32.const 0
  vqt = i64.const 1000000
  vqw = i32.atomic.wait vq vq0 vqt
  br 1()
}}
block 3 (vwoke: i32) {{
  vr = i64.extend_i32_s vwoke
  return vr
  }}
}}
data 16484 "jit"
data 16488 "prc"
data {BLOB_OFF} "{blob_data}"
"#,
        len = blob.len(),
    )
}

/// A host procedure whose every call the host punts: its job answers the argument plus 100. Forkable,
/// so a spawn can re-grant it to a child.
fn punting_proc() -> temen_interp::HostProcGrant {
    fn handler() -> temen_interp::ProcHandler {
        temen_interp::ProcHandler::Offloadable(Box::new(|_op, args, _out| {
            let a = *args.first().unwrap_or(&0);
            temen_interp::OffloadOutcome::Offload(Box::new(move || a + 100))
        }))
    }
    temen_interp::HostProcGrant::new(handler(), temen_interp::CapState::Stateless).forkable(
        std::sync::Arc::new(|_| {
            Some(temen_interp::ForkedProc {
                handler: handler(),
                signal: None,
                refork: None,
                exit: None,
                exec_remap: None,
                state: temen_interp::CapState::Stateless,
                abandon: None,
            })
        }),
    )
}

/// #3158 review B1 — a punted call's wait outside the pool, from a serve handler's shell on a pool
/// worker. The shell holds no lane, and while it runs the current domain is its serving child's: a
/// wait that gave back and re-took that domain's lanes re-took a lane the shell never held, and
/// under a cap of 1 the run then waited for it for good. The wait now gives back and re-takes only
/// the lanes its context holds as a vCPU (none, in a pool task), and the run answers `5 + 105`.
#[test]
fn a_punt_from_a_resumed_handler_under_a_lane_cap_of_one_completes() {
    let unit = encode_module(&parse_module(LANE_UNIT).expect("parse unit"));
    let parent = parse_module(&lane_parent()).expect("parse parent");
    verify_module(&parent).expect("verify parent");
    let server = parse_module(&lane_server(&unit)).expect("parse server");
    verify_module(&server).expect("verify server");
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut host = Host::new();
        host.set_lane_cap(1);
        let ih = host.grant_instantiator();
        let jh = grant_jit(&mut host, &server, 0);
        let bh = host.grant_budget(-1, 1 << 20, -1);
        let sh = host.grant_module(&server);
        let ph = host.grant_host_proc(punting_proc());
        let out = temen_run::jit_cap_run(
            &parent,
            0,
            &[ih, jh, bh, sh, ph].map(i64::from),
            &MemLayout::image(Vec::new()),
            DEFAULT_RESERVED_LOG2,
            0,
            &mut host,
            None,
        )
        .map(|(o, _)| o);
        let _ = tx.send(format!("{out:?}"));
    });
    let out = rx
        .recv_timeout(std::time::Duration::from_secs(30))
        .expect("the run completes: it hung waiting for a lane");
    assert_eq!(out, "Ok(Returned([110]))");
}
