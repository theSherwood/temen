//! #2322 — a detached child whose code runs on a guest fiber joins a child of its own from that
//! fiber, on the tree-walk oracle, the bytecode drivers and Cranelift alike.
//!
//! A join blocks its whole vCPU on every engine: the oracle parks the vCPU, the bytecode drivers the
//! task, and an OS vCPU on Cranelift its thread. Inside a Cranelift child task the fiber's park is
//! handed up to the task, so the fiber's resumer does not run on, and never sees `FIBER_PARKED`, while
//! the join waits. Each resumer drives its fiber the way JACL's scheduler drives its jobs, resuming
//! again on `FIBER_PARKED` (3) or a `suspend`, and counts every `FIBER_PARKED` it sees into its result,
//! so an engine that reports the join's park to a resumer shows up as a different answer. The joining
//! fiber runs under the child's root, and again under a fiber of its own, so the park has to pass
//! through a guest fiber's resume too. An outcome already in is returned at once.
//!
//! Before #2322 Cranelift trapped `ThreadFault` at every one of these joins: a task's join was refused
//! from any stack but the task's own, before the outcome was even read.

#[path = "../../temen-interp/tests/support/drivers.rs"]
mod drivers;
#[path = "../../temen-interp/tests/support/rec.rs"]
mod rec;

use drivers::{Driver, Ran};
use temen_interp::{Host, MemLayout, Value};
use temen_ir::{trap_code, Module, SpawnRec};
use temen_jit::{JitError, JitOutcome};

/// Where the root's record for the child sits.
const CHILD_REC: u64 = 17408;
/// Where the record for the child's own children sits, in the child's window: the program's data
/// segments seed it. The fiber writes the child's `"budget"` into it, and each child's module.
const GRANDCHILD_REC: u64 = 17536;
/// The name of a detached child's own budget, which it pays for its children from.
const BUDGET_NAME: u64 = 17920;
/// Where the names `"g3"`–`"g6"` sit, two bytes apart: the child images of funcs 3–6, which the root
/// grants the child under those names and the child spawns its own children from.
const MODULE_NAMES: u64 = 17952;
/// Where the root lays the grant list that hands the child those four modules.
const GRANTS: u64 = 18432;

/// The address of the name `"g{func}"`.
fn module_name(func: u32) -> u64 {
    MODULE_NAMES + 2 * u64::from(func - 3)
}

/// Text-IR stores that make func `func`'s child image the module of the record at
/// [`GRANDCHILD_REC`]: resolve `"g{func}"` and store it at the record's `module` (`vmp`).
fn pick(func: u32) -> String {
    format!(
        "  vgn{func} = i64.const {name}
  vgl{func} = i64.const 2
  vgm{func} = self.resolve vgn{func} vgl{func}
  i32.store vmp vgm{func}
",
        name = module_name(func),
    )
}
/// How many back-edges a grandchild that runs a while takes before it returns.
const SPIN: i64 = 100_000;
/// What a resumer multiplies its count of `FIBER_PARKED` resumes by.
const PARKED: i64 = 1_000_000_000_000;

/// The program: the root (func 0) spawns func 1 detached through the spawn record at [`CHILD_REC`] (its
/// args: an `Instantiator`, the `Budget` that pays, and the child image of func 1), granting it the
/// child images of funcs 3–6 as `"g3"`–`"g6"` ([`GRANTS`]), and joins it. The
/// child drives `fiber` (func 2) on a fiber ([`drive`]), passing it the child's `Instantiator` — or,
/// with `outer`, drives func 7 on a fiber, which drives func 2 in turn, so the join's park has two
/// resumers to pass through. Funcs 3 and 4 spin, then return 42 and 7; func 5 returns 5 at once; func
/// 6 reaches `unreachable`.
fn program(fiber: &str, outer: bool) -> Module {
    let names: String = (3..=6)
        .map(|f| format!("data {} \"g{f}\"\n", module_name(f)))
        .collect();
    // The root resolves each module the host named and lays the grant record that hands it on.
    let grants: String = (3..=6)
        .map(|f| {
            let at = GRANTS + 16 * u64::from(f - 3);
            let rec = rec::grant(&format!("q{f}"), at, module_name(f), 2, &format!("vh{f}"));
            format!(
                "  vrn{f} = i64.const {name}\n  vrl{f} = i64.const 2\n  vh{f} = self.resolve vrn{f} vrl{f}\n{rec}",
                name = module_name(f),
            )
        })
        .collect();
    let src = format!(
        "memory 17
data {BUDGET_NAME} \"budget\"
{names}func (i32, i32, i32) -> (i64) {{
block 0 (vinst: i32, vbud: i32, vmod: i32) {{
  vrm = i64.const {rec_module}
  i32.store vrm vmod
  vrb = i64.const {rec_budget}
  i32.store vrb vbud
{grants}  vrp = i64.const {CHILD_REC}
  vch = call.cap 6 17 (i64) -> (i32) vinst (vrp)
  vj_st, vj = call.cap 6 1 (i32) -> (i64, i64) vinst (vch)
  return vj
  }}
}}
func (i64) -> (i64) {{
block 0 (vi: i64) {{
{drive_from_root}
}}
func (i64, i64) -> (i64) {{
block 0 (vsp: i64, varg: i64) {{
  vi = i32.wrap_i64 varg
  vnp = i64.const {BUDGET_NAME}
  vnl = i64.const 6
  vb = self.resolve vnp vnl
  vbp = i64.const {gc_budget}
  i32.store vbp vb
  vrp = i64.const {GRANDCHILD_REC}
  vmp = i64.const {gc_module}
{fiber}
}}
{spin_42}
{spin_7}
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  v = i64.const 5
  return v
  }}
}}
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  unreachable
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (vsp: i64, vi: i64) {{
{drive_from_fiber}
}}
{child_rec}{grandchild_rec}",
        rec_module = CHILD_REC + SpawnRec::MODULE_AT,
        rec_budget = CHILD_REC + SpawnRec::BUDGET_AT,
        gc_budget = GRANDCHILD_REC + SpawnRec::BUDGET_AT,
        gc_module = GRANDCHILD_REC + SpawnRec::MODULE_AT,
        drive_from_root = drive(if outer { 7 } else { 2 }),
        drive_from_fiber = drive(2),
        spin_42 = spinner(42),
        spin_7 = spinner(7),
        child_rec = rec::segment(
            CHILD_REC,
            &SpawnRec {
                grants_ptr: GRANTS,
                grants_n: 4,
                ..SpawnRec::default()
            }
        ),
        grandchild_rec = rec::segment(GRANDCHILD_REC, &SpawnRec::default()),
    );
    let m = temen_text::parse_module(&src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// The body of a function whose block 0 has its `Instantiator` in `vi`: run func `fiber` on a fiber,
/// resuming it with `vi` until it returns — again on `suspend` (0), and on `FIBER_PARKED` (3), which
/// it counts, as JACL's scheduler does — and return its result plus that count × [`PARKED`].
fn drive(fiber: u32) -> String {
    format!(
        "  vf = ref.func {fiber}
  vz = i64.const 0
  vh = cont.new vf vz
  br 1(vh, vi, vz)
}}
block 1 (h1: i64, i1: i64, n1: i64) {{
  s1, r1 = cont.resume h1 i1
  vthree = i32.const 3
  vparked = i32.eq s1 vthree
  br_if vparked 2(h1, i1, n1) 3(h1, i1, n1, s1, r1)
}}
block 2 (h2: i64, i2: i64, n2: i64) {{
  vone = i64.const 1
  n2b = i64.add n2 vone
  br 1(h2, i2, n2b)
}}
block 3 (h3: i64, i3: i64, n3: i64, s3: i32, r3: i64) {{
  vret = i32.const 1
  vdone = i32.eq s3 vret
  br_if vdone 4(n3, r3) 1(h3, i3, n3)
}}
block 4 (n4: i64, r4: i64) {{
  vk = i64.const {PARKED}
  vp = i64.mul n4 vk
  vout = i64.add vp r4
  return vout
  }}"
    )
}

/// A child entry that takes [`SPIN`] back-edges, then returns `value`.
fn spinner(value: i64) -> String {
    format!(
        "func (i64) -> (i64) {{
block 0 (v0: i64) {{
  vn = i64.const {SPIN}
  br 1(vn)
}}
block 1 (vc: i64) {{
  vone = i64.const 1
  vc1 = i64.sub vc vone
  vdone = i64.eqz vc1
  br_if vdone 2() 1(vc1)
}}
block 2 () {{
  v = i64.const {value}
  return v
  }}
}}"
    )
}

/// An `Instantiator`, a `Budget` with room for every window and fiber, and the child: func 1's child
/// image (#2219). The child images of funcs 3–6, which the child spawns, are named `"g3"`–`"g6"`
/// for the root to grant on. `lanes` is the run's lane cap (`-1`: unbounded).
fn setup(m: &Module, lanes: i64) -> (Host, Vec<Value>) {
    let mut h = Host::new();
    h.set_lane_cap(lanes);
    let inst = h.grant_instantiator();
    let budget = h.grant_budget(-1, -1, -1);
    let child = h.grant_module(&temen_ir::child_image_at(m, 1).expect("child image"));
    for f in 3..=6 {
        let g = h.grant_module(&temen_ir::child_image_at(m, f).expect("child image"));
        h.register_cap_name(&format!("g{f}"), g);
    }
    (h, [inst, budget, child].map(Value::I32).to_vec())
}

/// `run` on its own thread with a deadline, so an engine that never finishes fails the test.
fn bounded<T: Send + 'static>(what: String, run: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(run());
    });
    rx.recv_timeout(std::time::Duration::from_secs(120))
        .unwrap_or_else(|e| panic!("{what}: no result ({e})"))
}

/// Run the program with `fiber` as its joining fiber's body on `drivers` and on Cranelift, under lane
/// cap `lanes` — that fiber resumed by the child's root, then by another fiber — and assert each
/// returns `want`. A failure names every engine that disagreed.
fn on_every_engine(what: &str, fiber: &str, drivers: &[Driver], lanes: i64, want: i64) {
    let ran = Ran {
        result: Ok(vec![Value::I64(want)]),
        stdout: Vec::new(),
        stderr: Vec::new(),
    };
    let mut wrong = Vec::new();
    for outer in [false, true] {
        let m = program(fiber, outer);
        let at = if outer {
            "under a fiber"
        } else {
            "under the root"
        };
        for &d in drivers {
            let m = m.clone();
            let got = bounded(format!("{what}, {at}: {d:?}"), move || {
                drivers::run_on(d, &m, &|| setup(&m, lanes))
            });
            match got {
                Some(got) if got == ran => {}
                Some(got) => wrong.push(format!("  {at}, {d:?}: {got:?}")),
                None => wrong.push(format!("  {at}, {d:?}: declined the module")),
            }
        }
        let got = bounded(format!("{what}, {at}: Cranelift"), move || {
            let (mut host, args) = setup(&m, lanes);
            let args: Vec<i64> = args
                .iter()
                .map(|v| match v {
                    Value::I32(h) => i64::from(*h),
                    other => panic!("a handle, not {other:?}"),
                })
                .collect();
            let init = MemLayout::image(vec![0u8; 128 << 10]);
            temen_run::jit_cap_run(&m, 0, &args, &init, 0, 0, &mut host, None).map(|(o, _)| o)
        });
        match got {
            Ok(JitOutcome::Returned(v)) if v == [want] => {}
            Err(JitError::Unsupported(_)) => {} // a target without the child executor
            other => wrong.push(format!("  {at}, Cranelift: {other:?}")),
        }
    }
    assert!(
        wrong.is_empty(),
        "{what}: want {want} on every engine, but\n{}",
        wrong.join("\n")
    );
}

/// The fiber joins a child that is still running, then another while it runs:
/// `(value * 1000 + status) * 1000 + value`, the second join's status beside both values (#2066).
/// Under a run lane cap of 1 a grandchild cannot start until its parent's join gives the one lane
/// back, so each join finds its child running, on every engine: the park must block the child's
/// whole vCPU, its root seeing no `FIBER_PARKED`, and hold no lane while it waits.
#[test]
fn a_childs_fiber_joins_running_children_on_every_engine() {
    let fiber = format!(
        "{}  vg1 = call.cap 6 17 (i64) -> (i32) vi (vrp)
  vj1_st, vj1 = call.cap 6 1 (i32) -> (i64, i64) vi (vg1)
{}  vg2 = call.cap 6 17 (i64) -> (i32) vi (vrp)
  vj2_st, vj2 = call.cap 6 1 (i32) -> (i64, i64) vi (vg2)
  vk = i64.const 1000
  va = i64.mul vj1 vk
  vb2 = i64.add va vj2_st
  vc = i64.mul vb2 vk
  vd = i64.add vc vj2
  return vd
  }}",
        pick(3),
        pick(4)
    );
    on_every_engine(
        "a fiber joins a running child, then another",
        &fiber,
        &drivers::ALL,
        1,
        (42 * 1000) * 1000 + 7,
    );
}

/// The fiber spawns func `func`'s child image, `poll`s it until it has ended — `suspend`ing to its
/// resumer between probes — then runs `tail` with the child's handle `g3`, the `Instantiator` `i3` and
/// the terminal poll status `p3` in scope. The lanes are unbounded: a busy poll keeps its lane on every
/// engine.
fn poll_then(func: u32, tail: &str) -> String {
    let module = pick(func);
    format!(
        "{module}  vg = call.cap 6 17 (i64) -> (i32) vi (vrp)
  br 1(vi, vg)
}}
block 1 (i1: i32, g1: i32) {{
  vpoll = call.cap 6 9 (i32) -> (i32) i1 (g1)
  vzero = i32.const 0
  vrunning = i32.eq vpoll vzero
  br_if vrunning 2(i1, g1) 3(i1, g1, vpoll)
}}
block 2 (i2: i32, g2: i32) {{
  vy = i64.const 0
  vx = suspend vy
  br 1(i2, g2)
}}
block 3 (i3: i32, g3: i32, p3: i32) {{
  vp64 = i64.extend_i32_u p3
  vk = i64.const 1000
{tail}
  }}"
    )
}

/// The fiber joins a child whose outcome is already in (`poll` reports it returned):
/// `(poll * 1000 + status) * 1000 + value`.
#[test]
fn a_childs_fiber_joins_a_child_that_returned_on_every_engine() {
    let fiber = poll_then(
        5,
        "  vj_st, vj = call.cap 6 1 (i32) -> (i64, i64) i3 (g3)
  va = i64.mul vp64 vk
  vb2 = i64.add va vj_st
  vc = i64.mul vb2 vk
  vd = i64.add vc vj
  return vd",
    );
    on_every_engine(
        "a fiber joins a child that returned",
        &fiber,
        &drivers::ALL,
        -1,
        (1000 * 1000) + 5,
    );
}

/// The fiber joins a child whose outcome is already in, a trap: the join answers the trap's code
/// and reaps the child, and the fiber runs on (#2066). `poll * 1000 + status`.
#[test]
fn a_childs_fiber_joins_a_child_that_trapped_on_every_engine() {
    let fiber = poll_then(
        6,
        "  vj_st, vj = call.cap 6 1 (i32) -> (i64, i64) i3 (g3)
  va = i64.mul vp64 vk
  vd = i64.add va vj_st
  return vd",
    );
    on_every_engine(
        "a fiber joins a child that trapped",
        &fiber,
        &drivers::ALL,
        -1,
        2 * 1000 + trap_code::UNREACHABLE,
    );
}
