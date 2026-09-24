//! #1720 — **a run as data**: the plan → root-guest generator (`temen_browser::plan`, decision C of
//! #1717). A one-node plan is the nim phase driver byte for byte; a many-node plan spawns and joins
//! every node, each holding exactly the capabilities its grants name; a plan the root cannot honour is
//! refused by the generator. The guests are written inline, so nothing is staged.

use std::sync::{Arc, Mutex};
use temen_browser::plan::{run, Node, Plan};
use temen_interp::{ForkedProc, Host, HostProc, HostProcFork, Value};
use temen_ir::Module;

/// A one-node plan is the nim phase driver, byte for byte: the text `nimc`'s phases ran on before
/// the plan existed (it replaced `detached_parent_src`, compared equal on every phase shape).
#[test]
fn a_one_node_plan_is_the_phase_driver() {
    let src = Plan::single(20, &["a", "b"], &["fs", "stdout"])
        .root_src()
        .unwrap();
    assert_eq!(
        src,
        r#"memory 16
data 18432 "fs"
data 18448 "stdout"
data 20480 "\x02\x00\x00\x00\x00\x00\x00\x00\x61\x00\x62\x00"
func (i32, i32, i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32, v3: i32, v4: i32) {
  xr17408 = i64.const 8589953024
  or17408 = i64.const 17408
  i64.store or17408 xr17408
  h17408 = i64.extend_i32_u v3
  oh17408 = i64.const 17416
  i64.store oh17408 h17408
  xr17424 = i64.const 25769822224
  or17424 = i64.const 17424
  i64.store or17424 xr17424
  h17424 = i64.extend_i32_u v4
  oh17424 = i64.const 17432
  i64.store oh17424 h17424
  vmh = i64.extend_i32_u v1
  vmin = i64.extend_i32_u v2
  vgptr = i64.const 17408
  vgn = i64.const 2
  ventry = i64.const 0
  vlog = i64.const 20
  vq = i64.const 0
  vap = i64.const 20480
  val = i64.const 12
  vh = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vmin, vmh, vgptr, vgn, ventry, vlog, vq, vap, val)
  vr = call.cap 6 1 (i32) -> (i64) v0 (vh)
  return vr
  }
}
"#
    );
}

/// A child-entry node that reports `id` through the `tally` capability if it was granted one, and
/// returns `id` — or `0` if it could not resolve `tally`.
fn tally_node(id: i64) -> Module {
    let src = format!(
        r#"memory 16
data 16384 "tally"
func (i64) -> (i64) {{
block 0 (va: i64) {{
  vp = i64.const 16384
  vl = i64.const 5
  vh = self.resolve vp vl
  vz = i32.const 0
  vmiss = i32.lt_s vh vz
  br_if vmiss 1() 2(vh)
}}
block 1 () {{
  vzero = i64.const 0
  return vzero
}}
block 2 (vh2: i32) {{
  vid = i64.const {id}
  vr = call.cap 13 0 (i64) -> (i64) vh2 (vid)
  return vid
  }}
}}
"#
    );
    let m = temen_text::parse_module(&src).expect("parse the tally node");
    temen_verify::verify_module(&m).expect("verify the tally node");
    m
}

/// Run a two-node plan whose root holds one capability, `tally` (a log both nodes would append
/// to), granting it to the nodes that list it. Returns what the root returned and the log.
fn run_two(grants_a: &[&str], grants_b: &[&str]) -> (i64, Vec<i64>) {
    let node = |grants: &[&str]| Node {
        window_log2: 16,
        argv: vec![],
        env: vec![],
        grants: grants.iter().map(|g| g.to_string()).collect(),
    };
    let plan = Plan {
        caps: vec!["tally".into()],
        nodes: vec![node(grants_a), node(grants_b)],
    };
    let log: Arc<Mutex<Vec<i64>>> = Arc::default();
    let mint = {
        let log = Arc::clone(&log);
        move || -> HostProc {
            let log = Arc::clone(&log);
            Box::new(move |_op, args: &[i64], _mem, _| {
                log.lock().unwrap().push(args[0]);
                Ok(vec![0])
            })
        }
    };
    let mut host = Host::new();
    let fork: HostProcFork = {
        let mint = mint.clone();
        Arc::new(move |_pid| ForkedProc::shared(mint()))
    };
    let tally = host.grant_host_proc_forkable(mint(), fork);
    let (a, b) = (tally_node(1), tally_node(2));
    let out = run(&plan, &[&a, &b], host, &[tally]).expect("the plan runs");
    let got = match out.first() {
        Some(Value::I64(v)) => *v,
        other => panic!("the root returns one i64, got {other:?}"),
    };
    let log = log.lock().unwrap().clone();
    (got, log)
}

/// Every node is spawned and joined, each with exactly the capabilities its grants name, and the
/// root returns the last node's result.
#[test]
fn each_node_holds_only_its_grants_and_the_root_returns_the_last() {
    assert_eq!(
        run_two(&["tally"], &["tally"]),
        (2, vec![1, 2]),
        "both granted"
    );
    assert_eq!(
        run_two(&["tally"], &[]),
        (0, vec![1]),
        "the second node was granted nothing, so it cannot reach `tally`"
    );
    assert_eq!(run_two(&[], &["tally"]), (2, vec![2]));
}

/// A plan the root cannot honour is refused by the generator, naming why — never generated into
/// a root that would grant something else.
#[test]
fn a_plan_the_root_cannot_honour_is_refused() {
    let node = |grants: &[&str]| Node {
        window_log2: 16,
        argv: vec![],
        env: vec![],
        grants: grants.iter().map(|g| g.to_string()).collect(),
    };
    let plan = |caps: &[&str], nodes: Vec<Node>| Plan {
        caps: caps.iter().map(|c| c.to_string()).collect(),
        nodes,
    };
    let err = |p: Plan| p.root_src().unwrap_err();
    assert!(err(plan(&["fs"], vec![])).contains("at least one node"));
    assert!(err(plan(&["fs"], vec![node(&["stdout"])])).contains("\"stdout\""));
    assert!(err(plan(&["a\"b"], vec![node(&[])])).contains("ASCII"));
    assert!(err(plan(&["seventeen-chars-x"], vec![node(&[])])).contains("ASCII"));
    let wide: Vec<String> = (0..129).map(|i| format!("c{i}")).collect();
    let wide: Vec<&str> = wide.iter().map(String::as_str).collect();
    assert!(err(plan(&wide, vec![node(&[])])).contains("do not fit"));
}

/// A node the spawn refuses (here: an entry of no admitted shape — three params — so op 15 answers
/// `-EINVAL`) ends the run with a trap when the root joins the refused handle. It used to panic the
/// driver (`drive_op13` indexed its child list with the negative handle).
#[test]
fn a_refused_node_ends_the_run_with_a_trap_not_a_host_panic() {
    let m = temen_text::parse_module(
        "memory 16\nfunc (i64, i64, i64) -> (i64) {\nblock 0 (va: i64, vb: i64, vc: i64) {\n  return va\n  }\n}\n",
    )
    .expect("parse");
    let plan = Plan::single(16, &[], &[]);
    assert!(run(&plan, &[&m], Host::new(), &[]).is_err());
}

/// #1720 — a card module nests **as built**: `hello_c.temen` (a powerbox `_start: () -> i32`, the
/// on-ramp's own build, no `--child-entry` twin) runs as a one-node plan with `stdout`/`exit`
/// re-granted by name, and prints exactly what its direct on-ramp run prints.
#[test]
fn a_card_module_runs_nested_as_built() {
    let m = temen_encode::decode_module(include_bytes!("../web/assets/hello_c.temen"))
        .expect("decode hello_c.temen");
    let direct = temen_browser::onramp_exec(&m, b"");
    assert_eq!(direct.status, temen_browser::STATUS_OK);

    let mut host = Host::new();
    let sink = host.shared_stdout(); // a re-granted stdout writes into its granter's shared sink
    let stdout = host.grant_stream(temen_interp::StreamRole::Out);
    let exit = host.grant_exit();
    let window = m.memory.expect("a window").size_log2;
    let plan = Plan::single(window, &[], &["stdout", "exit"]);
    let out = run(&plan, &[&m], host, &[stdout, exit]).expect("the nested run completes");

    assert_eq!(
        *sink.lock().unwrap(),
        direct.stdout,
        "same stdout, nested or not"
    );
    assert_eq!(
        out,
        vec![Value::I64(direct.value)],
        "same result, widened to the join's i64"
    );
}
