//! temen#2168 — a run compiles each detached child **once** on the JIT. Every spawn of a child
//! instantiates the run's one compile of it (`temen_jit::ChildCodeCache`), each over its own
//! powerbox, window, budget cell and nursery, so a child that spawns spawns through its own.
//!
//! The root spawns the [`nester`] four times, each two deep: three times from an unmetered budget and
//! once from a metered one, and every level spawns and joins the next itself. That is twelve spawns
//! of one module, and every engine returns the same sum. On the JIT a run compiles the nester twice,
//! once for each shape: a metered child's code charges its fuel cell, and an unmetered one's does
//! not. Runs handed one `JitCodeCache` share those compiles, so a later run compiles none — the
//! root's included, though it spawns: each instance of its code stands up a nursery of its own
//! (temen#2389). A cache that keeps its code in a directory (temen#2452) carries that to a later
//! process.
//!
//! The process-wide compile counters move only for one test's runs: the other runs its in
//! processes of their own.

#[path = "support/detached.rs"]
mod detached;

use detached::{module, nester, rec, within_a_minute};
use std::path::PathBuf;
use temen_interp::{Host, Value};
use temen_ir::SpawnRec;
use temen_run::{instantiate, Backend, JitCodeCache, Limits, Outcome, RunConfig};

/// What the nester returns from depth 2 (`10 * (10 * 7 + 1) + 2`), times its four spawns.
const SUM: i64 = 4 * 712;

/// The root: spawn the nester at depth 2 four times from the record at 17408, the last from the
/// `"metered"` budget and the others from `"pool"`, granting it on as `"nester"`; join each and return
/// the sum of the results.
fn root() -> String {
    let child = SpawnRec {
        size_log2: 16,
        grants_ptr: 20064,
        grants_n: 1,
        args: (18432, 8),
        ..SpawnRec::default()
    };
    format!(
        r#"memory 17
data 20000 "spawner"
data 20016 "nester"
data 20032 "pool"
data 20048 "metered"
func () -> (i64) {{
block 0 () {{
  vs = i64.const 0
  vi = i64.const 0
  br 1(vs, vi)
}}
block 1 (vs1: i64, vi1: i64) {{
  vn = i64.const 4
  vdone = i64.eq vi1 vn
  br_if vdone 3(vs1) 2(vs1, vi1)
}}
block 2 (vs2: i64, vi2: i64) {{
  vthree = i64.const 3
  vlast = i64.eq vi2 vthree
  br_if vlast 5(vs2, vi2) 4(vs2, vi2)
}}
block 3 (vr: i64) {{
  return vr
}}
block 4 (vs4: i64, vi4: i64) {{
  vpp = i64.const 20032
  vpl = i64.const 4
  vpool = self.resolve vpp vpl
  br 6(vs4, vi4, vpool)
}}
block 5 (vs5: i64, vi5: i64) {{
  vmp = i64.const 20048
  vml = i64.const 7
  vmet = self.resolve vmp vml
  br 6(vs5, vi5, vmet)
}}
block 6 (vs6: i64, vi6: i64, vbud: i32) {{
  vip = i64.const 20000
  vil = i64.const 7
  vinst = self.resolve vip vil
  vnp = i64.const 20016
  vnl = i64.const 6
  vnest = self.resolve vnp vnl
  vdp = i64.const 18432
  vd = i64.const 2
  i64.store vdp vd
{grant}  vma = i64.const {module_at}
  i32.store vma vnest
  vba = i64.const {budget_at}
  i32.store vba vbud
  vrp = i64.const 17408
  vh = call.cap 6 17 (i64) -> (i32) vinst (vrp)
  vj_st, vj = call.cap 6 1 (i32) -> (i64, i64) vinst (vh)
  vs7 = i64.add vs6 vj
  vone = i64.const 1
  vi7 = i64.add vi6 vone
  br 1(vs7, vi7)
  }}
}}
{rec}"#,
        grant = rec::grant("g", 20064, 20016, 6, "vnest"),
        module_at = 17408 + SpawnRec::MODULE_AT,
        budget_at = 17408 + SpawnRec::BUDGET_AT,
        rec = rec::segment(17408, &child),
    )
}

/// The root's run on `backend` under `cfg`: its Instantiator, the nester, an unmetered budget and a
/// metered one, each by name.
fn run(backend: Backend, cfg: RunConfig) -> Outcome {
    within_a_minute(&format!("{backend:?}"), move || {
        let nest = module(&nester(false));
        let mut setup = |h: &mut Host| {
            let inst = h.grant_instantiator();
            h.register_cap_name("spawner", inst);
            let m = h.grant_module(&nest);
            h.register_cap_name("nester", m);
            let pool = h.grant_budget(-1, 1 << 24, -1);
            h.register_cap_name("pool", pool);
            let metered = h.grant_budget(1 << 40, 1 << 24, -1);
            h.register_cap_name("metered", metered);
        };
        instantiate(module(&root()))
            .expect("instantiate")
            .run_with_caps_and_host(backend, &cfg, &[], Some(&mut setup))
            .unwrap_or_else(|e| panic!("{backend:?}: {e}"))
            .outcome
    })
}

/// How many children, and how many roots, a JIT run under `cfg` compiles.
fn compiles(cfg: &RunConfig) -> (u64, u64) {
    let (children, roots) = (temen_jit::child_compiles(), temen_jit::module_compiles());
    assert_eq!(
        run(Backend::Jit, cfg.clone()),
        Outcome::Returned(vec![Value::I64(SUM)]),
        "Jit"
    );
    (
        temen_jit::child_compiles() - children,
        temen_jit::module_compiles() - roots,
    )
}

/// An unmetered root, so the budget a spawn names decides whether the child is metered.
fn unmetered() -> RunConfig {
    RunConfig {
        limits: Limits {
            fuel: Some(u64::MAX),
            ..Limits::default()
        },
        ..RunConfig::default()
    }
}

#[test]
fn a_run_compiles_each_detached_child_once_for_all_its_spawns() {
    let unmetered = unmetered();
    for backend in [Backend::TreeWalk, Backend::Bytecode] {
        assert_eq!(
            run(backend, unmetered.clone()),
            Outcome::Returned(vec![Value::I64(SUM)]),
            "{backend:?}"
        );
    }
    assert_eq!(
        compiles(&unmetered),
        (2, 1),
        "twelve spawns of one child: its metered code and its unmetered code; and the root"
    );

    let cache = JitCodeCache::default();
    let cached = RunConfig {
        jit_code: Some(cache.clone()),
        ..unmetered
    };
    assert_eq!(
        compiles(&cached),
        (2, 1),
        "the first run compiles all three"
    );
    assert_eq!(
        compiles(&cached),
        (0, 0),
        "a later run compiles none: the root that spawns them is shared too (temen#2389)"
    );
    assert_eq!(cache.children_compiled(), 2);
    assert_eq!(cache.compiled(), 1);
}

/// Where [`run_kept_here`] keeps its code, and writes what it compiled.
const KEPT_DIR: &str = "TEMEN_RUN_KEPT_DIR";
const KEPT_OUT: &str = "TEMEN_RUN_KEPT_OUT";

/// The half of [`a_later_process_compiles_none_of_what_an_earlier_one_kept`] that runs in a process
/// of its own: a JIT run handed a cache kept in `$KEPT_DIR`, which writes how many children and
/// roots it compiled.
#[test]
#[ignore = "run in a process of its own by a_later_process_compiles_none_of_what_an_earlier_one_kept"]
fn run_kept_here() {
    let (Some(dir), Some(out)) = (std::env::var_os(KEPT_DIR), std::env::var_os(KEPT_OUT)) else {
        return;
    };
    let cache = JitCodeCache::kept_in(PathBuf::from(dir)).expect("open the code directory");
    let kept = RunConfig {
        jit_code: Some(cache),
        ..unmetered()
    };
    let (children, roots) = compiles(&kept);
    std::fs::write(out, format!("{children} {roots}")).expect("write what this process compiled");
}

/// [`run_kept_here`] in a process of its own over `dir`: the children and the roots it compiled.
#[cfg(unix)]
fn in_a_process(dir: &std::path::Path) -> (u64, u64) {
    let out = dir.with_extension("out");
    let run = std::process::Command::new(std::env::current_exe().expect("this test binary"))
        .args(["--exact", "run_kept_here", "--ignored"])
        .env(KEPT_DIR, dir)
        .env(KEPT_OUT, &out)
        .output()
        .expect("run this test binary");
    assert!(
        run.status.success(),
        "{}\n{}{}",
        run.status,
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    let saw = std::fs::read_to_string(&out).expect("what the process compiled");
    let _ = std::fs::remove_file(&out);
    let (children, roots) = saw.split_once(' ').expect("two numbers");
    (children.parse().unwrap(), roots.parse().unwrap())
}

#[cfg(unix)]
#[test]
fn a_later_process_compiles_none_of_what_an_earlier_one_kept() {
    let dir = std::env::temp_dir().join(format!("temen-run-kept-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(
        in_a_process(&dir),
        (2, 1),
        "the first process compiles the nester's two shapes and the root"
    );
    assert_eq!(
        in_a_process(&dir),
        (0, 0),
        "a later process compiles none of them: it defines the code the first kept"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
