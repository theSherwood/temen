//! **The full nimony compiler, running on the SVM: Nim source → a module that RUNS** (NIM.md §3c/§3e,
//! the "compile Nim in the browser" capstone). This closes the chain the earlier slices left in
//! pieces: `nifler` (parse), `nimsem` (sema, driving nifler via `exec`), and `hexer` (lower) each run
//! as **real SVM guests** over a shared in-window `fs`, producing the program's + `system` module's
//! Leng (`.x.nif`); `svm_leng::link_whole_with_runtime` then links that to one verified, import-free
//! module, and it **executes on both engines to the correct value** (§9 interp/JIT parity). Every
//! nimony phase runs sandboxed; only the final IR link is the embedder's Rust — the same host-drives-
//! the-phases shape as the browser cards and `nim_backend_chain`, now carried through to a *run*.
//!
//! ```text
//! cargo run -q --release -p svm-run --example nim_e2e_chain -- \
//!     <nifler.svmb> <nimsem.svmb> <hexer.svmb> <libdir> <nimcache-dir> <main-stem> \
//!     <export> <expected> [arg...]
//! ```
//!
//! `<nimcache-dir>` is a native-bootstrapped `nimcache` (from one `nimony c`, giving the `.p.nif`
//! parse outputs, the stem bookkeeping, and `<main-stem>.build.nif` — nimony's own dependency-ordered
//! plan). The driver reads that plan, so it handles **any number of modules**: a program that
//! `import`s pulls in more compilation units, each its own nifler→nimsem→hexer unit, dependency-
//! ordered. It re-runs **sema and lowering on the SVM** for every module (the parse is checked too —
//! nifler.svmb re-parses the main and its bytes must match), then links and runs in one of two modes:
//!
//! - **compute** — `<export> <expected-i64> [i64 args…]`: link with the W3 compute shim and call the
//!   named exported proc, asserting the returned `i64` on the tree-walker and the JIT (§9 parity).
//! - **I/O** — `<io> <expected-stdout>`: a program that `write`s. Link through the nim→powerbox bridge
//!   (`svm_leng::link_nim_powerbox`), run `_start` under the powerbox (host grants stdout), and assert
//!   the captured **stdout** (`\n`/`\t` decoded) — a Nim program compiled by the SVM printing for real.

use std::path::Path;
use std::sync::Arc;

use svm_interp::Value;
use svm_ir::{LinkUnit, Module};
use svm_run::exec::{domain_exec_with_fs, DomainProgram};
use svm_run::{instantiate, Backend, HostCap, Limits, Outcome, RunConfig};

/// The W3 runtime shim's function indices, keyed by the bottom-edge C symbol each import lowers to
/// (longest-prefix wins so `atomicCompareExchangeN` isn't shadowed by a shorter atomic). Mirrors
/// `svm-leng/tests/nim_e2e.rs` — the same shim the native end-to-end links against.
const SHIM_BINDINGS: &[(&str, u32)] = &[
    ("cExitSys", 0),
    ("cGetpid", 1),
    ("cKill", 2),
    ("c_memcpy", 3),
    ("c_memcmp", 4),
    ("c_memset", 5),
    ("mmap", 6),
    ("atomicLoadN", 7),
    ("atomicStoreN", 8),
    ("atomicCompareExchangeN", 9),
    ("atomicExchangeN", 10),
    ("atomicAddFetch", 11),
    ("atomicSubFetch", 12),
    ("bswap64", 13),
    ("ctz64", 14),
    ("clz64", 15),
    ("cWriteErr", 16),
    ("dlopen", 17),
    ("dlclose", 18),
    ("dlsym", 19),
];

fn shim_index(name: &str) -> Option<u32> {
    SHIM_BINDINGS
        .iter()
        .filter(|(p, _)| name.starts_with(p))
        .max_by_key(|(p, _)| p.len())
        .map(|(_, i)| *i)
}

/// Recursively collect `(relative-key, bytes)` under `dir`, prefixing each key with `prefix`.
fn collect(dir: &Path, prefix: &str, out: &mut Vec<(String, Vec<u8>)>) {
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap_or_else(|e| panic!("read_dir {d:?}: {e}")) {
            let e = e.expect("entry");
            let p = e.path();
            if e.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                stack.push(p);
            } else {
                let rel = p
                    .strip_prefix(dir)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/");
                out.push((format!("{prefix}{rel}"), std::fs::read(&p).expect("read")));
            }
        }
    }
}

/// One shared in-window `fs` store across every phase: seed it once, hand each guest run its own grant
/// over the same store (`Arc<factory>`), and read produced files back through `handle.seed()`.
struct Memfs {
    factory: Arc<dyn Fn() -> svm_interp::HostProc + Send + Sync>,
    handle: svm_run::fs::MemFsHandle,
}

impl Memfs {
    fn new(files: Vec<(String, Vec<u8>)>) -> Self {
        let (factory, handle) = svm_run::fs::mem_fs_shared_factory(files, vec!["nimcache".into()]);
        Memfs {
            factory: Arc::new(factory) as Arc<dyn Fn() -> svm_interp::HostProc + Send + Sync>,
            handle,
        }
    }
    fn grant(&self) -> HostCap {
        let f = self.factory.clone();
        HostCap::host_proc(0, move || (f)())
    }
    fn read(&self, key: &str) -> Option<Vec<u8>> {
        let (files, _) = self.handle.seed();
        files.into_iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
}

fn cfg(args: &[&str], stdin: Vec<u8>) -> RunConfig {
    RunConfig {
        limits: Limits {
            fuel: None,
            deadline: None,
            max_fibers: 0,
            max_vcpus: 0,
        },
        stdin,
        memory_size_log2: None,
        args: args.iter().map(|s| s.as_bytes().to_vec()).collect(),
        env: vec![],
        ..RunConfig::default()
    }
}

/// One nimsem invocation from nimony's build plan (`<main>.build.nif`): the module it produces, the
/// exact args (`--isSystem` / `--isMain` / bare `<stem>.p.nif`), and the modules it depends on.
struct NimsemStep {
    stem: String,
    args: Vec<String>,
    deps: Vec<String>,
}

/// Return the balanced `(head …)` s-expr slices in `s` (each includes the parens). A tiny reader —
/// enough to walk nimony's pretty-printed NIF build plan without a full parser.
fn sexpr_blocks(s: &str, head: &str) -> Vec<String> {
    let needle = format!("({head}");
    let b = s.as_bytes();
    let mut out = vec![];
    let mut i = 0;
    while let Some(rel) = s[i..].find(&needle) {
        let start = i + rel;
        let (mut j, mut depth, mut in_str) = (start, 0i32, false);
        while j < b.len() {
            match b[j] {
                b'\\' if in_str => j += 1,
                b'"' => in_str = !in_str,
                b'(' if !in_str => depth += 1,
                b')' if !in_str => {
                    depth -= 1;
                    if depth == 0 {
                        j += 1;
                        break;
                    }
                }
                _ => {}
            }
            j += 1;
        }
        out.push(s[start..j].to_string());
        i = j;
    }
    out
}

/// The contents of every `"…"` string literal in `s`, with NIF `\HH` hex + `\\`/`\"` escapes decoded.
fn quoted(s: &str) -> Vec<String> {
    let b = s.as_bytes();
    let (mut out, mut i) = (vec![], 0);
    while i < b.len() {
        if b[i] == b'"' {
            let mut cur = String::new();
            i += 1;
            while i < b.len() && b[i] != b'"' {
                if b[i] == b'\\' && i + 2 < b.len() {
                    if let Ok(byte) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                        cur.push(byte as char);
                        i += 3;
                        continue;
                    }
                    cur.push(b[i + 1] as char);
                    i += 2;
                    continue;
                }
                cur.push(b[i] as char);
                i += 1;
            }
            out.push(cur);
        }
        i += 1;
    }
    out
}

/// Parse the `(do nimsem …)` steps out of `<main>.build.nif` — nimony's own dependency-ordered plan.
/// Each step's `(output …).s.nif` names the module; its `(input …).s.idx.nif`s are the modules that
/// must be semchecked first (the dependency edges, already in stem space — no path→stem resolution).
fn parse_nimsem_steps(build_nif: &str) -> Vec<NimsemStep> {
    let stem_of = |path: &str, suffix: &str| -> Option<String> {
        path.rsplit('/')
            .next()
            .and_then(|f| f.strip_suffix(suffix))
            .map(str::to_string)
    };
    let mut steps = vec![];
    for block in sexpr_blocks(build_nif, "do nimsem") {
        let args = sexpr_blocks(&block, "args")
            .first()
            .map(|a| quoted(a))
            .unwrap_or_default();
        let mut stem = None;
        let mut deps = vec![];
        for out in sexpr_blocks(&block, "output") {
            if let Some(s) = quoted(&out).first().and_then(|p| stem_of(p, ".s.nif")) {
                stem = Some(s);
            }
        }
        for inp in sexpr_blocks(&block, "input") {
            if let Some(d) = quoted(&inp).first().and_then(|p| stem_of(p, ".s.idx.nif")) {
                deps.push(d);
            }
        }
        if let Some(stem) = stem {
            steps.push(NimsemStep { stem, args, deps });
        }
    }
    steps
}

/// Dependency-order the steps (each module after every module it depends on) via DFS postorder over
/// the in-set edges. nimony's plan is already a DAG; this pins a valid processing order for the SVM
/// run without relying on the plan's textual order.
fn toposort(steps: &[NimsemStep]) -> Vec<usize> {
    let idx: std::collections::HashMap<&str, usize> = steps
        .iter()
        .enumerate()
        .map(|(i, s)| (s.stem.as_str(), i))
        .collect();
    let mut order = vec![];
    let mut seen = vec![0u8; steps.len()]; // 0 unvisited, 1 in-progress, 2 done
    fn visit(
        i: usize,
        steps: &[NimsemStep],
        idx: &std::collections::HashMap<&str, usize>,
        seen: &mut [u8],
        order: &mut Vec<usize>,
    ) {
        if seen[i] != 0 {
            return; // done, or a cycle edge (nimony's graph is acyclic; ignore back-edges defensively)
        }
        seen[i] = 1;
        for d in &steps[i].deps {
            if let Some(&j) = idx.get(d.as_str()) {
                visit(j, steps, idx, seen, order);
            }
        }
        seen[i] = 2;
        order.push(i);
    }
    for i in 0..steps.len() {
        visit(i, steps, &idx, &mut seen, &mut order);
    }
    order
}

fn main() {
    let mut a = std::env::args().skip(1);
    let mut next = |what: &str| a.next().unwrap_or_else(|| panic!("missing <{what}>"));
    let nifler_p = next("nifler.svmb");
    let nimsem_p = next("nimsem.svmb");
    let hexer_p = next("hexer.svmb");
    let libdir = next("libdir");
    let nimcache = next("nimcache-dir");
    let main_stem = next("main-stem");
    // Two run modes. Compute: `<export> <expected-i64> [i64 args…]` — link with the compute shim and
    // call an exported proc, asserting the returned i64. I/O: `<io> <expected-stdout>` — link through
    // the nim→powerbox bridge and run `_start` under the powerbox, asserting the captured **stdout**
    // (a program that `echo`s / `write`s). `\n`/`\t` in the expected stdout are decoded.
    let export = next("export");
    let expected_raw = next("expected");
    let io_mode = export == "<io>";
    let call_args: Vec<i64> = a.map(|s| s.parse().expect("arg is i64")).collect();

    let read = |p: &str| std::fs::read(p).unwrap_or_else(|e| panic!("read {p}: {e}"));
    let nifler = read(&nifler_p);
    let nimsem = read(&nimsem_p);
    let hexer = read(&hexer_p);
    let nc = Path::new(&nimcache);
    let prog_pnif = read(&nc.join(format!("{main_stem}.p.nif")).to_string_lossy());
    let proj = nc.parent().expect("nimcache has a parent dir");

    // The module compilation plan — nimony's own dependency graph (`<main>.build.nif`): every `.s.nif`
    // module, its nimsem args (`--isSystem`/`--isMain`/bare), and its dependency edges. A program that
    // `import`s pulls in more modules (each its own nifler→nimsem→hexer unit); the plan enumerates and
    // orders them, so this generalizes the two-module case to any dependency graph.
    let build_nif = String::from_utf8_lossy(&read(
        &nc.join(format!("{main_stem}.build.nif")).to_string_lossy(),
    ))
    .into_owned();
    let steps = parse_nimsem_steps(&build_nif);
    assert!(
        steps.iter().any(|s| s.stem == main_stem),
        "no nimsem step for the main module {main_stem} in the build plan"
    );
    let order = toposort(&steps);
    eprintln!(
        "plan: {} modules, dependency order [{}]",
        steps.len(),
        order
            .iter()
            .map(|&i| steps[i].stem.as_str())
            .collect::<Vec<_>>()
            .join(" → ")
    );

    // ---- shared memfs: stdlib sources (under lib/, plus a flattened std/ view) + every `.p.nif` -----
    let mut files = vec![];
    collect(Path::new(&libdir), "lib/", &mut files);
    let flat: Vec<(String, Vec<u8>)> = files
        .iter()
        .filter_map(|(k, v)| {
            k.strip_prefix("lib/std/")
                .map(|r| (format!("lib/{r}"), v.clone()))
        })
        .collect();
    files.extend(flat);
    // Seed the parsed nif of every module the plan mentions (bootstrapped by native nifler; nifler.svmb
    // re-parses the main below to prove the parse runs on the SVM too).
    for step in &steps {
        let p = nc.join(format!("{}.p.nif", step.stem));
        if let Ok(bytes) = std::fs::read(&p) {
            files.push((format!("nimcache/{}.p.nif", step.stem), bytes));
        }
    }
    // Seed every project `.nim` source at its portable name (`prog.nim`, `helper.nim`, …) so nimsem
    // can resolve a user `import ./helper` — it may re-parse the dependency's source (via the exec
    // nifler) when it doesn't already hold its `.s.idx.nif`.
    for e in std::fs::read_dir(proj).unwrap_or_else(|e| panic!("read_dir {proj:?}: {e}")) {
        let p = e.expect("entry").path();
        if p.extension().and_then(|x| x.to_str()) == Some("nim") {
            let name = p.file_name().unwrap().to_string_lossy().into_owned();
            files.push((name, std::fs::read(&p).expect("read .nim")));
        }
    }
    eprintln!("seeded {} files into the shared memfs", files.len());
    let fs = Memfs::new(files);

    // nifler.svmb registered under the names nimsem uses for argv[0] (deps.nim shells out to `nifler`).
    let nifler_inst = Arc::new(
        instantiate(svm_encode::decode_module(&nifler).expect("decode nifler.svmb"))
            .expect("instantiate nifler"),
    );
    let programs: Vec<DomainProgram> = ["nifler", "/bin/nifler"]
        .iter()
        .map(|n| DomainProgram {
            name: (*n).into(),
            instance: nifler_inst.clone(),
            limits: Limits::default(),
        })
        .collect();

    // ---- phase 1: nifler.svmb re-parses prog.nim on the SVM; its bytes must match the bootstrap ----
    {
        let run = instantiate(svm_encode::decode_module(&nifler).expect("decode nifler"))
            .expect("instantiate nifler")
            .run_with_caps(
                Backend::TreeWalk,
                &cfg(
                    &["nifler", "p", "/prog.nim", "/nimcache/reparse.p.nif"],
                    vec![],
                ),
                &[("fs", fs.grant())],
            )
            .expect("run nifler");
        assert!(
            matches!(run.outcome, Outcome::Exited(0) | Outcome::Returned(_)),
            "nifler: {:?}",
            run.outcome
        );
        let reparsed = fs
            .read("nimcache/reparse.p.nif")
            .expect("nifler wrote no reparse.p.nif");
        assert_eq!(
            reparsed, prog_pnif,
            "nifler.svmb re-parse must match the bootstrap .p.nif"
        );
        eprintln!(
            "phase 1  nifler.svmb: parsed prog.nim → {} B .p.nif (== bootstrap)",
            reparsed.len()
        );
    }

    // ---- phase 2: nimsem.svmb (sema), driving nifler.svmb via exec — each module in dependency order.
    // Every module runs with the exact args nimony's plan assigned it (`--isSystem` for the system
    // module, `--isMain` for the program, bare for imported modules); a module is semchecked only after
    // its dependencies, so the `.s.nif` each one reads back is already present.
    for &i in &order {
        let step = &steps[i];
        let mut argv = vec![
            "nimsem",
            "--define:nimNativeAlloc",
            "--define:nimNativeIo",
            "m",
        ];
        argv.extend(step.args.iter().map(|s| s.as_str()));
        let run = instantiate(svm_encode::decode_module(&nimsem).expect("decode nimsem"))
            .expect("instantiate nimsem")
            .run_with_caps(
                Backend::TreeWalk,
                &cfg(&argv, vec![]),
                &[
                    ("fs", fs.grant()),
                    ("exec", domain_exec_with_fs(programs.clone(), fs.grant())),
                ],
            )
            .expect("run nimsem");
        assert!(
            matches!(run.outcome, Outcome::Exited(0) | Outcome::Returned(_)),
            "nimsem {}: {:?}",
            step.stem,
            run.outcome
        );
        let snif = fs
            .read(&format!("nimcache/{}.s.nif", step.stem))
            .unwrap_or_else(|| panic!("nimsem wrote no {}.s.nif", step.stem));
        eprintln!(
            "phase 2  nimsem.svmb [{}]: {} → {} B .s.nif",
            step.args.first().map(String::as_str).unwrap_or("m"),
            step.stem,
            snif.len()
        );
    }

    // ---- phase 3: hexer.svmb (lower) — each semchecked module → Leng `.x.nif` ----------------------
    // The **main** module is lowered as native `nimony c` does — `hexer c --isMain --app:console
    // --outdir:nimcache/<main>` — so its `.x.nif` (written into the outdir) carries the C `main`/
    // `_start` app-entry glue the powerbox `_start` run needs. Other modules use a plain `hexer c`.
    let outdir = format!("nimcache/{main_stem}");
    let run_hexer = |stem: &str| -> Vec<u8> {
        let is_main = stem == main_stem;
        let s_nif = format!("nimcache/{stem}.s.nif");
        let outdir_arg = format!("--outdir:{outdir}");
        let argv: Vec<&str> = if is_main {
            vec![
                "hexer",
                "c",
                "--bits:64",
                "--cpu:le",
                "--flags:br",
                "--isMain",
                "--app:console",
                &outdir_arg,
                &s_nif,
            ]
        } else {
            vec!["hexer", "c", &s_nif]
        };
        let run = instantiate(svm_encode::decode_module(&hexer).expect("decode hexer"))
            .expect("instantiate hexer")
            .run_with_caps(
                Backend::TreeWalk,
                &cfg(&argv, vec![]),
                &[("fs", fs.grant())],
            )
            .expect("run hexer");
        assert!(
            matches!(run.outcome, Outcome::Exited(0) | Outcome::Returned(_)),
            "hexer {stem}: {:?}",
            run.outcome
        );
        // The main's `.x.nif` lands in the outdir; the others' next to their `.s.nif`.
        let key = if is_main {
            format!("{outdir}/{stem}.x.nif")
        } else {
            format!("nimcache/{stem}.x.nif")
        };
        let xnif = fs
            .read(&key)
            .unwrap_or_else(|| panic!("hexer wrote no {key}"));
        eprintln!("phase 3  hexer.svmb: {stem} → {} B Leng .x.nif", xnif.len());
        xnif
    };
    // Lower every module (dependency order); keep the produced Leng keyed by stem for linking.
    let leng: Vec<(String, String)> = order
        .iter()
        .map(|&i| {
            let stem = steps[i].stem.clone();
            let x = run_hexer(&stem);
            (stem, String::from_utf8_lossy(&x).into_owned())
        })
        .collect();

    // ---- phase 4 (host link + run): link the SVM-produced Leng, execute --------------------------
    // Order units **main first** (func 0 = the natural entry) and **system last**, with imported
    // modules between — the convention both linkers build on.
    let mut ordered: Vec<&(String, String)> = leng.iter().collect();
    ordered.sort_by_key(|(stem, _)| (*stem != main_stem, stem.starts_with("sysv")));
    let units: Vec<svm_leng::WholeModule> = ordered
        .iter()
        .map(|(stem, src)| svm_leng::WholeModule { stem, src })
        .collect();

    if io_mode {
        // I/O program: bridge nimony's bottom edge to the §3e powerbox (`sysWrite(fd,buf,len)` → the
        // STREAM `write` cap) via `link_nim_powerbox`, then run `_start` under the powerbox (host grants
        // stdout) and assert the captured **stdout** — the same bridge the `nim_hello` card ships.
        let m: Module = svm_leng::link_nim_powerbox(&units)
            .unwrap_or_else(|e| panic!("nim→powerbox link: {e}"));
        svm_verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
        assert!(
            svm_run::is_named_powerbox_entry(&m),
            "linked module is not a powerbox entry (paramless func-0 `_start`)"
        );
        eprintln!(
            "phase 4  linked (powerbox) → {} funcs, verified",
            m.funcs.len()
        );
        let want = expected_raw.replace("\\n", "\n").replace("\\t", "\t");
        let run = svm_run::run_powerbox(&m, &[]).unwrap_or_else(|e| panic!("run_powerbox: {e}"));
        let got = String::from_utf8_lossy(&run.stdout);
        assert_eq!(
            got.as_ref(),
            want,
            "SVM stdout must match the expected output (got {:?})",
            got
        );
        println!(
            "✅ FULL CHAIN ON SVM (I/O): nifler → nimsem → hexer (all sandboxed) → nim→powerbox link → \
             ran under the powerbox → stdout = {:?}. A Nim program compiled by the SVM prints for real.",
            got
        );
        return;
    }

    // Compute program: link with the W3 runtime shim (discover the system module's bottom-edge C
    // imports and bind them), then call an exported proc on both engines (§9 parity).
    let expected: i64 = expected_raw
        .parse()
        .expect("expected is i64 in compute mode");
    let sys_src = leng
        .iter()
        .find(|(s, _)| s.starts_with("sysv"))
        .map(|(s, src)| svm_leng::WholeModule { stem: s, src })
        .expect("no system module among the lowered units");
    let obj = svm_encode::decode_unit(
        &svm_leng::compile_whole_object(&sys_src)
            .unwrap_or_else(|e| panic!("compile system module: {e}")),
    )
    .expect("decode object");
    let exports: Vec<(String, u32)> = obj
        .imports
        .iter()
        .filter_map(|imp| shim_index(&imp.name).map(|i| (imp.name.clone(), i)))
        .collect();
    const SHIM: &str = include_str!("../../svm-leng/src/powerbox_compute_shim.svm.txt");
    let runtime = LinkUnit {
        module: svm_text::parse_module(SHIM).expect("runtime shim parses"),
        exports,
        ..Default::default()
    };
    let m: Module = svm_leng::link_whole_with_runtime(&units, vec![runtime])
        .unwrap_or_else(|e| panic!("link with runtime: {e}"));
    svm_verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    assert_eq!(m.imports.len(), 0, "every bottom-edge import is bound");
    eprintln!(
        "phase 4  linked → {} funcs, verified, import-free",
        m.funcs.len()
    );

    let idx = m
        .exports
        .iter()
        .find(|e| e.name.contains(&export))
        .unwrap_or_else(|| panic!("no export matching `{export}`"))
        .func;
    let seed = vec![0u8; 1 << 20];
    let ivals: Vec<Value> = call_args.iter().map(|&n| Value::I64(n)).collect();
    let mut fuel = 500_000_000u64;
    let (ir, _) = svm_interp::run_capture(&m, idx, &ivals, &mut fuel, &seed);
    let iword = match ir.expect("interp").as_slice() {
        [Value::I64(n)] => *n,
        o => panic!("interp result {o:?}"),
    };
    let (jout, _) = svm_jit::compile_and_run_capture(&m, idx, &call_args, &seed).expect("jit");
    let jword = match jout {
        svm_jit::JitOutcome::Returned(v) => v,
        o => panic!("jit: {o:?}"),
    };
    assert_eq!(vec![iword], jword, "§9 interp/JIT parity");
    assert_eq!(
        iword, expected,
        "{export}({call_args:?}) — SVM run must equal the expected result"
    );

    println!(
        "✅ FULL CHAIN ON SVM: nifler → nimsem(→nifler via exec) → hexer, all sandboxed guests → \
         linked → {export}({call_args:?}) = {iword} (interp == JIT == {expected}). The nimony \
         compiler runs on the SVM, and its output runs too."
    );
}
