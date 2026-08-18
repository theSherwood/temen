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
//!     <nifler.svmb> <nimsem.svmb> <hexer.svmb> <libdir> <nimcache-dir> <sys-stem> <prog-stem> \
//!     <export> <expected> [arg...]
//! ```
//!
//! `<nimcache-dir>` is a native-bootstrapped `nimcache` (from one `nimony c`, giving the `.p.nif`
//! parse outputs + the stem bookkeeping nifmake computes); the driver re-runs **sema and lowering on
//! the SVM** from there, so the parse output is checked (nifler.svmb re-parses `prog.nim` and its
//! bytes must match) and the two big phases actually execute in the sandbox. `<export>` is a proc the
//! linked module must expose; the driver calls it with `[arg...]` and asserts the result is
//! `<expected>` on the tree-walker and the JIT.

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

fn main() {
    let mut a = std::env::args().skip(1);
    let mut next = |what: &str| a.next().unwrap_or_else(|| panic!("missing <{what}>"));
    let nifler_p = next("nifler.svmb");
    let nimsem_p = next("nimsem.svmb");
    let hexer_p = next("hexer.svmb");
    let libdir = next("libdir");
    let nimcache = next("nimcache-dir");
    let sys_stem = next("sys-stem");
    let prog_stem = next("prog-stem");
    let export = next("export");
    let expected: i64 = next("expected").parse().expect("expected is i64");
    let call_args: Vec<i64> = a.map(|s| s.parse().expect("arg is i64")).collect();

    let read = |p: &str| std::fs::read(p).unwrap_or_else(|e| panic!("read {p}: {e}"));
    let nifler = read(&nifler_p);
    let nimsem = read(&nimsem_p);
    let hexer = read(&hexer_p);
    let nc = Path::new(&nimcache);
    let prog_pnif = read(&nc.join(format!("{prog_stem}.p.nif")).to_string_lossy());
    let sys_pnif = read(&nc.join(format!("{sys_stem}.p.nif")).to_string_lossy());
    let prog_nim = read(&nc.join("..").join("prog.nim").to_string_lossy());

    // ---- shared memfs: stdlib sources (under lib/, plus a flattened std/ view) + the two `.p.nif` ---
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
    files.push((format!("nimcache/{prog_stem}.p.nif"), prog_pnif.clone()));
    files.push((format!("nimcache/{sys_stem}.p.nif"), sys_pnif));
    files.push(("prog.nim".into(), prog_nim));
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

    // ---- phase 2: nimsem.svmb (sema), driving nifler.svmb via exec, over system then main ----------
    let run_nimsem = |mode: &str, stem: &str| {
        let run = instantiate(svm_encode::decode_module(&nimsem).expect("decode nimsem"))
            .expect("instantiate nimsem")
            .run_with_caps(
                Backend::TreeWalk,
                &cfg(
                    &[
                        "nimsem",
                        "--define:nimNativeAlloc",
                        "--define:nimNativeIo",
                        "m",
                        mode,
                        &format!("nimcache/{stem}.p.nif"),
                    ],
                    vec![],
                ),
                &[
                    ("fs", fs.grant()),
                    ("exec", domain_exec_with_fs(programs.clone(), fs.grant())),
                ],
            )
            .expect("run nimsem");
        assert!(
            matches!(run.outcome, Outcome::Exited(0) | Outcome::Returned(_)),
            "nimsem {mode}: {:?}",
            run.outcome
        );
        let snif = fs
            .read(&format!("nimcache/{stem}.s.nif"))
            .unwrap_or_else(|| panic!("nimsem wrote no {stem}.s.nif"));
        eprintln!(
            "phase 2  nimsem.svmb {mode}: {stem} → {} B .s.nif",
            snif.len()
        );
    };
    run_nimsem("--isSystem", &sys_stem);
    run_nimsem("--isMain", &prog_stem);

    // ---- phase 3: hexer.svmb (lower) — each semchecked module → Leng `.x.nif` ----------------------
    let run_hexer = |stem: &str| -> Vec<u8> {
        let run = instantiate(svm_encode::decode_module(&hexer).expect("decode hexer"))
            .expect("instantiate hexer")
            .run_with_caps(
                Backend::TreeWalk,
                &cfg(&["hexer", "c", &format!("nimcache/{stem}.s.nif")], vec![]),
                &[("fs", fs.grant())],
            )
            .expect("run hexer");
        assert!(
            matches!(run.outcome, Outcome::Exited(0) | Outcome::Returned(_)),
            "hexer {stem}: {:?}",
            run.outcome
        );
        let xnif = fs
            .read(&format!("nimcache/{stem}.x.nif"))
            .unwrap_or_else(|| panic!("hexer wrote no {stem}.x.nif"));
        eprintln!("phase 3  hexer.svmb: {stem} → {} B Leng .x.nif", xnif.len());
        xnif
    };
    let sys_x = run_hexer(&sys_stem);
    let prog_x = run_hexer(&prog_stem);

    // ---- phase 4 (host link + run): link the SVM-produced Leng with the W3 runtime shim, execute ---
    let prog_src = String::from_utf8_lossy(&prog_x).into_owned();
    let sys_src = String::from_utf8_lossy(&sys_x).into_owned();
    // Discover the bottom-edge C imports from the (self-contained) system module and bind them to the
    // shim; program-first ordering so func 0 is the natural entry (matches svm-leng's nim_e2e).
    let sys_unit = svm_leng::WholeModule {
        stem: &sys_stem,
        src: &sys_src,
    };
    let obj = svm_encode::decode_unit(
        &svm_leng::compile_whole_object(&sys_unit)
            .unwrap_or_else(|e| panic!("compile {sys_stem}: {e}")),
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
    let units = vec![
        svm_leng::WholeModule {
            stem: &prog_stem,
            src: &prog_src,
        },
        svm_leng::WholeModule {
            stem: &sys_stem,
            src: &sys_src,
        },
    ];
    let m: Module = svm_leng::link_whole_with_runtime(&units, vec![runtime])
        .unwrap_or_else(|e| panic!("link with runtime: {e}"));
    svm_verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    assert_eq!(m.imports.len(), 0, "every bottom-edge import is bound");
    eprintln!(
        "phase 4  linked → {} funcs, verified, import-free",
        m.funcs.len()
    );

    // Call `export` with `call_args` on both engines (§9 parity) and assert the result.
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
