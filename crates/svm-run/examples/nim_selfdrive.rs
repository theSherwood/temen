//! **Compile Nim on the SVM with NO native `nimony` bootstrap** (NIM.md §3c/§3e; #958 — the in-browser
//! card's headless core). `nim_e2e_chain` relied on a native `nimony c` to lay out the `nimcache`
//! (the `.p.nif` parse outputs, the module stems, and the `.build.nif` dependency plan). The browser
//! has no native nimony, so this driver plays **nifmake itself**: given only Nim source + the stdlib,
//! it computes each module's cache **stem** exactly as nimony does (`gear2/modnames.moduleSuffix`:
//! `name[0..3]` + base36 of `lib/tinyhashes.uhash(shortest-relative-path)`), crawls the `import`
//! dependency graph by parsing each module with `nifler.svmb` and reading its `.p.deps.nif`, then runs
//! `nimsem.svmb` (dependency-ordered) and `hexer.svmb` over the closure and links + runs the result —
//! every phase a sandboxed SVM guest, nothing native in the loop but the guest `.svmb` build.
//!
//! ```text
//! cargo run -q --release -p svm-run --example nim_selfdrive -- \
//!     <nifler.svmb> <nimsem.svmb> <hexer.svmb> <libdir> <proj-dir> <main.nim> <export> <expected> [arg...]
//! ```
//!
//! `<proj-dir>` holds the user's sources (`<main.nim>` + any `import`ed siblings); `<libdir>` is the
//! nimony stdlib. Modes match `nim_e2e_chain`: compute (`<export> <expected-i64> [args…]`) or I/O
//! (`<io> <expected-stdout>`). This is the exact logic the browser cdylib runs — proven here headless.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use svm_interp::Value;
use svm_ir::{LinkUnit, Module};
use svm_run::exec::{domain_exec_with_fs, DomainProgram};
use svm_run::{instantiate, Backend, HostCap, Limits, Outcome, RunConfig};

// ---- nimony's module-stem hash (gear2/modnames.nim + lib/tinyhashes.nim), reproduced exactly -------

/// `lib/tinyhashes.uhash` — a stable string hash that ends up in every NIF cache name.
fn uhash(s: &str) -> u32 {
    let mut h: u32 = 0;
    for c in s.bytes() {
        h = h.wrapping_add(c as u32);
        h = h.wrapping_add(h << 10);
        h ^= h >> 6;
    }
    h = h.wrapping_add(h << 3);
    h ^= h >> 11;
    h = h.wrapping_add(h << 15);
    h
}

fn base36(mut id: u32) -> String {
    const B36: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    let mut r = String::new();
    while id > 0 {
        r.push(B36[(id % 36) as usize] as char);
        id /= 36;
    }
    r
}

/// POSIX-ish `relativePath(path, base)` over absolute `/`-paths (both start with `/`). Enough for the
/// memfs layout (`/prog.nim`, `/lib/std/…`); emits `../` segments when `path` escapes `base`.
fn relative_path(path: &str, base: &str) -> String {
    let p: Vec<&str> = path
        .trim_matches('/')
        .split('/')
        .filter(|s| !s.is_empty())
        .collect();
    let b: Vec<&str> = base
        .trim_matches('/')
        .split('/')
        .filter(|s| !s.is_empty())
        .collect();
    let mut i = 0;
    while i < p.len() && i < b.len() && p[i] == b[i] {
        i += 1;
    }
    let mut out: Vec<&str> = vec![".."; b.len() - i];
    out.extend_from_slice(&p[i..]);
    out.join("/")
}

/// `gear2/modnames.moduleSuffix`: the module's cache stem — `name[0..3]` + base36(`uhash(rel)`), where
/// `rel` is the **shortest** of the file's path relative to the cwd and to each search path. `file` is
/// an absolute memfs path (`/prog.nim`, `/lib/std/syncio.nim`); `cwd`/`search_paths` are absolute too.
fn module_suffix(file: &str, cwd: &str, search_paths: &[&str]) -> String {
    let mut rel = relative_path(file, cwd);
    for s in search_paths {
        let c = relative_path(file, s);
        if c.len() < rel.len() {
            rel = c;
        }
    }
    let name = rel.rsplit('/').next().unwrap_or(&rel);
    let name = name.strip_suffix(".nim").unwrap_or(name);
    let mut stem: String = name.chars().take(3).collect();
    stem.push_str(&base36(uhash(&rel)));
    stem
}

// ---- the runtime shim binding table (mirrors nim_e2e_chain) ----------------------------------------

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

// ---- shared in-window memfs (mirrors nim_e2e_chain) ------------------------------------------------

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

fn cfg(args: &[&str]) -> RunConfig {
    RunConfig {
        limits: Limits {
            fuel: None,
            deadline: None,
            max_fibers: 0,
            max_vcpus: 0,
        },
        stdin: vec![],
        memory_size_log2: None,
        args: args.iter().map(|s| s.as_bytes().to_vec()).collect(),
        env: vec![],
        ..RunConfig::default()
    }
}

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

/// One module in the dependency closure: its cache `stem`, its nimony `role` (system / main /
/// imported), and the stems it depends on.
struct Mod {
    stem: String,
    role: Role,
    deps: Vec<String>,
}

#[derive(PartialEq, Clone, Copy)]
enum Role {
    System,
    Main,
    Import,
}

/// The balanced `(head …)` slice starting at the first `(head` in `s` (parens included), or `None`.
fn balanced(s: &str, head: &str) -> Option<String> {
    let start = s.find(&format!("({head}"))?;
    let b = s.as_bytes();
    let (mut j, mut depth) = (start, 0i32);
    while j < b.len() {
        match b[j] {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(s[start..=j].to_string());
                }
            }
            _ => {}
        }
        j += 1;
    }
    None
}

/// The **active** `import` targets in a `.p.deps.nif`, each as an absolute memfs file path.
/// `(infix / std syncio)` → `/lib/std/syncio.nim`; `(prefix \2E/ helper)` → `<importer-dir>/helper.nim`.
/// Imports carrying a `(when …)` guard are **platform-conditional** (e.g. `winlean` under
/// `defined(windows)`) and skipped — the same set native nimony resolves for a non-Windows target.
/// (Following an *active* conditional import isn't needed for the current fixtures; a refinement is to
/// evaluate the guard against the define set.)
fn parse_imports(deps_nif: &str, importer_dir: &str) -> Vec<String> {
    let mut out = vec![];
    // Walk every top-level (import …)/(fromimport …) block.
    for kw in ["import", "fromimport"] {
        let mut rest = deps_nif;
        while let Some(block) = balanced(rest, kw) {
            let adv = rest.find(&block).unwrap() + block.len();
            rest = &rest[adv..];
            if block.contains("(when") {
                continue; // platform-guarded — skip (matches native for non-Windows)
            }
            if let Some(inf) = balanced(&block, "infix") {
                // flat `(infix / std formatfloat)` → std/formatfloat
                let segs: Vec<&str> = inf
                    .trim_start_matches("(infix")
                    .trim_end_matches(')')
                    .split_whitespace()
                    .filter(|t| *t != "/" && !t.starts_with('('))
                    .collect();
                if !segs.is_empty() {
                    out.push(format!("/lib/{}.nim", segs.join("/")));
                }
            } else if let Some(pre) = balanced(&block, "prefix") {
                // `(prefix \2E/ helper)` → ./helper relative to the importer
                let segs: Vec<&str> = pre
                    .trim_start_matches("(prefix")
                    .trim_end_matches(')')
                    .split_whitespace()
                    .filter(|t| !t.contains("2E") && *t != "/" && !t.starts_with('\\'))
                    .collect();
                if !segs.is_empty() {
                    out.push(format!("{importer_dir}/{}.nim", segs.join("/")));
                }
            }
        }
    }
    out
}

fn main() {
    let mut a = std::env::args().skip(1);
    let mut next = |w: &str| a.next().unwrap_or_else(|| panic!("missing <{w}>"));
    let nifler_p = next("nifler.svmb");
    let nimsem_p = next("nimsem.svmb");
    let hexer_p = next("hexer.svmb");
    let libdir = next("libdir");
    let projdir = next("proj-dir");
    let main_nim = next("main.nim");
    let export = next("export");
    let expected_raw = next("expected");
    let io_mode = export == "<io>";
    let call_args: Vec<i64> = a.map(|s| s.parse().expect("arg is i64")).collect();

    let read = |p: &str| std::fs::read(p).unwrap_or_else(|e| panic!("read {p}: {e}"));
    let nifler = read(&nifler_p);
    let nimsem = read(&nimsem_p);
    let hexer = read(&hexer_p);

    // ---- seed the memfs: stdlib under lib/ (+ a flattened std/ view) and the project sources at / ----
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
    for e in std::fs::read_dir(&projdir).unwrap_or_else(|e| panic!("read_dir {projdir}: {e}")) {
        let p = e.expect("entry").path();
        if p.extension().and_then(|x| x.to_str()) == Some("nim") {
            let name = p.file_name().unwrap().to_string_lossy().into_owned();
            files.push((name, std::fs::read(&p).expect("read .nim")));
        }
    }
    let fs = Memfs::new(files);

    // nifler.svmb registered under the names nimsem uses for argv[0] (deps.nim shells out to `nifler`).
    let nifler_inst = Arc::new(
        instantiate(svm_encode::decode_module(&nifler).expect("decode nifler.svmb"))
            .expect("inst nifler"),
    );
    let programs: Vec<DomainProgram> = ["nifler", "/bin/nifler"]
        .iter()
        .map(|n| DomainProgram {
            name: (*n).into(),
            instance: nifler_inst.clone(),
            limits: Limits::default(),
        })
        .collect();

    let cwd = "/";
    let search = ["/lib"];
    let stem_of = |file: &str| module_suffix(file, cwd, &search);

    // ---- phase 1: parse + crawl the import closure with nifler.svmb (plays nifmake) -----------------
    // Every module (system + main + transitively imported) gets parsed to `nimcache/<stem>.p.nif`; its
    // `.p.deps.nif` names the next imports. `system` is nimony's implicit root (fixed stem sysvq0asl).
    let run_nifler = |file: &str, stem: &str| {
        let out = format!("/nimcache/{stem}.p.nif");
        let run = instantiate(svm_encode::decode_module(&nifler).expect("decode nifler"))
            .expect("inst nifler")
            .run_with_caps(
                Backend::TreeWalk,
                &cfg(&["nifler", "--portablePaths", "--deps", "parse", file, &out]),
                &[("fs", fs.grant())],
            )
            .expect("run nifler");
        assert!(
            matches!(run.outcome, Outcome::Exited(0) | Outcome::Returned(_)),
            "nifler {file}: {:?}",
            run.outcome
        );
    };

    let mut mods: BTreeMap<String, Mod> = BTreeMap::new();
    // worklist of (memfs file, role)
    let mut work = vec![
        ("/lib/std/system.nim".to_string(), Role::System),
        (format!("/{main_nim}"), Role::Main),
    ];
    while let Some((file, role)) = work.pop() {
        let stem = stem_of(&file);
        if mods.contains_key(&stem) {
            continue;
        }
        run_nifler(&file, &stem);
        let deps_nif = fs
            .read(&format!("nimcache/{stem}.p.deps.nif"))
            .map(|b| String::from_utf8_lossy(&b).into_owned())
            .unwrap_or_default();
        let dir = file
            .rsplit_once('/')
            .map(|(d, _)| d)
            .unwrap_or("")
            .to_string();
        let dir = if dir.is_empty() { "".to_string() } else { dir };
        let import_files = parse_imports(&deps_nif, &dir);
        let mut dep_stems = vec![];
        for imp in import_files {
            dep_stems.push(stem_of(&imp));
            work.push((imp, Role::Import));
        }
        eprintln!(
            "phase 1  nifler.svmb: {} → stem {stem} ({} import(s))",
            file,
            dep_stems.len()
        );
        mods.insert(
            stem.clone(),
            Mod {
                stem,
                role,
                deps: dep_stems,
            },
        );
    }

    // ---- dependency order (DFS postorder): a module after every module it depends on ---------------
    let order = {
        let stems: Vec<String> = mods.keys().cloned().collect();
        let mut seen: BTreeMap<String, u8> = BTreeMap::new();
        let mut order = vec![];
        fn visit(
            s: &str,
            mods: &BTreeMap<String, Mod>,
            seen: &mut BTreeMap<String, u8>,
            order: &mut Vec<String>,
        ) {
            if seen.get(s).copied().unwrap_or(0) != 0 {
                return;
            }
            seen.insert(s.to_string(), 1);
            if let Some(m) = mods.get(s) {
                for d in &m.deps {
                    visit(d, mods, seen, order);
                }
            }
            seen.insert(s.to_string(), 2);
            order.push(s.to_string());
        }
        for s in &stems {
            visit(s, &mods, &mut seen, &mut order);
        }
        // system is the implicit dependency of everything; force it first even if nothing imports it.
        order.sort_by_key(|s| mods.get(s).map(|m| m.role != Role::System).unwrap_or(true));
        order
    };
    let main_stem = mods
        .values()
        .find(|m| m.role == Role::Main)
        .map(|m| m.stem.clone())
        .expect("no main module");
    eprintln!(
        "crawl: {} modules, order [{}]",
        order.len(),
        order.join(" → ")
    );

    // ---- phase 2: nimsem.svmb (sema), dependency-ordered, driving nifler.svmb via exec -------------
    for stem in &order {
        let m = &mods[stem];
        let pnif = format!("nimcache/{stem}.p.nif");
        let mut argv = vec![
            "nimsem",
            "--define:nimNativeAlloc",
            "--define:nimNativeIo",
            "m",
        ];
        match m.role {
            Role::System => argv.push("--isSystem"),
            Role::Main => argv.push("--isMain"),
            Role::Import => {}
        }
        argv.push(&pnif);
        let run = instantiate(svm_encode::decode_module(&nimsem).expect("decode nimsem"))
            .expect("inst nimsem")
            .run_with_caps(
                Backend::TreeWalk,
                &cfg(&argv),
                &[
                    ("fs", fs.grant()),
                    ("exec", domain_exec_with_fs(programs.clone(), fs.grant())),
                ],
            )
            .expect("run nimsem");
        assert!(
            matches!(run.outcome, Outcome::Exited(0) | Outcome::Returned(_)),
            "nimsem {stem}: {:?}",
            run.outcome
        );
        let s = fs
            .read(&format!("nimcache/{stem}.s.nif"))
            .unwrap_or_else(|| panic!("nimsem wrote no {stem}.s.nif"));
        eprintln!("phase 2  nimsem.svmb: {stem} → {} B .s.nif", s.len());
    }

    // ---- phase 3: hexer.svmb (lower) — main gets the app-entry glue --------------------------------
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
            .expect("inst hexer")
            .run_with_caps(Backend::TreeWalk, &cfg(&argv), &[("fs", fs.grant())])
            .expect("run hexer");
        assert!(
            matches!(run.outcome, Outcome::Exited(0) | Outcome::Returned(_)),
            "hexer {stem}: {:?}",
            run.outcome
        );
        let key = if is_main {
            format!("{outdir}/{stem}.x.nif")
        } else {
            format!("nimcache/{stem}.x.nif")
        };
        let x = fs
            .read(&key)
            .unwrap_or_else(|| panic!("hexer wrote no {key}"));
        eprintln!("phase 3  hexer.svmb: {stem} → {} B Leng .x.nif", x.len());
        x
    };
    let leng: Vec<(String, String)> = order
        .iter()
        .map(|stem| {
            (
                stem.clone(),
                String::from_utf8_lossy(&run_hexer(stem)).into_owned(),
            )
        })
        .collect();

    // ---- phase 4: link + run (main first, system last) --------------------------------------------
    let mut ordered: Vec<&(String, String)> = leng.iter().collect();
    ordered.sort_by_key(|(stem, _)| (*stem != main_stem, stem.starts_with("sysv")));
    let units: Vec<svm_leng::WholeModule> = ordered
        .iter()
        .map(|(stem, src)| svm_leng::WholeModule { stem, src })
        .collect();

    if io_mode {
        let m: Module =
            svm_leng::link_nim_powerbox(&units).unwrap_or_else(|e| panic!("nim→powerbox: {e}"));
        svm_verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
        assert!(svm_run::is_named_powerbox_entry(&m), "not a powerbox entry");
        let want = expected_raw.replace("\\n", "\n").replace("\\t", "\t");
        let run = svm_run::run_powerbox(&m, &[]).unwrap_or_else(|e| panic!("run_powerbox: {e}"));
        let got = String::from_utf8_lossy(&run.stdout);
        assert_eq!(got.as_ref(), want, "stdout mismatch (got {got:?})");
        println!(
            "✅ COMPILED NIM ON SVM, NO NATIVE BOOTSTRAP (I/O): {} modules → stdout = {got:?}",
            order.len()
        );
        return;
    }

    let expected: i64 = expected_raw.parse().expect("expected is i64");
    let sys = leng
        .iter()
        .find(|(s, _)| s.starts_with("sysv"))
        .map(|(s, src)| svm_leng::WholeModule { stem: s, src })
        .expect("no system unit");
    let obj = svm_encode::decode_unit(
        &svm_leng::compile_whole_object(&sys).unwrap_or_else(|e| panic!("compile system: {e}")),
    )
    .expect("decode object");
    let exports: Vec<(String, u32)> = obj
        .imports
        .iter()
        .filter_map(|imp| shim_index(&imp.name).map(|i| (imp.name.clone(), i)))
        .collect();
    const SHIM: &str = include_str!("../../svm-leng/src/powerbox_compute_shim.svm.txt");
    let runtime = LinkUnit {
        module: svm_text::parse_module(SHIM).expect("shim parses"),
        exports,
        ..Default::default()
    };
    let m: Module = svm_leng::link_whole_with_runtime(&units, vec![runtime])
        .unwrap_or_else(|e| panic!("link: {e}"));
    svm_verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    assert_eq!(m.imports.len(), 0, "import-free");

    let idx = m
        .exports
        .iter()
        .find(|e| e.name.contains(&export))
        .unwrap_or_else(|| panic!("no export `{export}`"))
        .func;
    let seed = vec![0u8; 1 << 20];
    let ivals: Vec<Value> = call_args.iter().map(|&n| Value::I64(n)).collect();
    let mut fuel = 500_000_000u64;
    let (ir, _) = svm_interp::run_capture(&m, idx, &ivals, &mut fuel, &seed);
    let iword = match ir.expect("interp").as_slice() {
        [Value::I64(n)] => *n,
        o => panic!("interp {o:?}"),
    };
    let (jout, _) = svm_jit::compile_and_run_capture(&m, idx, &call_args, &seed).expect("jit");
    let jword = match jout {
        svm_jit::JitOutcome::Returned(v) => v,
        o => panic!("jit {o:?}"),
    };
    assert_eq!(vec![iword], jword, "§9 interp/JIT parity");
    assert_eq!(iword, expected, "{export}({call_args:?}) result");
    println!(
        "✅ COMPILED NIM ON SVM, NO NATIVE BOOTSTRAP: {} modules → {export}({call_args:?}) = {iword} \
         (interp == JIT == {expected})",
        order.len()
    );
}
