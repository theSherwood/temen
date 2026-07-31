//! **Tier-2 end-to-end: real Nim source runs on SVM.** Unlike the translator-unit tests (small
//! hand-written Leng snippets), these start from *Nim source* and drive the whole real toolchain —
//! `nimony c` (→ nifler → nimony → hexer) emits the program's and the `system` module's Leng
//! (`.x.nif`), `svm-leng` lowers and links them together with the W3 runtime shim, and the result
//! runs on **both engines** (§9 interp/JIT parity). Nothing mid-pipeline is committed: the fixture
//! is the Nim source, and the toolchain regenerates everything downstream, so the test can never
//! rot against a stale snapshot.
//!
//! **Toolchain gating.** These need the nimony toolchain (`nimony` + `nim` on `PATH`, or pointed to
//! by `NIMONY_BIN`/`NIM_BIN`). In CI a provisioning step builds it (see `.github/workflows/ci.yml`,
//! the `nim-e2e` job); locally they run if the toolchain is installed. When it is absent the tests
//! **skip** (print `SKIP` and return) rather than fail — the translator's own logic is covered by
//! the fast, toolchain-free unit tests.

use std::process::Command;
use svm_interp::Value;
use svm_ir::{LinkUnit, Module};

/// The runtime shim's function indices, keyed by the C symbol each bottom-edge import lowers to.
/// Longest-prefix wins so `atomicCompareExchangeN` is not shadowed by a shorter atomic name.
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

/// The shim function bound to a **bottom-edge C** import, or `None` for a cross-module nimony symbol
/// (`ini`, a proc defined in a sibling module) — those resolve against the other link units, not the
/// runtime shim. Longest-prefix wins so `atomicCompareExchangeN` isn't shadowed by a shorter atomic.
fn shim_index(name: &str) -> Option<u32> {
    SHIM_BINDINGS
        .iter()
        .filter(|(p, _)| name.starts_with(p))
        .max_by_key(|(p, _)| p.len())
        .map(|(_, i)| *i)
}

/// Locate the nimony toolchain. Honours `NIMONY_BIN`/`NIM_BIN` (directories holding `nimony`/`nim`),
/// else looks for `nimony` on `PATH`. Returns the `PATH` value to run the compiler under (nimony
/// shells out to `nim`), or `None` when the toolchain is not installed — the caller then skips.
fn toolchain_path() -> Option<String> {
    let path = std::env::var("PATH").unwrap_or_default();
    let mut prefix = Vec::new();
    if let Ok(d) = std::env::var("NIMONY_BIN") {
        prefix.push(d);
    }
    if let Ok(d) = std::env::var("NIM_BIN") {
        prefix.push(d);
    }
    let full = if prefix.is_empty() {
        path.clone()
    } else {
        format!("{}:{}", prefix.join(":"), path)
    };
    // Confirm `nimony` is actually runnable under this PATH.
    let ok = full
        .split(':')
        .any(|d| !d.is_empty() && std::path::Path::new(d).join("nimony").exists())
        || which("nimony", &full);
    ok.then_some(full)
}

fn which(bin: &str, path: &str) -> bool {
    path.split(':')
        .any(|d| !d.is_empty() && std::path::Path::new(d).join(bin).is_file())
}

/// Compile Nim `source` with `nimony c --isMain` in a throwaway directory and return every module's
/// Leng as `(stem, x_nif_text)` — the main program plus the `system` module (and any deps). The
/// stem is the `.x.nif` basename, exactly the qualifier `svm-leng` links symbols under.
fn compile_to_leng(nim_path: &str, source: &str) -> Vec<(String, String)> {
    // A per-source directory (hash of the program) so tests running in parallel never share a
    // `nimcache` — each `nimony c` gets its own throwaway tree.
    let tag = {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        source.hash(&mut h);
        h.finish()
    };
    let dir = std::env::temp_dir().join(format!("svm_nim_e2e_{}_{tag:016x}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("mk tempdir");
    let src = dir.join("prog.nim");
    std::fs::write(&src, source).expect("write prog.nim");

    let out = Command::new("nimony")
        .args(["c", "--isMain", "prog.nim"])
        .current_dir(&dir)
        .env("PATH", nim_path)
        .output()
        .expect("run nimony");
    assert!(
        out.status.success(),
        "nimony c failed:\n{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    let mut mods = Vec::new();
    collect_x_nif(&dir.join("nimcache"), &mut mods);
    assert!(
        mods.iter().any(|(s, _)| s.starts_with("sysv")),
        "expected the system module among {:?}",
        mods.iter().map(|(s, _)| s).collect::<Vec<_>>()
    );
    let _ = std::fs::remove_dir_all(&dir);
    mods
}

fn collect_x_nif(dir: &std::path::Path, out: &mut Vec<(String, String)>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            collect_x_nif(&p, out);
        } else if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
            if let Some(stem) = name.strip_suffix(".x.nif") {
                if out.iter().all(|(s, _)| s != stem) {
                    out.push((stem.to_string(), std::fs::read_to_string(&p).unwrap()));
                }
            }
        }
    }
}

/// Link the compiled Nim modules together with the W3 runtime shim into one verified, import-free
/// SVM module. The shim's exports (its 20 functions) are bound to whatever bottom-edge C imports the
/// modules actually reference — discovered from each module's compiled object, so the mangled atomic
/// symbol names never have to be hard-coded.
fn link_with_runtime(mods: &[(String, String)]) -> Module {
    // Discover the real import names from the **system module** only (stem `sysv…`). Every
    // bottom-edge C import (`mmap`, `memcpy`, the atomics, …) originates there, and — unlike a
    // program module that references a cross-module aggregate type such as `string.0.sysv…` — the
    // system module is self-contained, so it compiles standalone (no pooled types) for discovery.
    let mut import_names: Vec<String> = Vec::new();
    for (stem, src) in mods.iter().filter(|(stem, _)| stem.starts_with("sysv")) {
        let obj = svm_encode::decode_unit(
            &svm_leng::compile_whole_object(&svm_leng::WholeModule { stem, src })
                .unwrap_or_else(|e| panic!("compile {stem}: {e}")),
        )
        .expect("decode object");
        for imp in &obj.imports {
            if import_names.iter().all(|n| n != &imp.name) {
                import_names.push(imp.name.clone());
            }
        }
    }

    const SHIM: &str = include_str!("fixtures/system_runtime.svm.txt");
    let shim = svm_text::parse_module(SHIM).expect("runtime shim parses");
    // Bind only the bottom-edge C imports to the shim; cross-module nimony imports (`ini`, sibling
    // procs) are left for the other link units to resolve.
    let exports: Vec<(String, u32)> = import_names
        .iter()
        .filter_map(|n| shim_index(n).map(|i| (n.clone(), i)))
        .collect();
    let runtime = LinkUnit {
        module: shim,
        exports,
        ..Default::default()
    };

    let units: Vec<svm_leng::WholeModule> = mods
        .iter()
        .map(|(stem, src)| svm_leng::WholeModule { stem, src })
        .collect();
    let m = svm_leng::link_whole_with_runtime(&units, vec![runtime])
        .unwrap_or_else(|e| panic!("link with runtime: {e}"));
    svm_verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    assert_eq!(m.imports.len(), 0, "every bottom-edge import is bound");
    m
}

/// Run exported proc `export_substr` with `args` on both engines (§9 parity) and return the i64.
fn run_export(m: &Module, export_substr: &str, args: &[i64]) -> i64 {
    let idx = m
        .exports
        .iter()
        .find(|e| e.name.contains(export_substr))
        .unwrap_or_else(|| panic!("no export matching `{export_substr}`"))
        .func;
    let seed = vec![0u8; 1 << 20];
    let ivals: Vec<Value> = args.iter().map(|&n| Value::I64(n)).collect();
    let mut fuel = 500_000_000u64;
    let (ir, _) = svm_interp::run_capture(m, idx, &ivals, &mut fuel, &seed);
    let iword = match ir.expect("interp").as_slice() {
        [Value::I64(n)] => *n,
        o => panic!("unexpected {o:?}"),
    };
    let (jout, _) = svm_jit::compile_and_run_capture(m, idx, args, &seed).expect("jit");
    let jword = match jout {
        svm_jit::JitOutcome::Returned(v) => v,
        o => panic!("jit: {o:?}"),
    };
    assert_eq!(vec![iword], jword, "§9 interp/JIT parity");
    iword
}

/// Compile Nim `source`, link it with the runtime, and hand the linked module to `check` — or skip
/// (printing why) when the toolchain is not installed.
fn with_program(source: &str, check: impl FnOnce(&Module)) {
    let Some(path) = toolchain_path() else {
        eprintln!("SKIP: nimony toolchain not found (set NIMONY_BIN/NIM_BIN or install on PATH)");
        return;
    };
    let mods = compile_to_leng(&path, source);
    let m = link_with_runtime(&mods);
    check(&m);
}

/// Run exported proc `export_substr` with `args` on both engines, seeding the bump-allocator cursor
/// (window offset 8) to `heap_start`; returns the i64 result and the interp's final window (so a
/// follow-up call can continue from the advanced cursor).
fn run_export_seeded(m: &Module, export_substr: &str, args: &[i64], seed: &[u8]) -> (i64, Vec<u8>) {
    let idx = m
        .exports
        .iter()
        .find(|e| e.name.contains(export_substr))
        .unwrap_or_else(|| panic!("no export matching `{export_substr}`"))
        .func;
    let ivals: Vec<Value> = args.iter().map(|&n| Value::I64(n)).collect();
    let mut fuel = 500_000_000u64;
    let (ir, imem) = svm_interp::run_capture(m, idx, &ivals, &mut fuel, seed);
    let iword = match ir.expect("interp").as_slice() {
        [Value::I64(n)] => *n,
        o => panic!("unexpected {o:?}"),
    };
    let (jout, _) = svm_jit::compile_and_run_capture(m, idx, args, seed).expect("jit");
    let jword = match jout {
        svm_jit::JitOutcome::Returned(v) => v,
        o => panic!("jit: {o:?}"),
    };
    assert_eq!(vec![iword], jword, "§9 interp/JIT parity");
    (iword, imem)
}

#[test]
fn nim_addtwo_runs_on_svm() {
    with_program(
        "proc addTwo(a, b: int): int = a + b\nlet r = addTwo(2, 3)\n",
        |m| {
            assert_eq!(run_export(m, "addTwo", &[2, 3]), 5);
            assert_eq!(run_export(m, "addTwo", &[40, 2]), 42);
        },
    );
}

#[test]
fn nim_arithmetic_and_control_flow_runs_on_svm() {
    // A pure-integer routine with a loop and a branch — real Nim `while`/`if`, compiled through the
    // whole toolchain and run on both engines. sumTo(5) = 1+2+3+4+5 = 15; maxOf picks the larger.
    with_program(
        "proc sumTo(n: int): int =\n  result = 0\n  var i = 1\n  while i <= n:\n    result = result + i\n    i = i + 1\n\nproc maxOf(a, b: int): int =\n  if a > b: a else: b\n\nlet r = sumTo(5)\nlet s = maxOf(7, 3)\n",
        |m| {
            assert_eq!(run_export(m, "sumTo", &[5]), 15);
            assert_eq!(run_export(m, "sumTo", &[10]), 55);
            assert_eq!(run_export(m, "maxOf", &[7, 3]), 7);
            assert_eq!(run_export(m, "maxOf", &[3, 9]), 9);
        },
    );
}

#[test]
fn real_allocator_runs_end_to_end() {
    // The `system` module linked into any program is the real one, so its allocator is exercisable.
    // `osAllocPages` is the raw page source — it calls the bound `mmap`, which the shim serves from a
    // bump cursor at window offset 8. Two calls must return the seeded heap start and then one page
    // past it — real stdlib allocation running on both engines, with the `system` module sourced from
    // the toolchain (no committed artifact).
    with_program("proc noop() = discard\nnoop()\n", |m| {
        let mut window = vec![0u8; 1 << 20];
        window[8..16].copy_from_slice(&(1i64 << 19).to_le_bytes());
        let (first, after) = run_export_seeded(m, "osAllocPages.0.sysvq0asl", &[4096], &window);
        assert_eq!(first, 1 << 19, "first page is the seeded heap start");
        let (second, _) = run_export_seeded(m, "osAllocPages.0.sysvq0asl", &[4096], &after);
        assert_eq!(second, (1 << 19) + 4096, "second page bumped by one page");
    });
}
