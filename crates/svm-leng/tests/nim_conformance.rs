//! **Nim language conformance suite on the SVM** (#956, parent #954; NIM.md §3a). A gated fixture
//! matrix that measures the *envelope* of Nim language features which compile-and-run on the SVM —
//! generics, exceptions, closures, methods (dynamic dispatch), `seq`/`string`/`Table`, floats,
//! iterators, `case`/variant objects, `ref` + ARC destructors — each driven through the **whole real
//! toolchain** (`nimony c` → nifler → nimony → hexer emit Leng, `svm-leng` lowers + links with the W3
//! compute shim) and **run on both engines** (§9 interp/JIT parity), reducing to an `int` we check.
//!
//! Unlike the exact-value tests in `nim_e2e.rs`, this suite tolerates *known* fail-closed features: each
//! fixture carries an [`Expect`], and the test asserts the **measured** status matches it. A feature that
//! starts working (Fails→Runs) or regresses (Runs→Fails) both fail the test — the first prompts flipping
//! the expectation (and closing the feature's sub-issue), the second is a real regression. This is the
//! "green/red matrix, each red a filed ticket" the issue asks for — the driver + oracle surfacing #760's
//! remaining arms from the *language* side.
//!
//! **Toolchain gating.** Needs the nimony toolchain (`nimony` + `nim` on `PATH`, or `NIMONY_BIN`/
//! `NIM_BIN`); when absent the test **skips** (like `nim_e2e.rs`). CI's `nim-e2e` job provisions it.
//! Set `NIM_CONFORMANCE_MEASURE=1` to print the matrix and skip the assertion (re-baselining aid).
//!
//! Shares no code with `nim_e2e.rs` on purpose: that file's helpers `panic!` on any pipeline failure
//! (it asserts exact behavior), whereas this suite must catch a fail-closed and record it, so its
//! pipeline primitives return `Result`/status instead.

use std::panic::AssertUnwindSafe;
use std::process::Command;
use svm_interp::Value;
use svm_ir::{LinkUnit, Module};

// ---- toolchain gating (mirrors nim_e2e.rs) --------------------------------------------------------

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
    let ok = full
        .split(':')
        .any(|d| !d.is_empty() && std::path::Path::new(d).join("nimony").exists())
        || full
            .split(':')
            .any(|d| !d.is_empty() && std::path::Path::new(d).join("nimony").is_file());
    ok.then_some(full)
}

// ---- the runtime compute shim's bottom-edge bindings (mirrors nim_e2e.rs) -------------------------

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

// ---- the pipeline, Result-returning so a fail-closed is data, not a panic -------------------------

/// The pipeline stage a fixture reaches before failing — the routing hint for its sub-issue.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Stage {
    /// `nimony c` rejected the source — a nimony **front-end** gap (not svm-leng's).
    Frontend,
    /// `svm-leng` fail-closed lowering/linking the Leng — a #760 totality arm.
    Translate,
    /// The linked module failed verification.
    Verify,
    /// It ran but trapped, mismatched across engines, or returned the wrong value.
    Run,
}

#[derive(Clone, PartialEq, Eq, Debug)]
enum Status {
    Runs,
    Fails(Stage),
}

/// Compile Nim `source` with `nimony c --isMain` in a throwaway dir; return every module's Leng as
/// `(stem, x_nif_text)`, or `Err` if the **front end** rejected it.
fn compile_to_leng(nim_path: &str, source: &str) -> Result<Vec<(String, String)>, Stage> {
    let tag = {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        source.hash(&mut h);
        h.finish()
    };
    let dir = std::env::temp_dir().join(format!("svm_nim_conf_{}_{tag:016x}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("mk tempdir");
    std::fs::write(dir.join("prog.nim"), source).expect("write prog.nim");

    let out = Command::new("nimony")
        .args(["c", "--isMain", "prog.nim"])
        .current_dir(&dir)
        .env("PATH", nim_path)
        .output()
        .expect("run nimony");
    if !out.status.success() {
        let _ = std::fs::remove_dir_all(&dir);
        return Err(Stage::Frontend);
    }
    let mut mods = Vec::new();
    collect_x_nif(&dir.join("nimcache"), &mut mods);
    let _ = std::fs::remove_dir_all(&dir);
    // No system module ⇒ the emit went sideways; treat as a front-end miss (nothing to lower).
    if !mods.iter().any(|(s, _)| s.starts_with("sysv")) {
        return Err(Stage::Frontend);
    }
    Ok(mods)
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
                    let bytes = std::fs::read(&p).unwrap();
                    out.push((
                        stem.to_string(),
                        String::from_utf8_lossy(&bytes).into_owned(),
                    ));
                }
            }
        }
    }
}

/// Link the compiled modules with the W3 runtime shim into one verified, import-free module, or `Err`
/// with the stage that fail-closed (`Translate` for an `svm-leng` `Unsupported`, `Verify` for a bad
/// module). Wrapped in `catch_unwind` because a residual translate path may `panic!`/`unreachable!`
/// rather than return `Err` — for a conformance probe that is still a clean fail-closed, not an abort.
fn link_with_runtime(mods: &[(String, String)]) -> Result<Module, Stage> {
    let res = std::panic::catch_unwind(AssertUnwindSafe(|| link_inner(mods)));
    match res {
        Ok(inner) => inner,
        Err(_) => Err(Stage::Translate),
    }
}

fn link_inner(mods: &[(String, String)]) -> Result<Module, Stage> {
    let mut import_names: Vec<String> = Vec::new();
    for (stem, src) in mods.iter().filter(|(stem, _)| stem.starts_with("sysv")) {
        let obj = svm_leng::compile_whole_object(&svm_leng::WholeModule { stem, src })
            .map_err(|_| Stage::Translate)?;
        let obj = svm_encode::decode_unit(&obj).map_err(|_| Stage::Translate)?;
        for imp in &obj.imports {
            if import_names.iter().all(|n| n != &imp.name) {
                import_names.push(imp.name.clone());
            }
        }
    }

    const SHIM: &str = include_str!("../src/powerbox_compute_shim.svm.txt");
    let shim = svm_text::parse_module(SHIM).expect("runtime shim parses");
    let exports: Vec<(String, u32)> = import_names
        .iter()
        .filter_map(|n| shim_index(n).map(|i| (n.clone(), i)))
        .collect();
    let runtime = LinkUnit {
        module: shim,
        exports,
        ..Default::default()
    };

    let mut ordered: Vec<&(String, String)> = mods.iter().collect();
    ordered.sort_by_key(|(stem, _)| stem.starts_with("sysv"));
    let units: Vec<svm_leng::WholeModule> = ordered
        .iter()
        .map(|(stem, src)| svm_leng::WholeModule { stem, src })
        .collect();
    let m =
        svm_leng::link_whole_with_runtime(&units, vec![runtime]).map_err(|_| Stage::Translate)?;
    svm_verify::verify_module(&m).map_err(|_| Stage::Verify)?;
    if !m.imports.is_empty() {
        return Err(Stage::Verify);
    }
    Ok(m)
}

/// Run the C `main` (full init chain), read back an `int` global by name prefix, and require §9
/// interp/JIT parity. `Err(Stage::Run)` on a trap, a cross-engine mismatch, or a missing global.
fn run_main_read_global(m: &Module, global_substr: &str) -> Result<i64, Stage> {
    let res = std::panic::catch_unwind(AssertUnwindSafe(|| run_inner(m, global_substr)));
    match res {
        Ok(inner) => inner,
        Err(_) => Err(Stage::Run),
    }
}

fn run_inner(m: &Module, global_substr: &str) -> Result<i64, Stage> {
    let main = m
        .exports
        .iter()
        .find(|e| e.name == "main")
        .ok_or(Stage::Run)?
        .func;
    let off = m
        .data_exports
        .iter()
        .find(|e| e.name.starts_with(global_substr))
        .ok_or(Stage::Run)?
        .offset as usize;
    let entry_sp = svm_ir::powerbox_entry_sp(m) as i64;
    let heap_base = entry_sp + svm_ir::POWERBOX_STACK_RESERVE as i64;
    let mut seed = vec![0u8; 1 << 20];
    let brk = svm_ir::POWERBOX_HEAP_BRK as usize;
    seed[brk..brk + 8].copy_from_slice(&heap_base.to_le_bytes());
    let ivals = [
        Value::I64(entry_sp),
        Value::I32(0),
        Value::I64(0),
        Value::I64(0),
    ];
    let mut fuel = 500_000_000u64;
    let (ir, imem) = svm_interp::run_capture(m, main, &ivals, &mut fuel, &seed);
    if ir.is_err() || imem.len() < off + 8 {
        return Err(Stage::Run);
    }
    let iv = i64::from_le_bytes(imem[off..off + 8].try_into().unwrap());
    let (jout, jmem) = svm_jit::compile_and_run_capture(m, main, &[entry_sp, 0, 0, 0], &seed)
        .map_err(|_| Stage::Run)?;
    if !matches!(jout, svm_jit::JitOutcome::Returned(_)) || jmem.len() < off + 8 {
        return Err(Stage::Run);
    }
    let jv = i64::from_le_bytes(jmem[off..off + 8].try_into().unwrap());
    if iv != jv {
        return Err(Stage::Run); // §9 parity break
    }
    Ok(iv)
}

// ---- the fixture matrix ---------------------------------------------------------------------------

/// Whether a feature is expected to run end-to-end today, or fail closed (and where). Keep this column
/// honest: it is the committed baseline the test gates against.
#[derive(Clone, PartialEq, Eq, Debug)]
enum Expect {
    Runs,
    FailsClosed(Stage),
}

struct Fixture {
    /// The language feature this exercises (the matrix row label).
    feature: &'static str,
    /// Nim source; must end with a top-level `let r = <int expr>` — we read `r.0.` back.
    source: &'static str,
    /// The `int` value a *working* run must produce (checked only when it Runs).
    expected: i64,
    /// The committed baseline: does it run today, or fail closed (and at which stage)?
    expect: Expect,
    /// The sub-issue tracking a fail-closed feature (for the printed matrix); `None` when it Runs.
    ticket: Option<&'static str>,
}

/// Measure one fixture's actual end-to-end status (and, when it runs, its value).
fn measure(path: &str, f: &Fixture) -> (Status, Option<i64>) {
    let mods = match compile_to_leng(path, f.source) {
        Ok(m) => m,
        Err(s) => return (Status::Fails(s), None),
    };
    let m = match link_with_runtime(&mods) {
        Ok(m) => m,
        Err(s) => return (Status::Fails(s), None),
    };
    match run_main_read_global(&m, "r.0.") {
        Ok(v) if v == f.expected => (Status::Runs, Some(v)),
        Ok(v) => (Status::Fails(Stage::Run), Some(v)), // ran, wrong value
        Err(s) => (Status::Fails(s), None),
    }
}

/// The feature envelope (#956). Each fixture reduces its feature to an `int` global `r` so the same
/// run-and-read-back probe measures them all. `expect`/`ticket` are set from the measured baseline
/// below the fold — a fresh measurement is a one-liner: `NIM_CONFORMANCE_MEASURE=1 cargo test -p
/// svm-leng --test nim_conformance -- --nocapture`.
const FIXTURES: &[Fixture] = &[
    // ---- anchors: features already proven by nim_e2e.rs, here as regression sentinels ----
    Fixture {
        feature: "arithmetic (in a proc)",
        source: "proc calc(a, b: int): int = a * b + 6\nlet r = calc(6, 6)\n",
        expected: 42,
        expect: Expect::Runs,
        ticket: None,
    },
    Fixture {
        feature: "control-flow (while/if)",
        source: "proc sumTo(n: int): int =\n  result = 0\n  var i = 1\n  while i <= n:\n    result = result + i\n    i = i + 1\nlet r = sumTo(8) + 6\n",
        expected: 42,
        expect: Expect::Runs,
        ticket: None,
    },
    Fixture {
        feature: "seq[int] (alloc/add/index)",
        source: "proc build(n: int): int =\n  var s: seq[int] = @[]\n  var i = 0\n  while i < n:\n    s.add(i * i)\n    i = i + 1\n  result = 0\n  for x in s:\n    result = result + x\nlet r = build(5) + 12\n",
        expected: 42,
        expect: Expect::Runs,
        ticket: None,
    },
    // ---- the breadth envelope ----
    Fixture {
        feature: "generics",
        source: "proc pick[T](a, b: T): T = a\nlet r = pick(42, 7)\n",
        expected: 42,
        expect: Expect::Runs,
        ticket: None,
    },
    Fixture {
        feature: "exceptions (raise/try/except)",
        source: "proc mayFail(x: int): int =\n  if x < 0: raise newException(ValueError, \"neg\")\n  else: x * 2\nproc safe(x: int): int =\n  try: mayFail(x)\n  except ValueError: -1\nlet r = safe(21)\n",
        expected: 42,
        expect: Expect::FailsClosed(Stage::Frontend),
        ticket: Some("#980 (nimony front-end exceptions surface)"),
    },
    Fixture {
        feature: "closures (capture upvalue)",
        source: "proc outer(): int =\n  var n = 40\n  proc inner(): int {.closure.} = n + 2\n  inner()\nlet r = outer()\n",
        expected: 42,
        expect: Expect::Runs,
        ticket: None,
    },
    Fixture {
        feature: "methods (dynamic dispatch)",
        source: "type\n  Animal = ref object of RootObj\n  Dog = ref object of Animal\nmethod speak(a: Animal): int {.base.} = 0\nmethod speak(d: Dog): int = 42\nproc run(): int =\n  let a: Animal = Dog()\n  a.speak()\nlet r = run()\n",
        expected: 42,
        // Compiles + runs, but dispatch resolves to the *base* method (returns 0, not 42) — a silent
        // wrong-answer, both engines agreeing. See #979.
        expect: Expect::FailsClosed(Stage::Run),
        ticket: Some("#979 (method dispatch selects base)"),
    },
    Fixture {
        feature: "string (.len)",
        source: "let s = \"hello, world!\"\nlet r = s.len * 3 + 3\n",
        expected: 42,
        expect: Expect::Runs,
        ticket: None,
    },
    Fixture {
        feature: "Table[string,int]",
        source: "import std/tables\nproc run(): int =\n  var t = initTable[string, int]()\n  t[\"a\"] = 42\n  t[\"a\"]\nlet r = run()\n",
        expected: 42,
        // `std/tables` routines are `.raises`, so this fails the same front-end check as exceptions —
        // blocked transitively until the exceptions surface exists. Same root cause, tracked in #980.
        expect: Expect::FailsClosed(Stage::Frontend),
        ticket: Some("#980 (blocked by exceptions surface)"),
    },
    Fixture {
        feature: "floats (arith + int conv)",
        source: "let x = 3.5\nlet r = int(x * 12.0)\n",
        expected: 42,
        expect: Expect::Runs,
        ticket: None,
    },
    Fixture {
        feature: "custom iterators (yield)",
        source: "iterator countUp(n: int): int =\n  var i = 0\n  while i < n:\n    yield i\n    inc i\nproc run(): int =\n  result = 6\n  for x in countUp(9):\n    result = result + x\nlet r = run()\n",
        expected: 42,
        expect: Expect::Runs,
        ticket: None,
    },
    Fixture {
        feature: "variant objects (case)",
        source: "type\n  Kind = enum kA, kB\n  Node = object\n    case kind: Kind\n    of kA: a: int\n    of kB: b: int\nproc val(n: Node): int =\n  case n.kind\n  of kA: n.a\n  of kB: n.b\nlet r = val(Node(kind: kA, a: 42))\n",
        expected: 42,
        expect: Expect::Runs,
        ticket: None,
    },
    Fixture {
        feature: "ref object",
        source: "type Box = ref object\n  v: int\nproc run(): int =\n  var b = Box(v: 42)\n  b.v\nlet r = run()\n",
        expected: 42,
        expect: Expect::Runs,
        ticket: None,
    },
    Fixture {
        feature: "object + ARC destructor",
        source: "var freed {.global.}: int = 0\ntype Res = object\n  id: int\nproc `=destroy`(x: Res) =\n  freed = freed + x.id\nproc run() =\n  var x = Res(id: 42)\n  discard x\nrun()\nlet r = freed\n",
        expected: 42,
        expect: Expect::Runs,
        ticket: None,
    },
    // A top-level `let` whose initializer is an un-folded constant *arithmetic tree* — nimony emits it
    // as an inline `gvar` initializer `(add (mul 2 3) 36)` rather than folding to `42` or routing it
    // through the init chain (as a call-initializer like the anchors above is). svm-leng fail-closes on
    // that gvar-initializer shape. (Arithmetic in a *proc* body runs — see the anchor.) A #760 arm.
    Fixture {
        feature: "const-arith gvar initializer",
        source: "let r = 2 * 3 + 36\n",
        expected: 42,
        expect: Expect::FailsClosed(Stage::Translate),
        ticket: Some("#760 (svm-leng totality)"),
    },
];

#[test]
fn nim_conformance_matrix() {
    let Some(path) = toolchain_path() else {
        eprintln!("SKIP: nimony toolchain not found (set NIMONY_BIN/NIM_BIN or install on PATH)");
        return;
    };
    // Quiet the panic hook: the catch_unwind fail-closed paths would otherwise spew backtraces that
    // read as failures. Restored after measurement.
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let measured: Vec<(Status, Option<i64>)> = FIXTURES.iter().map(|f| measure(&path, f)).collect();
    std::panic::set_hook(prev);

    // Print the matrix.
    eprintln!("\n=== Nim → SVM conformance matrix (#956) ===");
    eprintln!("{:<34} {:<14} note", "feature", "status");
    eprintln!("{}", "-".repeat(72));
    let mut regressions = Vec::new();
    let mut improvements = Vec::new();
    for (f, (st, val)) in FIXTURES.iter().zip(&measured) {
        let status = match st {
            Status::Runs => "RUNS".to_string(),
            Status::Fails(s) => format!("fails:{s:?}").to_lowercase(),
        };
        let note = match (st, val) {
            (Status::Runs, _) => String::new(),
            (Status::Fails(Stage::Run), Some(v)) => format!("ran, got {v} (want {})", f.expected),
            (Status::Fails(_), _) => f.ticket.map(|t| t.to_string()).unwrap_or_default(),
        };
        eprintln!("{:<34} {:<14} {}", f.feature, status, note);

        let expected_runs = matches!(f.expect, Expect::Runs);
        let measured_runs = matches!(st, Status::Runs);
        if expected_runs && !measured_runs {
            regressions.push((f.feature, st.clone()));
        } else if !expected_runs && measured_runs {
            improvements.push(f.feature);
        } else if let (Expect::FailsClosed(want), Status::Fails(got)) = (&f.expect, st) {
            if want != got {
                // The feature still fails, but at a different stage than recorded — worth a nudge to
                // re-route its ticket, but not a hard failure.
                eprintln!(
                    "  note: {} now fails at {got:?}, baseline said {want:?}",
                    f.feature
                );
            }
        }
    }
    let runs = measured.iter().filter(|(s, _)| *s == Status::Runs).count();
    eprintln!("{}", "-".repeat(72));
    eprintln!(
        "{runs}/{} features run end-to-end on the SVM today\n",
        FIXTURES.len()
    );

    if std::env::var("NIM_CONFORMANCE_MEASURE").is_ok() {
        eprintln!("(measure mode: assertions skipped)");
        return;
    }
    assert!(
        regressions.is_empty(),
        "conformance REGRESSED (was expected to run, now fails): {regressions:?}"
    );
    assert!(
        improvements.is_empty(),
        "conformance IMPROVED — flip these fixtures' `expect` to `Runs` and close their sub-issues: {improvements:?}"
    );
}
