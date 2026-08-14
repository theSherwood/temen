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
                    // Lossy: a `.x.nif` may carry non-UTF-8 bytes in a string literal (the sweep hit
                    // this on `strutils`). Identity for the valid-UTF-8 milestone fixtures.
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

    const SHIM: &str = include_str!("../src/powerbox_compute_shim.svm.txt");
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

    // Order the **program module first** (the `system` module — stem `sysv…` — last), the convention
    // `link` builds on: the first unit's first proc is func 0, the natural entry, and the C `main`/init
    // chain lives in the program module. `collect_x_nif`'s directory order is filesystem-dependent, so
    // pin it here — an init-chain run through `main` is order-sensitive (a `system`-first layout mislays
    // the entry region). Stable sort keeps any multi-module program's own order.
    let mut ordered: Vec<&(String, String)> = mods.iter().collect();
    ordered.sort_by_key(|(stem, _)| stem.starts_with("sysv"));
    let units: Vec<svm_leng::WholeModule> = ordered
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

/// Run exported proc `export_substr` with `args` on both engines over the caller-provided `seed`
/// (which sets the bump-allocator cursor, the `POWERBOX_HEAP_BRK` word at window offset 32);
/// returns the i64 result and the interp's final window (so a follow-up call can continue from the
/// advanced cursor).
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
    // `osAllocPages` is the raw page source — it calls the bound `mmap`, which the shim serves from
    // the `POWERBOX_HEAP_BRK` bump cursor (window offset 32). Two calls must return the seeded heap
    // start and then one page past it — real stdlib allocation running on both engines, with the
    // `system` module sourced from the toolchain (no committed artifact).
    with_program("proc noop() = discard\nnoop()\n", |m| {
        let mut window = vec![0u8; 1 << 20];
        let brk = svm_ir::POWERBOX_HEAP_BRK as usize;
        window[brk..brk + 8].copy_from_slice(&(1i64 << 19).to_le_bytes());
        // `osAllocPages` is frame-needing under the funcref ABI — it can reach the OOM/abort path,
        // whose handler is an indirect call — so its signature is `($sp, size)`. Give it a data-stack
        // base above the seeded heap region (heap at 1<<19; this run bumps only two pages), well clear
        // of the heap and globals; `size` is the second arg.
        let sp = 0xC_0000; // 768 KiB
        let (first, after) = run_export_seeded(m, "osAllocPages.0.sysvq0asl", &[sp, 4096], &window);
        assert_eq!(first, 1 << 19, "first page is the seeded heap start");
        let (second, _) = run_export_seeded(m, "osAllocPages.0.sysvq0asl", &[sp, 4096], &after);
        assert_eq!(second, (1 << 19) + 4096, "second page bumped by one page");
    });
}

/// Run the program's C `main` — the **full init chain** (nimony's `ini`: guards, the `system`
/// module's init, the funcref-gvar static initializers the linker materialized, then the program
/// body) — on both engines, seeding the bump-allocator cursor at window offset 8, and read module
/// global `global_substr` back from each engine's final window. Asserts `main` returns 0 (runs to
/// completion) and the two engines agree (§9 parity); returns the global.
fn run_main_read_global(m: &Module, global_substr: &str) -> i64 {
    let main = m
        .exports
        .iter()
        .find(|e| e.name == "main")
        .expect("exportc main")
        .func;
    let off = m
        .data_exports
        .iter()
        .find(|e| e.name.starts_with(global_substr))
        .unwrap_or_else(|| panic!("no data export starting with `{global_substr}`"))
        .offset as usize;
    // Powerbox layout (the model svm-llvm's C on-ramp runs under): the data stack is based at
    // `powerbox_entry_sp` (page-aligned, above all globals), and the heap lives above the 1 MiB
    // stack reserve — so globals / data stack / heap are disjoint by construction and the allocator
    // can never stomp a live frame or the seq it's growing. The heap-brk word (`POWERBOX_HEAP_BRK`,
    // window offset 32) is the shim bump allocator's cursor; seed it to the heap base.
    let entry_sp = svm_ir::powerbox_entry_sp(m) as i64;
    let heap_base = entry_sp + svm_ir::POWERBOX_STACK_RESERVE as i64;
    let mut seed = vec![0u8; 1 << 20];
    let brk = svm_ir::POWERBOX_HEAP_BRK as usize;
    seed[brk..brk + 8].copy_from_slice(&heap_base.to_le_bytes());
    // C `main(argc, argv, envp)` with the leading frame `$sp` = the data-stack base; argc/argv/envp
    // are zero for a no-arg program.
    let ivals = [
        Value::I64(entry_sp),
        Value::I32(0),
        Value::I64(0),
        Value::I64(0),
    ];
    let mut fuel = 500_000_000u64;
    let (ir, imem) = svm_interp::run_capture(m, main, &ivals, &mut fuel, &seed);
    assert!(ir.is_ok(), "interp main: {ir:?}");
    let iv = i64::from_le_bytes(imem[off..off + 8].try_into().unwrap());
    let (jout, jmem) =
        svm_jit::compile_and_run_capture(m, main, &[entry_sp, 0, 0, 0], &seed).expect("jit");
    assert!(
        matches!(jout, svm_jit::JitOutcome::Returned(_)),
        "jit main: {jout:?}"
    );
    let jv = i64::from_le_bytes(jmem[off..off + 8].try_into().unwrap());
    assert_eq!(iv, jv, "§9 interp/JIT parity on the computed global");
    iv
}

#[test]
fn nim_heap_seq_program_runs_end_to_end() {
    // A real allocating program: build `@[i*i]` in a `seq[int]`, then sum it. Exercises the whole
    // stack from Nim source — the cross-module funcref-gvar calls (`oomHandler`), the whole-program
    // frame fixpoint (`alloc` takes `$sp`), the `data.funcref`-materialized funcref-gvar initializers
    // (so the init chain runs without a trap), and the real stdlib allocator over the runtime shim.
    // Entry is the C `main`, which runs the full init chain then `let r = sumSquares(4)`; we read `r`
    // back. Uses explicit index iteration (`while j < s.len: s[j]`).
    with_program(
        "proc sumSquares(n: int): int =\n\
         \x20 var s: seq[int] = @[]\n\
         \x20 var i = 0\n\
         \x20 while i < n:\n\
         \x20   s.add(i * i)\n\
         \x20   i = i + 1\n\
         \x20 result = 0\n\
         \x20 var j = 0\n\
         \x20 while j < s.len:\n\
         \x20   result = result + s[j]\n\
         \x20   j = j + 1\n\
         let r = sumSquares(4)\n",
        |m| {
            // 0*0 + 1*1 + 2*2 + 3*3 = 0 + 1 + 4 + 9 = 14.
            assert_eq!(run_main_read_global(m, "r.0."), 14, "sumSquares(4)");
        },
    );
}

#[test]
fn nim_for_in_seq_iterator_runs_end_to_end() {
    // The idiomatic `for x in s` openArray iterator — the same allocating program as above, but
    // iterating the seq with `for` instead of an explicit index. This exercises two fixes together:
    // the powerbox memory layout (data stack based at `powerbox_entry_sp`, heap above the reserve,
    // so the allocator can never stomp the seq — previously this corrupted `s.len` into a phantom
    // `+20` extra element), and unsigned-comparison lowering (the TLSF allocator's `uint32` bitmaps,
    // which a signed `le`/`lt` mis-ordered once the layout stopped masking the reallocation path).
    with_program(
        "proc sumSquares(n: int): int =\n\
         \x20 var s: seq[int] = @[]\n\
         \x20 var i = 0\n\
         \x20 while i < n:\n\
         \x20   s.add(i * i)\n\
         \x20   i = i + 1\n\
         \x20 result = 0\n\
         \x20 for x in s:\n\
         \x20   result = result + x\n\
         let r = sumSquares(5)\n",
        |m| {
            // 0 + 1 + 4 + 9 + 16 = 30 — no phantom trailing element.
            assert_eq!(
                run_main_read_global(m, "r.0."),
                30,
                "for x in s: sum of squares"
            );
        },
    );
}

#[test]
fn nim_seq_param_sum_runs_end_to_end() {
    // Pass a `seq[int]` across a function boundary: `build(n)` returns a seq, `sumSeq(a)` takes it and
    // sums it with `for x in a`. Exercises seq-as-parameter (aggregate by address + `toOpenArray`),
    // the real allocator building the seq, seq index/len, and the ARC destroy of the temporary — the
    // read + write + handoff paths the retired `real_seq_loop`/`real_seq_index` `.nif` fixtures
    // covered, now driven from Nim **source** against the real compiled `system` module.
    with_program(
        "proc build(n: int): seq[int] =\n\
         \x20 result = @[]\n\
         \x20 var i = 0\n\
         \x20 while i < n:\n\
         \x20   result.add(i)\n\
         \x20   i = i + 1\n\
         proc sumSeq(a: seq[int]): int =\n\
         \x20 result = 0\n\
         \x20 for x in a:\n\
         \x20   result = result + x\n\
         let r = sumSeq(build(4))\n",
        |m| {
            // build(4) = @[0,1,2,3]; sumSeq = 0+1+2+3 = 6.
            assert_eq!(run_main_read_global(m, "r.0."), 6, "sumSeq(build(4))");
        },
    );
}

/// **The end-to-end I/O milestone (Path B).** A real `std/syncio` program that writes to stdout —
/// compiled by the nimony toolchain, lowered and linked by svm-leng (this time *retaining* the raw
/// syscall leaves as manifest imports, `link_whole_with_runtime_manifest`), then run with `sysWrite`
/// bound to a capture buffer. It carries the whole Path-B chain — nimony → hexer → svm-leng →
/// manifest-link → run — end to end, exercising W1 (aggregate globals, variant objects, cross-module
/// sret, the `$`/string machinery), W2 (linking), and W3 (the syscall seam + the `$sp`-carrying
/// funcref ABI the at-exit stdout flush registered via `setExitFlush` depends on). Interp only: the
/// host-proc import binding is the tree-walker's path; the pure-compute e2e tests above cover §9
/// interp/JIT parity.
#[test]
fn real_write_program_prints_to_stdout() {
    let Some(path) = toolchain_path() else {
        eprintln!("SKIP: nimony toolchain not found (set NIMONY_BIN/NIM_BIN or install on PATH)");
        return;
    };
    let mods = compile_to_leng(
        &path,
        "import std/syncio\nwrite(stdout, \"hello, svm\\n\")\n",
    );
    assert_eq!(run_io_program(&mods), b"hello, svm\n", "captured stdout");
}

/// **The browser-shape I/O path.** Same real `std/syncio` program, but linked via
/// `svm_leng::link_nim_powerbox` — the nim→powerbox bottom-edge bridge — and run under the **standard**
/// `svm_run::run_powerbox`, the exact engine the browser's `svm_run_onramp` wraps. Unlike
/// `real_write_program_prints_to_stdout` (which hand-wires each nim syscall to a `svm_posix` op), here
/// the host binds only its own §3e powerbox caps: the bridge shims the compute leaves in and adapts
/// nimony's `sysWrite(fd,buf,len)` to the STREAM `write(buf,len)` cap, so a real Nim program prints to
/// the powerbox stdout with no custom personality. This is what makes a "run real Nim" playground card
/// possible.
#[test]
fn nim_write_runs_under_the_powerbox() {
    let Some(path) = toolchain_path() else {
        eprintln!("SKIP: nimony toolchain not found (set NIMONY_BIN/NIM_BIN or install on PATH)");
        return;
    };
    let mods = compile_to_leng(
        &path,
        "import std/syncio\nwrite(stdout, \"hello, svm\\n\")\n",
    );
    let units: Vec<svm_leng::WholeModule> = mods
        .iter()
        .map(|(stem, src)| svm_leng::WholeModule { stem, src })
        .collect();
    let m = svm_leng::link_nim_powerbox(&units).unwrap_or_else(|e| panic!("bridge link: {e}"));
    let run = svm_run::run_powerbox(&m, &[]).unwrap_or_else(|e| panic!("run_powerbox: {e}"));
    assert_eq!(
        run.stdout, b"hello, svm\n",
        "real Nim printed via the standard powerbox"
    );
}

/// Manifest-link `mods` (retaining `write`/`read`/`_exit`/… as bindable imports), then run the
/// `exportc` `main` on **both engines** with `sysWrite` bound to a stdout capture and the other
/// syscall leaves stubbed, asserting the two capture the same bytes (§9 interp/JIT parity on the
/// syscall seam). `main` is `($sp, argc, argv, envp) -> cint`, so it receives a data-stack base
/// above the globals; argc/argv/envp are zero (this program reads none). Returns the captured
/// stdout bytes.
fn run_io_program(mods: &[(String, String)]) -> Vec<u8> {
    // Bind the pure-compute bottom edge to the shim (as `link_with_runtime`), but via the *manifest*
    // link so the raw-syscall leaves survive as host-bound imports rather than fail-closing.
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
    // Link with a synthesized **powerbox `_start`** at function 0: it reads the post-link data-stack
    // base (`data.top` → `powerbox_entry_sp`, page-aligned above the globals) and calls the C-shaped
    // `main($sp, argc, argv, envp)` with `argc/argv/envp = 0` — a real powerbox entry, not a
    // hand-provided `$sp`. Injecting `_start` as a first link unit (rather than prepending it after
    // the link) keeps the program's `data.funcref` gvar initializers valid — the funcref-carrying
    // at-exit flush this very program registers would otherwise dispatch through a stale, off-by-one
    // index.
    let m = svm_leng::link_whole_powerbox_manifest(&units, vec![runtime])
        .unwrap_or_else(|e| panic!("powerbox manifest link: {e}"));
    svm_verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    // The merged module carries the §3e powerbox entry shape: a paramless `_start` at function 0
    // exported by name — exactly what `run_powerbox`/`instantiate` bind manifest slots against.
    assert!(
        svm_run::is_named_powerbox_entry(&m),
        "linked module is a powerbox entry (paramless func-0 `_start`)"
    );

    // Instantiate the manifest module through the reference embedding and run `_start` on both
    // engines under the **POSIX personality** — the raw-syscall leaves (`sysWrite.0.` …) bind by name
    // to `svm_posix`'s `write`/`read`/`open`/`close`/`lseek` ops, whose `write(fd, buf, n)` ABI is the
    // exact nimony bottom edge. This is the real load-a-manifest-module, host-grants-the-caps path
    // (`instantiate_with_imports` → `Instance::run`), not a hand-wired host-binding harness. Each
    // engine gets its own fresh personality (separate fd table + stdout buffer); a divergence in the
    // two captured streams is a real engine bug on the W3 syscall seam.
    let interp_out = run_io_capture(&m, svm_run::Backend::TreeWalk);
    let jit_out = run_io_capture(&m, svm_run::Backend::Jit);
    assert_eq!(
        interp_out, jit_out,
        "§9 interp/JIT parity on the POSIX syscall I/O seam"
    );
    interp_out
}

/// Map a retained nimony syscall import name to the POSIX-personality op it binds to. The nimony
/// bottom edge (`sysWrite {.importc: "write".}` …) is spelled by the *nim* symbol (`sysWrite.0.`),
/// not the C name, so the powerbox's by-C-name resolver doesn't reach it — this is the small
/// nimony→personality name map that lets the retained leaves bind to `svm_posix`'s fd-based ops
/// (whose signatures match the nim ABI exactly).
fn nim_posix_op(name: &str) -> u32 {
    if name.starts_with("sysWrite") {
        svm_posix::OP_WRITE
    } else if name.starts_with("sysRead") {
        svm_posix::OP_READ
    } else if name.starts_with("sysOpen") {
        svm_posix::OP_OPEN
    } else if name.starts_with("sysClose") {
        svm_posix::OP_CLOSE
    } else if name.starts_with("sysLseek") {
        svm_posix::OP_LSEEK
    } else {
        panic!("unmapped nimony syscall import `{name}` — extend nim_posix_op");
    }
}

/// Run the linked I/O program's powerbox `_start` (function 0) on `backend` through the reference
/// embedding (`Instance`), with every retained nim-name syscall import bound to a single shared
/// **POSIX personality**. Returns the bytes the guest `write`-to-fd-1'd (`Posix::stdout`). The
/// program uses no posix `malloc`/`mmap` (its own runtime shim serves those), so the personality's
/// heap arena is unused and passed empty.
fn run_io_capture(m: &Module, backend: svm_run::Backend) -> Vec<u8> {
    // One personality shared across every bound name (one fd table, one stdout buffer): the factory
    // closes over a single `inner`, so each per-name grant re-mints a handler over the same state.
    // `svm_posix::cap` hands back an opaque `impl Fn` (no `Clone`), so wrap it in an `Arc` and share
    // that across the per-name grant closures.
    let (posix, make) = svm_posix::cap(0, 0, Vec::new());
    let make = std::sync::Arc::new(make);
    let mut imports = svm_run::Imports::new();
    for imp in &m.imports {
        let make = std::sync::Arc::clone(&make);
        imports = imports.provide(
            imp.name.clone(),
            svm_run::HostCap::host_proc(nim_posix_op(&imp.name), move || (*make)()),
        );
    }
    let inst = svm_run::instantiate_with_imports(m.clone(), imports)
        .unwrap_or_else(|e| panic!("instantiate manifest module: {e}"));
    inst.run(backend, &svm_run::RunConfig::default())
        .unwrap_or_else(|e| panic!("run `_start` on {backend:?}: {e}"));
    posix.stdout()
}

/// Richer end-to-end I/O over the same chain — output that actually *formats*, exercising the
/// cross-module sret `$`(int)→string conversion and the `$sp`-carrying funcref ABI (the at-exit
/// flush) on top of W1/W2/W3: an `$`-formatted integer, a loop emitting one conversion per iteration,
/// and a `writeLine` pair. Same manifest-link + `sysWrite`-capture harness as the hello case.
#[test]
fn real_formatted_output_runs_end_to_end() {
    let Some(path) = toolchain_path() else {
        eprintln!("SKIP: nimony toolchain not found (set NIMONY_BIN/NIM_BIN or install on PATH)");
        return;
    };
    assert_eq!(
        run_io_program(&compile_to_leng(
            &path,
            "import std/syncio\nwrite(stdout, $(20 + 22))\n"
        )),
        b"42",
        "`$`(int)->string then write"
    );
    assert_eq!(
        run_io_program(&compile_to_leng(
            &path,
            "import std/syncio\nfor i in 0..2: write(stdout, $i)\n"
        )),
        b"012",
        "loop with a per-iteration `$` conversion"
    );
    assert_eq!(
        run_io_program(&compile_to_leng(
            &path,
            "import std/syncio\nwriteLine(stdout, \"line one\")\nwriteLine(stdout, \"line two\")\n"
        )),
        b"line one\nline two\n",
        "two writeLine calls"
    );
}

// ---------------------------------------------------------------------------------------------
// #760 W1 totality sweep (diagnostic, gated on NIM_SWEEP=1). Compiles a corpus of real Nim through
// the toolchain, links each program whole, and tallies the `Unsupported`/`Malformed` reasons —
// the ranked worklist for closing residual Leng totality to nimony scale. Not a CI gate.
// ---------------------------------------------------------------------------------------------

/// Like `compile_to_leng` but never panics: returns `Err(stderr)` if `nimony c` fails, so the sweep
/// can distinguish a frontend failure from a translator gap.
fn try_compile_to_leng(
    nim_path: &str,
    name: &str,
    source: &str,
) -> Result<Vec<(String, String)>, String> {
    let dir = std::env::temp_dir().join(format!("svm_nim_sweep_{}_{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    std::fs::write(dir.join("prog.nim"), source).map_err(|e| e.to_string())?;
    let out = Command::new("nimony")
        .args(["c", "--isMain", "prog.nim"])
        .current_dir(&dir)
        .env("PATH", nim_path)
        .output()
        .map_err(|e| e.to_string())?;
    if !out.status.success() {
        let msg = String::from_utf8_lossy(&out.stderr).to_string();
        let _ = std::fs::remove_dir_all(&dir);
        return Err(format!(
            "nimony: {}",
            msg.lines().rev().take(3).collect::<Vec<_>>().join(" | ")
        ));
    }
    let mut mods = Vec::new();
    collect_x_nif(&dir.join("nimcache"), &mut mods);
    let _ = std::fs::remove_dir_all(&dir);
    Ok(mods)
}

/// Collapse a translator error message to a stable bucket key: strip backtick-quoted specifics and
/// parenthetical detail so `named type `foo.0.`` and `named type `bar.1.`` land in one bucket.
fn bucket(msg: &str) -> String {
    let mut out = String::new();
    let mut in_tick = false;
    for c in msg.chars() {
        match c {
            '`' => {
                in_tick = !in_tick;
                if in_tick {
                    out.push_str("`…`");
                }
            }
            _ if in_tick => {}
            '0'..='9' => out.push('#'),
            _ => out.push(c),
        }
    }
    out.split(" (").next().unwrap_or(&out).trim().to_string()
}

#[test]
fn totality_sweep_760() {
    if std::env::var("NIM_SWEEP").is_err() {
        eprintln!("SKIP totality_sweep_760 (set NIM_SWEEP=1 to run)");
        return;
    }
    let path = toolchain_path().expect("toolchain required for the sweep");
    // Corpus: nimony's own examples + a batch of feature-probe programs. Extendable via NIM_SWEEP_DIR
    // (a directory of .nim files, each compiled --isMain).
    let mut corpus: Vec<(String, String)> = Vec::new();
    if let Ok(dir) = std::env::var("NIM_SWEEP_DIR") {
        for e in std::fs::read_dir(&dir).expect("sweep dir").flatten() {
            let p = e.path();
            if p.extension().and_then(|s| s.to_str()) == Some("nim") {
                let name = p.file_stem().unwrap().to_string_lossy().to_string();
                corpus.push((name, std::fs::read_to_string(&p).unwrap()));
            }
        }
    }
    // Built-in feature probes (always run).
    for (n, s) in FEATURE_PROBES {
        corpus.push((n.to_string(), s.to_string()));
    }
    // Stdlib import-drivers: a program that `import`s each named module pulls it + its deps through
    // the toolchain, exercising the translator at real stdlib scale. Set NIM_SWEEP_STD=1 to include.
    if std::env::var("NIM_SWEEP_STD").is_ok() {
        for m in STD_MODULES {
            corpus.push((format!("std_{m}"), format!("import std/{m}\n")));
        }
    }

    use std::collections::BTreeMap;
    let mut buckets: BTreeMap<String, (usize, String)> = BTreeMap::new(); // key -> (count, example prog)
    let (mut ok, mut nim_fail, mut link_ok) = (0usize, 0usize, 0usize);
    let total = corpus.len();
    for (i, (name, src)) in corpus.iter().enumerate() {
        eprintln!("[{}/{total}] {name} …", i + 1);
        match try_compile_to_leng(&path, name, src) {
            Err(e) => {
                nim_fail += 1;
                eprintln!("[nimony-fail] {name}: {e}");
            }
            Ok(mods) => {
                let units: Vec<svm_leng::WholeModule> = mods
                    .iter()
                    .map(|(stem, src)| svm_leng::WholeModule { stem, src })
                    .collect();
                // Manifest link with an *empty* runtime: every unresolved bottom-edge leaf is retained
                // as a manifest import instead of fail-closing the link, so only genuine translator
                // `Unsupported` gaps (which surface during translate, before link) are reported — not
                // "no unit exports `write`" noise.
                match svm_leng::link_whole_with_runtime_manifest(&units, Vec::new()) {
                    Ok(_) => {
                        ok += 1;
                        link_ok += 1;
                    }
                    Err(err) => {
                        eprintln!("[gap] {name}: {err}");
                        let (kind, msg) = match &err {
                            svm_leng::LengError::Unsupported(m) => ("Unsupported", m.clone()),
                            svm_leng::LengError::Malformed(m) => ("Malformed", m.clone()),
                            svm_leng::LengError::Parse(m) => ("Parse", m.clone()),
                        };
                        let key = format!("[{kind}] {}", bucket(&msg));
                        let e = buckets.entry(key).or_insert((0, name.clone()));
                        e.0 += 1;
                    }
                }
            }
        }
    }
    let mut ranked: Vec<_> = buckets.into_iter().collect();
    ranked.sort_by_key(|b| std::cmp::Reverse(b.1 .0));
    eprintln!(
        "\n===== #760 totality sweep: {} programs, {ok} translated, {nim_fail} nimony-fail =====",
        corpus.len()
    );
    eprintln!("(link_ok={link_ok})");
    for (key, (count, example)) in &ranked {
        eprintln!("{count:>3}  {key}   (e.g. {example})");
    }
    eprintln!("===== end sweep =====\n");
}

/// Feature-probe programs — one construct family each, to surface translator gaps without needing a
/// whole stdlib module. Kept small and self-contained.
const FEATURE_PROBES: &[(&str, &str)] = &[
    ("echo_int", "echo 42\n"),
    ("string_concat", "import std/syncio\nlet s = \"a\" & \"b\"\nwrite(stdout, s)\n"),
    ("seq_map", "import std/syncio\nvar s = @[1,2,3]\nvar t = 0\nfor x in s: t += x\nwrite(stdout, $t)\n"),
    ("object_variant", "type K = enum ka, kb\ntype N = object\n  case k: K\n  of ka: a: int\n  of kb: b: int\nvar n = N(k: ka, a: 5)\n"),
    ("closure", "proc mk(): proc(): int =\n  var c = 0\n  result = proc(): int =\n    c += 1\n    c\nlet f = mk()\ndiscard f()\n"),
    ("exceptions", "import std/syncio\ntry:\n  raise newException(ValueError, \"x\")\nexcept ValueError:\n  write(stdout, \"caught\")\n"),
    ("generic_proc", "proc id[T](x: T): T = x\nlet a = id(3)\nlet b = id(\"s\")\n"),
    ("float_math", "import std/syncio\nimport std/math\nlet x = sqrt(2.0)\nwrite(stdout, $x)\n"),
];

/// Stdlib modules the sweep exercises via `import std/<m>` drivers (NIM_SWEEP_STD=1). A spread across
/// strings, containers, numerics, and parsing — the constructs a real program (and nimony itself) hit.
const STD_MODULES: &[&str] = &[
    "strutils",
    "sequtils",
    "algorithm",
    "math",
    "tables",
    "sets",
    "hashes",
    "options",
    "deques",
    "heapqueue",
    "intsets",
    "bitops",
    "parseutils",
    "strformat",
    "unicode",
    "times",
    "json",
    "base64",
    "editdistance",
    "md5",
    "monotimes",
    "complex",
    "rationals",
    "assertions",
    "typetraits",
    "enumutils",
    "sugar",
    "setutils",
    "lists",
    "packedsets",
];
