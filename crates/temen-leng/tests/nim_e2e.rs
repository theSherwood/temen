//! **Tier-2 end-to-end: real Nim source runs on Temen.** Unlike the translator-unit tests (small
//! hand-written Leng snippets), these start from *Nim source* and drive the whole real toolchain —
//! `nimony c` (→ nifler → nimony → hexer) emits the program's and the `system` module's Leng
//! (`.x.nif`), `temen-leng` lowers and links them together with the W3 runtime shim, and the result
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
use temen_interp::Value;
use temen_ir::Module;

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
/// stem is the `.x.nif` basename, exactly the qualifier `temen-leng` links symbols under.
fn compile_to_leng(nim_path: &str, source: &str) -> Vec<(String, String)> {
    // A per-source directory (hash of the program) so tests running in parallel never share a
    // `nimcache` — each `nimony c` gets its own throwaway tree.
    let tag = {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        source.hash(&mut h);
        h.finish()
    };
    let dir = std::env::temp_dir().join(format!("temen_nim_e2e_{}_{tag:016x}", std::process::id()));
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
            // `<scratch>_d` is a **macro plugin's own sub-build** (`semos.buildPlugin`: a plugin is
            // a separate executable Nimony compiles at compile time, in its own cache directory, with
            // its own C `main`). Those modules belong to a different program, not this one — sweeping
            // them in is how `import std/macros` started failing with `DuplicateSymbol("main")` under
            // v0.6.2, which routes `parsegen`/`regex` through plugins. The sibling `_v` is the
            // validator's sem-only run of the same source; skip it too.
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name.ends_with("_d") || name.ends_with("_v") {
                continue;
            }
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
/// Temen module. The shim's exports (its 20 functions) are bound to whatever bottom-edge C imports the
/// modules actually reference — discovered from each module's compiled object, so the mangled atomic
/// symbol names never have to be hard-coded.
fn link_with_runtime(mods: &[(String, String)]) -> Module {
    // Order the **program module first** (the `system` module — stem `sysv…` — last), the convention
    // `link` builds on: the first unit's first proc is func 0, the natural entry, and the C `main`/init
    // chain lives in the program module. `collect_x_nif`'s directory order is filesystem-dependent, so
    // pin it here — an init-chain run through `main` is order-sensitive (a `system`-first layout mislays
    // the entry region). Stable sort keeps any multi-module program's own order.
    let mut ordered: Vec<&(String, String)> = mods.iter().collect();
    ordered.sort_by_key(|(stem, _)| stem.starts_with("sysv"));
    let units: Vec<temen_leng::WholeModule> = ordered
        .iter()
        .map(|(stem, src)| temen_leng::WholeModule { stem, src })
        .collect();
    // The compute-shim link unit comes from the library's own leaf table — one route through the
    // bottom edge. A test-local copy of that table went stale the moment a leaf was added.
    let runtime = temen_leng::nim_compute_shim_unit(&units).expect("compute shim unit");
    let m = temen_leng::link_whole_with_runtime(&units, vec![runtime])
        .unwrap_or_else(|e| panic!("link with runtime: {e}"));
    temen_verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
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
    let (ir, _) = temen_interp::run_capture(m, idx, &ivals, &mut fuel, &seed);
    let iword = match ir.expect("interp").as_slice() {
        [Value::I64(n)] => *n,
        o => panic!("unexpected {o:?}"),
    };
    let (jout, _) = temen_jit::compile_and_run_capture(m, idx, args, &seed).expect("jit");
    let jword = match jout {
        temen_jit::JitOutcome::Returned(v) => v,
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
/// (which sets the bump-allocator cursor, the `POWERBOX_HEAP_BRK` word — under the #964 guard the
/// shim reads it at `POWERBOX_NULL_GUARD + POWERBOX_HEAP_BRK`, the guard's scratch page);
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
    let (ir, imem) = temen_interp::run_capture(m, idx, &ivals, &mut fuel, seed);
    let iword = match ir.expect("interp").as_slice() {
        [Value::I64(n)] => *n,
        o => panic!("unexpected {o:?}"),
    };
    let (jout, _) = temen_jit::compile_and_run_capture(m, idx, args, seed).expect("jit");
    let jword = match jout {
        temen_jit::JitOutcome::Returned(v) => v,
        o => panic!("jit: {o:?}"),
    };
    assert_eq!(vec![iword], jword, "§9 interp/JIT parity");
    (iword, imem)
}

#[test]
fn nim_addtwo_runs_on_temen() {
    with_program(
        "proc addTwo(a, b: int): int = a + b\nlet r = addTwo(2, 3)\n",
        |m| {
            assert_eq!(run_export(m, "addTwo", &[2, 3]), 5);
            assert_eq!(run_export(m, "addTwo", &[40, 2]), 42);
        },
    );
}

#[test]
fn nim_arithmetic_and_control_flow_runs_on_temen() {
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
    // the `POWERBOX_HEAP_BRK` bump cursor. Under the #964 NULL guard the shim reads that cursor one
    // guard up, at `POWERBOX_NULL_GUARD + POWERBOX_HEAP_BRK` (the guard's scratch page, #1091), so
    // seed it there. Two calls must return the seeded heap start and then one page past it — real
    // stdlib allocation running on both engines, with the `system` module sourced from the toolchain.
    with_program("proc noop() = discard\nnoop()\n", |m| {
        let mut window = vec![0u8; 1 << 20];
        let brk = (temen_ir::POWERBOX_NULL_GUARD + temen_ir::POWERBOX_HEAP_BRK) as usize;
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
    // Powerbox layout (the model temen-llvm's C on-ramp runs under): the data stack is based at
    // `powerbox_entry_sp` (page-aligned, above all globals), and the heap lives above the 1 MiB
    // stack reserve — so globals / data stack / heap are disjoint by construction and the allocator
    // can never stomp a live frame or the seq it's growing. The heap-brk word is the shim bump
    // allocator's cursor; under the #964 guard the shim reads it at `POWERBOX_NULL_GUARD +
    // POWERBOX_HEAP_BRK` (the guard's scratch page, #1091), so seed it there to the heap base.
    let entry_sp = temen_ir::powerbox_entry_sp(m) as i64;
    let heap_base = entry_sp + temen_ir::POWERBOX_STACK_RESERVE as i64;
    let mut seed = vec![0u8; 1 << 20];
    let brk = (temen_ir::POWERBOX_NULL_GUARD + temen_ir::POWERBOX_HEAP_BRK) as usize;
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
    let (ir, imem) = temen_interp::run_capture(m, main, &ivals, &mut fuel, &seed);
    assert!(ir.is_ok(), "interp main: {ir:?}");
    let iv = i64::from_le_bytes(imem[off..off + 8].try_into().unwrap());
    let (jout, jmem) =
        temen_jit::compile_and_run_capture(m, main, &[entry_sp, 0, 0, 0], &seed).expect("jit");
    assert!(
        matches!(jout, temen_jit::JitOutcome::Returned(_)),
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
/// compiled by the nimony toolchain, lowered and linked by temen-leng (this time *retaining* the raw
/// syscall leaves as manifest imports, `link_whole_with_runtime_manifest`), then run with `sysWrite`
/// bound to a capture buffer. It carries the whole Path-B chain — nimony → hexer → temen-leng →
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
        "import std/syncio\nwrite(stdout, \"hello, temen\\n\")\n",
    );
    assert_eq!(run_io_program(&mods), b"hello, temen\n", "captured stdout");
}

/// **The browser-shape I/O path.** Same real `std/syncio` program, but linked via
/// `temen_leng::link_nim_powerbox` — the nim→powerbox bottom-edge bridge — and run under the **standard**
/// `temen_run::run_powerbox`, the exact engine the browser's `temen_run_onramp` wraps. Unlike
/// `real_write_program_prints_to_stdout` (which hand-wires each nim syscall to a `temen_posix` op), here
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
        "import std/syncio\nwrite(stdout, \"hello, temen\\n\")\n",
    );
    let units: Vec<temen_leng::WholeModule> = mods
        .iter()
        .map(|(stem, src)| temen_leng::WholeModule { stem, src })
        .collect();
    let m =
        temen_leng::link_nim_powerbox(&units, None).unwrap_or_else(|e| panic!("bridge link: {e}"));
    let run = temen_run::run_powerbox(&m, &[]).unwrap_or_else(|e| panic!("run_powerbox: {e}"));
    assert_eq!(
        run.stdout, b"hello, temen\n",
        "real Nim printed via the standard powerbox"
    );
}

/// **#763 — the guest can read its own command line.** The synthesized powerbox `_start`
/// ([`temen_leng`'s `synth_start_unit`]) used to hand `main` a literal `argc = 0` and the fixed
/// empty `argv`/`envp` vectors, so *every* nim program on this route saw no arguments however the
/// host was invoked: nimony's `std/cmdline` reads the `cmdCount`/`cmdLine` globals the generated
/// `main` parks its parameters in, so `paramCount()` returned `-1` and `paramStr(i)` returned `""`.
/// That silently capped the route at argv-free programs — a real tool (`nifler2 parse in out`) does
/// nothing without one.
///
/// `_start` now parses the §3e args buffer the host seeds at `module_args_base` into real
/// `argv[]`/`envp[]` arrays. The gate is end-to-end through the standard powerbox: hand
/// `run_powerbox_cfg` an argv and assert the guest prints back what it was given, `argv[0]`
/// included. `getEnv` covers the `envp[]` half — the same walk (#1422) that must never see a NULL
/// vector pointer.
#[test]
fn nim_reads_its_argv_and_env_under_the_powerbox() {
    let Some(path) = toolchain_path() else {
        eprintln!("SKIP: nimony toolchain not found (set NIMONY_BIN/NIM_BIN or install on PATH)");
        return;
    };
    let mods = compile_to_leng(
        &path,
        "import std/syncio\nimport std/cmdline\nimport std/envvars\n\
         write(stdout, $paramCount())\n\
         for i in 0..paramCount():\n\x20 write(stdout, \"|\")\n\x20 write(stdout, paramStr(i))\n\
         write(stdout, \"|\")\nwrite(stdout, getEnv(\"TEMEN_ARGV_PROBE\"))\n",
    );
    let units: Vec<temen_leng::WholeModule> = mods
        .iter()
        .map(|(stem, src)| temen_leng::WholeModule { stem, src })
        .collect();
    let m =
        temen_leng::link_nim_powerbox(&units, None).unwrap_or_else(|e| panic!("bridge link: {e}"));
    let run = temen_run::run_powerbox_cfg(
        &m,
        &[],
        &[
            b"prog".as_slice(),
            b"parse".as_slice(),
            b"/in.nim".as_slice(),
        ],
        &[b"TEMEN_ARGV_PROBE=seen".as_slice()],
        None,
        temen_run::Quota::default(),
    )
    .unwrap_or_else(|e| panic!("run_powerbox_cfg: {e}"));
    // `paramCount()` is `argc - 1` (nim excludes `argv[0]`), and `paramStr(0)` is the program name.
    assert_eq!(
        String::from_utf8_lossy(&run.stdout),
        "2|prog|parse|/in.nim|seen",
        "the guest sees the argv and envp the host seeded"
    );
}

/// An argv-free run must stay exactly as it was: the host seeds no buffer, `_start` reads the zeroed
/// region back as `argc = envc = 0`, and builds the one-entry NULL-terminated vectors by the same
/// code path. This is why there is no second, no-args entry to keep in step (INVARIANTS #15).
#[test]
fn nim_with_no_argv_sees_an_empty_command_line() {
    let Some(path) = toolchain_path() else {
        eprintln!("SKIP: nimony toolchain not found (set NIMONY_BIN/NIM_BIN or install on PATH)");
        return;
    };
    let mods = compile_to_leng(
        &path,
        "import std/syncio\nimport std/cmdline\nwrite(stdout, $paramCount())\n",
    );
    let units: Vec<temen_leng::WholeModule> = mods
        .iter()
        .map(|(stem, src)| temen_leng::WholeModule { stem, src })
        .collect();
    let m =
        temen_leng::link_nim_powerbox(&units, None).unwrap_or_else(|e| panic!("bridge link: {e}"));
    let run = temen_run::run_powerbox(&m, &[]).unwrap_or_else(|e| panic!("run_powerbox: {e}"));
    assert_eq!(run.stdout, b"-1", "`argc = 0` ⇒ nim's `paramCount()` is -1");
}

/// **#1400 — a cross-module call widens narrow args to the callee's param type.** `std/strutils`
/// (like `std/unicode` and `std/times`) has a proc that passes a narrow (`i32`) result to a
/// cross-module callee whose real parameter is `int` (`i64`). Pre-fix, `call_import` derived the
/// import signature from the *argument* types and never coerced them, so the unwidened `i32`
/// surfaced only after the link as a verify `TypeMismatch` (the module fails to compile in the
/// browser). The linker now pools each proc's declared param types ([`export_proc_params`]) and
/// `call_import` coerces every scalar arg to its param type — exactly as the local `call_arg` path
/// already did — so the linked module verifies. A bare `import std/strutils` pulls in the offending
/// proc; the whole-program link + verify is the gate.
#[test]
fn nim_strutils_cross_module_arg_widths_verify() {
    let Some(path) = toolchain_path() else {
        eprintln!("SKIP: nimony toolchain not found (set NIMONY_BIN/NIM_BIN or install on PATH)");
        return;
    };
    let mods = compile_to_leng(&path, "import std/strutils\n");
    let units: Vec<temen_leng::WholeModule> = mods
        .iter()
        .map(|(stem, src)| temen_leng::WholeModule { stem, src })
        .collect();
    let m =
        temen_leng::link_nim_powerbox(&units, None).unwrap_or_else(|e| panic!("bridge link: {e}"));
    temen_verify::verify_module(&m)
        .unwrap_or_else(|e| panic!("strutils linked module must verify (#1400): {e:?}"));
}

/// Compile `import std/<module>`, link it against the nim runtime, and verify — the shape both
/// cross-module width regressions ([`nim_json_cross_module_return_widths_verify`],
/// [`nim_frame_needing_module_init_verify`]) gate on. Skips cleanly without the toolchain.
fn link_verify_std_module(module: &str) {
    let Some(path) = toolchain_path() else {
        eprintln!("SKIP: nimony toolchain not found (set NIMONY_BIN/NIM_BIN or install on PATH)");
        return;
    };
    let mods = compile_to_leng(&path, &format!("import std/{module}\n"));
    let units: Vec<temen_leng::WholeModule> = mods
        .iter()
        .map(|(stem, src)| temen_leng::WholeModule { stem, src })
        .collect();
    let m = temen_leng::link_nim_powerbox(&units, None)
        .unwrap_or_else(|e| panic!("bridge link `{module}`: {e}"));
    temen_verify::verify_module(&m)
        .unwrap_or_else(|e| panic!("`{module}` linked module must verify: {e:?}"));
}

/// **#1498 — an *aggregate-returning* cross-module call coerces its args too.** `std/nifreader`'s
/// `openFromBuffer` calls `std/vfs`'s `initBlob`, which returns a `VfsBlob` (so it is an **sret**
/// callee) and whose last param is a defaulted `cleanup: proc {.nimcall.}` — an `i32` funcref. hexer
/// expands the default to a bare `(nil)`, which lowers as a pointer-width `i64` null.
///
/// #1400 taught the non-sret import path to coerce each arg to the callee's declared param type, but
/// the sret path was a **separate copy of the same loop** and never got it, so this `i64` reached the
/// `i32` param and the module failed to verify post-link with
/// `TypeMismatch { expected: I32, found: I64 }`. Both paths now share one `marshal_import_arg`.
///
/// `macros` and `nifply` are the two stdlib modules that reach this call; both were invisible until
/// #1490 fixed the sweep's module list, and both are gated here.
#[test]
fn nim_sret_cross_module_arg_widths_verify() {
    link_verify_std_module("macros");
    link_verify_std_module("nifply");
}

/// **#1404 — a cross-module call widens its result to the callee's real return type.** `std/json`'s
/// `getTok` calls `system.equalStrings`, which returns `bool` (`i32`), and uses the result in an
/// `i64` comparison. Pre-fix, `call_import` declared the import's *return* from the call site's
/// expected type (`i64`), but the linker binds the import to the real proc by name without
/// reconciling the return width — so after the link a narrow (`i32`) result sat in a wide (`i64`)
/// slot and the module failed to verify (`TypeMismatch`). The linker now pools each proc's real
/// return type ([`export_proc_params`]) and `call_import` declares the import with it and coerces the
/// result to the call site's expected type — the return-side twin of the #1400 arg fix.
#[test]
fn nim_json_cross_module_return_widths_verify() {
    link_verify_std_module("json");
}

/// **#1405 — a frame-needing module-init is pooled so cross-module init calls pass its `$sp`.** A
/// module whose top-level code uses a cross-module aggregate (`std/md5`, `std/monotimes`) compiles
/// its `ini` proc frame-needing (a hidden leading `$sp`), and `std/editdistance` calls a
/// frame-needing `unicode.size`. Pre-fix, the linker's frame fixpoint pre-scan
/// ([`proc_frame_nodes`]) ran without the pooled cross-module type layouts, so `proc_needs_frame`
/// under-reported the frame — the callee was emitted with a `$sp` the caller never passed
/// (`CallArgCountMismatch`). The pre-scan now imports the pooled types, exactly as the real
/// translation does, so its frame-need matches and the `$sp` is threaded.
#[test]
fn nim_frame_needing_module_init_verify() {
    link_verify_std_module("md5");
    link_verify_std_module("monotimes");
    link_verify_std_module("editdistance");
}

/// **#1054 — the nim→powerbox link is unit-order-independent.** A program with a `LongString` const
/// (a string literal ≥ 8 bytes, past hexer's small-string optimization) linked against the nim
/// runtime must print the same correct bytes **no matter what order the whole units are linked in**.
/// This exercises [`temen_leng::link_whole_powerbox_manifest`] *directly* (no `system`-first
/// reorder), so it pins the real fix — the guest heap seed in the synthesized `_start`
/// ([`temen_leng`]'s `synth_start_unit`), which places the allocator arena above all placed data.
/// Pre-fix, the arena started at 0 and overlapped static data: a heap allocation could reuse the
/// address of the program's `"the Temen"` const, and `add`/realloc then scribbled its length so
/// `write` dumped ~4 KiB of stray bytes — but *only* for orders that placed the const where an
/// allocation landed, which is why it looked like the linker was order-sensitive (it is not).
#[test]
fn nim_powerbox_link_is_unit_order_independent() {
    let Some(path) = toolchain_path() else {
        eprintln!("SKIP: nimony toolchain not found (set NIMONY_BIN/NIM_BIN or install on PATH)");
        return;
    };
    let mods = compile_to_leng(
        &path,
        "import std/syncio\n\nproc greet(name: string): string =\n  \"hello, \" & name & \"\\n\"\n\nwrite(stdout, greet(\"Nim\"))\nwrite(stdout, greet(\"the Temen\"))\n",
    );
    let units: Vec<temen_leng::WholeModule> = mods
        .iter()
        .map(|(stem, src)| temen_leng::WholeModule { stem, src })
        .collect();
    // The runtime ([compute shim, syscall adapter]) is order-independent — build it once.
    let runtime = temen_leng::nim_powerbox_runtime(&units).expect("build nim runtime");
    let expected: &[u8] = b"hello, Nim\nhello, the Temen\n";

    // Cover both extremes and a reversal: `system`-first (the previously-working order),
    // `system`-last (the previously-corrupting order), and the collected order reversed.
    let n = units.len();
    let sys_first = {
        let mut v: Vec<usize> = (0..n).collect();
        v.sort_by_key(|&i| !units[i].stem.starts_with("sysv"));
        v
    };
    let sys_last = {
        let mut v: Vec<usize> = (0..n).collect();
        v.sort_by_key(|&i| units[i].stem.starts_with("sysv"));
        v
    };
    let reversed: Vec<usize> = (0..n).rev().collect();

    for order in [&sys_first, &sys_last, &reversed] {
        let ordered: Vec<temen_leng::WholeModule> = order
            .iter()
            .map(|&i| temen_leng::WholeModule {
                stem: units[i].stem,
                src: units[i].src,
            })
            .collect();
        let m = temen_leng::link_whole_powerbox_manifest(&ordered, runtime.clone())
            .unwrap_or_else(|e| panic!("link (order {order:?}): {e}"));
        temen_verify::verify_module(&m)
            .unwrap_or_else(|e| panic!("verify (order {order:?}): {e:?}"));
        let run = temen_run::run_powerbox(&m, &[])
            .unwrap_or_else(|e| panic!("run_powerbox (order {order:?}): {e}"));
        assert_eq!(
            run.stdout, expected,
            "stdout must be correct for unit order {order:?} (heap must not overlap static data)"
        );
    }
}

/// #1060: the guest heap words are baked into the linked powerbox module's data image with the
/// **break** just above the data stack and the **ceiling** at the real window top (`1 << size_log2`),
/// so the heap spans the whole remaining window and the compute-shim `mmap` can fail closed past it —
/// independent of any runtime unit's `memory N` declaration.
#[test]
fn nim_powerbox_seeds_heap_words_to_window_top() {
    let Some(path) = toolchain_path() else {
        eprintln!("SKIP: nimony toolchain not found (set NIMONY_BIN/NIM_BIN or install on PATH)");
        return;
    };
    let mods = compile_to_leng(
        &path,
        "import std/syncio\nwrite(stdout, \"the Temen is here\\n\")\n",
    );
    let units: Vec<temen_leng::WholeModule> = mods
        .iter()
        .map(|(stem, src)| temen_leng::WholeModule { stem, src })
        .collect();
    let m = temen_leng::link_nim_powerbox(&units, None).unwrap_or_else(|e| panic!("link: {e}"));

    // Read an 8-byte little-endian word from the linked data image at window offset `off`.
    let read_word = |off: u64| -> u64 {
        let seg = m
            .data
            .iter()
            .find(|d| d.offset <= off && off + 8 <= d.offset + d.bytes.len() as u64)
            .unwrap_or_else(|| panic!("no data segment covers window offset {off:#x}"));
        let lo = (off - seg.offset) as usize;
        u64::from_le_bytes(seg.bytes[lo..lo + 8].try_into().unwrap())
    };

    let win = 1u64 << m.memory.expect("powerbox module has a window").size_log2;
    // #964/#1094: the link bakes the heap words one guard up, in the guard's scratch page — read them
    // at `scratch + POWERBOX_HEAP_BRK`/`TOP` (the DAP convention). The guard is unconditional now.
    let scratch = temen_ir::module_null_guard();
    let brk = read_word(scratch + temen_ir::POWERBOX_HEAP_BRK);
    let top = read_word(scratch + temen_ir::POWERBOX_HEAP_TOP);
    let entry_sp = temen_ir::powerbox_entry_sp(&m);

    assert_eq!(
        brk,
        entry_sp + temen_ir::POWERBOX_STACK_RESERVE,
        "heap break seeded just above the data stack"
    );
    assert_eq!(top, win, "heap ceiling seeded to the mapped window top");
    assert!(
        top > brk,
        "the heap spans a non-empty region [{brk:#x}, {top:#x})"
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
    let mut ordered: Vec<&(String, String)> = mods.iter().collect();
    ordered.sort_by_key(|(stem, _)| stem.starts_with("sysv"));
    let units: Vec<temen_leng::WholeModule> = ordered
        .iter()
        .map(|(stem, src)| temen_leng::WholeModule { stem, src })
        .collect();
    // Compute leaves only: the true syscalls stay **retained imports** for the POSIX personality to
    // bind by name below, so this deliberately does not take `nim_powerbox_runtime`'s adapter.
    let runtime = temen_leng::nim_compute_shim_unit(&units).expect("compute shim unit");
    // Link with a synthesized **powerbox `_start`** at function 0: it reads the post-link data-stack
    // base (`data.top` → `powerbox_entry_sp`, page-aligned above the globals) and calls the C-shaped
    // `main($sp, argc, argv, envp)` with `argc/argv/envp = 0` — a real powerbox entry, not a
    // hand-provided `$sp`. Injecting `_start` as a first link unit (rather than prepending it after
    // the link) keeps the program's `data.funcref` gvar initializers valid — the funcref-carrying
    // at-exit flush this very program registers would otherwise dispatch through a stale, off-by-one
    // index.
    let m = temen_leng::link_whole_powerbox_manifest(&units, vec![runtime])
        .unwrap_or_else(|e| panic!("powerbox manifest link: {e}"));
    temen_verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    // The merged module carries the §3e powerbox entry shape: a paramless `_start` at function 0
    // exported by name — exactly what `run_powerbox`/`instantiate` bind manifest slots against.
    assert!(
        temen_run::is_named_powerbox_entry(&m),
        "linked module is a powerbox entry (paramless func-0 `_start`)"
    );

    // Instantiate the manifest module through the reference embedding and run `_start` on both
    // engines under the **POSIX personality** — the raw-syscall leaves (`sysWrite.0.` …) bind by name
    // to `temen_posix`'s `write`/`read`/`open`/`close`/`lseek` ops, whose `write(fd, buf, n)` ABI is the
    // exact nimony bottom edge. This is the real load-a-manifest-module, host-grants-the-caps path
    // (`instantiate_with_imports` → `Instance::run`), not a hand-wired host-binding harness. Each
    // engine gets its own fresh personality (separate fd table + stdout buffer); a divergence in the
    // two captured streams is a real engine bug on the W3 syscall seam.
    let cfg = temen_run::RunConfig::default();
    let interp_out = run_io_capture(&m, temen_run::Backend::TreeWalk, &cfg, &[]).stdout();
    let jit_out = run_io_capture(&m, temen_run::Backend::Jit, &cfg, &[]).stdout();
    assert_eq!(
        interp_out, jit_out,
        "§9 interp/JIT parity on the POSIX syscall I/O seam"
    );
    interp_out
}

/// Map a retained nimony syscall import name to the POSIX-personality op it binds to. The nimony
/// bottom edge (`sysWrite {.importc: "write".}` …) is spelled by the *nim* symbol (`sysWrite.0.`),
/// not the C name, so the powerbox's by-C-name resolver doesn't reach it — this is the small
/// nimony→personality name map that lets the retained leaves bind to `temen_posix`'s fd-based ops
/// (whose signatures match the nim ABI exactly).
///
/// `sysOpen` is deliberately absent: C's `open` takes a NUL-terminated `char*` where the personality
/// wants `(ptr, len)`, so it goes through `temen_leng`'s `POSIX_OPEN_ADAPTER` at link and arrives
/// here as a bare `open` instead. Binding it here directly read the flags word as the path length.
fn nim_posix_op(name: &str) -> u32 {
    if name.starts_with("sysWrite") {
        temen_posix::OP_WRITE
    } else if name.starts_with("sysRead") {
        temen_posix::OP_READ
    } else if name.starts_with("sysClose") {
        temen_posix::OP_CLOSE
    } else if name.starts_with("sysLseek") {
        temen_posix::OP_LSEEK
    } else if name.starts_with("getcwd") {
        // Served for real on this route (`temen_leng::POSIX_SERVED_LEAVES`) rather than by the
        // compute shim's NULL-returning stub; `getcwd(buf, size) -> buf` is the C ABI unchanged.
        temen_posix::OP_GETCWD
    } else {
        panic!("unmapped nimony syscall import `{name}` — extend nim_posix_op");
    }
}

/// Run the linked I/O program's powerbox `_start` (function 0) on `backend` through the reference
/// embedding (`Instance`), with every retained nim-name syscall import bound to a single shared
/// **POSIX personality**. Returns the personality itself, so the caller reads back whatever it cares
/// about — the bytes the guest `write`-to-fd-1'd (`Posix::stdout`), or a file it wrote to the memfs.
/// `seed` stages the memfs before `_start` (the files the guest will `open`), and `config` carries
/// the argv/env the guest is run with. The program uses no posix `malloc`/`mmap` (its own runtime
/// shim serves those), so the personality's heap arena is unused and passed empty.
fn run_io_capture(
    m: &Module,
    backend: temen_run::Backend,
    config: &temen_run::RunConfig,
    seed: &[(&str, &[u8])],
) -> temen_posix::Posix {
    // One personality shared across every bound name (one fd table, one stdout buffer): the factory
    // closes over a single `inner`, so each per-name grant re-mints a handler over the same state.
    // `temen_posix::cap` hands back an opaque `impl Fn` (no `Clone`), so wrap it in an `Arc` and share
    // that across the per-name grant closures.
    let (posix, make) = temen_posix::cap(0, 0, Vec::new());
    let make = std::sync::Arc::new(make);
    let mut imports = temen_run::Imports::new();
    for imp in &m.imports {
        // The guest libc's `write` is a §3e **STREAM cap**, not a nim-name syscall leaf: it is the
        // ordinary powerbox stdout, granted as such. Everything else is a retained nimony syscall.
        if imp.name == "write" {
            imports = imports.provide(imp.name.clone(), temen_run::HostCap::stdout());
            continue;
        }
        // The POSIX open adapter's forward (`temen_leng::POSIX_OPEN_ADAPTER`): a bare `open` taking
        // the `(ptr, len, flags)` the personality's op wants.
        if imp.name == "open" {
            let make = std::sync::Arc::clone(&make);
            imports = imports.provide(
                imp.name.clone(),
                temen_run::HostCap::host_proc(temen_posix::OP_OPEN, move || (*make)()),
            );
            continue;
        }
        let make = std::sync::Arc::clone(&make);
        imports = imports.provide(
            imp.name.clone(),
            temen_run::HostCap::host_proc(nim_posix_op(&imp.name), move || (*make)()),
        );
    }
    for (path, bytes) in seed {
        posix.write_file(path, bytes);
    }
    let inst = temen_run::instantiate_with_imports(m.clone(), imports)
        .unwrap_or_else(|e| panic!("instantiate manifest module: {e}"));
    inst.run(backend, config)
        .unwrap_or_else(|e| panic!("run `_start` on {backend:?}: {e}"));
    posix
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

// -------------------------------------------------------------------------------------------------
// Run-correctness of covered stdlib modules (#1382). Verify-passing is necessary but not sufficient:
// a wrong-sign width conversion or a mis-threaded frame would verify yet compute the wrong bytes.
// These compile a program exercising real ops of a covered module, run it on **both engines** (§9
// interp/JIT parity, enforced inside `run_io_program`), and diff the output against the oracle
// captured from the native nimony toolchain — a true differential test of the leng codegen.
//
// Scope: the pure-compute containers/integer modules run today. The formatting/parsing modules
// (strutils/json/unicode/…) additionally need libc `c_snprintf`/`c_strtod`, math needs the libm
// transcendentals, and times/monotimes need clock syscalls — none yet provided by the runtime shim
// (tracked separately). Those modules `link`+`verify` (their own tests); running them waits on the
// runtime-provider work.
// -------------------------------------------------------------------------------------------------

// -------------------------------------------------------------------------------------------------
// Runtime providers (#1422): programs that need the **prebuilt guest libc**. `snprintf` (nim's
// `formatBiggestFloat`), `strtod`, and the libm transcendentals are far too large to hand-write as
// Temen text, so `link_nim_powerbox` links the same guest-C libc the chibicc playground card uses
// (`browser/web/assets/pg_libc.temeno`, built by `genlibc`). Each test diffs the program's real
// output against the oracle captured from the native nimony toolchain.
// -------------------------------------------------------------------------------------------------

/// The committed guest libc, or `None` when the asset is absent (the caller then skips).
fn guest_libc() -> Option<Vec<u8>> {
    std::fs::read("../../browser/web/assets/pg_libc.temeno").ok()
}

/// Compile `src`, link it against the guest libc, and run `_start` under the **standard powerbox**
/// (the same engine the browser's `temen_run_onramp` wraps), returning the bytes it wrote to stdout.
fn run_libc_program(src: &str) -> Option<Vec<u8>> {
    let path = toolchain_path()?;
    let libc = guest_libc()?;
    let mods = compile_to_leng(&path, src);
    let units: Vec<temen_leng::WholeModule> = mods
        .iter()
        .map(|(stem, src)| temen_leng::WholeModule { stem, src })
        .collect();
    let m = temen_leng::link_nim_powerbox(&units, Some(&libc))
        .unwrap_or_else(|e| panic!("nim→powerbox link (with libc): {e}"));
    dump_module("libc_program", &m);
    temen_verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    // The guest libc's file/heap caps resolve to stubs at link, so the program's manifest is the one
    // `write` STREAM cap — exactly as it is without the libc.
    assert_eq!(
        m.imports
            .iter()
            .map(|i| i.name.as_str())
            .collect::<Vec<_>>(),
        vec!["write"],
        "linking the guest libc must not widen the program's capability manifest"
    );
    let run = temen_run::run_powerbox(&m, &[]).unwrap_or_else(|e| panic!("run_powerbox: {e}"));
    Some(run.stdout)
}

/// **#1422 — `std/strutils` float formatting runs.** `formatBiggestFloat` calls C `snprintf` with
/// `%#.*g`/`%#.*e`/`%#.*f`; the guest libc serves it (varargs are clang-wasm-style on both sides, so
/// it binds with no shim). Output is diffed against the native-nimony oracle.
#[test]
fn real_strutils_float_formatting_runs() {
    let src = concat!(
        "import std/syncio\n",
        "import std/strutils\n",
        "write(stdout, formatFloat(3.14159, ffDecimal, 3))\n",
        "write(stdout, \"|\")\n",
        "write(stdout, formatFloat(2.5, ffScientific, 2))\n",
        "write(stdout, \"|\")\n",
        "write(stdout, formatFloat(1.0, ffDefault, -1))\n",
    );
    let Some(out) = run_libc_program(src) else {
        eprintln!("SKIP: nimony toolchain or guest libc asset absent");
        return;
    };
    assert_eq!(out, b"3.142|2.50e+00|1");
}

/// **#1422 — `std/parseutils` float parsing runs.** `parseBiggestFloat` falls back to C `strtod`
/// for anything its fast path won't take; the guest libc serves it. Diffed against the
/// native-nimony oracle (4 characters consumed, value 2.75).
#[test]
fn real_parseutils_strtod_runs() {
    let src = concat!(
        "import std/syncio\n",
        "import std/strutils\n",
        "import std/parseutils\n",
        "var f: BiggestFloat = 0.0\n",
        "let n = parseBiggestFloat(\"2.75xyz\", f)\n",
        "write(stdout, $n)\n",
        "write(stdout, \"|\")\n",
        "write(stdout, formatFloat(f, ffDecimal, 2))\n",
    );
    let Some(out) = run_libc_program(src) else {
        eprintln!("SKIP: nimony toolchain or guest libc asset absent");
        return;
    };
    assert_eq!(out, b"4|2.75");
}

/// **#1422 — `std/times` links and runs.** `times` (and `monotimes`) reach `std/posix`, whose
/// `clock_gettime`/`open`/`execve`/… leaves the compute shim now serves as fail-closed stubs — a
/// sandboxed program gets no ambient clock, filesystem or process table, so `clock_gettime` reports a
/// deterministic zero timespec. Before, those leaves were unbound and the program refused to start.
#[test]
fn real_times_module_runs() {
    let src = "import std/syncio\nimport std/times\nwrite(stdout, \"times ok\")\n";
    let Some(out) = run_libc_program(src) else {
        eprintln!("SKIP: nimony toolchain or guest libc asset absent");
        return;
    };
    assert_eq!(out, b"times ok");
}

/// **#1422 — `std/math` transcendentals run.** nim imports each as a `float64`/`float32` pair
/// (`sin`/`sinf`), and the guest libc now carries both, plus the hyperbolics built on `exp`/`log`/
/// `sqrt`. Values are printed through `formatFloat`, so this covers the libm *and* `snprintf` paths.
#[test]
fn real_math_transcendentals_run() {
    let src = concat!(
        "import std/syncio\n",
        "import std/strutils\n",
        "import std/math\n",
        "write(stdout, formatFloat(sin(0.5), ffDecimal, 6))\n",
        "write(stdout, \"|\")\n",
        "write(stdout, formatFloat(cos(0.5), ffDecimal, 6))\n",
        "write(stdout, \"|\")\n",
        "write(stdout, formatFloat(tanh(1.0), ffDecimal, 6))\n",
        "write(stdout, \"|\")\n",
        "write(stdout, formatFloat(arctan(1.0), ffDecimal, 6))\n",
        "write(stdout, \"|\")\n",
        "write(stdout, formatFloat(sinh(1.0), ffDecimal, 6))\n",
    );
    let Some(out) = run_libc_program(src) else {
        eprintln!("SKIP: nimony toolchain or guest libc asset absent");
        return;
    };
    assert_eq!(out, b"0.479426|0.877583|0.761594|0.785398|1.175201");
}

#[test]
fn real_tables_ops_run_correctly() {
    let Some(path) = toolchain_path() else {
        eprintln!("SKIP: nimony toolchain not found (set NIMONY_BIN/NIM_BIN or install on PATH)");
        return;
    };
    let src = concat!(
        "import std/syncio\n",
        "import std/tables\n",
        "var t = initTable[string, int]()\n",
        "t[\"a\"] = 2\n",
        "t[\"b\"] = 40\n",
        "write(stdout, $t.getOrDefault(\"a\"))\n",
        "write(stdout, \"|\")\n",
        "write(stdout, $(t.getOrDefault(\"a\") + t.getOrDefault(\"b\")))\n",
        "write(stdout, \"|\")\n",
        "write(stdout, $t.len)\n",
        "write(stdout, \"|\")\n",
        "write(stdout, $(\"b\" in t))\n",
        "write(stdout, \"|\")\n",
        "write(stdout, $(\"z\" in t))\n",
    );
    assert_eq!(
        run_io_program(&compile_to_leng(&path, src)),
        b"2|42|2|true|false",
        "tables ops match the native-nimony oracle on both engines"
    );
}

#[test]
fn real_sets_ops_run_correctly() {
    let Some(path) = toolchain_path() else {
        eprintln!("SKIP: nimony toolchain not found (set NIMONY_BIN/NIM_BIN or install on PATH)");
        return;
    };
    let src = concat!(
        "import std/syncio\n",
        "import std/sets\n",
        "var s = initHashSet[int]()\n",
        "s.incl(3)\n",
        "s.incl(7)\n",
        "s.incl(3)\n",
        "write(stdout, $s.len)\n",
        "write(stdout, \"|\")\n",
        "write(stdout, $(7 in s))\n",
        "write(stdout, \"|\")\n",
        "write(stdout, $(4 in s))\n",
    );
    assert_eq!(
        run_io_program(&compile_to_leng(&path, src)),
        b"2|true|false",
        "sets ops match the native-nimony oracle on both engines"
    );
}

#[test]
fn real_bitops_ops_run_correctly() {
    let Some(path) = toolchain_path() else {
        eprintln!("SKIP: nimony toolchain not found (set NIMONY_BIN/NIM_BIN or install on PATH)");
        return;
    };
    let src = concat!(
        "import std/syncio\n",
        "import std/bitops\n",
        "write(stdout, $countSetBits(0xF0F0'u32))\n",
        "write(stdout, \"|\")\n",
        "write(stdout, $(0b1010 shl 2))\n",
    );
    assert_eq!(
        run_io_program(&compile_to_leng(&path, src)),
        b"8|40",
        "bitops ops match the native-nimony oracle on both engines"
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
    let dir = std::env::temp_dir().join(format!("temen_nim_sweep_{}_{name}", std::process::id()));
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
                let units: Vec<temen_leng::WholeModule> = mods
                    .iter()
                    .map(|(stem, src)| temen_leng::WholeModule { stem, src })
                    .collect();
                // Manifest link with an *empty* runtime: every unresolved bottom-edge leaf is retained
                // as a manifest import instead of fail-closing the link, so only genuine translator
                // `Unsupported` gaps (which surface during translate, before link) are reported — not
                // "no unit exports `write`" noise.
                match temen_leng::link_whole_with_runtime_manifest(&units, Vec::new()) {
                    Ok(_) => {
                        ok += 1;
                        link_ok += 1;
                    }
                    Err(err) => {
                        eprintln!("[gap] {name}: {err}");
                        let (kind, msg) = match &err {
                            temen_leng::LengError::Unsupported(m) => ("Unsupported", m.clone()),
                            temen_leng::LengError::Malformed(m) => ("Malformed", m.clone()),
                            temen_leng::LengError::Parse(m) => ("Parse", m.clone()),
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

/// **#1422 — `std/os` path handling runs.** The whole `os` module used to fail to *link*: its posix
/// bottom edge (`stat`/`getcwd`/`c_getenv`/`fork`/…) had no provider, so even the **pure** half —
/// string-only path manipulation that touches no filesystem at all — was unreachable. With the
/// fail-closed posix stubs it links, and the path helpers give byte-identical answers to native
/// nimony (they are pure string functions; nothing here consults a real filesystem).
#[test]
fn real_os_path_helpers_run() {
    let src = concat!(
        "import std/syncio\n",
        "import std/os\n",
        "\n",
        "let j = joinPath(\"a/b\", \"c.txt\")\n",
        "let p = parentDir(\"a/b/c.txt\")\n",
        "let f = extractFilename(\"a/b/c.txt\")\n",
        "let e = changeFileExt(\"a/b/c.txt\", \"nif\")\n",
        "write(stdout, j & \"|\" & p & \"|\" & f & \"|\" & e)\n",
    );
    let Some(out) = run_libc_program(src) else {
        eprintln!("SKIP real_os_path_helpers_run (no toolchain / libc asset)");
        return;
    };
    // Native `nimony c --run` prints exactly this.
    assert_eq!(
        String::from_utf8_lossy(&out),
        "a/b/c.txt|a/b|c.txt|a/b/c.nif"
    );
}

/// **#1422 — `std/strtabs` runs.** `strtabs` reaches the posix edge only through `os`'s environment
/// helpers; the table itself is pure. Previously unlinkable for that reason alone.
#[test]
fn real_strtabs_runs() {
    let src = concat!(
        "import std/syncio\n",
        "import std/strtabs\n",
        "\n",
        "var t = newStringTable(modeCaseSensitive)\n",
        "t[\"a\"] = \"1\"\n",
        "t[\"b\"] = \"2\"\n",
        "write(stdout, t.getOrDefault(\"a\") & \"|\" & t.getOrDefault(\"b\") & \"|\" & $t.len & \"|\" & $t.hasKey(\"c\"))\n",
    );
    let Some(out) = run_libc_program(src) else {
        eprintln!("SKIP real_strtabs_runs (no toolchain / libc asset)");
        return;
    };
    assert_eq!(String::from_utf8_lossy(&out), "1|2|2|false");
}

/// **#1422 — `std/envvars` runs, and the sandbox's empty environment is *indistinguishable* from a
/// host where the variable is simply unset.** `c_getenv` stubs to a null pointer, so `getEnv` yields
/// `""` and `existsEnv` yields `false` — byte-identical to native nimony for an unset name. The guest
/// is told "absent", never handed the host's environment.
#[test]
fn real_envvars_report_an_empty_environment() {
    let src = concat!(
        "import std/syncio\n",
        "import std/envvars\n",
        "\n",
        "write(stdout, \"[\" & getEnv(\"TEMEN_NO_SUCH_VAR_12345\") & \"]|\" & $existsEnv(\"TEMEN_NO_SUCH_VAR_12345\"))\n",
    );
    let Some(out) = run_libc_program(src) else {
        eprintln!("SKIP real_envvars_report_an_empty_environment (no toolchain / libc asset)");
        return;
    };
    assert_eq!(String::from_utf8_lossy(&out), "[]|false");
}

/// **The epoll trio is bound, and it reports failure.** `run_libc_program` asserts the program's
/// manifest is exactly the one `write` cap, so this first of all pins that `epoll_create1`/`_ctl`/
/// `_wait` are no longer unbound leaves — the thing that kept `std/threadpool` and `std/parfor` from
/// linking at all.
///
/// It then *calls* all three, because "it linked" is the weaker claim: a leaf bound to the wrong
/// shim, or to one returning a plausible-looking 0, links exactly as cleanly. `-1|-1|-1` is the
/// contract — the facility is absent, and `epoll_wait` says so rather than answering "no events
/// ready" for an epoll set that was never created.
#[test]
fn real_epoll_leaves_report_failure() {
    let src = concat!(
        "import std/syncio\n",
        "\n",
        "proc epoll_create1(flags: cint): cint {.importc, header: \"<sys/epoll.h>\".}\n",
        "proc epoll_ctl(epfd: cint; op: cint; fd: cint; event: ptr int64): cint {.importc,\n",
        "    header: \"<sys/epoll.h>\".}\n",
        "proc epoll_wait(epfd: cint; events: ptr int64; maxevents: cint; timeout: cint): cint {.importc,\n",
        "    header: \"<sys/epoll.h>\".}\n",
        "\n",
        // nimony pointers are non-nullable (`nil` is rejected outright), so the two pointer-taking
        // leaves get a real buffer. The stubs never read it.
        "var ev: array[8, int64]\n",
        "let a = epoll_create1(0.cint)\n",
        "let b = epoll_ctl(0.cint, 1.cint, 2.cint, addr ev[0])\n",
        "let c = epoll_wait(0.cint, addr ev[0], 1.cint, 0.cint)\n",
        "write(stdout, $a & \"|\" & $b & \"|\" & $c)\n",
    );
    let Some(out) = run_libc_program(src) else {
        eprintln!("SKIP real_epoll_leaves_report_failure (no toolchain / libc asset)");
        return;
    };
    assert_eq!(String::from_utf8_lossy(&out), "-1|-1|-1");
}

/// **`std/threadpool` and `std/parfor` link and run.** Both were unlinkable for one reason only —
/// the three epoll leaves above had no provider — so nothing in either module was reachable, pure or
/// not. Importing them is the whole test: `run_libc_program` pins the manifest to the single `write`
/// cap, and reaching the `write` proves module-level initialization did not trap on the way.
///
/// This does **not** claim a working thread pool. `initPool()` is an explicit call, not a module
/// initializer, and a program that makes it gets nim's own `assert gIoFd >= 0` — see the epoll rows
/// in `COMPUTE_LEAVES` for why that loud failure is the intended outcome.
#[test]
fn real_threadpool_and_parfor_link_and_run() {
    for module in ["threadpool", "parfor"] {
        let src = format!("import std/syncio\nimport std/{module}\n\nwrite(stdout, \"ok\")\n");
        let Some(out) = run_libc_program(&src) else {
            eprintln!("SKIP real_threadpool_and_parfor_link_and_run (no toolchain / libc asset)");
            return;
        };
        assert_eq!(String::from_utf8_lossy(&out), "ok", "std/{module}");
    }
}

/// **`std/ioring` is blocked on an upstream declaration conflict**, and this pins the blocker so it
/// self-heals.
///
/// The module used to link and run (every descriptor call sits inside an explicit proc —
/// `initIoRing`, `listenTcp`, `submitRead` — so importing is safe and a program that opens a socket
/// gets -1). Under nimony v0.6.2 it cannot link, because **three stdlib modules declare the same C
/// `syscall` with different widths**:
///
/// - `std/posix/io_uring`: `proc syscall(arg: cint): cint {.importc: "syscall", varargs.}`
/// - `std/rawthreads`:     `proc syscall(arg: clong): clong {.varargs, importc: "syscall".}`
/// - `std/private/syslocks`: `proc syscall(number: clong): clong {.importc: "syscall", varargs.}`
///
/// `std/ioring` pulls in all three. On the C backend this is invisible — `<unistd.h>`'s prototype is
/// the one that matters and nim's `importc` just calls it — but in an object-link model the bottom
/// edge has one `syscall.0.` symbol, and after the varargs marshalling its shape is either
/// `(i32, i64) -> i32` or `(i64, i64) -> i64`. C's own prototype is `long syscall(long, ...)`, so
/// io_uring's is the inaccurate one.
///
/// Refusing is correct (#1524): picking a width would be the silent-widening this project made
/// fail-closed on purpose, and it is the kind of mismatch that reads a register the callee never
/// wrote. So this asserts the **blocker**, not a workaround — the moment upstream aligns the three
/// declarations the link succeeds, this test fails, and it goes back to asserting `"ok"`.
#[test]
fn real_ioring_blocked_on_conflicting_syscall_decls() {
    let Some(path) = toolchain_path() else {
        eprintln!("SKIP real_ioring_blocked_on_conflicting_syscall_decls (no toolchain)");
        return;
    };
    let mods = compile_to_leng(
        &path,
        "import std/syncio\nimport std/ioring\n\nwrite(stdout, \"ok\")\n",
    );
    let units: Vec<temen_leng::WholeModule> = mods
        .iter()
        .map(|(stem, src)| temen_leng::WholeModule { stem, src })
        .collect();
    let err = match temen_leng::link_nim_powerbox(&units, guest_libc().as_deref()) {
        Err(e) => e.to_string(),
        Ok(_) => panic!(
            "std/ioring now links — upstream aligned the three `syscall` declarations. Restore this \
             test to `run_libc_program` + assert_eq!(out, \"ok\")."
        ),
    };
    assert!(
        err.contains("syscall.0.") && err.contains("ImportShapeMismatch"),
        "expected the conflicting-`syscall` link refusal, got: {err}"
    );
}

/// **`htons` computes, it does not stub.** It rides in with `std/ioring`'s socket leaves but is not
/// one of them: it is a pure 16-bit byte swap a guest is perfectly entitled to perform, so binding it
/// to the -1 its neighbours return would silently hand `listenTcp` a wrong port rather than fail.
/// Values are checked against the swap itself — `80 -> 0x5000`, `0x1234 -> 0x3412` — so a regression
/// to a constant stub cannot pass.
#[test]
fn real_htons_byte_swaps() {
    let src = concat!(
        "import std/syncio\n",
        "\n",
        "proc htons(x: uint16): uint16 {.importc, header: \"<arpa/inet.h>\".}\n",
        "\n",
        "let a = htons(80\'u16)\n",
        "let b = htons(0x1234\'u16)\n",
        "let c = htons(0\'u16)\n",
        "write(stdout, $a & \"|\" & $b & \"|\" & $c)\n",
    );
    let Some(out) = run_libc_program(src) else {
        eprintln!("SKIP real_htons_byte_swaps (no toolchain / libc asset)");
        return;
    };
    assert_eq!(String::from_utf8_lossy(&out), "20480|13330|0");
}

/// **#1422 stage-3 runnability sweep.** The `#760` sweep above answers "does it *translate*?"; this
/// one answers "does it *run*?" — the question stage 3 is about. For every stdlib module in
/// [`discovered_std_modules`] it compiles a driver that imports the module, links it through the real
/// `link_nim_powerbox` **with the guest libc**, checks the result verifies and asks for nothing
/// beyond the one `write` stream cap, and then **runs it and requires the driver's "ok"**. A module
/// that links with an extra manifest entry has an unbound bottom-edge leaf and would fail to
/// instantiate in the playground; a module that links and then dies in its own start-up is no more
/// runnable, and only running it says so.
///
/// This is a **diagnostic, not a gate** — it drives the whole nimony toolchain once per module
/// (~20 minutes for the full list), so it is gated on `NIM_RUN_SWEEP=1` and stays out of CI. What
/// gates in CI is the handful of `real_*_runs` tests above: each one actually *runs* a program from
/// one of these modules and diffs the output against the native-nimony oracle, which is a stronger
/// claim than "it linked" and costs seconds. Run the sweep when you change the bottom edge, to see
/// what moved. `NIM_RUN_SWEEP_STRICT=1` narrows it to [`STD_MODULES`] and asserts instead of
/// reporting, for a bisect.
#[test]
fn runnability_sweep() {
    if std::env::var("NIM_RUN_SWEEP").is_err() {
        eprintln!("SKIP runnability_sweep (set NIM_RUN_SWEEP=1 to run)");
        return;
    }
    let Some(path) = toolchain_path() else {
        eprintln!("SKIP runnability_sweep (no nimony toolchain)");
        return;
    };
    let Some(libc) = guest_libc() else {
        eprintln!("SKIP runnability_sweep (no guest libc asset)");
        return;
    };
    let Some(stdlib) = nimony_stdlib_dir() else {
        eprintln!("SKIP runnability_sweep (cannot locate nimony lib/std)");
        return;
    };
    let strict = std::env::var("NIM_RUN_SWEEP_STRICT").is_ok();
    let all = discovered_std_modules(&stdlib);
    // A name in the green list that is not in `lib/std/` is a stale entry, not a passing module:
    // before #1490 seven of them sat in `STD_MODULES` and were silently counted as green because a
    // missing file fails at the nimony step and strict mode only asserted on `unrunnable`.
    let missing_green: Vec<&str> = STD_MODULES
        .iter()
        .copied()
        .filter(|m| !all.iter().any(|a| a == m))
        .collect();
    assert!(
        missing_green.is_empty(),
        "STD_MODULES names modules that are not in {stdlib:?}: {missing_green:?} \
         — remove them or fix the spelling; they are not green, they are absent"
    );
    let mut mods: Vec<&str> = if strict {
        STD_MODULES.to_vec()
    } else {
        all.iter().map(|s| s.as_str()).collect()
    };
    // `NIM_SWEEP_ONLY=a,b,c` narrows the run to a few modules. The full sweep drives the whole
    // toolchain once per module (~25 min); when you are chasing one bottom-edge change you want the
    // three modules it touches, not all of them.
    let only = std::env::var("NIM_SWEEP_ONLY").unwrap_or_default();
    if !only.is_empty() {
        let want: Vec<&str> = only
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        mods.retain(|m| want.contains(m));
        assert!(
            !mods.is_empty(),
            "NIM_SWEEP_ONLY={only:?} matched no module in {stdlib:?}"
        );
    }
    let mut runnable: Vec<&str> = Vec::new();
    let mut unrunnable: Vec<(&str, String)> = Vec::new();
    let mut nim_fail: Vec<(&str, String)> = Vec::new();
    for (i, m) in mods.iter().enumerate() {
        let m: &str = m;
        eprintln!("[{}/{}] std/{m} …", i + 1, mods.len());
        let src = format!("import std/syncio\nimport std/{m}\n\nwrite(stdout, \"ok\")\n");
        let leng = match try_compile_to_leng(&path, m, &src) {
            Ok(v) => v,
            Err(e) => {
                nim_fail.push((m, e));
                continue;
            }
        };
        let units: Vec<temen_leng::WholeModule> = leng
            .iter()
            .map(|(stem, src)| temen_leng::WholeModule { stem, src })
            .collect();
        match temen_leng::link_nim_powerbox(&units, Some(&libc)) {
            Err(e) => unrunnable.push((m, format!("link: {e}"))),
            Ok(module) => match temen_verify::verify_module(&module) {
                Err(e) => unrunnable.push((m, format!("verify: {e:?}"))),
                Ok(()) => {
                    // Report each unbound leaf **with its signature**: the name alone says a
                    // provider is missing, the shape says what to write. Binding one means adding a
                    // shim func of exactly this type, so printing it here is the difference between
                    // "go read the nim source" and "write this func".
                    let extra: Vec<String> = module
                        .imports
                        .iter()
                        .filter(|i| i.name != "write")
                        .map(|i| match i.shape {
                            temen_ir::ImportShape::Func(t) => match module.types.get(t as usize) {
                                Some(temen_ir::TypeEntry::Func(f)) => {
                                    format!("{} {:?} -> {:?}", i.name, f.params, f.results)
                                }
                                _ => format!("{} (bad type ref)", i.name),
                            },
                            _ => format!("{} (grouped)", i.name),
                        })
                        .collect();
                    if !extra.is_empty() {
                        unrunnable.push((m, format!("unbound leaves: {}", extra.join(", "))));
                    } else {
                        // **Then actually run it.** Linking with no extra manifest entry says the
                        // bottom edge is covered; it does not say the module survives its own
                        // start-up. `std/encodings` is the case in point: its `Dl.…` global is
                        // initialized by a `nimLoadLibrary` call chain ending in `nimDynlibCheck`,
                        // which `die(1)`s when the handle is nil — and the `dlopen` stub always
                        // returns nil. The moment leng can lower a call-initialized global,
                        // `encodings` would link cleanly and abort on every run, and a sweep that
                        // stopped at the manifest would report it green. Running the driver and
                        // requiring its "ok" closes that gap for every module at once.
                        match temen_run::run_powerbox(&module, &[]) {
                            Err(e) => unrunnable.push((m, format!("run: {e}"))),
                            Ok(run) if run.stdout != b"ok" => unrunnable.push((
                                m,
                                format!(
                                    "ran but printed {:?}",
                                    String::from_utf8_lossy(&run.stdout)
                                ),
                            )),
                            Ok(_) => runnable.push(m),
                        }
                    }
                }
            },
        }
    }
    eprintln!(
        "\n===== #1422 runnability sweep: {} runnable / {} unrunnable / {} nimony-fail =====",
        runnable.len(),
        unrunnable.len(),
        nim_fail.len()
    );
    for (m, why) in &nim_fail {
        eprintln!("  [nimony] std/{m}: {why}");
    }
    for (m, why) in &unrunnable {
        eprintln!("  [unrunnable] std/{m}: {why}");
    }
    eprintln!("  runnable: {}", runnable.join(" "));
    eprintln!("===== end runnability sweep =====\n");
    if !STD_EXCLUDED.is_empty() {
        eprintln!(
            "  excluded: {}",
            STD_EXCLUDED
                .iter()
                .map(|(n, why)| format!("{n} ({why})"))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    if strict {
        // Both buckets are failures for a module we claim to hold green: `unrunnable` means it
        // links but cannot run, `nim_fail` means the toolchain rejected it. Before #1490 only the
        // first was asserted, so a green-list entry that stopped resolving passed silently.
        assert!(
            unrunnable.is_empty(),
            "modules that compile but cannot run: {unrunnable:?}"
        );
        assert!(
            nim_fail.is_empty(),
            "green-list modules the nimony toolchain rejected: {nim_fail:?}"
        );
    }
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

/// Modules present in `lib/std/` that the sweep deliberately does **not** drive, each with the
/// reason. #1490: the denominator used to be a hand-written list, which drifted from the stdlib in
/// both directions — three names that no longer existed (reported every run as "nimony failed",
/// which reads like a front-end bug to chase) and seven real modules silently omitted while the doc
/// comment claimed the list was "every" module. It is now derived from the filesystem, so the only
/// way to leave something out is to say so here.
const STD_EXCLUDED: &[(&str, &str)] = &[(
    "system",
    "implicitly imported by every module; `import std/system` is not a thing",
)];

/// Every module in the vendored nimony `lib/std/`, minus [`STD_EXCLUDED`] — the denominator the
/// stage-3 runnability sweep reports against. Derived from the filesystem (#1490) so it cannot drift
/// from the stdlib it claims to enumerate, the same "the directory *is* the expectation" property
/// the `nim_diff` corpus relies on.
fn discovered_std_modules(dir: &std::path::Path) -> Vec<String> {
    let mut out: Vec<String> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read {dir:?}: {e}"))
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|x| x == "nim"))
        .map(|e| e.path().file_stem().unwrap().to_string_lossy().to_string())
        .filter(|m| !STD_EXCLUDED.iter().any(|(n, _)| n == m))
        .collect();
    out.sort();
    out
}

/// The vendored nimony stdlib's `lib/std/`, derived from `NIMONY_BIN` (`<dist>/bin`) or from
/// wherever `nimony` sits on `PATH`. `None` when no toolchain is present — the callers already skip.
fn nimony_stdlib_dir() -> Option<std::path::PathBuf> {
    let bin = std::env::var("NIMONY_BIN")
        .ok()
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::var("PATH").ok().and_then(|p| {
                p.split(':')
                    .map(std::path::PathBuf::from)
                    .find(|d| d.join("nimony").exists())
            })
        })?;
    let dir = bin.parent()?.join("lib").join("std");
    dir.is_dir().then_some(dir)
}

/// The subset the sweep **holds green** under `NIM_RUN_SWEEP_STRICT=1` — a spread across strings,
/// containers, numerics and parsing, the constructs a real program (and nimony itself) hits. Every
/// entry must exist in `lib/std/`; the sweep asserts that up front (#1490) rather than letting an
/// absent module read as a pass.
const STD_MODULES: &[&str] = &[
    // Runnable since #1443 lowered `cpuRelax`'s `{.emit.}` hint and bound the last four atomic leaves.
    "atomics",
    // Runnable since #1498 taught the sret import path to coerce args to the callee's param types.
    "macros",
    "nifply",
    // Runnable since #1499 made the generic atomic leaves bind by signature, not by name.
    "locks",
    "rlocks",
    "ticketlocks",
    // Runnable since the epoll trio was bound as fail-closed stubs — the residue of #1443, whose
    // `{.emit.}` half fixed the other four threading modules and left these two on `epoll_*`.
    "parfor",
    "threadpool",
    // Runnable since its posix/socket edge was stubbed and `htons` given a real byte swap.
    "ioring",
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
    "unicode",
    "times",
    "json",
    "base64",
    "editdistance",
    "md5",
    "monotimes",
    "complex",
    "assertions",
    "setutils",
];

/// **#1375 — the rest of `std/math` runs, not just the trigonometric family.** `LIBC_SERVED` first
/// listed only sin/cos/tan and their relatives, because the test written beside it
/// ([`real_math_transcendentals_run`]) called only those. Everything else — `sqrt`, `pow`, `exp`, the
/// logs, the rounding family, `hypot`, `arctan2`, `cbrt` — was left unbound at link, so a program
/// calling them compiled fine and then **trapped at run**, in the playground included.
///
/// A name missing from `LIBC_SERVED` is not a compile error, so only a test that actually *calls*
/// each function can catch it. Values go through `formatFloat` rather than `$`: `$` on a non-integral
/// float is itself broken (`$3.14` prints `17.966570549813729`, on `main` and well before this work
/// — filed separately), and this test is about libm, not about the formatter.
#[test]
fn real_math_powers_roots_and_rounding_run() {
    let src = concat!(
        "import std/syncio\n",
        "import std/strutils\n",
        "import std/math\n",
        "\n",
        "proc f(x: float): string = formatFloat(x, ffDecimal, 6)\n",
        "\n",
        "write(stdout, f(sqrt(2.0)) & \"|\" & f(pow(2.0, 10.0)) & \"|\" & f(exp(1.0)) & \"|\" &\n",
        "              f(ln(100.0)) & \"|\" & f(log10(100.0)) & \"|\" & f(log2(8.0)) & \"|\" &\n",
        "              f(floor(1.7)) & \"|\" & f(ceil(1.2)) & \"|\" & f(hypot(3.0, 4.0)) & \"|\" &\n",
        "              f(arctan2(1.0, 1.0)) & \"|\" & f(cbrt(27.0)))\n",
    );
    let Some(out) = run_libc_program(src) else {
        eprintln!("SKIP real_math_powers_roots_and_rounding_run (no toolchain / libc asset)");
        return;
    };
    // `nimony c --run` prints exactly this.
    assert_eq!(
        String::from_utf8_lossy(&out),
        "1.414214|1024.000000|2.718282|4.605170|2.000000|3.000000|1.000000|2.000000|5.000000|0.785398|3.000000"
    );
}

/// **#1472 — `$` on a float.** `$3.14` printed `17.966570549813729`: `system/formatfloat`'s
/// Schubfach dtoa is built on `uint64(sfHi32(x)) * cp`, and every `(u 32)` with bit 31 set was
/// **sign**-extended into its 64-bit context, so the whole digit generation ran on garbage. Values
/// whose shortest form falls out of the dtoa's special cases (`0.5`, `2.0`, `100.0`) stayed correct
/// and hid it — which is why a suite full of `formatFloat` assertions never noticed.
///
/// The widening fix is unit-tested toolchain-free in `tests/integer.rs`
/// (`unsigned_widening_zero_extends`); this is the end-to-end proof, against the native oracle.
#[test]
fn real_dollar_float_matches_native() {
    let vals = [
        "0.5", "1.5", "2.0", "3.0", "1.25", "0.1", "3.14", "10.0", "100.0", "0.25", "7.5", "1e10",
    ];
    let mut src = String::from("import std/syncio\n\n");
    for v in &vals {
        src.push_str(&format!("write(stdout, ${v})\nwrite(stdout, \"|\")\n"));
    }
    let Some(out) = run_libc_program(&src) else {
        eprintln!("SKIP real_dollar_float_matches_native (no toolchain / libc asset)");
        return;
    };
    // `nimony c --run` prints exactly this.
    assert_eq!(
        String::from_utf8_lossy(&out),
        "0.5|1.5|2.0|3.0|1.25|0.1|3.14|10.0|100.0|0.25|7.5|10000000000.0|"
    );
}

/// **#1472, the arithmetic underneath.** 64-bit work built on `uint32` halves — the shape
/// `roundToOdd` uses, and the one that corrupted the dtoa. Every value is diffed against native
/// nimony; before the fix, `hi32` and everything derived from it came back negative.
#[test]
fn real_unsigned_32bit_halves_match_native() {
    let src = concat!(
        "import std/syncio\n\n",
        "proc lo32(x: uint64): uint32 = cast[uint32](x)\n",
        "proc hi32(x: uint64): uint32 = cast[uint32](x shr 32)\n",
        "let g: uint64 = 0x81CEB32C4B43FCF5'u64\n",
        "let cp: uint32 = 0x1234567'u32\n",
        "let b01: uint64 = uint64(lo32(g)) * cp\n",
        "let b11: uint64 = uint64(hi32(g)) * cp\n",
        "let hi: uint64 = b11 + hi32(b01)\n",
        "write(stdout, $int(lo32(g)) & \"|\" & $int(hi32(g)) & \"|\" & $int(b01) & \"|\" &\n",
        "              $int(b11) & \"|\" & $int(hi))\n",
    );
    let Some(out) = run_libc_program(src) else {
        eprintln!("SKIP real_unsigned_32bit_halves_match_native (no toolchain / libc asset)");
        return;
    };
    assert_eq!(
        String::from_utf8_lossy(&out),
        "1262746869|2177807148|24104250456395667|41571600951734964|41571600957347172"
    );
}

// ---------------------------------------------------------------------------
// The nim differential corpus.
// ---------------------------------------------------------------------------

/// Run `src` with the **native** toolchain (`nimony c --isMain --run`) and return what it printed —
/// the oracle. `Err` carries the compiler's own diagnostic, so a corpus program outside nimony's
/// subset says so rather than looking like a Temen failure.
fn native_output(nim_path: &str, name: &str, src: &str) -> Result<String, String> {
    let dir = std::env::temp_dir().join(format!("temen_nim_diff_n_{}_{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    std::fs::write(dir.join("prog.nim"), src).map_err(|e| e.to_string())?;
    let out = Command::new("nimony")
        .args(["c", "--isMain", "--run", "prog.nim"])
        .current_dir(&dir)
        .env("PATH", nim_path)
        .output()
        .map_err(|e| e.to_string())?;
    let _ = std::fs::remove_dir_all(&dir);
    if !out.status.success() {
        return Err(format!(
            "native nimony: {}",
            String::from_utf8_lossy(&out.stderr)
                .lines()
                .rev()
                .take(3)
                .collect::<Vec<_>>()
                .join(" | ")
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Run `src` on **Temen**: the real toolchain to Leng, `link_nim_powerbox` against the guest libc,
/// then `_start` under the standard powerbox. Returns what the program printed **and how the run
/// ended**. Every failure mode (nimony, link, verify, trap) comes back as `Err` with its reason, so
/// the differential reports *where* a program diverged rather than panicking on the first one.
///
/// The outcome is returned alongside the bytes because a program that produces no output has told
/// you nothing about *why*: `Returned([I32(0)])` (ran to completion and printed nothing),
/// `Exited(127)` (panicked through `cAbort`) and a trap are three different bugs that a bare `""`
/// renders identical. That ambiguity is what made the v0.6.2 empty-output blocker expensive.
/// `NIM_DIFF_DUMP=<dir>` writes the linked module as text next to its bound import list. A program
/// that runs cleanly and prints nothing gives the Nim side no way to say why; reading the generated
/// IR for the write path is what found the dropped-`scope` miscompile, after a day of bisecting from
/// the guest side. `print_module` output is large (a hello-world links ~660 functions), so this is
/// opt-in. Call it BEFORE `verify_module`: a module that fails to verify is exactly the one whose IR
/// you need, and dumping after the `?` would never produce it.
///
/// One helper rather than one per link path, so every caller reports the same way.
fn dump_module(name: &str, m: &temen_ir::Module) {
    let Ok(dir) = std::env::var("NIM_DIFF_DUMP") else {
        return;
    };
    let mut txt = String::new();
    for i in &m.imports {
        txt.push_str(&format!(
            "; import {:?} shape {:?} sig {:?}\n",
            i.name,
            i.shape,
            import_sig_dbg(m, i)
        ));
    }
    txt.push_str(&temen_text::print_module(m));
    let _ = std::fs::write(format!("{dir}/{name}.temt"), txt);
}

/// The `(params, results)` behind an import's type index, for `NIM_DIFF_DUMP`.
fn import_sig_dbg(
    m: &temen_ir::Module,
    imp: &temen_ir::Import,
) -> Option<(Vec<temen_ir::ValType>, Vec<temen_ir::ValType>)> {
    let temen_ir::ImportShape::Func(t) = imp.shape else {
        return None;
    };
    match m.types.get(t as usize)? {
        temen_ir::TypeEntry::Func(f) => Some((f.params.clone(), f.results.clone())),
        _ => None,
    }
}

fn temen_output(
    nim_path: &str,
    libc: &[u8],
    name: &str,
    src: &str,
) -> Result<(String, String), String> {
    let mods = try_compile_to_leng(nim_path, name, src)?;
    let units: Vec<temen_leng::WholeModule> = mods
        .iter()
        .map(|(stem, src)| temen_leng::WholeModule { stem, src })
        .collect();
    let m = temen_leng::link_nim_powerbox(&units, Some(libc)).map_err(|e| format!("link: {e}"))?;
    dump_module(name, &m);
    temen_verify::verify_module(&m).map_err(|e| format!("verify: {e:?}"))?;
    let extra: Vec<&str> = m
        .imports
        .iter()
        .map(|i| i.name.as_str())
        .filter(|n| *n != "write")
        .collect();
    if !extra.is_empty() {
        // An unbound bottom-edge leaf. The program would trap at run with nothing to say, so name it
        // here — this is how the missing half of libm surfaced (#1375).
        return Err(format!("unbound leaves: {}", extra.join(", ")));
    }
    let run = temen_run::run_powerbox(&m, &[]).map_err(|e| format!("run: {e}"))?;
    Ok((
        String::from_utf8_lossy(&run.stdout).into_owned(),
        format!("{:?}", run.outcome),
    ))
}

/// **The nim differential corpus** — every `tests/nim_diff/*.nim` compiled and run twice, on Temen and
/// on native nimony, with the two outputs diffed byte for byte.
///
/// This exists because the wrong-answer bugs keep being found by accident. `$3.14` printing
/// `17.966570549813729` (#1472) sat in `main` behind a suite that only ever asserted `formatFloat`;
/// half of libm was unbound (#1375) behind a test that happened to call only the trigonometric
/// functions. Both compiled, verified, and ran — they just produced the wrong bytes, which no
/// link-level or feature-envelope check can see.
///
/// The oracle is the native toolchain rather than a transcribed constant, so **adding a case is
/// adding a file**: no expected value to work out by hand, and no risk of baking in a wrong one.
/// Keep each program deterministic (no clock, no addresses, no iteration order that nim does not
/// pin) and inside nimony's subset.
#[test]
fn nim_differential_corpus() {
    let Some(path) = toolchain_path() else {
        eprintln!("SKIP nim_differential_corpus (no nimony toolchain)");
        return;
    };
    let Some(libc) = guest_libc() else {
        eprintln!("SKIP nim_differential_corpus (no guest libc asset)");
        return;
    };
    let dir = std::path::Path::new("tests/nim_diff");
    let mut cases = nim_cases(dir);
    assert!(!cases.is_empty(), "no corpus programs in {dir:?}");
    // `NIM_DIFF_ONLY=a,b,c` narrows the run to a few cases. The full corpus drives the whole
    // toolchain twice per case (native oracle + temen), so chasing one divergence over the whole
    // corpus costs ~30 min of wall clock to reach the case you care about. Filtering here — rather
    // than in a hand-rolled probe harness — keeps the one code path: the case runs under exactly the
    // driver that reports it, oracle comparison included.
    let only = std::env::var("NIM_DIFF_ONLY").unwrap_or_default();
    if !only.is_empty() {
        let want: Vec<&str> = only
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        cases.retain(|c| {
            c.file_stem()
                .map(|s| want.contains(&&*s.to_string_lossy()))
                .unwrap_or(false)
        });
        assert_eq!(
            cases.len(),
            want.len(),
            "NIM_DIFF_ONLY named {want:?} but matched {:?}",
            cases
                .iter()
                .map(|c| c.file_stem().unwrap().to_string_lossy().to_string())
                .collect::<Vec<_>>()
        );
    }
    // `known_gaps/` holds programs that are *expected* to diverge, each naming its issue. They are
    // still run: a gap that quietly starts working should be promoted and its issue closed, so that
    // is reported as a failure too. A directory is the whole expectation mechanism — no per-case
    // enum to keep in sync.
    let gap_dir = dir.join("known_gaps");
    // `NIM_DIFF_ONLY` names cases, so a filtered run skips the gaps outright — they are a separate
    // expectation, and re-running them would dominate the wall clock the filter exists to cut.
    let gaps = if only.is_empty() {
        nim_cases(&gap_dir)
    } else {
        Vec::new()
    };

    let mut failures: Vec<String> = Vec::new();
    for case in &cases {
        let name = case.file_stem().unwrap().to_string_lossy().to_string();
        let src = std::fs::read_to_string(case).expect("read case");
        // Announce the case **before** running it, and time it. Reporting only on success makes a
        // slow or non-terminating case invisible: the suite just stops producing output, which reads
        // as "hung on the first case" no matter which case it actually is. That cost real debugging
        // time on the v0.6.2 bump.
        eprintln!("  {name}: …");
        let started = std::time::Instant::now();
        let want = match native_output(&path, &name, &src) {
            Ok(s) => s,
            // Outside nimony's own subset: the corpus program is wrong, not Temen. Say so loudly —
            // a case that never runs natively silently tests nothing.
            Err(e) => {
                failures.push(format!("{name}: does not run under native nimony — {e}"));
                continue;
            }
        };
        match temen_output(&path, &libc, &name, &src) {
            // The bytes are only half the answer. `native_output` rejects a program that exits
            // non-zero natively, so every corpus case ends cleanly on the oracle — a Temen run that
            // ends any other way has diverged even when it printed the right bytes. `Exited(127)` is
            // what a nim panic looks like once `cAbort` reaches the stubbed `kill`, and a program
            // that panics after printing its output would otherwise pass.
            Ok((got, outcome)) if got == want && !is_clean_exit(&outcome) => failures.push(
                format!("{name}: output matches but the run ended {outcome} (native exits 0)"),
            ),
            Ok((got, _)) if got == want => {
                eprintln!(
                    "  {name}: ok in {}ms ({:?})",
                    started.elapsed().as_millis(),
                    elide(&got)
                )
            }
            Ok((got, outcome)) => failures.push(format!(
                "{name}: OUTPUT DIFFERS ({outcome})\n       temen: {:?}\n      native: {:?}",
                elide(&got),
                elide(&want)
            )),
            Err(e) => failures.push(format!("{name}: {e}  (native prints {:?})", elide(&want))),
        }
    }
    for case in &gaps {
        let name = case.file_stem().unwrap().to_string_lossy().to_string();
        let src = std::fs::read_to_string(case).expect("read case");
        let Ok(want) = native_output(&path, &name, &src) else {
            failures.push(format!(
                "known_gaps/{name}: does not run under native nimony either"
            ));
            continue;
        };
        match temen_output(&path, &libc, &name, &src) {
            Ok((got, _)) if got == want => failures.push(format!(
                "known_gaps/{name}: NOW MATCHES native — the gap is fixed. Move it into \
                 tests/nim_diff/ and close the issue named in its header."
            )),
            Ok(_) | Err(_) => eprintln!("  known_gaps/{name}: still diverges (expected)"),
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} corpus programs diverged from native nimony:\n  - {}",
        failures.len(),
        cases.len() + gaps.len(),
        failures.join("\n  - ")
    );
}

/// Did the Temen run end the way a normally-terminating program does — `main` returning 0, or an
/// explicit `quit(0)`? Anything else (a non-zero status from `cExit`/`cAbort`, a returned non-zero)
/// is a divergence from the native oracle, which by construction exited 0.
fn is_clean_exit(outcome: &str) -> bool {
    outcome == "Returned([I32(0)])" || outcome == "Exited(0)"
}

/// The `.nim` programs in `dir`, sorted, or empty if the directory is absent.
fn nim_cases(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut v: Vec<std::path::PathBuf> = rd
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("nim"))
        .collect();
    v.sort();
    v
}

/// Trim a long output for a readable failure message, keeping both ends.
fn elide(s: &str) -> String {
    if s.len() <= 160 {
        return s.to_string();
    }
    format!("{}…{}", &s[..100], &s[s.len() - 40..])
}

/// **A nim program reads and writes real files** through the POSIX personality — the route
/// [`temen_leng::nim_posix_runtime`] links: the compute shim plus `POSIX_OPEN_ADAPTER`, with the
/// syscalls left as retained manifest imports the host binds to `temen_posix`'s fd ops over an
/// in-memory filesystem.
///
/// `link_nim_powerbox`'s bottom edge cannot do this at all (its `sysOpen` is a `{ return -1 }` stub
/// for stdout-only programs), so until now nothing on the leng route had ever opened a file. This is
/// the smallest program that does, and the gate for the ABI reconciliation the route needs: C's
/// `open` takes a NUL-terminated `char*` where every `temen_posix` path op takes `(ptr, len)`.
#[test]
fn nim_reads_and_writes_files_through_the_posix_personality() {
    let Some(path) = toolchain_path() else {
        eprintln!("SKIP: nimony toolchain not found (set NIMONY_BIN/NIM_BIN or install on PATH)");
        return;
    };
    let mods = compile_to_leng(
        &path,
        "import std/syncio\n\
         try:\n\
         \x20 let s = readFile(\"/in.txt\")\n\
         \x20 write(stdout, s)\n\
         \x20 writeFile(\"/out.txt\", s & \"!\")\n\
         except:\n\
         \x20 write(stdout, \"IO FAILED\")\n",
    );
    let units: Vec<temen_leng::WholeModule> = mods
        .iter()
        .map(|(stem, src)| temen_leng::WholeModule { stem, src })
        .collect();
    let runtime = temen_leng::nim_posix_runtime(&units).expect("nim posix runtime");
    let m = temen_leng::link_whole_powerbox_manifest(&units, runtime)
        .unwrap_or_else(|e| panic!("posix-route link: {e}"));
    temen_verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    let posix = run_io_capture(
        &m,
        temen_run::Backend::TreeWalk,
        &temen_run::RunConfig::default(),
        &[("/in.txt", b"hello from the memfs")],
    );
    assert_eq!(
        String::from_utf8_lossy(&posix.stdout()),
        "hello from the memfs",
        "the guest read a seeded memfs file and printed it"
    );
    assert_eq!(
        posix.read_file("/out.txt").as_deref(),
        Some(b"hello from the memfs!".as_slice()),
        "the guest wrote a new memfs file"
    );
}

/// **#763 spike — `nifler2` through the no-C path.** Today's `nifler.temen` is built
/// `nifler.nim → (stock nim c) → C → clang → bitcode → temen-llvm-translate`, and that clang hop is
/// exactly the "no C compiler" dependency the capstone exists to remove. v0.6.2 ships **nifler2**,
/// which hexer's own builder calls "a NIMONY program" — it has no stock-compiler dependency, so it
/// can go `nimony c → Leng → link_nim_powerbox` with no C anywhere.
///
/// This compiles the real `src/nifler2/nifler2.nim` **in the nimony tree** (its imports are relative,
/// so it cannot be copied into a scratch dir the way [`compile_to_leng`] does for a source string)
/// and links its whole `.x.nif` closure. It is the same route as the corpus — `collect_x_nif` then
/// `link_nim_powerbox` — parameterized by *where the source lives*, not a second copy of it.
///
/// Gated on `NIM_NIFLER2=1`: the compile is minutes and the closure is 10× the corpus, far past what
/// the per-PR suite should carry. Reports how far it gets rather than asserting, until it lands.
///
/// **Where it stands:** it runs the real parse. `nifler2 parse /in.nim /out.nif` reads the seeded
/// memfs file and writes a `/out.nif` whose header is byte-identical to the native nifler2's. The
/// body is not: the guest trips `[Assertion Failure] beginRead with unclosed tags` and emits the
/// header alone, where native emits the parsed `(stmts …)` — so `parseModule` leaves a tag open on
/// Temen. That is a translator correctness bug, and this is its reproducer.
///
/// Getting here took three edges, each of which looked like the last one's cause: a `MemoryFault`
/// that was really `argc = 0` (`_start` passed no argv, so `getopt()` saw nothing); then
/// `cannot read the input file`, which was `link_nim_powerbox`'s stdout-only `sysOpen` stub; then
/// the same message again from `getcwd` returning NULL. Two memory hypotheses were tested and
/// disproved along the way (run window 64 MiB/256 MiB/1 GiB, heap 11 MiB → 251 MiB, all identical),
/// which is what pointed at the bottom edge rather than the sizing.
///
/// **Knobs**, for bisecting from the outside: `NIM_NIFLER2_SRC` picks a different in-tree program
/// (a smaller probe against the same parser), `NIM_NIFLER2_ARGS` the guest's argv, `NIM_NIFLER2_SL`
/// its window.
#[test]
fn nifler2_links_through_leng() {
    if std::env::var("NIM_NIFLER2").is_err() {
        eprintln!("SKIP nifler2_links_through_leng (set NIM_NIFLER2=1)");
        return;
    }
    let Some(path) = toolchain_path() else {
        eprintln!("SKIP nifler2_links_through_leng (no nimony toolchain)");
        return;
    };
    // The nimony source root is the parent of the `bin/` the toolchain lives in.
    let bin = std::env::var("NIMONY_BIN").expect("NIMONY_BIN");
    let root = std::path::Path::new(&bin).parent().expect("nimony root");
    // `NIM_NIFLER2_SRC` points at a different program in the same tree — a smaller probe against the
    // same parser, compiled and linked by this one route rather than a copy of it.
    let rel = std::env::var("NIM_NIFLER2_SRC").unwrap_or_else(|_| "src/nifler2/nifler2.nim".into());
    let src = root.join(&rel);
    if !src.exists() {
        eprintln!("SKIP nifler2_links_through_leng ({src:?} absent — v0.6.2+ only)");
        return;
    }
    let out = std::env::temp_dir().join("nifler2_spike_bin");
    let started = std::time::Instant::now();
    let st = Command::new("nimony")
        .args([
            "c",
            "-d:release",
            "--silentMake",
            &format!("--out:{}", out.display()),
            &rel,
        ])
        .current_dir(root)
        .env("PATH", &path)
        .output()
        .expect("run nimony on nifler2");
    assert!(
        st.status.success(),
        "nimony c nifler2 failed:\n{}\n{}",
        String::from_utf8_lossy(&st.stdout),
        String::from_utf8_lossy(&st.stderr)
    );
    eprintln!("  nifler2: nimony c ok in {}s", started.elapsed().as_secs());

    let mut mods = Vec::new();
    collect_x_nif(&root.join("nimcache"), &mut mods);
    eprintln!("  nifler2: {} modules in the Leng closure", mods.len());
    let units: Vec<temen_leng::WholeModule> = mods
        .iter()
        .map(|(stem, src)| temen_leng::WholeModule { stem, src })
        .collect();
    // The **POSIX personality route**, not `link_nim_powerbox`: that one's `SYSCALL_ADAPTER` is a
    // stdout-only bottom edge (`sysOpen` → `-1`, `sysRead` → `0`), so a program that reads a file
    // cannot work on it at all. Linking against the compute shim *alone* leaves the true syscalls as
    // retained manifest imports, which `run_io_capture` binds to `temen_posix`'s real fd ops over an
    // in-memory filesystem — the same route `run_io_program` already uses for stdout, with files.
    let mut runtime = temen_leng::nim_posix_runtime(&units).expect("nim posix runtime");
    if let Some(libc) = guest_libc() {
        runtime.extend(temen_leng::nim_libc_units(&libc, &units).expect("guest libc units"));
    }
    match temen_leng::link_whole_powerbox_manifest(&units, runtime) {
        Ok(m) => {
            dump_module("nifler2", &m);
            let v = temen_verify::verify_module(&m);
            eprintln!(
                "  nifler2: LINKED — {} funcs, {} imports, verify {:?}",
                m.funcs.len(),
                m.imports.len(),
                v.map(|_| "ok")
            );
            // The five raw syscalls are *meant* to survive here — the POSIX personality binds them
            // by name at instantiation. Anything else is a leaf nothing serves.
            let unbound: Vec<&str> = m
                .imports
                .iter()
                .map(|i| i.name.as_str())
                .filter(|n| {
                    // `write` is the guest libc's §3e STREAM cap; `open` is the POSIX open adapter's
                    // forward. The rest are nimony syscall leaves. All bound at instantiation.
                    !["write", "open"].contains(n)
                        && !["sysWrite", "sysRead", "sysClose", "sysLseek", "getcwd"]
                            .iter()
                            .any(|p| n.starts_with(p))
                })
                .collect();
            eprintln!(
                "  nifler2: imports: {:?}",
                m.imports.iter().map(|i| &i.name).collect::<Vec<_>>()
            );
            if !unbound.is_empty() {
                eprintln!("  nifler2: unbound leaves: {}", unbound.join(", "));
                return;
            }
            if let Some(mem) = m.memory.as_ref() {
                let win = 1u64 << mem.size_log2;
                let brk = temen_ir::powerbox_entry_sp(&m) + temen_ir::POWERBOX_STACK_RESERVE;
                eprintln!(
                    "  nifler2: window 2^{} = {} MiB · heap [{}, {}) = {} MiB",
                    mem.size_log2,
                    win >> 20,
                    brk,
                    win,
                    (win - brk) >> 20
                );
            }
            nifler2_run_vs_native(&m, &out);
        }
        Err(e) => eprintln!("  nifler2: LINK FAILED — {e}"),
    }
}

/// Drive the Temen-linked nifler2 over an in-memory fs and diff its `.nif` against the **native**
/// nifler2 binary the same `nimony c` just produced — the same oracle shape as
/// `temen-run/tests/nifler_asset.rs`, which does this for the LLVM-built `nifler.temen`.
///
/// The guest runs on the **POSIX personality** ([`run_io_capture`]): the retained `sysOpen`/
/// `sysRead`/`sysWrite`/`sysClose`/`sysLseek` leaves bind to `temen_posix`'s real fd ops over an
/// in-memory filesystem seeded with the input, and the argv `_start` now marshals reaches nifler2's
/// own `paramStr` — the two things this route needed that the stdout-only `link_nim_powerbox` edge
/// could never supply.
#[cfg(test)]
fn nifler2_run_vs_native(m: &temen_ir::Module, native_bin: &std::path::Path) {
    use temen_run::{Backend, Limits, RunConfig};
    const SRC: &str = "let x = 5\n";

    // Native oracle: run in a scratch cwd with the input named `in.nim`, so the path nifler2 embeds
    // in the NIF header matches what the guest sees at `/in.nim` (`fs::norm` strips the leading `/`).
    let dir = std::env::temp_dir().join("nifler2_oracle");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("mk oracle dir");
    std::fs::write(dir.join("in.nim"), SRC).expect("write in.nim");
    // The oracle runs the **same argv** the guest gets (minus `argv[0]`), so a probe program with no
    // arguments is its own oracle on stdout and `nifler2 parse` is one on the written file.
    let argv: Vec<String> = std::env::var("NIM_NIFLER2_ARGS")
        .unwrap_or_else(|_| "nifler2 parse in.nim out.nif".into())
        .split_whitespace()
        .skip(1)
        .map(|a| {
            a.replace("/in.nim", "in.nim")
                .replace("/out.nif", "out.nif")
        })
        .collect();
    let st = Command::new(native_bin)
        .args(&argv)
        .current_dir(&dir)
        .output()
        .expect("run native nifler2");
    if !st.status.success() {
        eprintln!(
            "  nifler2: native oracle failed: {}",
            String::from_utf8_lossy(&st.stderr)
        );
    }
    let want = std::fs::read(dir.join("out.nif")).unwrap_or_default();
    if !st.stdout.is_empty() {
        eprintln!(
            "  nifler2: native stdout {:?}",
            elide(&String::from_utf8_lossy(&st.stdout))
        );
    }

    let cfg = RunConfig {
        limits: Limits {
            fuel: None,
            ..Limits::default()
        },
        // `NIM_NIFLER2_SL` overrides the window so the need can be *measured* rather than argued —
        // the same knob #1591 wanted and did not have.
        memory_size_log2: std::env::var("NIM_NIFLER2_SL")
            .ok()
            .and_then(|v| v.parse().ok()),
        args: std::env::var("NIM_NIFLER2_ARGS")
            .unwrap_or_else(|_| "nifler2 parse /in.nim /out.nif".into())
            .split_whitespace()
            .map(|a| a.as_bytes().to_vec())
            .collect(),
        ..RunConfig::default()
    };
    let posix = run_io_capture(m, Backend::TreeWalk, &cfg, &[("/in.nim", SRC.as_bytes())]);
    let out = posix.stdout();
    if !out.is_empty() {
        eprintln!(
            "  nifler2: stdout {:?}",
            elide(&String::from_utf8_lossy(&out))
        );
    }
    if want.is_empty() {
        return; // a probe run: stdout above is the whole comparison
    }
    match posix.read_file("/out.nif") {
        None => eprintln!("  nifler2: wrote no /out.nif"),
        Some(got) if got == want => eprintln!(
            "  nifler2: ✅ BYTE-IDENTICAL to native ({} bytes) — the real Nim parser, compiled with \
             no C compiler, runs on Temen",
            got.len()
        ),
        Some(got) => eprintln!(
            "  nifler2: DIFFERS — temen {} bytes, native {} bytes\n    temen:  {:?}\n    native: {:?}",
            got.len(),
            want.len(),
            elide(&String::from_utf8_lossy(&got)),
            elide(&String::from_utf8_lossy(&want))
        ),
    }
}
