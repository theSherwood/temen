//! **The self-hosted nimony lane** (#1609, #763): nimony's own toolchain, compiled to Temen with
//! **no C compiler anywhere**, builds a nim program on Temen, and the program runs.
//!
//! Every tool is a `.temen` module built by `build_nim_hello_temen --posix` (nimony → Leng →
//! `temen-leng`), bound to one shared `temen_posix` personality, so they all see one memfs.
//!
//! ```text
//! nim_selfhost_lane --sh <sh.ir> --nimony <nimony.temen> --nifmake <nifmake.temen>
//!                   --nimsem <nimsem.temen> --nifler <nifler2.temen> [--hexer <hexer.temen>]
//!                   [--expect <native-nimcache>] [--engine E1,E2,…]
//!                   <libdir> <prog.nim> <dump-dir>
//! ```
//!
//! 1. **The frontend is nimony's own.** `nimony check --isMain prog.nim` runs in-guest: the driver
//!    parses the program's dependency graph and writes its build plan, then runs **nifmake** over
//!    it, which forks and execs **nifler** and **nimsem** for every node, in dependency order,
//!    through the POSIX `/bin/sh` (`startProcess` → fork → `execve("/bin/sh", ["-c", cmd])`).
//!    nimsem spawns nifler itself for the files a module includes. The host orders nothing.
//!    nifmake is built from `patches/nimony/nifmake-builds-with-nimony.patch`: upstream it is a
//!    classic-Nim-only program.
//! 2. **hexer** (`--hexer`) lowers every semchecked module to Leng (`hexer c` → `.x.nif`). This
//!    is the one phase the host still sequences: `nimony c` would run it, then its C backend
//!    (dce → lengc → cc → link), and a Temen target has no use for that.
//! 3. **temen-leng** links the `.x.nif`s ([`temen_leng::link_nim_posix`], the same link every
//!    tool here is built with), and the program runs on a fresh personality. Its stdout goes to
//!    *our* stdout, so the caller can diff it against the native binary's.
//!
//! **Engines** (`--engine E1,E2,…`): E1 runs the frontend — the driver and every process it spawns;
//! then **each** engine runs hexer and the program. Every engine is held to the same bytes, so the
//! lane is an engine differential over a real compiler with native nimony as the oracle. The JIT
//! serves no fork/exec yet (#1768), so it covers the phases that do not spawn (and cannot be E1).
//!
//! `--expect` makes the run a **test**: every artifact of every phase must be native nimony's,
//! byte for byte, when native runs the same phases built by the same compiler (see
//! [`expect_native`]). Everything the run wrote under `nimcache/` is dumped to `<dump-dir>`.

use std::path::Path;
use std::sync::Arc;

fn collect(dir: &Path, prefix: &str, out: &mut Vec<(String, Vec<u8>)>) {
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in entries.flatten() {
            let p = e.path();
            if e.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                stack.push(p);
            } else if let Ok(bytes) = std::fs::read(&p) {
                let rel = p
                    .strip_prefix(dir)
                    .unwrap_or(&p)
                    .to_string_lossy()
                    .replace('\\', "/");
                out.push((format!("{prefix}{rel}"), bytes));
            }
        }
    }
}

/// Load and verify a no-C phase, reporting its shape the way the other nim drivers do.
fn phase(path: &str, what: &str) -> temen_ir::Module {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let m = temen_encode::decode_module(&bytes).unwrap_or_else(|e| panic!("decode {path}: {e:?}"));
    temen_verify::verify_module(&m).unwrap_or_else(|e| panic!("verify {path}: {e:?}"));
    eprintln!(
        "{what}: {} funcs, {} imports, window 2^{}",
        m.funcs.len(),
        m.imports.len(),
        m.memory.as_ref().map(|x| x.size_log2).unwrap_or(0),
    );
    m
}

/// Write everything the run built under `nimcache/` to `dir`.
///
/// When a phase disagrees with native the first question is *which* one diverged, and the way to
/// ask it is to hand these exact inputs to the native tool. Dumping on both the success and the
/// failure path means the answer is already on disk when the question comes up, rather than costing
/// another full run of tree-walked compilers.
fn dump_cache(posix: &temen_posix::Posix, dir: &str) {
    if let Err(e) = std::fs::create_dir_all(dir) {
        eprintln!("note: cannot create {dir}: {e}");
        return;
    }
    let mut n = 0usize;
    for name in posix.file_names() {
        let Some(rel) = name.strip_prefix("/nimcache/") else {
            continue;
        };
        if let Some(b) = posix.read_file(&name) {
            // The main module's `.x.nif` lives a level down, in `nimcache/<main>/`.
            let to = Path::new(&dir).join(rel);
            if let Some(parent) = to.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            match std::fs::write(to, &b) {
                Ok(()) => n += 1,
                Err(e) => eprintln!("note: cannot write {rel}: {e}"),
            }
        }
    }
    eprintln!("dumped {n} nimcache files to {dir}");
}

/// Load a chibicc-emitted **command module** (IR text, `--child-entry`) — the `/bin/sh` that turns
/// nimsem's shell-out into a real `execve`. Kept out of the asset pipeline deliberately: the shell
/// is host environment, the way a kernel is, and passing it in keeps #763's "no C compiler"
/// claim about *nimony's* toolchain honest and visible rather than quietly bundled.
fn command_module(path: &str, what: &str) -> temen_ir::Module {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let m = temen_text::parse_module(&text).unwrap_or_else(|e| panic!("parse {path}: {e:?}"));
    temen_verify::verify_module(&m).unwrap_or_else(|e| panic!("verify {path}: {e:?}"));
    eprintln!(
        "{what}: {} funcs, window 2^{}",
        m.funcs.len(),
        m.memory.as_ref().map(|x| x.size_log2).unwrap_or(0),
    );
    m
}

/// **hexer**: lower every semchecked module to Leng — `hexer c` once per module, as native nimony's
/// plan runs it (`<main>.final.build.nif`), with the same flags so each `.x.nif` is comparable to
/// native's. The main module's goes under `nimcache/<main>/`, where native puts it.
fn lower(
    hexer: &temen_ir::Module,
    posix: &temen_posix::Posix,
    make: &Arc<dyn Fn() -> temen_interp::HostProc + Send + Sync>,
    main: Option<&String>,
    engine: temen_run::Backend,
) {
    let mut stems: Vec<String> = posix
        .file_names()
        .iter()
        .filter_map(|n| n.strip_prefix("/nimcache/")?.strip_suffix(".s.nif"))
        .filter(|s| !s.contains('/'))
        .map(|s| s.to_string())
        .collect();
    stems.sort();
    for stem in &stems {
        let mut argv: Vec<String> = [
            "hexer",
            "c",
            "--bits:64",
            "--cpu:le",
            "--os:Linux",
            "--flags:br",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let x = if Some(stem) == main {
            argv.extend([
                "--isMain".to_string(),
                "--app:console".to_string(),
                format!("--outdir:nimcache/{stem}"),
            ]);
            format!("nimcache/{stem}/{stem}.x.nif")
        } else {
            format!("nimcache/{stem}.x.nif")
        };
        argv.push(format!("nimcache/{stem}.s.nif"));
        temen_run::nim_noc_run(hexer.clone(), posix, Arc::clone(make), &argv, &[], engine)
            .unwrap_or_else(|e| panic!("hexer on {stem}: {e}"));
        assert!(
            posix.read_file(&x).is_some(),
            "hexer returned cleanly but wrote no {x}"
        );
    }
    eprintln!(
        "✅ hexer lowered {} modules to Leng on {engine:?}",
        stems.len()
    );
}

/// **Link and run**: the `.x.nif`s hexer wrote, linked by [`temen_leng::link_nim_posix`] — the one
/// link every nim phase is itself built with — then run on a fresh personality. Returns its stdout.
fn link_and_run(posix: &temen_posix::Posix, engine: temen_run::Backend) -> Vec<u8> {
    let mut mods: Vec<(String, String)> = posix
        .file_names()
        .iter()
        .filter(|n| n.starts_with("/nimcache/"))
        .filter_map(|n| {
            let stem = Path::new(n).file_name()?.to_str()?.strip_suffix(".x.nif")?;
            Some((
                stem.to_string(),
                String::from_utf8_lossy(&posix.read_file(n)?).into_owned(),
            ))
        })
        .collect();
    // Program first, `system` last: the order the nim e2e tests link I/O programs in.
    mods.sort_by_key(|(stem, _)| stem.starts_with("sysv"));
    let units: Vec<temen_leng::WholeModule> = mods
        .iter()
        .map(|(stem, src)| temen_leng::WholeModule { stem, src })
        .collect();
    // The prebuilt guest libc, found where `build_nim_hello_temen` finds it.
    let libc = std::fs::read(
        std::env::var("TEMEN_PG_LIBC")
            .unwrap_or_else(|_| "browser/web/assets/pg_libc.temeno".to_string()),
    )
    .ok();
    let (px_names, px_sigs) = temen_posix::cap_vtable();
    let module = temen_leng::link_nim_posix(&units, (&px_names, &px_sigs), libc.as_deref())
        .unwrap_or_else(|e| panic!("link the program: {e}"));
    temen_verify::verify_module(&module).unwrap_or_else(|e| panic!("verify the program: {e:?}"));
    eprintln!(
        "linked the program: {} modules, {} funcs",
        units.len(),
        module.funcs.len()
    );
    let (run, make) = temen_posix::cap(0, 0, Vec::new());
    temen_run::nim_noc_run(
        module,
        &run,
        Arc::new(make),
        &["prog".to_string()],
        &[],
        engine,
    )
    .unwrap_or_else(|e| panic!("the program failed: {e}"));
    run.stdout()
}

/// `--expect`: the run's artifacts must be native nimony's, **byte for byte** — nothing is
/// normalized and nothing is tolerated.
///
/// * every `.s.nif` native wrote (and, when hexer ran, every `.x.nif`) must be here and the same;
/// * every `.p.nif` *this run* wrote must be native's. Native's driver also parses files nothing
///   ends up including, so its set is the larger one.
///
/// Two conditions make exactness the right bar. The lane script runs native nimony from a tree laid
/// out as this memfs is (`lib/` at its cwd), so both record the same paths. And its phases are the
/// ones **nimony** built natively from the same sources as these Temen builds
/// (`build_nim_hello_temen --native`): the same compiler, only the target differs. `nimony/bin`'s
/// own phases are built by classic Nim, a different compiler whose hash tables iterate in a
/// different order, and hexer's output shows it (#1753).
fn expect_native(posix: &temen_posix::Posix, native: &str, lowered: bool) {
    let mut theirs: Vec<(String, Vec<u8>)> = Vec::new();
    collect(Path::new(native), "", &mut theirs);
    let theirs: std::collections::HashMap<String, Vec<u8>> = theirs.into_iter().collect();
    let mut compared: Vec<String> = theirs
        .keys()
        .filter(|rel| rel.ends_with(".s.nif") || (lowered && rel.ends_with(".x.nif")))
        .cloned()
        .collect();
    compared.extend(
        posix
            .file_names()
            .iter()
            .filter_map(|n| n.strip_prefix("/nimcache/"))
            .filter(|rel| rel.ends_with(".p.nif"))
            .map(|rel| rel.to_string()),
    );
    compared.sort();
    assert!(
        compared.iter().any(|rel| rel.ends_with(".s.nif")),
        "{native} holds no .s.nif — not a nimcache?"
    );
    for rel in &compared {
        let want = theirs
            .get(rel)
            .unwrap_or_else(|| panic!("this run wrote {rel}; native nimony did not"));
        let got = posix
            .read_file(&format!("nimcache/{rel}"))
            .unwrap_or_else(|| panic!("native nimony wrote {rel}; this run did not"));
        if got == *want {
            continue;
        }
        let (g, w) = (String::from_utf8_lossy(&got), String::from_utf8_lossy(want));
        let (gl, wl): (Vec<_>, Vec<_>) = (g.lines().collect(), w.lines().collect());
        let at = gl
            .iter()
            .zip(&wl)
            .position(|(a, b)| a != b)
            .unwrap_or(gl.len().min(wl.len()));
        let show = |v: &[&str]| v[at.saturating_sub(2)..(at + 3).min(v.len())].join("\n");
        panic!(
            "{rel} differs from native nimony's at line {}\n--- temen ---\n{}\n--- native ---\n{}",
            at + 1,
            show(&gl),
            show(&wl)
        );
    }
    let count = |ext: &str| compared.iter().filter(|r| r.ends_with(ext)).count();
    eprintln!(
        "✅ matches native nimony byte for byte: {} .p.nif, {} .s.nif, {} .x.nif",
        count(".p.nif"),
        count(".s.nif"),
        count(".x.nif")
    );
}

fn main() {
    let mut a: Vec<String> = std::env::args().skip(1).collect();
    let mut flag = |name: &str| {
        let i = a.iter().position(|t| t == name)?;
        let v = a
            .get(i + 1)
            .cloned()
            .unwrap_or_else(|| panic!("{name} needs a value"));
        a.drain(i..=i + 1);
        Some(v)
    };
    let need = |v: Option<String>, name: &str| v.unwrap_or_else(|| panic!("{name} is required"));
    let sh_ir = need(flag("--sh"), "--sh");
    let nimony_p = need(flag("--nimony"), "--nimony");
    let nifmake_p = need(flag("--nifmake"), "--nifmake");
    let nimsem_p = need(flag("--nimsem"), "--nimsem");
    let nifler_p = need(flag("--nifler"), "--nifler");
    let hexer_p = flag("--hexer");
    let expect = flag("--expect");
    // `--engine E1,E2,…` (`tree`, `bytecode`, `jit`; default `tree`): E1 runs the frontend — the
    // driver and every process it spawns — and **each** engine then runs hexer and the program,
    // every one held to the same bytes. The JIT serves no fork/exec, so it cannot be E1.
    let engines: Vec<temen_run::Backend> = flag("--engine")
        .unwrap_or_else(|| "tree".to_string())
        .split(',')
        .map(|e| match e {
            "tree" => temen_run::Backend::TreeWalk,
            "bytecode" => temen_run::Backend::Bytecode,
            "jit" => temen_run::Backend::Jit,
            e => panic!("--engine {e}: expected tree, bytecode or jit"),
        })
        .collect();
    let engine = engines[0];
    // The frontend is the stage that spawns (driver → nifmake → every phase), and the JIT serves
    // no fork/exec yet: refuse it here rather than let the driver report its first spawn failed.
    assert!(
        engine != temen_run::Backend::Jit,
        "--engine: the JIT cannot run the frontend (no fork/exec, #1768); put it after an interpreter"
    );
    let [libdir, prog, dump] = &a[..] else {
        panic!(
            "usage: nim_selfhost_lane --sh <sh.ir> --nimony <nimony.temen> --nifmake \
             <nifmake.temen> --nimsem <nimsem.temen> --nifler <nifler2.temen> \
             [--hexer <hexer.temen>] [--expect <native-nimcache>] [--engine E1,E2,…] \
             <libdir> <prog.nim> <dump-dir>"
        );
    };

    // The command registry: `/bin/sh`, and each tool at `bin/<name>` — where the driver's
    // `findTool` looks, beside its own `bin/nimony` — and at `/bin/<name>`, where the shell's PATH
    // walk finds a bare `nifmake`.
    let mut commands: Vec<(String, temen_ir::Module)> =
        vec![("/bin/sh".to_string(), command_module(&sh_ir, "/bin/sh"))];
    for (name, path) in [
        ("nifmake", &nifmake_p),
        ("nimsem", &nimsem_p),
        ("nifler2", &nifler_p),
    ] {
        let m = phase(path, name);
        commands.push((format!("bin/{name}"), m.clone()));
        commands.push((format!("/bin/{name}"), m));
    }

    // One personality for every process: one memfs, one fd table, one stdout.
    let (posix, make) = temen_posix::cap(0, 0, Vec::new());
    let make: Arc<dyn Fn() -> temen_interp::HostProc + Send + Sync> = Arc::new(make);
    posix.set_env("PATH", "/bin");

    // Seed the stdlib under `lib/`, where native nimony has it beside its cwd, and the program at
    // the root: every artifact is derived in-guest. Sources go in first so everything a phase
    // derives from them is genuinely newer — the memfs stamps write order into `st_mtim`, which is
    // what the freshness checks in `deps.nim` and nifmake read.
    let mut seed: Vec<(String, Vec<u8>)> = Vec::new();
    collect(Path::new(libdir), "lib/", &mut seed);
    let prog_name = Path::new(prog)
        .file_name()
        .and_then(|n| n.to_str())
        .expect("utf-8 program name")
        .to_string();
    seed.push((
        prog_name.clone(),
        std::fs::read(prog).unwrap_or_else(|e| panic!("read {prog}: {e}")),
    ));
    for (k, v) in &seed {
        posix.write_file(k, v);
    }
    eprintln!("seeded {} files", seed.len());

    // 1. The frontend, run by nimony's own driver.
    let argv: Vec<String> = ["bin/nimony", "check", "--isMain", &prog_name]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let t0 = std::time::Instant::now();
    let outcome = temen_run::nim_noc_run(
        phase(&nimony_p, "nimony"),
        &posix,
        Arc::clone(&make),
        &argv,
        &commands,
        engine,
    );
    // The driver quits with `FAILURE: <cmd>` on any failed step; that is an ordinary exit, so ask
    // the memfs what was built. The build plan names the main module: `nimcache/<main>.build.nif`.
    let main_stem = posix.file_names().iter().find_map(|n| {
        n.strip_prefix("/nimcache/")?
            .strip_suffix(".build.nif")
            .filter(|s| !s.contains('.') && !s.contains('/'))
            .map(|s| s.to_string())
    });
    let built = main_stem
        .as_ref()
        .is_some_and(|m| posix.read_file(&format!("nimcache/{m}.s.nif")).is_some());
    if outcome.is_err() || !built {
        eprint!(
            "--- stdout ---\n{}--- stderr ---\n{}",
            String::from_utf8_lossy(&posix.stdout()),
            String::from_utf8_lossy(&posix.stderr())
        );
        // #1665 — a command that crashed reaps as status 128; this is where its trap survives.
        for t in temen_interp::last_twin_traps() {
            eprintln!("--- a command crashed ---\n{t}");
        }
        dump_cache(&posix, dump);
        panic!("`nimony check` did not semcheck {prog_name}: {outcome:?}");
    }
    let main_stem = main_stem.expect("built");
    let semchecked = posix
        .file_names()
        .iter()
        .filter(|n| n.starts_with("/nimcache/") && n.ends_with(".s.nif"))
        .count();
    eprintln!(
        "✅ nimony's driver semchecked {semchecked} modules in-guest in {:.0?} on {engine:?} \
         (main: {main_stem})",
        t0.elapsed()
    );

    // 2–3. hexer, then link and run — once per engine, each held to native.
    let Some(hp) = &hexer_p else {
        dump_cache(&posix, dump);
        if let Some(native) = &expect {
            expect_native(&posix, native, false);
        }
        return;
    };
    let hexer = phase(hp, "hexer");
    let mut first: Option<Vec<u8>> = None;
    for &e in &engines {
        lower(&hexer, &posix, &make, Some(&main_stem), e);
        dump_cache(&posix, dump);
        if let Some(native) = &expect {
            expect_native(&posix, native, true);
        }
        let stdout = link_and_run(&posix, e);
        eprintln!(
            "✅ on {e:?}, the program ran: {} bytes of stdout",
            stdout.len()
        );
        match &first {
            None => first = Some(stdout),
            Some(f) => assert_eq!(
                String::from_utf8_lossy(f),
                String::from_utf8_lossy(&stdout),
                "the program built on {e:?} prints something other than on {:?}",
                engines[0]
            ),
        }
    }
    use std::io::Write;
    std::io::stdout()
        .write_all(&first.expect("at least one engine"))
        .expect("write stdout");
}
