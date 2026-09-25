//! **The self-hosted nimony lane** (#1609, #763): nimony's own toolchain, compiled to Temen with
//! **no C compiler anywhere**, builds a nim program on Temen, and the program runs.
//!
//! Every tool is a `.temen` module built by `build_nim_hello_temen --posix` (nimony → Leng →
//! `temen-leng`), bound to one shared `temen_posix` personality, so they all see one memfs.
//!
//! ```text
//! nim_selfhost_lane --sh <sh.ir> --nimony <nimony.temen> --nifmake <nifmake.temen>
//!                   --nimsem <nimsem.temen> --nifler <nifler2.temen> --hexer <hexer.temen>
//!                   --temen-link <temen-link.temen> [--libc <libc.temeno>]
//!                   [--expect <native-nimcache>] [--engine E1,E2,…]
//!                   <libdir> <prog.nim> <dump-dir>
//! ```
//!
//! 1. **The whole build is nimony's own, in-guest.** `nimony t --isMain prog.nim` — the driver with
//!    its Temen backend (`patches/nimony/temen-backend.patch`) — parses the dependency graph,
//!    writes the build plan and runs **nifmake** over it. nifmake forks and execs every step in
//!    dependency order through the POSIX `/bin/sh`: **nifler** and **nimsem** per module, **hexer**
//!    (`.x.nif`) and its dead-code elimination (`.c.nif`), then **temen-link**
//!    (`demos/temen_link`), which links the whole program into `nimcache/<main>.temen/<prog>.temen`.
//!    The host seeds the sources and reads the result; it orders nothing.
//! 2. The linked module runs on a fresh personality. Its stdout goes to *our* stdout, so the caller
//!    can diff it against the native binary's.
//!
//! **Engines** (`--engine E1,E2,…`): E1 runs the build — the driver and every process it spawns.
//! Each later engine runs hexer over every module again, held to E1's bytes, and **every** engine
//! runs the program. Any engine can be E1: the JIT serves `fork`, `execve` and `waitpid` too (#1768).
//!
//! `--expect` makes the run a **test**: every artifact of every phase must be native nimony's,
//! byte for byte, when native runs the same phases built by the same compiler (see
//! [`expect_native`]) — and the module linked in-guest must be the one the host links from the same
//! `.c.nif`s. Everything the run wrote under `nimcache/` is dumped to `<dump-dir>`.

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

/// **hexer** again, on `engine`: lower every semchecked module to Leng as the driver's plan did
/// (`hexer c`, the same flags), host-sequenced. Every `.x.nif` must be the one the in-guest build
/// wrote.
fn relower(
    hexer: &temen_ir::Module,
    posix: &temen_posix::Posix,
    make: &Arc<dyn Fn() -> temen_interp::HostProc + Send + Sync>,
    main: &str,
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
        let x = if stem == main {
            argv.extend([
                "--isMain".to_string(),
                "--app:console".to_string(),
                format!("--outdir:nimcache/{stem}.temen"),
            ]);
            format!("nimcache/{stem}.temen/{stem}.x.nif")
        } else {
            format!("nimcache/{stem}.x.nif")
        };
        let built = posix
            .read_file(&x)
            .unwrap_or_else(|| panic!("the in-guest build wrote no {x}"));
        argv.push(format!("nimcache/{stem}.s.nif"));
        temen_run::nim_noc_run(
            hexer.clone(),
            posix,
            Arc::clone(make),
            &argv,
            &temen_run::ExecGrants::default(),
            engine,
        )
        .unwrap_or_else(|e| panic!("hexer on {stem}: {e}"));
        assert!(
            posix.read_file(&x).as_ref() == Some(&built),
            "hexer on {engine:?} wrote a different {x} than the in-guest build"
        );
    }
    eprintln!(
        "✅ hexer on {engine:?} relowered {} modules to the in-guest build's bytes",
        stems.len()
    );
}

/// The whole program's DCE'd Leng — what the plan hands `temen-link` — in the order it links them.
fn program_units(posix: &temen_posix::Posix, main: &str) -> Vec<(String, String)> {
    let dir = format!("/nimcache/{main}.temen/");
    let mut mods: Vec<(String, String)> = posix
        .file_names()
        .iter()
        .filter_map(|n| {
            let stem = n.strip_prefix(&dir)?.strip_suffix(".c.nif")?;
            Some((
                stem.to_string(),
                String::from_utf8_lossy(&posix.read_file(n)?).into_owned(),
            ))
        })
        .collect();
    // `temen-link`'s order: the main module first, the rest by stem.
    mods.sort_by(|a, b| (a.0 != main, &a.0).cmp(&(b.0 != main, &b.0)));
    mods
}

/// The module `temen-link` wrote in-guest must be the one the host links from the same `.c.nif`s
/// with the same call ([`temen_leng::link_nim_posix`]) — byte for byte, encoded.
fn expect_host_link(posix: &temen_posix::Posix, main: &str, libc: Option<&[u8]>, linked: &[u8]) {
    let mods = program_units(posix, main);
    let units: Vec<temen_leng::WholeModule> = mods
        .iter()
        .map(|(stem, src)| temen_leng::WholeModule { stem, src })
        .collect();
    let (names, sigs) = temen_posix::cap_vtable();
    let host = temen_leng::link_nim_posix(&units, (&names, &sigs), libc)
        .unwrap_or_else(|e| panic!("the host link: {e}"));
    assert!(
        temen_encode::encode_module(&host) == linked,
        "temen-link's module differs from the host's link of the same {} modules",
        units.len()
    );
    eprintln!(
        "✅ temen-link's module is the host's link of the same {} modules, byte for byte",
        units.len()
    );
}

/// Run the linked program on a fresh personality; its stdout.
fn run_program(module: &temen_ir::Module, engine: temen_run::Backend) -> Vec<u8> {
    let (run, make) = temen_posix::cap(0, 0, Vec::new());
    temen_run::nim_noc_run(
        module.clone(),
        &run,
        Arc::new(make),
        &["prog".to_string()],
        &temen_run::ExecGrants::default(),
        engine,
    )
    .unwrap_or_else(|e| panic!("the program failed on {engine:?}: {e}"));
    run.stdout()
}

/// `--expect`: the run's artifacts must be native nimony's, **byte for byte** — nothing is
/// normalized and nothing is tolerated.
///
/// * every `.s.nif`, `.x.nif` and `.c.nif` native wrote must be here and the same;
/// * every `.p.nif` *this run* wrote must be native's. Native's driver also parses files nothing
///   ends up including, so its set is the larger one.
///
/// Native builds with the C backend (`nimony c`), whose pipeline the Temen backend shares up to the
/// `.c.nif`: the same target, the same flags. The one difference is where the main module's
/// backend artifacts go — `nimcache/<main>/` there, `nimcache/<main>.temen/` here — so that directory
/// is compared across.
///
/// Two conditions make exactness the right bar. The lane script runs native nimony from a tree laid
/// out as this memfs is (`lib/` at its cwd), so both record the same paths. And its phases are the
/// ones **nimony** built natively from the same sources as these Temen builds
/// (`build_nim_hello_temen --native`): the same compiler, only the target differs. `nimony/bin`'s
/// own phases are built by classic Nim, a different compiler whose hash tables iterate in a
/// different order, and hexer's output shows it (#1753).
fn expect_native(posix: &temen_posix::Posix, native: &str, main: &str) {
    let mut theirs: Vec<(String, Vec<u8>)> = Vec::new();
    collect(Path::new(native), "", &mut theirs);
    let native_dir = format!("{main}/");
    let ours = |rel: &str| match rel.strip_prefix(&native_dir) {
        Some(file) => format!("{main}.temen/{file}"),
        None => rel.to_string(),
    };
    let theirs: std::collections::HashMap<String, Vec<u8>> =
        theirs.into_iter().map(|(rel, b)| (ours(&rel), b)).collect();
    let mut compared: Vec<String> = theirs
        .keys()
        .filter(|rel| {
            [".s.nif", ".x.nif", ".c.nif"]
                .iter()
                .any(|e| rel.ends_with(e))
        })
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
        compared.iter().any(|rel| rel.ends_with(".c.nif")),
        "{native} holds no .c.nif — not a `nimony c` nimcache?"
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
        "✅ matches native nimony byte for byte: {} .p.nif, {} .s.nif, {} .x.nif, {} .c.nif",
        count(".p.nif"),
        count(".s.nif"),
        count(".x.nif"),
        count(".c.nif")
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
    let hexer_p = need(flag("--hexer"), "--hexer");
    let link_p = need(flag("--temen-link"), "--temen-link");
    // The prebuilt guest libc (`snprintf`, `strtod`, libm): what `temen-link` links a program
    // against, found where `build_nim_hello_temen` finds it.
    let libc_p = flag("--libc")
        .or_else(|| std::env::var("TEMEN_PG_LIBC").ok())
        .unwrap_or_else(|| "browser/web/assets/pg_libc.temeno".to_string());
    let expect = flag("--expect");
    // `--engine E1,E2,…` (`tree`, `bytecode`, `jit`; default `tree`): E1 runs the build — the driver
    // and every process it spawns; each later engine reruns hexer, and every engine runs the
    // program.
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
    let [libdir, prog, dump] = &a[..] else {
        panic!(
            "usage: nim_selfhost_lane --sh <sh.ir> --nimony <nimony.temen> --nifmake \
             <nifmake.temen> --nimsem <nimsem.temen> --nifler <nifler2.temen> --hexer \
             <hexer.temen> --temen-link <temen-link.temen> [--libc <libc.temeno>] \
             [--expect <native-nimcache>] [--engine E1,E2,…] <libdir> <prog.nim> <dump-dir>"
        );
    };

    // The command registry: `/bin/sh`, and each tool at `bin/<name>` — where the driver's
    // `findTool` looks, beside its own `bin/nimony` — and at `/bin/<name>`, where the shell's PATH
    // walk finds a bare name.
    let hexer = phase(&hexer_p, "hexer");
    let mut commands: Vec<(String, temen_ir::Module)> =
        vec![("/bin/sh".to_string(), command_module(&sh_ir, "/bin/sh"))];
    for (name, m) in [
        ("nifmake", phase(&nifmake_p, "nifmake")),
        ("nimsem", phase(&nimsem_p, "nimsem")),
        ("nifler2", phase(&nifler_p, "nifler2")),
        ("hexer", hexer.clone()),
        ("temen-link", phase(&link_p, "temen-link")),
    ] {
        commands.push((format!("bin/{name}"), m.clone()));
        commands.push((format!("/bin/{name}"), m));
    }

    // One personality for every process: one memfs, one fd table, one stdout.
    let (posix, make) = temen_posix::cap(0, 0, Vec::new());
    let make: Arc<dyn Fn() -> temen_interp::HostProc + Send + Sync> = Arc::new(make);
    posix.set_env("PATH", "/bin");

    // Seed the stdlib under `lib/`, where native nimony has it beside its cwd, the guest libc where
    // `temen-link` looks for it (`/lib/temen/libc.temeno`), and the program at the root: every
    // artifact is derived in-guest. Sources go in first so everything a phase derives from them is
    // genuinely newer — the memfs stamps write order into `st_mtim`, which is what the freshness
    // checks in `deps.nim` and nifmake read.
    let mut seed: Vec<(String, Vec<u8>)> = Vec::new();
    collect(Path::new(libdir), "lib/", &mut seed);
    let libc = std::fs::read(&libc_p).unwrap_or_else(|e| panic!("read {libc_p}: {e}"));
    seed.push(("/lib/temen/libc.temeno".to_string(), libc.clone()));
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

    // 1. The build, run by nimony's own driver with its Temen backend.
    let argv: Vec<String> = ["bin/nimony", "t", "--isMain", &prog_name]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let t0 = std::time::Instant::now();
    // The toolchain may run what it builds (#763): nimony's compile-time evaluation builds a
    // program and execs it.
    let outcome = temen_run::nim_noc_run(
        phase(&nimony_p, "nimony"),
        &posix,
        Arc::clone(&make),
        &argv,
        &temen_run::ExecGrants {
            commands: &commands,
            built: true,
        },
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
    let out_name = Path::new(&prog_name)
        .with_extension("temen")
        .to_string_lossy()
        .into_owned();
    let linked = main_stem
        .as_ref()
        .and_then(|m| posix.read_file(&format!("nimcache/{m}.temen/{out_name}")));
    dump_cache(&posix, dump);
    let (Ok(()), Some(main_stem), Some(linked)) = (&outcome, main_stem, linked) else {
        eprint!(
            "--- stdout ---\n{}--- stderr ---\n{}",
            String::from_utf8_lossy(&posix.stdout()),
            String::from_utf8_lossy(&posix.stderr())
        );
        // #1665 — a command that crashed reaps as status 128; this is where its trap survives.
        for t in temen_interp::last_twin_traps() {
            eprintln!("--- a command crashed ---\n{t}");
        }
        panic!("`nimony t` did not build {prog_name}: {outcome:?}");
    };
    let modules = program_units(&posix, &main_stem).len();
    eprintln!(
        "✅ nimony's driver built {prog_name} in-guest in {:.0?} on {engine:?}: {modules} modules \
         linked by temen-link (main: {main_stem})",
        t0.elapsed()
    );
    if let Some(native) = &expect {
        expect_native(&posix, native, &main_stem);
        expect_host_link(&posix, &main_stem, Some(&libc), &linked);
    }
    let module =
        temen_encode::decode_module(&linked).unwrap_or_else(|e| panic!("decode {out_name}: {e:?}"));
    temen_verify::verify_module(&module).unwrap_or_else(|e| panic!("verify {out_name}: {e:?}"));

    // 2. The program, on every engine; each later engine reruns hexer first.
    let mut first: Option<Vec<u8>> = None;
    for (i, &e) in engines.iter().enumerate() {
        if i > 0 {
            relower(&hexer, &posix, &make, &main_stem, e);
        }
        let stdout = run_program(&module, e);
        eprintln!(
            "✅ on {e:?}, the program ran: {} bytes of stdout",
            stdout.len()
        );
        match &first {
            None => first = Some(stdout),
            Some(f) => assert_eq!(
                String::from_utf8_lossy(f),
                String::from_utf8_lossy(&stdout),
                "the program prints something else on {e:?} than on {:?}",
                engines[0]
            ),
        }
    }
    use std::io::Write;
    std::io::stdout()
        .write_all(&first.expect("at least one engine"))
        .expect("write stdout");
}
