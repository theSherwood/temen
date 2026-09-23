//! **The self-hosted nimony lane** (#1609, #763): nimony's own phases, compiled to Temen with **no
//! C compiler anywhere**, build a nim program on Temen, and the program runs.
//!
//! Every phase is a `.temen` module built by `build_nim_hello_temen --posix` (nimony → Leng →
//! `temen-leng`), bound to one shared `temen_posix` personality, so they all see one memfs: the
//! `.p.nif` nifler2 writes is the one nimsem reads back, and the `.s.nif` nimsem writes is hexer's
//! input.
//!
//! ```text
//! nim_selfhost_lane [--sh <sh.ir>] [--check <prog.nim | lib/std/x.nim>] [--hexer <hexer.temen>]
//!                   [--expect <native-nimcache>]
//!                   <nimsem.temen> <nifler2.temen> <libdir> <sys.p.nif> <sys-stem> <out.s.nif>
//! ```
//!
//! What runs, in the order `nimony c --isMain prog.nim` runs it (its `.build.nif` plan):
//!
//! 1. **nifler** parses every module. nimsem spawns it itself for the files a module *includes*
//!    (`system`'s two dozen, below); an *imported* module has a parse step of its own in the plan.
//! 2. **nimsem** semchecks `system`, then each dependency, then the program (`--isMain`).
//! 3. **hexer** (`--hexer`) lowers every semchecked module to Leng (`hexer c` → `.x.nif`).
//! 4. **temen-leng** links the `.x.nif`s ([`temen_leng::link_nim_posix`], the same link every nim
//!    phase is built with), and the program runs on a fresh personality. Its stdout goes to *our*
//!    stdout, so the caller can diff it against the native binary's.
//!
//! The host still does two jobs a real build gives to nimony's own tools. It **orders** the phases,
//! running each module's parse and semcheck steps when nimsem names the module as missing:
//! nifmake's job, driven by the plan `nimony c` writes. It **links**: temen-leng, Temen's own
//! backend, where native nimony runs `dce` → lengc → a C compiler. `lengc` is nimony's *C* emitter
//! and has no place on a Temen target. Running nimony's driver and nifmake in-guest is the next step.
//!
//! `--expect` makes the run a **test**: every `.s.nif` (and, with `--hexer`, every `.x.nif`) native
//! nimony wrote must exist here and match (byte for byte; see [`expect_native`]).
//!
//! **The shell-out.** `nimsem m` shells out to nifler for a file it needs parsed and finds no
//! current `.p.nif` for — in practice the files a module includes (`deps.nim`'s `execNifler` →
//! `os.execShellCmd` → `fork` + `execve("/bin/sh", ["-c", cmd])` + `waitpid`). There are two ways
//! to serve it:
//!
//! * **`--sh <sh.ir>`, the real path.** Registers a shell at `/bin/sh` and nifler2 at the spellings
//!   nimsem uses, and nimsem runs its own fork/exec/wait sequence. The shell must be the **POSIX
//!   build** of `demos/shell` — the one that runs a command as a process of the same personality,
//!   inheriting its fds, rather than as a §14 child with a grant list (#1662):
//!
//!   ```text
//!   cat demos/shell/{shim,ring,shell_main}.c > sh.c
//!   chibicc -cc1 --emit-ir --child-entry -DTEMEN_SHELL_POSIX -cc1-input sh.c -cc1-output sh.ir sh.c
//!   ```
//!
//!   `sh -c "bin/nifler …"` is one simple command, so the shell execs nifler in place: nimsem's
//!   shell-out is one fork and two image-replaces, and nimsem reaps nifler's own status.
//!
//!   **This is the working path** (#1668). nimsem runs its own shell-outs, and the result is the
//!   native compiler's, byte for byte. That needed every POSIX import of
//!   a nim program to be the personality's own name (`__px_*`), so an `execve`'d nifler binds exactly
//!   as a C command does, and exec to admit a powerbox `_start`.
//!
//! * **Without `--sh`, a hand-crank** — kept as the fallback and as a differential against the real
//!   path. Run nimsem; when it quits with `FAILURE: nifler … parse <src>
//!   <out>`, run nifler2 over the same memfs with exactly those arguments; run nimsem again. The
//!   guest names the file it wants, so nothing here guesses a path or a cache stem. It converges
//!   because each round makes one more dependency current, and it refuses to spin: a command it has
//!   already served, served again, is a hang, not progress. Every phase is still a no-C Temen
//!   module doing the real work — but the driver sequences them; nimsem is not spawning nifler.
//!
//! **A live cross-check on the cache-stem port.** `temen_run::nim_module_suffix` reimplements
//! nimony's `moduleSuffix`. Every request the guest makes is a chance to check it against the real
//! thing, so the driver computes the stem it would have predicted and reports a mismatch. It does
//! not fail on one: the guest's own answer is authoritative here, and a wrong prediction is a fact
//! about the port (most likely its search-path model), worth printing rather than dying on.

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

/// The `nifler … parse <src> <out>` argv inside a `FAILURE: <cmd>` line the guest quit with, if the
/// last thing it said was a shell-out it could not perform. `quoteShell` may have quoted the paths.
fn failed_nifler_command(out: &str) -> Option<Vec<String>> {
    let line = out
        .lines()
        .rev()
        .find(|l| l.starts_with("FAILURE: ") && l.contains("nifler"))?;
    let argv: Vec<String> = line["FAILURE: ".len()..]
        .split_whitespace()
        .map(|t| t.trim_matches(|c| c == '\'' || c == '"').to_string())
        .collect();
    (argv.len() >= 3).then_some(argv)
}

/// The `<stem>` in a `vfs: open failed: nimcache/<stem>.s.nif` line — a dependency nimsem needs
/// semchecked before it can continue. Last such line wins, as with the shell-out parser.
fn missing_semchecked(out: &str) -> Option<String> {
    out.lines()
        .rev()
        .find_map(|l| l.trim().strip_prefix("vfs: open failed: nimcache/"))
        .and_then(|r| r.strip_suffix(".s.nif"))
        .map(|s| s.to_string())
}

/// Write everything the run built under `nimcache/` to `<out>.cache`.
///
/// When the two nimsems disagree the first question is *which* phase diverged, and the only way to
/// ask it is to hand these exact `.p.nif` files to the native nimsem: if it errors on them too, the
/// parse is wrong; if it accepts them, the semantic check is. Dumping on both the success and the
/// give-up path means the answer is already on disk when the question comes up, rather than costing
/// another full run of a tree-walked 12,725-function compiler.
fn dump_cache(posix: &temen_posix::Posix, out_p: &str) {
    let dir = format!("{out_p}.cache");
    if let Err(e) = std::fs::create_dir_all(&dir) {
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

/// `.x.nif` declarations in a canonical order: the top-level items sorted, and the trailing index's
/// entries with their byte offsets masked (an offset is where an item landed, so order moves it).
///
/// For the one known divergence only (#1753): hexer compiled to Temen emits some of a
/// module's declarations in a different order than native hexer over the same input. Same items,
/// same bytes each. An `.x.nif` is compared exactly first; this is the fallback, and it is reported.
fn declarations(text: &str) -> Vec<String> {
    // `(.index@` — not `(.index`, which also matches the `(.indexat …)` header on line 2.
    let (body, index) = text.split_at(text.find("(.index@").unwrap_or(text.len()));
    let mut items: Vec<String> = Vec::new();
    for l in body.lines() {
        // A top-level item opens at one space of indent; everything deeper continues the last one.
        match items.last_mut() {
            Some(last) if !l.starts_with(" (") => {
                last.push('\n');
                last.push_str(l);
            }
            _ => items.push(l.to_string()),
        }
    }
    items.extend(index.lines().map(|l| {
        // ` (h <sym> 130)` → ` (h <sym> N)`, keeping however many parens close the line.
        let t = l.trim_end_matches(')');
        match t.rsplit_once(' ') {
            Some((head, n)) if !n.is_empty() && n.bytes().all(|c| c.is_ascii_digit()) => {
                format!("{head} N{}", &l[t.len()..])
            }
            _ => l.to_string(),
        }
    }));
    items.sort();
    items
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

/// Run nimsem until it produces `produced`: parse and semcheck the dependencies it names as missing
/// first (nifmake's job), and — only without `--sh` — serve its own nifler shell-outs from outside.
///
/// Success is **the artifact**, not the return: nim's `quit("FAILURE: …")` is an ordinary exit, so
/// a run that gave up looks exactly like one that finished. Ask the memfs instead.
///
/// One loop, two callers: `system` always, then `--check`'s target. A second copy would be a
/// second answer to "did this module check?", and the shell-out bookkeeping — the no-progress
/// guard especially — is the part worth having once.
#[allow(clippy::too_many_arguments)]
fn semcheck(
    nimsem: &temen_ir::Module,
    nifler: &temen_ir::Module,
    posix: &temen_posix::Posix,
    make: &Arc<dyn Fn() -> temen_interp::HostProc + Send + Sync>,
    commands: &[(String, temen_ir::Module)],
    nimsem_argv: &[String],
    produced: &str,
    out_p: &str,
    sources: &std::collections::HashMap<String, String>,
) -> usize {
    let mut served: Vec<String> = Vec::new();
    loop {
        let before = posix.stdout().len();
        let before_err = posix.stderr().len();
        let outcome = temen_run::nim_noc_run_with_commands(
            nimsem.clone(),
            posix,
            Arc::clone(make),
            nimsem_argv,
            commands,
        );
        let said = String::from_utf8_lossy(&posix.stdout()[before..]).into_owned();
        // A guest says *why* it could not run something on **stderr**, and nothing else here looks
        // there — so an exec that fails inside `/bin/sh` reads as a bare `FAILURE: bin/nifler …`
        // from nimsem with the actual diagnosis thrown away.
        let said_err = String::from_utf8_lossy(&posix.stderr()[before_err..]).into_owned();
        if posix.read_file(produced).is_some() {
            return served.len();
        }
        // A dependency that is parsed but not yet **semchecked**: `vfs: open failed:
        // nimcache/<stem>.s.nif`. The real toolchain has `nifmake` walking the graph in
        // topological order; here the guest names what it is missing, so parse and semcheck that
        // module, then let the caller retry. This is nifmake's job, so it is served in both modes,
        // `--sh` included: nimsem never runs nimsem. Each round makes one more dependency
        // current, so this converges for the same reason the shell-out loop does, and the
        // no-progress guard covers both.
        if let Some(stem) = missing_semchecked(&said) {
            let Some(src) = sources.get(&stem) else {
                eprint!("--- nimsem said ---\n{said}--- stderr ---\n{said_err}");
                dump_cache(posix, out_p);
                panic!(
                    "nimsem wants nimcache/{stem}.s.nif but no seeded source hashes to that stem \
                     — `nim_module_suffix` and the guest disagree, or the module is outside `lib/`"
                );
            };
            let key = format!("semcheck {stem}");
            assert!(
                !served.contains(&key),
                "no progress: {stem} was semchecked and nimsem still cannot open its .s.nif"
            );
            eprintln!("serving dependency semcheck: {stem} ({src})");
            let pnif = format!("nimcache/{stem}.p.nif");
            // An *imported* module is not nimsem's to parse: native nimony's plan gives every
            // module its own `nifler` step, and nimsem only spawns nifler for what a module
            // *includes*. So its parse is served here too, as that plan's step.
            if posix.read_file(&pnif).is_none() {
                let argv: Vec<String> =
                    ["nifler", "--portablePaths", "--deps", "parse", src, &pnif]
                        .iter()
                        .map(|s| s.to_string())
                        .collect();
                temen_run::nim_noc_run(nifler.clone(), posix, Arc::clone(make), &argv)
                    .unwrap_or_else(|e| panic!("nifler2 on {src}: {e}"));
            }
            let argv: Vec<String> = [
                "nimsem",
                "--define:nimNativeAlloc",
                "--define:nimNativeIo",
                "m",
                &pnif,
            ]
            .iter()
            .map(|s| s.to_string())
            .collect();
            semcheck(
                nimsem,
                nifler,
                posix,
                make,
                commands,
                &argv,
                &format!("nimcache/{stem}.s.nif"),
                out_p,
                sources,
            );
            served.push(key);
            continue;
        }
        if !commands.is_empty() {
            // With `/bin/sh` registered nimsem spawns nifler itself, so reaching here means the
            // real exec path did not work — serving it from outside would hide exactly the thing
            // this mode exists to prove.
            eprint!("--- nimsem said ---\n{said}--- stderr ---\n{said_err}");
            // #1665 — a command that *crashed* reaps as status 128, like one that exited 128; this
            // is the only place its trap and frames survive. (After an `execve` the frames name the
            // new image — `dump_func <image> <func> <block>` reads one.)
            for t in temen_interp::last_twin_traps() {
                eprintln!("--- a command crashed ---\n{t}");
            }
            dump_cache(posix, out_p);
            panic!(
                "nimsem wrote no {produced} with /bin/sh registered — the in-guest exec path \
                 failed: {outcome:?}\nThe shell reports a failed exec on stdout (`<cmd>: not \
                 found` / `cannot execute (errno N)`), shown above. Nothing there usually means \
                 /bin/sh is not the \
                 -DTEMEN_SHELL_POSIX build: the default build's manifest does not bind under a \
                 POSIX personality, so its own exec is refused."
            );
        }
        let Some(argv) = failed_nifler_command(&said) else {
            eprint!("--- nimsem said ---\n{said}--- stderr ---\n{said_err}");
            dump_cache(posix, out_p);
            panic!("nimsem wrote no {produced} and asked for no shell-out: {outcome:?}");
        };
        let cmd = argv.join(" ");
        assert!(
            !served.contains(&cmd),
            "no progress: nimsem asked for `{cmd}` again after it was served — the artifact it got \
             is not the one it is looking for"
        );

        // `parse <src> <out.p.nif>`: check the stem we would have predicted against the one the
        // guest actually wants. A mismatch is a fact about the port, not a reason to stop.
        if let Some(i) = argv.iter().position(|t| t == "parse") {
            if let (Some(src), Some(dst)) = (argv.get(i + 1), argv.get(i + 2)) {
                let want = Path::new(dst)
                    .file_name()
                    .and_then(|n| n.to_str())
                    .and_then(|n| n.strip_suffix(".p.nif"))
                    .unwrap_or("");
                let got = temen_run::nim_module_suffix(src, &["lib"]);
                if got != want {
                    eprintln!("note: nim_module_suffix({src}) = {got}, guest asked for {want}");
                }
            }
        }

        eprintln!("serving shell-out #{}: {cmd}", served.len() + 1);
        // argv[0] is whatever path the guest resolved `nifler` to; the program is ours.
        let mut run_argv = argv.clone();
        run_argv[0] = "nifler".to_string();
        temen_run::nim_noc_run(nifler.clone(), posix, Arc::clone(make), &run_argv)
            .unwrap_or_else(|e| panic!("nifler2 on `{cmd}`: {e}"));
        served.push(cmd);
    }
}

/// **hexer**: lower every semchecked module to Leng — `hexer c` once per module, as native nimony's
/// plan runs it (`<main>.final.build.nif`), with the same flags so each `.x.nif` is comparable to
/// native's. The main module's goes under `nimcache/<main>/`, where native puts it.
fn lower(
    hexer: &temen_ir::Module,
    posix: &temen_posix::Posix,
    make: &Arc<dyn Fn() -> temen_interp::HostProc + Send + Sync>,
    main: Option<&String>,
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
        temen_run::nim_noc_run(hexer.clone(), posix, Arc::clone(make), &argv)
            .unwrap_or_else(|e| panic!("hexer on {stem}: {e}"));
        assert!(
            posix.read_file(&x).is_some(),
            "hexer returned cleanly but wrote no {x}"
        );
    }
    eprintln!("✅ hexer lowered {} modules to Leng", stems.len());
}

/// **Link and run**: the `.x.nif`s hexer wrote, linked by [`temen_leng::link_nim_posix`] — the one
/// link every nim phase is itself built with — then run on a fresh personality. Returns its stdout.
fn link_and_run(posix: &temen_posix::Posix) -> Vec<u8> {
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
    temen_run::nim_noc_run(module, &run, Arc::new(make), &["prog".to_string()])
        .unwrap_or_else(|e| panic!("the program failed: {e}"));
    run.stdout()
}

/// `--expect`: every artifact native nimony wrote for a phase this run performed (`.s.nif` from
/// nimsem, and `.x.nif` from hexer when it ran) must be here, **byte for byte**. Nothing is
/// normalized: the lane script runs native nimony from a tree laid out as this memfs is (`lib/` at
/// its cwd), so both record the same paths. The one tolerance is [`declarations`]' — named, and
/// reported when used.
fn expect_native(posix: &temen_posix::Posix, native: &str, lowered: bool) {
    let mut want: Vec<(String, Vec<u8>)> = Vec::new();
    collect(Path::new(native), "", &mut want);
    want.retain(|(rel, _)| rel.ends_with(".s.nif") || (lowered && rel.ends_with(".x.nif")));
    want.sort();
    assert!(
        !want.is_empty(),
        "{native} holds no .s.nif — not a nimcache?"
    );
    let (mut exact, mut reordered) = (0, Vec::new());
    for (rel, w) in &want {
        let got = posix
            .read_file(&format!("nimcache/{rel}"))
            .unwrap_or_else(|| panic!("native nimony wrote {rel}; this run did not"));
        if got == *w {
            exact += 1;
            continue;
        }
        let (g, n) = (String::from_utf8_lossy(&got), String::from_utf8_lossy(w));
        if rel.ends_with(".x.nif") && declarations(&g) == declarations(&n) {
            reordered.push(rel.as_str());
            continue;
        }
        let (gl, nl): (Vec<_>, Vec<_>) = (g.lines().collect(), n.lines().collect());
        let at = gl
            .iter()
            .zip(&nl)
            .position(|(a, b)| a != b)
            .unwrap_or(gl.len().min(nl.len()));
        let show = |v: &[&str]| v[at.saturating_sub(2)..(at + 3).min(v.len())].join("\n");
        panic!(
            "{rel} differs from native nimony's at line {}\n--- temen ---\n{}\n--- native ---\n{}",
            at + 1,
            show(&gl),
            show(&nl)
        );
    }
    eprintln!(
        "✅ matches native nimony: {exact} of {} artifacts byte for byte",
        want.len()
    );
    if !reordered.is_empty() {
        eprintln!(
            "   the rest hold native's declarations in another order (#1753): {}",
            reordered.join(", ")
        );
    }
}

fn main() {
    let mut a: Vec<String> = std::env::args().skip(1).collect();
    // `--sh <sh.ir>`: register it at `/bin/sh` and let nimsem spawn nifler itself. Without it the
    // driver falls back to cranking the shell-out from outside (the pre-#1609 behaviour), which is
    // kept as the oracle the real path is checked against.
    let sh_ir = a.iter().position(|t| t == "--sh").map(|i| {
        let v = a.get(i + 1).cloned().expect("--sh needs a path");
        a.drain(i..=i + 1);
        v
    });
    // `--check prog.nim` / `--check lib/std/math.nim`: semcheck **that** module after `system`.
    // A host file is a **program**: it is seeded at the memfs root and checked as the main module
    // (`--isMain`), exactly as `nimony c --isMain prog.nim` run beside it treats it. A lib-relative
    // path names a stdlib module that is already seeded. The system module
    // is the special case (`--isSystem`, and its `.p.nif` arrives ready-made); every other module
    // is reached the ordinary way — nifler parses it, then nimsem is pointed at the result — and
    // the shell-out loop below resolves its imports exactly as it does for `system`'s.
    //
    // Needed because the divergence #1630 found lives in a module `system`'s closure never
    // reaches, and the only way to see it was a 10-minute nimsem build plus a Playwright run per
    // iteration.
    let check_mod = a.iter().position(|t| t == "--check").map(|i| {
        let v = a
            .get(i + 1)
            .cloned()
            .expect("--check needs a program or a lib-relative path");
        a.drain(i..=i + 1);
        v
    });
    // `--hexer <hexer.temen>`: lower every semchecked module to Leng, link, and run the program.
    let hexer_p = a.iter().position(|t| t == "--hexer").map(|i| {
        let v = a.get(i + 1).cloned().expect("--hexer needs a path");
        a.drain(i..=i + 1);
        v
    });
    // `--expect <native nimcache>`: the run checks itself — every artifact native nimony wrote for
    // a phase this run performed must be here and be the same, byte for byte ([`expect_native`]). What makes the self-hosted lane a *test* rather than a demo:
    // `scripts/ci/nim-selfhost-lane.sh` passes native nimony's own nimcache for the same program.
    let expect = a.iter().position(|t| t == "--expect").map(|i| {
        let v = a.get(i + 1).cloned().expect("--expect needs a directory");
        a.drain(i..=i + 1);
        v
    });
    let [nimsem_p, nifler_p, libdir, sys_pnif, sys_stem, out_p] = &a[..] else {
        panic!(
            "usage: nim_selfhost_lane [--sh <sh.ir>] [--check <prog.nim | lib/std/x.nim>] \
             [--hexer <hexer.temen>] [--expect <native-nimcache>] <nimsem.temen> <nifler2.temen> \
             <libdir> <sys.p.nif> <sys-stem> <out.s.nif>"
        );
    };

    let nimsem = phase(nimsem_p, "nimsem");
    let nifler = phase(nifler_p, "nifler2");
    // The command registry this run grants: the shell nimsem execs, and nifler for the shell to
    // exec in turn.
    //
    // nimsem spells nifler **`bin/nifler`** — relative to its cwd, from its own
    // `findTool`/`getAppDir` logic — which is the spelling that actually has to resolve. The other
    // three are registered because the registry is a flat name table and the cost of a spelling is
    // one row: a PATH walk for a bare `nifler`, and the absolute form, so a future nimsem that
    // resolves differently does not fail as a mystery. The guest names what it wants; we do not
    // guess which name that will be.
    let commands: Vec<(String, temen_ir::Module)> = match &sh_ir {
        Some(p) => {
            let sh = command_module(p, "/bin/sh");
            ["/bin/sh"]
                .iter()
                .map(|n| (n.to_string(), sh.clone()))
                .chain(
                    ["bin/nifler", "/bin/nifler", "nifler"]
                        .iter()
                        .map(|n| (n.to_string(), nifler.clone())),
                )
                .collect()
        }
        None => Vec::new(),
    };

    // One personality for both phases: one memfs, one fd table, one stdout.
    let (posix, make) = temen_posix::cap(0, 0, Vec::new());
    let make: Arc<dyn Fn() -> temen_interp::HostProc + Send + Sync> = Arc::new(make);

    // Seed the stdlib under `lib/`, both preserving `std/` and flattened (nimony resolves `std/x` as
    // `lib/x` as well as `lib/std/x`), then the system module's parsed nif. Sources go in first so
    // every artifact a phase derives from them is genuinely newer — the memfs stamps write order
    // into `st_mtim`, and that ordering is what the freshness checks in `deps.nim` and nifler's own
    // `parse` read.
    let mut seed: Vec<(String, Vec<u8>)> = Vec::new();
    collect(Path::new(libdir), "lib/", &mut seed);
    let flat: Vec<(String, Vec<u8>)> = seed
        .iter()
        .filter_map(|(k, v)| {
            k.strip_prefix("lib/std/")
                .map(|r| (format!("lib/{r}"), v.clone()))
        })
        .collect();
    seed.extend(flat);
    for (k, v) in &seed {
        posix.write_file(k, v);
    }
    // A program is a source like any other: seeded with them, before anything derived from it.
    let (check_mod, is_main) = match check_mod {
        Some(m) if Path::new(&m).is_file() => {
            let name = Path::new(&m)
                .file_name()
                .and_then(|n| n.to_str())
                .expect("utf-8 program name")
                .to_string();
            posix.write_file(
                &name,
                &std::fs::read(&m).unwrap_or_else(|e| panic!("read {m}: {e}")),
            );
            (Some(name), true)
        }
        other => (other, false),
    };
    posix.write_file(
        &format!("nimcache/{sys_stem}.p.nif"),
        &std::fs::read(sys_pnif).unwrap_or_else(|e| panic!("read {sys_pnif}: {e}")),
    );
    eprintln!("seeded {} stdlib files + the system .p.nif", seed.len());

    // `stem -> lib-relative source`, the inverse of `nim_module_suffix` over everything seeded.
    // The guest names a missing artifact by stem only, so serving it needs the way back; building
    // the map by hashing forward keeps one implementation of the naming rule (and the loop's
    // existing prediction check keeps proving it agrees with the guest).
    let sources: std::collections::HashMap<String, String> = seed
        .iter()
        .map(|(k, _)| k.as_str())
        .filter(|k| k.ends_with(".nim"))
        .map(|k| (temen_run::nim_module_suffix(k, &["lib"]), k.to_string()))
        .collect();

    if !commands.is_empty() {
        // The PATH the shell walks for a bare `nifler`.
        posix.set_env("PATH", "/bin");
    }

    // `--check`: parse the target with nifler first, so nimsem has a `.p.nif` to be pointed at.
    // Its own imports still resolve through the shell-out loop below.
    let target_stem = match &check_mod {
        Some(m) => {
            let stem = temen_run::nim_module_suffix(m, &["lib"]);
            let out = format!("nimcache/{stem}.p.nif");
            eprintln!("--check {m} -> {out}");
            let argv: Vec<String> = ["nifler", "--portablePaths", "--deps", "parse", m, &out]
                .iter()
                .map(|s| s.to_string())
                .collect();
            temen_run::nim_noc_run(nifler.clone(), &posix, Arc::clone(&make), &argv)
                .unwrap_or_else(|e| panic!("nifler2 on {m}: {e}"));
            stem
        }
        None => sys_stem.clone(),
    };

    // Semcheck `system` first, always: every other module needs its `.s.nif` on disk before it
    // can be checked at all (nimsem opens it by name and quits if it is missing). In the default
    // mode that *is* the run; under `--check` it is the prerequisite.
    let sys_argv: Vec<String> = [
        "nimsem",
        "--define:nimNativeAlloc",
        "--define:nimNativeIo",
        "m",
        "--isSystem",
        &format!("nimcache/{sys_stem}.p.nif"),
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let sys_produced = format!("nimcache/{sys_stem}.s.nif");
    let mut served = semcheck(
        &nimsem,
        &nifler,
        &posix,
        &make,
        &commands,
        &sys_argv,
        &sys_produced,
        out_p,
        &sources,
    );

    let produced = format!("nimcache/{target_stem}.s.nif");
    if check_mod.is_some() {
        let mut argv: Vec<String> = [
            "nimsem",
            "--define:nimNativeAlloc",
            "--define:nimNativeIo",
            "m",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        if is_main {
            argv.push("--isMain".to_string());
        }
        argv.push(format!("nimcache/{target_stem}.p.nif"));
        served += semcheck(
            &nimsem, &nifler, &posix, &make, &commands, &argv, &produced, out_p, &sources,
        );
    }

    let Some(b) = posix.read_file(&produced) else {
        dump_cache(&posix, out_p);
        eprint!(
            "--- nimsem stdout ---\n{}",
            String::from_utf8_lossy(&posix.stdout())
        );
        panic!("nimsem returned cleanly but wrote no {produced}");
    };
    std::fs::write(out_p, &b).unwrap_or_else(|e| panic!("write {out_p}: {e}"));
    // With `--sh`, nimsem runs its own nifler in the guest; the host serves nothing.
    let how = if commands.is_empty() {
        format!("after {served} host-served nifler2 run(s)")
    } else {
        "nimsem ran its own nifler through /bin/sh".to_string()
    };
    eprintln!("✅ {produced} ({} bytes), {how} → {out_p}", b.len());

    if let Some(hp) = &hexer_p {
        lower(
            &phase(hp, "hexer"),
            &posix,
            &make,
            is_main.then_some(&target_stem),
        );
    }
    dump_cache(&posix, out_p);
    if hexer_p.is_some() && is_main {
        let stdout = link_and_run(&posix);
        eprintln!(
            "✅ the program ran: {} bytes of stdout (on our stdout)",
            stdout.len()
        );
        use std::io::Write;
        std::io::stdout().write_all(&stdout).expect("write stdout");
    }
    if let Some(native) = &expect {
        expect_native(&posix, native, hexer_p.is_some());
    }
}
