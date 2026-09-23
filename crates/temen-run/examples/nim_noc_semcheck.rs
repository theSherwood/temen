//! **A full `nimsem m` semantic check on the no-C path** (#1609, #763).
//!
//! Both phases are `.temen` modules built by `build_nim_hello_temen --posix` — nimony → Leng →
//! `temen-leng`, **no C compiler anywhere** — bound to one shared `temen_posix` personality, so they
//! see one memfs: the `.p.nif` nifler2 writes is the one nimsem reads back.
//!
//! ```text
//! nim_noc_semcheck <nimsem.temen> <nifler2.temen> <libdir> <sys.p.nif> <sys-stem> <out.s.nif>
//! ```
//!
//! **The shell-out, cranked from outside.** `nimsem m` resolves its import graph by shelling out to
//! nifler for every dependency without a current `.p.nif` (`deps.nim`'s `execNifler` →
//! `os.execShellCmd` → `fork` + `execve("/bin/sh", ["-c", cmd])` + `waitpid`). The LLVM route serves
//! that from inside the guest: `nifler_shim.c` defines `system()` and drives the `exec` capability. A
//! no-C phase has no C shim, so it takes the real fork/exec path — and `temen-posix` exposes no
//! `execve` op and has no `/bin/sh` in its command registry (#1609 has the table of what exists).
//!
//! So this driver does what the shell would, one level out: run nimsem; when it quits with
//! `FAILURE: nifler … parse <src> <out>`, run nifler2 over the same memfs with exactly those
//! arguments; run nimsem again. The guest names the file it wants, so nothing here guesses a path or
//! a cache stem. It converges because each round makes one more dependency current, and it refuses
//! to spin: a command it has already served, served again, is a hang, not progress.
//!
//! Be precise about what this shows and what it does not. Every phase that runs is a Temen module
//! compiled with no C compiler, doing the real work — that is the point. But the *driver* sequences
//! the phases; nimsem is not spawning nifler itself.
//!
//! **`--sh <sh.ir>` is the real path, and it is one blocker short** (#1609). Passing a
//! chibicc-built shell registers it at `/bin/sh`, registers nifler2 at the spellings nimsem uses,
//! and drops the hand-crank — nimsem then takes its own fork/exec/wait sequence. Four of the five
//! things that needed were built and work: `OP_EXECVE` and `OP_WAIT4` exist, `execShellCmd`'s
//! whole leaf set is retained as real imports rather than fail-closed stubs, those imports are
//! bound to the personality, and this run installs the signal/caller-request door `fork` rides.
//!
//! The exec path itself now works: #1621 (the `call.import` arm dropped the caller request, so
//! `fork` answered `-ENOSYS`) is fixed, and so is the argv replacement behind it. Run this with a
//! three-line probe registered in place of the shell and nimsem forks, `execve`s, the image is
//! replaced, and the probe runs on the shared personality reading its `argc`/`argv`.
//!
//! What is left is **#1628**: `demos/shell` has only ever been compiled as a *root* module
//! (`c_to_ir`, never `--child-entry`), and does not start as a command — it never reaches `main`.
//! That is about the demo, not the VM. Until it is sorted `--sh` fails and the hand-crank below
//! remains the working path.
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
            match std::fs::write(Path::new(&dir).join(rel), &b) {
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

/// Run nimsem until it produces `produced`, serving its nifler shell-outs from outside.
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
        if !commands.is_empty() {
            // With `/bin/sh` registered nimsem spawns nifler itself, so reaching here means the
            // real exec path did not work — serving it from outside would hide exactly the thing
            // this mode exists to prove. See the module docs for the one blocker that remains.
            eprint!("--- nimsem said ---\n{said}--- stderr ---\n{said_err}");
            dump_cache(posix, out_p);
            panic!(
                "nimsem wrote no {produced} with /bin/sh registered — the in-guest exec path \
                 failed: {outcome:?}\nA shell that never reaches `main` is #1628; try a \
                 three-line probe command in its place to tell that apart from an exec fault."
            );
        }
        // A dependency that is parsed but not yet **semchecked**: `vfs: open failed:
        // nimcache/<stem>.s.nif`. The real toolchain has `nifmake` walking the graph in
        // topological order; here the guest names what it is missing, so serve it the same way the
        // shell-out rule serves a parse — semcheck that module, then let the caller retry. Each
        // round makes one more dependency current, so this converges for the same reason the
        // shell-out loop does, and the no-progress guard covers both.
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
    // `--check lib/std/math.nim`: semcheck **that** module instead of `system`. The system module
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
            .expect("--check needs a lib-relative path");
        a.drain(i..=i + 1);
        v
    });
    let [nimsem_p, nifler_p, libdir, sys_pnif, sys_stem, out_p] = &a[..] else {
        panic!(
            "usage: nim_noc_semcheck [--sh <sh.ir>] [--check <lib/std/x.nim>] <nimsem.temen> \
             <nifler2.temen> <libdir> <sys.p.nif> <sys-stem> <out.s.nif>"
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
        let argv: Vec<String> = [
            "nimsem",
            "--define:nimNativeAlloc",
            "--define:nimNativeIo",
            "m",
            &format!("nimcache/{target_stem}.p.nif"),
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        served += semcheck(
            &nimsem, &nifler, &posix, &make, &commands, &argv, &produced, out_p, &sources,
        );
    }

    dump_cache(&posix, out_p);
    match posix.read_file(&produced) {
        Some(b) => {
            std::fs::write(out_p, &b).unwrap_or_else(|e| panic!("write {out_p}: {e}"));
            eprintln!(
                "✅ {produced} ({} bytes) after {served} nifler2 run(s) → {out_p}",
                b.len()
            );
        }
        None => {
            eprint!(
                "--- nimsem stdout ---\n{}",
                String::from_utf8_lossy(&posix.stdout())
            );
            panic!("nimsem returned cleanly but wrote no {produced}");
        }
    }
}
