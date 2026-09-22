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
//! the phases; nimsem is not spawning nifler itself. Serving `fork`/`execve` so it can is the
//! remaining piece of #1609, and it is a process-model decision, not a missing line of code.
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

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let [nimsem_p, nifler_p, libdir, sys_pnif, sys_stem, out_p] = &a[..] else {
        panic!(
            "usage: nim_noc_semcheck <nimsem.temen> <nifler2.temen> <libdir> <sys.p.nif> \
             <sys-stem> <out.s.nif>"
        );
    };

    let nimsem = phase(nimsem_p, "nimsem");
    let nifler = phase(nifler_p, "nifler2");

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

    let nimsem_argv: Vec<String> = [
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

    // Success is **the artifact**, not the return: nim's `quit("FAILURE: …")` is an ordinary exit,
    // so a run that gave up looks exactly like one that finished. Ask the memfs instead.
    let produced = format!("nimcache/{sys_stem}.s.nif");
    let mut served: Vec<String> = Vec::new();
    loop {
        let before = posix.stdout().len();
        let outcome =
            temen_run::nim_noc_run(nimsem.clone(), &posix, Arc::clone(&make), &nimsem_argv);
        let said = String::from_utf8_lossy(&posix.stdout()[before..]).into_owned();
        if posix.read_file(&produced).is_some() {
            break;
        }
        let Some(argv) = failed_nifler_command(&said) else {
            eprint!("--- nimsem said ---\n{said}");
            dump_cache(&posix, out_p);
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
        temen_run::nim_noc_run(nifler.clone(), &posix, Arc::clone(&make), &run_argv)
            .unwrap_or_else(|e| panic!("nifler2 on `{cmd}`: {e}"));
        served.push(cmd);
    }

    dump_cache(&posix, out_p);
    match posix.read_file(&produced) {
        Some(b) => {
            std::fs::write(out_p, &b).unwrap_or_else(|e| panic!("write {out_p}: {e}"));
            eprintln!(
                "✅ {produced} ({} bytes) after {} nifler2 run(s) → {out_p}",
                b.len(),
                served.len()
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
