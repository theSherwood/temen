//! **Run a no-C-built nimony phase over real work, and diff it against native** (#1599, #763).
//!
//! The phase is a `.temen` built by `build_nim_hello_temen --posix` — nimony → Leng → `temen-leng`,
//! with **no C compiler anywhere**. Its retained syscall imports bind to one `temen_posix`
//! personality over an in-memory filesystem seeded with the nimony stdlib and a `nimcache`.
//!
//! ```text
//! nim_noc_phase <phase.temen> <libdir> <nimcache-dir> <out-file> -- <argv...>
//! ```
//!
//! **Why the seed matters.** nimsem parses a dependency it has no current `.p.nif` for by shelling
//! out (`deps.nim`'s `execNifler` → `os.execShellCmd`, which on posix is `fork` + `execve("/bin/sh",
//! ["-c", cmd])` + `waitpid` — nim has no libc `system()` binding). The LLVM route never reaches
//! that: `nifler_shim.c` intercepts `system()` and drives the `exec` capability. A no-C phase has no
//! C shim, so it takes the real fork/exec path, which this personality does not serve — there is no
//! `execve` op and argv[0] is a shell.
//!
//! Seeding the stdlib *and* a matching `nimcache` does **not** avoid that, and it is worth recording
//! why: the `.p.nif` stems are **path-derived**. The guest resolves `lib/system/basic_types.nim` and
//! wants `nimcache/basu363p61.p.nif`; a native run computed its stem from an absolute stdlib path
//! and produced a different one. The freshness check can never match, so `m` shells out however much
//! is seeded. Serving fork/exec is the real answer, and it is its own piece of work — `OP_FORK` is a
//! genuine return-twice clone, but no op exposes the `Host`'s FORK.md §8.6 `execve` image-replace,
//! and argv[0] is a shell.
//!
//! **What does run today, byte-exact.** `nimsem x <module>.s.nif` — index generation — is
//! self-contained: it walks the semchecked module and emits `(index (checksum …))`, no dependency
//! resolution, no shell-out. A no-C-built nimsem produces a `.idx.nif` **byte-identical to native**,
//! checksum included, which means it walked the whole 1.39 MB input and hashed it the same:
//!
//! ```text
//! cargo run --release -p temen-run --example nim_noc_phase -- \
//!   nimsem_noc.temen .nimtool/nimony/lib <proj>/nimcache /nimcache/<sys>.s.idx.nif \
//!   -- nimsem x nimcache/<sys>.s.nif
//! ```
//!
//! That is nimsem doing real compiler work, compiled with no C compiler. The full `m` semantic check
//! is a larger job than index generation and remains blocked on exec — stated plainly so the two are
//! not confused.

use std::path::Path;

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

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let split = argv.iter().position(|a| a == "--").unwrap_or(argv.len());
    let (head, tail) = argv.split_at(split);
    let guest_argv: Vec<String> = tail.iter().skip(1).cloned().collect();
    let [phase, libdir, nimcache, outfile] = head else {
        panic!("usage: nim_noc_phase <phase.temen> <libdir> <nimcache> <out> -- <argv...>");
    };

    let bytes = std::fs::read(phase).unwrap_or_else(|e| panic!("read {phase}: {e}"));
    let module = temen_encode::decode_module(&bytes).expect("decode phase .temen");
    temen_verify::verify_module(&module).expect("phase verifies");
    eprintln!(
        "phase: {} funcs, {} imports, window 2^{}",
        module.funcs.len(),
        module.imports.len(),
        module.memory.as_ref().map(|m| m.size_log2).unwrap_or(0),
    );

    // One personality shared across every bound name: one fd table, one memfs, one stdout buffer.
    let (posix, make) = temen_posix::cap(0, 0, Vec::new());
    let make: std::sync::Arc<dyn Fn() -> temen_interp::HostProc + Send + Sync> =
        std::sync::Arc::new(make);
    let (imports, unbound) = temen_run::nim_posix_imports(&module, &posix, make);
    assert!(
        unbound.is_empty(),
        "unbound nimony imports (extend `temen_run::nim_import_binding`): {unbound:?}"
    );

    let mut seed: Vec<(String, Vec<u8>)> = Vec::new();
    collect(Path::new(libdir), "lib/", &mut seed);
    // nimony resolves `std/x` as `lib/x` as well as `lib/std/x`; mirror the flattening the LLVM
    // drivers do so a dependency is found under either spelling.
    let flat: Vec<(String, Vec<u8>)> = seed
        .iter()
        .filter_map(|(k, v)| {
            k.strip_prefix("lib/std/")
                .map(|r| (format!("lib/{r}"), v.clone()))
        })
        .collect();
    seed.extend(flat);
    collect(Path::new(nimcache), "nimcache/", &mut seed);
    for (k, v) in &seed {
        posix.write_file(k, v);
    }
    eprintln!("seeded {} files into the memfs", seed.len());

    let cfg = temen_run::RunConfig {
        limits: temen_run::Limits {
            fuel: None,
            ..temen_run::Limits::default()
        },
        args: guest_argv.iter().map(|a| a.as_bytes().to_vec()).collect(),
        ..temen_run::RunConfig::default()
    };
    let inst = temen_run::instantiate_with_imports(module, imports)
        .unwrap_or_else(|e| panic!("instantiate: {e}"));
    let outcome = inst.run(temen_run::Backend::TreeWalk, &cfg);

    let out = posix.stdout();
    if !out.is_empty() {
        eprintln!("--- guest stdout ---\n{}", String::from_utf8_lossy(&out));
    }
    let err = posix.stderr();
    if !err.is_empty() {
        eprintln!("--- guest stderr ---\n{}", String::from_utf8_lossy(&err));
    }
    if let Err(e) = outcome {
        // The guest's own words first: a phase that rejected its input says so, and the trap alone
        // does not. Then the file list, which separates "wrote nothing" from "wrote it elsewhere".
        eprintln!("--- the guest wrote: {:?}", produced(&posix, &seed));
        panic!("run failed: {e}");
    }

    match posix.read_file(outfile) {
        Some(b) => {
            std::fs::write(format!("/tmp/{}", sanitize(outfile)), &b).ok();
            eprintln!(
                "✅ produced {outfile} ({} bytes) → /tmp/{}",
                b.len(),
                sanitize(outfile)
            );
        }
        None => {
            eprintln!("--- the guest wrote: {:?}", produced(&posix, &seed));
            panic!("no {outfile} produced");
        }
    }
}

/// What the run added to the memfs, beyond what was seeded.
fn produced(posix: &temen_posix::Posix, seed: &[(String, Vec<u8>)]) -> Vec<String> {
    let seeded: std::collections::HashSet<String> = seed
        .iter()
        .map(|(k, _)| {
            if k.starts_with('/') {
                k.clone()
            } else {
                format!("/{k}")
            }
        })
        .collect();
    posix
        .file_names()
        .into_iter()
        .filter(|n| !seeded.contains(n))
        .collect()
}

fn sanitize(p: &str) -> String {
    p.trim_start_matches('/').replace('/', "_")
}
