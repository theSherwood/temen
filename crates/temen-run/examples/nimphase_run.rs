//! Run a nimony compiler phase (`nifler`/`hexer`/`nimony`/…) on the Temen over an in-window `fs` memfs,
//! and dump every file the phase *produced* to an output dir — the generic guest half of the
//! nimony-on-Temen differentials (NIM.md §3c/§3e, "compile Nim in the browser"). The phase's whole-
//! program `.temen` runs `main(argc, argv)` with the given argv; the memfs is seeded from a fixture
//! dir (every file under it, at its relative path). After the run, any key in the shared store that
//! was NOT in the seed is a phase output — written under `<out-dir>` so a script can diff the whole
//! output set against the native phase on the same inputs.
//!
//! ```text
//! cargo run -q --release -p temen-run --example nimphase_run -- \
//!     <phase.temen> <fixture-dir> <out-dir> -- <argv0> <argv1> ...
//! ```
//!
//! `TEMEN_PHASE_BACKEND` selects the engine (treewalk default · bytecode · jit). The guest's cwd is `/`,
//! so a phase's `absolutePath("x.nif")` collapses to `/x.nif` → the cap-relative key `x.nif` (fs.rs
//! `norm`), matching the bare relative keys seeded here.

use std::collections::BTreeSet;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::exit;

use temen_run::{Backend, HostCap, Limits, Outcome, RunConfig};

/// Walk `dir` and return every regular file as `(relative-key, bytes)` — the memfs seed.
fn seed_dir(dir: &Path) -> Vec<(String, Vec<u8>)> {
    let mut out = vec![];
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).unwrap_or_else(|e| panic!("read_dir {d:?}: {e}")) {
            let entry = entry.expect("dir entry");
            let path = entry.path();
            if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                stack.push(path);
            } else {
                let key = path
                    .strip_prefix(dir)
                    .expect("strip fixture prefix")
                    .to_string_lossy()
                    .into_owned();
                out.push((key, std::fs::read(&path).expect("read fixture file")));
            }
        }
    }
    out
}

fn main() {
    let mut a = std::env::args().skip(1);
    let temen = a
        .next()
        .expect("usage: nimphase_run <phase.temen> <fixture-dir> <out-dir> -- <argv...>");
    let fixture = a.next().expect("missing <fixture-dir>");
    let out_dir = a.next().expect("missing <out-dir>");
    // Skip the `--` separator if present, the rest is argv.
    let argv: Vec<String> = a.skip_while(|s| s == "--").collect();
    assert!(!argv.is_empty(), "missing argv after --");

    let seed = seed_dir(Path::new(&fixture));
    let seed_keys: BTreeSet<String> = seed.iter().map(|(k, _)| k.clone()).collect();

    let (factory, handle) = temen_run::fs::mem_fs_shared_factory(seed, vec![]);
    let fs_cap = HostCap::host_proc(0, factory);

    let bytes = std::fs::read(&temen).unwrap_or_else(|e| panic!("read {temen}: {e}"));
    let module = temen_encode::decode_module(&bytes).expect("decode .temen");
    let inst = temen_run::instantiate(module).expect("instantiate (verifies)");

    let backend = match std::env::var("TEMEN_PHASE_BACKEND").as_deref() {
        Ok("bytecode") => Backend::Bytecode,
        Ok("jit") => Backend::Jit,
        Ok("treewalk") | Err(_) => Backend::TreeWalk,
        Ok(other) => panic!("TEMEN_PHASE_BACKEND: unknown engine {other:?}"),
    };

    let cfg = RunConfig {
        limits: Limits {
            fuel: None,
            deadline: None,
            max_fibers: 0,
            max_vcpus: 0,
        },
        stdin: vec![],
        memory_size_log2: None,
        args: argv.iter().map(|s| s.clone().into_bytes()).collect(),
        env: vec![],
        ..RunConfig::default()
    };

    let run = inst
        .run_with_caps(backend, &cfg, &[("fs", fs_cap)])
        .expect("run nimony phase guest");

    std::io::stderr().write_all(&run.stderr).unwrap();
    if !matches!(&run.outcome, Outcome::Returned(_) | Outcome::Exited(0)) {
        eprintln!("phase guest did not exit cleanly: {:?}", run.outcome);
        eprintln!("--- guest stream (stdout+stderr) ---");
        std::io::stderr().write_all(&run.stdout).unwrap();
        eprintln!("\n--- end guest stream ---");
        exit(1);
    }

    // Every store key that wasn't seeded is a phase output — write it under out-dir.
    let (files, _dirs) = handle.seed();
    let mut wrote = 0usize;
    for (key, bytes) in files {
        if seed_keys.contains(&key) {
            continue;
        }
        let dest = PathBuf::from(&out_dir).join(&key);
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent).expect("mkdir out subdir");
        }
        std::fs::write(&dest, &bytes).expect("write phase output");
        wrote += 1;
    }
    eprintln!("phase produced {wrote} file(s) → {out_dir}");
}
