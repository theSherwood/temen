//! **Drive a §14 child-entry phase (e.g. `nifler`/`hexer` `--child-entry`) over a shared memfs.** The
//! guest half of the "nimony in the browser" driver: spawn a child-entry `.temen` through the
//! [`temen_run::conductor`] — its own window, argv as the spawn's args payload — with a memfs seeded
//! from a fixture dir and re-granted as `"fs"`, plus a `stdout` Stream and an `exit` cap for its
//! `write`/`read`/`exit` imports, then dump every file the phase *wrote* to an output dir. The mechanism
//! proven in `child_entry_argv_fs` / `rust_driver_nifler` (temen-llvm tests), on a real compiled phase.
//!
//! ```text
//! cargo run -q --release -p temen-run --example spawn_child_fs -- \
//!     <child.temen> <fixture-dir> <out-dir> -- <argv0> <argv1> ...
//! ```
//!
//! Like `nimphase_run`, but the phase runs as a **confined child** (verify_module + spawn) rather than a
//! top-level powerbox program — so a Rust-on-Temen driver guest can fan phases out the same way. The
//! child's imports `exit`/`read`/`write`/`vm_map` bind by the reference policy to the re-granted
//! Exit/Stream and the auto-granted AddressSpace; `fs` resolves by name from the grant list.

use std::collections::BTreeSet;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::exit;
use std::sync::Arc;

use temen_interp::{run_with_host, ForkedProc, Host, HostProcFork, StreamRole, Value};

/// Walk `dir` into `(relative-key, bytes)` memfs seed entries.
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
        .expect("usage: spawn_child_fs <child.temen> <fixture-dir> <out-dir> -- <argv...>");
    let fixture = a.next().expect("missing <fixture-dir>");
    let out_dir = a.next().expect("missing <out-dir>");
    let argv: Vec<String> = a.skip_while(|s| s == "--").collect();
    assert!(!argv.is_empty(), "missing argv after --");

    let seed = seed_dir(Path::new(&fixture));
    let seed_keys: BTreeSet<String> = seed.iter().map(|(k, _)| k.clone()).collect();

    let bytes = std::fs::read(&temen).unwrap_or_else(|e| panic!("read {temen}: {e}"));
    let child = temen_encode::decode_module(&bytes).expect("decode child .temen");
    temen_verify::verify_module(&child).expect("child verifies");

    let argv: Vec<&str> = argv.iter().map(String::as_str).collect();
    let parent = temen_run::conductor(&["fs", "stdout", "exit"], &argv);

    let (factory, handle) = temen_run::fs::mem_fs_shared_factory(seed, vec![]);
    let factory = Arc::new(factory);

    let mut host = Host::new();
    let (fs_init, fs_init_state) = (*factory)();
    let fs_fork: HostProcFork = {
        let f = Arc::clone(&factory);
        Arc::new(move |_pid| {
            let (h, s) = (*f)();
            ForkedProc::shared(h, s)
        })
    };
    let fs_h = host.grant_host_proc_forkable(fs_init, fs_fork, fs_init_state);
    let sink = host.shared_stdout();
    let stdout_h = host.grant_stream(StreamRole::Out);
    let exit_h = host.grant_exit();
    let (inst, modh, budget) = temen_run::grant_conductor(&mut host, &child);

    let mut fuel = 400_000_000_000u64;
    let r = run_with_host(
        &parent,
        0,
        &[
            Value::I32(inst),
            Value::I32(modh),
            Value::I32(budget),
            Value::I32(fs_h),
            Value::I32(stdout_h),
            Value::I32(exit_h),
        ],
        &mut fuel,
        &mut host,
    );
    let stream = sink.lock().unwrap().clone();
    if !stream.is_empty() {
        eprintln!("--- child stdout/stderr ---");
        std::io::stderr().write_all(&stream).unwrap();
        eprintln!("\n--- end ---");
    }
    match &r {
        Ok(v) => eprintln!("child joined: {v:?}"),
        Err(t) => {
            eprintln!("child trapped: {t:?}");
            exit(1);
        }
    }

    // Every store key not in the seed is a phase output — dump it under out-dir.
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
