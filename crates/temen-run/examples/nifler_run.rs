//! Run `nifler`-the-guest (`nifler.temen`, a real nimony phase on-ramped to Temen) over an input Nim
//! file, and print the `.nif` it emits — the guest half of the nifler-on-Temen differential (NIM.md
//! §3c/§3e, "nimony in the browser" slice 1). It seeds an in-memory `fs` cap with the source, passes
//! `nifler p <in> <out>` as argv (the parse command), runs `main` on the tree-walker (the oracle
//! engine, CLAUDE.md), then reads the emitted `<out>` back out of the shared memfs and writes it to
//! stdout, so the driving script can diff it byte-for-byte against native nifler on the same source.
//!
//! ```text
//! cargo run -q --release -p temen-run --example nifler_run -- <nifler.temen> <input.nim> <out.nif>
//! ```
//!
//! The memfs maps an absolute guest path to its cap-relative key (fs.rs `norm`), so `/x.nim` resolves
//! against the seed key `x.nim`. `TEMEN_NIFLER_BACKEND` selects the engine (treewalk default).

use std::io::Write;
use std::process::exit;

use temen_run::{Backend, HostCap, Limits, Outcome, RunConfig};

fn main() {
    let mut a = std::env::args().skip(1);
    let temen = a
        .next()
        .expect("usage: nifler_run <nifler.temen> <input.nim> <out.nif>");
    let input = a.next().expect("missing <input.nim>");
    let out_name = a.next().expect("missing <out.nif>");

    let in_key = "in.nim".to_string();
    let out_key = out_name.trim_start_matches('/').to_string();
    let src = std::fs::read(&input).unwrap_or_else(|e| panic!("read {input}: {e}"));

    // A cross-domain shared memfs so we can read the emitted `.nif` back after the run (the handle
    // observes the same store the guest writes into).
    let (factory, handle) =
        temen_run::fs::mem_fs_shared_factory(vec![(in_key.clone(), src)], vec![]);
    let fs_cap = HostCap::host_proc(0, factory);

    let bytes = std::fs::read(&temen).unwrap_or_else(|e| panic!("read {temen}: {e}"));
    let module = temen_encode::decode_module(&bytes).expect("decode .temen");
    let inst = temen_run::instantiate(module).expect("instantiate (verifies)");

    let backend = match std::env::var("TEMEN_NIFLER_BACKEND").as_deref() {
        Ok("bytecode") => Backend::Bytecode,
        Ok("jit") => Backend::Jit,
        Ok("treewalk") | Err(_) => Backend::TreeWalk,
        Ok(other) => panic!("TEMEN_NIFLER_BACKEND: unknown engine {other:?}"),
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
        // nifler's parse command: `nifler p <in> <out>` (bridge.nim's `parse` action).
        args: vec![
            b"nifler".to_vec(),
            b"p".to_vec(),
            format!("/{in_key}").into_bytes(),
            format!("/{out_key}").into_bytes(),
        ],
        env: vec![],
        ..RunConfig::default()
    };

    let run = inst
        .run_with_caps(backend, &cfg, &[("fs", fs_cap)])
        .expect("run nifler guest");

    std::io::stderr().write_all(&run.stderr).unwrap();
    if !matches!(&run.outcome, Outcome::Returned(_) | Outcome::Exited(0)) {
        eprintln!("nifler guest did not exit cleanly: {:?}", run.outcome);
        eprintln!("--- guest stream (stdout+stderr) ---");
        std::io::stderr().write_all(&run.stdout).unwrap();
        eprintln!("\n--- end guest stream ---");
        exit(1);
    }

    // Read the emitted `.nif` back out of the shared store and forward it.
    let (files, _dirs) = handle.seed();
    let emitted = files
        .into_iter()
        .find(|(name, _)| name == &out_key)
        .map(|(_, bytes)| bytes)
        .unwrap_or_else(|| panic!("nifler wrote no `{out_key}` (files: none matched)"));
    std::io::stdout().write_all(&emitted).unwrap();
}
