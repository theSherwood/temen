//! **`hexer d`**: time nimony's whole-program dead-code elimination (`shoggoth d`, hexer's pass) on
//! Temen, the JIT workload #2147, #2535 and #2635 measured.
//!
//! ```text
//! hexer_d_bench <shoggoth.temen> <x-nif-dir> <runs> [jit|bytecode]
//! ```
//!
//! It runs `shoggoth d` over every `.x.nif` in `<x-nif-dir>` `1 + <runs>` times in one process and
//! prints each run's time. The first run includes the JIT's compile; the later ones run the cached
//! code (`nim_noc_run` with one [`temen_run::JitCodeCache`]). Every run must write the same outputs.
//!
//! **Inputs.** `shoggoth.temen` is the toolchain's (`scripts/nim-toolchain.sh <out>`). The `.x.nif`
//! files are a macro plugin's 51 modules, from a native build of a one-macro program in the
//! toolchain's tree (`<out>/nimony`), as `scripts/ci/nim-selfhost-lane.sh` builds it:
//!
//! ```text
//! M=<dir>; cp -r <out>/nimony/{bin,lib,src,doc} $M/
//! printf 'import std/[syncio, macros]\nmacro hello(): untyped =\n  result = newCall("echo", [newStrLitNode("hello")])\nhello()\n' >$M/prog.nim
//! (cd $M && ./bin/nimony c -d:temen --isMain prog.nim)
//! P=$(ls -d $M/nimcache/c/macro_hello_*); mkdir x
//! for c in $P/*.c.nif; do s=$(basename $c .c.nif); cp $(ls $P/$s.x.nif $M/nimcache/c/$s.x.nif 2>/dev/null | head -1) x/; done
//! ```
//!
//! **Measuring a change.** This VM's noise and code alignment (#2634) swamp a single run, so:
//! - *time*: alternate processes of the two builds (or modules), ten runs each, and compare medians
//!   over at least 100 runs per side;
//! - *instructions*: callgrind over a whole process with `<runs>` 2 minus one with 1 is one run's
//!   count. `HEXER_RESERVED_LOG2=33` gives each window an 8 GiB reservation, which valgrind can map.

use std::sync::Arc;
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let [shoggoth, dir, runs, rest @ ..] = &a[..] else {
        panic!("usage: hexer_d_bench <shoggoth.temen> <x-nif-dir> <runs> [jit|bytecode]");
    };
    let runs: usize = runs.parse().expect("<runs> is a number");
    let engine = match rest.first().map(String::as_str) {
        None | Some("jit") => temen_run::Backend::Jit,
        Some("bytecode") => temen_run::Backend::Bytecode,
        Some(e) => panic!("engine {e}: expected jit or bytecode"),
    };
    if let Ok(l) = std::env::var("HEXER_RESERVED_LOG2") {
        let l = l.parse().expect("HEXER_RESERVED_LOG2 is a log2");
        assert!(
            temen_ir::set_default_reserved_log2(l),
            "HEXER_RESERVED_LOG2 above the default"
        );
    }
    let bytes = std::fs::read(shoggoth).unwrap_or_else(|e| panic!("read {shoggoth}: {e}"));
    let module = temen_encode::decode_module(&bytes).expect("decode shoggoth");
    temen_verify::verify_module(&module).expect("shoggoth verifies");

    let mut files: Vec<String> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read {dir}: {e}"))
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".x.nif"))
        .collect();
    files.sort();
    let (posix, make) = temen_posix::cap(0, 0, Vec::new());
    let make = Arc::new(make);
    let mut argv = vec!["shoggoth".to_string(), "d".into(), "--outdir:out".into()];
    for f in &files {
        let b = std::fs::read(format!("{dir}/{f}")).expect("read an .x.nif");
        posix.write_file(&format!("in/{f}"), &b);
        argv.push(format!("in/{f}"));
    }
    eprintln!("shoggoth d over {} modules", files.len());

    let outputs = || {
        let mut names: Vec<String> = posix
            .file_names()
            .into_iter()
            .filter(|n| n.trim_start_matches('/').starts_with("out/"))
            .collect();
        names.sort();
        names
            .into_iter()
            .map(|n| (posix.read_file(&n), n))
            .collect::<Vec<_>>()
    };
    let code = temen_run::JitCodeCache::default();
    let mut first = None;
    for run in 0..=runs {
        for n in posix.file_names() {
            if n.trim_start_matches('/').starts_with("out/") {
                posix.remove_file(&n);
            }
        }
        let t = Instant::now();
        temen_run::nim_noc_run(
            module.clone(),
            &posix,
            make.clone(),
            &argv,
            &temen_run::ExecGrants::default(),
            engine,
            Some(&code),
            None,
        )
        .unwrap_or_else(|e| panic!("shoggoth d: {e}"));
        let dt = t.elapsed().as_secs_f64();
        let out = outputs();
        match &first {
            None => {
                println!("run 0 (compile + run): {dt:.3} s, {} outputs", out.len());
                first = Some(out);
            }
            Some(want) => {
                assert!(*want == out, "run {run} wrote different outputs than run 0");
                println!("run {run}: {dt:.4} s");
            }
        }
    }
}
