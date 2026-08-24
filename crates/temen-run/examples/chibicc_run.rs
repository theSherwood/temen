//! Run chibicc-the-guest (`chibicc.temen`, built by
//! `demos/chibicc_selfhost/build_chibicc_temen.sh`) on the Temen and print what it emits — the guest
//! half of the self-host differential (SELFHOST_C.md §7 step 5). It seeds an in-memory `fs` cap with
//! the source file (and, optionally, a host include dir mounted at `/include`), passes argv, runs
//! `main` on the tree-walker (the oracle engine, CLAUDE.md), and forwards the guest's stdout,
//! stderr, and exit code. The driving script (`run_selfhost_diff.sh`) compares this stdout
//! byte-for-byte against the native reference built from the *same* frontend sources.
//!
//! ```text
//! cargo run -q -p temen-run --example chibicc_run -- \
//!     <chibicc.temen> <host-input.c> <guest-argv-path> [host-include-dir]
//! ```
//!
//! `guest-argv-path` is what the guest sees as argv[1] (e.g. `/in.c`); the memfs maps an absolute
//! path to its cap-relative key, so `/in.c` resolves against the seed key `in.c` (fs.rs norm).

use std::process::exit;

use temen_run::{fs, Backend, Limits, Outcome, RunConfig};

fn main() {
    let mut a = std::env::args().skip(1);
    let temen = a
        .next()
        .expect("usage: chibicc_run <temen> <input.c> <argv-path> [include-dir]");
    let input = a.next().expect("missing <input.c>");
    let argv_path = a.next().expect("missing <argv-path>");
    let include_dir = a.next();
    // Any remaining args are forwarded to the guest chibicc as leading options (e.g. `-g0`,
    // `--data-page 65536`) — chibicc parses options before the input path.
    let extra: Vec<String> = a.collect();

    // Seed: the source at its cap-relative key (argv-path minus any leading '/'), plus every header
    // in the optional include dir under `include/` (mounted at the guest's fixed `/include`).
    let key = argv_path.trim_start_matches('/').to_string();
    let src = std::fs::read(&input).unwrap_or_else(|e| panic!("read {input}: {e}"));
    let mut files = vec![(key, src)];
    let mut dirs = vec![];
    if let Some(dir) = &include_dir {
        dirs.push("include".to_string());
        for entry in std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read_dir {dir}: {e}")) {
            let entry = entry.expect("dir entry");
            if entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
                let name = entry.file_name().to_string_lossy().into_owned();
                let bytes = std::fs::read(entry.path()).expect("read header");
                files.push((format!("include/{name}"), bytes));
            }
        }
    }

    let bytes = std::fs::read(&temen).unwrap_or_else(|e| panic!("read {temen}: {e}"));
    let module = temen_encode::decode_module(&bytes).expect("decode .temen");
    let inst = temen_run::instantiate(module).expect("instantiate (verifies)");

    // TEMEN_CHIBICC_BACKEND selects the engine: treewalk (the oracle, default), bytecode (what the
    // browser runs), or jit (Cranelift — the compiling-backend proxy for the browser's wasm-jit).
    // The self-host differential runs all three to prove the guest compiles identically on each.
    let backend = match std::env::var("TEMEN_CHIBICC_BACKEND").as_deref() {
        Ok("bytecode") => Backend::Bytecode,
        Ok("jit") => Backend::Jit,
        Ok("treewalk") | Err(_) => Backend::TreeWalk,
        Ok(other) => panic!("TEMEN_CHIBICC_BACKEND: unknown engine {other:?}"),
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
        args: {
            let mut v = vec![b"chibicc".to_vec()];
            v.extend(extra.iter().map(|s| s.clone().into_bytes()));
            v.push(argv_path.into_bytes());
            v
        },
        env: vec![],
        ..RunConfig::default()
    };
    let run = inst
        .run_with_caps(backend, &cfg, &[("fs", fs::mem_fs_seeded(files, dirs))])
        .expect("run chibicc guest");

    use std::io::Write;
    std::io::stderr().write_all(&run.stderr).unwrap();
    if !matches!(&run.outcome, Outcome::Returned(_) | Outcome::Exited(0)) {
        eprintln!("chibicc guest did not exit cleanly: {:?}", run.outcome);
        exit(1);
    }
    let ir = run.stdout;

    // TEMEN_CHIBICC_EXEC=1 runs the *full playground pipeline*: instead of printing the IR, parse it
    // (temen_text::parse_module — the same entry the browser cdylib's encode step uses), instantiate
    // the compiled program, run it on the same engine, and exit with its result. This is the local
    // mirror of the browser demo (compile in-Temen → encode → run in-Temen → surface the result).
    if std::env::var("TEMEN_CHIBICC_EXEC").as_deref() == Ok("1") {
        let text = String::from_utf8(ir).expect("guest IR is not UTF-8");
        let out_mod = temen_text::parse_module(&text).expect("parse compiled IR");
        let out_inst = temen_run::instantiate(out_mod).expect("instantiate compiled program");
        let out_cfg = RunConfig {
            limits: Limits {
                fuel: None,
                deadline: None,
                max_fibers: 0,
                max_vcpus: 0,
            },
            stdin: vec![],
            memory_size_log2: None,
            args: vec![],
            env: vec![],
            ..RunConfig::default()
        };
        let out_run = out_inst
            .run(backend, &out_cfg)
            .expect("run compiled program");
        std::io::stdout().write_all(&out_run.stdout).unwrap();
        std::io::stderr().write_all(&out_run.stderr).unwrap();
        match out_run.outcome {
            Outcome::Returned(vals) => exit(match vals.first() {
                Some(temen_run::Value::I32(n)) => *n & 0xff,
                _ => 0,
            }),
            Outcome::Exited(code) => exit(code & 0xff),
        }
    }

    // Default: forward the emitted IR (the differential diffs our stdout against native).
    std::io::stdout().write_all(&ir).unwrap();
    match run.outcome {
        Outcome::Returned(vals) => exit(match vals.first() {
            Some(temen_run::Value::I32(n)) => *n,
            _ => 0,
        }),
        Outcome::Exited(code) => exit(code),
    }
}
