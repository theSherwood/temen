//! Run chibicc-the-guest (`chibicc.svmb`, built by
//! `demos/chibicc_selfhost/build_chibicc_svmb.sh`) on the SVM and print what it emits — the guest
//! half of the self-host differential (SELFHOST_C.md §7 step 5). It seeds an in-memory `fs` cap with
//! the source file (and, optionally, a host include dir mounted at `/include`), passes argv, runs
//! `main` on the tree-walker (the oracle engine, CLAUDE.md), and forwards the guest's stdout,
//! stderr, and exit code. The driving script (`run_selfhost_diff.sh`) compares this stdout
//! byte-for-byte against the native reference built from the *same* frontend sources.
//!
//! ```text
//! cargo run -q -p svm-run --example chibicc_run -- \
//!     <chibicc.svmb> <host-input.c> <guest-argv-path> [host-include-dir]
//! ```
//!
//! `guest-argv-path` is what the guest sees as argv[1] (e.g. `/in.c`); the memfs maps an absolute
//! path to its cap-relative key, so `/in.c` resolves against the seed key `in.c` (fs.rs norm).

use std::process::exit;

use svm_run::{fs, Backend, Limits, Outcome, RunConfig};

fn main() {
    let mut a = std::env::args().skip(1);
    let svmb = a
        .next()
        .expect("usage: chibicc_run <svmb> <input.c> <argv-path> [include-dir]");
    let input = a.next().expect("missing <input.c>");
    let argv_path = a.next().expect("missing <argv-path>");
    let include_dir = a.next();

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

    let bytes = std::fs::read(&svmb).unwrap_or_else(|e| panic!("read {svmb}: {e}"));
    let module = svm_encode::decode_module(&bytes).expect("decode .svmb");
    let inst = svm_run::instantiate(module).expect("instantiate (verifies)");

    let cfg = RunConfig {
        limits: Limits {
            fuel: None,
            deadline: None,
            max_fibers: 0,
            max_vcpus: 0,
        },
        stdin: vec![],
        memory_size_log2: None,
        args: vec![b"chibicc".to_vec(), argv_path.into_bytes()],
        env: vec![],
    };
    let run = inst
        .run_with_caps(
            Backend::TreeWalk,
            &cfg,
            &[("fs", fs::mem_fs_seeded(files, dirs))],
        )
        .expect("run chibicc guest");

    // Forward exactly what the guest produced; the script diffs our stdout against native.
    use std::io::Write;
    std::io::stdout().write_all(&run.stdout).unwrap();
    std::io::stderr().write_all(&run.stderr).unwrap();
    match run.outcome {
        Outcome::Returned(vals) => {
            // main returned N → the on-ramp _start passes it to exit; treat non-empty i32 as code.
            let code = match vals.first() {
                Some(svm_run::Value::I32(n)) => *n,
                _ => 0,
            };
            exit(code);
        }
        Outcome::Exited(code) => exit(code),
    }
}
