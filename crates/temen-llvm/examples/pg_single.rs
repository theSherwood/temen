//! Boot the whole-program Postgres module natively (`postgres --single`) against an `initdb`'d data
//! dir, feeding SQL on stdin — the README's "Booting" driver as a runnable tool, so a browser-card
//! failure (`play/postgres` in `browser-test.mjs`) can be reproduced on the reference host with the
//! guest's stderr visible. Same translate options the playground build uses (`--stub-externs`,
//! `--host-page 65536`).
//!
//!   cargo run --release --example pg_single -- /tmp/temen_pg_cache/postgres_shimmed.bc \
//!       /tmp/temen_pg_cache/pgdata [sql-file] [bytecode|treewalk|jit]
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: pg_single <postgres_shimmed.bc|.ll> <datadir> [sql-file] [bytecode|treewalk|jit]");
        std::process::exit(2);
    }
    let sql = match args.get(3) {
        Some(p) => std::fs::read(p).expect("read sql file"),
        None => b"CREATE TABLE t (x int, s text);\n\
                  INSERT INTO t VALUES (1, 'one'), (2, 'two'), (3, 'three');\n\
                  SELECT * FROM t WHERE x > 1 ORDER BY x DESC;\n\
                  SELECT count(*), sum(x), avg(x) FROM t;\n"
            .to_vec(),
    };
    let backend = match args.get(4).map(String::as_str) {
        Some("treewalk") => temen_run::Backend::TreeWalk,
        Some("jit") => temen_run::Backend::Jit,
        _ => temen_run::Backend::Bytecode,
    };
    let opts = temen_llvm::TranslateOptions {
        stub_unresolved_externs: true,
        stack_page: 65536,
        child_entry: false,
    };
    let input = &args[1];
    let t0 = std::time::Instant::now();
    let t = if input.ends_with(".ll") {
        temen_llvm::translate_ll_path_with_options(input, opts)
    } else {
        temen_llvm::translate_bc_path_with_options(input, opts)
    }
    .expect("translate");
    eprintln!("translated in {:.1?}", t0.elapsed());
    let inst = temen_run::instantiate(t.module).expect("instantiate");
    let config = temen_run::RunConfig {
        stdin: sql,
        args: ["./postgres", "--single", "-D", ".", "postgres"]
            .iter()
            .map(|s| s.as_bytes().to_vec())
            .collect(),
        ..Default::default()
    };
    let datadir = PathBuf::from(&args[2]);
    let t1 = std::time::Instant::now();
    match inst.run_with_caps(backend, &config, &[("fs", temen_run::fs::host_fs(datadir))]) {
        Ok(run) => {
            eprintln!("--- outcome: {:?} in {:.1?}", run.outcome, t1.elapsed());
            println!(
                "--- stdout ({} bytes)\n{}",
                run.stdout.len(),
                String::from_utf8_lossy(&run.stdout)
            );
            eprintln!(
                "--- stderr ({} bytes)\n{}",
                run.stderr.len(),
                String::from_utf8_lossy(&run.stderr)
            );
        }
        Err(e) => {
            eprintln!("--- run error after {:.1?}:\n{e}", t1.elapsed());
            std::process::exit(1);
        }
    }
}
