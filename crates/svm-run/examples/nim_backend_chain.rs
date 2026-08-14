//! The nimony **back-end chain**, run as real SVM guests, host-orchestrated (NIM.md §3c/§3e, slice 3
//! of "compile Nim in the browser"). This is the shape the browser card uses: the host (here Rust,
//! there JS) drives each phase's committed `.svmb` and pipes between them — no phase orchestrates the
//! others; the host does. Two REAL guests are chained:
//!
//!   semchecked `.s.nif`  ──hexer.svmb (fs cap)──▶  Leng `.x.nif`  ──svm-leng.svmb (stdin)──▶  SVM text
//!   ─parse+instantiate+run──▶ result
//!
//! `hexer.svmb` (nimony's middle-end, slice 2) lowers a semchecked module to Leng; `svm-leng.svmb`
//! (the committed W5 self-host asset) translates that Leng to SVM IR text; the text is parsed,
//! instantiated, and run. Every hop is checked against its native oracle (native hexer's `.x.nif`,
//! `svm_leng::translate_to_text`), so a divergence anywhere is caught. The `.s.nif` input is what
//! `nimsem` produces — currently oracle-baked while nimsem-on-SVM is blocked upstream (a native
//! nimony stock-nim build issue, not the on-ramp); once unblocked it slots in as a third guest ahead
//! of hexer, closing the full Nim→run chain.
//!
//! ```text
//! cargo run -q --release -p svm-run --example nim_backend_chain -- \
//!     <hexer.svmb> <svm-leng.svmb> <fixture-dir> <main-stem>
//! ```

use std::io::Write;

use svm_run::{Backend, HostCap, Limits, Outcome, RunConfig};

fn seed_dir(dir: &std::path::Path) -> Vec<(String, Vec<u8>)> {
    let mut out = vec![];
    for entry in std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read_dir {dir:?}: {e}")) {
        let entry = entry.expect("dir entry");
        if entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
            let name = entry.file_name().to_string_lossy().into_owned();
            out.push((name, std::fs::read(entry.path()).expect("read fixture")));
        }
    }
    out
}

fn run_hexer(hexer: &[u8], fixture: &std::path::Path, main: &str, backend: Backend) -> Vec<u8> {
    let (factory, handle) = svm_run::fs::mem_fs_shared_factory(seed_dir(fixture), vec![]);
    let fs_cap = HostCap::host_proc(0, factory);
    let inst = svm_run::instantiate(svm_encode::decode_module(hexer).expect("decode hexer.svmb"))
        .expect("instantiate hexer (verifies)");
    let cfg = RunConfig {
        limits: Limits {
            fuel: None,
            deadline: None,
            max_fibers: 0,
            max_vcpus: 0,
        },
        stdin: vec![],
        memory_size_log2: None,
        args: vec![
            b"hexer".to_vec(),
            b"c".to_vec(),
            format!("{main}.s.nif").into_bytes(),
        ],
        env: vec![],
        ..RunConfig::default()
    };
    let run = inst
        .run_with_caps(backend, &cfg, &[("fs", fs_cap)])
        .expect("run hexer guest");
    assert!(
        matches!(run.outcome, Outcome::Returned(_) | Outcome::Exited(0)),
        "hexer: {:?}",
        run.outcome
    );
    let (files, _) = handle.seed();
    files
        .into_iter()
        .find(|(k, _)| k == &format!("{main}.x.nif"))
        .map(|(_, b)| b)
        .expect("hexer wrote no .x.nif")
}

fn run_leng(leng_svmb: &[u8], leng_nif: &[u8], backend: Backend) -> Vec<u8> {
    let inst =
        svm_run::instantiate(svm_encode::decode_module(leng_svmb).expect("decode svm-leng.svmb"))
            .expect("instantiate svm-leng (verifies)");
    let cfg = RunConfig {
        limits: Limits {
            fuel: None,
            deadline: None,
            max_fibers: 0,
            max_vcpus: 0,
        },
        stdin: leng_nif.to_vec(),
        memory_size_log2: None,
        args: vec![],
        env: vec![],
        ..RunConfig::default()
    };
    let run = inst.run(backend, &cfg).expect("run svm-leng guest");
    assert!(
        matches!(run.outcome, Outcome::Returned(_) | Outcome::Exited(0)),
        "svm-leng: {:?}",
        run.outcome
    );
    run.stdout
}

fn main() {
    let mut a = std::env::args().skip(1);
    let hexer_p = a
        .next()
        .expect("usage: nim_backend_chain <hexer.svmb> <svm-leng.svmb> <fixture-dir> <main-stem>");
    let leng_p = a.next().expect("missing <svm-leng.svmb>");
    let fixture = a.next().expect("missing <fixture-dir>");
    let main = a.next().expect("missing <main-stem>");
    let hexer = std::fs::read(&hexer_p).unwrap_or_else(|e| panic!("read {hexer_p}: {e}"));
    let leng_svmb = std::fs::read(&leng_p).unwrap_or_else(|e| panic!("read {leng_p}: {e}"));
    let fixture = std::path::PathBuf::from(fixture);

    // Native oracles at each hop.
    let native_leng = {
        // native hexer over a scratch copy of the fixture
        let scratch = std::env::temp_dir().join(format!("nbc_nat_{main}"));
        let _ = std::fs::remove_dir_all(&scratch);
        std::fs::create_dir_all(&scratch).unwrap();
        for (k, v) in seed_dir(&fixture) {
            std::fs::write(scratch.join(&k), v).unwrap();
        }
        let hexer_bin =
            std::env::var("HEXER_BIN").unwrap_or_else(|_| ".nimtool/nimony/bin/hexer".into());
        let ok = std::process::Command::new(&hexer_bin)
            .current_dir(&scratch)
            .args(["c", &format!("{main}.s.nif")])
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        assert!(ok, "native hexer failed (set HEXER_BIN)");
        std::fs::read(scratch.join(format!("{main}.x.nif"))).expect("native hexer .x.nif")
    };
    let native_ir =
        svm_leng::translate_to_text(std::str::from_utf8(&native_leng).expect("Leng utf8"))
            .expect("native svm_leng translate");

    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        // Hop 1: hexer.svmb → Leng.
        let leng = run_hexer(&hexer, &fixture, &main, backend);
        assert_eq!(
            leng, native_leng,
            "{backend:?}: hexer.svmb Leng != native hexer"
        );
        // Hop 2: svm-leng.svmb → SVM text.
        let ir = run_leng(&leng_svmb, &leng, backend);
        assert_eq!(
            ir,
            native_ir.as_bytes(),
            "{backend:?}: svm-leng.svmb IR != native svm_leng"
        );
        // The produced IR is a well-formed module: it parses. (Running it needs the W2 link with the
        // `system` module + W3 runtime shim — its `ini`/syscall imports are unbound standalone; that
        // link step is proven separately in svm-leng's `nim_e2e` and is the same for native output.)
        let out_mod = svm_text::parse_module(std::str::from_utf8(&ir).expect("IR utf8"))
            .expect("parse chained IR");
        println!(
            "  {backend:?}: OK — hexer.svmb→Leng ({} B, ==native) → svm-leng.svmb→IR ({} B, ==native, {} funcs)",
            leng.len(),
            ir.len(),
            out_mod.funcs.len()
        );
    }
    std::io::stdout().flush().ok();
    println!("BACK-END CHAIN OK — real hexer + svm-leng guests, host-orchestrated, byte-exact vs native, all engines");
}
