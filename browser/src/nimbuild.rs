//! **nimony's own driver on the browser's interpreter tier, natively** (#958). Runs
//! [`temen_browser::nim_build`], the function the wasm export `temen_nim_build` runs, over the
//! self-hosted lane's toolchain and one of its program trees, and reports how long the build took,
//! the process's peak resident memory, and what the run held at its end. A measurement, not a gate:
//! the engine's cost before the wasm factor, from the code the browser runs.
//!
//! ```text
//! nimbuild <toolchain-dir> <tree> <prog.nim> [<expect.temen>]
//! ```
//!
//! `<toolchain-dir>` holds what `scripts/ci/nim-selfhost-lane.sh` builds: `nimony.temen`,
//! `nifmake.temen`, `nimsem.temen`, `nifler2.temen`, `hexer.temen`, `temen-link.temen`, the shell's
//! `sh.ir`, and the guest libc as `libc.temeno`. The build runs in `<tree>` at its host path, as the
//! lane's does, over the same seed. `<expect.temen>`, when given, is the module the build must link.

use std::path::Path;
use std::time::Instant;

const TOOLS: [&str; 6] = [
    "nimony",
    "nifmake",
    "nimsem",
    "nifler2",
    "hexer",
    "temen-link",
];

fn read(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// Every file under `dir`, keyed by its path below `prefix`.
fn collect(dir: &Path, prefix: &str, out: &mut Vec<(String, Vec<u8>)>) {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .flatten()
        .collect();
    entries.sort_by_key(|e| e.file_name());
    for e in entries {
        let name = e.file_name().to_string_lossy().into_owned();
        if e.path().is_dir() {
            collect(&e.path(), &format!("{prefix}{name}/"), out);
        } else {
            out.push((format!("{prefix}{name}"), read(&e.path())));
        }
    }
}

fn peak_rss_mib() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            let line = s.lines().find(|l| l.starts_with("VmHWM:"))?;
            line.split_whitespace().nth(1)?.parse::<u64>().ok()
        })
        .map_or(0, |kib| kib / 1024)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (tools, tree, prog, expect) = match &args[..] {
        [t, d, p] => (t, d, p, None),
        [t, d, p, e] => (t, d, p, Some(e)),
        _ => panic!("usage: nimbuild <toolchain-dir> <tree> <prog.nim> [<expect.temen>]"),
    };
    let tools = Path::new(tools);
    let dir = std::fs::canonicalize(tree)
        .unwrap_or_else(|e| panic!("{tree}: {e}"))
        .to_string_lossy()
        .into_owned();

    let decode = |name: &str| {
        let m = temen_encode::decode_module(&read(&tools.join(format!("{name}.temen"))))
            .unwrap_or_else(|e| panic!("decode {name}: {e:?}"));
        temen_verify::verify_module(&m).unwrap_or_else(|e| panic!("verify {name}: {e:?}"));
        m
    };
    let modules: Vec<(String, temen_ir::Module)> =
        TOOLS.iter().map(|t| (t.to_string(), decode(t))).collect();
    let sh = String::from_utf8(read(&tools.join("sh.ir"))).expect("sh.ir is text");
    let sh = temen_text::parse_module(&sh).expect("parse sh.ir");
    temen_verify::verify_module(&sh).expect("verify sh.ir");

    // `/bin/sh`, and each tool where the driver looks for it (beside its own `bin/nimony`) and where
    // the shell's `PATH` walk finds a bare name.
    let paths: Vec<[String; 2]> = TOOLS
        .iter()
        .map(|t| [format!("{dir}/bin/{t}"), format!("/bin/{t}")])
        .collect();
    let mut commands: Vec<(&temen_ir::Module, Vec<&str>)> = vec![(&sh, vec!["/bin/sh"])];
    for ((_, m), [a, b]) in modules.iter().zip(&paths) {
        commands.push((m, vec![a.as_str(), b.as_str()]));
    }

    // The tree, at its host path, without its `bin/` (the toolchain, which the guest has as
    // commands) or its `nimcache/` (what a build writes); then the guest libc where `temen-link`
    // looks for it.
    let mut files: Vec<(String, Vec<u8>)> = Vec::new();
    let mut top: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{dir}: {e}"))
        .flatten()
        .collect();
    top.sort_by_key(|e| e.file_name());
    for e in top {
        let name = e.file_name().to_string_lossy().into_owned();
        match (name.as_str(), e.path().is_dir()) {
            ("bin" | "nimcache", _) => {}
            (_, true) => collect(&e.path(), &format!("{dir}/{name}/"), &mut files),
            (_, false) => files.push((format!("{dir}/{name}"), read(&e.path()))),
        }
    }
    files.push((
        "/lib/temen/libc.temeno".to_string(),
        read(&tools.join("libc.temeno")),
    ));
    let files: Vec<(&str, &[u8])> = files
        .iter()
        .map(|(p, b)| (p.as_str(), b.as_slice()))
        .collect();

    let argv: [&[u8]; 4] = [b"bin/nimony", b"t", b"--isMain", prog.as_bytes()];
    let driver = &modules[0].1;
    let t0 = Instant::now();
    let b = temen_browser::nim_build(driver, &commands, &files, &argv, &dir)
        .expect("the interpreter tier runs nimony");
    let secs = t0.elapsed().as_secs_f64();

    // `nimcache/<main>.temen/<prog>.temen`, `<main>` the driver's stem for the program.
    let out = format!(
        "/{}.temen",
        Path::new(prog).file_stem().unwrap().to_string_lossy()
    );
    let built = b
        .posix
        .file_names()
        .into_iter()
        .find(|n| n.starts_with(&format!("{dir}/nimcache/")) && n.ends_with(&out))
        .and_then(|n| b.posix.read_file(&n));
    eprintln!(
        "status {} exit {} in {secs:.1} s, peak RSS {} MiB, {:?}, {} files seeded",
        b.status,
        b.exit_code,
        peak_rss_mib(),
        b.footprint,
        files.len()
    );
    let Some(built) = built else {
        eprint!(
            "--- stdout ---\n{}--- stderr ---\n{}",
            String::from_utf8_lossy(&b.stdout),
            String::from_utf8_lossy(&b.stderr)
        );
        panic!("the build linked no {out}");
    };
    eprintln!("built {out}: {} bytes", built.len());
    if let Some(e) = expect {
        assert!(
            read(Path::new(e)) == built,
            "the module built on the browser's tier is not {e}"
        );
        eprintln!("✅ it is {e}, byte for byte");
    }
}
