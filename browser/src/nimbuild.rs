//! **nimony's own driver on the browser's interpreter tier, natively** (#958). Runs
//! [`temen_browser::nim_build`] — the session the wasm export `temen_nim_open` opens, driven to its
//! end — over the self-hosted lane's toolchain and a program tree, and reports how long the build
//! took, the process's peak resident memory, and what the run held at its end. A measurement, not a
//! gate: the engine's cost before the wasm factor, from the code the browser runs. It also writes
//! the library pack the browser seeds ([`temen_browser::library_pack`]).
//!
//! ```text
//! nimbuild <toolchain-dir> <tree> <prog.nim> [--at <dir>] [--lib <pack>] [--pack <pack>]
//!          [--expect <module.temen>] [--leaves]
//! ```
//!
//! `<toolchain-dir>` holds what `scripts/ci/nim-selfhost-lane.sh` builds: `nimony.temen`,
//! `nifmake.temen`, `nimsem.temen`, `nifler2.temen`, `hexer.temen`, the shell's `sh.ir`, and the
//! guest libc as `libc.temeno`. `<tree>` holds `lib/` and `<prog.nim>`; it is seeded at `--at` (by
//! default its host path, where the lane builds), without its `bin/` or `nimcache/`.
//!
//! - `--lib <pack>` seeds a library pack after the tree, so the build compiles only the program's
//!   own modules. The pack must come from a build at the same `--at`.
//! - `--pack <pack>` writes the library pack of this build: build a program that imports the
//!   library, and the pack holds everything it compiled of the library.
//! - `--expect <module.temen>` is the module the build must link.
//! - `--leaves` tiers up each leaf process at its entry and serves it by bouncing the entry, the
//!   native stand-in for running it emitted (#1896).

use std::path::Path;
use std::time::Instant;

/// The toolchain's guests. `temen-link` is not one of them: [`temen_browser::nim_build`] serves it
/// natively.
const TOOLS: [&str; 5] = ["nimony", "nifmake", "nimsem", "nifler2", "hexer"];

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
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let leaves = args
        .iter()
        .position(|a| a == "--leaves")
        .map(|i| args.remove(i))
        .is_some();
    let mut flag = |name: &str| {
        let i = args.iter().position(|a| a == name)?;
        let value = args
            .get(i + 1)
            .unwrap_or_else(|| panic!("{name} needs a value"))
            .clone();
        args.drain(i..=i + 1);
        Some(value)
    };
    let (at, lib, pack, expect) = (
        flag("--at"),
        flag("--lib"),
        flag("--pack"),
        flag("--expect"),
    );
    let [tools, tree, prog] = &args[..] else {
        panic!(
            "usage: nimbuild <toolchain-dir> <tree> <prog.nim> [--at <dir>] [--lib <pack>] \
             [--pack <pack>] [--expect <module.temen>] [--leaves]"
        );
    };
    let tools = Path::new(tools);
    let tree = std::fs::canonicalize(tree).unwrap_or_else(|e| panic!("{tree}: {e}"));
    let dir = at.unwrap_or_else(|| tree.to_string_lossy().into_owned());

    let decode = |name: &str| {
        let m = temen_encode::decode_module(&read(&tools.join(format!("{name}.temen"))))
            .unwrap_or_else(|e| panic!("decode {name}: {e:?}"));
        temen_verify::verify_module(&m).unwrap_or_else(|e| panic!("verify {name}: {e:?}"));
        m
    };
    let modules: Vec<temen_ir::Module> = TOOLS.iter().map(|t| decode(t)).collect();
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
    for (m, [a, b]) in modules.iter().zip(&paths) {
        commands.push((m, vec![a.as_str(), b.as_str()]));
    }

    // The tree, without its `bin/` (the toolchain, which the guest has as commands) or its
    // `nimcache/` (what a build writes); then the library pack, newer than the sources it was built
    // from; then the guest libc where `temen-link` looks for it.
    let mut files: Vec<(String, Vec<u8>)> = Vec::new();
    let mut top: Vec<_> = std::fs::read_dir(&tree)
        .unwrap_or_else(|e| panic!("{}: {e}", tree.display()))
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
    let lib = lib.map(|p| read(Path::new(&p)));
    let lib: Vec<(&str, &[u8])> = lib
        .as_deref()
        .map_or(Vec::new(), temen_browser::blob_entries);
    let libc = read(&tools.join("libc.temeno"));
    let files: Vec<(&str, &[u8])> = files
        .iter()
        .map(|(p, b)| (p.as_str(), b.as_slice()))
        .chain(lib.iter().copied())
        .chain([("/lib/temen/libc.temeno", libc.as_slice())])
        .collect();

    let argv: [&[u8]; 4] = [b"bin/nimony", b"t", b"--isMain", prog.as_bytes()];
    let t0 = Instant::now();
    let b = temen_browser::nim_build(&modules[0], &commands, &files, &argv, &dir, leaves)
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
        "status {} exit {} in {secs:.1} s, peak RSS {} MiB, {:?}, {} files seeded ({} from the library pack), {} leaf processes",
        b.status,
        b.exit_code,
        peak_rss_mib(),
        b.footprint,
        files.len(),
        lib.len(),
        b.leaves
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
            read(Path::new(&e)) == built,
            "the module built on the browser's tier is not {e}"
        );
        eprintln!("✅ it is {e}, byte for byte");
    }
    if let Some(p) = pack {
        let library = temen_browser::library_pack(&b.posix, &dir);
        let entries: Vec<(&str, &[u8])> = library
            .iter()
            .map(|(n, b)| (n.as_str(), b.as_slice()))
            .collect();
        let blob = temen_browser::registry_blob(&entries);
        std::fs::write(&p, &blob).unwrap_or_else(|e| panic!("{p}: {e}"));
        eprintln!(
            "wrote the library pack {p}: {} files, {} bytes",
            entries.len(),
            blob.len()
        );
    }
}
