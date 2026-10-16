//! **The nim card's toolchain** (`browser/web/assets/nimony.blob.gz`, #958, #2442): nimony's own
//! driver builds a program in-guest on the JIT, and this writes what the card ships. That is the
//! toolchain, nimony's library, and everything the build compiled of the library (its library pack),
//! so a card build compiles only its own modules. `scripts/rebuild-assets.sh` runs it (step
//! `nim_card`).
//!
//! ```text
//! nim_card_bundle <toolchain-dir> <tree> <prog.nim> --at <dir> --bundle <out> [--pack <out>]
//! ```
//!
//! `<toolchain-dir>` holds what `scripts/nim-toolchain.sh` builds: the tools' `.temen`, the shell's
//! `sh.ir`, and the guest libc as `libc.temeno`. `<tree>` holds `lib/`, `src/` and `<prog.nim>`. It
//! is seeded at `--at`, where the card builds, without its `bin/` or `nimcache/`.
//!
//! - `--bundle <out>` is a registry blob (the framing `temen_browser::blob_entries` reads) of three
//!   entries. `cwd` is `--at`, which the pack names. `commands` is the toolchain as the card's
//!   `temen_nim_open` takes it, the driver first: each tool at its two paths, then `/bin/sh`. `files`
//!   is what the build was seeded with before its program, then its library pack
//!   ([`temen_run::nim_library_pack`]), then the guest libc.
//! - `--pack <out>` is the library pack alone, which `nimbuild --lib` seeds.
//!
//! The guest has one CPU, so nifmake runs one job at a time. The pack, in the order the build wrote
//! it, then comes out the same from every build of the same inputs. `temen-link` runs on the host,
//! as the card serves it.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

/// The toolchain's guests, the driver first. `temen-link` runs on the host ([`temen_link`]).
const TOOLS: [&str; 5] = ["nimony", "nifmake", "nimsem", "nifler2", "shoggoth"];

fn read(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The toolchain's `temen-link`, run on the host over the files of the process that execs it:
/// [`temen_leng::link_command`], the function the in-guest command is built from.
fn temen_link(argv: &[String], files: &mut dyn temen_posix::CommandFiles) -> i32 {
    let (names, sigs) = temen_posix::cap_vtable();
    let args: Vec<&str> = argv.iter().skip(1).map(String::as_str).collect();
    let files = core::cell::RefCell::new(files);
    temen_leng::link_command(
        &args,
        (&names, &sigs),
        &mut |path| files.borrow_mut().read(path),
        &mut |path, bytes| files.borrow_mut().write(path, bytes),
    )
}

/// A registry blob: all integers little-endian, a `u32` entry count, then per entry a `u32` name
/// length, the name's UTF-8 bytes, a `u32` length and that many bytes. The browser reads it
/// (`temen_browser::blob_entries`).
fn registry_blob(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let len = |n: usize| {
        u32::try_from(n)
            .expect("a blob entry under 4 GiB")
            .to_le_bytes()
    };
    let mut out = len(entries.len()).to_vec();
    for (name, bytes) in entries {
        out.extend(len(name.len()));
        out.extend(name.as_bytes());
        out.extend(len(bytes.len()));
        out.extend(*bytes);
    }
    out
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut flag = |name: &str| {
        let i = args.iter().position(|a| a == name)?;
        let value = args
            .get(i + 1)
            .unwrap_or_else(|| panic!("{name} needs a value"))
            .clone();
        args.drain(i..=i + 1);
        Some(value)
    };
    let (at, bundle, pack) = (flag("--at"), flag("--bundle"), flag("--pack"));
    let ([tools, tree, prog], Some(dir)) = (&args[..], at) else {
        panic!(
            "usage: nim_card_bundle <toolchain-dir> <tree> <prog.nim> --at <dir> --bundle <out> \
             [--pack <out>]"
        );
    };
    let tools = Path::new(tools);

    let encoded: Vec<Vec<u8>> = TOOLS
        .iter()
        .map(|t| read(&tools.join(format!("{t}.temen"))))
        .collect();
    let decode = |(name, bytes): (&&str, &Vec<u8>)| {
        let m =
            temen_encode::decode_module(bytes).unwrap_or_else(|e| panic!("decode {name}: {e:?}"));
        temen_verify::verify_module(&m).unwrap_or_else(|e| panic!("verify {name}: {e:?}"));
        m
    };
    let modules: Vec<temen_ir::Module> = TOOLS.iter().zip(&encoded).map(decode).collect();
    let sh = String::from_utf8(read(&tools.join("sh.ir"))).expect("sh.ir is text");
    let sh = temen_text::parse_module(&sh).expect("parse sh.ir");
    temen_verify::verify_module(&sh).expect("verify sh.ir");

    // `/bin/sh`, and each tool where the driver looks for it (beside its own `bin/nimony`) and where
    // the shell's `PATH` walk finds a bare name.
    let paths: Vec<Vec<String>> = TOOLS
        .iter()
        .map(|t| vec![format!("{dir}/bin/{t}"), format!("/bin/{t}")])
        .collect();
    let mut commands = vec![(
        temen_interp::PreparedModule::new(sh.clone()),
        vec!["/bin/sh".to_string()],
    )];
    for (m, p) in modules.iter().zip(&paths) {
        commands.push((temen_interp::PreparedModule::new(m.clone()), p.clone()));
    }

    let tree_files = temen_run::nim_tree_files(Path::new(tree), &dir);
    let libc = read(&tools.join("libc.temeno"));
    let libc = ("/lib/temen/libc.temeno", libc.as_slice());

    let (posix, make) = temen_posix::cap(0, 0, Vec::new());
    posix.set_env("PATH", "/bin");
    posix.set_cwd(&dir);
    for path in [
        format!("{dir}/bin/temen-link"),
        "/bin/temen-link".to_string(),
    ] {
        posix.register_host_command(&path, Arc::new(temen_link));
    }
    for (path, bytes) in &tree_files {
        posix.write_file(path, bytes);
    }
    posix.write_file(libc.0, libc.1);

    // No `validator` ships with the toolchain, so the build says so up front instead of nimony
    // warning about it at every plugin it builds (the card passes the same).
    let argv: Vec<String> = ["bin/nimony", "t", "--novalidate", "--isMain", prog]
        .iter()
        .map(|a| a.to_string())
        .collect();
    let t0 = Instant::now();
    let outcome = temen_run::nim_noc_run(
        modules[0].clone(),
        &posix,
        Arc::new(make),
        &argv,
        &temen_run::ExecGrants {
            commands: &commands,
            built: true,
        },
        temen_run::Backend::Jit,
        None,
        Some(1),
    );
    let secs = t0.elapsed().as_secs_f64();

    // `nimcache/t/<main>/<prog>.temen`, `<main>` the driver's stem for the program.
    let built = format!(
        "{dir}/nimcache/t/{}/{}.temen",
        temen_run::nim_module_suffix(prog, &temen_run::NIMONY_TREE_PATHS),
        Path::new(prog).file_stem().unwrap().to_string_lossy()
    );
    let (Ok(_), Some(_)) = (&outcome, posix.read_file(&built)) else {
        eprint!(
            "--- stdout ---\n{}--- stderr ---\n{}",
            String::from_utf8_lossy(&posix.stdout()),
            String::from_utf8_lossy(&posix.stderr())
        );
        panic!("`nimony t` built no {built}: {outcome:?}");
    };
    let library = temen_run::nim_library_pack(&posix, &dir);
    eprintln!(
        "built {built} in-guest on the JIT in {secs:.1} s; its library pack is {} files",
        library.len()
    );
    let library: Vec<(&str, &[u8])> = library
        .iter()
        .map(|(n, b)| (n.as_str(), b.as_slice()))
        .collect();
    if let Some(p) = pack {
        std::fs::write(&p, registry_blob(&library)).unwrap_or_else(|e| panic!("{p}: {e}"));
    }
    if let Some(p) = bundle {
        let sh = temen_encode::encode_module(&sh);
        let named: Vec<String> = paths.iter().map(|p| p.join("\n")).collect();
        let commands: Vec<(&str, &[u8])> = named
            .iter()
            .map(String::as_str)
            .zip(encoded.iter().map(Vec::as_slice))
            .chain([("/bin/sh", sh.as_slice())])
            .collect();
        let program = format!("{dir}/{prog}");
        let seeded: Vec<(&str, &[u8])> = tree_files
            .iter()
            .filter(|(p, _)| *p != program)
            .map(|(p, b)| (p.as_str(), b.as_slice()))
            .chain(library.iter().copied())
            .chain([libc])
            .collect();
        let blob = registry_blob(&[
            ("cwd", dir.as_bytes()),
            ("commands", &registry_blob(&commands)),
            ("files", &registry_blob(&seeded)),
        ]);
        std::fs::write(&p, &blob).unwrap_or_else(|e| panic!("{p}: {e}"));
        eprintln!(
            "wrote the toolchain {p}: {} commands, {} files, {} bytes",
            commands.len(),
            seeded.len(),
            blob.len()
        );
    }
}
