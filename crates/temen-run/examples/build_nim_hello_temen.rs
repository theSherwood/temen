//! **Build the `nim_hello.temen` playground asset** — a real Nim program compiled to a runnable
//! powerbox `.temen` through the whole nimony → temen-leng pipeline, the analog of the chibicc /
//! `temen-leng` asset lanes but for a *compiled Nim program that runs* (not a compiler).
//!
//! Pipeline: `nimony c` (→ nifler → nimony → hexer, then hexer's dead-code elimination) emits the
//! program's modules as DCE'd Leng — the `.c.nif`s `nimony t` hands `temen-link` — then
//! `temen_leng::link_nim_powerbox` bridges nimony's bottom edge to the §3e powerbox (compute leaves →
//! shim, `sysWrite(fd,buf,len)` → the STREAM `write(buf,len)` cap), and the merged powerbox module is
//! verified and re-serialized with `temen_encode::encode_module` — the browser-loadable form the
//! playground's `temen_run_onramp` runs (host grants stdout, so the program prints for real).
//!
//! ```text
//! NIMONY_BIN=<abs>/nimony/bin NIM_BIN=<dir of nim> \
//!   cargo run --release -p temen-run --example build_nim_hello_temen -- \
//!     crates/temen-run/demos/nim_hello/hello.nim browser/web/assets/nim_hello.temen
//! ```
//!
//! Needs the nimony toolchain (the vendored submodules, built by `scripts/ci/provision-nimony.sh`).
//! Re-run + commit the asset whenever the IR/ABI/encoder or the bridge changes — the same
//! code-coupled asset discipline as `chibicc.temen` / `temen-leng.temen`.

use std::path::Path;
use std::process::Command;

fn main() {
    // `[--posix] [--root <tree>] [--nimcache <dir>] <src> <out.temen>`.
    //
    // The program is built for the Temen platform, `-d:temen`, as `nimony t` builds what it builds
    // (#763): a compiler built for Temen builds the programs it runs itself (compile-time
    // evaluation, plugins) for Temen too.
    //
    // `--root` compiles **in tree**, the way nimony is normally invoked: `<src>` is then relative to
    // `<tree>`, which is also the path nimony names the program's build directory by
    // (`nim_program_units`). A nimony phase's imports are relative so it cannot be copied to a
    // scratch dir, and running from the file's own directory instead creates a second `nimcache`
    // beside the source and names the program differently.
    //
    // `--nimcache` gives the build a cache of its own instead of `<tree>/nimcache`, which every
    // in-tree build shares: fine one build after another, a race when several run at once.
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let posix = argv.iter().any(|a| a == "--posix");
    let root = argv
        .iter()
        .position(|a| a == "--root")
        .and_then(|i| argv.get(i + 1).cloned());
    let nimcache = argv
        .iter()
        .position(|a| a == "--nimcache")
        .and_then(|i| argv.get(i + 1).cloned());
    let positional: Vec<&String> = {
        let mut skip_next = false;
        argv.iter()
            .filter(|a| {
                if skip_next {
                    skip_next = false;
                    return false;
                }
                if *a == "--root" || *a == "--nimcache" {
                    skip_next = true;
                    return false;
                }
                *a != "--posix"
            })
            .collect()
    };
    let [nim, out] = positional.as_slice() else {
        panic!(
            "usage: build_nim_hello_temen [--posix] [--root <tree>] [--nimcache <dir>] <prog.nim> \
             <out.temen>"
        );
    };
    let (nim, out) = (nim.to_string(), out.to_string());

    // Run `nimony c --isMain`; link the program's DCE'd `.c.nif` modules.
    let nim_path = Path::new(&nim);
    let root_path = root.as_deref().map(Path::new);
    let dir = root_path.unwrap_or_else(|| nim_path.parent().unwrap_or(Path::new(".")));
    let file: &Path = if root_path.is_some() {
        nim_path
    } else {
        Path::new(nim_path.file_name().expect("nim file name"))
    };
    let path_env = std::env::var("PATH").unwrap_or_default();
    let mut prefix = Vec::new();
    if let Ok(d) = std::env::var("NIMONY_BIN") {
        prefix.push(d);
    }
    if let Ok(d) = std::env::var("NIM_BIN") {
        prefix.push(d);
    }
    let full_path = if prefix.is_empty() {
        path_env
    } else {
        format!("{}:{}", prefix.join(":"), path_env)
    };
    // nimony resolves a relative `--nimcache:` against its own cwd (`dir`), so it gets the absolute
    // path the caller meant, relative to ours.
    let cache = match &nimcache {
        Some(c) => std::path::absolute(c).unwrap_or_else(|e| panic!("--nimcache {c}: {e}")),
        None => dir.join("nimcache"),
    };
    let mut nimony = Command::new("nimony");
    nimony.args(["c", "--isMain", "-d:temen"]);
    if nimcache.is_some() {
        nimony.arg(format!("--nimcache:{}", cache.display()));
    }
    let status = nimony
        .arg(file)
        .current_dir(dir)
        .env("PATH", &full_path)
        .status()
        .expect("run nimony (set NIMONY_BIN/NIM_BIN or put nimony on PATH)");
    assert!(status.success(), "nimony c failed");

    // The program's DCE'd modules, as the build wrote them to its own build directory: what
    // `nimony t` hands `temen-link`, so a module built here is the one an in-guest `nimony t` of
    // the same source links. They are named by the path nimony was handed — `file`, relative to its
    // cwd `dir` — so an in-tree build, whose `nimcache` holds other programs too, links its own.
    let given = file.to_str().expect("utf-8 program path");
    let mods = temen_run::nim_program_units(&cache, given).unwrap_or_else(|e| {
        panic!("{e}\n    for an in-tree source pass `--root <tree> <path-relative-to-tree>`")
    });
    assert!(
        mods.iter().any(|(s, _)| s.starts_with("sysv")),
        "no `system` module in {:?}",
        mods.iter().map(|(s, _)| s).collect::<Vec<_>>()
    );

    let texts: Vec<(&str, std::borrow::Cow<str>)> = mods
        .iter()
        .map(|(stem, nif)| (stem.as_str(), temen_leng::nif_text(nif)))
        .collect();
    let units: Vec<temen_leng::WholeModule> = texts
        .iter()
        .map(|(stem, src)| temen_leng::WholeModule { stem, src })
        .collect();
    // Two runtimes over one compute half (see `temen_leng::nim_posix_runtime`): the default folds
    // the syscalls onto the single STREAM `write` cap — right for a program that only prints — while
    // `--posix` forwards them to a real `temen_posix` personality under its own op names (`__px_*`,
    // #1668), so the result binds like any command — at root, or `execve`'d. A compiler phase needs the latter: it opens, reads and writes files, and (for
    // nimsem) spawns `nifler` through an `exec` cap.
    // The **prebuilt guest libc** (`LIBC_SERVED`): `snprintf`/`strtod`/libm, which no hand-written
    // shim reasonably carries. Without it those stay unbound manifest imports and the program cannot
    // be instantiated — which is how a real phase first failed here, on 24 unbound trig leaves
    // (`arctan.0.`, `cos.1.`, …) that `std/math` declares and nimsem's closure pulls in. The compute
    // shim does not serve libm; `nim_libc_units` does.
    let libc = std::env::var("TEMEN_PG_LIBC")
        .ok()
        .map(std::path::PathBuf::from)
        .or_else(|| {
            let d = Path::new("browser/web/assets/pg_libc.temeno");
            d.exists().then(|| d.to_path_buf())
        })
        .map(|p| std::fs::read(&p).unwrap_or_else(|e| panic!("read libc {p:?}: {e}")));
    let module = if posix {
        let (px_names, px_sigs) = temen_posix::cap_vtable();
        temen_leng::link_nim_posix(&units, (&px_names, &px_sigs), libc.as_deref())
            .unwrap_or_else(|e| panic!("nim→posix bridge: {e}"))
    } else {
        temen_leng::link_nim_powerbox(&units, libc.as_deref())
            .unwrap_or_else(|e| panic!("nim→powerbox bridge: {e}"))
    };
    // Verify before shipping (the escape-freedom floor, DESIGN §2a) and sanity-check the entry shape.
    temen_verify::verify_module(&module).unwrap_or_else(|e| panic!("verify: {e:?}"));
    assert!(
        temen_run::is_named_powerbox_entry(&module),
        "linked module is not a powerbox entry (paramless func-0 `_start`)"
    );
    let bytes = temen_encode::encode_module(&module);
    std::fs::write(&out, &bytes).unwrap_or_else(|e| panic!("write {out}: {e}"));
    eprintln!(
        "wrote {out} ({} bytes, {} funcs, {} imports) — a real Nim program as a runnable {} module",
        bytes.len(),
        module.funcs.len(),
        module.imports.len(),
        if posix {
            "posix-personality"
        } else {
            "powerbox"
        },
    );
}
