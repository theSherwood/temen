//! **Build the `nim_hello.temen` playground asset** — a real Nim program compiled to a runnable
//! powerbox `.temen` through the whole nimony → temen-leng pipeline, the analog of the chibicc /
//! `temen-leng` asset lanes but for a *compiled Nim program that runs* (not a compiler).
//!
//! Pipeline: `nimony c` (→ nifler → nimony → hexer) emits the program + `system` module Leng, then
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
    // `[--posix] [--root <tree>] <src> <out.temen>`.
    //
    // `--root` compiles **in tree**, the way nimony is normally invoked: `<src>` is then relative to
    // `<tree>`, which is also the string nimony records in each module's line info and therefore the
    // one `nim_program_closure` matches on. A nimony phase's imports are relative so it cannot be
    // copied to a scratch dir, and running from the file's own directory instead creates a second
    // `nimcache` beside the source and records a path that matches nothing.
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let posix = argv.iter().any(|a| a == "--posix");
    let root = argv
        .iter()
        .position(|a| a == "--root")
        .and_then(|i| argv.get(i + 1).cloned());
    let positional: Vec<&String> = {
        let mut skip_next = false;
        argv.iter()
            .filter(|a| {
                if skip_next {
                    skip_next = false;
                    return false;
                }
                if *a == "--root" {
                    skip_next = true;
                    return false;
                }
                *a != "--posix"
            })
            .collect()
    };
    let [nim, out] = positional.as_slice() else {
        panic!("usage: build_nim_hello_temen [--posix] [--root <tree>] <prog.nim> <out.temen>");
    };
    let (nim, out) = (nim.to_string(), out.to_string());

    // Run `nimony c --isMain`; collect the emitted `.x.nif` modules.
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
    let status = Command::new("nimony")
        .args(["c", "--isMain"])
        .arg(file)
        .current_dir(dir)
        .env("PATH", &full_path)
        .status()
        .expect("run nimony (set NIMONY_BIN/NIM_BIN or put nimony on PATH)");
    assert!(status.success(), "nimony c failed");

    let mut mods: Vec<(String, String)> = Vec::new();
    temen_run::collect_x_nif(&dir.join("nimcache"), &mut mods);
    // An **in-tree** build (a nimony phase, whose imports are relative so it cannot be copied to a
    // scratch dir) shares one `nimcache` with every other program built there. Narrow to this
    // program's own closure, or the link sees two `main`s.
    if let Some(note) = temen_run::nim_program_closure(&dir.join("nimcache"), &nim, &mut mods) {
        // A fallback means the closure belongs to some *other* program. Linking it anyway writes a
        // plausible artifact for the wrong source and reports success — which is what happened the
        // first time this ran: 581 funcs written where nimsem has ~12,725, under the message
        // "wrote ... a real Nim program". Refuse instead.
        panic!(
            "{note}\n    `{nim}` names no module in {:?} — refusing to link a different program's \
             closure. For an in-tree source pass `--root <tree> <path-relative-to-tree>`.",
            dir.join("nimcache")
        );
    }
    assert!(
        mods.iter().any(|(s, _)| s.starts_with("sysv")),
        "no `system` module in {:?}",
        mods.iter().map(|(s, _)| s).collect::<Vec<_>>()
    );

    let units: Vec<temen_leng::WholeModule> = mods
        .iter()
        .map(|(stem, src)| temen_leng::WholeModule { stem, src })
        .collect();
    // Two runtimes over one compute half (see `temen_leng::nim_posix_runtime`): the default folds
    // the syscalls onto the single STREAM `write` cap — right for a program that only prints — while
    // `--posix` leaves them as **retained manifest imports** a host binds to a real `temen_posix`
    // personality. A compiler phase needs the latter: it opens, reads and writes files, and (for
    // nimsem) spawns `nifler` through an `exec` cap.
    let module = if posix {
        let runtime = temen_leng::nim_posix_runtime(&units)
            .unwrap_or_else(|e| panic!("nim posix runtime: {e}"));
        temen_leng::link_whole_powerbox_manifest(&units, runtime)
            .unwrap_or_else(|e| panic!("nim→posix bridge: {e}"))
    } else {
        temen_leng::link_nim_powerbox(&units, None)
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
