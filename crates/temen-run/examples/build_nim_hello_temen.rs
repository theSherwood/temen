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
    let mut args = std::env::args().skip(1);
    let nim = args
        .next()
        .expect("usage: build_nim_hello_temen <prog.nim> <out.temen>");
    let out = args
        .next()
        .expect("usage: build_nim_hello_temen <prog.nim> <out.temen>");

    // Run `nimony c --isMain` in the source's directory; collect the emitted `.x.nif` modules.
    let nim_path = Path::new(&nim);
    let dir = nim_path.parent().unwrap_or(Path::new("."));
    let file = nim_path.file_name().expect("nim file name");
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
    collect_x_nif(&dir.join("nimcache"), &mut mods);
    assert!(
        mods.iter().any(|(s, _)| s.starts_with("sysv")),
        "no `system` module in {:?}",
        mods.iter().map(|(s, _)| s).collect::<Vec<_>>()
    );

    let units: Vec<temen_leng::WholeModule> = mods
        .iter()
        .map(|(stem, src)| temen_leng::WholeModule { stem, src })
        .collect();
    let module = temen_leng::link_nim_powerbox(&units)
        .unwrap_or_else(|e| panic!("nim→powerbox bridge: {e}"));
    // Verify before shipping (the escape-freedom floor, DESIGN §2a) and sanity-check the entry shape.
    temen_verify::verify_module(&module).unwrap_or_else(|e| panic!("verify: {e:?}"));
    assert!(
        temen_run::is_named_powerbox_entry(&module),
        "linked module is not a powerbox entry (paramless func-0 `_start`)"
    );
    let bytes = temen_encode::encode_module(&module);
    std::fs::write(&out, &bytes).unwrap_or_else(|e| panic!("write {out}: {e}"));
    eprintln!(
        "wrote {out} ({} bytes, {} funcs) — a real Nim program as a runnable powerbox module",
        bytes.len(),
        module.funcs.len()
    );
}

fn collect_x_nif(dir: &Path, out: &mut Vec<(String, String)>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            collect_x_nif(&p, out);
        } else if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
            if let Some(stem) = name.strip_suffix(".x.nif") {
                if out.iter().all(|(s, _)| s != stem) {
                    let bytes = std::fs::read(&p).unwrap();
                    out.push((
                        stem.to_string(),
                        String::from_utf8_lossy(&bytes).into_owned(),
                    ));
                }
            }
        }
    }
}
