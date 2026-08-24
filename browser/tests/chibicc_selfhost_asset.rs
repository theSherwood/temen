//! **The shipped `chibicc.temen` self-hosts, byte-for-byte** (SELFHOST_C.md §7 step 5 — the playground
//! capstone's asset gate). The playground's "chibicc compiles its own source" card runs the *committed*
//! `chibicc.temen` in `--emit-object` mode over chibicc's own cc1 TUs; this test pins that that asset
//! emits each TU's object **byte-identical to a native `chibicc --emit-object`** on the same source —
//! the same interp/JIT-agnostic fixpoint `crates/temen/tests/c_link.rs` proves for the *linked* whole-cc1,
//! here for the *shipped on-ramp asset* the browser actually loads (the artifact that was silently stale
//! for emit-object until this feature rebuilt it — so the gate exists to keep it from drifting again).
//!
//! Drives the cdylib's `temen_selfhost_emit_object_fs` bytecode entry — the exact path the card's
//! interpreter tier and the JIT tier's oracle take (the JIT is diffed against this bytecode output in
//! the browser). **Linux-only** (chibicc pulls glibc's header tree, hardcoded include paths) and
//! **fail-soft**: SKIPs without `chibicc.temen` or a native `chibicc`, exactly like the sibling gates.

#![cfg(target_os = "linux")]

use std::path::{Path, PathBuf};
use std::process::Command;

use temen_browser::{
    temen_selfhost_emit_object_fs, temen_stderr_len, temen_stderr_ptr, temen_stdout_len,
    temen_stdout_ptr,
};

const TUS: &[&str] = &[
    "strings.c",
    "hashmap.c",
    "unicode.c",
    "type.c",
    "tokenize.c",
];
const PRELUDE: &str = "crates/temen-run/demos/chibicc_selfhost/selfhost_prelude.h";

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .canonicalize()
        .unwrap()
}
fn native_chibicc() -> PathBuf {
    repo_root().join("frontend/chibicc/chibicc")
}

/// The TU's `#include` closure (mirrors `c_link.rs::chibicc_tu_closure`): `chibicc -M` over the
/// **relative** TU path so repo-local deps come back relative and system headers as `usr/...`, keyed to
/// match the guest's `-I` search + `base_file`. Plus the force-included prelude.
fn closure_image(tu_rel: &str) -> Vec<u8> {
    let root = repo_root();
    let out = Command::new(native_chibicc())
        .args(["-M", "-c", tu_rel, "-Ifrontend/chibicc"])
        .current_dir(&root)
        .output()
        .expect("run chibicc -M");
    assert!(out.status.success(), "chibicc -M failed on {tu_rel}");
    let text = String::from_utf8(out.stdout).unwrap();
    let deps = text.split_once(':').map(|(_, d)| d).unwrap_or("");
    let mut files: Vec<(String, Vec<u8>)> = Vec::new();
    let mut dirs = std::collections::BTreeSet::new();
    let mut seen = std::collections::BTreeSet::new();
    let mut add = |rel: String, real: &Path, dirs: &mut std::collections::BTreeSet<String>| {
        if !seen.insert(rel.clone()) {
            return;
        }
        let mut anc = Path::new(&rel).parent();
        while let Some(d) = anc {
            if d.as_os_str().is_empty() {
                break;
            }
            dirs.insert(d.to_string_lossy().into_owned());
            anc = d.parent();
        }
        files.push((rel, std::fs::read(real).unwrap()));
    };
    for tok in deps.split_whitespace() {
        if tok == "\\" {
            continue;
        }
        let real = if tok.starts_with('/') {
            PathBuf::from(tok)
        } else {
            root.join(tok.trim_start_matches("./"))
        };
        let rel = match real.strip_prefix(&root) {
            Ok(r) => r.to_string_lossy().into_owned(),
            Err(_) => real.to_string_lossy().trim_start_matches('/').to_string(),
        };
        add(rel, &real, &mut dirs);
    }
    add(PRELUDE.to_string(), &root.join(PRELUDE), &mut dirs);
    temen_fs::encode_image(&files, &dirs.into_iter().collect::<Vec<_>>())
}

/// The native reference object: `chibicc --emit-object` on the identical source with the *same relative*
/// flags the cdylib's `chibicc_selfhost_argv` uses (no `--data-page` — the object is canonical), so the
/// only variable is the substrate (guest libc + Temen interpreter vs system libc + native CPU).
fn native_object(tu_rel: &str) -> Vec<u8> {
    let root = repo_root();
    let out = root.join(format!(
        "target/selfhost_ref_{}.temt",
        tu_rel.replace(['/', '.'], "_")
    ));
    let status = Command::new(native_chibicc())
        .args([
            "-cc1",
            "-include",
            PRELUDE,
            "-Ifrontend/chibicc",
            "-Ifrontend/chibicc/include",
            "-I/usr/include/x86_64-linux-gnu",
            "-I/usr/include",
            "--emit-object",
            "-cc1-input",
            tu_rel,
            "-cc1-output",
            out.to_str().unwrap(),
            tu_rel,
        ])
        .current_dir(&root)
        .status()
        .expect("run native chibicc --emit-object");
    assert!(
        status.success(),
        "native chibicc --emit-object failed on {tu_rel}"
    );
    std::fs::read(&out).unwrap()
}

fn chibicc_temen() -> Option<temen_ir::Module> {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/web/assets/chibicc.temen");
    Some(temen_encode::decode_module(&std::fs::read(p).ok()?).expect("decode chibicc.temen"))
}

/// The card's bytecode entry: run `chibicc.temen --emit-object <tu>` over the seeded closure and read the
/// emitted object text off the stdout stash — exactly what `temen_selfhost_emit_object_fs` drives.
fn guest_object(chibicc: &[u8], image: &[u8], tu_rel: &str) -> (i64, Vec<u8>, Vec<u8>) {
    let rv = temen_selfhost_emit_object_fs(
        chibicc.as_ptr(),
        chibicc.len(),
        image.as_ptr(),
        image.len(),
        tu_rel.as_ptr(),
        tu_rel.len(),
        0,
    );
    // SAFETY: single-threaded test; the stash is valid until the next call. A capture may be empty
    // (null ptr / zero len) — `from_raw_parts` requires non-null even at len 0, so guard it.
    let read = |p: *const u8, n: usize| -> Vec<u8> {
        if p.is_null() || n == 0 {
            Vec::new()
        } else {
            unsafe { core::slice::from_raw_parts(p, n).to_vec() }
        }
    };
    let stdout = read(temen_stdout_ptr(), temen_stdout_len());
    let stderr = read(temen_stderr_ptr(), temen_stderr_len());
    (rv, stdout, stderr)
}

#[test]
fn chibicc_temen_self_compiles_byte_identical_to_native() {
    let Some(module) = chibicc_temen() else {
        eprintln!("SKIP: chibicc.temen absent");
        return;
    };
    if !native_chibicc().exists() {
        eprintln!("SKIP: native frontend/chibicc/chibicc not built (make -C frontend/chibicc)");
        return;
    }
    let chibicc = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/web/assets/chibicc.temen"
    ))
    .unwrap();
    // The asset must actually carry --emit-object (the stale-asset regression that motivated this gate):
    // a chibicc that ignores the flag treats it as the input file and emits nothing.
    assert!(
        temen_verify::verify_module(&module).is_ok(),
        "chibicc.temen verifies"
    );

    for tu in TUS {
        let tu_rel = format!("frontend/chibicc/{tu}");
        let image = closure_image(&tu_rel);
        let (rv, guest, stderr) = guest_object(&chibicc, &image, &tu_rel);
        assert!(
            !guest.is_empty(),
            "chibicc.temen --emit-object {tu} emitted nothing (rv {rv}, stderr {:?}) — stale asset?",
            String::from_utf8_lossy(&stderr)
        );
        let native = native_object(&tu_rel);
        assert_eq!(
            String::from_utf8_lossy(&guest),
            String::from_utf8_lossy(&native),
            "chibicc.temen self-compiled {tu} must be byte-identical to native chibicc --emit-object"
        );
    }
}
