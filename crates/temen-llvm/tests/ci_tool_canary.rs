//! CI canary against silent tool-skip rot (ISSUES.md, "Platform-coverage skips & caps" inventory).
//!
//! ~30 tests in this crate **auto-skip** when a pipeline tool is absent (`clang`/`cc` for the C/C++
//! corpus, `llvm-dis` for the textual reader, `llvm-as` for hand-written `.ll`, and
//! `rustc` / `llvm-link` / `opt` for the `peval_*` probes). That keeps contributor
//! machines unburdened — but it means that if a CI setup step ever rots (an apt package rename, a
//! rustup failure, a PATH change), the whole `temen-llvm` lane goes green while testing nothing.
//! That failure shape is not hypothetical: the TSan and ASan lanes ran *nothing* for two weeks in
//! June before anyone noticed (ISSUES.md I19/I20), because a lane that fails during setup looks
//! like a lane that passes its (never-run) tests.
//!
//! So: on CI (GitHub Actions sets `CI=true` on every runner) on Linux — the only platform whose CI
//! job installs this toolchain — every tool the auto-skips probe for must actually be runnable.
//! Anywhere else this test is a no-op.

use std::process::Command;

/// Same probe shape as `tests/common/mod.rs::tool_ok` — the canary must agree with the skips.
fn runnable(cmd: &str, args: &[&str]) -> bool {
    Command::new(cmd)
        .args(args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[test]
fn ci_has_every_tool_the_auto_skips_probe_for() {
    if std::env::var_os("CI").is_none() || !cfg!(target_os = "linux") {
        eprintln!(
            "note: not Linux CI — tool canary is a no-op (auto-skips stay permissive locally)"
        );
        return;
    }
    // Everything the in-crate skips (`have()`s / `toolchain_present()` / `llvm-dis` in the reader)
    // probe for, minus network fetchers (`curl`/`unzip`) and `make` (only `#[ignore]`d benches).
    let tools: &[(&str, &[&str])] = &[
        ("clang", &["--version"]),
        ("clang++", &["--version"]),
        ("cc", &["--version"]),
        ("llvm-dis", &["--version"]),
        ("llvm-as", &["--version"]),
        ("llvm-link", &["--version"]),
        ("opt", &["--version"]),
        ("llvm-objcopy", &["--version"]),
        ("ar", &["--version"]),
        ("rustc", &["--version"]),
    ];
    let missing: Vec<&str> = tools
        .iter()
        .filter(|(cmd, args)| !runnable(cmd, args))
        .map(|(cmd, _)| *cmd)
        .collect();
    assert!(
        missing.is_empty(),
        "CI runner is missing {missing:?} — the temen-llvm tests that need these would silently \
         auto-skip, so this lane would be green while testing nothing. Fix the CI setup step \
         (ci.yml `temen-llvm` job: `scripts/ci/install-llvm.sh` + PATH) — do not delete this canary."
    );
    // The peval probes feed the default `rustc`'s IR to `llvm-link`/`opt`, which only ingest IR of
    // their own LLVM version or older — so the pinned LLVM major must equal the stable rustc's.
    let rustc_llvm = llvm_major(&["rustc", "-vV"]);
    let link_llvm = llvm_major(&["llvm-link", "--version"]);
    assert_eq!(
        rustc_llvm, link_llvm,
        "rustc's LLVM major ({rustc_llvm:?}) != the installed llvm-link's ({link_llvm:?}) — the \
         peval probes would auto-skip. Bump `LLVM_MAJOR` in scripts/ci/install-llvm.sh to match \
         `RUST_STABLE`'s LLVM (`rustc -vV`), or vice versa."
    );
}

/// The LLVM major in a tool's version banner (`LLVM version: 22.1.6` / `LLVM version 22.1.6`).
fn llvm_major(cmd: &[&str]) -> Option<u32> {
    let out = Command::new(cmd[0]).args(&cmd[1..]).output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let rest = text.split("LLVM version").nth(1)?;
    rest.trim_start_matches([':', ' '])
        .split('.')
        .next()?
        .parse()
        .ok()
}
