//! **The nim card's crawl must not fail silently** (#1364).
//!
//! The whole-card orchestrator crawls each module with `nifler` before nimsem/hexer. When that crawl
//! produced nothing, three layers in a row swallowed it: `temen_run_nifler_crawl_fs` discarded the
//! run's outcome and returned `STATUS_OK`, the JS crawl `continue`d past the module, and a *dependent*
//! module's nimsem then died with `cannot open <other-stem>.s.nif` — a message naming the wrong module
//! and giving no cause. The worker's `catch` turned that into a silent fall back to the tree-walker,
//! i.e. the 8-18 minute Runs users reported as "the nim card is extremely slow".
//!
//! These tests pin the seam that makes such a failure *visible*: a crawl that parses nothing reports a
//! non-OK status and retains the guest's diagnostics, and a crawl that succeeds reports `STATUS_OK`.
//! They need no browser — `temen_run_nifler_crawl_fs` is the same entry the card drives.
//!
//! Gated on the committed `nifler.temen.gz` asset being present; SKIPs otherwise.

use std::io::Write;
use std::process::{Command, Stdio};

/// Inflate a committed `.gz` asset by shelling out to `gzip` — the same dependency-free approach the
/// `nimlink_asset` gate uses.
fn asset(name: &str) -> Option<Vec<u8>> {
    let gz = std::fs::read(format!("web/assets/{name}")).ok()?;
    let mut c = Command::new("gzip")
        .args(["-dc"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let mut stdin = c.stdin.take().unwrap();
    let w = std::thread::spawn(move || {
        let _ = stdin.write_all(&gz);
    });
    let out = c.wait_with_output().ok()?;
    w.join().ok()?;
    out.status.success().then_some(out.stdout)
}

/// The FFI's capture slots (`OUT`, `CRAWL_DIAG`) are process-global — the cdylib is single-threaded
/// in the browser, so they need no locking there, but `cargo test` runs these in parallel threads and
/// they would clobber each other's stash. One lock around the whole call-and-read sequence.
static CRAWL_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Run the card's crawl over one source, returning `(status, produced-bytes, diagnostics)`.
fn crawl(nifler: &[u8], file: &str, src: &str) -> (i32, usize, String) {
    let _guard = CRAWL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let out = "/nimcache/probe.p.nif";
    let rc = unsafe {
        temen_browser::temen_run_nifler_crawl_fs(
            nifler.as_ptr(),
            nifler.len(),
            file.as_ptr(),
            file.len(),
            out.as_ptr(),
            out.len(),
            src.as_ptr(),
            src.len(),
        )
    };
    let produced = temen_browser::temen_stdout_len();
    let dl = temen_browser::temen_run_nifler_crawl_diag();
    let diag = if dl == 0 {
        String::new()
    } else {
        // SAFETY: `temen_run_nifler_crawl_diag` just stashed `dl` bytes at the stdout slot.
        unsafe {
            String::from_utf8_lossy(std::slice::from_raw_parts(
                temen_browser::temen_stdout_ptr(),
                dl,
            ))
            .into_owned()
        }
    };
    (rc, produced, diag)
}

/// A source the shipped nifler parses: the crawl reports success and produces a `.p.nif`.
#[test]
fn a_parsable_source_crawls_ok() {
    let Some(nifler) = asset("nifler.temen.gz") else {
        eprintln!("SKIP a_parsable_source_crawls_ok (no nifler asset)");
        return;
    };
    let (rc, produced, _diag) = crawl(&nifler, "/p.nim", "let a = 1.5\nlet b = 2\n");
    assert_eq!(rc, 0, "a parsable source must crawl OK");
    assert!(produced > 0, "a successful crawl must produce a .p.nif");
}

/// **The seam this test file exists for.** A source the shipped nifler *cannot* parse must surface as
/// a non-OK status with a diagnostic — not as `STATUS_OK` and an empty output, which is what let a
/// stale asset masquerade as "the module simply has no imports" all the way to a bogus nimsem error.
///
/// The probe is a float literal that misses `parseBiggestFloat`'s fast path (>2 exponent digits), so
/// it reaches `c_strtod`. Whether *that* parses depends on the committed asset's vintage; what this
/// test pins is that either answer is **reported**, never swallowed.
#[test]
fn an_unparsable_source_reports_a_reason() {
    let Some(nifler) = asset("nifler.temen.gz") else {
        eprintln!("SKIP an_unparsable_source_reports_a_reason (no nifler asset)");
        return;
    };
    let (rc, produced, diag) = crawl(&nifler, "/p.nim", "let a = 1.5e100\n");
    if rc == 0 {
        assert!(produced > 0, "STATUS_OK must mean a .p.nif was produced");
        return; // the asset handles it — nothing to report, and that is also correct
    }
    assert!(
        produced == 0,
        "a non-OK crawl must not claim to have produced output"
    );
    assert!(
        !diag.trim().is_empty(),
        "a failed crawl must retain a diagnostic (got none) — without it the card cannot say why"
    );
}

/// **The committed `nifler.temen.gz` must parse the float literals the stdlib actually contains**
/// (#1364). This is the coherence check between the shipped asset and `nifler_shim.c`: the asset is
/// a *build artifact of that shim*, and nothing else notices when it goes stale.
///
/// It went stale once already, for nine days. The `strtod` shim (`nifler_shim.c`, 2026-09-11) rebuilt
/// `nimsem`/`hexer`/`nimsem_ce`/`hexer_ce` but **not** `nifler` — the one phase that lexes float
/// literals out of source. Every literal missing `parseBiggestFloat`'s fast path reached the
/// still-stubbed `c_strtod` and trapped `Unreachable`, so `import std/math` (or anything reaching
/// `std/fenv`) could not be crawled at all and those Runs fell back to the multi-minute tree-walker.
///
/// If this fails, the asset needs regenerating — not the test relaxing:
/// `TEMEN_NIFLER_EMIT_ASSET=1 bash crates/temen-run/demos/nifler_temen/build_nifler_temen.sh`.
#[test]
fn shipped_nifler_parses_the_float_literals_the_stdlib_uses() {
    let Some(nifler) = asset("nifler.temen.gz") else {
        eprintln!("SKIP (no nifler asset)");
        return;
    };
    // Each of these appears in `std/fenv` / `std/math` and misses `parseBiggestFloat`'s fast path
    // (>22 decimal exponent, or a significand past 2^53), so each reaches `c_strtod`.
    for src in [
        "let a = 1.5e100\n",
        "let a = 2.2204460492503131\n",
        "let a = 2.2250738585072014E-308\n",
        "let a = 1.17549435e-38'f32\n",
    ] {
        let (rc, produced, diag) = crawl(&nifler, "/p.nim", src);
        assert_eq!(rc, 0, "crawl failed on {src:?}: {diag}");
        assert!(produced > 0, "no .p.nif for {src:?}");
    }
}
