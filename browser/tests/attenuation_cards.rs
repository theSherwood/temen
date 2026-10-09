//! #1509 — the playground's two **per-child attenuation** cards, run natively through the same
//! on-ramp powerbox the page uses (`onramp_exec`: stdout/stdin/exit/memory/addrspace + a named
//! `instantiator`, bytecode engine, in-process confined children). Both cards spawn one child function
//! twice with different grant lists — A with `{"stdout"}`, B with none — and return `A*10 + B`; only A
//! can resolve the re-granted stream and print. The sources are read out of `play.js` itself, so the
//! gate runs exactly what the card runs (one source, no copy):
//!
//! - the **Temen** card: parsed from its text and run as-is;
//! - the **C** card: compiled by the committed `chibicc.temen` (the in-browser compiler — this is also
//!   the code-coupled gate for its `__vm_instantiate_rec`/`__vm_instantiate_join` builtins) against
//!   the seeded playground headers, `<temen/spawn.h>` included, linked against the heap unit, then
//!   run. Fail-soft: SKIPs if the asset isn't built.
use temen_browser::{
    onramp_exec, onramp_fs_exec, playground_include_files, STATUS_EXIT, STATUS_OK,
};

#[path = "support/pg_heap.rs"]
mod pg_heap;

const PLAY_JS: &str = include_str!("../web/play.js");

/// The `src` template literal of the `EXAMPLES` card keyed `key`, with the JS escapes undone.
fn card_src(key: &str) -> String {
    let i = PLAY_JS
        .find(key)
        .unwrap_or_else(|| panic!("card {key} not in play.js"));
    let j = PLAY_JS[i..].find("src: `").expect("card src") + i + 6;
    let k = PLAY_JS[j..].find("`,\n  },").expect("card src end") + j;
    PLAY_JS[j..k].replace("\\\\", "\\")
}

const EXPECT_VALUE: i64 = 10;

#[test]
fn temen_card_grants_stdout_to_one_child_only() {
    let src = card_src("'§14 attenuation: two children, two powerboxes (Temen)'");
    let m = temen_text::parse_module(&src).unwrap_or_else(|e| panic!("parse: {e:?}"));
    let run = onramp_exec(&m, b"");
    assert_eq!(
        run.status,
        STATUS_OK,
        "stderr: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert_eq!(
        run.value, EXPECT_VALUE,
        "A (granted) = 1, B (not granted) = 0"
    );
    assert_eq!(run.stdout, b"granted\n", "only the granted child printed");
}

#[test]
fn c_card_grants_stdout_to_one_child_only() {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/web/assets/chibicc.temen");
    let Ok(bytes) = std::fs::read(p) else {
        eprintln!("SKIP: chibicc.temen not built");
        return;
    };
    let chibicc = temen_encode::decode_module(&bytes).expect("decode chibicc.temen");
    let src = card_src("'§14 attenuation from C (chibicc + <temen/spawn.h>)'");
    let mut files: Vec<(String, Vec<u8>)> = playground_include_files();
    files.push(("in.c".to_string(), src.into_bytes()));
    let dirs = vec!["include".to_string(), "include/temen".to_string()];
    let image = temen_fs::encode_image(&files, &dirs);
    let compiled = onramp_fs_exec(
        &chibicc,
        &image,
        &[
            b"chibicc",
            b"--data-page",
            b"65536",
            b"--emit-object",
            b"/in.c",
        ],
        b"",
    );
    assert!(
        compiled.status == STATUS_OK || compiled.status == STATUS_EXIT,
        "compile status {} — stderr: {}",
        compiled.status,
        String::from_utf8_lossy(&compiled.stderr)
    );
    let ir = String::from_utf8(compiled.stdout).expect("IR is utf8");
    // The helper's spawn/join are static `call.cap`s on the Instantiator (interface 6).
    assert!(
        ir.contains("call.cap 6 17") && ir.contains("call.cap 6 1 "),
        "{ir:.300}"
    );
    let m = temen_text::parse_module(&ir).unwrap_or_else(|e| panic!("parse IR: {e:?}"));
    let run = onramp_exec(&pg_heap::link(&[], &m), b"");
    assert_eq!(
        run.status,
        STATUS_OK,
        "stderr: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert_eq!(run.value, EXPECT_VALUE);
    assert_eq!(
        String::from_utf8_lossy(&run.stdout),
        "granted\nchild A (granted stdout) returned 1\nchild B (no grants)      returned 0\n"
    );
}

/// The `detached` card (§5 op 15 over a pre-mapped SharedRegion) through the same `onramp_exec` the
/// page's on-ramp recipe calls: the parent finds `child` (its child image, #2219) and `budget` by name
/// and spawns the image detached — 1000 × 42 + 82. `budget` cannot cross into a §14 child yet, so a
/// card that spawns detached runs at the root (#1720); this pins that it still runs at all.
#[test]
fn detached_card_spawns_over_a_premapped_region() {
    let src = card_src("\n  detached: {");
    let m = temen_text::parse_module(&src).unwrap_or_else(|e| panic!("parse: {e:?}"));
    let run = onramp_exec(&m, b"");
    assert_eq!(run.status, STATUS_OK, "trap: {:?}", run.trap);
    assert_eq!(run.value, 42082);
}

/// The `inst` card under the grants of the parallel §14 recipe it runs on (`temen_par_root`): an
/// `Instantiator` over its window, the `budget` its children's windows spend, and its child image as
/// `"child"` (#2219). The page runs each child on a Worker of its own; here they share the bytecode
/// engine's cooperative driver, which admits them the same way. Eight copies return 5 each.
#[test]
fn inst_card_sums_eight_copies_of_itself() {
    let src = card_src("\n  inst: {");
    let m = std::sync::Arc::new(
        temen_text::parse_module(&src).unwrap_or_else(|e| panic!("parse: {e:?}")),
    );
    let mut host = temen_interp::Host::new();
    host.set_self_module(&m);
    let inst = host.grant_instantiator(0, 1 << 16);
    let budget = host.grant_budget(-1, 1 << 20, -1); // the page's `minter`
    host.grant_child_image();
    let mut fuel = 50_000_000u64;
    let args = [inst, budget].map(temen_interp::Value::I32);
    let r = temen_interp::bytecode::compile_and_run_with_host(&m, 0, &args, &mut fuel, &mut host)
        .expect("the bytecode engine runs the card");
    assert_eq!(r, Ok(vec![temen_interp::Value::I64(40)]));
}
