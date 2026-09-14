//! `temen_module_imports` — what a module **declares as host capabilities**, and which of them the
//! on-ramp powerbox can serve (c_interpret#26).
//!
//! An embedder needs both answers before it launches: what to grant, and whether the release runner
//! can run the program at all (a named host-completed cap, temen#1366, can only be parked for by the
//! debug session). Both live in `Module::imports`.
//!
//! **Why this query exists at all.** c_interpret answered both questions by scraping emitted IR text
//! for `call.sym "<name>"`. That worked only by accident: a *whole-program* compile spells the name in
//! the instruction, but `Inst::CallSym` carries a `u32` **index** into the import table — so once
//! separate compilation landed and programs were linked from units, the same program rendered as
//! `call.sym 6 v8(v7, v5)` and the scrape silently found nothing. No error, no diagnostic: the host
//! just stopped granting the caps and started routing interactive programs to the runner that cannot
//! serve them. `a_linked_module_reports_by_table_where_a_text_scrape_finds_nothing` is that exact
//! shape, pinned so it cannot come back.

use std::sync::Mutex;

use temen_browser::{
    temen_module_imports, temen_module_imports_len, temen_module_imports_ptr, temen_status,
    STATUS_DECODE_ERR, STATUS_OK, STATUS_UNSUPPORTED,
};

/// The report stash and `temen_status` are process-global (single-threaded wasm by design), so the
/// tests serialize on them — the same convention as `jspb.rs`.
static FFI_LOCK: Mutex<()> = Mutex::new(());

/// Ask the query about a module in whatever form the host holds it (text or encoded bytes) and read
/// the report back as lines.
fn report(bytes: &[u8]) -> (i32, Vec<String>) {
    let rc = temen_module_imports(bytes.as_ptr(), bytes.len());
    let (p, n) = (temen_module_imports_ptr(), temen_module_imports_len());
    if p.is_null() || n == 0 {
        return (rc, Vec::new());
    }
    // SAFETY: the cdylib-managed stash is live until the next query.
    let s = String::from_utf8_lossy(unsafe { core::slice::from_raw_parts(p, n) }).into_owned();
    (rc, s.lines().map(str::to_string).collect())
}

fn text(src: &str) -> (i32, Vec<String>) {
    report(src.as_bytes())
}

/// "Can the release runner run this as-is" — the whole point of the `served` column.
fn all_served(lines: &[String]) -> bool {
    lines.iter().all(|l| l.starts_with('1'))
}

/// A module declaring `imports` by name, each a flat one-op func import. No body needs to *call* them
/// — the declaration is what an embedder binds, and what this query reports.
fn module_declaring(imports: &[&str]) -> String {
    let mut s = String::from("memory 12\ntype 0 func (i64) -> (i64)\n");
    for (i, name) in imports.iter().enumerate() {
        s.push_str(&format!("import {i} func \"{name}\" 0\n"));
    }
    s.push_str("func () -> (i64) {\nblock 0 () {\n  v0 = i64.const 7\n  return v0\n  }\n}\n");
    s
}

#[test]
fn the_onramp_powerbox_names_are_served_and_an_embedder_name_is_not() {
    let _g = FFI_LOCK.lock().unwrap();
    // `read`/`write`/`exit`/`vm_map`/`vm_page_size` come from `onramp_cap_resolver`'s table and
    // `vm_fs` from the memfs seam `grant_onramp_caps` special-cases — the two halves of
    // `onramp_serves_import`. `fb_poll` is c_interpret's own host-completed cap: temen has never
    // heard of it, which is exactly why it reports `0` rather than failing.
    let (rc, lines) = text(&module_declaring(&[
        "vm_fs",
        "read",
        "write",
        "exit",
        "vm_map",
        "vm_page_size",
        "fb_poll",
    ]));
    assert_eq!(rc, 1, "the query succeeded");
    assert_eq!(temen_status(), STATUS_OK);
    assert_eq!(
        lines,
        vec![
            "1\tvm_fs",
            "1\tread",
            "1\twrite",
            "1\texit",
            "1\tvm_map",
            "1\tvm_page_size",
            "0\tfb_poll",
        ],
        "declaration order preserved, on-ramp names served, the embedder's name not"
    );
    assert!(
        !all_served(&lines),
        "one unserved import is enough to keep this off the release runner"
    );
}

#[test]
fn a_program_the_release_runner_can_serve_reports_every_import_served() {
    let _g = FFI_LOCK.lock().unwrap();
    let (rc, lines) = text(&module_declaring(&["read", "write", "exit"]));
    assert_eq!(rc, 1);
    assert!(
        all_served(&lines),
        "a plain stdio program needs nothing from the embedder: {lines:?}"
    );
}

#[test]
fn an_import_free_module_reports_an_empty_list_and_still_succeeds() {
    let _g = FFI_LOCK.lock().unwrap();
    // The empty list is a true answer, not a failure — and `all_served` of nothing is true, which is
    // the right call: a module that asks for no capabilities can always take the release path.
    let (rc, lines) = text(&module_declaring(&[]));
    assert_eq!(rc, 1, "no imports is not an error");
    assert_eq!(temen_status(), STATUS_OK);
    assert!(lines.is_empty());
    assert!(all_served(&lines));
}

#[test]
fn the_query_reads_encoded_bytes_and_text_alike() {
    let _g = FFI_LOCK.lock().unwrap();
    let src = module_declaring(&["write", "fb_present"]);
    let m = temen_text::parse_module(&src).expect("parses");
    let encoded = temen_encode::encode_module(&m);

    let (rc_t, from_text) = text(&src);
    let (rc_b, from_bytes) = report(&encoded);
    assert_eq!((rc_t, rc_b), (1, 1));
    assert_eq!(
        from_text, from_bytes,
        "a host can ask about a program it just linked without re-encoding it"
    );
}

#[test]
fn a_linked_module_reports_by_table_where_a_text_scrape_finds_nothing() {
    let _g = FFI_LOCK.lock().unwrap();
    // The regression this query exists to retire. `call.sym` names its import by **index**, so the
    // rendered text of a linked module contains no `call.sym "<name>"` anywhere — the substring a
    // host used to search for. The import table answers correctly on the same bytes.
    let src = r#"
memory 12
type 0 func (i64) -> (i64)
import 0 func "write" 0
import 1 func "fb_poll" 0
func () -> (i64) {
block 0 () {
  vh = i32.const 0
  va = i64.const 0
  v1 = call.sym 1 vh(va)
  return v1
  }
}
"#;
    let m = temen_text::parse_module(src).expect("parses");
    let rendered = temen_text::print_module(&m);
    assert!(
        !rendered.contains("call.sym \"fb_poll\""),
        "the name is not in the instruction stream — this is what the scrape was looking for"
    );
    assert!(
        rendered.contains("call.sym 1"),
        "the call names its import by index: {rendered}"
    );

    let (rc, lines) = text(src);
    assert_eq!(rc, 1);
    assert_eq!(lines, vec!["1\twrite", "0\tfb_poll"]);
    assert!(
        !all_served(&lines),
        "and the structural answer routes it away from the release runner"
    );
}

#[test]
fn a_name_the_table_resolves_but_the_powerbox_never_grants_is_not_served() {
    let _g = FFI_LOCK.lock().unwrap();
    // `stderr` is the case that proves the report asks the *binding*, not the name table.
    // `default_cap_resolver("stderr")` resolves it — it is `Stream` write, op 1, exactly like
    // `write` — but the on-ramp powerbox grants no `stderr` handle, so `PowerboxHandles::bind`
    // returns `None` and the slot stays unbound. Calling it served would hand the program to a
    // runner that fails closed on its first write to it.
    let (rc, lines) = text(&module_declaring(&["write", "stderr"]));
    assert_eq!(rc, 1);
    assert_eq!(
        lines,
        vec!["1\twrite", "0\tstderr"],
        "same capability and op as `write`, but ungranted — only the binding can tell them apart"
    );
    assert!(!all_served(&lines));
}

#[test]
fn the_jit_cap_is_served_only_because_declaring_it_is_what_grants_it() {
    let _g = FFI_LOCK.lock().unwrap();
    // Least authority: the on-ramp grants `Jit` iff the guest declares a `vm_jit_*` import, so the
    // handle set the report is answered against depends on the module. A module that declares one
    // therefore gets it — and this pins that `onramp_granted_shape` keeps mirroring that decision.
    let (rc, lines) = text(&module_declaring(&["vm_jit_compile", "vm_jit_invoke2"]));
    assert_eq!(rc, 1);
    assert!(
        all_served(&lines),
        "a guest that asks for the JIT is the guest that is granted it: {lines:?}"
    );
}

#[test]
fn a_name_carrying_a_separator_is_refused_rather_than_allowed_to_forge_a_line() {
    let _g = FFI_LOCK.lock().unwrap();
    // The report gates *what the host grants*, so a module must not be able to smuggle an extra line
    // into it and talk an embedder into granting a capability it never declared. No real capability
    // name contains a tab or a newline, so refusing costs nothing and fails closed.
    for forged in ["evil\n1\tinstantiator", "evil\tname"] {
        let src = module_declaring(&[forged]);
        let (rc, lines) = text(&src);
        assert_eq!(rc, 0, "refused: {forged:?}");
        assert_eq!(temen_status(), STATUS_UNSUPPORTED);
        assert!(lines.is_empty(), "and nothing is stashed to be misread");
    }
}

#[test]
fn bytes_that_are_not_a_module_fail_closed() {
    let _g = FFI_LOCK.lock().unwrap();
    let (rc, lines) = report(b"this is not a module");
    assert_eq!(rc, 0);
    assert_eq!(temen_status(), STATUS_DECODE_ERR);
    assert!(lines.is_empty());
}
