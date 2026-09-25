//! The committed guest libc's **capability edge** must be fully served by the nim link's stub unit.
//!
//! `nim_e2e.rs`'s `run_libc_program` asserts the real property — linking the guest libc must not widen
//! a nim program's capability manifest past the one `write` STREAM cap — but every test in that file
//! needs the real nimony toolchain and skips without one. So the coupling between
//! `browser/web/assets/pg_libc.temeno`'s import table and `LIBC_CAP_STUB_NAMES` was checked only in the
//! one CI job that has the toolchain, and a rename of the libc's stdin/stdout edge
//! (`write`/`read` -> `stream_write`/`stream_read`, when `<unistd.h>` gained real fd-dispatching
//! definitions and those names stopped being frontend builtins) passed every local check and eight
//! nim tests failed on CI.
//!
//! This needs no toolchain: it links the libc against its own stub unit and reads the merged import
//! table directly.

/// Every import the committed guest libc declares is resolved by the stub unit, aliased onto the
/// powerbox `write` cap, or one of the core's memory ops a nim program's heap grows with (`vm_map`,
/// `vm_page_size`), which every host binds to the program's window — so those are the only names
/// left unbound. A stub for a memory op would capture the program's own import of it at link.
#[test]
fn libc_cap_edge_is_fully_served() {
    let Ok(libc) = std::fs::read("../../browser/web/assets/pg_libc.temeno") else {
        eprintln!("SKIP: browser/web/assets/pg_libc.temeno absent");
        return;
    };
    // `&[]` for the nim units: with no program there are no libc *leaf* exports to map, which is fine
    // — this is about the cap edge, and the stub unit it returns is the one the real link uses.
    let units = temen_leng::nim_libc_units(&libc, &[]).expect("build libc + stub link units");
    let merged = temen_ir::link_with_manifest(&units).expect("link the libc against its cap stubs");
    let mut left: Vec<&str> = merged.imports.iter().map(|i| i.name.as_str()).collect();
    left.sort_unstable();
    assert_eq!(
        left,
        vec!["vm_map", "vm_page_size", "write"],
        "the guest libc has an import the stub unit does not serve — add it to \
         LIBC_CAP_STUB_NAMES (with a body of the right shape in LIBC_CAP_STUBS), or it becomes a \
         capability every nim program declares"
    );
}
