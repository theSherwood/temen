//! The playground's C heap, for tests that run C compiled with the seeded headers (#2172). The headers
//! only declare `malloc`/`free`/`calloc`/`realloc`: the heap unit (`web/assets/pg_heap.temeno`, dlmalloc
//! built by clang) defines them, and the card links it into every program. A test does the same: it
//! compiles its C as a program unit (`--emit-object`) and links it here. Even a program that never
//! allocates needs it, since `printf`'s stream writer can `realloc`.

/// The committed heap unit. A stale one (wire drift) fails to decode: `scripts/rebuild-assets.sh`.
fn heap() -> temen_ir::Module {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/web/assets/pg_heap.temeno");
    let bytes = std::fs::read(path).expect("read web/assets/pg_heap.temeno");
    temen_encode::decode_unit(&bytes)
        .expect("decode pg_heap.temeno — a stale asset? scripts/rebuild-assets.sh")
}

/// Link the program unit `prog` after `libs` and the heap unit, entering at `main`: the runnable
/// program. `libs` is empty for a unit with the headers' bodies compiled in, and the libc unit for one
/// compiled against their declarations (the card's order: libc, heap, program).
pub fn link(libs: &[&temen_ir::Module], prog: &temen_ir::Module) -> temen_ir::Module {
    let heap = heap();
    let mut mods = libs.to_vec();
    mods.push(&heap);
    let exports: Vec<Vec<(String, temen_ir::FuncIdx)>> = mods
        .iter()
        .map(|m| m.exports.iter().map(|e| (e.name.clone(), e.func)).collect())
        .collect();
    let units: Vec<temen_ir::LinkUnitRef<'_>> = mods
        .iter()
        .zip(&exports)
        .map(|(m, exports)| temen_ir::LinkUnitRef {
            module: m,
            exports,
            data_exports: &m.data_exports,
        })
        .collect();
    temen_browser::link_program_multi(&units, prog, "main").expect("link the program and the heap")
}
