//! Print one function of a `.temen` module: `dump_func <mod.temen> <func-idx> [block]`.
//! A trap backtrace names `func N block B inst I`; this is how you see what that instruction is.
fn main() {
    let mut a = std::env::args().skip(1);
    let path = a
        .next()
        .expect("usage: dump_func <mod.temen> <func> [block]");
    let fi: usize = a.next().expect("func idx").parse().expect("func idx");
    let only: Option<usize> = a.next().map(|s| s.parse().expect("block"));
    // `.ir` is temen-text; anything else is an encoded `.temen`. A trap backtrace names the same
    // indices either way, and a text module is what a chibicc/llvm build leaves on disk.
    let m = if path.ends_with(".ir") {
        temen_text::parse_module(&std::fs::read_to_string(&path).expect("read")).expect("parse")
    } else {
        temen_encode::decode_module(&std::fs::read(&path).expect("read")).expect("decode")
    };
    for (i, im) in m.imports.iter().enumerate() {
        eprintln!("import {i}: {} {:?}", im.name, im.mode);
    }
    let f = &m.funcs[fi];
    eprintln!(
        "func {fi}: {} params -> {} results, {} blocks",
        f.params.len(),
        f.results.len(),
        f.blocks.len()
    );
    for (bi, b) in f.blocks.iter().enumerate() {
        if only.is_some_and(|o| o != bi) {
            continue;
        }
        eprintln!("block {bi} ({} params):", b.params.len());
        for (ii, inst) in b.insts.iter().enumerate() {
            eprintln!("  inst {ii}: {inst:?}");
        }
        eprintln!("  term: {:?}", b.term);
    }
}
