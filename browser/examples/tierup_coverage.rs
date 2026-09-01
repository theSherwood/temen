//! Temporary probe (#889): per-card emitted-function counts through the pump's emit gate,
//! before vs after `outline_cap_calls` — the #887 tally re-run over the shipping assets.
fn main() {
    for path in std::env::args().skip(1) {
        let bytes = std::fs::read(&path).expect("read");
        let m = temen_encode::decode_module(&bytes).expect("decode");
        let n = m.funcs.len();
        let scalar = |t: &temen_ir::ValType| {
            matches!(
                t,
                temen_ir::ValType::I32
                    | temen_ir::ValType::I64
                    | temen_ir::ValType::F32
                    | temen_ir::ValType::F64
            )
        };
        let max_slots = temen_wasm_jit::XCALL_MAX_SLOTS; // cross-tier call scratch capacity (#1120 Slice 3)
        let tally = |m: &temen_ir::Module| -> (usize, usize) {
            let mut em = m.clone();
            if let Some(mc) = em.memory.as_mut() {
                mc.size_log2 = mc.size_log2.max(26);
            }
            let all_shimmable = em.funcs.iter().all(|f| {
                f.params.iter().all(scalar)
                    && f.results.iter().all(scalar)
                    && f.params.len().max(f.results.len()) <= max_slots
            });
            let r = if all_shimmable {
                temen_wasm_jit::compile_module_tierup_b2(&em, false, 10)
            } else {
                temen_wasm_jit::compile_module_tierup(&em, false)
            };
            match r {
                Ok((_, emit)) => (emit.iter().filter(|&&e| e).count(), emit.len()),
                Err(_) => (0, em.funcs.len()),
            }
        };
        let (base_emit, _) = tally(&m);
        let mut om = m.clone();
        temen_wasm_jit::outline_cap_calls(&mut om);
        let (out_emit, out_total) = tally(&om);
        let wrappers = out_total - n;
        let orig_emitted = out_emit; // wrappers never emit (they hold the cap op)
        println!(
            "{path}: funcs {n} | emitted before {base_emit} ({}%) | after {orig_emitted} ({}%) | wrappers {wrappers}",
            base_emit * 100 / n.max(1),
            orig_emitted * 100 / n.max(1),
        );
    }
}
