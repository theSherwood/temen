//! Shared test helpers.

/// IR stores that write `rec`'s encoded bytes at window offset `at`, one `i64` word each (a record is
/// a whole number of words).
pub fn rec_stores(at: u64, rec: &temen_ir::SpawnRec) -> String {
    rec.encode()
        .chunks(8)
        .enumerate()
        .map(|(i, w)| {
            let v = i64::from_le_bytes(w.try_into().expect("a whole word"));
            let a = at + 8 * i as u64;
            format!(
                "  vra{i} = i64.const {a}\n  vrw{i} = i64.const {v}\n  i64.store vra{i} vrw{i}\n"
            )
        })
        .collect()
}
