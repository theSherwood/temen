//! The op-13 JIT loop's test driver (`temen_op13jit_open_detached`, entry `(inst, module, budget)`):
//! it re-grants each of `grants` — resolved by name on its own powerbox — to the child under the same
//! name, spawns the child detached through an op-17 v1 record (its declared window, paid from the
//! budget), joins it into `vr`, then runs `tail`, which must leave the result in `vo` (empty: return
//! `vr`). Names sit at 18432 + 16·i and grant records at 17536 + 16·i, above the NULL guard.

pub fn driver(grants: &[&str], tail: &str) -> Vec<u8> {
    let mut data = String::new();
    let mut body = String::new();
    for (i, name) in grants.iter().enumerate() {
        let (np, rp) = (18432 + 16 * i as u64, 17536 + 16 * i as u64);
        data.push_str(&format!("data {np} \"{name}\"\n"));
        body.push_str(&format!(
            "  np{i} = i64.const {np}\n  nl{i} = i64.const {len}\n  h{i} = self.resolve np{i} nl{i}\n  \
             rp{i} = i64.const {rp}\n  rw{i} = i64.const {w}\n  i64.store rp{i} rw{i}\n  \
             rh{i} = i64.const {rh}\n  i32.store rh{i} h{i}\n",
            len = name.len(),
            w = np | ((name.len() as u64) << 32),
            rh = rp + 8,
        ));
    }
    let rec = temen_ir::SpawnRec {
        grants_ptr: 17536,
        grants_n: grants.len() as u64,
        ..temen_ir::SpawnRec::v1(0)
    };
    let (seg, stores) = temen_browser::plan::spawn_rec_ir(17408, &rec, "v1", "v2");
    let (tail, ret) = if tail.is_empty() {
        ("", "vr")
    } else {
        (tail, "vo")
    };
    let src = format!(
        "memory 16\n{data}{seg}func (i32, i32, i32) -> (i64) {{\nblock 0 (v0: i32, v1: i32, v2: i32) {{\n\
         {body}{stores}  vrp = i64.const 17408\n  vh = call.cap 6 17 (i64) -> (i32) v0 (vrp)\n  \
         vr = call.cap 6 1 (i32) -> (i64) v0 (vh)\n{tail}\n  return {ret}\n  }}\n}}\n"
    );
    let m = temen_text::parse_module(&src).expect("parse driver");
    temen_verify::verify_module(&m).expect("verify driver");
    temen_encode::encode_module(&m)
}
