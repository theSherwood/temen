//! #1109 runtime discoverability — `self.list` (CAP_SELF op 17) + `self.schema` (op 18), the
//! typed-shell story: enumerate what a domain holds, then read each interface's canonical op
//! names + signatures, all authority-neutral (D46: reflection ≠ amplification). Both ops use
//! `self.label`'s buffer contract: the full byte length is always returned; nothing is written
//! unless the whole answer fits (size with `cap = 0`, re-call to fill — no truncated rows).
//! Exercised on the shared dispatch (`Host::cap_dispatch_slots`) both fast backends funnel a
//! `call.cap` through, so the interpreter covers the JIT path too.

use temen_interp::{cap_id, GuestMem, Host};
use temen_ir::{FuncType, ValType};

/// A trivial flat window — the reflection ops only need `write_bytes` into a caller buffer.
struct VecMem(Vec<u8>);

impl GuestMem for VecMem {
    fn read_bytes(&self, ptr: u64, len: u64) -> Option<Vec<u8>> {
        let (p, l) = (ptr as usize, len as usize);
        self.0.get(p..p + l).map(<[u8]>::to_vec)
    }
    fn write_bytes(&mut self, ptr: u64, data: &[u8]) -> Option<()> {
        let p = ptr as usize;
        self.0.get_mut(p..p + data.len())?.copy_from_slice(data);
        Some(())
    }
}

fn sig(params: Vec<ValType>, results: Vec<ValType>) -> FuncType {
    FuncType { params, results }
}

fn names(ns: &[&str]) -> Vec<String> {
    ns.iter().map(|s| s.to_string()).collect()
}

/// One `self.list` row, decoded: `{ handle: i32 LE, type_id: u32 LE, name_len: u32 LE, name }`.
fn decode_list(buf: &[u8]) -> Vec<(i32, u32, String)> {
    let mut rows = Vec::new();
    let mut i = 0;
    while i < buf.len() {
        let h = i32::from_le_bytes(buf[i..i + 4].try_into().unwrap());
        let tid = u32::from_le_bytes(buf[i + 4..i + 8].try_into().unwrap());
        let nl = u32::from_le_bytes(buf[i + 8..i + 12].try_into().unwrap()) as usize;
        let name = String::from_utf8(buf[i + 12..i + 12 + nl].to_vec()).unwrap();
        rows.push((h, tid, name));
        i += 12 + nl;
    }
    rows
}

/// One `self.schema` row, decoded: `{ name_len, name, n_params, param bytes, n_results, result
/// bytes }` — type bytes are the wire's (0=i32 1=i64 2=f32 3=f64 4=v128 5=ref 6=cap).
fn decode_schema(buf: &[u8]) -> Vec<(String, Vec<u8>, Vec<u8>)> {
    let mut rows = Vec::new();
    let mut i = 0;
    while i < buf.len() {
        let nl = u32::from_le_bytes(buf[i..i + 4].try_into().unwrap()) as usize;
        let name = String::from_utf8(buf[i + 4..i + 4 + nl].to_vec()).unwrap();
        i += 4 + nl;
        let np = u32::from_le_bytes(buf[i..i + 4].try_into().unwrap()) as usize;
        let params = buf[i + 4..i + 4 + np].to_vec();
        i += 4 + np;
        let nr = u32::from_le_bytes(buf[i..i + 4].try_into().unwrap()) as usize;
        let results = buf[i + 4..i + 4 + nr].to_vec();
        i += 4 + nr;
        rows.push((name, params, results));
    }
    rows
}

#[test]
fn list_enumerates_named_grants_with_handle_and_type_id() {
    let mut h = Host::new();
    // A wired offer is a real grant with a real type_id; name it in the directory the way the
    // powerbox layer names every granted handle.
    let funcs: std::sync::Arc<[temen_ir::Func]> = vec![temen_ir::Func {
        params: vec![ValType::I64],
        results: vec![ValType::I64],
        blocks: vec![],
    }]
    .into();
    let cap = h
        .wire_offer_func(&funcs, &std::sync::Arc::from(Vec::new()), &[0])
        .expect("wire");
    h.register_cap_name("worker", cap);

    let mut mem = VecMem(vec![0u8; 4096]);
    // Size pass: cap 0 ⇒ length only, nothing written.
    let need = h
        .cap_dispatch_slots(temen_ir::CAP_SELF_TYPE_ID, 17, 0, &[64, 0], Some(&mut mem))
        .expect("list size")[0];
    assert!(need > 0, "one named grant => non-empty listing");
    assert!(
        mem.0.iter().all(|&b| b == 0),
        "cap 0 writes nothing (label contract)"
    );
    // Fill pass.
    let got = h
        .cap_dispatch_slots(
            temen_ir::CAP_SELF_TYPE_ID,
            17,
            0,
            &[64, need],
            Some(&mut mem),
        )
        .expect("list fill")[0];
    assert_eq!(got, need, "fill returns the same full length");
    let rows = decode_list(&mem.0[64..64 + need as usize]);
    let tid = h.type_id_of(cap).expect("wired cap has a type_id");
    assert_eq!(
        rows,
        vec![(cap, tid, "worker".to_string())],
        "row carries handle + type_id + name (discover→call needs no self.resolve)"
    );
}

#[test]
fn schema_returns_canonical_names_and_wire_type_bytes() {
    let mut h = Host::new();
    // A guest-declared named interface: names are half the intern key (#1109), so the schema
    // read-back is canonical, not first-interned-wins.
    let tid = h.intern_interface(
        &names(&["frob", "quux"]),
        &[
            sig(vec![ValType::I64, ValType::I32], vec![ValType::I64]),
            sig(vec![], vec![]),
        ],
    );
    let mut mem = VecMem(vec![0u8; 4096]);
    let need = h
        .cap_dispatch_slots(
            temen_ir::CAP_SELF_TYPE_ID,
            18,
            0,
            &[tid as i64, 0, 0],
            Some(&mut mem),
        )
        .expect("schema size")[0];
    assert!(need > 0);
    let got = h
        .cap_dispatch_slots(
            temen_ir::CAP_SELF_TYPE_ID,
            18,
            0,
            &[tid as i64, 0, need],
            Some(&mut mem),
        )
        .expect("schema fill")[0];
    assert_eq!(got, need);
    let rows = decode_schema(&mem.0[..need as usize]);
    assert_eq!(
        rows,
        vec![
            ("frob".to_string(), vec![1u8, 0u8], vec![1u8]), // (i64, i32) -> (i64)
            ("quux".to_string(), vec![], vec![]),            // () -> ()
        ],
        "op names + wire type bytes, in declared order"
    );

    // The pre-seeded builtin reads back its canonical named shape.
    let need = h
        .cap_dispatch_slots(
            temen_ir::CAP_SELF_TYPE_ID,
            18,
            0,
            &[cap_id::STREAM as i64, 0, 0],
            Some(&mut mem),
        )
        .expect("stream schema size")[0];
    h.cap_dispatch_slots(
        temen_ir::CAP_SELF_TYPE_ID,
        18,
        0,
        &[cap_id::STREAM as i64, 0, need],
        Some(&mut mem),
    )
    .expect("stream schema fill");
    let rows = decode_schema(&mem.0[..need as usize]);
    let op_names: Vec<&str> = rows.iter().map(|(n, _, _)| n.as_str()).collect();
    assert_eq!(
        op_names,
        vec!["read", "write", "close"],
        "builtin Stream's canonical op names"
    );

    // An id with no schema (a handle-typed builtin) is a probeable -EINVAL, never a trap.
    let r = h
        .cap_dispatch_slots(
            temen_ir::CAP_SELF_TYPE_ID,
            18,
            0,
            &[cap_id::EXIT as i64, 0, 0],
            Some(&mut mem),
        )
        .expect("no schema is probeable")[0];
    assert!(r < 0, "-EINVAL for a schema-less id, got {r}");
}
