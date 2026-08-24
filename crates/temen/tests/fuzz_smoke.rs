//! Stable-toolchain smoke fuzzing — the day-one security invariants from AGENTS.md
//! and `DESIGN.md` §2a, runnable without nightly/cargo-fuzz. A real libFuzzer target
//! lives in `fuzz/` (nightly); this catches the same panics in CI on stable.
//!
//! Invariants asserted (the "fail-closed, never crash" property of the escape-TCB):
//!   1. `decode_module` never panics / OOMs / hangs on arbitrary bytes.
//!   2. `verify_module` never panics on any decoded module.
//!   3. A *verified* module never panics the interpreter (bounded by fuel).
//!   4. Binary round-trip is identity on every decodable module; text round-trip is
//!      identity on every verified module (mirrors the `roundtrip` fuzz target).

use temen::default_args;
use temen_encode::{decode_module, decode_unit, encode_module, encode_unit, DecodeError};
use temen_interp::run;
use temen_verify::verify_module;

/// Tiny deterministic PRNG (xorshift64*) — no external deps.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn byte(&mut self) -> u8 {
        (self.next_u64() & 0xff) as u8
    }
    fn range(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() % n as u64) as usize
        }
    }
}

fn drive(bytes: &[u8]) {
    // Object dialect (v9): `decode_unit` shares the decoder and must fail-closed identically;
    // object round-trip is identity; and the header firewall holds — anything the unit path
    // decodes that the runnable path rejects is rejected *as an object* (flag bit 0), never
    // for a later reason (mirrors the `roundtrip`/`decode_verify` fuzz targets).
    if let Ok(u) = decode_unit(bytes) {
        assert_eq!(
            decode_unit(&encode_unit(&u)),
            Ok(u.clone()),
            "object round-trip"
        );
        match decode_module(bytes) {
            Ok(_) | Err(DecodeError::ObjectInput) => {}
            Err(e) => panic!("dialect divergence beyond the object flag: {e:?}"),
        }
    }
    // 1. Decode must fail-closed, never panic.
    if let Ok(m) = decode_module(bytes) {
        // 4a. Binary round-trip is identity on every decodable module.
        assert_eq!(
            decode_module(&encode_module(&m)),
            Ok(m.clone()),
            "binary round-trip"
        );
        // 2. Verify must never panic.
        if verify_module(&m).is_ok() {
            // 4b. Text round-trip is identity on every verified module.
            assert_eq!(
                temen_text::parse_module(&temen_text::print_module(&m)),
                Ok(m.clone()),
                "text round-trip"
            );
            // 3. A verified module is safe to interpret (bounded by fuel).
            for (fi, f) in m.funcs.iter().enumerate() {
                let args = default_args(&f.params);
                let mut fuel = 10_000u64;
                let _ = run(&m, fi as u32, &args, &mut fuel);
            }
        }
    }
}

#[test]
fn decode_verify_interp_never_panic_on_random_bytes() {
    let mut rng = Rng(0x9E3779B97F4A7C15);
    for _ in 0..200_000 {
        let len = rng.range(64);
        let mut buf = Vec::with_capacity(len);
        for _ in 0..len {
            buf.push(rng.byte());
        }
        drive(&buf);
    }
}

#[test]
fn mutated_valid_modules_never_panic() {
    // Start from valid encodings and flip bytes — exercises the decoder's recovery
    // paths closer to the valid manifold than pure random noise reaches.
    let seeds = [
        r#"func (i32,i32)->(i32){
block 0 (v0:i32,v1:i32) { v2 = i32.add v0 v1
 return v2 } }"#,
        r#"func ()->(i64){
block 0 () { v0 = i64.const 7
 return v0 } }"#,
    ];
    let mut rng = Rng(0xD1B54A32D192ED03);
    for seed in seeds {
        let m = temen_text::parse_module(seed).expect("seed parses");
        let base = encode_module(&m);
        for _ in 0..50_000 {
            let mut buf = base.clone();
            // Apply 1..=4 random single-byte mutations.
            let muts = 1 + rng.range(4);
            for _ in 0..muts {
                if !buf.is_empty() {
                    let i = rng.range(buf.len());
                    buf[i] = rng.byte();
                }
            }
            drive(&buf);
        }
    }

    // An **object** seed (v9 dialect): mutations reach the object-only sections
    // (`data.ptr`, data exports) and the link-form opcodes, which pure random noise
    // rarely constructs past the header.
    use temen::ir;
    let mut u = temen_text::parse_module(seeds[1]).expect("seed parses");
    u.memory = Some(ir::Memory { size_log2: 16 });
    u.data = vec![ir::Data {
        offset: 0,
        readonly: false,
        bytes: vec![0; 16],
    }];
    u.data_ptrs = vec![
        ir::DataPtr {
            at: 0,
            target: ir::DataPtrTarget::SelfOff(8),
        },
        ir::DataPtr {
            at: 8,
            target: ir::DataPtrTarget::Sym {
                name: "g".into(),
                addend: -2,
            },
        },
    ];
    u.data_exports = vec![ir::DataExport {
        name: "h".into(),
        offset: 4,
    }];
    u.funcs.push(ir::Func {
        params: vec![],
        results: vec![ir::ValType::I64],
        blocks: vec![ir::Block {
            params: vec![],
            insts: vec![
                ir::Inst::DataSelf { offset: 8 },
                ir::Inst::DataSym {
                    name: b"g".to_vec(),
                    addend: 3,
                },
                ir::Inst::DataTop,
            ],
            term: ir::Terminator::Return(vec![2]),
        }],
    });
    let base = encode_unit(&u);
    assert_eq!(decode_unit(&base), Ok(u.clone()), "object seed round-trips");
    let mut rng = Rng(0xA0761D6478BD642F);
    for _ in 0..50_000 {
        let mut buf = base.clone();
        let muts = 1 + rng.range(4);
        for _ in 0..muts {
            if !buf.is_empty() {
                let i = rng.range(buf.len());
                buf[i] = rng.byte();
            }
        }
        drive(&buf);
    }
}
