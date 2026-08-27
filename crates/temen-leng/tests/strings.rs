//! String-literal tests (NIM.md Phase 2 / W2). nimony's `string` is a small-string-optimized object
//! `{bytes: u64@0, more: ptr@8}` (confirmed from the system module's `basic_types.nim`): a **short**
//! literal packs its chars into the inline `bytes` word with a nil `more`, so it's an ordinary
//! `(oconstr string (kv bytes.0 <packed-u64>) (kv more.0 (nil)))` — no data segment at all.
//!
//! Two pieces make that lower: unsigned literals (`122511465736197u`) parse, and the external
//! `string` type's layout is available. `link_units` resolves the layout automatically from the
//! defining unit (see `link.rs::real_string_type_resolves_across_link`); this file exercises the
//! *manual* escape hatch, `translate_proc_with_types`, which supplies it as a prelude — here, the
//! *real* `string` def.

use temen_interp::Value;
use temen_leng::LengModule;

fn run(module: &temen_ir::Module, idx: u32, args: &[i64]) -> i64 {
    temen_verify::verify_module(module).unwrap_or_else(|e| panic!("verify: {e:?}"));
    let ivals: Vec<Value> = args.iter().map(|&n| Value::I64(n)).collect();
    let mut fuel = u64::MAX;
    let interp = temen_interp::run(module, idx, &ivals, &mut fuel).expect("interp run");
    let n = match interp.as_slice() {
        [Value::I64(n)] => *n,
        o => panic!("expected i64, got {o:?}"),
    };
    let jit = match temen_jit::compile_and_run(module, idx, args).expect("jit") {
        temen_jit::JitOutcome::Returned(v) => v,
        o => panic!("jit: {o:?}"),
    };
    assert_eq!(jit.as_slice(), &[n], "§9 interp/JIT parity");
    n
}

#[test]
fn sso_string_construct_and_read() {
    // Build an SSO string exactly as nimony does — pack "hello" (0x6f6c6c6568 = 478560413032) into
    // `bytes`, nil `more` — then read the `bytes` word back. Runs (no runtime: it's a plain object).
    let leng = "\
(stmts
 (type :Str.0. . (object . (fld :bytes.0 . (u 64)) (fld :more.0 . (ptr (i +64)))))
 (proc :mk.0 . (i +64) .
  (stmts .
   (var :s.0 . Str.0. (oconstr Str.0. (kv bytes.0 478560413032u) (kv more.0 (nil))))
   (ret (dot s.0 bytes.0 0)))))";
    let m = temen_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    let sp = 20480; // 4096 + 16384, clearing the unconditional NULL guard (#1094)
    assert_eq!(
        run(&m, 0, &[sp]),
        478560413032,
        "packed \"hello\" bytes read back"
    );
}

#[test]
fn long_string_const_materializes() {
    // A **long** string literal can't pack into the SSO `bytes` word: nimony emits it as a `const`
    // `LongString` blob `{fullLen, rc, capImpl, data:"…"}` that the `string` value's `more` field
    // points at (`(addr strlit…)`). This exercises the blob's materialization: the const's exact
    // bytes land in a data segment, and `(addr blob)` resolves to them. `getptr` returns the blob's
    // address; the window there must hold `[fullLen=5 | rc=0 | capImpl=0 | "hello"]`.
    let leng = "\
(stmts
 (type :LS.0. . (object . (fld :fullLen.0 . (i +64)) (fld :rc.0 . (i +64)) (fld :capImpl.0 . (i +64)) (fld :data.0 . (uarray (c 8)))))
 (const :blob.0. . LS.0. (oconstr LS.0. (kv fullLen.0 5) (kv rc.0 0) (kv capImpl.0 0) (kv data.0 \"hello\")))
 (proc :getptr.0. . (i +64) . (stmts . (ret (addr blob.0.)))))";
    let m = temen_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    temen_verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));

    let mut want = vec![0u8; 24];
    want[0] = 5; // fullLen@0
    want.extend_from_slice(b"hello"); // data@24

    // The const blob lands in the globals region, now based above the unconditional NULL guard
    // (#1094, `translate::globals_base`); the snapshot window must reach it.
    let seed = vec![0u8; 32768];
    let mut fuel = u64::MAX;
    let (ir, imem) = temen_interp::run_capture(&m, 0, &[], &mut fuel, &seed);
    let addr = match ir.expect("interp").as_slice() {
        [Value::I64(a)] => *a as usize,
        o => panic!("expected an address, got {o:?}"),
    };
    assert_eq!(
        &imem[addr..addr + want.len()],
        &want[..],
        "LongString blob (interp)"
    );

    let (jout, jmem) = temen_jit::compile_and_run_capture(&m, 0, &[], &seed).expect("jit");
    let jaddr = match jout {
        temen_jit::JitOutcome::Returned(v) => v[0] as usize,
        o => panic!("jit: {o:?}"),
    };
    assert_eq!(jaddr, addr, "§9 interp/JIT parity on the blob address");
    assert_eq!(
        &jmem[jaddr..jaddr + want.len()],
        &want[..],
        "LongString blob (jit)"
    );
}

/// Real nimony `greetLong(): string = "hello, this is a long string!"` (29 chars — too long to
/// pack into the SSO word). hexer emits it as a `const` `LongString` blob plus a `string` whose
/// `more` points at it (`(addr strlit…)`). Linked against a stand-in `system` unit carrying the
/// real `string`/`LongString` type defs (so the const's layout resolves cross-module) and no-op ARC
/// stubs (`=wasMoved`/`=destroy`), it runs end-to-end: the returned `string`'s `more` points at the
/// materialized blob, whose `fullLen` is 29 and whose data is the literal — identically on both
/// engines.
#[test]
fn real_long_string_runs_end_to_end() {
    const REAL: &str = include_str!("fixtures/real_longstring.leng.nif");
    let system = "\
(stmts
 (type :string.0. . (object . (fld :bytes.0 . (u 64)) (fld :more.0 . (ptr (i +64)))))
 (type :LongString.0. . (object . (fld :fullLen.0 . (i +64)) (fld :rc.0 . (i +64)) (fld :capImpl.0 . (i +64)) (fld :data.0 . (uarray (c 8)))))
 (proc :=wasMoved.2. (params (param :p.0 . (ptr (i +64)))) . . (stmts . (ret .)))
 (proc :=destroy.2. (params (param :s.0 . (ptr (i +64)))) . . (stmts . (ret .))))";
    let linked = temen_leng::link_units(&[
        LengModule {
            stem: "lonelga0b",
            src: REAL,
            names: &["greetLong.0."],
        },
        LengModule {
            stem: "sysvq0asl",
            src: system,
            names: &["=wasMoved.2.", "=destroy.2."],
        },
    ])
    .unwrap_or_else(|e| panic!("link: {e}"));
    temen_verify::verify_module(&linked).unwrap_or_else(|e| panic!("verify: {e:?}"));

    // greetLong (func 0) is frame-needing + aggregate-returning: ($sp, $sret) -> (). `sp`/`sret`
    // sit in the low scratch page below the globals; the window must reach past the globals base
    // (`POWERBOX_NULL_GUARD + POWERBOX_STACK_PAGE`, 32768, #1091) so the snapshot covers the
    // relocated const blob the test reads.
    let (sp, sret) = (18432i64, 16896usize); // both +16384 above the unconditional NULL guard (#1094)
    let seed = vec![0u8; 49152];
    let ivals = vec![Value::I64(sp), Value::I64(sret as i64)];
    let mut fuel = u64::MAX;
    let (ir, imem) = temen_interp::run_capture(&linked, 0, &ivals, &mut fuel, &seed);
    ir.expect("interp greetLong");
    let (_j, jmem) =
        temen_jit::compile_and_run_capture(&linked, 0, &[sp, sret as i64], &seed).expect("jit");
    let n = imem.len().min(jmem.len());
    assert_eq!(imem[..n], jmem[..n], "§9 interp/JIT window parity");

    let bytes = u64::from_le_bytes(imem[sret..sret + 8].try_into().unwrap());
    let more = u64::from_le_bytes(imem[sret + 8..sret + 16].try_into().unwrap()) as usize;
    assert_eq!(
        bytes, 2318350419654699262,
        "the SSO `bytes` word nimony emitted"
    );
    assert_ne!(more, 0, "long string: `more` points at the LongString blob");
    let full_len = u64::from_le_bytes(imem[more..more + 8].try_into().unwrap());
    assert_eq!(full_len, 29, "blob fullLen@0");
    assert_eq!(
        &imem[more + 24..more + 24 + 29],
        b"hello, this is a long string!",
        "blob data@24"
    );
}

#[test]
fn computed_deref_flexarray_chain() {
    // The real string-internal walk: `s.more.data[i]` where `s: ptr string`, so the flex array is
    // reached through a **computed** deref — `(pat (dot (deref (dot (deref s) more)) data) i)`.
    // This needs `lvalue_type` to see through the computed `(deref (dot …))` to the `LongString`,
    // then to its `data` flex array. nth(s, i) reads `s.more.data[i]`.
    let leng = "\
(stmts
 (type :string.0. . (object . (fld :bytes.0 . (u 64)) (fld :more.0 . (ptr LS.0.))))
 (type :LS.0. . (object . (fld :fullLen.0 . (i +64)) (fld :rc.0 . (i +64)) (fld :capImpl.0 . (i +64)) (fld :data.0 . (uarray (c 8)))))
 (proc :nth.0. (params (param :s.0 . (ptr string.0.)) (param :i.0 . (i +64))) (i +64) .
  (stmts . (ret (conv (i +64) (pat (dot (deref (dot (deref s.0) more.0 0)) data.0 0) i.0))))))";
    let m = temen_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    temen_verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));

    // Pointer targets raised +16384 above the unconditional NULL guard (#1094).
    let (s, b) = (16640usize, 16896usize);
    let mut seed = vec![0u8; 32768];
    seed[s + 8..s + 16].copy_from_slice(&(b as u64).to_le_bytes()); // string.more → blob
    seed[b + 24..b + 24 + 3].copy_from_slice(b"hey"); // blob.data@24
    let read = |i: i64| -> i64 {
        let mut fuel = u64::MAX;
        let (ir, _) = temen_interp::run_capture(
            &m,
            0,
            &[Value::I64(s as i64), Value::I64(i)],
            &mut fuel,
            &seed,
        );
        match ir.expect("interp").as_slice() {
            [Value::I64(n)] => *n,
            o => panic!("{o:?}"),
        }
    };
    assert_eq!(read(0), b'h' as i64, "s.more.data[0]");
    assert_eq!(read(2), b'y' as i64, "s.more.data[2]");
}

#[test]
fn const_to_const_pointer_relocates() {
    // A `const` `string` whose `more` field points at *another* `const` (a `LongString` blob) — like
    // a local string literal `prefix = "…"` in the system module. In link mode this is a `data.ptr`
    // relocation: `s.more` holds placeholder bytes the linker overwrites with the blob's placed
    // address. `getmore()` returns `s.more`; following it must land on the blob (fullLen 3, "abc").
    let leng = "\
(stmts
 (type :LS.0. . (object . (fld :fullLen.0 . (i +64)) (fld :rc.0 . (i +64)) (fld :capImpl.0 . (i +64)) (fld :data.0 . (uarray (c 8)))))
 (type :string.0. . (object . (fld :bytes.0 . (u 64)) (fld :more.0 . (ptr LS.0.))))
 (const :strlit.0. . LS.0. (oconstr LS.0. (kv fullLen.0 3) (kv rc.0 0) (kv capImpl.0 0) (kv data.0 \"abc\")))
 (const :s.0. . string.0. (oconstr string.0. (kv bytes.0 42) (kv more.0 (addr strlit.0.))))
 (proc :getmore.0. . (i +64) . (stmts . (ret (dot s.0. more.0 0)))))";
    // Link the single unit (compile → decode → link) so the `data.ptr` self-relocation resolves.
    let linked = temen_leng::link_units(&[LengModule {
        stem: "m",
        src: leng,
        names: &["getmore.0."],
    }])
    .unwrap_or_else(|e| panic!("link: {e}"));
    temen_verify::verify_module(&linked).unwrap_or_else(|e| panic!("verify: {e:?}"));

    // The window must reach past the globals base (`POWERBOX_NULL_GUARD + POWERBOX_STACK_PAGE`,
    // 32768, #1091) so the snapshot covers the relocated const blob the test reads back through `more`.
    let seed = vec![0u8; 49152];
    let mut fuel = u64::MAX;
    let (ir, imem) = temen_interp::run_capture(&linked, 0, &[], &mut fuel, &seed);
    let more = match ir.expect("interp").as_slice() {
        [Value::I64(a)] => *a as usize,
        o => panic!("{o:?}"),
    };
    assert_ne!(more, 0, "`more` relocated to the blob");
    let full_len = u64::from_le_bytes(imem[more..more + 8].try_into().unwrap());
    assert_eq!(full_len, 3, "blob fullLen — followed the relocated pointer");
    assert_eq!(&imem[more + 24..more + 27], b"abc", "blob data");
}

#[test]
fn const_array_table_indexes() {
    // A `const` array table `(aconstr T e0 e1 …)` — like `system`'s `fsLookupTable` (256 i8s) —
    // materializes into a data segment; `(at table idx)` reads an element. Here a small i8 table
    // [-1, 5, 100, -3]; `nth(i)` returns the signed element.
    let leng = "\
(stmts
 (type :Tbl.0. . (array (i 8) 4))
 (const :tbl.0. . Tbl.0. (aconstr Tbl.0. (suf -1 \"i8\") (suf 5 \"i8\") (suf 100 \"i8\") (suf -3 \"i8\")))
 (proc :nth.0. (params (param :i.0 . (i +64))) (i +64) .
  (stmts . (ret (conv (i +64) (at tbl.0. i.0))))))";
    let m = temen_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    temen_verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    assert_eq!(run(&m, 0, &[0]), -1, "tbl[0]");
    assert_eq!(run(&m, 0, &[1]), 5, "tbl[1]");
    assert_eq!(run(&m, 0, &[2]), 100, "tbl[2]");
    assert_eq!(run(&m, 0, &[3]), -3, "tbl[3] sign-extended");
}

#[test]
fn flexarray_data_indexing() {
    // `LongString.data` is an inline `uarray char` — a flexible array whose own address is the base.
    // `nth(p, i) = (deref p).data[i]`, read as a byte. Indexes off `p + 24` (data@24), no pointer
    // load, widened per char. Reads "hello" seeded at the blob.
    let leng = "\
(stmts
 (type :LS.0. . (object . (fld :fullLen.0 . (i +64)) (fld :rc.0 . (i +64)) (fld :capImpl.0 . (i +64)) (fld :data.0 . (uarray (c 8)))))
 (proc :nth.0. (params (param :p.0 . (ptr LS.0.)) (param :i.0 . (i +64))) (i +64) .
  (stmts . (ret (conv (i +64) (at (dot (deref p.0) data.0 0) i.0))))))";
    let m = temen_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    temen_verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));

    let b = 16640usize; // 256 + 16384, above the unconditional NULL guard (#1094)
    let mut seed = vec![0u8; 32768];
    seed[b + 24..b + 24 + 5].copy_from_slice(b"hello"); // data@24
    let read = |i: i64| -> i64 {
        let mut fuel = u64::MAX;
        let (ir, _) = temen_interp::run_capture(
            &m,
            0,
            &[Value::I64(b as i64), Value::I64(i)],
            &mut fuel,
            &seed,
        );
        let n = match ir.expect("interp").as_slice() {
            [Value::I64(n)] => *n,
            o => panic!("{o:?}"),
        };
        let jit = match temen_jit::compile_and_run_capture(&m, 0, &[b as i64, i], &seed)
            .expect("jit")
            .0
        {
            temen_jit::JitOutcome::Returned(v) => v,
            o => panic!("{o:?}"),
        };
        assert_eq!(jit, vec![n], "§9 parity");
        n
    };
    assert_eq!(read(0), b'h' as i64, "data[0]");
    assert_eq!(read(1), b'e' as i64, "data[1]");
    assert_eq!(read(4), b'o' as i64, "data[4]");
}

#[test]
fn unsigned_literal_over_i64_max() {
    // An SSO word can exceed i64::MAX; the `u` literal keeps the bit pattern.
    let leng = "\
(stmts
 (proc :big.0 . (i +64) . (stmts . (ret 18000000000000000000u))))";
    let m = temen_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    assert_eq!(run(&m, 0, &[]), 18000000000000000000u64 as i64);
}

/// Real nimony `hexer` output for `greet(): string = "hello"` — the genuine SSO oconstr
/// `(oconstr string (kv bytes.0 122511465736197u) (kv more.0 (nil)))` returned by sret. Supplying
/// the real `string` layout (its `system`-module def, prepended), it **translates and verifies**;
/// running it end-to-end needs the ARC `=wasMoved`/`=destroy` imports bound (the W3 runtime edge).
#[test]
fn real_nimony_short_string() {
    const REAL: &str = include_str!("fixtures/real_string.leng.nif");
    // The `string` type as the system module defines it (bytes: u64, more: ptr), under the global
    // name the module references.
    let string_ty =
        "(stmts (type :string.0.sysvq0asl . (object . (fld :bytes.0 . (u 64)) (fld :more.0 . (ptr (i +64))))))";
    let m = temen_leng::translate_proc_with_types(REAL, "greet.0.", string_ty)
        .unwrap_or_else(|e| panic!("translate real greet: {e}"));
    temen_verify::verify_module(&m).unwrap_or_else(|e| panic!("verify greet: {e:?}"));
}
