//! Linking against the **real `system` module** (NIM.md W2, Path A). The stand-in system objects in
//! `whole.rs`/`strings.rs` are no-op stubs; here a user program binds a `sysvq0asl` edge to a proc
//! **translated from the actual nimony `system` module** — real compiled stdlib code running on SVM.
//!
//! The keystone is `=wasMoved`, string's ARC "moved-from" reset: `s.bytes = 0` through a `ptr string`
//! (verbatim from `system/stringimpl.nim`, exportc `nimStrWasMoved`). A driver calls it across the
//! module boundary; the linker binds the import to the real proc. Both engines, §9 parity.
//!
//! The string ARC set — `=wasMoved`, `=destroy`, `=copy` (verbatim from `system/stringimpl.nim`) —
//! now translates and runs against `arcDec`/`dealloc`/`arcInc`/`copyMem` stubs. The system `ini`
//! chain (exception globals) remains ahead.

use svm_interp::Value;
use svm_leng::LengModule;

#[test]
fn real_system_wasmoved_runs() {
    const SYSTEM: &str = include_str!("fixtures/real_system_arc.leng.nif");
    // A driver whose `clr(p)` calls the real `=wasMoved` across the module boundary — hexer would
    // emit exactly this `=wasMoved.2.sysvq0asl` reference. The linker binds it to the system proc.
    let driver = "\
(stmts
 (proc :clr.0. (params (param :p.0 . (i +64))) . .
  (stmts . (call =wasMoved.2.sysvq0asl p.0))))";
    let linked = svm_leng::link_units(&[
        LengModule {
            stem: "drv",
            src: driver,
            names: &["clr.0."],
        },
        LengModule {
            stem: "sysvq0asl",
            src: SYSTEM,
            names: &["=wasMoved.2."],
        },
    ])
    .unwrap_or_else(|e| panic!("link against real system: {e}"));
    svm_verify::verify_module(&linked).unwrap_or_else(|e| panic!("verify: {e:?}"));

    // Seed a `string {bytes: 0xdeadbeef, more: 0}` at offset 256; `clr` must zero its `bytes` word.
    let s = 256usize;
    let mut seed = vec![0u8; 4096];
    seed[s..s + 8].copy_from_slice(&0xdeadbeefu64.to_le_bytes());
    seed[s + 8..s + 16].copy_from_slice(&7u64.to_le_bytes()); // `more` — must be left untouched

    // clr (func 0): (p) -> ().
    let ivals = [Value::I64(s as i64)];
    let mut fuel = u64::MAX;
    let (ir, imem) = svm_interp::run_capture(&linked, 0, &ivals, &mut fuel, &seed);
    ir.expect("interp clr");
    let (_j, jmem) = svm_jit::compile_and_run_capture(&linked, 0, &[s as i64], &seed).expect("jit");
    let n = imem.len().min(jmem.len());
    assert_eq!(imem[..n], jmem[..n], "§9 interp/JIT window parity");

    let bytes = u64::from_le_bytes(imem[s..s + 8].try_into().unwrap());
    let more = u64::from_le_bytes(imem[s + 8..s + 16].try_into().unwrap());
    assert_eq!(bytes, 0, "real =wasMoved zeroed string.bytes");
    assert_eq!(more, 7, "=wasMoved touches only `bytes`, not `more`");
}

/// The real `=destroy` (string ARC destructor, verbatim from `system/stringimpl.nim`): if the low
/// byte of `bytes` is the long-string tag `255`, it decrements the `LongString`'s refcount
/// (`arcDec(&s.more.rc)`) and, if that hits zero, `dealloc(s.more)`. Exercising it end-to-end
/// stresses everything this PR added — the `u8` tag load through a `(cast (ptr u8) …)`, and the
/// **typed-pointer walk** `s.more → LongString.rc`. Linked against `arcDec`/`dealloc` stubs; both
/// engines.
#[test]
fn real_system_destroy_walks_the_longstring() {
    const SYSTEM: &str = include_str!("fixtures/real_system_arc.leng.nif");
    // `arcDec(p)` writes a marker through its pointer (so we can prove it got `&s.more.rc`) and
    // returns true → `dealloc` runs. `dealloc` is a no-op. Empty stem: these are intra-`system`
    // names `=destroy` calls bare (`arcDec.0.`/`dealloc.1.`), so they must export unqualified.
    let stubs = "\
(stmts
 (proc :arcDec.0. (params (param :p.0 . (ptr (i +64)))) (bool) .
  (stmts . (asgn (deref p.0) 48879) (ret (true))))
 (proc :dealloc.1. (params (param :p.0 . (i +64))) . . (stmts . (ret .))))";
    let driver = "\
(stmts
 (proc :del.0. (params (param :p.0 . (i +64))) . .
  (stmts . (call =destroy.2.sysvq0asl p.0))))";
    let linked = svm_leng::link_units(&[
        LengModule {
            stem: "drv",
            src: driver,
            names: &["del.0."],
        },
        LengModule {
            stem: "sysvq0asl",
            src: SYSTEM,
            names: &["=destroy.2."],
        },
        LengModule {
            stem: "",
            src: stubs,
            names: &["arcDec.0.", "dealloc.1."],
        },
    ])
    .unwrap_or_else(|e| panic!("link real =destroy: {e}"));
    svm_verify::verify_module(&linked).unwrap_or_else(|e| panic!("verify: {e:?}"));

    // A "long" string at `s`: low byte of `bytes` = 255 (the tag), `more` → a LongString blob at `b`.
    let (s, b) = (256usize, 512usize);
    let mut seed = vec![0u8; 4096];
    seed[s] = 255; // bytes low byte = long-string tag
    seed[s + 8..s + 16].copy_from_slice(&(b as u64).to_le_bytes()); // more → blob

    let mut fuel = u64::MAX;
    let (ir, imem) = svm_interp::run_capture(&linked, 0, &[Value::I64(s as i64)], &mut fuel, &seed);
    ir.expect("interp del");
    let (_j, jmem) = svm_jit::compile_and_run_capture(&linked, 0, &[s as i64], &seed).expect("jit");
    let n = imem.len().min(jmem.len());
    assert_eq!(imem[..n], jmem[..n], "§9 interp/JIT window parity");

    // arcDec wrote its marker at `&s.more.rc` = blob + 8 (rc@8) — proving the typed-pointer walk
    // `s.more → LongString.rc` computed the right address.
    let rc = u64::from_le_bytes(imem[b + 8..b + 16].try_into().unwrap());
    assert_eq!(rc, 48879, "arcDec ran on &s.more.rc (blob+8)");
}

/// The same real `=destroy` on a **short** string (low byte ≠ the `255` tag) takes the other branch:
/// it must do nothing — no refcount touch, no dealloc. Proves the `u8` tag comparison actually gates
/// the ARC path.
#[test]
fn real_system_destroy_short_string_is_noop() {
    const SYSTEM: &str = include_str!("fixtures/real_system_arc.leng.nif");
    let stubs = "\
(stmts
 (proc :arcDec.0. (params (param :p.0 . (ptr (i +64)))) (bool) .
  (stmts . (asgn (deref p.0) 48879) (ret (true))))
 (proc :dealloc.1. (params (param :p.0 . (i +64))) . . (stmts . (ret .))))";
    let driver = "\
(stmts
 (proc :del.0. (params (param :p.0 . (i +64))) . .
  (stmts . (call =destroy.2.sysvq0asl p.0))))";
    let linked = svm_leng::link_units(&[
        LengModule {
            stem: "drv",
            src: driver,
            names: &["del.0."],
        },
        LengModule {
            stem: "sysvq0asl",
            src: SYSTEM,
            names: &["=destroy.2."],
        },
        LengModule {
            stem: "",
            src: stubs,
            names: &["arcDec.0.", "dealloc.1."],
        },
    ])
    .unwrap_or_else(|e| panic!("link: {e}"));

    let (s, b) = (256usize, 512usize);
    let mut seed = vec![0u8; 4096];
    seed[s] = 0x68; // low byte of a packed SSO word ("h") — not the long tag
    seed[s + 8..s + 16].copy_from_slice(&(b as u64).to_le_bytes());

    let mut fuel = u64::MAX;
    let (ir, imem) = svm_interp::run_capture(&linked, 0, &[Value::I64(s as i64)], &mut fuel, &seed);
    ir.expect("interp del");
    // The blob's rc stays 0 — the short-string branch never called arcDec.
    let rc = u64::from_le_bytes(imem[b + 8..b + 16].try_into().unwrap());
    assert_eq!(rc, 0, "short string: no arcDec, no dealloc");
}

/// The real `=copy` (string copy-assign, verbatim from `system/stringimpl.nim`) — the most complex
/// string ARC op: it reads both operands' SSO length bytes, may release the destination's old
/// `LongString` (`=destroy`-style `arcDec`/`dealloc`), retains the source's when shared (`arcInc`),
/// and `copyMem`s the 16-byte string header across. It links against the four ARC/mem imports and
/// runs the short-string path here: `copyMem` writes a marker at the destination header so we can
/// confirm the copy site was reached — proving the whole proc lowered and executed. Both engines.
#[test]
fn real_system_copy_runs_short_path() {
    const SYSTEM: &str = include_str!("fixtures/real_system_arc.leng.nif");
    // Stubs for everything `=copy` calls (empty stem — intra-`system` names). `copyMem(d, s, n)`
    // writes a marker through its destination pointer; the rest are inert.
    let stubs = "\
(stmts
 (proc :arcDec.0. (params (param :p.0 . (ptr (i +64)))) (bool) . (stmts . (ret (false))))
 (proc :arcInc.0. (params (param :p.0 . (ptr (i +64)))) . . (stmts . (ret .)))
 (proc :dealloc.1. (params (param :p.0 . (i +64))) . . (stmts . (ret .)))
 (proc :copyMem.0. (params (param :d.0 . (ptr (i +64))) (param :s.0 . (i +64)) (param :n.0 . (i +64))) . .
  (stmts . (asgn (deref d.0) 48879))))";
    // `cp(dest, src)` calls the real `=copy` across the boundary; `dest` is a `ptr string`, `src` a
    // `string` (by address).
    let driver = "\
(stmts
 (proc :cp.0. (params (param :dest.0 . (i +64)) (param :src.0 . (i +64))) . .
  (stmts . (call =copy.2.sysvq0asl dest.0 src.0))))";
    let linked = svm_leng::link_units(&[
        LengModule {
            stem: "drv",
            src: driver,
            names: &["cp.0."],
        },
        LengModule {
            stem: "sysvq0asl",
            src: SYSTEM,
            names: &["=copy.2."],
        },
        LengModule {
            stem: "",
            src: stubs,
            names: &["arcDec.0.", "arcInc.0.", "dealloc.1.", "copyMem.0."],
        },
    ])
    .unwrap_or_else(|e| panic!("link real =copy: {e}"));
    svm_verify::verify_module(&linked).unwrap_or_else(|e| panic!("verify: {e:?}"));

    // Two short strings (low byte of `bytes` ≤ 14 → short path, ≠ 255 → no dealloc): `dest` at D,
    // `src` at S. `=copy` should reach `copyMem(&dest.bytes, …)`, marking dest.bytes.
    let (d, s) = (256usize, 512usize);
    let mut seed = vec![0u8; 4096];
    seed[d] = 5; // dest SSO length
    seed[s] = 5; // src SSO length
    let mut fuel = u64::MAX;
    let (ir, imem) = svm_interp::run_capture(
        &linked,
        0,
        &[Value::I64(d as i64), Value::I64(s as i64)],
        &mut fuel,
        &seed,
    );
    ir.expect("interp cp");
    let (_j, jmem) =
        svm_jit::compile_and_run_capture(&linked, 0, &[d as i64, s as i64], &seed).expect("jit");
    let n = imem.len().min(jmem.len());
    assert_eq!(imem[..n], jmem[..n], "§9 interp/JIT window parity");
    let mark = u64::from_le_bytes(imem[d..d + 8].try_into().unwrap());
    assert_eq!(mark, 48879, "=copy reached copyMem(&dest.bytes, …)");
}

/// **The end-to-end milestone (NIM.md §3b / Path A):** a *real* nimony program running against
/// *real* compiled `system` ARC code. `greetLong(): string = "hello, this is a long string!"` — the
/// genuine `hexer` output — is linked against the **real** `=wasMoved` and `=destroy` (verbatim from
/// `system/stringimpl.nim`), not the no-op stubs the earlier long-string test used. On entry the
/// default `result` is zeroed by the real `=wasMoved` and left untouched by the real `=destroy` (its
/// SSO tag byte ≠ 255 → the short-string branch, no `arcDec`/`dealloc`); then `result` becomes the
/// long-string value whose `more` points at the `LongString` blob. The blob is a `const` in
/// greetLong's own data, and the linker **relocates** `result.more` to its placed address — so the
/// returned string reads back as the literal, identically on both engines. `arcDec`/`dealloc` are
/// linked (never reached on this path).
#[test]
fn real_long_string_end_to_end_against_real_arc() {
    const GREET: &str = include_str!("fixtures/real_longstring.leng.nif");
    const SYSTEM: &str = include_str!("fixtures/real_system_arc.leng.nif");
    let stubs = "\
(stmts
 (proc :arcDec.0. (params (param :p.0 . (ptr (i +64)))) (bool) . (stmts . (ret (false))))
 (proc :dealloc.1. (params (param :p.0 . (i +64))) . . (stmts . (ret .))))";
    let linked = svm_leng::link_units(&[
        LengModule {
            stem: "lonelga0b",
            src: GREET,
            names: &["greetLong.0."],
        },
        LengModule {
            stem: "sysvq0asl",
            src: SYSTEM,
            names: &["=wasMoved.2.", "=destroy.2."],
        },
        LengModule {
            stem: "",
            src: stubs,
            names: &["arcDec.0.", "dealloc.1."],
        },
    ])
    .unwrap_or_else(|e| panic!("link greetLong against real system ARC: {e}"));
    svm_verify::verify_module(&linked).unwrap_or_else(|e| panic!("verify: {e:?}"));

    // greetLong (func 0) is frame-needing + aggregate-returning: ($sp, $sret) -> (). `sp`/`sret`
    // sit in the low scratch page below the globals; the window must reach past the globals base
    // (`POWERBOX_STACK_PAGE`, 16384) so the snapshot covers the relocated const `LongString` blob.
    let (sp, sret) = (2048i64, 512usize);
    let seed = vec![0u8; 32768];
    let ivals = vec![Value::I64(sp), Value::I64(sret as i64)];
    let mut fuel = u64::MAX;
    let (ir, imem) = svm_interp::run_capture(&linked, 0, &ivals, &mut fuel, &seed);
    ir.expect("interp greetLong");
    let (_j, jmem) =
        svm_jit::compile_and_run_capture(&linked, 0, &[sp, sret as i64], &seed).expect("jit");
    let n = imem.len().min(jmem.len());
    assert_eq!(imem[..n], jmem[..n], "§9 interp/JIT window parity");

    // The returned `string`: `bytes` is the SSO word nimony packed; `more` was *relocated* by the
    // linker to point at the const `LongString` blob (fullLen 29, data = the literal).
    let bytes = u64::from_le_bytes(imem[sret..sret + 8].try_into().unwrap());
    let more = u64::from_le_bytes(imem[sret + 8..sret + 16].try_into().unwrap()) as usize;
    assert_eq!(bytes, 2318350419654699262, "packed SSO `bytes` word");
    assert_ne!(more, 0, "`more` relocated to the LongString blob");
    let full_len = u64::from_le_bytes(imem[more..more + 8].try_into().unwrap());
    assert_eq!(full_len, 29, "blob fullLen");
    assert_eq!(
        &imem[more + 24..more + 24 + 29],
        b"hello, this is a long string!",
        "the literal, read back through real ARC + a relocated pointer"
    );
}

/// A **`ref`-type ARC hook** — nimony compiles a `ref object` destructor/`=dup` whose parameter is
/// typed by a *named pointer alias* (`RootRef = (ptr t.0.IAref…)`), not an inline `(ptr …)`. Real
/// bytes: the `=dup` hook (from `system/arcops.nim`) takes a bare `RootRef` `src`, and if non-nil
/// does `arcInc(&src.r)` before returning it. Without pointer-alias resolution the bare-named param
/// is a plain `i64` scalar and `(deref src)` fails "not a known pointer". This drives the real hook
/// against an `arcInc` stub that bumps the refcount word, on both engines.
#[test]
fn real_ref_type_dup_hook_increments_refcount() {
    const R: &str = include_str!("fixtures/real_refhook.leng.nif");
    let m = svm_leng::translate(R).unwrap_or_else(|e| panic!("translate ref hook: {e}"));
    svm_verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));

    // A ref target at offset 256 whose `r` (refcount) word is 5. dupDriver derefs the aliased
    // pointer, `arcInc` bumps `r` (5 → 6), and reads it back through the returned (== src) pointer.
    let r = 256usize;
    let mut seed = vec![0u8; 4096];
    seed[r..r + 8].copy_from_slice(&5u64.to_le_bytes());

    // dupDriver is func 2 (arcInc = 0, =dup = 1): (refptr) -> incremented refcount.
    let mut fuel = u64::MAX;
    let (ir, imem) = svm_interp::run_capture(&m, 2, &[Value::I64(r as i64)], &mut fuel, &seed);
    let ival = ir.expect("interp dupDriver");
    let (jout, jmem) = svm_jit::compile_and_run_capture(&m, 2, &[r as i64], &seed).expect("jit");
    let jval = match jout {
        svm_jit::JitOutcome::Returned(v) => v,
        o => panic!("jit: {o:?}"),
    };
    let iword = match ival.as_slice() {
        [Value::I64(n)] => *n,
        o => panic!("unexpected {o:?}"),
    };
    assert_eq!(vec![iword], jval, "§9 interp/JIT parity");
    let n = imem.len().min(jmem.len());
    assert_eq!(imem[..n], jmem[..n], "§9 interp/JIT window parity");
    assert_eq!(
        iword, 6,
        "arcInc bumped the refcount through the aliased ref pointer"
    );
    let rc = u64::from_le_bytes(imem[r..r + 8].try_into().unwrap());
    assert_eq!(rc, 6, "the refcount word was incremented in place");
}
