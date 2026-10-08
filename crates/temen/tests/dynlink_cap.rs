//! Guest-driven **host-assisted dynamic linking** (DESIGN.md §22): the `Jit` capability's new
//! `compile_linked` op (op 5). A guest submits a serialized unit that still carries **unresolved §7
//! imports** plus a **symbol-table buffer** (`name → slot`); the host resolves the imports by name
//! against that table, re-verifies, and compiles — all in-sandbox, driven by guest code. This is the
//! cap-op the guest-side `vm_dlopen` will call; the harness-level `dynlink_repl.rs` is its spec.
//!
//! Every test is **differential**: the exact same guest runs on the reference interpreter and the
//! JIT (with a byte-identical powerbox), and the outcomes + final memory must agree — the op behaves
//! identically on both backends, like every other `Jit` op.

use temen_encode::{encode_module, encode_symbol_table};
use temen_interp::{run_capture_reserved_with_host, Host, MemLayout, Value};
use temen_ir::{Resolved, SymbolTable, DEFAULT_RESERVED_LOG2};
use temen_jit::{JitOutcome, TrapKind};
use temen_run::{grant_jit, jit_cap_run};
use temen_text::parse_module;
use temen_verify::verify_module;

/// A self-contained blob (no imports): parse, verify, encode.
fn blob(src: &str) -> Vec<u8> {
    let m = parse_module(src).expect("parse blob");
    verify_module(&m).expect("verify blob");
    encode_module(&m)
}

/// A unit serialized **with its §7 imports still unresolved** — `verify` would reject it (imports
/// are resolved before verify), so we only parse + encode. This is the `.so` a loader resolves.
fn unresolved_blob(src: &str) -> Vec<u8> {
    encode_module(&parse_module(src).expect("parse unresolved blob"))
}

/// Run `guest_src`'s func 0 on both backends over the prepared `init` window image (handle = arg 0,
/// then `user_args`), reserving a `2^table_log2`-slot `call.dyn` table identically on each;
/// assert the outcomes + final memory agree and return the JIT's `(outcome, final_mem)`.
fn diff(guest_src: &str, init: &[u8], user_args: &[i64], table_log2: u8) -> (JitOutcome, Vec<u8>) {
    let m = parse_module(guest_src).expect("parse guest");
    verify_module(&m).expect("verify guest");

    // Interpreter.
    let mut host_i = Host::new();
    let h_i = grant_jit(&mut host_i, &m, table_log2);
    let mut iargs = vec![Value::I32(h_i)];
    iargs.extend(user_args.iter().map(|&a| Value::I32(a as i32)));
    let mut fuel = 50_000_000u64;
    let (ires, imem) = run_capture_reserved_with_host(
        &m,
        0,
        &iargs,
        &mut fuel,
        init,
        DEFAULT_RESERVED_LOG2,
        &mut host_i,
    );

    // JIT — a fresh Host configured identically (deterministic grant ⇒ identical handle value).
    let mut host_j = Host::new();
    let h_j = grant_jit(&mut host_j, &m, table_log2);
    assert_eq!(h_i, h_j, "identical powerbox setup mints identical handles");
    let mut jargs = vec![h_j as i64];
    jargs.extend_from_slice(user_args);
    let (jout, jmem) = jit_cap_run(
        &m,
        0,
        &jargs,
        &MemLayout::image(init.to_vec()),
        DEFAULT_RESERVED_LOG2,
        table_log2,
        &mut host_j,
        None,
    )
    .expect("jit run");

    match (&ires, &jout) {
        (Ok(vals), JitOutcome::Returned(slots)) => {
            assert_eq!(vals.len(), slots.len(), "result arity");
            for (v, s) in vals.iter().zip(slots) {
                let iv = match v {
                    Value::I32(x) => *x as i64,
                    Value::I64(x) => *x,
                    other => panic!("scalar result expected, got {other:?}"),
                };
                assert_eq!(iv, *s, "interp {ires:?} != jit {jout:?}");
            }
        }
        // The same trap on both engines: one wire code (#1735).
        (Err(t), JitOutcome::Trapped(k)) if t.code() == k.code() => {}
        other => panic!("backends disagree: {other:?}"),
    }
    assert_eq!(imem, jmem.bytes(), "final memory must be byte-identical");
    (jout, jmem.bytes().to_vec())
}

// Window layout shared by the guests below (all above the #1094 NULL guard at 16384).
const SVC_OFF: usize = 20480;
const UNIT_OFF: usize = 22528;
const SYMTAB_OFF: usize = 24576;

/// `service(a, b) = a*a + b` — a self-contained unit the guest installs into the table.
const SERVICE: &str = "memory 16\nfunc (i32, i32) -> (i32) {\n\
    block 0 (v0: i32, v1: i32) {\n  v2 = i32.mul v0 v0\n  v3 = i32.add v2 v1\n  return v3\n  }\n}\n";

/// `unit(a, b) = F(a, b) + 100`, where `F` is an **unresolved import** the loader binds by name.
const UNIT: &str = "memory 16\nfunc (i32, i32) -> (i32) {\n\
    block 0 (v0: i32, v1: i32) {\n  v2 = i32.const 0\n\
    \x20 v3 = call.sym \"F\" (i32, i32) -> (i32) v2 (v0, v1)\n\
    \x20 v4 = i32.const 100\n  v5 = i32.add v3 v4\n  return v5\n  }\n}\n";

/// Seed an init image with the service at [`SVC_OFF`] and the unit at [`UNIT_OFF`].
fn seed_service_and_unit(svc: &[u8], unit: &[u8]) -> Vec<u8> {
    let mut init = vec![0u8; UNIT_OFF + unit.len()];
    init[SVC_OFF..SVC_OFF + svc.len()].copy_from_slice(svc);
    init[UNIT_OFF..UNIT_OFF + unit.len()].copy_from_slice(unit);
    init
}

/// The guest for the full install→link→invoke flow (shared by the success case and the type-mismatch
/// case): compile the service at [`SVC_OFF`], install it, build the symbol table binding `"F"` to the
/// **install slot** at [`SYMTAB_OFF`] (`[count=1, namelen=1, 'F', kind=0/Slot, slot]`; `'F'`=70, a
/// slot < 128 is one uleb byte), `compile_linked` the unit at [`UNIT_OFF`] against it, and invoke
/// `unit(a, b)`. The slot is read back from install (the loader's real pattern — never hard-coded).
fn install_link_invoke_guest(svc_len: usize, unit_len: usize) -> String {
    format!(
        "memory 16\nfunc (i32, i32, i32) -> (i32) {{\nblock 0 (v0: i32, v1: i32, v2: i32) {{\n\
         \x20 v3 = i64.const {svc_off}\n  v4 = i64.const {svc_len}\n\
         \x20 v5 = call.cap 11 0 (i64, i64) -> (i64) v0 (v3, v4)\n\
         \x20 v6 = call.cap 11 3 (i64) -> (i64) v0 (v5)\n\
         \x20 v7 = i64.const {st}\n  v8 = i32.const 1\n  i32.store8 v7 v8\n\
         \x20 v9 = i64.const {st1}\n  i32.store8 v9 v8\n\
         \x20 v10 = i64.const {st2}\n  v11 = i32.const 70\n  i32.store8 v10 v11\n\
         \x20 v12 = i64.const {st3}\n  v13 = i32.const 0\n  i32.store8 v12 v13\n\
         \x20 v14 = i64.const {st4}\n  v15 = i32.wrap_i64 v6\n  i32.store8 v14 v15\n\
         \x20 v16 = i64.const {unit_off}\n  v17 = i64.const {unit_len}\n\
         \x20 v18 = i64.const {st}\n  v19 = i64.const 5\n\
         \x20 v20 = call.cap 11 5 (i64, i64, i64, i64) -> (i64) v0 (v16, v17, v18, v19)\n\
         \x20 v21 = call.cap 11 1 (i64, i32, i32) -> (i32) v0 (v20, v1, v2)\n\
         \x20 return v21\n  }}\n}}\n",
        svc_off = SVC_OFF,
        unit_off = UNIT_OFF,
        st = SYMTAB_OFF,
        st1 = SYMTAB_OFF + 1,
        st2 = SYMTAB_OFF + 2,
        st3 = SYMTAB_OFF + 3,
        st4 = SYMTAB_OFF + 4,
    )
}

/// The flagship: the **full guest-driven REPL flow**, end to end, on both backends. The guest
/// compiles `service`, installs it (getting a table slot), **builds a symbol table in its own
/// window** binding `"F"` to that slot, `compile_linked`s the unit — which imports `F` by name —
/// against it, and finally invokes the unit. The unit reaches the installed service purely by name:
/// `unit(5, 2) = service(5, 2) + 100 = (25 + 2) + 100 = 127`. Nothing is hand-resolved in the
/// harness; the *guest* delivers the symbol table and the *host* does the rewrite-then-verify.
#[test]
fn guest_compiles_links_and_invokes_by_name_across_backends() {
    let svc = blob(SERVICE);
    let unit = unresolved_blob(UNIT);
    let init = seed_service_and_unit(&svc, &unit);
    let guest = install_link_invoke_guest(svc.len(), unit.len());

    let (out, _) = diff(&guest, &init, &[5, 2], 4);
    assert!(
        matches!(out, JitOutcome::Returned(ref s) if s == &[127]),
        "the guest-linked unit reached the installed service by name: expected 127, got {out:?}"
    );
}

/// The security edge: linking a symbol to a slot holding a **wrong-typed** function does not produce
/// a type-confused call — it **traps** at the call site. Here the same guest installs a *one*-arg
/// service `(i32)->(i32)` and binds `"F"` (which the unit imports as `(i32,i32)->(i32)`) to it. The
/// link succeeds (resolution only rewrites the import to `call.dyn <slot>` and re-verifies — a
/// `call.dyn` is well-typed IR), but at invoke the slot's `type_id` doesn't match the call's, so
/// the masked dispatch faults `IndirectCallType`, identically on both backends. The loader cannot be
/// tricked into an out-of-type dispatch — the §3c table check carries the safety, exactly as for any
/// slot the guest already controls.
#[test]
fn linking_to_a_wrong_typed_slot_traps_not_confuses() {
    // A service of the WRONG arity for the import: (i32) -> (i32).
    let svc = blob("memory 16\nfunc (i32) -> (i32) {\nblock 0 (v0: i32) {\n  return v0\n  }\n}\n");
    let unit = unresolved_blob(UNIT); // imports F as (i32, i32) -> (i32)
    let init = seed_service_and_unit(&svc, &unit);
    let guest = install_link_invoke_guest(svc.len(), unit.len());

    let (out, _) = diff(&guest, &init, &[5, 2], 4);
    assert!(
        matches!(out, JitOutcome::Trapped(TrapKind::IndirectCallType)),
        "a mis-typed link must trap (IndirectCallType), never dispatch type-confused: got {out:?}"
    );
}

/// A guest that only `compile_linked`s the unit (at [`UNIT_OFF`]) against the symbol table at
/// [`SYMTAB_OFF`] and returns the raw result (handle or `-errno`) — for the fail-closed cases.
fn compile_linked_only(unit_len: usize, symtab_len: usize) -> String {
    format!(
        "memory 16\nfunc (i32) -> (i64) {{\nblock 0 (v0: i32) {{\n\
         \x20 v1 = i64.const {unit_off}\n  v2 = i64.const {unit_len}\n\
         \x20 v3 = i64.const {st}\n  v4 = i64.const {symtab_len}\n\
         \x20 v5 = call.cap 11 5 (i64, i64, i64, i64) -> (i64) v0 (v1, v2, v3, v4)\n\
         \x20 return v5\n  }}\n}}\n",
        unit_off = UNIT_OFF,
        st = SYMTAB_OFF,
    )
}

/// Seed an init image with the unit at [`UNIT_OFF`] and a raw symbol-table byte string at
/// [`SYMTAB_OFF`].
fn seed_unit_and_symtab(unit: &[u8], symtab: &[u8]) -> Vec<u8> {
    let end = (UNIT_OFF + unit.len()).max(SYMTAB_OFF + symtab.len());
    let mut init = vec![0u8; end];
    init[UNIT_OFF..UNIT_OFF + unit.len()].copy_from_slice(unit);
    init[SYMTAB_OFF..SYMTAB_OFF + symtab.len()].copy_from_slice(symtab);
    init
}

/// Fail-closed: the unit imports `F`, but the symbol table is **empty** (`count = 0`) — `F` is
/// unresolvable, so the resolve fails before verify/compile: `-EINVAL`, identically on both backends.
#[test]
fn compile_linked_unresolved_symbol_fails_closed() {
    let unit = unresolved_blob(UNIT);
    let symtab = encode_symbol_table(&SymbolTable::default()); // a single `0` count byte
    let init = seed_unit_and_symtab(&unit, &symtab);
    let guest = compile_linked_only(unit.len(), symtab.len());
    let (out, _) = diff(&guest, &init, &[], 0);
    assert!(
        matches!(out, JitOutcome::Returned(ref s) if s == &[-22]),
        "an unresolved import must fail closed (-EINVAL) on both backends, got {out:?}"
    );
}

/// Fail-closed: a **malformed** symbol table (a bad `kind` byte) is rejected by the host decoder
/// before any IR is touched: `-EINVAL`, identically on both backends.
#[test]
fn compile_linked_malformed_symtab_fails_closed() {
    let unit = unresolved_blob(UNIT);
    // count=1, namelen=1, 'F', kind=9 (unknown) → the decoder rejects it.
    let symtab = vec![1u8, 1, 70, 9];
    let init = seed_unit_and_symtab(&unit, &symtab);
    let guest = compile_linked_only(unit.len(), symtab.len());
    let (out, _) = diff(&guest, &init, &[], 0);
    assert!(
        matches!(out, JitOutcome::Returned(ref s) if s == &[-22]),
        "a malformed symbol table must fail closed (-EINVAL) on both backends, got {out:?}"
    );
}

/// A static symbol table (built host-side) resolves the same way as the guest-built one: binding
/// `F → Slot(1)` lets `compile_linked` succeed (a real handle, ≥ 0) even before the slot is filled —
/// resolution bakes the slot into a `call.dyn`; a still-empty slot only traps at *invoke* time.
#[test]
fn compile_linked_with_a_resolvable_table_returns_a_handle() {
    let unit = unresolved_blob(UNIT);
    let mut table = SymbolTable::default();
    table.funcs.insert("F".into(), Resolved::Slot(1));
    let symtab = encode_symbol_table(&table);
    let init = seed_unit_and_symtab(&unit, &symtab);
    let guest = compile_linked_only(unit.len(), symtab.len());
    // Reserve a table (log2=4) so Slot(1) is a valid index the verifier/compile accept.
    let (out, _) = diff(&guest, &init, &[], 4);
    assert!(
        matches!(out, JitOutcome::Returned(ref s) if s[0] >= 0),
        "a resolvable import compiles to a handle on both backends, got {out:?}"
    );
}

// ---- A link unit with globals of its own (#2167) ------------------------------------------------
// The guest names a room in its window for the unit's data; the host relocates the unit there and
// writes its initial data image through the window's checked accessor. The guests below declare a
// 2^17 window, larger than the unit's own 2^16, which the unit then runs in (the linker's rule).

/// Where the guest keeps a global of its own (`base`, an `i32`) that the unit references by name.
const BASE_OFF: usize = 26624;
/// Where the guest takes the `unit_info` reply.
const INFO_OFF: usize = 28672;
/// The room the guest names for the unit's data.
const ROOM: u64 = 32768;
/// A room past the window's backed prefix, never mapped.
const UNMAPPED: u64 = 1 << 20;

/// `bump(_, by)`: add `by` to the unit's own `count` (initially 41) through a pointer-valued
/// initializer (`slot = &count`), and return it plus the guest's `base` — an initialized global, a
/// relocated data pointer and a data symbol of the guest's, in one 32-byte data image.
const DATA_UNIT: &str = "memory 16\n\
    data 16 \"\\x29\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\"\n\
    data.ptr 24 self 16\n\
    func (i64, i64) -> (i64) {\nblock 0 (v0: i64, v1: i64) {\n\
    \x20 v2 = data.self 24\n  v3 = i64.load v2\n  v4 = i32.load v3\n  v5 = i32.wrap_i64 v1\n\
    \x20 v6 = i32.add v4 v5\n  i32.store v3 v6\n  v7 = data.sym \"base\" 0\n  v8 = i32.load v7\n\
    \x20 v9 = i32.add v6 v8\n  v10 = i64.extend_i32_s v9\n  return v10\n  }\n}\n\
    export 0 data \"count\" 16\n";

/// A link unit as its object-dialect bytes (the pre-link form a loader submits).
fn object(src: &str) -> Vec<u8> {
    temen_encode::encode_unit(&parse_module(src).expect("parse link unit"))
}

/// The guest's symbol table for [`DATA_UNIT`]: `base` and the unit's room.
fn data_symtab(place: Option<u64>) -> Vec<u8> {
    let mut table = SymbolTable {
        place,
        ..SymbolTable::default()
    };
    table.data.insert("base".into(), BASE_OFF as u64);
    encode_symbol_table(&table)
}

/// An init image with the unit at [`UNIT_OFF`], the symbol table at [`SYMTAB_OFF`] and the guest's
/// `base = 1000` at [`BASE_OFF`].
fn seed_data_unit(unit: &[u8], symtab: &[u8]) -> Vec<u8> {
    let mut init = seed_unit_and_symtab(unit, symtab);
    init.resize(INFO_OFF, 0);
    init[BASE_OFF..BASE_OFF + 4].copy_from_slice(&1000i32.to_le_bytes());
    init
}

/// A 2^17-window guest that `compile_linked`s the unit against the table and returns the result;
/// with `invoke`, it then calls the unit with `by = 1` and `by = 2` and returns `r1 * 10000 + r2`.
fn data_guest(unit_len: usize, symtab_len: usize, invoke: bool) -> String {
    let tail = if invoke {
        "  v6 = i64.const 0\n  v7 = i64.const 1\n\
         \x20 v8 = call.cap 11 1 (i64, i64, i64) -> (i64) v0 (v5, v6, v7)\n\
         \x20 v9 = i64.const 2\n\
         \x20 v10 = call.cap 11 1 (i64, i64, i64) -> (i64) v0 (v5, v6, v9)\n\
         \x20 v11 = i64.const 10000\n  v12 = i64.mul v8 v11\n  v13 = i64.add v12 v10\n\
         \x20 return v13\n"
    } else {
        "  return v5\n"
    };
    format!(
        "memory 17\nfunc (i32) -> (i64) {{\nblock 0 (v0: i32) {{\n\
         \x20 v1 = i64.const {UNIT_OFF}\n  v2 = i64.const {unit_len}\n\
         \x20 v3 = i64.const {SYMTAB_OFF}\n  v4 = i64.const {symtab_len}\n\
         \x20 v5 = call.cap 11 5 (i64, i64, i64, i64) -> (i64) v0 (v1, v2, v3, v4)\n\
         {tail}  }}\n}}\n"
    )
}

/// The returned scalar of a differential run.
fn result(out: &JitOutcome) -> i64 {
    match out {
        JitOutcome::Returned(s) if s.len() == 1 => s[0],
        other => panic!("expected one result, got {other:?}"),
    }
}

/// The unit's globals live in the room the guest named, initialized from its data image, and its
/// code reaches them — and the guest's own `base` — at their relocated addresses, identically on
/// both backends down to the last byte of the window.
#[test]
fn a_link_units_globals_live_in_the_room_the_guest_names() {
    let unit = object(DATA_UNIT);
    let symtab = data_symtab(Some(ROOM));
    let init = seed_data_unit(&unit, &symtab);
    let (out, mem) = diff(&data_guest(unit.len(), symtab.len(), true), &init, &[], 0);
    // count: 41 + 1 = 42, then + 2 = 44; each call adds base (1000).
    assert_eq!(result(&out), 1042 * 10000 + 1044);
    let room = ROOM as usize;
    let mut want = [0u8; 32];
    want[16..20].copy_from_slice(&44i32.to_le_bytes());
    want[24..32].copy_from_slice(&(ROOM + 16).to_le_bytes());
    assert_eq!(
        mem[room..room + 32],
        want,
        "the room holds the unit's data, relocated"
    );
}

/// The room is the guest's to name, but the host writes it only where the guest could write itself:
/// an unmapped room is `-EFAULT` with nothing installed or written, and a room that wraps the address
/// space is refused before anything runs.
#[test]
fn a_room_the_guest_cannot_write_fails_closed() {
    let unit = object(DATA_UNIT);
    for place in [UNMAPPED, u64::MAX - 8] {
        let symtab = data_symtab(Some(place));
        let init = seed_data_unit(&unit, &symtab);
        let (out, _) = diff(&data_guest(unit.len(), symtab.len(), false), &init, &[], 0);
        assert_eq!(result(&out), -14, "room {place:#x}");
    }
}

/// What a running window cannot take is refused (`-EINVAL`), identically on both backends: a unit
/// with data and no room, a unit declaring a larger window, a runnable module carrying data (its
/// offsets are absolute), function indices baked into data, and thread-locals.
#[test]
fn what_a_running_window_cannot_take_is_refused() {
    let module_with_data = blob(
        "memory 17\ndata 16400 \"\\x29\"\nfunc (i64, i64) -> (i64) {\n\
         block 0 (v0: i64, v1: i64) {\n  return v1\n  }\n}\n",
    );
    let cases = [
        (object(DATA_UNIT), data_symtab(None)),
        (
            object(&DATA_UNIT.replace("memory 16", "memory 18")),
            data_symtab(Some(ROOM)),
        ),
        (module_with_data, data_symtab(Some(ROOM))),
        (
            object(&format!("data.funcref 16\n{DATA_UNIT}")),
            data_symtab(Some(ROOM)),
        ),
        (
            object(&format!("data tls 0 \"\\x01\"\n{DATA_UNIT}")),
            data_symtab(Some(ROOM)),
        ),
    ];
    for (i, (unit, symtab)) in cases.iter().enumerate() {
        let init = seed_data_unit(unit, symtab);
        let (out, _) = diff(&data_guest(unit.len(), symtab.len(), false), &init, &[], 0);
        assert_eq!(result(&out), -22, "case {i}");
    }
}

/// `unit_info` (op 6) tells a loader what to place: the room (32 bytes) and the exported `count` at
/// offset 16. The reply is written only when it fits, and its length is returned either way; a
/// malformed unit is `-EINVAL` and a buffer the guest cannot write is `-EFAULT`.
#[test]
fn unit_info_describes_a_link_units_data() {
    let unit = object(DATA_UNIT);
    let init = seed_data_unit(&unit, &[]);
    let reply = [32u8, 1, 5, b'c', b'o', b'u', b'n', b't', 16];
    let info = |ir_len: usize, buf: u64, cap: i64| {
        let guest = format!(
            "memory 17\nfunc (i32) -> (i64) {{\nblock 0 (v0: i32) {{\n\
             \x20 v1 = i64.const {UNIT_OFF}\n  v2 = i64.const {ir_len}\n\
             \x20 v3 = i64.const {buf}\n  v4 = i64.const {cap}\n\
             \x20 v5 = call.cap 11 6 (i64, i64, i64, i64) -> (i64) v0 (v1, v2, v3, v4)\n\
             \x20 return v5\n  }}\n}}\n"
        );
        let (out, mem) = diff(&guest, &init, &[], 0);
        (result(&out), mem[INFO_OFF..INFO_OFF + reply.len()].to_vec())
    };
    let none = vec![0u8; reply.len()];
    assert_eq!(
        info(unit.len(), INFO_OFF as u64, 0),
        (9, none.clone()),
        "too small: the length"
    );
    assert_eq!(info(unit.len(), INFO_OFF as u64, 64), (9, reply.to_vec()));
    assert_eq!(
        info(3, INFO_OFF as u64, 64),
        (-22, none.clone()),
        "malformed"
    );
    assert_eq!(
        info(unit.len(), UNMAPPED, 64),
        (-14, none),
        "unwritable buffer"
    );
}

/// Where the freeze/thaw guest keeps the unit's code handle between its two phases.
const CODE_SLOT: usize = 30720;

/// The freeze/thaw guest: func 0 links [`DATA_UNIT`], keeps its code handle at [`CODE_SLOT`] and
/// calls `bump(1)`; func 1 calls `bump(2)` through the kept handle.
fn freeze_thaw_guest(unit_len: usize, symtab_len: usize) -> String {
    format!(
        "memory 17\nfunc (i32) -> (i64) {{\nblock 0 (v0: i32) {{\n\
         \x20 v1 = i64.const {UNIT_OFF}\n  v2 = i64.const {unit_len}\n\
         \x20 v3 = i64.const {SYMTAB_OFF}\n  v4 = i64.const {symtab_len}\n\
         \x20 v5 = call.cap 11 5 (i64, i64, i64, i64) -> (i64) v0 (v1, v2, v3, v4)\n\
         \x20 v6 = i64.const {CODE_SLOT}\n  i64.store v6 v5\n\
         \x20 v7 = i64.const 0\n  v8 = i64.const 1\n\
         \x20 v9 = call.cap 11 1 (i64, i64, i64) -> (i64) v0 (v5, v7, v8)\n\
         \x20 return v9\n  }}\n}}\n\
         func (i32) -> (i64) {{\nblock 0 (v0: i32) {{\n\
         \x20 v1 = i64.const {CODE_SLOT}\n  v2 = i64.load v1\n\
         \x20 v3 = i64.const 0\n  v4 = i64.const 2\n\
         \x20 v5 = call.cap 11 1 (i64, i64, i64) -> (i64) v0 (v2, v3, v4)\n\
         \x20 return v5\n  }}\n}}\n"
    )
}

/// A loaded unit's globals survive freeze → restore: its code persists in the linked form (its data
/// addresses are constants), and its data lives in the window, so a thawed domain carries on where it
/// stopped — on the tree-walker and on the JIT alike.
#[test]
fn a_link_units_globals_survive_freeze_and_thaw() {
    let unit = object(DATA_UNIT);
    let symtab = data_symtab(Some(ROOM));
    let mut init = seed_data_unit(&unit, &symtab);
    init.resize(1 << 17, 0);
    let m = parse_module(&freeze_thaw_guest(unit.len(), symtab.len())).expect("parse guest");
    verify_module(&m).expect("verify guest");

    let mut host = Host::new();
    let h = grant_jit(&mut host, &m, 0);
    let mut fuel = 50_000_000u64;
    let (r1, window) = run_capture_reserved_with_host(
        &m,
        0,
        &[Value::I32(h)],
        &mut fuel,
        &init,
        DEFAULT_RESERVED_LOG2,
        &mut host,
    );
    assert_eq!(r1.expect("phase 1"), [Value::I64(1042)]);
    let artifact = temen_snapshot::freeze(&m, &window, &host).expect("freeze");

    let thaw = || {
        let mut thost = Host::new();
        let window = temen_snapshot::restore(&artifact, &m, &mut thost).expect("restore");
        (thost, window)
    };
    // count was 42 at the freeze: 42 + 2 + base.
    let (mut thost, restored) = thaw();
    let mut fuel = 50_000_000u64;
    let (r2, _) = run_capture_reserved_with_host(
        &m,
        1,
        &[Value::I32(h)],
        &mut fuel,
        &restored,
        DEFAULT_RESERVED_LOG2,
        &mut thost,
    );
    assert_eq!(r2.expect("phase 2 on the tree-walker"), [Value::I64(1044)]);
    let (mut jhost, restored) = thaw();
    let (out, _) = jit_cap_run(
        &m,
        1,
        &[h as i64],
        &MemLayout::image(restored),
        DEFAULT_RESERVED_LOG2,
        0,
        &mut jhost,
        None,
    )
    .expect("jit run");
    assert_eq!(result(&out), 1044, "phase 2 on the JIT");
}
