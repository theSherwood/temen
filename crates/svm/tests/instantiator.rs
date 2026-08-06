//! §14 `Instantiator` (iface 6) — VM-in-VM nesting on the interpreter's §12 executor. A parent
//! guest holding an `Instantiator` capability `instantiate`s a child confined to a power-of-two
//! sub-window of its own window, then `join`s it (parking only the calling fiber). The child runs
//! the same module's entry on the same M:N executor, confined by trap-confinement to its slice, with an
//! attenuated powerbox over its own window — an `Instantiator` (so it can recurse) and an
//! `AddressSpace` (so it can manage its own pages) — and a fuel quota. These tests pin: the child
//! runs and its result returns through `join`; its writes land **only** in its sub-window of the
//! shared parent backing (confinement — the parent sees the superset); nesting **composes to depth
//! 2**; a child manages its **own** pages via its `AddressSpace`; an out-of-range carve is rejected;
//! and a child trap propagates to the parent on `join`.
//!
//! The JIT path is deferred (it has no in-process executor to spawn a child vCPU into) — an
//! `instantiate` there resolves to a `CapFault`, like any unsupported capability.

use svm_interp::{run_capture_reserved_with_host, Host, Trap, Value};
use svm_text::parse_module;
use svm_verify::verify_module;

/// A module whose func 0 (parent) instantiates func 1 (child) in a 4 KiB sub-window at `off`,
/// joins it, and returns the child's result. The child stores a marker at its offset 0 and at an
/// in-window offset (1695) inside its 4 KiB slice, then returns 42.
fn nest_src(off: u64, size_log2: u64) -> String {
    format!(
        "memory 17\n\
         func (i32) -> (i64) {{\n\
         block 0 (v0: i32) {{\n\
         \x20 v1 = i64.const 1\n\
         \x20 v2 = i64.const {off}\n\
         \x20 v3 = i64.const {size_log2}\n\
         \x20 v4 = i64.const 0\n\
         \x20 v5 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)\n\
         \x20 v6 = cap.call 6 1 (i32) -> (i64) v0 (v5)\n\
         \x20 return v6\n\
           }}\n\
         }}\n\
         func (i64) -> (i64) {{\n\
         block 0 (v0: i64) {{\n\
         \x20 v1 = i64.const 0\n\
         \x20 v2 = i32.const 171\n\
         \x20 i32.store8 v1 v2\n\
         \x20 v3 = i64.const 1695\n\
         \x20 v4 = i32.const 200\n\
         \x20 i32.store8 v3 v4\n\
         \x20 v5 = i64.const 42\n\
         \x20 return v5\n\
           }}\n\
         }}\n"
    )
}

/// Run `src`'s entry with an `Instantiator` granted over the whole `1<<size_log2`-byte window,
/// seeded with a non-zero pattern, returning the entry result and the final window snapshot.
fn run_nested(src: &str, win_log2: u8) -> (Result<Vec<Value>, Trap>, Vec<u8>) {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let mut host = Host::new();
    let h = host.grant_instantiator(0, 1u64 << win_log2);
    let init: Vec<u8> = (0..(1u64 << win_log2))
        .map(|i| (i as u8).wrapping_mul(31) ^ 0xa5)
        .collect();
    let mut fuel = 5_000_000u64;
    run_capture_reserved_with_host(&m, 0, &[Value::I32(h)], &mut fuel, &init, 0, &mut host)
}

#[test]
fn instantiate_child_runs_confined_and_joins() {
    const OFF: u64 = 64 << 10; // a 64 KiB-aligned slot in the 128 KiB window
    const SIZE: u64 = 4096; // size_log2 = 12
    let (res, mem) = run_nested(&nest_src(OFF, 12), 17);

    // The child ran on the executor while the parent was parked on `join`, and its result returned.
    assert_eq!(
        res.expect("run ok"),
        vec![Value::I64(42)],
        "child result via join"
    );

    // Its two stores landed in the child's sub-window of the *shared parent backing* — at child
    // offset 0 (→ window OFF) and at in-window child offset 1695.
    let init: Vec<u8> = (0..(128u64 << 10))
        .map(|i| (i as u8).wrapping_mul(31) ^ 0xa5)
        .collect();
    assert_eq!(
        mem[OFF as usize], 171,
        "child store at its offset 0 missing"
    );
    assert_eq!(
        mem[(OFF + 1695) as usize],
        200,
        "child in-window store missing"
    );

    // Confinement: every byte *outside* the child's slice is exactly as the parent seeded it — the
    // child could not reach the parent's other memory.
    for i in 0..(128u64 << 10) {
        if !(OFF..OFF + SIZE).contains(&i) {
            assert_eq!(
                mem[i as usize], init[i as usize],
                "child escaped to parent byte {i}"
            );
        }
    }
}

#[test]
fn nesting_composes_to_depth_two() {
    // Depth-2 VM-in-VM, the headline §14 property — confinement composes at depth-independent cost.
    // func 0 (parent) instantiates func 1 (child) in a 4 KiB window at 64 KiB; the child — handed an
    // `Instantiator` over *its* window — instantiates func 2 (grandchild) in a 1 KiB window at child
    // offset 2 KiB (→ window 64 KiB + 2 KiB). Each level writes a marker into its own slice; the
    // parent (the superset) sees all of them, and an in-window store at each level lands in that
    // level's window — proving the grandchild is confined to its 1 KiB, nested two deep.
    let src = "memory 17\n\
         func (i32) -> (i64) {\n\
         block 0 (v0: i32) {\n\
         \x20 ; spawn via record (op 17): entry=1 off=65536 sl=12 quota=0\n\
         \x20 q0v0 = i64.const 4294967296\n\
         \x20 q0v1 = i64.const 65536\n\
         \x20 q0v2 = i64.const -4294967284\n\
         \x20 q0v3 = i64.const 4294967295\n\
         \x20 q0v4 = i64.const 0\n\
         \x20 q0a0 = i64.const 1152\n\
         \x20 i64.store q0a0 q0v0\n\
         \x20 q0a1 = i64.const 1160\n\
         \x20 i64.store q0a1 q0v1\n\
         \x20 q0a2 = i64.const 1168\n\
         \x20 i64.store q0a2 q0v2\n\
         \x20 q0a3 = i64.const 1176\n\
         \x20 i64.store q0a3 q0v3\n\
         \x20 q0a4 = i64.const 1184\n\
         \x20 i64.store q0a4 q0v4\n\
         \x20 q0a5 = i64.const 1192\n\
         \x20 i64.store q0a5 q0v4\n\
         \x20 q0a6 = i64.const 1200\n\
         \x20 i64.store q0a6 q0v4\n\
         \x20 v5 = cap.call 6 17 (i64) -> (i32) v0 (q0a0)\n\
         \x20 v6 = cap.call 6 1 (i32) -> (i64) v0 (v5)\n\
         \x20 return v6\n\
           }\n\
         }\n\
         func (i64) -> (i64) {\n\
         block 0 (v0: i64) {\n\
         \x20 v1 = i32.wrap_i64 v0\n\
         \x20 v2 = i64.const 0\n\
         \x20 v3 = i32.const 171\n\
         \x20 i32.store8 v2 v3\n\
         \x20 ; spawn via record (op 17): entry=2 off=2048 sl=10 quota=0\n\
         \x20 q1v0 = i64.const 8589934592\n\
         \x20 q1v1 = i64.const 2048\n\
         \x20 q1v2 = i64.const -4294967286\n\
         \x20 q1v3 = i64.const 4294967295\n\
         \x20 q1v4 = i64.const 0\n\
         \x20 q1a0 = i64.const 1216\n\
         \x20 i64.store q1a0 q1v0\n\
         \x20 q1a1 = i64.const 1224\n\
         \x20 i64.store q1a1 q1v1\n\
         \x20 q1a2 = i64.const 1232\n\
         \x20 i64.store q1a2 q1v2\n\
         \x20 q1a3 = i64.const 1240\n\
         \x20 i64.store q1a3 q1v3\n\
         \x20 q1a4 = i64.const 1248\n\
         \x20 i64.store q1a4 q1v4\n\
         \x20 q1a5 = i64.const 1256\n\
         \x20 i64.store q1a5 q1v4\n\
         \x20 q1a6 = i64.const 1264\n\
         \x20 i64.store q1a6 q1v4\n\
         \x20 v8 = cap.call 6 17 (i64) -> (i32) v1 (q1a0)\n\
         \x20 v9 = cap.call 6 1 (i32) -> (i64) v1 (v8)\n\
         \x20 return v9\n\
           }\n\
         }\n\
         func (i64) -> (i64) {\n\
         block 0 (v0: i64) {\n\
         \x20 v1 = i64.const 0\n\
         \x20 v2 = i32.const 200\n\
         \x20 i32.store8 v1 v2\n\
         \x20 v3 = i64.const 671\n\
         \x20 v4 = i32.const 222\n\
         \x20 i32.store8 v3 v4\n\
         \x20 v5 = i64.const 77\n\
         \x20 return v5\n\
           }\n\
         }\n";
    let (res, mem) = run_nested(src, 17);
    assert_eq!(
        res.expect("run ok"),
        vec![Value::I64(77)],
        "grandchild result via two joins"
    );

    let init: Vec<u8> = (0..(128u64 << 10))
        .map(|i| (i as u8).wrapping_mul(31) ^ 0xa5)
        .collect();
    const CHILD: u64 = 64 << 10; // child window [64 KiB, 68 KiB)
    const GRAND: u64 = CHILD + 2048; // grandchild window [66 KiB, 67 KiB)
    assert_eq!(mem[CHILD as usize], 171, "child marker missing");
    assert_eq!(mem[GRAND as usize], 200, "grandchild marker missing");
    // The grandchild's in-window store at offset 671 lands in its 1 KiB window, not the child's.
    assert_eq!(
        mem[(GRAND + 671) as usize],
        222,
        "grandchild in-window store escaped its 1 KiB slice"
    );
    // Confinement: every byte outside the child's 4 KiB window is exactly as the parent seeded
    // it — except the parent's own 56-byte spawn record at 1152 (the child's record at
    // child-relative 1216 lands inside its exempted window).
    for i in 0..(128u64 << 10) {
        if !(CHILD..CHILD + 4096).contains(&i) && !(1152..1152 + 56).contains(&i) {
            assert_eq!(
                mem[i as usize], init[i as usize],
                "depth-2 nest escaped to byte {i}"
            );
        }
    }
}

#[test]
fn child_manages_its_own_pages_via_address_space() {
    // A two-arg child receives its starter caps `(Instantiator, AddressSpace)`. It uses the
    // `AddressSpace` (iface 5) to `unmap` the first 16 KiB of its **own** 64 KiB window — a §14
    // sub-window page op, which works now that the prot map is keyed window-relative. The unmap
    // decommits (zeroes) exactly the child's first 16 KiB of the shared parent backing and returns 0;
    // the rest of the child's window, and the entire parent outside it, stay as the parent seeded.
    const CHILD: u64 = 64 << 10; // child window [64 KiB, 128 KiB)
    const SPAN: u64 = 16 << 10; // unmap the child's first 16 KiB (a whole multiple of any host page)
    let src = "memory 18\n\
         func (i32) -> (i64) {\n\
         block 0 (v0: i32) {\n\
         \x20 ; spawn via record (op 17): entry=1 off=65536 sl=16 quota=0\n\
         \x20 q2v0 = i64.const 4294967296\n\
         \x20 q2v1 = i64.const 65536\n\
         \x20 q2v2 = i64.const -4294967280\n\
         \x20 q2v3 = i64.const 4294967295\n\
         \x20 q2v4 = i64.const 0\n\
         \x20 q2a0 = i64.const 1280\n\
         \x20 i64.store q2a0 q2v0\n\
         \x20 q2a1 = i64.const 1288\n\
         \x20 i64.store q2a1 q2v1\n\
         \x20 q2a2 = i64.const 1296\n\
         \x20 i64.store q2a2 q2v2\n\
         \x20 q2a3 = i64.const 1304\n\
         \x20 i64.store q2a3 q2v3\n\
         \x20 q2a4 = i64.const 1312\n\
         \x20 i64.store q2a4 q2v4\n\
         \x20 q2a5 = i64.const 1320\n\
         \x20 i64.store q2a5 q2v4\n\
         \x20 q2a6 = i64.const 1328\n\
         \x20 i64.store q2a6 q2v4\n\
         \x20 v5 = cap.call 6 17 (i64) -> (i32) v0 (q2a0)\n\
         \x20 v6 = cap.call 6 1 (i32) -> (i64) v0 (v5)\n\
         \x20 return v6\n\
           }\n\
         }\n\
         func (i64, i64) -> (i64) {\n\
         block 0 (v0: i64, v1: i64) {\n\
         \x20 v2 = i32.wrap_i64 v1\n\
         \x20 v3 = i64.const 0\n\
         \x20 v4 = i64.const 16384\n\
         \x20 v5 = cap.call 5 1 (i64, i64) -> (i64) v2 (v3, v4)\n\
         \x20 return v5\n\
           }\n\
         }\n";
    let (res, mem) = run_nested(src, 18); // 256 KiB window so a 64 KiB child fits at 64 KiB
    assert_eq!(
        res.expect("run ok"),
        vec![Value::I64(0)],
        "child unmap should succeed"
    );

    let init: Vec<u8> = (0..(256u64 << 10))
        .map(|i| (i as u8).wrapping_mul(31) ^ 0xa5)
        .collect();
    for i in 0..(256u64 << 10) {
        if (CHILD..CHILD + SPAN).contains(&i) {
            assert_eq!(
                mem[i as usize], 0,
                "child's unmapped page byte {i} not decommitted"
            );
        } else if !(1280..1280 + 56).contains(&i) {
            // (1280..1336 is the parent's own 56-byte spawn record.)
            assert_eq!(
                mem[i as usize], init[i as usize],
                "byte {i} outside the child's unmap changed"
            );
        }
    }
}

#[test]
fn instantiate_rejects_out_of_range_carve() {
    // A child window at offset 128 KiB does not fit in the 128 KiB holder → instantiate returns a
    // negative handle; the parent returns it without joining (joining a bad handle would fault).
    let src = "memory 17\n\
         func (i32) -> (i64) {\n\
         block 0 (v0: i32) {\n\
         \x20 ; spawn via record (op 17): entry=1 off=131072 sl=12 quota=0\n\
         \x20 q3v0 = i64.const 4294967296\n\
         \x20 q3v1 = i64.const 131072\n\
         \x20 q3v2 = i64.const -4294967284\n\
         \x20 q3v3 = i64.const 4294967295\n\
         \x20 q3v4 = i64.const 0\n\
         \x20 q3a0 = i64.const 1344\n\
         \x20 i64.store q3a0 q3v0\n\
         \x20 q3a1 = i64.const 1352\n\
         \x20 i64.store q3a1 q3v1\n\
         \x20 q3a2 = i64.const 1360\n\
         \x20 i64.store q3a2 q3v2\n\
         \x20 q3a3 = i64.const 1368\n\
         \x20 i64.store q3a3 q3v3\n\
         \x20 q3a4 = i64.const 1376\n\
         \x20 i64.store q3a4 q3v4\n\
         \x20 q3a5 = i64.const 1384\n\
         \x20 i64.store q3a5 q3v4\n\
         \x20 q3a6 = i64.const 1392\n\
         \x20 i64.store q3a6 q3v4\n\
         \x20 v5 = cap.call 6 17 (i64) -> (i32) v0 (q3a0)\n\
         \x20 v6 = i64.extend_i32_s v5\n\
         \x20 return v6\n\
           }\n\
         }\n\
         func (i64) -> (i64) {\n\
         block 0 (v0: i64) {\n\
         \x20 v1 = i64.const 0\n\
         \x20 return v1\n\
           }\n\
         }\n";
    let (res, _mem) = run_nested(src, 17);
    assert_eq!(
        res.expect("run ok"),
        vec![Value::I64(-22)],
        "an out-of-range carve must return -EINVAL, not a live child"
    );
}

#[test]
fn child_trap_propagates_on_join() {
    // The child traps (`unreachable`); `join` must surface it as the parent's trap, not a value.
    let src = "memory 17\n\
         func (i32) -> (i64) {\n\
         block 0 (v0: i32) {\n\
         \x20 ; spawn via record (op 17): entry=1 off=0 sl=12 quota=0\n\
         \x20 q4v0 = i64.const 4294967296\n\
         \x20 q4v1 = i64.const 0\n\
         \x20 q4v2 = i64.const -4294967284\n\
         \x20 q4v3 = i64.const 4294967295\n\
         \x20 q4a0 = i64.const 4352\n\
         \x20 i64.store q4a0 q4v0\n\
         \x20 q4a1 = i64.const 4360\n\
         \x20 i64.store q4a1 q4v1\n\
         \x20 q4a2 = i64.const 4368\n\
         \x20 i64.store q4a2 q4v2\n\
         \x20 q4a3 = i64.const 4376\n\
         \x20 i64.store q4a3 q4v3\n\
         \x20 q4a4 = i64.const 4384\n\
         \x20 i64.store q4a4 q4v1\n\
         \x20 q4a5 = i64.const 4392\n\
         \x20 i64.store q4a5 q4v1\n\
         \x20 q4a6 = i64.const 4400\n\
         \x20 i64.store q4a6 q4v1\n\
         \x20 v5 = cap.call 6 17 (i64) -> (i32) v0 (q4a0)\n\
         \x20 v6 = cap.call 6 1 (i32) -> (i64) v0 (v5)\n\
         \x20 return v6\n\
           }\n\
         }\n\
         func (i64) -> (i64) {\n\
         block 0 (v0: i64) {\n\
         \x20 unreachable\n\
           }\n\
         }\n";
    let (res, _mem) = run_nested(src, 17);
    assert!(
        res.is_err(),
        "a child trap must propagate through join, got {res:?}"
    );
}
