//! **Pure-IR C bottom-edge runtime** (NIM.md §3b, W3 — issue #761): the compute leaves of the Nim
//! stdlib's `{.importc.}` surface (`memcpy`/`memset`/`bswap`/`clz`/`ctz`/`popcount`/the `__atomic_*`
//! family) lowered to plain Temen ops, so they bind as ordinary linked functions — no host capability.
//! This pins each against a native oracle and asserts **interp == JIT == native** (§9/§18); the
//! allocator/syscall leaves (which *do* need the Memory cap / POSIX) are out of scope. `memcmp` (a
//! byte-compare loop) rounds out the compute set.

use temen_interp::Value;
use temen_ir::{LinkUnit, Memory, Module};
use temen_leng::{bottom_edge_index, bottom_edge_runtime};

/// The runtime with a window declared (its `mem.*`/load/store leaves need one to run standalone; under
/// a real link the program's larger window wins by max — `temen_ir::link`).
fn runtime() -> Module {
    let mut m = bottom_edge_runtime();
    m.memory = Some(Memory { size_log2: 16 });
    temen_verify::verify_module(&m).expect("bottom-edge runtime verifies");
    m
}

/// Run `idx` on both engines over a seeded window; assert the return words and the post-run window
/// agree (§9 interp/JIT parity), and return `(return words, interp window)`.
fn run_both(m: &Module, idx: u32, args: &[i64], seed: &[u8]) -> (Vec<i64>, Vec<u8>) {
    let ivals: Vec<Value> = args.iter().map(|&n| Value::I64(n)).collect();
    let mut fuel = u64::MAX;
    let (ir, iwin) = temen_interp::run_capture(m, idx, &ivals, &mut fuel, seed);
    let iwords: Vec<i64> = ir
        .expect("interp")
        .iter()
        .map(|v| match v {
            Value::I64(n) => *n,
            Value::I32(n) => *n as i64,
            o => panic!("unexpected {o:?}"),
        })
        .collect();
    let (jout, jwin) = temen_jit::compile_and_run_capture(m, idx, args, seed).expect("jit");
    let jwords = match jout {
        temen_jit::JitOutcome::Returned(v) => v,
        o => panic!("jit: {o:?}"),
    };
    assert_eq!(iwords, jwords, "§9 interp/JIT return parity (idx {idx})");
    // Window parity over the meaningful (seeded) region — the engines may capture different
    // trailing extents, but every byte we set or touch must agree.
    let n = seed.len().min(iwin.len()).min(jwin.len());
    assert_eq!(
        iwin[..n],
        jwin[..n],
        "§9 interp/JIT window parity (idx {idx})"
    );
    (iwords, iwin)
}

#[test]
fn index_map_matches_c_and_nim_spellings() {
    // Both the C name and nimony's stem-qualified spelling resolve (substring match).
    assert_eq!(bottom_edge_index("memcpy"), Some(0));
    assert_eq!(bottom_edge_index("copyMem_memcpy.0.sysvq0asl"), Some(0));
    assert_eq!(bottom_edge_index("memset"), Some(1));
    assert_eq!(bottom_edge_index("__builtin_bswap64"), Some(2));
    assert_eq!(bottom_edge_index("__builtin_clzll"), Some(3));
    assert_eq!(bottom_edge_index("__builtin_ctzll"), Some(4));
    assert_eq!(bottom_edge_index("__builtin_popcountll"), Some(5));
    assert_eq!(bottom_edge_index("__atomic_add_fetch_8"), Some(6));
    assert_eq!(bottom_edge_index("__atomic_sub_fetch_8"), Some(7));
    assert_eq!(bottom_edge_index("__atomic_fetch_add_8"), Some(8));
    assert_eq!(bottom_edge_index("__atomic_exchange_n"), Some(9));
    assert_eq!(bottom_edge_index("__atomic_load_8"), Some(10));
    assert_eq!(bottom_edge_index("__atomic_store_8"), Some(11));
    assert_eq!(bottom_edge_index("memcmp"), Some(12));
    assert_eq!(bottom_edge_index("nimCmpMem_memcmp.0.sysvq0asl"), Some(12));
    assert_eq!(bottom_edge_index("mmap"), None); // an allocator leaf — not provided here
}

#[test]
fn bitcount_and_bswap_leaves_match_native() {
    let m = runtime();
    for x in [
        0i64,
        1,
        -1,
        0x0000_0100_0000_0000,
        0x1234_5678_9abc_def0u64 as i64,
        i64::MIN,
        0xff,
    ] {
        assert_eq!(
            run_both(&m, 2, &[x], &[]).0,
            [x.swap_bytes()],
            "bswap64 {x:#x}"
        );
        assert_eq!(
            run_both(&m, 3, &[x], &[]).0,
            [(x as u64).leading_zeros() as i64],
            "clzll {x:#x}"
        );
        assert_eq!(
            run_both(&m, 4, &[x], &[]).0,
            [(x as u64).trailing_zeros() as i64],
            "ctzll {x:#x}"
        );
        assert_eq!(
            run_both(&m, 5, &[x], &[]).0,
            [(x as u64).count_ones() as i64],
            "popcountll {x:#x}"
        );
    }
}

#[test]
fn memcpy_and_memset_match_native() {
    let m = runtime();
    let (dst, src, n) = (256usize, 128usize, 24usize);
    let mut seed = vec![0u8; 4096];
    let payload: Vec<u8> = (0..n)
        .map(|i| (i as u8).wrapping_mul(37).wrapping_add(1))
        .collect();
    seed[src..src + n].copy_from_slice(&payload);

    // memcpy(dst, src, n) -> dst; the n bytes land at dst, byte-identical to the source.
    let (ret, win) = run_both(&m, 0, &[dst as i64, src as i64, n as i64], &seed);
    assert_eq!(ret, [dst as i64], "memcpy returns dst");
    assert_eq!(&win[dst..dst + n], &payload[..], "memcpy copied the bytes");

    // memset(dst, 0xAB, n) -> dst; n bytes of 0xAB at dst.
    let (ret, win) = run_both(&m, 1, &[dst as i64, 0xAB, n as i64], &seed);
    assert_eq!(ret, [dst as i64], "memset returns dst");
    assert!(
        win[dst..dst + n].iter().all(|&b| b == 0xAB),
        "memset filled 0xAB"
    );
}

#[test]
fn imports_bind_to_the_runtime_via_link() {
    // A "user" module (the shape a translated Nim module has) whose calls to the C bottom edge are
    // Temen imports; linking against the runtime binds each import to a pure-IR function — the #761 point.
    const USER: &str = r#"
memory 16
import 0 "memcpy" (i64, i64, i64) -> (i64)
import 1 "__builtin_clzll" (i64) -> (i64)
func (i64, i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64, v2: i64) {
  v3 = call.import 0 (v0, v1, v2)
  v4 = call.import 1 (v3)
  return v4
  }
}
"#;
    let user = temen_text::parse_module(USER).expect("user module parses");
    // Bind each of the user's imports to its runtime function (the linker resolves by export name).
    let rt_exports: Vec<(String, u32)> = user
        .imports
        .iter()
        .map(|imp| {
            (
                imp.name.clone(),
                bottom_edge_index(&imp.name).expect("import is a provided leaf"),
            )
        })
        .collect();
    let units = vec![
        LinkUnit {
            module: user,
            exports: vec![("run".to_string(), 0)],
            ..Default::default()
        },
        LinkUnit {
            module: bottom_edge_runtime(),
            exports: rt_exports,
            ..Default::default()
        },
    ];
    let linked = temen_ir::link(&units).expect("link user + bottom-edge runtime");
    temen_verify::verify_module(&linked).expect("linked module verifies (imports all bound)");
    assert!(linked.imports.is_empty(), "every bottom-edge import bound");

    let entry = linked
        .exports
        .iter()
        .find(|e| e.name == "run")
        .expect("run exported")
        .func;

    // run(dst=256, src=128, 8): memcpy 8 bytes 128->256 (returns 256), then clzll(256) = 55.
    let mut seed = vec![0u8; 4096];
    seed[128..136].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
    let (ret, win) = run_both(&linked, entry, &[256, 128, 8], &seed);
    assert_eq!(ret, [55], "clzll(memcpy(...)=256) = 55");
    assert_eq!(
        &win[256..264],
        &[1, 2, 3, 4, 5, 6, 7, 8],
        "memcpy ran through the link"
    );
}

#[test]
fn single_thread_atomics_match_native() {
    let m = runtime();
    let p = 512usize;
    let seed = |v: i64| {
        let mut s = vec![0u8; 4096];
        s[p..p + 8].copy_from_slice(&v.to_le_bytes());
        s
    };
    let word = |win: &[u8]| i64::from_le_bytes(win[p..p + 8].try_into().unwrap());

    // add_fetch: returns new, stores new.
    let (ret, win) = run_both(&m, 6, &[p as i64, 5], &seed(10));
    assert_eq!((ret[0], word(&win)), (15, 15), "add_fetch");
    // sub_fetch.
    let (ret, win) = run_both(&m, 7, &[p as i64, 4], &seed(10));
    assert_eq!((ret[0], word(&win)), (6, 6), "sub_fetch");
    // fetch_add: returns old, stores new.
    let (ret, win) = run_both(&m, 8, &[p as i64, 5], &seed(10));
    assert_eq!((ret[0], word(&win)), (10, 15), "fetch_add");
    // exchange: returns old, stores val.
    let (ret, win) = run_both(&m, 9, &[p as i64, 99], &seed(10));
    assert_eq!((ret[0], word(&win)), (10, 99), "exchange");
    // load: returns *p.
    let (ret, _) = run_both(&m, 10, &[p as i64], &seed(42));
    assert_eq!(ret, [42], "load");
    // store: *p = val (void).
    let (ret, win) = run_both(&m, 11, &[p as i64, 77], &seed(10));
    assert!(ret.is_empty(), "store returns void");
    assert_eq!(word(&win), 77, "store wrote *p");
}

#[test]
fn memcmp_matches_native() {
    let m = runtime();
    let (a, b, n) = (256usize, 512usize, 16usize);
    // Oracle: the first differing unsigned byte's `a[i] - b[i]` (signed), else 0.
    let oracle = |x: &[u8], y: &[u8]| -> i64 {
        (0..n)
            .find(|&i| x[i] != y[i])
            .map_or(0, |i| x[i] as i64 - y[i] as i64)
    };
    let base: Vec<u8> = (0..n as u8).collect();
    let mut a_gt = base.clone();
    a_gt[5] = 200; // differs at byte 5, a > b
    let mut a_lo = base.clone();
    a_lo[0] = 1;
    let mut b_lo = base.clone();
    b_lo[0] = 250; // differ at byte 0, a < b
    let mut a_last = base.clone();
    a_last[n - 1] = 0xff; // differ at the last byte
    let cases: [(&[u8], &[u8]); 4] = [
        (&base, &base),   // equal → 0
        (&a_gt, &base),   // positive
        (&a_lo, &b_lo),   // negative
        (&a_last, &base), // last-byte difference
    ];
    for (x, y) in cases {
        let mut seed = vec![0u8; 4096];
        seed[a..a + n].copy_from_slice(x);
        seed[b..b + n].copy_from_slice(y);
        let (ret, _) = run_both(&m, 12, &[a as i64, b as i64, n as i64], &seed);
        assert_eq!(ret, [oracle(x, y)], "memcmp {x:?} vs {y:?}");
    }
    // n = 0 → 0 regardless of contents (no bytes compared).
    let (ret, _) = run_both(&m, 12, &[a as i64, b as i64, 0], &vec![0xAA; 4096]);
    assert_eq!(ret, [0], "memcmp n=0");
}
