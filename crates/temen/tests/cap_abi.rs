//! #2353 — the capability-ABI fuzz harness (`support/cap_abi.rs`) driven from seeds on stable: the
//! same `fuzz_one` the `cap_abi` fuzz target runs, so the test suite keeps what the fuzzer finds
//! fixed, and checks that the generated guests reach every capability in the powerbox.

#[path = "support/cap_abi.rs"]
mod cap_abi;

#[global_allocator]
static ALLOC: cap_abi::Bounded = cap_abi::Bounded;

/// `n` inputs from a fixed xorshift64* stream, each 64 to 1 KiB long.
fn seeds(n: usize) -> impl Iterator<Item = Vec<u8>> {
    let mut x: u64 = 0x2353_cafe_f00d_d00d;
    let mut next = move || {
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    };
    (0..n).map(move |_| {
        let len = 64 + next() as usize % 960;
        (0..len).map(|_| next() as u8).collect()
    })
}

/// Every seed's guest runs to an answer, a trap or a refusal on every engine, and the host survives
/// it. A panic names the seed and prints its guest; an abort (a refused allocation) does not unwind,
/// so `CAP_ABI_TRACE=1` prints each seed's number before it runs.
#[test]
fn random_capability_calls_never_crash_the_host() {
    let (mut completed, mut runs) = (0, 0);
    let mut answered = std::collections::BTreeSet::new();
    for (i, seed) in seeds(1500).enumerate() {
        if std::env::var_os("CAP_ABI_TRACE").is_some() {
            eprintln!("seed {i}");
        }
        let r = std::panic::catch_unwind(|| cap_abi::fuzz_one(&seed)).unwrap_or_else(|e| {
            eprintln!("seed {i} panicked; its guest:\n{}", cap_abi::guest(&seed));
            std::panic::resume_unwind(e)
        });
        runs += r.completed + r.trapped;
        completed += r.completed;
        // A result at all is an answer: a call the host refused would have trapped the run.
        answered.extend(r.results.iter().map(|(t, _)| *t));
    }
    assert!(
        completed * 10 >= runs,
        "{completed} of {runs} runs completed"
    );
    // What random seeds reach. A spawn takes a record naming live handles, which the coverage-guided
    // corpus builds and random bytes rarely do.
    for t in [
        temen_ir::cap_id::STREAM,
        temen_ir::cap_id::ADDRESS_SPACE,
        temen_ir::cap_id::BUDGET,
        temen_ir::cap_id::CLOCK,
        temen_ir::cap_id::MODULE_LOADER,
        temen_ir::cap_id::HOST_PROC,
        temen_ir::cap_id::JIT,
        temen_ir::CAP_SELF_TYPE_ID,
    ] {
        assert!(
            answered.contains(&t),
            "no call of type {t} answered: {answered:?}"
        );
    }
}

/// What the fuzzer found, kept found: each input runs through the harness again.
#[test]
fn the_fuzzers_finds_stay_fixed() {
    for input in [
        // `Jit.uninstall` declaring no result: the bytecode engine wrote its answer past the frame.
        &[0x00, 0xc0, 0x70, 0x75, 0x32, 0x64, 0x6f, 0x2c][..],
        // `waitpid` of the caller's own pid: it locked the caller's own process, deadlocking.
        &[
            0x00, 0x41, 0x6b, 0x64, 0x69, 0xdc, 0xdc, 0x3e, 0x01, 0x00, 0xdc, 0x3e, 0x01, 0x00,
        ],
        &[
            0x00, 0x41, 0x6b, 0x64, 0x69, 0xdc, 0x05, 0xdc, 0xb8, 0x01, 0x00,
        ],
        // A parked call declaring no result (a `fork`, a `waitpid`) answered past the frame on the
        // bytecode engine, the parent's answer and the fork twin's alike.
        &[
            0x00, 0x01, 0x6d, 0x64, 0x95, 0xf3, 0x6d, 0x6b, 0x64, 0x69, 0x72, 0x00, 0x7c, 0x8d,
            0x7c, 0x7c, 0x69, 0x00, 0x00, 0x00, 0x00,
        ],
        &[
            0x00, 0x00, 0x74, 0x63, 0xd9, 0x6e, 0x6f, 0x65, 0x74, 0x63, 0x77, 0x64, 0x6e, 0x6c,
            0x69, 0x6e, 0x6b,
        ],
        &[
            0x00, 0x01, 0x6d, 0x64, 0x95, 0xf3, 0xff, 0xff, 0xff, 0x30, 0xed, 0x7c, 0x7c, 0x7c,
            0x00, 0x8f, 0xe3, 0x39, 0x8e, 0x03, 0xe3, 0x38, 0x8e, 0x03, 0x77, 0x72, 0x72, 0x6e,
        ],
        // `kill` and `waitpid` of the group `-i64::MIN`: negating the pid overflowed.
        &[
            0x01, 0x00, 0xdb, 0x00, 0xdb, 0xdb, 0x19, 0xdf, 0xbe, 0x46, 0x5c, 0xb9, 0xdf, 0x01,
            0x19,
        ],
        &[
            0x00, 0x41, 0x6b, 0x64, 0x69, 0xdc, 0x05, 0xf0, 0x1c, 0xc1, 0x01, 0x00,
        ],
        // `SharedRegion.unmap` of a span past the window on the Cranelift JIT: its region hook
        // heard the span before the unmap refused it, and `base + win_off` overflowed (#2400).
        &[
            0x00, 0x01, 0x6d, 0x64, 0x95, 0xf3, 0x2a, 0x7c, 0x7c, 0x7c, 0x49, 0x49, 0x49, 0x49,
            0x49, 0x48, 0x49, 0x49, 0x49, 0x49, 0x7f, 0x69, 0x72, 0x31,
        ],
        // An `unmap` of 88 GiB of the reservation: every engine walked each page of the span, 16 s
        // of host CPU for a few fuel (#2370).
        &[
            0x00, 0x3f, 0xc1, 0xc1, 0xd1, 0xc1, 0xc1, 0x08, 0x2c, 0xfe, 0xff, 0xff, 0x3f, 0x03,
            0x16, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0a,
        ],
    ] {
        cap_abi::fuzz_one(input);
    }
}

/// The guest `src` (taking the POSIX personality's handle) on both interpreters.
fn on_both(src: &str) -> [Option<Result<Vec<temen_interp::Value>, temen_interp::Trap>>; 2] {
    use temen_interp::{bytecode, run_with_host, Host, Value};
    let m = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let host = || {
        let mut h = Host::new();
        let (posix, _) = temen_run::posix::posix_cap(1 << 15, 1 << 16, Vec::new());
        let px = posix.install(&mut h, 0);
        (h, [Value::I32(px)])
    };
    let (mut h, args) = host();
    let tw = run_with_host(&m, 0, &args, &mut 1_000_000, &mut h);
    let (mut h, args) = host();
    let bc = bytecode::compile_and_run_with_host(&m, 0, &args, &mut 1_000_000, &mut h);
    [Some(tw), bc]
}

/// #2353 — a process is never its own child: `waitpid` of its own pid is `-ECHILD`, and a pid past
/// `i32` names no process (cut down, it named the caller's own, whose lock the op holds).
#[test]
fn a_process_waiting_on_itself_is_echild() {
    let [tw, bc] = on_both(
        "memory 16\nfunc (i32) -> (i64, i64) {\nblock 0 (vp: i32) {\n\
           vpid = call.cap 13 44 () -> (i64) vp ()\n  vz = i64.const 0\n\
           vw = call.cap 13 28 (i64, i64, i64) -> (i64) vp (vpid, vz, vz)\n\
           vhi = i64.const 4294967296\n  vfar = i64.add vpid vhi\n\
           vk = call.cap 13 31 (i64, i64) -> (i64) vp (vfar, vz)\n  return vw vk\n  }\n}\n",
    );
    use temen_interp::Value::I64;
    let want = Ok(vec![
        I64(temen_ir::errno::ECHILD),
        I64(temen_ir::errno::ESRCH),
    ]);
    assert_eq!(tw, Some(want.clone()), "tree-walker");
    assert_eq!(bc, Some(want), "bytecode");
}

/// One call to make: its type, op, the place of its handle in the powerbox, and its args (type,
/// value).
type Call<'a> = (u32, u32, usize, &'a [(&'a str, i64)]);

/// A guest making one `call.cap {type_id} {op}` on the powerbox handle at `cap`, with constant
/// `args` (type, value), declaring `results` `i64` results. It returns the second result when it
/// declares two, else `7777`, a value defined after the call. Its function 1 is the harness's child
/// entry, as every generated guest's is.
fn one_call(
    type_id: u32,
    op: u32,
    cap: usize,
    args: &[(&str, i64)],
    results: usize,
) -> temen_ir::Module {
    let mut body = String::new();
    for (i, (ty, v)) in args.iter().enumerate() {
        body.push_str(&format!("  a{i} = {ty}.const {v}\n"));
    }
    let params: Vec<&str> = args.iter().map(|(ty, _)| *ty).collect();
    let names: Vec<String> = (0..args.len()).map(|i| format!("a{i}")).collect();
    let outs: Vec<String> = (0..results).map(|i| format!("r{i}")).collect();
    let bind = match outs.is_empty() {
        true => String::new(),
        false => format!("{} = ", outs.join(", ")),
    };
    let src =
        format!(
        "memory 20\nfunc ({}) -> (i64) {{\nblock 0 ({}) {{\n{body}  {bind}call.cap {type_id} {op} \
         ({}) -> ({}) p{cap} ({})\n  vr = i64.const 7777\n  return {}\n  }}\n}}\n",
        ["i32"; 15].join(", "),
        (0..15).map(|i| format!("p{i}: i32")).collect::<Vec<_>>().join(", "),
        params.join(", "),
        vec!["i64"; results].join(", "),
        names.join(", "),
        if results == 2 { "r1" } else { "vr" },
    ) + cap_abi::CHILD_ENTRY;
    let m = temen_text::parse_module(&src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// #2361 — every engine defines the values a `call.cap` declares, and numbers the rest of its frame
/// by them. A call to an op an engine serves itself (`temen_ir::served_contract`) declaring any
/// other count of results traps `CapFault` before the op runs. One the host serves defines each
/// result it declares: the host's, then zero. The tree-walker used to push one value whatever such
/// a call declared, which shifted every later value of its frame.
#[test]
fn every_engine_defines_the_results_a_call_declares() {
    use temen_ir::{cap_id, trap_code::CAP_FAULT, CAP_SELF_TYPE_ID as SELF};
    // Each op with the args its contract reads.
    let served: [Call; 11] = [
        (cap_id::JIT, 3, 14, &[("i64", 5)]),
        (cap_id::JIT, 4, 14, &[("i64", 999)]),
        (cap_id::INSTANTIATOR, 1, 3, &[("i32", 7)]),
        (cap_id::INSTANTIATOR, 9, 3, &[("i32", 7)]),
        (cap_id::INSTANTIATOR, 14, 3, &[("i32", 7), ("i64", 0)]),
        (cap_id::INSTANTIATOR, 17, 3, &[("i64", 0)]),
        (SELF, 9, 0, &[]),
        (SELF, 12, 0, &[("i64", 99)]),
        (SELF, 13, 0, &[]),
        (SELF, 14, 0, &[("i64", 0); 5]),
        (SELF, 16, 0, &[("i64", 4096)]),
    ];
    for (type_id, op, cap, args) in served {
        let (_, contract) = temen_ir::served_contract(type_id, op).expect("a served op");
        for results in (0..=2).filter(|&n| n != contract.len()) {
            let outcomes = cap_abi::on_every_engine(&one_call(type_id, op, cap, args, results));
            assert!(
                outcomes.iter().all(|o| *o == Some(Err(CAP_FAULT))),
                "({type_id}, {op}) declaring {results} results: {outcomes:?}"
            );
        }
    }
    // A stream write, the clock, the address space's page size, a `Jit.release`: one result each.
    let host_served: [Call; 4] = [
        (cap_id::STREAM, 1, 0, &[("i64", 4096), ("i64", 0)]),
        (cap_id::CLOCK, 0, 8, &[]),
        (cap_id::ADDRESS_SPACE, 3, 4, &[]),
        (cap_id::JIT, 2, 14, &[("i64", 5)]),
    ];
    for (type_id, op, cap, args) in host_served {
        for (results, want) in [(0, 7777), (2, 0)] {
            let outcomes = cap_abi::on_every_engine(&one_call(type_id, op, cap, args, results));
            assert!(
                outcomes.iter().all(|o| *o == Some(Ok(vec![want]))),
                "({type_id}, {op}) declaring {results} results: {outcomes:?}"
            );
        }
    }
}
