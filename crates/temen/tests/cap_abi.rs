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

/// Every seed's guest runs to an answer, a trap or a refusal on both engines, and the host survives
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
