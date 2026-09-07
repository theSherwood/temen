//! Host-side fixture generator: parse an Temen IR text module and write its `temen-encode` binary form
//! to the path in `argv[1]`. `run.mjs` feeds the bytes to the wasm `temen_run` entry, so the wasm
//! build is exercised on the **real decode path** (not an embedded module).

use std::io::Write;

/// Same "alu" LCG recurrence as the embedded smoke kernel in `lib.rs` — so the encoded-module path
/// can be checked against the same hand-derived anchors.
const ALU: &str = r#"
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 0
  v2 = i64.const 0
  br 1(v0, v1, v2)
}
block 1 (v3: i64, v4: i64, v5: i64) {
  v6 = i64.lt_s v5 v3
  br_if v6 2(v3, v4, v5) 3(v4)
}
block 2 (v7: i64, v8: i64, v9: i64) {
  v10 = i64.const 6364136223846793005
  v11 = i64.mul v8 v10
  v12 = i64.const 1442695040888963407
  v13 = i64.add v11 v12
  v14 = i64.add v13 v9
  v15 = i64.const 1
  v16 = i64.add v9 v15
  br 1(v7, v14, v16)
}
block 3 (v17: i64) {
  return v17
  }
}
"#;

/// #1312 — the **growing-allocator** guest, the reduced `__temen_malloc` shape the JS coop driver is
/// checked against by `browser-coop-grow-test.mjs`.
///
/// `_start` calls a heavy leaf (heavy enough to clear the leaf tier-up size floor, so the run really
/// does reach the emitted tier) which first bounces to a helper that `vm_map`s a page **past the 32
/// MiB cooperative run window**, then stores through the freshly mapped address, reads it back, and
/// folds it into a sum. The result is written to stdout so a harness can compare byte-for-byte with
/// the plain bytecode path.
///
/// `_start` also `thread.spawn`s a worker, which is what routes the guest to the **cooperative**
/// driver: the single-vCPU whole-program path declines a genuinely threaded guest at the spawn event
/// (the same lever `coop_tierup_driver.rs`'s guests use). Without it `runJitModule` would serve this
/// module on the single-shot tier and never exercise the coop window at all.
///
/// The interpreter oracle reserves `DEFAULT_RESERVED_LOG2` and grows on demand, so the map succeeds
/// there. The cooperative tier used to clamp its reservation to the run window, making the same map
/// `-EINVAL`; like the real synthesized allocator this guest does not check that, so the following
/// store faulted and the run trapped with zero tier-ups.
///
/// Handles are the on-ramp powerbox's grant order (stdout, stdin, exit, memory), so the guest needs
/// no imports. `PROBE` is 32 MiB + 16 — inside the page mapped at the window's end.
fn grow_past_window() -> String {
    // Replay `grant_onramp_caps`'s grant order (stdout, stdin, exit, memory) against a fresh `Host`
    // to learn the handle values, rather than hardcoding them — the same trick the native coop
    // differential uses. Deterministic per session, so an import-free guest can `call.cap` them.
    let (stdout, memory) = {
        let mut h = temen_interp::Host::new();
        let out = h.grant_stream(temen_interp::StreamRole::Out);
        let _stdin = h.grant_stream(temen_interp::StreamRole::In);
        let _exit = h.grant_exit();
        (out, h.grant_memory())
    };
    const WIN: u64 = 1 << 25; // JIT_RUN_WIN_LOG2 — the cooperative run window
    const PROBE: u64 = WIN + 16;
    const SLOT: u64 = 32768 + 2048; // clear of the #1094 NULL guard and the args region
                                    // The leaf's body is padded with a long arithmetic chain so its estimated emitted size clears
                                    // `MIN_TIERUP_EMITTED_FN_BYTES`; without that the driver would keep it on the interpreter and the
                                    // test would pass without ever exercising the emitted tier.
    let mut chain = String::new();
    for i in 0..400 {
        chain.push_str(&format!(
            "  vk{i} = i64.const {k}\n  vm{i} = i64.mul vacc{i} vk{i}\n  vx{i} = i64.xor vm{i} vk{i}\n  vacc{next} = i64.add vx{i} vk{i}\n",
            i = i,
            next = i + 1,
            k = 3 + (i as u64 % 7),
        ));
    }
    format!(
        r#"memory 16
func () -> (i64) {{
block 0 () {{
  vz = i64.const 0
  vt = thread.spawn 3 vz vz
  vseed = i64.const 11
  vr = call 1 (vseed)
  vj = thread.join vt
  vtot = i64.add vr vj
  vsl = i64.const {SLOT}
  i64.store vsl vtot
  vout = i32.const {stdout}
  vlen8 = i64.const 8
  vw = call.cap 0 1 (i64, i64) -> (i64) vout (vsl, vlen8)
  return vtot
  }}
}}
func (i64) -> (i64) {{
block 0 (vacc0: i64) {{
  vg = call 2 (vacc0)
  i64.store vg vacc0
  vld = i64.load vg
{chain}  vsum = i64.add vacc400 vld
  return vsum
  }}
}}
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  vas = i32.const {memory}
  voff = i64.const {WIN}
  vlen = i64.const 16384
  vprot = i32.const 3
  vr = call.cap 5 0 (i64, i64, i32) -> (i64) vas (voff, vlen, vprot)
  vprobe = i64.const {PROBE}
  return vprobe
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (vsp: i64, varg: i64) {{
  vseed = i64.const 29
  vr = call 1 (vseed)
  return vr
  }}
}}
export 0 func "_start" 0
"#
    )
}

fn main() {
    let mut args = std::env::args().skip(1);
    let out = args.next().unwrap_or_else(|| "alu.temenc".into());
    // `genfixture <out> [kind]` — `alu` (the default) or `grow_past_window` (#1312).
    let kind = args.next().unwrap_or_else(|| "alu".into());
    let src = match kind.as_str() {
        "alu" => ALU.to_string(),
        "grow_past_window" => grow_past_window(),
        other => panic!("unknown fixture kind {other:?} (want `alu` or `grow_past_window`)"),
    };
    let m = temen_text::parse_module(&src).expect("parse fixture module");
    temen_verify::verify_module(&m).expect("verify fixture module");
    let bytes = temen_encode::encode_module(&m);
    let mut f = std::fs::File::create(&out).expect("create fixture file");
    f.write_all(&bytes).expect("write fixture");
    eprintln!(
        "wrote {} bytes of encoded IR ({kind}) to {out}",
        bytes.len()
    );
}
