//! #1296 slice 3 — a §14 **op-13 child holds a re-granted `Jit`** on the browser's JS-orchestrated
//! op-13 loop (`temen_op13jit_*`), the tier the playground's children actually run on. The driver
//! program re-grants its `"jit"` (the loop's parent host holds one, validator + wasm emitter armed)
//! into the child by name beside `"fs"`; the child resolves it, `compile`s a unit, `invoke`s it over
//! its own window, and `install`s it into its **own** dispatch table.
//!
//! Two seams are pinned, in the order the servicer takes them:
//! - **Emitted child, bounced Jit ops** (`child_jit_ops_persist_across_bounces`): the child is staged
//!   emitted (`OP13JIT_CHILD`); its Jit-op leaf bounces through `temen_onramp_jit_run_call_interp`
//!   (what the emitted `f0`'s `env.call_interp` does). Bounced twice, the second `install` lands in
//!   the **next** slot — the run's dispatch table persists across bounces (a throwaway table would
//!   hand out the same slot again, and the first unit would be unreachable).
//! - **Declined child, interpreter inline** (`declined_child_jit_ops_run_inline`): a `SharedRegion` op
//!   in the module gates the emit off; the same child runs on the interpreter inside `_step` and
//!   reports the same value (the oracle for the emitted seam).

use std::sync::Mutex;

use temen_browser::{
    temen_onramp_jit_run_call_interp, temen_op13jit_close, temen_op13jit_open_named,
    temen_op13jit_result, temen_op13jit_step, OP13JIT_CHILD, OP13JIT_DONE,
};

// The op-13 loop state is process-global (`OP13_JIT`, `JIT_RUN`): serialize the tests.
static LOCK: Mutex<()> = Mutex::new(());

/// Where the child stages its unit blob inside its own 32-KiB window (above the NULL guard + the
/// name scratch at 16392).
const BLOB_OFF: i64 = 20480;

/// The driver (`memory 16`, entry `(inst, module, fs)`): two 16-byte grant records at 17408 —
/// `"fs"` (its third entry arg) and `"jit"` (resolved by name on its own powerbox) — then op 13 into
/// the 32-KiB buddy-half carve at 32768 with `grants_n = 2`, join, return the child's result.
fn driver() -> Vec<u8> {
    let src = r#"memory 16
data 18432 "fs"
data 18448 "jit"
func (i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32) {
  w = i64.const 8589953024
  o = i64.const 17408
  i64.store o w
  hf = i64.extend_i32_u v2
  ohf = i64.const 17416
  i64.store ohf hf
  np = i64.const 18448
  nl = i64.const 3
  hj = self.resolve np nl
  w1 = i64.const 12884920336
  o1 = i64.const 17424
  i64.store o1 w1
  hj64 = i64.extend_i32_u hj
  ohj = i64.const 17432
  i64.store ohj hj64
  vmh = i64.extend_i32_u v1
  vgptr = i64.const 17408
  vgn = i64.const 2
  ventry = i64.const 0
  voff = i64.const 32768
  vsl = i64.const 15
  vq = i64.const 0
  vh = call.cap 6 13 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vmh, vgptr, vgn, ventry, voff, vsl, vq)
  vr = call.cap 6 1 (i32) -> (i64) v0 (vh)
  return vr
  }
}
"#;
    let m = temen_text::parse_module(src).expect("parse driver");
    temen_verify::verify_module(&m).expect("verify driver");
    temen_encode::encode_module(&m)
}

/// The unit the child compiles: `(a, b) -> a + b`, declaring the CHILD's memory (`memory 15`) — the
/// re-granted table's memory-match precondition resolves against the child's own module.
fn unit_blob() -> Vec<u8> {
    let src = "memory 15\nfunc (i32, i32) -> (i32) {\nblock 0 (v0: i32, v1: i32) {\n  v2 = i32.add v0 v1\n  return v2\n  }\n}\n";
    let m = temen_text::parse_module(src).expect("parse unit");
    temen_verify::verify_module(&m).expect("verify unit");
    temen_encode::encode_module(&m)
}

/// The child (`memory 15`, the carve): `f0(sp, as)` (emitted) = `f1()`; `f1` (a cross-tier leaf)
/// resolves `"jit"`, stages the blob, compiles it, invokes `(3, 4)`, installs it, and returns
/// `invoke * 1000 + slot`. `region_op` appends an unreachable §13 `SharedRegion` op so the emit
/// declines and the child runs on the interpreter inline.
fn child(region_op: bool) -> Vec<u8> {
    let blob = unit_blob();
    let mut stores = String::new();
    for (i, chunk) in blob.chunks(8).enumerate() {
        let mut w = [0u8; 8];
        w[..chunk.len()].copy_from_slice(chunk);
        stores.push_str(&format!(
            "  bp{i} = i64.const {}\n  bw{i} = i64.const {}\n  i64.store bp{i} bw{i}\n",
            BLOB_OFF + (i as i64) * 8,
            i64::from_le_bytes(w),
        ));
    }
    let extra = if region_op {
        "func () -> (i64) {\nblock 0 () {\n  vh = i32.const 0\n  va = i64.const 0\n  vr = call.cap 4 0 (i64, i64) -> (i64) vh (va, va)\n  return vr\n  }\n}\n"
    } else {
        ""
    };
    let src = format!(
        r#"memory 15
func (i64, i64) -> (i64) {{
block 0 (vsp: i64, vas: i64) {{
  vr = call 1 ()
  return vr
  }}
}}
func () -> (i64) {{
block 0 () {{
  vname = i64.const 7629162
  vnp = i64.const 16392
  i64.store vnp vname
  vl3 = i64.const 3
  hj = self.resolve vnp vl3
{stores}  vb = i64.const {BLOB_OFF}
  vl = i64.const {}
  vc = call.cap 11 0 (i64, i64) -> (i64) hj (vb, vl)
  va = i32.const 3
  vb4 = i32.const 4
  vr = call.cap 11 1 (i64, i32, i32) -> (i32) hj (vc, va, vb4)
  vslot = call.cap 11 3 (i64) -> (i64) hj (vc)
  vr64 = i64.extend_i32_s vr
  vk = i64.const 1000
  vm = i64.mul vr64 vk
  vsum = i64.add vm vslot
  return vsum
  }}
}}
{extra}"#,
        blob.len()
    );
    let m = temen_text::parse_module(&src).expect("parse child");
    temen_verify::verify_module(&m).expect("verify child");
    temen_encode::encode_module(&m)
}

fn open(child: &[u8]) {
    let d = driver();
    // SAFETY: live byte slices for the duration of the call.
    let st = unsafe { temen_op13jit_open_named(d.as_ptr(), d.len(), child.as_ptr(), child.len()) };
    assert_eq!(st, 0, "op-13 loop opens over the jit-granting driver");
}

/// Split a child result `invoke * 1000 + slot`.
fn parts(v: i64) -> (i64, i64) {
    (v / 1000, v % 1000)
}

#[test]
fn child_jit_ops_persist_across_bounces() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    open(&child(false));
    assert_eq!(
        temen_op13jit_step(),
        OP13JIT_CHILD,
        "the child emits and is staged for the JS driver"
    );
    // Service the leaf's bounce the way the emitted `f0` would (`env.call_interp(1, [])`): the Jit
    // ops run on the interpreter against the child's re-granted table over the child's carve.
    let mut scratch = [0u8; 16];
    assert_eq!(
        temen_onramp_jit_run_call_interp(1, scratch.as_mut_ptr()),
        0,
        "the leaf compiles, invokes and installs over the marshaled host + carve"
    );
    let first = i64::from_le_bytes(scratch[..8].try_into().unwrap());
    // The unit answered 3 + 4; the install landed in a padding slot of the child's own table (past
    // its functions — on this tier the emit's outlined leaf wrappers count among them, so the exact
    // index is the staged program's, not the source module's).
    let (inv, slot) = parts(first);
    assert_eq!(inv, 7, "invoke 3+4 = 7 (got {first})");
    assert!(
        slot >= 2,
        "install lands past the child's functions (got {first})"
    );
    // Bounce again: the run's dispatch table persisted, so the second install takes the NEXT slot
    // (a throwaway per-bounce table would hand out slot 2 again — and lose the first unit).
    let mut scratch = [0u8; 16];
    assert_eq!(temen_onramp_jit_run_call_interp(1, scratch.as_mut_ptr()), 0);
    let second = i64::from_le_bytes(scratch[..8].try_into().unwrap());
    assert_eq!(
        second,
        first + 1,
        "the install table persists across cross-tier bounces"
    );
    temen_op13jit_close();
}

#[test]
fn declined_child_jit_ops_run_inline() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    open(&child(true));
    assert_eq!(
        temen_op13jit_step(),
        OP13JIT_DONE,
        "the declined child runs on the interpreter inline and the driver returns"
    );
    // This variant has three functions (f0, f1, the unreachable region-op gate) and a 16-slot table
    // (the driver's `Jit` reservation, carried by the grant), so the interpreter's first install is
    // slot 3 — the padding starts right past the program's functions, as on every engine.
    assert_eq!(
        temen_op13jit_result(),
        7003,
        "the interpreter-inline child: invoke 3+4 = 7, install at slot 3"
    );
    temen_op13jit_close();
}
