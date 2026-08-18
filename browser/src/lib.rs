//! SVM **bytecode interpreter as a wasm guest** — the browser entry point (see `BROWSER.md`).
//!
//! Exports for a wasm host (browser / any runtime):
//!   * [`run_guest`] — a self-contained, no-import smoke probe (an embedded compute kernel), used by
//!     the wasm32 anchors in `run.mjs`.
//!   * [`svm_alloc`]/[`svm_dealloc`] — the host allocates a buffer in linear memory (no fixed cap),
//!     writes an **encoded SVM IR module** (the `svm-encode` binary form) into it, and frees it
//!     after the run.
//!   * [`svm_run`] — the production shape: `svm_run(ptr, len, arg)` decodes the module at
//!     `[ptr, len)`, runs function 0 on the **bytecode engine** with a **deny-all `Host`**
//!     (compute-only), and returns its first `i64` result. **Fail-closed:** a module the engine
//!     can't compile yields `STATUS_UNSUPPORTED` rather than any tree-walker fallback.
//!   * [`svm_run_pb`] — the **powerbox**: streams/clock/exit, I/O marshalled through allocations.
//!     `svm_run_live` (feature `live`) instead binds those to real host imports.
//!
//! Status of the last run is read separately via [`svm_status`] (a single `i64` return can't
//! disambiguate an error from a guest result of the same value).

// Every `#[no_mangle] extern "C"` export here is a wasm-host FFI boundary that, by construction,
// dereferences host-provided pointers (module bytes, the shared window, vCPU handles); each documents
// its host contract in a `SAFETY:` note. That is exactly the pattern `not_unsafe_ptr_arg_deref` warns
// about, so allow it crate-wide for these boundary functions.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::alloc::Layout;

#[cfg(feature = "live")]
use svm_interp::HostProc;
use svm_interp::{bytecode, Host, StreamRole, Trap, Value};

// The `webgpu` capability's host import (browser: `navigator.gpu` via `webgpu_op`). Wasm-only — native
// builds (the Rust reactor tests) have no such import, so the cap is simply not granted there.
#[cfg(target_arch = "wasm32")]
mod webgpu;

// ---- self-contained smoke probe (no host imports) --------------------------------------------

/// In-wasm roundtrip probe: parse → **encode** → **decode** → run, entirely inside the sandbox, so
/// the production `svm-encode` decode path (which `svm_run` relies on) is exercised on whatever
/// target this is built for — incl. wasm64 via `wasmtime --invoke run_roundtrip`. Returns the ALU
/// result for `arg = 1` (`1442695040888963407`), or `i64::MIN` on any failure.
#[no_mangle]
pub extern "C" fn run_roundtrip() -> i64 {
    let Ok(m) = svm_text::parse_module(ALU) else {
        return i64::MIN;
    };
    let bytes = svm_encode::encode_module(&m);
    let Ok(m2) = svm_encode::decode_module(&bytes) else {
        return i64::MIN;
    };
    let mut fuel = u64::MAX;
    match bytecode::compile_and_run(&m2, 0, &[Value::I64(1)], &mut fuel) {
        Some(Ok(vals)) => match vals.first() {
            Some(Value::I64(x)) => *x,
            _ => i64::MIN,
        },
        _ => i64::MIN,
    }
}

/// The §ROI-spike "alu" hash recurrence: loops `n` times mixing an LCG, returns the accumulator.
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

/// Parse the embedded guest, run it on the bytecode engine with arg `n`, return its i64 result.
/// `i64::MIN` is the in-band failure sentinel (parse/compile/trap).
#[no_mangle]
pub extern "C" fn run_guest(n: i64) -> i64 {
    let Ok(m) = svm_text::parse_module(ALU) else {
        return i64::MIN;
    };
    let mut fuel = u64::MAX;
    match bytecode::compile_and_run(&m, 0, &[Value::I64(n)], &mut fuel) {
        Some(Ok(vals)) => match vals.first() {
            Some(Value::I64(x)) => *x,
            _ => i64::MIN,
        },
        _ => i64::MIN,
    }
}

/// A self-contained **concurrency** smoke probe: 8 vCPUs each `atomic.rmw.add` a shared counter
/// 500× on the bytecode engine's cooperative `drive`, returning `4000` on every interleaving.
/// No host imports — usable via `wasmtime --invoke run_threads` to exercise the scheduler on wasm64.
const THREADS: &str = r#"
memory 16
func () -> (i64) {
block 0 () {
  v0 = i64.const 0
  br 1(v0)
}
block 1 (v1: i64) {
  v2 = i64.const 8
  v3 = i64.lt_u v1 v2
  br_if v3 2(v1) 3()
}
block 2 (v4: i64) {
  v5 = i64.const 500
  v6 = thread.spawn 1 v5 v5
  v7 = i64.const 4
  v8 = i64.mul v4 v7
  v9 = i64.const 16
  v10 = i64.add v9 v8
  i32.store v10 v6
  v11 = i64.const 1
  v12 = i64.add v4 v11
  br 1(v12)
}
block 3 () {
  v13 = i64.const 0
  br 4(v13)
}
block 4 (v14: i64) {
  v15 = i64.const 8
  v16 = i64.lt_u v14 v15
  br_if v16 5(v14) 6()
}
block 5 (v17: i64) {
  v18 = i64.const 4
  v19 = i64.mul v17 v18
  v20 = i64.const 16
  v21 = i64.add v20 v19
  v22 = i32.load v21
  v23 = thread.join v22
  v24 = i64.const 1
  v25 = i64.add v17 v24
  br 4(v25)
}
block 6 () {
  v26 = i64.const 0
  v27 = i64.atomic.load v26
  return v27
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, v0: i64) {
  br 1(v0)
}
block 1 (v1: i64) {
  v2 = i64.const 0
  v3 = i64.eq v1 v2
  br_if v3 3() 2(v1)
}
block 2 (v4: i64) {
  v5 = i64.const 0
  v6 = i64.const 1
  v7 = i64.atomic.rmw.add v5 v6
  v8 = i64.const -1
  v9 = i64.add v4 v8
  br 1(v9)
}
block 3 () {
  v10 = i64.const 0
  return v10
  }
}
"#;

/// Run the embedded concurrency probe; returns `4000`, or `i64::MIN` on any failure.
#[no_mangle]
pub extern "C" fn run_threads() -> i64 {
    let Ok(m) = svm_text::parse_module(THREADS) else {
        return i64::MIN;
    };
    let mut fuel = u64::MAX;
    match bytecode::compile_and_run(&m, 0, &[], &mut fuel) {
        Some(Ok(vals)) => match vals.first() {
            Some(Value::I64(x)) => *x,
            _ => i64::MIN,
        },
        _ => i64::MIN,
    }
}

/// A self-contained **fork** smoke probe (FORK.md §9 / OPS_PARITY.md `clone_caller`/`reap`): the
/// `SRC_TWIN` topology — a manager spawns a server + a guest, the guest `fork()`s (explicit-mode
/// `clone_caller(100, 200)`), the twin runs over a **private window** (`Mem::fork_private`) +
/// **duplicated powerbox** (`Host::fork_powerbox`), and the parent `wait`s it (`reap`). Confirms the
/// **browser fork story**: fork rides the portable bytecode cooperative `drive` (the path this crate
/// runs every multi-domain guest on), over primitives already exercised on wasm32 — the wasm-JIT
/// tier-up is orthogonal (a per-Worker compute accelerator; cap/serve/fork ops leaf-fold to the
/// interp). Returns `100` (the original's reply) **iff** both replies (`100` + `200`) reached the
/// shared stdout — i.e. the twin genuinely ran; `i64::MIN` on any failure.
const FORK_TWIN: &str = r#"
memory 18
type 0 func (i64) -> (i64)
type 1 interface { fork: 0, wait: 0 }
export 0 interface "svc" 1 { fork: 2, wait: 3 }
data 300 "svc"
data 310 "o"
func (i32, i32) -> (i64) {
block 0 (v0: i32, vout: i32) {
  vlog = i64.const 12
  vq = i64.const 0
  q1v0 = i64.const 4294967296
  q1v1 = i64.const 131072
  q1v2 = i64.const -4294967284
  q1v3 = i64.const 4294967295
  q1v4 = i64.const 0
  q1a0 = i64.const 1216
  i64.store q1a0 q1v0
  q1a1 = i64.const 1224
  i64.store q1a1 q1v1
  q1a2 = i64.const 1232
  i64.store q1a2 q1v2
  q1a3 = i64.const 1240
  i64.store q1a3 q1v3
  q1a4 = i64.const 1248
  i64.store q1a4 q1v4
  q1a5 = i64.const 1256
  i64.store q1a5 q1v4
  q1a6 = i64.const 1264
  i64.store q1a6 q1v4
  vs = cap.call 6 17 (i64) -> (i32) v0 (q1a0)
  vz0 = i64.const 0
  vcap = cap.call 6 14 (i32, i64) -> (i32) v0 (vs, vz0)
  va0 = i64.const 256
  vnp = i32.const 300
  i32.store va0 vnp
  va1 = i64.const 260
  vnl = i32.const 3
  i32.store va1 vnl
  va2 = i64.const 264
  i32.store va2 vcap
  va3 = i64.const 272
  vnp2 = i32.const 310
  i32.store va3 vnp2
  va4 = i64.const 276
  vnl2 = i32.const 1
  i32.store va4 vnl2
  va5 = i64.const 280
  i32.store va5 vout
  q2v0 = i64.const 17179869184
  q2v1 = i64.const 135168
  q2v2 = i64.const -4294967284
  q2v3 = i64.const 4294967295
  q2v4 = i64.const 0
  q2v5 = i64.const 256
  q2v6 = i64.const 2
  q2a0 = i64.const 1280
  i64.store q2a0 q2v0
  q2a1 = i64.const 1288
  i64.store q2a1 q2v1
  q2a2 = i64.const 1296
  i64.store q2a2 q2v2
  q2a3 = i64.const 1304
  i64.store q2a3 q2v3
  q2a4 = i64.const 1312
  i64.store q2a4 q2v4
  q2a5 = i64.const 1320
  i64.store q2a5 q2v5
  q2a6 = i64.const 1328
  i64.store q2a6 q2v6
  vc = cap.call 6 17 (i64) -> (i32) v0 (q2a0)
  vjc = cap.call 6 1 (i32) -> (i64) v0 (vc)
  return vjc
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  br 1()
  }
block 1 () {
  vz = i32.const 0
  vn = cap.call 4294967295 10 () -> (i64) vz ()
  br 1()
  }
}
func (i64) -> (i64) {
block 0 (vx: i64) {
  vz = i32.const 0
  vro = i64.const 100
  vrt = i64.const 200
  vt = cap.call 4294967295 11 (i64, i64) -> (i64) vz (vro, vrt)
  return vt
  }
}
func (i64) -> (i64) {
block 0 (vpid: i64) {
  vz = i32.const 0
  vt = cap.call 4294967295 12 (i64) -> (i64) vz (vpid)
  return vt
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vsvc = i64.const 6518387
  vzero = i64.const 0
  i64.store vzero vsvc
  voname = i64.const 111
  va8 = i64.const 8
  i64.store va8 voname
  vp0 = i64.const 0
  vl3 = i64.const 3
  vhsvc = cap.self.resolve vp0 vl3
  vp8 = i64.const 8
  vl1 = i64.const 1
  vho = cap.self.resolve vp8 vl1
  br 1(vhsvc, vho)
  }
block 1 (vhsvc: i32, vho: i32) {
  varg = i64.const 7
  vr = cap.call 268435456 0 (i64) -> (i64) vhsvc (varg)
  v200 = i64.const 200
  vistwin = i64.eq vr v200
  br_if vistwin 4(vr, vho) 2(vr, vhsvc, vho)
  }
block 2 (vr: i64, vhsvc: i32, vho: i32) {
  vpid3 = i64.const 3
  vstatus = cap.call 268435456 1 (i64) -> (i64) vhsvc (vpid3)
  veagain = i64.const -11
  viseagain = i64.eq vstatus veagain
  br_if viseagain 2(vr, vhsvc, vho) 3(vr, vstatus, vhsvc, vho)
  }
block 3 (vr: i64, vstatus: i64, vhsvc: i32, vho: i32) {
  vechild = i64.const -10
  visechild = i64.eq vstatus vechild
  br_if visechild 1(vhsvc, vho) 4(vr, vho)
  }
block 4 (vr: i64, vho: i32) {
  vp16 = i64.const 16
  i64.store vp16 vr
  vlen = i64.const 8
  vw = cap.call 0 1 (i64, i64) -> (i64) vho (vp16, vlen)
  return vr
  }
}
"#;

/// Run the embedded fork probe on the browser build; returns `100` (both copies wrote their reply to
/// the shared stdout — the twin genuinely ran), or `i64::MIN` on any failure. `wasmtime --invoke
/// run_fork` exercises the fork substrate on wasm.
#[no_mangle]
pub extern "C" fn run_fork() -> i64 {
    let Ok(m) = svm_text::parse_module(FORK_TWIN) else {
        return i64::MIN;
    };
    let m = std::sync::Arc::new(m);
    let mut host = Host::new();
    host.set_self_module(&m);
    let inst = host.grant_instantiator(0, 1u64 << 18);
    let sink = host.shared_stdout();
    let out_h = host.grant_stream(StreamRole::Out);
    let mut fuel = 40_000_000u64;
    let r = match bytecode::compile_and_run_with_host(
        &m,
        0,
        &[Value::I32(inst), Value::I32(out_h)],
        &mut fuel,
        &mut host,
    ) {
        Some(Ok(vals)) => match vals.first() {
            Some(Value::I64(x)) => *x,
            _ => return i64::MIN,
        },
        _ => return i64::MIN,
    };
    // The original resumed past the fork with reply_orig (100), and the shared sink carries BOTH
    // replies — proof the twin ran over its private window + duplicated powerbox.
    let bytes = sink.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let mut vals: Vec<i64> = bytes
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap_or([0; 8])))
        .collect();
    vals.sort_unstable();
    if r == 100 && vals == [100, 200] {
        100
    } else {
        i64::MIN
    }
}

// ---- production entry: run an encoded guest module -------------------------------------------

/// `svm_run` completed and returned a guest `i64`.
pub const STATUS_OK: i32 = 0;
/// The bytes at the scratch buffer were not a well-formed encoded module.
pub const STATUS_DECODE_ERR: i32 = 1;
/// Fail-closed: the bytecode engine doesn't drive some op the module uses (no tree-walker fallback).
pub const STATUS_UNSUPPORTED: i32 = 2;
/// The guest trapped (masking/confinement violation, fuel exhaustion, explicit trap, …).
pub const STATUS_TRAP: i32 = 3;
/// The guest returned, but not a single `i64` (compute-only v1 only surfaces `i64`).
pub const STATUS_BAD_RESULT: i32 = 4;

/// Most recent status (a `STATUS_*` code), read via [`svm_status`] after any run entry.
static mut LAST_STATUS: i32 = STATUS_OK;

// ---- linear-memory allocator: the host manages I/O buffers of arbitrary size ------------------
//
// Replaces the old fixed scratch buffers. The host calls [`svm_alloc`] to reserve `len` bytes in
// *this module's* linear memory (the Rust allocator grows it as needed — no 1 MiB cap), writes the
// encoded module / stdin there, passes the `(ptr, len)` to a run entry, then [`svm_dealloc`]s it.
// Allocations are plain bytes (alignment 1), so `dealloc` only needs the same `len`.

/// Allocate `len` bytes (alignment 1) in linear memory; returns the pointer (null for `len == 0` or
/// on allocation failure). Pair every non-null result with a [`svm_dealloc`] of the same `len`.
#[no_mangle]
pub extern "C" fn svm_alloc(len: usize) -> *mut u8 {
    match Layout::from_size_align(len, 1) {
        Ok(layout) if len != 0 => unsafe { std::alloc::alloc(layout) },
        _ => core::ptr::null_mut(),
    }
}

/// Free a [`svm_alloc`]ation — `ptr`/`len` must match the original request. No-op for a null `ptr`
/// or `len == 0`. (Do **not** call this on the `svm_stdout_ptr`/`svm_stderr_ptr` buffers: those are
/// cdylib-managed, reclaimed on the next [`svm_run_pb`].)
#[no_mangle]
pub extern "C" fn svm_dealloc(ptr: *mut u8, len: usize) {
    if ptr.is_null() || len == 0 {
        return;
    }
    if let Ok(layout) = Layout::from_size_align(len, 1) {
        unsafe { std::alloc::dealloc(ptr, layout) };
    }
}

/// `1` on a 64-bit (`wasm64`/`memory64`) build, `0` on `wasm32` — so a host harness knows whether
/// the pointer/length ABI values are `i64` (BigInt) or `i32`.
#[no_mangle]
pub extern "C" fn svm_abi_is64() -> i32 {
    (core::mem::size_of::<usize>() == 8) as i32
}

/// Status of the most recent run entry (one of the `STATUS_*` codes).
#[no_mangle]
pub extern "C" fn svm_status() -> i32 {
    // SAFETY: single-threaded wasm; plain `i32` read.
    unsafe { LAST_STATUS }
}

/// Decode the `len` bytes at `ptr` as an SVM IR module, run function 0 on the bytecode engine with
/// `args` and a deny-all `Host`, and return its first `i64` result (`0` on any non-`OK` status —
/// read [`svm_status`] to disambiguate). Sets [`LAST_STATUS`]. Shared by [`svm_run`]/[`svm_run0`].
fn run_at(ptr: *const u8, len: usize, args: &[Value]) -> i64 {
    let set = |s: i32| unsafe { LAST_STATUS = s };
    // SAFETY: the host guarantees `[ptr, ptr+len)` is a live `svm_alloc`ation it just filled.
    let bytes = unsafe { core::slice::from_raw_parts(ptr, len) };
    let m = match svm_encode::decode_module(bytes) {
        Ok(m) => m,
        Err(_) => {
            set(STATUS_DECODE_ERR);
            return 0;
        }
    };
    let mut fuel = u64::MAX;
    let mut host = svm_interp::Host::new(); // deny-all powerbox (compute-only)
    match bytecode::compile_and_run_with_host(&m, 0, args, &mut fuel, &mut host) {
        None => {
            set(STATUS_UNSUPPORTED);
            0
        }
        Some(Err(_)) => {
            set(STATUS_TRAP);
            0
        }
        Some(Ok(vals)) => match vals.first() {
            Some(Value::I64(x)) => {
                set(STATUS_OK);
                *x
            }
            _ => {
                set(STATUS_BAD_RESULT);
                0
            }
        },
    }
}

/// Run the encoded module at `[ptr, ptr+len)` passing a single `i64` argument (the common shape).
#[no_mangle]
pub extern "C" fn svm_run(ptr: *const u8, len: usize, arg: i64) -> i64 {
    run_at(ptr, len, &[Value::I64(arg)])
}

/// Run the encoded module at `[ptr, ptr+len)` with **no** arguments — e.g. the `() -> (i64)` thread
/// kernels that spawn/join cooperatively on the engine's `drive`.
#[no_mangle]
pub extern "C" fn svm_run0(ptr: *const u8, len: usize) -> i64 {
    run_at(ptr, len, &[])
}

/// `verify_module` rejected the decoded module ([`svm_prep_bench`]).
pub const STATUS_VERIFY_ERR: i32 = 6;

/// **Benchmark entry: the safe module-load path a browser must run before it can execute a guest.**
/// Decode the module at `[ptr, ptr+len)`, `verify_module` it (the escape-freedom TCB gate — never
/// skippable, however trusted the producer), and `bytecode::compile_module` it (the interpreter's
/// per-module cold cost). Sets [`svm_status`]; returns the function count (`0` on any error). The host
/// times this call to measure the **module-prep tax inside the wasm sandbox** — decode + verify +
/// compile of a *pre-translated, pre-resolved* `.svmb` — the wasm counterpart to the native
/// `prep_svmb` example. This is the one-time cost a fast-loading demo pays per page load (translation
/// is done at build time); its ratio to the native example is the sandbox tax on loading (see
/// `BOOTSPEED.md`). No powerbox run happens here. Driven by `browser/bench_prep.mjs`.
#[no_mangle]
pub extern "C" fn svm_prep_bench(ptr: *const u8, len: usize) -> i64 {
    let set = |s: i32| unsafe { LAST_STATUS = s };
    // SAFETY: the host guarantees `[ptr, ptr+len)` is a live `svm_alloc`ation it just filled.
    let bytes = unsafe { core::slice::from_raw_parts(ptr, len) };
    let m = match svm_encode::decode_module(bytes) {
        Ok(m) => m,
        Err(_) => {
            set(STATUS_DECODE_ERR);
            return 0;
        }
    };
    if svm_verify::verify_module(&m).is_err() {
        set(STATUS_VERIFY_ERR);
        return 0;
    }
    if bytecode::compile_module(&m.funcs).is_none() {
        set(STATUS_UNSUPPORTED);
        return 0;
    }
    set(STATUS_OK);
    m.funcs.len() as i64
}

/// **Benchmark entry: run an arbitrary kernel function under the LLVM-frontend ABI.** Decode the
/// module at `[mod_ptr, mod_len)`, run function `func` on the bytecode engine with the frontend's
/// `(sp, n)` calling convention — `(sp, n)` for a ≥2-param entry, `(n)` for a 1-param one — under a
/// deny-all `Host`, and return its first result widened to `i64` (`0` on any non-`OK` status; read
/// [`svm_status`]). Each argument is coerced to its declared `ValType` so a 32-bit `n` param (the
/// `cross_engine` kernels) and a 64-bit one (the `embench` kernels, `long n`) both run correctly.
///
/// This is the seam the cross-engine benchmark uses to time the **bytecode engine running inside
/// wasm** (`crates/svm-llvm/examples/cross_engine.rs`'s `svm-bytecode-wasm` row, driven via
/// `browser/bench.mjs`) on the *same* LLVM-frontend IR the native `svm-bytecode` row runs — isolating
/// the cost of the wasm sandbox over the interpreter. `svm_run`/`svm_run0` only reach function 0 with
/// a fixed arity, so a dedicated entry is needed to drive a kernel exported at an arbitrary index.
#[no_mangle]
pub extern "C" fn svm_run_bench(
    mod_ptr: *const u8,
    mod_len: usize,
    func: u32,
    sp: i64,
    n: i64,
) -> i64 {
    let set = |s: i32| unsafe { LAST_STATUS = s };
    // SAFETY: the host guarantees `[mod_ptr, mod_len)` is a live `svm_alloc`ation it just filled.
    let bytes = unsafe { core::slice::from_raw_parts(mod_ptr, mod_len) };
    let m = match svm_encode::decode_module(bytes) {
        Ok(m) => m,
        Err(_) => {
            set(STATUS_DECODE_ERR);
            return 0;
        }
    };
    let Some(f) = m.funcs.get(func as usize) else {
        set(STATUS_UNSUPPORTED);
        return 0;
    };
    // Frontend ABI: the entry is `func(sp, n)`; a 1-param entry (e.g. a hand-written text kernel)
    // takes just `n`. Coerce each value to the declared param type (i32 vs i64 `n`); pad any extra
    // params with 0 of their type.
    let supplied: &[i64] = if f.params.len() >= 2 { &[sp, n] } else { &[n] };
    let args: Vec<Value> = f
        .params
        .iter()
        .enumerate()
        .map(|(i, ty)| {
            let raw = supplied.get(i).copied().unwrap_or(0);
            match ty {
                svm_ir::ValType::I32 => Value::I32(raw as i32),
                _ => Value::I64(raw),
            }
        })
        .collect();
    let mut fuel = u64::MAX;
    let mut host = Host::new(); // deny-all powerbox (compute-only)
    match bytecode::compile_and_run_with_host(&m, func, &args, &mut fuel, &mut host) {
        None => {
            set(STATUS_UNSUPPORTED);
            0
        }
        Some(Err(_)) => {
            set(STATUS_TRAP);
            0
        }
        Some(Ok(vals)) => match vals.first() {
            Some(Value::I64(x)) => {
                set(STATUS_OK);
                *x
            }
            Some(Value::I32(x)) => {
                set(STATUS_OK);
                *x as i64
            }
            _ => {
                set(STATUS_BAD_RESULT);
                0
            }
        },
    }
}

// ---- shared-memory window: run the engine over a caller-owned region of *this* linear memory ----
//
// THREADS.md step 4. `svm_run` runs over a window the engine backs internally; `svm_run_shared` runs
// over a window the **host** carves out of this module's linear memory (`[win_ptr, win_size)`, via
// `svm_alloc`). Built as a wasm threads module (shared memory + `+atomics`), that linear memory is
// the host's `SharedArrayBuffer`, so the window lives in shared memory — the substrate the parallel
// mode's per-vCPU Workers will all execute over. Today still cooperative (one thread); the only
// change from `svm_run` is *where the guest window lives*. Stateless (no `static mut`), so two
// Workers running it over **disjoint** windows don't race on engine ABI globals.

/// Decode the module at `[mod_ptr, mod_len)` and run function 0 over the guest window
/// `[win_ptr, win_ptr+win_size)` of this module's linear memory (a `Region::shared`; `win_size` must
/// cover the module's `memory` size). Returns the guest's `i64` result, or `i64::MIN` on
/// decode/unsupported/trap/non-`i64`. The host reads the guest's memory effects directly from the
/// window region afterward.
#[no_mangle]
pub extern "C" fn svm_run_shared(
    mod_ptr: *const u8,
    mod_len: usize,
    win_ptr: *mut u8,
    win_size: usize,
    arg: i64,
) -> i64 {
    // SAFETY: the host guarantees `[mod_ptr, mod_len)` is a live `svm_alloc`ation it just filled.
    let bytes = unsafe { core::slice::from_raw_parts(mod_ptr, mod_len) };
    let Ok(m) = svm_encode::decode_module(bytes) else {
        return i64::MIN;
    };
    // SAFETY: the host guarantees `[win_ptr, win_size)` is a live `svm_alloc`ed region of this linear
    // memory used solely as this guest window for the call. The `unsafe` borrow lives here in the
    // embedder; the engine stays `#![forbid(unsafe_code)]` and just takes the `Arc<Region>`.
    let back = std::sync::Arc::new(unsafe { svm_interp::Region::shared(win_ptr, win_size as u64) });
    let arity = m.funcs.first().map_or(0, |f| f.params.len());
    let args: &[Value] = if arity >= 1 { &[Value::I64(arg)] } else { &[] };
    let mut fuel = u64::MAX;
    match bytecode::compile_and_run_capture_over(&m, 0, args, &mut fuel, &[], back) {
        Some((Ok(vals), _)) => match vals.first() {
            Some(Value::I64(x)) => *x,
            Some(Value::I32(x)) => *x as i64,
            _ => i64::MIN,
        },
        _ => i64::MIN,
    }
}

// ==== THREADS.md step 4c-wasm — the host-orchestrated parallel driver =============================
//
// wasm32 has no `thread::spawn`, so one guest's `thread.spawn`ed vCPUs are distributed across **Web
// Workers** by the JS host: each Worker runs **one** vCPU via the engine's resumable `Vcpu` API
// (`svm_par_run` → an event the host services → deliver the result → run again) over the **one** shared
// linear-memory window. The host services the events with real cross-Worker primitives: `thread.spawn`
// → start a Worker, `thread.join` → `Atomics.wait` on the child's completion slot, `memory.wait`/
// `notify` → `Atomics.wait`/`notify` on the futex word — so this is genuinely parallel, the native
// `bytecode_vcpu_orchestration.rs` test being its differential oracle.
//
// `VcpuProgram` is compiled once and shared **read-only** across Workers by pointer (it is `Sync`, and
// under `--shared-memory` every Worker's instance sees the same linear memory, so a `Box::leak`ed
// program built by one Worker is valid in all). Each `Vcpu` is `'static` here: the program outlives the
// run (never freed), so the borrow is sound — the `unsafe` of asserting that lives in this embedder.

/// Allocate `len` bytes **16-aligned** (so windows / futex words / completion slots are naturally
/// aligned for `Atomics` / the engine's hardware atomics, which `svm_alloc`'s align-1 does not
/// guarantee). Leaked for the run (the parallel demo never frees; the process exits). Null on `len==0`.
#[no_mangle]
pub extern "C" fn svm_par_alloc(len: usize) -> *mut u8 {
    match Layout::from_size_align(len, 16) {
        Ok(layout) if len != 0 => unsafe { std::alloc::alloc_zeroed(layout) },
        _ => core::ptr::null_mut(),
    }
}

/// Event codes returned by [`svm_par_run`] — the host switches on these (operands via `svm_par_ev_*`).
pub const PAR_DONE: i32 = 0;
pub const PAR_TRAP: i32 = 1;
pub const PAR_SPAWN: i32 = 2;
pub const PAR_JOIN: i32 = 3;
pub const PAR_WAIT: i32 = 4;
pub const PAR_NOTIFY: i32 = 5;
pub const PAR_INSTANTIATE: i32 = 6;
/// wasm-JIT tier-up (browser wasm-JIT threads slice): the vCPU reached a `Call` to a JIT-eligible
/// function. `svm_par_ev_a` = the func index; `svm_par_ev_b` = the window's committed extent, which
/// the Worker MUST write to the emitted module's `"mapped"` global before the call (#717 host sync —
/// over today's fully-mapped par window it equals the emit-time default, so the write is idempotent;
/// over a grown window it is what keeps the emitted bounds check in lockstep with the interpreter).
/// `svm_par_tierup_argv_ptr`/`_len` give the marshalled i64 args. The Worker runs the emitted
/// `f{func}` and calls `svm_par_deliver_tierup`/`_trap`.
pub const PAR_TIERUP: i32 = 7;
/// §22 guest-JIT **real codegen** (BROWSER.md § "wasm-JIT tier", slice 5): a guest's `Jit.invoke`
/// surfaces here (codegen mode on — [`svm_par_powerbox_jit_codegen`]) so the Worker runs the
/// submitted unit on **emitted wasm** (`svm_par_jit_unit_wasm_ptr`/`_len` — one immutable module per
/// run) instead of the interpreter. `svm_par_jit_code` keys the Worker's per-unit instance cache;
/// `svm_par_jit_argv_ptr`/`_len` give the args as i64 slots, `svm_par_jit_param_types_ptr` their wasm
/// types (i32/i64) so the Worker marshals each to a JS `Number`/`BigInt`. `svm_par_ev_b` = the
/// window's committed extent, which the Worker MUST write to the unit instance's `"mapped"` global
/// before the call (#717 host sync, same contract as [`PAR_TIERUP`] — an invoke whose window state
/// the scalar cannot represent never surfaces here; it is serviced on the interpreter instead). The
/// Worker runs the emitted `f{entry}(win, env, …args)` and calls `svm_par_deliver_jit_invoke`/`_trap`.
pub const PAR_JIT_INVOKE: i32 = 8;

/// A boxed resumable vCPU plus the operands of its last [`svm_par_run`] event (flattened to four
/// `i64`s the host reads via [`svm_par_ev_a`]–[`svm_par_ev_d`]).
pub struct ParVcpu {
    inner: bytecode::Vcpu<'static>,
    a: i64,
    b: i64,
    c: i64,
    d: i64,
    /// The marshalled arguments of a pending [`PAR_TIERUP`] event (raw i64 slots) — read by the
    /// Worker via [`svm_par_tierup_argv_ptr`]/[`svm_par_tierup_argv_len`] to call the emitted region.
    tierup_argv: Vec<i64>,
    /// The marshalled arguments of a pending [`PAR_JIT_INVOKE`] event (raw i64 slots) — read by the
    /// Worker via [`svm_par_jit_argv_ptr`]/[`svm_par_jit_argv_len`] to call the emitted §22 unit.
    jit_argv: Vec<i64>,
    /// The code handle of a pending [`PAR_JIT_INVOKE`] (the Worker caches one emitted instance per unit).
    jit_code: i32,
    /// Per-arg / per-result **scalar type codes** of a pending [`PAR_JIT_INVOKE`] (`0` = i32, `1` =
    /// i64, `2` = f32, `3` = f64) so the Worker marshals each i64 slot to/from the wasm type the
    /// emitted `f{entry}` uses: an i32 arg is a JS `Number`, an i64 a `BigInt`, a float the *value*
    /// the slot's bits reinterpret to. Read via [`svm_par_jit_param_types_ptr`] /
    /// [`svm_par_jit_result_types_ptr`] — a §22 unit need not be all-i64.
    jit_param_types: Vec<u8>,
    jit_result_types: Vec<u8>,
    /// The emitted wasm of a pending [`PAR_JIT_INVOKE`]'s **runtime-compiled** unit (the shared-host
    /// path, [`svm_par_powerbox_jit_runtime`]): the JS host reads its bytes via
    /// [`svm_par_jit_code_wasm_ptr`]/[`svm_par_jit_code_wasm_len`] to instantiate the unit once,
    /// caching the instance by [`jit_code`](ParVcpu::jit_code). `None` for the fixed-unit codegen path
    /// (that reads the run-wide [`JIT_UNIT_WASM`] stash). The `Arc` keeps the bytes alive for the read.
    jit_wasm: Option<std::sync::Arc<[u8]>>,
    /// #750 paged runs: the page-state table for a pending [`PAR_TIERUP`], rebuilt from the live
    /// page map at each event ([`bytecode::build_pagestate_table`]). Its bytes live in this
    /// module's linear memory, so [`svm_par_tierup_pagestate_ptr`] IS the address the Worker
    /// writes to the emitted module's `"pagestate"` global. Empty on unpaged runs.
    pagestate: Vec<u8>,
}

/// SVM scalar `ValType` → the Worker's marshalling type code (`0` = i32, `1` = i64, `2` = f32, `3` =
/// f64). `None` for `v128` (the Worker has no lane marshalling — such a unit stays on the interp).
fn scalar_type_code(t: svm_ir::ValType) -> Option<u8> {
    match t {
        svm_ir::ValType::I32 => Some(0),
        svm_ir::ValType::I64 => Some(1),
        svm_ir::ValType::F32 => Some(2),
        svm_ir::ValType::F64 => Some(3),
        _ => None,
    }
}

/// Box a freshly-built vCPU as a [`ParVcpu`] (event operands zeroed, no pending tier-up args).
fn par_box(inner: bytecode::Vcpu<'static>) -> *mut ParVcpu {
    Box::into_raw(Box::new(ParVcpu {
        inner,
        a: 0,
        b: 0,
        c: 0,
        d: 0,
        tierup_argv: Vec::new(),
        jit_argv: Vec::new(),
        jit_code: 0,
        jit_param_types: Vec::new(),
        jit_result_types: Vec::new(),
        jit_wasm: None,
        pagestate: Vec::new(),
    }))
}

/// Attach the tier-up bitmap (if published) — only the **plain compute paths** (root / `thread.spawn`
/// child over the primary module + window) tier up; §14/§22 orchestration roots and confined children
/// run different modules/windows, so they stay on the interpreter.
fn with_tierup(inner: bytecode::Vcpu<'static>) -> bytecode::Vcpu<'static> {
    match par_jit_eligible() {
        Some(e) => {
            let inner = inner.with_jit_eligible(e);
            // #750: a paged eligible set needs the dispatch to skip the scalar decline (the
            // per-event page-state table carries the fidelity the scalar cannot).
            if par_jit_paged() {
                inner.with_jit_page_checked()
            } else {
                inner
            }
        }
        None => inner,
    }
}

/// The JIT tier-up eligibility bitmap for this instance's guest (per-Worker: each computes its own
/// from the module bytes via [`svm_par_enable_jit`], since an `Arc` can't cross Worker instances).
static mut PAR_JIT_ELIGIBLE: Option<std::sync::Arc<[bool]>> = None;

/// Clone the published tier-up bitmap, if any.
fn par_jit_eligible() -> Option<std::sync::Arc<[bool]>> {
    // SAFETY: single-threaded per instance (the page, or one Worker) — same access model as `WASMJIT_MOD`.
    unsafe { (*core::ptr::addr_of!(PAR_JIT_ELIGIBLE)).clone() }
}

/// #750: whether this instance's tier-up module was emitted **paged**
/// ([`svm_par_enable_jit_paged`]) — the vCPUs then skip the scalar decline and every `PAR_TIERUP`
/// carries a freshly built page-state table (operand `b` = its coverage).
static mut PAR_JIT_PAGED: bool = false;

fn par_jit_paged() -> bool {
    // SAFETY: single-threaded per instance — same access model as `PAR_JIT_ELIGIBLE`.
    unsafe { *core::ptr::addr_of!(PAR_JIT_PAGED) }
}

// ==== I22 fix: emit each per-Worker codegen unit exactly ONCE per run ============================
// `svm_par_enable_jit` / `_jit_codegen` / `_inst_codegen` each emit wasm and `stash()` it into a
// `static mut`. JS instantiates every Worker against ONE shared linear memory, so those statics are a
// SINGLE shared copy — NOT "per instance" as the older SAFETY comments claimed. So N Workers each
// calling `enable_*` in their own setup, concurrently, raced on `stash()`'s `dealloc(old_ptr)`: two
// Workers read the same `old_ptr` and both freed it → double-free / use-after-free → heap corruption →
// a later `memory access out of bounds` or a panic=abort `unreachable` (ISSUES.md I22).
//
// Fix: emit exactly once per run. Every run's page-side powerbox publisher bumps `PAR_RUN_GEN` (always
// single-threaded — the previous run's Workers are terminated before the next run publishes). Each
// `enable_*` runs its emit under `CODEGEN_LOCK` and only if it hasn't already run for the current
// generation; later Workers skip the emit and reuse the shared stash (identical bytes either way). The
// stash is thus written once per run and never freed mid-run, so the Workers' reads of the emitted
// bytes are stable. A SPIN-lock (not a `Mutex`) so the page's own `enable_*` call — which happens on
// the main thread inside `svm_par_powerbox_jit_codegen` — can never hit a forbidden `Atomics.wait`; it
// is always uncontended (no Worker is alive yet), so it acquires without spinning.
//
// The emit runs while the lock is held, so under `panic = "abort"` a compile panic would leave the
// lock stuck and the run's other Workers spinning — but (i) the emitters are pure and only ever see
// fixed corpus/constant modules that compile cleanly, and (ii) the I22 retry reloads the page with a
// *fresh* `WebAssembly.Memory`, which re-initialises `CODEGEN_LOCK` to `false`, so even that case
// self-heals. (The pre-fix double-free is what actually produced the panics; this removes it.)
static PAR_RUN_GEN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
static CODEGEN_LOCK: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Bump the run generation. Call once, page-side, at the start of every powerbox publisher.
fn par_run_gen_bump() {
    PAR_RUN_GEN.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
}

/// RAII spin-lock over the codegen stashes: `Acquire` on lock, `Release` on unlock, so a stash the
/// first Worker writes under the lock is visible to a later Worker that acquires it.
struct CodegenGuard;
impl CodegenGuard {
    fn acquire() -> Self {
        use std::sync::atomic::Ordering;
        while CODEGEN_LOCK
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            core::hint::spin_loop();
        }
        CodegenGuard
    }
    /// Current run generation (read while the guard is held).
    fn generation(&self) -> u32 {
        PAR_RUN_GEN.load(std::sync::atomic::Ordering::Relaxed)
    }
}
impl Drop for CodegenGuard {
    fn drop(&mut self) {
        CODEGEN_LOCK.store(false, std::sync::atomic::Ordering::Release);
    }
}

// Per-stash "already emitted for this run generation" + the result to hand back to a Worker that
// arrives after the first has emitted (init `u32::MAX` = "no run yet"; every real gen is < that).
static TIERUP_DONE_GEN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(u32::MAX);
static TIERUP_RESULT: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);
static JIT_CG_DONE_GEN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(u32::MAX);
static JIT_CG_RESULT: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);
static INST_CG_DONE_GEN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(u32::MAX);
static INST_CG_RESULT: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);

/// Enable wasm-JIT **tier-up** for the module at `[mod_ptr, mod_len)` (`BROWSER.md` § "wasm-JIT
/// tier", per-Worker JIT): emit the tier-up module and compute which functions the interpreter
/// should surface as [`PAR_TIERUP`] (the browser then runs the emitted `f{func}` on the Worker
/// instead of interpreting). Unlike the whole-module `svm_wasmjit_compile`, this does **not** need
/// the guest's func 0 to be JITtable — the guest keeps running on the resumable interpreter (which
/// drives `thread.spawn`/`join`, atomics, `memory.wait`), and only a direct `Call` to an emitted
/// pure region tiers up. So a compute leaf reachable **only** through `thread.spawn` still tiers up,
/// which is the whole point of the threads tier ([`svm_wasm_jit::compile_jit`] with
/// [`svm_wasm_jit::Shape::Threaded`]).
///
/// A function is eligible iff it is **emitted** (in-subset, all its calls route) **and** has an
/// **all-i64** signature — so the Worker passes every arg / reads every result as a plain `BigInt`
/// i64 slot with no per-param type info (which the emitted `WebAssembly.Module` doesn't expose to
/// JS). Non-i64 scalar params (i32, floats) are a later refinement. On success this stashes the
/// emitted wasm (read via [`svm_wasmjit_ptr`]/[`svm_wasmjit_len`]) and the decoded module (for the
/// cross-tier [`svm_wasmjit_call_interp`]), so the Worker needs only this one call — no separate
/// `svm_wasmjit_compile`. Returns `1` when at least one function tier-ups, else `0` (everything
/// interprets). Call on **every** instance (page + each Worker) before building vCPUs, same bytes.
#[no_mangle]
pub extern "C" fn svm_par_enable_jit(mod_ptr: *const u8, mod_len: usize) -> i32 {
    use std::sync::atomic::Ordering;
    par_install_panic_capture(); // I22: capture a setup-time engine panic's FILE:LINE (not a bare `unreachable`)
                                 // I22: emit once per run under CODEGEN_LOCK; a later Worker reuses the shared stash (see the
                                 // CodegenGuard note above). The shared statics `WASMJIT`/`WASMJIT_MOD`/`PAR_JIT_ELIGIBLE` are a
                                 // single copy across Workers, so a concurrent re-emit double-freed the stash.
    let guard = CodegenGuard::acquire();
    let generation = guard.generation();
    if TIERUP_DONE_GEN.load(Ordering::Relaxed) == generation {
        return TIERUP_RESULT.load(Ordering::Relaxed);
    }
    let result = (|| {
        // SAFETY: the host guarantees `[mod_ptr, mod_len)` is a live `svm_alloc`ation it just filled.
        let bytes = unsafe { core::slice::from_raw_parts(mod_ptr, mod_len) };
        let Ok(m) = svm_encode::decode_module(bytes) else {
            return 0;
        };
        // Emit the tier-up module against the shared linear memory (the browser threads build) and take
        // its per-function emit set. The `Threaded` shape is interpreter-driven by construction (no
        // single top-level frame — vCPUs enter via `thread.spawn`), so `compile_jit` picks tier-up. `Err`
        // only if the assembler itself rejects the set — treat as "no tier-up" (fail-closed: the guest
        // keeps interpreting).
        let Ok(svm_wasm_jit::Artifact {
            wasm,
            emitted: emit,
            ..
        }) = svm_wasm_jit::compile_jit(&m, svm_wasm_jit::Shape::Threaded, true)
        else {
            return 0;
        };
        let all_i64 = |ts: &[svm_ir::ValType]| ts.iter().all(|t| *t == svm_ir::ValType::I64);
        let eligible: Vec<bool> = m
            .funcs
            .iter()
            .enumerate()
            .map(|(i, f)| emit[i] && all_i64(&f.params) && all_i64(&f.results))
            .collect();
        if !eligible.iter().any(|&e| e) {
            return 0; // nothing safely tier-up-able → leave everything on the interpreter
        }
        // SAFETY: written once per run while CODEGEN_LOCK is held (this closure runs only on the
        // first Worker of the run); Workers then read it stable for the run.
        unsafe {
            stash(&mut *core::ptr::addr_of_mut!(WASMJIT), wasm);
            *core::ptr::addr_of_mut!(WASMJIT_MOD) = Some(m);
            *core::ptr::addr_of_mut!(PAR_JIT_ELIGIBLE) = Some(std::sync::Arc::from(eligible));
            *core::ptr::addr_of_mut!(PAR_JIT_PAGED) = false;
        }
        1
    })();
    TIERUP_RESULT.store(result, Ordering::Relaxed);
    TIERUP_DONE_GEN.store(generation, Ordering::Relaxed);
    result
}

/// [`svm_par_enable_jit`], but the tier-up module is emitted **paged** (#750,
/// `compile_module_tierup_paged` with this instance's software page size): `unmap`/`protect`
/// guests keep their pure leaves eligible, and every emitted access consults the per-event
/// page-state table (see [`svm_par_tierup_pagestate_ptr`]; the Worker writes its base to the
/// emitted `"pagestate"` global and event operand `b` — the table's coverage — to `"mapped"`).
/// A run calls exactly ONE of the two enable entries (they share the once-per-run stash).
/// Same contract otherwise: call on every instance before building vCPUs, same bytes.
#[no_mangle]
pub extern "C" fn svm_par_enable_jit_paged(mod_ptr: *const u8, mod_len: usize) -> i32 {
    use std::sync::atomic::Ordering;
    par_install_panic_capture();
    let guard = CodegenGuard::acquire();
    let generation = guard.generation();
    if TIERUP_DONE_GEN.load(Ordering::Relaxed) == generation {
        return TIERUP_RESULT.load(Ordering::Relaxed);
    }
    let result = (|| {
        // SAFETY: the host guarantees `[mod_ptr, mod_len)` is a live `svm_alloc`ation it just filled.
        let bytes = unsafe { core::slice::from_raw_parts(mod_ptr, mod_len) };
        let Ok(m) = svm_encode::decode_module(bytes) else {
            return 0;
        };
        let page_log2 = svm_interp::host_page_size().trailing_zeros() as u8;
        let Ok((wasm, emit)) = svm_wasm_jit::compile_module_tierup_paged(&m, true, page_log2)
        else {
            return 0;
        };
        let all_i64 = |ts: &[svm_ir::ValType]| ts.iter().all(|t| *t == svm_ir::ValType::I64);
        let eligible: Vec<bool> = m
            .funcs
            .iter()
            .enumerate()
            .map(|(i, f)| emit[i] && all_i64(&f.params) && all_i64(&f.results))
            .collect();
        if !eligible.iter().any(|&e| e) {
            return 0; // nothing safely tier-up-able → leave everything on the interpreter
        }
        // SAFETY: written once per run while CODEGEN_LOCK is held; Workers read it stable.
        unsafe {
            stash(&mut *core::ptr::addr_of_mut!(WASMJIT), wasm);
            *core::ptr::addr_of_mut!(WASMJIT_MOD) = Some(m);
            *core::ptr::addr_of_mut!(PAR_JIT_ELIGIBLE) = Some(std::sync::Arc::from(eligible));
            *core::ptr::addr_of_mut!(PAR_JIT_PAGED) = true;
        }
        1
    })();
    TIERUP_RESULT.store(result, Ordering::Relaxed);
    TIERUP_DONE_GEN.store(generation, Ordering::Relaxed);
    result
}

fn first_i64(vals: &[Value]) -> i64 {
    match vals.first() {
        Some(Value::I64(x)) => *x,
        Some(Value::I32(x)) => *x as i64,
        _ => 0,
    }
}

/// Compile the module at `[mod_ptr, mod_len)` into a shareable [`bytecode::VcpuProgram`], returned as a
/// leaked pointer (lives for the run; shared read-only across Workers). Null on decode/unsupported.
#[no_mangle]
pub extern "C" fn svm_par_compile(
    mod_ptr: *const u8,
    mod_len: usize,
) -> *mut bytecode::VcpuProgram {
    // SAFETY: the host guarantees `[mod_ptr, mod_len)` is a live `svm_alloc`ation it just filled.
    let bytes = unsafe { core::slice::from_raw_parts(mod_ptr, mod_len) };
    let Ok(m) = svm_encode::decode_module(bytes) else {
        return core::ptr::null_mut();
    };
    match bytecode::VcpuProgram::compile(&m) {
        Some(p) => Box::into_raw(Box::new(p)),
        None => core::ptr::null_mut(),
    }
}

/// Borrow a `*mut VcpuProgram` as `&'static` (the program outlives the run). SAFETY: the host keeps it
/// alive for the whole run and never frees it before the last `Vcpu` over it.
unsafe fn prog_ref(prog: *mut bytecode::VcpuProgram) -> &'static bytecode::VcpuProgram {
    &*prog
}

// ---- §22 guest-JIT across Workers: a Rust-side shared powerbox (THREADS.md 4c-domain C2) ---------
// The powerbox (a `Host` with the `Jit` cap + the host-compiled unit) is built once and **leaked** into
// the shared linear memory; its pointer is published in a process-wide `static` which — under
// `--shared-memory` — lives in that shared memory, so every Worker's instance reads the same value
// (the same mechanism the `Box::leak`ed `VcpuProgram` uses, but a `static` instead of a JS-threaded
// pointer). A worker vCPU's `Jit.install`/`uninstall`/`invoke` is then serviced **inside**
// [`svm_par_run`] against this powerbox + the shared `Domain` — so the JS host services no new events
// (it never sees a JIT op, needs no new glue). During the run the powerbox is read-only (the unit is
// compiled at setup, before any spawn), so the concurrent `&Host` reads need no lock; the install/
// dispatch mutation lives in the `Domain`, which is already interior-mutable + thread-safe.

/// The shared §22 powerbox: a `Host` with the `Jit` cap granted + [`JIT_SERVICE`] host-compiled, plus
/// the handles the root guest receives as `(jit, code)`.
struct ParPowerbox {
    host: Host,
    jit: i32,
    code: i32,
}

/// The leaked [`ParPowerbox`] pointer (or `0`), shared across Workers via shared linear memory.
static PAR_PB: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// `2^4 = 16` dispatch-table slots — the `Jit` table reservation matched by [`svm_par_compile_jit`]
/// and the powerbox grant so guest `install` lands in range (mirrors [`jit_exec`]).
const PAR_JIT_TABLE_LOG2: u8 = 4;

/// `2^10 = 1024` dispatch-table slots for the on-ramp `Jit` grant — matches svm-run's
/// `CLI_JIT_TABLE_LOG2`. A self-hosted guest (the JACL compiler) binds a staged unit's `Slot` imports
/// to its own functions by index, so the table must cover the host program's function count (~800).
const ONRAMP_JIT_TABLE_LOG2: u8 = 10;

/// Build the **shared powerbox** for a §22-JIT run: grant the `Jit` cap (16-slot table) on a fresh
/// `Host`, host-compile [`JIT_SERVICE`] into it, then leak it and publish the pointer for every Worker.
/// `guest`'s declared memory sizes the domain (the validator's memory-match precondition). Returns `1`
/// on success, `0` on decode / parse / compile failure. Call **once** (on the main thread) before the
/// run; the published pointer outlives it.
#[no_mangle]
pub extern "C" fn svm_par_powerbox(guest_ptr: *const u8, guest_len: usize) -> i32 {
    par_run_gen_bump(); // I22: one bump per run — gates the once-per-run codegen emit (see CodegenGuard)
                        // SAFETY: the host guarantees `[guest_ptr, guest_len)` is a live allocation it just filled.
    let bytes = unsafe { core::slice::from_raw_parts(guest_ptr, guest_len) };
    let Ok(m) = svm_encode::decode_module(bytes) else {
        return 0;
    };
    let service = match svm_text::parse_module(JIT_SERVICE) {
        Ok(s) => svm_encode::encode_module(&s),
        Err(_) => return 0,
    };
    let mut host = Host::new();
    let jit = host.grant_jit_with_table(m.memory.map(|mc| mc.size_log2), PAR_JIT_TABLE_LOG2);
    host.set_jit_validator(browser_jit_validator);
    let code = match host.jit_compile(jit, &service) {
        Ok(Ok(c)) => c.handle,
        _ => return 0,
    };
    let pb = Box::into_raw(Box::new(ParPowerbox { host, jit, code }));
    PAR_PB.store(pb as usize, std::sync::atomic::Ordering::Release);
    // Last-published run recipe wins (a page runs several kinds back to back).
    PAR_INST.store(0, std::sync::atomic::Ordering::Release);
    PAR_IO.store(0, std::sync::atomic::Ordering::Release);
    PAR_JIT.store(0, std::sync::atomic::Ordering::Release);
    PAR_JIT_CODEGEN.store(false, std::sync::atomic::Ordering::Release); // this is the interp JIT path
    1
}

/// Codegen mode: when set, a guest's `Jit.invoke` of the emitted unit surfaces as [`PAR_JIT_INVOKE`]
/// so the Worker runs it on wasm; else the invoke is serviced in-Rust on the interpreter (as before).
static PAR_JIT_CODEGEN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// The emitted wasm of the run's single §22 unit (stashed once at [`svm_par_powerbox_jit_codegen`]
/// setup; immutable + shared across Workers, each instantiates its own instance). `(null, 0)` ⇒ none.
static mut JIT_UNIT_WASM: (*mut u8, usize) = (core::ptr::null_mut(), 0);

fn par_jit_codegen() -> bool {
    PAR_JIT_CODEGEN.load(std::sync::atomic::Ordering::Acquire)
}

/// A **float** §22 unit for the real-codegen proof: `fservice(a, b) = a*b + 100.0`, all `f64` — so
/// the Worker marshals args from the slot bits to JS `Number`s and the `f64` result back to its bits
/// (the ABI generalization to floats). `fservice(6.0, 7.0) = 142.0`.
const JIT_SERVICE_FLOAT: &str = r#"memory 16
func (f64, f64) -> (f64) {
block 0 (v0: f64, v1: f64) {
  v2 = f64.mul v0 v1
  v3 = f64.const 100.0
  v4 = f64.add v2 v3
  return v4
  }
}
"#;

/// Which §22 unit the codegen powerbox host-compiles + emits: `0` = the i32 [`JIT_SERVICE`] (the
/// default, matching the interp `#jit` item), `1` = the f64 [`JIT_SERVICE_FLOAT`]. The JS host sets
/// this (via [`svm_par_jit_codegen_service`]) before the run to exercise int vs float marshalling.
static PAR_JIT_SERVICE: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);

/// Select the codegen unit for the next run (`0` = i32 service, `1` = f64 service). Set on every
/// instance (page + each Worker) before enabling codegen, with the same value.
#[no_mangle]
pub extern "C" fn svm_par_jit_codegen_service(kind: i32) {
    PAR_JIT_SERVICE.store(kind, std::sync::atomic::Ordering::Release);
}

fn codegen_service_src() -> &'static str {
    if PAR_JIT_SERVICE.load(std::sync::atomic::Ordering::Acquire) == 1 {
        JIT_SERVICE_FLOAT
    } else {
        JIT_SERVICE
    }
}

/// Toggle codegen mode on the current §22 powerbox (`on != 0` ⇒ `Jit.invoke` runs on emitted wasm;
/// `0` ⇒ the interpreter services it in-Rust). Lets a host run the **same** guest + unit both ways
/// for a differential (the emitted region must match the interpreter). Set by
/// [`svm_par_powerbox_jit_codegen`]; a host that wants the interpreter path flips it off.
#[no_mangle]
pub extern "C" fn svm_par_jit_set_codegen(on: i32) {
    PAR_JIT_CODEGEN.store(on != 0, std::sync::atomic::Ordering::Release);
}

/// Enable §22 real codegen for the run: emit the run's unit (the scalar service selected by
/// [`codegen_service_src`] — i32 [`JIT_SERVICE`] or f64 [`JIT_SERVICE_FLOAT`]) into the shared
/// [`JIT_UNIT_WASM`] stash and set codegen mode. Every Worker calls this in its setup (like
/// [`svm_par_enable_jit`] for tier-up), but — since the stash is a single shared copy across Workers
/// (I22) — only the **first** caller of the run actually emits (under `CODEGEN_LOCK`); the rest reuse
/// it. Returns `1` on success, `0` if the unit is outside the emitter subset.
#[no_mangle]
pub extern "C" fn svm_par_enable_jit_codegen() -> i32 {
    use std::sync::atomic::Ordering;
    par_install_panic_capture(); // I22: capture a setup-time engine panic's FILE:LINE (not a bare `unreachable`)
    let guard = CodegenGuard::acquire();
    let generation = guard.generation();
    if JIT_CG_DONE_GEN.load(Ordering::Relaxed) == generation {
        return JIT_CG_RESULT.load(Ordering::Relaxed);
    }
    let result = (|| {
        let Ok(service_m) = svm_text::parse_module(codegen_service_src()) else {
            return 0;
        };
        // The §22 codegen service unit is a fixed, fully-in-subset scalar function, so `compile_jit`
        // emits it whole and wasm-driven (rooted at func 0). Defensively require that — the §22 path
        // runs the unit as emitted `f0`, so an interpreter-driven fallback would be a bug; fail closed.
        let Ok(svm_wasm_jit::Artifact {
            wasm,
            drive: svm_wasm_jit::DriveMode::WasmDriven { .. },
            ..
        }) = svm_wasm_jit::compile_jit(&service_m, svm_wasm_jit::Shape::Batch { entry: 0 }, true)
        else {
            return 0;
        };
        // SAFETY: written once per run while CODEGEN_LOCK is held; Workers then read it stable.
        unsafe { stash(&mut *core::ptr::addr_of_mut!(JIT_UNIT_WASM), wasm) };
        PAR_JIT_CODEGEN.store(true, Ordering::Release);
        1
    })();
    JIT_CG_RESULT.store(result, Ordering::Relaxed);
    JIT_CG_DONE_GEN.store(generation, Ordering::Relaxed);
    result
}

/// Build the **shared powerbox** for a §22 **real-codegen** run: like [`svm_par_powerbox`] but the
/// host-compiled unit is the scalar service selected by [`codegen_service_src`] (i32 [`JIT_SERVICE`]
/// or f64 [`JIT_SERVICE_FLOAT`]), and its wasm is emitted (via
/// [`svm_wasm_jit::compile_jit`] with [`svm_wasm_jit::Shape::Batch`], shared memory) + stashed so a guest `Jit.invoke`
/// runs the emitted region on the Worker instead of the interpreter. Returns `1` on success, `0` on
/// decode/parse/compile/emit failure (fail-closed: the caller keeps the interpreter). Call **once**
/// (on the main thread) before the run.
#[no_mangle]
pub extern "C" fn svm_par_powerbox_jit_codegen(guest_ptr: *const u8, guest_len: usize) -> i32 {
    par_run_gen_bump(); // I22: one bump per run — gates the once-per-run codegen emit (see CodegenGuard)
                        // SAFETY: the host guarantees `[guest_ptr, guest_len)` is a live allocation it just filled.
    let bytes = unsafe { core::slice::from_raw_parts(guest_ptr, guest_len) };
    let Ok(m) = svm_encode::decode_module(bytes) else {
        return 0;
    };
    let Ok(service_m) = svm_text::parse_module(codegen_service_src()) else {
        return 0;
    };
    let service = svm_encode::encode_module(&service_m);
    let mut host = Host::new();
    let jit = host.grant_jit_with_table(m.memory.map(|mc| mc.size_log2), PAR_JIT_TABLE_LOG2);
    host.set_jit_validator(browser_jit_validator);
    let code = match host.jit_compile(jit, &service) {
        Ok(Ok(c)) => c.handle,
        _ => return 0,
    };
    // Emit the unit wasm on **this** (page) instance too, so a single-vCPU run driven on the page
    // works; each Worker emits its own copy via [`svm_par_enable_jit_codegen`] (per-instance stash).
    // Fail-closed if the unit is outside the emitter subset — then there is nothing to run on wasm.
    if svm_par_enable_jit_codegen() != 1 {
        return 0;
    }
    let pb = Box::into_raw(Box::new(ParPowerbox { host, jit, code }));
    PAR_PB.store(pb as usize, std::sync::atomic::Ordering::Release);
    PAR_INST.store(0, std::sync::atomic::Ordering::Release);
    PAR_IO.store(0, std::sync::atomic::Ordering::Release);
    PAR_JIT.store(0, std::sync::atomic::Ordering::Release);
    1
}

/// Pointer / length of the run's emitted §22 unit wasm (see [`svm_par_powerbox_jit_codegen`]).
#[no_mangle]
pub extern "C" fn svm_par_jit_unit_wasm_ptr() -> *const u8 {
    unsafe { (*core::ptr::addr_of!(JIT_UNIT_WASM)).0 }
}
#[no_mangle]
pub extern "C" fn svm_par_jit_unit_wasm_len() -> usize {
    unsafe { (*core::ptr::addr_of!(JIT_UNIT_WASM)).1 }
}

/// Borrow the published powerbox (`None` until [`svm_par_powerbox`] ran). The pointer is published with
/// `Release`; this `Acquire` load pairs with it so the `Host` it built is visible to this Worker.
fn par_pb() -> Option<&'static ParPowerbox> {
    let p = PAR_PB.load(std::sync::atomic::Ordering::Acquire) as *const ParPowerbox;
    // SAFETY: once published the powerbox is leaked (never freed) and read-only for the run, so the
    // shared `&'static` is sound (concurrent `&self` reads only).
    unsafe { p.as_ref() }
}

/// Resolve a code-handle's unit funcs under authority `handle` against the powerbox (the `install` /
/// `invoke` service): a forged / cross-domain / wrong-type handle is an inert `CapFault` → trap.
fn par_resolve_unit(
    pb: &ParPowerbox,
    handle: i32,
    code: i32,
) -> Result<std::sync::Arc<[svm_ir::Func]>, Trap> {
    let domain = pb.host.resolve_jit_domain(handle)?;
    let (cd, cu) = pb.host.resolve_jit_code(code)?;
    if cd != domain {
        return Err(Trap::CapFault);
    }
    pb.host.jit_unit_funcs(cd, cu).ok_or(Trap::CapFault)
}

// ---- §14 instantiate across Workers (THREADS.md 4c-domain §14-D2) -------------------------------
// The §14 root powerbox lives **in the root vCPU** (unlike the §22 JIT powerbox, which the vCPU asks
// the host to resolve against): §14 resolves its `Instantiator` authority in-Vm during `resume`, so
// the grant must be in the vCPU's own `Host`. This static only carries the *recipe* — the authority
// range and the optional granted module — published once by the main thread so the root Worker can
// build its powerbox deterministically. Confined children never touch it: their attenuated powerbox
// is built inside `Vcpu::new_confined_child`, so no authority ever crosses JS (the `PAR_INSTANTIATE`
// event operands are inert integers).

/// The §14 run recipe: `Instantiator` authority over `[0, win_size)` + an optional `Module` grant.
struct ParInstCfg {
    win_size: u64,
    module: Option<svm_ir::Module>,
}

/// The leaked [`ParInstCfg`] pointer (or `0`), shared across Workers via shared linear memory.
static PAR_INST: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Publish the §14 run recipe: the root's `Instantiator` will span `[0, win_size)`; a non-empty
/// `[mod_ptr, mod_len)` is decoded as the **granted module** for `instantiate_module` (`0` len ⇒ no
/// grant). Returns `1`, or `0` on a bad module. Call once (on the main thread) before the run.
#[no_mangle]
pub extern "C" fn svm_par_powerbox_inst(win_size: u64, mod_ptr: *const u8, mod_len: usize) -> i32 {
    par_run_gen_bump(); // I22: one bump per run — gates the once-per-run codegen emit (see CodegenGuard)
    let module = if mod_len == 0 {
        None
    } else {
        // SAFETY: the host guarantees `[mod_ptr, mod_len)` is a live allocation it just filled.
        let bytes = unsafe { core::slice::from_raw_parts(mod_ptr, mod_len) };
        match svm_encode::decode_module(bytes) {
            Ok(m) => Some(m),
            Err(_) => return 0,
        }
    };
    let cfg = Box::into_raw(Box::new(ParInstCfg { win_size, module }));
    PAR_INST.store(cfg as usize, std::sync::atomic::Ordering::Release);
    // Last-published run recipe wins (a page runs several kinds back to back).
    PAR_PB.store(0, std::sync::atomic::Ordering::Release);
    PAR_IO.store(0, std::sync::atomic::Ordering::Release);
    PAR_JIT.store(0, std::sync::atomic::Ordering::Release);
    1
}

/// Borrow the published §14 recipe (`None` until [`svm_par_powerbox_inst`] ran). Leaked + read-only,
/// as [`par_pb`].
fn par_inst() -> Option<&'static ParInstCfg> {
    let p = PAR_INST.load(std::sync::atomic::Ordering::Acquire) as *const ParInstCfg;
    // SAFETY: once published the recipe is leaked (never freed) and read-only for the run.
    unsafe { p.as_ref() }
}

// ---- §14 instantiate_module **real codegen** (BROWSER.md § "wasm-JIT tier", slice 5) -------------
// A confined executor child whose granted module is fully in-subset runs its entry on **emitted
// wasm** on its own Worker (the module "compiles on push") instead of the bytecode interpreter — the
// child fills the same completion slot the parent `join`s, so no engine change is needed. The granted
// module is emitted once per instance (each Worker computes its own copy from the shared recipe, like
// the tier-up bitmap); a child entry that uses a `cap.call` (a nested `instantiate`, an address-space
// op) is **not** in-subset, so it stays on the interpreter (fail-closed).

/// The emitted wasm of the run's granted §14 unit (per-instance stash; `(null, 0)` ⇒ none).
static mut INST_UNIT_WASM: (*mut u8, usize) = (core::ptr::null_mut(), 0);
/// The granted unit's per-function eligibility ([`svm_wasm_jit::compile_nested`]'s `emitted` bitmap):
/// `f{i}` is emitted + safe to call directly. A confined child whose entry is eligible runs on wasm;
/// else it interprets.
static mut INST_ELIGIBLE: Option<Vec<bool>> = None;

/// Enable §14 real codegen for the run: emit the granted unit ([`ParInstCfg::module`]) to wasm and
/// stash it + the per-function eligibility. Called by each Worker before it builds a confined child
/// (like [`svm_par_enable_jit_codegen`]), but — the stash is a single shared copy across Workers (I22)
/// — only the **first** caller of the run emits (under `CODEGEN_LOCK`); the rest reuse it. Returns
/// `1` on success, `0` if there is no granted module or it is outside the emitter subset.
#[no_mangle]
pub extern "C" fn svm_par_enable_inst_codegen() -> i32 {
    use std::sync::atomic::Ordering;
    par_install_panic_capture(); // I22: capture a setup-time engine panic's FILE:LINE (not a bare `unreachable`)
    let guard = CodegenGuard::acquire();
    let generation = guard.generation();
    if INST_CG_DONE_GEN.load(Ordering::Relaxed) == generation {
        return INST_CG_RESULT.load(Ordering::Relaxed);
    }
    let result = (|| {
        let Some(cfg) = par_inst() else {
            return 0;
        };
        let Some(m) = &cfg.module else {
            return 0;
        };
        // §14 VM-in-VM codegen via the library's single nested front door ([`compile_nested`]): it
        // picks the drive mode from the IR and always yields a runnable artifact. A cap-using entry
        // (`cap.call 6 0/1` instantiate/join, or a `thread.spawn`) emits, its bounce arriving at the
        // Worker via the `env.instantiate`/`env.join`/`env.thread_*` imports (serviced through the same
        // confined-child completion-slot protocol as the interpreter path); a fiber-bearing unit falls
        // to an interpreter-driven tier-up. Either way `emitted[i]` is the sound "safe to call `f{i}`
        // directly" signal the Worker gates on (`svm_par_inst_eligible`): a fiber reachable from the
        // entry drops it from `emitted` (the tier-up fixpoint), and a `thread.spawn`ed fiber runs in its
        // own spawned interpreter vCPU — never across the emitted frame. The Worker offers the whole
        // nested import set unconditionally, so the uniform layout `compile_nested` emits just works.
        // ADDRESS_SPACE wrappers are NOT outlined here (the browser's `call_interp` carries no powerbox
        // yet), so a `sub`/`page_size` entry stays interpreter-driven; a `map`/`unmap`/`protect` unit
        // emits nothing and interprets wholly (mask-only confinement can't honor page state).
        let Ok(svm_wasm_jit::Artifact {
            wasm,
            emitted: eligible,
            ..
        }) = svm_wasm_jit::compile_nested(m, true)
        else {
            return 0;
        };
        // SAFETY: written once per run while CODEGEN_LOCK is held; Workers then read it stable.
        unsafe {
            stash(&mut *core::ptr::addr_of_mut!(INST_UNIT_WASM), wasm);
            *core::ptr::addr_of_mut!(INST_ELIGIBLE) = Some(eligible);
        }
        1
    })();
    INST_CG_RESULT.store(result, Ordering::Relaxed);
    INST_CG_DONE_GEN.store(generation, Ordering::Relaxed);
    result
}

/// Pointer / length of this instance's emitted §14 unit wasm (see [`svm_par_enable_inst_codegen`]).
#[no_mangle]
pub extern "C" fn svm_par_inst_unit_wasm_ptr() -> *const u8 {
    unsafe { (*core::ptr::addr_of!(INST_UNIT_WASM)).0 }
}
#[no_mangle]
pub extern "C" fn svm_par_inst_unit_wasm_len() -> usize {
    unsafe { (*core::ptr::addr_of!(INST_UNIT_WASM)).1 }
}

/// Whether the granted unit's function `entry` is emitted (safe to run `f{entry}` on wasm). `0` when
/// codegen isn't enabled, `entry` is out of range, or that function is out of the emitter subset.
#[no_mangle]
pub extern "C" fn svm_par_inst_eligible(entry: u32) -> i32 {
    // SAFETY: single-reader per instance; set by `svm_par_enable_inst_codegen`.
    let e = unsafe { (*core::ptr::addr_of!(INST_ELIGIBLE)).as_ref() };
    e.and_then(|v| v.get(entry as usize))
        .copied()
        .map_or(0, |b| b as i32)
}

/// The granted unit's `entry` param count (1 or 2 — the instantiator/address-space cap handles a pure
/// unit ignores). The Worker passes this many `0` args to the emitted `f{entry}`. `0` if no recipe.
#[no_mangle]
pub extern "C" fn svm_par_inst_nparams(entry: u32) -> usize {
    par_inst()
        .and_then(|c| c.module.as_ref())
        .and_then(|m| m.funcs.get(entry as usize))
        .map_or(0, |f| f.params.len())
}

// ---- 4d: host I/O across Workers — the run's shared powerbox ------------------------------------
// THREADS.md 4d: one `Mutex<Host>`, leaked into the shared linear memory (the same cross-Worker
// sharing as `PAR_PB`/`PAR_INST`), attached to **every** vCPU of the run
// ([`bytecode::Vcpu::with_shared_host`]) — so a worker vCPU's `cap.call` (host I/O) dispatches
// in-engine under the lock, `drive_parallel`'s 4c-host model, with no JS in the loop at all: the
// `Host` is fully virtual (stdout is an in-memory buffer the page reads back after the run).

/// The shared I/O powerbox: the `Mutex<Host>` every vCPU dispatches through, plus the handles the
/// root guest receives as its args.
struct ParIoCfg {
    host: std::sync::Mutex<Host>,
    /// The `Stream(Out)` handle (the root's single entry arg).
    out: i32,
}

/// The leaked [`ParIoCfg`] pointer (or `0`), shared across Workers via shared linear memory.
static PAR_IO: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Publish the run's **shared I/O powerbox**: a fresh `Host` granted a `Stream(Out)`, wrapped in the
/// `Mutex` every vCPU will dispatch `cap.call` through. The root is seeded with `[out_handle]`
/// (`svm_par_root`); read the accumulated stdout back after the run via [`svm_par_stdout_len`] +
/// [`svm_par_stdout_ptr`]. Call once (on the main thread) before the run; last-published run recipe
/// wins (the §22/§14 recipes are cleared, and vice versa).
#[no_mangle]
pub extern "C" fn svm_par_powerbox_io() -> i32 {
    par_run_gen_bump(); // I22: one bump per run — gates the once-per-run codegen emit (see CodegenGuard)
    let mut host = Host::new();
    let out = host.grant_stream(StreamRole::Out);
    let cfg = Box::into_raw(Box::new(ParIoCfg {
        host: std::sync::Mutex::new(host),
        out,
    }));
    PAR_IO.store(cfg as usize, std::sync::atomic::Ordering::Release);
    PAR_INST.store(0, std::sync::atomic::Ordering::Release);
    PAR_PB.store(0, std::sync::atomic::Ordering::Release);
    PAR_JIT.store(0, std::sync::atomic::Ordering::Release);
    1
}

/// Clear every published run recipe — the next run is **plain** (compute-only, no powerbox). The
/// recipes are last-published-wins for back-to-back runs of *different* kinds; a plain run after a
/// powerbox run (the playground can run modes in any order) needs this explicit "none" publish, or
/// the stale recipe would seed the new root with args its entry doesn't take.
#[no_mangle]
pub extern "C" fn svm_par_powerbox_none() {
    par_run_gen_bump(); // I22: one bump per run — gates the once-per-run codegen emit (see CodegenGuard)
    PAR_PB.store(0, std::sync::atomic::Ordering::Release);
    PAR_INST.store(0, std::sync::atomic::Ordering::Release);
    PAR_IO.store(0, std::sync::atomic::Ordering::Release);
    PAR_JIT.store(0, std::sync::atomic::Ordering::Release);
    PAR_JIT_CODEGEN.store(false, std::sync::atomic::Ordering::Release);
}

/// Borrow the published I/O powerbox (`None` until [`svm_par_powerbox_io`] ran). Leaked; interior
/// mutability is the `Mutex` (cross-Worker-safe on wasm atomics, like the `Domain`'s `ModuleSource`).
fn par_io() -> Option<&'static ParIoCfg> {
    let p = PAR_IO.load(std::sync::atomic::Ordering::Acquire) as *const ParIoCfg;
    // SAFETY: once published the powerbox is leaked (never freed); all access is via the `Mutex`.
    unsafe { p.as_ref() }
}

// ---- §22 runtime `Jit.compile` on the wasm tier (single-Worker) ----------------------------------
// Unlike `ParPowerbox` (a *fixed* unit host-compiled at setup, its code handle handed to the guest),
// this powerbox lets the guest build an IR blob **at runtime**, `compile` it — minting a unit **and
// emitting its wasm** in this host, because the `Jit` grant carries the browser validator + emitter and
// the vCPU dispatches its `cap.call`s through this shared `Mutex<Host>` (`with_shared_host`) — then
// `invoke` it on that emitted wasm. The shared `Mutex` is also the seam threaded compile will serialize
// on (DESIGN.md §22), so this is the single-Worker step of the same design, not a throwaway.

/// The runtime-`Jit.compile` powerbox recipe: the shared host (grant + validator + emitter) and the
/// `Jit` domain handle the root is seeded with.
struct ParJitCfg {
    host: std::sync::Mutex<Host>,
    /// The `Jit` domain handle (the root's single entry arg).
    jit: i32,
}

/// The leaked [`ParJitCfg`] pointer (or `0`), shared across Workers via shared linear memory.
static PAR_JIT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Publish the runtime-`Jit.compile` powerbox: a fresh `Host` granted `Jit` (memory-match precondition
/// from the guest's declared memory) with [`browser_jit_validator`] + [`browser_jit_wasm_emitter`]
/// installed, wrapped in the shared `Mutex` the vCPU dispatches `cap.call` through. The root is seeded
/// `[jit]` ([`svm_par_root`]); the guest builds an IR blob, `compile`s it (emitting wasm), and
/// `invoke`s it on the emitted region. Codegen on by default (flip with [`svm_par_jit_set_codegen`] to
/// run the interpreter path for a differential). Call once (on the main thread) before the run; the
/// other run recipes are cleared (last-published-wins). Returns `1`, or `0` on a bad guest module.
#[no_mangle]
pub extern "C" fn svm_par_powerbox_jit_runtime(guest_ptr: *const u8, guest_len: usize) -> i32 {
    par_run_gen_bump(); // I22: one bump per run
                        // SAFETY: the host guarantees `[guest_ptr, guest_len)` is a live allocation it just filled.
    let bytes = unsafe { core::slice::from_raw_parts(guest_ptr, guest_len) };
    let Ok(m) = svm_encode::decode_module(bytes) else {
        return 0;
    };
    let mut host = Host::new();
    host.set_jit_validator(browser_jit_validator);
    host.set_jit_wasm_emitter(browser_jit_wasm_emitter);
    let jit = host.grant_jit_with_table(m.memory.map(|mc| mc.size_log2), PAR_JIT_TABLE_LOG2);
    let cfg = Box::into_raw(Box::new(ParJitCfg {
        host: std::sync::Mutex::new(host),
        jit,
    }));
    PAR_JIT.store(cfg as usize, std::sync::atomic::Ordering::Release);
    PAR_PB.store(0, std::sync::atomic::Ordering::Release);
    PAR_INST.store(0, std::sync::atomic::Ordering::Release);
    PAR_IO.store(0, std::sync::atomic::Ordering::Release);
    PAR_JIT_CODEGEN.store(true, std::sync::atomic::Ordering::Release);
    1
}

/// Borrow the published runtime-`Jit.compile` powerbox (`None` until [`svm_par_powerbox_jit_runtime`]).
fn par_jit_rt() -> Option<&'static ParJitCfg> {
    let p = PAR_JIT.load(std::sync::atomic::Ordering::Acquire) as *const ParJitCfg;
    // SAFETY: once published the powerbox is leaked (never freed); all access is via the `Mutex`.
    unsafe { p.as_ref() }
}

/// Resolve a `JitInvoke` event's authority against the shared runtime-compile host (mirrors
/// [`par_resolve_unit`]) and hand back the unit's funcs + its emitted wasm.
#[allow(clippy::type_complexity)]
fn par_resolve_unit_rt(
    h: &Host,
    handle: i32,
    code: i32,
) -> Result<(std::sync::Arc<[svm_ir::Func]>, Option<std::sync::Arc<[u8]>>), Trap> {
    let domain = h.resolve_jit_domain(handle)?;
    let (cd, cu) = h.resolve_jit_code(code)?;
    if cd != domain {
        return Err(Trap::CapFault);
    }
    let funcs = h.jit_unit_funcs(cd, cu).ok_or(Trap::CapFault)?;
    Ok((funcs, h.jit_unit_wasm(cd, cu)))
}

/// §22 **Model B2 cross-Worker** mirror registry: `slot → the code handle installed there` (or `-1`
/// empty), shared across every Worker (one arena / one set of statics behind the shared memory). The
/// shared interpreter `Domain` dispatch table is atomics-in-memory but has no slot→emitted-wasm link;
/// this records it at the (Rust-serviced) `install`/`uninstall` sites so a Worker can rebuild its own
/// `WebAssembly.Table` from `(slot → code → svm_par_jit_code_wasm_by_handle)`. Sized lazily to the
/// grant reservation `1 << PAR_JIT_TABLE_LOG2`.
static PAR_JIT_SLOT_CODE: std::sync::Mutex<Vec<i32>> = std::sync::Mutex::new(Vec::new());

fn par_jit_slot_record(slot: usize, code: i32) {
    let mut v = PAR_JIT_SLOT_CODE.lock().unwrap_or_else(|e| e.into_inner());
    let need = 1usize << PAR_JIT_TABLE_LOG2;
    if v.len() < need {
        v.resize(need, -1);
    }
    if let Some(e) = v.get_mut(slot) {
        *e = code;
    }
}

fn par_jit_slot_clear(slot: usize) {
    let mut v = PAR_JIT_SLOT_CODE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(e) = v.get_mut(slot) {
        *e = -1;
    }
}

/// When on, the runtime-`Jit.compile` emitter emits **Model B2** units (importing the shared reserved
/// funcref table) instead of local-table units — so an installed unit's `call_indirect` dispatches to
/// other installed units through the per-Worker table mirror. Off by default (the emitted-unit shape
/// changes, so the JS host must provide `env.__indirect_function_table`).
static PAR_JIT_B2: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn par_jit_b2() -> bool {
    PAR_JIT_B2.load(std::sync::atomic::Ordering::Relaxed)
}

/// Toggle Model-B2 emission for the runtime-compile tier (see [`PAR_JIT_B2`]). The JS host sets this
/// on iff it provides each Worker a shared `WebAssembly.Table` import.
#[no_mangle]
pub extern "C" fn svm_par_jit_set_b2(on: i32) {
    PAR_JIT_B2.store(on != 0, std::sync::atomic::Ordering::Relaxed);
}

/// The §22 `Jit` dispatch-table reservation (`log2` of the slot count) — the size the JS host makes
/// each per-Worker `WebAssembly.Table`, matching the emitted `call_indirect` mask `idx & (2^n - 1)`.
#[no_mangle]
pub extern "C" fn svm_par_jit_table_log2() -> u32 {
    PAR_JIT_TABLE_LOG2 as u32
}

/// The code handle installed at dispatch-table `slot` (or `-1` if empty) — the mirror map a Worker
/// reads to rebuild its per-Worker `WebAssembly.Table` (§22 B2 cross-Worker).
#[no_mangle]
pub extern "C" fn svm_par_jit_slot_code(slot: u32) -> i32 {
    let v = PAR_JIT_SLOT_CODE.lock().unwrap_or_else(|e| e.into_inner());
    v.get(slot as usize).copied().unwrap_or(-1)
}

/// Emitted-wasm length for **any** code handle in the runtime domain (not just the pending invoke's),
/// so a Worker can instantiate a slot's unit it hasn't itself invoked. `0` if none. The bytes (via
/// [`svm_par_jit_code_wasm_by_handle_ptr`]) live in the shared host's heap = shared linear memory,
/// held for the process, so the returned pointer stays valid.
#[no_mangle]
pub extern "C" fn svm_par_jit_code_wasm_by_handle_len(handle: i32) -> usize {
    par_jit_rt()
        .and_then(|cfg| {
            let g = cfg.host.lock().unwrap_or_else(|e| e.into_inner());
            g.resolve_jit_code(handle)
                .ok()
                .and_then(|(cd, cu)| g.jit_unit_wasm(cd, cu))
                .map(|w| w.len())
        })
        .unwrap_or(0)
}

/// Pointer to the emitted-wasm for `handle` (see [`svm_par_jit_code_wasm_by_handle_len`]).
#[no_mangle]
pub extern "C" fn svm_par_jit_code_wasm_by_handle_ptr(handle: i32) -> *const u8 {
    par_jit_rt()
        .and_then(|cfg| {
            let g = cfg.host.lock().unwrap_or_else(|e| e.into_inner());
            g.resolve_jit_code(handle)
                .ok()
                .and_then(|(cd, cu)| g.jit_unit_wasm(cd, cu))
                .map(|w| w.as_ptr())
        })
        .unwrap_or(core::ptr::null())
}

/// Live-vCPU counter across Workers — the browser path's anti-bomb **backstop** (the native drivers
/// give the spawner a clean `ThreadFault`; here a construction past the cap returns null and the JS
/// host fails the run — cruder, but it bounds Worker creation). Incremented by the `svm_par_*` vCPU
/// constructors, decremented by [`svm_par_free`].
static PAR_LIVE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
/// Far above any legitimate fan-out (a tab with 256 live Workers is already pathological), far below
/// a Worker bomb's ambition.
const PAR_MAX_VCPUS: u32 = 256;

/// Admit one vCPU under the live cap (decrementing back out on refusal).
fn par_vcpu_admit() -> bool {
    use std::sync::atomic::Ordering;
    if PAR_LIVE.fetch_add(1, Ordering::AcqRel) >= PAR_MAX_VCPUS {
        PAR_LIVE.fetch_sub(1, Ordering::AcqRel);
        return false;
    }
    true
}

/// Un-admit a vCPU that failed to construct (the success path decrements via [`svm_par_free`]).
fn par_vcpu_retire() {
    PAR_LIVE.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
}

/// Like [`svm_par_compile`], but reserve the `Jit` dispatch table (matching the powerbox grant) so a
/// guest `install` lands in range. Use this (not [`svm_par_compile`]) for a §22-JIT run.
#[no_mangle]
pub extern "C" fn svm_par_compile_jit(
    mod_ptr: *const u8,
    mod_len: usize,
) -> *mut bytecode::VcpuProgram {
    // SAFETY: the host guarantees `[mod_ptr, mod_len)` is a live `svm_alloc`ation it just filled.
    let bytes = unsafe { core::slice::from_raw_parts(mod_ptr, mod_len) };
    let Ok(m) = svm_encode::decode_module(bytes) else {
        return core::ptr::null_mut();
    };
    match bytecode::VcpuProgram::compile_with_jit_table(&m, PAR_JIT_TABLE_LOG2) {
        Some(p) => Box::into_raw(Box::new(p)),
        None => core::ptr::null_mut(),
    }
}

/// Build the **root** vCPU (function `func`) over the shared window `[win_ptr, win_size)`; it seeds +
/// data-initialises the window (the once). Returns a boxed [`ParVcpu`] pointer, null on a bad func.
#[no_mangle]
pub extern "C" fn svm_par_root(
    prog: *mut bytecode::VcpuProgram,
    win_ptr: *mut u8,
    win_size: usize,
    func: u32,
) -> *mut ParVcpu {
    if !par_vcpu_admit() {
        return core::ptr::null_mut();
    }
    // SAFETY: the host guarantees `[win_ptr, win_size)` is a live shared window for the run.
    let back = std::sync::Arc::new(unsafe { svm_interp::Region::shared(win_ptr, win_size as u64) });
    // A §14 run builds the root's **own** powerbox from the published recipe (`Instantiator` +
    // optional `Module` grant; §14 resolves authority in-Vm, so the grants must live in the vCPU's
    // host) and seeds the root with the handles. A §22-JIT run seeds `(jit, code)` from the shared
    // powerbox; a 4d I/O run attaches the shared `Mutex<Host>` and seeds `[out]`; a plain run gets
    // no args. Signatures unchanged either way — the JS host just calls the matching
    // `svm_par_powerbox*` first.
    if let Some(cfg) = par_inst() {
        let mut host = Host::new();
        let inst = host.grant_instantiator(0, cfg.win_size);
        let mut args = vec![Value::I32(inst)];
        if let Some(m) = &cfg.module {
            args.push(Value::I32(host.grant_module(m)));
        }
        // SAFETY: `prog` is a live program pointer the host keeps alive for the run.
        return match bytecode::Vcpu::new_root_with_powerbox(
            unsafe { prog_ref(prog) },
            func,
            &args,
            back,
            &[],
            host,
        ) {
            Ok(inner) => par_box(inner),
            Err(_) => {
                par_vcpu_retire();
                core::ptr::null_mut()
            }
        };
    }
    // A §22 **runtime-compile** run: the guest's `Jit` authority + validator + emitter live in the
    // shared `Mutex<Host>` it dispatches `cap.call`s through (`with_shared_host`), so its runtime
    // `compile` mints + emits into that host. Seed `[jit]` only — the guest compiles its own unit.
    if let Some(cfg) = par_jit_rt() {
        // SAFETY: `prog` is a live program pointer the host keeps alive for the run.
        return match bytecode::Vcpu::new_root(
            unsafe { prog_ref(prog) },
            func,
            &[Value::I32(cfg.jit)],
            back,
            &[],
        ) {
            Ok(inner) => par_box(inner.with_shared_host(&cfg.host)),
            Err(_) => {
                par_vcpu_retire();
                core::ptr::null_mut()
            }
        };
    }
    let (args, io): (Vec<Value>, Option<&'static ParIoCfg>) = match (par_io(), par_pb()) {
        (Some(io), _) => (vec![Value::I32(io.out)], Some(io)),
        (None, Some(pb)) => (vec![Value::I32(pb.jit), Value::I32(pb.code)], None),
        (None, None) => (Vec::new(), None),
    };
    // SAFETY: `prog` is a live program pointer the host keeps alive for the run.
    match bytecode::Vcpu::new_root(unsafe { prog_ref(prog) }, func, &args, back, &[]) {
        Ok(inner) => {
            let inner = match io {
                Some(io) => inner.with_shared_host(&io.host),
                None => inner,
            };
            par_box(with_tierup(inner))
        }
        Err(_) => {
            par_vcpu_retire();
            core::ptr::null_mut()
        }
    }
}

/// Build a `thread.spawn`ed **child** vCPU (`func(sp, arg)`) over the **same** shared window — it does
/// not re-seed (the window is already live). Called on the child's Worker. Null on a bad func.
/// `module` is the spawning frame's module from the `PAR_SPAWN` event (`ev_a >> 32`) — `func`
/// resolves there and the child's root frame starts there (module-0 for plain guests).
#[no_mangle]
pub extern "C" fn svm_par_child(
    prog: *mut bytecode::VcpuProgram,
    win_ptr: *mut u8,
    win_size: usize,
    module: u32,
    func: u32,
    sp: i64,
    arg: i64,
) -> *mut ParVcpu {
    if !par_vcpu_admit() {
        return core::ptr::null_mut();
    }
    // A thread shares its SPAWNER's window — which for a §14 confined spawner is its carve, not the
    // root window — so the mask derives from the passed window size (a power of two by construction:
    // the root window or a carve), not the guest module's declared memory. Equal for root-window
    // spawns; the confinement fix for carve spawns (CONSOLIDATION.md §11 slice 3).
    if !win_size.is_power_of_two() {
        par_vcpu_retire();
        return core::ptr::null_mut();
    }
    let sl = win_size.trailing_zeros() as u8;
    // SAFETY: the host guarantees `[win_ptr, win_size)` is the same live shared window.
    let back = std::sync::Arc::new(unsafe { svm_interp::Region::shared(win_ptr, win_size as u64) });
    let args = [Value::I64(sp), Value::I64(arg)];
    // SAFETY: `prog` is a live program pointer the host keeps alive for the run.
    match bytecode::Vcpu::new_child_sized(unsafe { prog_ref(prog) }, module, func, &args, back, sl)
    {
        Ok(inner) => {
            // A §22 **runtime-compile** run shares the JIT `Mutex<Host>` across every vCPU (mirroring
            // the root, `svm_par_root`), so a worker `thread.spawn`ed onto this Worker can `compile` /
            // `invoke` against the *same* domain: a unit compiled on any Worker is invokable here, and
            // its emitted bytes (in the shared host's heap = shared linear memory) are read locally and
            // instantiated per-Worker. Concurrent `compile`s serialize on the `Mutex` (DESIGN.md §22).
            if let Some(cfg) = par_jit_rt() {
                return par_box(inner.with_shared_host(&cfg.host));
            }
            // A 4d I/O run shares one powerbox across every vCPU (worker `cap.call` = host I/O).
            let inner = match par_io() {
                Some(io) => inner.with_shared_host(&io.host),
                None => inner,
            };
            par_box(with_tierup(inner))
        }
        Err(_) => {
            par_vcpu_retire();
            core::ptr::null_mut()
        }
    }
}

/// Build a §14 **confined executor child** vCPU (THREADS.md 4c-domain §14-D2) over the parent's carve
/// `[carve_ptr, carve_ptr + 2^size_log2)` — the operands of a [`PAR_INSTANTIATE`] event, shuttled
/// verbatim by the JS host (`carve_ptr` = the parent Worker's window pointer + the event's `carve`).
/// Per DESIGN.md §14 a sub-window is indistinguishable from a top-level window, so the carve region
/// simply *is* the child's window; the attenuated powerbox and the child's own dispatch table are
/// built in-engine ([`bytecode::Vcpu::new_confined_child`]) — no authority crosses JS. Called on the
/// child's Worker. Null on a bad module/entry.
#[no_mangle]
pub extern "C" fn svm_par_child_confined(
    prog: *mut bytecode::VcpuProgram,
    carve_ptr: *mut u8,
    size_log2: u32,
    module: u32,
    entry: u32,
    fuel: i64,
) -> *mut ParVcpu {
    if size_log2 >= 64 || !par_vcpu_admit() {
        return core::ptr::null_mut();
    }
    // SAFETY: the host guarantees the carve is inside the parent's live window (the engine validated
    // it before surfacing the event); aliasing views of the shared memory are the §13 data plane.
    let back =
        std::sync::Arc::new(unsafe { svm_interp::Region::shared(carve_ptr, 1u64 << size_log2) });
    // SAFETY: `prog` is a live program pointer the host keeps alive for the run.
    // (No shared-host attach: a §14 confined child's powerbox is its own attenuated one, built
    // in-engine — its capability set never includes the run's I/O grants.)
    match bytecode::Vcpu::new_confined_child(
        unsafe { prog_ref(prog) },
        module,
        entry,
        back,
        size_log2 as u8,
        fuel as u64,
    ) {
        Ok(inner) => par_box(inner),
        Err(_) => {
            par_vcpu_retire();
            core::ptr::null_mut()
        }
    }
}

/// Pointer / length of the accumulated stdout in the run's shared I/O powerbox (4d). Call `len`
/// **first** — it snapshots the buffer under the powerbox lock into a stable stash `ptr` then reads —
/// after the run completes (the root's `done`; a mid-run call sees a prefix). `0` when no
/// [`svm_par_powerbox_io`] was published.
#[no_mangle]
pub extern "C" fn svm_par_stdout_len() -> usize {
    let Some(io) = par_io() else { return 0 };
    let bytes = {
        let g = io.host.lock().unwrap_or_else(|e| e.into_inner());
        g.stdout.clone()
    };
    // SAFETY: the stash slot is only touched from the main thread (the JS host reads results after
    // the run), matching the `svm_run_pb` accessors' single-reader contract.
    unsafe { stash(&mut *core::ptr::addr_of_mut!(PAR_OUT), bytes) };
    unsafe { (*core::ptr::addr_of!(PAR_OUT)).1 }
}
#[no_mangle]
pub extern "C" fn svm_par_stdout_ptr() -> *const u8 {
    // SAFETY: as above — main-thread single-reader stash.
    unsafe { (*core::ptr::addr_of!(PAR_OUT)).0 }
}
/// The stashed 4d stdout snapshot (`svm_par_stdout_len` fills it; `_ptr` reads it).
static mut PAR_OUT: (*mut u8, usize) = (core::ptr::null_mut(), 0);

// ---- I22 diagnostics: capture a Rust panic's location+message ----------------------------------
// `panic = "abort"` lowers a Rust panic to a wasm `unreachable`, which reaches the JS host as a bare
// `[pageerror] unreachable` with no location — the exact signature of the Jul 12 nightly `real-browser`
// flake (ISSUES.md I22). A `unreachable` trap unwinds to the host but leaves the instance's memory
// intact, so a panic hook can stash the message here and the worker.js trap handler reads it back
// AFTER the trap via the accessors below. No new wasm import needed (the threads build instantiates
// with only `env.memory`). Alloc-free in the hook (formats into a stack buffer); the one heap alloc is
// the `Box`ed closure at install, once.
const PAR_PANIC_CAP: usize = 512;
static mut PAR_PANIC_BUF: [u8; PAR_PANIC_CAP] = [0; PAR_PANIC_CAP];
static PAR_PANIC_LEN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(target_arch = "wasm32")]
static PAR_PANIC_ONCE: std::sync::Once = std::sync::Once::new();

/// Install the panic-capture hook once per shared-memory image. wasm-only: on native this is a no-op
/// so the default hook (backtraces, `#[should_panic]` test output) is untouched.
fn par_install_panic_capture() {
    #[cfg(target_arch = "wasm32")]
    PAR_PANIC_ONCE.call_once(|| {
        std::panic::set_hook(Box::new(|info| {
            use std::io::Write;
            let mut buf = [0u8; PAR_PANIC_CAP];
            let mut cur = std::io::Cursor::new(&mut buf[..]);
            let _ = write!(cur, "{info}"); // Display = "panicked at FILE:LINE:COL:\nMESSAGE"; truncates on overflow
            let n = cur.position() as usize;
            // SAFETY: fixed static buffer. A concurrent double-panic may interleave bytes, but we only
            // need one legible message; publish `len` last (Release) so a reader sees a written prefix.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    buf.as_ptr(),
                    core::ptr::addr_of_mut!(PAR_PANIC_BUF) as *mut u8,
                    n,
                );
            }
            PAR_PANIC_LEN.store(n, std::sync::atomic::Ordering::Release);
        }));
    });
}

/// Pointer to the captured-panic buffer (read `svm_par_last_panic_len` bytes). Valid after any trap.
#[no_mangle]
pub extern "C" fn svm_par_last_panic_ptr() -> *const u8 {
    core::ptr::addr_of!(PAR_PANIC_BUF) as *const u8
}
/// Length of the last captured panic message (0 = none captured this image).
#[no_mangle]
pub extern "C" fn svm_par_last_panic_len() -> usize {
    PAR_PANIC_LEN.load(std::sync::atomic::Ordering::Acquire)
}

/// Advance the vCPU until it finishes, traps, or hits a host-serviced event; returns a `PAR_*` code.
/// The host reads operands via `svm_par_ev_a`–`d`, services the event, calls the matching `deliver`,
/// then calls `svm_par_run` again.
#[no_mangle]
pub extern "C" fn svm_par_run(v: *mut ParVcpu) -> i32 {
    par_install_panic_capture(); // I22: so a mid-run engine panic self-identifies (FILE:LINE) not a bare `unreachable`
                                 // SAFETY: `v` is a live `ParVcpu` from `svm_par_root`/`svm_par_child`, owned by this Worker.
    let v = unsafe { &mut *v };
    // Loop so §22 JIT events (serviced in-Rust against the shared powerbox) never surface to the JS
    // host — it only ever sees the multi-vCPU events `spawn`/`join`/`wait`/`notify` (+ `done`/`trap`).
    loop {
        match v.inner.run() {
            bytecode::VcpuEvent::Done(vals) => {
                v.a = first_i64(&vals);
                return PAR_DONE;
            }
            bytecode::VcpuEvent::Trapped(_) => return PAR_TRAP,
            // wasm-JIT tier-up: hand the func index + marshalled args to the Worker, which runs the
            // emitted `f{func}` and delivers the results (`svm_par_deliver_tierup`) or a trap.
            // Operand `b` carries the window's scalar committed extent — the Worker writes it to the
            // emitted module's `"mapped"` global before the call, so the emitted bounds check admits
            // exactly what the interpreter would over a `vm_map`-grown window (#717 host sync).
            bytecode::VcpuEvent::TierUp { func, argv, mapped } => {
                v.a = func as i64;
                if par_jit_paged() {
                    // #750: rebuild the page-state table from the live map (frozen while emitted
                    // code runs); operand `b` becomes the table's COVERAGE — the value the Worker
                    // writes to the emitted `"mapped"` global — and the table bytes are read via
                    // `svm_par_tierup_pagestate_ptr`/`_len` (their address in this module's linear
                    // memory IS the `"pagestate"` global's value: one shared memory, zero copies).
                    let info = v.inner.mem_map_info().unwrap_or((1, 0, 0, Vec::new()));
                    let (table, cover) = bytecode::build_pagestate_table(&info);
                    v.pagestate = table;
                    v.b = cover as i64;
                } else {
                    v.b = mapped as i64;
                }
                v.tierup_argv = argv.into_vec();
                return PAR_TIERUP;
            }
            bytecode::VcpuEvent::Spawn {
                func,
                sp,
                arg,
                module,
            } => {
                // Pack `(module << 32) | func` exactly as the INSTANTIATE event does: the spawning
                // frame's module rides to the child Worker so the child resolves `func` there (an
                // installed §22 unit spawning its own functions — CONSOLIDATION.md §11).
                v.a = ((module as i64) << 32) | func as i64;
                v.b = sp;
                v.c = arg;
                return PAR_SPAWN;
            }
            bytecode::VcpuEvent::Join { handle } => {
                v.a = handle as i64;
                return PAR_JOIN;
            }
            bytecode::VcpuEvent::Wait {
                addr,
                expected,
                width,
                timeout,
            } => {
                v.a = addr as i64;
                v.b = expected as i64;
                v.c = width as i64;
                v.d = timeout as i64;
                return PAR_WAIT;
            }
            bytecode::VcpuEvent::Notify { addr, count } => {
                v.a = addr as i64;
                v.b = count as i64;
                return PAR_NOTIFY;
            }
            // §22 guest-JIT serviced in-Rust against the shared powerbox + `Domain` (THREADS.md
            // 4c-domain C2): resolve the unit (the powerbox holds authority) and deliver it; the vCPU
            // installs / invokes against the shared `Domain`, then we loop. Without a powerbox (a
            // non-JIT run) a JIT op is fail-closed, exactly as before this seam existed.
            bytecode::VcpuEvent::JitInstall { handle, code } => {
                // Resolve the unit via whichever powerbox this run holds — fixed-unit (`par_pb`) or
                // runtime-compile (`par_jit_rt`). Both surface a `JitInstall` event; the runtime path
                // was previously unwired here (it would trap). The filled slot is recorded per code
                // handle so each Worker can mirror the shared `Domain` into its own `WebAssembly.Table`
                // (§22 Model B2 cross-Worker — funcrefs can't cross Workers).
                let resolved = if let Some(pb) = par_pb() {
                    par_resolve_unit(pb, handle, code)
                } else if let Some(cfg) = par_jit_rt() {
                    let g = cfg.host.lock().unwrap_or_else(|e| e.into_inner());
                    par_resolve_unit_rt(&g, handle, code).map(|(f, _)| f)
                } else {
                    return PAR_TRAP;
                };
                if let Some(slot) = v.inner.deliver_jit_install(resolved) {
                    par_jit_slot_record(slot, code);
                }
            }
            bytecode::VcpuEvent::JitUninstall { handle, .. } => {
                let authorized = if let Some(pb) = par_pb() {
                    pb.host.resolve_jit_domain(handle).map(|_| ())
                } else if let Some(cfg) = par_jit_rt() {
                    let g = cfg.host.lock().unwrap_or_else(|e| e.into_inner());
                    g.resolve_jit_domain(handle).map(|_| ())
                } else {
                    return PAR_TRAP;
                };
                if let Some(slot) = v.inner.deliver_jit_uninstall(authorized) {
                    par_jit_slot_clear(slot); // keep each Worker's table mirror exact (stale call traps)
                }
            }
            bytecode::VcpuEvent::JitInvoke {
                handle,
                code,
                argv,
                params,
                results,
                mapped,
            } => {
                // Scalar arg/result type codes (i32/i64/f32/f64) the JS host marshals each i64 slot
                // by; `None` if any operand is v128 (no lane marshalling — the unit stays on interp).
                let codes = |ts: &[svm_ir::ValType]| {
                    ts.iter()
                        .map(|t| scalar_type_code(*t))
                        .collect::<Option<Vec<u8>>>()
                };
                let (ptypes, rtypes) = (codes(&params), codes(&results));
                // #717 host sync: operand `b` carries the committed extent for the Worker to write
                // to the emitted unit's `"mapped"` global before `f0`. An unrepresentable window
                // state (`None`) declines codegen below — the interpreted delivery honors the full
                // page map (fail-closed, same contract as PAR_TIERUP).
                if let Some(h) = mapped {
                    v.b = h as i64;
                }
                if let Some(cfg) = par_jit_rt() {
                    // §22 **runtime-compile** path: the guest compiled its *own* unit into the shared
                    // host, its wasm emitted there; run it on that per-unit wasm (JS instantiates it
                    // keyed by the code handle, reading `svm_par_jit_code_wasm_ptr`/`_len`). Authority
                    // resolves through the same host — a forged / cross-domain handle traps identically.
                    // Codegen off / v128 / a unit outside the emitter subset ⇒ the interpreter services it.
                    let resolved = {
                        let g = cfg.host.lock().unwrap_or_else(|e| e.into_inner());
                        par_resolve_unit_rt(&g, handle, code)
                    };
                    match resolved {
                        Err(t) => v.inner.deliver_jit_invoke(Err(t)),
                        Ok((funcs, wasm)) => {
                            let codegen = par_jit_codegen()
                                && wasm.is_some()
                                && ptypes.is_some()
                                && rtypes.is_some()
                                && mapped.is_some();
                            if codegen {
                                v.jit_argv = argv.into_vec();
                                v.jit_code = code;
                                v.jit_param_types = ptypes.unwrap();
                                v.jit_result_types = rtypes.unwrap();
                                v.jit_wasm = wasm;
                                return PAR_JIT_INVOKE;
                            }
                            v.inner.deliver_jit_invoke(Ok(funcs));
                        }
                    }
                } else {
                    // Fixed-unit codegen path (slice 5): the run's single unit was host-compiled at
                    // setup and emitted to the run-wide `JIT_UNIT_WASM` stash. Authority still resolves
                    // through the powerbox — a forged / cross-domain handle must trap identically.
                    match par_pb() {
                        None => return PAR_TRAP,
                        Some(pb) => {
                            let codegen = par_jit_codegen()
                                && svm_par_jit_unit_wasm_len() > 0
                                && ptypes.is_some()
                                && rtypes.is_some()
                                && mapped.is_some();
                            if codegen {
                                match par_resolve_unit(pb, handle, code) {
                                    Ok(_) => {
                                        v.jit_argv = argv.into_vec();
                                        v.jit_code = code;
                                        v.jit_param_types = ptypes.unwrap();
                                        v.jit_result_types = rtypes.unwrap();
                                        return PAR_JIT_INVOKE;
                                    }
                                    Err(t) => v.inner.deliver_jit_invoke(Err(t)),
                                }
                            } else {
                                v.inner
                                    .deliver_jit_invoke(par_resolve_unit(pb, handle, code));
                            }
                        }
                    }
                }
            }
            // §14 confined executor child (THREADS.md 4c-domain §14-D2): all authority-bearing work
            // already happened in-Vm — the operands are inert integers the JS host shuttles into a
            // new Worker running `svm_par_child_confined` over `[win + carve, +2^size_log2)`, joined
            // through the same completion-slot protocol as `PAR_SPAWN`.
            bytecode::VcpuEvent::Instantiate {
                module,
                entry,
                carve,
                size_log2,
                fuel,
            } => {
                v.a = ((module as i64) << 32) | entry as i64;
                v.b = carve as i64;
                v.c = size_log2 as i64;
                v.d = fuel as i64;
                return PAR_INSTANTIATE;
            }
            // Blocking stdin is a single-threaded interactive-session feature (the Postgres console
            // runs on its own owned-host `Vcpu`, not the parallel driver); a worker vCPU never sets it.
            bytecode::VcpuEvent::StdinPark => return PAR_TRAP,
        }
    }
}

macro_rules! par_ev_getter {
    ($name:ident, $field:ident) => {
        /// Read an operand of the last [`svm_par_run`] event.
        #[no_mangle]
        pub extern "C" fn $name(v: *mut ParVcpu) -> i64 {
            // SAFETY: `v` is a live `ParVcpu` owned by this Worker.
            unsafe { (*v).$field }
        }
    };
}
par_ev_getter!(svm_par_ev_a, a);
par_ev_getter!(svm_par_ev_b, b);
par_ev_getter!(svm_par_ev_c, c);
par_ev_getter!(svm_par_ev_d, d);

/// Deliver a `thread.spawn` handle (after `PAR_SPAWN`).
#[no_mangle]
pub extern "C" fn svm_par_deliver_handle(v: *mut ParVcpu, handle: i32) {
    // SAFETY: `v` is a live `ParVcpu` awaiting a delivery.
    unsafe { (*v).inner.deliver_handle(handle) };
}

/// Deliver a `memory.wait` code / `memory.notify` count (after `PAR_WAIT` / `PAR_NOTIFY`).
#[no_mangle]
pub extern "C" fn svm_par_deliver_code(v: *mut ParVcpu, code: i32) {
    // SAFETY: `v` is a live `ParVcpu` awaiting a delivery.
    unsafe { (*v).inner.deliver_code(code) };
}

/// Deliver a joined child's result (after `PAR_JOIN`): `val` is its first return value, or — if
/// `is_trap != 0` — the child trapped and the joiner traps on its next `svm_par_run`.
#[no_mangle]
pub extern "C" fn svm_par_deliver_join(v: *mut ParVcpu, val: i64, is_trap: i32) {
    // SAFETY: `v` is a live `ParVcpu` awaiting a delivery.
    let v = unsafe { &mut *v };
    if is_trap != 0 {
        v.inner.deliver_join(Err(Trap::ThreadFault));
    } else {
        v.inner.deliver_join(Ok(vec![Value::I64(val)]));
    }
}

/// Pointer to the marshalled tier-up args (raw i64 slots) after a [`PAR_TIERUP`] event — the Worker
/// reads `svm_par_tierup_argv_len` of them to call the emitted `f{func}`.
#[no_mangle]
pub extern "C" fn svm_par_tierup_argv_ptr(v: *mut ParVcpu) -> *const i64 {
    // SAFETY: `v` is a live `ParVcpu`; the buffer lives until the next event overwrites it.
    unsafe { (*v).tierup_argv.as_ptr() }
}

/// #750: the pending [`PAR_TIERUP`]'s page-state table base (paged runs only — see
/// [`svm_par_enable_jit_paged`]). The bytes live in this module's linear memory, so this pointer
/// is exactly the value the Worker writes to the emitted module's `"pagestate"` global.
#[no_mangle]
pub extern "C" fn svm_par_tierup_pagestate_ptr(v: *mut ParVcpu) -> *const u8 {
    // SAFETY: `v` is a live `ParVcpu`; the buffer lives until the next event overwrites it.
    unsafe { (*v).pagestate.as_ptr() }
}

/// Byte length of the pending tier-up's page-state table (`0` on an unpaged run — the Worker
/// skips the `"pagestate"` write, which the unpaged module doesn't export anyway).
#[no_mangle]
pub extern "C" fn svm_par_tierup_pagestate_len(v: *mut ParVcpu) -> usize {
    // SAFETY: `v` is a live `ParVcpu`.
    unsafe { (*v).pagestate.len() }
}

/// Number of tier-up args (see [`svm_par_tierup_argv_ptr`]).
#[no_mangle]
pub extern "C" fn svm_par_tierup_argv_len(v: *mut ParVcpu) -> usize {
    // SAFETY: `v` is a live `ParVcpu`.
    unsafe { (*v).tierup_argv.len() }
}

/// Deliver the results of a tier-up region (after `PAR_TIERUP`): `[results_ptr, n)` are the emitted
/// `f{func}`'s i64 result slots. The vCPU resumes with them in the awaiting call's dst.
#[no_mangle]
pub extern "C" fn svm_par_deliver_tierup(v: *mut ParVcpu, results_ptr: *const i64, n: usize) {
    // SAFETY: `v` is a live `ParVcpu` awaiting a delivery; `[results_ptr, n)` is a live host buffer.
    let v = unsafe { &mut *v };
    let results = unsafe { core::slice::from_raw_parts(results_ptr, n) };
    v.inner.deliver_tierup(results);
}

/// Deliver a **trap** from a tier-up region (the emitted `f{func}` threw — memory fault / fuel /
/// div-by-zero / `unreachable`). The vCPU traps on its next `svm_par_run`, as if interp had trapped.
#[no_mangle]
pub extern "C" fn svm_par_deliver_tierup_trap(v: *mut ParVcpu) {
    // SAFETY: `v` is a live `ParVcpu` awaiting a delivery.
    unsafe { (*v).inner.deliver_tierup_trap(Trap::Unreachable) };
}

/// The code handle of a pending [`PAR_JIT_INVOKE`] — the Worker keys its per-unit emitted-instance
/// cache by this (one wasm instance per submitted unit; args differ per invoke).
#[no_mangle]
pub extern "C" fn svm_par_jit_code(v: *mut ParVcpu) -> i32 {
    // SAFETY: `v` is a live `ParVcpu`.
    unsafe { (*v).jit_code }
}

/// Pointer to the marshalled §22 invoke args (raw i64 slots) after a [`PAR_JIT_INVOKE`] event — the
/// Worker reads `svm_par_jit_argv_len` of them to call the emitted unit's `f{entry}`.
#[no_mangle]
pub extern "C" fn svm_par_jit_argv_ptr(v: *mut ParVcpu) -> *const i64 {
    // SAFETY: `v` is a live `ParVcpu`; the buffer lives until the next event overwrites it.
    unsafe { (*v).jit_argv.as_ptr() }
}

/// Number of §22 invoke args (see [`svm_par_jit_argv_ptr`]).
#[no_mangle]
pub extern "C" fn svm_par_jit_argv_len(v: *mut ParVcpu) -> usize {
    // SAFETY: `v` is a live `ParVcpu`.
    unsafe { (*v).jit_argv.len() }
}

/// Per-arg **scalar type codes** of a pending [`PAR_JIT_INVOKE`] (`0` = i32, `1` = i64, `2` = f32,
/// `3` = f64), one byte per arg — the Worker reads them to marshal each i64 slot to the wasm type the
/// emitted `f{entry}` uses. Length equals [`svm_par_jit_argv_len`].
#[no_mangle]
pub extern "C" fn svm_par_jit_param_types_ptr(v: *mut ParVcpu) -> *const u8 {
    // SAFETY: `v` is a live `ParVcpu`; the buffer lives until the next event overwrites it.
    unsafe { (*v).jit_param_types.as_ptr() }
}

/// Per-result **scalar type codes** of a pending [`PAR_JIT_INVOKE`] (same encoding as
/// [`svm_par_jit_param_types_ptr`]) — the Worker marshals each emitted-`f{entry}` result back to its
/// i64 result slot (a float's *bits*, an integer's value) for [`svm_par_deliver_jit_invoke`].
#[no_mangle]
pub extern "C" fn svm_par_jit_result_types_ptr(v: *mut ParVcpu) -> *const u8 {
    // SAFETY: `v` is a live `ParVcpu`; the buffer lives until the next event overwrites it.
    unsafe { (*v).jit_result_types.as_ptr() }
}

/// Number of §22 invoke results (see [`svm_par_jit_result_types_ptr`]).
#[no_mangle]
pub extern "C" fn svm_par_jit_result_types_len(v: *mut ParVcpu) -> usize {
    // SAFETY: `v` is a live `ParVcpu`.
    unsafe { (*v).jit_result_types.len() }
}

/// Pointer / length of a pending [`PAR_JIT_INVOKE`]'s **runtime-compiled** unit wasm (the shared-host
/// path, [`svm_par_powerbox_jit_runtime`]): the JS host instantiates it once and caches the instance by
/// [`jit_code`](ParVcpu::jit_code). `(null, 0)` for the fixed-unit codegen path (that reads the run-wide
/// [`svm_par_jit_unit_wasm_ptr`] stash instead). The bytes stay valid until the next `svm_par_run`.
#[no_mangle]
pub extern "C" fn svm_par_jit_code_wasm_ptr(v: *mut ParVcpu) -> *const u8 {
    // SAFETY: `v` is a live `ParVcpu`.
    unsafe {
        (*v).jit_wasm
            .as_ref()
            .map_or(core::ptr::null(), |w| w.as_ptr())
    }
}
#[no_mangle]
pub extern "C" fn svm_par_jit_code_wasm_len(v: *mut ParVcpu) -> usize {
    // SAFETY: `v` is a live `ParVcpu`.
    unsafe { (*v).jit_wasm.as_ref().map_or(0, |w| w.len()) }
}

/// Deliver the results of a §22 unit run on emitted wasm (after `PAR_JIT_INVOKE`): `[results_ptr, n)`
/// are the emitted `f{entry}`'s i64 result slots. The vCPU resumes with them in the invoke's dst —
/// identical to the interpreter having run the unit.
#[no_mangle]
pub extern "C" fn svm_par_deliver_jit_invoke(v: *mut ParVcpu, results_ptr: *const i64, n: usize) {
    // SAFETY: `v` is a live `ParVcpu` awaiting a delivery; `[results_ptr, n)` is a live host buffer.
    let v = unsafe { &mut *v };
    let results = unsafe { core::slice::from_raw_parts(results_ptr, n) };
    v.inner.deliver_jit_invoke_vals(results);
}

/// Deliver a **trap** from a §22 unit run on emitted wasm (the emitted region threw). The vCPU traps
/// on its next `svm_par_run`, as if the interpreted invoke had trapped.
#[no_mangle]
pub extern "C" fn svm_par_deliver_jit_invoke_trap(v: *mut ParVcpu) {
    // SAFETY: `v` is a live `ParVcpu` awaiting a delivery.
    unsafe { (*v).inner.deliver_jit_invoke_trap(Trap::Unreachable) };
}

/// Free a finished vCPU.
#[no_mangle]
pub extern "C" fn svm_par_free(v: *mut ParVcpu) {
    if !v.is_null() {
        // SAFETY: `v` came from `Box::into_raw` in `svm_par_root`/`svm_par_child` and is freed once.
        drop(unsafe { Box::from_raw(v) });
        par_vcpu_retire(); // the live-cap admit from this vCPU's constructor
    }
}

// ---- host powerbox: console + clock, marshalled through host-allocated memory ----------------
//
// Beyond compute-only: grant the guest a real capability set (stdin/stdout/stderr streams, a
// monotonic clock, and exit). The `Host` powerbox is already self-contained and **deterministic** —
// stream writes accumulate in `Host::stdout`/`stderr`, `read` draws from `Host::stdin`, and
// `Clock.now` is a strictly-increasing counter — so no wasm host *imports* are needed: I/O crosses
// the boundary the same way the module does, through `svm_alloc`ed memory. The host writes stdin to
// an allocation it passes in; the captured streams come back as cdylib-managed allocations the host
// reads (via the `*_ptr`/`*_len` exports) before the next call. The cdylib stays import-free.

/// The guest called `Exit.exit(code)` (a non-error trap); read the code via [`svm_exit_code`].
pub const STATUS_EXIT: i32 = 5;

/// A captured RGBA framebuffer a guest presented through the `display` capability: `width`×`height`
/// pixels, `rgba` exactly `width*height*4` bytes (R,G,B,A per pixel, row-major, top row first — the
/// `<canvas>` `ImageData` layout, so the browser blits it with a single `putImageData`). This is the
/// foundation of the graphical demos (the framebuffer output path Doom rides).
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// Outcome of a [`powerbox_exec`] run: the status (a `STATUS_*` code), the `i64`-widened return value
/// (when `STATUS_OK`), the exit code (when `STATUS_EXIT`), the bytes the guest wrote to its stdout /
/// stderr streams, and the last framebuffer it presented via the `display` capability (`None` if it
/// presented none — the common case; only the graphical on-ramp guests use it).
pub struct PbOutcome {
    pub status: i32,
    pub value: i64,
    pub exit_code: i32,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub framebuffer: Option<Frame>,
}

/// The canonical names of the browser powerbox's capabilities, in grant order — the vocabulary a
/// powerbox guest resolves against via `cap.self.resolve` (F7) / labels via `cap.self.label` (F9). The
/// browser ABI grants `(stdout, stdin, exit, stderr, clock)` by arity (its set differs from `svm-run`'s
/// fixed §3e prefix after slot 3, since the capabilities differ), so the names follow that order.
const POWERBOX_CAP_NAMES: [&str; 5] = ["stdout", "stdin", "exit", "stderr", "clock"];

/// Run `m`'s function 0 under the **browser powerbox**, seeding `stdin` and capturing the streams.
///
/// Capabilities are granted by the entry's **arity** (so `hello.svmt`'s 3-handle `(out, in, exit)`
/// shape works unchanged), in this order — the browser embedder's ABI:
///
/// | param # | capability        | `cap.call` type_id |
/// |---------|-------------------|--------------------|
/// | 1       | `Stream(Out)`     | 0 (op 1 = write)   |
/// | 2       | `Stream(In)`      | 0 (op 0 = read)    |
/// | 3       | `Exit`            | 1 (op 0 = exit)    |
/// | 4       | `Stream(Err)`     | 0 (op 1 = write)   |
/// | 5       | `Clock`           | 2 (op 0 = now)     |
///
/// Shared verbatim by the wasm [`svm_run_pb`] export and the native `gencorpus` ground truth, so the
/// differential compares the *same* logic on both builds.
pub fn powerbox_exec(m: &svm_ir::Module, stdin: &[u8]) -> PbOutcome {
    let arity = m.funcs.first().map_or(0, |f| f.params.len());
    let mut host = Host::new();
    host.stdin = stdin.to_vec();
    let mut slots: Vec<Value> = Vec::new();
    if arity >= 1 {
        slots.push(Value::I32(host.grant_stream(StreamRole::Out)));
    }
    if arity >= 2 {
        slots.push(Value::I32(host.grant_stream(StreamRole::In)));
    }
    if arity >= 3 {
        slots.push(Value::I32(host.grant_exit()));
    }
    if arity >= 4 {
        slots.push(Value::I32(host.grant_stream(StreamRole::Err)));
    }
    if arity >= 5 {
        slots.push(Value::I32(host.grant_clock()));
    }
    // §7 register each granted capability under its canonical name (F7/F9, PR #118) so a guest can
    // `cap.self.resolve` / `cap.self.label` it at runtime — mirroring `svm-run`'s powerbox so the
    // browser stays a faithful twin. Names parallel the grant order above; only the `arity` actually
    // granted are registered.
    for (name, slot) in POWERBOX_CAP_NAMES.iter().zip(&slots) {
        if let Value::I32(handle) = slot {
            host.register_cap_name(name, *handle);
        }
    }
    let mut fuel = u64::MAX;
    let (status, value, exit_code) =
        match bytecode::compile_and_run_with_host(m, 0, &slots, &mut fuel, &mut host) {
            None => (STATUS_UNSUPPORTED, 0, 0),
            Some(Err(Trap::Exit(code))) => (STATUS_EXIT, 0, code),
            Some(Err(_)) => (STATUS_TRAP, 0, 0),
            Some(Ok(vals)) => match vals.first() {
                Some(Value::I64(x)) => (STATUS_OK, *x, 0),
                Some(Value::I32(x)) => (STATUS_OK, *x as i64, 0),
                _ => (STATUS_BAD_RESULT, 0, 0),
            },
        };
    PbOutcome {
        status,
        value,
        exit_code,
        stdout: host.stdout,
        stderr: host.stderr,
        framebuffer: None, // the browser-corpus powerbox grants no `display` cap
    }
}

/// The canonical names of the **on-ramp** powerbox prefix, in grant order — the fixed §3e `VM_CAP_*`
/// vocabulary the LLVM on-ramp's synthesized `_start` expects (and `svm-run` grants). This differs
/// from [`POWERBOX_CAP_NAMES`] after slot 3: the hand-written browser corpus uses `(stderr, clock)`
/// at slots 4/5, but an on-ramp guest wants `(memory, addrspace)` there — `memory` is what `malloc`
/// grows the heap through, so Lua/SQLite need it. See `LLVM.md` §N (the powerbox on-ramp).
const ONRAMP_CAP_NAMES: [&str; 5] = ["stdout", "stdin", "exit", "memory", "addrspace"];

/// The reference host's §7 capability-import name policy — a browser-side twin of `svm-run`'s
/// `default_cap_resolver`. The on-ramp emits `call.sym "<name>"` for each libc→capability shim
/// (`write`/`read`/`exit`/`vm_map`/…); this lowers each name to the `(type_id, op)` its `cap.call`
/// runs, so the resolved module verifies and runs. The **handle** (which stream/region) is supplied
/// by the powerbox stash, not this map — `write`/`read` share `Stream`, differing only by handle.
fn onramp_cap_resolver(name: &str) -> Option<svm_ir::ResolvedCap> {
    use svm_interp::cap_id;
    let (type_id, op): (u32, u32) = match name {
        "write" => (cap_id::STREAM, 1),
        "read" => (cap_id::STREAM, 0),
        "exit" => (cap_id::EXIT, 0),
        "vm_map" => (cap_id::ADDRESS_SPACE, 0),
        "vm_unmap" => (cap_id::ADDRESS_SPACE, 1),
        "vm_protect" => (cap_id::ADDRESS_SPACE, 2),
        "vm_page_size" => (cap_id::ADDRESS_SPACE, 3),
        "vm_region_create" => (cap_id::ADDRESS_SPACE, 5),
        "vm_region_map" => (cap_id::SHARED_REGION, 0),
        "vm_region_unmap" => (cap_id::SHARED_REGION, 1),
        "vm_region_page_size" => (cap_id::SHARED_REGION, 3),
        // Guest-driven JIT (§22) — the macro-staging on-ramp grants the Jit cap; mirrors
        // svm-run's default_cap_resolver so a compiler-guest's `__vm_jit_*` builtins bind.
        "vm_jit_compile" => (cap_id::JIT, 0),
        "vm_jit_compile_linked" => (cap_id::JIT, 5),
        "vm_jit_invoke2" => (cap_id::JIT, 1),
        "vm_jit_release" => (cap_id::JIT, 2),
        "vm_jit_install" => (cap_id::JIT, 3),
        "vm_jit_uninstall" => (cap_id::JIT, 4),
        _ => return None,
    };
    Some(svm_ir::ResolvedCap { type_id, op })
}

/// Gate an on-ramp module (IMPORTS.md phase 4): the runtime never rewrites. A module that declares
/// imports must carry the **powerbox entry shape** — a paramless func 0 exported as `_start`
/// (`svm-run`'s `is_named_powerbox_entry`) — so its manifest slots can bind at instantiation
/// ([`grant_onramp_caps`] installs the bindings; `call.import` dispatches through them). An
/// import-bearing module without that shape **fails closed** (the pre-manifest `resolve_imports`
/// rewrite died with phase 4). An import-free module passes as-is: its entry runs with no args
/// (missing params zero-seed, the `Session` convention) and reaches capabilities only by name via
/// `cap.self.resolve`.
fn onramp_check(m: &svm_ir::Module) -> Result<(), ()> {
    let named_entry = m.funcs.first().is_some_and(|f| f.params.is_empty())
        && m.exports.iter().any(|e| e.name == "_start" && e.func == 0);
    if m.imports.is_empty() || named_entry {
        Ok(())
    } else {
        Err(())
    }
}

/// A shared **keyboard event queue** (the `keyboard` capability's backing): the host pushes packed
/// key events, the guest drains them via `__vm_cap_resolve("keyboard")` + `poll`. `Arc<Mutex<…>>` so
/// the cap's `HostProc` closure and the host/reactor driver share one queue. Packed event layout:
/// `(pressed << 16) | (keycode & 0xffff)` — `pressed` is 1 (down) / 0 (up); `poll` returns `-1` when
/// empty (the doomgeneric `DG_GetKey` shape: pump until empty each frame).
type KeyQueue = std::sync::Arc<std::sync::Mutex<std::collections::VecDeque<i32>>>;

/// Grant the **on-ramp powerbox** onto `host` for module `m`: the §3e prefix
/// (`stdout, stdin, exit, memory, addrspace`), each registered under its `cap.self.resolve` name,
/// plus the two by-name graphical `HostProc` capabilities every on-ramp run carries — `display` (op 0 =
/// `present(ptr, w, h)`, copies `w*h*4` RGBA bytes out of the window into the returned frame cell) and
/// `keyboard` (op 0 = `poll()`, dequeues one packed event from the returned queue, or `-1`).
///
/// The on-ramp entry is the phase-4 powerbox shape (mirroring `svm-run`'s `grant_caps`): a
/// **paramless** `_start` whose manifest imports bind to slot bindings at instantiation, and which
/// resolves any further capability by name via `cap.self.resolve`. The whole prefix is granted and
/// registered under its canonical names; a guest resolves only the names it uses, so registering
/// the full prefix is a harmless superset — one that resolves neither graphical cap is unaffected
/// (the queue stays empty, the frame cell `None`). The entry receives **no** handle arguments (the
/// positional slot-order delivery died in phase 4). Shared by [`onramp_exec`] and the per-frame
/// [`OnrampReactor`], so both grant the identical powerbox.
fn grant_onramp_caps(
    host: &mut Host,
    m: &svm_ir::Module,
    fs: Option<(String, Vec<u8>)>,
) -> (std::sync::Arc<std::sync::Mutex<Option<Frame>>>, KeyQueue) {
    let win = m.memory.map_or(0, |mc| 1u64 << mc.size_log2);
    let handles: [i32; 5] = [
        host.grant_stream(StreamRole::Out),
        host.grant_stream(StreamRole::In),
        host.grant_exit(),
        host.grant_memory(),
        host.grant_address_space(0, win),
    ];
    for (name, handle) in ONRAMP_CAP_NAMES.iter().zip(&handles) {
        host.register_cap_name(name, *handle);
    }
    // §22 guest-driven JIT: grant the `Jit` cap **iff** the guest declares a `__vm_jit_*` import
    // (principle of least authority — a plain on-ramp guest gets no Jit). The JACL self-hosted
    // compiler uses it to expand macros in-guest. Match svm-run's powerbox grant so a self-hosted
    // guest behaves identically: a 1024-slot dispatch table (a staged unit's `Slot` imports call back
    // into the host program's ~800 functions by index) and fiber hosting (a staged macro runs on the
    // compiler's scheduler root, which suspends). `browser_jit_validator` verifies every submitted
    // unit — the security hinge, so this stays "as secure as wasm".
    let jit_h: Option<i32> = if m.imports.iter().any(|im| im.name.starts_with("vm_jit_")) {
        let h = host.grant_jit_with_table(m.memory.map(|mc| mc.size_log2), ONRAMP_JIT_TABLE_LOG2);
        host.set_jit_validator(browser_jit_validator);
        host.set_jit_hosts_fibers(true);
        Some(h)
    } else {
        None
    };
    // IMPORTS.md phase 4: a manifest-carrying module executes its `call.import`s through
    // instantiation-time slot bindings — import `i`'s name maps to `(type_id, op)` via the
    // on-ramp policy and to the granted handle by interface. A name outside the policy (or the
    // dynamic-only SharedRegion ops) leaves its slot unbound — fail-closed at dispatch.
    if !m.imports.is_empty() {
        use svm_interp::cap_id;
        let bindings = m
            .imports
            .iter()
            .map(|im| {
                let Some(cap) = onramp_cap_resolver(&im.name) else {
                    return svm_interp::BoundImport::rebindable(0, 0, None);
                };
                let handle = match (cap.type_id, cap.op) {
                    (cap_id::STREAM, 1) => handles[0],
                    (cap_id::STREAM, _) => handles[1],
                    (cap_id::EXIT, _) => handles[2],
                    // One kind post-§4 (op-keyed like Stream): vm_map family → the
                    // whole-window grant, sub/region_create → the sized one.
                    (cap_id::ADDRESS_SPACE, 0..=3) => handles[3],
                    (cap_id::ADDRESS_SPACE, _) => handles[4],
                    (cap_id::JIT, _) => match jit_h {
                        Some(h) => h,
                        None => return svm_interp::BoundImport::rebindable(0, 0, None),
                    },
                    _ => return svm_interp::BoundImport::rebindable(0, 0, None),
                };
                svm_interp::BoundImport::required(cap.type_id, cap.op, handle)
            })
            .collect();
        host.set_import_bindings(bindings);
    }
    // `display` — the framebuffer output waist (Doom slice 1). `present(ptr, w, h)` copies the frame out.
    let frame: std::sync::Arc<std::sync::Mutex<Option<Frame>>> =
        std::sync::Arc::new(std::sync::Mutex::new(None));
    {
        let frame = std::sync::Arc::clone(&frame);
        let handle = host.grant_host_proc(Box::new(move |op, args, mem, _| {
            if op != 0 {
                return Ok(vec![-1]); // only present(0) is defined
            }
            let ptr = args.first().copied().unwrap_or(0);
            let w = args.get(1).copied().unwrap_or(0);
            let h = args.get(2).copied().unwrap_or(0);
            // Bound the dimensions so a bad (or hostile) call can't ask us to read/allocate wildly.
            if !(1..=8192).contains(&w) || !(1..=8192).contains(&h) {
                return Ok(vec![-1]);
            }
            let n = (w as u64) * (h as u64) * 4;
            match mem.and_then(|m| m.read_bytes(ptr as u64, n)) {
                Some(rgba) => {
                    *frame.lock().unwrap() = Some(Frame {
                        width: w as u32,
                        height: h as u32,
                        rgba,
                    });
                    Ok(vec![0])
                }
                None => Ok(vec![-1]), // ptr/len outside the window
            }
        }));
        host.register_cap_name("display", handle);
    }
    // `keyboard` — the input waist (Doom slice 2). `poll()` dequeues one packed event, or `-1` if empty.
    let keys: KeyQueue =
        std::sync::Arc::new(std::sync::Mutex::new(std::collections::VecDeque::new()));
    {
        let keys = std::sync::Arc::clone(&keys);
        let handle = host.grant_host_proc(Box::new(move |op, _args, _mem, _| {
            if op != 0 {
                return Ok(vec![-1]); // only poll(0) is defined
            }
            Ok(vec![keys
                .lock()
                .unwrap()
                .pop_front()
                .map_or(-1, |e| e as i64)])
        }));
        host.register_cap_name("keyboard", handle);
    }
    // `webgpu` — a GPU render surface, serviced (in the browser) against `navigator.gpu` via the
    // `webgpu_op` host import (`src/webgpu.rs`). The guest ships a WGSL shader once (op 0) and asks the
    // host to present a frame each tick (op 1); the parallel pixel work runs on the GPU and only tiny
    // scalars + the shader source cross the boundary — the guest never holds a GPU pointer (§2a). Only
    // granted in the wasm build (native has no GPU import); a guest resolves `-1` and skips elsewhere.
    #[cfg(target_arch = "wasm32")]
    {
        let handle = host.grant_host_proc(Box::new(move |op, args, mem, _| {
            match op {
                // set_shader(wgsl_ptr, wgsl_len) → 0 (compiled) / -1 (bad ptr or compile error)
                0 => {
                    let ptr = args.first().copied().unwrap_or(0);
                    let len = args.get(1).copied().unwrap_or(0);
                    if !(0..=1 << 20).contains(&len) {
                        return Ok(vec![-1]);
                    }
                    let Some(wgsl) = mem.and_then(|m| m.read_bytes(ptr as u64, len as u64)) else {
                        return Ok(vec![-1]);
                    };
                    // SAFETY: wasm-only import; `wgsl` outlives the synchronous call.
                    let r = unsafe { webgpu::webgpu_op(0, 0, 0, 0, wgsl.as_ptr(), wgsl.len()) };
                    Ok(vec![r])
                }
                // present(frame, w, h) → 0
                1 => {
                    let a = args.first().copied().unwrap_or(0);
                    let b = args.get(1).copied().unwrap_or(0);
                    let c = args.get(2).copied().unwrap_or(0);
                    let r = unsafe { webgpu::webgpu_op(1, a, b, c, core::ptr::null(), 0) };
                    Ok(vec![r])
                }
                _ => Ok(vec![-1]),
            }
        }));
        host.register_cap_name("webgpu", handle);
    }
    // `fs` — a read-only in-memory file (Doom slice 4: the WAD read path). Granted only when the host
    // supplies one file; a guest that resolves no `fs` cap (bounce/life) is unaffected. The op
    // protocol mirrors the native `doom_diff` differential's in-memory WAD server (and the reused
    // `lua_files_stdio.c` FILE shim): 0 open(nameptr,namelen,flags)→fd|-2(ENOENT), 1 read(fd,buf,len)
    // →n, 3 seek(fd,whence,off)→pos (whence 0=SET/1=CUR/2=END), 2 write(fd,…)→len (discard-accept),
    // 4 close→0. `fd` indexes a per-open cursor, so a guest that opens the file more than once is fine.
    if let Some((name, data)) = fs {
        let mut cursors: Vec<u64> = Vec::new();
        let handle = host.grant_host_proc(Box::new(move |op, args, mem, _| match op {
            0 => {
                let requested = mem
                    .and_then(|m| m.read_bytes(args[0] as u64, args[1] as u64))
                    .unwrap_or_default();
                if String::from_utf8_lossy(&requested).contains(name.as_str()) {
                    cursors.push(0);
                    Ok(vec![(cursors.len() - 1) as i64]) // fd = index into `cursors`
                } else {
                    Ok(vec![-2]) // ENOENT → the guest's fopen returns NULL (defaults/skips)
                }
            }
            1 => {
                let (buf, want) = (args[1] as u64, args[2] as u64);
                let Some(&cur) = cursors.get(args[0] as usize) else {
                    return Ok(vec![-1]);
                };
                let end = (cur + want).min(data.len() as u64);
                if end > cur {
                    if let Some(mem) = mem {
                        let _ = mem.write_bytes(buf, &data[cur as usize..end as usize]);
                    }
                }
                cursors[args[0] as usize] = end;
                Ok(vec![(end - cur) as i64])
            }
            3 => {
                let (whence, off) = (args[1], args[2]);
                let Some(cur) = cursors.get(args[0] as usize).copied() else {
                    return Ok(vec![-1]);
                };
                let base = match whence {
                    1 => cur as i64,
                    2 => data.len() as i64,
                    _ => 0,
                };
                cursors[args[0] as usize] = (base + off).max(0) as u64;
                Ok(vec![cursors[args[0] as usize] as i64])
            }
            2 => Ok(vec![args.get(2).copied().unwrap_or(0)]), // write: discard-accept
            _ => Ok(vec![0]),                                 // close et al.
        }));
        host.register_cap_name("fs", handle);
    }
    (frame, keys)
}

/// Run `m`'s function 0 under the **on-ramp powerbox** — the ABI `svm-llvm`'s synthesized `_start`
/// expects, so a `.svmb` straight off `svm-llvm-translate` (Lua, SQLite, …) runs unchanged. This is
/// the twin of [`powerbox_exec`] with the fixed §3e `VM_CAP_*` grant prefix instead of the browser
/// corpus's `(…, stderr, clock)` set: [`grant_onramp_caps`] grants `stdout, stdin, exit, memory,
/// addrspace` (mirroring `svm-run`'s `grant_powerbox_prefix`) and registers each under its name, and
/// the by-name `_start` resolves what it needs via `cap.self.resolve`.
///
/// The entry is the phase-4 powerbox shape ([`onramp_check`]): a paramless `_start` whose manifest
/// imports bind at instantiation, taking **no** handle arguments — the positional (slot-order
/// handle-args) entry form died in phase 4 and an import-bearing module without the manifest entry
/// shape is fail-closed (`STATUS_UNSUPPORTED`). The `fs` capability (SQLite Phase B, Lua
/// `files.lua`) is a `host_proc` resolved by name — a Stage-1 follow-on, not part of this prefix.
pub fn onramp_exec(m: &svm_ir::Module, stdin: &[u8]) -> PbOutcome {
    let unsupported = || PbOutcome {
        status: STATUS_UNSUPPORTED,
        value: 0,
        exit_code: 0,
        stdout: Vec::new(),
        stderr: Vec::new(),
        framebuffer: None,
    };
    if onramp_check(m).is_err() {
        return unsupported();
    }
    let mut host = Host::new();
    host.stdin = stdin.to_vec();
    // Grant the powerbox prefix + the `display`/`keyboard` graphical caps (shared with the reactor). A
    // single-shot run drains no keys, and `frame` captures the last frame the guest presented (if any).
    // No `fs` file: a single-shot on-ramp guest reads its input from stdin, not a served file.
    let (frame, _keys) = grant_onramp_caps(&mut host, m, None);
    let mut fuel = u64::MAX;
    // The bytecode engine services a `vm_jit_*`-importing guest (the JACL self-hosted compiler) too:
    // it lowers the guest's `call.import` §22 ops to the driver's `Op::JitInvoke`/`install`/`uninstall`
    // just like a static `cap.call (JIT, op)`, and multiplexes the guest's scheduler cooperatively
    // (no OS threads — so this runs on the wasm32 cdylib, unlike the tree-walker's thread pool). A
    // C guest that grows a large heap with sub-64-KiB `vm_map`s runs unchanged now that the interp's
    // software page size is 4 KiB on wasm (see `host_page_size`) — no per-guest window bump needed.
    let (status, value, exit_code) =
        match bytecode::compile_and_run_with_host(m, 0, &[], &mut fuel, &mut host) {
            None => (STATUS_UNSUPPORTED, 0, 0),
            Some(Err(Trap::Exit(code))) => (STATUS_EXIT, 0, code),
            Some(Err(_)) => (STATUS_TRAP, 0, 0),
            Some(Ok(vals)) => match vals.first() {
                Some(Value::I64(x)) => (STATUS_OK, *x, 0),
                Some(Value::I32(x)) => (STATUS_OK, *x as i64, 0),
                _ => (STATUS_BAD_RESULT, 0, 0),
            },
        };
    let framebuffer = frame.lock().unwrap().take();
    PbOutcome {
        status,
        value,
        exit_code,
        stdout: host.stdout,
        stderr: host.stderr,
        framebuffer,
    }
}

/// On-ramp powerbox **with the §22 `Jit` capability granted** — for the self-hosted JACL
/// compiler-guest (`jacl_compiler.svmb`), which expands macros in-guest by compiling each macro body
/// with `vm_jit_compile_linked` and running it with `vm_jit_invoke2`. This now delegates to
/// [`onramp_exec`], which grants the `Jit` cap conditionally (any guest importing `vm_jit_*`) via
/// [`grant_onramp_caps`] and runs a Jit-importing guest on the tree-walker so its import-bound
/// `invoke`/`install` reach the driver (see `onramp_exec`). Kept as a named entry for callers that
/// specifically mean "run the compiler-guest".
pub fn onramp_jit_exec(m: &svm_ir::Module, stdin: &[u8]) -> PbOutcome {
    onramp_exec(m, stdin)
}

/// Run `m`'s function 0 under the **POSIX personality** (POSIX.md / STAGE1.md) instead of the fixed
/// on-ramp powerbox — the seam that lets the real `svm-posix` shell (and any chibicc program linking
/// the personality libc) run in the browser. [`svm_posix::grant`] registers one `HostProc` capability
/// implementing the libc/memfs surface (`read`/`write`/`open`/`opendir`/`getcwd`/…), and
/// [`svm_posix::bind`] binds the module's manifest imports to it **by name** (IMPORTS.md phase 4 —
/// slot `i` ↔ import `i`, bound at instantiation; the module bytes are never rewritten). `stdin`
/// preloads `read(0, …)`; the guest's `write(1, …)` accumulates in the personality's stdout, returned
/// here (the personality owns stdout, not the browser `Host`'s Stream cap).
///
/// The entry is the phase-4 shape ([`onramp_check`]): a paramless func 0 exported `_start`. A module
/// whose imports are not all POSIX names fails closed ([`svm_posix::bind`] returns `false`) — the
/// `Instantiator`/ring imports the shell's *concurrent* paths use are a later slice; this first slice
/// runs the sequential personality (files, redirects, in-process pipelines) only.
///
/// Runs on the **bytecode** engine (the browser's interpreter tier), the same engine [`onramp_exec`]
/// uses; the personality's `HostProc` dispatches through the guest window `bytecode` hands it.
pub fn onramp_posix_exec(m: &svm_ir::Module, stdin: &[u8]) -> PbOutcome {
    let unsupported = || PbOutcome {
        status: STATUS_UNSUPPORTED,
        value: 0,
        exit_code: 0,
        stdout: Vec::new(),
        stderr: Vec::new(),
        framebuffer: None,
    };
    if onramp_check(m).is_err() {
        return unsupported();
    }
    let mut host = Host::new();
    // The personality's `malloc` hands out the top 64 KiB of the guest window (window offsets, as the
    // `c_shell` harness grants it); the shell's static data + stack sit below. `stdin` seeds `read(0)`.
    let win = m.memory.map_or(0, |mc| 1u64 << mc.size_log2);
    let heap_base = win.saturating_sub(64 << 10);
    let (px_h, posix) = svm_posix::grant(&mut host, heap_base, win, stdin.to_vec());
    // Bind every manifest import to the personality by name; fail closed on a non-POSIX import.
    if !svm_posix::bind(m, &mut host, px_h) {
        return unsupported();
    }
    let mut fuel = u64::MAX;
    let (status, value, exit_code) =
        match bytecode::compile_and_run_with_host(m, 0, &[], &mut fuel, &mut host) {
            None => (STATUS_UNSUPPORTED, 0, 0),
            Some(Err(Trap::Exit(code))) => (STATUS_EXIT, 0, code),
            Some(Err(_)) => (STATUS_TRAP, 0, 0),
            Some(Ok(vals)) => match vals.first() {
                Some(Value::I64(x)) => (STATUS_OK, *x, 0),
                Some(Value::I32(x)) => (STATUS_OK, *x as i64, 0),
                _ => (STATUS_BAD_RESULT, 0, 0),
            },
        };
    PbOutcome {
        status,
        value,
        exit_code,
        stdout: posix.stdout(),
        stderr: posix.stderr(),
        framebuffer: None, // the personality grants no `display` cap
    }
}

/// Run the **`svm-posix` shell** (STAGE1.md; `crates/svm/tests/c_shell.rs`) — a real command
/// interpreter compiled by chibicc onto the personality — with `stdin` as the script. This is the
/// playground's shell card: the same module bytes the differential test runs, executed in the browser.
///
/// Runs on the **bytecode** engine ([`bytecode::compile_and_run_with_host`]) — the browser's
/// single-threaded, wasm-safe interpreter tier. (The tree-walk `drive` uses OS worker threads + a wall
/// clock, neither of which exists under `wasm32-unknown-unknown`, so it can't run in the browser.) The
/// shell *statically* carries `Instantiator`/`SharedRegion` cap.calls (its external-command / ring
/// paths), but the sequential Stage-0 surface never **executes** them — with no commands registered,
/// `exec_lookup` misses and pipelines fall back to the in-window memfs — so the module runs cleanly;
/// only the *reserved-window* bytecode entry statically refuses such modules, and this plain entry does
/// not. Cross-checked against the tree-walk/JIT oracle by `crates/svm/tests/c_shell.rs`'s bytecode arm.
///
/// Grants match the differential's setup and order so the shell's `cap.self` reflection discovers the
/// same interfaces: a forwardable `stdout` `Stream`, an `Instantiator` + `AddressSpace` over the whole
/// window (inert this slice — see above), and the POSIX personality itself (its captured stdout is the
/// shell's output). The personality heap is the top 64 KiB (the shell never `malloc`s).
pub fn posix_shell_exec(m: &svm_ir::Module, stdin: &[u8]) -> PbOutcome {
    posix_shell_exec_with(m, stdin, &[])
}

/// As [`posix_shell_exec`], plus a **PATH registry** of external commands `(name, module)` — each
/// granted as a `Module` and registered so an unknown command name in the script is `exec`'d as a
/// separate compiled child (op 13, STAGE1.md §5) instead of `<cmd>: not found`. Registering the
/// `__stage` ring-filter runner here is what makes `cat f | sort | uniq`-style pipelines take the
/// **concurrent ring path** (op 11 + `SharedRegion` + futex) rather than sequential memfs staging.
/// Grant order + the shared-stdout unification mirror `c_shell.rs`'s `setup` exactly, so a run here
/// discovers the same handles as the byte-checked differential and its output (shell builtins + child
/// stages) lands in one captured stdout.
pub fn posix_shell_exec_with(
    m: &svm_ir::Module,
    stdin: &[u8],
    cmds: &[(&str, &svm_ir::Module)],
) -> PbOutcome {
    let win = m.memory.map_or(0, |mc| 1u64 << mc.size_log2);
    let mut host = Host::new();
    // Grant order mirrors `c_shell.rs`'s `setup` (shared stdout sink, Stream, Instantiator,
    // AddressSpace, the command Modules, then the personality) so a run here discovers the same handles
    // as the tested one, and the shell's fd-1 writes + each child's re-granted `Stream` share one sink.
    let sink = host.shared_stdout();
    let out_h = host.grant_stream(StreamRole::Out);
    let (in_h, in_fifo) = host.grant_input_pipe();
    let _inst = host.grant_instantiator(0, win);
    let _as = host.grant_address_space(0, win);
    let cmd_handles: Vec<(&str, i32, u8)> = cmds
        .iter()
        .map(|(n, cm)| {
            (
                *n,
                host.grant_module(cm),
                cm.memory.map_or(0, |mm| mm.size_log2),
            )
        })
        .collect();
    let heap_base = win.saturating_sub(64 << 10);
    let (_px, posix) = svm_posix::grant(&mut host, heap_base, win, stdin.to_vec());
    posix.set_stdout_sink(sink);
    posix.set_exec_stdout(out_h);
    posix.set_exec_stdin(in_h, in_fifo);
    for (n, h, wl) in &cmd_handles {
        posix.register_command(n, *h, *wl);
    }
    let mut fuel = 200_000_000u64;
    let (status, value, exit_code) =
        match bytecode::compile_and_run_with_host(m, 0, &[], &mut fuel, &mut host) {
            Some(Ok(vals)) => match vals.first() {
                Some(Value::I64(x)) => (STATUS_OK, *x, 0),
                Some(Value::I32(x)) => (STATUS_OK, *x as i64, 0),
                _ => (STATUS_OK, 0, 0), // a shell that loops to EOF returns nothing meaningful
            },
            Some(Err(Trap::Exit(code))) => (STATUS_EXIT, 0, code),
            Some(Err(_)) => (STATUS_TRAP, 0, 0),
            None => (STATUS_UNSUPPORTED, 0, 0),
        };
    PbOutcome {
        status,
        value,
        exit_code,
        stdout: posix.stdout(),
        stderr: posix.stderr(),
        framebuffer: None,
    }
}

/// Build the §3e powerbox args blob — `{ argc:u32-LE, envc:u32-LE }` then packed NUL-terminated
/// strings — for seeding at `POWERBOX_ARGS_BASE` (the browser twin of `svm-run`'s `build_args_blob`,
/// no env). The on-ramp `_start` parses it into `argc`/`argv`.
fn pg_args_blob(argv: &[&[u8]]) -> Vec<u8> {
    let mut blob = Vec::new();
    blob.extend_from_slice(&(argv.len() as u32).to_le_bytes());
    blob.extend_from_slice(&0u32.to_le_bytes()); // envc = 0
    for s in argv {
        blob.extend_from_slice(s);
        blob.push(0);
    }
    blob
}

/// Shared Postgres powerbox setup (used by the one-shot [`pg_exec`] and the persistent [`PgSession`]):
/// gate the module shape, grant `stdout/stdin/exit/memory` (registered by name and bound to the
/// manifest slots — the paramless `_start` takes no handle args) + the in-memory `fs` cap over the
/// data `image`, and build the `--single` argv image. Returns `(host, init_mem, fs handle)` or a
/// `STATUS_*` on failure. `host`'s stdin is left empty and non-blocking; the caller sets those per
/// run mode.
fn pg_setup(
    m: &svm_ir::Module,
    image: &[u8],
    argv: &[&[u8]],
) -> Result<(Host, Vec<u8>, svm_fs::MemFsHandle), i32> {
    // IMPORTS.md phase 4: an import-bearing module must be a manifest module (paramless exported
    // `_start`) — the runtime binds slots, it never rewrites. Fail closed otherwise.
    onramp_check(m).map_err(|_| STATUS_UNSUPPORTED)?;
    let mut host = Host::new();
    // Grant the on-ramp powerbox prefix — stdout, stdin, exit, memory (the heap-growth cap `malloc`
    // uses) — and register each **by name** (`cap.self.resolve`). Granted directly — not via
    // `grant_onramp_caps`, whose graphical `display`/`keyboard`/`webgpu` caps would add a host
    // import a headless Postgres neither needs nor can satisfy.
    let out = host.grant_stream(StreamRole::Out);
    host.register_cap_name("stdout", out);
    let inp = host.grant_stream(StreamRole::In);
    host.register_cap_name("stdin", inp);
    let exit = host.grant_exit();
    host.register_cap_name("exit", exit);
    let memory = host.grant_memory();
    host.register_cap_name("memory", memory);
    // IMPORTS.md phase 4: a manifest-carrying module executes its `call.import`s through
    // instantiation-time slot bindings — map each import name via the on-ramp policy onto the four
    // granted handles (`Stream` disambiguated by op). A name outside this headless powerbox (e.g.
    // the dynamic-only SharedRegion ops) leaves its slot unbound — fail-closed at dispatch.
    if !m.imports.is_empty() {
        use svm_interp::cap_id;
        let bindings = m
            .imports
            .iter()
            .map(|im| {
                let Some(cap) = onramp_cap_resolver(&im.name) else {
                    return svm_interp::BoundImport::rebindable(0, 0, None);
                };
                let handle = match (cap.type_id, cap.op) {
                    (cap_id::STREAM, 1) => out,
                    (cap_id::STREAM, _) => inp,
                    (cap_id::EXIT, _) => exit,
                    (cap_id::ADDRESS_SPACE, _) => memory,
                    _ => return svm_interp::BoundImport::rebindable(0, 0, None),
                };
                svm_interp::BoundImport::required(cap.type_id, cap.op, handle)
            })
            .collect();
        host.set_import_bindings(bindings);
    }
    // Mount the shipped data image as an in-memory `fs` cap (decode is fail-closed). The **shared**
    // mount hands back a `MemFsHandle`, so a persistent session can snapshot the live data dir back out
    // later ([`svm_pg_snapshot`]); the one-shot `pg_exec` simply drops it.
    let (files, dirs) = svm_fs::decode_image(image).map_err(|_| STATUS_DECODE_ERR)?;
    let (fs_hostfn, fs_handle) = svm_fs::mem_fs_seeded_shared(files, dirs);
    let fsh = host.grant_host_proc(fs_hostfn);
    host.register_cap_name("fs", fsh);
    // Seed the caller's `argv` at the powerbox args base (Postgres: a slashed `argv[0]` so
    // `find_my_exec` resolves; chibicc: `["chibicc", "/in.c"]`). #964: a `__null_guard`-marked
    // module reads its args one guard higher — place the blob where its `_start` looks.
    let blob = pg_args_blob(argv);
    let base = svm_ir::module_args_base(m) as usize;
    let mut init_mem = vec![0u8; base + blob.len()];
    init_mem[base..].copy_from_slice(&blob);
    Ok((host, init_mem, fs_handle))
}

/// Run **PostgreSQL `--single`** in the wasm sandbox: mount the data-image `image` on the `fs` cap
/// (`svm_fs::mem_fs_seeded_handler` — a real in-memory filesystem, no host fs), seed the `--single`
/// argv, and run the module's `_start` on the **reserved-window** bytecode engine (Postgres grows its
/// heap through the `memory` cap into the reserved tail). `stdin` is the SQL script; the backend's
/// output comes back on the captured `stdout`. The one entry that boots a *real database* in the
/// browser — and the direct in-wasm measurement of the guest boot (BOOTSPEED.md). The `stdout, stdin,
/// exit, memory, fs` caps are reached by name (`cap.self.resolve`) or through the module's manifest
/// slot bindings — the paramless `_start` takes no handle args (IMPORTS.md phase 4).
pub fn pg_exec(m: &svm_ir::Module, image: &[u8], stdin: &[u8]) -> PbOutcome {
    onramp_fs_exec(m, image, &PG_SINGLE_ARGV, stdin)
}

/// The Postgres `--single` argv (a slashed `argv[0]` so `find_my_exec` resolves).
const PG_SINGLE_ARGV: [&[u8]; 5] = [b"./postgres", b"--single", b"-D", b".", b"postgres"];

/// Generic on-ramp run with a seeded multi-file `fs` cap + caller-supplied `argv` — the shape
/// [`pg_exec`] (Postgres) and [`svm_run_onramp_fs`] (chibicc-the-guest) share. Mounts the memfs
/// `image` on the `fs` cap, seeds `argv` at `POWERBOX_ARGS_BASE`, feeds `stdin`, and runs `_start`
/// on the reserved-window engine (the guest may grow a heap through the `memory` cap). Unlike
/// [`onramp_exec`], the guest reads its input from served files, not stdin — chibicc `fopen`s
/// `/in.c` + `/include/*.h`.
pub fn onramp_fs_exec(m: &svm_ir::Module, image: &[u8], argv: &[&[u8]], stdin: &[u8]) -> PbOutcome {
    let unsupported = |status: i32| PbOutcome {
        status,
        value: 0,
        exit_code: 0,
        stdout: Vec::new(),
        stderr: Vec::new(),
        framebuffer: None,
    };
    let (mut host, init_mem, _fs) = match pg_setup(m, image, argv) {
        Ok(setup) => setup,
        Err(status) => return unsupported(status),
    };
    host.stdin = stdin.to_vec();
    let mut fuel = u64::MAX;
    let (status, value, exit_code) = match bytecode::compile_and_run_capture_reserved_with_host(
        m,
        0,
        &[],
        &mut fuel,
        &init_mem,
        svm_ir::DEFAULT_RESERVED_LOG2,
        &mut host,
    ) {
        None => (STATUS_UNSUPPORTED, 0, 0),
        Some((Err(Trap::Exit(code)), _)) => (STATUS_EXIT, 0, code),
        Some((Err(_), _)) => (STATUS_TRAP, 0, 0),
        Some((Ok(vals), _)) => match vals.first() {
            Some(Value::I64(x)) => (STATUS_OK, *x, 0),
            Some(Value::I32(x)) => (STATUS_OK, *x as i64, 0),
            _ => (STATUS_BAD_RESULT, 0, 0),
        },
    };
    PbOutcome {
        status,
        value,
        exit_code,
        stdout: host.stdout,
        stderr: host.stderr,
        framebuffer: None,
    }
}

/// Like [`onramp_fs_exec`], but after the run **reads one named file back out of the seeded memfs** —
/// for a guest phase whose real output is a *file* it wrote through the `fs` cap, not stdout. `nifler
/// p /in.nim /out.p.nif` (NIM.md §3c/§3e, "nimony in the browser" slice 4) parses to a `.p.nif` file;
/// [`onramp_fs_exec`] drops the fs handle, so this keeps it and, after the run, seeds it back out and
/// returns the bytes at `out_key` (the memfs strips a leading `/`, so pass the slashless key). The
/// file is `Vec::new()` if the phase never wrote it (a parse error — the caller shows the guest's
/// stderr instead). The run's own `stdout`/`stderr` still ride the returned [`PbOutcome`].
fn onramp_fs_exec_readback(
    m: &svm_ir::Module,
    image: &[u8],
    argv: &[&[u8]],
    stdin: &[u8],
    out_key: &str,
) -> (PbOutcome, Vec<u8>) {
    let unsupported = |status: i32| PbOutcome {
        status,
        value: 0,
        exit_code: 0,
        stdout: Vec::new(),
        stderr: Vec::new(),
        framebuffer: None,
    };
    let (mut host, init_mem, fs) = match pg_setup(m, image, argv) {
        Ok(setup) => setup,
        Err(status) => return (unsupported(status), Vec::new()),
    };
    host.stdin = stdin.to_vec();
    let mut fuel = u64::MAX;
    let (status, value, exit_code) = match bytecode::compile_and_run_capture_reserved_with_host(
        m,
        0,
        &[],
        &mut fuel,
        &init_mem,
        svm_ir::DEFAULT_RESERVED_LOG2,
        &mut host,
    ) {
        None => (STATUS_UNSUPPORTED, 0, 0),
        Some((Err(Trap::Exit(code)), _)) => (STATUS_EXIT, 0, code),
        Some((Err(_), _)) => (STATUS_TRAP, 0, 0),
        Some((Ok(vals), _)) => match vals.first() {
            Some(Value::I64(x)) => (STATUS_OK, *x, 0),
            Some(Value::I32(x)) => (STATUS_OK, *x as i64, 0),
            _ => (STATUS_BAD_RESULT, 0, 0),
        },
    };
    // Read the emitted file back out of the live store (the `MemFsHandle` observes what the guest wrote).
    let (files, _dirs) = fs.seed();
    let produced = files
        .into_iter()
        .find(|(k, _)| k == out_key)
        .map(|(_, v)| v)
        .unwrap_or_default();
    (
        PbOutcome {
            status,
            value,
            exit_code,
            stdout: host.stdout,
            stderr: host.stderr,
            framebuffer: None,
        },
        produced,
    )
}

/// **Boot Postgres in wasm.** Decode + verify the module at `[mod_ptr, mod_len)`, mount the data image
/// at `[img_ptr, img_len)` on the `fs` cap, feed the SQL at `[stdin_ptr, stdin_len)`, and run. Sets
/// [`svm_status`]/[`svm_exit_code`]; the backend's output is read back via `svm_stdout_ptr`/`_len`.
/// Returns the guest's `i64` result (`0` on any non-`OK`/`EXIT`). Driven by `browser/bench_pg.mjs`.
#[no_mangle]
pub extern "C" fn svm_run_pg(
    mod_ptr: *const u8,
    mod_len: usize,
    img_ptr: *const u8,
    img_len: usize,
    stdin_ptr: *const u8,
    stdin_len: usize,
) -> i64 {
    let set = |s: i32| unsafe { LAST_STATUS = s };
    // SAFETY: the host guarantees each range is a live `svm_alloc`ation it just filled.
    let bytes = unsafe { core::slice::from_raw_parts(mod_ptr, mod_len) };
    let image = unsafe { core::slice::from_raw_parts(img_ptr, img_len) };
    let stdin: &[u8] = if stdin_ptr.is_null() || stdin_len == 0 {
        &[]
    } else {
        unsafe { core::slice::from_raw_parts(stdin_ptr, stdin_len) }
    };
    let m = match svm_encode::decode_module(bytes) {
        Ok(m) => m,
        Err(_) => {
            set(STATUS_DECODE_ERR);
            return 0;
        }
    };
    if svm_verify::verify_module(&m).is_err() {
        set(STATUS_VERIFY_ERR);
        return 0;
    }
    let out = pg_exec(&m, image, stdin);
    set(out.status);
    // SAFETY: single-threaded wasm; the capture slots are read back only via the export accessors.
    unsafe {
        stash(&mut *core::ptr::addr_of_mut!(OUT), out.stdout);
        stash(&mut *core::ptr::addr_of_mut!(ERR), out.stderr);
        EXIT_CODE = out.exit_code;
    }
    out.value
}

/// The built-in playground libc headers, seeded under `/include` for the C-compiler card (SELFHOST_C.md
/// §7). They are guest C compiled *into* the user's program on `#include` — `printf`/`puts`/… format
/// over the powerbox's ambient `write`, `malloc` is a bump allocator, `str*`/`ctype` are pure. Nothing
/// is linked, and a header costs nothing unless the program includes it. Source: `browser/playground-include/`.
pub fn playground_include_files() -> Vec<(String, Vec<u8>)> {
    const HEADERS: &[(&str, &str)] = &[
        (
            "include/stdio.h",
            include_str!("../playground-include/stdio.h"),
        ),
        (
            "include/string.h",
            include_str!("../playground-include/string.h"),
        ),
        (
            "include/stdlib.h",
            include_str!("../playground-include/stdlib.h"),
        ),
        (
            "include/stdarg.h",
            include_str!("../playground-include/stdarg.h"),
        ),
        (
            "include/ctype.h",
            include_str!("../playground-include/ctype.h"),
        ),
        (
            "include/stdbool.h",
            include_str!("../playground-include/stdbool.h"),
        ),
        (
            "include/stdint.h",
            include_str!("../playground-include/stdint.h"),
        ),
        (
            "include/stddef.h",
            include_str!("../playground-include/stddef.h"),
        ),
        (
            "include/limits.h",
            include_str!("../playground-include/limits.h"),
        ),
        (
            "include/errno.h",
            include_str!("../playground-include/errno.h"),
        ),
        (
            "include/assert.h",
            include_str!("../playground-include/assert.h"),
        ),
        (
            "include/math.h",
            include_str!("../playground-include/math.h"),
        ),
        (
            "include/xmmintrin.h",
            include_str!("../playground-include/xmmintrin.h"),
        ),
        // The §12 threading layer (INTERACTIVE_EMBEDDING.md slice 9): pthreads + POSIX semaphores
        // over the VM's futex/atomics builtins. Seeded from the *frontend's* bundled copies —
        // one source of truth, since the guest chibicc lowers the same `__vm_*` builtins the
        // native frontend does.
        (
            "include/pthread.h",
            include_str!("../../frontend/chibicc/include/pthread.h"),
        ),
        (
            "include/semaphore.h",
            include_str!("../../frontend/chibicc/include/semaphore.h"),
        ),
        // C11 atomics (INTERACTIVE_EMBEDDING.md): the playground's own `<stdatomic.h>` maps the
        // atomic ops to the **real** VM atomic builtins (not plain `*p += v` like the frontend's
        // display header), so a lock-free `atomic_fetch_add` counter stays correct under any
        // interleaving — the "atomic counter survives chaos mode" lesson.
        (
            "include/stdatomic.h",
            include_str!("../playground-include/stdatomic.h"),
        ),
        // The system-header surface chibicc's *own* sources #include (SELFHOST_C.md §7, stage-2). Most
        // are thin stubs — the sandbox has no processes/globbing/wall-clock — present so `chibicc.h`
        // parses; `<time.h>` returns a fixed 1970 epoch (for the `__DATE__`/`__TIME__` macros), and the
        // fs syscalls (`open`/`read` in `<unistd.h>`) match `<stdio.h>`'s `fopen`/`fread`.
        (
            "include/stdnoreturn.h",
            include_str!("../playground-include/stdnoreturn.h"),
        ),
        (
            "include/strings.h",
            include_str!("../playground-include/strings.h"),
        ),
        (
            "include/glob.h",
            include_str!("../playground-include/glob.h"),
        ),
        (
            "include/libgen.h",
            include_str!("../playground-include/libgen.h"),
        ),
        (
            "include/unistd.h",
            include_str!("../playground-include/unistd.h"),
        ),
        (
            "include/time.h",
            include_str!("../playground-include/time.h"),
        ),
        (
            "include/sys/stat.h",
            include_str!("../playground-include/sys/stat.h"),
        ),
        (
            "include/sys/types.h",
            include_str!("../playground-include/sys/types.h"),
        ),
        (
            "include/sys/wait.h",
            include_str!("../playground-include/sys/wait.h"),
        ),
    ];
    HEADERS
        .iter()
        .map(|(k, v)| ((*k).to_string(), v.as_bytes().to_vec()))
        .collect()
}

/// The chibicc card's argv (shared by the bytecode [`svm_run_onramp_fs`] and the JIT
/// [`svm_onramp_jit_run_open_fs`]). `--data-page 65536`: the compiled program runs in the browser
/// (64 KiB wasm host page), so its read-only globals must not share a host page with writable data
/// (D40). Debug info is **off by default** (the `debug.*` waist is ~a third of the emitted IR, so a
/// clean run compiles far less IR); pass `debug_info` (a `-g` flag) only when the user opts into
/// source-level debugging (the DAP panel maps C `file:line`/locals through chibicc's debug section).
fn chibicc_card_argv(debug_info: bool) -> Vec<&'static [u8]> {
    let mut argv: Vec<&'static [u8]> = vec![b"chibicc", b"--data-page", b"65536"];
    if debug_info {
        argv.push(b"-g");
    }
    argv.push(b"/in.c");
    argv
}

/// The **self-host** card's argv (SELFHOST_C.md §5): compile one of chibicc's *own* cc1 TUs to a
/// linkable **object** unit (`--emit-object`, `cc -c`), reading the TU + its full system-header closure
/// from the seeded memfs. Mirrors `guest_emit_object` in `crates/svm/tests/c_link.rs` — relative `-I`s
/// (the fs cap refuses absolute paths), the self-host prelude force-included (chibicc's parser can't
/// ingest modern glibc's ISO-C23 `strtoul`/… redirects), output to stdout (no out arg). `tu` is the
/// memfs-relative input path (e.g. `frontend/chibicc/hashmap.c`). Borrows `tu`; caller keeps it alive.
fn chibicc_selfhost_argv(tu: &[u8], debug_info: bool) -> Vec<&[u8]> {
    // No `--data-page`: an emit-object *unit* is relinked (the linker page-aligns each unit's data, D40),
    // so the object stays canonical — byte-identical to the proven native path (`c_link.rs`, which passes
    // no data-page), which is what the CI gate diffs against.
    let mut argv: Vec<&[u8]> = vec![
        b"chibicc",
        b"--emit-object",
        b"-include",
        b"crates/svm-run/demos/chibicc_selfhost/selfhost_prelude.h",
        b"-Ifrontend/chibicc",
        b"-Ifrontend/chibicc/include",
        b"-Iusr/include/x86_64-linux-gnu",
        b"-Iusr/include",
    ];
    if debug_info {
        argv.push(b"-g");
    }
    argv.push(tu);
    argv
}

/// Assemble the chibicc card's memfs image: the user's source `src` at `in.c`, the built-in playground
/// libc headers under `include/` ([`playground_include_files`]), plus any caller headers from the
/// optional `encode_image` blob at `[img_ptr, img_len)` (which win on a key clash). Shared by the
/// bytecode and JIT card entries so both seed an identical filesystem. `Err(STATUS_DECODE_ERR)` if the
/// caller image doesn't decode.
fn chibicc_card_image(img_ptr: *const u8, img_len: usize, src: &[u8]) -> Result<Vec<u8>, i32> {
    // The guest maps the absolute `/in.c`/`/include/*` back to these cap-relative keys (`in.c`,
    // `include/…`); chibicc's fixed `/include` search path resolves the headers.
    let (mut files, mut dirs) = if img_len == 0 {
        (Vec::new(), Vec::new())
    } else {
        // SAFETY: the host guarantees `[img_ptr, img_len)` is a live allocation it just filled.
        let image = unsafe { core::slice::from_raw_parts(img_ptr, img_len) };
        svm_fs::decode_image(image).map_err(|_| STATUS_DECODE_ERR)?
    };
    // Seed the built-in playground libc headers under `/include` so a compiled program can
    // `#include <stdio.h>` etc. — a caller-supplied image (same key) takes precedence.
    for (key, bytes) in playground_include_files() {
        if !files.iter().any(|(k, _)| *k == key) {
            files.push((key, bytes));
        }
    }
    if !dirs.iter().any(|d| d == "include") {
        dirs.push("include".to_string());
    }
    // The seeded headers include `sys/*.h` (the stage-2 system-header stubs), so register `include/sys`.
    if !dirs.iter().any(|d| d == "include/sys") {
        dirs.push("include/sys".to_string());
    }
    // Split the editor buffer into a **multi-file** project: the compile targets `/in.c` (the text
    // before the first marker), and each `//// file: NAME` marker seeds a sibling file the entry can
    // `#include "NAME"` (chibicc resolves quote-includes against the source's own directory, `/`). No
    // marker ⇒ the whole buffer is `/in.c`, exactly as before. Any `NAME` with a `/` seeds under that
    // directory (its parent dirs are registered so the memfs can hold it).
    for (key, bytes) in split_multifile_source(src) {
        if let Some((dir, _)) = key.rsplit_once('/') {
            // Register every ancestor directory of a nested file (e.g. `a/b/c.h` → `a`, `a/b`).
            let mut acc = String::new();
            for seg in dir.split('/') {
                if !acc.is_empty() {
                    acc.push('/');
                }
                acc.push_str(seg);
                if !dirs.contains(&acc) {
                    dirs.push(acc.clone());
                }
            }
        }
        // A seeded file wins over any same-key caller-image entry (the editor is the source of truth).
        files.retain(|(k, _)| *k != key);
        files.push((key, bytes));
    }
    Ok(svm_fs::encode_image(&files, &dirs))
}

/// Split a card editor buffer into memfs files on `//// file: NAME` marker lines. The text before the
/// first marker is the entry (`in.c`); each marker begins a new file `NAME` (a leading `/` is trimmed;
/// a `NAME` with slashes nests). No marker ⇒ a single `in.c` (the whole buffer). Used by both the
/// bytecode and JIT card entries so multi-file behaves identically across tiers.
pub fn split_multifile_source(src: &[u8]) -> Vec<(String, Vec<u8>)> {
    // A line is a marker iff, ignoring leading whitespace, it is `////` followed by (optional space)
    // `file:` (case-insensitive) then a non-empty filename.
    fn marker_name(line: &str) -> Option<String> {
        let rest = line.trim_start().strip_prefix("////")?.trim_start();
        // Case-insensitive `file:` prefix.
        let after = rest.get(..5).filter(|p| p.eq_ignore_ascii_case("file:"))?;
        let _ = after;
        let name = rest[5..].trim().trim_start_matches('/').trim();
        if name.is_empty() {
            None
        } else {
            Some(name.to_string())
        }
    }

    let text = String::from_utf8_lossy(src);
    let mut files: Vec<(String, Vec<u8>)> = Vec::new();
    let mut cur_name = "in.c".to_string();
    let mut cur = String::new();
    for line in text.split_inclusive('\n') {
        if let Some(name) = marker_name(line.trim_end_matches('\n')) {
            files.push((cur_name, cur.into_bytes()));
            cur_name = name;
            cur = String::new();
        } else {
            cur.push_str(line);
        }
    }
    files.push((cur_name, cur.into_bytes()));
    files
}

/// **Run chibicc-the-guest to compile a C source** — the playground C-compiler demo (SELFHOST_C.md
/// §7 step 5). Decode + verify the compiler module at `[mod_ptr, mod_len)`, build an in-memory `fs`
/// mounting the user's source at `in.c` (the guest opens `/in.c`), the built-in playground libc
/// headers under `include/` ([`playground_include_files`] — `<stdio.h>` etc.), plus, if `img_len > 0`,
/// any caller headers from the `encode_image` blob at `[img_ptr, img_len)` (which win on a key clash).
/// Seeds `argv = ["chibicc", "/in.c"]` and runs. The emitted SVM-IR **text** comes back on
/// `svm_stdout_ptr`/`_len`, ready to hand to [`svm_parse`] → a runnable module. The seeded headers are
/// guest C compiled in on `#include`, so a `printf` program prints (over the powerbox's ambient
/// `write`) instead of trapping on an unresolved call. Sets [`svm_status`]/[`svm_exit_code`]; returns
/// the guest's `i64` result (`0` on any non-`OK`/`EXIT`).
#[no_mangle]
pub extern "C" fn svm_run_onramp_fs(
    mod_ptr: *const u8,
    mod_len: usize,
    img_ptr: *const u8,
    img_len: usize,
    src_ptr: *const u8,
    src_len: usize,
    debug_info: i32,
) -> i64 {
    let set = |s: i32| unsafe { LAST_STATUS = s };
    // SAFETY: the host guarantees each range is a live `svm_alloc`ation it just filled.
    let bytes = unsafe { core::slice::from_raw_parts(mod_ptr, mod_len) };
    let src = unsafe { core::slice::from_raw_parts(src_ptr, src_len) };
    let m = match svm_encode::decode_module(bytes) {
        Ok(m) => m,
        Err(_) => {
            set(STATUS_DECODE_ERR);
            return 0;
        }
    };
    if svm_verify::verify_module(&m).is_err() {
        set(STATUS_VERIFY_ERR);
        return 0;
    }
    let image = match chibicc_card_image(img_ptr, img_len, src) {
        Ok(image) => image,
        Err(status) => {
            set(status);
            return 0;
        }
    };
    let argv = chibicc_card_argv(debug_info != 0);
    let out = onramp_fs_exec(&m, &image, &argv, &[]);
    set(out.status);
    // SAFETY: single-threaded wasm; the capture slots are read back only via the export accessors.
    unsafe {
        stash(&mut *core::ptr::addr_of_mut!(OUT), out.stdout);
        stash(&mut *core::ptr::addr_of_mut!(ERR), out.stderr);
        EXIT_CODE = out.exit_code;
    }
    out.value
}

/// **Compile Nim in the browser — the nimony front-end card** (NIM.md §3c/§3e, "nimony in the browser"
/// slice 4). Run `nifler.svmb` — the *first real nimony compiler phase* (Nim source → parsed NIF),
/// itself a Nim program on-ramped to SVM through the C on-ramp (slice 1) — over the editor's Nim.
/// Decode + verify the phase module at `[mod_ptr, mod_len)`, seed an in-memory `fs` cap with the user's
/// source at `in.nim`, run `nifler p /in.nim /out.p.nif` (the parse command), and hand the emitted
/// `.p.nif` **text** back on `svm_stdout_ptr`/`_len` — the same real nifler that parses Nim natively,
/// now running client-side in the sandbox on the reader's own code. Unlike the pre-built
/// `nim (Nim → SVM, runs)` card (whose front-end ran at *build* time), this runs a front-end phase
/// **in the browser**; unlike the `svm-leng` back-end card (Leng → IR), this is the front edge (Nim →
/// NIF). The guest reaches only the seeded `fs` — no ambient authority. Sets [`svm_status`]/
/// [`svm_exit_code`]; returns the guest's `i64` result (`0` on any non-`OK`/`EXIT`). On a parse error
/// (`nifler` wrote no `.p.nif`) the guest's own stderr rides `svm_stderr_ptr`/`_len` and the stdout
/// capture is empty, so the card can surface the diagnostic.
#[no_mangle]
pub extern "C" fn svm_run_nifler_fs(
    mod_ptr: *const u8,
    mod_len: usize,
    src_ptr: *const u8,
    src_len: usize,
) -> i64 {
    let set = |s: i32| unsafe { LAST_STATUS = s };
    // SAFETY: the host guarantees each range is a live `svm_alloc`ation it just filled.
    let bytes = unsafe { core::slice::from_raw_parts(mod_ptr, mod_len) };
    let src = unsafe { core::slice::from_raw_parts(src_ptr, src_len) };
    let m = match svm_encode::decode_module(bytes) {
        Ok(m) => m,
        Err(_) => {
            set(STATUS_DECODE_ERR);
            return 0;
        }
    };
    if svm_verify::verify_module(&m).is_err() {
        set(STATUS_VERIFY_ERR);
        return 0;
    }
    // Seed the source as `in.nim`; nifler parses `/in.nim` and writes `/out.p.nif` (both memfs keys,
    // slashless in the store). The emitted `.p.nif` is the file we read back and show.
    let image = svm_fs::encode_image(&[("in.nim".to_string(), src.to_vec())], &[]);
    let argv: [&[u8]; 4] = [b"nifler", b"p", b"/in.nim", b"/out.p.nif"];
    let (out, produced) = onramp_fs_exec_readback(&m, &image, &argv, &[], "out.p.nif");
    set(out.status);
    // SAFETY: single-threaded wasm; the capture slots are read back only via the export accessors.
    // The produced `.p.nif` is the visible output (stdout slot); the guest's diagnostics ride stderr.
    unsafe {
        stash(&mut *core::ptr::addr_of_mut!(OUT), produced);
        stash(&mut *core::ptr::addr_of_mut!(ERR), out.stderr);
        EXIT_CODE = out.exit_code;
    }
    out.value
}

/// **Self-host card — bytecode tier** (SELFHOST_C.md §7 step 5, the capstone). Run `chibicc.svmb` in
/// `--emit-object` mode over one of chibicc's *own* cc1 TUs, seeded from `[img_ptr, img_len)` — the
/// committed closure image (`chibicc_selfhost.img`: the TU sources + their glibc header closure +
/// `selfhost_prelude.h`) — and emit that TU's linkable **object** unit as SVM-IR **text** on
/// `svm_stdout_ptr`/`_len`. `[tu_ptr, tu_len)` is the memfs-relative TU path. Unlike
/// [`svm_run_onramp_fs`] (which merges the playground libc + seeds `/in.c`), the image is passed
/// **raw** — the self-host closure is self-contained. The wasm-JIT twin is
/// [`svm_selfhost_jit_emit_object_fs`]. Sets [`svm_status`]/[`svm_exit_code`]; returns the guest result.
#[no_mangle]
pub extern "C" fn svm_selfhost_emit_object_fs(
    mod_ptr: *const u8,
    mod_len: usize,
    img_ptr: *const u8,
    img_len: usize,
    tu_ptr: *const u8,
    tu_len: usize,
    debug_info: i32,
) -> i64 {
    let set = |s: i32| unsafe { LAST_STATUS = s };
    // SAFETY: the host guarantees each range is a live `svm_alloc`ation it just filled.
    let bytes = unsafe { core::slice::from_raw_parts(mod_ptr, mod_len) };
    let image = unsafe { core::slice::from_raw_parts(img_ptr, img_len) };
    let tu = unsafe { core::slice::from_raw_parts(tu_ptr, tu_len) };
    let m = match svm_encode::decode_module(bytes) {
        Ok(m) => m,
        Err(_) => {
            set(STATUS_DECODE_ERR);
            return 0;
        }
    };
    if svm_verify::verify_module(&m).is_err() {
        set(STATUS_VERIFY_ERR);
        return 0;
    }
    let argv = chibicc_selfhost_argv(tu, debug_info != 0);
    let out = onramp_fs_exec(&m, image, &argv, &[]);
    set(out.status);
    // SAFETY: single-threaded wasm; the capture slots are read back only via the export accessors.
    unsafe {
        stash(&mut *core::ptr::addr_of_mut!(OUT), out.stdout);
        stash(&mut *core::ptr::addr_of_mut!(ERR), out.stderr);
        EXIT_CODE = out.exit_code;
    }
    out.value
}

// ==== persistent interactive Postgres session (the browser console) ===============================
// `svm_run_pg` boots a *fresh* backend per call, runs the SQL to EOF, and lets it exit — every query
// pays the multi-second boot and loses all state. A `PgSession` keeps ONE backend alive: boot to the
// `backend>` prompt once, then each query pushes SQL onto the (now **blocking**) stdin and resumes the
// vCPU until it parks at the next read — so queries after the first are sub-second and DDL/DML persist
// across them, exactly like a real `psql` session. This rides the [`bytecode::Vcpu::set_stdin_blocking`]
// park (a `read` on an exhausted buffer suspends instead of returning EOF) over the same reserved
// window `svm_run_pg` uses; Postgres `--single` is single-threaded, so the only events are the stdin
// park, a clean exit, or a trap.

/// A live single-user Postgres backend suspended at a stdin read. Owns its leaked [`bytecode::VcpuProgram`]
/// (the vCPU borrows it `'static`; reclaimed when the session is replaced/closed) and tracks how much of
/// the backend's cumulative stdout has already been handed back, so each query returns only its delta.
struct PgSession {
    /// The compiled program, leaked so `vcpu` can borrow it `'static`; reclaimed in [`pg_close_session`].
    prog: *mut bytecode::VcpuProgram,
    vcpu: bytecode::Vcpu<'static>,
    /// Bytes of `vcpu`'s cumulative stdout already returned (delta cursor).
    stdout_pos: usize,
    /// The backend exited or trapped — no further queries are possible.
    ended: bool,
    /// Live handle onto the session's `mem_fs` data dir, so [`svm_pg_snapshot`] can serialize the
    /// current database (tables, WAL, catalogs) back out for the host to persist across reloads.
    fs_snap: svm_fs::MemFsHandle,
}

/// The one live session (single-threaded wasm ⇒ a plain static). `None` until [`svm_pg_open`].
static mut PG_SESSION: Option<PgSession> = None;

/// Drop the live session (if any) and **reclaim** its leaked program. Order matters: the vCPU (which
/// borrows the program `'static`) must drop before the program box is reclaimed.
fn pg_close_session() {
    // SAFETY: single-threaded wasm; exclusive access to the session static.
    unsafe {
        if let Some(s) = (*core::ptr::addr_of_mut!(PG_SESSION)).take() {
            let prog = s.prog;
            drop(s); // drops `vcpu` (releasing the `'static` borrow); the raw `prog` ptr is a no-op drop
            drop(Box::from_raw(prog)); // reclaim the leaked VcpuProgram
        }
    }
}

/// Advance the session's vCPU to its next stop. Returns [`STATUS_OK`] when it parks at a stdin read
/// (ready for the next query), [`STATUS_EXIT`] on a clean guest exit, [`STATUS_TRAP`] on a trap, and
/// [`STATUS_UNSUPPORTED`] for any other event (a `--single` backend spawns/JITs nothing). Marks the
/// session `ended` on anything but a park.
fn pg_pump(s: &mut PgSession) -> i32 {
    match s.vcpu.run() {
        bytecode::VcpuEvent::StdinPark => STATUS_OK,
        bytecode::VcpuEvent::Done(_) => {
            s.ended = true;
            STATUS_EXIT
        }
        bytecode::VcpuEvent::Trapped(_) => {
            s.ended = true;
            STATUS_TRAP
        }
        _ => {
            s.ended = true;
            STATUS_UNSUPPORTED
        }
    }
}

/// Stash the session's stdout **delta** (bytes since the last hand-back) into the `OUT` buffer the
/// `svm_stdout_ptr`/`_len` accessors expose, advancing the cursor.
fn pg_flush_stdout(s: &mut PgSession) {
    let out = &s.vcpu.host_mut().stdout;
    let delta = out.get(s.stdout_pos..).unwrap_or(&[]).to_vec();
    s.stdout_pos = out.len();
    // SAFETY: single-threaded wasm; read back only via the export accessors.
    unsafe { stash(&mut *core::ptr::addr_of_mut!(OUT), delta) };
}

/// **Open a persistent Postgres session.** Decode + verify the module at `[mod_ptr, mod_len)`, mount the
/// data image at `[img_ptr, img_len)` on the `fs` cap, and boot `postgres --single` to its `backend>`
/// prompt — leaving it **suspended at the first stdin read** (blocking stdin) rather than running to
/// exit. Replaces any prior session. Sets [`svm_status`]; the banner + prompt land in `svm_stdout_*`.
/// Returns `0` on a ready backend, else the negative `STATUS_*`. Drive with [`svm_pg_query`], end with
/// [`svm_pg_close`].
#[no_mangle]
pub extern "C" fn svm_pg_open(
    mod_ptr: *const u8,
    mod_len: usize,
    img_ptr: *const u8,
    img_len: usize,
) -> i32 {
    let set = |s: i32| unsafe { LAST_STATUS = s };
    pg_close_session(); // a fresh open supersedes any live session
                        // SAFETY: the host guarantees each range is a live `svm_alloc`ation it just filled.
    let bytes = unsafe { core::slice::from_raw_parts(mod_ptr, mod_len) };
    let image = unsafe { core::slice::from_raw_parts(img_ptr, img_len) };
    let m = match svm_encode::decode_module(bytes) {
        Ok(m) => m,
        Err(_) => {
            set(STATUS_DECODE_ERR);
            return -STATUS_DECODE_ERR;
        }
    };
    if svm_verify::verify_module(&m).is_err() {
        set(STATUS_VERIFY_ERR);
        return -STATUS_VERIFY_ERR;
    }
    let (mut host, init_mem, fs_snap) = match pg_setup(&m, image, &PG_SINGLE_ARGV) {
        Ok(setup) => setup,
        Err(status) => {
            set(status);
            return -status;
        }
    };
    host.set_stdin_blocking(true); // the read at the prompt parks instead of returning EOF
    let prog = match bytecode::VcpuProgram::compile(&m) {
        Some(p) => Box::into_raw(Box::new(p)),
        None => {
            set(STATUS_UNSUPPORTED);
            return -STATUS_UNSUPPORTED;
        }
    };
    // SAFETY: `prog` is leaked here and only reclaimed by `pg_close_session` after its vCPU drops, so the
    // `&'static` borrow is valid for the session's whole life.
    let vcpu = match bytecode::Vcpu::new_root_reserved_with_powerbox(
        unsafe { &*prog },
        0,
        &[],
        &init_mem,
        host,
        svm_ir::DEFAULT_RESERVED_LOG2,
    ) {
        Ok(v) => v,
        Err(_) => {
            // SAFETY: nothing borrows `prog` yet — the vCPU build failed — so reclaim it directly.
            unsafe { drop(Box::from_raw(prog)) };
            set(STATUS_TRAP);
            return -STATUS_TRAP;
        }
    };
    let mut session = PgSession {
        prog,
        vcpu,
        stdout_pos: 0,
        ended: false,
        fs_snap,
    };
    let status = pg_pump(&mut session); // run boot to the first stdin park (the prompt)
    pg_flush_stdout(&mut session); // hand back the banner + prompt
    set(status);
    // SAFETY: single-threaded wasm; exclusive access to the session static.
    unsafe { *core::ptr::addr_of_mut!(PG_SESSION) = Some(session) };
    if status == STATUS_OK {
        0
    } else {
        -status
    }
}

/// **Run one query on the open session.** Push the SQL at `[sql_ptr, sql_len)` (a trailing newline is
/// added if absent, so `--single` executes it) onto the backend's stdin and resume until it parks at the
/// next prompt. Sets [`svm_status`]; the query's output (result rows + the next `backend>`) lands in
/// `svm_stdout_*` as a **delta** (just this query's bytes). Returns `0` on a ready backend, else the
/// negative `STATUS_*` (incl. [`STATUS_UNSUPPORTED`] if no session is open or it already ended).
#[no_mangle]
pub extern "C" fn svm_pg_query(sql_ptr: *const u8, sql_len: usize) -> i32 {
    let set = |s: i32| unsafe { LAST_STATUS = s };
    // SAFETY: single-threaded wasm; exclusive access to the session static.
    let Some(session) = (unsafe { (*core::ptr::addr_of_mut!(PG_SESSION)).as_mut() }) else {
        set(STATUS_UNSUPPORTED);
        return -STATUS_UNSUPPORTED;
    };
    if session.ended {
        set(STATUS_UNSUPPORTED);
        return -STATUS_UNSUPPORTED;
    }
    // SAFETY: the host guarantees `[sql_ptr, sql_len)` is a live `svm_alloc`ation it just filled.
    let sql: &[u8] = if sql_ptr.is_null() || sql_len == 0 {
        &[]
    } else {
        unsafe { core::slice::from_raw_parts(sql_ptr, sql_len) }
    };
    session.vcpu.push_stdin(sql);
    if sql.last() != Some(&b'\n') {
        session.vcpu.push_stdin(b"\n"); // flush the statement to `--single`'s line reader
    }
    let status = pg_pump(session);
    pg_flush_stdout(session);
    set(status);
    if status == STATUS_OK {
        0
    } else {
        -status
    }
}

/// **Snapshot the open session's database** to a shippable data image. Serializes the live `mem_fs`
/// data dir — every file the backend has written (heap tables, indexes, WAL, catalogs) — into the same
/// [`svm_fs::encode_image`] blob [`svm_pg_open`] mounts, so the host can persist it (e.g. IndexedDB) and
/// reopen from it on the next visit: Postgres runs its normal startup recovery over the snapshot and all
/// committed state comes back. Best taken while the backend is parked at its prompt (between queries),
/// when the fs is quiescent — the natural resting state of an idle session. The bytes land in a
/// cdylib-managed allocation exposed by `svm_pg_snapshot_ptr`/`_len`, valid until the next snapshot (do
/// **not** `svm_dealloc` it). Sets [`svm_status`]; returns `0` on success, `-STATUS_UNSUPPORTED` if no
/// session is open.
#[no_mangle]
pub extern "C" fn svm_pg_snapshot() -> i32 {
    let set = |s: i32| unsafe { LAST_STATUS = s };
    // SAFETY: single-threaded wasm; exclusive access to the session static. A snapshot only reads the
    // fs handle, so an `ended` (exited/trapped) session is still serializable — its data dir holds the
    // final committed state, which recovery replays like any crash-consistent image.
    let Some(session) = (unsafe { (*core::ptr::addr_of!(PG_SESSION)).as_ref() }) else {
        set(STATUS_UNSUPPORTED);
        return -STATUS_UNSUPPORTED;
    };
    let image = session.fs_snap.image();
    // SAFETY: single-threaded wasm; read back only via the `svm_pg_snapshot_*` accessors.
    unsafe { stash(&mut *core::ptr::addr_of_mut!(PG_SNAP), image) };
    set(STATUS_OK);
    0
}

/// Close the open Postgres session (drop the backend + reclaim its program). Idempotent; a no-op when
/// none is open. The next [`svm_pg_open`] starts a fresh backend.
#[no_mangle]
pub extern "C" fn svm_pg_close() {
    pg_close_session();
}

/// A live per-frame **reactor** over an on-ramp guest — the interactive/graphical run model (the path
/// Doom rides), the browser twin of `svm-run`'s reactor `Session`. Instantiate once: run `_start`
/// (func 0) to stash the granted handles and run the C initializer, then call the guest's exported
/// `tick` once per host-driven frame. State (globals/BSS within the 256 KiB `SNAP_CAP` window)
/// **persists** between frames via the snapshot round-trip. Each `tick` presents a frame through the
/// `display` capability (captured into `frame`) and drains input through the `keyboard` capability
/// (`keys`, fed by the host). Single-threaded; the guest keeps its per-frame state in globals/BSS (a
/// grown `malloc` heap above the window is **not** persisted yet — the same slice-1 reactor scope as
/// `svm-run`, and the reason Doom itself needs the heap-persistence follow-on).
pub struct OnrampReactor {
    /// The persistent single-vCPU instance — its guest window (globals, BSS, **and** the grown heap)
    /// stays live between frames, so heavy-heap guests (Life, eventually Doom) keep their state.
    inst: bytecode::Reactor,
    host: Host,
    /// The reactor calling convention's data-stack base (`powerbox_entry_sp`), passed to each `tick`.
    entry_sp: u64,
    tick: svm_ir::FuncIdx,
    frame: std::sync::Arc<std::sync::Mutex<Option<Frame>>>,
    keys: KeyQueue,
    /// The `Debug` string of the last frame's trap (diagnostic; `None` until a `tick` traps).
    last_trap: Option<String>,
}

impl OnrampReactor {
    /// Open a reactor over `m`: grant the powerbox (prefix + `display`/`keyboard`), bind its
    /// manifest import slots, and run the entry once (init) over a **live** window kept for the
    /// per-frame `tick` calls. `Err(status)` if an import-bearing module lacks the manifest entry
    /// shape (fail-closed, IMPORTS.md phase 4), there is no exported `tick`, the module is outside
    /// the engine's subset, or the entry traps.
    pub fn open(m: &svm_ir::Module) -> Result<OnrampReactor, i32> {
        Self::open_inner(m, None)
    }

    /// Like [`open`](Self::open) but also grant an `fs` capability serving one read-only file `data`
    /// under the name `name` (matched as a substring of the guest's `open` path, like the native
    /// `doom_diff` differential). This is the WAD read path: Doom's `_start` (`doomgeneric_Create`)
    /// reads its IWAD through the `fs` cap during init, so the file must be served before `_start`
    /// runs — which this does, since [`grant_onramp_caps`] grants it ahead of the `_start` call.
    pub fn open_with_fs(
        m: &svm_ir::Module,
        name: String,
        data: Vec<u8>,
    ) -> Result<OnrampReactor, i32> {
        Self::open_inner(m, Some((name, data)))
    }

    fn open_inner(m: &svm_ir::Module, fs: Option<(String, Vec<u8>)>) -> Result<OnrampReactor, i32> {
        onramp_check(m).map_err(|_| STATUS_UNSUPPORTED)?;
        // The per-frame entry: the guest's exported `tick` (reactor convention `(sp) -> …`).
        let tick = m.resolve_export("tick").ok_or(STATUS_UNSUPPORTED)?;
        let entry_sp = svm_ir::powerbox_entry_sp(m);
        let mut host = Host::new();
        let (frame, keys) = grant_onramp_caps(&mut host, m, fs);
        let mut inst = bytecode::Reactor::open(m).ok_or(STATUS_UNSUPPORTED)?;
        // Run the entry (func 0) once on the live window with no args (phase 4: the manifest slot
        // bindings deliver the capabilities) to run the C initializer. The window (globals/BSS/heap)
        // then persists for every `tick`.
        let mut fuel = u64::MAX;
        match inst.call(0, &[], &mut fuel, &mut host) {
            Ok(_) => {}
            Err(_) => return Err(STATUS_TRAP),
        }
        Ok(OnrampReactor {
            inst,
            host,
            entry_sp,
            tick,
            frame,
            keys,
            last_trap: None,
        })
    }

    /// Run one frame: call the guest's `tick` on the **live** window (all prior-frame state — globals,
    /// BSS, heap — intact), returning `(status, stdout-delta)`. `STATUS_OK` = keep going; `STATUS_EXIT`
    /// = the guest called `Exit`; `STATUS_TRAP` = a trap. The presented frame (if any) is read via
    /// [`take_frame`](Self::take_frame).
    pub fn frame(&mut self) -> (i32, Vec<u8>) {
        let stdout_before = self.host.stdout.len();
        let args = [Value::I64(self.entry_sp as i64)];
        let mut fuel = u64::MAX;
        let status = match self.inst.call(self.tick, &args, &mut fuel, &mut self.host) {
            Ok(_) => STATUS_OK,
            Err(Trap::Exit(_)) => STATUS_EXIT,
            Err(t) => {
                self.last_trap = Some(format!("{t:?}"));
                STATUS_TRAP
            }
        };
        let delta = self.host.stdout[stdout_before..].to_vec();
        (status, delta)
    }

    /// The `Debug` string of the last frame's trap (diagnostic), or `""` if none.
    pub fn last_trap(&self) -> &str {
        self.last_trap.as_deref().unwrap_or("")
    }

    /// Take the frame the last `tick` presented through `display` (`None` if it presented none).
    pub fn take_frame(&self) -> Option<Frame> {
        self.frame.lock().unwrap().take()
    }

    /// Enqueue a key event for the guest to `poll` through the `keyboard` capability next frame.
    /// `pressed` is 1 (down) / 0 (up); `keycode` is the platform key id (e.g. a JS `keyCode`).
    pub fn push_key(&self, keycode: i32, pressed: i32) {
        self.keys
            .lock()
            .unwrap()
            .push_back(((pressed & 1) << 16) | (keycode & 0xffff));
    }
}

/// Like [`OnrampReactor`], but the guest window lives in a **caller-provided region of this module's
/// own linear memory** (a [`Region::shared`](svm_interp::Region::shared) over `[win_ptr, win_size)`)
/// rather than a window the engine backs internally. That relocation is the substrate the wasm-JIT
/// **reactor** tier needs (BROWSER.md § "wasm-JIT tier", slice 5b): the emitted `tick` — a JS-compiled
/// wasm module that imports `env.memory` = *this* cdylib's linear memory — must read and write the same
/// bytes the interpreter seeds, so `_start` (interpreter), the per-frame `tick`, and any cross-tier
/// interpreter bounce all operate over **one** window. Backed by the resumable, tier-up-capable
/// [`bytecode::VcpuReactor`]; with no eligibility set (this slice) every `tick` interprets, so it is a
/// faithful, **byte-identical** substitute for [`OnrampReactor`] — the differential the reactor tests
/// assert — with the window merely relocated into shared linear memory. The emitted-`tick` path (slice
/// 5c) rides the same window and shared host.
pub struct SharedOnrampReactor {
    reactor: bytecode::VcpuReactor,
    /// The powerbox (granted caps + stashed `_start` handles), shared across `open` and every `frame`
    /// so the `display`/`keyboard`/`fs` caps and their state persist frame-to-frame.
    host: std::sync::Mutex<Host>,
    /// Keep-alive for an **owned** backing (the native/test path allocates it here); `None` when the
    /// window is caller-owned (the FFI path hands in a pointer into this module's linear memory). The
    /// `Region::shared` in `reactor`/`_back` addresses these bytes, so this must outlive the reactor —
    /// a `Box<[u8]>`'s heap allocation is stable across moves of the struct.
    _backing: Option<Box<[u8]>>,
    /// The shared backing region (kept so its lifetime is tied to the reactor's).
    _back: std::sync::Arc<svm_interp::Region>,
    entry_sp: u64,
    tick: svm_ir::FuncIdx,
    frame: std::sync::Arc<std::sync::Mutex<Option<Frame>>>,
    keys: KeyQueue,
    last_trap: Option<String>,
}

impl SharedOnrampReactor {
    /// Open a shared-window reactor over `m` with an **owned** backing of `1 << win_log2` bytes
    /// (allocated + kept alive here) — the native/test entry. `win_log2` must be ≥ the module's mapped
    /// size and large enough for the guest's grown heap. See [`open_shared`](Self::open_shared) for the
    /// FFI entry that borrows a caller-owned window.
    pub fn open_owned(m: &svm_ir::Module, win_log2: u8) -> Result<SharedOnrampReactor, i32> {
        Self::open_owned_inner(m, win_log2, None)
    }

    /// Like [`open_owned`](Self::open_owned) but also grant an `fs` capability serving one read-only
    /// file `data` under `name` (the WAD read path — see [`OnrampReactor::open_with_fs`]).
    pub fn open_owned_with_fs(
        m: &svm_ir::Module,
        win_log2: u8,
        name: String,
        data: Vec<u8>,
    ) -> Result<SharedOnrampReactor, i32> {
        Self::open_owned_inner(m, win_log2, Some((name, data)))
    }

    fn open_owned_inner(
        m: &svm_ir::Module,
        win_log2: u8,
        fs: Option<(String, Vec<u8>)>,
    ) -> Result<SharedOnrampReactor, i32> {
        let win_size = 1u64 << win_log2;
        let mut backing = vec![0u8; win_size as usize].into_boxed_slice();
        let ptr = backing.as_mut_ptr();
        // SAFETY: `backing` (a `Box<[u8]>` of `win_size` bytes) is owned by the returned struct and its
        // heap allocation is pointer-stable across the struct's moves, so `[ptr, win_size)` stays valid
        // and exclusively this reactor's window for its whole lifetime.
        let back = std::sync::Arc::new(unsafe { svm_interp::Region::shared(ptr, win_size) });
        Self::open_over(m, back, Some(backing), fs)
    }

    /// Open a shared-window reactor over a **caller-owned** window `[win_ptr, win_ptr+win_size)` of this
    /// module's linear memory — the FFI entry (the host `svm_alloc`s the window and keeps it live for
    /// the reactor's lifetime).
    ///
    /// # Safety
    /// `[win_ptr, win_size)` must be a live region of this module's linear memory, used solely as this
    /// reactor's window and kept valid (not freed, not reused) until the reactor is dropped.
    pub unsafe fn open_shared(
        m: &svm_ir::Module,
        win_ptr: *mut u8,
        win_size: u64,
        fs: Option<(String, Vec<u8>)>,
    ) -> Result<SharedOnrampReactor, i32> {
        let back = std::sync::Arc::new(svm_interp::Region::shared(win_ptr, win_size));
        Self::open_over(m, back, None, fs)
    }

    fn open_over(
        m: &svm_ir::Module,
        back: std::sync::Arc<svm_interp::Region>,
        backing: Option<Box<[u8]>>,
        fs: Option<(String, Vec<u8>)>,
    ) -> Result<SharedOnrampReactor, i32> {
        onramp_check(m).map_err(|_| STATUS_UNSUPPORTED)?;
        let tick = m.resolve_export("tick").ok_or(STATUS_UNSUPPORTED)?;
        let entry_sp = svm_ir::powerbox_entry_sp(m);
        let mut host = Host::new();
        let (frame, keys) = grant_onramp_caps(&mut host, m, fs);
        // Run the entry (func 0) once over the shared window with no args (phase 4: the manifest
        // slot bindings deliver the capabilities) to run the C initializer, seeding +
        // data-initialising the window (the once). The window then persists in the shared backing
        // for every `tick`.
        let host = std::sync::Mutex::new(host);
        let reactor =
            bytecode::VcpuReactor::open(m, back.clone(), &host, &[]).map_err(|_| STATUS_TRAP)?;
        Ok(SharedOnrampReactor {
            reactor,
            host,
            _backing: backing,
            _back: back,
            entry_sp,
            tick,
            frame,
            keys,
            last_trap: None,
        })
    }

    /// Run one frame: call the guest's `tick` on the **live** shared window (all prior-frame state
    /// intact), returning `(status, stdout-delta)`. `STATUS_OK` = keep going; `STATUS_EXIT` = the guest
    /// called `Exit`; `STATUS_TRAP` = a trap. This slice interprets `tick` (no eligibility set), so the
    /// `service` closure is never invoked. The presented frame (if any) is read via
    /// [`take_frame`](Self::take_frame).
    pub fn frame(&mut self) -> (i32, Vec<u8>) {
        let stdout_before = self.host.lock().unwrap().stdout.len();
        let args = [Value::I64(self.entry_sp as i64)];
        // No JIT eligibility set → `service` is unreachable; propagate a trap if one ever surfaced.
        let result = self.reactor.frame(
            self.tick,
            &args,
            &self.host,
            |_func, _argv, _mapped, _info| Err(Trap::Malformed),
        );
        let status = match result {
            Ok(_) => STATUS_OK,
            Err(Trap::Exit(_)) => STATUS_EXIT,
            Err(t) => {
                self.last_trap = Some(format!("{t:?}"));
                STATUS_TRAP
            }
        };
        let delta = self.host.lock().unwrap().stdout[stdout_before..].to_vec();
        (status, delta)
    }

    /// The `Debug` string of the last frame's trap (diagnostic), or `""` if none.
    pub fn last_trap(&self) -> &str {
        self.last_trap.as_deref().unwrap_or("")
    }

    /// Take the frame the last `tick` presented through `display` (`None` if it presented none).
    pub fn take_frame(&self) -> Option<Frame> {
        self.frame.lock().unwrap().take()
    }

    /// Enqueue a key event for the guest to `poll` through the `keyboard` capability next frame.
    pub fn push_key(&self, keycode: i32, pressed: i32) {
        self.keys
            .lock()
            .unwrap()
            .push_back(((pressed & 1) << 16) | (keycode & 0xffff));
    }
}

/// A **wasm-JIT reactor** over an on-ramp guest (BROWSER.md § "wasm-JIT tier", slice 5c): the guest's
/// whole `tick` runs on **emitted wasm** each frame instead of the interpreter, over the same shared
/// window as [`SharedOnrampReactor`]. The host (JS in the browser, `wasmi` in the native differential)
/// compiles the emitted module ([`emitted_wasm`](Self::emitted_wasm)), instantiates it against *this*
/// module's linear memory, and calls `f{tick}(win, env, entry_sp)` per frame; the emitted code bounces
/// each call to a non-emitted (cross-tier) helper back through [`run_cross_tier`](Self::run_cross_tier),
/// which runs that callee on the interpreter over the shared window **with the powerbox** (so
/// `display`/`keyboard`/`fs`/`exit` all resolve) — its memory effects landing in the bytes the emitted
/// code reads.
///
/// The mapped window is enlarged at open (`win_log2`) so the guest's heap growth stays **within**
/// mapped — Doom `vm_map`s its zone heap to ~11 MiB, above its native 4 MiB mapped window; keeping it
/// inside mapped lets the emitter's static-`mapped` confinement mask cover every emitted access, and
/// lets a fresh-`Mem` cross-tier run reach the whole live window without committed-page tracking. The
/// guest stays confined to the (larger, still power-of-two) window.
pub struct JitOnrampReactor {
    /// The import-resolved, window-enlarged module — cross-tier callees are interpreted from it.
    module: svm_ir::Module,
    /// The module compiled **once** for cross-tier runs. Recompiling per `env.call_interp` bounce
    /// (a handful per frame) otherwise dominates the frame — for Doom, ~6 ms × 3 ≈ 19 ms of a 20 ms
    /// frame; cached, a cross-tier call is just build-window + interpret.
    program: bytecode::SharedProgram,
    /// The powerbox (granted caps + `_start`'s stashed handles), shared across `_start` and every
    /// per-frame cross-tier callee so caps + their state persist.
    host: Host,
    /// Keep-alive for an owned backing (native path); `None` when the window is caller-owned (FFI).
    _backing: Option<Box<[u8]>>,
    back: std::sync::Arc<svm_interp::Region>,
    /// The window base as a byte offset in this module's linear memory — the emitted `f{tick}`'s `win`
    /// argument (the address the emitted code masks its accesses against).
    win_base: usize,
    entry_sp: u64,
    tick: svm_ir::FuncIdx,
    /// The emitted wasm for the whole `tick` (the host compiles + runs it) and the per-function emitted
    /// bitmap (`emitted[i]` ⇒ `f{i}` runs on wasm; the rest bounce through `run_cross_tier`).
    emitted_wasm: Vec<u8>,
    emitted: Vec<bool>,
    frame: std::sync::Arc<std::sync::Mutex<Option<Frame>>>,
    keys: KeyQueue,
    last_trap: Option<String>,
}

impl JitOnrampReactor {
    /// Open a wasm-JIT reactor over `m` with an **owned** backing of `1 << win_log2` bytes (native
    /// path). `shared_memory` selects the emitted `env.memory` import's shared flag (`true` for the
    /// browser threads build; `false` for the `wasmi` differential — the codegen is otherwise
    /// identical). `Err(status)` if imports don't resolve, there is no `tick`, `_start` traps, or the
    /// `tick` isn't wasm-JIT-emittable (it falls back to [`SharedOnrampReactor`]).
    pub fn open_owned_jit(
        m: &svm_ir::Module,
        win_log2: u8,
        shared_memory: bool,
        fs: Option<(String, Vec<u8>)>,
    ) -> Result<JitOnrampReactor, i32> {
        let win_size = 1u64 << win_log2;
        let mut backing = vec![0u8; win_size as usize].into_boxed_slice();
        let ptr = backing.as_mut_ptr();
        // SAFETY: `backing` is owned by the returned struct and its heap allocation is pointer-stable
        // across the struct's moves, so `[ptr, win_size)` stays valid + exclusive for the run.
        let win_base = ptr as usize;
        let back = std::sync::Arc::new(unsafe { svm_interp::Region::shared(ptr, win_size) });
        Self::open_over_jit(
            m,
            back,
            Some(backing),
            win_base,
            win_log2,
            shared_memory,
            fs,
        )
    }

    /// Open a wasm-JIT reactor over a **caller-owned** window `[win_ptr, win_ptr+win_size)` of this
    /// module's linear memory (the FFI path). `win_size` must equal `1 << win_log2`.
    ///
    /// # Safety
    /// `[win_ptr, win_size)` must be a live region of this module's linear memory, used solely as this
    /// reactor's window and kept valid until the reactor is dropped.
    pub unsafe fn open_shared_jit(
        m: &svm_ir::Module,
        win_ptr: *mut u8,
        win_size: u64,
        win_log2: u8,
        shared_memory: bool,
        fs: Option<(String, Vec<u8>)>,
    ) -> Result<JitOnrampReactor, i32> {
        let win_base = win_ptr as usize;
        let back = std::sync::Arc::new(svm_interp::Region::shared(win_ptr, win_size));
        Self::open_over_jit(m, back, None, win_base, win_log2, shared_memory, fs)
    }

    fn open_over_jit(
        m: &svm_ir::Module,
        back: std::sync::Arc<svm_interp::Region>,
        backing: Option<Box<[u8]>>,
        win_base: usize,
        win_log2: u8,
        shared_memory: bool,
        fs: Option<(String, Vec<u8>)>,
    ) -> Result<JitOnrampReactor, i32> {
        onramp_check(m).map_err(|_| STATUS_UNSUPPORTED)?;
        let mut module = m.clone();
        // Hoist inline `cap.call`s into cross-tier wrapper functions so a hot `tick` that interleaves
        // compute with a once-per-frame present/poll cap call still emits (its hot path runs on wasm;
        // only the cap wrapper bounces to the interpreter). Mutates the module BOTH tiers use: the
        // emitter reads it below, and `run_cross_tier` runs the wrappers on the interpreter.
        svm_wasm_jit::outline_cap_calls(&mut module);
        // Enlarge the mapped window to cover the guest's grown heap (see the struct docs).
        if let Some(mc) = module.memory.as_mut() {
            if (mc.size_log2 as u32) < win_log2 as u32 {
                mc.size_log2 = win_log2;
            }
        }
        let tick = module.resolve_export("tick").ok_or(STATUS_UNSUPPORTED)?;
        let entry_sp = svm_ir::powerbox_entry_sp(&module);
        let mut host = Host::new();
        let (frame, keys) = grant_onramp_caps(&mut host, &module, fs);
        // Compile the module **once** — reused for the entry and every per-frame cross-tier bounce.
        let program = bytecode::SharedProgram::compile(&module).ok_or(STATUS_UNSUPPORTED)?;
        // Run the entry (func 0) once over the shared window with no args (phase 4: the manifest
        // slot bindings deliver the capabilities), servicing `cap.call`s (Doom's WAD read) inline
        // against the powerbox. The window then persists in `back` for every frame.
        let mut fuel = u64::MAX;
        match program.run_over(0, &[], &mut fuel, back.clone(), &mut host, true) {
            Ok(_) => {}
            Err(_) => return Err(STATUS_TRAP),
        }
        // Emit the whole `tick`, wasm-driven (cross-tier helpers routed to `env.call_interp`). The
        // front door derives the strategy: a `tick` whose reachable set can suspend is *not*
        // wasm-drivable (a JITted frame can't unwind across a stack switch), so it reports
        // `InterpDriven` instead of emitting a reactor this driver couldn't run — fall back to the
        // pure interpreter then, exactly as when the `tick` is out of subset.
        let artifact = svm_wasm_jit::compile_jit(
            &module,
            svm_wasm_jit::Shape::Reactor { entry: tick },
            shared_memory,
        )
        .map_err(|_| STATUS_UNSUPPORTED)?;
        let svm_wasm_jit::DriveMode::WasmDriven { .. } = artifact.drive else {
            return Err(STATUS_UNSUPPORTED);
        };
        let (emitted_wasm, emitted) = (artifact.wasm, artifact.emitted);
        Ok(JitOnrampReactor {
            module,
            program,
            host,
            _backing: backing,
            back,
            win_base,
            entry_sp,
            tick,
            emitted_wasm,
            emitted,
            frame,
            keys,
            last_trap: None,
        })
    }

    /// The window base as a byte offset in this module's linear memory — the emitted `f{tick}`'s `win`.
    pub fn win_base(&self) -> usize {
        self.win_base
    }

    /// The emitted wasm module for the whole `tick` — the host compiles + instantiates it against this
    /// module's linear memory and calls the exported `f{tick}(win, env, entry_sp)`.
    pub fn emitted_wasm(&self) -> &[u8] {
        &self.emitted_wasm
    }

    /// The per-function emitted bitmap: `emitted[i]` ⇒ `f{i}` runs on wasm (the rest bounce through
    /// [`run_cross_tier`](Self::run_cross_tier)).
    pub fn emitted(&self) -> &[bool] {
        &self.emitted
    }

    /// The reactor calling-convention data-stack base, passed as the emitted `f{tick}`'s `sp` argument.
    pub fn entry_sp(&self) -> u64 {
        self.entry_sp
    }

    /// The SVM index of the exported `tick` — the emitted export name is `f{tick}`.
    pub fn tick(&self) -> svm_ir::FuncIdx {
        self.tick
    }

    /// **Cross-tier bounce.** Run non-emitted `func(args)` on the interpreter over the shared window
    /// with the powerbox — the callback the emitted `tick`'s `env.call_interp` drives. Memory effects
    /// land in the shared window (the bytes the emitted code reads); `cap.call`s resolve against the
    /// persistent host (so a `display.present` populates the frame cell, `keyboard.poll` drains input).
    /// `Err(Trap::Exit)` is the guest's `Exit`; any other `Err` is a trap.
    pub fn run_cross_tier(&mut self, func: u32, args: &[Value]) -> Result<Vec<Value>, Trap> {
        let mut fuel = u64::MAX;
        // Use the once-compiled `program` (no per-call recompile — the frame's dominant cost otherwise).
        self.program.run_over(
            func,
            args,
            &mut fuel,
            self.back.clone(),
            &mut self.host,
            false,
        )
    }

    /// The signature of cross-tier `func` (the host marshals `env.call_interp`'s i64 arg/result slots
    /// per these types).
    pub fn func_sig(&self, func: u32) -> (&[svm_ir::ValType], &[svm_ir::ValType]) {
        let f = &self.module.funcs[func as usize];
        (&f.params, &f.results)
    }

    /// Record a trap's `Debug` string (diagnostic); returns `""` if none.
    pub fn last_trap(&self) -> &str {
        self.last_trap.as_deref().unwrap_or("")
    }

    /// Set the last-trap diagnostic (the host records the trap that unwound a frame's emitted `tick`).
    pub fn set_last_trap(&mut self, s: String) {
        self.last_trap = Some(s);
    }

    /// Take the frame the last `tick` presented through `display` (`None` if it presented none).
    pub fn take_frame(&self) -> Option<Frame> {
        self.frame.lock().unwrap().take()
    }

    /// Enqueue a key event for the guest to `poll` through the `keyboard` capability next frame.
    pub fn push_key(&self, keycode: i32, pressed: i32) {
        self.keys
            .lock()
            .unwrap()
            .push_back(((pressed & 1) << 16) | (keycode & 0xffff));
    }
}

/// A **single-shot** wasm-JIT run of an on-ramp module — the run-to-completion twin of
/// [`JitOnrampReactor`] (which drives an exported `tick` per frame). Here the whole program *is* func 0
/// (`_start`), so we emit **that** and run it once: the paramless `_start` reads its capabilities
/// through the manifest slot bindings / `cap.self.resolve` (phase 4 — no handle params), seeds the
/// heap, and calls `main(sp)` — all on emitted wasm, with the 47/103 cross-tier helpers (Lua/SQLite)
/// bouncing to the interpreter through `env.call_interp` over the same window (so
/// `write`/`read`/`exit` resolve against the powerbox). Unlike the reactor, `_start` is **not**
/// pre-run on the interpreter; instead the `.data`/`.rodata` segments are materialized into the
/// window up front (the emitted `_start` seeds only the heap), then `f0(win, env)` runs the
/// program. `stdout`/`stderr`/`exit_code` are read back from the host afterward, exactly as
/// [`onramp_exec`] captures them — so the two tiers are a stdout/exit differential.
pub struct JitOnrampRun {
    module: svm_ir::Module,
    program: bytecode::SharedProgram,
    host: Host,
    _backing: Option<Box<[u8]>>,
    back: std::sync::Arc<svm_interp::Region>,
    win_base: usize,
    emitted_wasm: Vec<u8>,
    emitted: Vec<bool>,
    frame: std::sync::Arc<std::sync::Mutex<Option<Frame>>>,
    /// The emitted `f0`'s trailing `...slots` args. Empty for the paramless `_start` (IMPORTS.md phase
    /// 4); the warm+JIT `eval_run` entry (WASM_AOT.md) carries one `I64` — the powerbox `sp` — here.
    slots: Vec<Value>,
    last_trap: Option<String>,
    /// Set when a cross-tier bounce returns `Trap::Exit(code)` (the guest called `exit`), unwinding the
    /// emitted `f0`; `exited` distinguishes "exited with code 0" from "returned 0".
    exit_code: i32,
    exited: bool,
    /// The value the emitted `f0` returned (the guest's top-level result), reported by the JS driver via
    /// [`svm_onramp_jit_run_report`]. Meaningful only when the run *returned* (not exited/trapped) — it is
    /// what [`svm_run_onramp`]'s `value` is on the interpreter, so the two tiers agree on the result.
    returned_value: i64,
    /// Set when the emitted `f0` unwound on a **trap** (a wasm `unreachable`, or a cross-tier bounce that
    /// trapped rather than `exit`ed) instead of returning. The JS driver can't tell an `exit` unwind from
    /// a trap unwind on its own, so it reports "threw"; combined with `exited` (set Rust-side on a
    /// cross-tier `Exit`), a throw that did not `exit` is a trap. Keeps the runner from reporting a
    /// truncated run as `STATUS_OK` (INVARIANT 9: a fast backend never runs wrong — it traps or declines).
    trapped: bool,
}

/// How a single-shot JIT run feeds its guest — the twin of [`onramp_exec`] (stdin) vs
/// [`onramp_fs_exec`] (a seeded memfs + argv). `Stdin` grants the plain on-ramp powerbox
/// ([`grant_onramp_caps`]); `Fs` grants the headless powerbox with a mounted `fs` image and seeds
/// `argv` at `POWERBOX_ARGS_BASE` (the chibicc-in-the-browser card: `/in.c` + `/include/*.h`).
enum RunInput {
    Stdin(Vec<u8>),
    Fs {
        image: Vec<u8>,
        argv: Vec<Vec<u8>>,
        stdin: Vec<u8>,
    },
}

impl JitOnrampRun {
    /// Open a single-shot JIT run over a **caller-owned** window (the FFI path: `win_ptr` addresses this
    /// module's own linear memory). `win_size == 1 << win_log2`.
    ///
    /// # Safety
    /// `[win_ptr, win_size)` must be a live region of this module's linear memory, used solely as this
    /// run's window and kept valid until the run is dropped.
    pub unsafe fn open_shared_run(
        m: &svm_ir::Module,
        win_ptr: *mut u8,
        win_size: u64,
        win_log2: u8,
        shared_memory: bool,
        stdin: Vec<u8>,
    ) -> Result<JitOnrampRun, i32> {
        let win_base = win_ptr as usize;
        let back = std::sync::Arc::new(svm_interp::Region::shared(win_ptr, win_size));
        Self::open_over_run(
            m,
            back,
            None,
            win_ptr,
            win_size,
            win_base,
            win_log2,
            shared_memory,
            RunInput::Stdin(stdin),
        )
    }

    /// Open a single-shot JIT run over an **owned** window (heap-backed, pointer-stable for the run).
    /// `win_log2` is a **minimum** — the window is grown to the module's declared `size_log2` when
    /// larger (Lua declares 64 MiB), so the allocated window always equals the size the emitter masks
    /// to; a smaller allocation would fault any access into the module's upper address range.
    pub fn open_owned_run(
        m: &svm_ir::Module,
        win_log2: u8,
        shared_memory: bool,
        stdin: Vec<u8>,
    ) -> Result<JitOnrampRun, i32> {
        Self::open_owned_run_with(m, win_log2, shared_memory, RunInput::Stdin(stdin))
    }

    /// Like [`open_owned_run`](Self::open_owned_run), but the guest reads its input from a seeded
    /// **memfs** `image` (mounted on the `fs` cap) with `argv` seeded at `POWERBOX_ARGS_BASE` — the
    /// single-shot JIT twin of [`onramp_fs_exec`]. This is the chibicc-in-the-browser card's fast tier:
    /// chibicc `fopen`s `/in.c` + `/include/*.h`, emits SVM-IR text on stdout.
    pub fn open_owned_run_fs(
        m: &svm_ir::Module,
        win_log2: u8,
        shared_memory: bool,
        image: &[u8],
        argv: &[&[u8]],
        stdin: Vec<u8>,
    ) -> Result<JitOnrampRun, i32> {
        Self::open_owned_run_with(
            m,
            win_log2,
            shared_memory,
            RunInput::Fs {
                image: image.to_vec(),
                argv: argv.iter().map(|a| a.to_vec()).collect(),
                stdin,
            },
        )
    }

    /// Open a single-shot JIT run over a **caller-owned** window with a seeded memfs (the FFI twin of
    /// [`open_owned_run_fs`](Self::open_owned_run_fs)).
    ///
    /// # Safety
    /// As [`open_shared_run`](Self::open_shared_run): `[win_ptr, win_size)` must be a live region of this
    /// module's linear memory, used solely as this run's window, valid until the run is dropped.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn open_shared_run_fs(
        m: &svm_ir::Module,
        win_ptr: *mut u8,
        win_size: u64,
        win_log2: u8,
        shared_memory: bool,
        image: &[u8],
        argv: &[&[u8]],
        stdin: Vec<u8>,
    ) -> Result<JitOnrampRun, i32> {
        let win_base = win_ptr as usize;
        let back = std::sync::Arc::new(svm_interp::Region::shared(win_ptr, win_size));
        Self::open_over_run(
            m,
            back,
            None,
            win_ptr,
            win_size,
            win_base,
            win_log2,
            shared_memory,
            RunInput::Fs {
                image: image.to_vec(),
                argv: argv.iter().map(|a| a.to_vec()).collect(),
                stdin,
            },
        )
    }

    fn open_owned_run_with(
        m: &svm_ir::Module,
        win_log2: u8,
        shared_memory: bool,
        input: RunInput,
    ) -> Result<JitOnrampRun, i32> {
        let declared = m.memory.map_or(0, |mc| mc.size_log2);
        let win_log2 = win_log2.max(declared);
        let win_size = 1u64 << win_log2;
        let mut backing = vec![0u8; win_size as usize].into_boxed_slice();
        let ptr = backing.as_mut_ptr();
        let win_base = ptr as usize;
        // SAFETY: `backing` is owned by the returned struct and pointer-stable across its moves, so
        // `[ptr, win_size)` stays valid + exclusive for the run.
        let back = std::sync::Arc::new(unsafe { svm_interp::Region::shared(ptr, win_size) });
        Self::open_over_run(
            m,
            back,
            Some(backing),
            ptr,
            win_size,
            win_base,
            win_log2,
            shared_memory,
            input,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn open_over_run(
        m: &svm_ir::Module,
        back: std::sync::Arc<svm_interp::Region>,
        backing: Option<Box<[u8]>>,
        win_ptr: *mut u8,
        win_size: u64,
        win_base: usize,
        win_log2: u8,
        shared_memory: bool,
        input: RunInput,
    ) -> Result<JitOnrampRun, i32> {
        onramp_check(m).map_err(|_| STATUS_UNSUPPORTED)?;
        let mut module = m.clone();
        svm_wasm_jit::outline_cap_calls(&mut module);
        // Enlarge the mapped window to cover the guest's heap (fixed — emitted code can't grow it).
        if let Some(mc) = module.memory.as_mut() {
            if (mc.size_log2 as u32) < win_log2 as u32 {
                mc.size_log2 = win_log2;
            }
        }
        // Build the powerbox + the window-prefix seed (`init_mem`, the argv blob for the `Fs` path)
        // from the input shape. `frame` is only ever populated by a `display.present` — kept for
        // struct parity; a compiler/compute guest never presents.
        let (host, init_mem, frame): (Host, Vec<u8>, _) = match input {
            RunInput::Stdin(stdin) => {
                let mut host = Host::new();
                host.stdin = stdin;
                // The powerbox prefix (stdout/stdin/exit/…) bound to the manifest slots and registered
                // by name; `display` too (unused by a pure compute guest, present for parity with
                // `onramp_exec`). No `fs` (input comes from stdin).
                let (frame, _keys) = grant_onramp_caps(&mut host, &module, None);
                (host, Vec::new(), frame)
            }
            RunInput::Fs { image, argv, stdin } => {
                // The headless memfs powerbox (`fs` image + argv at POWERBOX_ARGS_BASE), exactly as the
                // bytecode `onramp_fs_exec` builds it — the `MemFsHandle` is dropped (no snapshot here).
                let argv_refs: Vec<&[u8]> = argv.iter().map(|a| a.as_slice()).collect();
                let (mut host, init_mem, _fsh) = pg_setup(&module, &image, &argv_refs)?;
                host.stdin = stdin;
                let frame = std::sync::Arc::new(std::sync::Mutex::new(None));
                (host, init_mem, frame)
            }
        };
        // Compile once — reused for every cross-tier bounce.
        let program = bytecode::SharedProgram::compile(&module).ok_or(STATUS_UNSUPPORTED)?;
        // Materialize the window before the emitted `_start` runs (the interpreter does this at
        // instantiation; the emitted `_start` seeds only the heap + stashes handles): first the argv
        // prefix (`init_mem`, empty for stdin), then `.data`/`.rodata`. Data segments start at the
        // module's data page (65 KiB), so they never overlap the argv prefix at POWERBOX_ARGS_BASE.
        // SAFETY: `[win_ptr, win_size)` is a live window (owned backing or the caller's linear memory).
        unsafe {
            let win = core::slice::from_raw_parts_mut(win_ptr, win_size as usize);
            if init_mem.len() <= win.len() {
                win[..init_mem.len()].copy_from_slice(&init_mem);
            }
            for seg in &module.data {
                let off = seg.offset as usize;
                let end = off.saturating_add(seg.bytes.len());
                if end <= win.len() {
                    win[off..end].copy_from_slice(&seg.bytes);
                }
            }
        }
        // Emit rooted at func 0 (`_start`), wasm-driven; cross-tier helpers route to `env.call_interp`.
        // The front door reports `InterpDriven` (→ fall back to the pure interpreter) if `_start` is out
        // of subset or its reachable set can suspend — this driver can only run a wasm-driven artifact.
        let artifact = svm_wasm_jit::compile_jit(
            &module,
            svm_wasm_jit::Shape::Batch { entry: 0 },
            shared_memory,
        )
        .map_err(|_| STATUS_UNSUPPORTED)?;
        let svm_wasm_jit::DriveMode::WasmDriven { .. } = artifact.drive else {
            return Err(STATUS_UNSUPPORTED);
        };
        let (emitted_wasm, emitted) = (artifact.wasm, artifact.emitted);
        Ok(JitOnrampRun {
            module,
            program,
            host,
            _backing: backing,
            back,
            win_base,
            emitted_wasm,
            emitted,
            frame,
            slots: Vec::new(),
            last_trap: None,
            exit_code: 0,
            exited: false,
            returned_value: 0,
            trapped: false,
        })
    }

    /// Open a **warm+JIT** `eval_run` run (WASM_AOT.md warm+JIT) over the caller-owned warm-session
    /// window. Three differences from [`open_shared_run`]: the emit is rooted at the module's `eval_run`
    /// export (not `_start`), the window is **not** re-seeded with data segments (the caller restores the
    /// warm image before each drive), and the entry's `sp` rides along as the emitted `f0`'s trailing
    /// slot. Returns [`STATUS_UNSUPPORTED`] if `eval_run` isn't wasm-drivable — the caller then evaluates
    /// on the interpreter warm path ([`svm_warm_eval`]).
    ///
    /// # Safety
    /// `back` must alias the live warm-session window `[win_ptr, 1 << win_log2)`, kept valid until the
    /// run is dropped (it is owned by the [`WarmSession`] that holds this run, freed in `svm_warm_close`).
    #[allow(clippy::too_many_arguments)]
    unsafe fn open_warm_eval(
        m: &svm_ir::Module,
        back: std::sync::Arc<svm_interp::Region>,
        win_ptr: *mut u8,
        win_log2: u8,
        shared_memory: bool,
        eval_fn: svm_ir::FuncIdx,
        entry_sp: u64,
    ) -> Result<JitOnrampRun, i32> {
        onramp_check(m).map_err(|_| STATUS_UNSUPPORTED)?;
        let win_base = win_ptr as usize;
        let mut module = m.clone();
        svm_wasm_jit::outline_cap_calls(&mut module);
        // The warm module already declares the mapped window; keep the belt-and-braces enlarge for parity
        // with `open_over_run` (a no-op when `size_log2 == win_log2`).
        if let Some(mc) = module.memory.as_mut() {
            if (mc.size_log2 as u32) < win_log2 as u32 {
                mc.size_log2 = win_log2;
            }
        }
        // The powerbox the interpreter warm path grants (`svm_warm_eval`) — a fresh host is re-granted per
        // Run via [`reset_warm`]; this one seeds `open`, replaced before the first drive.
        let mut host = Host::new();
        let (frame, _keys) = grant_onramp_caps(&mut host, &module, None);
        // Compiled once — reused for every cross-tier bounce (`write`/`read`/`exit` off the emitted eval).
        let program = bytecode::SharedProgram::compile(&module).ok_or(STATUS_UNSUPPORTED)?;
        // Emit rooted at `eval_run` (not `_start`); the reachable-set / concurrency gates are unchanged,
        // so a driver whose eval can suspect or leaves the subset declines to the interpreter.
        let artifact = svm_wasm_jit::compile_jit(
            &module,
            svm_wasm_jit::Shape::Batch { entry: eval_fn },
            shared_memory,
        )
        .map_err(|_| STATUS_UNSUPPORTED)?;
        let svm_wasm_jit::DriveMode::WasmDriven { .. } = artifact.drive else {
            return Err(STATUS_UNSUPPORTED);
        };
        let (emitted_wasm, emitted) = (artifact.wasm, artifact.emitted);
        Ok(JitOnrampRun {
            module,
            program,
            host,
            _backing: None,
            back,
            win_base,
            emitted_wasm,
            emitted,
            frame,
            slots: vec![Value::I64(entry_sp as i64)],
            last_trap: None,
            exit_code: 0,
            exited: false,
            returned_value: 0,
            trapped: false,
        })
    }

    /// Reset a cached warm+JIT run for a fresh Run: rebuild the powerbox (a clean `Host` + `grant_onramp_
    /// caps`, so captured streams and the frame cell start empty) and clear the finish flags. The caller
    /// restores the warm image into the window separately (`svm_warm_jit_prepare`); together they give the
    /// same fresh-per-Run state the interpreter warm path gets, so no guest state crosses Runs.
    fn reset_warm(&mut self, stdin: Vec<u8>) {
        let mut host = Host::new();
        host.stdin = stdin;
        let (frame, _keys) = grant_onramp_caps(&mut host, &self.module, None);
        self.host = host;
        self.frame = frame;
        self.exit_code = 0;
        self.exited = false;
        self.returned_value = 0;
        self.trapped = false;
        self.last_trap = None;
    }

    /// The emitted wasm (the host compiles + instantiates it, then calls `f0(win, env, ...slots)` once).
    pub fn emitted_wasm(&self) -> &[u8] {
        &self.emitted_wasm
    }
    /// The window base as a byte offset in this module's linear memory — the emitted `f0`'s `win`.
    pub fn win_base(&self) -> usize {
        self.win_base
    }
    /// The emitted `f0`'s trailing `...slots` args. Empty for the paramless `_start` (IMPORTS.md phase
    /// 4 — capabilities arrive via the manifest slot bindings); the warm+JIT `eval_run` entry carries one
    /// `I64` (the powerbox `sp`), passed by the warm-JIT driver as the emitted `f0`'s third argument.
    pub fn slots(&self) -> &[Value] {
        &self.slots
    }
    /// The per-function emitted bitmap (`emitted[i]` ⇒ `f{i}` runs on wasm; the rest are cross-tier).
    pub fn emitted(&self) -> &[bool] {
        &self.emitted
    }
    /// The signature of cross-tier `func` (the host marshals `env.call_interp`'s i64 slots per these).
    pub fn func_sig(&self, func: u32) -> (&[svm_ir::ValType], &[svm_ir::ValType]) {
        let f = &self.module.funcs[func as usize];
        (&f.params, &f.results)
    }

    /// **Cross-tier bounce.** Run non-emitted `func(args)` on the interpreter over the shared window with
    /// the powerbox (so `write`/`read`/`exit` resolve). A `Trap::Exit(code)` is stashed (the guest called
    /// `exit`) so `f0` unwinds and the run reports `STATUS_EXIT` with that code.
    pub fn run_cross_tier(&mut self, func: u32, args: &[Value]) -> Result<Vec<Value>, Trap> {
        let mut fuel = u64::MAX;
        let r = self.program.run_over(
            func,
            args,
            &mut fuel,
            self.back.clone(),
            &mut self.host,
            false,
        );
        if let Err(Trap::Exit(code)) = &r {
            self.exit_code = *code;
            self.exited = true;
        }
        r
    }

    /// The captured streams / exit — read after the emitted `f0` returns or unwinds (same contract as
    /// [`onramp_exec`]). `value` is `f0`'s return (meaningful when it returned rather than `exit`ed).
    pub fn stdout(&self) -> &[u8] {
        &self.host.stdout
    }
    pub fn stderr(&self) -> &[u8] {
        &self.host.stderr
    }
    pub fn exited(&self) -> bool {
        self.exited
    }
    pub fn exit_code(&self) -> i32 {
        self.exit_code
    }
    /// The value the emitted `f0` returned (guest top-level result); meaningful only for a returned run.
    pub fn returned_value(&self) -> i64 {
        self.returned_value
    }
    /// Whether the emitted `f0` unwound on a trap rather than returning or `exit`ing.
    pub fn trapped(&self) -> bool {
        self.trapped
    }
    /// Record the JS driver's observation of how the emitted `f0` finished: `value` is its return
    /// (used only when it returned), and `threw` is whether it unwound. A throw that did not set
    /// `exited` (a cross-tier `Exit`) is a trap. A cross-tier `Exit` already set `exited`, so `threw`
    /// there is subsumed by the exit.
    pub fn record_outcome(&mut self, threw: bool, value: i64) {
        self.returned_value = value;
        if threw && !self.exited {
            self.trapped = true;
        }
    }
    /// Take the frame the run presented through `display`, if any.
    pub fn take_frame(&self) -> Option<Frame> {
        self.frame.lock().unwrap().take()
    }
    pub fn last_trap(&self) -> &str {
        self.last_trap.as_deref().unwrap_or("")
    }
    pub fn set_last_trap(&mut self, s: String) {
        self.last_trap = Some(s);
    }
}

/// Outcome of a [`capture_exec`] run: the status, the `i64`-widened return value (when `STATUS_OK`),
/// and the **final window image** — the first `init.len()` bytes of the guest's memory after the run.
pub struct CapOutcome {
    pub status: i32,
    pub value: i64,
    pub snapshot: Vec<u8>,
}

/// Run `m`'s function 0 over a window seeded with `init` (deny-all `Host`), and capture the final
/// window image. This is the "host hands in a buffer, the guest transforms it in place, the host
/// reads it back" shape: [`bytecode::compile_and_run_capture`] snapshots the first `init.len()`
/// bytes of memory after the run. Shared verbatim by the wasm [`svm_run_capture`] export and the
/// native `gencorpus` ground truth, so the differential compares identical logic.
pub fn capture_exec(m: &svm_ir::Module, init: &[u8], arg: i64) -> CapOutcome {
    let mut fuel = u64::MAX;
    match bytecode::compile_and_run_capture(m, 0, &[Value::I64(arg)], &mut fuel, init) {
        None => CapOutcome {
            status: STATUS_UNSUPPORTED,
            value: 0,
            snapshot: Vec::new(),
        },
        Some((r, snapshot)) => {
            let (status, value) = match r {
                Err(_) => (STATUS_TRAP, 0),
                Ok(vals) => match vals.first() {
                    Some(Value::I64(x)) => (STATUS_OK, *x),
                    Some(Value::I32(x)) => (STATUS_OK, *x as i64),
                    _ => (STATUS_BAD_RESULT, 0),
                },
            };
            CapOutcome {
                status,
                value,
                snapshot,
            }
        }
    }
}

/// Run `m`'s function 0 with an `Instantiator` (iface 6) granted over `[0, 128 KiB)` — the §14
/// **nested-child** seam: function 0 may `instantiate`/`join` confined child domains over power-of-two
/// sub-windows of that range (a child runs on the cooperative executor, confined by masking to its
/// slice, joinable through the shared thread machinery). Returns `(status, i64-widened value)`.
/// Shared by the wasm [`svm_run_nested`] export and the native `gencorpus` ground truth.
pub fn instantiate_exec(m: &svm_ir::Module) -> (i32, i64) {
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, 128 << 10);
    let mut fuel = 5_000_000u64;
    match bytecode::compile_and_run_with_host(m, 0, &[Value::I32(inst)], &mut fuel, &mut host) {
        None => (STATUS_UNSUPPORTED, 0),
        Some(Err(_)) => (STATUS_TRAP, 0),
        Some(Ok(vals)) => match vals.first() {
            Some(Value::I64(x)) => (STATUS_OK, *x),
            Some(Value::I32(x)) => (STATUS_OK, *x as i64),
            _ => (STATUS_BAD_RESULT, 0),
        },
    }
}

/// Captured stdout / stderr of the most recent [`svm_run_pb`], as cdylib-managed allocations
/// `(ptr, len)`. Each is a leaked boxed slice (exact length, alignment 1) freed when the next
/// [`svm_run_pb`] replaces it — so the host reads it via the `*_ptr`/`*_len` exports *before* the
/// next call and never frees it itself.
static mut OUT: (*mut u8, usize) = (core::ptr::null_mut(), 0);
static mut ERR: (*mut u8, usize) = (core::ptr::null_mut(), 0);
static mut EXIT_CODE: i32 = 0;
/// The value the guest's top-level function returned on the most recent run (the `value` in
/// [`svm_run_onramp`]'s outcome, and the single-shot JIT run's captured `f0` return). Read via
/// [`svm_run_value`] so both tiers surface the same result for a *returned* run — the parity the
/// interpreter oracle defines (INVARIANT 9).
static mut RUN_VALUE: i64 = 0;
/// Captured data image of the most recent [`svm_pg_snapshot`] (same cdylib-managed lifetime as `OUT`:
/// a leaked boxed slice, valid until the next snapshot; read via `svm_pg_snapshot_ptr`/`_len`).
static mut PG_SNAP: (*mut u8, usize) = (core::ptr::null_mut(), 0);
/// Captured final window image of the most recent [`svm_run_capture`] (same cdylib-managed lifetime
/// as `OUT`/`ERR`: valid until the next `svm_run_capture`).
static mut SNAP: (*mut u8, usize) = (core::ptr::null_mut(), 0);
/// Captured framebuffer (RGBA) the most recent [`svm_run_onramp`] guest presented via the `display`
/// capability, plus its dimensions. `(null, 0)` / `0`×`0` when the guest presented no frame. Same
/// cdylib-managed lifetime as `OUT` (valid until the next `svm_run_onramp`; the host reads it via the
/// `svm_framebuffer_*` exports and never frees it).
static mut FB: (*mut u8, usize) = (core::ptr::null_mut(), 0);
static mut FB_W: u32 = 0;
static mut FB_H: u32 = 0;

/// Replace the capture in `slot` with `data`, freeing the previous allocation. Empty `data` stores
/// `(null, 0)`. The stored allocation is a boxed slice — exactly `len` bytes, alignment 1 — so it is
/// freed with the matching `Layout`.
fn stash(slot: &mut (*mut u8, usize), data: Vec<u8>) {
    let (old_ptr, old_len) = *slot;
    if !old_ptr.is_null() && old_len != 0 {
        if let Ok(layout) = Layout::from_size_align(old_len, 1) {
            unsafe { std::alloc::dealloc(old_ptr, layout) };
        }
    }
    *slot = if data.is_empty() {
        (core::ptr::null_mut(), 0)
    } else {
        let boxed = data.into_boxed_slice(); // shrink-to-fit: capacity == len, alignment 1
        let len = boxed.len();
        (Box::into_raw(boxed) as *mut u8, len)
    };
}

// ---- Per-function call profiler FFI (opt-in `callprof` feature; tier-up break-even measurement) ---
// Not present in a shipped build. Arm with `svm_callprof_reset(n_funcs)`, run a guest via
// `svm_run_onramp`, then `svm_callprof_dump()` and read the buffer (LE `u64` per function).
#[cfg(feature = "callprof")]
static mut CALLPROF: (*mut u8, usize) = (core::ptr::null_mut(), 0);
#[cfg(feature = "callprof")]
#[no_mangle]
pub extern "C" fn svm_callprof_reset(n: usize) {
    bytecode::callprof_reset(n);
}
#[cfg(feature = "callprof")]
#[no_mangle]
pub extern "C" fn svm_callprof_dump() {
    let counts = bytecode::callprof_snapshot();
    let mut bytes = Vec::with_capacity(counts.len() * 8);
    for c in counts {
        bytes.extend_from_slice(&c.to_le_bytes());
    }
    unsafe { stash(&mut *core::ptr::addr_of_mut!(CALLPROF), bytes) };
}
#[cfg(feature = "callprof")]
#[no_mangle]
pub extern "C" fn svm_callprof_ptr() -> *const u8 {
    unsafe { (*core::ptr::addr_of!(CALLPROF)).0 }
}
#[cfg(feature = "callprof")]
#[no_mangle]
pub extern "C" fn svm_callprof_len() -> usize {
    unsafe { (*core::ptr::addr_of!(CALLPROF)).1 }
}

/// Decode the module at `[mod_ptr, mod_len)` and run function 0 under the **powerbox** (see
/// [`powerbox_exec`]): grant streams/clock/exit, seed stdin from `[stdin_ptr, stdin_len)` (a null /
/// zero-length range ⇒ empty stdin), capture the streams + exit code, and return the guest's `i64`
/// result (`0` on any non-`OK`/`EXIT` status). Read [`svm_status`] / [`svm_exit_code`] /
/// `svm_stdout_ptr`+`svm_stdout_len` / `svm_stderr_ptr`+`svm_stderr_len` afterward. Sets
/// [`LAST_STATUS`].
#[no_mangle]
pub extern "C" fn svm_run_pb(
    mod_ptr: *const u8,
    mod_len: usize,
    stdin_ptr: *const u8,
    stdin_len: usize,
) -> i64 {
    let set = |s: i32| unsafe { LAST_STATUS = s };
    // SAFETY: the host guarantees both ranges are live `svm_alloc`ations it just filled.
    let bytes = unsafe { core::slice::from_raw_parts(mod_ptr, mod_len) };
    let stdin: &[u8] = if stdin_ptr.is_null() || stdin_len == 0 {
        &[]
    } else {
        unsafe { core::slice::from_raw_parts(stdin_ptr, stdin_len) }
    };
    let m = match svm_encode::decode_module(bytes) {
        Ok(m) => m,
        Err(_) => {
            set(STATUS_DECODE_ERR);
            return 0;
        }
    };
    let out = powerbox_exec(&m, stdin);
    set(out.status);
    // SAFETY: single-threaded wasm; the capture slots are read back only via the export accessors.
    unsafe {
        stash(&mut *core::ptr::addr_of_mut!(OUT), out.stdout);
        stash(&mut *core::ptr::addr_of_mut!(ERR), out.stderr);
        EXIT_CODE = out.exit_code;
    }
    out.value
}

/// Decode the module at `[mod_ptr, mod_len)` and run function 0 under the **on-ramp powerbox** (see
/// [`onramp_exec`]) — the ABI a `.svmb` off `svm-llvm-translate` expects, so real C/C++ guests (Lua,
/// SQLite) run unchanged. Same capture/accessor contract as [`svm_run_pb`]: seed stdin from
/// `[stdin_ptr, stdin_len)` (null / zero-length ⇒ empty), read the streams via
/// `svm_stdout_ptr`+`svm_stdout_len` / `svm_stderr_ptr`+`svm_stderr_len`, the exit code via
/// [`svm_exit_code`], and the status via [`svm_status`]. Returns the guest's `i64` result. The
/// captures share `OUT`/`ERR`/`EXIT_CODE` with `svm_run_pb` — read them before the next call either
/// export makes.
#[no_mangle]
pub extern "C" fn svm_run_onramp(
    mod_ptr: *const u8,
    mod_len: usize,
    stdin_ptr: *const u8,
    stdin_len: usize,
) -> i64 {
    let set = |s: i32| unsafe { LAST_STATUS = s };
    // SAFETY: the host guarantees both ranges are live `svm_alloc`ations it just filled.
    let bytes = unsafe { core::slice::from_raw_parts(mod_ptr, mod_len) };
    let stdin: &[u8] = if stdin_ptr.is_null() || stdin_len == 0 {
        &[]
    } else {
        unsafe { core::slice::from_raw_parts(stdin_ptr, stdin_len) }
    };
    let m = match svm_encode::decode_module(bytes) {
        Ok(m) => m,
        Err(_) => {
            set(STATUS_DECODE_ERR);
            return 0;
        }
    };
    let out = onramp_exec(&m, stdin);
    set(out.status);
    let (fb_rgba, fb_w, fb_h) = match out.framebuffer {
        Some(f) => (f.rgba, f.width, f.height),
        None => (Vec::new(), 0, 0),
    };
    // SAFETY: single-threaded wasm; the capture slots are read back only via the export accessors.
    unsafe {
        stash(&mut *core::ptr::addr_of_mut!(OUT), out.stdout);
        stash(&mut *core::ptr::addr_of_mut!(ERR), out.stderr);
        stash(&mut *core::ptr::addr_of_mut!(FB), fb_rgba);
        FB_W = fb_w;
        FB_H = fb_h;
        EXIT_CODE = out.exit_code;
    }
    out.value
}

// ===== warm-runtime snapshot: init once, restore-per-Run for a two-phase on-ramp guest ============
//
// WASM_AOT.md § "warm-runtime snapshot". A language on-ramp (QuickJS…) rebuilds its whole runtime on
// every Run; for a trivial program that fixed init dominates the wall clock. This session runs the
// guest's `warmup` export **once** (init runtime+context into statics), snapshots the post-init guest
// image, then restores that image before each `eval_run` — so every Run evaluates the user's code over
// a warm runtime it did not have to rebuild. Fresh-per-Run isolation holds: each Run restores the SAME
// program-independent image into the window (no guest state crosses Runs). The native prototype is
// `crates/svm-llvm/examples/qjs_snapshot.rs`; this is its browser twin, a stateful session like
// `PgSession`. Requires a two-phase driver module exporting `warmup` + `eval_run` (both `(i64 sp)`).
//
// Memory model (the wrinkle the native prototype de-risked): the on-ramp heap grows **above** the
// declared window (`heap_base = 1 << declared_log2`), and that `vm_map` growth's mapped-width state
// lives in the `Mem`, not the window bytes — so we run under a **larger mapped window**
// (`WARM_MAPPED_LOG2`) that keeps the whole heap inside the mapped region, making the guest image a
// contiguous byte range we snapshot/restore by plain copy.

/// The mapped-window log2 the warm session runs the guest under, overriding the module's declared size
/// so the on-ramp heap stays inside the mapped region (see the memory-model note above). 2^26 = 64 MiB:
/// a few MiB of globals/stack + generous heap headroom for QuickJS. Matches the native prototype.
const WARM_MAPPED_LOG2: u8 = 26;

/// A live warm-runtime snapshot session over an owned window: the compiled program, the warm image, and
/// the window it restores into. Single-threaded wasm ⇒ held in a plain static ([`WARM_SESSION`]).
struct WarmSession {
    prog: bytecode::SharedProgram,
    /// The module (memory patched to the mapped window) — re-granted onto a fresh host per eval so the
    /// deterministic powerbox handles match the snapshot's window-relative state.
    module: svm_ir::Module,
    /// The warmup image's explicit page-state entries (#816, the `Mem::map_info` encoding): the
    /// on-ramp's `protect`ed rodata inside the prefix and the `vm_map`-grown heap tail alike.
    /// Re-established (without zeroing) before every eval, so the guest restores to the same
    /// mapped geometry — and the same write protections — instead of faulting.
    prots: Vec<(u64, u8)>,
    /// Mapped window size, `1 << WARM_MAPPED_LOG2` (also the owned backing size).
    win: u64,
    /// The powerbox data-stack base (`powerbox_entry_sp`), passed as each entry's `sp` arg.
    entry_sp: u64,
    eval_fn: svm_ir::FuncIdx,
    /// The owned window backing (kept alive for the session; `back` aliases it).
    win_ptr: *mut u8,
    win_layout: Layout,
    back: std::sync::Arc<svm_interp::Region>,
    /// The program-independent warm image — the live prefix `[0, brk)` captured after `warmup`.
    image: Vec<u8>,
    /// High-water of bytes any prior eval may have dirtied (≥ `image.len()`): the restore zeroes
    /// `[image.len(), dirty_end)` so a re-Run sees the same zero tail `warmup` left above the heap.
    dirty_end: usize,
    /// #964: the module's NULL guard (`0` = legacy layout) — a marked module's powerbox low scratch
    /// (heap bump words included) sits one guard up, so every brk read/seed offsets by this.
    scratch: u64,
    /// The cached warm+JIT run (WASM_AOT.md warm+JIT): `eval_run` emitted to wasm **once**, then driven
    /// per Run over the restored warm image. `None` until [`svm_warm_jit_open`]; reusing the emit across
    /// Runs is what keeps a warm+JIT Run off the ~one-time cdylib emit. Held here (not in a global) so
    /// [`svm_warm_close`] tears it down while its window alias is still valid, before the window is freed.
    jit: Option<Box<JitOnrampRun>>,
}

/// The one live warm session (single-threaded wasm ⇒ a plain static). `None` until [`svm_warm_open`].
static mut WARM_SESSION: Option<WarmSession> = None;

/// Read the on-ramp guest heap bump pointer (`POWERBOX_HEAP_BRK`, shifted one guard up on the #964
/// marked layout — pass the module's `scratch` base) from a window image.
fn warm_read_brk(win: &[u8], scratch: u64) -> usize {
    let o = (scratch + svm_ir::POWERBOX_HEAP_BRK) as usize;
    i64::from_le_bytes(win[o..o + 8].try_into().unwrap()) as usize
}

/// Open a warm session over the two-phase driver module at `[mod_ptr, mod_len)`: run `warmup` once and
/// keep its post-init guest image for [`svm_warm_eval`]. Returns the live-image byte length on success
/// (≥ 0), or `-1` with [`svm_status`] set (`UNSUPPORTED` if the module isn't a warm-snapshot driver —
/// no `warmup`/`eval_run` exports, or its declared window ≥ the mapped window; `TRAP` if `warmup`
/// traps). Closes any prior session first. Drive with [`svm_warm_eval`], end with [`svm_warm_close`].
#[no_mangle]
pub extern "C" fn svm_warm_open(mod_ptr: *const u8, mod_len: usize) -> i64 {
    let set = |s: i32| unsafe { LAST_STATUS = s };
    svm_warm_close();
    // SAFETY: the host guarantees `[mod_ptr, mod_len)` is a live `svm_alloc`ation it just filled.
    let bytes = unsafe { core::slice::from_raw_parts(mod_ptr, mod_len) };
    let m = match svm_encode::decode_module(bytes) {
        Ok(m) => m,
        Err(_) => {
            set(STATUS_DECODE_ERR);
            return -1;
        }
    };
    let Some(mc) = m.memory else {
        set(STATUS_UNSUPPORTED);
        return -1;
    };
    let declared_log2 = mc.size_log2;
    // The backing (and the run's reservation) is `1 << WARM_MAPPED_LOG2`; the declared window must
    // fit inside it. The heap may now `vm_map`-grow into `[declared, backing)` (#816) — the module
    // is no longer over-sized to swallow the heap.
    if declared_log2 > WARM_MAPPED_LOG2 {
        set(STATUS_UNSUPPORTED);
        return -1;
    }
    let heap_base = 1u64 << declared_log2;
    let (Some(warmup_fn), Some(eval_fn)) =
        (m.resolve_export("warmup"), m.resolve_export("eval_run"))
    else {
        set(STATUS_UNSUPPORTED);
        return -1;
    };
    let entry_sp = svm_ir::powerbox_entry_sp(&m);
    let win = 1u64 << WARM_MAPPED_LOG2;
    let Some(prog) = bytecode::SharedProgram::compile(&m) else {
        set(STATUS_UNSUPPORTED);
        return -1;
    };
    let Ok(layout) = Layout::from_size_align(win as usize, 8) else {
        set(STATUS_UNSUPPORTED);
        return -1;
    };
    // SAFETY: non-zero 8-aligned size; the buffer is this session's window, freed in `svm_warm_close`.
    let win_ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    if win_ptr.is_null() {
        set(STATUS_TRAP);
        return -1;
    }
    // Seed the on-ramp heap bump words (`_start` normally does this): brk = top = heap_base.
    // #964: a marked module's heap words sit one guard up.
    let scratch = svm_ir::module_null_guard(&m).unwrap_or(0);
    // SAFETY: `win_ptr` owns `win` zeroed bytes; no engine run is in flight (sole access here).
    unsafe {
        let w = core::slice::from_raw_parts_mut(win_ptr, win as usize);
        let hb = (heap_base as i64).to_le_bytes();
        let (b, t) = (
            (scratch + svm_ir::POWERBOX_HEAP_BRK) as usize,
            (scratch + svm_ir::POWERBOX_HEAP_TOP) as usize,
        );
        w[b..b + 8].copy_from_slice(&hb);
        w[t..t + 8].copy_from_slice(&hb);
    }
    // SAFETY: `[win_ptr, win)` is this session's exclusive window; the engine takes the `Arc<Region>`.
    let back = std::sync::Arc::new(unsafe { svm_interp::Region::shared(win_ptr, win) });
    let mut host = Host::new();
    let _ = grant_onramp_caps(&mut host, &m, None);
    let mut fuel = u64::MAX;
    // The reservation is clamped to the backing (#816): a `map` past `win` fails with `-EINVAL`
    // instead of minting pages whose writes the backing silently drops.
    let (ran, pages) = prog.run_over_grown(
        warmup_fn,
        &[Value::I64(entry_sp as i64)],
        &mut fuel,
        back.clone(),
        &mut host,
        true, // seed the module's data segments once, here
        WARM_MAPPED_LOG2,
        None,
    );
    // Fail closed on a warmup that trapped OR aliased a §13 region page (a byte restore cannot
    // reproduce an alias; the page falls back to cold runs).
    let prots = match (&ran, pages) {
        (Ok(_) | Err(Trap::Exit(_)), Some(entries)) => entries,
        _ => {
            drop(back);
            // SAFETY: no alias remains (back dropped, run returned); free the window.
            unsafe { std::alloc::dealloc(win_ptr, layout) };
            set(STATUS_TRAP);
            return -1;
        }
    };
    // Capture the live prefix `[0, brk)` — everything above brk is still the zero `warmup` left.
    // SAFETY: `win_ptr` owns `win` bytes; read the post-warmup image, no run in flight.
    let (image, live) = unsafe {
        let w = core::slice::from_raw_parts(win_ptr, win as usize);
        let live = warm_read_brk(w, scratch).min(win as usize);
        (w[..live].to_vec(), live)
    };
    // SAFETY: single-threaded wasm; the session is read back only via the warm exports.
    unsafe {
        *core::ptr::addr_of_mut!(WARM_SESSION) = Some(WarmSession {
            prog,
            module: m,
            prots,
            win,
            entry_sp,
            eval_fn,
            win_ptr,
            win_layout: layout,
            back,
            image,
            dirty_end: live,
            scratch,
            jit: None,
        });
    }
    set(STATUS_OK);
    live as i64
}

/// Evaluate `[stdin_ptr, stdin_len)` (the user's source) over the warm session: restore the snapshot
/// into the window, run `eval_run`, and stage its stdout/stderr/exit into the shared capture slots
/// (read via `svm_stdout_ptr`/`_len`, `svm_stderr_ptr`/`_len`, `svm_exit_code`; status via
/// [`svm_status`]). Returns the guest's `i64` result, or `-1` with `UNSUPPORTED` if no session is open.
#[no_mangle]
pub extern "C" fn svm_warm_eval(stdin_ptr: *const u8, stdin_len: usize) -> i64 {
    let set = |s: i32| unsafe { LAST_STATUS = s };
    // SAFETY: single-threaded wasm; exclusive access to the session for this call.
    let Some(s) = (unsafe { (*core::ptr::addr_of_mut!(WARM_SESSION)).as_mut() }) else {
        set(STATUS_UNSUPPORTED);
        return -1;
    };
    let stdin: &[u8] = if stdin_ptr.is_null() || stdin_len == 0 {
        &[]
    } else {
        // SAFETY: the host guarantees `[stdin_ptr, stdin_len)` is a live `svm_alloc`ation it filled.
        unsafe { core::slice::from_raw_parts(stdin_ptr, stdin_len) }
    };
    // Restore the warm image, and zero the tail any prior eval grew the heap into — so this Run sees
    // byte-identical warm state (fresh-per-Run isolation).
    // SAFETY: `win_ptr` owns `win ≥ dirty_end` bytes; no engine run is in flight (sole access here).
    unsafe {
        let w = core::slice::from_raw_parts_mut(s.win_ptr, s.win as usize);
        w[..s.image.len()].copy_from_slice(&s.image);
        w[s.image.len()..s.dirty_end].fill(0);
    }
    let mut host = Host::new();
    host.stdin = stdin.to_vec();
    let _ = grant_onramp_caps(&mut host, &s.module, None);
    let mut fuel = u64::MAX;
    // Re-establish the warmup image's page-state entries (no zeroing — the memcpy above restored
    // the bytes), so a `vm_map`-grown warm heap is addressable again — and its `protect`ed rodata
    // write-protected again (#816). Every eval starts from the SAME captured map — an eval's own
    // remaps never accumulate (fresh-per-Run isolation).
    let (ran, eval_pages) = s.prog.run_over_grown(
        s.eval_fn,
        &[Value::I64(s.entry_sp as i64)],
        &mut fuel,
        s.back.clone(),
        &mut host,
        false, // window already carries the warm image — do not re-seed
        WARM_MAPPED_LOG2,
        Some(&s.prots),
    );
    let (status, value, exit_code) = match ran {
        Err(Trap::Exit(code)) => (STATUS_EXIT, 0, code),
        Err(_) => (STATUS_TRAP, 0, 0),
        Ok(vals) => match vals.first() {
            Some(Value::I64(x)) => (STATUS_OK, *x, 0),
            Some(Value::I32(x)) => (STATUS_OK, *x as i64, 0),
            _ => (STATUS_OK, 0, 0),
        },
    };
    // Track the eval's heap high-water so the next restore zeroes exactly what it dirtied — both
    // the brk it advanced and any page it `vm_map`-grew past the warm extent (freshly-mapped pages
    // are zeroed at map time, but the guest may have written them).
    // SAFETY: `win_ptr` owns `win` bytes; read the post-eval brk, no run in flight.
    unsafe {
        let w = core::slice::from_raw_parts(s.win_ptr, s.win as usize);
        // Any page the eval left committed (`Rw`, kind 1) may carry its writes — zero to the top
        // of the highest one on the next restore (page size from the engine's map_info encoding).
        let grown = eval_pages
            .as_deref()
            .unwrap_or(&[])
            .iter()
            .filter(|&&(_, kind)| kind == 1)
            .map(|&(off, _)| off.saturating_add(svm_interp::host_page_size()))
            .max()
            .unwrap_or(0)
            .min(s.win) as usize;
        s.dirty_end = s
            .dirty_end
            .max(warm_read_brk(w, s.scratch).min(s.win as usize))
            .max(grown);
    }
    set(status);
    // SAFETY: single-threaded wasm; the capture slots are read back only via the export accessors.
    unsafe {
        stash(&mut *core::ptr::addr_of_mut!(OUT), host.stdout);
        stash(&mut *core::ptr::addr_of_mut!(ERR), host.stderr);
        EXIT_CODE = exit_code;
    }
    value
}

/// Tear down the warm session (free its window), if any. Idempotent; the next [`svm_warm_open`] starts
/// a fresh one.
#[no_mangle]
pub extern "C" fn svm_warm_close() {
    // SAFETY: single-threaded wasm; take the session and free its owned window.
    unsafe {
        if let Some(s) = (*core::ptr::addr_of_mut!(WARM_SESSION)).take() {
            let (win_ptr, layout) = (s.win_ptr, s.win_layout);
            drop(s); // drops `back` + the cached warm+JIT run (both alias `win_ptr`) before the free
            std::alloc::dealloc(win_ptr, layout);
        }
    }
}

// ===== warm+JIT: run the warm session's `eval_run` on the emitted-wasm tier ========================
//
// WASM_AOT.md warm+JIT. The interpreter warm path ([`svm_warm_eval`]) already skips the QuickJS runtime
// rebuild, but evaluates on the bytecode interpreter — so a compute-heavy program still pays interpreter
// speed for the eval itself. This tier emits the module's `eval_run` to wasm **once** and drives it over
// the restored warm image each Run, so the eval runs near-native while init stays paid-once. The emit is
// cached in the [`WarmSession`] (a warm+JIT Run never re-pays the cdylib emit); only the powerbox host +
// the window image are reset per Run, preserving the card's fresh-per-Run isolation. Cross-tier bounces
// (`write`/`read`/`exit`) route to the run's interpreter over the same window, exactly as the single-shot
// JIT run does. The JS driver mirrors `wasmjit-module.js`'s `driveJitRun`, but passes the entry `sp` as
// the emitted `f0`'s third argument (an `i64` slot) and reuses the cached compiled Module across Runs.

/// Borrow the open warm session's cached warm+JIT run, if built.
fn warm_jit_ref() -> Option<&'static JitOnrampRun> {
    // SAFETY: single-threaded wasm; the run is touched only by these export accessors, no run in flight.
    unsafe {
        (*core::ptr::addr_of!(WARM_SESSION))
            .as_ref()
            .and_then(|s| s.jit.as_deref())
    }
}

/// Emit the open warm session's `eval_run` to wasm and cache it for the warm+JIT drive. Idempotent — a
/// second call reuses the cached emit (returns `0`). `shared != 0` ⇒ the emitted module imports a shared
/// memory (the cross-origin-isolated threads build), matching the memory the host instantiates it against.
/// Returns `0`, else a negative `STATUS_*` (also in [`LAST_STATUS`]): [`STATUS_UNSUPPORTED`] if no warm
/// session is open or `eval_run` isn't wasm-drivable (the page then evaluates via [`svm_warm_eval`]).
#[no_mangle]
pub extern "C" fn svm_warm_jit_open(shared: i32) -> i32 {
    let set = |s: i32| unsafe { LAST_STATUS = s };
    // SAFETY: single-threaded wasm; exclusive access to the session for this call.
    let Some(s) = (unsafe { (*core::ptr::addr_of_mut!(WARM_SESSION)).as_mut() }) else {
        set(STATUS_UNSUPPORTED);
        return -STATUS_UNSUPPORTED;
    };
    if s.jit.is_some() {
        set(STATUS_OK);
        return 0;
    }
    // SAFETY: `s.back` aliases the live session window `[s.win_ptr, s.win)`, valid for the session's life.
    let built = unsafe {
        JitOnrampRun::open_warm_eval(
            &s.module,
            s.back.clone(),
            s.win_ptr,
            WARM_MAPPED_LOG2,
            shared != 0,
            s.eval_fn,
            s.entry_sp,
        )
    };
    match built {
        Ok(run) => {
            s.jit = Some(Box::new(run));
            set(STATUS_OK);
            0
        }
        Err(status) => {
            set(status);
            -status
        }
    }
}

/// Restore the warm image into the window and reset the cached warm+JIT run's powerbox for a fresh Run
/// (seeding stdin from `[stdin_ptr, stdin_len)`). Call before each drive. Returns `0`, else
/// `-STATUS_UNSUPPORTED` if no warm+JIT run is open.
#[no_mangle]
pub extern "C" fn svm_warm_jit_prepare(stdin_ptr: *const u8, stdin_len: usize) -> i32 {
    let set = |s: i32| unsafe { LAST_STATUS = s };
    // SAFETY: single-threaded wasm; exclusive access to the session for this call.
    let Some(s) = (unsafe { (*core::ptr::addr_of_mut!(WARM_SESSION)).as_mut() }) else {
        set(STATUS_UNSUPPORTED);
        return -STATUS_UNSUPPORTED;
    };
    if s.jit.is_none() {
        set(STATUS_UNSUPPORTED);
        return -STATUS_UNSUPPORTED;
    }
    // Restore the program-independent warm image, zeroing any tail a prior eval grew into — byte-identical
    // warm state each Run (identical to [`svm_warm_eval`]'s restore).
    // SAFETY: `win_ptr` owns `win ≥ dirty_end` bytes; no engine run is in flight (sole access here).
    unsafe {
        let w = core::slice::from_raw_parts_mut(s.win_ptr, s.win as usize);
        w[..s.image.len()].copy_from_slice(&s.image);
        w[s.image.len()..s.dirty_end].fill(0);
    }
    let stdin: Vec<u8> = if stdin_ptr.is_null() || stdin_len == 0 {
        Vec::new()
    } else {
        // SAFETY: the host guarantees `[stdin_ptr, stdin_len)` is a live `svm_alloc`ation it filled.
        unsafe { core::slice::from_raw_parts(stdin_ptr, stdin_len) }.to_vec()
    };
    s.jit.as_mut().unwrap().reset_warm(stdin);
    set(STATUS_OK);
    0
}

/// Pointer / length of the emitted `eval_run` wasm bytes (valid until the warm session is closed).
#[no_mangle]
pub extern "C" fn svm_warm_jit_wasm_ptr() -> *const u8 {
    warm_jit_ref().map_or(core::ptr::null(), |r| r.emitted_wasm().as_ptr())
}
#[no_mangle]
pub extern "C" fn svm_warm_jit_wasm_len() -> usize {
    warm_jit_ref().map_or(0, |r| r.emitted_wasm().len())
}
/// The window base as a byte offset in this module's linear memory — the emitted `f0`'s `win` arg.
#[no_mangle]
pub extern "C" fn svm_warm_jit_win_ptr() -> usize {
    warm_jit_ref().map_or(0, |r| r.win_base())
}
/// The entry `sp` the emitted entry takes as its trailing `i64` slot (the powerbox data-stack base).
#[no_mangle]
pub extern "C" fn svm_warm_jit_entry_sp() -> i64 {
    warm_jit_ref().map_or(0, |r| match r.slots().first() {
        Some(Value::I64(x)) => *x,
        Some(Value::I32(x)) => *x as i64,
        _ => 0,
    })
}

/// The emitted **export index of the warm+JIT entry** — the driver must call `f{this}`, NOT `f0`. The
/// emit exports one `f{svm_idx}` per SVM function, and the warm+JIT emit is rooted at the module's
/// `eval_run` (not the cold `_start`, which is func 0). So the entry export is `f{eval_fn}` where
/// `eval_fn` is `eval_run`'s SVM function index. Driving `f0` instead runs the cold `_start`
/// (init + eval), which for a driver whose init re-runs on the restored image traps or diverges (#865).
/// Valid after [`svm_warm_jit_open`]; 0 (a harmless `f0`) if no warm session is open.
#[no_mangle]
pub extern "C" fn svm_warm_jit_entry_func() -> u32 {
    // SAFETY: single-threaded wasm; shared read of the session static.
    unsafe {
        (*core::ptr::addr_of!(WARM_SESSION))
            .as_ref()
            .map_or(0, |s| s.eval_fn)
    }
}

/// **Cross-tier bounce** for the warm+JIT run — the emitted `f0`'s `env.call_interp(func, args_ptr)`
/// relays here (identical contract to [`svm_onramp_jit_run_call_interp`], over the warm run's
/// window/powerbox).
#[no_mangle]
pub extern "C" fn svm_warm_jit_call_interp(func: u32, args_ptr: *mut u8) -> i32 {
    // SAFETY: single-threaded wasm; exclusive access to the run for this call.
    let Some(run) = (unsafe {
        (*core::ptr::addr_of_mut!(WARM_SESSION))
            .as_mut()
            .and_then(|s| s.jit.as_deref_mut())
    }) else {
        return STATUS_UNSUPPORTED;
    };
    let (params, results) = {
        let (p, r) = run.func_sig(func);
        (p.to_vec(), r.to_vec())
    };
    // SAFETY: the host guarantees `args_ptr` addresses the signature's full slot span (the env scratch).
    let args: Vec<Value> = params
        .iter()
        .zip(slot_offs(&params))
        .map(|(t, o)| unsafe { read_slot_value(*t, args_ptr, o) })
        .collect();
    match run.run_cross_tier(func, &args) {
        Ok(vals) => {
            let offs = slot_offs(&results);
            for (i, v) in vals.iter().enumerate() {
                if i >= results.len() {
                    break;
                }
                // SAFETY: `args_ptr + offs[i]` is within the env scratch (result slots overlay arg slots).
                if !unsafe { write_slot_value(v, args_ptr, offs[i]) } {
                    return STATUS_TRAP;
                }
            }
            0
        }
        Err(Trap::Exit(_)) => STATUS_EXIT,
        Err(t) => {
            run.set_last_trap(format!("{t:?}"));
            STATUS_TRAP
        }
    }
}

/// Record how the emitted warm `f0` finished (see [`svm_onramp_jit_run_report`]). Call before
/// [`svm_warm_jit_finish`].
#[no_mangle]
pub extern "C" fn svm_warm_jit_report(threw: i32, value: i64) {
    // SAFETY: single-threaded wasm; exclusive access to the run.
    if let Some(run) = unsafe {
        (*core::ptr::addr_of_mut!(WARM_SESSION))
            .as_mut()
            .and_then(|s| s.jit.as_deref_mut())
    } {
        run.record_outcome(threw != 0, value);
    }
}

/// Capture the finished warm+JIT run's streams / exit / value into the shared `OUT`/`ERR`/`EXIT_CODE`/
/// `RUN_VALUE` slots (read via the usual `svm_stdout_*` / `svm_exit_code` / `svm_run_value` accessors),
/// and advance the session's heap high-water so the next [`svm_warm_jit_prepare`] zeroes the right tail.
/// Same status contract as [`svm_onramp_jit_run_finish`] — so warm+JIT and the interpreter warm path
/// agree on result + exit + trap (INVARIANT 9). Call once after `f0` returns/unwinds (and after
/// [`svm_warm_jit_report`]). Returns the `STATUS_*`.
#[no_mangle]
pub extern "C" fn svm_warm_jit_finish() -> i32 {
    // SAFETY: single-threaded wasm; exclusive access to the session for this call.
    let Some(s) = (unsafe { (*core::ptr::addr_of_mut!(WARM_SESSION)).as_mut() }) else {
        return STATUS_UNSUPPORTED;
    };
    let Some(run) = s.jit.as_deref() else {
        return STATUS_UNSUPPORTED;
    };
    let stdout = run.stdout().to_vec();
    let stderr = run.stderr().to_vec();
    let (status, code, value) = if run.exited() {
        (STATUS_EXIT, run.exit_code(), 0)
    } else if run.trapped() {
        (STATUS_TRAP, 0, 0)
    } else {
        (STATUS_OK, 0, run.returned_value())
    };
    // Track the eval's heap high-water so the next restore zeroes exactly what it dirtied (mirrors
    // [`svm_warm_eval`]).
    // SAFETY: `win_ptr` owns `win` bytes; read the post-eval brk, no run in flight.
    unsafe {
        let w = core::slice::from_raw_parts(s.win_ptr, s.win as usize);
        s.dirty_end = s
            .dirty_end
            .max(warm_read_brk(w, s.scratch).min(s.win as usize));
    }
    // SAFETY: single-threaded wasm; the capture slots are read back only via the export accessors.
    unsafe {
        stash(&mut *core::ptr::addr_of_mut!(OUT), stdout);
        stash(&mut *core::ptr::addr_of_mut!(ERR), stderr);
        EXIT_CODE = code;
        RUN_VALUE = value;
        LAST_STATUS = status;
    }
    status
}

/// Decode the module at `[mod_ptr, mod_len)` and run function 0 under the **POSIX personality** (see
/// [`onramp_posix_exec`]) — the entry the real `svm-posix` shell runs through in the playground. Same
/// capture/accessor contract as [`svm_run_onramp`]: seed stdin from `[stdin_ptr, stdin_len)`, read the
/// captured streams via `svm_stdout_ptr`+`svm_stdout_len` / `svm_stderr_ptr`+`svm_stderr_len`, the
/// exit code via [`svm_exit_code`], and the status via [`svm_status`]. Returns the guest's `i64`
/// result. Shares the `OUT`/`ERR`/`EXIT_CODE` capture slots with `svm_run_onramp` — read them before
/// the next call either export makes.
#[no_mangle]
pub extern "C" fn svm_run_onramp_posix(
    mod_ptr: *const u8,
    mod_len: usize,
    stdin_ptr: *const u8,
    stdin_len: usize,
) -> i64 {
    let set = |s: i32| unsafe { LAST_STATUS = s };
    // SAFETY: the host guarantees both ranges are live `svm_alloc`ations it just filled.
    let bytes = unsafe { core::slice::from_raw_parts(mod_ptr, mod_len) };
    let stdin: &[u8] = if stdin_ptr.is_null() || stdin_len == 0 {
        &[]
    } else {
        unsafe { core::slice::from_raw_parts(stdin_ptr, stdin_len) }
    };
    let m = match svm_encode::decode_module(bytes) {
        Ok(m) => m,
        Err(_) => {
            set(STATUS_DECODE_ERR);
            return 0;
        }
    };
    let out = onramp_posix_exec(&m, stdin);
    set(out.status);
    // SAFETY: single-threaded wasm; the capture slots are read back only via the export accessors.
    unsafe {
        stash(&mut *core::ptr::addr_of_mut!(OUT), out.stdout);
        stash(&mut *core::ptr::addr_of_mut!(ERR), out.stderr);
        EXIT_CODE = out.exit_code;
    }
    out.value
}

/// Parse the shell's **PATH-registry blob** at `[ptr, len)` into `(name, module)` pairs. Layout, all
/// integers little-endian: a `u32` entry count, then per entry a `u32` name length + that many UTF-8
/// name bytes + a `u32` module length + that many encoded-module bytes. It bundles the `__stage`
/// ring-filter runner and every external command (`primes`, …) into one buffer so `svm_run_shell` takes
/// a single extra arg. Defensive: a truncated or malformed blob, or an entry whose module fails to
/// decode, drops that entry (and everything after a length that overruns) rather than trapping — the
/// shell still runs, just without the affected command. Returns owned `(String, Module)`s.
fn parse_shell_cmds(bytes: &[u8]) -> Vec<(String, svm_ir::Module)> {
    let mut out = Vec::new();
    let rd_u32 = |b: &[u8], at: usize| -> Option<usize> {
        b.get(at..at + 4)
            .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]) as usize)
    };
    let count = match rd_u32(bytes, 0) {
        Some(c) => c,
        None => return out,
    };
    let mut off = 4usize;
    for _ in 0..count {
        let Some(nlen) = rd_u32(bytes, off) else {
            break;
        };
        off += 4;
        let Some(name_bytes) = bytes.get(off..off + nlen) else {
            break;
        };
        let Ok(name) = core::str::from_utf8(name_bytes) else {
            break;
        };
        off += nlen;
        let Some(mlen) = rd_u32(bytes, off) else {
            break;
        };
        off += 4;
        let Some(mod_bytes) = bytes.get(off..off + mlen) else {
            break;
        };
        off += mlen;
        if let Ok(m) = svm_encode::decode_module(mod_bytes) {
            out.push((name.to_string(), m));
        }
    }
    out
}

/// Decode the module at `[mod_ptr, mod_len)` and run it as the **`svm-posix` shell** (see
/// [`posix_shell_exec`]) with `[stdin_ptr, stdin_len)` as the script — the playground's shell card.
/// `[cmds_ptr, cmds_len)`, when non-empty, is the **PATH-registry blob** ([`parse_shell_cmds`]): the
/// `__stage` ring-filter runner (so `cat f | sort | uniq` takes the **concurrent ring path** — op 11 +
/// `SharedRegion` + futex) and any **external commands** (`primes N`, …) the shell `exec`s as op-13
/// §14 children. Pass `cmds_len = 0` to run bare (memfs pipelines, no external commands). Same
/// capture/accessor contract as [`svm_run_onramp`]: read the captured stdout via `svm_stdout_ptr`+
/// `svm_stdout_len`, the exit code via [`svm_exit_code`], the status via [`svm_status`]. Returns the
/// guest's `i64` result. Shares the `OUT`/`ERR`/`EXIT_CODE` capture slots with the other run exports —
/// read them before the next call.
#[no_mangle]
pub extern "C" fn svm_run_shell(
    mod_ptr: *const u8,
    mod_len: usize,
    stdin_ptr: *const u8,
    stdin_len: usize,
    cmds_ptr: *const u8,
    cmds_len: usize,
) -> i64 {
    let set = |s: i32| unsafe { LAST_STATUS = s };
    // SAFETY: the host guarantees both ranges are live `svm_alloc`ations it just filled.
    let bytes = unsafe { core::slice::from_raw_parts(mod_ptr, mod_len) };
    let stdin: &[u8] = if stdin_ptr.is_null() || stdin_len == 0 {
        &[]
    } else {
        unsafe { core::slice::from_raw_parts(stdin_ptr, stdin_len) }
    };
    let m = match svm_encode::decode_module(bytes) {
        Ok(m) => m,
        Err(_) => {
            set(STATUS_DECODE_ERR);
            return 0;
        }
    };
    // The PATH registry (`__stage` + external commands). A truncated/undecodable blob is non-fatal:
    // `parse_shell_cmds` drops the bad entries and the shell runs with whatever registered.
    let owned = if cmds_ptr.is_null() || cmds_len == 0 {
        Vec::new()
    } else {
        let cb = unsafe { core::slice::from_raw_parts(cmds_ptr, cmds_len) };
        parse_shell_cmds(cb)
    };
    let cmds: Vec<(&str, &svm_ir::Module)> = owned.iter().map(|(n, m)| (n.as_str(), m)).collect();
    let out = posix_shell_exec_with(&m, stdin, &cmds);
    set(out.status);
    // SAFETY: single-threaded wasm; the capture slots are read back only via the export accessors.
    unsafe {
        stash(&mut *core::ptr::addr_of_mut!(OUT), out.stdout);
        stash(&mut *core::ptr::addr_of_mut!(ERR), out.stderr);
        EXIT_CODE = out.exit_code;
    }
    out.value
}

/// **In-browser link + run of a frontend-emitted program** (docs/SVM_BROWSER_PLAN.md option (b)):
/// the live-editing path, language-agnostic. Given a **program** unit and a **library** unit —
/// each either SVM-IR **text** or a **binary object** (`.svmo` bytes, the v9 object dialect;
/// told apart by the `SVM\0` magic, so the two params mix freely) — and the name of the export
/// to run, this loads both, links them (`link_with_manifest`), wraps the named entry in a
/// powerbox `_start` (`synth_manifest_start`), verifies, and runs it through the same on-ramp
/// powerbox as [`svm_run_onramp`] — so freshly-emitted source runs without a native link/encode
/// step. Results are read back through the same accessors (`svm_stdout_ptr`/`_len`,
/// `svm_status`, `svm_exit_code`).
///
/// This is the generic browser counterpart to the native link path: **nothing here is specific to
/// any source language.** A program's own-data addresses ride in its module text as `data.self
/// <offset>` instructions (resolved to the program unit's assigned window base by the linker), so no
/// separate relocation buffer is passed — the self-describing link forms replaced the old
/// `(func, block, inst)` reloc table. A frontend's own wire format (what it names its entry export,
/// which blob is its runtime) stays entirely on the caller's side.
///
/// The program is linked as unit 1 (its func 0 exported under `entry_name`) against the library as
/// unit 0 (re-exporting the library's own inline exports), so the program's calls into the library
/// resolve by name.
#[no_mangle]
pub extern "C" fn svm_link_run(
    prog_ptr: *const u8,
    prog_len: usize,
    lib_ptr: *const u8,
    lib_len: usize,
    entry_ptr: *const u8,
    entry_len: usize,
    stdin_ptr: *const u8,
    stdin_len: usize,
) -> i64 {
    let set = |s: i32| unsafe { LAST_STATUS = s };
    let slice = |p: *const u8, n: usize| -> &'static [u8] {
        if p.is_null() || n == 0 {
            &[]
        } else {
            unsafe { core::slice::from_raw_parts(p, n) }
        }
    };
    let entry_name = match core::str::from_utf8(slice(entry_ptr, entry_len)) {
        Ok(s) => s,
        Err(_) => {
            set(STATUS_DECODE_ERR);
            return 0;
        }
    };
    let stdin = slice(stdin_ptr, stdin_len);

    // A unit is binary iff it opens with the container magic (`SVM\0`) — text IR can't start
    // with a NUL, so the sniff is unambiguous. Binary rides `decode_unit` (the object dialect;
    // a resolved runnable module is a degenerate unit and loads fine), text rides the parser.
    let load_unit = |bytes: &[u8]| -> Option<svm_ir::Module> {
        if bytes.starts_with(b"SVM\0") {
            svm_encode::decode_unit(bytes).ok()
        } else {
            svm_text::parse_module(core::str::from_utf8(bytes).ok()?).ok()
        }
    };
    let program = match load_unit(slice(prog_ptr, prog_len)) {
        Some(m) => m,
        None => {
            set(STATUS_DECODE_ERR);
            return 0;
        }
    };
    let lib = match load_unit(slice(lib_ptr, lib_len)) {
        Some(m) => m,
        None => {
            set(STATUS_DECODE_ERR);
            return 0;
        }
    };
    let lib_exports: Vec<(String, svm_ir::FuncIdx)> = lib
        .exports
        .iter()
        .map(|e| (e.name.clone(), e.func))
        .collect();

    let linked = match svm_ir::link_with_manifest(&[
        svm_ir::LinkUnit {
            module: lib,
            exports: lib_exports,
            ..Default::default()
        },
        svm_ir::LinkUnit {
            module: program,
            exports: vec![(entry_name.to_string(), 0)],
            ..Default::default()
        },
    ]) {
        Ok(m) => m,
        Err(_) => {
            set(STATUS_UNSUPPORTED);
            return 0;
        }
    };
    let entry = match linked.resolve_export(entry_name) {
        Some(e) => e,
        None => {
            set(STATUS_UNSUPPORTED);
            return 0;
        }
    };
    let module = match svm_ir::synth_manifest_start(linked, entry, false) {
        Ok(m) => m,
        Err(_) => {
            set(STATUS_UNSUPPORTED);
            return 0;
        }
    };
    // Verify before running: a program that references an undefined proc links to an unresolvable
    // manifest import / out-of-range target, which would otherwise fault deep in the engine. Reject
    // it cleanly (STATUS_UNSUPPORTED) so a typo can't take down the playground's wasm instance.
    if svm_verify::verify_module(&module).is_err() {
        set(STATUS_UNSUPPORTED);
        return 0;
    }

    let out = onramp_exec(&module, stdin);
    set(out.status);
    // SAFETY: single-threaded wasm; capture slots read back only via the export accessors.
    unsafe {
        stash(&mut *core::ptr::addr_of_mut!(OUT), out.stdout);
        stash(&mut *core::ptr::addr_of_mut!(ERR), out.stderr);
        EXIT_CODE = out.exit_code;
    }
    out.value
}

/// Pointer / length of the RGBA framebuffer the most recent [`svm_run_onramp`] guest presented via
/// the `display` capability (`(null, 0)` if none). `svm_framebuffer_width`/`_height` give its
/// dimensions; `len` is `width*height*4`. Valid until the next `svm_run_onramp`; do not `svm_dealloc`.
#[no_mangle]
pub extern "C" fn svm_framebuffer_ptr() -> *const u8 {
    unsafe { (*core::ptr::addr_of!(FB)).0 }
}
#[no_mangle]
pub extern "C" fn svm_framebuffer_len() -> usize {
    unsafe { (*core::ptr::addr_of!(FB)).1 }
}
#[no_mangle]
pub extern "C" fn svm_framebuffer_width() -> u32 {
    unsafe { FB_W }
}
#[no_mangle]
pub extern "C" fn svm_framebuffer_height() -> u32 {
    unsafe { FB_H }
}

/// The live per-frame [`OnrampReactor`] (interactive/graphical guests: bounce, eventually Doom).
/// `None` until [`svm_onramp_open`]; single-threaded wasm, so a plain static is sound.
static mut REACTOR: Option<OnrampReactor> = None;

/// Open a per-frame **reactor** over the on-ramp module at `[mod_ptr, mod_len)` (an interactive guest
/// exporting `tick`): decode, grant the powerbox, run `_start`. Returns `0` on success, else a
/// negative `STATUS_*`; also sets [`LAST_STATUS`]. Replaces any prior reactor. Drive it with
/// [`svm_onramp_frame`], feed input with [`svm_onramp_key`], and read each frame via the
/// `svm_framebuffer_*` exports; close with [`svm_onramp_close`].
#[no_mangle]
pub extern "C" fn svm_onramp_open(mod_ptr: *const u8, mod_len: usize) -> i32 {
    let set = |s: i32| unsafe { LAST_STATUS = s };
    // SAFETY: the host guarantees `[mod_ptr, mod_len)` is a live `svm_alloc`ation it just filled.
    let bytes = unsafe { core::slice::from_raw_parts(mod_ptr, mod_len) };
    let m = match svm_encode::decode_module(bytes) {
        Ok(m) => m,
        Err(_) => {
            set(STATUS_DECODE_ERR);
            return -STATUS_DECODE_ERR;
        }
    };
    match OnrampReactor::open(&m) {
        Ok(r) => {
            // SAFETY: single-threaded wasm; the reactor is touched only by these export accessors.
            unsafe { *core::ptr::addr_of_mut!(REACTOR) = Some(r) };
            set(STATUS_OK);
            0
        }
        Err(status) => {
            set(status);
            -status
        }
    }
}

/// Like [`svm_onramp_open`] but also grant an `fs` capability serving one read-only file — the bytes
/// at `[data_ptr, data_len)` under the name at `[name_ptr, name_len)` (the WAD read path Doom needs:
/// its `_start` reads the IWAD through `fs`). The host `svm_alloc`s and fills both buffers before the
/// call and frees them after (the file bytes are copied into the reactor's `fs` server). Returns `0`
/// on success, else a negative `STATUS_*`; also sets [`LAST_STATUS`]. Replaces any prior reactor.
#[no_mangle]
pub extern "C" fn svm_onramp_open_fs(
    mod_ptr: *const u8,
    mod_len: usize,
    name_ptr: *const u8,
    name_len: usize,
    data_ptr: *const u8,
    data_len: usize,
) -> i32 {
    let set = |s: i32| unsafe { LAST_STATUS = s };
    // SAFETY: the host guarantees each `[ptr, len)` is a live `svm_alloc`ation it just filled.
    let bytes = unsafe { core::slice::from_raw_parts(mod_ptr, mod_len) };
    let name = unsafe { core::slice::from_raw_parts(name_ptr, name_len) };
    let data = unsafe { core::slice::from_raw_parts(data_ptr, data_len) };
    let m = match svm_encode::decode_module(bytes) {
        Ok(m) => m,
        Err(_) => {
            set(STATUS_DECODE_ERR);
            return -STATUS_DECODE_ERR;
        }
    };
    let name = String::from_utf8_lossy(name).into_owned();
    match OnrampReactor::open_with_fs(&m, name, data.to_vec()) {
        Ok(r) => {
            // SAFETY: single-threaded wasm; the reactor is touched only by these export accessors.
            unsafe { *core::ptr::addr_of_mut!(REACTOR) = Some(r) };
            set(STATUS_OK);
            0
        }
        Err(status) => {
            set(status);
            -status
        }
    }
}

/// Advance the open reactor by one frame: call the guest's `tick`, stash the presented frame (read
/// via `svm_framebuffer_*`) and any stdout delta (read via `svm_stdout_*`), and return the frame
/// status (`0` = keep going, [`STATUS_EXIT`] = the guest exited, else a trap). Returns
/// [`STATUS_UNSUPPORTED`] if no reactor is open. Sets [`LAST_STATUS`].
#[no_mangle]
pub extern "C" fn svm_onramp_frame() -> i32 {
    let set = |s: i32| unsafe { LAST_STATUS = s };
    // SAFETY: single-threaded wasm; exclusive access to the reactor for this call.
    let reactor = unsafe { (*core::ptr::addr_of_mut!(REACTOR)).as_mut() };
    let Some(reactor) = reactor else {
        set(STATUS_UNSUPPORTED);
        return STATUS_UNSUPPORTED;
    };
    let (status, stdout_delta) = reactor.frame();
    let (fb_rgba, fb_w, fb_h) = match reactor.take_frame() {
        Some(f) => (f.rgba, f.width, f.height),
        None => (Vec::new(), 0, 0),
    };
    set(status);
    // SAFETY: single-threaded wasm; the capture slots are read back only via the export accessors.
    unsafe {
        stash(&mut *core::ptr::addr_of_mut!(FB), fb_rgba);
        FB_W = fb_w;
        FB_H = fb_h;
        stash(&mut *core::ptr::addr_of_mut!(OUT), stdout_delta);
    }
    status
}

/// Enqueue a key event for the open reactor's guest to `poll` next frame (`pressed`: 1 = down,
/// 0 = up; `keycode`: the platform key id, e.g. a JS `keyCode`). No-op if no reactor is open.
#[no_mangle]
pub extern "C" fn svm_onramp_key(keycode: i32, pressed: i32) {
    // SAFETY: single-threaded wasm; shared read of the reactor's key queue.
    if let Some(reactor) = unsafe { (*core::ptr::addr_of!(REACTOR)).as_ref() } {
        reactor.push_key(keycode, pressed);
    }
}

/// Close the open reactor, freeing its instance. Idempotent.
#[no_mangle]
pub extern "C" fn svm_onramp_close() {
    // SAFETY: single-threaded wasm; exclusive access to drop the reactor.
    unsafe { *core::ptr::addr_of_mut!(REACTOR) = None };
}

/// Diagnostic: stash the open reactor's last-trap `Debug` string into [`OUT`] and return its length
/// (`0` if no reactor / no trap). Read the bytes via [`svm_stdout_ptr`]. Lets the page surface *why* a
/// reactor `tick` trapped (the `Trap` variant), not just the `STATUS_TRAP` code.
#[no_mangle]
pub extern "C" fn svm_onramp_trap_len() -> usize {
    // SAFETY: single-threaded wasm; shared read of the reactor.
    let s = unsafe { (*core::ptr::addr_of!(REACTOR)).as_ref() }.map_or("", |r| r.last_trap());
    let bytes = s.as_bytes().to_vec();
    let len = bytes.len();
    // SAFETY: single-threaded wasm; the stash is read back only via `svm_stdout_ptr`.
    unsafe { stash(&mut *core::ptr::addr_of_mut!(OUT), bytes) };
    len
}

// ---- cross-tier `env.call_interp` slot ABI (shared by the three servicers below) ------------------
//
// The emitter (svm-wasm-jit `emit_slot_store` / `emit_slot_load`) packs each cross-tier arg/result
// into the env scratch at a **running slot offset** computed from the callee's signature: scalars
// (`i32` widened, `f32` low 4 bytes, `i64`/`f64` whole) take one 8-byte slot, a `v128` takes two
// (16 raw little-endian bytes, #749). These helpers are the host end of that single encoding, so
// all three `*_call_interp` servicers decode/encode identically. Only `ref`/`cap` values are
// unmarshallable (excluded by `marshallable_sig`; a servicer fails closed if one appears).

/// Running byte offsets of each value of a signature side in the scratch (params and results each
/// start at 0 — result slots overlay arg slots). Mirrors the emitter's `slot_off`.
fn slot_offs(types: &[svm_ir::ValType]) -> Vec<usize> {
    let mut offs = Vec::with_capacity(types.len());
    let mut off = 0usize;
    for t in types {
        offs.push(off);
        off += if *t == svm_ir::ValType::V128 { 16 } else { 8 };
    }
    offs
}

/// Decode the scratch value at `args_ptr + off` to a [`Value`] of the callee's param type.
///
/// SAFETY: the caller guarantees `args_ptr + off` addresses the full slot(s) of `ty` (8 bytes, 16
/// for `v128`) inside the env scratch.
unsafe fn read_slot_value(ty: svm_ir::ValType, args_ptr: *const u8, off: usize) -> Value {
    let raw8 = || -> u64 {
        let mut b = [0u8; 8];
        unsafe { core::ptr::copy_nonoverlapping(args_ptr.add(off), b.as_mut_ptr(), 8) };
        u64::from_le_bytes(b)
    };
    match ty {
        svm_ir::ValType::I32 => Value::I32(raw8() as i32),
        svm_ir::ValType::I64 => Value::I64(raw8() as i64),
        svm_ir::ValType::F32 => Value::F32(f32::from_bits(raw8() as u32)),
        svm_ir::ValType::F64 => Value::F64(f64::from_bits(raw8())),
        svm_ir::ValType::V128 => {
            let mut b = [0u8; 16];
            unsafe { core::ptr::copy_nonoverlapping(args_ptr.add(off), b.as_mut_ptr(), 16) };
            Value::V128(b)
        }
        _ => Value::I64(raw8() as i64), // ref/cap never reach a cross-tier leaf; decode defensively
    }
}

/// Encode a cross-tier result [`Value`] into its slot(s) at `args_ptr + off`. `false` for a
/// `ref`/`cap` result (not marshallable — the caller fails the cross-tier call closed).
///
/// SAFETY: the caller guarantees `args_ptr + off` addresses the full slot(s) of `v`'s type (8
/// bytes, 16 for `v128`) inside the env scratch.
unsafe fn write_slot_value(v: &Value, args_ptr: *mut u8, off: usize) -> bool {
    let put8 = |raw: u64| {
        let b = raw.to_le_bytes();
        unsafe { core::ptr::copy_nonoverlapping(b.as_ptr(), args_ptr.add(off), 8) };
    };
    match v {
        Value::I32(x) => put8(*x as u32 as u64),
        Value::I64(x) => put8(*x as u64),
        Value::F32(x) => put8(x.to_bits() as u64),
        Value::F64(x) => put8(x.to_bits()),
        Value::V128(b) => unsafe {
            core::ptr::copy_nonoverlapping(b.as_ptr(), args_ptr.add(off), 16)
        },
        _ => return false,
    }
    true
}

#[cfg(test)]
mod xcall_slot_tests {
    use super::*;

    /// The two-slot `v128` scratch layout (#749): running offsets skip 16 bytes past a `v128`, and
    /// every marshallable value round-trips write→read bit-exactly at its offset — the host half of
    /// the encoding the emitter's `slot_off`/`emit_slot_store`/`emit_slot_load` produce (the wasmi
    /// differential in svm-wasm-jit `cross_tier.rs` pins the emitter half against this same layout).
    #[test]
    fn v128_two_slot_layout_round_trips() {
        use svm_ir::ValType;
        let sig = [
            ValType::V128,
            ValType::I64,
            ValType::F32,
            ValType::V128,
            ValType::I32,
        ];
        assert_eq!(slot_offs(&sig), vec![0, 16, 24, 32, 48]);

        let vals = [
            Value::V128(*b"0123456789abcdef"),
            Value::I64(-7),
            Value::F32(1.5),
            Value::V128([0xAA; 16]),
            Value::I32(-1),
        ];
        let mut scratch = [0u8; 56];
        let p = scratch.as_mut_ptr();
        for (v, off) in vals.iter().zip(slot_offs(&sig)) {
            // SAFETY: `scratch` covers the signature's full 56-byte slot span.
            assert!(unsafe { write_slot_value(v, p, off) });
        }
        for ((ty, v), off) in sig.iter().zip(&vals).zip(slot_offs(&sig)) {
            // SAFETY: same span as above.
            let got = unsafe { read_slot_value(*ty, p, off) };
            assert_eq!(format!("{got:?}"), format!("{v:?}"), "round-trip at {off}");
        }
    }
}

// ---- the wasm-JIT reactor (Doom's whole `tick` on emitted wasm) — BROWSER.md §"wasm-JIT tier" 5d ---
//
// Unlike the interpreter reactor (`svm_onramp_*`), the per-frame `tick` runs as a **JS-compiled**
// emitted wasm module (`svm_wasmjit`), instantiated by `play.js` against this cdylib's own linear
// memory. Each frame the page calls the emitted `f{tick}(win, env, sp)` directly; its cross-tier
// helpers relay `env.call_interp` back to [`svm_onramp_jit_call_interp`], which runs the callee on the
// interpreter over the same window with the powerbox (so `display`/`keyboard`/`fs`/`exit` resolve).
// The window lives in a Rust-owned `Box` inside linear memory; the page reads its base
// ([`svm_onramp_jit_win_ptr`]) for the emitted `win` argument and reads the emitted bytes to compile.

/// The live wasm-JIT reactor. `None` until [`svm_onramp_jit_open_fs`]; single-threaded wasm.
static mut JIT_REACTOR: Option<JitOnrampReactor> = None;

/// The JIT reactor's mapped-window log2 — 16 MiB, covering Doom's grown zone heap so the emitter's
/// static-`mapped` confinement mask holds (see [`JitOnrampReactor`]).
const JIT_WIN_LOG2: u8 = 24;

/// Open a **wasm-JIT reactor** over the on-ramp module at `[mod_ptr, mod_len)` with **no `fs` file** —
/// the reactor analogue of [`svm_onramp_open`], with the whole `tick` emitted to wasm (for interactive
/// guests that need no served file: bounce/life/mandelzoom). Decodes, enlarges the window, runs
/// `_start` on the interpreter, and emits the whole `tick`. Returns `0` on success, else a negative
/// `STATUS_*` (also set in [`LAST_STATUS`]) — notably [`STATUS_UNSUPPORTED`] if the `tick` isn't
/// wasm-JIT-emittable (the page falls back to the interp reactor). Otherwise identical to
/// [`svm_onramp_jit_open_fs`]; drive/close with the same `svm_onramp_jit_*` exports.
#[no_mangle]
pub extern "C" fn svm_onramp_jit_open(mod_ptr: *const u8, mod_len: usize) -> i32 {
    let set = |s: i32| unsafe { LAST_STATUS = s };
    // SAFETY: the host guarantees `[mod_ptr, mod_len)` is a live `svm_alloc`ation it just filled.
    let bytes = unsafe { core::slice::from_raw_parts(mod_ptr, mod_len) };
    let m = match svm_encode::decode_module(bytes) {
        Ok(m) => m,
        Err(_) => {
            set(STATUS_DECODE_ERR);
            return -STATUS_DECODE_ERR;
        }
    };
    // The play threads build imports a **shared** memory, so the emitted module must too. No `fs` file.
    match JitOnrampReactor::open_owned_jit(&m, JIT_WIN_LOG2, true, None) {
        Ok(r) => {
            // SAFETY: single-threaded wasm; the reactor is touched only by these export accessors.
            unsafe { *core::ptr::addr_of_mut!(JIT_REACTOR) = Some(r) };
            set(STATUS_OK);
            0
        }
        Err(status) => {
            set(status);
            -status
        }
    }
}

/// Open a **wasm-JIT reactor** over the on-ramp module at `[mod_ptr, mod_len)`, granting an `fs`
/// capability that serves the file `[data_ptr, data_len)` under the name `[name_ptr, name_len)` (Doom's
/// WAD). Decodes, enlarges the window, runs `_start` on the interpreter, and emits the whole `tick`.
/// Returns `0` on success, else a negative `STATUS_*` (also set in [`LAST_STATUS`]) — notably
/// [`STATUS_UNSUPPORTED`] if the `tick` isn't wasm-JIT-emittable (the page falls back to the interp
/// reactor). Replaces any prior JIT reactor. After success, read [`svm_onramp_jit_wasm_ptr`]/`_len`
/// (emitted bytes), [`svm_onramp_jit_win_ptr`], [`svm_onramp_jit_entry_sp`], [`svm_onramp_jit_tick`],
/// and [`svm_onramp_jit_env_bytes`] to set up the emitted module; drive it with
/// [`svm_onramp_jit_call_interp`] + [`svm_onramp_jit_present`]; feed input with
/// [`svm_onramp_jit_key`]; close with [`svm_onramp_jit_close`].
#[no_mangle]
pub extern "C" fn svm_onramp_jit_open_fs(
    mod_ptr: *const u8,
    mod_len: usize,
    name_ptr: *const u8,
    name_len: usize,
    data_ptr: *const u8,
    data_len: usize,
) -> i32 {
    let set = |s: i32| unsafe { LAST_STATUS = s };
    // SAFETY: the host guarantees each `[ptr, len)` is a live `svm_alloc`ation it just filled.
    let bytes = unsafe { core::slice::from_raw_parts(mod_ptr, mod_len) };
    let name = unsafe { core::slice::from_raw_parts(name_ptr, name_len) };
    let data = unsafe { core::slice::from_raw_parts(data_ptr, data_len) };
    let m = match svm_encode::decode_module(bytes) {
        Ok(m) => m,
        Err(_) => {
            set(STATUS_DECODE_ERR);
            return -STATUS_DECODE_ERR;
        }
    };
    let name = String::from_utf8_lossy(name).into_owned();
    // The play threads build imports a **shared** memory, so the emitted module must too.
    match JitOnrampReactor::open_owned_jit(&m, JIT_WIN_LOG2, true, Some((name, data.to_vec()))) {
        Ok(r) => {
            // SAFETY: single-threaded wasm; the reactor is touched only by these export accessors.
            unsafe { *core::ptr::addr_of_mut!(JIT_REACTOR) = Some(r) };
            set(STATUS_OK);
            0
        }
        Err(status) => {
            set(status);
            -status
        }
    }
}

/// Pointer / length of the emitted `tick` wasm bytes (valid until the reactor is replaced/closed; the
/// page copies them out and `WebAssembly.compile`s them). `(null, 0)` if no reactor is open.
#[no_mangle]
pub extern "C" fn svm_onramp_jit_wasm_ptr() -> *const u8 {
    unsafe { (*core::ptr::addr_of!(JIT_REACTOR)).as_ref() }
        .map_or(core::ptr::null(), |r| r.emitted_wasm().as_ptr())
}
#[no_mangle]
pub extern "C" fn svm_onramp_jit_wasm_len() -> usize {
    unsafe { (*core::ptr::addr_of!(JIT_REACTOR)).as_ref() }.map_or(0, |r| r.emitted_wasm().len())
}

/// The window base as a byte offset in this module's linear memory — the emitted `f{tick}`'s `win`.
#[no_mangle]
pub extern "C" fn svm_onramp_jit_win_ptr() -> usize {
    unsafe { (*core::ptr::addr_of!(JIT_REACTOR)).as_ref() }.map_or(0, |r| r.win_base())
}
/// The reactor calling-convention data-stack base — the emitted `f{tick}`'s `sp` argument.
#[no_mangle]
pub extern "C" fn svm_onramp_jit_entry_sp() -> i64 {
    unsafe { (*core::ptr::addr_of!(JIT_REACTOR)).as_ref() }.map_or(0, |r| r.entry_sp() as i64)
}
/// The SVM index of the exported `tick` — the emitted export is `f{tick}`.
#[no_mangle]
pub extern "C" fn svm_onramp_jit_tick() -> u32 {
    unsafe { (*core::ptr::addr_of!(JIT_REACTOR)).as_ref() }.map_or(0, |r| r.tick())
}
/// The `env` cell size (fuel counter + cross-tier scratch) the page must `svm_alloc` for the emitted
/// module's `env` argument.
#[no_mangle]
pub extern "C" fn svm_onramp_jit_env_bytes() -> usize {
    svm_wasm_jit::ENV_CELL_BYTES
}

/// **Cross-tier bounce** — the emitted `tick`'s `env.call_interp(func, args_ptr)` relays here. Runs
/// non-emitted `func` on the interpreter over the shared window with the powerbox, marshalling its i64
/// arg/result slots at `args_ptr` (in linear memory). Returns `0` on success, [`STATUS_EXIT`] if the
/// callee `Exit`ed, else [`STATUS_TRAP`] — the page throws on any nonzero to unwind the emitted `tick`.
#[no_mangle]
pub extern "C" fn svm_onramp_jit_call_interp(func: u32, args_ptr: *mut u8) -> i32 {
    // SAFETY: single-threaded wasm; exclusive access to the reactor for this call.
    let Some(reactor) = (unsafe { (*core::ptr::addr_of_mut!(JIT_REACTOR)).as_mut() }) else {
        return STATUS_UNSUPPORTED;
    };
    let (params, results) = {
        let (p, r) = reactor.func_sig(func);
        (p.to_vec(), r.to_vec())
    };
    // SAFETY: the host guarantees `args_ptr` addresses the signature's full slot span (the env scratch).
    let args: Vec<Value> = params
        .iter()
        .zip(slot_offs(&params))
        .map(|(t, o)| unsafe { read_slot_value(*t, args_ptr, o) })
        .collect();
    match reactor.run_cross_tier(func, &args) {
        Ok(vals) => {
            let offs = slot_offs(&results);
            for (i, v) in vals.iter().enumerate() {
                if i >= results.len() {
                    break;
                }
                // SAFETY: `args_ptr + offs[i]` is within the env scratch (result slots overlay arg slots).
                if !unsafe { write_slot_value(v, args_ptr, offs[i]) } {
                    return STATUS_TRAP;
                }
            }
            0
        }
        Err(Trap::Exit(_)) => STATUS_EXIT,
        Err(t) => {
            reactor.set_last_trap(format!("{t:?}"));
            STATUS_TRAP
        }
    }
}

/// Stash the frame the last emitted `tick` presented through `display` into the `svm_framebuffer_*`
/// slots (the page blits it after each frame). Returns `1` if a frame was presented, else `0`.
#[no_mangle]
pub extern "C" fn svm_onramp_jit_present() -> i32 {
    // SAFETY: single-threaded wasm; shared read of the reactor.
    let frame =
        unsafe { (*core::ptr::addr_of!(JIT_REACTOR)).as_ref() }.and_then(|r| r.take_frame());
    let (rgba, w, h) = match frame {
        Some(f) => (f.rgba, f.width, f.height),
        None => return 0,
    };
    // SAFETY: single-threaded wasm; the capture slots are read back only via the `svm_framebuffer_*`.
    unsafe {
        stash(&mut *core::ptr::addr_of_mut!(FB), rgba);
        FB_W = w;
        FB_H = h;
    }
    1
}

/// Enqueue a key event for the JIT reactor's guest to `poll` next frame (`pressed`: 1 = down, 0 = up).
#[no_mangle]
pub extern "C" fn svm_onramp_jit_key(keycode: i32, pressed: i32) {
    // SAFETY: single-threaded wasm; shared read of the reactor's key queue.
    if let Some(r) = unsafe { (*core::ptr::addr_of!(JIT_REACTOR)).as_ref() } {
        r.push_key(keycode, pressed);
    }
}

/// Diagnostic: stash the JIT reactor's last-trap string into [`OUT`] and return its length (`0` if
/// none). Read via [`svm_stdout_ptr`].
#[no_mangle]
pub extern "C" fn svm_onramp_jit_trap_len() -> usize {
    // SAFETY: single-threaded wasm; shared read of the reactor.
    let s = unsafe { (*core::ptr::addr_of!(JIT_REACTOR)).as_ref() }.map_or("", |r| r.last_trap());
    let bytes = s.as_bytes().to_vec();
    let len = bytes.len();
    // SAFETY: single-threaded wasm; the stash is read back only via `svm_stdout_ptr`.
    unsafe { stash(&mut *core::ptr::addr_of_mut!(OUT), bytes) };
    len
}

/// Close the open JIT reactor, freeing it (and its window `Box`). Idempotent.
#[no_mangle]
pub extern "C" fn svm_onramp_jit_close() {
    // SAFETY: single-threaded wasm; exclusive access to drop the reactor.
    unsafe { *core::ptr::addr_of_mut!(JIT_REACTOR) = None };
}

// ---- single-shot module wasm-JIT run (Lua/SQLite on emitted wasm) — the run-to-completion twin ------
//
// The module-demo analogue of the reactor FFI above: the whole program is func 0 (`_start`), emitted and
// run once as `f0(win, env, ...slots)`. The page compiles [`svm_onramp_jit_run_wasm_ptr`]/`_len`,
// instantiates against the cdylib's linear memory with `env.call_interp` → [`svm_onramp_jit_run_call_interp`],
// calls `f0` with [`svm_onramp_jit_run_win_ptr`] + the [`svm_onramp_jit_run_slot`] handles, then
// [`svm_onramp_jit_run_finish`] captures stdout/stderr/exit into the shared `OUT`/`ERR`/`EXIT_CODE`.

/// The live single-shot JIT run. `None` until [`svm_onramp_jit_run_open`]; single-threaded wasm.
static mut JIT_RUN: Option<JitOnrampRun> = None;

/// The single-shot run's fixed window log2 — 32 MiB, holding Lua/SQLite's heap (the emitted run can't
/// grow it, so it must be sized up front).
const JIT_RUN_WIN_LOG2: u8 = 25;

/// Open a **single-shot wasm-JIT run** over the on-ramp module at `[mod_ptr, mod_len)` (Lua/SQLite/hello):
/// resolve imports, outline cap-calls, grant the powerbox (seeding stdin from `[stdin_ptr, stdin_len)`),
/// materialize `.data`, and emit rooted at `_start`. Returns `0`, else a negative `STATUS_*` (also set in
/// [`LAST_STATUS`]) — notably [`STATUS_UNSUPPORTED`] if `_start` isn't emittable (the page falls back to
/// [`svm_run_onramp`]). Replaces any prior run. Drive it with the `svm_onramp_jit_run_*` exports below.
#[no_mangle]
pub extern "C" fn svm_onramp_jit_run_open(
    mod_ptr: *const u8,
    mod_len: usize,
    stdin_ptr: *const u8,
    stdin_len: usize,
    shared: i32,
) -> i32 {
    let set = |s: i32| unsafe { LAST_STATUS = s };
    // SAFETY: the host guarantees `[mod_ptr, mod_len)` is a live `svm_alloc`ation it just filled.
    let bytes = unsafe { core::slice::from_raw_parts(mod_ptr, mod_len) };
    let stdin: Vec<u8> = if stdin_ptr.is_null() || stdin_len == 0 {
        Vec::new()
    } else {
        // SAFETY: same host guarantee for the stdin range.
        unsafe { core::slice::from_raw_parts(stdin_ptr, stdin_len) }.to_vec()
    };
    let m = match svm_encode::decode_module(bytes) {
        Ok(m) => m,
        Err(_) => {
            set(STATUS_DECODE_ERR);
            return -STATUS_DECODE_ERR;
        }
    };
    // `shared != 0` ⇒ the emitted module imports a **shared** memory (the threads/cross-origin-isolated
    // build); `0` for the plain single-threaded build (e.g. GitHub Pages, no COOP/COEP), where the host
    // instantiates the emitted module against a non-shared `WebAssembly.Memory`. The flag must match the
    // memory the host actually provides — a mismatch fails instantiation.
    match JitOnrampRun::open_owned_run(&m, JIT_RUN_WIN_LOG2, shared != 0, stdin) {
        Ok(r) => {
            // SAFETY: single-threaded wasm; the run is touched only by these export accessors.
            unsafe { *core::ptr::addr_of_mut!(JIT_RUN) = Some(r) };
            set(STATUS_OK);
            0
        }
        Err(status) => {
            set(status);
            -status
        }
    }
}

/// Open a **single-shot wasm-JIT run** of the chibicc compiler card: decode the compiler module at
/// `[mod_ptr, mod_len)`, assemble the same memfs [`svm_run_onramp_fs`] does (the user's source at
/// `/in.c`, the built-in libc headers + any caller headers under `/include`), and emit `_start` as
/// wasm — so the browser runs chibicc's compile on the **wasm-JIT** instead of the bytecode
/// interpreter (the compiler's `fopen`/`write`/`exit` bounce cross-tier). The fast twin of
/// `svm_run_onramp_fs`; drive it with the same `svm_onramp_jit_run_*` exports (call
/// [`svm_onramp_jit_run_finish`] for the emitted IR on `svm_stdout_ptr`). `debug_info != 0` compiles
/// with `-g` (source-level debug section) — off by default, as in `svm_run_onramp_fs`. Returns `0`,
/// else a negative `STATUS_*` (also in [`LAST_STATUS`]) — notably [`STATUS_UNSUPPORTED`] if `_start`
/// isn't emittable (the page falls back to [`svm_run_onramp_fs`]).
#[no_mangle]
pub extern "C" fn svm_onramp_jit_run_open_fs(
    mod_ptr: *const u8,
    mod_len: usize,
    img_ptr: *const u8,
    img_len: usize,
    src_ptr: *const u8,
    src_len: usize,
    debug_info: i32,
) -> i32 {
    let set = |s: i32| unsafe { LAST_STATUS = s };
    // SAFETY: the host guarantees each range is a live `svm_alloc`ation it just filled.
    let bytes = unsafe { core::slice::from_raw_parts(mod_ptr, mod_len) };
    let src = unsafe { core::slice::from_raw_parts(src_ptr, src_len) };
    let m = match svm_encode::decode_module(bytes) {
        Ok(m) => m,
        Err(_) => {
            set(STATUS_DECODE_ERR);
            return -STATUS_DECODE_ERR;
        }
    };
    if svm_verify::verify_module(&m).is_err() {
        set(STATUS_VERIFY_ERR);
        return -STATUS_VERIFY_ERR;
    }
    let image = match chibicc_card_image(img_ptr, img_len, src) {
        Ok(image) => image,
        Err(status) => {
            set(status);
            return -status;
        }
    };
    let argv = chibicc_card_argv(debug_info != 0);
    // The play threads build imports a **shared** memory, so the emitted module must too.
    match JitOnrampRun::open_owned_run_fs(&m, JIT_RUN_WIN_LOG2, true, &image, &argv, Vec::new()) {
        Ok(r) => {
            // SAFETY: single-threaded wasm; the run is touched only by these export accessors.
            unsafe { *core::ptr::addr_of_mut!(JIT_RUN) = Some(r) };
            set(STATUS_OK);
            0
        }
        Err(status) => {
            set(status);
            -status
        }
    }
}

/// The self-host card's window: **2^27 = 128 MiB**, larger than the single-file card's
/// [`JIT_RUN_WIN_LOG2`] (32 MiB). chibicc's largest TU, `codegen_ir.c`, emits ~1.2 MB of IR and its
/// compile working set overruns 32 MiB — the run traps `unreachable` mid-emit (measured). 128 MiB clears
/// every cc1 TU (all byte-identical to native); the max-memory cap is 1 GiB, so this stays well within.
const SELFHOST_WIN_LOG2: u8 = 27;

/// **Self-host card — wasm-JIT tier** (the fast twin of [`svm_selfhost_emit_object_fs`]). Emit
/// `chibicc.svmb`'s `_start` to wasm and run it in `--emit-object` mode over one of chibicc's own cc1
/// TUs (seeded from the raw closure image `[img_ptr, img_len)`, memfs-relative TU path
/// `[tu_ptr, tu_len)`), so the browser compiles chibicc's own source on emitted wasm — every cc1 TU,
/// giants included, in a few hundred ms (SELFHOST_C.md). Drives the same finish path as the shipping JIT
/// card ([`svm_onramp_jit_run_finish`] → the object text on `svm_stdout_*`). Runs in a 128 MiB window
/// ([`SELFHOST_WIN_LOG2`]). Returns `0`, else a negative `STATUS_*` (also in [`LAST_STATUS`]).
#[no_mangle]
pub extern "C" fn svm_selfhost_jit_emit_object_fs(
    mod_ptr: *const u8,
    mod_len: usize,
    img_ptr: *const u8,
    img_len: usize,
    tu_ptr: *const u8,
    tu_len: usize,
    debug_info: i32,
) -> i32 {
    let set = |s: i32| unsafe { LAST_STATUS = s };
    // SAFETY: the host guarantees each range is a live `svm_alloc`ation it just filled.
    let bytes = unsafe { core::slice::from_raw_parts(mod_ptr, mod_len) };
    let image = unsafe { core::slice::from_raw_parts(img_ptr, img_len) };
    let tu = unsafe { core::slice::from_raw_parts(tu_ptr, tu_len) };
    let m = match svm_encode::decode_module(bytes) {
        Ok(m) => m,
        Err(_) => {
            set(STATUS_DECODE_ERR);
            return -STATUS_DECODE_ERR;
        }
    };
    if svm_verify::verify_module(&m).is_err() {
        set(STATUS_VERIFY_ERR);
        return -STATUS_VERIFY_ERR;
    }
    let argv = chibicc_selfhost_argv(tu, debug_info != 0);
    match JitOnrampRun::open_owned_run_fs(&m, SELFHOST_WIN_LOG2, true, image, &argv, Vec::new()) {
        Ok(r) => {
            // SAFETY: single-threaded wasm; the run is touched only by these export accessors.
            unsafe { *core::ptr::addr_of_mut!(JIT_RUN) = Some(r) };
            set(STATUS_OK);
            0
        }
        Err(status) => {
            set(status);
            -status
        }
    }
}

/// Pointer / length of the emitted `_start` wasm bytes (valid until the run is replaced/closed).
#[no_mangle]
pub extern "C" fn svm_onramp_jit_run_wasm_ptr() -> *const u8 {
    unsafe { (*core::ptr::addr_of!(JIT_RUN)).as_ref() }
        .map_or(core::ptr::null(), |r| r.emitted_wasm().as_ptr())
}
#[no_mangle]
pub extern "C" fn svm_onramp_jit_run_wasm_len() -> usize {
    unsafe { (*core::ptr::addr_of!(JIT_RUN)).as_ref() }.map_or(0, |r| r.emitted_wasm().len())
}
/// The window base as a byte offset in this module's linear memory — the emitted `f0`'s `win`.
#[no_mangle]
pub extern "C" fn svm_onramp_jit_run_win_ptr() -> usize {
    unsafe { (*core::ptr::addr_of!(JIT_RUN)).as_ref() }.map_or(0, |r| r.win_base())
}
/// The `env` cell size the page must `svm_alloc` for the emitted module's `env` argument.
#[no_mangle]
pub extern "C" fn svm_onramp_jit_run_env_bytes() -> usize {
    svm_wasm_jit::ENV_CELL_BYTES
}
/// The number of capability-handle params `_start` takes — the emitted `f0`'s trailing `...slots` args.
#[no_mangle]
pub extern "C" fn svm_onramp_jit_run_slot_count() -> usize {
    unsafe { (*core::ptr::addr_of!(JIT_RUN)).as_ref() }.map_or(0, |r| r.slots().len())
}
/// The `i`-th capability handle `_start` takes as a param (`0` if out of range / no run).
#[no_mangle]
pub extern "C" fn svm_onramp_jit_run_slot(i: usize) -> i32 {
    unsafe { (*core::ptr::addr_of!(JIT_RUN)).as_ref() }.map_or(0, |r| match r.slots().get(i) {
        Some(Value::I32(x)) => *x,
        Some(Value::I64(x)) => *x as i32,
        _ => 0,
    })
}

/// **Cross-tier bounce** — the emitted `f0`'s `env.call_interp(func, args_ptr)` relays here (identical
/// contract to [`svm_onramp_jit_call_interp`], but over the single-shot run's window/powerbox).
#[no_mangle]
pub extern "C" fn svm_onramp_jit_run_call_interp(func: u32, args_ptr: *mut u8) -> i32 {
    // SAFETY: single-threaded wasm; exclusive access to the run for this call.
    let Some(run) = (unsafe { (*core::ptr::addr_of_mut!(JIT_RUN)).as_mut() }) else {
        return STATUS_UNSUPPORTED;
    };
    let (params, results) = {
        let (p, r) = run.func_sig(func);
        (p.to_vec(), r.to_vec())
    };
    // SAFETY: the host guarantees `args_ptr` addresses the signature's full slot span (the env scratch).
    let args: Vec<Value> = params
        .iter()
        .zip(slot_offs(&params))
        .map(|(t, o)| unsafe { read_slot_value(*t, args_ptr, o) })
        .collect();
    match run.run_cross_tier(func, &args) {
        Ok(vals) => {
            let offs = slot_offs(&results);
            for (i, v) in vals.iter().enumerate() {
                if i >= results.len() {
                    break;
                }
                // SAFETY: `args_ptr + offs[i]` is within the env scratch (result slots overlay arg slots).
                if !unsafe { write_slot_value(v, args_ptr, offs[i]) } {
                    return STATUS_TRAP;
                }
            }
            0
        }
        // The guest `exit`ed — unwind the emitted `f0`; `svm_onramp_jit_run_finish` reports the code.
        Err(Trap::Exit(_)) => STATUS_EXIT,
        Err(t) => {
            run.set_last_trap(format!("{t:?}"));
            STATUS_TRAP
        }
    }
}

/// Record how the emitted `f0` finished, from the JS driver's vantage: `value` is `f0`'s return (the
/// guest's top-level result, meaningful only when it *returned*), and `threw != 0` iff the call unwound.
/// Call this before [`svm_onramp_jit_run_finish`]. The driver can't tell an `exit` unwind from a trap
/// unwind, so it reports "threw" and this pairs it with the Rust-side `exited` flag (set on a cross-tier
/// `Exit`): a throw that did not `exit` is a trap. Optional — a caller that skips it gets the legacy
/// "returned, value 0" reading (`jit-profile.mjs` doesn't care about the value).
#[no_mangle]
pub extern "C" fn svm_onramp_jit_run_report(threw: i32, value: i64) {
    // SAFETY: single-threaded wasm; exclusive access to the run.
    if let Some(run) = unsafe { (*core::ptr::addr_of_mut!(JIT_RUN)).as_mut() } {
        run.record_outcome(threw != 0, value);
    }
}

/// Capture the finished run's streams into the shared `OUT`/`ERR`/`EXIT_CODE`/`RUN_VALUE` + any presented
/// frame into the `svm_framebuffer_*` slots, so the page reads them via the usual [`svm_stdout_ptr`] /
/// [`svm_exit_code`] / [`svm_run_value`] / `svm_framebuffer_*` accessors — identical to
/// [`svm_run_onramp`]'s contract, so the interpreter and the wasm-JIT agree on result + exit + trap
/// (INVARIANT 9). Call once after `f0` returns or unwinds (and after [`svm_onramp_jit_run_report`]).
/// Returns [`STATUS_EXIT`] if the guest `exit`ed, [`STATUS_TRAP`] if the emitted run unwound on a trap
/// (a wasm `unreachable` / a cross-tier bounce that trapped — never a truncated `STATUS_OK`), else
/// [`STATUS_OK`] with the returned value in `RUN_VALUE`.
#[no_mangle]
pub extern "C" fn svm_onramp_jit_run_finish() -> i32 {
    // SAFETY: single-threaded wasm; exclusive access to the run.
    let Some(run) = (unsafe { (*core::ptr::addr_of!(JIT_RUN)).as_ref() }) else {
        return STATUS_UNSUPPORTED;
    };
    let stdout = run.stdout().to_vec();
    let stderr = run.stderr().to_vec();
    // Exit is checked first (a cross-tier `Exit` sets both `exited` and, via the JS driver, `trapped`);
    // then a trap; then a clean return carrying the guest's result value. This mirrors `svm_run_onramp`'s
    // `Exit` / other-`Err` / `Ok(value)` arms exactly, so a program has the same status + value on both
    // tiers.
    let (status, code, value) = if run.exited() {
        (STATUS_EXIT, run.exit_code(), 0)
    } else if run.trapped() {
        (STATUS_TRAP, 0, 0)
    } else {
        (STATUS_OK, 0, run.returned_value())
    };
    let (fb_rgba, fb_w, fb_h) = match run.take_frame() {
        Some(f) => (f.rgba, f.width, f.height),
        None => (Vec::new(), 0, 0),
    };
    // SAFETY: single-threaded wasm; the capture slots are read back only via the export accessors.
    unsafe {
        stash(&mut *core::ptr::addr_of_mut!(OUT), stdout);
        stash(&mut *core::ptr::addr_of_mut!(ERR), stderr);
        stash(&mut *core::ptr::addr_of_mut!(FB), fb_rgba);
        FB_W = fb_w;
        FB_H = fb_h;
        EXIT_CODE = code;
        RUN_VALUE = value;
        LAST_STATUS = status;
    }
    status
}

/// Diagnostic: stash the single-shot run's last-trap string into [`OUT`] and return its length.
#[no_mangle]
pub extern "C" fn svm_onramp_jit_run_trap_len() -> usize {
    let s = unsafe { (*core::ptr::addr_of!(JIT_RUN)).as_ref() }.map_or("", |r| r.last_trap());
    let bytes = s.as_bytes().to_vec();
    let len = bytes.len();
    // SAFETY: single-threaded wasm; the stash is read back only via `svm_stdout_ptr`.
    unsafe { stash(&mut *core::ptr::addr_of_mut!(OUT), bytes) };
    len
}

/// Close the open single-shot run, freeing it (and its window `Box`). Idempotent.
#[no_mangle]
pub extern "C" fn svm_onramp_jit_run_close() {
    // SAFETY: single-threaded wasm; exclusive access to drop the run.
    unsafe { *core::ptr::addr_of_mut!(JIT_RUN) = None };
}

/// Pointer / length of the captured stdout from the most recent [`svm_run_pb`] (valid until the next
/// `svm_run_pb`; do not `svm_dealloc` it).
#[no_mangle]
pub extern "C" fn svm_stdout_ptr() -> *const u8 {
    unsafe { (*core::ptr::addr_of!(OUT)).0 }
}
#[no_mangle]
pub extern "C" fn svm_stdout_len() -> usize {
    unsafe { (*core::ptr::addr_of!(OUT)).1 }
}
/// Pointer / length of the data image from the most recent [`svm_pg_snapshot`] (valid until the next
/// snapshot; do not `svm_dealloc` it).
#[no_mangle]
pub extern "C" fn svm_pg_snapshot_ptr() -> *const u8 {
    unsafe { (*core::ptr::addr_of!(PG_SNAP)).0 }
}
#[no_mangle]
pub extern "C" fn svm_pg_snapshot_len() -> usize {
    unsafe { (*core::ptr::addr_of!(PG_SNAP)).1 }
}
/// Pointer / length of the captured stderr from the most recent [`svm_run_pb`] (same lifetime rule).
#[no_mangle]
pub extern "C" fn svm_stderr_ptr() -> *const u8 {
    unsafe { (*core::ptr::addr_of!(ERR)).0 }
}
#[no_mangle]
pub extern "C" fn svm_stderr_len() -> usize {
    unsafe { (*core::ptr::addr_of!(ERR)).1 }
}
/// Exit code from the most recent [`svm_run_pb`] (valid when [`svm_status`] is [`STATUS_EXIT`]).
#[no_mangle]
pub extern "C" fn svm_exit_code() -> i32 {
    unsafe { EXIT_CODE }
}

/// The value the guest's top-level function returned on the most recent single-shot JIT run (valid when
/// [`svm_status`] / the finish status is [`STATUS_OK`]; `0` after an exit or trap). This is the same
/// result `svm_run_onramp` returns on the interpreter, so the page shows an identical value on both
/// tiers for a returned program (INVARIANT 9).
#[no_mangle]
pub extern "C" fn svm_run_value() -> i64 {
    unsafe { RUN_VALUE }
}

/// Decode the module at `[mod_ptr, mod_len)` and run function 0 (single `i64` `arg`, deny-all
/// `Host`) over a window **seeded** with `[init_ptr, init_len)`, then capture the final window image
/// (see [`capture_exec`]). Returns the guest's `i64` result; sets [`LAST_STATUS`]. The captured image
/// (the first `init_len` bytes of memory after the run) is read via [`svm_snapshot_ptr`] /
/// [`svm_snapshot_len`] and is cdylib-managed (valid until the next call; do not `svm_dealloc` it).
#[no_mangle]
pub extern "C" fn svm_run_capture(
    mod_ptr: *const u8,
    mod_len: usize,
    init_ptr: *const u8,
    init_len: usize,
    arg: i64,
) -> i64 {
    let set = |s: i32| unsafe { LAST_STATUS = s };
    // SAFETY: the host guarantees both ranges are live `svm_alloc`ations it just filled.
    let bytes = unsafe { core::slice::from_raw_parts(mod_ptr, mod_len) };
    let init: &[u8] = if init_ptr.is_null() || init_len == 0 {
        &[]
    } else {
        unsafe { core::slice::from_raw_parts(init_ptr, init_len) }
    };
    let m = match svm_encode::decode_module(bytes) {
        Ok(m) => m,
        Err(_) => {
            set(STATUS_DECODE_ERR);
            return 0;
        }
    };
    let out = capture_exec(&m, init, arg);
    set(out.status);
    // SAFETY: single-threaded wasm; the slot is read back only via the export accessors.
    unsafe { stash(&mut *core::ptr::addr_of_mut!(SNAP), out.snapshot) };
    out.value
}

/// Pointer / length of the captured final window image from the most recent [`svm_run_capture`].
#[no_mangle]
pub extern "C" fn svm_snapshot_ptr() -> *const u8 {
    unsafe { (*core::ptr::addr_of!(SNAP)).0 }
}
#[no_mangle]
pub extern "C" fn svm_snapshot_len() -> usize {
    unsafe { (*core::ptr::addr_of!(SNAP)).1 }
}

// ---- playground: in-browser SVM-text front end (parse → verify → encode) ------------------------

/// Compile the **SVM text** at `[src_ptr, src_len)` (UTF-8) into the `svm-encode` binary form the
/// `svm_run*` / `svm_par_*` entries consume: parse (`svm-text`) → verify (`svm-verify`) → encode.
/// Returns `1` and stashes the encoded module bytes, or `0` and stashes a UTF-8 error message
/// (which stage failed and why). Read the stash via [`svm_parse_ptr`] + [`svm_parse_len`] before
/// the next call — this is the playground's front end, so rejects must come back as *messages*,
/// not statuses.
#[no_mangle]
pub extern "C" fn svm_parse(src_ptr: *const u8, src_len: usize) -> i32 {
    let bytes: &[u8] = if src_ptr.is_null() || src_len == 0 {
        &[]
    } else {
        // SAFETY: the host guarantees `[src_ptr, src_len)` is a live allocation it just filled.
        unsafe { core::slice::from_raw_parts(src_ptr, src_len) }
    };
    // SAFETY: single-threaded main-thread use; the slot is read back only via the accessors below.
    let put = |ok: i32, data: Vec<u8>| -> i32 {
        unsafe { stash(&mut *core::ptr::addr_of_mut!(PARSE), data) };
        ok
    };
    let src = match core::str::from_utf8(bytes) {
        Ok(s) => s,
        Err(e) => return put(0, format!("source is not UTF-8: {e}").into_bytes()),
    };
    let m = match svm_text::parse_module(src) {
        Ok(m) => m,
        // `ParseError`'s Display already carries the "parse error: " prefix.
        Err(e) => return put(0, format!("{e}").into_bytes()),
    };
    if let Err(e) = svm_verify::verify_module(&m) {
        return put(0, format!("verify error: {e:?}").into_bytes());
    }
    put(1, svm_encode::encode_module(&m))
}

/// Pointer / length of the most recent [`svm_parse`] output (module bytes on `1`, error text on `0`).
#[no_mangle]
pub extern "C" fn svm_parse_ptr() -> *const u8 {
    unsafe { (*core::ptr::addr_of!(PARSE)).0 }
}
#[no_mangle]
pub extern "C" fn svm_parse_len() -> usize {
    unsafe { (*core::ptr::addr_of!(PARSE)).1 }
}
/// The stashed [`svm_parse`] output (same cdylib-managed lifetime as `OUT`/`ERR`).
static mut PARSE: (*mut u8, usize) = (core::ptr::null_mut(), 0);

// ---- Debug Adapter Protocol (DEBUGGING.md) — the debugger, over the bytecode engine --------------
// The playground drives the same `svm-dap` server the CLI/editor use, but selecting the **bytecode**
// backend (`"engine":"bytecode"` in `launch`) so it debugs the engine that actually ships, never the
// tree-walker (which stays the differential oracle). The wire is trivial: JS sends a DAP request JSON,
// the cdylib parses it, calls the pure `DapServer::handle`, and stashes the reply (a JSON array of one
// response + any events) for the JS to read back — the same request→messages logic the `dap.rs` tests
// drive, minus the `Content-Length` framing (`run_stdio`, unused in wasm).

/// The live DAP session's server (single-threaded, main-thread only, like every stash here).
static mut DAP_SERVER: Option<svm_dap::DapServer> = None;
/// The stashed reply of the most recent [`svm_dap_request`] (cdylib-managed, like `PARSE`).
static mut DAP_OUT: (*mut u8, usize) = (core::ptr::null_mut(), 0);

/// Start a fresh debug session (drop any prior server). Call before the `initialize` request so each
/// session begins clean.
#[no_mangle]
pub extern "C" fn svm_dap_reset() -> i32 {
    // SAFETY: single-threaded main-thread state, like `PARSE`/`OUT`.
    unsafe {
        *core::ptr::addr_of_mut!(DAP_SERVER) = Some(svm_dap::DapServer::new());
    }
    0
}

/// Feed one DAP request (a JSON object at `[ptr, len)`) to the session server and stash the JSON array
/// of reply messages (`[response, event…]`) for [`svm_dap_response_ptr`] + [`svm_dap_response_len`].
/// Returns `0`, or `-1` if the request isn't valid UTF-8 / JSON. Lazily creates the server if
/// [`svm_dap_reset`] wasn't called first.
#[no_mangle]
pub extern "C" fn svm_dap_request(ptr: *const u8, len: usize) -> i32 {
    let bytes: &[u8] = if ptr.is_null() || len == 0 {
        &[]
    } else {
        // SAFETY: the host guarantees `[ptr, len)` is a live allocation it just filled.
        unsafe { core::slice::from_raw_parts(ptr, len) }
    };
    // SAFETY: single-reader stash on the main thread, like the `svm_parse` accessors.
    let put = |data: Vec<u8>| unsafe { stash(&mut *core::ptr::addr_of_mut!(DAP_OUT), data) };
    let text = match core::str::from_utf8(bytes) {
        Ok(s) => s,
        Err(_) => {
            put(Vec::new());
            return -1;
        }
    };
    let req = match svm_dap::parse(text) {
        Some(j) => j,
        None => {
            put(Vec::new());
            return -1;
        }
    };
    // SAFETY: single-threaded; the server is only ever touched from the main thread.
    let server = unsafe {
        let slot = &mut *core::ptr::addr_of_mut!(DAP_SERVER);
        if slot.is_none() {
            *slot = Some(svm_dap::DapServer::new());
        }
        slot.as_mut().unwrap()
    };
    // A top-level JSON **array** batches requests (INTERACTIVE_EMBEDDING.md, the step+reads
    // bundle): each element is handled in order and the reply messages concatenate into the one
    // reply array — a step + N state reads in a single FFI crossing. A single object stays the
    // one-request pump; the reply shape is identical either way (`web/dap.js` already consumes an
    // array and filters by `type`).
    let reply = match &req {
        svm_dap::Json::Arr(reqs) => {
            let mut msgs = Vec::new();
            for r in reqs {
                msgs.extend(server.handle(r));
            }
            svm_dap::Json::Arr(msgs)
        }
        _ => svm_dap::Json::Arr(server.handle(&req)),
    }
    .to_string()
    .into_bytes();
    put(reply);
    0
}

/// Pointer / length of the most recent [`svm_dap_request`] reply (a JSON array of DAP messages).
#[no_mangle]
pub extern "C" fn svm_dap_response_ptr() -> *const u8 {
    unsafe { (*core::ptr::addr_of!(DAP_OUT)).0 }
}
#[no_mangle]
pub extern "C" fn svm_dap_response_len() -> usize {
    unsafe { (*core::ptr::addr_of!(DAP_OUT)).1 }
}

// ---- W3 in the browser: run-mode memory profiling (INTERACTIVE_EMBEDDING.md slice 4) -------------
// The debug tier feeds the host-side models through the access sink (`svm-dap`); a **non-debug**
// profiling run uses the W3 instrumentation pass instead: rewrite the module so every memory op
// announces itself on a host-fn capability — the `svm-run` `with_mem_hooks` twin, reproduced here
// because the cdylib depends on neither `svm-run` nor its OS-bound PAL (`svm-opt` is pure IR and
// wasm-clean) — feed the same `MemModel`, and stash its stats JSON. The rewrite never meets a
// debugger: run-mode entries inspect nothing, so the inserted ops are invisible by construction.

/// The stashed stats JSON of the most recent [`svm_mem_profile`] run.
static mut MEMPROF: (*mut u8, usize) = (core::ptr::null_mut(), 0);

/// The `svm-run` `decode_mem_event` twin (the op/arg layout is owned by
/// `svm_opt::instrument::mem_hook_op`). Drift between the twins is pinned by
/// `browser/tests/mem_profile.rs`, which compares this hook-fed model against a sink-fed one on
/// the same guest, stats-for-stats.
fn decode_mem_event(op: u32, args: &[i64]) -> Option<svm_interp::MemEvent> {
    use svm_interp::MemEvent as E;
    use svm_opt::instrument::mem_hook_op as k;
    let a = |i: usize| args.get(i).copied().map(|v| v as u64);
    Some(match (op, args.len()) {
        (k::LOAD, 2) => E::Load {
            addr: a(0)?,
            width: args[1] as u32,
        },
        (k::STORE, 2) => E::Store {
            addr: a(0)?,
            width: args[1] as u32,
        },
        (k::ATOMIC_LOAD, 2) => E::AtomicLoad {
            addr: a(0)?,
            width: args[1] as u32,
        },
        (k::ATOMIC_STORE, 2) => E::AtomicStore {
            addr: a(0)?,
            width: args[1] as u32,
        },
        (k::ATOMIC_RMW, 2) => E::AtomicRmw {
            addr: a(0)?,
            width: args[1] as u32,
        },
        (k::ATOMIC_CMPXCHG, 2) => E::AtomicCmpxchg {
            addr: a(0)?,
            width: args[1] as u32,
        },
        (k::COPY, 3) => E::Copy {
            dst: a(0)?,
            src: a(1)?,
            len: a(2)?,
        },
        (k::FILL, 2) => E::Fill {
            dst: a(0)?,
            len: a(1)?,
        },
        _ => return None,
    })
}

/// Profile `[ptr, len)`'s module (function 0, deny-all plus the hook grant — a compute guest):
/// instrument with the W3 pass, re-verify (fail-closed like every rewrite), run on the bytecode
/// engine feeding a [`svm_dap::models::MemModel`] (geometry from the args; `0` = the teaching
/// default), and stash the stats JSON for [`svm_mem_profile_stats_ptr`]. Returns `0` on a clean
/// run, `1` on a guest trap (stats still stashed — the final event is the attempted faulting
/// access), `-1` undecodable, `-2` manifest-carrying (fail-closed: the hook grant would occupy
/// import slot 0 — IMPORTS.md §2.1, the `svm-run` rule), `-3` failed re-verification, `-4`
/// outside the bytecode subset.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn svm_mem_profile(
    ptr: *const u8,
    len: usize,
    l1_sets: u32,
    l1_ways: u32,
    l1_line: u32,
    l2_sets: u32,
    l2_ways: u32,
    l2_line: u32,
    page_size: u32,
) -> i32 {
    use std::sync::{Arc, Mutex};
    let put = |data: Vec<u8>| unsafe { stash(&mut *core::ptr::addr_of_mut!(MEMPROF), data) };
    // SAFETY: the host guarantees `[ptr, len)` is a live allocation it just filled.
    let bytes: &[u8] = unsafe { core::slice::from_raw_parts(ptr, len) };
    let Ok(m) = svm_encode::decode_module(bytes) else {
        put(Vec::new());
        return -1;
    };
    if !m.imports.is_empty() {
        put(Vec::new());
        return -2;
    }
    // Discover the handle the hook grant will mint (grants are deterministic; the run host below
    // grants the hook first, so a scratch first-grant yields the exact baked-in value).
    let handle = {
        let mut scratch = Host::new();
        scratch.grant_host_proc(Box::new(|_, _, _, _| Ok(vec![])))
    };
    let spec = svm_opt::instrument::MemHookSpec {
        type_id: svm_interp::cap_id::HOST_PROC,
        handle,
    };
    let (im, _stats) = svm_opt::instrument::instrument_mem_hooks(&m, spec);
    if let Err(e) = svm_verify::verify_module(&im) {
        put(format!("{e:?}").into_bytes()); // the error text, for diagnostics
        return -3;
    }
    let d = svm_dap::models::MemModelCfg::default();
    let dim = |v: u32, def: u64| if v == 0 { def } else { v as u64 };
    let cfg = svm_dap::models::MemModelCfg {
        l1: svm_dap::models::CacheCfg {
            sets: dim(l1_sets, d.l1.sets),
            ways: if l1_ways == 0 {
                d.l1.ways
            } else {
                l1_ways as usize
            },
            line: dim(l1_line, d.l1.line),
        },
        l2: svm_dap::models::CacheCfg {
            sets: dim(l2_sets, d.l2.sets),
            ways: if l2_ways == 0 {
                d.l2.ways
            } else {
                l2_ways as usize
            },
            line: dim(l2_line, d.l2.line),
        },
        page_size: dim(page_size, d.page_size),
    };
    let model = Arc::new(Mutex::new(svm_dap::models::MemModel::new(cfg)));
    model
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .set_snapshots(false); // run-mode: forward-only clock, no seeks
    let feed = Arc::clone(&model);
    let mut host = Host::new();
    let mut n: u64 = 0; // the profile's event clock (hooks carry none; forward-only)
    let h = host.grant_host_proc(Box::new(move |op, args, _mem, _| {
        if let Some(ev) = decode_mem_event(op, args) {
            n += 1;
            feed.lock()
                .unwrap_or_else(|e| e.into_inner())
                .observe(n, 0, ev);
        }
        Ok(vec![])
    }));
    debug_assert_eq!(h, handle, "the hook grant is the first grant");
    let mut fuel = u64::MAX;
    let res = bytecode::compile_and_run_with_host(&im, 0, &[], &mut fuel, &mut host);
    let status = match &res {
        None => {
            put(Vec::new());
            return -4;
        }
        Some(Ok(_)) => 0,
        Some(Err(_)) => 1,
    };
    let json = model
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .stats_json()
        .to_string();
    put(json.into_bytes());
    status
}

/// Pointer / length of the most recent [`svm_mem_profile`] stats JSON.
#[no_mangle]
pub extern "C" fn svm_mem_profile_stats_ptr() -> *const u8 {
    unsafe { (*core::ptr::addr_of!(MEMPROF)).0 }
}
#[no_mangle]
pub extern "C" fn svm_mem_profile_stats_len() -> usize {
    unsafe { (*core::ptr::addr_of!(MEMPROF)).1 }
}

// ---- wasm-JIT tier (BROWSER.md § "wasm-JIT tier"), slice 2: emit + expose to the JS host ---------

/// Trap codes the emitted wasm delivers through its `env.trap` import — re-exported from the
/// emitter so the JS linker names them without hard-coding. `f{i}` calls `env.trap(code)` then
/// `unreachable`; because the JS host calls the emitted function **directly** (not via this
/// cdylib), that `unreachable` surfaces as a catchable `RuntimeError` at the JS boundary — the host
/// reads the code it recorded to classify the trap (exactly the slice-1 differential model).
pub const WASMJIT_TRAP_OUT_OF_FUEL: i32 = svm_wasm_jit::TRAP_OUT_OF_FUEL;
pub const WASMJIT_TRAP_MEMORY_FAULT: i32 = svm_wasm_jit::TRAP_MEMORY_FAULT;

/// Compile the encoded SVM module at `[mod_ptr, mod_len)` to a **WebAssembly module** (the wasm-JIT
/// tier). Returns `1` and stashes the emitted wasm bytes when the whole module is JIT-eligible (its
/// every function is in the emitter's v1 subset), or `0` when it is not — the fail-closed signal for
/// the host to keep running the module on the bytecode interpreter (`svm_run`). Read the bytes via
/// [`svm_wasmjit_ptr`] + [`svm_wasmjit_len`] before the next call.
///
/// The emitted module imports `env.memory` + `env.trap`; the host instantiates it against **this
/// cdylib's own linear memory** (its exported `memory`) so an `svm_alloc`ed window/`env` cell is
/// addressable in both, then calls the exported `f{i}(win, env, ...args)` directly. `size_log2` of
/// the module's declared memory bakes the guard bound into the emitted confinement, so the host
/// need only size the window ≥ `1 << size_log2`.
#[no_mangle]
pub extern "C" fn svm_wasmjit_compile(mod_ptr: *const u8, mod_len: usize) -> i32 {
    // The browser default: entry func 0, shared memory (`shared = 1`) — the emitted module links
    // against this cdylib's shared linear memory (the threads build). See [`svm_wasmjit_compile_full`].
    svm_wasmjit_compile_full(mod_ptr, mod_len, 0, 1)
}

/// [`svm_wasmjit_compile`] with the JIT entry and memory-shared flag exposed. `entry` is the SVM
/// function the host will call (the emitted export is `f{entry}`; the cross-engine bench runs an
/// arbitrary kernel, not always func 0). `shared` selects the `env.memory` import's shared flag —
/// `1` for the browser threads build (shared memory), `0` for a plain cdylib (the bench, whose
/// exported memory is non-shared); it must match the memory the host links against.
#[no_mangle]
pub extern "C" fn svm_wasmjit_compile_full(
    mod_ptr: *const u8,
    mod_len: usize,
    entry: u32,
    shared: i32,
) -> i32 {
    let bytes: &[u8] = if mod_ptr.is_null() || mod_len == 0 {
        &[]
    } else {
        // SAFETY: the host guarantees `[mod_ptr, mod_len)` is a live allocation it just filled.
        unsafe { core::slice::from_raw_parts(mod_ptr, mod_len) }
    };
    let Ok(m) = svm_encode::decode_module(bytes) else {
        return 0;
    };
    // This FFI services cross-tier leaves on a *throwaway* window ([`svm_wasmjit_call_interp`], via
    // `bytecode::compile_and_run` with no window), so it can only accept guests whose cross-tier
    // callees are memory-free leaves — the `mixed_ok` condition. `compile_jit`'s wasm-driven path can
    // *also* emit reactor guests with memory/cap-touching cross-tier callees over a *shared* window,
    // which this callback can't service — so gate on `mixed_ok` first and fail closed otherwise. For a
    // `mixed_ok` guest the emitted wasm is byte-identical to the old `compile_module_mixed_entry`
    // (reactor's cross-tier set collapses to exactly the memory-free leaves). Emits `f{entry}`; a
    // fully-in-subset guest is the special case with no leaves.
    if !svm_wasm_jit::analyze_from(&m, entry).mixed_ok {
        return 0;
    }
    match svm_wasm_jit::compile_jit(&m, svm_wasm_jit::Shape::Batch { entry }, shared != 0) {
        Ok(svm_wasm_jit::Artifact {
            wasm,
            drive: svm_wasm_jit::DriveMode::WasmDriven { .. },
            ..
        }) => {
            // SAFETY: single-reader stash on the main thread, like the `svm_parse` accessors.
            unsafe { stash(&mut *core::ptr::addr_of_mut!(WASMJIT), wasm) };
            // Keep the decoded module for the cross-tier callback (it runs an interp leaf).
            unsafe { *core::ptr::addr_of_mut!(WASMJIT_MOD) = Some(m) };
            1
        }
        _ => 0,
    }
}

/// Emit a **§22 Model B2** unit: like [`svm_wasmjit_compile`] but the module *imports* one shared
/// `env.__indirect_function_table` (sized `1 << table_log2` = the `Jit` grant's reservation) instead
/// of declaring a private one, and populates no slots — the JS host owns the shared
/// `WebAssembly.Table`, writing each unit's `f0` funcref into its slot on `install` (`table.set`) and
/// nulling it on `uninstall`. So an installed unit is a funcref another instance's `call_indirect`
/// reaches through the one shared table (`svm_wasm_jit::compile_module_b2`; the native differential is
/// `crates/svm-wasm-jit/tests/b2_install.rs`). Emitted bytes are stashed exactly like
/// [`svm_wasmjit_compile`] (read via [`svm_wasmjit_ptr`]/[`svm_wasmjit_len`]); `0` if the module is
/// outside the emitter subset. `shared` matches the linked memory's shared flag as in
/// [`svm_wasmjit_compile_full`].
#[no_mangle]
pub extern "C" fn svm_wasmjit_compile_b2(
    mod_ptr: *const u8,
    mod_len: usize,
    table_log2: u32,
    shared: i32,
) -> i32 {
    let bytes: &[u8] = if mod_ptr.is_null() || mod_len == 0 {
        &[]
    } else {
        // SAFETY: the host guarantees `[mod_ptr, mod_len)` is a live allocation it just filled.
        unsafe { core::slice::from_raw_parts(mod_ptr, mod_len) }
    };
    let Ok(m) = svm_encode::decode_module(bytes) else {
        return 0;
    };
    match svm_wasm_jit::compile_module_b2(&m, shared != 0, table_log2) {
        Ok(wasm) => {
            // SAFETY: single-reader stash on the main thread, like the `svm_parse` accessors.
            unsafe { stash(&mut *core::ptr::addr_of_mut!(WASMJIT), wasm) };
            // Keep the decoded module for `svm_wasmjit_init_window` (its data segments) + call_interp.
            unsafe { *core::ptr::addr_of_mut!(WASMJIT_MOD) = Some(m) };
            1
        }
        Err(_) => 0,
    }
}

/// Pointer / length of the most recent [`svm_wasmjit_compile`] output (emitted wasm bytes).
#[no_mangle]
pub extern "C" fn svm_wasmjit_ptr() -> *const u8 {
    unsafe { (*core::ptr::addr_of!(WASMJIT)).0 }
}
#[no_mangle]
pub extern "C" fn svm_wasmjit_len() -> usize {
    unsafe { (*core::ptr::addr_of!(WASMJIT)).1 }
}
/// The stashed emitted-wasm bytes (same cdylib-managed lifetime as `OUT`/`ERR`).
static mut WASMJIT: (*mut u8, usize) = (core::ptr::null_mut(), 0);
/// The decoded module of the most recent [`svm_wasmjit_compile`], for [`svm_wasmjit_call_interp`].
static mut WASMJIT_MOD: Option<svm_ir::Module> = None;

/// Bytes the host must allocate for the `env` cell — the fuel counter plus the cross-tier scratch
/// (`env.call_interp` marshals its i64 arg/result slots there). The JS linker sizes the `env`
/// allocation with this.
#[no_mangle]
pub extern "C" fn svm_wasmjit_env_bytes() -> usize {
    svm_wasm_jit::ENV_CELL_BYTES
}

/// Materialize the most recent [`svm_wasmjit_compile`] module's **data segments** into the window at
/// `[win_ptr, win_ptr + win_size)` — the emitted code only loads/stores, so the host must lay the
/// module's initialized data into the window before running `f{entry}` (exactly what the
/// interpreter's window init does). Writes each `data.bytes` at `data.offset`, clamped to the
/// window. Call once, after allocating the window, before the first run.
#[no_mangle]
pub extern "C" fn svm_wasmjit_init_window(win_ptr: *mut u8, win_size: usize) {
    // SAFETY: set by the preceding `svm_wasmjit_compile`; single-threaded page use.
    let Some(m) = (unsafe { (*core::ptr::addr_of!(WASMJIT_MOD)).as_ref() }) else {
        return;
    };
    for seg in &m.data {
        let off = seg.offset as usize;
        let end = off.saturating_add(seg.bytes.len());
        if end > win_size {
            continue; // a segment past the window is the host's error; skip rather than corrupt
        }
        // SAFETY: `[win_ptr, win_ptr+win_size)` is a live host allocation; `[off, end) ⊆ window`.
        unsafe {
            core::ptr::copy_nonoverlapping(seg.bytes.as_ptr(), win_ptr.add(off), seg.bytes.len());
        }
    }
}

/// Service one cross-tier call (BROWSER.md § "wasm-JIT tier", slice 3c). The emitted mixed-tier
/// module calls this (via its `env.call_interp` import, relayed by the JS host) when JITted code
/// reaches an **interp leaf**: `func` is the SVM function index, `args_ptr` points at its i64 arg
/// slots in linear memory. Runs the leaf on the **bytecode interpreter** (the leaf is memory-free by
/// construction — see the emitter's `interp_leaf`), writes its i64 result slots back over the same
/// `args_ptr`, and returns `0`; on a trap returns `1` so the JS host throws (unwinding the emitted
/// wasm to the top-level `f0` caller — the slice-1/2 trap model).
#[no_mangle]
pub extern "C" fn svm_wasmjit_call_interp(func: u32, args_ptr: *mut u8) -> i32 {
    // SAFETY: `WASMJIT_MOD` is set by the preceding `svm_wasmjit_compile`; single-threaded page use.
    let Some(m) = (unsafe { (*core::ptr::addr_of!(WASMJIT_MOD)).as_ref() }) else {
        return 1;
    };
    let Some(callee) = m.funcs.get(func as usize) else {
        return 1;
    };
    let nresults = callee.results.len();
    // SAFETY: the host guarantees `args_ptr` addresses the signature's full slot span (the env
    // scratch, sized by `svm_wasmjit_env_bytes`).
    let args: Vec<Value> = callee
        .params
        .iter()
        .zip(slot_offs(&callee.params))
        .map(|(t, o)| unsafe { read_slot_value(*t, args_ptr, o) })
        .collect();
    let mut fuel = u64::MAX;
    match bytecode::compile_and_run(m, func, &args, &mut fuel) {
        Some(Ok(vals)) if vals.len() == nresults => {
            let offs = slot_offs(&callee.results);
            for (i, v) in vals.iter().enumerate() {
                // SAFETY: `args_ptr + offs[i]` is within the env scratch (result slots overlay args).
                if !unsafe { write_slot_value(v, args_ptr, offs[i]) } {
                    return 1; // ref/cap result: not marshallable through the scratch
                }
            }
            0
        }
        _ => 1, // trap, unsupported, or arity mismatch → the host throws
    }
}

/// Run `m`'s function 0 under a deterministic **3-cap powerbox** — `Stream(Out)` (type 0), `Exit`
/// (type 1), and a host-fn (type 13), granted in that order — so the §7 reflection ops
/// `cap.self.count` / `cap.self.get` see a fixed, known capability table. Passes `arg` only if the
/// entry takes one. Returns `(status, i64-widened value)`. Shared by [`svm_run_reflect`] and
/// `gencorpus`.
pub fn reflect_exec(m: &svm_ir::Module, arg: i64) -> (i32, i64) {
    let mut host = Host::new();
    let _ = host.grant_stream(StreamRole::Out); // handle 0, type_id 0
    let _ = host.grant_exit(); // handle 1, type_id 1
    let _ = host.grant_host_proc(Box::new(|_op, _args, _mem, _| Ok(vec![0]))); // handle 2, type_id 13
    let arity = m.funcs.first().map_or(0, |f| f.params.len());
    let args: Vec<Value> = if arity >= 1 {
        vec![Value::I32(arg as i32)]
    } else {
        Vec::new()
    };
    let mut fuel = 1_000_000u64;
    match bytecode::compile_and_run_with_host(m, 0, &args, &mut fuel, &mut host) {
        None => (STATUS_UNSUPPORTED, 0),
        Some(Err(_)) => (STATUS_TRAP, 0),
        Some(Ok(vals)) => match vals.first() {
            Some(Value::I64(x)) => (STATUS_OK, *x),
            Some(Value::I32(x)) => (STATUS_OK, *x as i64),
            _ => (STATUS_BAD_RESULT, 0),
        },
    }
}

// The **canonical** §22 `compile_linked` symbol-table wire form (mirrors `svm-run::decode_symbol_table`,
// DESIGN.md §22): a LEB128 stream `count`, then per entry `name` (uleb len + UTF-8 bytes), a `kind`
// byte, and its payload — `0` = `Slot(uleb)` (a shared `call_indirect` table slot: the *dynamic*-link
// case a guest loader uses to bind a submitted unit's imports to functions of the host program it runs
// inside — e.g. the JACL self-hosted compiler-guest binding a staged macro's `call.sym` imports to its
// own `jaclrt` runtime funcs), `1` = `Cap(uleb type_id, uleb op)` (a host capability). Empty bytes ⇒
// the closed-blob `compile` op (no bindings), so a unit with imports fails closed. This must match the
// producer (the on-ramp/`svm-llvm` guest loader) byte-for-byte, so it is NOT a browser-private form.

/// A minimal fail-closed LEB128 cursor for [`decode_symtab`] (never panics / over-reads).
struct SymCursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl SymCursor<'_> {
    fn byte(&mut self) -> Option<u8> {
        let b = *self.bytes.get(self.pos)?;
        self.pos += 1;
        Some(b)
    }
    /// Unsigned LEB128 → `u64` (max 10 bytes; rejects overflow / truncation).
    fn uleb(&mut self) -> Option<u64> {
        let (mut result, mut shift) = (0u64, 0u32);
        loop {
            let b = self.byte()?;
            if shift >= 64 || (shift == 63 && b & 0x7f > 1) {
                return None;
            }
            result |= ((b & 0x7f) as u64) << shift;
            if b & 0x80 == 0 {
                return Some(result);
            }
            shift += 7;
        }
    }
    fn u32(&mut self) -> Option<u32> {
        u32::try_from(self.uleb()?).ok()
    }
    fn string(&mut self) -> Option<String> {
        let n = usize::try_from(self.uleb()?).ok()?;
        let end = self.pos.checked_add(n)?;
        let s = core::str::from_utf8(self.bytes.get(self.pos..end)?).ok()?;
        self.pos = end;
        Some(s.to_string())
    }
}

/// Build a `compile_linked` symbol table (canonical wire form; used by the reference `Jit` tests).
fn encode_symtab(entries: &[(&str, svm_ir::Resolved)]) -> Vec<u8> {
    fn uleb(out: &mut Vec<u8>, mut v: u64) {
        loop {
            let b = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                out.push(b);
                break;
            }
            out.push(b | 0x80);
        }
    }
    let mut out = Vec::new();
    uleb(&mut out, entries.len() as u64);
    for (name, r) in entries {
        uleb(&mut out, name.len() as u64);
        out.extend_from_slice(name.as_bytes());
        match r {
            svm_ir::Resolved::Slot(slot) => {
                out.push(0);
                uleb(&mut out, *slot as u64);
            }
            svm_ir::Resolved::Cap(cap) => {
                out.push(1);
                uleb(&mut out, cap.type_id as u64);
                uleb(&mut out, cap.op as u64);
            }
            svm_ir::Resolved::Func(_) => {
                unreachable!("Func is not deliverable via the symbol table")
            }
        }
    }
    out
}

/// Decode a canonical `compile_linked` symbol table; `None` (fail-closed) on any malformation.
fn decode_symtab(bytes: &[u8]) -> Option<Vec<(String, svm_ir::Resolved)>> {
    // The closed-blob `compile` op passes no table (`&[]`) — the empty table (resolves nothing).
    if bytes.is_empty() {
        return Some(Vec::new());
    }
    let mut c = SymCursor { bytes, pos: 0 };
    let count = c.uleb()?;
    let mut out = Vec::new();
    for _ in 0..count {
        let name = c.string()?;
        let resolved = match c.byte()? {
            0 => svm_ir::Resolved::Slot(c.u32()?),
            1 => svm_ir::Resolved::Cap(svm_ir::ResolvedCap {
                type_id: c.u32()?,
                op: c.u32()?,
            }),
            _ => return None, // unknown kind
        };
        out.push((name, resolved));
    }
    // Trailing bytes ⇒ a length mismatch — reject rather than silently ignore (fail-closed).
    (c.pos == bytes.len()).then_some(out)
}

/// The browser's [`svm_interp::JitValidator`] — the §22 security hinge for the guest-driven `Jit`
/// cap: decode the symbol table → `decode_module` (fail-closed) → resolve named imports against the
/// table (`Slot`/`Cap`) → `verify_module` (the escape-freedom gate) → the memory-match precondition →
/// reject data segments and threads/futex ops. A pure-Rust replica of `svm-run`'s canonical validator
/// (same symtab wire form), so it builds for wasm with no Cranelift dep.
///
/// **Fibers are admitted** (#845 — the §22 renegotiated 2026-07-30 split, matching the canonical
/// gate in `svm-run`): `cont.*`/`suspend` switch stacks within the domain on the caller's thread,
/// so a unit that runs its own scheduler to completion never parks across the synchronous invoke.
/// *Emitted* execution of a fiber-using unit stays fail-closed with no gate here: `compile_jit`'s
/// `reachable_concurrency` guard never yields `WasmDriven` for one, so both wasm emitters return
/// `None` and the invoke runs on the interpreter (whose nested eval services the fiber ops).
fn browser_jit_validator(
    bytes: &[u8],
    mem_log2: Option<u8>,
    symtab: &[u8],
) -> Result<std::sync::Arc<[svm_ir::Func]>, i64> {
    const EINVAL: i64 = -22;
    let Some(table) = decode_symtab(symtab) else {
        return Err(EINVAL);
    };
    let Ok(m) = svm_encode::decode_module(bytes) else {
        return Err(EINVAL);
    };
    // Bind named imports via the table (a Slot → `call_indirect`, a Cap → `cap.call`); an unresolved
    // import ⇒ fail closed (the module is re-verified after the rewrite).
    let resolve = |name: &str| table.iter().find(|(n, _)| n == name).map(|(_, r)| *r);
    let Ok(m) = svm_ir::resolve_imports_with(&m, resolve) else {
        return Err(EINVAL);
    };
    if svm_verify::verify_module(&m).is_err() {
        return Err(EINVAL);
    }
    if m.memory.map(|mc| mc.size_log2) != mem_log2 {
        return Err(EINVAL); // declared memory must equal the parent window
    }
    if !m.data.is_empty()
        || m.funcs.is_empty()
        || m.funcs.iter().any(|f| f.uses_threads() || f.uses_futex())
    {
        return Err(EINVAL);
    }
    Ok(m.funcs.into())
}

/// The wasm-JIT emitter the runtime-`Jit.compile` path installs ([`svm_par_powerbox_jit_runtime`],
/// via [`Host::set_jit_wasm_emitter`]): emit a **validated closed unit**'s entry as `f0(win, env,
/// args…)` against **shared** memory (the browser's `SharedArrayBuffer`), or `None` if it is outside
/// the emitter subset — then `invoke` runs on the interpreter, fail-closed. A bare `fn`
/// (`svm_interp::JitWasmEmitter`), so the core stores only the opaque bytes. The unit was already
/// decode+verify+precondition-gated by [`browser_jit_validator`]; this re-decodes those same bytes.
fn browser_jit_wasm_emitter(blob: &[u8]) -> Option<Vec<u8>> {
    let m = svm_encode::decode_module(blob).ok()?;
    if par_jit_b2() {
        // §22 Model B2 cross-Worker: emit a unit that imports the shared reserved funcref table, so its
        // `call_indirect` dispatches (at native wasm speed) to units installed in the per-Worker table
        // mirror. Whole-module in-subset only; otherwise fail-closed to the interpreter (`.ok()`).
        return svm_wasm_jit::compile_module_b2(&m, true, PAR_JIT_TABLE_LOG2 as u32).ok();
    }
    match svm_wasm_jit::compile_jit(&m, svm_wasm_jit::Shape::Batch { entry: 0 }, true) {
        Ok(svm_wasm_jit::Artifact {
            wasm,
            drive: svm_wasm_jit::DriveMode::WasmDriven { .. },
            ..
        }) => Some(wasm),
        // Interp-driven / unsupported ⇒ nothing to run as `f0`; the unit invokes on the interpreter.
        _ => None,
    }
}

/// A unit the guest-JIT path installs and calls: `service(a, b) = a*b + 100`. Host-compiled (the
/// bytecode entry builds memory from the module, so no in-guest blob seeding is needed).
const JIT_SERVICE: &str = r#"memory 16
func (i32, i32) -> (i32) {
block 0 (v0: i32, v1: i32) {
  v2 = i32.mul v0 v1
  v3 = i32.const 100
  v4 = i32.add v2 v3
  return v4
  }
}
"#;

/// Run `m`'s function 0 with a **`Jit`** cap (iface 11) and a host-compiled [`JIT_SERVICE`] unit:
/// the guest receives `(jit_handle, code_handle, a, b)`, `install`s the unit into its dispatch table
/// (op 3), then `call_indirect`s it — guest-driven code loading, **interpreted** (the bytecode engine
/// lowers the submitted unit to bytecode; no native backend). `a=6, b=7`. Returns `(status, value)`.
pub fn jit_exec(m: &svm_ir::Module) -> (i32, i64) {
    let service = match svm_text::parse_module(JIT_SERVICE) {
        Ok(s) => svm_encode::encode_module(&s),
        Err(_) => return (STATUS_BAD_RESULT, 0),
    };
    let mut host = Host::new();
    let jit = host.grant_jit_with_table(m.memory.map(|mc| mc.size_log2), 4); // 2^4 = 16-slot table
    host.set_jit_validator(browser_jit_validator);
    let code = match host.jit_compile(jit, &service) {
        Ok(Ok(c)) => c.handle,
        _ => return (STATUS_TRAP, 0),
    };
    let args = [
        Value::I32(jit),
        Value::I32(code),
        Value::I32(6),
        Value::I32(7),
    ];
    let mut fuel = 50_000_000u64;
    match bytecode::compile_and_run_with_host(m, 0, &args, &mut fuel, &mut host) {
        None => (STATUS_UNSUPPORTED, 0),
        Some(Err(_)) => (STATUS_TRAP, 0),
        Some(Ok(vals)) => match vals.first() {
            Some(Value::I64(x)) => (STATUS_OK, *x),
            Some(Value::I32(x)) => (STATUS_OK, *x as i64),
            _ => (STATUS_BAD_RESULT, 0),
        },
    }
}

/// A separately-compiled unit with a **named import** `"clock"`, resolved by `compile_linked`'s
/// symbol table to a host capability — a plugin reaching a host service by name. `clock.now` first
/// reads `0`, so the unit returns `0 + 777 = 777` once linked. (Declares `memory 16` to satisfy the
/// memory-match precondition against the parent window.)
const DL_UNIT: &str = r#"memory 16
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = call.sym "clock" () -> (i64) v0 ()
  v2 = i64.const 777
  v3 = i64.add v1 v2
  return v3
  }
}
"#;

/// Run `m`'s function 0 with a `Jit` cap, a `Clock` cap, and a host-`compile_linked` [`DL_UNIT`]:
/// **dynamic linking** — the unit's named import `"clock"` is bound (via the symbol table) to the
/// `Clock` capability `(type_id 2, op 0)` before verify, lowering `call.sym "clock"` to a real
/// `cap.call 2 0`. The guest receives `(jit, code, clock)`, installs the unit and `call_indirect`s it
/// passing the clock handle → `777`. With `link == false` the symbol table is empty, so the import is
/// unresolved and `compile_linked` fails closed (`STATUS_TRAP`). Returns `(status, value)`.
pub fn dynlink_exec(m: &svm_ir::Module, link: bool) -> (i32, i64) {
    let unit = match svm_text::parse_module(DL_UNIT) {
        Ok(u) => svm_encode::encode_module(&u),
        Err(_) => return (STATUS_BAD_RESULT, 0),
    };
    let mut host = Host::new();
    let jit = host.grant_jit_with_table(m.memory.map(|mc| mc.size_log2), 4);
    host.set_jit_validator(browser_jit_validator);
    let clock = host.grant_clock();
    // Bind "clock" → the Clock cap (iface 2, op 0) iff linking; otherwise an empty table (fail-closed).
    let symtab = if link {
        encode_symtab(&[(
            "clock",
            svm_ir::Resolved::Cap(svm_ir::ResolvedCap { type_id: 2, op: 0 }),
        )])
    } else {
        Vec::new()
    };
    let code = match host.jit_compile_linked(jit, &unit, &symtab) {
        Ok(Ok(c)) => c.handle,
        _ => return (STATUS_TRAP, 0), // unresolved import ⇒ compile_linked fails closed
    };
    let args = [Value::I32(jit), Value::I32(code), Value::I32(clock)];
    let mut fuel = 50_000_000u64;
    match bytecode::compile_and_run_with_host(m, 0, &args, &mut fuel, &mut host) {
        None => (STATUS_UNSUPPORTED, 0),
        Some(Err(_)) => (STATUS_TRAP, 0),
        Some(Ok(vals)) => match vals.first() {
            Some(Value::I64(x)) => (STATUS_OK, *x),
            Some(Value::I32(x)) => (STATUS_OK, *x as i64),
            _ => (STATUS_BAD_RESULT, 0),
        },
    }
}

/// Decode the module at `[mod_ptr, mod_len)` and run function 0 with a dynamically-**linked** unit
/// (see [`dynlink_exec`]); `link != 0` binds the unit's `"clock"` import, `0` leaves it unresolved
/// (fail-closed). Returns the guest's `i64` result; sets [`LAST_STATUS`].
#[no_mangle]
pub extern "C" fn svm_run_dynlink(mod_ptr: *const u8, mod_len: usize, link: i32) -> i64 {
    let set = |s: i32| unsafe { LAST_STATUS = s };
    // SAFETY: the host guarantees `[mod_ptr, mod_len)` is a live `svm_alloc`ation it just filled.
    let bytes = unsafe { core::slice::from_raw_parts(mod_ptr, mod_len) };
    let m = match svm_encode::decode_module(bytes) {
        Ok(m) => m,
        Err(_) => {
            set(STATUS_DECODE_ERR);
            return 0;
        }
    };
    let (status, value) = dynlink_exec(&m, link != 0);
    set(status);
    value
}

/// Decode the module at `[mod_ptr, mod_len)` and run function 0 under the **guest-JIT** powerbox (see
/// [`jit_exec`]). Returns the guest's `i64` result; sets [`LAST_STATUS`].
#[no_mangle]
pub extern "C" fn svm_run_jit(mod_ptr: *const u8, mod_len: usize) -> i64 {
    let set = |s: i32| unsafe { LAST_STATUS = s };
    // SAFETY: the host guarantees `[mod_ptr, mod_len)` is a live `svm_alloc`ation it just filled.
    let bytes = unsafe { core::slice::from_raw_parts(mod_ptr, mod_len) };
    let m = match svm_encode::decode_module(bytes) {
        Ok(m) => m,
        Err(_) => {
            set(STATUS_DECODE_ERR);
            return 0;
        }
    };
    let (status, value) = jit_exec(&m);
    set(status);
    value
}

/// Run an **already-instrumented** (durability-transformed) module's function 0 over a durable
/// `window` (its low bytes carry the state word `NORMAL`/`UNWINDING`/`REWINDING` + the shadow region),
/// with a `Clock` cap seeded to `clock_v`. Single-vCPU / single-fiber freeze/thaw is *driven by the
/// transform's emitted IR* (DURABILITY.md §2) — the engine just runs it. Returns `(status, value,
/// final-window snapshot, clock_after)`. Shared by [`svm_run_durable`] and `gencorpus`.
pub fn durable_run(inst: &svm_ir::Module, window: &[u8], clock_v: i64) -> (i32, i64, Vec<u8>, i64) {
    let mut host = Host::new();
    host.set_durable(true);
    let clk = host.grant_clock();
    host.clock_ns = clock_v;
    let mut fuel = 1_000_000u64;
    match bytecode::compile_and_run_capture_reserved_with_host(
        inst,
        0,
        &[Value::I32(clk)],
        &mut fuel,
        window,
        17, // SIZE_LOG2 = 128 KiB ≥ the durable reserve
        &mut host,
    ) {
        None => (STATUS_UNSUPPORTED, 0, Vec::new(), host.clock_ns),
        Some((r, snap)) => {
            let (status, value) = match r {
                Err(_) => (STATUS_TRAP, 0),
                Ok(vals) => match vals.first() {
                    Some(Value::I64(x)) => (STATUS_OK, *x),
                    Some(Value::I32(x)) => (STATUS_OK, *x as i64),
                    _ => (STATUS_BAD_RESULT, 0),
                },
            };
            (status, value, snap, host.clock_ns)
        }
    }
}

/// Decode the **instrumented** module at `[mod_ptr, mod_len)`, run function 0 over the durable window
/// at `[init_ptr, init_len)` (the state word lives in those bytes) with the clock seeded to `clock`
/// (see [`durable_run`]). The final window image is captured to the snapshot slot
/// (`svm_snapshot_ptr`/`svm_snapshot_len`). Returns the guest's `i64` result; sets [`LAST_STATUS`].
#[no_mangle]
pub extern "C" fn svm_run_durable(
    mod_ptr: *const u8,
    mod_len: usize,
    init_ptr: *const u8,
    init_len: usize,
    clock: i64,
) -> i64 {
    let set = |s: i32| unsafe { LAST_STATUS = s };
    // SAFETY: the host guarantees both ranges are live `svm_alloc`ations it just filled.
    let bytes = unsafe { core::slice::from_raw_parts(mod_ptr, mod_len) };
    let window = unsafe { core::slice::from_raw_parts(init_ptr, init_len) };
    let m = match svm_encode::decode_module(bytes) {
        Ok(m) => m,
        Err(_) => {
            set(STATUS_DECODE_ERR);
            return 0;
        }
    };
    let (status, value, snap, _clk) = durable_run(&m, window, clock);
    set(status);
    // SAFETY: single-threaded wasm; read back only via the snapshot accessors.
    unsafe { stash(&mut *core::ptr::addr_of_mut!(SNAP), snap) };
    value
}

/// Run `m`'s function 0 with a host-granted **`SharedRegion`** (iface 4, 64 KiB) as its sole cap —
/// the §13 host-backed memory object a guest `map`s into its window (op 0), aliasing the same backing
/// at multiple offsets (the magic-ring-buffer primitive); op 2 `len`, op 3 `page_size`. Returns
/// `(status, i64-widened value)`. Shared by [`svm_run_region`] and `gencorpus`.
pub fn region_exec(m: &svm_ir::Module) -> (i32, i64) {
    let mut host = Host::new();
    let h = host.grant_shared_region(1 << 16); // 64 KiB, comfortably larger than any host page
    let mut fuel = 5_000_000u64;
    match bytecode::compile_and_run_with_host(m, 0, &[Value::I32(h)], &mut fuel, &mut host) {
        None => (STATUS_UNSUPPORTED, 0),
        Some(Err(_)) => (STATUS_TRAP, 0),
        Some(Ok(vals)) => match vals.first() {
            Some(Value::I64(x)) => (STATUS_OK, *x),
            Some(Value::I32(x)) => (STATUS_OK, *x as i64),
            _ => (STATUS_BAD_RESULT, 0),
        },
    }
}

/// Decode the module at `[mod_ptr, mod_len)` and run function 0 with a `SharedRegion` cap (see
/// [`region_exec`]). Returns the guest's `i64` result; sets [`LAST_STATUS`].
#[no_mangle]
pub extern "C" fn svm_run_region(mod_ptr: *const u8, mod_len: usize) -> i64 {
    let set = |s: i32| unsafe { LAST_STATUS = s };
    // SAFETY: the host guarantees `[mod_ptr, mod_len)` is a live `svm_alloc`ation it just filled.
    let bytes = unsafe { core::slice::from_raw_parts(mod_ptr, mod_len) };
    let m = match svm_encode::decode_module(bytes) {
        Ok(m) => m,
        Err(_) => {
            set(STATUS_DECODE_ERR);
            return 0;
        }
    };
    let (status, value) = region_exec(&m);
    set(status);
    value
}

/// Decode the module at `[mod_ptr, mod_len)` and run function 0 under a fixed 3-cap powerbox, so §7
/// reflection (`cap.self.count`/`get`) is deterministic (see [`reflect_exec`]). Returns the guest's
/// `i64` result; sets [`LAST_STATUS`].
#[no_mangle]
pub extern "C" fn svm_run_reflect(mod_ptr: *const u8, mod_len: usize, arg: i64) -> i64 {
    let set = |s: i32| unsafe { LAST_STATUS = s };
    // SAFETY: the host guarantees `[mod_ptr, mod_len)` is a live `svm_alloc`ation it just filled.
    let bytes = unsafe { core::slice::from_raw_parts(mod_ptr, mod_len) };
    let m = match svm_encode::decode_module(bytes) {
        Ok(m) => m,
        Err(_) => {
            set(STATUS_DECODE_ERR);
            return 0;
        }
    };
    let (status, value) = reflect_exec(&m, arg);
    set(status);
    value
}

/// Decode the module at `[mod_ptr, mod_len)` and run function 0 under the **nested-child** powerbox
/// (an `Instantiator` over `[0, 128 KiB)`; see [`instantiate_exec`]): function 0 may `instantiate`
/// confined child guests over sub-windows and `join` them. Returns the guest's `i64` result; sets
/// [`LAST_STATUS`].
#[no_mangle]
pub extern "C" fn svm_run_nested(mod_ptr: *const u8, mod_len: usize) -> i64 {
    let set = |s: i32| unsafe { LAST_STATUS = s };
    // SAFETY: the host guarantees `[mod_ptr, mod_len)` is a live `svm_alloc`ation it just filled.
    let bytes = unsafe { core::slice::from_raw_parts(mod_ptr, mod_len) };
    let m = match svm_encode::decode_module(bytes) {
        Ok(m) => m,
        Err(_) => {
            set(STATUS_DECODE_ERR);
            return 0;
        }
    };
    let (status, value) = instantiate_exec(&m);
    set(status);
    value
}

/// Self-contained powerbox probe (no host buffers, so usable via `wasmtime --invoke run_powerbox`):
/// run a greeting guest that writes 17 bytes to stdout, then an `exit(42)` guest, and return `17`
/// iff both the captured stdout length **and** the exit code are correct on this target — i.e. the
/// stream-write/capture and exit-trap paths work on wasm64. Returns `-1` on any mismatch.
#[no_mangle]
pub extern "C" fn run_powerbox() -> i64 {
    const HELLO: &str = r#"
memory 16
data 0 "hello, powerbox!\n"
func (i32, i32, i32) -> (i32) {
block 0 (v0: i32, v1: i32, v2: i32) {
  v3 = i64.const 0
  v4 = i64.const 17
  v5 = cap.call 0 1 (i64, i64) -> (i64) v0(v3, v4)
  v6 = i32.const 0
  return v6
  }
}
"#;
    const EXIT: &str = r#"
func (i32, i32, i32) -> (i32) {
block 0 (v0: i32, v1: i32, v2: i32) {
  v3 = i32.const 42
  cap.call 1 0 (i32) -> () v2(v3)
  v4 = i32.const 0
  return v4
  }
}
"#;
    let (Ok(hm), Ok(em)) = (svm_text::parse_module(HELLO), svm_text::parse_module(EXIT)) else {
        return -1;
    };
    let h = powerbox_exec(&hm, &[]);
    let e = powerbox_exec(&em, &[]);
    if h.status == STATUS_OK
        && h.stdout == b"hello, powerbox!\n"
        && e.status == STATUS_EXIT
        && e.exit_code == 42
    {
        h.stdout.len() as i64
    } else {
        -1
    }
}

/// Self-contained capture probe (seeds its own window, so usable via `wasmtime --invoke run_capture`):
/// run an in-place "add `arg` to each i64 word" guest over a 16-word window whose word 0 is `1000`,
/// with `arg = 7`, and return word 0 of the **captured final image** — `1007` iff seeding, the
/// in-place writes, and the snapshot all work on this target. Returns `-1` on any mismatch.
#[no_mangle]
pub extern "C" fn run_capture() -> i64 {
    const ADDK: &str = r#"
memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 0
  br 1(v0, v1)
}
block 1 (v2: i64, v3: i64) {
  v4 = i64.const 128
  v5 = i64.lt_u v3 v4
  br_if v5 2(v2, v3) 3()
}
block 2 (v6: i64, v7: i64) {
  v8 = i64.load v7
  v9 = i64.add v8 v6
  i64.store v7 v9
  v10 = i64.const 8
  v11 = i64.add v7 v10
  br 1(v6, v11)
}
block 3 () {
  v12 = i64.const 0
  v13 = i64.load v12
  return v13
  }
}
"#;
    let Ok(m) = svm_text::parse_module(ADDK) else {
        return -1;
    };
    // Seed 16 i64 words: word 0 = 1000, the rest 0.
    let mut init = [0u8; 128];
    init[..8].copy_from_slice(&1000i64.to_le_bytes());
    let out = capture_exec(&m, &init, 7);
    if out.status != STATUS_OK || out.snapshot.len() != 128 {
        return -1;
    }
    // Word 0 of the captured image should be 1000 + 7 = 1007.
    i64::from_le_bytes(out.snapshot[..8].try_into().unwrap())
}

/// Self-contained nested-child probe (so usable via `wasmtime --invoke run_instantiate`): a parent
/// `instantiate`s a confined child in a 4 KiB sub-window at 64 KiB, the child writes a marker into
/// the shared backing and returns 42, the parent joins and reads the marker back — returning
/// `42 * 1000 + 123 = 42123` iff confined child execution + the shared data plane work on this
/// target. Returns `-1` on any mismatch.
#[no_mangle]
pub extern "C" fn run_instantiate() -> i64 {
    const SHARED: &str = r#"memory 17
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const 1
  v2 = i64.const 65536
  v3 = i64.const 12
  v4 = i64.const 0
  v5 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)
  v6 = cap.call 6 1 (i32) -> (i64) v0 (v5)
  v7 = i64.const 65543
  v8 = i32.load8_u v7
  v9 = i64.extend_i32_u v8
  v10 = i64.const 1000
  v11 = i64.mul v6 v10
  v12 = i64.add v11 v9
  return v12
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 7
  v2 = i32.const 123
  i32.store8 v1 v2
  v3 = i64.const 42
  return v3
  }
}
"#;
    let Ok(m) = svm_text::parse_module(SHARED) else {
        return -1;
    };
    match instantiate_exec(&m) {
        (STATUS_OK, v) => v,
        _ => -1,
    }
}

/// Self-contained SIMD probe (`wasmtime --invoke run_simd`): splat 21 into an `i64x2`, add lanewise,
/// extract lane 0 → `42`. Returns `-1` on any mismatch.
#[no_mangle]
pub extern "C" fn run_simd() -> i64 {
    const S: &str = r#"
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64x2.splat v0
  v2 = i64x2.add v1 v1
  v3 = i64x2.extract_lane 0 v2
  return v3
  }
}
"#;
    let Ok(m) = svm_text::parse_module(S) else {
        return -1;
    };
    let mut fuel = u64::MAX;
    match bytecode::compile_and_run(&m, 0, &[Value::I64(21)], &mut fuel) {
        Some(Ok(vals)) => match vals.first() {
            Some(Value::I64(x)) => *x,
            _ => -1,
        },
        _ => -1,
    }
}

/// Self-contained durability probe (`wasmtime --invoke run_durable`): instrument a single-fiber
/// program that reads the clock twice (each an unwind point), run it NORMAL over a fresh durable
/// window with the clock seeded to 1000 → `1000 + 1001 = 2001`. Proves the freeze/thaw transform's
/// emitted IR runs on this target. Returns `-1` on any mismatch.
#[no_mangle]
pub extern "C" fn run_durable() -> i64 {
    const SRC: &str = r#"memory 17
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = cap.call 2 0 () -> (i64) v0 ()
  v2 = cap.call 2 0 () -> (i64) v0 ()
  v3 = i64.add v1 v2
  return v3
  }
}
"#;
    let Ok(m) = svm_text::parse_module(SRC) else {
        return -1;
    };
    let Ok(inst) = svm_durable::transform_module(&m) else {
        return -1;
    };
    let mut win = svm_durable::init_durable_window(1 << 17);
    svm_durable::write_state(&mut win, svm_durable::STATE_NORMAL);
    match durable_run(&inst, &win, 1000) {
        (STATUS_OK, v, _, _) => v,
        _ => -1,
    }
}

/// Self-contained dynamic-linking probe (`wasmtime --invoke run_dynlink`): a unit's named import
/// `"clock"` is resolved by `compile_linked`'s symbol table to the Clock cap; the guest installs and
/// calls it → `clock.now (0) + 777 = 777`. Returns `-1` on any mismatch.
#[no_mangle]
pub extern "C" fn run_dynlink() -> i64 {
    const G: &str = r#"memory 16
func (i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32) {
  v3 = i64.extend_i32_u v1
  v4 = cap.call 11 3 (i64) -> (i64) v0 (v3)
  v5 = i32.wrap_i64 v4
  v6 = call_indirect (i32) -> (i64) v5 (v2)
  return v6
  }
}
"#;
    let Ok(m) = svm_text::parse_module(G) else {
        return -1;
    };
    match dynlink_exec(&m, true) {
        (STATUS_OK, v) => v,
        _ => -1,
    }
}

/// Self-contained guest-JIT probe (`wasmtime --invoke run_jit`): a guest installs a host-compiled
/// unit (`a*b+100`) into its dispatch table and `call_indirect`s it with `(6, 7)` → `142`. Proves
/// guest-driven code loading (validated + interpreted, no native backend) works. `-1` on mismatch.
#[no_mangle]
pub extern "C" fn run_jit() -> i64 {
    const G: &str = r#"memory 16
func (i32, i32, i32, i32) -> (i32) {
block 0 (v0: i32, v1: i32, v2: i32, v3: i32) {
  v4 = i64.extend_i32_u v1
  v5 = cap.call 11 3 (i64) -> (i64) v0 (v4)
  v6 = i32.wrap_i64 v5
  v7 = call_indirect (i32, i32) -> (i32) v6 (v2, v3)
  return v7
  }
}
"#;
    let Ok(m) = svm_text::parse_module(G) else {
        return -1;
    };
    match jit_exec(&m) {
        (STATUS_OK, v) => v,
        _ => -1,
    }
}

/// Self-contained SharedRegion probe (`wasmtime --invoke run_region`): map a host region at two
/// window offsets, store a marker through one and load it through the other → `0x0123456789abcdef`
/// (`81985529216486895`) iff the mappings alias the same backing. Returns `-1` on any mismatch.
#[no_mangle]
pub extern "C" fn run_region() -> i64 {
    const R: &str = r#"memory 17
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = cap.call 4 3 () -> (i64) v0 ()
  v2 = i64.const 0
  v3 = i32.const 3
  v4 = cap.call 4 0 (i64, i64, i64, i32) -> (i64) v0 (v2, v2, v1, v3)
  v5 = cap.call 4 0 (i64, i64, i64, i32) -> (i64) v0 (v1, v2, v1, v3)
  v6 = i64.const 81985529216486895
  i64.store v2 v6
  v7 = i64.load v1
  return v7
  }
}
"#;
    let Ok(m) = svm_text::parse_module(R) else {
        return -1;
    };
    match region_exec(&m) {
        (STATUS_OK, v) => v,
        _ => -1,
    }
}

/// Self-contained reflection probe (`wasmtime --invoke run_reflect`): under the fixed 3-cap powerbox,
/// `cap.self.count` reports `3`. Returns `-1` on any mismatch.
#[no_mangle]
pub extern "C" fn run_reflect() -> i64 {
    const R: &str = r#"
func () -> (i32) {
block 0 () {
  v0 = cap.self.count
  return v0
  }
}
"#;
    let Ok(m) = svm_text::parse_module(R) else {
        return -1;
    };
    match reflect_exec(&m, 0) {
        (STATUS_OK, v) => v,
        _ => -1,
    }
}

/// Self-contained GC-roots probe (`wasmtime --invoke run_gcroots`): a `gc.roots` scan over an
/// activation holding the in-range constants `{4096, 5000}` (one duplicated, one out of range)
/// returns the root count `2`. Returns `-1` on any mismatch.
#[no_mangle]
pub extern "C" fn run_gcroots() -> i64 {
    const G: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  va = i64.const 4096
  vb = i64.const 5000
  vc = i64.const 5000
  vd = i64.const 9000
  vlo = i64.const 4096
  vhi = i64.const 8192
  vmask = i64.const -1
  vbuf = i64.const 0
  vcap = i64.const 64
  vt = gc.roots vlo vhi vmask vbuf vcap
  return vt
  }
}
"#;
    let Ok(m) = svm_text::parse_module(G) else {
        return -1;
    };
    let init = [0u8; 4096];
    match capture_exec(&m, &init, 0) {
        out if out.status == STATUS_OK => out.value,
        _ => -1,
    }
}

/// Self-contained tail-call probe (`wasmtime --invoke run_tailcall`): a tail-recursive factorial via
/// `return_call` (O(1) window reuse) returns `5! = 120`. Returns `-1` on any mismatch.
#[no_mangle]
pub extern "C" fn run_tailcall() -> i64 {
    const T: &str = r#"
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 1
  return_call 1(v0, v1)
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 1
  v3 = i64.lt_s v0 v2
  br_if v3 1(v1) 2(v0, v1)
}
block 1 (v4: i64) {
  return v4
}
block 2 (v5: i64, v6: i64) {
  v7 = i64.mul v6 v5
  v8 = i64.const -1
  v9 = i64.add v5 v8
  return_call 1(v9, v7)
  }
}
"#;
    let Ok(m) = svm_text::parse_module(T) else {
        return -1;
    };
    let mut fuel = u64::MAX;
    match bytecode::compile_and_run(&m, 0, &[Value::I64(5)], &mut fuel) {
        Some(Ok(vals)) => match vals.first() {
            Some(Value::I64(x)) => *x,
            _ => -1,
        },
        _ => -1,
    }
}

/// Self-contained fiber probe (`wasmtime --invoke run_fiber`): a §12 continuation (`cont.new`/
/// `cont.resume`) runs to completion, resumed with 7 and returning `7 + 100`. Returns `107` iff
/// cooperative continuation switching works on this target, else `-1`.
#[no_mangle]
pub extern "C" fn run_fiber() -> i64 {
    const FIB: &str = r#"
func () -> (i64) {
block 0 () {
  v0 = ref.func 1
  v1 = i64.const 0
  v2 = cont.new v0 v1
  v3 = i64.const 7
  v4, v5 = cont.resume v2 v3
  return v5
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  v0 = i64.const 100
  v1 = i64.add varg v0
  return v1
  }
}
"#;
    let Ok(m) = svm_text::parse_module(FIB) else {
        return -1;
    };
    let mut fuel = u64::MAX;
    match bytecode::compile_and_run(&m, 0, &[], &mut fuel) {
        Some(Ok(vals)) => match vals.first() {
            Some(Value::I64(x)) => *x,
            _ => -1,
        },
        _ => -1,
    }
}

/// Self-contained coroutine probe (`wasmtime --invoke run_coroutine`): a §14 coroutine confined to a
/// sub-window is resumed three times, yielding 100, 210, then returning 1019. Returns
/// `100 + 210 + 1019 + RETURNED*1_000_000 = 1001329` iff `spawn_coroutine`/`resume`/`yield` work on
/// this target, else `-1`.
#[no_mangle]
pub extern "C" fn run_coroutine() -> i64 {
    const CORO: &str = r#"memory 17
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const 1
  v2 = i64.const 65536
  v3 = i64.const 16
  v4 = i64.const 0
  v5 = cap.call 6 2 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)
  v6 = i64.const 0
  v7, v8 = cap.call 6 3 (i32, i64) -> (i32, i64) v0 (v5, v6)
  v9 = i64.const 10
  v10, v11 = cap.call 6 3 (i32, i64) -> (i32, i64) v0 (v5, v9)
  v12 = i64.const 20
  v13, v14 = cap.call 6 3 (i32, i64) -> (i32, i64) v0 (v5, v12)
  v15 = i64.add v8 v11
  v16 = i64.add v15 v14
  v17 = i64.extend_i32_s v13
  v18 = i64.const 1000000
  v19 = i64.mul v17 v18
  v20 = i64.add v16 v19
  return v20
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i32.wrap_i64 v0
  v2 = i64.const 0
  v3 = i32.const 7
  i32.store8 v2 v3
  v4 = i64.const 100
  v5 = cap.call 7 0 (i64) -> (i64) v1 (v4)
  v6 = i64.const 200
  v7 = i64.add v6 v5
  v8 = cap.call 7 0 (i64) -> (i64) v1 (v7)
  v9 = i64.const 999
  v10 = i64.add v9 v8
  return v10
  }
}
"#;
    let Ok(m) = svm_text::parse_module(CORO) else {
        return -1;
    };
    match instantiate_exec(&m) {
        (STATUS_OK, v) => v,
        _ => -1,
    }
}

/// Self-contained scalar-float probe (`wasmtime --invoke run_float`): reinterpret the f64 bits of
/// `4.0`, take `sqrt(|·|)`, and return the result's i64 bits — `4611686018427387904` (the bits of
/// `2.0`) iff f64 reinterpret/abs/sqrt round-trip bit-exactly on this target, else `-1`.
#[no_mangle]
pub extern "C" fn run_float() -> i64 {
    const SQRT: &str = r#"
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = f64.reinterpret_i64 v0
  v2 = f64.abs v1
  v3 = f64.sqrt v2
  v4 = i64.reinterpret_f64 v3
  return v4
  }
}
"#;
    let Ok(m) = svm_text::parse_module(SQRT) else {
        return -1;
    };
    let mut fuel = u64::MAX;
    // arg = bits(4.0) = 0x4010000000000000; sqrt(4.0) = 2.0 = bits 0x4000000000000000.
    let arg = 0x4010000000000000u64 as i64;
    match bytecode::compile_and_run(&m, 0, &[Value::I64(arg)], &mut fuel) {
        Some(Ok(vals)) => match vals.first() {
            Some(Value::I64(x)) => *x,
            _ => -1,
        },
        _ => -1,
    }
}

// ---- live host imports: bind capabilities to real host functions ----------------------------
//
// Everything above keeps the cdylib import-free by buffering I/O. This (feature-gated) entry instead
// bridges guest capabilities to **real wasm imports**, so a guest's writes reach the live host
// console *as they happen* and the clock reads real host time. The seam is `Host::grant_host_proc`
// (iface 13) — the designed extension point: a closure supplies the capability's semantics, here by
// calling out to the imported host function. The guest sees only a masked, type-checked handle.

#[cfg(feature = "live")]
pub mod live {
    use super::*;

    // The host functions the embedder must supply (module `svm_host`). `host_write` receives a
    // pointer into *this module's* linear memory (the bytes the guest wrote, copied out of its
    // window into a Rust buffer that lives on the wasm heap), so JS reads them as
    // `new Uint8Array(memory.buffer, ptr, len)`. `host_now_ns` returns real host time.
    #[link(wasm_import_module = "svm_host")]
    extern "C" {
        /// `host_write(stream, ptr, len)` — `stream` 0 = stdout, 1 = stderr.
        fn host_write(stream: i32, ptr: *const u8, len: usize);
        /// `host_now_ns() -> i64` — host wall/monotonic clock, nanoseconds.
        fn host_now_ns() -> i64;
    }

    const EFAULT: i64 = -14;
    const EINVAL: i64 = -22;

    /// Decode the module at `[mod_ptr, mod_len)` and run function 0 with a **host-backed** powerbox:
    /// `(console, clock)` capabilities (both iface `HOST_PROC` = 13) bridged to the imports above.
    /// The guest calls `cap.call 13 1 (i64,i64,i64) -> (i64) v<console>(stream, ptr, len)` to write
    /// live, and `cap.call 13 0 () -> (i64) v<clock>()` to read the host clock. Returns the guest's
    /// `i64` result; sets [`LAST_STATUS`].
    #[no_mangle]
    pub extern "C" fn svm_run_live(mod_ptr: *const u8, mod_len: usize) -> i64 {
        // SAFETY: the host guarantees `[mod_ptr, mod_len)` is a live `svm_alloc`ation it just filled.
        let bytes = unsafe { core::slice::from_raw_parts(mod_ptr, mod_len) };
        let set = |s: i32| unsafe { LAST_STATUS = s };
        let m = match svm_encode::decode_module(bytes) {
            Ok(m) => m,
            Err(_) => {
                set(STATUS_DECODE_ERR);
                return 0;
            }
        };
        let mut host = Host::new();
        // console (param 1): op 1 = write(stream, ptr, len) → reads the guest window, forwards live.
        let console: HostProc = Box::new(|op, args, mem, _| {
            if op != 1 {
                return Ok(vec![EINVAL]);
            }
            let (Some(&stream), Some(&ptr), Some(&n)) = (args.first(), args.get(1), args.get(2))
            else {
                return Ok(vec![EINVAL]);
            };
            let Some(m) = mem else {
                return Ok(vec![EFAULT]);
            };
            match m.read_bytes(ptr as u64, n as u64) {
                // The copied bytes live on this module's wasm heap; hand their pointer to the host.
                Some(buf) => {
                    unsafe { host_write(stream as i32, buf.as_ptr(), buf.len()) };
                    Ok(vec![n])
                }
                None => Ok(vec![EFAULT]),
            }
        });
        // clock (param 2): op 0 = now() → real host time.
        let clock: HostProc = Box::new(|op, _args, _mem, _| {
            if op != 0 {
                return Ok(vec![EINVAL]);
            }
            Ok(vec![unsafe { host_now_ns() }])
        });
        let arity = m.funcs.first().map_or(0, |f| f.params.len());
        let mut slots: Vec<Value> = Vec::new();
        if arity >= 1 {
            slots.push(Value::I32(host.grant_host_proc(console)));
        }
        if arity >= 2 {
            slots.push(Value::I32(host.grant_host_proc(clock)));
        }
        // §7 register the live caps under canonical names (F7/F9, PR #118) so the guest can
        // `cap.self.resolve`/`label` them at runtime, matching the fixed-powerbox path.
        for (name, slot) in ["console", "clock"].iter().zip(&slots) {
            if let Value::I32(handle) = slot {
                host.register_cap_name(name, *handle);
            }
        }
        let mut fuel = u64::MAX;
        match bytecode::compile_and_run_with_host(&m, 0, &slots, &mut fuel, &mut host) {
            None => {
                set(STATUS_UNSUPPORTED);
                0
            }
            Some(Err(Trap::Exit(code))) => {
                set(STATUS_EXIT);
                unsafe { EXIT_CODE = code };
                0
            }
            Some(Err(_)) => {
                set(STATUS_TRAP);
                0
            }
            Some(Ok(vals)) => match vals.first() {
                Some(Value::I64(x)) => {
                    set(STATUS_OK);
                    *x
                }
                Some(Value::I32(x)) => {
                    set(STATUS_OK);
                    *x as i64
                }
                _ => {
                    set(STATUS_BAD_RESULT);
                    0
                }
            },
        }
    }
}

// ===== single-shot leaf tier-up drive (#809): InterpDriven on-ramp runs with emitted hot leaves ====
//
// The single-shot wasm-JIT run above requires a `WasmDriven` `_start`; a module whose entry can't
// emit (post-#784 that notably includes every `vm_map`-growing C guest — the mapping function keeps
// the module `InterpDriven`) previously fell all the way back to the pure bytecode path, discarding
// its tier-up-**eligible** pure leaves. This drive closes that gap: the interpreter owns `_start`
// (caps serviced inline — no outlining, no cross-tier bounces), and each direct call to an eligible
// leaf surfaces as a TIERUP event the JS host runs on emitted wasm, with the #717 live-`mapped`
// sync per call (the event carries the window's committed extent; a sparse state declines to the
// interpreter at the dispatch). The window is the owned single-shot buffer with the reservation
// clamped to it (#816's lesson: over-growth fails probeably instead of silently dropping writes).

/// The live single-shot tier-up run. `None` until [`svm_onramp_tierup_open`]; single-threaded wasm.
static mut TIERUP_RUN: Option<TierupRun> = None;

/// [`svm_onramp_tierup_run`] event codes: the run finished (statuses + capture slots staged), a
/// TIERUP awaits servicing (operands via the `svm_onramp_tierup_*` accessors), it trapped
/// (statuses staged; the page re-runs on the interpreter oracle — INVARIANT 9 refusal), or a §22
/// `Jit.invoke` awaits servicing on its unit's emitted wasm (#835 — operands via the
/// `svm_onramp_tierup_jit_*` accessors + the shared `mapped`/`argv` ones).
pub const TIERUP_RUN_DONE: i32 = 0;
pub const TIERUP_RUN_TIERUP: i32 = 1;
pub const TIERUP_RUN_TRAP: i32 = 2;
pub const TIERUP_RUN_JIT_INVOKE: i32 = 3;

/// The single-shot pump's §22 unit-emit parameters (#835), read by [`onramp_tierup_unit_emitter`]
/// (a bare `fn` — [`Host::set_jit_wasm_emitter`] stores no closure state): the run's memory-share
/// flag and its window log2. Stored at [`svm_onramp_tierup_open`]; single-threaded wasm.
static TIERUP_UNIT_SHARED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static TIERUP_UNIT_WIN_LOG2: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// The wasm emitter the single-shot pump installs for a `vm_jit_*`-importing guest (#835/#846):
/// emit a validated unit whole-module in **Model B2** shape (`compile_module_b2` — its
/// `call_indirect` dispatches through the driver's shared funcref table, so a **linked** unit's
/// Slot callbacks reach installed units / eligible program `f{i}`s natively and everything else
/// through a live-state bounce trampoline), or `None` if any function is outside the integer
/// subset — then the invoke runs on the interpreter, fail-closed. The unit's mask is bumped to the
/// pump's run window first — the driver convention everywhere ([`svm_onramp_tierup_open`] bumps
/// the main module the same way): the unit declares the guest's memory (the validator's
/// memory-match precondition), but the run window is larger, and a declared-size mask would alias
/// `vm_map`-grown addresses.
fn onramp_tierup_unit_emitter(blob: &[u8]) -> Option<Vec<u8>> {
    let mut m = svm_encode::decode_module(blob).ok()?;
    let win_log2 = TIERUP_UNIT_WIN_LOG2.load(std::sync::atomic::Ordering::Relaxed);
    if let Some(mc) = m.memory.as_mut() {
        mc.size_log2 = mc.size_log2.max(win_log2);
    }
    let shared = TIERUP_UNIT_SHARED.load(std::sync::atomic::Ordering::Relaxed);
    svm_wasm_jit::compile_module_b2(&m, shared, ONRAMP_JIT_TABLE_LOG2 as u32).ok()
}

struct TierupRun {
    vcpu: bytecode::Vcpu<'static>,
    /// The leaked program the vCPU borrows; reboxed + dropped at close.
    prog: *mut bytecode::VcpuProgram,
    /// The owned window buffer — lives in this module's linear memory, so the emitted leaves
    /// address it directly through the one shared `env.memory`.
    backing: Box<[u8]>,
    emitted_wasm: Vec<u8>,
    /// Pending TIERUP operands (`mapped`/`argv` double as the pending JIT_INVOKE's — one event
    /// is pending at a time).
    func: u32,
    mapped: u64,
    argv: Vec<i64>,
    /// Pending JIT_INVOKE operands (#835): the invoked unit's code handle (the JS host's
    /// instance-cache key), its emitted wasm, and the per-arg/-result scalar type codes the JS
    /// host marshals the i64 slots by.
    jit_code: i32,
    jit_wasm: Option<std::sync::Arc<[u8]>>,
    jit_param_types: Vec<u8>,
    jit_result_types: Vec<u8>,
    /// #846 — the driver-table state. `slot_codes[s]` is the code handle installed at dispatch
    /// slot `s` (`-1` empty/natural), recorded at the inline install/uninstall arms so the JS host
    /// can mirror the engine table into its `WebAssembly.Table` at each event boundary.
    slot_codes: Vec<i32>,
    /// The program functions' signatures (slot = index in the natural prefix) — the shim
    /// generator's signature source.
    sigs: Vec<(Vec<svm_ir::ValType>, Vec<svm_ir::ValType>)>,
    /// The last generated bounce-shim module ([`svm_onramp_tierup_shim_wasm`]) / the last
    /// by-handle unit-wasm fetch ([`svm_onramp_tierup_jit_wasm_by_handle`]) — each valid until the
    /// next call of its accessor.
    shim_wasm: Vec<u8>,
    jit_wasm_by_handle: Option<std::sync::Arc<[u8]>>,
    /// A bounce callback's trap ([`svm_onramp_tierup_call_interp`]), staged so the JS host's
    /// unwind-then-`deliver_jit_trap` resolves the invoke with the *real* trap — an `Exit` must
    /// end the run as `STATUS_EXIT`, exactly as the interpreted invoke would, not as a refusal.
    pending_bounce_trap: Option<Trap>,
    /// The guest's top-level result, staged at DONE.
    value: i64,
    frame: std::sync::Arc<std::sync::Mutex<Option<Frame>>>,
}

/// Open a **leaf tier-up run** over the on-ramp module at `[mod_ptr, mod_len)` with stdin seeded:
/// the interpreter drives `_start`; every all-i64 tier-up-eligible function is emitted for the JS
/// host to run on TIERUP events. A `vm_jit_*`-importing guest (#835, the JACL compiler shape) is
/// admitted too: its runtime-compiled units get a wasm emitter, and a codegen-eligible `Jit.invoke`
/// surfaces as a [`TIERUP_RUN_JIT_INVOKE`] event. Returns `0`, else a negative `STATUS_*` (also in
/// [`svm_status`]) — [`STATUS_UNSUPPORTED`] when nothing could ever run emitted (no eligible leaf
/// and no `Jit` import), or the guest uses threads/futex, whose events this single-vCPU pump cannot
/// service (fibers are serviced in-engine); the page then runs the plain bytecode path. Replaces
/// any prior run. Drive with [`svm_onramp_tierup_run`] + the deliver calls; close with
/// [`svm_onramp_tierup_close`].
#[no_mangle]
pub extern "C" fn svm_onramp_tierup_open(
    mod_ptr: *const u8,
    mod_len: usize,
    stdin_ptr: *const u8,
    stdin_len: usize,
    shared: i32,
) -> i32 {
    let set = |s: i32| unsafe { LAST_STATUS = s };
    svm_onramp_tierup_close();
    // SAFETY: the host guarantees `[mod_ptr, mod_len)` is a live `svm_alloc`ation it just filled.
    let bytes = unsafe { core::slice::from_raw_parts(mod_ptr, mod_len) };
    let Ok(m) = svm_encode::decode_module(bytes) else {
        set(STATUS_DECODE_ERR);
        return -STATUS_DECODE_ERR;
    };
    if onramp_check(&m).is_err() || m.memory.is_none() {
        set(STATUS_UNSUPPORTED);
        return -STATUS_UNSUPPORTED;
    }
    // #926 slice 1 — **no static concurrency gate at open.** The old `any(uses_threads ||
    // uses_futex)` refusal was whole-module, so it also rejected guests whose concurrency ops are
    // *linked but dead* — the JACL compiler-guest (jaclrt's scheduler/GC links thread/futex ops
    // that never run with `POOL_WORKERS=1`) is the motivating case (#839): single-threaded at
    // runtime, but refused for code it never reaches. The gate is unnecessary: a concurrency op
    // that *actually executes* surfaces a `Spawn`/`Join`/`Wait`/`Notify` event, and the run loop's
    // catch-all already declines it to `TIERUP_RUN_TRAP` → the page re-runs on the interpreter
    // (which multiplexes them cooperatively — INVARIANT 9). So the **runtime** event, not a static
    // scan, is the real gate: a guest that never reaches its concurrency ops (JACL) now tiers up;
    // one that does declines cleanly at the op, exactly as the static refusal did but without
    // rejecting dead code. Fibers were already admitted (`step_vcpu` services `cont.*`/`suspend`
    // internally — §22 renegotiated 2026-07-30); §22 `vm_jit_*` guests too (#835). (Servicing the
    // concurrency events on a cooperative multi-vCPU scheduler instead of declining is #926 slice 2
    // — deferred until a guest that genuinely spawns at runtime needs it.)
    let jit_importer = m.imports.iter().any(|im| im.name.starts_with("vm_jit_"));
    let declared = m.memory.map_or(0, |mc| mc.size_log2);
    let win_log2 = JIT_RUN_WIN_LOG2.max(declared);
    // Emit with the mask bumped to the run window (the driver convention everywhere): the emitted
    // `"mapped"` default is then the full window, which is exactly why the per-call sync below is
    // mandatory — the event's committed extent narrows it to what the interpreter admits (#717).
    let mut emit_m = m.clone();
    if let Some(mc) = emit_m.memory.as_mut() {
        mc.size_log2 = win_log2;
    }
    // #880 parity gate for the shared-table world: every dispatch-table target must be reachable
    // from the emitted tier — natively or through a bounce shim — but the i64-slot transport
    // carries scalars only and at most the env scratch's worth of them. A guest with any
    // non-shimmable signature (v128 / over-arity) emits in the old **local**-table mode instead
    // (the emitter's own indirect restriction then applies), so a null shared-table slot can never
    // diverge from the interpreter's dispatch.
    let scalar = |t: &svm_ir::ValType| {
        matches!(
            t,
            svm_ir::ValType::I32
                | svm_ir::ValType::I64
                | svm_ir::ValType::F32
                | svm_ir::ValType::F64
        )
    };
    let max_slots = (svm_wasm_jit::ENV_CELL_BYTES - 16) / 8;
    let all_shimmable = m.funcs.iter().all(|f| {
        f.params.iter().all(scalar)
            && f.results.iter().all(scalar)
            && f.params.len().max(f.results.len()) <= max_slots
    });
    // #880: emit the main module over the **shared reserved table** — `call_indirect`-bearing
    // functions tier up (the language-runtime dispatch-loop shape), their indirect calls reaching
    // installed units natively (old→new) and interpreter-resident targets through the live bounce.
    let emitted_res = if all_shimmable {
        svm_wasm_jit::compile_module_tierup_b2(&emit_m, shared != 0, ONRAMP_JIT_TABLE_LOG2 as u32)
    } else {
        svm_wasm_jit::compile_module_tierup(&emit_m, shared != 0)
    };
    let Ok((wasm, emit)) = emitted_res else {
        set(STATUS_UNSUPPORTED);
        return -STATUS_UNSUPPORTED;
    };
    // The JS host marshals every arg/result as a plain i64 slot (same limit as the par tier-up).
    let all_i64 = |ts: &[svm_ir::ValType]| ts.iter().all(|t| *t == svm_ir::ValType::I64);
    let eligible: Vec<bool> = m
        .funcs
        .iter()
        .enumerate()
        .map(|(i, f)| emit[i] && all_i64(&f.params) && all_i64(&f.results))
        .collect();
    // Nothing for the emitted tier to ever run → refuse (the page's bytecode path is strictly
    // simpler). A `vm_jit_*` importer stays eligible even with no emittable leaf: its win is the
    // *runtime-compiled units* running emitted (#835 — the JACL macro-staging shape).
    if !eligible.iter().any(|&e| e) && !jit_importer {
        set(STATUS_UNSUPPORTED);
        return -STATUS_UNSUPPORTED;
    }
    // The engine's dispatch table must match the emitted `call_indirect` mask
    // (`1 << ONRAMP_JIT_TABLE_LOG2`) — a natural-size table would both number install slots
    // differently and wrap wild indices differently (#846/#880). Sized for every pump guest.
    let Some(prog) = bytecode::VcpuProgram::compile_with_jit_table(&m, ONRAMP_JIT_TABLE_LOG2)
    else {
        set(STATUS_UNSUPPORTED);
        return -STATUS_UNSUPPORTED;
    };
    let prog: &'static bytecode::VcpuProgram = Box::leak(Box::new(prog));
    let mut backing = vec![0u8; 1usize << win_log2].into_boxed_slice();
    let win_ptr = backing.as_mut_ptr();
    // SAFETY: `backing` is owned by the session and pointer-stable across its moves (boxed slice).
    let back =
        std::sync::Arc::new(unsafe { svm_interp::Region::shared(win_ptr, 1u64 << win_log2) });
    let mut host = Host::new();
    host.stdin = if stdin_ptr.is_null() || stdin_len == 0 {
        Vec::new()
    } else {
        // SAFETY: the host guarantees the stdin range is a live `svm_alloc`ation it just filled.
        unsafe { core::slice::from_raw_parts(stdin_ptr, stdin_len) }.to_vec()
    };
    let (frame, _keys) = grant_onramp_caps(&mut host, &m, None);
    // The unit-emit/shim parameters ride statics (the emitter and the shim generator are bare
    // `fn`s) — stored for every open, since shims serve non-jit guests' emitted leaves too (#880).
    TIERUP_UNIT_SHARED.store(shared != 0, std::sync::atomic::Ordering::Relaxed);
    TIERUP_UNIT_WIN_LOG2.store(win_log2, std::sync::atomic::Ordering::Relaxed);
    // #835: arm the §22 unit wasm-emitter for a `vm_jit_*` importer (`grant_onramp_caps` granted
    // the `Jit` cap + validator). Gated on the same shimmable-signature bound as the shared-table
    // emit above — a unit's dispatch depends on shims covering every interpreter-resident slot.
    if jit_importer && all_shimmable {
        host.set_jit_wasm_emitter(onramp_tierup_unit_emitter);
    }
    // Declared prefix mapped, reservation clamped to the buffer: the guest `vm_map`-grows into
    // `[declared, 1 << win_log2)`, and an over-grow fails with `-EINVAL` (#816).
    let vcpu = match bytecode::Vcpu::new_root_reserved_over_with_powerbox(
        prog,
        0,
        &[],
        &[],
        host,
        win_log2,
        back,
    ) {
        Ok(v) => v.with_jit_eligible(std::sync::Arc::from(eligible.into_boxed_slice())),
        Err(_) => {
            // SAFETY: the leaked program has no borrower yet; rebox and drop it.
            drop(unsafe { Box::from_raw(prog as *const _ as *mut bytecode::VcpuProgram) });
            set(STATUS_TRAP);
            return -STATUS_TRAP;
        }
    };
    // SAFETY: single-threaded wasm; the session is read back only via the tier-up exports.
    unsafe {
        *core::ptr::addr_of_mut!(TIERUP_RUN) = Some(TierupRun {
            vcpu,
            prog: prog as *const _ as *mut bytecode::VcpuProgram,
            backing,
            emitted_wasm: wasm,
            func: 0,
            mapped: 0,
            argv: Vec::new(),
            jit_code: 0,
            jit_wasm: None,
            jit_param_types: Vec::new(),
            jit_result_types: Vec::new(),
            slot_codes: vec![-1; 1usize << ONRAMP_JIT_TABLE_LOG2],
            sigs: m
                .funcs
                .iter()
                .map(|f| (f.params.clone(), f.results.clone()))
                .collect(),
            shim_wasm: Vec::new(),
            jit_wasm_by_handle: None,
            pending_bounce_trap: None,
            value: 0,
            frame,
        });
    }
    set(STATUS_OK);
    0
}

/// Pump the open tier-up run: interpret until it finishes ([`TIERUP_RUN_DONE`] — statuses, stdout/
/// stderr, exit code, and framebuffer staged in the shared capture slots), traps
/// ([`TIERUP_RUN_TRAP`] — statuses staged), reaches an eligible call ([`TIERUP_RUN_TIERUP`] —
/// run the emitted `f{func}` and deliver), or reaches a codegen-eligible §22 `Jit.invoke`
/// ([`TIERUP_RUN_JIT_INVOKE`] — instantiate the unit's wasm from the `svm_onramp_tierup_jit_*`
/// accessors, cached per code handle, run its `f0` and deliver via
/// [`svm_onramp_tierup_deliver_jit`]). The host contract per TIERUP/JIT_INVOKE: write
/// [`svm_onramp_tierup_mapped`] to the emitted module's/unit's `"mapped"` global (and arm the
/// fuel env cell) before calling `f{func}(win, env, ...argv)` / `f0(win, env, ...argv)`.
#[no_mangle]
pub extern "C" fn svm_onramp_tierup_run() -> i32 {
    // SAFETY: single-threaded wasm; exclusive access to the session for this call.
    let Some(s) = (unsafe { (*core::ptr::addr_of_mut!(TIERUP_RUN)).as_mut() }) else {
        unsafe { LAST_STATUS = STATUS_UNSUPPORTED };
        return TIERUP_RUN_TRAP;
    };
    let (status, value, exit_code, ev) = loop {
        match s.vcpu.run() {
            bytecode::VcpuEvent::TierUp { func, argv, mapped } => {
                s.func = func;
                s.mapped = mapped;
                s.argv = argv.into_vec();
                return TIERUP_RUN_TIERUP;
            }
            // §22 install/uninstall mutate only engine/host state — resolve authority against this
            // run's own powerbox and deliver inline; the JS host is involved only when a unit *runs*.
            // `svm_par_run`'s arms minus the B2 table mirror: an installed unit's `call_indirect`
            // dispatch stays interpreted here (inline in the caller's frames, where fibers work).
            bytecode::VcpuEvent::JitInstall { handle, code } => {
                let resolved = par_resolve_unit_rt(s.vcpu.host_mut(), handle, code).map(|(f, _)| f);
                // #846: mirror `slot → code` so the JS host can rebuild its `WebAssembly.Table`
                // at the next event boundary (installs only ever happen between events — a unit
                // with a `cap.call` never emits, so the install itself always runs interpreted).
                if let Some(slot) = s.vcpu.deliver_jit_install(resolved) {
                    if let Some(e) = s.slot_codes.get_mut(slot) {
                        *e = code;
                    }
                }
            }
            bytecode::VcpuEvent::JitUninstall { handle, .. } => {
                let authorized = s.vcpu.host_mut().resolve_jit_domain(handle).map(|_| ());
                if let Some(slot) = s.vcpu.deliver_jit_uninstall(authorized) {
                    if let Some(e) = s.slot_codes.get_mut(slot) {
                        *e = -1; // keep the JS table mirror exact (a stale slot must trap)
                    }
                }
            }
            // §22 `Jit.invoke` (#835): a runtime-compiled unit with emitted wasm, all-scalar
            // operands, and a representable window state runs on the JS host (the pending operands
            // staged for the `svm_onramp_tierup_jit_*` accessors); anything else is serviced on the
            // interpreter — fail-closed, it honors the full page map (the PAR_JIT_INVOKE contract).
            bytecode::VcpuEvent::JitInvoke {
                handle,
                code,
                argv,
                params,
                results,
                mapped,
            } => {
                let codes = |ts: &[svm_ir::ValType]| {
                    ts.iter()
                        .map(|t| scalar_type_code(*t))
                        .collect::<Option<Vec<u8>>>()
                };
                let (ptypes, rtypes) = (codes(&params), codes(&results));
                match par_resolve_unit_rt(s.vcpu.host_mut(), handle, code) {
                    Err(t) => s.vcpu.deliver_jit_invoke(Err(t)),
                    Ok((funcs, wasm)) => {
                        if let (Some(w), Some(pt), Some(rt), Some(h)) =
                            (wasm, ptypes, rtypes, mapped)
                        {
                            s.jit_code = code;
                            s.mapped = h;
                            s.argv = argv.into_vec();
                            s.jit_wasm = Some(w);
                            s.jit_param_types = pt;
                            s.jit_result_types = rt;
                            return TIERUP_RUN_JIT_INVOKE;
                        }
                        s.vcpu.deliver_jit_invoke(Ok(funcs));
                    }
                }
            }
            bytecode::VcpuEvent::Done(vals) => match vals.first() {
                Some(Value::I64(x)) => break (STATUS_OK, *x, 0, TIERUP_RUN_DONE),
                Some(Value::I32(x)) => break (STATUS_OK, *x as i64, 0, TIERUP_RUN_DONE),
                None => break (STATUS_OK, 0, 0, TIERUP_RUN_DONE),
                _ => break (STATUS_BAD_RESULT, 0, 0, TIERUP_RUN_DONE),
            },
            bytecode::VcpuEvent::Trapped(Trap::Exit(code)) => {
                break (STATUS_EXIT, 0, code, TIERUP_RUN_DONE)
            }
            bytecode::VcpuEvent::Trapped(_) => break (STATUS_TRAP, 0, 0, TIERUP_RUN_TRAP),
            // A concurrency event (`Spawn`/`Join`/`Wait`/`Notify`) or a `StdinPark` this single-vCPU
            // pump doesn't service: decline to the interpreter (#926 slice 1 — the real gate, now
            // that open no longer scans for dead concurrency ops). Fail closed like the reactor.
            _ => break (STATUS_TRAP, 0, 0, TIERUP_RUN_TRAP),
        }
    };
    s.value = value;
    let host = s.vcpu.host_mut();
    let stdout = std::mem::take(&mut host.stdout);
    let stderr = std::mem::take(&mut host.stderr);
    let fb = s.frame.lock().unwrap().take();
    let (fb_rgba, fb_w, fb_h) = match fb {
        Some(f) => (f.rgba, f.width, f.height),
        None => (Vec::new(), 0, 0),
    };
    // SAFETY: single-threaded wasm; the capture slots are read back only via the export accessors.
    unsafe {
        stash(&mut *core::ptr::addr_of_mut!(OUT), stdout);
        stash(&mut *core::ptr::addr_of_mut!(ERR), stderr);
        stash(&mut *core::ptr::addr_of_mut!(FB), fb_rgba);
        FB_W = fb_w;
        FB_H = fb_h;
        EXIT_CODE = exit_code;
        RUN_VALUE = value;
        LAST_STATUS = status;
    }
    ev
}

/// The pending TIERUP's function index.
#[no_mangle]
pub extern "C" fn svm_onramp_tierup_func() -> i32 {
    // SAFETY: single-threaded wasm; read of the pending operands.
    unsafe { (*core::ptr::addr_of!(TIERUP_RUN)).as_ref() }.map_or(0, |s| s.func as i32)
}

/// The pending TIERUP's/JIT_INVOKE's committed extent — the value for the emitted `"mapped"`
/// global (#717).
#[no_mangle]
pub extern "C" fn svm_onramp_tierup_mapped() -> i64 {
    // SAFETY: single-threaded wasm; read of the pending operands.
    unsafe { (*core::ptr::addr_of!(TIERUP_RUN)).as_ref() }.map_or(0, |s| s.mapped as i64)
}

/// The pending TIERUP's/JIT_INVOKE's marshalled i64 args (base pointer; valid until the next event).
#[no_mangle]
pub extern "C" fn svm_onramp_tierup_argv_ptr() -> *const i64 {
    // SAFETY: single-threaded wasm; read of the pending operands.
    unsafe { (*core::ptr::addr_of!(TIERUP_RUN)).as_ref() }
        .map_or(core::ptr::null(), |s| s.argv.as_ptr())
}

/// Number of pending TIERUP args (see [`svm_onramp_tierup_argv_ptr`]).
#[no_mangle]
pub extern "C" fn svm_onramp_tierup_argv_len() -> usize {
    // SAFETY: single-threaded wasm; read of the pending operands.
    unsafe { (*core::ptr::addr_of!(TIERUP_RUN)).as_ref() }.map_or(0, |s| s.argv.len())
}

/// The emitted tier-up module's bytes (compile + instantiate against this module's memory).
#[no_mangle]
pub extern "C" fn svm_onramp_tierup_wasm_ptr() -> *const u8 {
    // SAFETY: single-threaded wasm; read of the session stash.
    unsafe { (*core::ptr::addr_of!(TIERUP_RUN)).as_ref() }
        .map_or(core::ptr::null(), |s| s.emitted_wasm.as_ptr())
}

#[no_mangle]
pub extern "C" fn svm_onramp_tierup_wasm_len() -> usize {
    // SAFETY: single-threaded wasm; read of the session stash.
    unsafe { (*core::ptr::addr_of!(TIERUP_RUN)).as_ref() }.map_or(0, |s| s.emitted_wasm.len())
}

/// The run window's base address in this module's linear memory (the emitted `f{i}`s' `win` arg).
#[no_mangle]
pub extern "C" fn svm_onramp_tierup_win_ptr() -> *const u8 {
    // SAFETY: single-threaded wasm; read of the session stash.
    unsafe { (*core::ptr::addr_of!(TIERUP_RUN)).as_ref() }
        .map_or(core::ptr::null(), |s| s.backing.as_ptr())
}

/// The run window's byte length (`1 << win_log2` — declared bumped to [`JIT_RUN_WIN_LOG2`]).
#[no_mangle]
pub extern "C" fn svm_onramp_tierup_win_len() -> usize {
    // SAFETY: single-threaded wasm; read of the session stash.
    unsafe { (*core::ptr::addr_of!(TIERUP_RUN)).as_ref() }.map_or(0, |s| s.backing.len())
}

/// The guest's top-level result once [`TIERUP_RUN_DONE`] is reached.
#[no_mangle]
pub extern "C" fn svm_onramp_tierup_value() -> i64 {
    // SAFETY: single-threaded wasm; read of the session stash.
    unsafe { (*core::ptr::addr_of!(TIERUP_RUN)).as_ref() }.map_or(0, |s| s.value)
}

/// Deliver the emitted `f{func}`'s i64 result slots for the pending TIERUP.
#[no_mangle]
pub extern "C" fn svm_onramp_tierup_deliver(rptr: *const i64, n: usize) {
    // SAFETY: single-threaded wasm; `[rptr, n)` is a live `svm_alloc`ation the host just filled.
    if let Some(s) = unsafe { (*core::ptr::addr_of_mut!(TIERUP_RUN)).as_mut() } {
        let vals = if rptr.is_null() || n == 0 {
            &[][..]
        } else {
            unsafe { core::slice::from_raw_parts(rptr, n) }
        };
        s.vcpu.deliver_tierup(vals);
    }
}

/// Deliver a trap from the emitted `f{func}` (a wasm trap or an SVM fault) for the pending TIERUP.
/// When the unwind came from a **bounce** callback's trap (#880 — a `call_indirect`-bearing leaf
/// hit a shim whose callback trapped), the staged real trap is delivered instead, so a callback's
/// `exit` ends the run as `STATUS_EXIT` exactly as the interpreted call would.
#[no_mangle]
pub extern "C" fn svm_onramp_tierup_deliver_trap() {
    // SAFETY: single-threaded wasm; exclusive access to the session.
    if let Some(s) = unsafe { (*core::ptr::addr_of_mut!(TIERUP_RUN)).as_mut() } {
        let t = s.pending_bounce_trap.take().unwrap_or(Trap::Unreachable);
        s.vcpu.deliver_tierup_trap(t);
    }
}

// ---- pending JIT_INVOKE operands (#835) — the §22 unit half of the pump ----------------------

/// The pending JIT_INVOKE's code handle — the JS host keys its per-unit instance cache by this
/// (one wasm instance per compiled unit; args differ per invoke — worker.js's `jitUnitFor` shape).
#[no_mangle]
pub extern "C" fn svm_onramp_tierup_jit_code() -> i32 {
    // SAFETY: single-threaded wasm; read of the pending operands.
    unsafe { (*core::ptr::addr_of!(TIERUP_RUN)).as_ref() }.map_or(0, |s| s.jit_code)
}

/// The pending JIT_INVOKE's emitted unit wasm (instantiate against this module's memory; the bytes
/// stay valid until the next event — the session holds its own `Arc`).
#[no_mangle]
pub extern "C" fn svm_onramp_tierup_jit_wasm_ptr() -> *const u8 {
    // SAFETY: single-threaded wasm; read of the pending operands.
    unsafe { (*core::ptr::addr_of!(TIERUP_RUN)).as_ref() }.map_or(core::ptr::null(), |s| {
        s.jit_wasm
            .as_ref()
            .map_or(core::ptr::null(), |w| w.as_ptr())
    })
}

#[no_mangle]
pub extern "C" fn svm_onramp_tierup_jit_wasm_len() -> usize {
    // SAFETY: single-threaded wasm; read of the pending operands.
    unsafe { (*core::ptr::addr_of!(TIERUP_RUN)).as_ref() }
        .map_or(0, |s| s.jit_wasm.as_ref().map_or(0, |w| w.len()))
}

/// Per-arg scalar type codes of the pending JIT_INVOKE (`0` = i32, `1` = i64, `2` = f32, `3` =
/// f64), one byte per arg — the JS host marshals each [`svm_onramp_tierup_argv_ptr`] i64 slot to
/// the wasm type the unit's `f0` declares. Length equals [`svm_onramp_tierup_argv_len`].
#[no_mangle]
pub extern "C" fn svm_onramp_tierup_jit_param_types_ptr() -> *const u8 {
    // SAFETY: single-threaded wasm; read of the pending operands.
    unsafe { (*core::ptr::addr_of!(TIERUP_RUN)).as_ref() }
        .map_or(core::ptr::null(), |s| s.jit_param_types.as_ptr())
}

/// Per-result scalar type codes of the pending JIT_INVOKE (encoding as
/// [`svm_onramp_tierup_jit_param_types_ptr`]) — the JS host marshals each `f0` result back to its
/// i64 slot (a float's bits, an integer's value) for [`svm_onramp_tierup_deliver_jit`].
#[no_mangle]
pub extern "C" fn svm_onramp_tierup_jit_result_types_ptr() -> *const u8 {
    // SAFETY: single-threaded wasm; read of the pending operands.
    unsafe { (*core::ptr::addr_of!(TIERUP_RUN)).as_ref() }
        .map_or(core::ptr::null(), |s| s.jit_result_types.as_ptr())
}

/// Number of pending JIT_INVOKE results (see [`svm_onramp_tierup_jit_result_types_ptr`]).
#[no_mangle]
pub extern "C" fn svm_onramp_tierup_jit_result_types_len() -> usize {
    // SAFETY: single-threaded wasm; read of the pending operands.
    unsafe { (*core::ptr::addr_of!(TIERUP_RUN)).as_ref() }.map_or(0, |s| s.jit_result_types.len())
}

/// Deliver the unit `f0`'s i64 result slots for the pending JIT_INVOKE — the vCPU resumes exactly
/// as if the interpreter had run the unit ([`Vcpu::deliver_jit_invoke_vals`]'s contract).
#[no_mangle]
pub extern "C" fn svm_onramp_tierup_deliver_jit(rptr: *const i64, n: usize) {
    // SAFETY: single-threaded wasm; `[rptr, n)` is a live `svm_alloc`ation the host just filled.
    if let Some(s) = unsafe { (*core::ptr::addr_of_mut!(TIERUP_RUN)).as_mut() } {
        let vals = if rptr.is_null() || n == 0 {
            &[][..]
        } else {
            unsafe { core::slice::from_raw_parts(rptr, n) }
        };
        s.vcpu.deliver_jit_invoke_vals(vals);
    }
}

/// Deliver a trap from the pending JIT_INVOKE's emitted unit (a wasm trap or an SVM fault) — the
/// vCPU traps on its next pump, as an interpreted invoke trap would. When the unwind was caused by
/// a **bounce** callback's trap ([`svm_onramp_tierup_call_interp`] returned nonzero), the staged
/// real trap is delivered instead — so a callback's `exit` ends the run as `STATUS_EXIT`, exactly
/// as the interpreted invoke would, not as a refusal (#846).
#[no_mangle]
pub extern "C" fn svm_onramp_tierup_deliver_jit_trap() {
    // SAFETY: single-threaded wasm; exclusive access to the session.
    if let Some(s) = unsafe { (*core::ptr::addr_of_mut!(TIERUP_RUN)).as_mut() } {
        let t = s.pending_bounce_trap.take().unwrap_or(Trap::Unreachable);
        s.vcpu.deliver_jit_invoke_trap(t);
    }
}

// ---- #846: the driver table + live-state bounce (the §22 linked-unit half) -------------------

/// Service one **cross-tier bounce** out of an emitted §22 unit: the JS host is inside the pending
/// JIT_INVOKE's `f0` (or a table trampoline it called), and the emitted code reached a target with
/// no wasm body. `target` is the dispatch-table slot (= the function's own index for the natural
/// prefix — the value the trampoline was generated with); `[args_ptr, …)` are the i64 arg slots in
/// the `env.call_interp` scratch, overwritten with the result slots. Runs the target on a nested
/// interpretation over the run's **live** window/powerbox/fuel ([`Vcpu::bounce_call`] — fibers
/// persist across the invoke's bounces). Returns `0`, or `1` on a trap (staged — the JS host must
/// throw to unwind the emitted frames, then [`svm_onramp_tierup_deliver_jit_trap`] delivers it).
#[no_mangle]
pub extern "C" fn svm_onramp_tierup_call_interp(target: u32, args_ptr: *mut u8) -> i32 {
    // SAFETY: single-threaded wasm; the vCPU is parked on the pending invoke (not executing), so
    // this re-entrant borrow of the session is exclusive. The host guarantees `args_ptr` addresses
    // the env cell's scratch (≥ `ENV_CELL_BYTES - 16` bytes — the emitted call sites and the shim
    // bodies both spill there).
    let Some(s) = (unsafe { (*core::ptr::addr_of_mut!(TIERUP_RUN)).as_mut() }) else {
        return 1;
    };
    let max_slots = (svm_wasm_jit::ENV_CELL_BYTES - 16) / 8;
    let io = unsafe { core::slice::from_raw_parts_mut(args_ptr as *mut i64, max_slots) };
    match s.vcpu.bounce_call(target, io) {
        Ok(_) => 0,
        Err(t) => {
            s.pending_bounce_trap = Some(t);
            1
        }
    }
}

/// The window's committed scalar extent **right now** — the JS host re-syncs it to every live
/// instance's `"mapped"` global after each bounce (a callback may have `vm_map`-grown the window
/// mid-invoke; the fan-out makes growth visible exactly when the interpreted path would see it).
#[no_mangle]
pub extern "C" fn svm_onramp_tierup_mapped_now() -> i64 {
    // SAFETY: single-threaded wasm; read of the parked session.
    unsafe { (*core::ptr::addr_of!(TIERUP_RUN)).as_ref() }
        .map_or(0, |s| s.vcpu.window_scalar_extent() as i64)
}

/// The dispatch-table size (log2) a `vm_jit_*` run's table is built with — the JS host sizes its
/// `WebAssembly.Table` to `1 << this` (the emitted `call_indirect` mask).
#[no_mangle]
pub extern "C" fn svm_onramp_tierup_table_log2() -> u32 {
    ONRAMP_JIT_TABLE_LOG2 as u32
}

/// The guest program's function count — the dispatch table's **natural prefix** (`slot i < nfuncs`
/// dispatches program function `i`; slots at or past it hold installed units or trap empty).
#[no_mangle]
pub extern "C" fn svm_onramp_tierup_nfuncs() -> usize {
    // SAFETY: single-threaded wasm; read of the session stash.
    unsafe { (*core::ptr::addr_of!(TIERUP_RUN)).as_ref() }.map_or(0, |s| s.sigs.len())
}

/// The code handle installed at dispatch-table `slot` (`-1` empty/natural) — the mirror the JS
/// host rebuilds its table from at each event boundary (worker.js's `svm_par_jit_slot_code` shape).
#[no_mangle]
pub extern "C" fn svm_onramp_tierup_slot_code(slot: u32) -> i32 {
    // SAFETY: single-threaded wasm; read of the session stash.
    unsafe { (*core::ptr::addr_of!(TIERUP_RUN)).as_ref() }.map_or(-1, |s| {
        s.slot_codes.get(slot as usize).copied().unwrap_or(-1)
    })
}

/// Emitted-wasm length for **any** compiled unit by code handle (`0` if none — the unit is
/// interpreter-only) — so the JS host can instantiate an *installed* slot's unit it hasn't itself
/// invoked (worker.js's `svm_par_jit_code_wasm_by_handle` shape). The bytes (via
/// [`svm_onramp_tierup_jit_wasm_by_handle_ptr`]) stay valid until the next call.
#[no_mangle]
pub extern "C" fn svm_onramp_tierup_jit_wasm_by_handle_len(code: i32) -> usize {
    // SAFETY: single-threaded wasm; exclusive access to the session stash.
    let Some(s) = (unsafe { (*core::ptr::addr_of_mut!(TIERUP_RUN)).as_mut() }) else {
        return 0;
    };
    let h = s.vcpu.host_mut();
    s.jit_wasm_by_handle = h
        .resolve_jit_code(code)
        .ok()
        .and_then(|(cd, cu)| h.jit_unit_wasm(cd, cu));
    s.jit_wasm_by_handle.as_ref().map_or(0, |w| w.len())
}

/// Pointer to the emitted wasm the last [`svm_onramp_tierup_jit_wasm_by_handle_len`] resolved.
#[no_mangle]
pub extern "C" fn svm_onramp_tierup_jit_wasm_by_handle_ptr() -> *const u8 {
    // SAFETY: single-threaded wasm; read of the session stash.
    unsafe { (*core::ptr::addr_of!(TIERUP_RUN)).as_ref() }.map_or(core::ptr::null(), |s| {
        s.jit_wasm_by_handle
            .as_ref()
            .map_or(core::ptr::null(), |w| w.as_ptr())
    })
}

/// Generate the **bounce-shim module** for dispatch-table `slot` — a standalone one-function wasm
/// module (`export "t"`, [`svm_wasm_jit::emit_slot_trampoline`]) with the slot's occupant's
/// env-prepended signature, whose body bounces to [`svm_onramp_tierup_call_interp`] with `slot`
/// baked in. The JS host `table.set`s its instance's `"t"` into the slot, so an emitted unit's
/// `call_indirect` to an interpreter-resident target lands on the live-state bounce. Returns the
/// module's byte length (`0` = no shim: empty slot, or a signature the transport can't carry — the
/// open-time `all_shimmable` gate makes the latter unreachable for a run whose units emit). Bytes
/// via [`svm_onramp_tierup_shim_ptr`], valid until the next call.
#[no_mangle]
pub extern "C" fn svm_onramp_tierup_shim_wasm(slot: u32) -> usize {
    // SAFETY: single-threaded wasm; exclusive access to the session stash.
    let Some(s) = (unsafe { (*core::ptr::addr_of_mut!(TIERUP_RUN)).as_mut() }) else {
        return 0;
    };
    let sig = if (slot as usize) < s.sigs.len() {
        Some(s.sigs[slot as usize].clone())
    } else {
        let code = s.slot_codes.get(slot as usize).copied().unwrap_or(-1);
        if code < 0 {
            None
        } else {
            let h = s.vcpu.host_mut();
            h.resolve_jit_code(code)
                .ok()
                .and_then(|(cd, cu)| h.jit_unit_funcs(cd, cu))
                .and_then(|fs| fs.first().map(|f| (f.params.clone(), f.results.clone())))
        }
    };
    let Some((params, results)) = sig else {
        s.shim_wasm.clear();
        return 0;
    };
    let shared = TIERUP_UNIT_SHARED.load(std::sync::atomic::Ordering::Relaxed);
    match svm_wasm_jit::emit_slot_trampoline(&params, &results, slot, shared) {
        Ok(w) => {
            s.shim_wasm = w;
            s.shim_wasm.len()
        }
        Err(_) => {
            s.shim_wasm.clear();
            0
        }
    }
}

/// Pointer to the shim module the last [`svm_onramp_tierup_shim_wasm`] generated.
#[no_mangle]
pub extern "C" fn svm_onramp_tierup_shim_ptr() -> *const u8 {
    // SAFETY: single-threaded wasm; read of the session stash.
    unsafe { (*core::ptr::addr_of!(TIERUP_RUN)).as_ref() }
        .map_or(core::ptr::null(), |s| s.shim_wasm.as_ptr())
}

/// Close the open tier-up run, freeing its window and program. Idempotent.
#[no_mangle]
pub extern "C" fn svm_onramp_tierup_close() {
    // SAFETY: single-threaded wasm; take the session; the vCPU (borrowing the leaked program) is
    // dropped before the program is reboxed and freed.
    unsafe {
        if let Some(s) = (*core::ptr::addr_of_mut!(TIERUP_RUN)).take() {
            let prog = s.prog;
            drop(s);
            drop(Box::from_raw(prog));
        }
    }
}

// ============================================================================================
// #926 slice 2 — the **cooperative** tier-up driver (`svm_coop_*`).
//
// The single-vCPU `svm_onramp_tierup_*` driver above declines a guest that actually reaches a
// `Spawn`/`Join`/`Wait`/`Notify` event (its `run()` catch-all → `TIERUP_RUN_TRAP`), because one
// `Vcpu` can't multiplex threads. This driver wraps [`bytecode::CoopRun`] instead — the cooperative
// multi-vCPU scheduler — so a real `thread.spawn` guest tiers up its hot leaves on **one** wasm
// thread, no Workers (the JACL-with-runtime-threads shape). It is a strict simplification of the
// single-vCPU FFI: `CoopRun` owns its `Domain`/window/powerbox (no leaked program), and services
// `Spawn`/`Join`/`Wait`/`Notify`/fibers/`Jit.install`/`Jit.invoke` **internally**, so the only
// events that reach the host are tier-up and the run's end — the pump loop has no concurrency or
// §22 arms. Emitted `Jit.invoke` acceleration (a host event) is a later refinement; here units run
// interpreted inside the pump. The cross-tier `call_interp` bounce is routed to the tiering-up
// task's env by [`CoopRun::bounce`] (the confinement hinge).
//
// The capture/status statics (`OUT`/`ERR`/`FB`/`EXIT_CODE`/`RUN_VALUE`/`LAST_STATUS`) and the
// `svm_alloc`/`svm_stdout_*` accessors are shared with the single-vCPU driver — only one runs at a
// time (single-threaded wasm; tests serialize on the FFI lock).
// ============================================================================================

/// Cooperative-driver event codes (returned by [`svm_coop_run`]) — the `CoopEvent` subset that
/// reaches the host. Distinct constants from the `TIERUP_RUN_*` set for clarity, though the values
/// coincide.
pub const COOP_RUN_DONE: i32 = 0;
pub const COOP_RUN_TIERUP: i32 = 1;
pub const COOP_RUN_TRAP: i32 = 2;
pub const COOP_RUN_JIT_INVOKE: i32 = 3;

/// The live cooperative tier-up session — the `CoopRun` plus the host-facing operand/capture state
/// (mirrors the relevant fields of `TierupRun`). The `backing` box owns the window `CoopRun`'s `Mem`
/// addresses through a raw-pointer `Region`; both drop together at [`svm_coop_close`].
struct CoopTierupRun {
    run: bytecode::CoopRun,
    /// The owned window buffer — in this module's linear memory, so emitted leaves address it through
    /// the one shared `env.memory`. Pointer-stable across the struct's moves (boxed slice).
    backing: Box<[u8]>,
    emitted_wasm: Vec<u8>,
    /// Pending TIERUP / JIT_INVOKE operands (one event is pending at a time, so `mapped`/`argv` are
    /// shared between them).
    func: u32,
    mapped: u64,
    argv: Vec<i64>,
    /// Pending JIT_INVOKE operands (#926 slice 2e): the invoked unit's code handle (the JS host's
    /// instance-cache key), its emitted wasm, and the per-arg/-result scalar type codes.
    jit_code: i32,
    jit_wasm: Option<std::sync::Arc<[u8]>>,
    jit_param_types: Vec<u8>,
    jit_result_types: Vec<u8>,
    /// #926 slice 2f — the B2 driver-table state (the twin of `TierupRun`'s). `sigs[i]` is program
    /// function `i`'s signature (the shim generator's source for a slot in the natural prefix); the
    /// slot→code mirror itself lives on the engine's `CoopSched` (read via [`bytecode::CoopRun::slot_code`],
    /// since coop `Jit.install` happens inside the pump). `shim_wasm`/`jit_wasm_by_handle` each hold the
    /// last generated bounce-shim / by-handle unit-wasm, valid until the next call of its accessor.
    sigs: Vec<(Vec<svm_ir::ValType>, Vec<svm_ir::ValType>)>,
    shim_wasm: Vec<u8>,
    jit_wasm_by_handle: Option<std::sync::Arc<[u8]>>,
    /// A bounce callback's staged trap (see [`svm_coop_call_interp`] / [`svm_coop_deliver_trap`]).
    pending_bounce_trap: Option<Trap>,
    /// The guest's top-level result, staged at DONE.
    value: i64,
    frame: std::sync::Arc<std::sync::Mutex<Option<Frame>>>,
}

static mut COOP_RUN: Option<CoopTierupRun> = None;

/// Open a cooperative tier-up run over the guest module `[mod_ptr, mod_len)` (stdin optional,
/// `shared` = SharedArrayBuffer memory). Returns `0`/`STATUS_OK` on success, a negative `STATUS_*`
/// on refusal (decode error, an op outside the engine subset, or nothing for the emitted tier to
/// run). Idempotent open: closes any prior run first.
#[no_mangle]
pub extern "C" fn svm_coop_open(
    mod_ptr: *const u8,
    mod_len: usize,
    stdin_ptr: *const u8,
    stdin_len: usize,
    shared: i32,
) -> i32 {
    let set = |s: i32| unsafe { LAST_STATUS = s };
    svm_coop_close();
    // SAFETY: the host guarantees `[mod_ptr, mod_len)` is a live `svm_alloc`ation it just filled.
    let bytes = unsafe { core::slice::from_raw_parts(mod_ptr, mod_len) };
    let Ok(m) = svm_encode::decode_module(bytes) else {
        set(STATUS_DECODE_ERR);
        return -STATUS_DECODE_ERR;
    };
    if onramp_check(&m).is_err() || m.memory.is_none() {
        set(STATUS_UNSUPPORTED);
        return -STATUS_UNSUPPORTED;
    }
    let declared = m.memory.map_or(0, |mc| mc.size_log2);
    let win_log2 = JIT_RUN_WIN_LOG2.max(declared);
    // Emit with the mask bumped to the run window (the driver convention), so the emitted `"mapped"`
    // default is the full window; the per-tier-up `mapped` event narrows it to the committed extent.
    let mut emit_m = m.clone();
    if let Some(mc) = emit_m.memory.as_mut() {
        mc.size_log2 = win_log2;
    }
    // #926 slice 2f — B2 vs non-B2 emit, exactly the single-shot pump's gate ([`svm_onramp_tierup_open`]).
    // A guest whose every function has a shimmable signature (scalar operands, arity ≤ the env
    // scratch's slot count) emits over the **shared reserved table** (Model B2): `call_indirect`-bearing
    // functions tier up (the language-runtime dispatch-loop shape), their indirect calls reaching
    // installed §22 units natively (old→new) and interpreter-resident targets through the live bounce.
    // A non-shimmable guest (v128 / over-arity) emits in the old local-table mode, where a null
    // shared-table slot can never diverge from the interpreter's dispatch. An unsupported shape declines.
    let scalar = |t: &svm_ir::ValType| {
        matches!(
            t,
            svm_ir::ValType::I32
                | svm_ir::ValType::I64
                | svm_ir::ValType::F32
                | svm_ir::ValType::F64
        )
    };
    let max_slots = (svm_wasm_jit::ENV_CELL_BYTES - 16) / 8;
    let all_shimmable = m.funcs.iter().all(|f| {
        f.params.iter().all(scalar)
            && f.results.iter().all(scalar)
            && f.params.len().max(f.results.len()) <= max_slots
    });
    let emitted_res = if all_shimmable {
        svm_wasm_jit::compile_module_tierup_b2(&emit_m, shared != 0, ONRAMP_JIT_TABLE_LOG2 as u32)
    } else {
        svm_wasm_jit::compile_module_tierup(&emit_m, shared != 0)
    };
    let (wasm, emit) = match emitted_res {
        Ok(x) => x,
        Err(_) => {
            set(STATUS_UNSUPPORTED);
            return -STATUS_UNSUPPORTED;
        }
    };
    // A function tiers up iff the emitter emitted it and its signature is all-i64 (the i64-slot
    // transport the host marshals by) — exactly the single-vCPU gate.
    let all_i64 = |ts: &[svm_ir::ValType]| ts.iter().all(|t| *t == svm_ir::ValType::I64);
    let eligible: Vec<bool> = m
        .funcs
        .iter()
        .enumerate()
        .map(|(i, f)| emit[i] && all_i64(&f.params) && all_i64(&f.results))
        .collect();
    // #835/#926: a `vm_jit_*` importer stays eligible even with no emittable leaf — its win is the
    // runtime-compiled §22 units running emitted (the JACL macro-staging shape). A guest with neither
    // an eligible leaf nor a jit importer has nothing for the emitted tier to ever run → decline.
    let jit_importer = m.imports.iter().any(|im| im.name.starts_with("vm_jit_"));
    if !eligible.iter().any(|&e| e) && !jit_importer {
        set(STATUS_UNSUPPORTED);
        return -STATUS_UNSUPPORTED;
    }
    let mut backing = vec![0u8; 1usize << win_log2].into_boxed_slice();
    let win_ptr = backing.as_mut_ptr();
    // SAFETY: `backing` is owned by the session and pointer-stable across its moves (boxed slice).
    let back =
        std::sync::Arc::new(unsafe { svm_interp::Region::shared(win_ptr, 1u64 << win_log2) });
    let mut host = Host::new();
    host.stdin = if stdin_ptr.is_null() || stdin_len == 0 {
        Vec::new()
    } else {
        // SAFETY: the host guarantees the stdin range is a live `svm_alloc`ation it just filled.
        unsafe { core::slice::from_raw_parts(stdin_ptr, stdin_len) }.to_vec()
    };
    let (frame, _keys) = grant_onramp_caps(&mut host, &m, None);
    // #926 slice 2f: a B2 main module masks `call_indirect` against `1 << ONRAMP_JIT_TABLE_LOG2`, so the
    // engine's dispatch table must be the same size — a natural-size table would number install slots
    // and wrap wild indices differently (#846/#880). `CoopRun` builds the domain with
    // `host.jit_table_log2()`, so force it here (a `vm_jit_*` importer's `grant_onramp_caps` already set
    // it to this floor; `set_jit_table_log2` takes the max, so this is idempotent for that case).
    if all_shimmable {
        host.set_jit_table_log2(ONRAMP_JIT_TABLE_LOG2);
    }
    // #926 slice 2e/2f: arm the §22 unit wasm-emitter for a `vm_jit_*` importer so its runtime-compiled
    // units run emitted (the pump then surfaces a `CoopEvent::JitInvoke` when a unit has emitted wasm;
    // an interpreter-only unit falls back to the inline service). Gated on `all_shimmable` like the
    // shared-table emit — a unit emits B2 (`onramp_tierup_unit_emitter`), so its `call_indirect`
    // dispatch depends on the driver table covering every interpreter-resident slot with a shim; a
    // non-shimmable guest's units run interpreted instead (fail-closed, matching the single-shot pump).
    // Reuses the single-vCPU emitter and its shared parameters — only one driver runs at a time.
    TIERUP_UNIT_SHARED.store(shared != 0, std::sync::atomic::Ordering::Relaxed);
    TIERUP_UNIT_WIN_LOG2.store(win_log2, std::sync::atomic::Ordering::Relaxed);
    if jit_importer && all_shimmable {
        host.set_jit_wasm_emitter(onramp_tierup_unit_emitter);
    }
    let tierup = bytecode::TierUpConfig {
        eligible: std::sync::Arc::from(eligible.into_boxed_slice()),
        page_checked: false,
    };
    // `CoopRun` owns its `Domain`; the window is built over `back` with the reservation clamped to
    // the run window (`vm_map`-grow into `[declared, 1 << win_log2)`, an over-grow `-EINVAL`s).
    let run = match bytecode::CoopRun::new_over(
        &m,
        0,
        &[],
        u64::MAX,
        host,
        Some(tierup),
        &[],
        win_log2,
        back,
    ) {
        Some(Ok(r)) => r,
        Some(Err(_)) => {
            set(STATUS_TRAP);
            return -STATUS_TRAP;
        }
        None => {
            set(STATUS_UNSUPPORTED);
            return -STATUS_UNSUPPORTED;
        }
    };
    // SAFETY: single-threaded wasm; the session is read back only via the coop exports.
    unsafe {
        *core::ptr::addr_of_mut!(COOP_RUN) = Some(CoopTierupRun {
            run,
            backing,
            emitted_wasm: wasm,
            func: 0,
            mapped: 0,
            argv: Vec::new(),
            jit_code: 0,
            jit_wasm: None,
            jit_param_types: Vec::new(),
            jit_result_types: Vec::new(),
            sigs: m
                .funcs
                .iter()
                .map(|f| (f.params.clone(), f.results.clone()))
                .collect(),
            shim_wasm: Vec::new(),
            jit_wasm_by_handle: None,
            pending_bounce_trap: None,
            value: 0,
            frame,
        });
    }
    set(STATUS_OK);
    0
}

/// Pump the cooperative run to its next host event: `COOP_RUN_TIERUP` (read the operands via the
/// getters, run the emitted `f{func}`, then [`svm_coop_deliver`]/[`svm_coop_deliver_trap`]) or
/// `COOP_RUN_DONE`/`COOP_RUN_TRAP` (the run ended — result + stdout/stderr/framebuffer are captured
/// into the shared accessor slots). All concurrency is multiplexed inside the pump.
#[no_mangle]
pub extern "C" fn svm_coop_run() -> i32 {
    // SAFETY: single-threaded wasm; exclusive access to the session for this call.
    let Some(s) = (unsafe { (*core::ptr::addr_of_mut!(COOP_RUN)).as_mut() }) else {
        unsafe { LAST_STATUS = STATUS_UNSUPPORTED };
        return COOP_RUN_TRAP;
    };
    let (status, value, exit_code, ev) = match s.run.run() {
        bytecode::CoopEvent::TierUp { func, argv, mapped } => {
            s.func = func;
            s.mapped = mapped;
            s.argv = argv.into_vec();
            return COOP_RUN_TIERUP;
        }
        bytecode::CoopEvent::JitInvoke {
            code,
            wasm,
            argv,
            params,
            results,
            mapped,
        } => {
            // The unit is emittable + all-scalar (the pump gated surfacing on it), so every type
            // maps to a scalar code; the host marshals the i64 slots by these.
            let codes = |ts: &[svm_ir::ValType]| {
                ts.iter()
                    .map(|t| scalar_type_code(*t).unwrap_or(0))
                    .collect::<Vec<u8>>()
            };
            s.jit_code = code;
            s.mapped = mapped;
            s.argv = argv.into_vec();
            s.jit_param_types = codes(&params);
            s.jit_result_types = codes(&results);
            s.jit_wasm = Some(wasm);
            return COOP_RUN_JIT_INVOKE;
        }
        bytecode::CoopEvent::Done(vals) => match vals.first() {
            Some(Value::I64(x)) => (STATUS_OK, *x, 0, COOP_RUN_DONE),
            Some(Value::I32(x)) => (STATUS_OK, *x as i64, 0, COOP_RUN_DONE),
            None => (STATUS_OK, 0, 0, COOP_RUN_DONE),
            _ => (STATUS_BAD_RESULT, 0, 0, COOP_RUN_DONE),
        },
        bytecode::CoopEvent::Trapped(Trap::Exit(code)) => (STATUS_EXIT, 0, code, COOP_RUN_DONE),
        bytecode::CoopEvent::Trapped(_) => (STATUS_TRAP, 0, 0, COOP_RUN_TRAP),
    };
    s.value = value;
    let host = s.run.host_mut();
    let stdout = std::mem::take(&mut host.stdout);
    let stderr = std::mem::take(&mut host.stderr);
    let fb = s.frame.lock().unwrap().take();
    let (fb_rgba, fb_w, fb_h) = match fb {
        Some(f) => (f.rgba, f.width, f.height),
        None => (Vec::new(), 0, 0),
    };
    // SAFETY: single-threaded wasm; the capture slots are read back only via the export accessors.
    unsafe {
        stash(&mut *core::ptr::addr_of_mut!(OUT), stdout);
        stash(&mut *core::ptr::addr_of_mut!(ERR), stderr);
        stash(&mut *core::ptr::addr_of_mut!(FB), fb_rgba);
        FB_W = fb_w;
        FB_H = fb_h;
        EXIT_CODE = exit_code;
        RUN_VALUE = value;
        LAST_STATUS = status;
    }
    ev
}

/// The pending TIERUP's function index.
#[no_mangle]
pub extern "C" fn svm_coop_func() -> i32 {
    unsafe { (*core::ptr::addr_of!(COOP_RUN)).as_ref() }.map_or(0, |s| s.func as i32)
}

/// The pending TIERUP's committed extent — the value for the emitted `"mapped"` global (#717).
#[no_mangle]
pub extern "C" fn svm_coop_mapped() -> i64 {
    unsafe { (*core::ptr::addr_of!(COOP_RUN)).as_ref() }.map_or(0, |s| s.mapped as i64)
}

/// The pending TIERUP's marshalled i64 args (base pointer; valid until the next event).
#[no_mangle]
pub extern "C" fn svm_coop_argv_ptr() -> *const i64 {
    unsafe { (*core::ptr::addr_of!(COOP_RUN)).as_ref() }
        .map_or(core::ptr::null(), |s| s.argv.as_ptr())
}

/// Number of pending TIERUP args (see [`svm_coop_argv_ptr`]).
#[no_mangle]
pub extern "C" fn svm_coop_argv_len() -> usize {
    unsafe { (*core::ptr::addr_of!(COOP_RUN)).as_ref() }.map_or(0, |s| s.argv.len())
}

/// The emitted tier-up module's bytes (compile + instantiate against this module's memory).
#[no_mangle]
pub extern "C" fn svm_coop_wasm_ptr() -> *const u8 {
    unsafe { (*core::ptr::addr_of!(COOP_RUN)).as_ref() }
        .map_or(core::ptr::null(), |s| s.emitted_wasm.as_ptr())
}

#[no_mangle]
pub extern "C" fn svm_coop_wasm_len() -> usize {
    unsafe { (*core::ptr::addr_of!(COOP_RUN)).as_ref() }.map_or(0, |s| s.emitted_wasm.len())
}

/// The run window's base address in this module's linear memory (the emitted `f{i}`s' `win` arg).
#[no_mangle]
pub extern "C" fn svm_coop_win_ptr() -> *const u8 {
    unsafe { (*core::ptr::addr_of!(COOP_RUN)).as_ref() }
        .map_or(core::ptr::null(), |s| s.backing.as_ptr())
}

/// The run window's byte length (`1 << win_log2`).
#[no_mangle]
pub extern "C" fn svm_coop_win_len() -> usize {
    unsafe { (*core::ptr::addr_of!(COOP_RUN)).as_ref() }.map_or(0, |s| s.backing.len())
}

/// The guest's top-level result once [`COOP_RUN_DONE`] is reached.
#[no_mangle]
pub extern "C" fn svm_coop_value() -> i64 {
    unsafe { (*core::ptr::addr_of!(COOP_RUN)).as_ref() }.map_or(0, |s| s.value)
}

/// Deliver the emitted `f{func}`'s i64 result slots for the pending TIERUP, resuming the paused task.
#[no_mangle]
pub extern "C" fn svm_coop_deliver(rptr: *const i64, n: usize) {
    // SAFETY: single-threaded wasm; `[rptr, n)` is a live `svm_alloc`ation the host just filled.
    if let Some(s) = unsafe { (*core::ptr::addr_of_mut!(COOP_RUN)).as_mut() } {
        let vals = if rptr.is_null() || n == 0 {
            &[][..]
        } else {
            unsafe { core::slice::from_raw_parts(rptr, n) }
        };
        s.run.deliver_tierup(vals);
    }
}

/// Deliver a trap from the emitted `f{func}` for the pending TIERUP. A bounce callback's staged trap
/// (see [`svm_coop_call_interp`]) is delivered in preference, so a callback's `exit` ends the run as
/// `STATUS_EXIT` exactly as the interpreted call would.
#[no_mangle]
pub extern "C" fn svm_coop_deliver_trap() {
    // SAFETY: single-threaded wasm; exclusive access to the session.
    if let Some(s) = unsafe { (*core::ptr::addr_of_mut!(COOP_RUN)).as_mut() } {
        let t = s.pending_bounce_trap.take().unwrap_or(Trap::Unreachable);
        s.run.deliver_tierup_trap(t);
    }
}

// ---- #926 slice 2e: pending JIT_INVOKE operands (the §22 runtime-unit half of the coop pump) ----

/// The pending JIT_INVOKE unit's code handle (the JS host's instance-cache key).
#[no_mangle]
pub extern "C" fn svm_coop_jit_code() -> i32 {
    unsafe { (*core::ptr::addr_of!(COOP_RUN)).as_ref() }.map_or(0, |s| s.jit_code)
}

/// The pending JIT_INVOKE unit's emitted wasm bytes (compile + instantiate its `f0`).
#[no_mangle]
pub extern "C" fn svm_coop_jit_wasm_ptr() -> *const u8 {
    unsafe { (*core::ptr::addr_of!(COOP_RUN)).as_ref() }
        .and_then(|s| s.jit_wasm.as_ref())
        .map_or(core::ptr::null(), |w| w.as_ptr())
}

#[no_mangle]
pub extern "C" fn svm_coop_jit_wasm_len() -> usize {
    unsafe { (*core::ptr::addr_of!(COOP_RUN)).as_ref() }
        .and_then(|s| s.jit_wasm.as_ref())
        .map_or(0, |w| w.len())
}

/// The pending JIT_INVOKE's per-arg scalar type codes (length is [`svm_coop_argv_len`], as the args).
#[no_mangle]
pub extern "C" fn svm_coop_jit_param_types_ptr() -> *const u8 {
    unsafe { (*core::ptr::addr_of!(COOP_RUN)).as_ref() }
        .map_or(core::ptr::null(), |s| s.jit_param_types.as_ptr())
}

/// The pending JIT_INVOKE's per-result scalar type codes (marshal the emitted `f0`'s returns by these).
#[no_mangle]
pub extern "C" fn svm_coop_jit_result_types_ptr() -> *const u8 {
    unsafe { (*core::ptr::addr_of!(COOP_RUN)).as_ref() }
        .map_or(core::ptr::null(), |s| s.jit_result_types.as_ptr())
}

#[no_mangle]
pub extern "C" fn svm_coop_jit_result_types_len() -> usize {
    unsafe { (*core::ptr::addr_of!(COOP_RUN)).as_ref() }.map_or(0, |s| s.jit_result_types.len())
}

/// Deliver the emitted unit `f0`'s i64 result slots for the pending JIT_INVOKE, resuming the task.
#[no_mangle]
pub extern "C" fn svm_coop_deliver_jit(rptr: *const i64, n: usize) {
    // SAFETY: single-threaded wasm; `[rptr, n)` is a live `svm_alloc`ation the host just filled.
    if let Some(s) = unsafe { (*core::ptr::addr_of_mut!(COOP_RUN)).as_mut() } {
        let vals = if rptr.is_null() || n == 0 {
            &[][..]
        } else {
            unsafe { core::slice::from_raw_parts(rptr, n) }
        };
        s.run.deliver_jit_invoke_vals(vals);
    }
}

/// Deliver a trap from the emitted unit for the pending JIT_INVOKE (a bounce callback's staged trap in
/// preference, so a callback's `exit` ends the run as `STATUS_EXIT` exactly as interpreted).
#[no_mangle]
pub extern "C" fn svm_coop_deliver_jit_trap() {
    // SAFETY: single-threaded wasm; exclusive access to the session.
    if let Some(s) = unsafe { (*core::ptr::addr_of_mut!(COOP_RUN)).as_mut() } {
        let t = s.pending_bounce_trap.take().unwrap_or(Trap::Unreachable);
        s.run.deliver_jit_invoke_trap(t);
    }
}

/// The emitted tier-up region's cross-tier `env.call_interp(target, args_ptr)`: bounce into the
/// interp-resident leaf `target` over the **tiering-up task's** window/powerbox (routed by
/// [`CoopRun::bounce`]). `args_ptr` is the env scratch (i64 slots, args→results in place). Returns
/// `0` on success, `1` on a callback trap (staged for [`svm_coop_deliver_trap`]).
#[no_mangle]
pub extern "C" fn svm_coop_call_interp(target: u32, args_ptr: *mut u8) -> i32 {
    let Some(s) = (unsafe { (*core::ptr::addr_of_mut!(COOP_RUN)).as_mut() }) else {
        return 1;
    };
    let max_slots = (svm_wasm_jit::ENV_CELL_BYTES - 16) / 8;
    // SAFETY: the host passes the env scratch, at least `max_slots` i64s wide.
    let io = unsafe { core::slice::from_raw_parts_mut(args_ptr as *mut i64, max_slots) };
    match s.run.bounce(target, io) {
        Ok(_) => 0,
        Err(t) => {
            s.pending_bounce_trap = Some(t);
            1
        }
    }
}

/// The run window's committed scalar extent right now — the #717 value the host re-syncs to every
/// emitted instance's `"mapped"` global after a [`svm_coop_call_interp`] bounce.
#[no_mangle]
pub extern "C" fn svm_coop_mapped_now() -> i64 {
    unsafe { (*core::ptr::addr_of!(COOP_RUN)).as_ref() }
        .map_or(0, |s| s.run.window_scalar_extent() as i64)
}

// ---- #926 slice 2f: the B2 driver-table accessors (the twins of the single-shot `svm_onramp_tierup_*`
// set) the JS host rebuilds its shared `WebAssembly.Table` from at each event boundary. ----

/// The dispatch-table size (log2) the coop run's shared table is built with — the JS host sizes its
/// `WebAssembly.Table` to `1 << this` (the emitted `call_indirect` mask). `0` (a 1-slot table) for a
/// non-shimmable guest, which emits in local-table mode and never dispatches through the shared table.
#[no_mangle]
pub extern "C" fn svm_coop_table_log2() -> u32 {
    // SAFETY: single-threaded wasm; read of the session's host.
    unsafe { (*core::ptr::addr_of_mut!(COOP_RUN)).as_mut() }
        .map_or(0, |s| s.run.host_mut().jit_table_log2() as u32)
}

/// The guest program's function count — the dispatch table's **natural prefix** (`slot i < nfuncs`
/// dispatches program function `i`; slots at or past it hold installed §22 units or trap empty).
#[no_mangle]
pub extern "C" fn svm_coop_nfuncs() -> usize {
    unsafe { (*core::ptr::addr_of!(COOP_RUN)).as_ref() }.map_or(0, |s| s.sigs.len())
}

/// The §22 code handle installed at dispatch-table `slot` (`-1` empty/natural) — the mirror the JS
/// host rebuilds its table from (from [`bytecode::CoopRun::slot_code`], since coop install happens
/// inside the pump).
#[no_mangle]
pub extern "C" fn svm_coop_slot_code(slot: u32) -> i32 {
    unsafe { (*core::ptr::addr_of!(COOP_RUN)).as_ref() }.map_or(-1, |s| s.run.slot_code(slot))
}

/// Emitted-wasm length for **any** compiled unit by code handle (`0` if none — the unit is
/// interpreter-only), so the JS host can instantiate an *installed* slot's unit it hasn't itself
/// invoked. The bytes (via [`svm_coop_jit_wasm_by_handle_ptr`]) stay valid until the next call.
#[no_mangle]
pub extern "C" fn svm_coop_jit_wasm_by_handle_len(code: i32) -> usize {
    // SAFETY: single-threaded wasm; exclusive access to the session.
    let Some(s) = (unsafe { (*core::ptr::addr_of_mut!(COOP_RUN)).as_mut() }) else {
        return 0;
    };
    let h = s.run.host_mut();
    s.jit_wasm_by_handle = h
        .resolve_jit_code(code)
        .ok()
        .and_then(|(cd, cu)| h.jit_unit_wasm(cd, cu));
    s.jit_wasm_by_handle.as_ref().map_or(0, |w| w.len())
}

/// Pointer to the emitted wasm the last [`svm_coop_jit_wasm_by_handle_len`] resolved.
#[no_mangle]
pub extern "C" fn svm_coop_jit_wasm_by_handle_ptr() -> *const u8 {
    unsafe { (*core::ptr::addr_of!(COOP_RUN)).as_ref() }.map_or(core::ptr::null(), |s| {
        s.jit_wasm_by_handle
            .as_ref()
            .map_or(core::ptr::null(), |w| w.as_ptr())
    })
}

/// Generate the **bounce-shim module** for dispatch-table `slot` — a standalone one-function wasm
/// module (`export "t"`, [`svm_wasm_jit::emit_slot_trampoline`]) with the slot occupant's env-prepended
/// signature, whose body bounces to [`svm_coop_call_interp`] with `slot` baked in. The JS host
/// `table.set`s its instance's `"t"` into the slot, so an emitted `call_indirect` to an
/// interpreter-resident target lands on the live-state bounce. Returns the module's byte length
/// (`0` = no shim: empty slot, or a signature the transport can't carry — the open-time `all_shimmable`
/// gate makes the latter unreachable for a run whose units emit). Bytes via [`svm_coop_shim_ptr`],
/// valid until the next call.
#[no_mangle]
pub extern "C" fn svm_coop_shim_wasm(slot: u32) -> usize {
    // SAFETY: single-threaded wasm; exclusive access to the session.
    let Some(s) = (unsafe { (*core::ptr::addr_of_mut!(COOP_RUN)).as_mut() }) else {
        return 0;
    };
    let sig = if (slot as usize) < s.sigs.len() {
        Some(s.sigs[slot as usize].clone())
    } else {
        let code = s.run.slot_code(slot);
        if code < 0 {
            None
        } else {
            let h = s.run.host_mut();
            h.resolve_jit_code(code)
                .ok()
                .and_then(|(cd, cu)| h.jit_unit_funcs(cd, cu))
                .and_then(|fs| fs.first().map(|f| (f.params.clone(), f.results.clone())))
        }
    };
    let Some((params, results)) = sig else {
        s.shim_wasm.clear();
        return 0;
    };
    let shared = TIERUP_UNIT_SHARED.load(std::sync::atomic::Ordering::Relaxed);
    match svm_wasm_jit::emit_slot_trampoline(&params, &results, slot, shared) {
        Ok(w) => {
            s.shim_wasm = w;
            s.shim_wasm.len()
        }
        Err(_) => {
            s.shim_wasm.clear();
            0
        }
    }
}

/// Pointer to the shim module the last [`svm_coop_shim_wasm`] generated.
#[no_mangle]
pub extern "C" fn svm_coop_shim_ptr() -> *const u8 {
    unsafe { (*core::ptr::addr_of!(COOP_RUN)).as_ref() }
        .map_or(core::ptr::null(), |s| s.shim_wasm.as_ptr())
}

/// Close the open cooperative run, freeing its window. Idempotent. `CoopRun` owns everything it
/// borrows (no leaked program), so dropping the session frees it all.
#[no_mangle]
pub extern "C" fn svm_coop_close() {
    // SAFETY: single-threaded wasm; take + drop the session.
    unsafe {
        *core::ptr::addr_of_mut!(COOP_RUN) = None;
    }
}
