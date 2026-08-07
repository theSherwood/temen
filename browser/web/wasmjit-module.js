// Single-shot **wasm-JIT module runner** — the run-to-completion twin of `wasmjit-reactor.js` (which
// drives a `tick` per frame). An on-ramp module's whole program is func 0 (`_start`); the cdylib emits
// it and this module compiles + runs `f0(win, env, ...slots)` **once** against the cdylib's own shared
// linear memory, servicing the cross-tier helpers through `env.call_interp`. After the run,
// `svm_onramp_jit_run_finish` captures stdout/stderr/exit into the shared slots — read them via the
// usual `svm_stdout_*` / `svm_exit_code` accessors, exactly like the interpreter `svm_run_onramp` path.
//
// Two openers share the same drive loop: `runJitModule` feeds the guest from stdin (Lua/SQLite/hello),
// `runJitCompiler` feeds it from a seeded memfs + argv (the chibicc card: `/in.c` + `/include/*.h`).
// Both return the run status (0 = returned, 5 = exited) or throw if `_start` isn't emittable (the caller
// falls back to the interpreter). Synchronous guest work; only the initial `WebAssembly.compile` is async.

// Drive an already-opened single-shot JIT run to completion: copy the emitted `_start` out, compile +
// instantiate it against the cdylib's shared `memory`, and call `f0(win, env, ...slots)` once. Returns
// the finish status. The caller must have opened the run (`svm_onramp_jit_run_open*`) already.
async function driveJitRun(ex, memory) {
  const u8 = () => new Uint8Array(memory.buffer);
  // Copy the emitted bytes out (a later svm_alloc could move the stash), read the window base + the
  // powerbox handle slots `_start` takes as params, and the env-cell size.
  const wptr = Number(ex.svm_onramp_jit_run_wasm_ptr());
  const wlen = ex.svm_onramp_jit_run_wasm_len();
  const emitted = u8().slice(wptr, wptr + wlen);
  const win = Number(ex.svm_onramp_jit_run_win_ptr());
  const envBytes = ex.svm_onramp_jit_run_env_bytes();
  const slots = [];
  for (let i = 0, n = ex.svm_onramp_jit_run_slot_count(); i < n; i++) {
    slots.push(ex.svm_onramp_jit_run_slot(i));
  }

  // `env.call_interp` relays each cross-tier call to the cdylib; a nonzero status (exit/trap) throws to
  // unwind the emitted `f0` (the browser's JS import model — `Exit` and real traps both caught below).
  const module = await WebAssembly.compile(emitted);
  const instance = await WebAssembly.instantiate(module, {
    env: {
      memory,
      trap: () => {},
      call_interp: (func, argsPtr) => {
        if (ex.svm_onramp_jit_run_call_interp(func, argsPtr) !== 0) throw new Error('cross-tier stop');
      },
    },
  });
  const f0 = instance.exports.f0;
  if (typeof f0 !== 'function') {
    ex.svm_onramp_jit_run_close();
    throw new Error('emitted module has no f0 export');
  }

  const env = Number(ex.svm_alloc(envBytes));
  new DataView(memory.buffer).setBigInt64(env, 1n << 60n, true); // huge dispatcher-fuel budget
  // Capture how `f0` finished so the runner reports it with parity to the interpreter: its return value
  // (the guest's top-level result) when it returns, and whether it *threw*. The cdylib pairs `threw` with
  // its own `exited` flag (set on a cross-tier `Exit`) — a throw that didn't exit is a trap, so the run
  // reports STATUS_TRAP instead of a truncated STATUS_OK (INVARIANT 9).
  let threw = 0;
  let value = 0n;
  try {
    // f0(win, env, ...cap-handle slots) — runs `_start` (→ main) to completion on emitted wasm. Its
    // return is the guest's result (an i32/i64, or undefined for a void `_start`); normalize to i64.
    const r = f0(win, env, ...slots);
    value = r === undefined || r === null ? 0n : BigInt(r);
  } catch {
    // The emitted `f0` unwound — a cross-tier `exit` (expected for a guest that calls exit) or a trap
    // (a wasm `unreachable` / a cross-tier bounce that trapped). `svm_onramp_jit_run_finish` tells which.
    threw = 1;
  }
  ex.svm_dealloc(env, envBytes);
  ex.svm_onramp_jit_run_report(threw, value); // record the return value + throw before capturing
  const status = ex.svm_onramp_jit_run_finish(); // capture stdout/stderr/exit/value into the shared slots
  ex.svm_onramp_jit_run_close();
  // A trap on the emitted tier is a refusal, not a result: throw so the caller runs the guest on the
  // interpreter oracle instead of surfacing a truncated run (INVARIANT 9 — diverge toward refusal).
  if (status === 3 /* STATUS_TRAP */) throw new Error('emitted run trapped (declined to the interpreter)');
  return status;
}

// Run an on-ramp module whose input is **stdin** (Lua/SQLite/hello) on the wasm-JIT.
export async function runJitModule(ex, memory, moduleBytes, stdinBytes) {
  const u8 = () => new Uint8Array(memory.buffer);
  // Hand the module (+ optional stdin) to the cdylib: decode, outline, grant powerbox, emit `_start`.
  const modP = Number(ex.svm_alloc(moduleBytes.length));
  u8().set(moduleBytes, modP);
  let stdinP = 0;
  const stdinLen = stdinBytes ? stdinBytes.length : 0;
  if (stdinLen) {
    stdinP = Number(ex.svm_alloc(stdinLen));
    u8().set(stdinBytes, stdinP);
  }
  // shared=1: this demo instantiates the emitted module against the cdylib's **shared** memory
  // (cross-origin-isolated threads build). A plain single-threaded host passes 0.
  const opened = ex.svm_onramp_jit_run_open(modP, moduleBytes.length, stdinP, stdinLen, 1);
  ex.svm_dealloc(modP, moduleBytes.length);
  if (stdinP) ex.svm_dealloc(stdinP, stdinLen);
  if (opened !== 0) {
    throw new Error(`JIT module open failed: status ${ex.svm_status()} (2 = _start not emittable)`);
  }
  return driveJitRun(ex, memory);
}

// Run the **chibicc compiler** on the wasm-JIT: feed it the user's C `srcBytes` (seeded at `/in.c`) plus
// the built-in libc headers under `/include`, and emit its `_start`. The cdylib assembles the memfs +
// argv (`svm_onramp_jit_run_open_fs`, sharing the bytecode card's `chibicc_card_image`), so this driver
// just hands over the module + source. The emitted SVM-IR comes back on `svm_stdout_*` after finish.
export async function runJitCompiler(ex, memory, moduleBytes, srcBytes, debugInfo = 0) {
  const u8 = () => new Uint8Array(memory.buffer);
  const modP = Number(ex.svm_alloc(moduleBytes.length));
  const srcP = Number(ex.svm_alloc(srcBytes.length));
  u8().set(moduleBytes, modP);
  u8().set(srcBytes, srcP);
  // Empty header image (0, 0) — the cdylib seeds the built-in playground headers itself. `debugInfo`
  // selects chibicc's `-g` debug section (off by default, matching the bytecode `svm_run_onramp_fs`).
  const opened = ex.svm_onramp_jit_run_open_fs(modP, moduleBytes.length, 0, 0, srcP, srcBytes.length, debugInfo);
  ex.svm_dealloc(modP, moduleBytes.length);
  ex.svm_dealloc(srcP, srcBytes.length);
  if (opened !== 0) {
    throw new Error(`JIT compiler open failed: status ${ex.svm_status()} (2 = _start not emittable)`);
  }
  return driveJitRun(ex, memory);
}

// Run the **self-host** compile on the wasm-JIT (SELFHOST_C.md §7 step 5): chibicc.svmb compiles one of
// its own cc1 TUs (`tuBytes`, a memfs-relative path like `frontend/chibicc/hashmap.c`) to a linkable
// object, reading the TU + its glibc header closure from `imgBytes` (the committed closure image). Same
// shape as `runJitCompiler` but through `svm_selfhost_jit_emit_object_fs` (raw image + `--emit-object`
// argv, 128 MiB window for the giants). The emitted object text comes back on `svm_stdout_*` after finish.
export async function runJitSelfhost(ex, memory, moduleBytes, imgBytes, tuBytes, debugInfo = 0) {
  const u8 = () => new Uint8Array(memory.buffer);
  const modP = Number(ex.svm_alloc(moduleBytes.length));
  const imgP = Number(ex.svm_alloc(imgBytes.length));
  const tuP = Number(ex.svm_alloc(tuBytes.length));
  u8().set(moduleBytes, modP); u8().set(imgBytes, imgP); u8().set(tuBytes, tuP);
  const opened = ex.svm_selfhost_jit_emit_object_fs(modP, moduleBytes.length, imgP, imgBytes.length, tuP, tuBytes.length, debugInfo);
  ex.svm_dealloc(modP, moduleBytes.length);
  ex.svm_dealloc(imgP, imgBytes.length);
  ex.svm_dealloc(tuP, tuBytes.length);
  if (opened !== 0) {
    throw new Error(`JIT self-host open failed: status ${ex.svm_status()} (2 = _start not emittable)`);
  }
  return driveJitRun(ex, memory);
}
