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
  try {
    // f0(win, env, ...cap-handle slots) — runs `_start` (→ main) to completion on emitted wasm.
    f0(win, env, ...slots);
  } catch {
    // A cross-tier `exit`/trap unwound `f0` (expected for a guest that calls exit); the finish status
    // and `svm_onramp_jit_run_trap_len` distinguish a clean exit from a real trap.
  }
  ex.svm_dealloc(env, envBytes);
  const status = ex.svm_onramp_jit_run_finish(); // capture stdout/stderr/exit into the shared slots
  ex.svm_onramp_jit_run_close();
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
  const opened = ex.svm_onramp_jit_run_open(modP, moduleBytes.length, stdinP, stdinLen);
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
