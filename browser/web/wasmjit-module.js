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

// Slice-1 browser cache (WASM_AOT.md): a cross-Run cache of the compiled `WebAssembly.Module`, keyed
// by a caller-supplied **stable module identity** (a URL / asset name — the same content key the
// native `svm_run::CompiledCache` uses, here supplied by the caller so no per-Run hashing is needed).
// The emitted `_start` is a pure function of the guest module (stdin/source are guest *input*, fed
// through memfs/stdin, never baked into the code), so the compiled Module is reused verbatim on every
// later Run of the same module — skipping `WebAssembly.compile` (V8 codegen). Only *code* is cached: a
// fresh instance, window, and env cell are built per Run, so no guest state crosses Runs. A missing
// key (undefined) disables caching for that call. Bounded so a session that Runs many distinct modules
// can't grow it without limit.
const jitModuleCache = new Map();
const JIT_MODULE_CACHE_MAX = 16;
function cacheGet(key) {
  return key === undefined ? undefined : jitModuleCache.get(key);
}
function cachePut(key, mod) {
  if (key === undefined) return;
  // Simple LRU-ish bound: drop the oldest insertion when full (Map preserves insertion order).
  if (jitModuleCache.size >= JIT_MODULE_CACHE_MAX && !jitModuleCache.has(key)) {
    jitModuleCache.delete(jitModuleCache.keys().next().value);
  }
  jitModuleCache.set(key, mod);
}
// A parallel cross-Run cache of the **instantiated** emitted module (issue #803). The compiled Module
// skips V8 codegen; caching the instance additionally skips the per-Run `WebAssembly.instantiate` of the
// ~12.7 MB emitted module *and* the copy of its bytes out of wasm memory (only read on a compile miss).
// Safe because the emitted entry `f0(win, env, ...slots)` takes win/env/sp as **arguments** and holds no
// state between calls — the reactor already reuses one instance across every frame the same way — and its
// linear memory is the imported (stable) engine memory. Keyed by the same stable module identity as
// `jitModuleCache`; a fresh `win`/`sp` and the dynamic cross-tier bounce are supplied per Run by the
// caller, so a reused instance always runs against current state.
const jitInstanceCache = new Map();
// Test/telemetry hook: how many WebAssembly.compile calls the cache has served vs skipped.
export const jitCacheStats = { compiles: 0, hits: 0 };
export function jitCacheClear() {
  jitModuleCache.clear();
  jitInstanceCache.clear();
  jitCacheStats.compiles = 0;
  jitCacheStats.hits = 0;
}

// Get-or-build the emitted module's instance for `cacheKey`, reusing it across Runs. `readEmitted` is
// invoked **only on a compile miss** (so a hit skips the ~12.7 MB byte copy), and `callInterp` becomes
// the instance's `env.call_interp`. Returns the emitted `f0`. A `hits`/`compiles` bump mirrors the
// Module-cache accounting (an instance hit is a compile skip). `cacheKey === undefined` disables caching.
async function cachedInstanceF0(memory, cacheKey, readEmitted, callInterp) {
  const cached = cacheKey === undefined ? undefined : jitInstanceCache.get(cacheKey);
  if (cached) {
    jitCacheStats.hits++;
    return cached.f0;
  }
  let module = cacheGet(cacheKey);
  if (module === undefined) {
    module = await WebAssembly.compile(readEmitted());
    cachePut(cacheKey, module);
    jitCacheStats.compiles++;
  }
  const instance = await WebAssembly.instantiate(module, {
    env: { memory, trap: () => {}, call_interp: callInterp },
  });
  const f0 = instance.exports.f0;
  if (typeof f0 !== 'function') throw new Error('emitted module has no f0 export');
  if (cacheKey !== undefined) {
    if (jitInstanceCache.size >= JIT_MODULE_CACHE_MAX && !jitInstanceCache.has(cacheKey)) {
      jitInstanceCache.delete(jitInstanceCache.keys().next().value);
    }
    jitInstanceCache.set(cacheKey, { instance, f0 });
  }
  return f0;
}

// Drive an already-opened single-shot JIT run to completion: get the emitted `_start`'s instance (compiled
// + instantiated against the cdylib's shared `memory`, reused across Runs) and call `f0(win, env, ...slots)`
// once. Returns the finish status. The caller must have opened the run (`svm_onramp_jit_run_open*`) already.
// `cacheKey` (optional) is a stable identity of the guest module; when given, the compiled Module and its
// instance are reused across Runs (see `cachedInstanceF0`).
async function driveJitRun(ex, memory, cacheKey) {
  const u8 = () => new Uint8Array(memory.buffer);
  // Read the window base + the powerbox handle slots `_start` takes as params, and the env-cell size.
  const win = Number(ex.svm_onramp_jit_run_win_ptr());
  const envBytes = ex.svm_onramp_jit_run_env_bytes();
  const slots = [];
  for (let i = 0, n = ex.svm_onramp_jit_run_slot_count(); i < n; i++) {
    slots.push(ex.svm_onramp_jit_run_slot(i));
  }

  // `env.call_interp` relays each cross-tier call to the cdylib; a nonzero status (exit/trap) throws to
  // unwind the emitted `f0` (the browser's JS import model — `Exit` and real traps both caught below).
  // Reuse the compiled Module **and** the instance across Runs of the same guest module (issue #803):
  // the emitted bytes are copied out of wasm memory only on a compile miss, and one instance serves every
  // Run. Safe even though this path re-opens a fresh window each Run — `win` is passed to `f0` per Run,
  // and the cross-tier bounce routes to the current cdylib run, so a reused instance runs against current
  // state (the warm path reuses the same way, and the reactor reuses one instance across every frame).
  let f0;
  try {
    f0 = await cachedInstanceF0(
      memory,
      cacheKey,
      () => {
        // Copy the emitted bytes out (a later svm_alloc could move the stash).
        const wptr = Number(ex.svm_onramp_jit_run_wasm_ptr());
        const wlen = ex.svm_onramp_jit_run_wasm_len();
        return u8().slice(wptr, wptr + wlen);
      },
      (func, argsPtr) => {
        if (ex.svm_onramp_jit_run_call_interp(func, argsPtr) !== 0) throw new Error('cross-tier stop');
      },
    );
  } catch (e) {
    ex.svm_onramp_jit_run_close();
    throw e;
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

// Marshal one i64 slot to the wasm type an emitted unit's `f0` declares, and a JS return back to its
// i64 slot — worker.js's `jitArg`/`jitRes` twins for the single-shot pump (#835). Type codes:
// 0 = i32 (JS Number), 1 = i64 (BigInt), 2 = f32, 3 = f64 (Numbers via the slot's float bits).
const f64buf = new DataView(new ArrayBuffer(8));
const tierupJitArg = (slot, tc) => tc === 0 ? Number(BigInt.asIntN(32, slot))
  : tc === 1 ? slot
  : (f64buf.setBigInt64(0, slot, true), tc === 2 ? f64buf.getFloat32(0, true) : f64buf.getFloat64(0, true));
const tierupJitRes = (ret, tc) => tc === 0 || tc === 1 ? BigInt(ret)
  : (tc === 2 ? f64buf.setFloat32(0, ret, true) : f64buf.setFloat64(0, ret, true), f64buf.getBigInt64(0, true));

// Drive an already-opened **leaf tier-up** run (#809) — the InterpDriven complement to `driveJitRun`:
// the interpreter owns `_start` (it `vm_map`s / does host I/O, so the whole-program emit declined), and
// each tier-up-eligible pure leaf surfaces as a TIERUP event serviced here on the emitted module. The
// per-event contract (worker.js's PAR_TIERUP, single-shot edition): re-arm the emitted `"fuel"` global,
// write the event's committed-extent snapshot to `"mapped"` (#717 host sync), call
// `f{func}(win, env, ...i64 args)` over the cdylib's shared memory (the run window lives inside it — no
// copies), and deliver the i64 result slots (or a trap) back to the parked vCPU. On DONE the cdylib has
// staged stdout/stderr/exit/value into the usual shared slots. A trapped run throws so the caller falls
// back to the interpreter oracle (INVARIANT 9 — diverge toward refusal).
//
// A `vm_jit_*`-importing guest (#835, the JACL compiler shape) additionally surfaces JIT_INVOKE
// events: a guest-compiled §22 unit runs here as `f0(win, env, ...args)` on its own emitted wasm
// (instantiated once per code handle — worker.js's `jitUnitFor` shape), args/results marshalled by
// the event's scalar type codes, with the same per-call `"mapped"` sync. The engine only surfaces
// invokes whose unit emitted and whose window state is representable; everything else it services
// on the interpreter itself, so this handler never has to decline.
async function driveTierupRun(ex, memory, cacheKey) {
  const u8 = () => new Uint8Array(memory.buffer);
  const i64 = () => new BigInt64Array(memory.buffer);
  const win = Number(ex.svm_onramp_tierup_win_ptr());

  // Compile + instantiate the emitted leaves once per Run, reused across every TIERUP. The compiled
  // Module is cached across Runs under a key distinct from the whole-program `_start` emit (a different
  // artifact: same guest, leaf-only exports). No instance cache: the eligible-leaf set is per-open, and
  // a Run holds its instance for all its events anyway. An emitted leaf never crosses tiers (eligibility
  // excludes cap-calls and calls into unemitted code), so `call_interp` throwing is fail-closed.
  const tierupKey = cacheKey === undefined ? undefined : `${cacheKey}#tierup`;
  let module = cacheGet(tierupKey);
  if (module === undefined) {
    const wptr = Number(ex.svm_onramp_tierup_wasm_ptr());
    const wlen = ex.svm_onramp_tierup_wasm_len();
    module = await WebAssembly.compile(u8().slice(wptr, wptr + wlen));
    cachePut(tierupKey, module);
    jitCacheStats.compiles++;
  } else {
    jitCacheStats.hits++;
  }
  const instance = await WebAssembly.instantiate(module, {
    env: {
      memory,
      trap: () => {},
      call_interp: () => { throw new Error('unexpected cross-tier call from an emitted leaf'); },
    },
  });
  const emitted = instance.exports;
  const envCell = Number(ex.svm_alloc(ex.svm_wasmjit_env_bytes()));

  // ---- #846: the linked-unit driver — one shared funcref table + the live-state bounce --------
  // §22 units emit in Model B2 shape: their `call_indirect` dispatches through this table, whose
  // slots the driver populates from the engine's mirror at each event boundary — an installed
  // unit's emitted `f0`, an eligible program function's `f{i}` from the main emitted module, or a
  // **bounce shim** (engine-generated trampoline) for interpreter-resident targets. The bounce
  // (`env.call_interp`) runs the target on a nested interpretation over the run's LIVE
  // window/powerbox/fuel, then fans the fresh `"mapped"` extent out to every live instance (a
  // callback may have vm_map-grown the window mid-invoke). Proven observably identical to
  // `onramp_exec` by tests/tierup_driver.rs's `TierupDriver` (wasmi playing this file's role).
  const tsize = 1 << ex.svm_onramp_tierup_table_log2();
  const table = new WebAssembly.Table({ initial: tsize, maximum: tsize, element: 'anyfunc' });
  const mappedGlobals = []; // every live instance's "mapped" — the post-bounce fan-out set
  const fuelGlobals = [];
  const registerGlobals = (exports) => {
    if (exports.mapped) mappedGlobals.push(exports.mapped);
    if (exports.fuel) fuelGlobals.push(exports.fuel);
  };
  registerGlobals(emitted);
  const unitImports = () => ({ env: {
    memory,
    __indirect_function_table: table,
    trap: () => {},
    call_interp: (target, argsPtr) => {
      const rc = ex.svm_onramp_tierup_call_interp(target, argsPtr);
      // #717 fan-out: growth a bounced callback made is visible from the next instruction on.
      const now = ex.svm_onramp_tierup_mapped_now();
      for (const g of mappedGlobals) g.value = now;
      if (rc !== 0) throw new Error('bounce trap'); // unwind to the deliver below
    },
  } });
  // Per-code-handle unit instances and per-(slot, occupant) bounce shims; per-Run only. Async
  // instantiation (the drive loop awaits it) — a macro unit can exceed the main thread's sync
  // compile allowance.
  const jitUnits = new Map();
  const shims = new Map();
  const instantiateUnit = async (bytes) => {
    const inst = await WebAssembly.instantiate(await WebAssembly.compile(bytes), unitImports());
    registerGlobals(inst.exports);
    return inst.exports;
  };
  const shimFor = async (slot, code) => {
    const key = `${slot}#${code}`;
    let f = shims.get(key);
    if (f === undefined) {
      const len = ex.svm_onramp_tierup_shim_wasm(slot);
      if (len === 0) return null;
      const bytes = u8().slice(Number(ex.svm_onramp_tierup_shim_ptr()), Number(ex.svm_onramp_tierup_shim_ptr()) + len);
      f = (await instantiateUnit(bytes))['t'];
      shims.set(key, f);
    }
    return f;
  };
  const unitFor = async (code, bytes) => {
    let unit = jitUnits.get(code);
    if (unit === undefined) {
      unit = await instantiateUnit(bytes);
      jitUnits.set(code, unit);
    }
    return unit;
  };
  // Rebuild the table from the engine's slot mirror. Installs only happen between events (a unit
  // with a cap.call never emits), so an event-boundary sync is exact for the whole event.
  const nfuncs = ex.svm_onramp_tierup_nfuncs();
  const syncTable = async () => {
    for (let slot = 0; slot < tsize; slot++) {
      let entry = null;
      if (slot < nfuncs) {
        entry = emitted['f' + slot] ?? await shimFor(slot, -2);
      } else {
        const code = ex.svm_onramp_tierup_slot_code(slot);
        if (code >= 0) {
          const len = ex.svm_onramp_tierup_jit_wasm_by_handle_len(code);
          entry = len > 0
            ? (await unitFor(code, u8().slice(Number(ex.svm_onramp_tierup_jit_wasm_by_handle_ptr()),
                                              Number(ex.svm_onramp_tierup_jit_wasm_by_handle_ptr()) + len)))['f0']
            : await shimFor(slot, code);
        }
      }
      table.set(slot, entry);
    }
  };

  try {
    for (;;) {
      const ev = ex.svm_onramp_tierup_run();
      if (ev === 3 /* TIERUP_RUN_JIT_INVOKE */) {
        // A guest-compiled §22 unit with emitted wasm: sync the table, instantiate once per code
        // handle, then `f0(win, env, ...args)` with the per-event `"mapped"`/fuel sync fanned out
        // to every live instance (a unit's table dispatch may run any of them).
        await syncTable();
        const code = ex.svm_onramp_tierup_jit_code();
        const unit = await unitFor(code, u8().slice(Number(ex.svm_onramp_tierup_jit_wasm_ptr()),
                                                    Number(ex.svm_onramp_tierup_jit_wasm_ptr()) + ex.svm_onramp_tierup_jit_wasm_len()));
        const argvPtr = Number(ex.svm_onramp_tierup_argv_ptr());
        const n = ex.svm_onramp_tierup_argv_len();
        const ptypes = new Uint8Array(memory.buffer, Number(ex.svm_onramp_tierup_jit_param_types_ptr()), n);
        const args = [];
        for (let i = 0; i < n; i++) args.push(tierupJitArg(i64()[(argvPtr >> 3) + i], ptypes[i]));
        const mapped = ex.svm_onramp_tierup_mapped(); // #717 host sync, as TIERUP below
        for (const g of mappedGlobals) g.value = mapped;
        for (const g of fuelGlobals) g.value = 1n << 61n;
        new DataView(memory.buffer).setBigInt64(envCell, 1n << 61n, true);
        try {
          const ret = unit['f0'](win, envCell, ...args);
          const rets = ret === undefined ? [] : Array.isArray(ret) ? ret : [ret];
          const rn = ex.svm_onramp_tierup_jit_result_types_len();
          const rtypes = new Uint8Array(memory.buffer, Number(ex.svm_onramp_tierup_jit_result_types_ptr()), rn);
          const rlen = Math.max(1, rets.length) * 8;
          const rptr = Number(ex.svm_alloc(rlen));
          for (let i = 0; i < rets.length; i++) i64()[(rptr >> 3) + i] = tierupJitRes(rets[i], rtypes[i]);
          ex.svm_onramp_tierup_deliver_jit(rptr, rets.length);
          ex.svm_dealloc(rptr, rlen);
        } catch {
          // A wasm trap — or a bounce callback's trap/exit, which the engine staged and
          // `deliver_jit_trap` resolves as the real trap kind (an exit ends the run as EXIT).
          ex.svm_onramp_tierup_deliver_jit_trap();
        }
        continue;
      }
      if (ev !== 1 /* TIERUP_RUN_TIERUP */) break; // 0 = done (slots staged), 2 = trapped (status 3)
      const func = ex.svm_onramp_tierup_func();
      const argvPtr = Number(ex.svm_onramp_tierup_argv_ptr());
      const n = ex.svm_onramp_tierup_argv_len();
      const args = [];
      for (let i = 0; i < n; i++) args.push(i64()[(argvPtr >> 3) + i]);
      // #717 host sync: the event's committed extent → the emitted `"mapped"` global, so the bounds
      // check admits exactly what the interpreter's page map does for this call.
      emitted.mapped.value = ex.svm_onramp_tierup_mapped();
      emitted.fuel.value = 1n << 61n; // re-arm the fuel global across events on the reused instance
      new DataView(memory.buffer).setBigInt64(envCell, 1n << 61n, true);
      try {
        const ret = emitted['f' + func](win, envCell, ...args);
        const rets = ret === undefined ? [] : Array.isArray(ret) ? ret : [ret];
        const rlen = Math.max(1, rets.length) * 8;
        const rptr = Number(ex.svm_alloc(rlen));
        for (let i = 0; i < rets.length; i++) i64()[(rptr >> 3) + i] = BigInt(rets[i]);
        ex.svm_onramp_tierup_deliver(rptr, rets.length);
        ex.svm_dealloc(rptr, rlen);
      } catch {
        ex.svm_onramp_tierup_deliver_trap();
      }
    }
  } finally {
    ex.svm_dealloc(envCell, ex.svm_wasmjit_env_bytes());
    ex.svm_onramp_tierup_close();
  }
  const status = ex.svm_status();
  if (status === 3 /* STATUS_TRAP */) throw new Error('tier-up run trapped (declined to the interpreter)');
  return status;
}

// Run an on-ramp module whose input is **stdin** (Lua/SQLite/hello) on the wasm-JIT.
export async function runJitModule(ex, memory, moduleBytes, stdinBytes, cacheKey) {
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
  // `_start` not whole-program-emittable (an InterpDriven guest — it `vm_map`s, streams, …): try the
  // leaf tier-up run (#809) before giving the buffers up — the interpreter drives `_start`, its
  // eligible pure leaves run on emitted wasm, and a `vm_jit_*` guest's runtime-compiled §22 units run
  // emitted too (#835). Refused (nothing emittable ever / threads/futex) → fall through to the throw
  // and the caller's plain-interpreter fallback.
  let tierup = false;
  if (opened !== 0 && ex.svm_onramp_tierup_open &&
      ex.svm_onramp_tierup_open(modP, moduleBytes.length, stdinP, stdinLen, 1) === 0) {
    tierup = true;
  }
  ex.svm_dealloc(modP, moduleBytes.length);
  if (stdinP) ex.svm_dealloc(stdinP, stdinLen);
  if (tierup) return driveTierupRun(ex, memory, cacheKey);
  if (opened !== 0) {
    throw new Error(`JIT module open failed: status ${ex.svm_status()} (2 = _start not emittable)`);
  }
  return driveJitRun(ex, memory, cacheKey);
}

// Run the warm session's `eval_run` on the **warm+JIT** tier (WASM_AOT.md warm+JIT). The warm snapshot
// (`svm_warm_open`) has already paid the QuickJS runtime init once; this evaluates the user's code on
// emitted wasm over the restored warm image, so a compute-heavy program runs the eval near-native while
// init stays paid-once. The engine emits `eval_run` on the first Run and caches it (a warm+JIT Run never
// re-pays the cdylib emit); the compiled `WebAssembly.Module` is cached across Runs too (keyed by
// `cacheKey`). Differs from `driveJitRun` only in the accessors it drives and in passing the entry `sp`
// as the emitted `f0`'s third argument (an i64 slot ⇒ a BigInt). Assumes `svm_warm_open` already
// succeeded for this module. Returns the run status (0 = returned, 5 = exited); throws if `eval_run`
// isn't wasm-drivable or the run traps (the caller falls back to the interpreter warm path).
export async function runWarmJit(ex, memory, stdinBytes, cacheKey, shared = 1) {
  const u8 = () => new Uint8Array(memory.buffer);
  // Emit `eval_run` (idempotent — cached in the warm session after the first Run).
  if (ex.svm_warm_jit_open(shared) !== 0) {
    throw new Error(`warm-JIT open failed: status ${ex.svm_status()} (2 = eval_run not emittable)`);
  }
  // Per-Run: restore the warm image + reset the run's powerbox, feeding the editor text as stdin.
  let stdinP = 0;
  const stdinLen = stdinBytes ? stdinBytes.length : 0;
  if (stdinLen) {
    stdinP = Number(ex.svm_alloc(stdinLen));
    u8().set(stdinBytes, stdinP);
  }
  const prepared = ex.svm_warm_jit_prepare(stdinP, stdinLen);
  if (stdinP) ex.svm_dealloc(stdinP, stdinLen);
  if (prepared !== 0) throw new Error(`warm-JIT prepare failed: status ${ex.svm_status()}`);

  const win = Number(ex.svm_warm_jit_win_ptr());
  const sp = ex.svm_warm_jit_entry_sp(); // i64 export ⇒ BigInt; passed straight as f0's i64 slot
  const envBytes = ex.svm_onramp_jit_run_env_bytes();

  // Reuse the instance across Runs (issue #803): a hit skips both the byte copy and instantiate, so a
  // warm Run collapses to `prepare` + the eval. The emit is stable and window-independent (`win`/`sp` are
  // passed per Run), so one instance serves every Run of this warm session.
  const f0 = await cachedInstanceF0(
    memory,
    cacheKey,
    () => {
      const wptr = Number(ex.svm_warm_jit_wasm_ptr());
      const wlen = ex.svm_warm_jit_wasm_len();
      return u8().slice(wptr, wptr + wlen);
    },
    (func, argsPtr) => {
      if (ex.svm_warm_jit_call_interp(func, argsPtr) !== 0) throw new Error('cross-tier stop');
    },
  );

  const env = Number(ex.svm_alloc(envBytes));
  new DataView(memory.buffer).setBigInt64(env, 1n << 60n, true); // huge dispatcher-fuel budget
  let threw = 0;
  let value = 0n;
  let trapError = null;
  try {
    // f0(win, env, sp) — runs `eval_run(sp)` over the restored warm image on emitted wasm. Its return is
    // the guest's top-level result (an i32/i64); normalize to i64.
    const r = f0(win, env, sp);
    value = r === undefined || r === null ? 0n : BigInt(r);
  } catch (e) {
    threw = 1;
    trapError = e; // keep it — the trap kind + wasm location live here (issue #865)
  }
  ex.svm_dealloc(env, envBytes);
  ex.svm_warm_jit_report(threw, value);
  const status = ex.svm_warm_jit_finish();
  if (status === 3 /* STATUS_TRAP */) throw warmJitTrapError(trapError);
  return status;
}

// The last warm+JIT trap, captured for diagnosis (issue #865) — `{ kind, frames }` where `frames` are the
// emitted wasm frames `fN@0xoff` (innermost first), or `null` if the last run didn't trap. Test/telemetry
// hook: a decline used to be a bare "trapped" with no location; this exposes the trap KIND (the V8
// RuntimeError message, e.g. "null function or function signature mismatch" = a `call_indirect` to a
// null/mismatched table slot) and WHERE (which emitted functions), the way the bytecode tier reports.
export let lastWarmJitTrap = null;

// Build a diagnosable decline error from the caught `f0` trap. A wasm-level trap is a `WebAssembly.
// RuntimeError` whose message is the trap kind and whose stack carries `wasm-function[N]:0xoff` frames
// (the emitted function index + byte offset). Our own cross-tier unwind (`env.call_interp` returned
// nonzero → we threw 'cross-tier stop') has no wasm location — the real trap is on the interpreter side,
// recorded by `svm_warm_jit_call_interp`'s `last_trap`.
function warmJitTrapError(e) {
  if (e instanceof WebAssembly.RuntimeError) {
    const frames = String(e.stack || '')
      .split('\n')
      .map((l) => l.match(/wasm-function\[(\d+)\]:0x([0-9a-fA-F]+)/))
      .filter(Boolean)
      .map((m) => `f${m[1]}@0x${m[2]}`);
    lastWarmJitTrap = { kind: e.message, frames };
    const where = frames.length ? ` at ${frames[0]}${frames.length > 1 ? ` (from ${frames.slice(1, 6).join(' ← ')})` : ''}` : '';
    return new Error(`emitted warm run trapped: ${e.message}${where}`);
  }
  lastWarmJitTrap = { kind: (e && e.message) || 'cross-tier stop', frames: [] };
  return new Error(`emitted warm run trapped (declined to the interpreter): ${(e && e.message) || e}`);
}

// **Pre-warm** the warm+JIT `eval_run` for `cacheKey` — emit + `WebAssembly.compile` + instantiate **and
// dry-run it once** (empty input, over the restored image), so the first real `runWarmJit` is instant.
// Called during pre-warm (off the main thread), so all of that cost is hidden.
//
// Why the dry run matters (and why compile+instantiate alone did not): V8 compiles a wasm module's
// **function bodies lazily — on first call**, not during `WebAssembly.compile`. So caching the compiled
// Module + instance still left the *first* `f0()` call paying ~1.5 s of function compilation on the
// user's first Run (measured: run1 4.7 s vs run2 2.3 s even with the instance primed). Making one `f0`
// call here forces that compilation now. Empty input is language-agnostic (QuickJS/Lua) and still enters
// the interpreter, warming the hot functions; the next real Run does `svm_warm_jit_prepare` (restores the
// image + resets the powerbox), so the dry run leaves no state and fresh-per-Run isolation holds.
//
// Best-effort: returns false if `eval_run` isn't wasm-drivable or the dry run traps (the card then just
// uses warm-interp), true once compiled + warmed.
export async function primeWarmJit(ex, memory, cacheKey, shared = 1) {
  try {
    // A full dry Run with no stdin: opens/emits `eval_run`, compiles + instantiates (cached under
    // `cacheKey`), and — the point — makes the first `f0` call so V8 compiles the bodies now.
    await runWarmJit(ex, memory, null, cacheKey, shared);
    return true;
  } catch {
    return false; // eval_run not emittable / trapped → card stays warm-interp
  }
}

// Run the **chibicc compiler** on the wasm-JIT: feed it the user's C `srcBytes` (seeded at `/in.c`) plus
// the built-in libc headers under `/include`, and emit its `_start`. The cdylib assembles the memfs +
// argv (`svm_onramp_jit_run_open_fs`, sharing the bytecode card's `chibicc_card_image`), so this driver
// just hands over the module + source. The emitted SVM-IR comes back on `svm_stdout_*` after finish.
export async function runJitCompiler(ex, memory, moduleBytes, srcBytes, debugInfo = 0, cacheKey) {
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
  return driveJitRun(ex, memory, cacheKey);
}

// Run the **self-host** compile on the wasm-JIT (SELFHOST_C.md §7 step 5): chibicc.svmb compiles one of
// its own cc1 TUs (`tuBytes`, a memfs-relative path like `frontend/chibicc/hashmap.c`) to a linkable
// object, reading the TU + its glibc header closure from `imgBytes` (the committed closure image). Same
// shape as `runJitCompiler` but through `svm_selfhost_jit_emit_object_fs` (raw image + `--emit-object`
// argv, 128 MiB window for the giants). The emitted object text comes back on `svm_stdout_*` after finish.
export async function runJitSelfhost(ex, memory, moduleBytes, imgBytes, tuBytes, debugInfo = 0, cacheKey) {
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
  return driveJitRun(ex, memory, cacheKey);
}
