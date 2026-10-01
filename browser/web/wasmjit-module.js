// Single-shot **wasm-JIT module runner** — the run-to-completion twin of `wasmjit-reactor.js` (which
// drives a `tick` per frame). An on-ramp module's whole program is func 0 (`_start`); the cdylib emits
// it and this module compiles + runs `f0(win, env, ...slots)` **once** against the cdylib's own shared
// linear memory, servicing the cross-tier helpers through `env.call_interp`. After the run,
// `temen_onramp_jit_run_finish` captures stdout/stderr/exit into the shared slots — read them via the
// usual `temen_stdout_*` / `temen_exit_code` accessors, exactly like the interpreter `temen_run_onramp` path.
//
// Two openers share the same drive loop: `runJitModule` feeds the guest from stdin (Lua/SQLite/hello),
// `runJitCompiler` feeds it from a seeded memfs + argv (the chibicc card: `/in.c` + `/include/*.h`).
// Both return the run status (0 = returned, 5 = exited) or throw if `_start` isn't emittable (the caller
// falls back to the interpreter). Synchronous guest work; only the initial `WebAssembly.compile` is async.

// Slice-1 browser cache (WASM_AOT.md): a cross-Run cache of the compiled `WebAssembly.Module`, keyed
// by a caller-supplied **stable module identity** (a URL / asset name — the same content key the
// native `temen_run::CompiledCache` uses, here supplied by the caller so no per-Run hashing is needed).
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
// `reuseInstance = false` (#1285, a detached child): reuse the compiled Module by `cacheKey` but always
// instantiate afresh — an Instance is bound to its imported memory, and each detached child has its own.
async function cachedInstanceF0(memory, cacheKey, readEmitted, callInterp, entryName = 'f0', reuseInstance = true) {
  // The emit exports one `f{temen_idx}` per Temen function; `entryName` picks the one this run drives. The
  // single-shot `_start` path is `f0`; the warm+JIT path drives `eval_run`'s export (`f{eval_fn}`), NOT
  // `f0` (= the cold `_start`) — see `runWarmJit` (#865).
  const pick = (instance) => {
    const f = instance.exports[entryName];
    if (typeof f !== 'function') throw new Error(`emitted module has no ${entryName} export`);
    return f;
  };
  const cached = cacheKey === undefined || !reuseInstance ? undefined : jitInstanceCache.get(cacheKey);
  if (cached) {
    jitCacheStats.hits++;
    return { f0: pick(cached.instance), instance: cached.instance };
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
  if (cacheKey !== undefined && reuseInstance) {
    if (jitInstanceCache.size >= JIT_MODULE_CACHE_MAX && !jitInstanceCache.has(cacheKey)) {
      jitInstanceCache.delete(jitInstanceCache.keys().next().value);
    }
    jitInstanceCache.set(cacheKey, { instance });
  }
  return { f0: pick(instance), instance };
}

// Drive an already-opened single-shot JIT run to completion: get the emitted `_start`'s instance (compiled
// + instantiated against the cdylib's shared `memory`, reused across Runs) and call `f0(win, env, ...slots)`
// once. Returns the finish status. The caller must have opened the run (`temen_onramp_jit_run_open*`) already.
// `cacheKey` (optional) is a stable identity of the guest module; when given, the compiled Module and its
// instance are reused across Runs (see `cachedInstanceF0`).
export async function driveJitRun(ex, memory, cacheKey) {
  const u8 = () => new Uint8Array(memory.buffer);
  // Read the window base + the powerbox handle slots `_start` takes as params, and the env-cell size.
  const win = Number(ex.temen_onramp_jit_run_win_ptr());
  const envBytes = ex.temen_onramp_jit_run_env_bytes();
  const slots = [];
  for (let i = 0, n = ex.temen_onramp_jit_run_slot_count(); i < n; i++) {
    slots.push(ex.temen_onramp_jit_run_slot(i));
  }

  // `env.call_interp` relays each cross-tier call to the cdylib; a nonzero status (exit/trap) throws to
  // unwind the emitted `f0` (the browser's JS import model — `Exit` and real traps both caught below).
  // Reuse the compiled Module **and** the instance across Runs of the same guest module (issue #803):
  // the emitted bytes are copied out of wasm memory only on a compile miss, and one instance serves every
  // Run. Safe even though this path re-opens a fresh window each Run — `win` is passed to `f0` per Run,
  // and the cross-tier bounce routes to the current cdylib run, so a reused instance runs against current
  // state (the warm path reuses the same way, and the reactor reuses one instance across every frame).
  let f0, instance;
  // #1153: the emitted `"mapped"` bound, re-synced after each `vm_map`-growing bounce so a grown store
  // admits (parity with the coop tier — the single-shot on-ramp path no longer pre-sizes a fixed window).
  // #1201: a PAGED run (the guest `unmap`s/`protect`s its pages) also exports `"pagestate"` — the cdylib
  // rebuilds the page-state table after each bounce and `"mapped"` is then its coverage; re-point both
  // before `f0` and after every bounce (the `syncPaged` contract of the §14 codegen Worker / coop driver).
  let mappedGlobal = null, pagestateGlobal = null;
  const syncGlobals = () => {
    if (pagestateGlobal) pagestateGlobal.value = Number(ex.temen_onramp_jit_run_pagestate_ptr());
    if (mappedGlobal) mappedGlobal.value = ex.temen_onramp_jit_run_mapped();
  };
  try {
    ({ f0, instance } = await cachedInstanceF0(
      memory,
      cacheKey,
      () => {
        // Copy the emitted bytes out (a later temen_alloc could move the stash).
        const wptr = Number(ex.temen_onramp_jit_run_wasm_ptr());
        const wlen = ex.temen_onramp_jit_run_wasm_len();
        return u8().slice(wptr, wptr + wlen);
      },
      (func, argsPtr) => {
        const st = ex.temen_onramp_jit_run_call_interp(func, argsPtr);
        if (st !== 0) throw new Error('cross-tier stop');
        // A `vm_map` grow in the bounce advanced the run's committed extent — re-sync the emitted
        // `"mapped"` (the `driveCoopTierupRun` scalar pattern; on-ramp guests grow scalar, no paged
        // pagestate on the mask-only emit; a paged run re-points its table too). Inert until the
        // globals are registered just below (and on a cached instance the first Run's closure carries
        // them, reading the current run's state via the FFI each time).
        syncGlobals();
      },
    ));
  } catch (e) {
    ex.temen_onramp_jit_run_close();
    throw e;
  }
  mappedGlobal = instance.exports.mapped ?? null;
  pagestateGlobal = instance.exports.pagestate ?? null;
  // #1153: reset the bound to THIS run's committed extent before `f0` runs. A cached instance (issue
  // #803) carries the prior Run's grown `"mapped"`; each Run re-opens a cold window at the declared
  // extent, so without this reset an early emitted access (before the Run's first `vm_map` bounce)
  // could admit against a stale-high bound. `temen_onramp_jit_run_mapped` reads the current run.
  syncGlobals();

  const env = Number(ex.temen_alloc(envBytes));
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
    // (a wasm `unreachable` / a cross-tier bounce that trapped). `temen_onramp_jit_run_finish` tells which.
    threw = 1;
  }
  ex.temen_dealloc(env, envBytes);
  ex.temen_onramp_jit_run_report(threw, value); // record the return value + throw before capturing
  const status = ex.temen_onramp_jit_run_finish(); // capture stdout/stderr/exit/value into the shared slots
  let trapMsg = '';
  if (status === 3) {
    const tl = ex.temen_onramp_jit_run_trap_len();
    trapMsg = tl ? new TextDecoder().decode(u8().slice(Number(ex.temen_stdout_ptr()), Number(ex.temen_stdout_ptr()) + tl)) : '';
  }
  ex.temen_onramp_jit_run_close();
  // A trap on the emitted tier is a refusal, not a result: throw so the caller runs the guest on the
  // interpreter oracle instead of surfacing a truncated run (INVARIANT 9 — diverge toward refusal).
  if (status === 3 /* STATUS_TRAP */) throw new Error('emitted run trapped (declined to the interpreter): ' + trapMsg);
  return status;
}

// Drive an opened **detached** single-shot run (#1285, DETACHED_JIT.md §3.1): the emitted `_start`
// instance is bound to the CHILD's own `WebAssembly.Memory` (`childMemory`, registered as the run's foreign
// memory), not the engine's. `win` is the host header page the cdylib reports; the `env` bounce cell lives
// at child offset 0 (the emitted code stores its scratch slots through ITS memory), so each `call_interp`
// mirrors the cell into an engine-side scratch for the bounce and back after. A paged run's `pagestate`
// table is copied into the header at `temen_detached_pagestate_off()` on every sync. Everything else —
// `"mapped"` re-sync, report/finish/close, trap ⇒ decline — is `driveJitRun`'s contract.
export async function driveDetachedRun(ex, memory, childMemory, cacheKey) {
  const eu8 = () => new Uint8Array(memory.buffer);
  const cu8 = () => new Uint8Array(childMemory.buffer);
  const win = Number(ex.temen_onramp_jit_run_win_ptr()); // = temen_detached_header_bytes()
  const envBytes = ex.temen_onramp_jit_run_env_bytes();
  const pagestateOff = ex.temen_detached_pagestate_off();
  if (envBytes > pagestateOff) throw new Error('env cell does not fit the detached header');
  const slots = [];
  for (let i = 0, n = ex.temen_onramp_jit_run_slot_count(); i < n; i++) slots.push(ex.temen_onramp_jit_run_slot(i));
  const env = 0; // the header page starts the child memory; the cell sits at its bottom
  const scratch = Number(ex.temen_alloc(envBytes)); // engine-side mirror of the cell for bounces
  let mappedGlobal = null, pagestateGlobal = null;
  const syncGlobals = () => {
    if (pagestateGlobal) {
      const p = Number(ex.temen_onramp_jit_run_pagestate_ptr()), n = ex.temen_onramp_jit_run_pagestate_len();
      if (pagestateOff + n > win) throw new Error('pagestate table does not fit the detached header');
      cu8().set(eu8().subarray(p, p + n), pagestateOff);
      pagestateGlobal.value = pagestateOff;
    }
    if (mappedGlobal) mappedGlobal.value = ex.temen_onramp_jit_run_mapped();
  };
  let f0, instance;
  try {
    ({ f0, instance } = await cachedInstanceF0(
      childMemory,
      cacheKey,
      () => {
        const wptr = Number(ex.temen_onramp_jit_run_wasm_ptr());
        const wlen = ex.temen_onramp_jit_run_wasm_len();
        return eu8().slice(wptr, wptr + wlen);
      },
      (func, argsPtr) => {
        eu8().set(cu8().subarray(env, env + envBytes), scratch);
        const st = ex.temen_onramp_jit_run_call_interp(func, scratch + (argsPtr - env));
        // Copy back even on a stop: the cell is the child's, and a later read must see the results.
        cu8().set(eu8().subarray(scratch, scratch + envBytes), env);
        if (st !== 0) throw new Error('cross-tier stop');
        syncGlobals();
      },
      'f0',
      false,
    ));
  } catch (e) {
    ex.temen_dealloc(scratch, envBytes);
    ex.temen_onramp_jit_run_close();
    throw e;
  }
  mappedGlobal = instance.exports.mapped ?? null;
  pagestateGlobal = instance.exports.pagestate ?? null;
  syncGlobals();
  new DataView(childMemory.buffer).setBigInt64(env, 1n << 60n, true); // dispatcher-fuel budget
  let threw = 0, value = 0n, unwound = '';
  try {
    const r = f0(win, env, ...slots);
    value = r === undefined || r === null ? 0n : BigInt(r);
  } catch (e) {
    threw = 1;
    unwound = String(e && e.message || e); // a wasm trap, a bounce stop, or a host-import exception
  }
  ex.temen_dealloc(scratch, envBytes);
  ex.temen_onramp_jit_run_report(threw, value);
  const status = ex.temen_onramp_jit_run_finish();
  let trapMsg = '';
  if (status === 3) {
    const tl = ex.temen_onramp_jit_run_trap_len();
    trapMsg = tl ? new TextDecoder().decode(eu8().slice(Number(ex.temen_stdout_ptr()), Number(ex.temen_stdout_ptr()) + tl)) : '';
  }
  ex.temen_onramp_jit_run_close();
  if (status === 3 /* STATUS_TRAP */) throw new Error(`emitted detached run trapped (declined to the interpreter): ${trapMsg || unwound}`);
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

// The **cooperative tier-up driver** (#926 slice 2; since #1026 the ONE fallback tier when the
// whole-program emit declines): the single-thread, no-Worker host loop for an InterpDriven guest —
// single-vCPU or genuinely threaded alike. It wraps the `temen_coop_*` cdylib (`CoopRun`), whose
// scheduler multiplexes every vCPU of the run — the root and its `thread.spawn` descendants — on
// this one wasm thread and services concurrency, fibers, and §22 install/invoke **internally**, so
// only tier-up, a §22 `Jit.invoke` of an emitted unit, and the run's end reach here.
//
// #1312: this run's window **grows** — a guest allocator's `vm_map` past the declared window commits
// real memory rather than being refused — and growing it reallocates, so the window's base can move
// mid-run. Nothing here may cache the base or the length: `eventWin()` reads the base per event, and
// the `call_interp` bounce below republishes it to every live instance's `"win"` global (the emitted
// code reloads its `win` from that global after each call). The per-event
// contract (worker.js's PAR_TIERUP shape): sync the B2 shared driver table (`call_indirect` tiers
// up, #880) when its generation advances, re-arm `"fuel"`, write the event's `"mapped"` sync (#717;
// a paged run also points `"pagestate"` at the live table, #1009), call `f{func}(win, env, ...args)`
// over the cdylib's shared memory, and deliver the results (or the trap) back to the parked vCPU;
// `env.call_interp` is the live-state bounce. Proven observably identical to `onramp_exec` by
// tests/coop_tierup_driver.rs (wasmi playing this file's role).
/// #1896 — whether this host can **suspend** a leaf's emitted frames where a call in it parks: JSPI
/// (`WebAssembly.Suspending` / `WebAssembly.promising`). Pass it to `temen_nim_open`: without it the
/// engine offers no leaf that can park, and such a process runs interpreted instead.
export const suspendsLeaves = typeof WebAssembly.Suspending === 'function'
  && typeof WebAssembly.promising === 'function';

export async function driveCoopTierupRun(ex, memory, cacheKey, counts = {}) {
  // #1896: how many processes ran whole on the emitted tier, and how many parked calls in them
  // resumed. A caller that wants to know passes the object to fill.
  counts.leaves = 0;
  counts.resumes = 0;
  const u8 = () => new Uint8Array(memory.buffer);
  const i64 = () => new BigInt64Array(memory.buffer);
  // #816 env-routed tier-up: `win` is PER EVENT — the pending task's window base (root backing for
  // a root-env task, backing + carve offset for a §14 confined child). Read inside each event arm
  // via temen_coop_tierup_win_ptr(); never cache it across events.
  const eventWin = () => Number(ex.temen_coop_tierup_win_ptr());

  const mappedGlobals = []; // every live instance's "mapped" — the post-bounce fan-out set (#717)
  const fuelGlobals = [];
  const pagestateGlobals = []; // #1009 paged: the "pagestate" base globals (only a paged main module has one)
  // #1312: every live instance's "win" — the live window BASE. The coop run's window backing grows
  // on a guest `vm_map`, and growing it reallocates, so the base can move mid-run. The emitted entry
  // publishes its own `win` argument here, so this set only has to be written when the base actually
  // changed: after a bounce that may have grown the window (below). Every emitted function reloads
  // its `win` local from this global after each call, so the write takes effect immediately.
  const winGlobals = [];
  const registerGlobals = (exports) => {
    if (exports.mapped) mappedGlobals.push(exports.mapped);
    if (exports.fuel) fuelGlobals.push(exports.fuel);
    if (exports.pagestate) pagestateGlobals.push(exports.pagestate);
    if (exports.win) winGlobals.push(exports.win);
  };
  // #846/#880 on the cooperative path — the shared driver table (Model B2): the main module and every
  // §22 unit `call_indirect` through it, and the driver populates its slots from the engine's mirror at
  // each event boundary (an installed unit's emitted `f0`, an emitted program function's `f{i}`, or a
  // bounce shim for an interpreter-resident target). `env.call_interp` bounces a cross-tier helper back
  // through the cooperative live-state bounce (routed to the tiering-up task's env), then fans the fresh
  // "mapped" extent out to every live instance. A non-shimmable guest reports `table_log2 == 0` (a
  // 1-slot table) and emits in local-table mode, so the shared table is inert for it.
  const tsize = 1 << ex.temen_coop_table_log2();
  const table = new WebAssembly.Table({ initial: tsize, maximum: tsize, element: 'anyfunc' });
  const bounce = (target, argsPtr) => {
    // #1627: a spilling run hands the bounce the words its emitted frames pushed, `[base, cursor)`.
    const spillLen = spillBase
      ? (new DataView(memory.buffer).getUint32(envCell + spillOff, true) - spillBase) / 8
      : 0;
    return ex.temen_coop_call_interp(target, argsPtr, spillLen);
  };
  // What a bounce's return fans out to the live instances — and a parked call's (#1896).
  const afterBounce = (rc) => {
    // #1233: the bounce may have been a `Jit.install`/`uninstall` issued from the emitted frame
    // itself (Forth's outer interpreter defining a word, then `call.dyn`ing it) — the slot mirror
    // moved mid-event, and the frame's next `call_indirect` must find the new occupant, not a stale
    // or empty slot. Rebuild synchronously, before the globals fan-out below primes any instance
    // this creates (a word unit is tiny; one over the sync compile budget gets a bounce shim now and
    // its emitted unit at the next event-boundary sync).
    if (rc === 0 && ex.temen_coop_table_gen() !== syncedGen) syncTableSync();
    // #1009 paged: the grow rebuilt the page-state table (in `call_interp`) — fan the fresh coverage
    // to "mapped" and re-point "pagestate"; else the #717 scalar extent (the pump's twin).
    if (ex.temen_coop_paged()) {
      const cover = ex.temen_coop_mapped();
      for (const g of mappedGlobals) g.value = cover;
      const ps = Number(ex.temen_coop_pagestate_ptr());
      for (const g of pagestateGlobals) g.value = ps;
    } else {
      const now = ex.temen_coop_mapped_now();
      for (const g of mappedGlobals) g.value = now;
    }
    // #1312: the bounce ran interpreted guest code, which may have `vm_map`-grown the window. A
    // grow reallocates the backing and can MOVE it, so publish the current base to every live
    // instance — the emitted frame reloads its `win` from this global on return from the bounce.
    // Read it fresh here (never cached): this is the one point in the run where it can change.
    const base = Number(ex.temen_coop_tierup_win_ptr());
    for (const g of winGlobals) g.value = base;
  };
  const callInterp = (target, argsPtr) => {
    const rc = bounce(target, argsPtr);
    afterBounce(rc);
    if (rc !== 0) throw new Error('bounce trap'); // unwind to the deliver below
  };
  // #1896 — a leaf's `call_interp` under JSPI. A call that parks (`2`) suspends the leaf's frames on
  // the promise returned here, and the driver runs on; `COOP_RUN_RESUME` resolves it once the call
  // has returned, with its results in the call's scratch.
  let parking = null; // the running leaf's parked call: its scratch, and what resumes its frames
  let parked = () => {}; // tells `settle` the running leaf parked
  const leafCallInterp = suspendsLeaves
    ? new WebAssembly.Suspending((target, argsPtr) => {
      const rc = bounce(target, argsPtr);
      if (rc === 2) {
        return new Promise((resolve) => {
          parking = { argsPtr, resolve };
          parked();
        });
      }
      afterBounce(rc);
      if (rc !== 0) throw new Error('bounce trap');
    })
    : callInterp;
  const unitImports = (call_interp = callInterp) => ({ env: {
    memory,
    __indirect_function_table: table,
    trap: () => {},
    call_interp,
  } });

  // The run's own emit (program 0). A nimony build (`temen_nim_open`) emits none of its own: its
  // programs are the leaf images its processes exec (#1896), each instantiated at its first TIERUP —
  // and neither does a run whose root program runs whole as a leaf (#1954): its program 0 is that leaf.
  let emitted = {};
  if (ex.temen_coop_wasm_len() > 0) {
    const coopKey = cacheKey === undefined ? undefined : `${cacheKey}#coop`;
    let module = cacheGet(coopKey);
    if (module === undefined) {
      const wptr = Number(ex.temen_coop_wasm_ptr());
      const wlen = ex.temen_coop_wasm_len();
      module = await WebAssembly.compile(u8().slice(wptr, wptr + wlen));
      cachePut(coopKey, module);
      jitCacheStats.compiles++;
    } else {
      jitCacheStats.hits++;
    }
    emitted = (await WebAssembly.instantiate(module, unitImports())).exports;
  }
  // #1954: a program is a leaf image iff the engine emitted one for it — the engine's answer, not a
  // guess from its index (a root leaf is program 0).
  const isLeaf = (m) => ex.temen_coop_leaf_wasm_len(m) > 0;
  const programs = new Map();
  if (ex.temen_coop_wasm_len() > 0) programs.set(0, emitted);
  const programFor = async (m) => {
    let p = programs.get(m);
    if (p === undefined && !isLeaf(m)) p = emitted;
    if (p === undefined) {
      const ptr = Number(ex.temen_coop_leaf_wasm_ptr(m));
      const bytes = u8().slice(ptr, ptr + ex.temen_coop_leaf_wasm_len(m));
      // A leaf image's program index is the run's own, so the cross-Run cache keys its emitted bytes
      // by content. Holding the compiled Module keeps its code for the next build in this page: V8
      // otherwise frees a Module nothing references, and the next build compiles it cold again.
      const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
      const key = `leaf:${Array.from(digest, (b) => b.toString(16).padStart(2, '0')).join('')}`;
      let module = cacheGet(key);
      if (module === undefined) {
        module = await WebAssembly.compile(bytes);
        cachePut(key, module);
        jitCacheStats.compiles++;
      } else {
        jitCacheStats.hits++;
      }
      p = (await WebAssembly.instantiate(module, unitImports(leafCallInterp))).exports;
      registerGlobals(p);
      programs.set(m, p);
    }
    return p;
  };
  const envCell = Number(ex.temen_alloc(ex.temen_wasmjit_env_bytes()));
  // #1627: a collecting guest's emitted frames push their live words to a spill stack named by the
  // env cell's cursor pair; every event is an outermost entry, so each one re-arms it at the base.
  const spillBytes = ex.temen_coop_spill_bytes();
  const spillBase = spillBytes ? Number(ex.temen_coop_spill_ptr()) : 0;
  const spillOff = ex.temen_wasmjit_spill_sp_off();
  const armEnv = () => {
    const dv = new DataView(memory.buffer);
    dv.setBigInt64(envCell, 1n << 61n, true);
    if (spillBase) {
      dv.setUint32(envCell + spillOff, spillBase, true);
      dv.setUint32(envCell + spillOff + 4, spillBase + spillBytes, true);
    }
  };
  registerGlobals(emitted);
  // Per-code-handle unit instances (a runtime-compiled §22 unit runs emitted on JIT_INVOKE — the
  // JACL macro-staging shape). Async instantiation: a macro unit can exceed the sync compile budget.
  const jitUnits = new Map();
  const shims = new Map();
  const instantiateUnit = async (bytes) => {
    const inst = await WebAssembly.instantiate(await WebAssembly.compile(bytes), unitImports());
    registerGlobals(inst.exports);
    return inst.exports;
  };
  // #1233: the synchronous twin, for a rebuild inside a bounce (no event boundary to await at). An
  // instance created mid-event never passes the per-event fuel re-arm — budget it now.
  const instantiateUnitSync = (bytes) => {
    const inst = new WebAssembly.Instance(new WebAssembly.Module(bytes), unitImports());
    registerGlobals(inst.exports);
    if (inst.exports.fuel) inst.exports.fuel.value = 1n << 61n;
    return inst.exports;
  };
  const shimBytes = (slot) => {
    const len = ex.temen_coop_shim_wasm(slot);
    if (len === 0) return null;
    return u8().slice(Number(ex.temen_coop_shim_ptr()), Number(ex.temen_coop_shim_ptr()) + len);
  };
  const shimFor = async (slot, code) => {
    const key = `${slot}#${code}`;
    let f = shims.get(key);
    if (f === undefined) {
      const bytes = shimBytes(slot);
      if (bytes === null) return null;
      f = (await instantiateUnit(bytes))['t'];
      shims.set(key, f);
    }
    return f;
  };
  // Bounce shims are under 200 bytes each and a table rebuild instantiates one per interpreter-
  // resident slot (~200 on the JACL compiler card). Going through the ASYNC compile queue for them is
  // pathological: V8 can park one such `WebAssembly.instantiate` promise for seconds behind its own
  // background work on the big emitted module (measured: the SECOND warm-coop run's rebuild took
  // 6.2 s for 224 shims, one of them 6.15 s, while the first and third took ~50 ms — the playground's
  // tier-up mode failed its second compile on this). Synchronous instantiation is immune (~25 ms for
  // all 224) and a shim is far under the main-thread sync-compile budget; a shim that isn't (never
  // seen) falls back to the async path.
  const shimForFast = async (slot, code) => {
    try {
      return shimForSync(slot, code);
    } catch {
      return shimFor(slot, code);
    }
  };
  const shimForSync = (slot, code) => {
    const key = `${slot}#${code}`;
    let f = shims.get(key);
    if (f === undefined) {
      const bytes = shimBytes(slot);
      if (bytes === null) return null;
      f = instantiateUnitSync(bytes)['t'];
      shims.set(key, f);
    }
    return f;
  };
  // `key`: a surfaced JIT_INVOKE's code handle (a Number — live for that invoke), or an installed
  // slot's `(domain, unit)` identity (`temen_coop_slot_unit`, a BigInt) — distinct key types, one
  // cache. An installed slot is never keyed by handle: the guest revokes it right after `install`.
  // #1378 again, on the unit path: a guest-compiled §22 unit (a JACL macro body: ~1.1 KB) went through
  // the ASYNC compile queue, and V8 parks that behind its background work on the just-compiled emitted
  // module — measured 2.9 s / 3.7 s for the tour's two macro invokes on the first two warm-coop runs,
  // 88 ms on the third. Instantiate synchronously when the unit is under the main-thread sync-compile
  // budget (a `new WebAssembly.Module` over it throws — then the async path, as before).
  const unitFor = async (key, bytes) => {
    let unit = jitUnits.get(key);
    if (unit === undefined) {
      try {
        unit = instantiateUnitSync(bytes);
      } catch {
        unit = await instantiateUnit(bytes);
      }
      jitUnits.set(key, unit);
    }
    return unit;
  };
  // The bytes of the unit installed at `slot` (`null` = interpreter-only), fetched **by slot** —
  // the guest typically `release`s the code handle right after `install` (the unit stays installed,
  // only its handle dies), so a by-handle fetch would come back empty and null the slot (#1233).
  const slotUnitBytes = (slot) => {
    const len = ex.temen_coop_jit_wasm_by_slot_len(slot);
    if (len === 0) return null;
    const p = Number(ex.temen_coop_jit_wasm_by_handle_ptr());
    return u8().slice(p, p + len);
  };
  // Rebuild the shared table from the engine's slot mirror whenever its generation moved: at each
  // event boundary (`syncTable`) and — #1233 — inside `env.call_interp` after a bounce that installed
  // or uninstalled (`syncTableSync`). A slot in the natural prefix holds the emitted program `f{slot}`
  // (or a bounce shim if that function stayed interpreted); a slot past it holds an installed unit's
  // `f0` (fetched by slot, cached by unit identity) or a shim for an interpreter-resident target.
  // Exactly `driveTierupRun`'s `syncTable`, over the `temen_coop_*` accessors.
  const nfuncs = ex.temen_coop_nfuncs();
  // #1009: rebuild the table only when the slot mirror changed (a §22 install/uninstall bumps
  // `temen_coop_table_gen`) — a card that never installs syncs the table once, not per tier-up.
  let syncedGen = -1;
  const syncTable = async () => {
    const gen = ex.temen_coop_table_gen();
    if (gen === syncedGen) return;
    for (let slot = 0; slot < tsize; slot++) {
      let entry = null;
      if (slot < nfuncs) {
        entry = emitted['f' + slot] ?? await shimForFast(slot, -2);
      } else {
        const uid = ex.temen_coop_slot_unit(slot);
        if (uid >= 0n) {
          const cached = jitUnits.get(uid);
          if (cached !== undefined) entry = cached['f0'];
          else {
            const bytes = slotUnitBytes(slot);
            entry =
              bytes !== null ? (await unitFor(uid, bytes))['f0'] : await shimForFast(slot, uid);
          }
        }
      }
      table.set(slot, entry);
    }
    syncedGen = gen;
  };
  const syncTableSync = () => {
    const gen = ex.temen_coop_table_gen();
    if (gen === syncedGen) return;
    for (let slot = 0; slot < tsize; slot++) {
      let entry = null;
      if (slot < nfuncs) {
        entry = emitted['f' + slot] ?? shimForSync(slot, -2);
      } else {
        const uid = ex.temen_coop_slot_unit(slot);
        if (uid >= 0n) {
          const cached = jitUnits.get(uid);
          if (cached !== undefined) entry = cached['f0'];
          else {
            const bytes = slotUnitBytes(slot);
            if (bytes !== null) {
              // Over the sync compile budget ⇒ a shim now; `syncTable` upgrades it at the next event.
              try { const u = instantiateUnitSync(bytes); jitUnits.set(uid, u); entry = u['f0']; }
              catch { entry = shimForSync(slot, uid); }
            } else entry = shimForSync(slot, uid);
          }
        }
      }
      table.set(slot, entry);
    }
    syncedGen = gen;
  };

  const deliver = (ret) => {
    const rets = ret === undefined ? [] : Array.isArray(ret) ? ret : [ret];
    const rlen = Math.max(1, rets.length) * 8;
    const rptr = Number(ex.temen_alloc(rlen));
    for (let i = 0; i < rets.length; i++) i64()[(rptr >> 3) + i] = BigInt(rets[i]);
    ex.temen_coop_deliver(rptr, rets.length);
    ex.temen_dealloc(rptr, rlen);
  };
  // #1896 — run a leaf's frames, its entry or its frames resuming, until the leaf ends or a call in
  // it parks: deliver the end, or hold the frames (`suspended`, by task) until the call returns.
  const suspended = new Map();
  const settle = async (task, go) => {
    const parks = new Promise((r) => { parked = r; });
    const run = go();
    const end = await Promise.race([
      run.then((ret) => ({ ret }), () => ({ trapped: true })),
      parks.then(() => ({ parks: true })),
    ]);
    parked = () => {};
    if (end.parks) {
      suspended.set(task, { run, ...parking });
      parking = null;
    } else if (end.trapped) {
      ex.temen_coop_deliver_trap();
    } else {
      deliver(end.ret);
    }
  };

  try {
    for (;;) {
      const ev = ex.temen_coop_run();
      if (ev === 4 /* COOP_RUN_RESUME */) {
        // #1896: a leaf's parked call returned. Its results go to the call's scratch, the globals
        // fan out as after any bounce, and the leaf's frames run on.
        counts.resumes++;
        const task = ex.temen_coop_task();
        const leaf = suspended.get(task);
        suspended.delete(task);
        const from = Number(ex.temen_coop_argv_ptr()) >> 3;
        for (let i = 0; i < ex.temen_coop_argv_len(); i++) {
          i64()[(leaf.argsPtr >> 3) + i] = i64()[from + i];
        }
        await syncTable();
        afterBounce(0);
        for (const g of fuelGlobals) g.value = 1n << 61n;
        armEnv();
        await settle(task, () => {
          leaf.resolve();
          return leaf.run;
        });
        continue;
      }
      if (ev === 3 /* COOP_RUN_JIT_INVOKE */) {
        // A guest-compiled §22 unit with emitted wasm: sync the table (its `call_indirect` may reach
        // installed units / program `f{i}`s / bounce shims), instantiate once per code handle, then
        // `f0(win, env, ...args)` with the per-event "mapped"/fuel sync fanned to every live instance.
        await syncTable();
        const code = ex.temen_coop_jit_code();
        const wptr = Number(ex.temen_coop_jit_wasm_ptr());
        const unit = await unitFor(code, u8().slice(wptr, wptr + ex.temen_coop_jit_wasm_len()));
        const argvPtr = Number(ex.temen_coop_argv_ptr());
        const n = ex.temen_coop_argv_len();
        const ptypes = new Uint8Array(memory.buffer, Number(ex.temen_coop_jit_param_types_ptr()), n);
        const args = [];
        for (let i = 0; i < n; i++) args.push(tierupJitArg(i64()[(argvPtr >> 3) + i], ptypes[i]));
        const mapped = ex.temen_coop_mapped();
        for (const g of mappedGlobals) g.value = mapped;
        for (const g of fuelGlobals) g.value = 1n << 61n;
        // #1334 paged: the unit's `call_indirect` can reach a paged program `f{i}` — point its page
        // check at the table the engine refreshed for this event (as the TIERUP arm does below).
        if (ex.temen_coop_paged()) {
          const ps = Number(ex.temen_coop_pagestate_ptr());
          for (const g of pagestateGlobals) g.value = ps;
        }
        armEnv();
        try {
          const ret = unit['f0'](eventWin(), envCell, ...args);
          const rets = ret === undefined ? [] : Array.isArray(ret) ? ret : [ret];
          const rn = ex.temen_coop_jit_result_types_len();
          const rtypes = new Uint8Array(memory.buffer, Number(ex.temen_coop_jit_result_types_ptr()), rn);
          const rlen = Math.max(1, rets.length) * 8;
          const rptr = Number(ex.temen_alloc(rlen));
          for (let i = 0; i < rets.length; i++) i64()[(rptr >> 3) + i] = tierupJitRes(rets[i], rtypes[i]);
          ex.temen_coop_deliver_jit(rptr, rets.length);
          ex.temen_dealloc(rptr, rlen);
        } catch {
          ex.temen_coop_deliver_jit_trap();
        }
        continue;
      }
      if (ev !== 1 /* COOP_RUN_TIERUP */) break; // 0 = done (slots staged), 2 = trapped (status 3)
      // #880: a tiered-up leaf's `call_indirect` dispatches through the shared table — sync it before
      // running the region (the per-event "mapped"/fuel fan-out covers every instance it may reach).
      await syncTable();
      const func = ex.temen_coop_func();
      // Before the per-event fan-out below, so a program instantiated now gets this event's sync.
      const m = ex.temen_coop_module();
      const leaf = isLeaf(m);
      if (leaf) counts.leaves++;
      const program = await programFor(m);
      const argvPtr = Number(ex.temen_coop_argv_ptr());
      const n = ex.temen_coop_argv_len();
      const args = [];
      for (let i = 0; i < n; i++) args.push(i64()[(argvPtr >> 3) + i]);
      // #717 host sync: the event's committed extent → every live "mapped" global, so the emitted
      // bounds checks admit exactly what the interpreter's page map does for this call.
      const tmapped = ex.temen_coop_mapped();
      for (const g of mappedGlobals) g.value = tmapped;
      for (const g of fuelGlobals) g.value = 1n << 61n; // re-arm across events on the reused instance
      // #1009 paged: point the emitted page check at the freshly rebuilt table (base can move as the
      // pump's Vec reallocates, so read it every event). Empty set / `_paged==0` on an unpaged run.
      if (ex.temen_coop_paged()) {
        const ps = Number(ex.temen_coop_pagestate_ptr());
        for (const g of pagestateGlobals) g.value = ps;
      }
      armEnv();
      // #1896: a leaf program runs under JSPI where there is one, so that a call in it can park.
      if (suspendsLeaves && leaf) {
        const entry = WebAssembly.promising(program['f' + func]);
        await settle(ex.temen_coop_task(), () => entry(eventWin(), envCell, ...args));
        continue;
      }
      try {
        deliver(program['f' + func](eventWin(), envCell, ...args));
      } catch {
        ex.temen_coop_deliver_trap();
      }
    }
  } finally {
    ex.temen_dealloc(envCell, ex.temen_wasmjit_env_bytes());
    ex.temen_coop_close();
  }
  const status = ex.temen_status();
  if (status === 3 /* STATUS_TRAP */) {
    throw new Error('cooperative tier-up run trapped (declined to the interpreter)');
  }
  return status;
}

// A **registry blob** (the cdylib's `registry_blob`): all integers little-endian, a u32 entry count,
// then per entry a u32 name length, the name (UTF-8), a u32 length and the bytes. Its entries, each
// `[name, bytes]` with `bytes` a view into `blob`.
export function blobEntries(blob) {
  const dv = new DataView(blob.buffer, blob.byteOffset, blob.byteLength);
  const dec = new TextDecoder();
  const out = [];
  let o = 4;
  for (let n = dv.getUint32(0, true); n > 0; n--) {
    const nl = dv.getUint32(o, true);
    const name = dec.decode(blob.subarray(o + 4, o + 4 + nl));
    const bl = dv.getUint32(o + 4 + nl, true);
    o += 8 + nl;
    out.push([name, blob.subarray(o, o + bl)]);
    o += bl;
  }
  return out;
}

// The registry blob of `entries`, each `[name, bytes]` ([`blobEntries`]'s inverse).
export function registryBlob(entries) {
  const enc = new TextEncoder();
  const named = entries.map(([n, b]) => [enc.encode(n), b]);
  const out = new Uint8Array(named.reduce((t, [n, b]) => t + 8 + n.length + b.length, 4));
  const dv = new DataView(out.buffer);
  dv.setUint32(0, named.length, true);
  let o = 4;
  for (const [n, b] of named) {
    dv.setUint32(o, n.length, true);
    out.set(n, o + 4);
    dv.setUint32(o + 4 + n.length, b.length, true);
    out.set(b, o + 8 + n.length);
    o += 8 + n.length + b.length;
  }
  return out;
}

// #958 — nimony's toolchain as the nim card ships it (`web/assets/nimony.blob.gz`, written by
// `nimbuild --bundle`): a registry blob of `cwd`, the directory a build with it runs in (its library
// pack names it); `commands`, as `temen_nim_open` takes them, nimony first; and `files`, what a build
// seeds before its program — nimony's library, the library pack, the guest libc.
export function nimToolchain(bundle) {
  const e = new Map(blobEntries(bundle));
  return {
    cwd: new TextDecoder().decode(e.get('cwd')),
    commands: e.get('commands'),
    nimony: blobEntries(e.get('commands'))[0][1],
    files: blobEntries(e.get('files')),
  };
}

// #958 — compile `source` as `prog.nim` with nimony's own driver and run what it builds:
// `nimony t -r --isMain prog.nim`, one `temen_nim_open` session over `toolchain` ([`nimToolchain`])
// that `driveCoopTierupRun` runs. The process tree (nimony, nifmake, nimsem, the shell) runs on the
// interpreter, each leaf process (nifler2, hexer, the program itself) whole on the emitted tier
// (#1896), and temen-link natively. nimony prints nothing when a build succeeds, so what the session
// prints is the program's. When the build fails, nothing is linked and what it printed is nimony's
// diagnostics. Resolves `{ status, exit, stdout, stderr, built, leaves, resumes }`: `built` is
// whether the program was linked, and so ran.
export async function nimCompileRun(ex, memory, toolchain, source) {
  const enc = new TextEncoder();
  const text = (p, n) => new TextDecoder().decode(new Uint8Array(memory.buffer, p, n).slice());
  const held = [];
  const put = (bytes) => {
    const p = Number(ex.temen_alloc(bytes.length));
    new Uint8Array(memory.buffer).set(bytes, p);
    held.push([p, bytes.length]);
    return [p, bytes.length];
  };
  const { cwd } = toolchain;
  const files = registryBlob([...toolchain.files, [`${cwd}/prog.nim`, enc.encode(source)]]);
  const argv = enc.encode(['bin/nimony', 't', '-r', '--isMain', 'prog.nim'].map((a) => `${a}\0`).join(''));
  const args = [put(toolchain.nimony), put(toolchain.commands), put(files), put(argv), put(enc.encode(cwd))];
  const opened = ex.temen_nim_open(...args.flat(), suspendsLeaves ? 1 : 0);
  // The session holds its own copies of what it was opened with.
  for (const [p, n] of held.splice(0)) ex.temen_dealloc(p, n);
  if (opened !== 0) throw new Error(`temen_nim_open: status ${ex.temen_status()}`);
  const counts = {};
  const status = await driveCoopTierupRun(ex, memory, undefined, counts);
  const stdout = text(Number(ex.temen_stdout_ptr()), ex.temen_stdout_len());
  const stderr = text(Number(ex.temen_stderr_ptr()), ex.temen_stderr_len());
  const exit = ex.temen_exit_code();
  // `nimcache/<stem>.temen/prog.temen`, `<stem>` nimony's for `prog.nim`, which lands on the stdout
  // slot (read it only after the call: the slot moves).
  const stemLen = ex.temen_nim_module_suffix(...put(enc.encode('prog.nim')));
  const stem = text(Number(ex.temen_stdout_ptr()), stemLen);
  const built = Number(ex.temen_nim_file(...put(enc.encode(`${cwd}/nimcache/${stem}.temen/prog.temen`)))) >= 0;
  for (const [p, n] of held.splice(0)) ex.temen_dealloc(p, n);
  return { status, exit, stdout, stderr, built, ...counts };
}

// Run an on-ramp module whose input is **stdin** (Lua/SQLite/hello) on the wasm-JIT.
export async function runJitModule(ex, memory, moduleBytes, stdinBytes, cacheKey, shared = 1) {
  const u8 = () => new Uint8Array(memory.buffer);
  // Hand the module (+ optional stdin) to the cdylib: decode, outline, grant powerbox, emit `_start`.
  const modP = Number(ex.temen_alloc(moduleBytes.length));
  u8().set(moduleBytes, modP);
  let stdinP = 0;
  const stdinLen = stdinBytes ? stdinBytes.length : 0;
  if (stdinLen) {
    stdinP = Number(ex.temen_alloc(stdinLen));
    u8().set(stdinBytes, stdinP);
  }
  // `shared`: 1 (default) instantiates the emitted module against the cdylib's **shared** memory
  // (cross-origin-isolated threads build); a plain single-threaded host (e.g. the JACL playground's
  // non-threads cdylib) passes 0 — a shared-mode emit LinkErrors against a non-shared memory. Same
  // knob as `runWarmCoop`.
  const opened = ex.temen_onramp_jit_run_open(modP, moduleBytes.length, stdinP, stdinLen, shared);
  // `_start` not whole-program-emittable (an InterpDriven guest — it `vm_map`s, streams,
  // `thread.spawn`s, hosts fibers, …): try the **cooperative** tier-up driver before giving the
  // buffers up — its scheduler multiplexes every vCPU of the run on this one wasm thread, the
  // interpreter drives `_start`, eligible pure leaves run on emitted wasm, and a `vm_jit_*` guest's
  // runtime-compiled §22 units run emitted too (#835). This is the ONE fallback tier (#1026: the
  // single-vCPU pump it used to try first was a strict subset, and slower). Refused (nothing
  // emittable ever) → fall through to the throw and the caller's plain-interpreter fallback.
  let coop = false;
  if (opened !== 0 && ex.temen_coop_open &&
      ex.temen_coop_open(modP, moduleBytes.length, stdinP, stdinLen, shared) === 0) {
    coop = true;
  }
  ex.temen_dealloc(modP, moduleBytes.length);
  if (stdinP) ex.temen_dealloc(stdinP, stdinLen);
  if (coop) return driveCoopTierupRun(ex, memory, cacheKey);
  if (opened !== 0) {
    throw new Error(`JIT module open failed: status ${ex.temen_status()} (2 = _start not emittable)`);
  }
  return driveJitRun(ex, memory, cacheKey);
}

// Run the warm session's `eval_run` on the **warm+JIT** tier (WASM_AOT.md warm+JIT). The warm snapshot
// (`temen_warm_open`) has already paid the QuickJS runtime init once; this evaluates the user's code on
// emitted wasm over the restored warm image, so a compute-heavy program runs the eval near-native while
// init stays paid-once. The engine emits `eval_run` on the first Run and caches it (a warm+JIT Run never
// re-pays the cdylib emit); the compiled `WebAssembly.Module` is cached across Runs too (keyed by
// `cacheKey`). Differs from `driveJitRun` only in the accessors it drives and in passing the entry `sp`
// as the emitted `f0`'s third argument (an i64 slot ⇒ a BigInt). Assumes `temen_warm_open` already
// succeeded for this module. Returns the run status (0 = returned, 5 = exited); throws if `eval_run`
// isn't wasm-drivable or the run traps (the caller falls back to the interpreter warm path).
export async function runWarmJit(ex, memory, stdinBytes, cacheKey, shared = 1) {
  const u8 = () => new Uint8Array(memory.buffer);
  // Emit `eval_run` (idempotent — cached in the warm session after the first Run).
  if (ex.temen_warm_jit_open(shared) !== 0) {
    throw new Error(`warm-JIT open failed: status ${ex.temen_status()} (2 = eval_run not emittable)`);
  }
  // Per-Run: restore the warm image + reset the run's powerbox, feeding the editor text as stdin.
  let stdinP = 0;
  const stdinLen = stdinBytes ? stdinBytes.length : 0;
  if (stdinLen) {
    stdinP = Number(ex.temen_alloc(stdinLen));
    u8().set(stdinBytes, stdinP);
  }
  const prepared = ex.temen_warm_jit_prepare(stdinP, stdinLen);
  if (stdinP) ex.temen_dealloc(stdinP, stdinLen);
  if (prepared !== 0) throw new Error(`warm-JIT prepare failed: status ${ex.temen_status()}`);

  const win = Number(ex.temen_warm_jit_win_ptr());
  const sp = ex.temen_warm_jit_entry_sp(); // i64 export ⇒ BigInt; passed straight as the entry's i64 slot
  const envBytes = ex.temen_onramp_jit_run_env_bytes();
  // Drive the emitted `eval_run` export — `f{eval_fn}`, NOT `f0` (#865). `f0` is the cold `_start`
  // (init + eval); driving it re-runs the guest's init over the restored warm image, which for Tcl
  // re-enters `Tcl_FindExecutable` → an encoding-proc `call_indirect` trap (and for any driver defeats
  // the "init paid once" warm contract). The entry export index comes from the engine.
  const entryName = `f${ex.temen_warm_jit_entry_func()}`;

  // Reuse the instance across Runs (issue #803): a hit skips both the byte copy and instantiate, so a
  // warm Run collapses to `prepare` + the eval. The emit is stable and window-independent (`win`/`sp` are
  // passed per Run), so one instance serves every Run of this warm session.
  const { f0: entry } = await cachedInstanceF0(
    memory,
    cacheKey,
    () => {
      const wptr = Number(ex.temen_warm_jit_wasm_ptr());
      const wlen = ex.temen_warm_jit_wasm_len();
      return u8().slice(wptr, wptr + wlen);
    },
    (func, argsPtr) => {
      if (ex.temen_warm_jit_call_interp(func, argsPtr) !== 0) throw new Error('cross-tier stop');
    },
    entryName,
  );

  const env = Number(ex.temen_alloc(envBytes));
  new DataView(memory.buffer).setBigInt64(env, 1n << 60n, true); // huge dispatcher-fuel budget
  let threw = 0;
  let value = 0n;
  let trapError = null;
  try {
    // entry(win, env, sp) — runs `eval_run(sp)` over the restored warm image on emitted wasm. Its return
    // is the guest's top-level result (an i32/i64); normalize to i64.
    const r = entry(win, env, sp);
    value = r === undefined || r === null ? 0n : BigInt(r);
  } catch (e) {
    threw = 1;
    trapError = e; // keep it — the trap kind + wasm location live here (issue #865)
  }
  ex.temen_dealloc(env, envBytes);
  ex.temen_warm_jit_report(threw, value);
  const status = ex.temen_warm_jit_finish();
  if (status === 3 /* STATUS_TRAP */) throw warmJitTrapError(trapError);
  return status;
}

// Run the warm session's `eval_run` on the **warm-coop** tier (#816 item 4): the cooperative
// tier-up drive over the restored warm image, for a page-managing / InterpDriven eval the
// WasmDriven `runWarmJit` declines (its open throws). The engine emits the module's leaves + cap
// wrappers once (cached in the warm session, like the warm+JIT emit; the compiled
// `WebAssembly.Module` is cached across Runs under `cacheKey#coop`); each Run is prepare (restore
// image + re-establish the captured page map + arm `eval_run` on the coop scheduler) + the standard
// `driveCoopTierupRun` event loop — the interpreter owns the eval, eligible pure leaves run on
// emitted wasm with the per-event `mapped`/page-state sync carrying the warm image's grown heap and
// protected rodata. Assumes `temen_warm_open` already succeeded. Returns the run status; throws if
// the module has nothing for the emitted tier or the run traps (the caller falls back to
// `temen_warm_eval`, the interpreter warm path).
export async function runWarmCoop(ex, memory, stdinBytes, cacheKey, shared = 1) {
  const u8 = () => new Uint8Array(memory.buffer);
  if (ex.temen_warm_coop_open(shared) !== 0) {
    throw new Error(`warm-coop open failed: status ${ex.temen_status()} (2 = nothing emittable)`);
  }
  let stdinP = 0;
  const stdinLen = stdinBytes ? stdinBytes.length : 0;
  if (stdinLen) {
    stdinP = Number(ex.temen_alloc(stdinLen));
    u8().set(stdinBytes, stdinP);
  }
  const prepared = ex.temen_warm_coop_prepare(stdinP, stdinLen);
  if (stdinP) ex.temen_dealloc(stdinP, stdinLen);
  if (prepared !== 0) throw new Error(`warm-coop prepare failed: status ${ex.temen_status()}`);
  return driveCoopTierupRun(ex, memory, cacheKey);
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
// recorded by `temen_warm_jit_call_interp`'s `last_trap`.
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
// the interpreter, warming the hot functions; the next real Run does `temen_warm_jit_prepare` (restores the
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
// argv (`temen_onramp_jit_run_open_fs`, sharing the bytecode card's `chibicc_card_image`), so this driver
// just hands over the module + source. The emitted TEMEN-IR comes back on `temen_stdout_*` after finish.
// `flags` picks how to compile: bit 0 = `-g`, bit 1 = a linkable **program unit** against libc
// declarations only (#1392). chibicc's emitted `_start` is independent of both (the source and argv are
// fed through the memfs, not baked into the code), so `cacheKey` stays valid across them.
export async function runJitCompiler(ex, memory, moduleBytes, srcBytes, flags = 0, cacheKey) {
  const u8 = () => new Uint8Array(memory.buffer);
  const modP = Number(ex.temen_alloc(moduleBytes.length));
  const srcP = Number(ex.temen_alloc(srcBytes.length));
  u8().set(moduleBytes, modP);
  u8().set(srcBytes, srcP);
  // Empty header image (0, 0) — the cdylib seeds the built-in playground headers itself. `flags` is the
  // same word the bytecode `temen_run_onramp_fs` takes.
  const opened = ex.temen_onramp_jit_run_open_fs(modP, moduleBytes.length, 0, 0, srcP, srcBytes.length, flags);
  ex.temen_dealloc(modP, moduleBytes.length);
  ex.temen_dealloc(srcP, srcBytes.length);
  if (opened !== 0) {
    throw new Error(`JIT compiler open failed: status ${ex.temen_status()} (2 = _start not emittable)`);
  }
  return driveJitRun(ex, memory, cacheKey);
}

// Run the **self-host** compile on the wasm-JIT (SELFHOST_C.md §7 step 5): chibicc.temen compiles one of
// its own cc1 TUs (`tuBytes`, a memfs-relative path like `frontend/chibicc/hashmap.c`) to a linkable
// object, reading the TU + its glibc header closure from `imgBytes` (the committed closure image). Same
// shape as `runJitCompiler` but through `temen_selfhost_jit_emit_object_fs` (raw image + `--emit-object`
// argv, 128 MiB window for the giants). The emitted object text comes back on `temen_stdout_*` after finish.
export async function runJitSelfhost(ex, memory, moduleBytes, imgBytes, tuBytes, debugInfo = 0, cacheKey) {
  const u8 = () => new Uint8Array(memory.buffer);
  const modP = Number(ex.temen_alloc(moduleBytes.length));
  const imgP = Number(ex.temen_alloc(imgBytes.length));
  const tuP = Number(ex.temen_alloc(tuBytes.length));
  u8().set(moduleBytes, modP); u8().set(imgBytes, imgP); u8().set(tuBytes, tuP);
  const opened = ex.temen_selfhost_jit_emit_object_fs(modP, moduleBytes.length, imgP, imgBytes.length, tuP, tuBytes.length, debugInfo);
  ex.temen_dealloc(modP, moduleBytes.length);
  ex.temen_dealloc(imgP, imgBytes.length);
  ex.temen_dealloc(tuP, tuBytes.length);
  if (opened !== 0) {
    throw new Error(`JIT self-host open failed: status ${ex.temen_status()} (2 = _start not emittable)`);
  }
  return driveJitRun(ex, memory, cacheKey);
}
