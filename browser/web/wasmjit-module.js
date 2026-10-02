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

import {
  cacheGet, cachePut, jitCacheStats, moduleCacheClear, JIT_MODULE_CACHE_MAX, driveCoopTierupRun,
  suspendsLeaves,
} from './coop-driver.js';

export { jitCacheStats, driveCoopTierupRun, suspendsLeaves };

// A parallel cross-Run cache of the **instantiated** emitted module (issue #803). The compiled Module
// skips V8 codegen; caching the instance additionally skips the per-Run `WebAssembly.instantiate` of the
// ~12.7 MB emitted module *and* the copy of its bytes out of wasm memory (only read on a compile miss).
// Safe because the emitted entry `f0(win, env, ...slots)` takes win/env/sp as **arguments** and holds no
// state between calls — the reactor already reuses one instance across every frame the same way — and its
// linear memory is the imported (stable) engine memory. Keyed by the same stable module identity as
// `jitModuleCache`; a fresh `win`/`sp` and the dynamic cross-tier bounce are supplied per Run by the
// caller, so a reused instance always runs against current state.
const jitInstanceCache = new Map();
export function jitCacheClear() {
  moduleCacheClear();
  jitInstanceCache.clear();
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
  const status = await driveCoopTierupRun(ex, memory, { counts });
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
  if (coop) return driveCoopTierupRun(ex, memory, { cacheKey });
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
  return driveCoopTierupRun(ex, memory, { cacheKey });
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
