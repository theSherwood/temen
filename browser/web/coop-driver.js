// The **cooperative tier-up driver** (`driveCoopTierupRun`) — the JS host loop over the cdylib's
// `temen_coop_*` session, and the one driver every embedder runs it with (#1954): the playground's
// runners (`wasmjit-module.js`), the nimony card, and an embedder such as c_interpret, which stages
// this file beside the engine. It depends on nothing but the cdylib's exports and the platform, so it
// can be copied as is.
//
// The session runs one program: on the interpreter, with hot regions or (#1954) the whole root program
// running on emitted wasm. `driveCoopTierupRun(ex, memory, opts)` pumps it to the end and returns the
// run's status; what the embedder adds is in `opts` (see the function).

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
export const JIT_MODULE_CACHE_MAX = 16;
export function cacheGet(key) {
  return key === undefined ? undefined : jitModuleCache.get(key);
}
export function cachePut(key, mod) {
  if (key === undefined) return;
  // Simple LRU-ish bound: drop the oldest insertion when full (Map preserves insertion order).
  if (jitModuleCache.size >= JIT_MODULE_CACHE_MAX && !jitModuleCache.has(key)) {
    jitModuleCache.delete(jitModuleCache.keys().next().value);
  }
  jitModuleCache.set(key, mod);
}
// Test/telemetry hook: how many WebAssembly.compile calls the caches have served vs skipped.
export const jitCacheStats = { compiles: 0, hits: 0 };
export function moduleCacheClear() {
  jitModuleCache.clear();
  jitCacheStats.compiles = 0;
  jitCacheStats.hits = 0;
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

//
// `opts` (all optional):
// - `cacheKey`: a stable identity for the run's region emit, to reuse its compiled Module across runs.
// - `counts`: an object to fill with what ran on the emitted tier (#1896): `leaves`, how many programs
//   ran whole there, and `resumes`, how many parked calls in them resumed.
// - `budget` (#1954): pump in slices of about this many interpreted ops (`temen_coop_run_for`), so the
//   embedder gets control back between them. Emitted code is not counted: a leaf runs until it returns
//   or parks.
// - `onOutput(stdout, stderr)`: a sliced run's output (`Uint8Array`s), handed over as each pump return
//   produces it. Without `budget`, output is read from the stdout/stderr slots once the run is over.
// - `onSlice()`: called, and awaited, when a slice is spent. Resolve to carry on; resolve to `false` to
//   stop the run here (the session is closed and the driver returns `null`). This is where an embedder
//   yields to its event loop, and where it waits out a Pause.
// - `onCapPark({ id, index, args })`: a call to a declared host-completed cap (`temen_coop_open`'s
//   caps) parked the run: `index` names the cap, `args` are its arguments. Resolve to the call's
//   result (a Number or BigInt) and the run goes on; resolve to `null` to stop it, as `onSlice` can.
//   The run's window is readable meanwhile through `temen_coop_read`.
// - `trapDeclines` (default `true`): a trapped run throws ("declined to the interpreter"), for a host
//   that then re-runs the program interpreted. `false` returns the trap status (3) instead, with the
//   trap's name and fault address in the `temen_trap_*`/`temen_fault_addr` slots.
//
// The driver closes the session before it returns; the run's value is `temen_run_value`, its files
// `temen_coop_fs_image`.
export async function driveCoopTierupRun(ex, memory, opts = {}) {
  const { cacheKey, counts = {}, budget, onOutput, onSlice, onCapPark, trapDeclines = true } = opts;
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

  // A sliced run's output since the last pump return, to the embedder.
  const handOutput = () => {
    const take = (ptr, len) => {
      const n = Number(len);
      return n ? u8().slice(Number(ptr), Number(ptr) + n) : new Uint8Array(0);
    };
    const out = take(ex.temen_stdout_ptr(), ex.temen_stdout_len());
    const err = take(ex.temen_stderr_ptr(), ex.temen_stderr_len());
    if (out.length || err.length) onOutput(out, err);
  };
  let stopped = false;
  try {
    for (;;) {
      const ev = budget === undefined ? ex.temen_coop_run() : ex.temen_coop_run_for(BigInt(budget));
      if (budget !== undefined && onOutput) handOutput();
      if (ev === 5 /* COOP_RUN_PAUSED */) {
        // #1954: the slice is spent — the embedder's turn (its event loop, or a Pause).
        if (onSlice && (await onSlice()) === false) {
          stopped = true;
          break;
        }
        continue;
      }
      if (ev === 6 /* COOP_RUN_CAP_PARK */) {
        // #1954: a declared cap call parked the run — `[id, cap index, args…]` — until it is answered.
        const n = ex.temen_coop_cap_len();
        const words = new BigInt64Array(memory.buffer, Number(ex.temen_coop_cap_ptr()), n).slice();
        const value = onCapPark
          ? await onCapPark({ id: words[0], index: Number(words[1]), args: Array.from(words.slice(2)) })
          : null;
        if (value === null || value === undefined) {
          stopped = true;
          break;
        }
        ex.temen_coop_deliver_cap(words[0], BigInt(value));
        continue;
      }
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
  if (stopped) return null;
  const status = ex.temen_status();
  if (status === 3 /* STATUS_TRAP */ && trapDeclines) {
    throw new Error('cooperative tier-up run trapped (declined to the interpreter)');
  }
  return status;
}

