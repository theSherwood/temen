// The **cooperative tier-up driver** (`driveCoopTierupRun`) — the JS host loop over the cdylib's
// `temen_coop_*` session, and the one driver every embedder runs it with (#1954): the playground's
// runners (`wasmjit-module.js`), the nimony card, and an embedder such as c_interpret, which stages
// this file beside the engine. It depends on nothing but the cdylib's exports, the platform and
// `foreign-mem.js` (a window's memory of its own), so the two can be copied as they are.
//
// The session runs one program: on the interpreter, with hot regions or (#1954) the whole root program
// running on emitted wasm. `driveCoopTierupRun(ex, memory, opts)` pumps it to the end and returns the
// run's status; what the embedder adds is in `opts` (see the function).

import { foreignMemory, headerOf } from './foreign-mem.js';

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
export const tierupJitArg = (slot, tc) => tc === 0 ? Number(BigInt.asIntN(32, slot))
  : tc === 1 ? slot
  : (f64buf.setBigInt64(0, slot, true), tc === 2 ? f64buf.getFloat32(0, true) : f64buf.getFloat64(0, true));
export const tierupJitRes = (ret, tc) => tc === 0 || tc === 1 ? BigInt(ret)
  : (tc === 2 ? f64buf.setFloat32(0, ret, true) : f64buf.setFloat64(0, ret, true), f64buf.getBigInt64(0, true));

// The **cooperative tier-up driver** (#926 slice 2; since #1026 the ONE fallback tier when the
// whole-program emit declines): the single-thread, no-Worker host loop for an InterpDriven guest —
// single-vCPU or genuinely threaded alike. It wraps the `temen_coop_*` cdylib (`CoopRun`), whose
// scheduler multiplexes every vCPU of the run — the root and its `thread.spawn` descendants — on
// this one wasm thread and services concurrency, fibers, and §22 install/invoke **internally**, so
// only tier-up, a §22 `Jit.invoke` of an emitted unit, and the run's end reach here.
//
// #1312: this run's window **grows** — a guest allocator's `vm_map` past the declared window commits
// real memory rather than being refused. A window in a memory of its own (#1414 5d: a root's or a
// detached child's, on the threads build) grows in place, past a header whose end never moves. One in
// the engine's memory (a process's, and every window of the plain build) grows by reallocating, so
// its base can move mid-run. Nothing here may cache that base or the length: `eventWin()` reads the
// base per event, and the `call_interp` bounce below republishes it to every live instance's `"win"`
// global (the emitted code reloads its `win` from that global after each call). The per-event
// contract (worker.js's PAR_TIERUP shape): sync the B2 shared driver table (`call_indirect` tiers
// up, #880) when its generation advances, re-arm `"fuel"`, write the event's `"mapped"` sync (#717;
// a paged run also points `"pagestate"` at the live table, #1009), call `f{func}(win, env, ...args)`
// over the cdylib's shared memory, and deliver the results (or the trap) back to the parked vCPU;
// `env.call_interp` is the live-state bounce. Proven observably identical to `onramp_exec` by
// tests/coop_tierup_driver.rs (wasmi playing this file's role).
// #1419, #2579 — the page's half of a JS capability, a call answered inside the call: the engine's
// `temen_host.js_cap_call(slot, op, argsPtr, nArgs, mem)` import reaches `globalThis.__temen_js_cap_call`.
// A handler never sees the guest's memory: `mem` is a bounds-checked view of the calling guest's
// window (`temen_jspb_read`/`temen_jspb_write` in `src/jspb.rs`), live only for the call — in a child,
// the child's window. `null` for a guest that declares no memory.
const capEnc = new TextEncoder();
const capDec = new TextDecoder();
export function windowView(ex, memory, mem) {
  if (!mem) return null;
  const N = ex.temen_abi_is64() === 1 ? BigInt : Number; // usize params: i64 on wasm64, i32 on wasm32
  return {
    /** The guest's bytes at `[ptr, ptr+len)`, or `null` if that is not wholly inside its window. */
    bytes(ptr, len) {
      const p = ex.temen_jspb_read(mem, BigInt(ptr), N(len));
      if (!p) return null;
      return new Uint8Array(memory.buffer, Number(p), Number(len)).slice();
    },
    /** The same range decoded as UTF-8 (`''` when out of bounds). */
    str(ptr, len) {
      const b = this.bytes(ptr, len);
      return b ? capDec.decode(b) : '';
    },
    /** Write bytes (or a string) into the guest's window at `ptr`. `true` iff it landed. */
    write(ptr, data) {
      const bytes = typeof data === 'string' ? capEnc.encode(data) : data;
      const buf = ex.temen_alloc(N(bytes.length));
      if (!buf) return false;
      new Uint8Array(memory.buffer).set(bytes, Number(buf));
      const r = ex.temen_jspb_write(mem, BigInt(ptr), buf, N(bytes.length));
      ex.temen_dealloc(buf, N(bytes.length));
      return r === 0;
    },
  };
}

// The `__temen_js_cap_call` servicer over `onCap(slot, op, args, mem)`: `args` are the call's i64
// arguments as BigInts and `mem` its window view. It returns the call's one i64 (a Number or BigInt;
// nothing is 0). A handler that throws answers -EIO (-5): it must not unwind through the engine.
export function jsCapServicer(ex, memory, onCap) {
  return (slot, op, argsPtr, nArgs, mem) => {
    try {
      const n = Number(nArgs);
      const args = n ? Array.from(new BigInt64Array(memory.buffer, Number(argsPtr), n)) : [];
      return BigInt(onCap(slot, op, args, windowView(ex, memory, mem)) ?? 0);
    } catch (e) {
      console.error(`JS capability ${slot} (op ${op}):`, e);
      return -5n;
    }
  };
}

/// #1896 — whether this host can **suspend** a leaf's emitted frames where a call in it parks: JSPI
/// (`WebAssembly.Suspending` / `WebAssembly.promising`). Pass it to `temen_nim_open`: without it the
/// engine offers no leaf that can park, and such a process runs interpreted instead.
export const suspendsLeaves = typeof WebAssembly.Suspending === 'function'
  && typeof WebAssembly.promising === 'function';

// The **emitted tier** one host thread runs emitted code with (#1414 B6-3b): the shared dispatch table
// that code calls through (Model B2, #880), what fills it — the run's region emit, installed §22 units,
// and bounce shims for slots whose function stays on the interpreter — the globals every live
// instance shares, the env cell, the bounce back into the interpreter, and how a trap in emitted code
// is named. The coop driver keeps one for its run, and each thread of the parallel driver one on its
// Worker. `x` reads the run it serves:
// - `tableLog2()`, `nfuncs()`: the table's size, and its natural prefix (the program's functions);
// - `tableGen()`, `slotUnit(slot)`: the table's generation, and the `(domain, unit)` installed at a
//   slot (a BigInt; negative for none);
// - `slotUnitBytes(slot)`, `shimBytes(slot)`: the emitted wasm of a slot's installed unit, and a
//   slot's bounce shim, each `null` when there is none;
// - `callInterp(target, argsPtr, spillLen)`: one bounce into the interpreter (`0` returned, `1`
//   trapped, `2` parked: #1896);
// - `paged()`, `mapped()`, `mappedNow()`, `pagestatePtr()`, `winPtr()`: the window as emitted code
//   sees it — whether it is page-checked, the event's extent (a paged run's table coverage), the
//   extent now, the page-state table, and the base;
// - `deliverFaultAddr(addr)`: where a memory fault in emitted code faulted;
// - `spillBytes()`, `spillPtr()`: the spill stack a collecting guest's frames push to (#1627);
// - `fuelOut()` (optional): emitted code spent its fuel. Return more for every instance to run on
//   with, or nothing to let the code trap `OutOfFuel`; without it, it traps.
// - `envCell` (optional): where in `memory` the env cell is, when the host places it (in the header
//   of a window's own memory: #1414 ED-2, 5d, `windowTier`); without it, the tier allocates one in
//   the engine's memory.
// Everything here is synchronous but `syncTable` and `unitFor`, which the coop driver awaits at its
// event boundaries; a host that cannot await uses `syncTableSync` and `unitForSync`.
// #2126 — a native wasm trap (no `env.trap`) still names itself in its `RuntimeError` message. The
// wording is the engine's, not the spec's, so only messages that say exactly one of the interpreter's
// traps are named (their `temen_ir::trap_code`); any other stays unnamed (`0`). ("integer overflow" is
// not one: SpiderMonkey and JSC say it for an out-of-range float→int conversion too, which the
// interpreter calls `BadConversion`.)
const NATIVE_TRAPS = [
  [/^(divide by zero|remainder by zero|integer divide by zero|division by zero)$/i, 1], // DIV_BY_ZERO
  [/^divide result unrepresentable$/i, 2], // INT_OVERFLOW
  [/^(float unrepresentable in integer range|invalid conversion to integer|out of bounds trunc operation)$/i, 3], // BAD_CONVERSION
];
export const nativeTrap = (e) => {
  if (!(e instanceof WebAssembly.RuntimeError)) return 0;
  const hit = NATIVE_TRAPS.find(([re]) => re.test(e.message));
  return hit ? hit[1] : 0;
};

export function emittedTier(ex, memory, x) {
  const mappedGlobals = []; // every live instance's "mapped" — the post-bounce fan-out set (#717)
  const fuelGlobals = [];
  const pagestateGlobals = []; // #1009 paged: the "pagestate" base globals (only a paged main module has one)
  // #1312: every live instance's "win" — the live window BASE. A window in the engine's memory grows
  // on a guest `vm_map` by reallocating, so its base can move mid-run. The emitted entry
  // publishes its own `win` argument here, so this set only has to be written when the base actually
  // changed: after a bounce that may have grown the window (below). Every emitted function reloads
  // its `win` local from this global after each call, so the write takes effect immediately.
  const winGlobals = [];
  const register = (exports) => {
    if (exports.mapped) mappedGlobals.push(exports.mapped);
    if (exports.fuel) fuelGlobals.push(exports.fuel);
    if (exports.pagestate) pagestateGlobals.push(exports.pagestate);
    if (exports.win) winGlobals.push(exports.win);
  };
  // #846/#880 — the shared driver table (Model B2): the region emit and every §22 unit
  // `call_indirect` through it, and the host populates its slots from the run's dispatch table (an
  // installed unit's emitted `f0`, an emitted program function's `f{i}`, or a bounce shim for an
  // interpreter-resident target). `env.call_interp` bounces a cross-tier helper back through the
  // live-state bounce (routed to the tiering-up task's env), then fans the fresh "mapped" extent out
  // to every live instance. A non-shimmable guest reports `table_log2 == 0` (a 1-slot table) and
  // emits in local-table mode, so the shared table is inert for it.
  const tsize = 1 << x.tableLog2();
  const table = new WebAssembly.Table({ initial: tsize, maximum: tsize, element: 'anyfunc' });
  const envBytes = ex.temen_wasmjit_env_bytes();
  const placed = x.envCell !== undefined;
  const envCell = placed ? x.envCell : Number(ex.temen_alloc(envBytes));
  // #1627: a collecting guest's emitted frames push their live words to a spill stack named by the
  // env cell's cursor pair; every event is an outermost entry, so each one re-arms it at the base.
  const spillBytes = x.spillBytes();
  const spillBase = spillBytes ? Number(x.spillPtr()) : 0;
  const spillOff = ex.temen_wasmjit_spill_sp_off();
  const faultOff = ex.temen_wasmjit_fault_off();
  // #1822 — the code the running emitted frames last passed to `env.trap` before aborting (a memory
  // fault, spent fuel, a spill overflow), handed to the trap deliver so the guest sees that trap. Each
  // event's entry resets it (`armEnv`); `0` — no `env.trap` — is a native wasm trap (`nativeTrap`).
  let lastTrap = 0;
  const recordTrap = (code) => { lastTrap = code; };
  // The `env.trap` every instance made here imports. Emitted code calls it with `OUT_OF_FUEL` when
  // its fuel runs out, and runs on if the counter was refilled.
  const trap = x.fuelOut
    ? (code) => {
      const more = code === 11 /* temen_ir::trap_code::OUT_OF_FUEL */ ? x.fuelOut() : undefined;
      if (more === undefined) return recordTrap(code);
      for (const g of fuelGlobals) g.value = more;
    }
    : recordTrap;
  const armEnv = () => {
    lastTrap = 0;
    const dv = new DataView(memory.buffer);
    dv.setBigInt64(envCell, 1n << 61n, true);
    if (spillBase) {
      dv.setUint32(envCell + spillOff, spillBase, true);
      dv.setUint32(envCell + spillOff + 4, spillBase + spillBytes, true);
    }
  };
  const bounce = (target, argsPtr) => {
    // #1627: a spilling run hands the bounce the words its emitted frames pushed, `[base, cursor)`.
    const spillLen = spillBase
      ? (new DataView(memory.buffer).getUint32(envCell + spillOff, true) - spillBase) / 8
      : 0;
    return x.callInterp(target, argsPtr, spillLen);
  };
  // The region emit's exports, whose `f{i}` fill the table's natural prefix (`setProgram`).
  let program = {};
  // #1009: rebuild the table only when the run's dispatch table changed (a §22 install/uninstall
  // moves its generation) — a card that never installs syncs the table once, not per tier-up.
  let syncedGen = -1;
  // What a bounce's return fans out to the live instances — and a parked call's (#1896).
  const afterBounce = (rc) => {
    // #1233: the bounce may have been a `Jit.install`/`uninstall` issued from the emitted frame
    // itself (Forth's outer interpreter defining a word, then `call.dyn`ing it) — the table moved
    // mid-event, and the frame's next `call_indirect` must find the new occupant, not a stale or
    // empty slot. Rebuild synchronously, before the globals fan-out below primes any instance this
    // creates (a word unit is tiny; one over the sync compile budget gets a bounce shim now and its
    // emitted unit at the next event-boundary sync).
    if (rc === 0 && x.tableGen() !== syncedGen) syncTableSync();
    // #1009 paged: the grow rebuilt the page-state table (in `call_interp`) — fan the fresh coverage
    // to "mapped" and re-point "pagestate"; else the #717 scalar extent (the pump's twin).
    if (x.paged()) {
      const cover = x.mapped();
      for (const g of mappedGlobals) g.value = cover;
      const ps = Number(x.pagestatePtr());
      for (const g of pagestateGlobals) g.value = ps;
    } else {
      const now = x.mappedNow();
      for (const g of mappedGlobals) g.value = now;
    }
    // #1312: the bounce ran interpreted guest code, which may have `vm_map`-grown the window. In the
    // engine's memory a grow reallocates the backing and can MOVE it, so publish the current base to
    // every live instance — the emitted frame reloads its `win` from this global on return from the
    // bounce. Read it fresh here (never cached): this is the one point in the run where it can change.
    // (A window in a memory of its own grows in place: its base is the header's end, always.)
    const base = Number(x.winPtr());
    for (const g of winGlobals) g.value = base;
  };
  const callInterp = (target, argsPtr) => {
    const rc = bounce(target, argsPtr);
    afterBounce(rc);
    if (rc !== 0) throw new Error('bounce trap'); // unwind to the deliver below
  };
  const imports = (call_interp = callInterp, onTrap = trap) => ({ env: {
    memory,
    __indirect_function_table: table,
    trap: onTrap,
    call_interp,
  } });
  // The trap emitted frames ended with (`e`, what they threw): their `env.trap` code, else the native
  // trap's. A memory fault's guard left its faulting address in the env cell (#2126), which the run
  // reports as the interpreter would.
  const trapOf = (e) => {
    const code = lastTrap || nativeTrap(e);
    if (code === 8 /* MEMORY_FAULT */) {
      x.deliverFaultAddr(new DataView(memory.buffer).getBigInt64(envCell + faultOff, true));
    }
    return code;
  };
  // Per-code-handle unit instances (a runtime-compiled §22 unit runs emitted on JIT_INVOKE — the
  // JACL macro-staging shape). Async instantiation: a macro unit can exceed the sync compile budget.
  const jitUnits = new Map();
  const shims = new Map();
  const instantiateUnit = async (bytes) => {
    const inst = await WebAssembly.instantiate(await WebAssembly.compile(bytes), imports());
    register(inst.exports);
    return inst.exports;
  };
  // #1233: the synchronous twin, for a rebuild inside a bounce (no event boundary to await at). An
  // instance created mid-event never passes the per-event fuel re-arm — budget it now.
  const instantiateUnitSync = (bytes) => {
    const inst = new WebAssembly.Instance(new WebAssembly.Module(bytes), imports());
    register(inst.exports);
    if (inst.exports.fuel) inst.exports.fuel.value = 1n << 61n;
    return inst.exports;
  };
  const shimFor = async (slot, code) => {
    const key = `${slot}#${code}`;
    let f = shims.get(key);
    if (f === undefined) {
      const bytes = x.shimBytes(slot);
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
      const bytes = x.shimBytes(slot);
      if (bytes === null) return null;
      f = instantiateUnitSync(bytes)['t'];
      shims.set(key, f);
    }
    return f;
  };
  // `key`: a surfaced JIT_INVOKE's code handle (a Number — live for that invoke), or an installed
  // slot's `(domain, unit)` identity (`slotUnit`, a BigInt) — distinct key types, one cache. An
  // installed slot is never keyed by handle: the guest revokes it right after `install`.
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
  // `unitFor` for a host that cannot await: a Worker, which may compile a module of any size.
  const unitForSync = (key, bytes) => {
    let unit = jitUnits.get(key);
    if (unit === undefined) {
      unit = instantiateUnitSync(bytes);
      jitUnits.set(key, unit);
    }
    return unit;
  };
  // Rebuild the shared table from the run's dispatch table whenever its generation moved: at each
  // event boundary (`syncTable`) and — #1233 — inside `env.call_interp` after a bounce that installed
  // or uninstalled (`syncTableSync`). A slot in the natural prefix holds the emitted program `f{slot}`
  // (or a bounce shim if that function stayed interpreted); a slot past it holds an installed unit's
  // `f0` (fetched by slot, cached by unit identity) or a shim for an interpreter-resident target.
  const nfuncs = x.nfuncs();
  const syncTable = async () => {
    const gen = x.tableGen();
    if (gen === syncedGen) return;
    for (let slot = 0; slot < tsize; slot++) {
      let entry = null;
      if (slot < nfuncs) {
        entry = program['f' + slot] ?? await shimForFast(slot, -2);
      } else {
        const uid = x.slotUnit(slot);
        if (uid >= 0n) {
          const cached = jitUnits.get(uid);
          if (cached !== undefined) entry = cached['f0'];
          else {
            const bytes = x.slotUnitBytes(slot);
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
    const gen = x.tableGen();
    if (gen === syncedGen) return;
    for (let slot = 0; slot < tsize; slot++) {
      let entry = null;
      if (slot < nfuncs) {
        entry = program['f' + slot] ?? shimForSync(slot, -2);
      } else {
        const uid = x.slotUnit(slot);
        if (uid >= 0n) {
          const cached = jitUnits.get(uid);
          if (cached !== undefined) entry = cached['f0'];
          else {
            const bytes = x.slotUnitBytes(slot);
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
  return {
    memory, imports, register, mappedGlobals, fuelGlobals, pagestateGlobals, envCell, armEnv, bounce,
    afterBounce, callInterp, recordTrap, trapOf, unitFor, unitForSync, syncTable, syncTableSync,
    // The region emit's exports, whose `f{i}` fill the table's natural prefix.
    program: () => program,
    setProgram: (exports) => {
      program = exports;
      register(exports);
    },
    free: () => {
      if (!placed) ex.temen_dealloc(envCell, envBytes);
    },
  };
}

// The emitted tier over a window's own memory (#1414 ED-2, 5d), a root's or a detached child's, on
// either executor: `emittedTier` over `windowMemory`, in context 0 of the memory's header
// (`headerOf`), its window from the header's end (`win`). A root's code calls through the run's shared
// table, so its `x` names the table as `emittedTier`'s does; a child's module dispatches through a
// table of its own functions, so its `x` leaves the table out and its shared table stays empty. The
// engine reads a bounce's slots from its own memory, so a bounce copies the call scratch there and
// back, and the words a spilling emit pushed to the engine's spill buffer, which `gc.roots` scans. A
// page-checked emit reads its page-state table from the context: `pagestateAt()` copies the engine's
// table in, when it changed, and answers where. Besides `emittedTier`'s, `x` has `scratch()`, the
// engine address of `temen_wasmjit_call_scratch_bytes()` bytes the bounces copy through, the
// caller's to free; `pagestate()`, the `[ptr, len, stamp]` of the table in the engine's memory, the
// stamp changing whenever the engine rebuilds it (`undefined`: copy it every time); and, for a
// spilling emit, `spillBytes()` and `spillPtr()`, the engine's spill buffer.
export function windowTier(ex, memory, windowMemory, x) {
  const h = headerOf(windowMemory);
  const env = h.context0;
  if (ex.temen_wasmjit_env_bytes() > h.tableOff) throw new Error('env cell does not fit the window header');
  const tableAt = env + h.tableOff, spillAt = env + h.spillOff;
  const scratchBytes = ex.temen_wasmjit_call_scratch_bytes();
  const spillBytes = Math.min(h.spillBytes, x.spillBytes?.() ?? 0);
  const span = (m, at, n) => new Uint8Array(m.buffer, at, n);
  // What a bounce copies between, viewed once: a view of a shared memory stays valid as the memory
  // grows, the header never moves, and neither does the engine's scratch nor its spill buffer while
  // they are held. Made again only when one of them is not the one viewed.
  let views = null;
  const viewsFor = (argsPtr, scratch) => {
    if (views?.argsPtr !== argsPtr || views.scratch !== scratch) {
      views = {
        argsPtr, scratch, win: span(windowMemory, argsPtr, scratchBytes), eng: span(memory, scratch, scratchBytes),
      };
    }
    return views;
  };
  let spills = null;
  const spillsFor = (ptr) => {
    if (spills?.ptr !== ptr) spills = { ptr, win: span(windowMemory, spillAt, spillBytes), eng: span(memory, ptr, spillBytes) };
    return spills;
  };
  let copied; // the stamp of the table the context holds
  const pagestateAt = () => {
    const [p, n, stamp] = x.pagestate();
    if (n > h.tableBytes) throw new Error('pagestate table does not fit the window header');
    if (stamp === undefined || stamp !== copied) {
      span(windowMemory, tableAt, n).set(span(memory, p, n));
      copied = stamp;
    }
    return tableAt;
  };
  const tier = emittedTier(ex, windowMemory, {
    tableLog2: () => 0,
    nfuncs: () => 0,
    slotUnit: () => -1n,
    slotUnitBytes: () => null,
    shimBytes: () => null,
    ...x,
    envCell: env,
    callInterp: (target, argsPtr, spillLen) => {
      const scratch = x.scratch();
      const v = viewsFor(argsPtr, scratch);
      v.eng.set(v.win);
      if (spillLen) {
        const sp = spillsFor(Number(x.spillPtr()));
        sp.eng.set(sp.win.subarray(0, spillLen * 8));
      }
      const rc = x.callInterp(target, scratch, spillLen);
      v.win.set(v.eng);
      return rc;
    },
    pagestatePtr: pagestateAt,
    winPtr: () => h.base,
    spillBytes: () => spillBytes,
    spillPtr: () => spillAt,
  });
  return { ...tier, win: h.base, pagestateAt };
}

// Free what a driver `session` kept (see `driveCoopTierupRun`): the emitted tiers it made, and with
// them the leaves instantiated over them.
export function endSession(session) {
  session.host?.free();
  session.host = undefined;
  session.windows = undefined;
}

//
// `opts` (all optional):
// - `cacheKey`: a stable identity for the run's region emit, to reuse its compiled Module across runs.
// - `counts`: an object to fill with what ran on the emitted tier (#1896): `leaves`, how many programs
//   ran whole there, and `resumes`, how many parked calls in them resumed.
// - `budget` (#1954): pump in slices of about this many interpreted ops (`temen_coop_run_for`), so the
//   embedder gets control back between them. Where the host suspends emitted frames (JSPI), a leaf is
//   sliced too: its fuel counter is armed with `leafBudget` (default `budget`) units — one per function
//   entry and taken back-edge — and when it runs out the leaf's frames suspend for `onSlice`, then run
//   on with a fresh slice. Elsewhere emitted code is not counted: a leaf runs until it returns or parks.
// - `leafBudget`: a leaf's slice, in emitted safepoints (see `budget`).
// - `onOutput(stdout, stderr)`: the run's output (`Uint8Array`s) as it writes it, handed over each time
//   the driver gets control back: at each pump return (each slice, with `budget`), and at a sliced
//   leaf's checkpoints. The stdout/stderr slots hold all of it once the run is over, as they do without
//   `onOutput`.
// - `onSlice()`: called, and awaited, when a slice is spent — interpreted or a leaf's. Resolve to carry on; resolve to `false` to
//   stop the run here (the session is closed and the driver returns `null`). This is where an embedder
//   yields to its event loop, and where it waits out a Pause.
// - `onCapPark({ id, index, op, args })`: a call to a declared host-completed cap (`temen_coop_open`'s
//   caps) parked its task, in any domain of the run (a child the cap was re-granted to files its
//   calls here too): `index` names the cap, `op` is the op the call dispatched with (0 for a flat
//   `call.sym` import, whose own op rides in `args[0]`), `args` are its arguments. Resolve to the call's
//   result (a Number or BigInt) and the driver delivers it before its next pump; resolve to `null`
//   to stop the run, as `onSlice` can. The run goes on meanwhile (#2557): its other tasks run, and
//   its timers fire in the page's time. With nothing left to run, the driver waits for an answer
//   or the run's next timer, whichever comes first. The run's window is readable meanwhile
//   through `temen_coop_read`.
// - `onCap(slot, op, args, mem)` (#2579): a call to a JS capability the embedder staged before the
//   open (`temen_onramp_stage_js_cap`, which returned `slot`), answered inside the call: `op` is the
//   guest's op, `args` its arguments (BigInts) and `mem` the calling guest's window, a child's
//   included (`windowView`). Return its result (a Number or BigInt); throwing answers -EIO. It is
//   installed for the run's duration, and runs synchronously: from the pump, or from a bounce out
//   of emitted code.
// - `trapDeclines` (default `true`): a trapped run throws ("declined to the interpreter"), for a host
//   that then re-runs the program interpreted. `false` returns the trap status (3) instead, with the
//   trap's name and fault address in the `temen_trap_*`/`temen_fault_addr` slots. That is exact for a
//   trap on the interpreter. A trap in emitted code (a hot region, or a leaf: `counts.leaves > 0`) is
//   named by the code its `env.trap` reported (#1822: a memory fault, spent fuel, a spill overflow),
//   but a native wasm trap (division by zero, an `unreachable`) reads as `Unreachable`, and no emitted
//   trap has a fault address. A host that shows those re-runs such a run interpreted, as c_interpret
//   does (replaying the cap answers the first run got).
// - `session` (#1414 S4-4): an object the caller keeps across runs of one program — a reactor's frames.
//   The driver keeps the emitted tiers it makes in it, with each leaf it instantiates over them, so a
//   run after the first instantiates nothing. A session's leaves never park (the engine offers a
//   reactor's leaf only when it cannot), so they run without JSPI, over imports the session keeps.
//   Free it with `endSession`.
//
// The driver closes the session before it returns; the run's value is `temen_run_value`, its files
// `temen_coop_fs_image`.
export async function driveCoopTierupRun(ex, memory, opts = {}) {
  const {
    cacheKey, counts = {}, budget, leafBudget = budget, onOutput, onSlice, onCapPark, onCap,
    trapDeclines = true, session,
  } = opts;
  // #1896: how many processes ran whole on the emitted tier, and how many parked calls in them
  // resumed. A caller that wants to know passes the object to fill.
  counts.leaves = 0;
  counts.resumes = 0;
  const u8 = () => new Uint8Array(memory.buffer);
  const i64 = () => new BigInt64Array(memory.buffer);
  // #1414 5d — the tier an event runs on is its window's memory's. A root's window and a detached
  // child's are memories of their own (`windowTier`), one tier per memory, made at the memory's first
  // event and kept as long as the memory: `windows` lets go of a tier once the registry lets go of its
  // memory. A root's tier calls through the run's shared table; a child's doesn't. A window still in
  // the engine's memory (a process's, and every window of the plain build) runs on the engine's tier,
  // `emittedTier` over the engine's memory, where the window is per event: the pending task's base
  // (`temen_coop_tierup_win_ptr`), read in each event arm, never cached across events.
  const eventWin = () => Number(ex.temen_coop_tierup_win_ptr());
  const runTable = {
    tableLog2: () => ex.temen_coop_table_log2(),
    nfuncs: () => ex.temen_coop_nfuncs(),
    tableGen: () => ex.temen_coop_table_gen(),
    slotUnit: (slot) => ex.temen_coop_slot_unit(slot),
    // The bytes of the unit installed at `slot` (`null` = interpreter-only), fetched **by slot** —
    // the guest typically `release`s the code handle right after `install` (the unit stays installed,
    // only its handle dies), so a by-handle fetch would come back empty and null the slot (#1233).
    slotUnitBytes: (slot) => {
      const len = ex.temen_coop_jit_wasm_by_slot_len(slot);
      if (len === 0) return null;
      const p = Number(ex.temen_coop_jit_wasm_by_handle_ptr());
      return u8().slice(p, p + len);
    },
    shimBytes: (slot) => {
      const len = ex.temen_coop_shim_wasm(slot);
      if (len === 0) return null;
      return u8().slice(Number(ex.temen_coop_shim_ptr()), Number(ex.temen_coop_shim_ptr()) + len);
    },
    spillBytes: () => ex.temen_coop_spill_bytes(),
    spillPtr: () => ex.temen_coop_spill_ptr(),
  };
  const run = {
    tableGen: runTable.tableGen,
    callInterp: (target, argsPtr, spillLen) => ex.temen_coop_call_interp(target, argsPtr, spillLen),
    paged: () => ex.temen_coop_paged(),
    mapped: () => ex.temen_coop_mapped(),
    mappedNow: () => ex.temen_coop_mapped_now(),
    deliverFaultAddr: (addr) => ex.temen_coop_deliver_fault_addr(addr),
  };
  let engine = session?.host;
  const engineTier = () => {
    engine ??= emittedTier(ex, memory, {
      ...runTable,
      ...run,
      pagestatePtr: () => ex.temen_coop_pagestate_ptr(),
      winPtr: () => ex.temen_coop_tierup_win_ptr(),
    });
    if (session) session.host = engine;
    return engine;
  };
  // The engine scratch a window tier's bounces copy through: one at a time runs, so they share it. It
  // is made at the first bounce of a run and freed at its end; a session's tiers find the next run's.
  const scratchBytes = ex.temen_wasmjit_call_scratch_bytes();
  const held = session ? (session.scratch ??= { at: 0 }) : { at: 0 };
  const scratch = () => (held.at ||= Number(ex.temen_alloc(scratchBytes)));
  const pagestate = () => [
    Number(ex.temen_coop_pagestate_ptr()), ex.temen_coop_pagestate_len(), ex.temen_coop_pagestate_stamp(),
  ];
  const windows = session ? (session.windows ??= new WeakMap()) : new WeakMap();
  const windowOf = (wm) => {
    let tier = windows.get(wm);
    if (tier === undefined) {
      const root = ex.temen_coop_tierup_root() !== 0;
      tier = windowTier(ex, memory, wm, { ...(root ? runTable : {}), ...run, scratch, pagestate });
      tier.child = !root;
      windows.set(wm, tier);
    }
    return tier;
  };

  // #1896 — a leaf's `call_interp` under JSPI. A call that parks (`2`) suspends the leaf's frames on
  // the promise returned here, and the driver runs on; `COOP_RUN_RESUME` resolves it once the call
  // has returned, with its results in the call's scratch, in its tier's memory.
  let parking = null; // the running leaf's parked call: its tier, its scratch, and what resumes its frames
  let parked = () => {}; // tells `settle` the running leaf parked
  const leafCallInterp = (tier) => suspendsLeaves
    ? new WebAssembly.Suspending((target, argsPtr) => {
      const rc = tier.bounce(target, argsPtr);
      if (rc === 2) {
        return new Promise((resolve) => {
          parking = { tier, argsPtr, resolve };
          parked();
        });
      }
      tier.afterBounce(rc);
      if (rc !== 0) throw new Error('bounce trap');
    })
    : tier.callInterp;
  // #1954 — a sliced leaf's budget checkpoint under JSPI. Its emitted code calls `env.trap(OUT_OF_FUEL)`
  // when its fuel counter runs out and aborts only if the counter is still negative afterwards; this
  // import suspends the leaf's frames there, hands over what its calls printed, gives the embedder
  // its turn (`onSlice`), and refills the counter, so the frames run on. `onSlice` resolving `false`
  // drops the frames instead and tells `settle` the run is stopped. Any other trap code is recorded
  // (`recordTrap`) and returns, and the emitted code aborts as it always does.
  const sliceLeaves = suspendsLeaves && budget !== undefined;
  const leafFuel = BigInt(leafBudget ?? 0);
  const leafFuelGlobals = []; // the sliced leaves' "fuel" globals, armed with `leafFuel` per event
  let stopLeaf = () => {}; // tells `settle` the embedder stopped the run at a leaf's checkpoint
  const leafTrap = (tier) => sliceLeaves
    ? new WebAssembly.Suspending(async (code) => {
      if (code !== 11 /* temen_ir::trap_code::OUT_OF_FUEL */) return tier.recordTrap(code);
      handOutput();
      if (onSlice && (await onSlice()) === false) {
        stopLeaf();
        return new Promise(() => {}); // never resumed: the session closes under the frames
      }
      for (const g of leafFuelGlobals) g.value = leafFuel;
    })
    : tier.recordTrap;

  // The run's own emit (program 0), compiled once for the run and instantiated over each tier that
  // runs it. A nimony build (`temen_nim_open`) emits none of its own: its programs are the leaf images
  // its processes exec (#1896), each instantiated at its first TIERUP — and neither does a run whose
  // root program runs whole as a leaf (#1954): its program 0 is that leaf.
  let region = null;
  if (ex.temen_coop_wasm_len() > 0) {
    const coopKey = cacheKey === undefined ? undefined : `${cacheKey}#coop`;
    region = cacheGet(coopKey);
    if (region === undefined) {
      const wptr = Number(ex.temen_coop_wasm_ptr());
      const wlen = ex.temen_coop_wasm_len();
      region = await WebAssembly.compile(u8().slice(wptr, wptr + wlen));
      cachePut(coopKey, region);
      jitCacheStats.compiles++;
    } else {
      jitCacheStats.hits++;
    }
  }
  // What this run has instantiated over each tier: its programs, by index (the run's own), and its
  // leaves' JSPI imports. A child's tier runs its granted module alone; any other tier fills its
  // shared table's natural prefix from the run's own emit, when there is one.
  const drives = new WeakMap();
  const driveOf = async (tier) => {
    let d = drives.get(tier);
    if (d === undefined) {
      d = { programs: new Map(), leafImports: null };
      drives.set(tier, d);
      if (!tier.child) {
        const emitted = region ? (await WebAssembly.instantiate(region, tier.imports())).exports : {};
        tier.setProgram(emitted);
        if (region) d.programs.set(0, emitted);
      }
    }
    return d;
  };
  // A program the engine emitted on its own (#1896, #2248): a leaf image, run whole from its entry
  // (a root leaf is program 0, #1954), or a granted module a child runs, whose functions tier up as
  // the run's own do. Any other program is the run's own emit. The engine's answers, not guesses
  // from the index.
  const isLeaf = (m) => ex.temen_coop_program_leaf(m) !== 0;
  // Program `m`'s compiled module and its cache key. A program's index is the run's own, so the
  // cross-Run cache keys it by the engine's key: its image's digest and how it was emitted (#2087). A
  // later build's image of the same command is found without reading, hashing or compiling its
  // bytes. Holding the compiled Module keeps its code for the next build in this page: V8 otherwise
  // frees a Module nothing references, and the next build compiles it cold again.
  const compiled = async (m) => {
    const kn = ex.temen_coop_program_key(m);
    const kp = Number(ex.temen_coop_program_key_ptr());
    const key = `program:${Array.from(u8().subarray(kp, kp + kn), (b) => b.toString(16).padStart(2, '0')).join('')}`;
    let module = cacheGet(key);
    if (module === undefined) {
      const ptr = Number(ex.temen_coop_program_wasm_ptr(m));
      const bytes = u8().slice(ptr, ptr + ex.temen_coop_program_wasm_len(m));
      module = await WebAssembly.compile(bytes);
      cachePut(key, module);
      jitCacheStats.compiles++;
    } else {
      jitCacheStats.hits++;
    }
    return [key, module];
  };
  // Program `m` over `tier`. A session keeps its tiers' leaves, by key, for its next runs.
  const programFor = async (tier, m) => {
    const d = await driveOf(tier);
    let p = d.programs.get(m);
    if (p === undefined && ex.temen_coop_program_wasm_len(m) === 0) p = tier.program();
    if (p === undefined) {
      const [key, module] = await compiled(m);
      const leaf = isLeaf(m);
      p = session && leaf ? tier.leaves?.get(key) : undefined;
      if (p === undefined) {
        const imports = leaf && !session
          ? (d.leafImports ??= tier.imports(leafCallInterp(tier), leafTrap(tier)))
          : tier.imports();
        p = (await WebAssembly.instantiate(module, imports)).exports;
        if (leaf && sliceLeaves && p.fuel) leafFuelGlobals.push(p.fuel);
        tier.register(p);
        if (session && leaf) (tier.leaves ??= new Map()).set(key, p);
      }
      d.programs.set(m, p);
    }
    return p;
  };
  // The tier the pending event runs on, made ready for this run.
  const pendingTier = async () => {
    const mem = ex.temen_coop_tierup_mem();
    const tier = mem < 0 ? engineTier() : windowOf(foreignMemory(memory, mem));
    await driveOf(tier);
    return tier;
  };
  // Where the pending event's window starts in its tier's memory.
  const winOf = (tier) => (tier === engine ? eventWin() : tier.win);
  // #1009 paged: point the tier's page checks at the event's table — the engine's (its base can move
  // as the pump's Vec reallocates, so read it every event), or a window tier's copy in its header.
  const pointPagestate = (tier) => {
    if (!ex.temen_coop_paged()) return;
    const ps = tier === engine ? Number(ex.temen_coop_pagestate_ptr()) : tier.pagestateAt();
    for (const g of tier.pagestateGlobals) g.value = ps;
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
  // Resolves to `true` when the embedder stopped the run at one of the leaf's budget checkpoints.
  const settle = async (task, tier, go) => {
    const parks = new Promise((r) => { parked = r; });
    const stops = new Promise((r) => { stopLeaf = r; });
    const run = go();
    const end = await Promise.race([
      run.then((ret) => ({ ret }), (e) => ({ trapped: e })),
      parks.then(() => ({ parks: true })),
      stops.then(() => ({ stopped: true })),
    ]);
    parked = () => {};
    stopLeaf = () => {};
    if (end.stopped) return true;
    if (end.parks) {
      suspended.set(task, { run, ...parking });
      parking = null;
    } else if ('trapped' in end) {
      ex.temen_coop_deliver_trap(tier.trapOf(end.trapped));
    } else {
      deliver(end.ret);
    }
    return false;
  };

  // The run's output as it writes it, queued by the `stream_chunk` import's sink (#1414 S5a) and handed
  // to the embedder when the driver has control.
  const queued = [[], []];
  const sink = globalThis.__temen_stream_chunk;
  if (onOutput) globalThis.__temen_stream_chunk = (fd, seq, bytes) => queued[fd === 2 ? 1 : 0].push(bytes);
  const capCall = globalThis.__temen_js_cap_call;
  if (onCap) globalThis.__temen_js_cap_call = jsCapServicer(ex, memory, onCap);
  const handOutput = () => {
    if (!onOutput || !(queued[0].length || queued[1].length)) return;
    const [out, err] = queued.map((q) => {
      const all = new Uint8Array(q.reduce((n, b) => n + b.length, 0));
      q.reduce((at, b) => (all.set(b, at), at + b.length), 0);
      q.length = 0;
      return all;
    });
    onOutput(out, err);
  };
  // #2557: the answers `onCapPark` gave, in the order they came, each delivered before the next
  // pump; `answered` ends an idle wait when one comes.
  const answers = [];
  let answered = () => {};
  const deliverAnswers = () => {
    while (answers.length) {
      const a = answers.shift();
      if (!a.ok) throw a.error;
      if (a.value === null || a.value === undefined) return false;
      ex.temen_coop_deliver_cap(a.id, BigInt(a.value));
    }
    return true;
  };
  let stopped = false;
  try {
    for (;;) {
      if (!deliverAnswers()) {
        stopped = true;
        break;
      }
      const ev = budget === undefined ? ex.temen_coop_run() : ex.temen_coop_run_for(BigInt(budget));
      handOutput();
      if (ev === 5 /* COOP_RUN_PAUSED */) {
        // #1954: the slice is spent — the embedder's turn (its event loop, or a Pause).
        if (onSlice && (await onSlice()) === false) {
          stopped = true;
          break;
        }
        continue;
      }
      if (ev === 6 /* COOP_RUN_CAP_PARK */) {
        // #1954: a declared cap call parked its task — `[id, cap index, op, args…]` — until it
        // is answered. #2557: the run goes on meanwhile, and the answer is delivered once it comes.
        if (!onCapPark) {
          stopped = true;
          break;
        }
        const n = ex.temen_coop_cap_len();
        const words = new BigInt64Array(memory.buffer, Number(ex.temen_coop_cap_ptr()), n).slice();
        const id = words[0];
        const call = { id, index: Number(words[1]), op: Number(words[2]), args: Array.from(words.slice(3)) };
        Promise.resolve(onCapPark(call))
          .then((value) => ({ id, ok: true, value }), (error) => ({ id, ok: false, error }))
          .then((a) => {
            answers.push(a);
            answered();
          });
        continue;
      }
      if (ev === 7 /* COOP_RUN_IDLE */) {
        // #2557: nothing can run until an answer comes, or until the run's next timer (`-1`: none).
        // Wait for whichever comes first, then let the time that passed pass in the run.
        if (answers.length === 0) {
          const timer = ex.temen_coop_idle_timer();
          const began = performance.now();
          await new Promise((wake) => {
            answered = wake;
            // A delay past 2^31 - 1 ms would fire at once: wait that long, and look again.
            if (timer >= 0n) setTimeout(wake, Math.min(Number(timer) / 1e6, 0x7fffffff));
          });
          answered = () => {};
          ex.temen_coop_pass_time(BigInt(Math.round((performance.now() - began) * 1e6)));
        }
        continue;
      }
      if (ev === 4 /* COOP_RUN_RESUME */) {
        // #1896: a leaf's parked call returned. Its results go to the call's scratch, the globals
        // fan out as after any bounce, and the leaf's frames run on.
        counts.resumes++;
        const task = ex.temen_coop_task();
        const leaf = suspended.get(task);
        suspended.delete(task);
        const { tier } = leaf;
        const from = Number(ex.temen_coop_argv_ptr()) >> 3;
        const to = new BigInt64Array(tier.memory.buffer);
        for (let i = 0; i < ex.temen_coop_argv_len(); i++) to[(leaf.argsPtr >> 3) + i] = i64()[from + i];
        await tier.syncTable();
        tier.afterBounce(0);
        for (const g of tier.fuelGlobals) g.value = 1n << 61n;
        for (const g of leafFuelGlobals) g.value = leafFuel;
        tier.armEnv();
        const stop = await settle(task, tier, () => {
          leaf.resolve();
          return leaf.run;
        });
        if (stop) {
          stopped = true;
          break;
        }
        continue;
      }
      if (ev === 3 /* COOP_RUN_JIT_INVOKE */) {
        // A guest-compiled §22 unit with emitted wasm: sync the table (its `call_indirect` may reach
        // installed units / program `f{i}`s / bounce shims), instantiate once per code handle, then
        // `f0(win, env, ...args)` with the per-event "mapped"/fuel sync fanned to every live instance.
        const tier = await pendingTier();
        await tier.syncTable();
        const code = ex.temen_coop_jit_code();
        const wptr = Number(ex.temen_coop_jit_wasm_ptr());
        const unit = await tier.unitFor(code, u8().slice(wptr, wptr + ex.temen_coop_jit_wasm_len()));
        const argvPtr = Number(ex.temen_coop_argv_ptr());
        const n = ex.temen_coop_argv_len();
        const ptypes = new Uint8Array(memory.buffer, Number(ex.temen_coop_jit_param_types_ptr()), n);
        const args = [];
        for (let i = 0; i < n; i++) args.push(tierupJitArg(i64()[(argvPtr >> 3) + i], ptypes[i]));
        const mapped = ex.temen_coop_mapped();
        for (const g of tier.mappedGlobals) g.value = mapped;
        for (const g of tier.fuelGlobals) g.value = 1n << 61n;
        // #1334 paged: the unit's `call_indirect` can reach a paged program `f{i}` — point its page
        // check at the table the engine refreshed for this event (as the TIERUP arm does below).
        pointPagestate(tier);
        tier.armEnv();
        try {
          const ret = unit['f0'](winOf(tier), tier.envCell, ...args);
          const rets = ret === undefined ? [] : Array.isArray(ret) ? ret : [ret];
          const rn = ex.temen_coop_jit_result_types_len();
          const rtypes = new Uint8Array(memory.buffer, Number(ex.temen_coop_jit_result_types_ptr()), rn);
          const rlen = Math.max(1, rets.length) * 8;
          const rptr = Number(ex.temen_alloc(rlen));
          for (let i = 0; i < rets.length; i++) i64()[(rptr >> 3) + i] = tierupJitRes(rets[i], rtypes[i]);
          ex.temen_coop_deliver_jit(rptr, rets.length);
          ex.temen_dealloc(rptr, rlen);
        } catch (e) {
          ex.temen_coop_deliver_jit_trap(tier.trapOf(e));
        }
        continue;
      }
      if (ev !== 1 /* COOP_RUN_TIERUP */) break; // 0 = done (slots staged), 2 = trapped (status 3)
      // #1414 5d: the call runs on its window's memory's tier.
      const tier = await pendingTier();
      // #880: a tiered-up leaf's `call_indirect` dispatches through the shared table — sync it before
      // running the region (the per-event "mapped"/fuel fan-out covers every instance it may reach).
      await tier.syncTable();
      const func = ex.temen_coop_func();
      // Before the per-event fan-out below, so a program instantiated now gets this event's sync.
      const m = ex.temen_coop_module();
      const leaf = isLeaf(m);
      if (leaf) counts.leaves++;
      const program = await programFor(tier, m);
      const argvPtr = Number(ex.temen_coop_argv_ptr());
      const n = ex.temen_coop_argv_len();
      const args = [];
      for (let i = 0; i < n; i++) args.push(i64()[(argvPtr >> 3) + i]);
      // #717 host sync: the event's committed extent → every live "mapped" global, so the emitted
      // bounds checks admit exactly what the interpreter's page map does for this call.
      const tmapped = ex.temen_coop_mapped();
      for (const g of tier.mappedGlobals) g.value = tmapped;
      for (const g of tier.fuelGlobals) g.value = 1n << 61n; // re-arm across events on the reused instance
      for (const g of leafFuelGlobals) g.value = leafFuel; // #1954: a sliced leaf's own slice
      // #1009 paged: point the emitted page check at the freshly rebuilt table. Empty set /
      // `_paged==0` on an unpaged run.
      pointPagestate(tier);
      tier.armEnv();
      // #1896: a leaf program runs under JSPI where there is one, so that a call in it can park — but
      // not a session's, which cannot.
      if (suspendsLeaves && leaf && !session) {
        const entry = WebAssembly.promising(program['f' + func]);
        if (await settle(ex.temen_coop_task(), tier, () => entry(winOf(tier), tier.envCell, ...args))) {
          stopped = true;
          break;
        }
        continue;
      }
      const fuel = program.fuel, armed = fuel?.value;
      try {
        const ret = program['f' + func](winOf(tier), tier.envCell, ...args);
        // #2474: the work the call did, for the run's tier-up profile.
        if (fuel && armed > fuel.value) ex.temen_coop_set_emitted_work(armed - fuel.value);
        deliver(ret);
      } catch (e) {
        ex.temen_coop_deliver_trap(tier.trapOf(e));
      }
    }
  } finally {
    if (!session) engine?.free();
    if (held.at) {
      ex.temen_dealloc(held.at, scratchBytes);
      held.at = 0;
    }
    ex.temen_coop_close();
    handOutput();
    if (onOutput) globalThis.__temen_stream_chunk = sink;
    if (onCap) globalThis.__temen_js_cap_call = capCall;
  }
  if (stopped) return null;
  const status = ex.temen_status();
  if (status === 3 /* STATUS_TRAP */ && trapDeclines) {
    throw new Error('cooperative tier-up run trapped (declined to the interpreter)');
  }
  return status;
}

