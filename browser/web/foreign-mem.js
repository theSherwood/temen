// Foreign memories (#1284, DETACHED_JIT.md §3.3): the engine cdylib addresses a detached child's own
// `WebAssembly.Memory` through the `temen_host.foreign_*` imports below — `Region::Foreign` in
// `temen-mem` calls one import per access, and the bytes never enter the engine's linear memory except
// through the copy the call performs. This module is the registry (id → Memory) plus the import table.
// Every instantiation of the THREADS engine build supplies `...foreignImports(memory)` in `temen_host`
// (par.js, the Workers, engine-imports.mjs); the plain build imports none of these.
//
// ABI (all u32, no BigInt on the hot path): offsets are region-relative and already bounded by the
// engine; `ab` points at 16 bytes of ENGINE memory holding the atomic's operands `a` (0..8) and `b`
// (8..16) little-endian, and receives the old value at 0..8. `kind`: 0 load · 1 store(a) ·
// 2 add · 3 sub · 4 and · 5 or · 6 xor · 7 xchg (operand a) · 8 cmpxchg(expected a, replacement b).
// Widths 4/8; `off` naturally aligned. A registry is per agent (page or Worker): a Memory registered
// here is addressable by the engine instance(s) of THIS agent.

/**
 * Adopt a shared memory this agent was handed — a Worker's copy of the engine memory, or of a detached
 * child's — by making its length current. Call it once, before instantiating over the memory or viewing
 * its `buffer`. Returns the memory. It lives here, not in `engine-mem.js`, because every Worker loads
 * this module anyway: one more import in a Worker's module graph costs each Worker birth a fetch.
 *
 * V8 tells every agent sharing a memory when it grows, but an agent still deserializing the memory when
 * another thread grows it can miss the notice (#1996). Its `buffer`, and every instance it then builds
 * over the memory, keep the old length until the next grow. Loads and stores into the missed pages
 * still work, but atomics, `memory.fill`/`copy` and `memory.size` check the stale length and trap, and a
 * view over the stale `buffer` ends early. `grow(0)` makes V8 re-read the length for this agent and its
 * instances on the spot.
 */
export function adoptMemory(memory) {
  memory.grow(0);
  return memory;
}

const mems = []; // id -> { m: WebAssembly.Memory, base: byte offset of region offset 0 within it }
// Views of each registered memory, by id, refreshed when stale (`foreignImports`).
const u8s = [], i32s = [], i64s = [];

/**
 * Register a child `WebAssembly.Memory`; returns the id the engine names it by. `base` is where the
 * engine's region offset 0 lands in the memory — a detached child's window starts one host header page
 * in (`temen_detached_header_bytes()`), so its region is `[base, …)` and the header below stays the
 * host's (DETACHED_JIT.md §3.1). `id` is the next free one, or the id another agent gave the memory,
 * for a Worker the memory was handed to with it (#1414 B6-3c): such a Worker registers it first, so
 * the ids it mints after cannot meet it.
 */
export function registerForeign(memory, base = 0, id = mems.length) {
  if (mems[id] !== undefined) throw new Error(`foreign memory ${id} is already registered`);
  mems[id] = { m: adoptMemory(memory), base }; // a detached child's memory, shared by its threads' Workers
  return id;
}

/** Let go of the memory registered as `id`, and of this agent's views of it. Its id is not reused. */
export function releaseForeign(id) {
  mems[id] = u8s[id] = i32s[id] = i64s[id] = undefined;
}

/** The registered Memory for `id` (e.g. to read a result out). */
export function foreignMemory(id) {
  return mems[id].m;
}

/** The `temen_host` import entries for an engine instance whose linear memory is `engineMemory`. */
export function foreignImports(engineMemory) {
  // Every threads-engine instantiation comes through here, so this is where an agent handed the engine
  // memory adopts it: before the views below and the instance built over these imports (#1996).
  adoptMemory(engineMemory);
  // Views are cached and NEVER refreshed through `.buffer` on the hot path: measured in Chromium, the
  // `WebAssembly.Memory.buffer` getter costs ~90 ns, more than the whole wasm↔JS call. A view over
  // SHARED memory is never detached by a grow — it just stays short — so "does this access fit in the
  // cached view" is the staleness test, and only a miss (after a grow) pays `.buffer` + a new view.
  // With this, one import call is ~30 ns (vs ~110 ns allocating views per call, ~180 ns re-reading
  // `.buffer`), i.e. ×4 (byte) to ×7 (8-byte word) over a direct linear-memory access (#1284's gate).
  let eu8 = new Uint8Array(engineMemory.buffer);
  let edv = new DataView(engineMemory.buffer);
  const eng = (end) => {
    if (end > eu8.byteLength) {
      eu8 = new Uint8Array(engineMemory.buffer);
      edv = new DataView(engineMemory.buffer);
    }
    return eu8;
  };
  // `end` is the memory-absolute end of the access (region offset + base + length).
  const child = (id, end) => {
    const v = u8s[id];
    if (v !== undefined && end <= v.byteLength) return v;
    const buf = mems[id].m.buffer;
    i32s[id] = new Int32Array(buf);
    i64s[id] = new BigInt64Array(buf);
    return (u8s[id] = new Uint8Array(buf));
  };
  const RMW = [null, null, 'add', 'sub', 'and', 'or', 'xor', 'exchange'];
  return {
    foreign_read: (id, off, dst, len) => {
      off += mems[id].base;
      const e = eng(dst + len), c = child(id, off + len);
      if (len <= 16) for (let i = 0; i < len; i++) e[dst + i] = c[off + i];
      else e.set(c.subarray(off, off + len), dst);
    },
    foreign_write: (id, off, src, len) => {
      off += mems[id].base;
      const e = eng(src + len), c = child(id, off + len);
      if (len <= 16) for (let i = 0; i < len; i++) c[off + i] = e[src + i];
      else c.set(e.subarray(src, src + len), off);
    },
    foreign_fill: (id, off, len, b) => {
      off += mems[id].base;
      child(id, off + len).fill(b, off, off + len);
    },
    foreign_copy: (id, dst, src, len) => {
      const base = mems[id].base;
      dst += base; src += base;
      child(id, Math.max(dst, src) + len).copyWithin(dst, src, src + len);
    },
    // Mint a fresh shared child memory of `initial`..`maximum` bytes (header included) and register it
    // with `base` as the region origin (#1286). Returns the id, or -1 if the constructor refused.
    foreign_mint: (base, initial, maximum) => {
      try {
        const m = new WebAssembly.Memory({
          initial: Math.ceil(initial / 65536), maximum: Math.ceil(maximum / 65536), shared: true,
        });
        return registerForeign(m, base);
      } catch { return -1; }
    },
    // Make `len` region bytes addressable: grow the memory by whole pages. 1 = ok, 0 = refused (at the
    // memory's `maximum`). The cached views go stale-short and refresh on their next miss.
    foreign_grow: (id, len) => {
      const { m, base } = mems[id];
      const need = base + len, have = m.buffer.byteLength;
      if (need <= have) return 1;
      const pages = Math.ceil((need - have) / 65536);
      try { m.grow(pages); return 1; } catch { return 0; }
    },
    foreign_atomic: (id, kind, off, width, ab) => {
      off += mems[id].base;
      eng(ab + 16);
      child(id, off + width);
      let old;
      if (width === 8) {
        const v = i64s[id], i = off / 8;
        const a = edv.getBigInt64(ab, true), b = edv.getBigInt64(ab + 8, true);
        if (kind === 0) old = Atomics.load(v, i);
        else if (kind === 1) { old = 0n; Atomics.store(v, i, a); }
        else if (kind === 8) old = Atomics.compareExchange(v, i, a, b);
        else old = Atomics[RMW[kind]](v, i, a);
        edv.setBigUint64(ab, BigInt.asUintN(64, old), true);
      } else {
        const v = i32s[id], i = off / 4;
        const a = edv.getInt32(ab, true), b = edv.getInt32(ab + 8, true);
        if (kind === 0) old = Atomics.load(v, i);
        else if (kind === 1) { old = 0; Atomics.store(v, i, a); }
        else if (kind === 8) old = Atomics.compareExchange(v, i, a, b);
        else old = Atomics[RMW[kind]](v, i, a);
        edv.setUint32(ab, old >>> 0, true);
        edv.setUint32(ab + 4, 0, true);
      }
    },
  };
}
