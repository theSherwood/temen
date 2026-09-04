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

const mems = [];

/** Register a child `WebAssembly.Memory`; returns the id the engine names it by. */
export function registerForeign(memory) {
  mems.push(memory);
  return mems.length - 1;
}

/** The registered Memory for `id` (e.g. to grow it or read a result out). */
export function foreignMemory(id) {
  return mems[id];
}

/** The `temen_host` import entries for an engine instance whose linear memory is `engineMemory`. */
export function foreignImports(engineMemory) {
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
  const u8s = [], i32s = [], i64s = [];
  const child = (id, end) => {
    const v = u8s[id];
    if (v !== undefined && end <= v.byteLength) return v;
    const buf = mems[id].buffer;
    i32s[id] = new Int32Array(buf);
    i64s[id] = new BigInt64Array(buf);
    return (u8s[id] = new Uint8Array(buf));
  };
  const RMW = [null, null, 'add', 'sub', 'and', 'or', 'xor', 'exchange'];
  return {
    foreign_read: (id, off, dst, len) => {
      const e = eng(dst + len), c = child(id, off + len);
      if (len <= 16) for (let i = 0; i < len; i++) e[dst + i] = c[off + i];
      else e.set(c.subarray(off, off + len), dst);
    },
    foreign_write: (id, off, src, len) => {
      const e = eng(src + len), c = child(id, off + len);
      if (len <= 16) for (let i = 0; i < len; i++) c[off + i] = e[src + i];
      else c.set(e.subarray(src, src + len), off);
    },
    foreign_fill: (id, off, len, b) => {
      child(id, off + len).fill(b, off, off + len);
    },
    foreign_copy: (id, dst, src, len) => {
      child(id, Math.max(dst, src) + len).copyWithin(dst, src, src + len);
    },
    foreign_atomic: (id, kind, off, width, ab) => {
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
