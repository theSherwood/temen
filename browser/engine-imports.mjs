import { adoptMemory } from './web/foreign-mem.js';

// The import object every instantiation of the **engine** wasm (`temen_browser.wasm`) must supply.
// Besides the optional shared `memory` (the threads build imports it; the plain build owns its own),
// the wasm32 build imports `temen_host.webgpu_op` — the `webgpu` capability's host seam (a guest ships a
// WGSL shader / asks the host to present a frame). Only the playground's main thread (`web/par.js`)
// services it for real against `navigator.gpu`; every other instantiation — the corpus/bench
// differentials, the parallel-Worker vCPUs — has no GPU surface, so it passes this no-op stub and a
// guest that resolves the `webgpu` cap there simply gets -1 back and skips. Returns a BigInt (i64).
//
// (Emitted wasm-JIT *units* are a different module with their own imports — `env.{memory,trap,
// call_interp}`, no `temen_host` — so they do NOT use this.)
export function engineImports(memory) {
  // #1284 `Region::Foreign`: the threads build imports `temen_host.foreign_*` (detached children's own
  // memories, `web/foreign-mem.js`). No Node harness registers a foreign memory, so these stubs only have
  // to satisfy the import — a call would mean a bug, hence they throw. Kept in step with foreign-mem.js.
  const noForeign = () => { throw new Error('foreign memory access in a harness without foreign-mem.js'); };
  const foreign = Object.fromEntries(
    ['foreign_read', 'foreign_write', 'foreign_load', 'foreign_store', 'foreign_fill', 'foreign_copy', 'foreign_atomic', 'foreign_grow',
      'foreign_mint'].map((n) => [n, noForeign]));
  // `js_cap_call` is the JS-defined powerbox's seam (`web/powerbox.js`): one guest capability call,
  // dispatched to the function the page bound to that slot. Delegated to the same `globalThis` hook
  // the playground uses (`web/par.js`), so a harness that imports `powerbox.js` drives the real seam;
  // with no servicer installed a call is -ENOSYS, exactly as the engine documents.
  // `stream_chunk` is the live output tee (#1414 S5a), delegated to the same `globalThis` sink the
  // playground's import object calls (`web/par.js`): the coop driver installs one for an `onOutput`, so
  // a harness that drives it streams through the real seam; with none installed a write lands only in
  // the run's buffers. It copies the chunk out of the imported `memory`, so a sink needs the threads
  // build: the plain build's memory is its own, which this object never sees.
  // `x2_spawn`/`x2_now_ms` are the parallel driver's platform (#1414 B6), which starts each thread
  // of a run as a Worker. A harness starts none: a refused `x2_spawn` ends such a run with
  // `ThreadFault`.
  const imports = {
    temen_host: {
      ...foreign,
      webgpu_op: () => -1n,
      stream_chunk: (fd, seq, ptr, len) => {
        const h = globalThis.__temen_stream_chunk;
        if (h) h(fd, seq >>> 0, new Uint8Array(memory.buffer, ptr >>> 0, len >>> 0).slice());
      },
      x2_spawn: () => -1,
      x2_now_ms: () => performance.timeOrigin + performance.now(),
      x2_tierup: () => -1,
      x2_invoke: () => -1,
      js_cap_call: (slot, op, argsPtr, nArgs, mem) => {
        const h = globalThis.__temen_js_cap_call;
        return h ? BigInt(h(Number(slot), Number(op), argsPtr, nArgs, mem)) : -38n;
      },
    },
  };
  // A Worker handed the memory adopts it before instantiating over it (`adoptMemory`, #1996).
  if (memory) imports.env = { memory: adoptMemory(memory) };
  return imports;
}
