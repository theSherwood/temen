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
  // `stdout_chunk` is the live-stdout tee seam (`temen_run_onramp_stream`): the page appends each chunk
  // as the guest writes it. Headless probes don't stream, so a no-op stub satisfies the import.
  // #1284 `Region::Foreign`: the threads build imports `temen_host.foreign_*` (detached children's own
  // memories, `web/foreign-mem.js`). No Node harness registers a foreign memory, so these stubs only have
  // to satisfy the import — a call would mean a bug, hence they throw. Kept in step with foreign-mem.js.
  const noForeign = () => { throw new Error('foreign memory access in a harness without foreign-mem.js'); };
  const foreign = Object.fromEntries(
    ['foreign_read', 'foreign_write', 'foreign_fill', 'foreign_copy', 'foreign_atomic', 'foreign_grow', 'foreign_mint'].map((n) => [n, noForeign]));
  // `js_cap_call` is the JS-defined powerbox's seam (`web/powerbox.js`): one guest capability call,
  // dispatched to the function the page bound to that slot. Delegated to the same `globalThis` hook
  // the playground uses (`web/par.js`), so a harness that imports `powerbox.js` drives the real seam;
  // with no servicer installed a call is -ENOSYS, exactly as the engine documents.
  const imports = {
    temen_host: {
      ...foreign,
      webgpu_op: () => -1n,
      stdout_chunk: () => {},
      js_cap_call: (slot, op, argsPtr, nArgs, mem) => {
        const h = globalThis.__temen_js_cap_call;
        return h ? BigInt(h(Number(slot), Number(op), argsPtr, nArgs, mem)) : -38n;
      },
    },
  };
  if (memory) imports.env = { memory };
  return imports;
}
