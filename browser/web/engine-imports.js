// The import object every instantiation of the **engine** wasm (`temen_browser.wasm`) supplies: the
// page's (`par.js`), each Worker's, and the Node harnesses' (`../engine-imports.mjs`). One definition
// (#2579, INVARIANTS #15): each `temen_host` seam reaches a `globalThis` hook its host may install,
// and does nothing without one. A host whose seam must do more passes its own in `hooks`: the
// executor-2 Worker's thread platform and output relay (`worker.js`).
//
// `memory` is the shared memory the threads build imports; the plain build owns its own, so a host
// of it passes none. (Emitted wasm-JIT *units* are a different module with their own imports —
// `env.{memory,trap,call_interp}`, no `temen_host` — so they do not use this.)
import { foreignImports } from './foreign-mem.js';

export function engineImports(memory, hooks = {}) {
  const temen_host = {
    // #1284 `Region::Foreign`: detached children's own memories; only the threads build imports them.
    ...(memory ? foreignImports(memory) : {}),
    // The `webgpu` capability's seam: a page installs a servicer backed by its <canvas> and
    // `navigator.gpu` (play.js). i64 args arrive as BigInt; it returns a BigInt.
    webgpu_op: (op, a, b, c, ptr, len) => {
      const h = globalThis.__temen_webgpu_op;
      return h ? BigInt(h(op, a, b, c, ptr, len, memory)) : -1n;
    },
    // One call to a JS capability (#1419, #2579): `slot` names it, `mem` is the opaque handle to the
    // calling guest's window. The coop driver installs a servicer for a run's `onCap`, and
    // `definePowerbox` for its run (`web/coop-driver.js`'s `jsCapServicer`). -ENOSYS with none.
    js_cap_call: (slot, op, argsPtr, nArgs, mem) => {
      const h = globalThis.__temen_js_cap_call;
      return h ? BigInt(h(Number(slot), Number(op), argsPtr, nArgs, mem)) : -38n;
    },
    // The live output tee (#1414 S5a): each write to a run's stdout (`fd` 1) or stderr (2), as
    // `[ptr, ptr+len)` in linear memory, numbered by `seq`. The coop driver installs a sink for an
    // `onOutput`; it gets a fresh copy of the chunk (the memory may move after).
    stream_chunk: (fd, seq, ptr, len) => {
      const h = globalThis.__temen_stream_chunk;
      if (h) h(fd, seq >>> 0, new Uint8Array(memory.buffer, ptr >>> 0, len >>> 0).slice());
    },
    // The parallel driver's platform (#1414 B6): only an executor-2 Worker starts a run's threads.
    // Anywhere else a refused `x2_spawn` ends such a run with `ThreadFault`.
    x2_spawn: () => -1,
    x2_now_ms: () => performance.timeOrigin + performance.now(),
    x2_tierup: () => -1,
    x2_invoke: () => -1,
    ...hooks,
  };
  return memory ? { env: { memory }, temen_host } : { temen_host };
}
