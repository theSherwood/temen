// **Define a powerbox in JavaScript.** A powerbox is the set of capabilities a guest is handed; on
// every other path in this playground they are implemented Rust-side (the fixed §3e stdout/stdin/exit
// prefix, the on-ramp's `display`/`keyboard`/`fs`). Here the page defines them:
//
//   const pb = definePowerbox(eng, {
//     'js.log':  (args, mem) => { console.log(mem.str(args[0], args[1])); return args[1]; },
//     'js.now':  () => BigInt(Date.now()),
//   });
//   const { value, error } = pb.run(moduleBytes);
//
// A guest reaches each one by name — `call.sym "js.log" (i64, i64) -> (i64) v0 (ptr, len)` — and the
// module's import manifest binds name → capability at instantiation (IMPORTS.md §2.1), fail-closed:
// an import no key here defines refuses the run before a single guest op executes.
//
// The JS never sees the guest's memory. `mem` is a bounds-checked view of the calling guest's window
// (`temen_jspb_read`/`temen_jspb_write` in `browser/src/jspb.rs`) that is live **only** for the
// duration of one call — exactly the confinement the built-in capabilities get.
//
// A handler receives `(args, mem)`: `args` are the call's `i64` arguments as BigInt, `mem` is the
// window view (or `null` for a module that declares no memory). Return a number or BigInt — the
// capability's single `i64` result, negative for an errno the guest can branch on. Returning nothing
// means 0; throwing means `-EIO` (-5), never a page crash and never a wasm trap. The view and the
// servicer are the coop driver's (`onCap`, #2579): one seam for every JS capability.

import { jsCapServicer } from './coop-driver.js';

const enc = new TextEncoder();
const dec = new TextDecoder();

/**
 * Bind `caps` (name → handler) as a guest's powerbox on the loaded engine `eng`, and return
 * `{ names, run(moduleBytes) }`. `run` gives back `{ value, status, error }` — `value` is the
 * entry's `i64` result (BigInt), `status` the engine's `temen_status()` code, and `error` a message
 * when the run was refused or trapped (`''` on success).
 */
export function definePowerbox(eng, caps) {
  const { ex, memory } = eng;
  const N = ex.temen_abi_is64() === 1 ? BigInt : Number;
  const names = Object.keys(caps);

  // One JS function per slot, in bind order — the key the engine dispatches on.
  const servicer = jsCapServicer(ex, memory, (slot, op, args, mem) => {
    const fn = caps[names[slot]];
    return fn ? fn(args, mem) : -38n; // -ENOSYS: a slot with no handler
  });

  const readErr = () => {
    const p = ex.temen_jspb_error_ptr(), n = Number(ex.temen_jspb_error_len());
    // `.slice()` first: the threads build's memory is a SharedArrayBuffer, and TextDecoder refuses a
    // view onto shared memory.
    return n ? dec.decode(new Uint8Array(memory.buffer, Number(p), n).slice()) : '';
  };

  return {
    names,
    /** Run an encoded module (a `temen_parse` product or a fetched `.temenc`) under these capabilities. */
    run(moduleBytes) {
      // Re-declare on every run: the engine's registry is process-global, so the run and the names
      // it was bound with can never drift apart.
      ex.temen_jspb_reset();
      for (const name of names) {
        const b = enc.encode(name);
        const p = ex.temen_alloc(N(b.length));
        new Uint8Array(memory.buffer).set(b, Number(p));
        const slot = ex.temen_jspb_bind(p, N(b.length));
        ex.temen_dealloc(p, N(b.length));
        if (slot < 0) throw new Error(`cannot bind capability ${JSON.stringify(name)}`);
      }
      const p = ex.temen_alloc(N(moduleBytes.length));
      new Uint8Array(memory.buffer).set(moduleBytes, Number(p));
      const prev = globalThis.__temen_js_cap_call;
      globalThis.__temen_js_cap_call = servicer;
      try {
        const value = ex.temen_jspb_run(p, N(moduleBytes.length));
        return { value, status: ex.temen_status(), error: readErr() };
      } finally {
        globalThis.__temen_js_cap_call = prev;
        ex.temen_dealloc(p, N(moduleBytes.length));
      }
    },
  };
}
