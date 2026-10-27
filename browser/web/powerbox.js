// **Define a powerbox in JavaScript.** A powerbox is the set of capabilities a guest is handed. Every
// on-ramp run is handed the playground's Rust-side ones (the §3e stdout/stdin/exit prefix, the
// on-ramp's `display`/`keyboard`/`fs`); here the page adds its own, one JS function each:
//
//   const pb = definePowerbox(eng, {
//     'js.log':  (args, mem) => { console.log(mem.str(args[0], args[1])); return args[1]; },
//     'js.now':  () => BigInt(Date.now()),
//   });
//   const { value, error } = pb.run(module);
//
// A guest reaches each one by name — `call.sym "js.log" (i64, i64) -> (i64) v0 (ptr, len)` — and the
// module's import manifest binds name → capability at instantiation (IMPORTS.md §2.1), fail-closed:
// an import nothing grants refuses the run before a single guest op executes.
//
// `run` stages the functions as the next open's JS capabilities (`temen_onramp_stage_js_cap`, #2579)
// and runs the module as any on-ramp run is run (`temen_run_onramp`), so they join that run's
// powerbox; a card that runs nested is granted them by name, over its own window.
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
// `temen_status()` codes (`browser/src/lib.rs`): a clean return, and a trap.
const STATUS_OK = 0;
const STATUS_TRAP = 3;

/**
 * Add `caps` (name → handler) to a guest's powerbox on the loaded engine `eng`, and return
 * `{ names, run(moduleBytes) }`. `run` gives back `{ value, status, error }` — `value` is the
 * entry's `i64` result (BigInt), `status` the engine's `temen_status()` code, and `error` a message
 * when the run was refused or did not return (`''` on success).
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

  // What a stash export holds, decoded. `.slice()` first: the threads build's memory is a
  // SharedArrayBuffer, and TextDecoder refuses a view onto shared memory.
  const stashed = (len, ptr) => {
    const n = Number(ex[len]()); // the length first: it stashes what the pointer then names
    return n ? dec.decode(new Uint8Array(memory.buffer, Number(ex[ptr]()), n).slice()) : '';
  };
  // Why a run did not end well: the notes of a refusal (an import nothing grants names itself
  // there), else its trap.
  const runError = (status) =>
    stashed('temen_notes_len', 'temen_notes_ptr') ||
    (status === STATUS_TRAP ? `trap: ${stashed('temen_trap_len', 'temen_trap_ptr')}` : `run failed: status ${status}`);

  return {
    names,
    /** Run an encoded module (a `temen_parse` product or a fetched `.temenc`) under these capabilities. */
    run(moduleBytes) {
      // Staged for this run alone: the run and the names its slots dispatch on can never drift apart.
      ex.temen_onramp_stage_clear();
      for (const name of names) {
        const b = enc.encode(name);
        const p = ex.temen_alloc(N(b.length));
        new Uint8Array(memory.buffer).set(b, Number(p));
        const slot = ex.temen_onramp_stage_js_cap(p, N(b.length));
        ex.temen_dealloc(p, N(b.length));
        if (slot < 0) throw new Error(`cannot stage capability ${JSON.stringify(name)}`);
      }
      const p = ex.temen_alloc(N(moduleBytes.length));
      new Uint8Array(memory.buffer).set(moduleBytes, Number(p));
      const prev = globalThis.__temen_js_cap_call;
      globalThis.__temen_js_cap_call = servicer;
      try {
        const value = ex.temen_run_onramp(p, N(moduleBytes.length), N(0), N(0));
        const status = ex.temen_status();
        return { value, status, error: status === STATUS_OK ? '' : runError(status) };
      } finally {
        globalThis.__temen_js_cap_call = prev;
        ex.temen_dealloc(p, N(moduleBytes.length));
      }
    },
  };
}
