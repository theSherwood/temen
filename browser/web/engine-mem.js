// The engine's shared linear memory, and the one place its ceiling is decided.
//
// The threads build links with `--import-memory`, so the **host** constructs the engine's
// `WebAssembly.Memory` and picks its `maximum`. That maximum is the real ceiling: dlmalloc grows into
// it, every in-engine guest window lives in it, and a card that outgrows it dies in `Paged::zero` with
// no room (#1543). It is therefore policy, not a build constant — a phone and a workstation want
// different answers, and a test that needs headroom should be able to ask for it.
//
// One thing stays a build constant: `--max-memory` bakes a `maximum` into the module's memory **import
// declaration**, and instantiation rejects any supplied memory whose maximum exceeds it
// (`LinkError: memory import has a larger maximum size`). It is a ceiling on the ceiling. The build
// therefore declares the wasm32 architectural limit (65536 pages = 4 GiB, the largest a 32-bit memory
// can be) so it never binds again, and everything below is the host's call.
//
// `declaredMaxPages` reads that declaration straight out of the module bytes so a host can ask for
// what it wants and still run against a build that declares less — an older build, or CI before a
// raised `--max-memory` lands. Asking for more than the build allows then clamps instead of throwing.

/** Pages (64 KiB each) the host asks for by default: 2 GiB. */
export const ENGINE_MAX_PAGES = 32768;

/** Pages the engine's memory starts at (dlmalloc grows from here): 128 MiB. */
export const ENGINE_INITIAL_PAGES = 2048;

const PAGE_BITS = 16;

/** Decode one LEB128 at `b[i]`, returning `[value, next]`. */
function leb(b, i) {
  let v = 0, shift = 0;
  for (;;) {
    const byte = b[i++];
    v += (byte & 0x7f) * 2 ** shift;
    if ((byte & 0x80) === 0) return [v, i];
    shift += 7;
  }
}

/**
 * The `maximum` (in pages) the module's imported memory declares, or `undefined` if it imports no
 * memory or declares no maximum. Walks the section table to the import section (id 2) and reads the
 * memory import's limits — a few hundred bytes of parsing, no compile needed.
 */
export function declaredMaxPages(wasmBytes) {
  const b = wasmBytes;
  if (b.length < 8 || b[0] !== 0x00 || b[1] !== 0x61 || b[2] !== 0x73 || b[3] !== 0x6d) return undefined;
  let i = 8;
  while (i < b.length) {
    const id = b[i++];
    let size;
    [size, i] = leb(b, i);
    const end = i + size;
    if (id !== 2) { i = end; continue; } // not the import section
    let count;
    [count, i] = leb(b, i);
    for (let n = 0; n < count; n++) {
      let len;
      [len, i] = leb(b, i); i += len;            // module name
      [len, i] = leb(b, i); i += len;            // field name
      const kind = b[i++];
      if (kind === 0x00) { [, i] = leb(b, i); }  // func: type index
      else if (kind === 0x03) { i += 2; }        // global: valtype + mutability
      else {
        if (kind === 0x01) i++;                  // table: element type, then limits
        const flags = b[i++];                    // bit 0 = has max, bit 1 = shared
        [, i] = leb(b, i);                       // minimum
        if ((flags & 0x01) === 0) {
          if (kind === 0x02) return undefined;   // a memory import with no declared maximum
          continue;                              // a table with no maximum — keep scanning
        }
        let max;
        [max, i] = leb(b, i);
        if (kind === 0x02) return max;
      }
    }
    return undefined;
  }
  return undefined;
}

/**
 * The engine's shared memory. `wasmBytes` is the build it will be instantiated against: the requested
 * maximum is clamped to what that build declares, so a host asking for more than the build allows gets
 * the build's ceiling instead of a `LinkError`. Pass `maxPages` to override the default policy.
 */
export function engineMemory(wasmBytes, { initial = ENGINE_INITIAL_PAGES, maxPages = ENGINE_MAX_PAGES } = {}) {
  const declared = wasmBytes ? declaredMaxPages(wasmBytes) : undefined;
  const maximum = declared === undefined ? maxPages : Math.min(maxPages, declared);
  return { memory: new WebAssembly.Memory({ initial, maximum, shared: true }), maxPages: maximum };
}

/** Bytes a page count addresses — for a host reporting or budgeting against the ceiling. */
export const pagesToBytes = (pages) => pages * 2 ** PAGE_BITS;
