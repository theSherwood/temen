// Real-browser (V8) differential for #1155 (invariant 14, **code-origin axis**): a §22 **guest-JIT
// unit** — code the guest compiled at runtime with `vm_jit_compile` — running on emitted wasm over a
// `vm_map`-GROWN window, byte-identical to the interpreter oracle. The guest (routed to the cooperative
// driver by a `thread.spawn`) grows its window `[64 KiB, 80 KiB)`, `vm_jit_compile`s a unit whose body
// stores into and loads back a **grown** page, `vm_jit_invoke2`s it, and streams the result. Run through
// the shipped `runJitModule` → `driveCoopTierupRun` coop driver (the unit emitted + JIT_INVOKE'd on V8,
// its `"mapped"` bound synced from the grown extent), the stdout must match the interpreter
// (`temen_run_onramp`) byte-for-byte.
//
// This is the first in-browser exercise of the guest-runtime-JIT growth path: the native
// `coop_tierup_driver.rs` proves the mechanism against a *reimplemented* driver; this proves the SHIPPED
// `wasmjit-module.js` driver in real V8. Non-vacuity: `jitCacheStats.compiles` must show the unit
// actually emitted (a silent interpreter fallback would make the differential pass trivially).
import { startServer } from './serve.mjs';
import { fileURLToPath } from 'node:url';
import { dirname } from 'node:path';
const ROOT = dirname(fileURLToPath(import.meta.url));

async function loadChromium() {
  for (const s of ['playwright', '/opt/node22/lib/node_modules/playwright/index.js']) {
    try { const m = await import(s); return m.chromium ?? m.default?.chromium; } catch {}
  }
  throw new Error('playwright not found');
}
const chromium = await loadChromium();
const { server, port } = await startServer(ROOT);
const browser = await chromium.launch({ args: process.env.CI ? ['--no-sandbox'] : [] });
const page = await browser.newPage();
// Only real JS/page errors gate the run — a benign resource 404 (a deploy-built play card asset absent
// in this job, e.g. `tcl_snapshot.temen`) is not this test's concern.
const errors = [];
const benign = (t) => /Failed to load resource|status of 404/i.test(t);
page.on('pageerror', (e) => errors.push(String(e)));
page.on('console', (m) => { if (m.type() === 'error' && !benign(m.text())) errors.push(m.text()); });
await page.goto(`http://127.0.0.1:${port}/web/play.html`);

// The unit the guest compiles at runtime: `f(x) = x + UNIT_K`, storing the sum into `[x+8]` (a grown
// page when `x` lands above the declared window) and loading it back — so its emitted run exercises the
// grown-window `"mapped"` bound. Mirrors `coop_tierup_driver.rs::unit_blob`.
const UNIT_SRC = `memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  vk = i64.const 90909
  vsum = i64.add v0 vk
  vp = i64.const 8
  vaddr = i64.add v0 vp
  i64.store vaddr vsum
  vld = i64.load vaddr
  return vld
  }
}
`;

const res = await page.evaluate(async ({ unitSrc }) => {
  const { loadEngine } = await import('./par.js');
  const { runJitModule, jitCacheStats } = await import('./wasmjit-module.js');
  const eng = await loadEngine();
  const ex = eng.ex, memory = eng.memory;
  const u8 = () => new Uint8Array(memory.buffer);
  const slice = (p, n) => u8().slice(p, p + n);
  const parse = (src) => {
    const b = new TextEncoder().encode(src);
    const p = ex.temen_alloc(b.length);
    u8().set(b, p);
    const okp = ex.temen_parse(p, b.length);
    const outp = slice(ex.temen_parse_ptr(), ex.temen_parse_len());
    ex.temen_dealloc(p, b.length);
    if (okp !== 1) throw new Error('parse: ' + new TextDecoder().decode(outp));
    return outp;
  };
  const readStdout = () => slice(Number(ex.temen_stdout_ptr()), ex.temen_stdout_len());
  const b64 = (a) => btoa(String.fromCharCode(...a));

  const unitBytes = parse(unitSrc);
  // Stage the unit's encoded bytes into the guest window at BLOB_BASE as little-endian i64 stores, so
  // the guest is self-contained and reads the blob for `vm_jit_compile` (like coop_tierup_driver.rs).
  const BLOB_BASE = 36864, SLOT = 34816, PROBE = 65552; // PROBE → the unit stores at 65560, a grown page
  let stores = '';
  for (let i = 0; i * 8 < unitBytes.length; i++) {
    let word = 0n;
    for (let k = 0; k < 8 && i * 8 + k < unitBytes.length; k++) word |= BigInt(unitBytes[i * 8 + k]) << BigInt(8 * k);
    stores += `  va${i} = i64.const ${BLOB_BASE + i * 8}\n  vv${i} = i64.const ${BigInt.asIntN(64, word)}\n  i64.store va${i} vv${i}\n`;
  }
  // func 0 `_start`: `thread.spawn` a trivial worker (routes to the cooperative driver), `vm_map`-grow
  // [64 KiB, 80 KiB), stage + `vm_jit_compile` the unit, `vm_jit_invoke2` it with a grown-page probe,
  // join, stream the sum. All caps arrive as manifest imports (host-agnostic — no baked handles).
  const guestSrc = `memory 16
import 0 "write" (i64, i64) -> (i64)
import 1 "vm_map" (i64, i64, i32) -> (i64)
import 2 "vm_jit_compile" (i64, i64) -> (i64)
import 3 "vm_jit_invoke2" (i64, i64) -> (i64)
func () -> (i64) {
block 0 () {
  vz = i64.const 0
  vt = thread.spawn 1 vz vz
  voff = i64.const 65536
  vlen = i64.const 16384
  vprot = i32.const 3
  vg = call.import 1 (voff, vlen, vprot)
${stores}  vbp = i64.const ${BLOB_BASE}
  vbl = i64.const ${unitBytes.length}
  vcode = call.import 2 (vbp, vbl)
  vprobe = i64.const ${PROBE}
  vres = call.import 3 (vcode, vprobe)
  vj = thread.join vt
  vsum = i64.add vres vj
  vsl = i64.const ${SLOT}
  i64.store vsl vsum
  vlen8 = i64.const 8
  vw = call.import 0 (vsl, vlen8)
  return vsum
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vz2 = i64.const 0
  return vz2
  }
}
export 0 func "_start" 0
`;
  const guest = parse(guestSrc);

  // Interpreter oracle: run the whole guest on the bytecode engine (units interpreted, no emit).
  const gp = ex.temen_alloc(guest.length);
  u8().set(guest, gp);
  ex.temen_run_onramp(gp, guest.length, 0, 0);
  const interpStatus = ex.temen_status();
  const interpOut = b64(readStdout());
  ex.temen_dealloc(gp, guest.length);

  // Emitted tier: `runJitModule` routes the threaded/vm_jit guest to `driveCoopTierupRun`; the unit is
  // emitted + JIT_INVOKE'd on V8 with the grown-window `"mapped"` sync.
  const compilesBefore = jitCacheStats.compiles;
  let jitErr = null, jitStatus = null, jitOut = '';
  try {
    jitStatus = await runJitModule(ex, memory, guest, new Uint8Array(), 'jit-runtime-grow');
    jitOut = b64(readStdout());
  } catch (e) { jitErr = String(e && e.message || e); }
  return { interpStatus, jitStatus, jitErr, interpOut, jitOut, unitCompiles: jitCacheStats.compiles - compilesBefore };
}, { unitSrc: UNIT_SRC });

await browser.close();
await new Promise((r) => server.close(r));

console.log('RESULT', JSON.stringify(res, null, 2));
if (errors.length) console.log('ERRORS', errors.slice(0, 5));

const sumOf = (b) => { const d = Buffer.from(b, 'base64'); return d.length >= 8 ? d.readBigInt64LE(0) : null; };
const WANT = 156461n; // PROBE + UNIT_K = 65552 + 90909 — proves the grown-page store/load worked
const ok = errors.length === 0
  && !res.jitErr
  && res.interpOut === res.jitOut       // byte-identical stdout: emitted §22-unit-grow ≡ interpreter
  && sumOf(res.interpOut) === WANT
  && res.unitCompiles >= 1;             // non-vacuity: the unit actually ran on emitted wasm
console.log(
  `  §22 guest-JIT unit over a vm_map-grown window: interp sum=${sumOf(res.interpOut)} jit sum=${res.jitErr ? 'ERR ' + res.jitErr : sumOf(res.jitOut)}` +
  ` · unit emitted=${res.unitCompiles >= 1 ? 'yes' : 'NO (vacuous!)'}` + (ok ? ' (== expected)' : '  <-- FAIL')
);
console.log(ok ? 'PASS — §22 guest-JIT unit over a vm_map-grown window ≡ interpreter (in V8)' : 'FAIL');
process.exit(ok ? 0 : 1);
