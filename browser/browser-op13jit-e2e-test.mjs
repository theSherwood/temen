// Real-browser (V8) end-to-end for the **JS-orchestrated §14 op-13 loop** (#1025 Path 1): a resumable
// driver marshals an `fs` grant to a confined child, and JS runs the child's `_start` on the **emitted
// wasm** tier (`driveJitRun`) over its carve. The child's `call.cap` leaf resolves the *marshaled* `fs`
// on the reactor cross-tier bounce and returns `40 + fs()` = 41; the shared counter ticks once. This is
// the browser realization of `nimc.rs::drive_op13` with the child tiered up — the nested phase on JIT.
import { startServer } from './serve.mjs';
import { fileURLToPath } from 'node:url';
import { dirname } from 'node:path';
import { existsSync } from 'node:fs';
const ROOT = dirname(fileURLToPath(import.meta.url));
async function loadChromium() {
  for (const s of ['playwright', '/opt/node22/lib/node_modules/playwright/index.js']) {
    try { const m = await import(s); return m.chromium ?? m.default?.chromium; } catch {}
  }
  throw new Error('playwright not found');
}
if (!existsSync(`${ROOT}/target/wasm32-unknown-unknown/release/temen_browser.wasm`)) {
  console.log('SKIP: threads wasm not built'); process.exit(0);
}
const chromium = await loadChromium();
const { server, port } = await startServer(ROOT);
const browser = await chromium.launch({ args: process.env.CI ? ['--no-sandbox'] : [] });
const page = await browser.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(String(e)));
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
await page.goto(`http://127.0.0.1:${port}/web/play.html`);

const res = await page.evaluate(async () => {
  const par = await import('./par.js');
  const { driveJitRun } = await import('./wasmjit-module.js');
  const eng = await par.loadEngine();
  const ex = eng.ex, memory = eng.memory;

  if (ex.temen_op13jit_open() !== 0) return { err: `open failed: status ${ex.temen_status?.() ?? '?'}` };
  let steps = 0, drove = 0;
  for (;;) {
    if (steps++ > 8) { ex.temen_op13jit_close(); return { err: 'loop did not terminate' }; }
    const s = ex.temen_op13jit_step();
    if (s === 0) break;             // OP13JIT_DONE
    if (s === 1) {                  // OP13JIT_CHILD — run the staged child on emitted wasm
      try {
        await driveJitRun(ex, memory, 'op13jit-child');
      } catch (e) {
        ex.temen_op13jit_close();
        return { err: `driveJitRun threw: ${String(e && e.message || e)}` };
      }
      ex.temen_op13jit_deliver();
      drove++;
      continue;
    }
    ex.temen_op13jit_close();
    return { err: `trap at step (code ${s})` };
  }
  const result = Number(ex.temen_op13jit_result());
  const counter = Number(ex.temen_op13jit_counter());
  ex.temen_op13jit_close();
  return { result, counter, drove };
});

await browser.close(); server.close();
console.log('RESULT', JSON.stringify(res, null, 2));
if (errors.length) console.log('ERRORS', errors.slice(0, 6));
const ok = !res.err && res.result === 41 && res.counter === 1 && res.drove === 1;
console.log(`  op13jit-e2e: result=${res.result} counter=${res.counter} childrenDriven=${res.drove}${res.err ? ` · ERR ${res.err}` : ''}`);
console.log(ok ? 'PASS — nested child ran on the EMITTED tier over its marshaled fs (driver joined 41)' : 'FAIL');
process.exit(ok ? 0 : 1);
