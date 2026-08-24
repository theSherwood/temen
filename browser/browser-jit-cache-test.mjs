// Slice-1 browser cache (WASM_AOT.md): validate + measure the cross-Run compiled-`WebAssembly.Module`
// cache in `wasmjit-module.js`. Correctness: re-Running the SAME module with a stable `cacheKey` must
// (a) produce byte-identical stdout every Run (no stale-code / state leak) and (b) compile exactly once
// then serve hits. Measurement: report the per-Run breakdown (cdylib emit vs `WebAssembly.compile` vs
// run) so we know whether skipping compile actually helps the "light script slower under JIT" footgun.
import { startServer } from './serve.mjs';
import { benignAssetMiss } from './play-test-errors.mjs';
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
const chromium = await loadChromium();
const { server, port } = await startServer(ROOT);
const browser = await chromium.launch({ args: process.env.CI ? ['--no-sandbox'] : [] });
const page = await browser.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(String(e)));
page.on('console', (m) => { if (m.type() === 'error' && !benignAssetMiss(m)) errors.push(m.text()); });
await page.goto(`http://127.0.0.1:${port}/web/play.html`);

// Light + heavy cases: hello_c is the footgun shape (tiny guest, emit/compile overhead dominates);
// qjs_repl is a real compute guest. Only test the ones whose asset is committed/staged.
const CASES = [
  { name: 'hello_c', stdin: '' },
  { name: 'qjs_repl', stdin: "let s=0; for(let i=0;i<200000;i++) s+=i; console.log('sum', s);\n" },
].filter((c) => existsSync(`${ROOT}/web/assets/${c.name}.temen`));

const res = await page.evaluate(async (cases) => {
  const par = await import('./par.js');
  const wj = await import('./wasmjit-module.js');
  const eng = await par.loadEngine();
  const dec = (p, n) => new TextDecoder().decode(new Uint8Array(eng.memory.buffer).slice(p, p + n));
  const readStdout = () => dec(Number(eng.ex.temen_stdout_ptr()), eng.ex.temen_stdout_len());

  const out = {};
  for (const { name, stdin } of cases) {
    const bytes = new Uint8Array(await (await fetch(`./assets/${name}.temen`)).arrayBuffer());
    const stdinBytes = new TextEncoder().encode(stdin);
    wj.jitCacheClear();

    // Run the same module 3× with a stable cacheKey; capture stdout each time + wall-clock.
    const stdouts = [];
    const times = [];
    for (let i = 0; i < 3; i++) {
      const t0 = performance.now();
      await wj.runJitModule(eng.ex, eng.memory, bytes, stdinBytes, name);
      times.push(performance.now() - t0);
      stdouts.push(readStdout());
    }

    // Isolate the raw WebAssembly.compile cost of this module's emitted _start (what the cache skips):
    // open once, copy the emitted bytes, time a bare compile, then close without running.
    let compileMs = null, emittedLen = null;
    {
      const modP = Number(eng.ex.temen_alloc(bytes.length));
      new Uint8Array(eng.memory.buffer).set(bytes, modP);
      const opened = eng.ex.temen_onramp_jit_run_open(modP, bytes.length, 0, 0, 1);
      eng.ex.temen_dealloc(modP, bytes.length);
      if (opened === 0) {
        const wptr = Number(eng.ex.temen_onramp_jit_run_wasm_ptr());
        const wlen = eng.ex.temen_onramp_jit_run_wasm_len();
        const emitted = new Uint8Array(eng.memory.buffer).slice(wptr, wptr + wlen);
        emittedLen = emitted.length;
        const c0 = performance.now();
        await WebAssembly.compile(emitted);
        compileMs = performance.now() - c0;
        eng.ex.temen_onramp_jit_run_close();
      }
    }

    out[name] = {
      stdouts,
      allEqual: stdouts.every((s) => s === stdouts[0]),
      stats: { ...wj.jitCacheStats },
      times,
      compileMs,
      emittedLen,
    };
  }
  return out;
}, CASES);

await browser.close();
await new Promise((r) => server.close(r));

let ok = true;
for (const [name, r] of Object.entries(res)) {
  const statsOk = r.stats.compiles === 1 && r.stats.hits === 2;
  const eqOk = r.allEqual;
  const pass = statsOk && eqOk;
  ok = ok && pass;
  const kb = r.emittedLen != null ? (r.emittedLen / 1024).toFixed(0) : '?';
  const t = r.times.map((x) => x.toFixed(0)).join('/');
  console.log(
    `  ${name}: identical-stdout=${eqOk} cache={compiles:${r.stats.compiles},hits:${r.stats.hits}}` +
    ` · runs ${t}ms · WebAssembly.compile=${r.compileMs?.toFixed(1)}ms for ${kb}KB emitted` +
    (pass ? '' : '  <-- FAIL')
  );
}
if (errors.length) { console.log('page errors:', errors); ok = false; }
console.log(ok ? 'PASS — cross-Run module cache: identical output, compiled once' : 'FAIL');
process.exit(ok ? 0 : 1);
