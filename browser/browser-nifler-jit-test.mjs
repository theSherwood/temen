// Real-browser (V8) differential for the **nifler card on the wasm-JIT**: parse one Nim program with
// nifler's `_start` emitted (`runJitNifler`, the card's tier) and on the interpreter
// (`temen_run_nifler_fs`, its fallback), and assert the `.p.nif` both write is byte-identical. V8, not
// wasmi, is where nifler's giant `_start` JITs: `tests/nifler_jit.rs` holds the same differential on
// wasmi, ignored because one function exceeds wasmi's register budget.
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
if (!existsSync(`${ROOT}/web/assets/nifler.temen.gz`)) {
  console.log('SKIP: web/assets/nifler.temen.gz absent (rebuild-assets.sh)');
  process.exit(0);
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
  const { runJitNifler } = await import('./wasmjit-module.js');
  const eng = await par.loadEngine();
  const u8 = () => new Uint8Array(eng.memory.buffer);
  const readOut = () => u8().slice(Number(eng.ex.temen_stdout_ptr()), Number(eng.ex.temen_stdout_ptr()) + eng.ex.temen_stdout_len());
  const gunzip = async (u) => new Uint8Array(await new Response(new Blob([u]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());

  const nifler = await gunzip(new Uint8Array(await (await fetch('./assets/nifler.temen.gz')).arrayBuffer()));
  const src = new TextEncoder().encode('import std/[syncio, math]\n\nproc fib(n: int): int =\n  if n < 2: n else: fib(n - 1) + fib(n - 2)\n\necho fib(10)\n');

  // The card's tier: nifler's `_start` emitted; the `.p.nif` comes back on the stdout slot.
  let jit = null, jitErr = null;
  try {
    await runJitNifler(eng.ex, eng.memory, nifler, src, './nifler-jit');
    jit = readOut();
  } catch (e) { jitErr = String(e && e.message || e); }

  // The interpreter oracle: the card's fallback, the same `.p.nif` on the stdout slot.
  const modP = Number(eng.ex.temen_alloc(nifler.length));
  const sP = Number(eng.ex.temen_alloc(src.length));
  { const v = u8(); v.set(nifler, modP); v.set(src, sP); }
  eng.ex.temen_run_nifler_fs(modP, nifler.length, sP, src.length);
  const status = eng.ex.temen_status();
  const oracle = readOut();
  eng.ex.temen_dealloc(modP, nifler.length);
  eng.ex.temen_dealloc(sP, src.length);

  const eq = jit !== null && jit.length === oracle.length && jit.every((x, i) => x === oracle[i]);
  return { jitErr, status, eq, jitLen: jit ? jit.length : 0, oracleLen: oracle.length,
    head: new TextDecoder().decode(oracle).slice(0, 80) };
});

await browser.close(); server.close();
if (errors.length) console.log('ERRORS', errors.slice(0, 5));
const ok = !res.jitErr && res.eq && res.oracleLen > 0 && (res.status === 0 || res.status === 5);
console.log(`  nifler-jit: .p.nif ≡ ${res.eq} (${res.jitLen}/${res.oracleLen} B), interpreter status ${res.status}` +
  `${res.jitErr ? ` · JIT ERROR ${res.jitErr}` : ''} · ${JSON.stringify(res.head)}`);
console.log(ok ? 'PASS — nifler on the wasm-JIT ≡ the interpreter (.p.nif)' : 'FAIL');
process.exit(ok ? 0 : 1);
