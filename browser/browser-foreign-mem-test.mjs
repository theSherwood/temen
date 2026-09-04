// Real-browser (V8) gate for **`Region::Foreign`** (#1284, DETACHED_JIT.md §3.3): the engine cdylib
// addresses a JS-owned `WebAssembly.Memory` — a detached child's future window — through the
// `temen_host.foreign_*` imports. Two checks: (1) the in-tree 20k-op differential (`temen_mem::differential`)
// runs the proxied region against the safe `Paged` reference over a REAL second shared Memory —
// byte-for-byte agreement through the JS atomics/copies; (2) the decline-path cost gate: per-access
// timings of `read_word`/`write_word`/`byte` over Foreign vs the flat `Shared` backing the interpreter
// uses today, reported as a ratio (the number #1284 asks for before slice 2 commits to proxying the
// interpreter's fallback bodies).
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
  console.log('SKIP: threads wasm absent'); process.exit(0);
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
  const { registerForeign } = await import('./foreign-mem.js');
  const eng = await par.loadEngine();
  const ex = eng.ex, memory = eng.memory;
  const readOut = () => new TextDecoder().decode(
    new Uint8Array(memory.buffer).slice(Number(ex.temen_stdout_ptr()), Number(ex.temen_stdout_ptr()) + ex.temen_stdout_len()));

  // (1) differential over a real second shared Memory: 3 pages (192 KiB), 20k ops, page 4096 (the
  // Paged reference's chunk boundary is exercised). 0 = agree; else the first divergence is in stdout.
  const size = 3 * 65536;
  const child = new WebAssembly.Memory({ initial: 3, maximum: 8, shared: true });
  const id = registerForeign(child);
  const diff = ex.temen_foreign_selftest(id, size, 20000);
  const diffMsg = diff === 0 ? '' : readOut();
  // A write from the engine lands in the CHILD memory (a fresh one, untouched by the differential), not
  // the engine's: the isolation half.
  const fresh = new WebAssembly.Memory({ initial: 1, maximum: 1, shared: true });
  const fid = registerForeign(fresh);
  const probe = new Uint32Array(fresh.buffer);
  const before = probe[0x2000 / 4];
  ex.temen_foreign_poke(fid, 0x2000, 0xC0FFEE);
  const landed = probe[0x2000 / 4] === 0xC0FFEE && before === 0;

  // (2) per-access cost: kind 0/1 = read_word+write_word over Shared/Foreign, 2/3 = byte()/set_byte()
  // over Shared/Foreign. Same seeded offset stream and checksum on both backings.
  const bench = (kind, iters) => {
    ex.temen_foreign_bench(id, size, kind, 1000); // warm
    const t0 = performance.now();
    const sum = ex.temen_foreign_bench(id, size, kind, iters);
    return { ns: (performance.now() - t0) * 1e6 / iters, sum: String(sum) };
  };
  const iters = 2_000_000;
  const wS = bench(0, iters), wF = bench(1, iters), bS = bench(2, iters), bF = bench(3, iters);
  return {
    diff, diffMsg, landed,
    word: { sharedNs: wS.ns, foreignNs: wF.ns, ratio: wF.ns / wS.ns, sumEq: wS.sum === wF.sum },
    byte: { sharedNs: bS.ns, foreignNs: bF.ns, ratio: bF.ns / bS.ns, sumEq: bS.sum === bF.sum },
  };
});

await browser.close(); server.close();
console.log('RESULT', JSON.stringify(res, null, 2));
if (errors.length) console.log('ERRORS', errors.slice(0, 6));
const ok = res.diff === 0 && res.landed && res.word.sumEq && res.byte.sumEq;
const f = (x) => x.toFixed(1);
console.log(`  foreign-mem: differential=${res.diff === 0 ? 'agree' : 'DIVERGE ' + res.diffMsg} landedInChild=${res.landed}`);
console.log(`  word access: Shared ${f(res.word.sharedNs)} ns · Foreign ${f(res.word.foreignNs)} ns · ×${f(res.word.ratio)}`);
console.log(`  byte access: Shared ${f(res.byte.sharedNs)} ns · Foreign ${f(res.byte.foreignNs)} ns · ×${f(res.byte.ratio)}`);
console.log(ok ? 'PASS — Region::Foreign over a real second WebAssembly.Memory agrees with the safe reference byte-for-byte' : 'FAIL');
process.exit(ok ? 0 : 1);
