// Real-browser differential for the **wasm-JIT reactor** (cap-call outlining): for each interactive
// reactor demo, open the SAME unmodified guest on the interpreter and on the wasm-JIT tier and assert
// the presented framebuffer is BYTE-IDENTICAL every frame — the "verified ⇒ same result on both tiers"
// contract that gates the emitter. A successful JIT open also proves the `tick` actually *emitted*
// (`openJitReactor` throws on a not-emittable fallback), i.e. the outlining did its job. bounce/life/
// mandelzoom auto-run deterministically, so the two tiers must produce the identical frame sequence.
import { startServer } from './serve.mjs';
import { benignAssetMiss } from './play-test-errors.mjs';
import { fileURLToPath } from 'node:url';
import { dirname } from 'node:path';
const ROOT = dirname(fileURLToPath(import.meta.url));
async function loadChromium() {
  for (const s of ['playwright', '/opt/node22/lib/node_modules/playwright/index.js']) {
    try { const m = await import(s); return m.chromium ?? m.default?.chromium; } catch {}
  }
  throw new Error('playwright not found — run `npm i playwright && npx playwright install chromium`');
}
const chromium = await loadChromium();
const { server, port } = await startServer(ROOT);
const browser = await chromium.launch({ args: process.env.CI ? ['--no-sandbox'] : [] });
const page = await browser.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(String(e)));
page.on('console', (m) => { if (m.type() === 'error' && !benignAssetMiss(m)) errors.push(m.text()); });
await page.goto(`http://127.0.0.1:${port}/web/play.html`);

const res = await page.evaluate(async () => {
  const par = await import('./par.js');
  const { openJitReactor } = await import('./wasmjit-reactor.js');
  const eng = await par.loadEngine();
  const NFRAMES = 30;
  // FNV-1a over the presented framebuffer (copied out of the shared memory — a plain view would be a
  // live alias). Tags with dimensions so a size divergence also shows.
  const hashFB = () => {
    const w = eng.ex.temen_framebuffer_width(), h = eng.ex.temen_framebuffer_height();
    const p = Number(eng.ex.temen_framebuffer_ptr());
    const px = new Uint8Array(eng.memory.buffer).slice(p, p + w * h * 4);
    let hsh = 0x811c9dc5;
    for (let i = 0; i < px.length; i++) { hsh ^= px[i]; hsh = Math.imul(hsh, 0x01000193) >>> 0; }
    return `${w}x${h}:${(hsh >>> 0).toString(16)}`;
  };
  // `file` = { name, data } served to the guest through the `fs` capability (the Uxn ROM), or null.
  const runInterp = (bytes, file) => {
    const p = eng.ex.temen_alloc(bytes.length); new Uint8Array(eng.memory.buffer).set(bytes, p);
    let opened;
    if (file) {
      const nameBytes = new TextEncoder().encode(file.name);
      const nameP = eng.ex.temen_alloc(nameBytes.length);
      const dataP = eng.ex.temen_alloc(file.data.length);
      const view = new Uint8Array(eng.memory.buffer);
      view.set(nameBytes, nameP);
      view.set(file.data, dataP);
      opened = eng.ex.temen_onramp_open_fs(p, bytes.length, nameP, nameBytes.length, dataP, file.data.length);
      eng.ex.temen_dealloc(nameP, nameBytes.length);
      eng.ex.temen_dealloc(dataP, file.data.length);
    } else {
      opened = eng.ex.temen_onramp_open(p, bytes.length);
    }
    eng.ex.temen_dealloc(p, bytes.length);
    if (opened !== 0) throw new Error(`interp open failed: ${opened}`);
    const hs = [];
    for (let i = 0; i < NFRAMES; i++) { if (eng.ex.temen_onramp_frame() !== 0) break; hs.push(hashFB()); }
    eng.ex.temen_onramp_close();
    return hs;
  };
  const runJit = async (bytes, file) => {
    // throws if tick isn't emittable
    const r = await openJitReactor(eng.ex, eng.memory, bytes, file && file.name, file && file.data);
    const hs = [];
    for (let i = 0; i < NFRAMES; i++) { if (r.frame() !== 0) break; hs.push(hashFB()); }
    r.close();
    return hs;
  };
  const out = {};
  // uxn: the Uxn VM + Varvara compositor as one tick(), over its demo ROM served as boot.rom (fs).
  const FILES = { uxn: { url: './assets/uxn_demo.rom', name: 'boot.rom' } };
  for (const name of ['bounce', 'life', 'mandelzoom', 'uxn']) {
    const bytes = new Uint8Array(await (await fetch(`./assets/${name}.temen`)).arrayBuffer());
    const file = FILES[name]
      ? { name: FILES[name].name, data: new Uint8Array(await (await fetch(FILES[name].url)).arrayBuffer()) }
      : null;
    let emitted = true, interpH = [], jitH = [];
    try {
      interpH = runInterp(bytes, file);
      jitH = await runJit(bytes, file);
    } catch (e) {
      out[name] = { error: e.message };
      continue;
    }
    const n = Math.min(interpH.length, jitH.length);
    let mismatch = -1;
    for (let i = 0; i < n; i++) if (interpH[i] !== jitH[i]) { mismatch = i; break; }
    out[name] = {
      emitted,
      frames: n,
      identical: mismatch === -1 && interpH.length === jitH.length && n > 0,
      firstMismatch: mismatch,
    };
  }
  return out;
});
console.log('RESULT', JSON.stringify(res));
if (errors.length) console.log('ERRORS', errors.slice(0, 5));
await browser.close(); server.close();

const demos = ['bounce', 'life', 'mandelzoom', 'uxn'];
const ok = errors.length === 0 && demos.every((n) => res[n] && res[n].identical);
for (const n of demos) {
  const r = res[n] || {};
  console.log(`  ${n}: ${r.error ? `ERROR ${r.error}` : `${r.frames} frames, JIT≡interp=${r.identical}`}`);
}
console.log(ok ? 'PASS — wasm-JIT reactor byte-identical to the interpreter' : 'FAIL');
process.exit(ok ? 0 : 1);
