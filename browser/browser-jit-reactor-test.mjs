// Real-browser differential for the reactor's **wasm-JIT tier** (#1414 S4-4; cap-call outlining): for
// each interactive reactor demo, run the SAME unmodified guest with its frames interpreted and with its
// `tick` offered to the emitted tier, and assert the presented framebuffer is BYTE-IDENTICAL every
// frame — the "verified ⇒ same result on both tiers" contract that gates the emitter. Every frame on the
// emitted tier must report that `tick` ran there (a frame the engine could not emit runs interpreted),
// i.e. the outlining did its job. bounce/life/mandelzoom auto-run deterministically, so the two tiers
// must produce the identical frame sequence.
import { startServer } from './serve.mjs';
import { benignAssetMiss } from './play-test-errors.mjs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { readFileSync } from 'node:fs';
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

// The Uxntal card's path needs the demo's SOURCE, which the dev server does not serve (it lives outside
// browser/): hand it to the page as an argument.
const talSource = readFileSync(join(ROOT, '..', 'crates', 'temen-run', 'demos', 'uxn', 'demo.tal'), 'utf8');
const res = await page.evaluate(async ({ talSource }) => {
  const par = await import('./par.js');
  const { openReactor } = await import('./reactor.js');
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
  // Open `bytes`, serving `file` = { name, data } through the `fs` capability (the Uxn ROM) or nothing.
  const open = (bytes, file) => {
    const r = openReactor(eng.ex, eng.memory, bytes, file);
    if (!r) throw new Error(`open failed: status ${eng.ex.temen_status()}`);
    return r;
  };
  // One frame on the emitted tier (`jit`) or interpreted. An emitted frame that kept going must have run
  // `tick` there: comparing the interpreter with itself would prove nothing.
  const frameOn = async (r, jit) => {
    const counts = {};
    const status = await r.frame(jit, counts);
    if (jit && status === 0 && counts.leaves === 0) throw new Error('tick did not run on the emitted tier');
    return status;
  };
  const runTier = async (bytes, file, jit) => {
    const r = open(bytes, file);
    const hs = [];
    try {
      for (let i = 0; i < NFRAMES; i++) { if (await frameOn(r, jit) !== 0) break; hs.push(hashFB()); }
    } finally {
      r.close();
    }
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
      interpH = await runTier(bytes, file, false);
      jitH = await runTier(bytes, file, true);
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
  // The `mouse` capability on both tiers: a click over the Uxn guest (kind 0, payload
  // (buttons << 24) | (x << 12) | y) must change the next frame on the interpreter AND on the wasm-JIT
  // tier, and the two post-click frames must still agree — the cap-call outlining carries the new
  // waist exactly like `keyboard`.
  {
    const bytes = new Uint8Array(await (await fetch('./assets/uxn.temen')).arrayBuffer());
    const file = { name: 'boot.rom', data: new Uint8Array(await (await fetch(FILES.uxn.url)).arrayBuffer()) };
    const click = [(1 << 24) | (200 << 12) | 100, (200 << 12) | 100];
    const withClick = async (jit) => {
      const r = open(bytes, file);
      await frameOn(r, jit); await frameOn(r, jit);
      const before = hashFB();
      eng.ex.temen_onramp_mouse(0, click[0]); eng.ex.temen_onramp_mouse(0, click[1]);
      await frameOn(r, jit);
      const after = hashFB();
      r.close();
      return { before, after };
    };
    try {
      const i = await withClick(false), j = await withClick(true);
      out.uxnMouse = { changed: i.before !== i.after && j.before !== j.after, tiersAgree: i.after === j.after, i, j };
    } catch (e) {
      out.uxnMouse = { error: e.message };
    }
  }
  // The Uxntal card's path on both tiers: the demo's SOURCE served as boot.tal is assembled in the guest
  // and must render the committed ROM's exact frames; an unassemblable source must surface its error on
  // stdout (`temen_stdout_*` after the frame) and exit, on both tiers.
  {
    const bytes = new Uint8Array(await (await fetch('./assets/uxn.temen')).arrayBuffer());
    const tal = new TextEncoder().encode(talSource);
    const bad = new TextEncoder().encode('|0100 #01 #02 ADD\n,&nope JMP BRK\n');
    const stdoutNow = () => {
      const n = eng.ex.temen_stdout_len(), p = eng.ex.temen_stdout_ptr();
      return new TextDecoder().decode(new Uint8Array(eng.memory.buffer).slice(p, p + n));
    };
    const run = async (tier, file, n) => {
      const r = open(bytes, file);
      const hs = [], outs = [];
      let status = 0;
      for (let i = 0; i < n; i++) {
        status = await frameOn(r, tier === 'jit');
        outs.push(stdoutNow());
        if (status !== 0) break;
        hs.push(hashFB());
      }
      r.close();
      return { hs, status, stdout: outs.join('') };
    };
    try {
      const romFile = { name: 'boot.rom', data: new Uint8Array(await (await fetch(FILES.uxn.url)).arrayBuffer()) };
      const fromRom = await run('interp', romFile, 10);
      const talI = await run('interp', { name: 'boot.tal', data: tal }, 10);
      const talJ = await run('jit', { name: 'boot.tal', data: tal }, 10);
      const badI = await run('interp', { name: 'boot.tal', data: bad }, 3);
      const badJ = await run('jit', { name: 'boot.tal', data: bad }, 3);
      const same = (a, b) => a.length === b.length && a.every((h, i) => h === b[i]);
      const errText = 'uxnasm: line 2: unknown reference: on-reset/nope';
      out.uxnTal = {
        assembled: same(fromRom.hs, talI.hs) && same(fromRom.hs, talJ.hs) && fromRom.hs.length === 10,
        errorReported: badI.status === 5 && badJ.status === 5 && badI.stdout.includes(errText) && badJ.stdout.includes(errText),
        badI: { status: badI.status, stdout: badI.stdout }, badJ: { status: badJ.status, stdout: badJ.stdout },
      };
    } catch (e) {
      out.uxnTal = { error: e.message };
    }
  }
  return out;
}, { talSource });
console.log('RESULT', JSON.stringify(res));
if (errors.length) console.log('ERRORS', errors.slice(0, 5));
await browser.close(); server.close();

const demos = ['bounce', 'life', 'mandelzoom', 'uxn'];
const mouseOk = !!(res.uxnMouse && res.uxnMouse.changed && res.uxnMouse.tiersAgree);
console.log(`  uxn mouse: ${res.uxnMouse && res.uxnMouse.error ? `ERROR ${res.uxnMouse.error}` : `click changes the frame on both tiers=${mouseOk}`}`);
const talOk = !!(res.uxnTal && res.uxnTal.assembled && res.uxnTal.errorReported);
console.log(`  uxn tal: ${res.uxnTal && res.uxnTal.error ? `ERROR ${res.uxnTal.error}` : `assembled-in-guest frames match the ROM on both tiers=${!!(res.uxnTal && res.uxnTal.assembled)}, a bad source reports + exits on both tiers=${!!(res.uxnTal && res.uxnTal.errorReported)}`}`);
const ok = errors.length === 0 && mouseOk && talOk && demos.every((n) => res[n] && res[n].identical);
for (const n of demos) {
  const r = res[n] || {};
  console.log(`  ${n}: ${r.error ? `ERROR ${r.error}` : `${r.frames} frames, JIT≡interp=${r.identical}`}`);
}
console.log(ok ? 'PASS — the reactor on the wasm-JIT tier is byte-identical to the interpreter' : 'FAIL');
process.exit(ok ? 0 : 1);
