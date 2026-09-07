// Real-browser differential for the **wasm-JIT reactor** (cap-call outlining): for each interactive
// reactor demo, open the SAME unmodified guest on the interpreter and on the wasm-JIT tier and assert
// the presented framebuffer is BYTE-IDENTICAL every frame — the "verified ⇒ same result on both tiers"
// contract that gates the emitter. A successful JIT open also proves the `tick` actually *emitted*
// (`openJitReactor` throws on a not-emittable fallback), i.e. the outlining did its job. bounce/life/
// mandelzoom auto-run deterministically, so the two tiers must produce the identical frame sequence.
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
  // The `mouse` capability on both tiers: a click over the Uxn guest (kind 0, payload
  // (buttons << 24) | (x << 12) | y) must change the next frame on the interpreter AND on the wasm-JIT
  // tier, and the two post-click frames must still agree — the cap-call outlining carries the new
  // waist exactly like `keyboard`.
  {
    const bytes = new Uint8Array(await (await fetch('./assets/uxn.temen')).arrayBuffer());
    const file = { name: 'boot.rom', data: new Uint8Array(await (await fetch(FILES.uxn.url)).arrayBuffer()) };
    const click = [(1 << 24) | (200 << 12) | 100, (200 << 12) | 100];
    const withClick = async (tier) => {
      let r = null;
      if (tier === 'jit') r = await openJitReactor(eng.ex, eng.memory, bytes, file.name, file.data);
      else {
        const p = eng.ex.temen_alloc(bytes.length); new Uint8Array(eng.memory.buffer).set(bytes, p);
        const nameBytes = new TextEncoder().encode(file.name);
        const nameP = eng.ex.temen_alloc(nameBytes.length), dataP = eng.ex.temen_alloc(file.data.length);
        const view = new Uint8Array(eng.memory.buffer); view.set(nameBytes, nameP); view.set(file.data, dataP);
        const opened = eng.ex.temen_onramp_open_fs(p, bytes.length, nameP, nameBytes.length, dataP, file.data.length);
        eng.ex.temen_dealloc(p, bytes.length); eng.ex.temen_dealloc(nameP, nameBytes.length); eng.ex.temen_dealloc(dataP, file.data.length);
        if (opened !== 0) throw new Error(`interp open failed: ${opened}`);
      }
      const frame = () => (r ? r.frame() : eng.ex.temen_onramp_frame());
      const push = (kind, payload) => (r ? eng.ex.temen_onramp_jit_mouse(kind, payload) : eng.ex.temen_onramp_mouse(kind, payload));
      frame(); frame();
      const before = hashFB();
      push(0, click[0]); push(0, click[1]);
      frame();
      const after = hashFB();
      if (r) r.close(); else eng.ex.temen_onramp_close();
      return { before, after };
    };
    try {
      const i = await withClick('interp'), j = await withClick('jit');
      out.uxnMouse = { changed: i.before !== i.after && j.before !== j.after, tiersAgree: i.after === j.after, i, j };
    } catch (e) {
      out.uxnMouse = { error: e.message };
    }
  }
  // The Uxntal card's path on both tiers: the demo's SOURCE served as boot.tal is assembled in the guest
  // and must render the committed ROM's exact frames; an unassemblable source must surface its error on
  // stdout (`temen_stdout_*` after the frame — on the JIT tier via temen_onramp_jit_present) and exit.
  {
    const bytes = new Uint8Array(await (await fetch('./assets/uxn.temen')).arrayBuffer());
    const tal = new TextEncoder().encode(talSource);
    const bad = new TextEncoder().encode('|0100 #01 #02 ADD\n,&nope JMP BRK\n');
    const openInterp = (file) => {
      const p = eng.ex.temen_alloc(bytes.length); new Uint8Array(eng.memory.buffer).set(bytes, p);
      const nameBytes = new TextEncoder().encode(file.name);
      const nameP = eng.ex.temen_alloc(nameBytes.length), dataP = eng.ex.temen_alloc(file.data.length);
      const view = new Uint8Array(eng.memory.buffer); view.set(nameBytes, nameP); view.set(file.data, dataP);
      const opened = eng.ex.temen_onramp_open_fs(p, bytes.length, nameP, nameBytes.length, dataP, file.data.length);
      eng.ex.temen_dealloc(p, bytes.length); eng.ex.temen_dealloc(nameP, nameBytes.length); eng.ex.temen_dealloc(dataP, file.data.length);
      if (opened !== 0) throw new Error(`interp open failed: ${opened}`);
    };
    const stdoutNow = () => {
      const n = eng.ex.temen_stdout_len(), p = eng.ex.temen_stdout_ptr();
      return new TextDecoder().decode(new Uint8Array(eng.memory.buffer).slice(p, p + n));
    };
    const run = async (tier, file, n) => {
      let r = null;
      if (tier === 'jit') r = await openJitReactor(eng.ex, eng.memory, bytes, file.name, file.data);
      else openInterp(file);
      const hs = [], outs = [];
      let status = 0;
      for (let i = 0; i < n; i++) {
        status = r ? r.frame() : eng.ex.temen_onramp_frame();
        outs.push(stdoutNow());
        if (status !== 0) break;
        hs.push(hashFB());
      }
      if (r) r.close(); else eng.ex.temen_onramp_close();
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
console.log(ok ? 'PASS — wasm-JIT reactor byte-identical to the interpreter' : 'FAIL');
process.exit(ok ? 0 : 1);
