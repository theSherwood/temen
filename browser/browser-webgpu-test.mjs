// Real-browser proof of the `webgpu` capability (#2665): the protocol temen serves natively over wgpu,
// served here by the engine against `navigator.gpu`, in headless Chromium.
//
// 1. `render.temen` — the exact-pixel program the native tests run (crates/temen-run/demos/webgpu/
//    render.c), built as a reactor. It renders one frame to a 64x64 canvas (instanced indexed quads with
//    additive blending, a scissored draw, a sampled texture), reads it back on a later tick, and checks
//    every pixel against its CPU model; it also checks that a malformed stream is refused whole and that
//    a WGSL error comes back as a value. So the same program, and the same oracle, pass on both hosts.
// 2. `gpu_shader.temen` — the playground's GPU card, for a dozen frames: every tick returns 0, and the
//    page reports no error.
//
// No adapter is a failure, not a skip (#2665): the lane runs with one.
import { startServer } from './serve.mjs';
import { fileURLToPath } from 'node:url';
import { dirname } from 'node:path';
const ROOT = dirname(fileURLToPath(import.meta.url));
async function loadChromium() {
  for (const s of ['playwright', '/opt/node22/lib/node_modules/playwright/index.js']) {
    try { const m = await import(s); return m.chromium ?? m.default?.chromium; } catch {}
  }
  throw new Error('no playwright');
}
const chromium = await loadChromium();
const { server, port } = await startServer(ROOT);
// Headless Chromium presents a WebGPU canvas only through ANGLE on Vulkan: without these flags every
// async completion after a pass on the canvas fails ("A valid external Instance reference no longer
// exists"). lavapipe (`mesa-vulkan-drivers`) is the Vulkan driver; WebGPU itself runs on SwiftShader.
const flags = ['--no-sandbox', '--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=vulkan',
  '--ignore-gpu-blocklist'];
const lavapipe = '/usr/share/vulkan/icd.d/lvp_icd.json';
const browser = await chromium.launch({ headless: true, args: flags,
  env: { ...process.env, VK_ICD_FILENAMES: process.env.VK_ICD_FILENAMES ?? lavapipe } });
const page = await browser.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(String(e)));
// A page asset that is absent here (a card built only at deploy) is the page's business,
// `check-play-assets.mjs`'s; one this test uses fails its own open.
page.on('console', (m) => {
  if (m.type() === 'error' && !m.text().startsWith('Failed to load resource')) errors.push(m.text());
});
await page.goto(`http://127.0.0.1:${port}/web/play.html`);
const res = await page.evaluate(async () => {
  const par = await import('./par.js');
  const wg = await import('./webgpu.js');
  const { openReactor } = await import('./reactor.js');
  if (!wg.webgpuAvailable()) return { error: 'no navigator.gpu' };
  const eng = await par.loadEngine();
  const text = () => {
    const p = Number(eng.ex.temen_stdout_ptr()), n = Number(eng.ex.temen_stdout_len());
    return new TextDecoder().decode(new Uint8Array(eng.memory.buffer).slice(p, p + n));
  };
  const asset = async (name) => new Uint8Array(await (await fetch(`./assets/${name}`)).arrayBuffer());
  // What a trapped open or tick says: its text, where stdout's buffer is.
  const trap = () => {
    const p = Number(eng.ex.temen_stdout_ptr()), n = Number(eng.ex.temen_onramp_trap_len());
    return new TextDecoder().decode(new Uint8Array(eng.memory.buffer).slice(p, p + n));
  };
  const canvas = (w, h) => {
    const c = document.createElement('canvas');
    c.width = w;
    c.height = h;
    document.body.append(c);
    return c;
  };

  // 1. The exact-pixel program, a tick at a time: the readback and WebGPU's errors arrive between ticks.
  await wg.initWebGPU(canvas(64, 64));
  const r = openReactor(eng.ex, eng.memory, await asset('render.temen'));
  if (!r) return { error: `render.temen did not open: status ${eng.ex.temen_status()}: ${trap()}` };
  let out = text();
  let ticks = 0;
  while (!out.includes('render: ') && ticks < 600) {
    await new Promise((k) => setTimeout(k, 5));
    const status = await r.frame(false);
    out += text();
    if (status !== 0) return { error: `render tick ${ticks} ended with status ${status}: ${trap()}`, out };
    ticks++;
  }
  r.close();
  wg.teardownWebGPU();

  // 2. The playground's card.
  await wg.initWebGPU(canvas(640, 480));
  const g = openReactor(eng.ex, eng.memory, await asset('gpu_shader.temen'));
  if (!g) return { error: `gpu_shader.temen did not open: status ${eng.ex.temen_status()}: ${trap()}`, out };
  const statuses = [];
  for (let i = 0; i < 12; i++) {
    statuses.push(await g.frame(false));
    await new Promise((k) => setTimeout(k, 5));
  }
  g.close();
  wg.teardownWebGPU();
  return { out, ticks, statuses };
});
await browser.close();
server.close();
console.log('RESULT', JSON.stringify(res));
if (errors.length) console.log('ERRORS', errors.slice(0, 5));
const out = res.out ?? '';
// The canvas's format is the browser's preferred one, RGBA or BGRA: either is a pass (3 or 5).
const want = [/surface: 64x64 format [35]: OK/, /refused: OK/, /frame: OK, 0 pixels differ/,
  /wgsl error reported: OK/, /render: OK/];
const ok = !res.error && want.every((w) => w.test(out)) && res.statuses?.every((s) => s === 0) &&
  errors.length === 0;
console.log(ok ? 'PASS' : 'FAIL');
process.exit(ok ? 0 : 1);
