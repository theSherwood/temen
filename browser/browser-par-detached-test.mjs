// Real-browser (V8) test for #1286 slice 3b: **concurrent detached children on the parallel Worker
// driver**. A root vCPU (on its own Worker) holds an `Instantiator`, a child `Module` and a
// `WindowMinter` (the §14 recipe + `minter`), and issues three §5 `instantiate_detached` (op 15) spawns
// back to back — each child lands in its OWN fresh shared `WebAssembly.Memory` on its OWN Worker (the
// spawning Worker mints + seeds the memory from the event's segment blob; the page relays the Worker
// start) — then a fourth the exhausted minter must refuse probeably (`-EINVAL`), then joins the three.
// Each child reads the 8-byte payload the root passed at `module_args_base()`, `vm_map`s a page PAST
// its declared 64 KiB window (the grow reaches the child memory through `Region::Foreign` →
// `foreign_grow`), stores/loads the word on the grown page and returns it + 1.
//
// The root returns Σ(word_i + 1) + h4 = (1001 + 2001 + 3001) - 22. Non-vacuity: `started === 4` — the
// root and three child Workers were all created (the children are spawned before any join, so all
// three run at once, each over its own memory).
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
const errors = [];
const benign = (t) => /Failed to load resource|status of 404/i.test(t);
page.on('pageerror', (e) => errors.push(String(e)));
page.on('console', (m) => { if (m.type() === 'error' && !benign(m.text())) errors.push(m.text()); });
await page.goto(`http://127.0.0.1:${port}/web/play.html`);

// The root: v0 Instantiator, v1 the child Module, v2 the WindowMinter. Three payload words at
// 18432/18440/18448 (above the 16 KiB NULL guard), three 9-arg spawns (payload `(addr, 8)`), one 7-arg
// spawn the minter refuses, three joins.
const OP15 = 'call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vmin, vmh, vz, vz, ve, vlog, vq, ';
const ROOT_SRC = `memory 16
func (i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32) {
  vb0 = i64.const 18432
  vw0 = i64.const 1000
  i64.store vb0 vw0
  vb1 = i64.const 18440
  vw1 = i64.const 2000
  i64.store vb1 vw1
  vb2 = i64.const 18448
  vw2 = i64.const 3000
  i64.store vb2 vw2
  vmh = i64.extend_i32_u v1
  vmin = i64.extend_i32_u v2
  vz = i64.const 0
  ve = i64.const 0
  vlog = i64.const 16
  vq = i64.const 0
  vl = i64.const 8
  vh0 = ${OP15}vb0, vl)
  vh1 = ${OP15}vb1, vl)
  vh2 = ${OP15}vb2, vl)
  vh3 = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vmin, vmh, vz, vz, ve, vlog, vq)
  vr0 = call.cap 6 1 (i32) -> (i64) v0 (vh0)
  vr1 = call.cap 6 1 (i32) -> (i64) v0 (vh1)
  vr2 = call.cap 6 1 (i32) -> (i64) v0 (vh2)
  vs0 = i64.add vr0 vr1
  vs1 = i64.add vs0 vr2
  vh3x = i64.extend_i32_s vh3
  vs = i64.add vs1 vh3x
  return vs
  }
}
`;
// The child: the payload word at module_args_base() (16384 + 128), a vm_map of [64 KiB, 80 KiB) past
// the declared window, the word stored + reloaded on the grown page, returned + 1.
const CHILD_SRC = `memory 16
import 0 "vm_map" (i64, i64, i32) -> (i64)

func (i64) -> (i64) {
block 0 (v0: i64) {
  vab = i64.const 16512
  va = i64.load vab
  voff = i64.const 65536
  vlen = i64.const 16384
  vprot = i32.const 3
  vg = call.import 0 (voff, vlen, vprot)
  vp = i64.const 65600
  i64.store vp va
  vld = i64.load vp
  vone = i64.const 1
  vs = i64.add vld vone
  return vs
  }
}
`;
const CHILD_LOG2 = 16, MINTER_QUOTA = 3 * (1 << CHILD_LOG2);

const res = await page.evaluate(async ({ rootSrc, childSrc, minter }) => {
  const { loadEngine, makeRunner } = await import('./par.js');
  const eng = await loadEngine();
  const ex = eng.ex, memory = eng.memory;
  const u8 = () => new Uint8Array(memory.buffer);
  const parse = (src) => {
    const b = new TextEncoder().encode(src);
    const p = ex.temen_alloc(b.length);
    u8().set(b, p);
    const ok = ex.temen_parse(p, b.length);
    const out = u8().slice(ex.temen_parse_ptr(), ex.temen_parse_ptr() + ex.temen_parse_len());
    ex.temen_dealloc(p, b.length);
    if (ok !== 1) throw new Error('parse: ' + new TextDecoder().decode(out));
    return out;
  };
  const root = parse(rootSrc), child = parse(childSrc);
  const run = makeRunner(eng);
  try {
    const { value, started } = await run(root, { inst: true, unit: child, minter });
    return { value: value.toString(), started };
  } catch (e) {
    return { err: String(e && e.message ? e.message : e) };
  }
}, { rootSrc: ROOT_SRC, childSrc: CHILD_SRC, minter: MINTER_QUOTA });

await browser.close();
await new Promise((r) => server.close(r));
console.log('RESULT', JSON.stringify(res));
if (errors.length) console.log('ERRORS', errors.slice(0, 5));
const EXPECT = String(1001 + 2001 + 3001 - 22);
const ok = errors.length === 0 && !res.err && res.value === EXPECT && res.started === 4;
console.log(`  detached children across Workers: value ${res.value}/${EXPECT} workers ${res.started}/4${res.err ? ` · ERR ${res.err}` : ''}`);
console.log(ok ? 'PASS — three detached children ran concurrently, each on its own Worker in its own WebAssembly.Memory, grew it on vm_map, and the exhausted minter refused a fourth' : 'FAIL');
process.exit(ok ? 0 : 1);
