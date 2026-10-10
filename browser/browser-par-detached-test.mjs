// Real-browser (V8) test for #1286 slice 3b: **concurrent detached children on the parallel Worker
// driver**. A root vCPU (on its own Worker) holds an `Instantiator`, a child `Module` and a
// `WindowMinter` (the §14 recipe + `minter`), and issues three §5 `instantiate_detached` (op 15) spawns
// back to back — each child lands in its OWN fresh shared `WebAssembly.Memory` on its OWN Worker (the
// spawning Worker mints the memory and posts it with the admitted child's ticket; the child's Worker
// starts it, and the engine seeds the window; the page relays the Worker start) — then a fourth the
// exhausted minter must refuse probeably (`-EINVAL`), then joins the three.
// Each child reads the 8-byte payload the root passed at `module_args_base()`, `vm_map`s a page PAST
// its declared 64 KiB window (the grow reaches the child memory through `Region::Foreign` →
// `foreign_grow`), stores/loads the word on the grown page and returns it + 1.
//
// The root returns Σ(word_i + 1) + h4 = (1001 + 2001 + 3001) - 22. Non-vacuity: `started === 4` — the
// root and three child Workers were all created (the children are spawned before any join, so all
// three run at once, each over its own memory).
//
// #1865: the same run again with `instCodegen` — each child's entry runs on EMITTED wasm bound to its
// own Memory. Its `vm_map` sits in a helper the emitted entry bounces to the child's own vCPU, so the
// emitted store on the grown page passes only if the Worker re-read the child's `"mapped"` after the
// bounce. Non-vacuity: `tierups === 3`, one per child that ran emitted.
//
// Two more #1865 pins. A root's child that runs the root's own child image (granted as `"child"`)
// stays on the interpreter under `instCodegen` even though the granted unit emits functions at the same
// indices: only a child whose program is the unit runs the unit's emit. And a carve spawn (op 0) fails
// closed.
//
// #2251: a plain root that returns its `self.parallelism` answers the page's core count
// (`navigator.hardwareConcurrency`), since this driver gives each vCPU its own Worker.
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
// the declared window (in func 1, which the emitted entry bounces to), the word stored + reloaded on
// the grown page, returned + 1.
const CHILD_SRC = `memory 16
import 0 "vm_map" (i64, i64, i32) -> (i64)

func (i64) -> (i64) {
block 0 (v0: i64) {
  vab = i64.const 16512
  va = i64.load vab
  vg = call 1 ()
  vp = i64.const 65600
  i64.store vp va
  vld = i64.load vp
  vone = i64.const 1
  vs = i64.add vld vone
  return vs
  }
}
func () -> (i64) {
block 0 () {
  voff = i64.const 65536
  vlen = i64.const 16384
  vprot = i32.const 3
  vg = call.import 0 (voff, vlen, vprot)
  return vg
  }
}
`;
// The budget pays for three children: each one's window and the 16 KiB it grows (#1909). The fourth
// spawn, made before any join, finds less than a window left, however far the three have grown.
const CHILD_LOG2 = 16, CHILD_GROWTH = 16384;
const MINTER_QUOTA = 3 * ((1 << CHILD_LOG2) + CHILD_GROWTH);
// #1865: the root spawns a copy of itself — its child image, which starts at its `_child` export
// (func 1, → 7), granted as `"child"` (#2219) — through an op-17 v1 record and joins it, with a
// granted unit whose func 1 (→ 9) is emitted. Only a child that runs the granted unit may run its
// emit, so with `instCodegen` this child stays interpreted: 7, no emitted children.
const SELF_ROOT_SRC = `memory 16
data 17504 "child"
export 0 func "_child" 1
func (i32, i32, i32) -> (i64) {
block 0 (v0: i32, vm: i32, vbud: i32) {
  vcp = i64.const 17504
  vcl = i64.const 5
  vchild = self.resolve vcp vcl
  vr0 = i64.const 17408
  vf0 = i64.const 1
  i64.store vr0 vf0
  vr2 = i64.const 17424
  vf2 = i64.const -4294967296
  i64.store vr2 vf2
  vr3 = i64.const 17432
  i32.store vr3 vchild
  vr3b = i64.const 17436
  i32.store vr3b vbud
  vr9 = i64.const 17480
  vnone = i32.const -1
  i32.store vr9 vnone
  vh = call.cap 6 17 (i64) -> (i32) v0 (vr0)
  vr = call.cap 6 1 (i32) -> (i64) v0 (vh)
  return vr
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v7 = i64.const 7
  return v7
  }
}
`;
const NINE_UNIT_SRC = `memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 1
  return v1
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v9 = i64.const 9
  return v9
  }
}
`;
// A §14 carve spawn (op 0) on the par driver fails closed, as a `CapFault` trap: its children are
// detached (#1865).
const CARVE_ROOT_SRC = `memory 20
func (i32) -> (i64) {
block 0 (v0: i32) {
  ve = i64.const 1
  voff = i64.const 65536
  vslog = i64.const 16
  vq = i64.const 0
  vh = call.cap 6 0 (i64, i64, i64, i64) -> (i32) v0 (ve, voff, vslog, vq)
  vr = call.cap 6 1 (i32) -> (i64) v0 (vh)
  return vr
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v5 = i64.const 5
  return v5
  }
}
`;

// #2251: `self.parallelism`, returned.
const CPUS_SRC = `memory 16
func () -> (i64) {
block 0 () {
  v0 = i32.const 0
  v1 = call.cap 4294967295 19 () -> (i64) v0 ()
  return v1
  }
}
`;

const res = await page.evaluate(async ({ rootSrc, childSrc, minter, selfRootSrc, nineUnitSrc, carveRootSrc, cpusSrc }) => {
  const { loadEngine, makeRunner } = await import('./par.js');
  const once = async (eng, codegen = false, src = { root: rootSrc, child: childSrc, minter }) => {
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
  const root = parse(src.root), child = src.child ? parse(src.child) : null;
  const run = makeRunner(eng);
  try {
    const { value, started, tierups } = await run(root, src.plain ? {} : codegen
      ? { instCodegen: true, unit: child, minter: src.minter, winSize: src.winSize }
      : { inst: true, unit: child, minter: src.minter, winSize: src.winSize });
    return { value: value.toString(), started, tierups };
  } catch (e) {
    return { err: String(e && e.message ? e.message : e) };
  }
  };
  const eng = await loadEngine();
  const first = await once(eng);
  // A second run on a fresh engine from `loadEngine(prev)` — how a host runs many guests without the
  // shared memory accumulating each run's allocations: the compiled module is reused, the memory is
  // new (none of the first run's window/stacks), and the run is the same.
  const fresh = await loadEngine(eng);
  const usedBytes = eng.memory.buffer.byteLength, freshBytes = fresh.memory.buffer.byteLength;
  const second = await once(fresh);
  const emitted = await once(await loadEngine(fresh), true);
  const self = await once(await loadEngine(fresh), true, { root: selfRootSrc, child: nineUnitSrc, minter: 1 << 16 });
  const carve = await once(await loadEngine(fresh), false, { root: carveRootSrc, child: null, minter: 0, winSize: 1 << 20 });
  const cpus = await once(await loadEngine(fresh), false, { root: cpusSrc, child: null, plain: true });
  const cores = String(navigator.hardwareConcurrency || 1);
  return { ...first, second, emitted, self, carve, cpus, cores, reused: fresh.module === eng.module, freshMemory: fresh.memory !== eng.memory && freshBytes < usedBytes };
}, { rootSrc: ROOT_SRC, childSrc: CHILD_SRC, minter: MINTER_QUOTA, selfRootSrc: SELF_ROOT_SRC, nineUnitSrc: NINE_UNIT_SRC, carveRootSrc: CARVE_ROOT_SRC, cpusSrc: CPUS_SRC });

await browser.close();
await new Promise((r) => server.close(r));
console.log('RESULT', JSON.stringify(res));
if (errors.length) console.log('ERRORS', errors.slice(0, 5));
const EXPECT = String(1001 + 2001 + 3001 - 22);
const again = res.second && !res.second.err && res.second.value === EXPECT && res.second.started === 4;
const em = res.emitted;
const emittedOk = em && !em.err && em.value === EXPECT && em.started === 4 && em.tierups === 3;
const sf = res.self, cv = res.carve;
const selfOk = sf && !sf.err && sf.value === '7' && sf.tierups === 0;
const carveOk = cv && cv.err === 'guest trap: CapFault';
const cpusOk = res.cpus && !res.cpus.err && res.cpus.value === res.cores;
const ok = errors.length === 0 && !res.err && res.value === EXPECT && res.started === 4 && again && res.reused && res.freshMemory && emittedOk && selfOk && carveOk && cpusOk;
console.log(`  detached children across Workers: value ${res.value}/${EXPECT} workers ${res.started}/4${res.err ? ` · ERR ${res.err}` : ''}`);
console.log(`  again on loadEngine(prev): value ${res.second?.value}/${EXPECT} · compiled module reused ${res.reused} · fresh memory ${res.freshMemory}`);
console.log(`  on emitted wasm: value ${em?.value}/${EXPECT} workers ${em?.started}/4 emitted children ${em?.tierups}/3${em?.err ? ` · ERR ${em.err}` : ''}`);
console.log(`  own-module child under instCodegen: value ${sf?.value}/7 emitted ${sf?.tierups}/0${sf?.err ? ` · ERR ${sf.err}` : ''}`);
console.log(`  carve spawn: ${carveOk ? 'fails closed' : `NOT refused (${cv?.err ?? cv?.value})`}`);
console.log(`  self.parallelism: ${res.cpus?.value}/${res.cores} (the page's core count)${res.cpus?.err ? ` · ERR ${res.cpus.err}` : ''}`);
console.log(ok ? 'PASS — three detached children ran concurrently, each on its own Worker in its own WebAssembly.Memory, grew it on vm_map, and the exhausted minter refused a fourth' : 'FAIL');
process.exit(ok ? 0 : 1);
