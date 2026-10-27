// Real-browser (V8) test for #1286 slice 3b: **concurrent detached children on the parallel driver**
// (executor 2, #1414 B6-3c). A root (on its own Worker) holds an `Instantiator`, a child `Module` and a
// `WindowMinter` (the §14 recipe + `minter`), and issues three §5 detached spawns (op-17 records)
// back to back — each child lands in its OWN fresh shared `WebAssembly.Memory`, its thread on its OWN
// Worker (the spawning thread's Worker mints the memory, and the Worker that runs the child's thread
// is handed it) — then a fourth the exhausted minter must refuse probeably (`-EINVAL`), then joins the
// three. Each child reads the 8-byte payload the root passed at `module_args_base()`, `vm_map`s a page
// PAST its declared 64 KiB window (the grow reaches the child memory through `Region::Foreign` → its
// accessor's `grow`), stores/loads the word on the grown page and returns it + 1.
//
// The root returns Σ(word_i + 1) + h4 = (1001 + 2001 + 3001) - 22. Non-vacuity: `started === 4` — the
// root's and three children's Workers were all created (the children are spawned before any join, so
// all three run at once), and `childMems === 3`, one thread over each child's own memory.
//
// #1865, #1414 ED-2: the same run again under `tierup` — each child's entry runs on its granted
// module's EMITTED wasm over its own Memory. Its `vm_map` sits in a helper the emitted entry bounces to
// the interpreter, so the emitted store on the grown page passes only if the Worker re-read the
// child's `"mapped"` after the bounce. Non-vacuity: `tierups === 3`, one per child that ran emitted.
//
// Two more #1865 pins. A root's child that runs the root's own child image (granted as `"child"`)
// runs that image's emit under `tierup` and answers its func 1 (7), though the granted unit's func 1
// answers 9: a child runs the emit of its own program only. And a carve spawn (op 0) fails closed.
//
// #2251: a plain root that returns its `self.parallelism` answers the page's core count
// (`navigator.hardwareConcurrency`): each of the run's threads gets a Worker of its own.
//
// #1414 5d: on the pump too, on the page's own engine (`temen_run_nested`), each child's window is a
// memory of its own. Two runs of a root that runs 90 children one after another make 180 children, more
// than a page can hold memories at once (Chromium holds about 125). When a child's window drops, the
// engine gives its memory back, and the page zeroes it and hands it to the next child, so the 180
// share one memory. Each child reads its word before it writes it, so a byte left by the child before
// would show in the sum. Non-vacuity: the runs minted one memory, which the page still holds.
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
// 18432/18440/18448 (above the 16 KiB NULL guard), three spawns with a payload `(addr, 8)`, one without
// that the minter refuses, three joins.
//
// `spawn(dst, at, payload)`: a spawn record (op 17) laid at `at` by stores, then the spawn into
// `dst`: a 64 KiB window, no pager, the child Module (v1) paid from the minter (v2), no region, and
// the 8-byte payload at `payload` if there is one. Every field left 0 is none.
const spawn = (dst, at, payload) => `  ${dst}a = i64.const ${at}
  ${dst}l = i32.const 16
  i32.store ${dst}a ${dst}l
  ${dst}n = i32.const -1
  i32.store ${dst}a ${dst}n offset=4
  i32.store ${dst}a v1 offset=8
  i32.store ${dst}a v2 offset=12
  i32.store ${dst}a ${dst}n offset=48
${payload === undefined ? '' : `  ${dst}p = i64.const ${payload}
  i64.store ${dst}a ${dst}p offset=32
  ${dst}s = i64.const 8
  i64.store ${dst}a ${dst}s offset=40
`}  ${dst} = call.cap 6 17 (i64) -> (i32) v0 (${dst}a)
`;
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
${spawn('vh0', 17408, 18432)}${spawn('vh1', 17496, 18440)}${spawn('vh2', 17584, 18448)}${spawn('vh3', 17672)}  vr0_st, vr0 = call.cap 6 1 (i32) -> (i64, i64) v0 (vh0)
  vr1_st, vr1 = call.cap 6 1 (i32) -> (i64, i64) v0 (vh1)
  vr2_st, vr2 = call.cap 6 1 (i32) -> (i64, i64) v0 (vh2)
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
// (func 1, → 7), granted as `"child"` (#2219) — through an op-17 record and joins it, with a
// granted unit whose func 1 (→ 9) is emitted. Only a child that runs the granted unit may run its
// emit, so under `tierup` this child runs its own image's emit and answers 7.
const SELF_ROOT_SRC = `memory 16
data 17504 "child"
export 0 func "_child" 1
func (i32, i32, i32) -> (i64) {
block 0 (v0: i32, vm: i32, vbud: i32) {
  vcp = i64.const 17504
  vcl = i64.const 5
  vchild = self.resolve vcp vcl
  vr0 = i64.const 17408
  vf0 = i64.const -4294967296
  i64.store vr0 vf0
  vr1 = i64.const 17416
  i32.store vr1 vchild
  vr1b = i64.const 17420
  i32.store vr1b vbud
  vr6 = i64.const 17456
  vnone = i32.const -1
  i32.store vr6 vnone
  vh = call.cap 6 17 (i64) -> (i32) v0 (vr0)
  vr_st, vr = call.cap 6 1 (i32) -> (i64, i64) v0 (vh)
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
// A §14 carve spawn (op 0) fails closed on the parallel driver as on every engine, a `CapFault` trap
// (#1867).
const CARVE_ROOT_SRC = `memory 20
func (i32) -> (i64) {
block 0 (v0: i32) {
  ve = i64.const 1
  voff = i64.const 65536
  vslog = i64.const 16
  vq = i64.const 0
  vh = call.cap 6 0 (i64, i64, i64, i64) -> (i32) v0 (ve, voff, vslog, vq)
  vr_st, vr = call.cap 6 1 (i32) -> (i64, i64) v0 (vh)
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

// #1414 5d: the root runs its `kid` export, a child image, 90 times one after another, each spawned
// detached by the record at 34816 and paid from the run's `budget`, and joined before the next. The
// child reads a word of its own window, stores 7 there, and returns the word it read plus the one it
// reads back: 7 in a zeroed window, so the root returns 630. (A run of `temen_run_nested` has fuel
// for 91 such children.)
const PUMP_CHILDREN = 90, PUMP_RUNS = 2;
const PUMP_ROOT_SRC = `memory 17
data ro 40960 "budget"
data ro 40976 "kid"
export 0 func "kid" 1
func (i32) -> (i64) {
block 0 (vinst: i32) {
  vbp = i64.const 40960
  vbl = i64.const 6
  vb = self.resolve vbp vbl
  vkp = i64.const 40976
  vkl = i64.const 3
  vk = self.resolve vkp vkl
  vrec = i64.const 34816
  vw0 = i64.const -4294967296
  i64.store vrec vw0
  i32.store vrec vk offset=8
  i32.store vrec vb offset=12
  vw6 = i64.const 4294967295
  i64.store vrec vw6 offset=48
  vi = i64.const 0
  vacc = i64.const 0
  br 1(vinst, vi, vacc)
}
block 1 (vin: i32, vi1: i64, vacc1: i64) {
  vrec1 = i64.const 34816
  vh = call.cap 6 17 (i64) -> (i32) vin (vrec1)
  vj_st, vj = call.cap 6 1 (i32) -> (i64, i64) vin (vh)
  vacc2 = i64.add vacc1 vj
  vone = i64.const 1
  vi2 = i64.add vi1 vone
  vn = i64.const ${PUMP_CHILDREN}
  vmore = i64.lt_s vi2 vn
  br_if vmore 1(vin, vi2, vacc2) 2(vacc2)
}
block 2 (vres: i64) {
  return vres
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  va = i64.const 16400
  vold = i64.load va
  v7 = i64.const 7
  i64.store va v7
  vl = i64.load va
  vs = i64.add vold vl
  return vs
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

const res = await page.evaluate(async ({ rootSrc, childSrc, minter, selfRootSrc, nineUnitSrc, carveRootSrc, cpusSrc, pumpRootSrc, pumpRuns }) => {
  const { loadEngine, makeRunner } = await import('./par.js');
  const { foreignMemory, registerForeign, releaseForeign } = await import('./foreign-mem.js');
  const once = async (eng, emitted = false, src = { root: rootSrc, child: childSrc, minter }) => {
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
  // The children's entries are short: the emitted runs' engines tier up every eligible call at its
  // start (#2474), as coop's own tests do.
  if (emitted) ex.temen_coop_set_hotness(0, 0);
  try {
    const { value, started, tierups, childMems } = await run(root, src.plain ? {}
      : { inst: true, unit: child, minter: src.minter, winSize: src.winSize, tierup: emitted });
    return { value: value.toString(), started, tierups, childMems };
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
  // The pump: the root's children, one after another, on this page's own engine, in two runs. The
  // engine's next foreign id before and after the runs counts the memories they minted, and those
  // still registered are the page's spares.
  const pumped = await (async () => {
    const pe = await loadEngine(fresh), pex = pe.ex;
    const pu8 = () => new Uint8Array(pe.memory.buffer);
    const b = new TextEncoder().encode(pumpRootSrc);
    const p = pex.temen_alloc(b.length);
    pu8().set(b, p);
    const parsed = pex.temen_parse(p, b.length);
    pex.temen_dealloc(p, b.length);
    const bytes = pu8().slice(pex.temen_parse_ptr(), pex.temen_parse_ptr() + pex.temen_parse_len());
    if (parsed !== 1) return { err: 'parse: ' + new TextDecoder().decode(bytes) };
    const nextId = () => {
      const id = registerForeign(pe.memory, new WebAssembly.Memory({ initial: 1, maximum: 1, shared: true }));
      releaseForeign(pe.memory, id);
      return id;
    };
    const m = pex.temen_alloc(bytes.length);
    pu8().set(bytes, m);
    const before = nextId();
    const runs = [];
    for (let r = 0; r < pumpRuns; r++) {
      const value = pex.temen_run_nested(m, bytes.length);
      runs.push(`${value}:${pex.temen_status()}`);
    }
    const after = nextId();
    const minted = after - before - 1;
    let held = 0;
    for (let id = before + 1; id < after; id++) if (foreignMemory(pe.memory, id) !== undefined) held++;
    return { runs, minted, held };
  })();
  return { ...first, second, emitted, self, carve, cpus, cores, pumped, reused: fresh.module === eng.module, freshMemory: fresh.memory !== eng.memory && freshBytes < usedBytes };
}, { rootSrc: ROOT_SRC, childSrc: CHILD_SRC, minter: MINTER_QUOTA, selfRootSrc: SELF_ROOT_SRC, nineUnitSrc: NINE_UNIT_SRC, carveRootSrc: CARVE_ROOT_SRC, cpusSrc: CPUS_SRC, pumpRootSrc: PUMP_ROOT_SRC, pumpRuns: PUMP_RUNS });

await browser.close();
await new Promise((r) => server.close(r));
console.log('RESULT', JSON.stringify(res));
if (errors.length) console.log('ERRORS', errors.slice(0, 5));
const EXPECT = String(1001 + 2001 + 3001 - 22);
const again = res.second && !res.second.err && res.second.value === EXPECT && res.second.started === 4;
const em = res.emitted;
const emittedOk = em && !em.err && em.value === EXPECT && em.started === 4 && em.tierups === 3;
const sf = res.self, cv = res.carve;
const selfOk = sf && !sf.err && sf.value === '7' && sf.tierups === 1;
const carveOk = cv && cv.err === 'guest trap: CapFault';
const cpusOk = res.cpus && !res.cpus.err && res.cpus.value === res.cores;
const pm = res.pumped;
const pumpRun = `${7 * PUMP_CHILDREN}:0`;
const pumpOk = pm && !pm.err && pm.runs.every((r) => r === pumpRun)
  && pm.minted === 1 && pm.held === 1;
const ok = errors.length === 0 && !res.err && res.value === EXPECT && res.started === 4 && res.childMems === 3 && again && res.reused && res.freshMemory && emittedOk && selfOk && carveOk && cpusOk && pumpOk;
console.log(`  detached children across Workers: value ${res.value}/${EXPECT} workers ${res.started}/4 threads over a child's own memory ${res.childMems}/3${res.err ? ` · ERR ${res.err}` : ''}`);
console.log(`  again on loadEngine(prev): value ${res.second?.value}/${EXPECT} · compiled module reused ${res.reused} · fresh memory ${res.freshMemory}`);
console.log(`  on emitted wasm: value ${em?.value}/${EXPECT} workers ${em?.started}/4 emitted children ${em?.tierups}/3${em?.err ? ` · ERR ${em.err}` : ''}`);
console.log(`  own-module child under tierup: value ${sf?.value}/7 emitted ${sf?.tierups}/1${sf?.err ? ` · ERR ${sf.err}` : ''}`);
console.log(`  carve spawn: ${carveOk ? 'fails closed' : `NOT refused (${cv?.err ?? cv?.value})`}`);
console.log(`  self.parallelism: ${res.cpus?.value}/${res.cores} (the page's core count)${res.cpus?.err ? ` · ERR ${res.cpus.err}` : ''}`);
console.log(`  on the pump: runs (value:status) ${pm?.runs?.join(' ')} of ${pumpRun} · memories minted ${pm?.minted}/1, held ${pm?.held}/1${pm?.err ? ` · ERR ${pm.err}` : ''}`);
console.log(ok ? 'PASS — three detached children ran concurrently, each on its own Worker in its own WebAssembly.Memory, grew it on vm_map, and the exhausted minter refused a fourth; on the pump, 180 children ran one after another, each in a memory of its own, which the page zeroed and handed to the next' : 'FAIL');
process.exit(ok ? 0 : 1);
