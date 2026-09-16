// Real-browser (V8) end-to-end for a **guest-issued detached spawn on the JS-orchestrated op-13 loop**
// (#1286): the resumable driver guest calls `Instantiator.instantiate_detached` (op 15, 9-arg form with the
// spawn-time args payload), the engine surfaces `VcpuEvent::InstantiateDetached`, the cdylib mints the child
// a fresh `WebAssembly.Memory` (`foreign_mint`) and stages it as OP13JIT_CHILD_DETACHED, JS drives the
// child's emitted `_start` bound to THAT memory (`driveDetachedRun`), and the joined result flows back to
// the driver. The child reads the argv word the payload carried, `self.attest`s (1 = tier 1, unexposed),
// `vm_map`-grows past its declared 64 KiB (the child memory grows with it), stores into and loads back the
// grown page, and returns `word + attest`. Expected value is fully determined; non-vacuity: the child
// emitted, its memory grew, the driver's own window never held the argv word at the child's address.
//
// Second case (#1527): the same loop with an op-15 **pre-mapped SharedRegion** (the 11-arg form). Two
// `WebAssembly.Memory`s cannot alias, so the servicer copies the region's bytes into the child memory at
// the offset before the emitted child starts and copies the span back into the region at deliver — a
// detached child runs to completion before its parent resumes, so that is observationally the alias.
// The driver mints a 64 KiB region, maps it at 65536 of its own window, stores 41 there, spawns the
// child with the region pre-mapped at 65536 of ITS window, joins, and reads the child's 82 back through
// its own mapping: 1000 × 42 + 82 = 42082. Non-vacuity: the child emitted (no region op of its own),
// the child memory holds 41 (copy-in) and 82 (the child's store) at the offset, and the driver's 82
// can only have arrived by copy-out.
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
const benign = (t) => /Failed to load resource|status of 404/i.test(t);
page.on('pageerror', (e) => errors.push(String(e)));
page.on('console', (m) => { if (m.type() === 'error' && !benign(m.text())) errors.push(m.text()); });
await page.goto(`http://127.0.0.1:${port}/web/play.html`);

const ARGV_WORD = 7306014452085450088n; // "hello-de" little-endian
// The driver (memory 16; entry `(Instantiator, Module, WindowMinter) -> i64`): stores the args blob
// `{argc 1, envc 0} "hello-detached\0"` as three words at 18432 (above its NULL guard), spawns the child
// DETACHED with the payload `(18432, 24)`, no grants, entry 0, window 2^16, then joins it.
const DRIVER = `memory 16
func (i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32) {
  vb0 = i64.const 18432
  vw0 = i64.const 1
  i64.store vb0 vw0
  vb1 = i64.const 18440
  vw1 = i64.const ${ARGV_WORD}
  i64.store vb1 vw1
  vb2 = i64.const 18448
  vw2 = i64.const 110386705817972
  i64.store vb2 vw2
  vmh = i64.extend_i32_u v1
  vmin = i64.extend_i32_u v2
  vz = i64.const 0
  ve = i64.const 0
  vlog = i64.const 16
  vq = i64.const 0
  vap = i64.const 18432
  val = i64.const 24
  vh = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vmin, vmh, vz, vz, ve, vlog, vq, vap, val)
  vr = call.cap 6 1 (i32) -> (i64) v0 (vh)
  return vr
  }
}
`;
// The child-entry child (memory 16, `(i64) -> (i64)`, its starter Instantiator unused): argv word at
// args_base + 8, attest, vm_map [64 KiB, 80 KiB), store/load the word on the grown page, return word+attest.
const CHILD = `memory 16
import 0 "vm_map" (i64, i64, i32) -> (i64)
func (i64) -> (i64) {
block 0 (v0: i64) {
  vab = i64.const 16520
  va = i64.load vab
  vz = i32.const 0
  vat = call.cap 4294967295 4 () -> (i64) vz ()
  voff = i64.const 65536
  vlen = i64.const 16384
  vprot = i32.const 3
  vg = call.import 0 (voff, vlen, vprot)
  vp = i64.const 65600
  i64.store vp va
  vld = i64.load vp
  vs = i64.add vld vat
  return vs
  }
}
`;

// #1527 — the pre-mapped driver (memory 17; entry `(Instantiator, Module, WindowMinter, AddressSpace)
// -> i64`: the fourth arg is what lets it mint a region): create 64 KiB, map at 65536, store 41, spawn
// the child detached with the region pre-mapped at 65536 of its `memory 17` window, join, read 82 back.
const DRIVER_PREMAP = `memory 17
func (i32, i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32, v3: i32) {
  vlen = i64.const 65536
  vrh64 = call.cap 5 5 (i64) -> (i64) v3 (vlen)
  vrh = i32.wrap_i64 vrh64
  vwo = i64.const 65536
  vro = i64.const 0
  vprot = i32.const 3
  vm0 = call.cap 4 0 (i64, i64, i64, i32) -> (i64) vrh (vwo, vro, vlen, vprot)
  vin = i64.const 41
  i64.store vwo vin
  vb = i64.extend_i32_u v2
  vmh = i64.extend_i32_u v1
  vz = i64.const 0
  vlog = i64.const 17
  vreg = i64.extend_i32_u vrh
  vc = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vb, vmh, vz, vz, vz, vlog, vz, vz, vz, vreg, vwo)
  vj = call.cap 6 1 (i32) -> (i64) v0 (vc)
  vk = i64.const 1000
  vm = i64.mul vj vk
  vob = i64.const 65544
  vo = i64.load vob
  vr = i64.add vm vo
  return vr
  }
}
`;
// The pre-mapped child (memory 17, `(i64) -> (i64)`): no handle, no `map` — reads 41 at 65536, stores
// 82 at 65544, returns 42. Pure loads/stores, so it emits.
const CHILD_PREMAP = `memory 17
func (i64) -> (i64) {
block 0 (v0: i64) {
  va = i64.const 65536
  vin = i64.load va
  vtwo = i64.const 2
  vout = i64.mul vin vtwo
  vb = i64.const 65544
  i64.store vb vout
  vone = i64.const 1
  vr = i64.add vin vone
  return vr
  }
}
`;

// Run both cases on one engine: open the detached op-13 loop over `(driver, child)`, step it — a
// `2` (OP13JIT_CHILD_DETACHED) stages an emitted child in a fresh memory that `driveDetachedRun` runs
// and `deliver` banks — and read `probe` words out of the child memory (guest offsets) at the end.
const res = await page.evaluate(async ({ cases }) => {
  const par = await import('./par.js');
  const { driveDetachedRun, jitCacheStats } = await import('./wasmjit-module.js');
  const { foreignMemory } = await import('./foreign-mem.js');
  const eng = await par.loadEngine();
  const ex = eng.ex, memory = eng.memory;
  const u8 = () => new Uint8Array(memory.buffer);
  const slice = (p, n) => u8().slice(p, p + n);
  const push = (bytes) => { const p = Number(ex.temen_alloc(bytes.length)); u8().set(bytes, p); return p; };
  const parse = (src) => {
    const b = new TextEncoder().encode(src);
    const p = push(b);
    const okp = ex.temen_parse(p, b.length);
    const outp = slice(ex.temen_parse_ptr(), ex.temen_parse_len());
    ex.temen_dealloc(p, b.length);
    if (okp !== 1) throw new Error('parse: ' + new TextDecoder().decode(outp));
    return outp;
  };
  const runCase = async ({ name, driverSrc, childSrc, probe }) => {
    const driver = parse(driverSrc), child = parse(childSrc);
    const dp = push(driver), cp = push(child);
    const opened = ex.temen_op13jit_open_detached(dp, driver.length, cp, child.length);
    ex.temen_dealloc(dp, driver.length); ex.temen_dealloc(cp, child.length);
    if (opened !== 0) return { name, err: `open failed: ${opened}` };
    const compilesBefore = jitCacheStats.compiles;
    const steps = [];
    let memId = -1, pagesAtStage = 0, pagesAfter = 0;
    for (let i = 0; i < 8; i++) {
      const s = ex.temen_op13jit_step();
      steps.push(s);
      if (s === 0) break;
      if (s === 2) {
        memId = ex.temen_op13jit_child_mem_id();
        const cm = foreignMemory(memId);
        pagesAtStage = cm.buffer.byteLength / 65536;
        try { await driveDetachedRun(ex, memory, cm, `op13jit-detached-test:${name}`); }
        catch (e) { ex.temen_op13jit_close(); return { name, err: `driveDetachedRun: ${String(e && e.message || e)}`, steps }; }
        pagesAfter = cm.buffer.byteLength / 65536;
        ex.temen_op13jit_deliver();
        continue;
      }
      ex.temen_op13jit_close();
      return { name, err: `unexpected step ${s}`, steps };
    }
    const result = String(ex.temen_op13jit_result());
    // Isolation half: the child's stores must be in the CHILD memory (guest offset + header), and the
    // driver's own window (engine memory) is not where the child ran.
    const header = ex.temen_detached_header_bytes();
    const cm = foreignMemory(memId);
    const dv = new DataView(cm.buffer);
    const probed = probe.map((off) => String(dv.getBigInt64(header + off, true)));
    ex.temen_op13jit_close();
    return { name, steps, result, memId, pagesAtStage, pagesAfter, probed, compiles: jitCacheStats.compiles - compilesBefore };
  };
  const out = [];
  for (const c of cases) out.push(await runCase(c));
  return out;
}, { cases: [
  { name: 'argv+grow', driverSrc: DRIVER, childSrc: CHILD, probe: [65600] },
  { name: 'premap', driverSrc: DRIVER_PREMAP, childSrc: CHILD_PREMAP, probe: [65536, 65544] },
] });

await browser.close();
await new Promise((r) => server.close(r));
console.log('RESULT', JSON.stringify(res, null, 2));
if (errors.length) console.log('ERRORS', errors.slice(0, 5));
const [grow, premap] = res;
const WANT = String(ARGV_WORD + 1n);
const growOk = !grow.err
  && grow.result === WANT
  && grow.steps.length === 2 && grow.steps[0] === 2 && grow.steps[1] === 0   // one detached stage, then done
  && grow.memId >= 0 && grow.pagesAtStage === 2 && grow.pagesAfter >= 3        // header+declared → grown
  && grow.probed[0] === String(ARGV_WORD)                                       // the store landed in the child memory
  && grow.compiles >= 1;
const premapOk = !premap.err
  && premap.result === '42082'
  && premap.steps.length === 2 && premap.steps[0] === 2 && premap.steps[1] === 0
  && premap.memId >= 0 && premap.memId !== grow.memId
  && premap.probed[0] === '41' && premap.probed[1] === '82'                     // copy-in landed; the child's store is in ITS memory
  && premap.compiles >= 1;                                                      // the pre-mapped child emitted
const ok = errors.length === 0 && growOk && premapOk;
console.log(`  op13jit detached: result=${grow.result} (want ${WANT}) steps=${JSON.stringify(grow.steps)} childMem#${grow.memId} pages ${grow.pagesAtStage}→${grow.pagesAfter} storedInChild=${grow.probed && grow.probed[0] === String(ARGV_WORD)} emitted=${grow.compiles}${grow.err ? ` · ERR ${grow.err}` : ''}`);
console.log(`  op13jit detached pre-mapped: result=${premap.result} (want 42082) steps=${JSON.stringify(premap.steps)} childMem#${premap.memId} child[65536]=${premap.probed && premap.probed[0]} child[65544]=${premap.probed && premap.probed[1]} emitted=${premap.compiles}${premap.err ? ` · ERR ${premap.err}` : ''}`);
console.log(ok ? 'PASS — a guest-issued op 15 on the op-13 loop ran its child on the emitted tier in a freshly minted WebAssembly.Memory (args payload + vm_map growth; a pre-mapped region by copy-in/copy-out)' : 'FAIL');
process.exit(ok ? 0 : 1);
