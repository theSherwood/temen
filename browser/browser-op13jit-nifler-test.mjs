// Real-browser (V8) end-to-end for the **JS-orchestrated op-13 loop on a REAL nifler phase child**
// (#1025 Path 1, the nifler scaling): a resumable driver marshals {fs, stdout, exit} to nifler_ce (the
// child-entry phase), and JS runs nifler_ce's `_start` on **emitted wasm** over its carve — it reads the
// source from the marshaled memfs and writes `.p.nif` back. The output must be byte-identical to the
// interpreter oracle (`temen_run_nifler_crawl_fs`, the top-level nifler on the tree-walker — the two
// nifler builds share the parser, proven by op13_nifler_crawl_matches_inline). This is `nimc.rs`'s phase-1
// nifler crawl, nested under an op-13 driver, tiered up to JIT in the browser.
import { startServer } from './serve.mjs';
import { fileURLToPath } from 'node:url';
import { dirname } from 'node:path';
import { existsSync, readFileSync, writeFileSync, rmSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
const ROOT = dirname(fileURLToPath(import.meta.url));
async function loadChromium() {
  for (const s of ['playwright', '/opt/node22/lib/node_modules/playwright/index.js']) {
    try { const m = await import(s); return m.chromium ?? m.default?.chromium; } catch {}
  }
  throw new Error('playwright not found');
}
const CE_GZ = `${ROOT}/../crates/temen-run/demos/nifler_temen/nifler_ce.temen.gz`;
const NIFLER_GZ = `${ROOT}/web/assets/nifler.temen.gz`;
if (!existsSync(CE_GZ) || !existsSync(NIFLER_GZ) ||
    !existsSync(`${ROOT}/target/wasm32-unknown-unknown/release/temen_browser.wasm`)) {
  console.log('SKIP: nifler_ce / nifler asset or threads wasm absent'); process.exit(0);
}
// Stage nifler_ce (gunzipped) as a temp served asset so the page can fetch its raw bytes.
const CE_TMP = `${ROOT}/web/assets/nifler_ce_test.temen`;
writeFileSync(CE_TMP, gunzipSync(readFileSync(CE_GZ)));

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
  const { driveJitRun } = await import('./wasmjit-module.js');
  const eng = await par.loadEngine();
  const ex = eng.ex, memory = eng.memory;
  const u8 = () => new Uint8Array(memory.buffer);
  const gunzip = async (u) => new Uint8Array(await new Response(new Blob([u]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
  const readOut = () => u8().slice(Number(ex.temen_stdout_ptr()), Number(ex.temen_stdout_ptr()) + ex.temen_stdout_len());

  const niflerCe = new Uint8Array(await (await fetch('./assets/nifler_ce_test.temen')).arrayBuffer());
  const nifler = await gunzip(new Uint8Array(await (await fetch('./assets/nifler.temen.gz')).arrayBuffer()));
  const src = new TextEncoder().encode('import std/[syncio, math]\n\nlet x = 1\necho x\n');
  const file = '/prog.nim', out = '/nimcache/prog.p.nif';
  const enc = new TextEncoder();

  // --- interpreter oracle: top-level nifler over the same source (route A's crawl-step FFI) ---
  const push = (bytes) => { const p = Number(ex.temen_alloc(bytes.length)); u8().set(bytes, p); return p; };
  const fb = enc.encode(file), ob = enc.encode(out);
  const mp = push(nifler), fp = push(fb), op = push(ob), sp = push(src);
  const bcStatus = ex.temen_run_nifler_crawl_fs(mp, nifler.length, fp, fb.length, op, ob.length, sp, src.length);
  const oracle = readOut();
  ex.temen_dealloc(mp, nifler.length); ex.temen_dealloc(fp, fb.length); ex.temen_dealloc(op, ob.length); ex.temen_dealloc(sp, src.length);

  // --- op-13 emitted: nifler_ce as a nested child on the JIT tier ---
  const cp = push(niflerCe), fp2 = push(fb), op2 = push(ob), sp2 = push(src);
  const opened = ex.temen_op13jit_phase_open(cp, niflerCe.length, fp2, fb.length, op2, ob.length, sp2, src.length);
  ex.temen_dealloc(cp, niflerCe.length); ex.temen_dealloc(fp2, fb.length); ex.temen_dealloc(op2, ob.length); ex.temen_dealloc(sp2, src.length);
  if (opened !== 0) return { err: `phase_open failed: ${opened}`, bcStatus };
  let steps = 0, drove = 0, driveErr = null;
  for (;;) {
    if (steps++ > 8) { ex.temen_op13jit_close(); return { err: 'loop did not terminate', bcStatus }; }
    const s = ex.temen_op13jit_step();
    if (s === 0) break;
    if (s === 1) {
      try { await driveJitRun(ex, memory, 'op13jit-nifler'); }
      catch (e) { driveErr = String(e && e.message || e); ex.temen_op13jit_close(); return { err: `driveJitRun: ${driveErr}`, bcStatus }; }
      ex.temen_op13jit_deliver(); drove++; continue;
    }
    ex.temen_op13jit_close();
    return { err: `trap at step ${s}`, bcStatus };
  }
  const result = Number(ex.temen_op13jit_result());
  const plen = ex.temen_op13jit_phase_output();
  const emitted = readOut().slice(0, plen);
  ex.temen_op13jit_close();

  const eq = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);
  return {
    bcStatus, result, drove,
    oracleLen: oracle.length, emittedLen: emitted.length,
    pnifEq: eq(oracle, emitted),
    head: new TextDecoder().decode(oracle.slice(0, 60)),
  };
});

await browser.close(); server.close();
try { rmSync(CE_TMP); } catch {}
console.log('RESULT', JSON.stringify(res, null, 2));
if (errors.length) console.log('ERRORS', errors.slice(0, 6));
const ok = !res.err && res.pnifEq && res.emittedLen > 0 && res.oracleLen > 0;
console.log(`  op13jit-nifler: .p.nif≡=${res.pnifEq} (emitted ${res.emittedLen}B / oracle ${res.oracleLen}B) driver=${res.result} childrenDriven=${res.drove}${res.err ? ` · ERR ${res.err}` : ''}`);
console.log(ok ? 'PASS — real nifler_ce ran nested on the EMITTED tier; .p.nif ≡ interpreter oracle' : 'FAIL');
process.exit(ok ? 0 : 1);
