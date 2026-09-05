// Real-browser (V8) end-to-end for the **JS-orchestrated op-13 loop on a REAL hexer phase child**
// (#1025 3a.3, extending the nifler tier-up to the second front-end phase via the phase-agnostic
// `temen_op13jit_phase_open_argv`): a resumable driver marshals {fs, stdout, exit} to hexer_ce (the
// child-entry phase), seeds the shared memfs with the committed semchecked system module
// (`sysvq0asl.s.nif` + its `.s.idx.nif`), and JS runs hexer_ce's `_start` on **emitted wasm** over its
// 256 MiB carve — it reads the `.s.nif` and writes the Leng `.x.nif`. The output must be byte-identical
// to the committed `sysvq0asl.x.nif` (hexer is deterministic for a fixed input — the same oracle the
// headless `rust_driver_hexer.rs` gate uses). This is `nimc.rs`'s phase-3 hexer, nested under an op-13
// driver, tiered up to JIT in the browser — the piece the ~180s card needs beyond nifler.
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
const FX = `${ROOT}/../crates/temen-run/demos/nim_frontend/fixtures`;
const CE_GZ = `${FX}/hexer_ce.temen.gz`;
const SNIF_GZ = `${FX}/sysvq0asl.s.nif.gz`;
const SIDX = `${FX}/sysvq0asl.s.idx.nif`;
const XNIF_GZ = `${FX}/sysvq0asl.x.nif.gz`;
if (![CE_GZ, SNIF_GZ, SIDX, XNIF_GZ].every(existsSync) ||
    !existsSync(`${ROOT}/target/wasm32-unknown-unknown/release/temen_browser.wasm`)) {
  console.log('SKIP: hexer_ce / .s.nif / .s.idx.nif / expected asset or threads wasm absent'); process.exit(0);
}
// Stage the (gunzipped) inputs + expected output as temp served assets so the page can fetch raw bytes.
const CE_TMP = `${ROOT}/web/assets/hexer_ce_test.temen`;
const SNIF_TMP = `${ROOT}/web/assets/hexer_snif_test.nif`;
const SIDX_TMP = `${ROOT}/web/assets/hexer_sidx_test.nif`;
const XNIF_TMP = `${ROOT}/web/assets/hexer_xnif_test.nif`;
writeFileSync(CE_TMP, gunzipSync(readFileSync(CE_GZ)));
writeFileSync(SNIF_TMP, gunzipSync(readFileSync(SNIF_GZ)));
writeFileSync(SIDX_TMP, readFileSync(SIDX));
writeFileSync(XNIF_TMP, gunzipSync(readFileSync(XNIF_GZ)));

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
  const { driveDetachedRun } = await import('./wasmjit-module.js');
  const { foreignMemory } = await import('./foreign-mem.js');
  const eng = await par.loadEngine();
  const ex = eng.ex, memory = eng.memory;
  const u8 = () => new Uint8Array(memory.buffer);
  const readOut = () => u8().slice(Number(ex.temen_stdout_ptr()), Number(ex.temen_stdout_ptr()) + ex.temen_stdout_len());
  const enc = new TextEncoder();
  const fetchBytes = async (u) => new Uint8Array(await (await fetch(u)).arrayBuffer());

  const hexerCe = await fetchBytes('./assets/hexer_ce_test.temen');
  const snif = await fetchBytes('./assets/hexer_snif_test.nif');
  const sidx = await fetchBytes('./assets/hexer_sidx_test.nif');
  const expected = await fetchBytes('./assets/hexer_xnif_test.nif');

  // Pack helpers for the phase-agnostic FFI: strings `[count][len,bytes]…`, files `[count][nlen,name,dlen,data]…` (u32 LE).
  const u32 = (n) => { const b = new Uint8Array(4); new DataView(b.buffer).setUint32(0, n, true); return b; };
  const cat = (arrs) => { const out = new Uint8Array(arrs.reduce((a, x) => a + x.length, 0)); let o = 0; for (const x of arrs) { out.set(x, o); o += x.length; } return out; };
  const packStrs = (ss) => cat([u32(ss.length), ...ss.flatMap((s) => { const b = enc.encode(s); return [u32(b.length), b]; })]);
  const packFiles = (fs) => cat([u32(fs.length), ...fs.flatMap(([n, d]) => { const nb = enc.encode(n); return [u32(nb.length), nb, u32(d.length), d]; })]);

  const argv = packStrs(['hexer', 'c', 'nimcache/sysvq0asl.s.nif']);
  const seeds = packFiles([['nimcache/sysvq0asl.s.nif', snif], ['nimcache/sysvq0asl.s.idx.nif', sidx]]);
  const outPath = enc.encode('nimcache/sysvq0asl.x.nif');

  const push = (bytes) => { const p = Number(ex.temen_alloc(bytes.length)); u8().set(bytes, p); return p; };
  const cp = push(hexerCe), ap = push(argv), sp = push(seeds), op = push(outPath);
  // carve_log2 = 28: hexer's no-GC system lowering peaks ~256 MiB (a small declared window would undersize it).
  const opened = ex.temen_op13jit_phase_open_argv(cp, hexerCe.length, ap, argv.length, sp, seeds.length, op, outPath.length);
  ex.temen_dealloc(cp, hexerCe.length); ex.temen_dealloc(ap, argv.length); ex.temen_dealloc(sp, seeds.length); ex.temen_dealloc(op, outPath.length);
  if (opened !== 0) return { err: `phase_open_argv failed: ${opened}` };

  let steps = 0, drove = 0;
  for (;;) {
    if (steps++ > 8) { ex.temen_op13jit_close(); return { err: 'loop did not terminate' }; }
    const s = ex.temen_op13jit_step();
    if (s === 0) break;
    if (s === 2) {                  // CHILD_DETACHED (#1288): hexer_ce runs in its own minted Memory
      try { await driveDetachedRun(ex, memory, foreignMemory(ex.temen_op13jit_child_mem_id()), 'op13jit-hexer'); }
      catch (e) { ex.temen_op13jit_close(); return { err: `driveDetachedRun: ${String(e && e.message || e)}` }; }
      ex.temen_op13jit_deliver(); drove++; continue;
    }
    ex.temen_op13jit_close();
    return { err: `trap at step ${s}` };
  }
  const result = Number(ex.temen_op13jit_result());
  const plen = ex.temen_op13jit_phase_output();
  const emitted = readOut().slice(0, plen);
  ex.temen_op13jit_close();

  const eq = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);
  return {
    result, drove,
    expectedLen: expected.length, emittedLen: emitted.length,
    xnifEq: eq(expected, emitted),
    head: new TextDecoder().decode(expected.slice(0, 60)),
  };
});

await browser.close(); server.close();
for (const f of [CE_TMP, SNIF_TMP, SIDX_TMP, XNIF_TMP]) { try { rmSync(f); } catch {} }
console.log('RESULT', JSON.stringify(res, null, 2));
if (errors.length) console.log('ERRORS', errors.slice(0, 6));
const ok = !res.err && res.xnifEq && res.emittedLen > 0 && res.expectedLen > 0;
console.log(`  op13jit-hexer: .x.nif≡=${res.xnifEq} (emitted ${res.emittedLen}B / expected ${res.expectedLen}B) driver=${res.result} childrenDriven=${res.drove}${res.err ? ` · ERR ${res.err}` : ''}`);
console.log(ok ? 'PASS — real hexer_ce ran DETACHED on the EMITTED tier in its own WebAssembly.Memory; .x.nif ≡ committed expected' : 'FAIL');
process.exit(ok ? 0 : 1);
