// Real-browser (V8) end-to-end for the **JS-orchestrated op-13 loop on a REAL nimsem phase child**
// (#1025 3a.3 — the heaviest front-end phase, extending the nifler/hexer tier-up to the 4-cap `exec`
// phase via `temen_op13jit_nimsem_open`): a resumable driver marshals {fs, stdout, exit, **exec**} to
// nimsem_ce (the child-entry phase). The `exec` cap is `make_exec` over the SAME shared memfs, running
// the top-level nifler on the interpreter — so nimsem-the-tiered-up-child shells out to nifler
// grandchildren (host-side) to parse the stdlib it imports, while nimsem's own sema (the dominant cost)
// runs on **emitted wasm** over its 256 MiB carve. It semchecks the system module (`--isSystem`) and
// writes `sysvq0asl.s.nif`, which must be byte-identical to the committed expected — the same oracle the
// headless `rust_driver_nimsem.rs` gate uses. This is `nimc.rs`'s phase-2 nimsem, nested under an op-13
// driver, tiered up to JIT in the browser — the dominant piece of the ~180s card.
import { startServer } from './serve.mjs';
import { fileURLToPath } from 'node:url';
import { dirname } from 'node:path';
import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync, writeFileSync, rmSync, mkdtempSync, readdirSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, relative, sep } from 'node:path';
import { gunzipSync } from 'node:zlib';
const ROOT = dirname(fileURLToPath(import.meta.url));
async function loadChromium() {
  for (const s of ['playwright', '/opt/node22/lib/node_modules/playwright/index.js']) {
    try { const m = await import(s); return m.chromium ?? m.default?.chromium; } catch {}
  }
  throw new Error('playwright not found');
}
const FX = `${ROOT}/../crates/temen-run/demos/nim_frontend/fixtures`;
const CE_GZ = `${FX}/nimsem_ce.temen.gz`;
const NIFLER_GZ = `${ROOT}/web/assets/nifler.temen.gz`;
const SYSLIB_GZ = `${FX}/syslib.tar.gz`;
const PNIF = `${FX}/sysvq0asl.p.nif`;
const SNIF_GZ = `${FX}/sysvq0asl.s.nif.gz`;
if (![CE_GZ, NIFLER_GZ, SYSLIB_GZ, PNIF, SNIF_GZ].every(existsSync) ||
    !existsSync(`${ROOT}/target/wasm32-unknown-unknown/release/temen_browser.wasm`)) {
  console.log('SKIP: nimsem_ce / nifler / syslib.tar.gz / .p.nif / expected asset or threads wasm absent'); process.exit(0);
}

// Pack helpers (node side) for the phase-agnostic FFI: files `[count][nlen,name,dlen,data]…` (u32 LE).
const u32 = (n) => { const b = new Uint8Array(4); new DataView(b.buffer).setUint32(0, n, true); return b; };
const cat = (arrs) => { const out = new Uint8Array(arrs.reduce((a, x) => a + x.length, 0)); let o = 0; for (const x of arrs) { out.set(x, o); o += x.length; } return out; };
const enc = new TextEncoder();
const packFiles = (fs) => cat([u32(fs.length), ...fs.flatMap(([n, d]) => { const nb = enc.encode(n); return [u32(nb.length), nb, u32(d.length), d]; })]);

// Unpack syslib.tar.gz (the 26-file stdlib import closure) into a temp dir, then seed both `lib/std/…`
// (as packed) and flattened `lib/…` — exactly as nimc/rust_driver_nimsem seeds the shared memfs.
const work = mkdtempSync(join(tmpdir(), 'nimsem-seeds-'));
try {
  const libdir = join(work, 'lib');
  execFileSync('mkdir', ['-p', libdir]);
  execFileSync('tar', ['xzf', SYSLIB_GZ, '-C', libdir]);
  const walk = (d, out) => { for (const e of readdirSync(d)) { const p = join(d, e); if (statSync(p).isDirectory()) walk(p, out); else out.push(p); } };
  const paths = []; walk(libdir, paths);
  const files = [];
  for (const p of paths) {
    const rel = relative(libdir, p).split(sep).join('/');
    files.push([`lib/${rel}`, new Uint8Array(readFileSync(p))]);
  }
  const flat = files
    .filter(([k]) => k.startsWith('lib/std/'))
    .map(([k, v]) => [`lib/${k.slice('lib/std/'.length)}`, v]);
  files.push(...flat);
  files.push(['nimcache/sysvq0asl.p.nif', new Uint8Array(readFileSync(PNIF))]);
  const seedsBlob = packFiles(files);

  // Stage the (gunzipped) inputs, prepacked seeds, and expected output as temp served assets.
  const CE_TMP = `${ROOT}/web/assets/nimsem_ce_test.temen`;
  const NIFLER_TMP = `${ROOT}/web/assets/nimsem_nifler_test.temen`;
  const SEEDS_TMP = `${ROOT}/web/assets/nimsem_seeds_test.bin`;
  const SNIF_TMP = `${ROOT}/web/assets/nimsem_snif_test.nif`;
  writeFileSync(CE_TMP, gunzipSync(readFileSync(CE_GZ)));
  writeFileSync(NIFLER_TMP, gunzipSync(readFileSync(NIFLER_GZ)));
  writeFileSync(SEEDS_TMP, seedsBlob);
  writeFileSync(SNIF_TMP, gunzipSync(readFileSync(SNIF_GZ)));

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
    // #1025 3a.3: nimsem needs a 256 MiB carve → a 512 MiB window, and its 5.7 MB module's cross-tier
    // `SharedProgram` + emitted wasm compile peaks ~700 MiB on top — over the default 1 GiB threads-module
    // ceiling (`web/par.js` `maxPages` == the build's `--max-memory`). Check the ceiling up front (a
    // non-destructive read — pre-*growing* would burn address space dlmalloc can't reclaim and OOM the
    // run itself): SKIP cleanly unless the memory can reach ~1.4 GiB. The byte-exact gate below activates
    // once the ceiling is raised to 2 GiB (the build's `--max-memory=2147483648` + `par.js`'s
    // `maxPages: 32768`). Headless `nim_phase_tierup_eligible.rs` already gates that nimsem *emits*
    // WasmDriven at zero memory cost — this is the end-to-end byte-exact leg.
    if ((eng.maxPages || 16384) < 22528) return { skip: true }; // 22528 pages = 1.4 GiB
    const u8 = () => new Uint8Array(memory.buffer);
    const readOut = () => u8().slice(Number(ex.temen_stdout_ptr()), Number(ex.temen_stdout_ptr()) + ex.temen_stdout_len());
    const enc = new TextEncoder();
    const fetchBytes = async (u) => new Uint8Array(await (await fetch(u)).arrayBuffer());

    const nimsemCe = await fetchBytes('./assets/nimsem_ce_test.temen');
    const nifler = await fetchBytes('./assets/nimsem_nifler_test.temen');
    const seeds = await fetchBytes('./assets/nimsem_seeds_test.bin'); // already in packed `[count][nlen,name,dlen,data]` format
    const expected = await fetchBytes('./assets/nimsem_snif_test.nif');

    const u32 = (n) => { const b = new Uint8Array(4); new DataView(b.buffer).setUint32(0, n, true); return b; };
    const cat = (arrs) => { const out = new Uint8Array(arrs.reduce((a, x) => a + x.length, 0)); let o = 0; for (const x of arrs) { out.set(x, o); o += x.length; } return out; };
    const packStrs = (ss) => cat([u32(ss.length), ...ss.flatMap((s) => { const b = enc.encode(s); return [u32(b.length), b]; })]);

    const argv = packStrs(['nimsem', '--define:nimNativeAlloc', '--define:nimNativeIo', 'm', '--isSystem', 'nimcache/sysvq0asl.p.nif']);
    const outPath = enc.encode('nimcache/sysvq0asl.s.nif');

    const push = (bytes) => { const p = Number(ex.temen_alloc(bytes.length)); u8().set(bytes, p); return p; };
    const cp = push(nimsemCe), np = push(nifler), ap = push(argv), sp = push(seeds), op = push(outPath);
    // carve_log2 = 28: nimsem's no-GC system semcheck peaks ~256 MiB (parent window rounds to 512 MiB).
    const opened = ex.temen_op13jit_nimsem_open(cp, nimsemCe.length, np, nifler.length, ap, argv.length, sp, seeds.length, op, outPath.length, 28);
    ex.temen_dealloc(cp, nimsemCe.length); ex.temen_dealloc(np, nifler.length);
    ex.temen_dealloc(ap, argv.length); ex.temen_dealloc(sp, seeds.length); ex.temen_dealloc(op, outPath.length);
    if (opened !== 0) return { err: `nimsem_open failed: ${opened}` };

    let steps = 0, drove = 0;
    for (;;) {
      if (steps++ > 8) { ex.temen_op13jit_close(); return { err: 'loop did not terminate' }; }
      const s = ex.temen_op13jit_step();
      if (s === 0) break;
      if (s === 1) {
        try { await driveJitRun(ex, memory, 'op13jit-nimsem'); }
        catch (e) { ex.temen_op13jit_close(); return { err: `driveJitRun: ${String(e && e.message || e)}` }; }
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
      snifEq: eq(expected, emitted),
      head: new TextDecoder().decode(expected.slice(0, 60)),
    };
  });

  await browser.close(); server.close();
  for (const f of [CE_TMP, NIFLER_TMP, SEEDS_TMP, SNIF_TMP]) { try { rmSync(f); } catch {} }
  if (res.skip) {
    console.log('SKIP: shared-memory ceiling < 1.4 GiB — nimsem tier-up needs the 2 GiB threads module (see test header)');
    process.exit(0);
  }
  console.log('RESULT', JSON.stringify(res, null, 2));
  if (errors.length) console.log('ERRORS', errors.slice(0, 6));
  const ok = !res.err && res.snifEq && res.emittedLen > 0 && res.expectedLen > 0;
  console.log(`  op13jit-nimsem: .s.nif≡=${res.snifEq} (emitted ${res.emittedLen}B / expected ${res.expectedLen}B) driver=${res.result} childrenDriven=${res.drove}${res.err ? ` · ERR ${res.err}` : ''}`);
  console.log(ok ? 'PASS — real nimsem_ce ran nested on the EMITTED tier (exec→nifler host-side); .s.nif ≡ committed expected' : 'FAIL');
  process.exit(ok ? 0 : 1);
} finally {
  try { rmSync(work, { recursive: true, force: true }); } catch {}
}
