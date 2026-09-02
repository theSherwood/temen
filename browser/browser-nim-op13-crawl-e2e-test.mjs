// Real-browser (V8) end-to-end: **drive the whole nim card through the op-13 loop** (#1025 Path 1). The
// nim compile's phase-1 nifler crawl runs each module as a §14 op-13 **nested** child (nifler_ce) on the
// emitted-wasm tier — a resumable driver marshals {fs,stdout,exit} across the op-13 bounce, the child reads
// the module source from the marshaled memfs and writes `.p.nif`, seeded into the accumulator
// `temen_compile_nim_fs` mounts. Then nimsem/hexer/link/run finish the compile. The card's output must be
// byte-identical whether phase-1 ran on the interpreter or as nested op-13 emitted children.
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
for (const a of ['nifler', 'nimsem', 'hexer']) {
  if (!existsSync(`${ROOT}/web/assets/${a}.temen.gz`)) { console.log(`SKIP: web/assets/${a}.temen.gz absent`); process.exit(0); }
}
if (!existsSync(`${ROOT}/web/assets/nim_stdlib.img.gz`) || !existsSync(CE_GZ)) { console.log('SKIP: stdlib / nifler_ce absent'); process.exit(0); }
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
  const { jitNimCrawlOp13 } = await import('./wasmjit-module.js');
  const eng = await par.loadEngine();
  const ex = eng.ex, memory = eng.memory;
  const u8 = () => new Uint8Array(memory.buffer);
  const gunzip = async (u) => new Uint8Array(await new Response(new Blob([u]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
  const fetchGz = async (p) => gunzip(new Uint8Array(await (await fetch(p)).arrayBuffer()));
  const readOut = () => u8().slice(Number(ex.temen_stdout_ptr()), Number(ex.temen_stdout_ptr()) + ex.temen_stdout_len());
  const readErr = () => u8().slice(Number(ex.temen_stderr_ptr()), Number(ex.temen_stderr_ptr()) + ex.temen_stderr_len());

  const [nifler, nimsem, hexer, stdlib] = await Promise.all([
    fetchGz('./assets/nifler.temen.gz'), fetchGz('./assets/nimsem.temen.gz'),
    fetchGz('./assets/hexer.temen.gz'), fetchGz('./assets/nim_stdlib.img.gz')]);
  const niflerCe = new Uint8Array(await (await fetch('./assets/nifler_ce_test.temen')).arrayBuffer());

  const source = new TextEncoder().encode('import std/syncio\n\nproc greet(name: string): string =\n  "hello, " & name & "\\n"\n\nwrite(stdout, greet("Nim"))\nwrite(stdout, greet("the Temen"))\n');
  const main = new TextEncoder().encode('prog.nim');

  const compile = async (op13CrawlFirst) => {
    let crawled = 0;
    if (op13CrawlFirst) {
      const r = await jitNimCrawlOp13(ex, memory, niflerCe, stdlib, '/prog.nim', source, 'nim-op13-nifler');
      crawled = r.crawled;
    } else {
      ex.temen_nim_precrawl_reset();
    }
    const np = Number(ex.temen_alloc(nifler.length));
    const smp = Number(ex.temen_alloc(nimsem.length));
    const hp = Number(ex.temen_alloc(hexer.length));
    const ip = Number(ex.temen_alloc(stdlib.length));
    const sp = Number(ex.temen_alloc(source.length));
    const mp = Number(ex.temen_alloc(main.length));
    { const v = u8(); v.set(nifler, np); v.set(nimsem, smp); v.set(hexer, hp); v.set(stdlib, ip); v.set(source, sp); v.set(main, mp); }
    ex.temen_compile_nim_fs(np, nifler.length, smp, nimsem.length, hp, hexer.length, ip, stdlib.length, sp, source.length, mp, main.length);
    const status = ex.temen_status();
    const out = readOut(), err = readErr();
    ex.temen_dealloc(np, nifler.length); ex.temen_dealloc(smp, nimsem.length); ex.temen_dealloc(hp, hexer.length);
    ex.temen_dealloc(ip, stdlib.length); ex.temen_dealloc(sp, source.length); ex.temen_dealloc(mp, main.length);
    return { status, out, err, crawled };
  };

  const base = await compile(false);   // interpreter phase-1
  const jit = await compile(true);     // op-13 nested emitted phase-1
  const dec = new TextDecoder();
  const eq = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);
  return {
    baseStatus: base.status, jitStatus: jit.status,
    baseOut: dec.decode(base.out), jitOut: dec.decode(jit.out),
    jitErr: dec.decode(jit.err).slice(0, 200),
    outEq: eq(base.out, jit.out), crawled: jit.crawled,
  };
});

await browser.close(); server.close();
try { rmSync(CE_TMP); } catch {}
console.log('RESULT', JSON.stringify(res, null, 2));
if (errors.length) console.log('ERRORS', errors.slice(0, 8));
const ok = res.baseStatus === 0 && res.jitStatus === 0 && res.outEq && res.crawled > 0 && res.baseOut.includes('hello, Nim');
console.log(`  nim-op13-crawl-e2e: status ${res.baseStatus}/${res.jitStatus} · out≡=${res.outEq} · op13-crawled=${res.crawled} modules`);
console.log(`  program stdout: ${JSON.stringify(res.jitOut.slice(0, 80))}`);
console.log(ok ? 'PASS — whole nim card compiled with phase-1 as nested op-13 emitted children ≡ interpreter' : 'FAIL');
process.exit(ok ? 0 : 1);
