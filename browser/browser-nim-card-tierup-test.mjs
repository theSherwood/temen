// Real-browser (V8) end-to-end for the **shipped nim card** (#958): drives the playground's own
// `snapshotClient.nimCompile` — the exact path play.js's nim card takes — with the committed toolchain
// (`web/assets/nimony.blob.gz`). The nim worker runs `nimony t --isMain prog.nim` and then the program it
// built, in one POSIX process tree (`nimCompileRun`). Asserts the program prints what native nimony's
// build of it prints, that the toolchain's leaf processes ran on emitted wasm rather than falling back to
// the interpreter, and that a program that does not compile reports nimony's diagnostic and runs nothing.
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
page.on('pageerror', (e) => errors.push(String(e)));
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
await page.goto(`http://127.0.0.1:${port}/web/play.html`, { waitUntil: 'load' });
// The snapshot client is created asynchronously after the engine loads (and only under cross-origin
// isolation); wait for the page to expose it before driving a compile.
await page.waitForFunction(() => !!globalThis.__snapshotClient, null, { timeout: 30000 })
  .catch(() => {});

const res = await page.evaluate(async () => {
  const client = globalThis.__snapshotClient;
  if (!client) return { err: 'snapshotClient not exposed (page lacks cross-origin isolation?)' };
  const gunzip = async (u) => new Uint8Array(await new Response(new Blob([u]).stream()
    .pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
  // The card's asset, as play.js's `runNimc` fetches it.
  const getAssets = async () => ({ bundle: await gunzip(new Uint8Array(await (await fetch('./assets/nimony.blob.gz')).arrayBuffer())) });
  // Uses three modules of the prebuilt library, and a float, which `formatFloat` prints through the
  // guest libc's `snprintf`.
  const source = 'import std/[syncio, strutils, tables, math]\n\n'
    + 'proc greet(name: string): string =\n  "hello, " & name & "\\n"\n\n'
    + 'var t = initTable[string, int]()\nt["k"] = 7\n'
    + 'write(stdout, greet("Nim"))\nwrite(stdout, greet("the Temen"))\n'
    + 'write(stdout, toUpperAscii("ab") & "|" & formatFloat(sqrt(4.0), ffDecimal, 1) & "|" & $t.len & "\\n")\n';
  const t0 = performance.now();
  const ran = await client.nimCompile(getAssets, source);
  const secs = (performance.now() - t0) / 1000;
  const failed = await client.nimCompile(getAssets, 'let x: int = "nope"\necho x\n');
  return { ran, failed, secs };
});

await browser.close(); server.close();
if (errors.length) console.log('ERRORS', errors.slice(0, 6));
if (res.err) { console.log(`FAIL: ${res.err}`); process.exit(1); }
const { ran, failed } = res;
// What native nimony's build of the program prints.
const want = 'hello, Nim\nhello, the Temen\nAB|2.0|1\n';
if (!ran.ok || !failed.ok) console.log(`  errors: ${ran.error} / ${failed.error}`);
console.log(`  ran:    ok ${ran.ok} · status ${ran.status} exit ${ran.exit} · built ${ran.built} · ` +
  `${ran.leaves} leaf processes emitted, ${ran.resumes} parked calls resumed · ${res.secs.toFixed(1)} s`);
console.log(`          stdout ${JSON.stringify(ran.stdout)}${ran.stderr ? ` · stderr ${JSON.stringify(ran.stderr.slice(0, 400))}` : ''}`);
console.log(`  failed: ok ${failed.ok} · exit ${failed.exit} · built ${failed.built} · stdout ${JSON.stringify((failed.stdout || '').slice(0, 200))}`);
const ok = ran.ok && ran.built && ran.exit === 0 && ran.stdout === want
  // The program and each of the toolchain's parsing and lowering runs are leaf processes.
  && ran.leaves >= 3
  && failed.ok && !failed.built && failed.exit !== 0
  && /prog\.nim\(1, \d+\) Error: type mismatch/.test(failed.stdout);
console.log(ok ? 'PASS — the shipped nim card compiled and ran on nimony\'s own driver, leaves on emitted wasm' : 'FAIL');
process.exit(ok ? 0 : 1);
