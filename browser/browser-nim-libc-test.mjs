// Real-browser gate for the nim **runtime providers** (#1422). A nim program that formats a float
// (`std/strutils`' `formatBiggestFloat`), parses one (`std/parseutils` → `strtod`), or calls a libm
// transcendental (`std/math`) compiles fine but could not *run*: those bottom-edge leaves had no
// implementation, so the linked module carried unbound manifest imports and refused to start. The
// nim->powerbox link now binds them against the same prebuilt guest-C libc the chibicc card uses
// (`assets/pg_libc.temeno`), which the host seeds through `temen_nim_libc_put`.
//
// This drives the **production FFI path** in a real browser: seed the libc exactly as `play.js` /
// `snapshot-worker.js` do, compile+run a program through `temen_compile_nim_fs`, and check the bytes
// against the native-nimony oracle. A regression (libc unseeded, leaf unbound, printf `*` support
// lost) shows up here as wrong output or a refused start, not as a silent slow path.
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
if (!existsSync(`${ROOT}/web/assets/pg_libc.temeno`)) {
  console.log('SKIP: pg_libc.temeno absent'); process.exit(0);
}
const chromium = await loadChromium();
const { server, port } = await startServer(ROOT);
const browser = await chromium.launch({ args: process.env.CI ? ['--no-sandbox'] : [] });
const page = await browser.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(String(e)));
await page.goto(`http://127.0.0.1:${port}/web/play.html`, { waitUntil: 'load' });

const res = await page.evaluate(async () => {
  const par = await import('./par.js');
  const gunzip = async (u) => new Uint8Array(await new Response(new Blob([u]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
  const fetchGz = async (p) => gunzip(new Uint8Array(await (await fetch(p)).arrayBuffer()));
  const [nifler, nimsem, hexer, stdlib] = await Promise.all([
    fetchGz('./assets/nifler.temen.gz'), fetchGz('./assets/nimsem.temen.gz'),
    fetchGz('./assets/hexer.temen.gz'), fetchGz('./assets/nim_stdlib.img.gz')]);
  const libc = new Uint8Array(await (await fetch('./assets/pg_libc.temeno')).arrayBuffer());
  const eng = await par.loadEngine();
  const ex = eng.ex, memory = eng.memory;
  const u8 = () => new Uint8Array(memory.buffer);
  const push = (b) => { const p = Number(ex.temen_alloc(b.length)); u8().set(b, p); return p; };
  // Seed the guest libc through the production FFI (as play.js / snapshot-worker.js do).
  const lp = push(libc); ex.temen_nim_libc_put(lp, libc.length); ex.temen_dealloc(lp, libc.length);
  // `%#.*f` / `%#.*e` float formatting — `formatBiggestFloat` -> the guest libc's `snprintf`. (The
  // libm transcendentals ride the same binding and are covered natively by
  // `temen-leng`'s `real_math_transcendentals_run`; `std/math` is left out here because this
  // single-shot interpreter card path has its own limit parsing it, unrelated to #1422.)
  const src = new TextEncoder().encode(
    'import std/syncio\nimport std/strutils\n' +
    'write(stdout, formatFloat(3.14159, ffDecimal, 3))\n' +
    'write(stdout, "|")\n' +
    'write(stdout, formatFloat(2.5, ffScientific, 2))\n');
  const main = new TextEncoder().encode('prog.nim');
  const np = push(nifler), smp = push(nimsem), hp = push(hexer), ip = push(stdlib), sp = push(src), mp = push(main);
  ex.temen_compile_nim_fs(np, nifler.length, smp, nimsem.length, hp, hexer.length,
    ip, stdlib.length, sp, src.length, mp, main.length);
  const status = ex.temen_status();
  const dec = new TextDecoder();
  const out = dec.decode(u8().slice(Number(ex.temen_stdout_ptr()), Number(ex.temen_stdout_ptr()) + ex.temen_stdout_len()));
  const err = dec.decode(u8().slice(Number(ex.temen_stderr_ptr()), Number(ex.temen_stderr_ptr()) + ex.temen_stderr_len()));
  return { status, out, err: err.slice(-400) };
});
await browser.close(); server.close();
if (errors.length) console.log('ERRORS', errors.slice(0, 6));
// The native-nimony oracle for the same program.
const WANT = '3.142|2.50e+00';
const ok = res.status === 0 && res.out === WANT;
console.log(`  nim-libc: status=${res.status} out=${JSON.stringify(res.out)} want=${JSON.stringify(WANT)}`);
if (!ok && res.err) console.log(`  stderr tail: ${res.err}`);
console.log(ok ? 'PASS — nim float formatting runs in the browser against the guest libc'
               : 'FAIL — the guest libc leaves are unbound or miscompiled');
process.exit(ok ? 0 : 1);
