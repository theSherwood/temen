// Real-browser (V8) end-to-end check for the **playground bash card** (#1080): load the same
// `temen_browser.wasm` the page runs, fetch `web/assets/bash.temen` + the /bin coreutils, and drive
// the `temen_run_bash` entry the card's Run calls — running a script as `bash -c` and asserting the
// captured stdout. This exercises the actual wasm path the "bash" card uses: **real GNU bash** on
// the bytecode cooperative engine under the temen-posix personality — fork per pipeline stage,
// `execve` image-replace per external command, CorePipe reads parking + EOF, blocking `waitpid`
// reaps — all inside Chromium.
//
// `bash.temen` is a **deploy-built asset** (GPLv3, never committed): the test SKIPs cleanly when it
// (or a coreutil, or a Chromium/Playwright install) is absent — run `node build-bash-assets.mjs`
// first to make it actually run. Run: node browser-bash-test.mjs
import { startServer } from './serve.mjs';
import { benignAssetMiss } from './play-test-errors.mjs';
import { fileURLToPath } from 'node:url';
import { dirname } from 'node:path';
import { existsSync } from 'node:fs';

const ROOT = dirname(fileURLToPath(import.meta.url));
const BINS = ['echo', 'cat', 'seq', 'head', 'sort', 'uniq'];
const ASSETS = ['bash.temen', ...BINS.map((b) => `bin_${b}.temen`)];
if (ASSETS.some((a) => !existsSync(`${ROOT}/web/assets/${a}`))) {
  console.log('– bash test skipped (web/assets/bash.temen or a bin_*.temen absent — run build-bash-assets.mjs)');
  process.exit(0);
}
async function loadChromium() {
  for (const s of ['playwright', '/opt/node22/lib/node_modules/playwright/index.js']) {
    try { const m = await import(s); return m.chromium ?? m.default?.chromium; } catch {}
  }
  throw new Error('playwright not found');
}
let chromium;
try { chromium = await loadChromium(); }
catch { console.log('– bash test skipped (playwright not found)'); process.exit(0); }

const { server, port } = await startServer(ROOT);
const browser = await chromium.launch({ args: process.env.CI ? ['--no-sandbox'] : [] });
const page = await browser.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(String(e)));
page.on('console', (m) => { if (m.type() === 'error' && !benignAssetMiss(m)) errors.push(m.text()); });
await page.goto(`http://127.0.0.1:${port}/web/play.html`);

// The script exercises bash's own expansion ($VAR, $((…))), a fork-twin external command mid-list
// (`seq 2; echo done` — fork → execve → blocking waitpid), a two-stage external pipeline
// (`echo … | cat` — CorePipe through exec), the three-stage `seq | head` shape, a builtin-fed
// `sort | uniq` pipeline, and a `for` loop — the #1080 surface the card advertises.
const SCRIPT =
  'echo hi from bash\n' +
  'N=w0rld\n' +
  'echo greet $N\n' +
  'echo $((6 * 7))\n' +
  'seq 2; echo done\n' +
  'echo piped | cat\n' +
  'seq 5 | head -n 3\n' +
  "printf 'b\\na\\nb\\n' | sort | uniq\n" +
  'for i in 1 2; do echo loop $i; done\n';
const EXPECT =
  'hi from bash\ngreet w0rld\n42\n1\n2\ndone\npiped\n1\n2\n3\na\nb\nloop 1\nloop 2\n';

const res = await page.evaluate(async ({ script, bins }) => {
  const par = await import('./par.js');
  const eng = await par.loadEngine();
  const dec = (p, n) => new TextDecoder().decode(new Uint8Array(eng.memory.buffer).slice(p, p + n));
  const fetchB = async (u) => new Uint8Array(await (await fetch(u)).arrayBuffer());
  const bytes = await fetchB('./assets/bash.temen');
  // Build the /bin registry blob temen_run_bash parses: u32 count, then per entry u32 name-len +
  // UTF-8 name + u32 module-len + module bytes (little-endian) — full paths as names.
  const reg = [];
  for (const b of bins) {
    reg.push({ name: new TextEncoder().encode(`/bin/${b}`), bytes: await fetchB(`./assets/bin_${b}.temen`) });
  }
  let total = 4;
  for (const c of reg) total += 4 + c.name.length + 4 + c.bytes.length;
  const blob = new Uint8Array(total);
  const dv = new DataView(blob.buffer);
  let o = 0; dv.setUint32(o, reg.length, true); o += 4;
  for (const c of reg) {
    dv.setUint32(o, c.name.length, true); o += 4; blob.set(c.name, o); o += c.name.length;
    dv.setUint32(o, c.bytes.length, true); o += 4; blob.set(c.bytes, o); o += c.bytes.length;
  }
  const cmdBytes = new TextEncoder().encode(script);
  const mp = eng.ex.temen_alloc(bytes.length); new Uint8Array(eng.memory.buffer).set(bytes, mp);
  const cp = eng.ex.temen_alloc(cmdBytes.length); new Uint8Array(eng.memory.buffer).set(cmdBytes, cp);
  const bp = eng.ex.temen_alloc(blob.length); new Uint8Array(eng.memory.buffer).set(blob, bp);
  eng.ex.temen_run_bash(mp, bytes.length, cp, cmdBytes.length, 0, 0, bp, blob.length);
  const status = eng.ex.temen_status();
  const exit = eng.ex.temen_exit_code();
  const stdout = dec(Number(eng.ex.temen_stdout_ptr()), eng.ex.temen_stdout_len());
  eng.ex.temen_dealloc(mp, bytes.length); eng.ex.temen_dealloc(cp, cmdBytes.length);
  eng.ex.temen_dealloc(bp, blob.length);
  return { status, exit, stdout };
}, { script: SCRIPT, bins: BINS });

await browser.close();
await new Promise((r) => server.close(r));

let ok = true;
if (errors.length) { console.error('page errors:', errors); ok = false; }
// 0 = OK, 5 = clean Exit (bash's exit_shell after the final builtin); exit code must be 0 either way.
if (res.status !== 0 && res.status !== 5) { console.error(`bash status ${res.status} (expected 0 or 5)`); ok = false; }
if (res.exit !== 0) { console.error(`bash exit code ${res.exit} (expected 0)`); ok = false; }
if (res.stdout !== EXPECT) {
  console.error(`stdout mismatch:\n  got:      ${JSON.stringify(res.stdout)}\n  expected: ${JSON.stringify(EXPECT)}`);
  ok = false;
}
if (ok) { console.log(`✓ bash card: temen_run_bash produced the expected output (${res.stdout.length}B, status ${res.status})`); process.exit(0); }
process.exit(1);
