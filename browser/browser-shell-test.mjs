// Real-browser (V8) end-to-end check for the **playground shell card** (STAGE1.md): load the same
// `svm_browser.wasm` the page runs, fetch `web/assets/shell.svmb` + its PATH registry
// (`stage_runner.svmb`, `primes.svmb`), and drive the `svm_run_shell` entry the card's Run calls —
// feeding a script as stdin and asserting the captured stdout. This exercises the actual wasm path the
// "Shell" card uses: the bytecode cooperative engine, with the `__stage` runner registered so a
// `cat | sort | uniq` pipeline runs as concurrent stages over shared-memory rings (op 11 +
// SharedRegion + futex), and the `primes` external command exec'd as an op-13 §14 child.
//
// Skipped cleanly if the assets or a Chromium/Playwright install is unavailable (like the other
// on-ramp browser checks). Run: node browser-shell-test.mjs
import { startServer } from './serve.mjs';
import { fileURLToPath } from 'node:url';
import { dirname } from 'node:path';
import { existsSync } from 'node:fs';

const ROOT = dirname(fileURLToPath(import.meta.url));
const ASSETS = ['shell.svmb', 'stage_runner.svmb', 'primes.svmb', 'upper.svmb'];
if (ASSETS.some((a) => !existsSync(`${ROOT}/web/assets/${a}`))) {
  console.log(`– shell test skipped (web/assets/{${ASSETS.join(',')}} absent — run build-onramp-assets.mjs)`);
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
catch { console.log('– shell test skipped (playwright not found)'); process.exit(0); }

const { server, port } = await startServer(ROOT);
const browser = await chromium.launch({ args: process.env.CI ? ['--no-sandbox'] : [] });
const page = await browser.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(String(e)));
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
await page.goto(`http://127.0.0.1:${port}/web/play.html`);

// The script exercises the read-eval loop, $VAR expansion, a redirect+cat over the memfs, a
// `cat | grep` pipeline, a **`cat | sort | uniq` concurrent ring pipeline**, the **`primes` external
// command** (a generator), the **`upper` filter** (an external command that reads stdin), and an
// if/test conditional — the shell surface the card advertises.
const SCRIPT =
  '# a comment line — ignored\n' +
  'echo hello from the sandbox\n' +
  'N=world\n' +
  'echo hi $N   # an inline comment\n' +
  'echo -- primes --\n' +
  'primes 10\n' +
  'echo -- upper --\n' +
  'echo shout it | upper\n' +
  'echo banana > fruits\n' +
  'echo apple >> fruits\n' +
  'echo banana >> fruits\n' +
  'cat fruits | grep a\n' +
  'echo -- sorted --\n' +
  'cat fruits | sort | uniq\n' +
  'if test -f fruits; then echo exists; fi\n';
// `primes 10` → the generator streams 2 3 5 7; `echo shout it | upper` → the filter reads its stdin
// and uppercases it; `grep a` matches all three lines (banana, apple, banana all contain 'a'); the
// ring `sort | uniq` then yields the deduped sorted pair.
const EXPECT =
  'hello from the sandbox\nhi world\n-- primes --\n2\n3\n5\n7\n-- upper --\nSHOUT IT\n' +
  'banana\napple\nbanana\n-- sorted --\napple\nbanana\nexists\n';

const res = await page.evaluate(async (script) => {
  const par = await import('./par.js');
  const eng = await par.loadEngine();
  const dec = (p, n) => new TextDecoder().decode(new Uint8Array(eng.memory.buffer).slice(p, p + n));
  const fetchB = async (u) => new Uint8Array(await (await fetch(u)).arrayBuffer());
  const bytes = await fetchB('./assets/shell.svmb');
  // Build the PATH-registry blob svm_run_shell parses: u32 count, then per entry u32 name-len + UTF-8
  // name + u32 module-len + module bytes (little-endian) — the __stage runner and the primes command.
  const reg = [
    { name: '__stage', bytes: await fetchB('./assets/stage_runner.svmb') },
    { name: 'primes', bytes: await fetchB('./assets/primes.svmb') },
    { name: 'upper', bytes: await fetchB('./assets/upper.svmb') },
  ].map((c) => ({ name: new TextEncoder().encode(c.name), bytes: c.bytes }));
  let total = 4;
  for (const c of reg) total += 4 + c.name.length + 4 + c.bytes.length;
  const blob = new Uint8Array(total);
  const dv = new DataView(blob.buffer);
  let o = 0; dv.setUint32(o, reg.length, true); o += 4;
  for (const c of reg) {
    dv.setUint32(o, c.name.length, true); o += 4; blob.set(c.name, o); o += c.name.length;
    dv.setUint32(o, c.bytes.length, true); o += 4; blob.set(c.bytes, o); o += c.bytes.length;
  }
  const stdinBytes = new TextEncoder().encode(script);
  const mp = eng.ex.svm_alloc(bytes.length); new Uint8Array(eng.memory.buffer).set(bytes, mp);
  const sp = eng.ex.svm_alloc(stdinBytes.length); new Uint8Array(eng.memory.buffer).set(stdinBytes, sp);
  const gp = eng.ex.svm_alloc(blob.length); new Uint8Array(eng.memory.buffer).set(blob, gp);
  eng.ex.svm_run_shell(mp, bytes.length, sp, stdinBytes.length, gp, blob.length);
  const status = eng.ex.svm_status();
  const stdout = dec(Number(eng.ex.svm_stdout_ptr()), eng.ex.svm_stdout_len());
  eng.ex.svm_dealloc(mp, bytes.length); eng.ex.svm_dealloc(sp, stdinBytes.length);
  eng.ex.svm_dealloc(gp, blob.length);
  return { status, stdout };
}, SCRIPT);

await browser.close();
await new Promise((r) => server.close(r));

let ok = true;
if (errors.length) { console.error('page errors:', errors); ok = false; }
if (res.status !== 0) { console.error(`shell status ${res.status} (expected 0)`); ok = false; }
if (res.stdout !== EXPECT) {
  console.error(`stdout mismatch:\n  got:      ${JSON.stringify(res.stdout)}\n  expected: ${JSON.stringify(EXPECT)}`);
  ok = false;
}
if (ok) { console.log(`✓ shell card: svm_run_shell produced the expected output (${res.stdout.length}B)`); process.exit(0); }
process.exit(1);
