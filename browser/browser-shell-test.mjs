// Real-browser (V8) end-to-end check for the **playground shell card** (STAGE1.md): load the same
// `svm_browser.wasm` the page runs, fetch `web/assets/shell.svmb` (the committed shell fixture the
// build step copies in), and drive the `svm_run_shell` entry the card's Run calls — feeding a script
// as stdin and asserting the captured stdout. This exercises the actual wasm path the "Shell" card
// uses (interpreter tier; the shell carries Instantiator cap.calls, so there is no wasm-JIT tier).
//
// Skipped cleanly if the asset or a Chromium/Playwright install is unavailable (like the other
// on-ramp browser checks). Run: node browser-shell-test.mjs
import { startServer } from './serve.mjs';
import { fileURLToPath } from 'node:url';
import { dirname } from 'node:path';
import { existsSync } from 'node:fs';

const ROOT = dirname(fileURLToPath(import.meta.url));
if (!existsSync(`${ROOT}/web/assets/shell.svmb`)) {
  console.log('– shell test skipped (web/assets/shell.svmb absent — run build-onramp-assets.mjs)');
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
// `cat | grep` pipeline, and an if/test conditional — the Stage-0 shell surface the card advertises.
const SCRIPT =
  'echo hello from the sandbox\n' +
  'N=world\n' +
  'echo hi $N\n' +
  'echo apple > fruits\n' +
  'echo banana >> fruits\n' +
  'cat fruits | grep a\n' +
  'if test -f fruits; then echo exists; fi\n';
const EXPECT = 'hello from the sandbox\nhi world\napple\nbanana\nexists\n';

const res = await page.evaluate(async (script) => {
  const par = await import('./par.js');
  const eng = await par.loadEngine();
  const dec = (p, n) => new TextDecoder().decode(new Uint8Array(eng.memory.buffer).slice(p, p + n));
  const bytes = new Uint8Array(await (await fetch('./assets/shell.svmb')).arrayBuffer());
  const stdinBytes = new TextEncoder().encode(script);
  const mp = eng.ex.svm_alloc(bytes.length); new Uint8Array(eng.memory.buffer).set(bytes, mp);
  const sp = eng.ex.svm_alloc(stdinBytes.length); new Uint8Array(eng.memory.buffer).set(stdinBytes, sp);
  eng.ex.svm_run_shell(mp, bytes.length, sp, stdinBytes.length);
  const status = eng.ex.svm_status();
  const stdout = dec(Number(eng.ex.svm_stdout_ptr()), eng.ex.svm_stdout_len());
  eng.ex.svm_dealloc(mp, bytes.length); eng.ex.svm_dealloc(sp, stdinBytes.length);
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
