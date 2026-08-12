// Chromium smoke for the **snapshot worker** (WASM_AOT.md warm+JIT · issue #804). Drives the real page:
// the QuickJS card's warm session runs on a dedicated Worker (its own engine + memory), pre-warmed off the
// main thread; each Run is a message round-trip. Asserts the card runs end-to-end on BOTH warm tiers
// (warm-snapshot + warm+JIT), that the work actually went through the worker (not a silent main-thread
// fallback — `globalThis.__snapshotWorkerRuns` increments), and that fresh-per-Run isolation holds.
//
// Reuses the wasm32 module built by the CI real-browser job (and `serve.mjs` for COOP/COEP). Run:
//   node snapshot-worker-test.mjs
import { startServer } from './serve.mjs';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
// The QuickJS two-phase snapshot asset is committed, but skip cleanly if it (or the engine) is absent.
if (!existsSync(join(HERE, 'web', 'assets', 'qjs_snapshot.svmb'))) {
  console.log('SKIP: web/assets/qjs_snapshot.svmb not built (QuickJS on-ramp asset absent)');
  process.exit(0);
}
if (!existsSync(join(HERE, 'target', 'wasm32-unknown-unknown', 'release', 'svm_browser.wasm'))) {
  console.log('SKIP: engine wasm not built (run the browser-real recipe first)');
  process.exit(0);
}

const chromium = (await import('playwright')).chromium;
const { server, port } = await startServer(process.cwd());
const browser = await chromium.launch({ args: ['--no-sandbox'] });
let failed = false;
const ok = (m) => console.log(`  ok: ${m}`);
const fail = (m) => { failed = true; console.log(`  FAIL: ${m}`); };

const QJS = '[data-demo="JavaScript (QuickJS — write & run JS)"]';
const setSrc = (page, code) =>
  page.evaluate(({ sel, code }) => document.querySelector(`${sel} .CodeMirror`).CodeMirror.setValue(code), { sel: QJS, code });
const workerRuns = (page) => page.evaluate(() => globalThis.__snapshotWorkerRuns || 0);
const runQjs = async (page, timeout = 30_000) => {
  await page.click(`${QJS} .run`);
  await page.waitForFunction(
    (sel) => ['done', 'error', 'stopped'].includes(document.querySelector(`${sel} .state`).dataset.state),
    QJS, { timeout },
  );
  return {
    state: await page.getAttribute(`${QJS} .state`, 'data-state'),
    label: (await page.textContent(`${QJS} .state`)) || '',
    stdout: (await page.textContent(`${QJS} .stdout`)) || '',
  };
};

try {
  const page = await browser.newPage();
  page.on('pageerror', (e) => fail(`pageerror: ${e.message}`));
  page.on('console', (m) => { if (m.type() === 'error') fail(`console.error: ${m.text()}`); });
  await page.goto(`http://127.0.0.1:${port}/web/play.html`, { waitUntil: 'load' });
  await page.waitForFunction(() => document.getElementById('engine-state').dataset.state === 'ready', { timeout: 30_000 });

  // 1. warm-snapshot (interpreter tier, the default): a plain program runs on the worker.
  await setSrc(page, 'console.log("hi", 6 * 7);\n');
  let before = await workerRuns(page);
  let r = await runQjs(page);
  if (r.state === 'done' && r.stdout.includes('hi 42') && r.label.includes('warm-snapshot')) ok(`warm-snapshot on the worker → ${JSON.stringify(r.stdout.trim())}`);
  else fail(`warm-snapshot: state=${r.state} label=${r.label} stdout=${JSON.stringify(r.stdout)}`);
  let after = await workerRuns(page);
  after > before ? ok(`ran on the worker (__snapshotWorkerRuns ${before}→${after})`) : fail(`did NOT run on the worker (counter stuck at ${after}) — silent main-thread fallback`);

  // 2. warm+JIT (tick the wasm-JIT toggle): a compute-heavy program runs the eval near-native on the worker.
  await page.check(`${QJS} .jit-label input`);
  await setSrc(page, 'let s=0;for(let i=0;i<200000;i++)s+=i;console.log("sum", s);\n');
  before = await workerRuns(page);
  r = await runQjs(page);
  if (r.state === 'done' && r.stdout.includes('sum 19999900000') && r.label.includes('warm+JIT')) ok(`warm+JIT on the worker → ${JSON.stringify(r.stdout.trim())}`);
  else fail(`warm+JIT: state=${r.state} label=${r.label} stdout=${JSON.stringify(r.stdout)}`);
  after = await workerRuns(page);
  after > before ? ok(`warm+JIT ran on the worker (counter ${before}→${after})`) : fail(`warm+JIT did NOT run on the worker (counter ${after})`);

  // 3. fresh-per-Run isolation: a `var` in one Run must not leak into the next (snapshot restored each Run).
  await setSrc(page, 'var leak = 4242; console.log("set", leak);\n');
  r = await runQjs(page);
  if (!r.stdout.includes('set 4242')) fail(`isolation setup: ${JSON.stringify(r.stdout)}`);
  await setSrc(page, 'try { console.log("leak?", typeof leak, leak); } catch (e) { console.log("clean", e.name); }\n');
  r = await runQjs(page);
  if (r.state === 'done' && r.stdout.includes('clean') && !r.stdout.includes('4242')) ok('fresh-per-Run isolation holds (no var leak across Runs)');
  else fail(`isolation: state=${r.state} stdout=${JSON.stringify(r.stdout)}`);

  const total = await workerRuns(page);
  total >= 4 ? ok(`all ${total} Runs went through the snapshot worker`) : fail(`only ${total} worker Runs (expected ≥4)`);
} catch (e) {
  fail(`exception: ${e.message}`);
} finally {
  await browser.close();
  await new Promise((r) => server.close(r));
}

console.log(failed ? 'FAIL — snapshot worker' : 'PASS — snapshot worker: warm-snapshot + warm+JIT on a dedicated worker, isolation holds');
process.exit(failed ? 1 : 0);
