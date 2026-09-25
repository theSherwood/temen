// Chromium end-to-end for the **Postgres session surviving a real page reload** (the persistence
// feature: snapshot the data dir → IndexedDB → reboot from it). This is the one layer the native /
// node round-trips can't reach: the actual `play.js` IndexedDB glue driven through a browser reload.
//
// Flow, all against the real playground page:
//   1. Run `CREATE TABLE` + `INSERT` — the first Run boots the backend and persists the data dir.
//   2. Wait for the IndexedDB save to land, then `page.reload()` (drops all wasm memory).
//   3. Run `SELECT` — the session boots *from the saved snapshot*, and the row is still there.
//   4. Run `\reset` — the saved database is cleared (IndexedDB key gone).
//
// Reuses `serve.mjs` (COOP/COEP) + the wasm32 threads build. SKIPs (exit 0) when the ~60 MB Postgres
// artifacts aren't staged — CI's real-browser job only has committed assets. Run locally:
//   node build-pg-assets.mjs && node pg-reload-test.mjs
import { startServer } from './serve.mjs';
import { benignAssetMiss } from './play-test-errors.mjs';
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const need = ['web/assets/postgres_resolved.temen', 'web/assets/pgdata.img',
  'target/wasm32-unknown-unknown/release/temen_browser.wasm'];
const missing = need.filter((p) => !existsSync(join(HERE, p)));
if (missing.length) {
  console.log(`SKIP: pg reload test — missing ${missing.join(', ')} (run \`node build-pg-assets.mjs\` + build the wasm)`);
  process.exit(0);
}

const PG = 'PostgreSQL (17.5 — write & run SQL)';
const KEY = './assets/postgres_resolved.temen'; // play.js keys the saved image by the module URL
const sel = `[data-demo="${PG}"]`;

// Resolve Playwright's `chromium` portably (CI installs it locally; this env has a global install).
async function loadChromium() {
  for (const spec of ['playwright', '/opt/node22/lib/node_modules/playwright/index.js']) {
    try {
      const m = await import(spec);
      const chromium = m.chromium ?? m.default?.chromium;
      if (chromium) return chromium;
    } catch {
      /* try the next resolution */
    }
  }
  throw new Error('playwright not found — run `npm i playwright && npx playwright install chromium`');
}
const chromium = await loadChromium();
const { server, port } = await startServer(process.cwd());
const browser = await chromium.launch({ args: ['--no-sandbox'] });
let failed = false;
const ok = (m) => console.log(`  ok: ${m}`);
const fail = (m) => { failed = true; console.log(`  FAIL: ${m}`); };

// #1135 diagnostics (no effect unless the reload stalls): say WHICH Chromium component is wedged. A
// renderer whose main (or IO) thread is merely stuck in JS/wasm or a user-space wait is SIGTERMed by
// Chromium's shutdown, so it cannot explain the CI signature (`browser.close()` taking exactly
// Playwright's 30 s kill); only a browser process that stops answering, or a child that is stopped /
// in uninterruptible sleep, can. So: did the new document commit, does the browser still answer CDP,
// does the page, and which Chromium threads are stopped (T/t) or in D state — with their wait channel.
async function describeStall(page, navigations) {
  try { await describeStallOn(page, navigations); } catch (e) { console.log(`  stall: diagnostics failed: ${e.message}`); }
}
async function describeStallOn(page, navigations) {
  const within = (p, ms) => Promise.race([
    p.then(() => 'answers', (e) => `error ${String(e.message).split('\n')[0]}`),
    new Promise((r) => setTimeout(() => r(`silent for ${ms} ms`), ms)),
  ]);
  const cdp = await within(browser.newBrowserCDPSession().then((s) => s.send('Browser.getVersion')), 5000);
  const eva = await within(page.evaluate(() => document.readyState), 5000);
  console.log(`  stall: new-document commits=${navigations} · browser CDP ${cdp} · page ${eva}`);
  const rd = (p) => { try { return readFileSync(p, 'utf8'); } catch { return ''; } };
  for (const pid of readdirSync('/proc').filter((d) => /^\d+$/.test(d))) {
    const cmd = rd(`/proc/${pid}/cmdline`).replace(/\0/g, ' ');
    if (!/headless_shell|chrom/.test(cmd.split(' ')[0])) continue;
    const odd = [];
    for (const tid of (() => { try { return readdirSync(`/proc/${pid}/task`); } catch { return []; } })()) {
      const st = rd(`/proc/${pid}/task/${tid}/stat`);
      const state = st.slice(st.lastIndexOf(')') + 2, st.lastIndexOf(')') + 3);
      if (/[DTt]/.test(state)) odd.push(`${st.slice(st.indexOf('(') + 1, st.lastIndexOf(')'))}:${state}:${rd(`/proc/${pid}/task/${tid}/wchan`)}`);
    }
    const type = (/--type=([\w-]+)/.exec(cmd) || [, 'browser'])[1];
    console.log(`  stall: ${type} ${pid} rss=${(/VmRSS:\s+(\d+)/.exec(rd(`/proc/${pid}/status`)) || [, '?'])[1]}kB ${odd.join(' ') || 'no stopped/D threads'}`);
  }
  for (const f of ['cpu', 'memory', 'io']) console.log(`  stall: pressure ${f}: ${rd(`/proc/pressure/${f}`).split('\n')[0]}`);
}

// Read the byte length stored under the session key (null if absent). Never *creates* the DB/store —
// it opens the existing one play.js made, so probing can't race play.js into a storeless database.
const savedSize = (page) => page.evaluate((k) => new Promise((resolve) => {
  const open = () => {
    const req = indexedDB.open('temen-pg');
    req.onsuccess = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains('sessions')) return resolve(null);
      const r = db.transaction('sessions', 'readonly').objectStore('sessions').get(k);
      r.onsuccess = () => resolve(r.result ? (r.result.byteLength ?? r.result.length ?? 0) : null);
      r.onerror = () => resolve(null);
    };
    req.onerror = () => resolve(null);
  };
  if (indexedDB.databases) {
    indexedDB.databases().then((list) => (list.some((d) => d.name === 'temen-pg') ? open() : resolve(null)));
  } else {
    open();
  }
}), KEY);

const waitEngine = (page) => page.waitForFunction(
  () => document.getElementById('engine-state').dataset.state === 'ready', { timeout: 30_000 });

// Set the card's editor, Run, and wait for the run to settle. A pre-click sentinel state avoids the
// "already done from the previous run" race (a fast query never passes through `running`).
async function runSql(page, sql, timeout = 120_000) {
  await page.evaluate(([s, q]) => {
    document.querySelector(`${s} .CodeMirror`).CodeMirror.setValue(q);
    document.querySelector(`${s} .state`).dataset.state = 'pending';
  }, [sel, sql]);
  await page.click(`${sel} .run`);
  await page.waitForFunction(
    (s) => ['done', 'error', 'stopped'].includes(document.querySelector(`${s} .state`).dataset.state),
    sel, { timeout });
  return page.evaluate((s) => ({
    state: document.querySelector(`${s} .state`).dataset.state,
    stdout: document.querySelector(`${s} .stdout`).textContent,
    log: document.querySelector(`${s} .log`).textContent,
  }), sel);
}

// Poll `savedSize` until the predicate holds (or time out).
async function waitSaved(page, pred, label, timeout = 20_000) {
  const t0 = Date.now();
  for (;;) {
    const sz = await savedSize(page);
    if (pred(sz)) return sz;
    if (Date.now() - t0 > timeout) throw new Error(`timed out waiting for ${label} (last size ${sz})`);
    await new Promise((r) => setTimeout(r, 250));
  }
}

try {
  const page = await browser.newPage();
  page.on('pageerror', (e) => fail(`pageerror: ${e.message}`));
  page.on('console', (m) => { if (m.type() === 'error' && !benignAssetMiss(m)) fail(`console.error: ${m.text()}`); });
  await page.goto(`http://127.0.0.1:${port}/web/play.html`, { waitUntil: 'load' });
  await waitEngine(page);

  // 1) First Run: boot the backend (pristine), create a table, insert a sentinel row.
  const r1 = await runSql(page, 'CREATE TABLE reload_probe (x int);\nINSERT INTO reload_probe VALUES (919191);');
  if (r1.state !== 'done') fail(`create/insert did not finish cleanly: state=${r1.state}`);
  else if (!r1.stdout.includes('backend>')) fail('no backend prompt after boot');
  else if (r1.stdout.includes('ERROR')) fail(`create/insert reported an error:\n${r1.stdout.slice(-300)}`);
  else ok('booted + created table + inserted row');

  // 2) The post-query snapshot lands in IndexedDB (this is what a reload will reboot from).
  const size = await waitSaved(page, (s) => s != null && s > 0, 'the session to be saved');
  ok(`data dir persisted to IndexedDB (${size} bytes)`);

  // 3) Reload the page — every byte of wasm linear memory is gone. Anything that survives came from IDB.
  // Wait on `domcontentloaded`, not `load`: the full `load` event blocks until every sub-resource
  // (the ~2.2 MB engine wasm + ~4.3 MB qjs asset, re-fetched from the COOP/COEP server) has settled,
  // which under CI runner load intermittently exceeds Playwright's 30 s default and flakes the reload
  // (ISSUES.md I56). The engine's own readiness is the real gate — `waitEngine` below waits for
  // `engine-state=ready`, and the persistence assertions (SELECT 919191, "restored") still must pass —
  // so relaxing the navigation wait hardens the flake without masking any real regression.
  let commits = 0;
  page.on('framenavigated', (f) => { if (f === page.mainFrame()) commits++; });
  try {
    await page.reload({ waitUntil: 'domcontentloaded', timeout: 60_000 });
  } catch (e) {
    // One-shot retry (ISSUES.md I56, second lever): even the relaxed 60 s `domcontentloaded` wait can
    // lapse under CI runner load. The snapshot is already durable in IndexedDB (step 2 asserted it),
    // so a second reload loses nothing — and the real gates below (`waitEngine` + the SELECT 919191 /
    // "restored" assertions) still must pass, so the retry cannot mask a genuine regression.
    console.log(`  reload timed out once (I56) — retrying: ${String(e.message).split('\n')[0]}`);
    await describeStall(page, commits);
    try {
      await page.reload({ waitUntil: 'domcontentloaded', timeout: 60_000 });
    } catch (e2) {
      await describeStall(page, commits);
      throw e2;
    }
  }
  await waitEngine(page);
  const r2 = await runSql(page, 'SELECT x FROM reload_probe;');
  if (r2.state !== 'done') fail(`select after reload did not finish cleanly: state=${r2.state}`);
  else if (!r2.log.includes('restored')) fail(`session was not restored from the saved image; log:\n${r2.log}`);
  else if (r2.stdout.includes('ERROR')) fail(`select after reload errored:\n${r2.stdout.slice(-300)}`);
  else if (!r2.stdout.includes('919191')) fail(`the row did not survive the reload; stdout:\n${r2.stdout.slice(-300)}`);
  else ok('after reload: session restored from IndexedDB and the row is still there');

  // 4) `\reset` wipes the saved database.
  const r3 = await runSql(page, '\\reset', 20_000);
  if (r3.state !== 'done') fail(`\\reset did not finish cleanly: state=${r3.state}`);
  else {
    await waitSaved(page, (s) => s == null, 'the saved database to be cleared');
    ok('`\\reset` cleared the saved database');
  }
} catch (e) {
  fail(`exception: ${e.message}`);
} finally {
  await browser.close();
  server.close();
}

console.log(failed ? 'pg reload test: FAILED' : 'pg reload test: PASSED');
process.exit(failed ? 1 : 0);
