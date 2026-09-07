// Real-browser (Chromium) end-to-end for the interactive bash card's **line editing** (#802 readline
// rung): the staged `bash.temen` is the READLINE variant, keystrokes reach the terminal one key at a
// time, and the pane is a (minimal) terminal model — so Backspace edits the line readline is
// composing, ArrowUp recalls the previous command from history, and the pane shows the EDITED line
// (readline erases with `\b \b`, which the pane interprets instead of printing raw control bytes).
//
// The session: type `echo abX`, Backspace, `c`, Enter → the pane's committed line is `$ echo abc`
// and the output `abc`; ArrowUp + Enter re-runs it (a second `abc`); `echo rc=$?` → `rc=0`; ^D ends
// the session with bash's `exit` farewell. Needs the deploy-built assets (bash.temen; GPLv3, never
// committed) and Playwright — SKIPs cleanly when they are absent. Run: node browser-bash-readline-test.mjs
import { startServer } from './serve.mjs';
import { benignAssetMiss } from './play-test-errors.mjs';
import { fileURLToPath } from 'node:url';
import { dirname } from 'node:path';
import { existsSync } from 'node:fs';

const ROOT = dirname(fileURLToPath(import.meta.url));
if (!existsSync(`${ROOT}/web/assets/bash.temen`)) {
  console.log('– bash readline test skipped (web/assets/bash.temen absent — run build-bash-assets.mjs)');
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
catch { console.log('– bash readline test skipped (playwright not found)'); process.exit(0); }

const { server, port } = await startServer(ROOT);
const browser = await chromium.launch({ args: process.env.CI ? ['--no-sandbox'] : [] });
const page = await browser.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(String(e)));
page.on('console', (m) => { if (m.type() === 'error' && !benignAssetMiss(m)) errors.push(m.text()); });
await page.goto(`http://127.0.0.1:${port}/web/play.html`);

const CARD = 'section[data-demo="bash -i (an interactive terminal)"]';
const runBtn = page.locator(`${CARD} button.run`);
const term = page.locator(`${CARD} input.term-input`);
const stdout = page.locator(`${CARD} pre.stdout`);
const state = page.locator(`${CARD} span.state`);
const waitPane = (re) => page.waitForFunction(
  ([sel, src]) => new RegExp(src, 'm').test(document.querySelector(sel).textContent),
  [`${CARD} pre.stdout`, re.source], { timeout: 60000 });

let ok = true;
try {
  await runBtn.waitFor({ state: 'attached' });
  await page.waitForFunction((sel) => !document.querySelector(sel).disabled, `${CARD} button.run`, { timeout: 60000 });
  await runBtn.click();
  await page.waitForFunction((sel) => !document.querySelector(sel).disabled, `${CARD} input.term-input`, { timeout: 60000 });
  await waitPane(/\$ $/); // readline printed the first prompt

  // Per-key typing with a Backspace edit: readline echoes each key, erases the `X` with `\b \b`
  // (rendered by the pane as an actual erase), and the committed line reads `echo abc`.
  await term.type('echo abX');
  await waitPane(/\$ echo abX$/);
  await term.press('Backspace');
  await term.type('c');
  await waitPane(/\$ echo abc$/);
  await term.press('Enter');
  await waitPane(/^\$ echo abc\nabc\n\$ $/);

  // ArrowUp recalls `echo abc` from history; Enter re-runs it.
  await term.press('ArrowUp');
  await waitPane(/\$ echo abc$/);
  await term.press('Enter');
  await page.waitForFunction(
    (sel) => (document.querySelector(sel).textContent.match(/(?:^|\n)abc\n/g)?.length ?? 0) >= 2,
    `${CARD} pre.stdout`, { timeout: 60000 });

  // A fresh command still runs cleanly.
  await term.type('echo rc=$?');
  await term.press('Enter');
  await waitPane(/^rc=0$/);

  // ^D ends the session with the farewell.
  await term.press('Control+d');
  await page.waitForFunction(
    (sel) => document.querySelector(sel).dataset.state === 'done',
    `${CARD} span.state`, { timeout: 60000 });
  const text = await stdout.textContent();
  if (!text.includes('exit')) { console.error(`no exit farewell: ${JSON.stringify(text.slice(-200))}`); ok = false; }
  if (/[\b\r]/.test(text)) { console.error(`raw control bytes leaked into the pane: ${JSON.stringify(text)}`); ok = false; }
} catch (e) {
  console.error(`bash readline session failed: ${e.message}`);
  console.error('stdout pane:', JSON.stringify(await stdout.textContent().catch(() => '<gone>')));
  console.error('state:', await state.textContent().catch(() => '<gone>'));
  ok = false;
}

await browser.close();
await new Promise((r) => server.close(r));
if (errors.length) { console.error('page errors:', errors); ok = false; }
if (ok) { console.log('✓ interactive bash card: readline line editing (Backspace) + history (ArrowUp) work end-to-end in the browser'); process.exit(0); }
process.exit(1);
