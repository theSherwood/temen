// Real-browser (Chromium) end-to-end for the interactive bash card's **background job launch**
// (#798 bg/&): drive the live `bash -i` session, run a command with `&`, and assert bash launches it
// in the background (`[n] pid`), returns to the prompt immediately (without waiting), streams the
// job's output, and posts the async `[n]+ Done` notification on the next prompt. Exercises the whole
// stack — bash forks + execve's the coreutil into its own process group without a terminal handoff or
// a blocking wait, the job runs to completion over CorePipes, and its SIGCHLD updates bash's job
// table — on the cooperative bytecode engine over one shared wasm memory.
//
// `bash.temen` + `bin_seq.temen` are deploy-built assets (GPLv3, never committed): the test SKIPs
// cleanly when they (or Playwright) are absent. Run: node browser-bash-bg-test.mjs
import { startServer } from './serve.mjs';
import { benignAssetMiss } from './play-test-errors.mjs';
import { fileURLToPath } from 'node:url';
import { dirname } from 'node:path';
import { existsSync } from 'node:fs';

const ROOT = dirname(fileURLToPath(import.meta.url));
const ASSETS = ['bash.temen', 'bin_seq.temen'];
if (ASSETS.some((a) => !existsSync(`${ROOT}/web/assets/${a}`))) {
  console.log('– bash bg test skipped (web/assets/bash.temen or bin_seq.temen absent — run build-bash-assets.mjs)');
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
catch { console.log('– bash bg test skipped (playwright not found)'); process.exit(0); }

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

let ok = true;
try {
  await runBtn.waitFor({ state: 'attached' });
  await page.waitForFunction((sel) => !document.querySelector(sel).disabled, `${CARD} button.run`, { timeout: 60000 });
  await runBtn.click();
  await page.waitForFunction((sel) => !document.querySelector(sel).disabled, `${CARD} input.term-input`, { timeout: 60000 });

  // Launch `seq 3` in the background: bash forks + execve's it into its own pgroup, prints `[1] <pid>`
  // and returns to the prompt WITHOUT waiting. The job's stdout (`1 2 3`) streams to the terminal.
  await term.fill('seq 3 &');
  await term.press('Enter');
  await page.waitForFunction(
    (sel) => { const t = document.querySelector(sel).textContent; return /\[1\]\s+\d+/.test(t) && /1\n2\n3/.test(t); },
    `${CARD} pre.stdout`, { timeout: 60000 });

  // The prompt is usable immediately (bash did not block on the background job): run a builtin. bash
  // posts the async completion notice `[1]+  Done  seq 3` for the finished job on this next prompt.
  await term.fill('echo afterbg=$?');
  await term.press('Enter');
  await page.waitForFunction(
    (sel) => document.querySelector(sel).textContent.includes('afterbg='),
    `${CARD} pre.stdout`, { timeout: 60000 });
  await page.waitForFunction(
    (sel) => { const t = document.querySelector(sel).textContent; return /\[1\]\+\s+Done\b/.test(t) && /Done\s+seq 3/.test(t); },
    `${CARD} pre.stdout`, { timeout: 60000 });

  // ^D on the empty line ends the session (no stopped/running jobs remain): bash prints its farewell.
  await term.press('Control+d');
  await page.waitForFunction(
    (sel) => document.querySelector(sel).dataset.state === 'done',
    `${CARD} span.state`, { timeout: 60000 });
  const text = await stdout.textContent();
  if (!text.includes('exit')) { console.error(`no exit farewell: ${JSON.stringify(text.slice(-200))}`); ok = false; }
} catch (e) {
  console.error(`bash bg session failed: ${e.message}`);
  console.error('stdout pane:', JSON.stringify(await stdout.textContent().catch(() => '<gone>')));
  console.error('state:', await state.textContent().catch(() => '<gone>'));
  ok = false;
}

await browser.close();
await new Promise((r) => server.close(r));
if (errors.length) { console.error('page errors:', errors); ok = false; }
if (ok) { console.log('✓ interactive bash card: `seq 3 &` background launch — [1] pid + streamed output + async [1]+ Done, clean exit'); process.exit(0); }
process.exit(1);
