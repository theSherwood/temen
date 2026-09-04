// Real-browser (Chromium) end-to-end for the interactive bash card's **^C (SIGINT)**: a terminal
// VINTR (^C) must interrupt the FOREGROUND job and return the shell to a usable prompt. This is the
// interrupt analog of the ^Z suspend (`browser-bash-jobs-test.mjs`) and the natural complement to the
// job-control quad (^Z/fg/bg/jobs/kill) — the card wires ^C (`play.js`: Ctrl+C → `[3]`/VINTR) but
// nothing exercised it end-to-end.
//
// Drives the live `bash -i` session: `cat` (no args) runs in the FOREGROUND and blocks reading the
// terminal, so the shell is parked in `waitpid`. `^C` fires the #797 line discipline's ISIG group-kill
// (SIGINT at the foreground pgroup); `cat` has no handler, so it's TERMINATED (default action,
// invariant 14) and the shell's `waitpid` reaps it WIFSIGNALED(SIGINT) and returns to the prompt.
// `$?` for a SIGINT-killed foreground job is 128+2 = 130 — the proof the job was interrupted (not
// exited, not still running) AND the prompt is usable. A second command then runs cleanly and `^D`
// exits.
//
// `bash.temen` + `bin_cat.temen` are deploy-built assets (GPLv3, never committed): the test SKIPs
// cleanly when they (or Playwright) are absent. Run: node browser-bash-ctrl-c-test.mjs
import { startServer } from './serve.mjs';
import { benignAssetMiss } from './play-test-errors.mjs';
import { fileURLToPath } from 'node:url';
import { dirname } from 'node:path';
import { existsSync } from 'node:fs';

const ROOT = dirname(fileURLToPath(import.meta.url));
const ASSETS = ['bash.temen', 'bin_cat.temen'];
if (ASSETS.some((a) => !existsSync(`${ROOT}/web/assets/${a}`))) {
  console.log('– bash ctrl-c test skipped (web/assets/bash.temen or bin_cat.temen absent — run build-bash-assets.mjs)');
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
catch { console.log('– bash ctrl-c test skipped (playwright not found)'); process.exit(0); }

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

  // `cat` (no args) runs in the FOREGROUND and blocks reading the terminal. Confirm it is actually
  // the live foreground reader BEFORE ^C (else the signal races cat's startup and is lost): feed it a
  // line — `cat` copies stdin to stdout, so the marker appears TWICE (line-discipline echo + cat's
  // copy). Two `RUNNING` occurrences prove cat is running and reading the terminal.
  await term.fill('cat');
  await term.press('Enter');
  await page.waitForTimeout(1500); // let cat fork+exec and reach its terminal read before feeding it
  await term.fill('RUNNING');
  await term.press('Enter');
  await page.waitForFunction(
    (sel) => (document.querySelector(sel).textContent.match(/RUNNING/g)?.length ?? 0) >= 2,
    `${CARD} pre.stdout`, { timeout: 60000 });

  // ^C: SIGINT at the foreground group terminates `cat` (default action) and the shell returns to the
  // prompt. `echo BACK=$?` proves it — a SIGINT-killed foreground job leaves `$?` = 128+2 = 130.
  await term.press('Control+c');
  await term.fill('echo BACK=$?');
  await term.press('Enter');
  await page.waitForFunction(
    (sel) => /(?:^|\n)BACK=130\b/.test(document.querySelector(sel).textContent),
    `${CARD} pre.stdout`, { timeout: 60000 });

  // The prompt is fully usable: a second command runs cleanly with a fresh $? = 0.
  await term.fill('echo AGAIN=$?');
  await term.press('Enter');
  await page.waitForFunction(
    (sel) => /(?:^|\n)AGAIN=0\b/.test(document.querySelector(sel).textContent),
    `${CARD} pre.stdout`, { timeout: 60000 });

  // ^D ends the session — no stopped/queued jobs, so it exits on the first ^D.
  await term.press('Control+d');
  await page.waitForFunction(
    (sel) => document.querySelector(sel).dataset.state === 'done',
    `${CARD} span.state`, { timeout: 60000 });
  const text = await stdout.textContent();
  if (!text.includes('exit')) { console.error(`no exit farewell: ${JSON.stringify(text.slice(-200))}`); ok = false; }
} catch (e) {
  console.error(`bash ctrl-c session failed: ${e.message}`);
  console.error('stdout pane:', JSON.stringify(await stdout.textContent().catch(() => '<gone>')));
  console.error('state:', await state.textContent().catch(() => '<gone>'));
  ok = false;
}

await browser.close();
await new Promise((r) => server.close(r));
if (errors.length) { console.error('page errors:', errors); ok = false; }
if (ok) { console.log('✓ interactive bash card: `^C` terminates a foreground `cat` ($?=130) and the shell returns to a usable prompt'); process.exit(0); }
process.exit(1);
