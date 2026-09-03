// Real-browser (Chromium) end-to-end for the interactive bash card's **kill of a stopped job**
// (#1215): `kill -9 %n` on a `SIGTTIN`-stopped background job must actually terminate it, so the job
// leaves the table and the session exits cleanly. Before the cooperative driver's default-action
// terminate path, the kill set the personality's `term_sig` but nothing finalized the domain — the
// stopped job stayed on the table (`kill -9` a no-op), and `^D` reported "There are stopped jobs".
//
// Drives the live `bash -i` session: `cat &` reads the terminal from a background pgroup and
// SIGTTIN-stops (`[1]+ Stopped cat`); `kill -9 %1` then terminates the stopped job (the loop-top
// kill sweep finalizes the benched domain, retired WIFSIGNALED); the completion notice for %1 lands
// on the next prompt, and `^D` exits cleanly with NO "There are stopped jobs".
//
// `bash.temen` + `bin_cat.temen` are deploy-built assets (GPLv3, never committed): the test SKIPs
// cleanly when they (or Playwright) are absent. Run: node browser-bash-kill-stopped-test.mjs
import { startServer } from './serve.mjs';
import { benignAssetMiss } from './play-test-errors.mjs';
import { fileURLToPath } from 'node:url';
import { dirname } from 'node:path';
import { existsSync } from 'node:fs';

const ROOT = dirname(fileURLToPath(import.meta.url));
const ASSETS = ['bash.temen', 'bin_cat.temen'];
if (ASSETS.some((a) => !existsSync(`${ROOT}/web/assets/${a}`))) {
  console.log('– bash kill-stopped test skipped (web/assets/bash.temen or bin_cat.temen absent — run build-bash-assets.mjs)');
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
catch { console.log('– bash kill-stopped test skipped (playwright not found)'); process.exit(0); }

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

  // `cat &` launches in the background and SIGTTIN-stops on its terminal read (#1198).
  await term.fill('cat &');
  await term.press('Enter');
  await page.waitForFunction(
    (sel) => /\[1\]\s+\d+/.test(document.querySelector(sel).textContent),
    `${CARD} pre.stdout`, { timeout: 60000 });
  await term.fill('echo stopped=$?');
  await term.press('Enter');
  await page.waitForFunction(
    (sel) => { const t = document.querySelector(sel).textContent; return /stopped=0\b/.test(t) && /\[1\]\+?\s+Stopped\b/.test(t) && /Stopped\s+cat\b/.test(t); },
    `${CARD} pre.stdout`, { timeout: 60000 });

  // `kill -9 %1` on the STOPPED job: SIGKILL fires even on a stopped process (#1215) and the
  // cooperative driver now finalizes the killed domain, so the job is really terminated. The `[1]+`
  // completion notice (Killed) lands on the next prompt, and the prompt stays usable.
  await term.fill('kill -9 %1');
  await term.press('Enter');
  await term.fill('echo cleared=$?');
  await term.press('Enter');
  await page.waitForFunction(
    (sel) => /cleared=0\b/.test(document.querySelector(sel).textContent),
    `${CARD} pre.stdout`, { timeout: 60000 });
  // bash's WIFSIGNALED job notice for the killed job — this build labels a signal death "Signal"
  // (others "Killed"/"Terminated"); any of them proves the job was terminated, not left stopped.
  await page.waitForFunction(
    (sel) => { const t = document.querySelector(sel).textContent; return /\[1\]\+?\s+(Killed|Terminated|Signal)\b.*\bcat\b/.test(t); },
    `${CARD} pre.stdout`, { timeout: 60000 });

  // ^D ends the session: with the killed job gone from the table there is NO "There are stopped jobs"
  // hold, so bash exits on the first ^D.
  await term.press('Control+d');
  await page.waitForFunction(
    (sel) => document.querySelector(sel).dataset.state === 'done',
    `${CARD} span.state`, { timeout: 60000 });
  const text = await stdout.textContent();
  if (/There are stopped jobs/.test(text)) { console.error(`kill -9 left the job stopped: ${JSON.stringify(text.slice(-200))}`); ok = false; }
  if (!text.includes('exit')) { console.error(`no exit farewell: ${JSON.stringify(text.slice(-200))}`); ok = false; }
} catch (e) {
  console.error(`bash kill-stopped session failed: ${e.message}`);
  console.error('stdout pane:', JSON.stringify(await stdout.textContent().catch(() => '<gone>')));
  console.error('state:', await state.textContent().catch(() => '<gone>'));
  ok = false;
}

await browser.close();
await new Promise((r) => server.close(r));
if (errors.length) { console.error('page errors:', errors); ok = false; }
if (ok) { console.log('✓ interactive bash card: `kill -9 %1` terminates a SIGTTIN-stopped bg job — [1]+ Killed, clean ^D exit (no stopped-jobs hold)'); process.exit(0); }
process.exit(1);
