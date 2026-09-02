// Real-browser (Chromium) end-to-end for the interactive bash card's **job table with several
// concurrent jobs** (#798 multiple concurrent jobs, the depth beyond #1171's single foreground job):
// drive the live `bash -i` session, suspend TWO `cat`s with ^Z so both sit in the job table, list them
// with `jobs` (asserting the [1]-/[2]+ previous/current markers), then resume each by job spec
// (`fg %1`, `fg %2`) and ^D it to exit. Exercises the whole stack — two exec'd children each in their
// own process group parked on stdin, two SIGTSTP stops reported through bash's foreground
// waitpid(WUNTRACED), bash's own job-table bookkeeping, and per-spec SIGCONT + terminal handoff — all
// on the cooperative bytecode engine over one shared wasm memory.
//
// `bash.temen` + `bin_cat.temen` are deploy-built assets (GPLv3, never committed): the test SKIPs
// cleanly when they (or Playwright) are absent. Run: node browser-bash-jobs-test.mjs
import { startServer } from './serve.mjs';
import { benignAssetMiss } from './play-test-errors.mjs';
import { fileURLToPath } from 'node:url';
import { dirname } from 'node:path';
import { existsSync } from 'node:fs';

const ROOT = dirname(fileURLToPath(import.meta.url));
const ASSETS = ['bash.temen', 'bin_cat.temen'];
if (ASSETS.some((a) => !existsSync(`${ROOT}/web/assets/${a}`))) {
  console.log('– bash jobs test skipped (web/assets/bash.temen or bin_cat.temen absent — run build-bash-assets.mjs)');
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
catch { console.log('– bash jobs test skipped (playwright not found)'); process.exit(0); }

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

  // Suspend two foreground `cat`s in turn: each blocks on its stdin read, ^Z stops it, bash reports
  // `[N]+ Stopped` and returns to the prompt. After both, the job table holds two stopped jobs.
  const suspendCat = async (wantStopped) => {
    await term.fill('cat');
    await term.press('Enter');
    await new Promise((r) => setTimeout(r, 800)); // let cat reach its blocking read
    await term.press('Control+z');
    await page.waitForFunction(
      ([sel, want]) => (document.querySelector(sel).textContent.match(/Stopped/g) || []).length >= want,
      [`${CARD} pre.stdout`, wantStopped], { timeout: 60000 });
  };
  await suspendCat(1);
  await suspendCat(2);

  // `jobs` lists BOTH jobs. bash marks the most-recently-stopped job current (`+`) and the one before
  // it previous (`-`): so job 2 is `[2]+` and job 1 is `[1]-`.
  await term.fill('jobs');
  await term.press('Enter');
  await page.waitForFunction(
    (sel) => { const t = document.querySelector(sel).textContent; return /\[1\]-\s+Stopped/.test(t) && /\[2\]\+\s+Stopped/.test(t); },
    `${CARD} pre.stdout`, { timeout: 60000 });

  // Resume each job BY SPEC and re-suspend it: `fg %n` re-foregrounds that exact job (bash echoes its
  // command line), then ^Z stops it again — the resume-to-block-then-restop round-trip, now selected
  // by job spec rather than "the current job". Doing both %1 and %2 proves spec selection reaches the
  // right group; re-stopping (rather than ^D-EOF) keeps the terminal open so the second spec still has
  // a live session to act on. Each ^Z adds one more `Stopped` line.
  const resumeAndRestop = async (spec, wantStopped) => {
    await term.fill(`fg %${spec}`);
    await term.press('Enter');
    await new Promise((r) => setTimeout(r, 1000)); // bash re-foregrounds the job; let its read reach block
    await term.press('Control+z');
    await page.waitForFunction(
      ([sel, want]) => (document.querySelector(sel).textContent.match(/Stopped/g) || []).length >= want,
      [`${CARD} pre.stdout`, wantStopped], { timeout: 60000 });
  };
  await resumeAndRestop(1, 3); // %1 resume + ^Z → third Stopped
  await resumeAndRestop(2, 4); // %2 resume + ^Z → fourth Stopped

  // Two stopped jobs remain: ^D at the prompt hits bash's exit guard ("There are stopped jobs.") and
  // does NOT exit; a second ^D exits anyway. This ends the session without EOF-racing a foreground cat.
  await term.press('Control+d');
  await page.waitForFunction(
    (sel) => document.querySelector(sel).textContent.includes('stopped jobs'),
    `${CARD} pre.stdout`, { timeout: 60000 });
  await term.press('Control+d');
  await page.waitForFunction(
    (sel) => document.querySelector(sel).dataset.state === 'done',
    `${CARD} span.state`, { timeout: 60000 });
  const text = await stdout.textContent();
  if (!text.includes('exit')) { console.error(`no exit farewell: ${JSON.stringify(text.slice(-200))}`); ok = false; }
} catch (e) {
  console.error(`bash jobs session failed: ${e.message}`);
  console.error('stdout pane:', JSON.stringify(await stdout.textContent().catch(() => '<gone>')));
  console.error('state:', await state.textContent().catch(() => '<gone>'));
  ok = false;
}

await browser.close();
await new Promise((r) => server.close(r));
if (errors.length) { console.error('page errors:', errors); ok = false; }
if (ok) { console.log('✓ interactive bash card: two ^Z-stopped jobs — jobs lists [1]-/[2]+, fg %1/%2 resume + re-^Z, stopped-jobs exit'); process.exit(0); }
process.exit(1);
