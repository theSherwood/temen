// Real-browser (Chromium) end-to-end check for the **interactive bash card** (#1122 route (b)):
// drive the actual play-page UI — click Run on the "bash -i" card, type commands at its terminal
// input, and assert the session's streamed output. Under the hood this exercises the whole stack:
// the session Worker blocking inside `temen_bash_session` (real GNU bash `-i` on the cooperative
// bytecode engine, its pump parked on the #1122 external-wake doorbell at the prompt), the control
// Worker feeding keystrokes through the #797 line discipline and poll-draining output, and an
// external command fork→execve'd mid-session — all over one shared wasm memory.
//
// `bash.temen` is a **deploy-built asset** (GPLv3, never committed): the test SKIPs cleanly when it
// (or a coreutil, or Playwright) is absent — run `node build-bash-assets.mjs` first to make it
// actually run. Run: node browser-bash-interactive-test.mjs
import { startServer } from './serve.mjs';
import { benignAssetMiss } from './play-test-errors.mjs';
import { fileURLToPath } from 'node:url';
import { dirname } from 'node:path';
import { existsSync } from 'node:fs';

const ROOT = dirname(fileURLToPath(import.meta.url));
const ASSETS = ['bash.temen', 'bin_seq.temen', 'bin_cat.temen'];
if (ASSETS.some((a) => !existsSync(`${ROOT}/web/assets/${a}`))) {
  console.log('– interactive bash test skipped (web/assets/bash.temen, bin_seq.temen or bin_cat.temen absent — run build-bash-assets.mjs)');
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
catch { console.log('– interactive bash test skipped (playwright not found)'); process.exit(0); }

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
  // Engine load enables Run; the session takes a moment to reach the first prompt.
  await runBtn.waitFor({ state: 'attached' });
  await page.waitForFunction((sel) => !document.querySelector(sel).disabled, `${CARD} button.run`, { timeout: 60000 });
  await runBtn.click();
  await page.waitForFunction((sel) => !document.querySelector(sel).disabled, `${CARD} input.term-input`, { timeout: 60000 });

  // A builtin at the prompt: the echoed line and its output stream back.
  await term.fill('echo hi from the terminal');
  await term.press('Enter');
  await page.waitForFunction(
    (sel) => document.querySelector(sel).textContent.includes('hi from the terminal'),
    `${CARD} pre.stdout`, { timeout: 60000 });

  // #1146 — ^C at the prompt: the parked terminal read is interrupted (SIGINT delivered into bash's
  // handler on the cooperative bytecode engine — the slice-1 safepoint redirect + slice-2a CorePipe
  // EINTR), so bash aborts the line and sets `$? = 130` (128 + SIGINT). Before async delivery landed
  // on the bytecode tier the ^C was swallowed and `$?` stayed 0.
  await term.press('Control+c');
  await term.fill('echo rc=$?');
  await term.press('Enter');
  await page.waitForFunction(
    (sel) => document.querySelector(sel).textContent.includes('rc=130'),
    `${CARD} pre.stdout`, { timeout: 60000 });

  // An external command mid-session: bash forks + execve's /bin/seq while interactive.
  await term.fill('seq 3');
  await term.press('Enter');
  await page.waitForFunction(
    (sel) => document.querySelector(sel).textContent.includes('1\n2\n3'),
    `${CARD} pre.stdout`, { timeout: 60000 });

  // #1171 — ^Z job control: run a foreground `cat` (blocks on its stdin read), then ^Z suspends it.
  // The line discipline raises SIGTSTP at the foreground group; the exec'd cat stops on its parked
  // read; its SIGCHLD wakes bash's foreground waitpid(WUNTRACED) — even though bash has no async
  // SIGCHLD delivery (no sigaltstack) — so bash prints `[N]+ Stopped` and returns to the prompt.
  await term.fill('cat');
  await term.press('Enter');
  await new Promise((r) => setTimeout(r, 800)); // let cat reach its blocking read
  await term.press('Control+z');
  await page.waitForFunction(
    (sel) => document.querySelector(sel).textContent.includes('Stopped'),
    `${CARD} pre.stdout`, { timeout: 60000 });
  // The prompt is fully usable again: a stopped `cat` must NOT steal this line — bash runs it.
  await term.fill('echo zdone=$?');
  await term.press('Enter');
  await page.waitForFunction(
    (sel) => document.querySelector(sel).textContent.includes('zdone='),
    `${CARD} pre.stdout`, { timeout: 60000 });

  // Resume the stopped cat with `fg`: bash re-foregrounds the job's pgrp and SIGCONTs it. That
  // SIGCONT is a *guest* `kill` syscall (unlike ^Z/^C, which arrive inline through the line
  // discipline), so on the coop engine its deferred wake must fire inline — before #1171's inline
  // fix it spawned an OS thread and trapped the wasm engine to a bare `unreachable`. Then ^D EOFs
  // the resumed cat so it exits and bash reaps it and returns to a clean prompt.
  await term.fill('fg');
  await term.press('Enter');
  await new Promise((r) => setTimeout(r, 800));
  await term.press('Control+d'); // EOF to the resumed cat → it exits, bash reaps it
  await new Promise((r) => setTimeout(r, 800));

  // ^D on the empty line ends the session: bash prints its `exit` farewell and the card settles.
  await term.press('Control+d');
  await page.waitForFunction(
    (sel) => document.querySelector(sel).dataset.state === 'done',
    `${CARD} span.state`, { timeout: 60000 });
  const text = await stdout.textContent();
  if (!text.includes('exit')) { console.error(`no exit farewell in output: ${JSON.stringify(text.slice(-200))}`); ok = false; }
} catch (e) {
  console.error(`interactive bash session failed: ${e.message}`);
  console.error('stdout pane:', JSON.stringify(await stdout.textContent().catch(() => '<gone>')));
  console.error('state:', await state.textContent().catch(() => '<gone>'));
  ok = false;
}

await browser.close();
await new Promise((r) => server.close(r));
if (errors.length) { console.error('page errors:', errors); ok = false; }
if (ok) { console.log('✓ interactive bash card: live session — prompt, builtin, ^C→rc=130, fork+exec\'d seq, ^Z suspends cat + usable prompt, fg resumes, ^D exit'); process.exit(0); }
process.exit(1);
