// Real-browser (Chromium) end-to-end for the interactive bash card's **`bg` of a terminal-reading
// job** (#1198 tail): after `cat &` SIGTTIN-stops (a background terminal read), `bg %1` resumes it —
// and the resumed `cat`, still not the terminal's foreground group, must SIGTTIN-**re-stop** on its
// re-issued read rather than STEAL the shell's input. The steal was the original #1198 symptom (a
// foreground ping-pong: the bg'd reader ate the next typed line, and the shell never got its prompt
// back). This drives the live UI to prove the shell keeps the prompt: a command typed after `bg %1`
// actually RUNS (its output appears), and `cat` bounces straight back to Stopped.
//
// `bg %1` continues the job (kill(-pgid, SIGCONT), no terminal handoff — the terminal stays with the
// shell); the resumed `cat` re-reads fd 0, `tty_background_check` fires again (its group ≠ the
// foreground), and it re-stops on the EMPTY terminal — before any keystroke — because the background
// check gates on process-group, not on data. So the next typed `echo` is read by BASH (run: its
// output prints), not consumed by `cat`. A second `[1]+ Stopped cat` confirms the re-stop, and
// `kill -9 %1` + `^D` tears the session down cleanly.
//
// `bash.temen` + `bin_cat.temen` + `bin_echo.temen` are deploy-built assets (GPLv3, never committed):
// the test SKIPs cleanly when they (or Playwright) are absent. Run: node browser-bash-bg-resume-test.mjs
import { startServer } from './serve.mjs';
import { benignAssetMiss } from './play-test-errors.mjs';
import { fileURLToPath } from 'node:url';
import { dirname } from 'node:path';
import { existsSync } from 'node:fs';

const ROOT = dirname(fileURLToPath(import.meta.url));
const ASSETS = ['bash.temen', 'bin_cat.temen', 'bin_echo.temen'];
if (ASSETS.some((a) => !existsSync(`${ROOT}/web/assets/${a}`))) {
  console.log('– bash bg-resume test skipped (web/assets/bash.temen, bin_cat.temen or bin_echo.temen absent — run build-bash-assets.mjs)');
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
catch { console.log('– bash bg-resume test skipped (playwright not found)'); process.exit(0); }

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

  // Snapshot how many `Stopped cat` notices have printed so far, so the assert below can wait for a
  // SECOND one (the re-stop after `bg`) rather than re-matching the first.
  const stoppedBefore = (await stdout.textContent()).match(/Stopped\s+cat\b/g)?.length ?? 0;

  // `bg %1`: resume the stopped job in the background (SIGCONT, no terminal handoff). The resumed
  // `cat` re-reads the terminal, is a background reader again, and SIGTTIN-re-stops on the empty
  // terminal — before the next keystroke — so it CANNOT steal the line typed next.
  await term.fill('bg %1');
  await term.press('Enter');
  // Wait for the `bg` acknowledgement (`[1]+ cat &`) so the next line isn't typed into a busy prompt.
  await page.waitForFunction(
    (sel) => /\[1\]\+?\s+cat\b/.test(document.querySelector(sel).textContent),
    `${CARD} pre.stdout`, { timeout: 60000 });

  // The proof the shell kept its prompt: this `echo` is read and RUN by BASH (its output `MARK=alive`
  // prints on its own line), not consumed by the bg'd `cat`. The typed line is echoed as
  // `echo MARK=alive`; only a real run prints `MARK=alive` at the start of a fresh line. Had `cat`
  // stolen the input, `MARK=alive` would never appear as command output.
  await term.fill('echo MARK=alive');
  await term.press('Enter');
  await page.waitForFunction(
    (sel) => /(?:^|\n)MARK=alive\b/.test(document.querySelector(sel).textContent),
    `${CARD} pre.stdout`, { timeout: 60000 });

  // A second command: (a) further proves the prompt stayed usable, and (b) flushes bash's DEFERRED
  // job notice — a re-stop is reported right before the *next* prompt, so `[1]+ Stopped cat` (the
  // proof `cat` bounced back to Stopped rather than exiting on a stolen line) prints ahead of this
  // command's output.
  await term.fill('echo MARK2=ok');
  await term.press('Enter');
  await page.waitForFunction(
    ({ sel, before }) => {
      const t = document.querySelector(sel).textContent;
      return /(?:^|\n)MARK2=ok\b/.test(t) && (t.match(/Stopped\s+cat\b/g)?.length ?? 0) > before;
    },
    { sel: `${CARD} pre.stdout`, before: stoppedBefore }, { timeout: 60000 });

  // Tear down: kill the re-stopped job, then `^D` exits with no "There are stopped jobs" hold.
  await term.fill('kill -9 %1');
  await term.press('Enter');
  await term.fill('echo cleared=$?');
  await term.press('Enter');
  await page.waitForFunction(
    (sel) => /cleared=0\b/.test(document.querySelector(sel).textContent),
    `${CARD} pre.stdout`, { timeout: 60000 });
  await term.press('Control+d');
  await page.waitForFunction(
    (sel) => document.querySelector(sel).dataset.state === 'done',
    `${CARD} span.state`, { timeout: 60000 });
  const text = await stdout.textContent();
  if (/There are stopped jobs/.test(text)) { console.error(`kill left the job stopped: ${JSON.stringify(text.slice(-200))}`); ok = false; }
  if (!text.includes('exit')) { console.error(`no exit farewell: ${JSON.stringify(text.slice(-200))}`); ok = false; }
} catch (e) {
  console.error(`bash bg-resume session failed: ${e.message}`);
  console.error('stdout pane:', JSON.stringify(await stdout.textContent().catch(() => '<gone>')));
  console.error('state:', await state.textContent().catch(() => '<gone>'));
  ok = false;
}

await browser.close();
await new Promise((r) => server.close(r));
if (errors.length) { console.error('page errors:', errors); ok = false; }
if (ok) { console.log('✓ interactive bash card: `bg %1` of a stopped `cat` re-stops it (no input steal) — the next command RUNS and cat bounces back to Stopped'); process.exit(0); }
process.exit(1);
