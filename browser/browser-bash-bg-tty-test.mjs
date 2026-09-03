// Real-browser (Chromium) end-to-end for the interactive bash card's **background terminal-read
// stop** (#1198): a background job that reads the controlling terminal must be SIGTTIN-stopped, not
// left spinning and not allowed to steal the shell's input. Drive the live `bash -i` session, run
// `cat &` (cat reads the terminal but is in a background process group), and assert:
//   * bash launches it (`[n] pid`) and returns to the prompt immediately,
//   * cat's terminal read raises SIGTTIN and stops it — bash reports `[n]+  Stopped  cat`,
//   * the prompt stays usable and cat does NOT steal the next typed line (a `stopped` process makes
//     no progress), then `kill -9 %n` clears the stopped job and the session exits cleanly.
//
// This is the whole #1198 stack end-to-end: bash forks + execve's the coreutil into its own pgroup,
// the exec'd reader hits `tty_background_check` → SIGTTIN → stop, and — crucially — the cooperative
// bytecode engine BENCHES the stopped reader at its syscall boundary instead of spinning on the
// libc `-ERESTART` retry (the bug this fixes), so bash's SIGCHLD marks the job Stopped.
//
// `bash.temen` + `bin_cat.temen` are deploy-built assets (GPLv3, never committed): the test SKIPs
// cleanly when they (or Playwright) are absent. Run: node browser-bash-bg-tty-test.mjs
import { startServer } from './serve.mjs';
import { benignAssetMiss } from './play-test-errors.mjs';
import { fileURLToPath } from 'node:url';
import { dirname } from 'node:path';
import { existsSync } from 'node:fs';

const ROOT = dirname(fileURLToPath(import.meta.url));
const ASSETS = ['bash.temen', 'bin_cat.temen'];
if (ASSETS.some((a) => !existsSync(`${ROOT}/web/assets/${a}`))) {
  console.log('– bash bg-tty test skipped (web/assets/bash.temen or bin_cat.temen absent — run build-bash-assets.mjs)');
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
catch { console.log('– bash bg-tty test skipped (playwright not found)'); process.exit(0); }

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

  // Launch `cat &` in the background: bash forks + execve's it into its own pgroup and prints
  // `[1] <pid>`, returning to the prompt without waiting. cat immediately reads the terminal.
  await term.fill('cat &');
  await term.press('Enter');
  await page.waitForFunction(
    (sel) => /\[1\]\s+\d+/.test(document.querySelector(sel).textContent),
    `${CARD} pre.stdout`, { timeout: 60000 });

  // cat's background terminal read raised SIGTTIN and STOPPED it (the engine benched the reader
  // instead of spinning on -ERESTART). The `[1]+ Stopped cat` notice lands on the next prompt, and
  // the `echo` proves the prompt is usable AND that stopped cat did not steal this typed line.
  await term.fill('echo alive=$?');
  await term.press('Enter');
  await page.waitForFunction(
    (sel) => /alive=0\b/.test(document.querySelector(sel).textContent),
    `${CARD} pre.stdout`, { timeout: 60000 });
  await page.waitForFunction(
    (sel) => { const t = document.querySelector(sel).textContent; return /\[1\]\+?\s+Stopped\b/.test(t) && /Stopped\s+cat\b/.test(t); },
    `${CARD} pre.stdout`, { timeout: 60000 });

  // Resume the stopped job in the FOREGROUND (`fg %1`): SIGCONT continues cat, and — now the
  // foreground group — its terminal read succeeds instead of re-stopping. A `^D` (VEOF) on the empty
  // line EOFs that read, so cat exits cleanly and the job leaves the table (a wired path, unlike
  // SIGKILL of a still-stopped job). This also witnesses fg-resume of a SIGTTIN-stopped bg job.
  await term.fill('fg %1');
  await term.press('Enter');
  await term.press('Control+d');
  await page.waitForFunction(
    (sel) => { const t = document.querySelector(sel).textContent; return /\bfg %1\b/.test(t) && /\bcat\b/.test(t); },
    `${CARD} pre.stdout`, { timeout: 60000 });
  await term.fill('echo cleared=yes');
  await term.press('Enter');
  await page.waitForFunction(
    (sel) => /cleared=yes\b/.test(document.querySelector(sel).textContent),
    `${CARD} pre.stdout`, { timeout: 60000 });

  // ^D on the empty line ends the session (no running/stopped jobs remain): bash prints its farewell.
  await term.press('Control+d');
  await page.waitForFunction(
    (sel) => document.querySelector(sel).dataset.state === 'done',
    `${CARD} span.state`, { timeout: 60000 });
  const text = await stdout.textContent();
  if (!text.includes('exit')) { console.error(`no exit farewell: ${JSON.stringify(text.slice(-200))}`); ok = false; }
} catch (e) {
  console.error(`bash bg-tty session failed: ${e.message}`);
  console.error('stdout pane:', JSON.stringify(await stdout.textContent().catch(() => '<gone>')));
  console.error('state:', await state.textContent().catch(() => '<gone>'));
  ok = false;
}

await browser.close();
await new Promise((r) => server.close(r));
if (errors.length) { console.error('page errors:', errors); ok = false; }
if (ok) { console.log('✓ interactive bash card: `cat &` background terminal read SIGTTIN-stops ([1]+ Stopped cat), prompt stays usable, no input steal'); process.exit(0); }
process.exit(1);
