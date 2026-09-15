// Chromium gate for the playground's **reactor save-states** — freeze a running guest to a §12 artifact,
// persist it in IndexedDB, and come back to it later (#1458, the #1454 moment design). Drives the real
// page: run the `bounce` reactor, save it at an exact frame, stop, load it back, and assert the guest
// resumes at the instant it was frozen on — then reload the page and assert the save is still there.
//
// Run on **both tiers**, because a save-state is one frontier across them (INVARIANTS #14): the
// interpreter freezes its own window, the wasm-JIT reactor freezes the emitted tier's, and the page
// remembers which one took a save so it comes back on that tier rather than on whatever the toggle says.
//
// **Scope.** This gates the page *wiring*: that the row reaches the engine's freeze/thaw exports on
// either tier, that the artifact round-trips through IndexedDB, that a thaw resumes the frozen instant
// rather than re-running `_start`, and that a save outlives a full page reload. The codec itself, the
// named-cap re-grant, and the refusal on a mismatched module are gated natively by
// `browser/tests/reactor_moment.rs`.
//
// The "resumes at the frozen instant" check is pixel-exact and needs no live-frame sampling: the run is
// paused on the scrub bar at a known frame before the save, so both the original and the thawed run are
// driven the same deterministic N frames forward from the same state, and the two canvases must match.
//
// Reuses the wasm32 module built by the CI real-browser job (and `serve.mjs` for COOP/COEP). Run:
//   cargo build --release --lib --target wasm32-unknown-unknown
//   node browser-play-savestate-test.mjs
import { startServer } from './serve.mjs';
import { benignAssetMiss } from './play-test-errors.mjs';

const chromium = (await import('playwright')).chromium;
const { server, port } = await startServer(process.cwd());
const browser = await chromium.launch({ args: ['--no-sandbox'] });
let failed = false;
const ok = (m) => console.log(`  ok: ${m}`);
const fail = (m) => { failed = true; console.log(`  FAIL: ${m}`); };

const DEMO = 'bounce (interactive — arrow keys)';
const sel = `[data-demo="${DEMO}"]`;
const SAVE_AT = 60; // a keyframe tick (stride 30), so the seek that parks us there re-runs zero frames
const AFTER = 10;   // frames run past the save, on both the original and the thawed run

// Park the scrub bar on `target` and hand back what is on the canvas there.
const seek = (page, target) => page.evaluate((t) => {
  const row = document.querySelector('.scrub:not([hidden]) .scrub-range');
  row.value = String(t);
  row.dispatchEvent(new Event('input', { bubbles: true }));
  const canvas = document.querySelector('.demo .canvas:not([hidden])');
  return { at: globalThis.__scrubState().viewing, png: canvas.toDataURL() };
}, target);

const row = (page) => page.evaluate((s) => {
  const r = document.querySelector(`${s} .savestate`);
  if (!r) return null;
  return {
    save: r.querySelector('.save-write').disabled,
    load: r.querySelector('.save-load').disabled,
    clear: r.querySelector('.save-clear').disabled,
    label: r.querySelector('.save-label').textContent,
  };
}, sel);

// `browser.newPage()` would mint a fresh incognito context each time — and with it a fresh, empty
// IndexedDB, which is precisely the thing under test. One context per round, reloaded in place.
const openPage = async (context) => {
  const page = await context.newPage();
  page.on('pageerror', (e) => fail(`pageerror: ${e.message}`));
  page.on('console', (m) => { if (m.type() === 'error' && !benignAssetMiss(m)) fail(`console.error: ${m.text()}`); });
  await page.goto(`http://127.0.0.1:${port}/web/play.html`, { waitUntil: 'load' });
  await ready(page);
  return page;
};
const ready = (page) => page.waitForFunction(
  () => document.getElementById('engine-state').dataset.state === 'ready',
  { timeout: 30_000 },
);

// Forget everything this page has stored, so a leftover save from an earlier round can't mask a failure.
const wipe = (page) => page.evaluate(() => new Promise((res) => {
  const r = indexedDB.deleteDatabase('temen-savestate');
  r.onsuccess = r.onerror = r.onblocked = () => res();
}));

// One full save/stop/load cycle on the named tier. `jit` drives the card's wasm-JIT toggle; the save
// record carries the tier, so the Load after a reload must come back on it whatever the toggle says.
async function round(tier, jit) {
  console.log(`[${tier}]`);
  const context = await browser.newContext();
  const page = await openPage(context);
  await wipe(page);
  await page.reload({ waitUntil: 'load' });
  await ready(page);

  // The row is present and offers nothing before a run: no reactor to freeze, nothing saved.
  const cold = await row(page);
  cold && cold.save && cold.load && cold.clear && cold.label === 'no saved state'
    ? ok('save row present and inert before any run')
    : fail(`cold row: ${JSON.stringify(cold)}`);

  await page.evaluate(({ s, on }) => {
    const box = document.querySelector(`${s} .jit-toggle`);
    if (box) box.checked = on;
  }, { s: sel, on: jit });

  // Run, then park the scrub bar on an exact frame so everything after this is deterministic.
  await page.click(`${sel} .run`);
  await page.waitForFunction(() => (globalThis.__scrubState()?.tick ?? 0) >= 75, { timeout: 60_000 });
  const ran = await page.evaluate((s) => document.querySelector(`${s} .log`).textContent, sel);
  ran.includes(jit ? 'wasm-JIT reactor opened' : 'reactor opened')
    ? ok(`running on the ${tier} tier`)
    : fail(`wrong tier: ${ran.split('\n').slice(-2).join(' | ')}`);

  const parked = await seek(page, SAVE_AT);
  parked.at === SAVE_AT ? ok(`parked at frame ${SAVE_AT}`) : fail(`park: ${JSON.stringify(parked)}`);
  const live = await row(page);
  !live.save ? ok('Save enabled for the live reactor') : fail(`live row: ${JSON.stringify(live)}`);

  // Freeze it. The artifact is the guest at frame 60 — the frame the bar is showing, not the frame the
  // run had reached before it was paused.
  await page.click(`${sel} .save-write`);
  await page.waitForFunction((s) => /^saved: /.test(document.querySelector(`${s} .save-label`).textContent),
    sel, { timeout: 15_000 });
  const saved = await row(page);
  !saved.load && !saved.clear && saved.label.includes(`frame ${SAVE_AT}`) && saved.label.includes(tier)
    ? ok(saved.label)
    : fail(`after save: ${JSON.stringify(saved)}`);

  // The reference: run the ORIGINAL guest `AFTER` frames past the save and photograph it.
  const reference = await seek(page, SAVE_AT + AFTER);
  reference.at === SAVE_AT + AFTER && reference.png !== parked.png
    ? ok(`reference frame ${SAVE_AT + AFTER} captured (and it moved)`)
    : fail(`reference: at=${reference.at} moved=${reference.png !== parked.png}`);

  // Stop: the guest is gone, the save is not.
  await page.click(`${sel} .stop`);
  const stopped = await row(page);
  stopped.save && !stopped.load
    ? ok('after Stop: Save goes inert, Load still offered')
    : fail(`after stop: ${JSON.stringify(stopped)}`);

  // The point of persisting it: the save survives a full page reload, on a page that has never run —
  // and comes back on the tier it was taken on, with the toggle left the other way round.
  await page.reload({ waitUntil: 'load' });
  await ready(page);
  await page.waitForFunction((s) => /^saved: /.test(document.querySelector(`${s} .save-label`).textContent),
    sel, { timeout: 15_000 });
  const reloaded = await row(page);
  !reloaded.load && reloaded.save && reloaded.label.includes(tier)
    ? ok(`save survives a reload: ${reloaded.label}`)
    : fail(`after reload: ${JSON.stringify(reloaded)}`);
  await page.evaluate(({ s, on }) => {
    const box = document.querySelector(`${s} .jit-toggle`);
    if (box) box.checked = on; // the *wrong* tier — the record must win
  }, { s: sel, on: !jit });

  // Load it back — a thaw, not a boot — and run the same `AFTER` frames forward. The thawed run's frame
  // 0 is the saved instant, so its frame `AFTER` must be the reference pixel for pixel.
  await page.click(`${sel} .save-load`);
  await page.waitForFunction((n) => (globalThis.__scrubState()?.tick ?? 0) >= n + 5, AFTER, { timeout: 60_000 });
  const thawLog = await page.evaluate((s) => document.querySelector(`${s} .log`).textContent, sel);
  thawLog.includes(jit ? 'wasm-JIT reactor thawed' : 'reactor thawed')
    ? ok(`thawed back onto the ${tier} tier despite the toggle`)
    : fail(`thaw tier: ${thawLog.split('\n').slice(-2).join(' | ')}`);

  const thawed = await seek(page, AFTER);
  thawed.at === AFTER && thawed.png === reference.png
    ? ok(`thawed guest resumed the frozen instant — frame ${AFTER} after the load ≡ frame ${SAVE_AT + AFTER} of the original`)
    : fail(`thaw mismatch: at=${thawed.at} same=${thawed.png === reference.png}`);

  // …and it is a real guest, not a frozen picture: it keeps moving.
  const moved = await seek(page, AFTER + 5);
  moved.png !== thawed.png
    ? ok('the thawed guest runs on from there')
    : fail('the thawed guest did not advance — it is a still, not a running reactor');

  await page.click(`${sel} .stop`);

  // Clear forgets it.
  await page.click(`${sel} .save-clear`);
  await page.waitForFunction((s) => document.querySelector(`${s} .save-label`).textContent === 'no saved state',
    sel, { timeout: 15_000 });
  const cleared = await row(page);
  cleared.load && cleared.clear
    ? ok('Clear forgets the save-state')
    : fail(`after clear: ${JSON.stringify(cleared)}`);
  await context.close();
}

try {
  await round('interpreter', false);
  await round('wasm-JIT', true);
} finally {
  await browser.close();
  server.close();
}

console.log(failed ? 'FAILED' : 'PASS');
process.exit(failed ? 1 : 0);
