// Chromium gate for the playground's **reactor scrub bar** — time travel over a running guest (#1457,
// the #1454 moment design). Drives the real page: run the `bounce` reactor, scrub back through the
// keyframe ladder, and assert the frames that come back are the frames that were there.
//
// **Scope.** This gates the page *wiring*: that a ladder is built while the run goes, that input is
// taped, that the DOM controls reach the engine's moments, that a seek is reproducible and agrees with
// stepping, that Resume abandons the scrubbed-past future, that the ring stays bounded, and that Stop
// tears down a paused reactor. What it deliberately does not re-prove is replay *fidelity* — that the
// frames a re-run produces are the frames the live run produced. That needs the original live frames
// sampled at an exact tick, which you cannot do from outside a 60 fps rAF loop; it is gated instead,
// rigorously and on both tiers, by `browser/tests/reactor_moment.rs` and `jit_reactor_moment.rs`.
//
// Reuses the wasm32 module built by the CI real-browser job (and `serve.mjs` for COOP/COEP). Run:
//   cargo build --release --lib --target wasm32-unknown-unknown
//   node browser-play-scrub-test.mjs
import { startServer } from './serve.mjs';
import { benignAssetMiss } from './play-test-errors.mjs';

const chromium = (await import('playwright')).chromium;
const { server, port } = await startServer(process.cwd());
const browser = await chromium.launch({ args: ['--no-sandbox'] });
let failed = false;
const ok = (m) => console.log(`  ok: ${m}`);
const fail = (m) => { failed = true; console.log(`  FAIL: ${m}`); };

const DEMO = 'bounce (interactive — arrow keys)';
const card = (name) => `[data-demo="${name}"]`;
const sel = card(DEMO);

try {
  const page = await browser.newPage();
  page.on('pageerror', (e) => fail(`pageerror: ${e.message}`));
  page.on('console', (m) => { if (m.type() === 'error' && !benignAssetMiss(m)) fail(`console.error: ${m.text()}`); });
  await page.goto(`http://127.0.0.1:${port}/web/play.html`, { waitUntil: 'load' });
  await page.waitForFunction(
    () => document.getElementById('engine-state').dataset.state === 'ready',
    { timeout: 30_000 },
  );

  // The scrub row exists on the reactor card and is hidden until that card's reactor is live.
  const before = await page.evaluate((s) => {
    const row = document.querySelector(`${s} .scrub`);
    return { present: !!row, hidden: row ? row.hidden : null };
  }, sel);
  before.present && before.hidden === true
    ? ok('scrub row present and hidden before the run')
    : fail(`scrub row before run: ${JSON.stringify(before)}`);

  // Run the reactor and let it get well past a couple of keyframe strides (stride is 30 frames).
  await page.click(`${sel} .run`);
  await page.waitForFunction(() => (globalThis.__scrubState()?.tick ?? 0) >= 75, { timeout: 30_000 });

  const live = await page.evaluate((s) => ({
    hidden: document.querySelector(`${s} .scrub`).hidden,
    state: globalThis.__scrubState(),
  }), sel);
  !live.hidden && live.state.keyframes.length >= 3 && live.state.keyframes[0] === 0
    ? ok(`ladder live: ${live.state.keyframes.length} keyframes at ${live.state.keyframes.join(',')}`)
    : fail(`ladder: ${JSON.stringify(live)}`);

  // Steer the guest mid-run, so the tape has input in it and a faithful re-run has to replay it.
  await page.keyboard.press('ArrowLeft');
  await page.waitForFunction(() => (globalThis.__scrubState()?.tick ?? 0) >= 100, { timeout: 30_000 });
  const taped = await page.evaluate(() => globalThis.__scrubState().taped);
  taped.length > 0
    ? ok(`input taped at frame(s) ${taped.join(',')}`)
    : fail('the steer was not taped — a re-run would replay an unsteered guest');

  // Seek to a frame, twice, from different starting points: the pixels must be identical. This is the
  // whole property — a moment plus the tape reproduces the frame, whatever path you took to it.
  const seek = (t) => page.evaluate((target) => {
    const row = document.querySelector('.scrub:not([hidden]) .scrub-range');
    row.value = String(target);
    row.dispatchEvent(new Event('input', { bubbles: true }));
    const canvas = document.querySelector('.demo .canvas:not([hidden])');
    return { at: globalThis.__scrubState().viewing, png: canvas.toDataURL() };
  }, t);

  const a = await seek(60);
  const detour = await seek(95);
  const b = await seek(60);
  a.at === 60 && b.at === 60 && a.png === b.png && a.png !== detour.png
    ? ok('seek(60) → seek(95) → seek(60) reproduces frame 60 exactly')
    : fail(`seek reproducibility: at=${a.at}/${b.at} same=${a.png === b.png} moved=${a.png !== detour.png}`);

  // Stepping one frame forward from 60 must equal seeking straight to 61 — the re-run and the live path
  // agree on what frame 61 is.
  const stepped = await page.evaluate((s) => {
    document.querySelector(`${s} .scrub .scrub-fwd`).click();
    const canvas = document.querySelector('.demo .canvas:not([hidden])');
    return { at: globalThis.__scrubState().viewing, png: canvas.toDataURL() };
  }, sel);
  const direct = await seek(61);
  stepped.at === 61 && direct.at === 61 && stepped.png === direct.png
    ? ok('step-forward from 60 ≡ seek straight to 61')
    : fail(`step vs seek: stepped=${stepped.at} direct=${direct.at} same=${stepped.png === direct.png}`);

  // Resume plays on from the scrubbed frame and abandons the future that was scrubbed past: the tick
  // count resumes from 61 rather than jumping back to 100, and the tape past it is gone.
  await page.click(`${sel} .scrub-resume`);
  await page.waitForFunction(() => (globalThis.__scrubState()?.tick ?? 0) >= 70, { timeout: 30_000 });
  const resumed = await page.evaluate(() => globalThis.__scrubState());
  resumed.viewing === null && resumed.tick >= 70 && resumed.tick < 100 && resumed.taped.every((t) => t <= 61)
    ? ok(`resumed live from the scrubbed frame (tick ${resumed.tick}, tape truncated)`)
    : fail(`resume: ${JSON.stringify(resumed)}`);

  // The ring is bounded — a long run must not accumulate 16 MiB keyframes forever.
  await page.waitForFunction(() => (globalThis.__scrubState()?.tick ?? 0) >= 340, { timeout: 60_000 });
  const ring = await page.evaluate(() => globalThis.__scrubState());
  ring.keyframes.length <= 8 && ring.keyframes[0] > 0
    ? ok(`ladder ring bounded at ${ring.keyframes.length} keyframes (oldest now frame ${ring.keyframes[0]})`)
    : fail(`ring: ${JSON.stringify(ring)}`);

  // Stopping a paused reactor still tears it down (a paused run has no pending frame callback, which is
  // exactly the case the old "no rAF ⇒ nothing open" shortcut got wrong).
  await seek(ring.keyframes[0]);
  await page.click(`${sel} .stop`);
  const stopped = await page.evaluate((s) => ({
    scrub: globalThis.__scrubState(),
    hidden: document.querySelector(`${s} .scrub`).hidden,
    stopDisabled: document.querySelector(`${s} .stop`).disabled,
  }), sel);
  stopped.scrub === null && stopped.hidden
    ? ok('Stop tears down a paused reactor and releases its keyframes')
    : fail(`stop while paused: ${JSON.stringify(stopped)}`);
} finally {
  await browser.close();
  server.close();
}

console.log(failed ? 'FAILED' : 'PASS');
process.exit(failed ? 1 : 0);
