// Mainline wasm-JIT tier-up over a LIVE window (WASM_AOT.md slice 2 + the slice-0 JACL residual). The
// existing `tierup` index case covers the `thread.spawn` shape (fresh activation, argv marshalled); this
// covers the shape the JACL consumer's SVM_BROWSER_TIERUP_FINDINGS.md flagged and that slice 2 turns on
// for SVM-text compute cards: a **mainline** root frame that loops calling a hot leaf which reads a DATA
// SEGMENT out of the live, mid-computation window. Run the SAME guest plain (all-interp) and with tier-up
// across real Workers; both must return the same value AND tier-up must actually fire (tierups > 0) — a
// value match with tierups==0 would be a vacuous pass (nothing tiered up). A stale/mis-based window or
// unmaterialized data would make the emitted leaf compute the wrong sum, caught here.
import { startServer } from './serve.mjs';
import { fileURLToPath } from 'node:url';
import { dirname } from 'node:path';
const ROOT = dirname(fileURLToPath(import.meta.url));
async function loadChromium() {
  for (const s of ['playwright', '/opt/node22/lib/node_modules/playwright/index.js']) {
    try { const m = await import(s); return m.chromium ?? m.default?.chromium; } catch {}
  }
  throw new Error('playwright not found');
}
const chromium = await loadChromium();
const { server, port } = await startServer(ROOT);
const browser = await chromium.launch({ args: process.env.CI ? ['--no-sandbox'] : [] });
const page = await browser.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(String(e)));
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
await page.goto(`http://127.0.0.1:${port}/web/play.html`);

// Mainline root (func 0) loops N times calling the eligible leaf (func 1), which **round-trips through
// the live window** — stores i at window addr 4096, loads it back, ·7 — so the emitted leaf reads/writes
// the same mid-computation window the interpreter root holds (the JACL "mainline over a live window"
// shape). func 1 is a pure all-i64 leaf → tier-up-eligible; func 0 interp-drives and its `call 1` tiers
// up. The **safety criterion is differential**: the tier-up value must equal the all-interp value
// (INVARIANT 9), and tier-up must actually fire. Σ_{i=0}^{N-1} i·7 = 7·N·(N-1)/2 is the expected value
// when the window round-trip is faithful (reported informationally; the differential is the gate).
const N = 50000;
const WANT = BigInt(7 * N * (N - 1) / 2); // 8749825000
const SRC = `
memory 16
func () -> (i64) {
block 0 () {
  vi0 = i64.const 0
  vacc0 = i64.const 0
  br 1 (vi0, vacc0)
}
block 1 (vi: i64, vacc: i64) {
  vn = i64.const ${N}
  vlt = i64.lt_u vi vn
  br_if vlt 2(vi, vacc) 3(vacc)
}
block 2 (vi2: i64, vacc2: i64) {
  vr = call 1 (vi2)
  vacc3 = i64.add vacc2 vr
  v1 = i64.const 1
  vi3 = i64.add vi2 v1
  br 1(vi3, vacc3)
}
block 3 (vaccf: i64) {
  return vaccf
}
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  va = i64.const 4096
  i64.store va v0
  vld = i64.load va
  vk = i64.const 7
  vm = i64.mul vld vk
  return vm
}
}
`;

const res = await page.evaluate(async ({ src }) => {
  const par = await import('./par.js');
  const eng = await par.loadEngine();
  const run = par.makeRunner(eng);
  const u8 = () => new Uint8Array(eng.memory.buffer);

  // Parse SVM text → module bytes in-sandbox, exactly like the playground's SVM-text card.
  const srcBytes = new TextEncoder().encode(src);
  const p = eng.ex.svm_alloc(srcBytes.length);
  u8().set(srcBytes, p);
  const ok = eng.ex.svm_parse(p, srcBytes.length);
  const out = u8().slice(eng.ex.svm_parse_ptr(), eng.ex.svm_parse_ptr() + eng.ex.svm_parse_len());
  eng.ex.svm_dealloc(p, srcBytes.length);
  if (ok !== 1) return { parseError: new TextDecoder().decode(out) };
  const guest = out;

  const plain = await run(guest, {});                 // all-interpreter
  const tiered = await run(guest, { tierup: true });  // mainline tier-up over the live window
  return {
    plain: plain.value.toString(),
    tiered: tiered.value.toString(),
    tierups: tiered.tierups,
    workers: tiered.started,
  };
}, { src: SRC });

await browser.close();
await new Promise((r) => server.close(r));

let ok = true;
if (res.parseError) {
  console.log('  parse error:', res.parseError);
  ok = false;
} else {
  // The gate is the differential (tier-up ≡ interpreter) plus non-vacuity (tier-up actually fired).
  const diffOk = res.plain === res.tiered;
  const firedOk = res.tierups > 0;
  ok = diffOk && firedOk;
  const wantNote = res.tiered === WANT.toString() ? ' (== expected window round-trip)' : '';
  console.log(
    `  mainline tier-up: interp=${res.plain} tier-up=${res.tiered}${wantNote}` +
    ` · ${res.tierups} regions on emitted wasm across ${res.workers} Worker(s)` +
    (ok ? '' : `  <-- FAIL${diffOk ? '' : ' (interp≠tier-up — INVARIANT 9 violation)'}${firedOk ? '' : ' (nothing tiered up)'}`)
  );
}
if (errors.length) { console.log('page errors:', errors); ok = false; }
console.log(ok ? 'PASS — mainline tier-up over a live window ≡ interpreter' : 'FAIL');
process.exit(ok ? 0 : 1);
