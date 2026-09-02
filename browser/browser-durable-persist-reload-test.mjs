// Real-browser (V8) end-to-end for #816 Slice C (invariant 14, **durability axis**, cross-host leg):
// a durability-instrumented, `vm_map`-GROWN guest **frozen to a §12 snapshot artifact**, persisted to
// **IndexedDB**, and — after a genuine **page reload** into a fresh WebAssembly instance — **thawed
// and resumed to completion**, with the grown-page content surviving the whole round-trip.
//
// This is the shipped-path proof of the browser "persist a warmed/grown guest across a reload"
// consumer: the native oracle (`crates/temen/tests/durable_grown_snapshot_resume.rs`) pins the
// mechanism; this proves the `temen_durable_freeze` / `temen_durable_thaw_resume` cdylib FFI drives
// it in real V8, across the IndexedDB persistence boundary and a real reload (fresh linear memory).
//
// The guest: grow `[128 KiB, 192 KiB)` Rw, store marker 77 into a grown page (143367, above the
// declared 128 KiB window), self-flip the state word to UNWINDING just before the clock read (the
// `multipoint.rs` freeze device), then reload the marker after the call and return clock + marker.
// Baseline (clock 42): 42 + 77 = 119. A correct thaw REPLAYS the captured clock (42), so seeding the
// thaw clock to 9999 and still getting 119 proves both the replay AND that the grown-page marker
// rode the snapshot artifact across the reload (a lost grown page would give 42; a re-issued clock 9999+).
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
// Only real JS/page errors gate the run — a benign resource 404 (a deploy-built play card asset
// absent in this job, e.g. `tcl_snapshot.temen`) is not this test's concern.
const errors = [];
const benign = (t) => /Failed to load resource|status of 404/i.test(t);
page.on('pageerror', (e) => errors.push(String(e)));
page.on('console', (m) => { if (m.type() === 'error' && !benign(m.text())) errors.push(m.text()); });

const GUEST_SRC = `memory 17
func (i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32) {
  voff = i64.const 131072
  vlen = i64.const 65536
  vprot = i32.const 3
  vg = call.cap 5 0 (i64, i64, i32) -> (i64) v0 (voff, vlen, vprot)
  vaddr = i64.const 143367
  vmark = i64.const 77
  i64.store vaddr vmark
  vsa = i64.const 16384
  vsu = i32.const 1
  i32.store vsa vsu
  vz = i32.const 0
  vc = call.cap 2 0 (i32) -> (i64) v1 (vz)
  vld = i64.load vaddr
  vsum = i64.add vc vld
  return vsum
  }
}
`;
const RESERVED_LOG2 = 20; // 1 MiB mask domain (declared 128 KiB, grows into the tail)
const FREEZE_CLOCK = 42;
const THAW_CLOCK = 9999; // different on purpose — a re-issue instead of a replay would surface it
const WANT = 119; // 42 (replayed clock) + 77 (grown-page marker reloaded across the reload)
const DB = 'temen-slice-c', STORE = 'kv', KEY = 'snapshot';

// ---- Phase A: freeze the grown durable guest and persist the snapshot artifact to IndexedDB. ----
await page.goto(`http://127.0.0.1:${port}/web/play.html`);
const a = await page.evaluate(async ({ src, reserved, clock, db, store, key }) => {
  const { loadEngine } = await import('./par.js');
  const eng = await loadEngine();
  const ex = eng.ex, memory = eng.memory;
  const u8 = () => new Uint8Array(memory.buffer);
  const slice = (p, n) => u8().slice(p, p + n);
  const parse = (s) => {
    const b = new TextEncoder().encode(s);
    const p = ex.temen_alloc(b.length);
    u8().set(b, p);
    const ok = ex.temen_parse(p, b.length);
    const out = slice(ex.temen_parse_ptr(), ex.temen_parse_len());
    ex.temen_dealloc(p, b.length);
    if (ok !== 1) throw new Error('parse: ' + new TextDecoder().decode(out));
    return out;
  };
  const idbPut = (val) => new Promise((res, rej) => {
    const r = indexedDB.open(db, 1);
    r.onupgradeneeded = () => r.result.createObjectStore(store);
    r.onsuccess = () => { const tx = r.result.transaction(store, 'readwrite');
      tx.objectStore(store).put(val, key); tx.oncomplete = () => res(); tx.onerror = () => rej(tx.error); };
    r.onerror = () => rej(r.error);
  });
  const guest = parse(src);
  const gp = ex.temen_alloc(guest.length);
  u8().set(guest, gp);
  const status = ex.temen_durable_freeze(gp, guest.length, BigInt(clock), reserved);
  ex.temen_dealloc(gp, guest.length);
  // Copy the artifact OUT of wasm memory (structured-clone into IndexedDB — independent of this
  // instance's linear memory, which the reload discards).
  const art = slice(Number(ex.temen_durable_art_ptr()), ex.temen_durable_art_len());
  await idbPut(art);
  return { status, artLen: art.length };
}, { src: GUEST_SRC, reserved: RESERVED_LOG2, clock: FREEZE_CLOCK, db: DB, store: STORE, key: KEY });

// ---- The reload: a fresh page context + a fresh WebAssembly instance (new linear memory). The
// snapshot bytes survive only because they live in IndexedDB, not in the discarded wasm memory. ----
await page.reload();

// ---- Phase B: thaw the artifact from IndexedDB into the fresh instance and resume to completion. ----
const b = await page.evaluate(async ({ src, clock, db, store, key }) => {
  const { loadEngine } = await import('./par.js');
  const eng = await loadEngine();
  const ex = eng.ex, memory = eng.memory;
  const u8 = () => new Uint8Array(memory.buffer);
  const slice = (p, n) => u8().slice(p, p + n);
  const parse = (s) => {
    const b = new TextEncoder().encode(s);
    const p = ex.temen_alloc(b.length);
    u8().set(b, p);
    const ok = ex.temen_parse(p, b.length);
    const out = slice(ex.temen_parse_ptr(), ex.temen_parse_len());
    ex.temen_dealloc(p, b.length);
    if (ok !== 1) throw new Error('parse: ' + new TextDecoder().decode(out));
    return out;
  };
  const idbGet = () => new Promise((res, rej) => {
    const r = indexedDB.open(db, 1);
    r.onupgradeneeded = () => r.result.createObjectStore(store);
    r.onsuccess = () => { const tx = r.result.transaction(store, 'readonly');
      const g = tx.objectStore(store).get(key); g.onsuccess = () => res(g.result); g.onerror = () => rej(g.error); };
    r.onerror = () => rej(r.error);
  });
  const art = await idbGet();
  if (!art) return { error: 'artifact missing from IndexedDB after reload' };
  const guest = parse(src); // same raw guest — the FFI re-instruments it (deterministic digest)
  const gp = ex.temen_alloc(guest.length);
  u8().set(guest, gp);
  const ap = ex.temen_alloc(art.length);
  u8().set(art, ap);
  const result = Number(ex.temen_durable_thaw_resume(gp, guest.length, ap, art.length, BigInt(clock)));
  const status = ex.temen_status();
  ex.temen_dealloc(gp, guest.length);
  ex.temen_dealloc(ap, art.length);
  return { status, result, artLen: art.length };
}, { src: GUEST_SRC, clock: THAW_CLOCK, db: DB, store: STORE, key: KEY });

await browser.close();
await new Promise((r) => server.close(r));

console.log('RESULT', JSON.stringify({ freeze: a, thaw: b }, null, 2));
if (errors.length) console.log('ERRORS', errors.slice(0, 5));

const ok = errors.length === 0
  && a.status === 0          // STATUS_OK: the guest actually froze mid-run (non-vacuous)
  && a.artLen > 32           // a real snapshot artifact was produced and persisted
  && !b.error
  && b.status === 0          // STATUS_OK: the thaw resumed to completion
  && b.result === WANT;      // grown-page marker + clock replay survived the reload
console.log(
  `  freeze status=${a.status} artLen=${a.artLen} · thaw status=${b.status} result=${b.error || b.result}` +
  ` (want ${WANT})` + (ok ? '  ✓' : '  <-- FAIL')
);
console.log(ok ? 'PASS — grown durable guest persisted to IndexedDB and resumed across a reload (in V8)' : 'FAIL');
process.exit(ok ? 0 : 1);
