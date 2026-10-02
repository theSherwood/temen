// #1954 — **an embedder's Release run through the shared JS driver**: `driveCoopTierupRun`'s hooks
// over a tier-up session opened with declared host-completed caps, a root-leaf mode and no regions —
// the shape c_interpret's Release button runs. The cases (`coop-embedder-cases.mjs`) run under Node
// and then in Chromium through Playwright when it is at hand.
//
// A program that asks the embedder `ping` 30 times gets each answer through `onCapPark` (answered
// asynchronously) and ends with the right value and stdout, handed over through `onOutput`. Where
// the host suspends (Chromium's JSPI) its root runs whole as one leaf and each park resumes the
// leaf's frames; where it cannot (Node), it runs interpreted. Resolving a park to `null` stops the
// run; so does `onSlice` resolving `false` on a program that never ends, after its output has streamed.
// A trap with `trapDeclines: false` returns its status, name and fault address; by default it throws.
// Under JSPI a leaf is sliced at its own safepoints (step 4): a leaf that never ends stops at a budget
// checkpoint when `onSlice` says so, and a long one hands its output over from inside, as it runs.
// A fault in a leaf's emitted code reports the trap its `env.trap` named (#1822), as the interpreter's.
//
// Usage:  node browser-coop-embedder-test.mjs [module.wasm]   (build the threads cdylib first)

import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { driveCoopTierupRun, suspendsLeaves } from './web/coop-driver.js';
import { engineImports } from './engine-imports.mjs';
import { runCases, pong, PINGS } from './coop-embedder-cases.mjs';
import { startServer } from './serve.mjs';

const ROOT = dirname(fileURLToPath(import.meta.url));
const wasmPath = process.argv[2]
  ?? join(ROOT, 'target/wasm32-unknown-unknown/release/temen_browser.wasm');
if (!existsSync(wasmPath)) {
  console.error(`SKIP: ${wasmPath} missing`);
  process.exit(0);
}

const fail = (msg) => { console.error(`FAIL: ${msg}`); process.exitCode = 1; };
let sum = 0n;
for (let i = 0n; i < BigInt(PINGS); i++) sum += pong(i);
const le8 = (v) => {
  const b = new Uint8Array(8);
  new DataView(b.buffer).setBigInt64(0, v, true);
  return new TextDecoder().decode(b);
};

const check = (host, r, suspends) => {
  const ran = (name, x, leaves, resumes) => {
    if (x.status !== 0 || x.value !== sum || x.out !== le8(sum)) {
      fail(`${host}: ${name}: status ${x.status}, value ${x.value}, ${x.out.length} stdout bytes (want 0, ${sum})`);
    }
    if (x.parks !== PINGS) fail(`${host}: ${name}: ${x.parks} parks (want ${PINGS})`);
    if (x.counts.leaves !== leaves || x.counts.resumes !== resumes) {
      fail(`${host}: ${name}: ${x.counts.leaves} leaves and ${x.counts.resumes} resumes (want ${leaves} and ${resumes})`);
    }
  };
  ran('pings', r.pings, suspends ? 1 : 0, suspends ? PINGS : 0);
  ran('pings, interpreted', r.pingsInterpreted, 0, 0);
  if (r.stopAtPark.status !== null || r.stopAtPark.parks !== 3) {
    fail(`${host}: a park answered null stops the run: status ${r.stopAtPark.status}, ${r.stopAtPark.parks} parks`);
  }
  if (r.spin.status !== null || r.spin.slices !== 20 || r.spin.out !== 'hi\n') {
    fail(`${host}: onSlice false stops an endless run: status ${r.spin.status}, ${r.spin.slices} slices, out ${JSON.stringify(r.spin.out)}`);
  }
  const ticks = 'tick\n'.repeat(50);
  const ticked = (name, x, leaves) => {
    if (x.status !== 0 || x.value !== 50n || x.out !== ticks || x.counts.leaves !== leaves) {
      fail(`${host}: ${name}: status ${x.status}, value ${x.value}, ${x.out.length} bytes out, ${x.counts.leaves} leaves`);
    }
    if (x.slices < 10 || x.chunks < 10) {
      fail(`${host}: ${name}: sliced and streamed as it ran: ${x.slices} slices, ${x.chunks} chunks`);
    }
  };
  ticked('ticks, interpreted', r.ticks, 0);
  if (suspends) {
    ticked('ticks, a sliced leaf', r.ticksLeaf, 1);
    const sl = r.spinLeaf;
    if (sl.status !== null || sl.slices !== 20 || sl.out !== 'hi\n' || sl.counts.leaves !== 1) {
      fail(`${host}: onSlice false stops an endless leaf at a checkpoint: status ${sl.status}, `
        + `${sl.slices} slices, out ${JSON.stringify(sl.out)}, ${sl.counts.leaves} leaves`);
    }
    const fl = r.faultsLeaf;
    if (fl.status !== 3 || fl.trap !== 'MemoryFault' || fl.out !== 'before\n' || fl.counts.leaves !== 1) {
      fail(`${host}: a fault in a leaf is its trap, not Unreachable (#1822): status ${fl.status}, `
        + `trap ${fl.trap}, out ${JSON.stringify(fl.out)}, ${fl.counts.leaves} leaves`);
    }
  }
  const f = r.faults;
  if (f.status !== 3 || f.trap !== 'MemoryFault' || f.addr !== 8n || f.out !== 'before\n') {
    fail(`${host}: a reported trap: status ${f.status}, trap ${f.trap}, addr ${f.addr}, out ${JSON.stringify(f.out)}`);
  }
  if (!/declined to the interpreter/.test(r.faultsDecline)) {
    fail(`${host}: a trap declines by default: ${r.faultsDecline}`);
  }
};

const mod = await WebAssembly.compile(readFileSync(wasmPath));
const memory = new WebAssembly.Memory({ initial: 2048, maximum: 16384, shared: true });
const { exports: ex } = await WebAssembly.instantiate(mod, engineImports(memory));
check('node', await runCases({ ex, memory, drive: driveCoopTierupRun, suspends: suspendsLeaves }),
  suspendsLeaves);

async function loadChromium() {
  for (const s of ['playwright', '/opt/node22/lib/node_modules/playwright/index.js']) {
    try { const m = await import(s); return m.chromium ?? m.default?.chromium; } catch {}
  }
  return null;
}
const chromium = await loadChromium();
if (chromium === null) {
  console.log('– the Chromium pass is skipped (playwright not found)');
} else {
  const { server, port } = await startServer(ROOT);
  const browser = await chromium.launch({ args: process.env.CI ? ['--no-sandbox'] : [] });
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', (e) => errors.push(String(e)));
    await page.goto(`http://127.0.0.1:${port}/web/play.html`);
    const { results, suspends } = await page.evaluate(async () => {
      const par = await import('./par.js');
      const { driveCoopTierupRun: drive, suspendsLeaves: suspends } = await import('./coop-driver.js');
      const cases = await import('../coop-embedder-cases.mjs');
      const { ex, memory } = await par.loadEngine(null);
      return { results: await cases.runCases({ ex, memory, drive, suspends }), suspends };
    });
    if (!suspends) fail('chromium: no JSPI (WebAssembly.Suspending / WebAssembly.promising)');
    check('chromium', results, suspends);
    if (errors.length) fail(`chromium: page errors: ${errors}`);
  } finally {
    await browser.close();
    server.close();
  }
}
if (process.exitCode) process.exit(process.exitCode);
console.log('ok — an embedder\'s Release run: declared caps answered through onCapPark, output through '
  + 'onOutput, slices through onSlice, stops at a park and at a slice, and a trap reported or declined'
  + `${chromium ? '; in Chromium the parking root ran as one leaf, suspended and resumed, and a leaf '
    + 'was sliced at its own budget checkpoints' : ''}`);
