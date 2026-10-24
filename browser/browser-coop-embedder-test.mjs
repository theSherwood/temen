// #1954 — **an embedder's Release run through the shared JS driver**: `driveCoopTierupRun`'s hooks
// over a tier-up session opened with declared host-completed caps, a root-leaf mode and no regions —
// the shape c_interpret's Release button runs. The cases (`coop-embedder-cases.mjs`) run under Node
// and then in Chromium through Playwright when it is at hand.
//
// A program that asks the embedder `ping` 30 times gets each answer through `onCapPark` (answered
// asynchronously) and ends with the right value and stdout, handed over through `onOutput`. Where
// the host suspends (Chromium's JSPI) its root runs whole as one leaf and each park resumes the
// leaf's frames; where it cannot (Node), it runs interpreted. The run goes on while an answer is owed:
// a root's timer fires in the page's time before an answer that comes later (#2557), and a lone wait
// takes its time (#2439). Resolving a park to `null` stops the run; so does `onSlice` resolving
// `false` on a program that never ends, after its output has streamed. JS capabilities the embedder
// staged are answered inside the call through `onCap`, with the guest's window, interpreted and from
// a leaf's bounce (#2579); executor 2 refuses a run that has one.
// A trap with `trapDeclines: false` returns its status, name and fault address; by default it throws.
// Under JSPI a leaf is sliced at its own safepoints (step 4): a leaf that never ends stops at a budget
// checkpoint when `onSlice` says so, and a long one hands its output over from inside, as it runs.
// A fault in a leaf's emitted code reports the trap its `env.trap` named (#1822), as the interpreter's.
//
// #1414 5b-2 — in Chromium the cases get an x2 row: the same embedder's runs on executor 2, each task
// on a Worker of its own, under the on-ramp powerbox. The page answers each call as it is told of it
// while the run's other tasks go on, including the thread whose answer a spinning root waits for
// (#2531's shape), and the root's timer fires while an answer is owed (#2557). It feeds stdin later:
// as the run idles on its reads, before its Worker opens it, and more at once than the engine's inbox
// holds. A park answered `null` stops the run, and a fault reports its address, as on the pump.
//
// Usage:  node browser-coop-embedder-test.mjs [module.wasm]   (build the threads cdylib first)

import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { driveCoopTierupRun, suspendsLeaves } from './web/coop-driver.js';
import { engineImports } from './engine-imports.mjs';
import { runCases, runX2Cases, pong, PINGS, bigFeed, LONE_WAIT_MS, JS_CAPS_OUT } from './coop-embedder-cases.mjs';
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
  // #2557: the root's timer fired while the answer was owed, so it read the thread's flag still down.
  const t = r.timerWhileOwed;
  const timed = 20000n + pong(7n);
  if (t.status !== 0 || t.value !== timed || t.parks !== 1) {
    fail(`${host}: a timer fires while an answer is owed: status ${t.status}, value ${t.value}, `
      + `${t.parks} parks (want 0, ${timed}, 1)`);
  }
  // #2439: a lone wait timed out (2) after its time, not at once.
  const w = r.loneWait;
  if (w.status !== 0 || w.value !== 2n || w.ms < LONE_WAIT_MS) {
    fail(`${host}: a lone ${LONE_WAIT_MS} ms wait: status ${w.status}, value ${w.value}, `
      + `after ${w.ms.toFixed(1)} ms`);
  }
  // #2579: the staged JS capabilities answered inside each call, and the uppercasing landed in the
  // guest's window: interpreted, and from the root's leaf.
  const jsCapped = (name, x, leaves) => {
    if (x.status !== 0 || x.value !== 12n || x.out !== JS_CAPS_OUT || x.capCalls !== 2
      || x.parks !== 0 || x.counts.leaves !== leaves) {
      fail(`${host}: ${name}: status ${x.status}, value ${x.value}, out ${JSON.stringify(x.out)}, `
        + `${x.capCalls} cap calls, ${x.parks} parks, ${x.counts.leaves} leaves `
        + `(want 0, 12, ${JSON.stringify(JS_CAPS_OUT)}, 2, 0, ${leaves})`);
    }
  };
  jsCapped('JS capabilities, interpreted', r.jsCaps, 0);
  jsCapped('JS capabilities from a leaf', r.jsCapsLeaf, 1);
  if (r.x2RefusesJsCaps === false) fail(`${host}: executor 2 took a run with a staged JS capability`);
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
    // #2126: the leaf's guard recorded the faulting address, the one the interpreted run reports.
    if (fl.addr !== 8n) fail(`${host}: a fault in a leaf reports its address (#2126): addr ${fl.addr}`);
    const dl = r.dividesLeaf;
    if (dl.status !== 3 || dl.trap !== 'DivByZero' || dl.out !== 'before\n' || dl.counts.leaves !== 1) {
      fail(`${host}: a native trap in a leaf is named from its message (#2126): status ${dl.status}, `
        + `trap ${dl.trap}, out ${JSON.stringify(dl.out)}, ${dl.counts.leaves} leaves`);
    }
  }
  const f = r.faults;
  if (f.status !== 3 || f.trap !== 'MemoryFault' || f.addr !== 8n || f.out !== 'before\n') {
    fail(`${host}: a reported trap: status ${f.status}, trap ${f.trap}, addr ${f.addr}, out ${JSON.stringify(f.out)}`);
  }
  if (r.divides.status !== 3 || r.divides.trap !== 'DivByZero') {
    fail(`${host}: an interpreted divide by zero: status ${r.divides.status}, trap ${r.divides.trap}`);
  }
  if (!/declined to the interpreter/.test(r.faultsDecline)) {
    fail(`${host}: a trap declines by default: ${r.faultsDecline}`);
  }
};

// The x2 row (Chromium only): the same embedder's runs on executor 2.
const checkX2 = (r) => {
  const ran = (name, x, want, parks) => {
    if (x.error !== undefined || x.value !== want) {
      fail(`chromium x2: ${name}: value ${x.value}, error ${x.error} (want ${want})`);
    }
    if (x.parks !== parks) fail(`chromium x2: ${name}: ${x.parks} parks (want ${parks})`);
  };
  ran('pings', r.pings, sum, PINGS);
  if (r.pings.out !== le8(sum)) fail(`chromium x2: pings: stdout ${JSON.stringify(r.pings.out)}`);
  if (r.stopAtPark.error !== 'stopped' || r.stopAtPark.parks !== 3) {
    fail(`chromium x2: a park answered null stops the run: ${r.stopAtPark.error}, ${r.stopAtPark.parks} parks`);
  }
  // The root never stops spinning until the answer reaches its thread: only an answer that reaches
  // the run while its Worker watches it ends this run.
  ran('a thread\'s call while the root spins (#2531)', r.threadPing, pong(7n), 1);
  ran('a timer fires while an answer is owed (#2557)', r.timerWhileOwed, 20000n + pong(7n), 1);
  const echoed = (name, x, out) => {
    if (x.error !== undefined || x.value !== BigInt(out.length) || x.out !== out) {
      fail(`chromium x2: ${name}: value ${x.value}, error ${x.error}, stdout ${JSON.stringify(x.out)}`);
    }
  };
  echoed('stdin fed as the run idles', r.echoLater, 'hello world\n');
  if (r.echoLater.idles < 2) fail(`chromium x2: the fed run idled ${r.echoLater.idles} times (want 2 or more)`);
  echoed('stdin fed before the run opens', r.echoEarly, 'early\n');
  echoed('stdin fed past what the inbox holds', r.echoBig, bigFeed());
  const f = r.faults;
  if (!/MemoryFault/.test(f.error ?? '') || f.faultAddr !== 8n || f.out !== 'before\n') {
    fail(`chromium x2: a reported trap: ${f.error}, addr ${f.faultAddr}, out ${JSON.stringify(f.out)}`);
  }
  // #1414 5c: a root the leaf emitter takes runs whole from its entry, its one tier-up, on the
  // Worker that runs it, which holds its frames while a call in them waits for the page.
  const leaf = (name, x) => {
    if (!x.leaf || x.tierups !== 1) {
      fail(`chromium x2: ${name}: leaf ${x.leaf}, ${x.tierups} tier-ups (want the root as a leaf, 1)`);
    }
  };
  ran('a root leaf\'s pings', r.leafPings, sum, PINGS);
  if (r.leafPings.out !== le8(sum)) fail(`chromium x2: a root leaf's pings: stdout ${JSON.stringify(r.leafPings.out)}`);
  leaf('pings', r.leafPings);
  echoed('a root leaf\'s stdin fed as it idles', r.leafEcho, 'hello world\n');
  leaf('stdin fed later', r.leafEcho);
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
    const { results, suspends, x2 } = await page.evaluate(async () => {
      const par = await import('./par.js');
      const { driveCoopTierupRun: drive, suspendsLeaves: suspends } = await import('./coop-driver.js');
      const cases = await import('../coop-embedder-cases.mjs');
      const eng = await par.loadEngine(null);
      const results = await cases.runCases({ ex: eng.ex, memory: eng.memory, drive, suspends });
      return { results, suspends, x2: await cases.runX2Cases({ par, eng }) };
    });
    if (!suspends) fail('chromium: no JSPI (WebAssembly.Suspending / WebAssembly.promising)');
    check('chromium', results, suspends);
    checkX2(x2);
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
    + 'was sliced at its own budget checkpoints; and on executor 2 the page answered each call while '
    + 'the run went on, and fed stdin as it idled' : ''}`);
