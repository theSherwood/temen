// Chromium smoke for the **snapshot worker** (WASM_AOT.md warm+JIT · issues #804/#805). Drives the real
// page: each warm card (QuickJS, Lua) runs its snapshot on a dedicated Worker (own engine + memory),
// pre-warmed off the main thread; each Run is a message round-trip. Asserts every warm card runs
// end-to-end on BOTH warm tiers (warm-snapshot + warm+JIT), that the work went *through the worker* (not
// a silent main-thread fallback — `globalThis.__snapshotWorkerRuns` increments), and that fresh-per-Run
// isolation holds. One worker per module means both cards stay warm at once.
//
// Reuses the wasm32 module built by the CI real-browser job (and `serve.mjs` for COOP/COEP). Run:
//   node snapshot-worker-test.mjs
import { startServer } from './serve.mjs';
import { benignAssetMiss } from './play-test-errors.mjs';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
if (!existsSync(join(HERE, 'target', 'wasm32-unknown-unknown', 'release', 'svm_browser.wasm'))) {
  console.log('SKIP: engine wasm not built (run the browser-real recipe first)');
  process.exit(0);
}

// Each warm card: its committed snapshot asset, and the three programs (plain / compute-heavy / a
// leak-check pair) with the exact stdout each should print. Only cards whose asset is committed run.
const CARDS = [
  {
    name: 'JavaScript (QuickJS — write & run JS)', asset: 'qjs_snapshot.svmb',
    plain: ['console.log("hi", 6 * 7);\n', 'hi 42'],
    heavy: ['let s=0;for(let i=0;i<200000;i++)s+=i;console.log("sum", s);\n', 'sum 19999900000'],
    leakSet: 'var leak = 4242; console.log("set", leak);\n',
    leakGet: 'try { console.log("leak?", typeof leak, leak); } catch (e) { console.log("clean", e.name); }\n',
    leakClean: (out) => out.includes('clean') && !out.includes('4242'),
  },
  {
    name: 'Lua (5.4.7 — write & run)', asset: 'lua_snapshot.svmb',
    plain: ['print("hi", 6 * 7)\n', 'hi\t42'],
    heavy: ['local s=0; for i=1,200000 do s=s+i end; print("sum", s)\n', 'sum\t20000100000'],
    leakSet: 'leaked = 4242; print("set", leaked)\n',
    leakGet: 'print("leak?", leaked)\n',
    leakClean: (out) => out.includes('nil') && !out.includes('4242'),
  },
  {
    // Tcl's warm+JIT declines (the 162-TU interp's emitted eval_run traps), so it's warm-snapshot-only
    // (`noJit`): no wasm-JIT toggle, no JIT pre-compile. The `tcl_snapshot.svmb` asset is deploy-built,
    // so this entry is filtered out (skipped) in the committed-asset CI job.
    name: 'Tcl (8.6 — write & run)', asset: 'tcl_snapshot.svmb', noJit: true,
    plain: ['puts [expr 6*7]\n', '42'],
    leakSet: 'set leaked 4242; puts "set $leaked"\n',
    leakGet: 'puts "exists? [info exists leaked]"\n',
    leakClean: (out) => out.includes('exists? 0') && !out.includes('4242'),
  },
].filter((c) => existsSync(join(HERE, 'web', 'assets', c.asset)));

const chromium = (await import('playwright')).chromium;
const { server, port } = await startServer(process.cwd());
const browser = await chromium.launch({ args: ['--no-sandbox'] });
let failed = false;
const ok = (m) => console.log(`  ok: ${m}`);
const fail = (m) => { failed = true; console.log(`  FAIL: ${m}`); };

try {
  const page = await browser.newPage();
  page.on('pageerror', (e) => fail(`pageerror: ${e.message}`));
  page.on('console', (m) => { if (m.type() === 'error' && !benignAssetMiss(m)) fail(`console.error: ${m.text()}`); });
  await page.goto(`http://127.0.0.1:${port}/web/play.html`, { waitUntil: 'load' });
  await page.waitForFunction(() => document.getElementById('engine-state').dataset.state === 'ready', { timeout: 30_000 });

  const sel = (c) => `[data-demo="${c.name}"]`;
  const setSrc = (c, code) => page.evaluate(({ s, code }) => document.querySelector(`${s} .CodeMirror`).CodeMirror.setValue(code), { s: sel(c), code });
  const workerRuns = () => page.evaluate(() => globalThis.__snapshotWorkerRuns || 0);
  const run = async (c, timeout = 30_000) => {
    await page.click(`${sel(c)} .run`);
    await page.waitForFunction((s) => ['done', 'error', 'stopped'].includes(document.querySelector(`${s} .state`).dataset.state), sel(c), { timeout });
    return {
      state: await page.getAttribute(`${sel(c)} .state`, 'data-state'),
      label: (await page.textContent(`${sel(c)} .state`)) || '',
      stdout: (await page.textContent(`${sel(c)} .stdout`)) || '',
    };
  };

  // Prewarm pre-compiles the warm+JIT `eval_run` off the main thread, so the FIRST wasm-JIT Run is a
  // cache hit rather than paying the ~12.7 MB `WebAssembly.compile`. Before running anything, wait for
  // each card's worker to finish pre-compiling and assert it compiled exactly once with no eval yet
  // (compiles=1, hits=0) — i.e. the compile happened during pre-warm, not on the first Run.
  const statsOf = (c) => page.evaluate((u) => globalThis.__snapshotClient?.stats(u), `./assets/${c.asset}`);
  for (const c of CARDS) {
    if (c.noJit) continue; // warm-snapshot-only cards (Tcl) don't pre-compile a JIT
    const tag = c.name.split(' ')[0];
    let st = null;
    for (let i = 0; i < 160; i++) { // up to ~40s: engine load + warmup + the background ~12.7 MB compile
      st = await statsOf(c);
      if (st && st.ok && st.jitPrimed) break;
      await new Promise((r) => setTimeout(r, 250));
    }
    st && st.ok && st.jitPrimed && st.compiles === 1 && st.hits === 0
      ? ok(`${tag}: prewarm pre-compiled warm+JIT (compiles=1, hits=0 before any Run → first JIT Run is a cache hit)`)
      : fail(`${tag}: prewarm did NOT prime the JIT compile: ${JSON.stringify(st)}`);
  }

  for (const c of CARDS) {
    const tag = c.name.split(' ')[0];
    // warm-snapshot (JIT toggle off, the default): a plain program runs on the card's worker.
    if (!c.noJit) await page.uncheck(`${sel(c)} .jit-label input`).catch(() => {});
    await setSrc(c, c.plain[0]);
    let before = await workerRuns();
    let r = await run(c);
    if (r.state === 'done' && r.stdout.includes(c.plain[1]) && r.label.includes('warm-snapshot')) ok(`${tag}: warm-snapshot on the worker → ${JSON.stringify(r.stdout.trim())}`);
    else fail(`${tag} warm-snapshot: state=${r.state} label=${r.label} stdout=${JSON.stringify(r.stdout)}`);
    (await workerRuns()) > before ? ok(`${tag}: ran on the worker (counter +1)`) : fail(`${tag}: did NOT run on the worker — silent main-thread fallback`);

    // warm+JIT (tick the toggle): a compute-heavy program evals near-native on the worker. Skipped for a
    // warm-snapshot-only card (Tcl) — it has no wasm-JIT toggle.
    if (!c.noJit) {
      await page.check(`${sel(c)} .jit-label input`);
      await setSrc(c, c.heavy[0]);
      before = await workerRuns();
      r = await run(c);
      if (r.state === 'done' && r.stdout.includes(c.heavy[1]) && r.label.includes('warm+JIT')) ok(`${tag}: warm+JIT on the worker → ${JSON.stringify(r.stdout.trim())}`);
      else fail(`${tag} warm+JIT: state=${r.state} label=${r.label} stdout=${JSON.stringify(r.stdout)}`);
      (await workerRuns()) > before ? ok(`${tag}: warm+JIT ran on the worker (counter +1)`) : fail(`${tag}: warm+JIT did NOT run on the worker`);
    }

    // fresh-per-Run isolation: a global in one Run must not leak into the next (snapshot restored each Run).
    await setSrc(c, c.leakSet);
    r = await run(c);
    if (!r.stdout.includes('set') || !r.stdout.includes('4242')) fail(`${tag} isolation setup: ${JSON.stringify(r.stdout)}`);
    await setSrc(c, c.leakGet);
    r = await run(c);
    r.state === 'done' && c.leakClean(r.stdout) ? ok(`${tag}: fresh-per-Run isolation holds (no leak across Runs)`) : fail(`${tag} isolation: state=${r.state} stdout=${JSON.stringify(r.stdout)}`);
  }

  // After all the Runs, each JIT-capable card's compile count is still 1 — the warm+JIT Run reused the
  // pre-compiled instance instead of recompiling (a hit was recorded). Warm-snapshot-only cards (Tcl)
  // never compiled a JIT (compiles=0).
  for (const c of CARDS) {
    const tag = c.name.split(' ')[0];
    const st = await statsOf(c);
    if (c.noJit) {
      st && st.ok && st.compiles === 0 ? ok(`${tag}: warm-snapshot-only — no JIT compiled (compiles=0)`) : fail(`${tag}: unexpected JIT compile: ${JSON.stringify(st)}`);
    } else {
      st && st.ok && st.compiles === 1 && st.hits >= 1
        ? ok(`${tag}: warm+JIT Run reused the pre-compiled instance (compiles=1, hits=${st.hits})`)
        : fail(`${tag}: unexpected JIT recompile: ${JSON.stringify(st)}`);
    }
  }

  const total = await workerRuns();
  const expect = CARDS.reduce((n, c) => n + (c.noJit ? 3 : 4), 0); // noJit cards do 3 Runs (no warm+JIT)
  total >= expect ? ok(`all ${total} Runs across ${CARDS.length} warm card(s) went through the snapshot workers`) : fail(`only ${total} worker Runs (expected ≥${expect})`);
} catch (e) {
  fail(`exception: ${e.message}`);
} finally {
  await browser.close();
  await new Promise((r) => server.close(r));
}

console.log(failed ? 'FAIL — snapshot worker' : 'PASS — snapshot worker: warm-snapshot + warm+JIT on dedicated workers (QuickJS + Lua), isolation holds');
process.exit(failed ? 1 : 0);
