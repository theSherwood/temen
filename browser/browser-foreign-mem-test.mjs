// Real-browser (V8) gate for **`Region::Foreign`** (#1284, DETACHED_JIT.md §3.3): the engine cdylib
// addresses a JS-owned `WebAssembly.Memory`, a detached child's window, through the functions of a small
// accessor instance over it, which `web/foreign-mem.js` puts in the engine's own function table: no
// JavaScript runs on any access (#1414 5d). Checks, on a page with multi-memory and again on one told to
// do without it (`__temenNoMultiMemory`, as Safari must): (1) the in-tree 20k-op differential
// (`temen_mem::differential`) runs the proxied region against the safe `Paged` reference over a REAL
// second shared Memory, byte for byte; (2) a write from the engine lands in the child's memory, not the
// engine's; (3) a span moves by the copier where the page has multi-memory and by the engine's own word
// loops where it hasn't (the table's `read`/`write` slots); (4) of the foreign imports, the engine takes
// only `foreign_mint`. Then the cost gate: per-access timings of words, bytes, atomics and 64-byte spans
// over Foreign vs the flat `Shared` backing, on the multi-memory page, reported as a ratio.
import { startServer } from './serve.mjs';
import { fileURLToPath } from 'node:url';
import { dirname } from 'node:path';
import { existsSync } from 'node:fs';
const ROOT = dirname(fileURLToPath(import.meta.url));
async function loadChromium() {
  for (const s of ['playwright', '/opt/node22/lib/node_modules/playwright/index.js']) {
    try { const m = await import(s); return m.chromium ?? m.default?.chromium; } catch {}
  }
  throw new Error('playwright not found');
}
if (!existsSync(`${ROOT}/target/wasm32-unknown-unknown/release/temen_browser.wasm`)) {
  console.log('SKIP: threads wasm absent'); process.exit(0);
}
const chromium = await loadChromium();
const { server, port } = await startServer(ROOT);
const browser = await chromium.launch({ args: process.env.CI ? ['--no-sandbox'] : [] });
const errors = [];

/** Run the checks on a fresh page; `noMultiMemory` makes it move spans as a browser without multi-memory. */
async function runPage({ noMultiMemory, bench }) {
  const page = await browser.newPage();
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  if (noMultiMemory) await page.addInitScript(() => { globalThis.__temenNoMultiMemory = true; });
  await page.goto(`http://127.0.0.1:${port}/web/play.html`);
  const res = await page.evaluate(async (bench) => {
    const par = await import('./par.js');
    const { registerForeign } = await import('./foreign-mem.js');
    const eng = await par.loadEngine();
    const ex = eng.ex, memory = eng.memory;
    const readOut = () => new TextDecoder().decode(
      new Uint8Array(memory.buffer).slice(Number(ex.temen_stdout_ptr()), Number(ex.temen_stdout_ptr()) + ex.temen_stdout_len()));
    const foreignImports = WebAssembly.Module.imports(eng.module)
      .filter((i) => i.name.startsWith('foreign_')).map((i) => i.name);

    // (1) differential over a real second shared Memory: 3 pages (192 KiB), 20k ops, page 4096 (the
    // Paged reference's chunk boundary is exercised). 0 = agree; else the first divergence is in stdout.
    const size = 3 * 65536;
    const child = new WebAssembly.Memory({ initial: 3, maximum: 8, shared: true });
    const id = registerForeign(child);
    const diff = ex.temen_foreign_selftest(id, size, 20000);
    const diffMsg = diff === 0 ? '' : readOut();
    // (2) a write from the engine lands in the CHILD memory (a fresh one, untouched by the
    // differential), not the engine's: the isolation half.
    const fresh = new WebAssembly.Memory({ initial: 1, maximum: 1, shared: true });
    const fid = registerForeign(fresh);
    const probe = new Uint32Array(fresh.buffer);
    const before = probe[0x2000 / 4];
    ex.temen_foreign_poke(fid, 0x2000, 0xC0FFEE);
    const landed = probe[0x2000 / 4] === 0xC0FFEE && before === 0;
    // (3) the span slots, `read` and `write`, the first two of the 8 each memory takes. `fid` is the
    // last registered, so the table ends at its slots.
    const table = ex.__indirect_function_table, lo = table.length - 8;
    const wordSpans = table.get(lo) === ex.temen_foreign_read_words
      && table.get(lo + 1) === ex.temen_foreign_write_words;

    // The cost gate: kind 0/1 = write_word + read_word over Shared/Foreign, 2/3 = set_byte + byte, 4/5 =
    // an 8-byte atomic fetch-add, 6/7 = a 64-byte write_from + read_into. Same seeded offset stream and
    // checksum on both backings.
    const cost = {};
    if (bench) {
      const time = (kind, iters) => {
        ex.temen_foreign_bench(id, size, kind, 1000); // warm
        const t0 = performance.now();
        const sum = ex.temen_foreign_bench(id, size, kind, iters);
        return { ns: (performance.now() - t0) * 1e6 / iters, sum: String(sum) };
      };
      for (const [name, k, iters] of [['word', 0, 2e6], ['byte', 2, 2e6], ['atomic', 4, 1e6], ['span64', 6, 1e6]]) {
        const sh = time(k, iters), fo = time(k + 1, iters);
        cost[name] = { sharedNs: sh.ns, foreignNs: fo.ns, ratio: fo.ns / sh.ns, sumEq: sh.sum === fo.sum };
      }
    }
    return { foreignImports, diff, diffMsg, landed, wordSpans, cost };
  }, bench);
  await page.close();
  return res;
}

const mm = await runPage({ noMultiMemory: false, bench: true });
const nomm = await runPage({ noMultiMemory: true, bench: false });
await browser.close(); server.close();
console.log('RESULT', JSON.stringify({ mm, nomm }, null, 2));
if (errors.length) console.log('ERRORS', errors.slice(0, 6));
const f = (x) => x.toFixed(1);
let ok = mm.foreignImports.join() === 'foreign_mint';
console.log(`  foreign imports: ${mm.foreignImports.join(', ')}`);
for (const [label, r, words] of [['multi-memory', mm, false], ['no multi-memory', nomm, true]]) {
  ok &&= r.diff === 0 && r.landed && r.wordSpans === words;
  console.log(`  ${label}: differential=${r.diff === 0 ? 'agree' : 'DIVERGE ' + r.diffMsg} landedInChild=${r.landed} spans=${r.wordSpans ? 'word loops' : 'copier'}`);
}
for (const [name, c] of Object.entries(mm.cost)) {
  ok &&= c.sumEq;
  console.log(`  ${name.padEnd(6)} access: Shared ${f(c.sharedNs)} ns · Foreign ${f(c.foreignNs)} ns · ×${f(c.ratio)}`);
}
console.log(ok ? 'PASS — Region::Foreign over a real second WebAssembly.Memory agrees with the safe reference byte-for-byte, with and without multi-memory' : 'FAIL');
process.exit(ok ? 0 : 1);
