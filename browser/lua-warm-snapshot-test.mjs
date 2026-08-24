// V8 (Node) check of the **Lua warm-runtime snapshot** driver (issue #805), the Lua twin of
// `warm-snapshot-test.mjs`. Drives the shipping engine FFI (`temen_warm_open`/`temen_warm_eval`) plus the
// warm+JIT path (`runWarmJit`) over the committed `lua_snapshot.temen` (built from
// `lua_snapshot_harness.c`), and asserts:
//   - warm `eval_run` over the restored snapshot matches the cold `_start` (`temen_run_onramp`) output
//     byte-for-byte while skipping the Lua runtime rebuild,
//   - warm+JIT (`eval_run` emitted to wasm) matches too,
//   - fresh-per-Run isolation holds (a global in one Run can't leak into the next).
//
//   node lua-warm-snapshot-test.mjs [temen_browser.wasm] [lua_snapshot.temen]
//
// Exits 0 on success, 1 on any mismatch; SKIPs cleanly if the asset isn't built.
import { readFileSync, existsSync } from 'node:fs';
import { engineImports } from './engine-imports.mjs';
import { runWarmJit } from './web/wasmjit-module.js';

const wasmPath = process.argv[2] ?? 'target/wasm32-unknown-unknown/release/temen_browser.wasm';
const modPath = process.argv[3] ?? 'web/assets/lua_snapshot.temen';
if (!existsSync(modPath)) {
  console.log(`SKIP: ${modPath} not built (Lua on-ramp asset absent)`);
  process.exit(0);
}

const mod = await WebAssembly.compile(readFileSync(wasmPath));
const memory = new WebAssembly.Memory({ initial: 2048, maximum: 16384, shared: true });
const ex = (await WebAssembly.instantiate(mod, engineImports(memory))).exports;
const membuf = () => (ex.memory ?? memory).buffer;
const put = (bytes) => { const p = ex.temen_alloc(bytes.length); new Uint8Array(membuf()).set(bytes, Number(p)); return { p, len: bytes.length, free: () => ex.temen_dealloc(p, bytes.length) }; };
const readStdout = () => { const p = Number(ex.temen_stdout_ptr()), l = Number(ex.temen_stdout_len()); return p && l ? Buffer.from(new Uint8Array(membuf(), p, l)).toString() : ''; };
const fail = (m) => { console.error(`FAIL: ${m}`); process.exit(1); };
const enc = (s) => Buffer.from(s);
const modBytes = readFileSync(modPath);

function cold(js) { const m = put(modBytes); const s = put(enc(js)); ex.temen_run_onramp(m.p, m.len, s.p, s.len); const out = readStdout(); const st = ex.temen_status(); s.free(); m.free(); return { out, st }; }
function warmInterp(js) { const s = put(enc(js)); ex.temen_warm_eval(s.p, s.len); const out = readStdout(); const st = ex.temen_status(); s.free(); return { out, st }; }
async function warmJit(js) { const st = await runWarmJit(ex, ex.memory ?? memory, enc(js), `${modPath}#eval`, 1); return { out: readStdout(), st }; }

const m = put(modBytes);
const live = Number(ex.temen_warm_open(m.p, m.len));
m.free();
if (live < 0 || ex.temen_status() !== 0) fail(`temen_warm_open: status ${ex.temen_status()}`);
console.error(`warm session opened: live image ${(live / (1 << 20)).toFixed(2)} MiB`);

const programs = [
  ['print', 'print("hi", 6 * 7)\n'],
  ['string.format', 'print(string.format("%d %s %.2f", 42, "x", 3.14159))\n'],
  ['table.sort', 'local t = {5,3,8,1,9,2}; table.sort(t); print(table.concat(t, ","))\n'],
  ['coroutine', 'local co = coroutine.wrap(function() for i=1,3 do coroutine.yield(i*i) end end); print(co(), co(), co())\n'],
  ['200k loop', 'local s=0; for i=1,200000 do s=s+i end; print("sum", s)\n'],
];

console.log(`\n${'program'.padEnd(16)}${'cold≡warm'.padStart(11)}${'≡warm+JIT'.padStart(11)}`);
let allOk = true;
for (const [name, js] of programs) {
  const c = cold(js);
  if (c.st !== 0) fail(`cold ${name}: status ${c.st}`);
  const wi = warmInterp(js);
  const wj = await warmJit(js);
  const interpOk = c.out === wi.out && wi.st === 0;
  const jitOk = c.out === wj.out && (wj.st === 0 || wj.st === 5);
  allOk = allOk && interpOk && jitOk;
  console.log(`${name.padEnd(16)}${(interpOk ? 'OK' : 'MISMATCH').padStart(11)}${(jitOk ? 'OK' : 'MISMATCH').padStart(11)}`);
  if (!interpOk || !jitOk) { console.log(`  cold:     ${JSON.stringify(c.out)}`); console.log(`  warm:     ${JSON.stringify(wi.out)}`); console.log(`  warm+JIT: ${JSON.stringify(wj.out)}`); }
}

// Fresh-per-Run isolation: a global in one Run must not leak into the next (snapshot restored each Run).
const r1 = warmInterp('leaked = 4242; print("set", leaked)\n');
const r2 = warmInterp('print("leak?", leaked)\n');
const isolated = r1.out.includes('set') && r1.out.includes('4242') && r2.out.includes('nil') && !r2.out.includes('4242');
allOk = allOk && isolated;
console.log(`\nfresh-per-Run isolation: ${isolated ? 'OK — no global leak across Runs' : 'LEAK!'}`);
if (!isolated) { console.log(`  run1: ${JSON.stringify(r1.out)}`); console.log(`  run2: ${JSON.stringify(r2.out)}`); }

ex.temen_warm_close();
if (!allOk) fail('Lua warm snapshot parity/isolation mismatch');
console.log('\nOK: Lua warm snapshot — warm eval_run matches cold _start (interp + JIT) byte-for-byte, isolation holds');
