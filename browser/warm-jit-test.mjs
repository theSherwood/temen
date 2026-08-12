// V8 (Node) end-to-end check of the **warm+JIT** eval tier (WASM_AOT.md warm+JIT), driving the shipping
// engine FFI the playground uses (`svm_warm_open` + `svm_warm_jit_*`) plus the browser driver
// (`runWarmJit` from wasmjit-module.js). The interpreter warm path (`warm-snapshot-test.mjs`) already
// proves eval_run-over-the-snapshot ≡ cold `_start`; this proves the **emitted-wasm** eval_run agrees
// with the interpreter warm eval byte-for-byte, keeps fresh-per-Run isolation, and — the point of the
// tier — accelerates a compute-heavy eval while init stays paid-once.
//
//   node warm-jit-test.mjs [svm_browser.wasm] [qjs_snapshot.svmb]
//
// Exits 0 on success (all parity + isolation checks pass), 1 on any mismatch.
import { readFileSync, existsSync } from 'node:fs';
import { performance } from 'node:perf_hooks';
import { engineImports } from './engine-imports.mjs';
import { runWarmJit, jitCacheStats } from './web/wasmjit-module.js';

const wasmPath = process.argv[2] ?? 'target/wasm32-unknown-unknown/release/svm_browser.wasm';
const modPath = process.argv[3] ?? 'web/assets/qjs_snapshot.svmb';

if (!existsSync(modPath)) {
  console.log(`SKIP: ${modPath} not built (QuickJS on-ramp asset absent)`);
  process.exit(0);
}

const mod = await WebAssembly.compile(readFileSync(wasmPath));
const memory = new WebAssembly.Memory({ initial: 2048, maximum: 16384, shared: true });
const ex = (await WebAssembly.instantiate(mod, engineImports(memory))).exports;
const membuf = () => (ex.memory ?? memory).buffer;
const is64 = ex.svm_abi_is64() === 1;
const N = (x) => (is64 ? BigInt(x) : Number(x));
const put = (bytes) => {
  const p = ex.svm_alloc(N(bytes.length));
  new Uint8Array(membuf()).set(bytes, Number(p));
  return { p, len: N(bytes.length), free: () => ex.svm_dealloc(p, N(bytes.length)) };
};
const readStdout = () => {
  const p = Number(ex.svm_stdout_ptr());
  const l = Number(ex.svm_stdout_len());
  return p && l ? Buffer.from(new Uint8Array(membuf(), p, l)).toString() : '';
};
const fail = (m) => { console.error(`FAIL: ${m}`); process.exit(1); };

const modBytes = readFileSync(modPath);
const enc = (s) => Buffer.from(s);

// WARM-INTERP: restore the snapshot, then `eval_run` on the bytecode interpreter.
function warmInterp(js) {
  const s = put(enc(js));
  const t = performance.now();
  Number(ex.svm_warm_eval(s.p, s.len));
  const ms = performance.now() - t;
  const out = readStdout();
  const status = ex.svm_status();
  s.free();
  return { out, ms, status };
}
// WARM+JIT: restore the snapshot, then `eval_run` on emitted wasm (the whole runWarmJit round-trip).
async function warmJit(js) {
  const t = performance.now();
  const status = await runWarmJit(ex, ex.memory ?? memory, enc(js), `${modPath}#eval`, 1);
  const ms = performance.now() - t;
  return { out: readStdout(), ms, status };
}

// Open the warm session once (runs `warmup`, snapshots the post-init image).
{
  const m = put(modBytes);
  const live = Number(ex.svm_warm_open(m.p, m.len));
  m.free();
  if (live < 0 || ex.svm_status() !== 0) fail(`svm_warm_open: status ${ex.svm_status()}`);
  console.error(`warm session opened: live image ${(live / (1 << 20)).toFixed(1)} MiB`);
}

const programs = [
  ['trivial', '1;\n'],
  [
    "user's program",
    'function fib(n){return n<2?n:fib(n-1)+fib(n-2);}\n' +
      'console.log("fib", Array.from({length:11},(_,i)=>fib(i)).join(" "));\n' +
      'const xs=[5,3,8,1,9,2,7];\n' +
      'console.log("sorted", xs.slice().sort((a,b)=>a-b).join(","));\n' +
      'console.log("json", JSON.stringify({ok:true,nums:[1,2,3],nested:{pi:Math.PI}}));\n',
  ],
  ['500k loop', 'let s=0;for(let i=0;i<500000;i++)s+=i;console.log("sum",s);\n'],
];

// Prime the compiled-Module cache with one throwaway Run: the first warm+JIT Run pays the one-time
// `WebAssembly.compile` of the ~12.7 MiB emitted eval_run (V8 codegen), reused on every later Run. Time
// it separately so the per-program numbers below reflect steady-state (cache-hit) warm+JIT, not the
// first-Run codegen — the same way the card behaves after its first JIT Run.
{
  const t = performance.now();
  await warmJit('1;\n');
  console.error(`warm+JIT primed: one-time WebAssembly.compile of the emitted eval_run took ${(performance.now() - t).toFixed(0)} ms (amortized across Runs)`);
}

console.log(`\n${'program'.padEnd(16)}${'interp ms'.padStart(11)}${'warm+JIT ms'.padStart(13)}${'speedup'.padStart(9)}  parity`);
let allOk = true;
for (const [name, js] of programs) {
  const ci = warmInterp(js);
  if (ci.status !== 0) fail(`warm-interp ${name}: status ${ci.status}`);
  const cj = await warmJit(js);
  if (cj.status !== 0 && cj.status !== 5) fail(`warm+JIT ${name}: status ${cj.status}`);
  const parity = ci.out === cj.out;
  allOk &&= parity;
  console.log(
    `${name.padEnd(16)}${ci.ms.toFixed(0).padStart(11)}${cj.ms.toFixed(0).padStart(13)}${(ci.ms / cj.ms).toFixed(1).padStart(8)}x  ${parity ? 'OK' : 'MISMATCH!'}`,
  );
  if (!parity) {
    console.log(`  interp:   ${JSON.stringify(ci.out)}`);
    console.log(`  warm+JIT: ${JSON.stringify(cj.out)}`);
  }
}

// Fresh-per-Run isolation: a `var` declared in one Run must not leak into the next (the snapshot is
// restored each Run). Declare `leak`, then read it — the second Run should throw a ReferenceError, not
// see the prior value. Two consecutive warm+JIT Runs.
{
  const r1 = await warmJit('var leak = 12345; console.log("set", leak);\n');
  const r2 = await warmJit('try { console.log("leaked?", typeof leak, leak); } catch (e) { console.log("clean", e.name); }\n');
  const isolated = r1.out.includes('set 12345') && r2.out.includes('clean') && !r2.out.includes('12345');
  allOk &&= isolated;
  console.log(`\nfresh-per-Run isolation (warm+JIT): ${isolated ? 'OK — no var leak across Runs' : 'LEAK!'}`);
  if (!isolated) { console.log(`  run1: ${JSON.stringify(r1.out)}`); console.log(`  run2: ${JSON.stringify(r2.out)}`); }
}

// Instance cache (issue #803): every warm+JIT Run above shares one cacheKey, so exactly the first Run
// compiled the emitted eval_run and every later Run reused the cached instance (skipping compile +
// instantiate + the byte copy). Assert that accounting so a regression that re-instantiates per Run is
// caught, not just tolerated.
{
  const runs = 1 /* primer */ + programs.length + 2 /* isolation */;
  const cacheOk = jitCacheStats.compiles === 1 && jitCacheStats.hits === runs - 1;
  allOk &&= cacheOk;
  console.log(
    `\ninstance cache: compiles=${jitCacheStats.compiles} hits=${jitCacheStats.hits} ` +
    `(expected compiles=1, hits=${runs - 1}) — ${cacheOk ? 'OK — one compile, rest reused' : 'REGRESSION: re-instantiating per Run'}`,
  );
}

ex.svm_warm_close();

if (!allOk) fail('warm+JIT parity/isolation/instance-cache mismatch');
console.log('\nOK: warm+JIT — emitted eval_run matches warm-interp byte-for-byte, isolation + instance cache hold');
