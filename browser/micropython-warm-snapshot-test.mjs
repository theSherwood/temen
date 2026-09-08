// V8 (Node) check of the **MicroPython warm-runtime snapshot** driver (PYTHON.md slice B, #1327) —
// the MicroPython twin of `lua-warm-snapshot-test.mjs`. Drives the shipping engine FFI
// (`temen_warm_open` / `temen_warm_eval`) plus the warm+JIT path (`runWarmJit`) over the committed
// `micropython_snapshot.temen` (built from `micropython_snapshot.c`), and asserts:
//   - warm `eval_run` over the restored snapshot matches the cold `_start` (`temen_run_onramp`) output
//     byte-for-byte while skipping the MicroPython interpreter bring-up,
//   - warm+JIT (`eval_run` emitted to wasm) matches too, or declines cleanly to warm-interp (as Lua/Tcl
//     do — the nlr/`mp_embed_exec_str` setjmp routes `eval_run` to InterpDriven),
//   - fresh-per-Run isolation holds (a global bound in one Run can't leak into the next).
//
//   node micropython-warm-snapshot-test.mjs [temen_browser.wasm] [micropython_snapshot.temen]
//
// Exits 0 on success, 1 on any mismatch; SKIPs cleanly if the asset isn't built.
import { readFileSync, existsSync } from 'node:fs';
import { engineImports } from './engine-imports.mjs';
import { runWarmJit } from './web/wasmjit-module.js';

const wasmPath = process.argv[2] ?? 'target/wasm32-unknown-unknown/release/temen_browser.wasm';
const modPath = process.argv[3] ?? 'web/assets/micropython_snapshot.temen';
if (!existsSync(modPath)) {
  console.log(`SKIP: ${modPath} not built (MicroPython on-ramp asset absent)`);
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

function cold(py) { const m = put(modBytes); const s = put(enc(py)); ex.temen_run_onramp(m.p, m.len, s.p, s.len); const out = readStdout(); const st = ex.temen_status(); s.free(); m.free(); return { out, st }; }
function warmInterp(py) { const s = put(enc(py)); ex.temen_warm_eval(s.p, s.len); const out = readStdout(); const st = ex.temen_status(); s.free(); return { out, st }; }

let warmJitDeclined = false;
async function warmJit(py) {
  try {
    const st = await runWarmJit(ex, ex.memory ?? memory, enc(py), `${modPath}#eval`, 1);
    return { out: readStdout(), st, tier: 'jit' };
  } catch (e) {
    if (ex.temen_status() !== 2) throw e; // only STATUS_UNSUPPORTED is the drivability decline; re-raise real traps
    warmJitDeclined = true;
    const wi = warmInterp(py);
    return { out: wi.out, st: wi.st, tier: 'interp' };
  }
}

const m = put(modBytes);
const live = Number(ex.temen_warm_open(m.p, m.len));
m.free();
if (live < 0 || ex.temen_status() !== 0) fail(`temen_warm_open: status ${ex.temen_status()}`);
console.error(`warm session opened: live image ${(live / (1 << 20)).toFixed(2)} MiB`);

const programs = [
  ['print', 'print("hi", 6 * 7)\n'],
  ['comprehension', 'print([x * x for x in range(6)])\n'],
  ['dict', 'd = {"a": 1, "b": 2}; print(sum(d.values()))\n'],
  ['recursion', 'def fib(n):\n    return n if n < 2 else fib(n-1)+fib(n-2)\nprint([fib(i) for i in range(10)])\n'],
  ['float', 'print(round(2 ** 0.5, 6), 3.0 / 2)\n'],
  ['exception', 'try:\n    1/0\nexcept Exception as e:\n    print(repr(e))\n'],
  ['200k loop', 's = 0\nfor i in range(200000):\n    s += i\nprint("sum", s)\n'],
];

console.log(`\n${'program'.padEnd(16)}${'cold≡warm'.padStart(11)}${'≡warm+JIT'.padStart(11)}`);
let allOk = true;
for (const [name, py] of programs) {
  const c = cold(py);
  if (c.st !== 0) fail(`cold ${name}: status ${c.st}`);
  const wi = warmInterp(py);
  const wj = await warmJit(py);
  const interpOk = c.out === wi.out && wi.st === 0;
  const jitOk = c.out === wj.out && (wj.st === 0 || wj.st === 5);
  allOk = allOk && interpOk && jitOk;
  const jitLabel = jitOk ? (wj.tier === 'jit' ? 'OK' : 'OK↩interp') : 'MISMATCH';
  console.log(`${name.padEnd(16)}${(interpOk ? 'OK' : 'MISMATCH').padStart(11)}${jitLabel.padStart(11)}`);
  if (!interpOk || !jitOk) { console.log(`  cold:     ${JSON.stringify(c.out)}`); console.log(`  warm:     ${JSON.stringify(wi.out)}`); console.log(`  warm+JIT: ${JSON.stringify(wj.out)}`); }
}

// Fresh-per-Run isolation: a global bound in one Run must not leak into the next (snapshot restored
// each Run). Run 2 references the name → MicroPython raises NameError over the post-warmup image.
const r1 = warmInterp('leaked = 4242\nprint("set", leaked)\n');
const r2 = warmInterp('print("leak?", leaked)\n');
const isolated = r1.out.includes('set') && r1.out.includes('4242') && r2.out.includes('NameError') && !r2.out.includes('4242');
allOk = allOk && isolated;
console.log(`\nfresh-per-Run isolation: ${isolated ? 'OK — no global leak across Runs' : 'LEAK!'}`);
if (!isolated) { console.log(`  run1: ${JSON.stringify(r1.out)}`); console.log(`  run2: ${JSON.stringify(r2.out)}`); }

ex.temen_warm_close();
if (!allOk) fail('MicroPython warm snapshot parity/isolation mismatch');
if (warmJitDeclined) {
  console.log('\nnote: warm+JIT declined (MicroPython eval_run reaches mp_embed_exec_str\'s nlr setjmp ⇒');
  console.log('      InterpDriven); the warm card falls back to the warm interpreter, as production does.');
}
console.log('\nOK: MicroPython warm snapshot — warm eval_run matches cold _start byte-for-byte, isolation holds');
