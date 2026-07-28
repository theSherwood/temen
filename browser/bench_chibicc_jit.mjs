// chibicc compile-time — **bytecode interpreter vs wasm-JIT**, timed on Node's V8 (the browser
// engine), no Playwright. The fast-iteration twin of the playground C-compiler card: for each C
// program it runs chibicc.svmb's compile pass twice — once on the bytecode interpreter
// (`svm_run_onramp_fs`) and once on the wasm-JIT (`runJitCompiler`, which emits chibicc's `_start` to
// wasm, `WebAssembly.compile`s it, and runs it, bouncing cap-calls cross-tier) — asserts the emitted
// SVM-IR is byte-identical across tiers, and reports the wall-clock speedup. Runs on the **threads**
// wasm32 cdylib (imported shared memory, exactly the page's build) on Node's WebAssembly.
//
// Usage:  node bench_chibicc_jit.mjs [module.wasm]   (build the threads cdylib first — see browser-test.mjs)
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { performance } from 'node:perf_hooks';
import { runJitCompiler } from './web/wasmjit-module.js';
import { engineImports } from './engine-imports.mjs';

const ROOT = dirname(fileURLToPath(import.meta.url));
const wasmPath = process.argv[2] ?? join(ROOT, 'target/wasm32-unknown-unknown/release/svm_browser.wasm');
const chibiccPath = join(ROOT, 'web/assets/chibicc.svmb');
if (!existsSync(chibiccPath)) {
  console.log('SKIP: web/assets/chibicc.svmb not built (run build-onramp-assets.mjs)');
  process.exit(0);
}

const mod = await WebAssembly.compile(readFileSync(wasmPath));
if (!WebAssembly.Module.imports(mod).some((i) => i.kind === 'memory')) {
  console.log('FAIL: expected the threads cdylib (imported shared memory) — build with the threads flags');
  process.exit(1);
}
const memory = new WebAssembly.Memory({ initial: 2048, maximum: 16384, shared: true });
const { exports: ex } = await WebAssembly.instantiate(mod, engineImports(memory));
const compiler = new Uint8Array(readFileSync(chibiccPath));

const dec = new TextDecoder();
const readStdout = () => dec.decode(new Uint8Array(memory.buffer).slice(
  Number(ex.svm_stdout_ptr()), Number(ex.svm_stdout_ptr()) + Number(ex.svm_stdout_len())));

// Compile `src` on the **bytecode interpreter** (`svm_run_onramp_fs`, debug off), returning its IR.
function bytecodeCompile(srcBytes) {
  const p = Number(ex.svm_alloc(compiler.length));
  const sp = Number(ex.svm_alloc(srcBytes.length));
  const u8 = new Uint8Array(memory.buffer);
  u8.set(compiler, p);
  u8.set(srcBytes, sp);
  ex.svm_run_onramp_fs(p, compiler.length, 0, 0, sp, srcBytes.length, 0);
  const ir = readStdout();
  ex.svm_dealloc(p, compiler.length);
  ex.svm_dealloc(sp, srcBytes.length);
  return ir;
}

// The programs, small → larger, so the fixed JIT cost (emit + WebAssembly.compile of chibicc's ~1.2 MB
// `_start`) shows up against the interpreter's per-program work.
const PROGRAMS = [
  ['return-only (tiny)', 'int main(void){ int x = 7 * 6; return x; }'],
  ['printf loop', '#include <stdio.h>\nint main(void){ for(int i=1;i<=20;i++) printf("i=%d sq=%d\\n", i, i*i); return 0; }'],
  ['string + stdlib', '#include <stdio.h>\n#include <string.h>\n#include <stdlib.h>\n' +
    'int main(void){ char b[64]; for(int i=0;i<30;i++){ snprintf(b,sizeof b,"n%d",i); } ' +
    'char *p=malloc(32); strcpy(p,"chibicc"); printf("%s %lu\\n", p, (unsigned long)strlen(p)); return 0; }'],
  ['nested loops + floats', '#include <stdio.h>\nint main(void){ double s=0; ' +
    'for(int i=0;i<40;i++) for(int j=0;j<40;j++) s += (i*j)/3.0; printf("%.3f\\n", s); return 0; }'],
];

const REPS = 5;
const median = (xs) => xs.slice().sort((a, b) => a - b)[Math.floor(xs.length / 2)];

let failed = false;
console.log(`module: ${wasmPath}\nchibicc: ${chibiccPath} (${compiler.length}B)\n`);

// The emitted wasm is chibicc's `_start` — the SAME module for every program (the user's C is *data*
// in the seeded memfs, not part of the emitted code), so V8 code-caches its `WebAssembly.compile`
// after the first call. Time that one-time cold emit+compile before the warm per-program loop.
{
  const src = new TextEncoder().encode('int main(void){ return 0; }');
  const t = performance.now();
  await runJitCompiler(ex, memory, compiler, src, 0);
  console.log(`cold JIT warm-up (one-time emit + WebAssembly.compile of chibicc's _start): ${(performance.now() - t).toFixed(0)}ms`);
  console.log('(paid once per page load; every compile after reuses V8\'s cached module)\n');
}

console.log(`${'program'.padEnd(24)} ${'IR'.padStart(7)}  ${'bytecode'.padStart(10)}  ${'wasm-JIT'.padStart(10)}  speedup  match`);

for (const [name, src] of PROGRAMS) {
  const srcBytes = new TextEncoder().encode(src);
  // Warm each tier once (first-touch alloc), then time REPS and take the median (steady-state:
  // repeated edit→Run in the card, chibicc's emitted module already V8-cached).
  const irB0 = bytecodeCompile(srcBytes);
  await runJitCompiler(ex, memory, compiler, srcBytes, 0);
  const irJ0 = readStdout();
  const match = irB0 === irJ0 && irB0.includes('func');

  const bt = [], jt = [];
  for (let i = 0; i < REPS; i++) {
    let t = performance.now(); bytecodeCompile(srcBytes); bt.push(performance.now() - t);
    t = performance.now(); await runJitCompiler(ex, memory, compiler, srcBytes, 0); jt.push(performance.now() - t);
  }
  const b = median(bt), j = median(jt);
  const speedup = (b / j).toFixed(2);
  if (!match) failed = true;
  console.log(`${name.padEnd(24)} ${(irB0.length + 'B').padStart(7)}  ${(b.toFixed(0) + 'ms').padStart(10)}  ` +
    `${(j.toFixed(0) + 'ms').padStart(10)}  ${(speedup + '×').padStart(7)}  ${match ? 'ok' : 'DIVERGED'}`);
}

console.log(`\n${failed ? 'FAIL (a tier diverged)' : 'PASS — both tiers emit byte-identical IR'}`);
process.exit(failed ? 1 : 0);
