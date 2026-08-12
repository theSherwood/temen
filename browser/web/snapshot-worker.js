// The **snapshot worker** (WASM_AOT.md warm+JIT · issue #804). Owns the warm-runtime snapshot session for
// the playground's warm cards (QuickJS) on a dedicated Web Worker with its **own private engine instance
// and memory** — so the ~one-time QuickJS `warmup` (`svm_warm_open`) and every eval run entirely off the
// main thread. The main thread pre-warms at page load and dispatches each Run as a message; it never
// blocks on the ~0.9 s warmup or a compute-heavy eval.
//
// Why private memory (not the page's shared engine memory): `worker.js`'s concurrent path documents a
// "rare shared-memory race (a double-free)" in the shared setup. This Worker instantiates the engine over
// a fresh memory of its own and allocates only there, so its warm session can't race the main thread's
// allocator. Main ↔ worker communicate only by messages (source string in; stdout/status/value out).
import { runWarmJit, primeWarmJit, jitCacheStats } from './wasmjit-module.js';

let ex = null; // the worker's own engine exports
let memory = null; // the worker's own (private) shared WebAssembly.Memory
let warmUrl = null; // the module URL the warm session is currently open for (null ⇒ none)
// The background warm+JIT pre-compile (issue: the first wasm-JIT Run paid the ~12.7 MB `WebAssembly.
// compile`). `jitPrimePromise` is the in-flight (or settled) `primeWarmJit`; `jitPrimed` flips true once
// it settles (compiled, or known non-emittable). A wasm-JIT eval awaits it so the first Run is a cache hit.
let jitPrimePromise = null;
let jitPrimed = false;

const u8 = () => new Uint8Array(memory.buffer);
// Read a captured stream out of the worker's memory. `.slice` copies to a non-shared buffer (TextDecoder
// rejects a SharedArrayBuffer-backed view), exactly as play.js's `readModuleStdout` does.
const readStr = (ptrFn, lenFn) => {
  const p = Number(ex[ptrFn]());
  const l = Number(ex[lenFn]());
  return p && l ? new TextDecoder().decode(u8().slice(p, p + l)) : '';
};
const readStdout = () => readStr('svm_stdout_ptr', 'svm_stdout_len');
const readStderr = () => readStr('svm_stderr_ptr', 'svm_stderr_len');

// Open (or reuse) the warm session for `url`'s module `bytes` — runs `warmup` once and snapshots the
// post-init image (mirrors play.js's `ensureWarmSession`). Returns true on success; false if the module
// isn't a warm-snapshot driver, `warmup` traps, or a re-open is needed but no bytes were supplied.
function ensureWarm(url, bytes) {
  if (warmUrl === url) return true;
  if (!bytes) return false;
  const p = Number(ex.svm_alloc(bytes.length));
  u8().set(bytes, p);
  const live = Number(ex.svm_warm_open(p, bytes.length));
  ex.svm_dealloc(p, bytes.length);
  if (live < 0 || ex.svm_status() !== 0) {
    warmUrl = null;
    return false;
  }
  warmUrl = url;
  return true;
}

// Evaluate `source` over the open warm session — warm+JIT when `jit`, else warm-interp. A warm+JIT
// decline/trap falls back to the warm interpreter, so the result is always the snapshot's eval.
async function evalWarm(source, jit) {
  const stdinBytes = source ? new TextEncoder().encode(source) : null;
  if (jit) {
    try {
      const status = await runWarmJit(ex, memory, stdinBytes, `${warmUrl}#eval`, 1);
      return { tier: 'warm+JIT', status, value: Number(ex.svm_run_value()), stdout: readStdout(), stderr: readStderr() };
    } catch {
      // decline / trap → fall through to the warm interpreter
    }
  }
  let stdinP = 0;
  const len = stdinBytes ? stdinBytes.length : 0;
  if (len) {
    stdinP = Number(ex.svm_alloc(len));
    u8().set(stdinBytes, stdinP);
  }
  const value = Number(ex.svm_warm_eval(stdinP, len));
  const status = ex.svm_status();
  const stdout = readStdout();
  const stderr = readStderr();
  if (stdinP) ex.svm_dealloc(stdinP, len);
  return { tier: 'warm-snapshot', status, value, stdout, stderr };
}

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === 'init') {
      // The main thread posts the already-compiled engine `WebAssembly.Module` (structured-cloneable),
      // so the worker skips a second fetch+compile. It instantiates over its OWN shared memory (threads
      // build imports one) with the no-op webgpu stub (a compute guest never presents).
      memory = new WebAssembly.Memory({ initial: 2048, maximum: 16384, shared: true });
      ({ exports: ex } = await WebAssembly.instantiate(msg.module, {
        env: { memory },
        svm_host: { webgpu_op: () => -1n },
      }));
      self.postMessage({ type: 'ready' });
      return;
    }
    if (msg.type === 'prewarm') {
      const ok = ensureWarm(msg.url, msg.bytes);
      // Reply as soon as warm-interp is ready (the ~0.9 s `warmup`) — do NOT block on the JIT pre-compile,
      // so an early interpreter Run isn't delayed by it.
      self.postMessage({ type: 'reply', id: msg.id, ok, status: ok ? 0 : ex.svm_status() });
      // Then pre-compile the warm+JIT `eval_run` in the background (the ~12.7 MB `WebAssembly.compile`), so
      // the first wasm-JIT Run is a cache hit instead of paying that compile. Best-effort; a non-emittable
      // eval just leaves the card on warm-interp. Skipped when `primeJit` is false (a card whose warm+JIT
      // declines, e.g. Tcl — no point compiling a module that traps).
      if (ok && msg.primeJit !== false && !jitPrimePromise) {
        jitPrimePromise = primeWarmJit(ex, memory, `${warmUrl}#eval`, 1)
          .catch(() => false)
          .finally(() => { jitPrimed = true; });
      }
      return;
    }
    if (msg.type === 'eval') {
      if (!ensureWarm(msg.url, msg.bytes)) {
        self.postMessage({ type: 'reply', id: msg.id, ok: false, error: 'warm session unavailable' });
        return;
      }
      // A wasm-JIT Run waits for the in-flight pre-compile (if any) so it reuses the primed instance
      // rather than kicking off a fresh ~12.7 MB compile on this Run.
      if (msg.jit && jitPrimePromise) { try { await jitPrimePromise; } catch { /* prime failed → runWarmJit falls back */ } }
      const r = await evalWarm(msg.source, msg.jit);
      self.postMessage({ type: 'reply', id: msg.id, ok: true, ...r });
      return;
    }
    if (msg.type === 'stats') {
      // Test/telemetry hook: whether the warm+JIT pre-compile has settled, and the worker's JIT cache
      // accounting (a primed instance ⇒ the first wasm-JIT Run is a cache hit, not a fresh compile).
      self.postMessage({ type: 'reply', id: msg.id, ok: true, jitPrimed, compiles: jitCacheStats.compiles, hits: jitCacheStats.hits });
      return;
    }
  } catch (err) {
    self.postMessage({ type: 'reply', id: msg.id, ok: false, error: String((err && err.message) || err) });
  }
};
