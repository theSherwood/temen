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
import { runWarmJit } from './wasmjit-module.js';

let ex = null; // the worker's own engine exports
let memory = null; // the worker's own (private) shared WebAssembly.Memory
let warmUrl = null; // the module URL the warm session is currently open for (null ⇒ none)

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
      self.postMessage({ type: 'reply', id: msg.id, ok, status: ok ? 0 : ex.svm_status() });
      return;
    }
    if (msg.type === 'eval') {
      if (!ensureWarm(msg.url, msg.bytes)) {
        self.postMessage({ type: 'reply', id: msg.id, ok: false, error: 'warm session unavailable' });
        return;
      }
      const r = await evalWarm(msg.source, msg.jit);
      self.postMessage({ type: 'reply', id: msg.id, ok: true, ...r });
      return;
    }
  } catch (err) {
    self.postMessage({ type: 'reply', id: msg.id, ok: false, error: String((err && err.message) || err) });
  }
};
