// Main-thread client for the **snapshot workers** (issues #804/#805). Wraps a pool of Web Workers — one
// per warm module URL — in a promise-based request/reply protocol. Each worker owns its own engine
// instance + private memory + one warm session, so multiple warm cards (QuickJS, Lua) stay warm
// simultaneously (the engine holds a single warm session per instance, so a shared worker would evict one
// card's snapshot when the other pre-warms). The playground pre-warms each warm card on load (off the
// main thread, with a card indicator) and routes its Runs here, so neither the ~one-time runtime warmup
// nor a compute-heavy eval ever blocks the UI.
export class SnapshotClient {
  // `engineModule` is the already-compiled engine `WebAssembly.Module` (from `loadEngine`), posted to
  // each worker so it need not re-fetch/compile.
  constructor(engineModule) {
    this._engineModule = engineModule;
    this._workers = new Map(); // url -> worker record
  }

  // Get-or-spawn the dedicated worker for `url` (one engine + memory + warm session per module).
  _workerFor(url) {
    let w = this._workers.get(url);
    if (w) return w;
    const worker = new Worker(new URL('./snapshot-worker.js', import.meta.url), { type: 'module' });
    w = { worker, seq: 0, pending: new Map(), chunks: new Map(), prewarm: null };
    w.ready = new Promise((resolve, reject) => {
      w._resolveReady = resolve;
      w._rejectReady = reject;
    });
    worker.onmessage = (e) => {
      const m = e.data;
      if (m.type === 'ready') {
        w._resolveReady();
        return;
      }
      if (m.type === 'stdout-chunk') {
        // A live stdout chunk from a streaming Run — hand it to that request's sink as it arrives.
        const onChunk = w.chunks.get(m.id);
        if (onChunk) onChunk(m.bytes);
        return;
      }
      if (m.type === 'reply') {
        w.chunks.delete(m.id);
        const resolve = w.pending.get(m.id);
        if (resolve) {
          w.pending.delete(m.id);
          resolve(m);
        }
      }
    };
    worker.onerror = (e) => {
      const err = new Error(e.message || 'snapshot worker error');
      w._rejectReady(err); // no-op once ready has settled
      // Fail every in-flight request so its caller can fall back / report, instead of hanging forever.
      for (const [id, resolve] of w.pending) resolve({ ok: false, error: err.message });
      w.pending.clear();
    };
    worker.postMessage({ type: 'init', module: this._engineModule });
    this._workers.set(url, w);
    return w;
  }

  // Whether `url`'s pre-warm has been kicked off (⇒ its Run won't pay the first-Run warmup).
  isWarming(url) {
    const w = this._workers.get(url);
    return !!(w && w.prewarm);
  }

  _request(w, type, payload, onChunk) {
    const id = ++w.seq;
    if (onChunk) w.chunks.set(id, onChunk); // live stdout chunks for this request (streaming Runs)
    return new Promise((resolve) => {
      w.pending.set(id, resolve);
      w.worker.postMessage({ type, id, ...payload });
    });
  }

  // Open the warm session for `url` on its worker (idempotent per URL). `getBytes()` fetches the module
  // bytes and is awaited only the first time. `primeJit` (default true) also pre-compiles the warm+JIT
  // tier in the worker (skip it for cards whose warm+JIT declines, e.g. Tcl). Resolves `{ ok, status }`.
  prewarm(url, getBytes, primeJit = true) {
    const w = this._workerFor(url);
    if (w.prewarm) return w.prewarm;
    w.prewarm = (async () => {
      await w.ready;
      const bytes = await getBytes();
      return this._request(w, 'prewarm', { url, bytes, primeJit });
    })();
    return w.prewarm;
  }

  // Evaluate `source` over `url`'s warm session (pre-warming first if needed), on `url`'s worker. `jit`
  // picks the warm+JIT tier. Resolves `{ ok, tier, status, value, stdout, stderr }`, or `{ ok:false,
  // error }` (the caller then falls back to the main-thread path).
  async evalWarm(url, getBytes, source, jit, onChunk) {
    const warm = await this.prewarm(url, getBytes);
    if (!warm.ok) return warm;
    return this._request(this._workers.get(url), 'eval', { url, source, jit }, onChunk);
  }

  // The reserved worker key for the nim full-compile card (its own engine instance, separate from any
  // warm-card worker). A non-URL sentinel so it never collides with a warm module URL.
  static NIMC_KEY = '__nimc__';

  // The reserved worker for **streamed plain-module Runs** (nim/C on-ramp guests): off the main thread so
  // the page paints each stdout chunk as the guest writes it (the worker's synchronous run still delivers
  // its `postMessage`s to the main thread's event loop). One shared worker — module Runs are independent
  // and short — keyed by a non-URL sentinel.
  static STREAM_KEY = '__stream__';

  // Run a plain on-ramp module off the main thread with **live stdout**: `onChunk(Uint8Array)` fires for
  // each write as it happens; resolves `{ ok, status, value, stdout, stderr }` (the full captured output),
  // or `{ ok:false, error }` (the caller then falls back to the synchronous main-thread path). `stdin` is
  // an optional `Uint8Array`.
  async runStream(bytes, stdin, onChunk) {
    const w = this._workerFor(SnapshotClient.STREAM_KEY);
    await w.ready;
    const id = ++w.seq;
    if (onChunk) w.chunks.set(id, onChunk);
    return new Promise((resolve) => {
      w.pending.set(id, resolve);
      w.worker.postMessage({ type: 'runStream', id, bytes, stdin: stdin || null });
    });
  }

  // Run a module's `_start` on the **wasm-JIT tier** off the main thread with **live stdout** (#1141);
  // `onChunk(Uint8Array)` fires per write. Resolves `{ ok, status, value, stdout, stderr }`, or
  // `{ ok:false, error }` on a JIT decline/trap (the caller then falls back to the interpreter path).
  // `cacheKey` caches the emitted Module across Runs (same as the main-thread `runJitModule`).
  async runJitStream(bytes, stdin, cacheKey, onChunk) {
    const w = this._workerFor(SnapshotClient.STREAM_KEY);
    await w.ready;
    const id = ++w.seq;
    if (onChunk) w.chunks.set(id, onChunk);
    return new Promise((resolve) => {
      w.pending.set(id, resolve);
      w.worker.postMessage({ type: 'runJitStream', id, bytes, stdin: stdin || null, cacheKey });
    });
  }

  // Compile a whole Nim program off the main thread. `getAssets()` resolves the four phase buffers
  // `{ nifler, nimsem, hexer, stdlib }` (fetched + inflated by the caller); they're posted to the nim
  // worker once and cached there, so subsequent Runs ship only `source`. Resolves `{ ok, status,
  // stdout, stderr }`, or `{ ok:false, error }` (the caller then falls back to the main-thread path).
  async nimCompile(getAssets, source, main = 'prog.nim', onChunk) {
    const w = this._workerFor(SnapshotClient.NIMC_KEY);
    await w.ready;
    if (!w.nimAssets) {
      w.nimAssets = (async () => {
        const a = await getAssets();
        return this._request(w, 'nimAssets', a);
      })();
    }
    let loaded;
    try {
      loaded = await w.nimAssets;
    } catch (e) {
      w.nimAssets = null; // a fetch/inflate failure: clear the cached (rejected) upload so a later Run retries
      throw e; // surfaces to runNimc's catch (fetch/build-hint error state)
    }
    if (!loaded.ok) {
      w.nimAssets = null; // worker rejected the upload: let a later Run retry it
      return { ok: false, error: loaded.error || 'nim assets failed to load' };
    }
    return this._request(w, 'nimCompile', { source, main }, onChunk);
  }

  // Abandon an in-flight nim compile: terminate the nim worker (a runaway guest can't be interrupted
  // cooperatively) and drop its record so the next `nimCompile` respawns a fresh engine and re-sends
  // assets. Pending requests on it never resolve — the caller drops them.
  cancelNim() {
    const w = this._workers.get(SnapshotClient.NIMC_KEY);
    if (!w) return;
    w.worker.terminate();
    this._workers.delete(SnapshotClient.NIMC_KEY);
  }

  // Query `url`'s worker for its warm+JIT pre-compile state — `{ ok, jitPrimed, compiles, hits }` (for
  // tests/telemetry). Resolves `{ ok:false }` if no worker exists for `url` yet.
  stats(url) {
    const w = this._workers.get(url);
    if (!w) return Promise.resolve({ ok: false });
    return this._request(w, 'stats', {});
  }
}
