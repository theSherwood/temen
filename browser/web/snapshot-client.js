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
    w = { worker, seq: 0, pending: new Map(), prewarm: null };
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
      if (m.type === 'reply') {
        const resolve = w.pending.get(m.id);
        if (resolve) {
          w.pending.delete(m.id);
          resolve(m);
        }
      }
    };
    worker.onerror = (e) => w._rejectReady(new Error(e.message || 'snapshot worker error'));
    worker.postMessage({ type: 'init', module: this._engineModule });
    this._workers.set(url, w);
    return w;
  }

  // Whether `url`'s pre-warm has been kicked off (⇒ its Run won't pay the first-Run warmup).
  isWarming(url) {
    const w = this._workers.get(url);
    return !!(w && w.prewarm);
  }

  _request(w, type, payload) {
    const id = ++w.seq;
    return new Promise((resolve) => {
      w.pending.set(id, resolve);
      w.worker.postMessage({ type, id, ...payload });
    });
  }

  // Open the warm session for `url` on its worker (idempotent per URL). `getBytes()` fetches the module
  // bytes and is awaited only the first time. Resolves `{ ok, status }`.
  prewarm(url, getBytes) {
    const w = this._workerFor(url);
    if (w.prewarm) return w.prewarm;
    w.prewarm = (async () => {
      await w.ready;
      const bytes = await getBytes();
      return this._request(w, 'prewarm', { url, bytes });
    })();
    return w.prewarm;
  }

  // Evaluate `source` over `url`'s warm session (pre-warming first if needed), on `url`'s worker. `jit`
  // picks the warm+JIT tier. Resolves `{ ok, tier, status, value, stdout, stderr }`, or `{ ok:false,
  // error }` (the caller then falls back to the main-thread path).
  async evalWarm(url, getBytes, source, jit) {
    const warm = await this.prewarm(url, getBytes);
    if (!warm.ok) return warm;
    return this._request(this._workers.get(url), 'eval', { url, source, jit });
  }

  // Query `url`'s worker for its warm+JIT pre-compile state — `{ ok, jitPrimed, compiles, hits }` (for
  // tests/telemetry). Resolves `{ ok:false }` if no worker exists for `url` yet.
  stats(url) {
    const w = this._workers.get(url);
    if (!w) return Promise.resolve({ ok: false });
    return this._request(w, 'stats', {});
  }
}
