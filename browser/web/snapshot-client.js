// Main-thread client for the **snapshot worker** (issue #804). Wraps the Worker in a promise-based
// request/reply protocol and tracks per-URL pre-warm so each warm session is opened once. The playground
// pre-warms a warm card on load (off the main thread, with a card indicator) and routes its Runs here, so
// neither the ~one-time QuickJS warmup nor a compute-heavy eval ever blocks the UI.
export class SnapshotClient {
  // `engineModule` is the already-compiled engine `WebAssembly.Module` (from `loadEngine`), posted to the
  // worker so it need not re-fetch/compile. Construction is cheap; `ready()` resolves once the worker has
  // instantiated its own engine.
  constructor(engineModule) {
    this._worker = new Worker(new URL('./snapshot-worker.js', import.meta.url), { type: 'module' });
    this._seq = 0;
    this._pending = new Map(); // id -> resolve
    this._warmed = new Map(); // url -> Promise<reply> (dedup pre-warm per module)
    this._ready = new Promise((resolve, reject) => {
      this._resolveReady = resolve;
      this._rejectReady = reject;
    });
    this._worker.onmessage = (e) => {
      const m = e.data;
      if (m.type === 'ready') {
        this._resolveReady();
        return;
      }
      if (m.type === 'reply') {
        const resolve = this._pending.get(m.id);
        if (resolve) {
          this._pending.delete(m.id);
          resolve(m);
        }
      }
    };
    this._worker.onerror = (e) => this._rejectReady(new Error(e.message || 'snapshot worker error'));
    this._worker.postMessage({ type: 'init', module: engineModule });
  }

  ready() {
    return this._ready;
  }

  // Whether `url`'s warm session pre-warm has been kicked off (⇒ its Run won't pay the first-Run warmup).
  isWarming(url) {
    return this._warmed.has(url);
  }

  _request(type, payload) {
    const id = ++this._seq;
    return new Promise((resolve) => {
      this._pending.set(id, resolve);
      this._worker.postMessage({ type, id, ...payload });
    });
  }

  // Open the warm session for `url` (idempotent per URL). `getBytes()` fetches the module bytes and is
  // awaited only the first time. Resolves `{ ok, status }`.
  prewarm(url, getBytes) {
    let p = this._warmed.get(url);
    if (p) return p;
    p = (async () => {
      await this._ready;
      const bytes = await getBytes();
      return this._request('prewarm', { url, bytes });
    })();
    this._warmed.set(url, p);
    return p;
  }

  // Evaluate `source` over `url`'s warm session (pre-warming first if needed), on the worker. `jit` picks
  // the warm+JIT tier. Resolves `{ ok, tier, status, value, stdout, stderr }`, or `{ ok:false, error }`
  // (the caller then falls back to the main-thread path).
  async evalWarm(url, getBytes, source, jit) {
    const warm = await this.prewarm(url, getBytes);
    if (!warm.ok) return warm;
    return this._request('eval', { url, source, jit });
  }
}
