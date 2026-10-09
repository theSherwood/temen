// The **snapshot worker** (WASM_AOT.md warm+JIT · issue #804). Owns the warm-runtime snapshot session for
// the playground's warm cards (QuickJS) on a dedicated Web Worker with its **own private engine instance
// and memory** — so the ~one-time QuickJS `warmup` (`temen_warm_open`) and every eval run entirely off the
// main thread. The main thread pre-warms at page load and dispatches each Run as a message; it never
// blocks on the ~0.9 s warmup or a compute-heavy eval.
//
// Why private memory (not the page's shared engine memory): `worker.js`'s concurrent path documents a
// "rare shared-memory race (a double-free)" in the shared setup. This Worker instantiates the engine over
// a fresh memory of its own and allocates only there, so its warm session can't race the main thread's
// allocator. Main ↔ worker communicate only by messages (source string in; stdout/status/value out).
import { foreignImports } from './foreign-mem.js';
import { engineMemory } from './engine-mem.js';
import { runWarmJit, runWarmCoop, primeWarmJit, jitCacheStats, runJitModule, nimToolchain, nimCompileRun } from './wasmjit-module.js';

let ex = null; // the worker's own engine exports
let memory = null; // the worker's own (private) shared WebAssembly.Memory
let warmUrl = null; // the module URL the warm session is currently open for (null ⇒ none)
// The background warm+JIT pre-compile (issue: the first wasm-JIT Run paid the ~12.7 MB `WebAssembly.
// compile`). `jitPrimePromise` is the in-flight (or settled) `primeWarmJit`; `jitPrimed` flips true once
// it settles (compiled, or known non-emittable). A wasm-JIT eval awaits it so the first Run is a cache hit.
let jitPrimePromise = null;
let jitPrimed = false;
// Whether this worker's warm+JIT emit outlined the hot `eval_run` (#1120 hot-function outlining). Set from
// the `prewarm` message before the first `temen_warm_jit_open`; reported in `stats` so an A/B harness can
// confirm which variant a Run used.
let warmSplit = false;
// The background warm-**interp** pre-run. Even the interpreter eval path has a first-call cost: V8
// compiles the engine's `eval_run` interpreter functions lazily on the first real eval, and `warmup`
// (Tcl_Init / QuickJS init) doesn't exercise all of them — so the user's first warm-snapshot Run paid
// that codegen (measured ~0.1–0.7 s with the deployed engine). A dry eval at pre-warm compiles them off
// the main thread; a warm-interp Run awaits it so its first Run reuses the warmed code.
let interpWarmPromise = null;
// nimony's toolchain (`nimToolchain` over the nim card's `nimony.blob.gz`), kept after the first
// `nimAssets` message so each `nimCompile` Run posts only its source.
let nim = null;
// The engine holds one nim session at a time, so each `nimCompile` waits for the one before it.
let nimTurn = Promise.resolve();
// Active only during a streaming Run (`runStream`, `runJitStream`, `eval`): a fn that relays one stdout
// chunk to the main thread. `null` at rest so the `stdout_chunk` import is a no-op for warm/prime dry runs.
let chunkSink = null;

const u8 = () => new Uint8Array(memory.buffer);
// Read a captured stream out of the worker's memory. `.slice` copies to a non-shared buffer (TextDecoder
// rejects a SharedArrayBuffer-backed view), exactly as play.js's `readModuleStdout` does.
const readStr = (ptrFn, lenFn) => {
  const p = Number(ex[ptrFn]());
  const l = Number(ex[lenFn]());
  return p && l ? new TextDecoder().decode(u8().slice(p, p + l)) : '';
};
const readStdout = () => readStr('temen_stdout_ptr', 'temen_stdout_len');
const readStderr = () => readStr('temen_stderr_ptr', 'temen_stderr_len');

// Open (or reuse) the warm session for `url`'s module `bytes` — runs `warmup` once and snapshots the
// post-init image (mirrors play.js's `ensureWarmSession`). Returns true on success; false if the module
// isn't a warm-snapshot driver, `warmup` traps, or a re-open is needed but no bytes were supplied.
function ensureWarm(url, bytes) {
  if (warmUrl === url) return true;
  if (!bytes) return false;
  const p = Number(ex.temen_alloc(bytes.length));
  u8().set(bytes, p);
  const live = Number(ex.temen_warm_open(p, bytes.length));
  ex.temen_dealloc(p, bytes.length);
  if (live < 0 || ex.temen_status() !== 0) {
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
      return { tier: 'warm+JIT', status, value: Number(ex.temen_run_value()), stdout: readStdout(), stderr: readStderr() };
    } catch {
      // whole-program decline / trap → the cooperative tier-up drive (#816 item 4): a page-managing
      // eval (grows/protects during the eval, so it isn't WasmDriven) still runs its eligible pure
      // leaves on emitted wasm. driveCoopTierupRun stages the result in the same run-value/stdout slots.
    }
    try {
      const status = await runWarmCoop(ex, memory, stdinBytes, `${warmUrl}#coop`, 1);
      return { tier: 'warm-coop', status, value: Number(ex.temen_run_value()), stdout: readStdout(), stderr: readStderr() };
    } catch {
      // coop decline / trap → fall through to the warm interpreter (always correct, just unaccelerated)
    }
  }
  let stdinP = 0;
  const len = stdinBytes ? stdinBytes.length : 0;
  if (len) {
    stdinP = Number(ex.temen_alloc(len));
    u8().set(stdinBytes, stdinP);
  }
  const value = Number(ex.temen_warm_eval(stdinP, len));
  const status = ex.temen_status();
  const stdout = readStdout();
  const stderr = readStderr();
  if (stdinP) ex.temen_dealloc(stdinP, len);
  return { tier: 'warm-snapshot', status, value, stdout, stderr };
}

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === 'init') {
      // The main thread posts the already-compiled engine `WebAssembly.Module` (structured-cloneable),
      // so the worker skips a second fetch+compile. It instantiates over its OWN shared memory (threads
      // build imports one) with the no-op webgpu stub (a compute guest never presents).
      // The ceiling the spawning engine settled on (`web/engine-mem.js`). A client that sends none is
      // an older one: fall back to the historical 1 GiB, which every build declares at least.
      ({ memory } = engineMemory(null, { maxPages: msg.maxPages ?? 16384 }));
      ({ exports: ex } = await WebAssembly.instantiate(msg.module, {
        env: { memory },
        temen_host: {
          ...foreignImports(memory),
          webgpu_op: () => -1n,
          // No page-defined powerbox in a snapshot worker (`web/powerbox.js` runs on the main thread).
          js_cap_call: () => -38n,
          // No parallel-driver run here (#1414 B6): a refused `x2_spawn` ends one with `ThreadFault`.
          x2_spawn: () => -1,
          x2_now_ms: () => performance.timeOrigin + performance.now(),
          x2_tierup: () => -1,
          x2_invoke: () => -1,
          // The live-stdout tee (`temen_run_onramp_stream`): while a streaming Run is active, `chunkSink`
          // relays each write to the main thread; the worker's run stays synchronous, so the main thread
          // paints the chunks as they arrive (a Worker's postMessage delivers even while it computes).
          stdout_chunk: (ptr, len) => {
            if (chunkSink) chunkSink(new Uint8Array(memory.buffer, Number(ptr), Number(len)).slice());
          },
        },
      }));
      self.postMessage({ type: 'ready' });
      return;
    }
    if (msg.type === 'prewarm') {
      // #1120: outline the hot `eval_run` when requested, BEFORE the first `temen_warm_jit_open` (below)
      // caches the emit — the engine's split toggle is a global read at open time. Default off keeps the
      // shipping warm cards byte-identical. `temen_warm_jit_set_split` is absent on pre-#1131 engines.
      warmSplit = !!msg.split;
      if (ex.temen_warm_jit_set_split) ex.temen_warm_jit_set_split(warmSplit ? 1 : 0);
      const ok = ensureWarm(msg.url, msg.bytes);
      // Reply as soon as warm-interp is ready (the ~0.9 s `warmup`) — do NOT block on the JIT pre-compile,
      // so an early interpreter Run isn't delayed by it.
      self.postMessage({ type: 'reply', id: msg.id, ok, status: ok ? 0 : ex.temen_status() });
      if (ok && !interpWarmPromise) {
        // First, a dry warm-**interp** eval (empty input) so V8 compiles the interpreter eval path now,
        // off the main thread — the first real warm-snapshot Run then pays no first-call codegen.
        interpWarmPromise = evalWarm('', false).catch(() => {});
        // Then pre-compile the warm+**JIT** `eval_run` (the ~12.7 MB `WebAssembly.compile` + a dry f0
        // call), so the first wasm-JIT Run is a cache hit. Chained AFTER the interp warm so the two never
        // touch the warm session concurrently (each restores + runs over the shared image). Best-effort;
        // skipped when `primeJit` is false (a card whose warm+JIT declines, e.g. Tcl).
        if (msg.primeJit !== false && !jitPrimePromise) {
          jitPrimePromise = interpWarmPromise
            .then(() => primeWarmJit(ex, memory, `${warmUrl}#eval`, 1))
            .catch(() => false)
            .finally(() => { jitPrimed = true; });
        }
      }
      return;
    }
    if (msg.type === 'eval') {
      if (!ensureWarm(msg.url, msg.bytes)) {
        self.postMessage({ type: 'reply', id: msg.id, ok: false, error: 'warm session unavailable' });
        return;
      }
      // Wait for the relevant background warm-up so this Run reuses it (and never races the session): a
      // wasm-JIT Run waits for the JIT pre-compile; a warm-interp Run waits for the dry interp pre-run.
      if (msg.jit) { if (jitPrimePromise) { try { await jitPrimePromise; } catch { /* prime failed → runWarmJit falls back */ } } }
      else if (interpWarmPromise) { try { await interpWarmPromise; } catch { /* dry eval failed → just run */ } }
      // Live-stream the eval's stdout to the page (#1142): the tee on the warm host (both tiers) fires
      // `stdout_chunk`, relayed here for the duration of the eval.
      chunkSink = (bytes) => self.postMessage({ type: 'stdout-chunk', id: msg.id, bytes }, [bytes.buffer]);
      let r;
      try {
        r = await evalWarm(msg.source, msg.jit);
      } finally {
        chunkSink = null;
      }
      self.postMessage({ type: 'reply', id: msg.id, ok: true, ...r });
      return;
    }
    if (msg.type === 'runStream') {
      // Run a plain on-ramp module (nim/C guest) off the main thread with **live stdout**: the tee
      // (`temen_run_onramp_stream`) posts each write to the page as the guest produces it, so a long or
      // chatty program's output appears progressively instead of in one dump at the end. The final
      // captured stdout is returned too (the page overwrites with it for an exact end state).
      const mod = msg.bytes;
      const stdin = msg.stdin && msg.stdin.length ? msg.stdin : null;
      const mp = Number(ex.temen_alloc(mod.length));
      const sp = stdin ? Number(ex.temen_alloc(stdin.length)) : 0;
      const view = u8();
      view.set(mod, mp);
      if (sp) view.set(stdin, sp);
      chunkSink = (bytes) => self.postMessage({ type: 'stdout-chunk', id: msg.id, bytes }, [bytes.buffer]);
      // The guest's result is this export's return value (like `temen_run_onramp` on the main
      // thread) — it never fills the `temen_run_value` slot, which the JIT/warm tiers own; reading
      // that slot here reported 0 for every streamed module (#1509's Temen card caught it).
      let value;
      try {
        value = Number(ex.temen_run_onramp_stream(mp, mod.length, sp, stdin ? stdin.length : 0));
      } finally {
        chunkSink = null;
      }
      const status = ex.temen_status();
      // Return the framebuffer too (a `display`-cap guest like gradient presents one), so a streamed
      // run stays feature-equivalent to the main-thread path — the page blits it from these bytes.
      const fbw = Number(ex.temen_framebuffer_width());
      const fbh = Number(ex.temen_framebuffer_height());
      let fb = null;
      if (fbw && fbh) {
        const fp = Number(ex.temen_framebuffer_ptr());
        const fl = Number(ex.temen_framebuffer_len());
        fb = { w: fbw, h: fbh, rgba: u8().slice(fp, fp + fl) };
      }
      ex.temen_dealloc(mp, mod.length);
      if (sp) ex.temen_dealloc(sp, stdin.length);
      self.postMessage(
        { type: 'reply', id: msg.id, ok: true, status, value, stdout: readStdout(), stderr: readStderr(), fb },
        fb ? [fb.rgba.buffer] : [],
      );
      return;
    }
    if (msg.type === 'runJitStream') {
      // Run a module's `_start` on the **wasm-JIT tier** off the main thread, with **live stdout** (#1141):
      // `runJitModule` emits `_start`, drives `f0`, and bounces cross-tier `write`s to the interpreter — the
      // tee on that run's host fires `stdout_chunk`, relayed here as the guest writes. A JIT decline/trap
      // throws → reply `ok:false` so the page falls back to the interpreter path.
      const mod = msg.bytes;
      const stdin = msg.stdin && msg.stdin.length ? msg.stdin : null;
      chunkSink = (bytes) => self.postMessage({ type: 'stdout-chunk', id: msg.id, bytes }, [bytes.buffer]);
      let status;
      try {
        status = await runJitModule(ex, memory, mod, stdin, msg.cacheKey);
      } catch (err) {
        chunkSink = null;
        self.postMessage({ type: 'reply', id: msg.id, ok: false, error: String((err && err.message) || err) });
        return;
      } finally {
        chunkSink = null;
      }
      const value = Number(ex.temen_run_value());
      self.postMessage({ type: 'reply', id: msg.id, ok: true, status, value, stdout: readStdout(), stderr: readStderr() });
      return;
    }
    if (msg.type === 'nimAssets') {
      nim = nimToolchain(msg.bundle);
      self.postMessage({ type: 'reply', id: msg.id, ok: true });
      return;
    }
    if (msg.type === 'nimCompile') {
      // Compile a whole Nim program with nimony's own driver on THIS worker, and run it
      // (`nimCompileRun`), so a build — or a program that never returns — stalls only this worker,
      // never the page.
      if (!nim) {
        self.postMessage({ type: 'reply', id: msg.id, ok: false, error: 'nim assets not loaded' });
        return;
      }
      const run = nimTurn.then(() => nimCompileRun(ex, memory, nim, msg.source));
      nimTurn = run.catch(() => {});
      const r = await run;
      // The compiled-module cache's counters and the engine's emits, so a caller can see a later build
      // reuse this worker's leaf images: compiled, and emitted (#2087).
      const cache = { compiles: jitCacheStats.compiles, hits: jitCacheStats.hits, emits: ex.temen_coop_leaf_emits() };
      self.postMessage({ type: 'reply', id: msg.id, ok: true, ...r, ...cache });
      return;
    }
    if (msg.type === 'stats') {
      // Test/telemetry hook: whether the warm+JIT pre-compile has settled, and the worker's JIT cache
      // accounting (a primed instance ⇒ the first wasm-JIT Run is a cache hit, not a fresh compile).
      self.postMessage({ type: 'reply', id: msg.id, ok: true, jitPrimed, split: warmSplit, compiles: jitCacheStats.compiles, hits: jitCacheStats.hits });
      return;
    }
  } catch (err) {
    self.postMessage({ type: 'reply', id: msg.id, ok: false, error: String((err && err.message) || err) });
  }
};
