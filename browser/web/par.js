// Shared page-side Worker orchestration — one Temen guest's vCPUs across real Web Workers over one
// shared `WebAssembly.Memory`. Extracted from `main.js` (THREADS.md 4c-wasm) so the validation page
// and the playground (`play.js`) drive the exact same machinery. The page creates every Worker (no
// nested Workers) and never blocks (a browser bans main-thread `Atomics.wait`); the Workers do all
// the blocking (`worker.js`).

// Resolved relative to this module's URL (not the document root) so the page works both at the
// origin root (local `serve.mjs`) and under a subpath (GitHub Pages serves a project site at
// `/<repo>/`). The deployed site keeps the same `web/` + `target/…` layout, so `../target/…`
// resolves correctly under either base.
import { foreignImports } from './foreign-mem.js';
import { engineMemory } from './engine-mem.js';
const WASM = new URL('../target/wasm32-unknown-unknown/release/temen_browser.wasm', import.meta.url);
const STACK = 1 << 20;
const roundUp = (n, a) => (a > 1 ? Math.ceil(n / a) * a : n);

export async function fetchBytes(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: ${r.status}`);
  return new Uint8Array(await r.arrayBuffer());
}

// Compile + instantiate the threads wasm build over a fresh shared memory. Requires cross-origin
// isolation (the caller checks `self.crossOriginIsolated` first for a friendlier message).
//
// `prev` (an engine this returned earlier) reuses its compiled module: only a new memory and
// instance are made (~0.5 ms, against ~40 ms to fetch + compile). That is how a host runs many
// guests without accumulating memory: a run never frees what it allocates in the shared memory (its
// window, the vCPUs' stacks — some allocated inside Workers the page can only `terminate()`, with no
// signal that they have stopped), so the host drops the whole engine after a run and takes a fresh
// one for the next; the old memory goes with its Workers. Each memory holds an address-space
// reservation until the page collects it, though, so a fresh engine for every one of many quick runs
// runs out of them: the validation page (`main.js`) keeps its engine until a run ends other than by
// its root returning.
export async function loadEngine(prev = null, { maxPages: askPages } = {}) {
  const bytes = prev ? prev.bytes : await fetchBytes(WASM);
  const module = prev ? prev.module : await WebAssembly.compile(bytes);
  if (!WebAssembly.Module.imports(module).some((i) => i.kind === 'memory')) {
    throw new Error('not a threads build (no imported memory)');
  }
  // The engine's ceiling is the host's call, not the build's (`web/engine-mem.js`): `maxPages` asks
  // for one (the default otherwise), clamped to the `--max-memory` this build declares, so asking for
  // more than an older build allows yields that build's ceiling instead of a `LinkError`.
  const { memory, maxPages } = engineMemory(bytes, { maxPages: askPages });
  // The wasm imports `temen_host.webgpu_op` (the `webgpu` capability's host seam). It is a no-op unless a
  // page installs a real servicer on `globalThis.__temen_webgpu_op` (play.js does, backed by the page's
  // <canvas> + `navigator.gpu`). i64 args arrive as BigInt; the handler gets the shared `memory` so it
  // can read the WGSL bytes at `ptr`/`len`. Returns a BigInt (the wasm import result is i64).
  const importObj = {
    env: { memory },
    temen_host: {
      ...foreignImports(memory), // #1284: detached children's own memories (Region::Foreign)
      webgpu_op: (op, a, b, c, ptr, len) => {
        const h = globalThis.__temen_webgpu_op;
        return h ? BigInt(h(op, a, b, c, ptr, len, memory)) : -1n;
      },
      // The JS-defined powerbox's seam (`web/powerbox.js`, `src/jspb.rs`): one guest capability call —
      // `slot` says which name the page bound, `mem` is the opaque handle to the calling guest's
      // window. A no-op (-ENOSYS) unless a page installs a servicer on `globalThis.__temen_js_cap_call`
      // (`definePowerbox` does, for the duration of its run). Returns a BigInt (the import is i64).
      js_cap_call: (slot, op, argsPtr, nArgs, mem) => {
        const h = globalThis.__temen_js_cap_call;
        return h ? BigInt(h(Number(slot), Number(op), argsPtr, nArgs, mem)) : -38n;
      },
      // The live-stdout tee (`temen_run_onramp_stream`): each write's [ptr,ptr+len) in linear memory.
      // A no-op unless a page installs a sink on `globalThis.__temen_stdout_chunk` (play.js does during a
      // streaming Run) — it receives a fresh copy of the chunk (the wasm memory may move after).
      stdout_chunk: (ptr, len) => {
        const h = globalThis.__temen_stdout_chunk;
        if (h) h(new Uint8Array(memory.buffer, Number(ptr), Number(len)).slice());
      },
      // The parallel driver's platform (#1414 B6): it runs on a Worker (`worker.js`), never on the
      // page, which cannot block — so the page starts no thread of its own.
      x2_spawn: () => -1,
      x2_now_ms: () => performance.timeOrigin + performance.now(),
      x2_tierup: () => -1,
      x2_invoke: () => -1,
    },
  };
  const { exports: ex } = await WebAssembly.instantiate(module, importObj);
  return { module, bytes, memory, ex, maxPages };
}

// Build the runner over a loaded engine. The returned `runAcrossWorkers(guest, opts)` runs one guest
// on the parallel driver (executor 2, #1414 B6): the whole run is one in-Rust call on the root's
// Worker, and each of its threads runs on a Worker of its own. It resolves `{ value, exit, started,
// tierups, childMems }`. `guest` is the **encoded module bytes** (a fetched `.temenc` or an in-browser
// `temen_parse` product). Options:
//   `jit`     ⇒ build the Rust-side §22 powerbox: a host-compiled unit the root is handed;
//   `jitCodegen` ⇒ the same, with a guest's `Jit.invoke` running the unit on emitted wasm on its
//               thread's Worker; `jitService` selects the unit (0 = i32 `service`, 1 = f64 `fservice`);
//   `jitRuntime` ⇒ publish the runtime-`Jit.compile` powerbox (the guest compiles its own units, in
//               one shared host); `jitRuntimeCodegen` runs their invokes on emitted wasm, and
//               `jitBlobs` stages the unit blobs into the window, where the guest's `compile (ptr,
//               len)` reads them;
//   `inst`    ⇒ publish the §14 recipe (root `Instantiator` over the window + the optional granted
//               `unit` module bytes + a `WindowMinter` of `minter` bytes when > 0, #1286) — a root's
//               detached spawn starts each child on its own `WebAssembly.Memory`, each of its threads
//               on a Worker of its own;
//   `io`      ⇒ publish the 4d shared I/O powerbox (a `Mutex<Host>` in shared memory every thread
//               dispatches `call.cap` through; read stdout back via `temen_par_stdout_*` after);
//   `onramp`  ⇒ publish the **on-ramp** powerbox the same way (#152): the grants a single-threaded
//               on-ramp run gets, manifest imports included, shared by every thread — so an on-ramp
//               `.temen` whose runtime spawns threads runs them on real Workers. `stdin` (bytes)
//               seeds its stdin, and `env` (an array of `KEY=VALUE` strings, #1777) its §3e
//               environment — how a runtime learns per-run settings such as JACL's worker count;
//               an `exit(code)` from any thread ends the run with `exit` = code;
//   none      ⇒ the recipes are explicitly cleared (`temen_par_powerbox_none`) so a plain compute run
//               isn't seeded by a previous run's recipe;
//   `tierup`  ⇒ each thread runs its task's tier-ups on its own Worker, over a window of `winSize`
//               that cannot grow; a detached child's root task runs its granted module's emit over
//               the child's own memory (#1414 ED-2). `tierups` counts the emitted calls.
//   `winSize` sizes the window; `signal` (an `AbortSignal`) stops the run: every Worker is
//   terminated and the promise rejects. NOTE a stop tears down Workers mid-run — shared state (the
//   engine's locks) may be left unusable; reload the page after a stop.
// A run that ends in a trap rejects with an Error whose `faultAddr` is where a memory fault faulted,
// window-relative (`-1n` for none, #2275). The page services no authority: the powerboxes are
// in-Rust, and a Worker only hosts a thread the engine started.
export function makeRunner({ module, memory, ex }) {
  const u8 = () => new Uint8Array(memory.buffer);
  const tlsSize = ex.__tls_size.value, tlsAlign = ex.__tls_align.value || 1;

  return async function runAcrossWorkers(guest, { jit = false, jitCodegen = false, jitService = 0, inst = false, io = false, onramp = false, stdin = null, env = null, tierup = false, unit = null, minter = 0, winSize = 1 << 16, signal = null, jitRuntime = false, jitRuntimeCodegen = false, jitBlobs = [] } = {}) {
    const gptr = ex.temen_par_alloc(guest.length);
    u8().set(guest, gptr);
    // #2251: every thread gets its own Worker, so a guest's `self.parallelism` is the page's core count.
    ex.temen_par_set_workers(globalThis.navigator?.hardwareConcurrency || 1);
    if (jit && ex.temen_par_powerbox(gptr, guest.length) !== 1) throw new Error('temen_par_powerbox failed');
    if (jitCodegen) {
      ex.temen_par_jit_codegen_service(jitService);
      if (ex.temen_par_powerbox_jit_codegen(gptr, guest.length) !== 1) throw new Error('temen_par_powerbox_jit_codegen failed');
    }
    if (jitRuntime) {
      if (ex.temen_par_powerbox_jit_runtime(gptr, guest.length) !== 1) throw new Error('temen_par_powerbox_jit_runtime failed');
      ex.temen_par_jit_set_codegen(jitRuntimeCodegen ? 1 : 0);
    }
    if (io && ex.temen_par_powerbox_io() !== 1) throw new Error('temen_par_powerbox_io failed');
    if (onramp) {
      const sin = stdin || new Uint8Array(0);
      const sptr = sin.length ? ex.temen_par_alloc(sin.length) : 0;
      if (sin.length) u8().set(sin, sptr);
      // Always set — an empty list clears — so one run's environment never reaches the next.
      const envb = new TextEncoder().encode((env || []).map((e) => `${e}\0`).join(''));
      const eptr = envb.length ? ex.temen_par_alloc(envb.length) : 0;
      if (envb.length) u8().set(envb, eptr);
      if (ex.temen_set_run_env(eptr, envb.length) !== 0) throw new Error('temen_set_run_env refused the environment (an entry without `=`, or too large)');
      if (ex.temen_par_powerbox_onramp(gptr, guest.length, sptr, sin.length) !== 1) throw new Error('temen_par_powerbox_onramp failed (not an on-ramp module)');
    }
    if (inst) {
      let uptr = 0, ulen = 0;
      if (unit) {
        uptr = ex.temen_par_alloc(unit.length);
        u8().set(unit, uptr);
        ulen = unit.length;
      }
      if (ex.temen_par_powerbox_inst(BigInt(winSize), uptr, ulen, BigInt(minter)) !== 1) {
        throw new Error('temen_par_powerbox_inst failed');
      }
    }
    if (!jit && !jitCodegen && !io && !onramp && !inst && !jitRuntime) ex.temen_par_powerbox_none();
    // A runtime-compile guest reads its unit blobs from the window: they are staged here, and the run
    // copies them into its own window (`seedLen` bytes from `win`).
    const seedLen = jitBlobs.reduce((n, b) => Math.max(n, b.off + b.bytes.length), 0);
    const win = seedLen ? ex.temen_par_alloc(seedLen) : 0;
    if (seedLen && win === 0) throw new Error(`could not allocate ${seedLen} bytes to stage the unit blobs`);
    for (const b of jitBlobs) u8().set(b.bytes, win + b.off);

    // A shared i32 cell every Worker atomically bumps each time it runs an emitted call (a tier-up or
    // a §22 invoke), so the caller can prove the seam actually fired (a result match alone can't tell
    // "ran emitted wasm" from "silently interpreted"). Read back after the run.
    const counted = tierup || jitCodegen || jitRuntimeCodegen;
    const tierupCell = counted ? ex.temen_par_alloc(4) : 0;

    // A thread's Worker closes itself once its thread is out of the engine, and the page leaves it
    // be: the thread counts itself out of the run before it is done in the engine (it still drops
    // what it held). Chromium stops a terminated Worker that is still running about two seconds
    // later, wherever it is, and one stopped inside the engine's allocator holds its lock for good.
    // Only a stop tears these down too, as the note on `signal` above says.
    let root = null;
    const threads = new Set();
    // `childMems`: how many threads ran over a detached child's own memory.
    let started = 0, childMems = 0;
    try {
      const { value = null, exit = null } = await new Promise((resolve, reject) => {
        if (signal) {
          if (signal.aborted) return reject(new Error('stopped'));
          signal.addEventListener('abort', () => reject(new Error('stopped')), { once: true });
        }
        const startWorker = (cfg) => {
          started++;
          const w = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
          if (cfg.role === 'x2thread') threads.add(w);
          else root = w;
          w.onmessage = (e) => {
            const m = e.data;
            if (m.kind === 'x2spawn') {
              // A thread of the run: the boxed closure at `start`, on a Worker of its own, with the
              // detached child's memory its window lives in, if any (B6-3c).
              if (m.x2Mem) childMems++;
              startWorker({ role: 'x2thread', start: m.start, stackTop: m.stackTop, tlsBase: m.tlsBase, x2Mem: m.x2Mem, x2MemId: m.x2MemId });
            } else if (m.kind === 'done') {
              resolve({ value: BigInt(m.value) });
            } else if (m.kind === 'exit') {
              resolve({ exit: m.code });
            } else if (m.kind === 'trap' || m.kind === 'fail') {
              reject(Object.assign(new Error(m.why || 'guest trap'), { faultAddr: m.faultAddr }));
            }
          };
          w.onerror = (e) => reject(new Error(e.message || 'worker error'));
          w.postMessage({ module, memory, win, winSize, tierup, gptr, glen: guest.length, tierupCell, seedLen, ...cfg });
        };
        // The run starts on a Worker of its own (the page can't Atomics.wait).
        const stackTop = ex.temen_par_alloc(STACK) + STACK;
        const tlsBase = tlsSize > 0 ? roundUp(ex.temen_par_alloc(tlsSize + tlsAlign), tlsAlign) : 0;
        startWorker({ role: 'x2root', stackTop, tlsBase });
      });
      const tierups = counted ? Atomics.load(new Int32Array(memory.buffer), tierupCell >> 2) : 0;
      return { value, exit, started, tierups, childMems };
    } finally {
      if (root) root.terminate();
      if (signal && signal.aborted) for (const w of threads) w.terminate();
    }
  };
}

// Read back the accumulated stdout of the last 4d I/O or on-ramp run (empty string when no I/O
// powerbox ran). `slice` (not `subarray`) copies out of the SharedArrayBuffer — TextDecoder rejects
// shared views. `readParStdoutBytes` is the raw form, for binary output.
export function readParStdoutBytes({ memory, ex }) {
  const len = ex.temen_par_stdout_len();
  const ptr = ex.temen_par_stdout_ptr();
  return new Uint8Array(memory.buffer).slice(ptr, ptr + len);
}
export function readParStdout(eng) {
  return new TextDecoder().decode(readParStdoutBytes(eng));
}
// The on-ramp run's stderr (the shared host's, as the single-threaded on-ramp run returns it).
export function readParStderr({ memory, ex }) {
  const len = ex.temen_par_stderr_len();
  const ptr = ex.temen_par_stderr_ptr();
  return new TextDecoder().decode(new Uint8Array(memory.buffer).slice(ptr, ptr + len));
}
