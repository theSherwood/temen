// Shared page-side Worker orchestration — one Temen guest's vCPUs across real Web Workers over one
// shared `WebAssembly.Memory`. Extracted from `main.js` (THREADS.md 4c-wasm) so the validation page
// and the playground (`play.js`) drive the exact same machinery. The page creates every Worker (no
// nested Workers) and never blocks (a browser bans main-thread `Atomics.wait`); the Workers do all
// the blocking (`worker.js`).

// Resolved relative to this module's URL (not the document root) so the page works both at the
// origin root (local `serve.mjs`) and under a subpath (GitHub Pages serves a project site at
// `/<repo>/`). The deployed site keeps the same `web/` + `target/…` layout, so `../target/…`
// resolves correctly under either base.
import { engineImports } from './engine-imports.js';
import { adoptEngine, registerForeign, releaseForeign } from './foreign-mem.js';
import { engineMemory } from './engine-mem.js';
import { jsCapServicer } from './coop-driver.js';
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
  const { exports: ex } = await WebAssembly.instantiate(module, engineImports(memory));
  adoptEngine(ex, memory); // detached children's own memories, through this instance's table
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
//   `winSize` sizes the window; `signal` (an `AbortSignal`) stops the run: the engine ends it as one
//   whose fuel ran out (`temen_x2_stop`), its Workers leave as after any run, and the promise rejects
//   with `stopped`. The engine stays usable. Only a run that has not ended `STOP_GRACE_MS` after it
//   was told has its Workers torn down; that error carries `broken`, as the engine's locks may then be
//   left held;
//   `onOutput(stdout, stderr)` (`Uint8Array`s) gets the run's output as its threads write it, in the
//   order its buffers hold it, as the coop driver's does;
//   `caps` (names) ⇒ with `onramp`, the host-answered caps the page answers (#1414 5b-2): a call to
//   one comes to `onCapPark({ id, index, op, args })`, which resolves to the call's result (a Number or
//   BigInt) while the run's other tasks go on, or to `null` to stop the run, as the pump's driver's
//   does;
//   `onCap(slot, op, args, mem)` ⇒ with `onramp`, the JS capabilities the page staged for the run
//   (`temen_onramp_stage_js_cap`, #2579), answered inside each call as the pump's driver answers
//   them: the calling task's Worker sends the call here and waits inside it. The call runs with
//   its domain's host let go (#2645), so the run's other tasks make their host calls meanwhile.
//   `mem` is the calling task's window, a detached child's included. `onCap` reaches the engine
//   only through `mem`, since the page may not wait on the engine's locks;
//   `capModule` (a module URL) ⇒ instead of `onCap`, the same capabilities answered by the module's
//   `onCap` export, which every Worker of the run imports: each call is answered in the calling
//   task's own Worker, without a trip to the page (#2645). The module's JS state is each Worker's
//   own, and it has a Worker's APIs, not the page's;
//   `stdinLater` ⇒ with `onramp`, the page feeds stdin as the run goes, through `feed(bytes)` and
//   `end()` on the run this returns: a read of empty stdin waits for more until `end()`. `onIdle()`
//   is called each time no task can run until the page acts, by answering or feeding.
// A run that ends in a trap rejects with an Error whose `faultAddr` is where a memory fault faulted,
// window-relative (`-1n` for none, #2275). The page services no authority: the powerboxes are
// in-Rust, and a Worker only hosts a thread the engine started. The page's main thread cannot wait
// on the engine's locks, so what it gives a run as it goes (answers, input) it leaves in the engine's
// memory with atomics alone (`temen_x2_answer`, `temen_x2_feed`), and wakes the Worker that pumps the
// run to take it (`temen_x2_wake`).
export function makeRunner({ module, memory, ex }) {
  const u8 = () => new Uint8Array(memory.buffer);
  const tlsSize = ex.__tls_size.value, tlsAlign = ex.__tls_align.value || 1;

  // `ctl` gets what the run's caller feeds it with while it goes (`stdinLater`).
  const runOn = async (guest, { jit = false, jitCodegen = false, jitService = 0, inst = false, io = false, onramp = false, stdin = null, env = null, tierup = false, unit = null, minter = 0, winSize = 1 << 16, signal = null, onOutput = null, jitRuntime = false, jitRuntimeCodegen = false, jitBlobs = [], caps = [], onCapPark = null, onCap = null, capModule = null, stdinLater = false, onIdle = null } = {}, ctl) => {
    if ((caps.length || stdinLater || onCap || capModule) && !onramp) {
      throw new Error('`caps`, `onCap`, `capModule` and `stdinLater` need the on-ramp powerbox (`onramp`)');
    }
    if (onCap && capModule) throw new Error('JS capabilities are answered by `onCap` or by `capModule`, not both');
    // A stop asked for a run before this one is withdrawn before this one starts, and what the page
    // left for an earlier run is dropped.
    ex.temen_x2_stop(0);
    // Once the run is over, neither its `signal` nor a stop it is still asking reaches the next run,
    // and nothing more is left for it.
    let finished = false, unlisten = null;
    // #1414 5b-2: what the page leaves for the run, in order. Each `put` leaves one message in the
    // engine's inbox, or answers `false` while the inbox is full. Then the Worker is woken to take
    // what is there, and the put is asked again a moment later. Once everything is left, the Worker
    // is woken to take it.
    const pending = [];
    const wake = () => {
      if (!finished && !ex.temen_x2_wake()) setTimeout(wake, 1);
    };
    const flush = () => {
      for (; pending.length && !finished; pending.shift()) {
        if (!pending[0]()) {
          wake();
          return void setTimeout(flush, 1);
        }
      }
      wake();
    };
    const give = (put) => {
      if (finished) return;
      pending.push(put);
      if (pending.length === 1) flush();
    };
    if (stdinLater) {
      const feedCap = ex.temen_x2_feed_cap();
      ctl.feed = (bytes) => {
        for (let at = 0; at < bytes.length; at += feedCap) {
          const chunk = bytes.slice(at, at + feedCap);
          give(() => {
            u8().set(chunk, Number(ex.temen_x2_feed_ptr()));
            return ex.temen_x2_feed(chunk.length) === 1;
          });
        }
      };
      ctl.end = () => give(() => ex.temen_x2_feed_end() === 1);
    }
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
      const names = new TextEncoder().encode(caps.join('\n'));
      const nptr = names.length ? ex.temen_par_alloc(names.length) : 0;
      if (names.length) u8().set(names, nptr);
      if (ex.temen_par_powerbox_onramp(gptr, guest.length, sptr, sin.length, nptr, names.length, stdinLater ? 1 : 0) !== 1) {
        throw new Error('temen_par_powerbox_onramp failed (not an on-ramp module, or a cap name that is not UTF-8)');
      }
    }
    if (inst) {
      let uptr = 0, ulen = 0;
      if (unit) {
        uptr = ex.temen_par_alloc(unit.length);
        u8().set(unit, uptr);
        ulen = unit.length;
      }
      if (ex.temen_par_powerbox_inst(uptr, ulen, BigInt(minter)) !== 1) {
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
    // Only a stop that the run outlasts tears these down, as the note on `signal` above says.
    let root = null;
    const threads = new Set();
    // `childMems`: how many threads ran over a detached child's own memory.
    let started = 0, childMems = 0;
    // The run's output as its Workers post it, handed on in `seq` order (see `stream_chunk`).
    const output = onOutput && outputInOrder(onOutput);
    // #2579: `onCap` over the calling task's window. A detached child's window is its own memory,
    // which the page addresses only once it has registered it under the id the child's Workers name
    // it by. Ids are each minting Worker's own, so two children's may meet: the page holds one
    // registration per id, the last caller's (`held`: id → its Worker), and lets them all go after.
    const jsCap = onCap && jsCapServicer(ex, memory, onCap);
    const held = new Map();
    const jsCall = (w, cfg, m) => {
      if (finished || !jsCap) return -38n; // -ENOSYS: no servicer, or the run is over
      if (cfg.x2Mem && held.get(cfg.x2MemId) !== w) {
        if (held.has(cfg.x2MemId)) releaseForeign(cfg.x2MemId);
        registerForeign(cfg.x2Mem, ex.temen_detached_header_bytes(), cfg.x2MemId);
        held.set(cfg.x2MemId, w);
      }
      return jsCap(m.slot, m.op, m.argsPtr, m.nArgs, m.mem);
    };
    let tornDown = false;
    try {
      const { value = null, exit = null } = await new Promise((resolve, reject) => {
        // `failed`: why the page stopped the run, when an `onCapPark` threw.
        let stopped = false, grace = null, failed = null;
        const stop = () => {
          stopped = true;
          // Asked until the run's lock was free to be told: it never waits on the page.
          const ask = () => {
            if (!finished && !ex.temen_x2_stop(1)) setTimeout(ask, 1);
          };
          ask();
          grace = setTimeout(() => {
            tornDown = true;
            reject(Object.assign(new Error('stopped'), { broken: true }));
          }, STOP_GRACE_MS);
        };
        if (signal) {
          if (signal.aborted) return reject(new Error('stopped'));
          signal.addEventListener('abort', stop, { once: true });
          unlisten = () => signal.removeEventListener('abort', stop);
        }
        // The run is over: the rest of its output, and how it ended — a stop's, when one was asked.
        const end = (settle) => {
          clearTimeout(grace);
          if (output) output.end(readParStdoutBytes({ memory, ex }), readParStderrBytes({ memory, ex }));
          if (stopped) reject(failed ?? new Error('stopped'));
          else settle();
        };
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
            } else if (m.kind === 'out') {
              if (output) output.chunk(m.fd, m.seq, m.bytes);
            } else if (m.kind === 'jscap') {
              // #2579: a JS capability call, which its thread waits inside: answered in its
              // Worker's cell (worker.js's `js_cap_call`), at once, and always, even with -EIO.
              let answer = -5n;
              try {
                answer = BigInt(jsCall(w, cfg, m));
              } catch (e) {
                console.error('JS capability call:', e);
              }
              new BigInt64Array(memory.buffer, m.cell + 8, 1)[0] = answer;
              const flag = new Int32Array(memory.buffer, m.cell, 1);
              Atomics.store(flag, 0, 1);
              Atomics.notify(flag, 0);
            } else if (m.kind === 'cap') {
              // #1414 5b-2: a call to a declared host-answered cap, which the page answers when it
              // can while the run goes on. An answer of `null` stops the run, and so does one that
              // is not an integer, with its error.
              Promise.resolve()
                .then(() => (onCapPark ? onCapPark({ id: m.id, index: m.index, op: m.op, args: m.args }) : null))
                .then((value) => {
                  if (value === null || value === undefined) return stop();
                  const answer = BigInt(value);
                  give(() => ex.temen_x2_answer(m.id, answer) === 1);
                })
                .catch((e) => {
                  failed = e;
                  stop();
                });
            } else if (m.kind === 'idle') {
              if (onIdle) onIdle();
            } else if (m.kind === 'done') {
              end(() => resolve({ value: BigInt(m.value) }));
            } else if (m.kind === 'exit') {
              end(() => resolve({ exit: m.code }));
            } else if (m.kind === 'trap' || m.kind === 'fail') {
              end(() => reject(Object.assign(new Error(m.why || 'guest trap'), { faultAddr: m.faultAddr })));
            }
          };
          w.onerror = (e) => reject(new Error(e.message || 'worker error'));
          w.postMessage({ module, memory, win, winSize, tierup, gptr, glen: guest.length, tierupCell, seedLen, stream: !!onOutput, capModule, ...cfg });
        };
        // The run starts on a Worker of its own (the page can't Atomics.wait).
        const stackTop = ex.temen_par_alloc(STACK) + STACK;
        const tlsBase = tlsSize > 0 ? roundUp(ex.temen_par_alloc(tlsSize + tlsAlign), tlsAlign) : 0;
        startWorker({ role: 'x2root', stackTop, tlsBase });
      });
      const tierups = counted ? Atomics.load(new Int32Array(memory.buffer), tierupCell >> 2) : 0;
      return { value, exit, started, tierups, childMems };
    } finally {
      finished = true;
      if (unlisten) unlisten();
      if (root) root.terminate();
      if (tornDown) for (const w of threads) w.terminate();
      for (const id of held.keys()) releaseForeign(id);
    }
  };
  // The run, which resolves as above, with `feed(bytes)` and `end()` for a run fed later.
  const fedLater = (f) => {
    if (!f) throw new Error('`feed` and `end` need a run fed later (`stdinLater`)');
    return f;
  };
  return function runAcrossWorkers(guest, opts) {
    const ctl = {};
    return Object.assign(runOn(guest, opts, ctl), {
      feed: (bytes) => fedLater(ctl.feed)(bytes),
      end: () => fedLater(ctl.end)(),
    });
  };
}

// How long a stopped run may take to end before its Workers are torn down (`signal` above).
const STOP_GRACE_MS = 5000;

// `onOutput` for a run whose chunks arrive from several Workers: each is handed on once every chunk
// numbered before it has been, so the stream reads as the run's buffers do. `end` hands on what the
// buffers hold past what was handed on, the chunks still on their way included; any that come later
// are dropped.
export function outputInOrder(onOutput) {
  const early = new Map();
  const sent = [0, 0, 0]; // bytes handed on, by fd
  let next = 1, over = false;
  const give = (fd, bytes) => {
    sent[fd] += bytes.length;
    if (fd === 2) onOutput(new Uint8Array(0), bytes);
    else onOutput(bytes, new Uint8Array(0));
  };
  return {
    chunk(fd, seq, bytes) {
      if (over) return;
      early.set(seq, { fd, bytes });
      for (let c; (c = early.get(next)); next++) {
        early.delete(next);
        give(c.fd, c.bytes);
      }
    },
    end(out, err) {
      over = true;
      const rest = [out.subarray(sent[1]), err.subarray(sent[2])];
      if (rest[0].length || rest[1].length) onOutput(rest[0], rest[1]);
    },
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
export function readParStderrBytes({ memory, ex }) {
  const len = ex.temen_par_stderr_len();
  const ptr = ex.temen_par_stderr_ptr();
  return new Uint8Array(memory.buffer).slice(ptr, ptr + len);
}
export function readParStderr(eng) {
  return new TextDecoder().decode(readParStderrBytes(eng));
}
