// #1122 — the interactive bash card's Workers (two roles over one shared linear memory):
//
//   session  — makes the ONE blocking call `temen_bash_session(...)`: real GNU bash runs `-i` on
//              the cooperative bytecode engine inside this Worker for the whole session, its pump
//              parked on the external-wake doorbell whenever bash waits at the prompt. A browser
//              permits that block only off the main thread — this Worker IS the terminal's
//              "process".
//   control  — the non-blocking side: it delivers keystrokes into the live session
//              (`temen_bash_feed` → the #797 feed-time line discipline in shared memory) and
//              polls `temen_bash_drain` for new stdout/stderr, posting each chunk to the page.
//              A separate Worker (not the page) because these calls take real Mutexes — under
//              contention they `Atomics.wait`, which the main thread bans.
//
// Both roles instantiate the SAME threads-build module over the SAME shared memory (statics —
// including the session control block — live in that memory), each with its own stack + TLS
// block, exactly like `worker.js`'s per-vCPU bootstrap. One synchronous message handler owns the
// whole lifecycle: keystrokes that land while the instantiate is still in flight are BUFFERED and
// fed on ready (the page enables its input the moment the Workers exist, so this race is real).

let ex = null;
let memory = null;
let drainBuf = 0;
const DRAIN_CAP = 1 << 16;
const pending = []; // keystroke batches that arrived before the instantiate finished
const dec = new TextDecoder();
const u8 = () => new Uint8Array(memory.buffer); // re-taken per use: the shared memory can grow

const feed = (bytes) => {
  const p = ex.temen_alloc(bytes.length);
  u8().set(bytes, p);
  ex.temen_bash_feed(p, bytes.length);
  ex.temen_dealloc(p, bytes.length);
};

const drainOnce = () => {
  for (const kind of [0, 1]) {
    // Loop each stream dry (a burst larger than the buffer spans several drains). `slice` (not
    // `subarray`): TextDecoder rejects views over a SharedArrayBuffer.
    for (;;) {
      const n = ex.temen_bash_drain(kind, drainBuf, DRAIN_CAP);
      if (n === 0) break;
      postMessage({ kind: kind === 0 ? 'out' : 'err', text: dec.decode(u8().slice(drainBuf, drainBuf + n)) });
      if (n < DRAIN_CAP) break;
    }
  }
};

async function init(cfg) {
  const { module, role, stackTop, tlsBase, modPtr, modLen, binsPtr, binsLen } = cfg;
  memory = cfg.memory;
  let exports;
  try {
    ({ exports } = await WebAssembly.instantiate(module, {
      env: { memory },
      temen_host: { webgpu_op: () => -1n }, // no GPU surface in a bash Worker
    }));
    exports.__stack_pointer.value = stackTop;
    if (exports.__tls_size.value > 0) exports.__wasm_init_tls(tlsBase);
  } catch (err) {
    postMessage({ kind: 'fail', why: String(err) });
    return;
  }

  if (role === 'session') {
    try {
      const rc = exports.temen_bash_session(modPtr, modLen, binsPtr, binsLen);
      postMessage({ kind: 'exit', rc });
    } catch (err) {
      postMessage({ kind: 'fail', why: String(err) });
    }
    return;
  }

  // control: publish `ex`, flush the pre-ready keystrokes, then poll-drain (~25 fps is plenty).
  ex = exports;
  drainBuf = ex.temen_alloc(DRAIN_CAP);
  for (const b of pending.splice(0)) feed(b);
  const timer = setInterval(() => {
    try {
      drainOnce();
      const rc = ex.temen_bash_exited();
      if (rc >= 0) {
        clearInterval(timer);
        drainOnce(); // one last sweep so the farewell after the exit mark is not lost
        postMessage({ kind: 'done', rc });
      }
    } catch (err) {
      clearInterval(timer);
      postMessage({ kind: 'fail', why: String(err) });
    }
  }, 40);
}

self.onmessage = (e) => {
  const m = e.data;
  if (m.role) {
    init(m); // async; keystrokes arriving meanwhile buffer below
  } else if (m.kind === 'keys' && m.bytes && m.bytes.length) {
    if (ex) feed(m.bytes);
    else pending.push(m.bytes);
  }
};
