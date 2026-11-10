// The page's executor for the `webgpu` capability (#2665). The engine decodes and checks every stream
// a guest submits (`crates/temen-gpu`), as the native host does, and hands this file only the WebGPU
// calls a checked stream stands for: JSON steps `[target, method, args, into?]` (`temen_gpu::js`).
// This file makes them by name against `navigator.gpu`. It knows nothing of WebGPU's API beyond the
// three helpers below, so the protocol has one implementation, in Rust (INVARIANTS #15).
//
// `initWebGPU(canvas)` brings up a device and the canvas context, and installs the executor on
// `globalThis.__temen_webgpu`, which the engine's `temen_host.webgpu_*` imports call
// (`engine-imports.js`). Device creation, the one async step, is awaited before a guest runs, so every
// call is synchronous, which the reactor on the main thread requires. A readback resolves later; the
// guest asks for it on a later tick (`-EAGAIN` until then). WebGPU reports its errors through error
// scopes, which resolve later too, so a guest reads them with its `error` op.
//
// Each capability handler is a session of its own: its ids, the results it registers, its readbacks
// and its first error. Two domains granted the capability never see each other's objects.

let G = null; // { device, ctx, format, canvas, sessions: Map<number, Session> }

export function webgpuAvailable() {
  return typeof navigator !== 'undefined' && !!navigator.gpu;
}

// `format` is the canvas's format, `navigator.gpu`'s preferred one unless a caller names another.
export async function initWebGPU(canvas, { format } = {}) {
  if (!navigator.gpu) throw new Error('WebGPU is not available in this browser');
  const adapter =
    (await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })) ||
    (await navigator.gpu.requestAdapter({ forceFallbackAdapter: true }));
  if (!adapter) throw new Error('no WebGPU adapter');
  const device = await adapter.requestDevice();
  const ctx = canvas.getContext('webgpu');
  const fmt = format ?? navigator.gpu.getPreferredCanvasFormat();
  // COPY_SRC lets a guest read its own frame back.
  ctx.configure({
    device,
    format: fmt,
    alphaMode: 'opaque',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });
  G = { device, ctx, format: fmt, canvas, sessions: new Map() };
  globalThis.__temen_webgpu = { submit, read, error, surface, drop };
}

export function teardownWebGPU() {
  globalThis.__temen_webgpu = null;
  try {
    G?.device.destroy();
  } catch {
    /* ignore */
  }
  G = null;
}

function session(id) {
  let s = G.sessions.get(id);
  if (!s) {
    s = { objs: new Map(), regs: {}, reads: new Map(), error: null };
    G.sessions.set(id, s);
  }
  return s;
}

// A copy of `[ptr, ptr + len)` of the engine's memory: WebGPU and `TextDecoder` refuse a view over the
// threads build's SharedArrayBuffer, and the memory may move as it grows.
const copy = (memory, ptr, len) => new Uint8Array(memory.buffer, Number(ptr), Number(len)).slice();

const note = (s, e) => {
  if (e && s.error === null) s.error = e.message ?? String(e);
};

// `{"$": n}` is object `n` or register `n`; `{"@": [offset, length]}` is those bytes of `data`.
function resolve(s, v, data) {
  if (Array.isArray(v)) return v.map((x) => resolve(s, x, data));
  if (v === null || typeof v !== 'object') return v;
  if ('$' in v) return typeof v.$ === 'number' ? s.objs.get(v.$) : s.regs[v.$];
  if ('@' in v) return data.subarray(v['@'][0], v['@'][0] + v['@'][1]);
  const out = {};
  for (const k in v) out[k] = resolve(s, v[k], data);
  return out;
}

// The executor's own steps: what WebGPU in a page does differently from WebGPU natively.
function helpers(s) {
  return {
    // A copy of a buffer's range, mapped once it has arrived, for the guest's `read`.
    readback(src, offset, size) {
      const staging = G.device.createBuffer({
        size,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      const enc = G.device.createCommandEncoder();
      enc.copyBufferToBuffer(s.objs.get(src), offset, staging, 0, size);
      G.device.queue.submit([enc.finish()]);
      const entry = { bytes: null };
      s.reads.set(src, entry);
      staging.mapAsync(GPUMapMode.READ).then(
        () => {
          entry.bytes = new Uint8Array(staging.getMappedRange()).slice();
          staging.destroy();
        },
        (e) => {
          if (s.reads.get(src) === entry) s.reads.delete(src);
          note(s, e);
          staging.destroy();
        },
      );
    },
    // A page presents its canvas when the task that drew it ends; the frame's ids go.
    present(texture, view) {
      s.objs.delete(texture);
      s.objs.delete(view);
    },
    destroy(id) {
      s.objs.get(id)?.destroy?.();
      s.objs.delete(id);
      s.reads.delete(id);
    },
  };
}

function target(s, t) {
  if (typeof t === 'number') return s.objs.get(t);
  if (t === 'device') return G.device;
  if (t === 'queue') return G.device.queue;
  if (t === 'ctx') return G.ctx;
  if (t === 'js') return helpers(s);
  return s.regs[t];
}

function submit(id, memory, json, jsonLen, blob, blobLen) {
  const s = session(id);
  const steps = JSON.parse(new TextDecoder().decode(copy(memory, json, jsonLen)));
  const data = copy(memory, blob, blobLen);
  G.device.pushErrorScope('out-of-memory');
  G.device.pushErrorScope('validation');
  try {
    for (const [t, method, args, into] of steps) {
      const r = target(s, t)[method](...resolve(s, args, data));
      if (typeof into === 'number') s.objs.set(into, r);
      else if (into !== undefined) s.regs[into] = r;
    }
  } catch (e) {
    note(s, e);
  }
  // A scope that cannot resolve (a lost device) is the guest's error too, never the page's.
  for (let i = 0; i < 2; i++) G.device.popErrorScope().then((e) => note(s, e), (e) => note(s, e));
  return 0;
}

function read(id, memory, buffer, dst, cap) {
  const s = G.sessions.get(id);
  const e = s?.reads.get(buffer);
  if (!e) return -2;
  if (!e.bytes) return -11;
  if (Number(cap) >= e.bytes.length) {
    new Uint8Array(memory.buffer, Number(dst), e.bytes.length).set(e.bytes);
    s.reads.delete(buffer);
  }
  return e.bytes.length;
}

function error(id, memory, dst, cap) {
  const s = G.sessions.get(id);
  if (!s || s.error === null) return 0;
  const b = new TextEncoder().encode(s.error);
  const n = Math.min(b.length, Number(cap));
  new Uint8Array(memory.buffer, Number(dst), n).set(b.subarray(0, n));
  s.error = null;
  return b.length;
}

function surface(memory, dims, name, cap) {
  const d = new Uint32Array(memory.buffer, Number(dims), 2);
  d[0] = G.canvas.width;
  d[1] = G.canvas.height;
  const b = new TextEncoder().encode(G.format);
  const n = Math.min(b.length, Number(cap));
  new Uint8Array(memory.buffer, Number(name), n).set(b.subarray(0, n));
  return b.length;
}

function drop(id) {
  const s = G?.sessions.get(id);
  if (!s) return;
  for (const o of s.objs.values()) o?.destroy?.();
  G.sessions.delete(id);
}
