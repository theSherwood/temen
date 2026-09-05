// THREADS/BROWSER step 4c-wasm in a REAL browser — the per-vCPU Web Worker. One guest vCPU runs here
// through the engine's resumable `Vcpu` API (`temen_par_run` → a host-serviced event → deliver → run
// again) over the ONE shared linear memory. This is the browser twin of `threads-spawn.mjs`'s
// `worker()`: the only differences are init delivery (a `postMessage` instead of Node `workerData`)
// and that a spawn request is posted to the page (which creates every Worker — no nested Workers).
//
// The host services events with genuine browser primitives: `thread.join` → `Atomics.wait` on the
// child's completion slot; `memory.wait`/`notify` → `Atomics.wait`/`notify` on the futex word. A Worker
// (not the page) is the only place a browser permits a blocking `Atomics.wait`.

import { foreignImports, registerForeign } from './foreign-mem.js';
const STACK = 1 << 20; // per-Worker stack
const SLOT = 16; // completion slot: [done:i32 @0][result:i64 @8]
const roundUp = (n, a) => (a > 1 ? Math.ceil(n / a) * a : n);
// Event codes — must match browser/src/lib.rs PAR_*.
const DONE = 0, TRAP = 1, SPAWN = 2, JOIN = 3, WAIT = 4, NOTIFY = 5, INSTANTIATE = 6, TIERUP = 7, JIT_INVOKE = 8,
  INSTANTIATE_DETACHED = 9;

// §22 codegen arg/result marshalling by scalar type code (0=i32, 1=i64, 2=f32, 3=f64): the engine
// carries every arg/result as a raw i64 slot; the Worker converts to/from the JS value the emitted
// wasm function uses (an integer's value, a float's bits reinterpreted). A tiny scratch DataView does
// the float bit-casts.
const _sdv = new DataView(new ArrayBuffer(8));
const jitArg = (slot, tc) => tc === 0 ? Number(BigInt.asIntN(32, slot)) // i32 → Number
  : tc === 1 ? slot // i64 → BigInt
  : tc === 2 ? (_sdv.setInt32(0, Number(BigInt.asIntN(32, slot)), true), _sdv.getFloat32(0, true)) // f32
  : (_sdv.setBigInt64(0, slot, true), _sdv.getFloat64(0, true)); // f64
const jitRes = (ret, tc) => tc === 0 ? BigInt(ret) // i32 value
  : tc === 1 ? ret // i64
  : tc === 2 ? (_sdv.setFloat32(0, ret, true), BigInt(_sdv.getUint32(0, true))) // f32 bits (zero-ext)
  : (_sdv.setFloat64(0, ret, true), _sdv.getBigInt64(0, true)); // f64 bits

self.onmessage = async (e) => {
  const { module, memory, prog, win, winSize, role, func, sp, arg, slot, stackTop, tlsBase,
    smod, entry, slog, fuel, tierup, gptr, glen, tierupCell, jitCodegen, jitService, instCodegen,
    jitB2, jitRuntime, tierupPaged, childMem } = e.data;
  // I22 liveness backstop. The `temen_par_run` loop below already catches host traps, but the SETUP +
  // codegen calls before it (WebAssembly.instantiate, temen_par_enable_jit / _jit_codegen /
  // _inst_codegen, temen_par_child*) are the ones a rare shared-memory race actually trips (a double-free
  // in the shared codegen stash → `memory access out of bounds` or a panic=abort `unreachable`). An
  // uncaught trap there rejects this async onmessage, and a Worker's unhandled rejection does NOT fire
  // `Worker.onerror` on the page — so a child that dies here never fills its completion slot and the
  // root's join `Atomics.wait` hangs the whole page (the 30s-timeout flake). Wrap the entire body so
  // ANY trap becomes a clean vCPU trap: wake any joiner, and report `fail` with the captured panic site.
  let ex;
  try {
  // The engine imports `temen_host.webgpu_op` (the `webgpu` capability's host seam). A Worker vCPU has
  // no GPU surface (the playground's GPU reactor runs on the main thread via par.js), so stub it to a
  // no-op — a guest that resolves the `webgpu` cap here gets -1 and skips. Without it the instantiate
  // fails with "Import temen_host: module is not an object or function".
  // `stdout_chunk` (the live-stdout tee) is likewise stubbed — a Worker vCPU streams no card output.
  ({ exports: ex } = await WebAssembly.instantiate(module, { env: { memory }, temen_host: { ...foreignImports(memory), webgpu_op: () => -1n, stdout_chunk: () => {} } }));
  ex.__stack_pointer.value = stackTop; // this Worker's private stack...
  if (ex.__tls_size.value > 0) ex.__wasm_init_tls(tlsBase); // ...and TLS block (per 4b)
  // Views over the shared memory, refreshed when stale: the shared WebAssembly.Memory can GROW
  // mid-run (any Worker's in-wasm allocation — e.g. a §14 module compile+push), and views created
  // before a growth don't cover the new region (an Atomics access past the old length throws).
  let i32v = new Int32Array(memory.buffer), i64v = new BigInt64Array(memory.buffer);
  const i32 = () =>
    i32v.byteLength === memory.buffer.byteLength ? i32v : (i32v = new Int32Array(memory.buffer));
  const i64 = () =>
    i64v.byteLength === memory.buffer.byteLength ? i64v : (i64v = new BigInt64Array(memory.buffer));
  const tlsSize = ex.__tls_size.value, tlsAlign = ex.__tls_align.value || 1;
  // A §5 'detached' child (#1286) runs over its OWN shared `WebAssembly.Memory` (`childMem`, minted +
  // seeded by the spawning Worker below), reached by the engine through `Region::Foreign` — so its
  // futex words live there, one host header page in, not in the engine memory at `win`. The completion
  // slot protocol stays in the engine memory (`i32()`); only WAIT/NOTIFY switch to these views.
  const fmem = childMem || memory, fbase = childMem ? ex.temen_detached_header_bytes() : win;
  let fi32v = new Int32Array(fmem.buffer);
  const fi32 = () => fi32v.byteLength === fmem.buffer.byteLength ? fi32v : (fi32v = new Int32Array(fmem.buffer));

  // A §14 'confined' child's `win`/`winSize` are already its carve (the parent's window + the event's
  // offset) — a confined child is just a child with a shifted, smaller window (DESIGN.md §14).
  // wasm-JIT tier-up (threads slice): this Worker enables the tier-up bitmap in this instance —
  // `temen_par_enable_jit` emits the tier-up module (a pure leaf reachable only via `thread.spawn`
  // still emits, since the guest keeps interpreting), stashes its bytes + the decoded module (so a
  // cross-tier leaf's `call_interp` works), and reports whether anything tier-ups. This Worker then
  // instantiates the emitted module against the ONE shared memory (each Worker instantiates its own —
  // wasm tables aren't shareable across Workers). On PAR_TIERUP it calls `f{func}` here.
  let emitted = null, envCell = 0;
  // #750: `tierupPaged` opts the run into the paged tier — unmap/protect guests keep their pure
  // leaves eligible; each TIERUP then also carries a page-state table (see the handler below).
  const enableJit = tierupPaged ? ex.temen_par_enable_jit_paged : ex.temen_par_enable_jit;
  if (tierup && enableJit(gptr, glen) === 1) {
    const wptr = Number(ex.temen_wasmjit_ptr()), wlen = ex.temen_wasmjit_len();
    const bytes = new Uint8Array(memory.buffer).slice(wptr, wptr + wlen);
    const emod = await WebAssembly.instantiate(await WebAssembly.compile(bytes), {
      env: {
        memory,
        trap: () => {}, // an TEMEN-specific fault; the following `unreachable` throws, caught below
        call_interp: (f, argsPtr) => { if (ex.temen_wasmjit_call_interp(f, argsPtr) !== 0) throw new Error('cross-tier trap'); },
      },
    });
    emitted = emod.exports;
    envCell = Number(ex.temen_par_alloc(ex.temen_wasmjit_env_bytes())); // fuel counter + cross-tier scratch
  }

  // §22 guest-JIT real codegen (BROWSER.md slice 5): the run's single §22 unit was emitted + stashed
  // once at powerbox setup (temen_par_powerbox_jit_codegen); every Worker instantiates its own instance
  // against the ONE shared memory. On PAR_JIT_INVOKE this Worker runs the emitted `f0(win, env, args)`
  // instead of the interpreter. A `new WebAssembly.Module`/`Instance` here is synchronous (the unit is
  // small) so it needs no await inside the event loop.
  let jitUnit = null, jitEnvCell = 0;
  if (jitCodegen) ex.temen_par_jit_codegen_service(jitService | 0); // 0=i32, 1=f64 service (per-instance)
  if (jitCodegen && ex.temen_par_enable_jit_codegen() === 1 && ex.temen_par_jit_unit_wasm_len() > 0) {
    const wptr = Number(ex.temen_par_jit_unit_wasm_ptr()), wlen = ex.temen_par_jit_unit_wasm_len();
    const bytes = new Uint8Array(memory.buffer).slice(wptr, wptr + wlen);
    const umod = new WebAssembly.Module(bytes);
    const uinst = new WebAssembly.Instance(umod, {
      env: {
        memory,
        trap: () => {},
        call_interp: (f, argsPtr) => { if (ex.temen_wasmjit_call_interp(f, argsPtr) !== 0) throw new Error('cross-tier trap'); },
      },
    });
    jitUnit = uinst.exports;
    jitEnvCell = Number(ex.temen_par_alloc(ex.temen_wasmjit_env_bytes()));
  }

  // §22 Model B2 cross-Worker (BROWSER.md § "wasm-JIT tier"): a runtime-`Jit.compile`d unit's
  // `call_indirect` must reach units another Worker `install`ed. wasm funcrefs can't cross Workers,
  // so this Worker holds its OWN funcref table mirroring the shared interpreter `Domain`'s slot→unit
  // map, and instantiates each installed unit locally (the emitted units import this table — the Rust
  // emitter runs in B2 mode, `temen_par_jit_set_b2`). Enabled by `jitB2` (the page sets both).
  //   Verified end-to-end by the CI-gated `jitb2` work item (main.js item 12): 8 Workers each
  //   runtime-compile + `install` a unit into the shared table (raced slots) and dispatch it on
  //   B2-emitted wasm through this mirror, interp ≡ B2 codegen ≡ 56. The emitter-level cross-instance
  //   semantics are pinned native by `crates/temen-wasm-jit/tests/b2_install.rs`.
  let jitTable = null;
  const jitInstCache = new Map(); // code handle → instance.exports (per-Worker instantiation)
  if (jitB2) {
    const size = 1 << ex.temen_par_jit_table_log2();
    jitTable = new WebAssembly.Table({ initial: size, maximum: size, element: 'anyfunc' });
  }
  if ((jitB2 || jitRuntime) && !jitEnvCell) jitEnvCell = Number(ex.temen_par_alloc(ex.temen_wasmjit_env_bytes()));
  // Instantiate a unit's emitted bytes importing this Worker's shared table, or null if not emitted.
  const jitInstantiate = (bytes) =>
    new WebAssembly.Instance(new WebAssembly.Module(bytes), {
      env: {
        memory,
        trap: () => {},
        call_interp: (f, a) => { if (ex.temen_wasmjit_call_interp(f, a) !== 0) throw new Error('cross-tier trap'); },
        __indirect_function_table: jitTable,
      },
    }).exports;
  // Get-or-instantiate the unit for a code handle (cached per Worker); null if it has no emitted wasm.
  const jitUnitFor = (code) => {
    let inst = jitInstCache.get(code);
    if (inst) return inst;
    const len = ex.temen_par_jit_code_wasm_by_handle_len(code);
    if (len === 0) return null;
    const ptr = Number(ex.temen_par_jit_code_wasm_by_handle_ptr(code));
    inst = jitInstantiate(new Uint8Array(memory.buffer).slice(ptr, ptr + len));
    jitInstCache.set(code, inst);
    return inst;
  };
  // Mirror the shared `Domain` slot→unit map into this Worker's table: `f0` of the installed unit, or
  // null for an empty/uninstalled slot (so a stale `call_indirect` traps). Called before each invoke.
  const jitSyncTable = () => {
    const size = 1 << ex.temen_par_jit_table_log2();
    for (let slot = 0; slot < size; slot++) {
      const code = ex.temen_par_jit_slot_code(slot);
      if (code < 0) { jitTable.set(slot, null); continue; }
      const inst = jitUnitFor(code);
      jitTable.set(slot, inst ? inst['f0'] : null);
    }
  };

  // §14 instantiate real codegen (BROWSER.md slice 5 + VM-in-VM): a confined child whose granted-unit
  // entry is eligible runs it on EMITTED WASM here and fills the completion slot the parent joins — no
  // vCPU. The unit's data segments were materialized into the carve by the parent before this event,
  // so `f{entry}(win=carveBase, env, …cap-handle args a pure unit ignores)` reads them. With the
  // nested emit (`compile_module_nested`) a cap-using entry is ALSO eligible: its `call.cap 6 0/1`
  // (instantiate/join) arrives here as the `env.instantiate`/`env.join` imports, serviced through the
  // SAME confined-child completion-slot protocol as the interpreter's INSTANTIATE/JOIN arms below —
  // the grandchild spawns on its own Worker (page relay), and `env.join` blocks on its slot with
  // `Atomics.wait` (legal in a Worker). A non-nested (2-import) unit simply ignores the extra keys.
  if (role === 'confined' && instCodegen && ex.temen_par_enable_inst_codegen() === 1
      && ex.temen_par_inst_eligible(entry) === 1) {
    const wptr = Number(ex.temen_par_inst_unit_wasm_ptr()), wlen = ex.temen_par_inst_unit_wasm_len();
    const bytes = new Uint8Array(memory.buffer).slice(wptr, wptr + wlen);
    // #1151 Slice 2c: the child vCPU is built anyway — never `temen_par_run`, it services every
    // `env.call_interp` leaf over the carve with the child's OWN powerbox (`temen_par_inst_call_interp`
    // → `bounce_call`), so a leaf may store / `map` / `unmap` / `protect` exactly as the interpreter
    // path would run it. On a paged unit (the module reaches a page op) the emitted accesses consult a
    // page-state table re-synced from this vCPU after each bounce. Its starter cap handles (the entry
    // args) come from the argv stash — no longer inert zeros.
    const cv = ex.temen_par_child_confined(prog, win, slog, smod, entry, BigInt(fuel));
    if (cv === 0) {
      Atomics.store(i32(), slot >> 2, 2); Atomics.notify(i32(), slot >> 2);
      self.postMessage({ kind: 'fail', why: 'confined child vcpu build failed (codegen path)' });
      return;
    }
    const paged = ex.temen_par_inst_paged() === 1;
    let uexports = null;
    const syncPaged = () => {
      uexports.mapped.value = ex.temen_par_ev_b(cv);
      uexports.pagestate.value = Number(ex.temen_par_tierup_pagestate_ptr(cv));
    };
    const childSlots = []; // env.instantiate handle (index) → grandchild completion slot ptr
    const threadSlots = []; // env.thread_spawn handle (index) → thread completion slot ptr
    const uinst = new WebAssembly.Instance(new WebAssembly.Module(bytes), {
      env: {
        memory,
        trap: () => {},
        call_interp: (f, a) => {
          if (ex.temen_par_inst_call_interp(cv, f, a) !== 0) throw new Error('cross-tier trap');
          if (paged) syncPaged();
        },
        // §14 VM-in-VM spawn bounce. The emitted parent does no confinement itself, so the engine's
        // `event_instantiate` carve checks are replicated here: the grandchild's power-of-two carve
        // must be aligned and lie inside THIS child's own window (confinement composes); a violation
        // throws → this child's slot reads trapped, exactly as the interpreter traps the parent.
        // The `inst` handle arg is inert (0n) on the emitted tier — authority is this child's §14
        // construction itself (every confined child holds an attenuated Instantiator), mirroring the
        // native harness (crates/temen-wasm-jit/tests/nested_vm.rs).
        instantiate: (cwin, _inst, centry, off, cslog, quota) => {
          const gsize = 1 << Number(cslog), goff = Number(off);
          // #1206: a carve may not dip into this child's own NULL guard (`[0, 16 KiB)`, seeded on any
          // window of at least the guard's size — the engine's `carve_fits` rule, `POWERBOX_NULL_GUARD`).
          const guard = winSize >= 16384 ? 16384 : 0;
          if (gsize > winSize || (goff & (gsize - 1)) !== 0 || goff + gsize > winSize || goff < guard)
            throw new Error('bad nested carve');
          const gslot = ex.temen_par_alloc(SLOT);
          const gstackTop = ex.temen_par_alloc(STACK) + STACK;
          const gtlsBase = tlsSize > 0 ? roundUp(ex.temen_par_alloc(tlsSize + tlsAlign), tlsAlign) : 0;
          // Fuel: min(quota, parent's) — the emitted tier tracks fuel coarsely (the env-cell
          // counter), so "parent's" is this child's own granted fuel from its init cfg.
          const pf = BigInt(fuel);
          const gfuel = quota > 0n && quota < pf ? quota : pf;
          self.postMessage({
            kind: 'spawn', role: 'confined', smod, entry: Number(centry), slog: Number(cslog),
            fuel: gfuel.toString(), win: cwin + goff, winSize: gsize,
            slot: gslot, stackTop: gstackTop, tlsBase: gtlsBase,
          });
          const h = childSlots.length;
          childSlots.push(gslot);
          return h;
        },
        // §14 VM-in-VM join: block on the grandchild's completion slot — the same wait the
        // interpreter JOIN arm does — and surface its result (or trap) to the emitted parent.
        join: (_inst, child) => {
          const gslot = childSlots[child];
          if (gslot === undefined) throw new Error('join of unknown child');
          Atomics.wait(i32(), gslot >> 2, 0);
          if (Atomics.load(i32(), gslot >> 2) === 2) throw new Error('nested child trapped');
          return i64()[(gslot + 8) >> 3];
        },
        // §11 slice 3 — thread/futex ops from an EMITTED unit, serviced through the same spawn
        // relay + completion-slot protocol as the interpreter's SPAWN/JOIN arms. The spawned vCPU
        // runs the granted unit's own `func` (smod — this Worker knows its module), over THIS
        // child's window (a thread shares its spawner's window = the carve).
        thread_spawn: (func, sp, arg) => {
          const tslot = ex.temen_par_alloc(SLOT);
          const tstackTop = ex.temen_par_alloc(STACK) + STACK;
          const ttlsBase = tlsSize > 0 ? roundUp(ex.temen_par_alloc(tlsSize + tlsAlign), tlsAlign) : 0;
          self.postMessage({
            kind: 'spawn', smod, func, sp: sp.toString(), arg: arg.toString(),
            win, winSize, fuel,
            slot: tslot, stackTop: tstackTop, tlsBase: ttlsBase,
          });
          const h = threadSlots.length;
          threadSlots.push(tslot);
          return h;
        },
        thread_join: (h) => {
          const tslot = threadSlots[h];
          if (tslot === undefined) throw new Error('join of unknown thread');
          Atomics.wait(i32(), tslot >> 2, 0);
          if (Atomics.load(i32(), tslot >> 2) === 2) throw new Error('unit thread trapped');
          return i64()[(tslot + 8) >> 3];
        },
        // Futex over the shared window (addr confined by the window mask, as the engine does).
        mem_wait: (cwin, addr, expected, timeout, is64) => {
          const a = cwin + (Number(addr) & (winSize - 1));
          const ms = timeout <= 0n ? Infinity : Number(timeout) / 1e6;
          const r = is64
            ? Atomics.wait(i64(), a >> 3, expected, ms)
            : Atomics.wait(i32(), a >> 2, Number(BigInt.asIntN(32, expected)), ms);
          return r === 'ok' ? 0 : r === 'not-equal' ? 1 : 2;
        },
        mem_notify: (cwin, addr, count) => {
          const a = cwin + (Number(addr) & (winSize - 1));
          return Atomics.notify(i32(), a >> 2, count >>> 0);
        },
      },
    });
    const envCell = Number(ex.temen_par_alloc(ex.temen_wasmjit_env_bytes()));
    new DataView(memory.buffer).setBigInt64(envCell, 1n << 61n, true); // ample fuel
    // #1123 slice 2/3 — per-event window routing: set the emitted child's live `mapped` global to ITS
    // carve size (`winSize = 1 << slog`), not its declared memory. The emitted trap check reads this
    // global live (#717), so this both confines the child to the carve (an access past `winSize` faults,
    // fail-closed, regardless of how the carve compares to `1 << declared`) and lets a `carve > declared`
    // child use its whole carve — heap growth into `[1 << declared, winSize)` (a malloc child, e.g. a nim
    // phase). Matches the interpreter confined child (`mapped == carve`) and the headless wasmi servicer
    // (crates/temen-wasm-jit/tests/nested_emitted_child.rs). `mapped` is exported by every emitted module.
    uexports = uinst.exports;
    if (paged) {
      // The page-state table + its coverage (the value for `mapped`), seeded from the child's live map.
      ex.temen_par_inst_pagestate_sync(cv);
      syncPaged();
    } else {
      uinst.exports.mapped.value = BigInt(winSize);
    }
    // The entry args: the child's starter cap handles, staged by `temen_par_child_confined`.
    const nargs = Number(ex.temen_par_tierup_argv_len(cv)), aptr = Number(ex.temen_par_tierup_argv_ptr(cv));
    const args = [];
    for (let i = 0; i < nargs; i++) args.push(i64()[(aptr >> 3) + i]);
    if (tierupCell) Atomics.add(i32(), tierupCell >> 2, 1); // count emitted children (non-vacuity)
    try {
      const ret = uinst.exports['f' + entry](win, envCell, ...args);
      i64()[(slot + 8) >> 3] = BigInt(ret); // publish result...
      Atomics.store(i32(), slot >> 2, 1); // ...set done flag...
      Atomics.notify(i32(), slot >> 2); // ...and wake the joiner
    } catch {
      Atomics.store(i32(), slot >> 2, 2); // 2 = trapped (the joiner traps on deliver_join)
      Atomics.notify(i32(), slot >> 2);
    }
    ex.temen_par_free(cv);
    return;
  }

  const v = role === 'root'
    ? ex.temen_par_root(prog, win, winSize, func)
    : role === 'confined'
      ? ex.temen_par_child_confined(prog, win, slog, smod, entry, BigInt(fuel))
      : role === 'detached'
        ? ex.temen_par_child_detached(prog, registerForeign(childMem, fbase), slog, smod, entry, BigInt(fuel))
        : ex.temen_par_child(prog, win, winSize, smod | 0, func, BigInt(sp), BigInt(arg));
  if (v === 0) { self.postMessage({ kind: 'fail', why: 'vcpu build failed' }); return; }

  const handles = []; // local spawn handle (index) → child completion slot ptr

  for (;;) {
    // I22 hang site. A host wasm trap escaping `temen_par_run` — `memory access out of bounds`, or
    // `unreachable` from a panic=abort engine panic — unwinds into this async `onmessage`, rejecting
    // it. A Worker's unhandled rejection does NOT fire `Worker.onerror` on the page, so par.js's
    // promise would never settle: the vCPU's DOM item would sit `pending` until the harness's 30s
    // `waitForFunction` times out (the silent-flake signature). Convert it into a structured failure —
    // wake any joiner (a non-root vCPU's completion slot) so a parent's `Atomics.wait` doesn't
    // cascade-hang, then report `fail` with the trap text so the page/harness self-identifies.
    let evc;
    try {
      evc = ex.temen_par_run(v);
    } catch (err) {
      if (role !== 'root') {
        const iv = new Int32Array(memory.buffer);
        Atomics.store(iv, slot >> 2, 2); // 2 = trapped
        Atomics.notify(iv, slot >> 2);
      }
      let why = `vcpu ${role} host trap: ${err && err.message ? err.message : err}`;
      // If the trap was a panic=abort engine panic (surfaces as `unreachable`), the Rust panic hook
      // stashed FILE:LINE + message; the trap left memory intact, so read it back here (I22 (a)).
      try {
        const plen = ex.temen_par_last_panic_len ? ex.temen_par_last_panic_len() : 0;
        if (plen > 0) {
          const p = Number(ex.temen_par_last_panic_ptr());
          why += ` | panic: ${new TextDecoder().decode(new Uint8Array(memory.buffer).slice(p, p + plen))}`;
        }
      } catch { /* accessor absent (older build) or read failed — the trap text alone still ships */ }
      self.postMessage({ kind: 'fail', why });
      return; // don't temen_par_free(v): the instance just trapped; the page terminates this Worker
    }
    if (evc === DONE) {
      const value = ex.temen_par_ev_a(v); // i64 → BigInt
      i64()[(slot + 8) >> 3] = value; // publish result...
      Atomics.store(i32(), slot >> 2, 1); // ...set done flag...
      Atomics.notify(i32(), slot >> 2); // ...and wake a joiner
      if (role === 'root') self.postMessage({ kind: 'done', value: value.toString() });
      ex.temen_par_free(v);
      return;
    }
    if (evc === TRAP) {
      Atomics.store(i32(), slot >> 2, 2); // 2 = trapped
      Atomics.notify(i32(), slot >> 2);
      if (role === 'root') self.postMessage({ kind: 'trap' });
      ex.temen_par_free(v);
      return;
    }
    if (evc === SPAWN) {
      // ev_a packs (spawning frame's module << 32) | func, as the INSTANTIATE event does — the
      // child resolves `func` in that module (an installed §22 unit spawns its own functions).
      const cam = ex.temen_par_ev_a(v);
      const csmod = Number(cam >> 32n), cfunc = Number(BigInt.asUintN(32, cam));
      const csp = ex.temen_par_ev_b(v), carg = ex.temen_par_ev_c(v);
      // Allocate the child's completion slot + stack + TLS, then ask the page to start its Worker.
      const cslot = ex.temen_par_alloc(SLOT);
      const cstackTop = ex.temen_par_alloc(STACK) + STACK;
      const ctlsBase = tlsSize > 0 ? roundUp(ex.temen_par_alloc(tlsSize + tlsAlign), tlsAlign) : 0;
      self.postMessage({
        kind: 'spawn', smod: csmod, func: cfunc, sp: csp.toString(), arg: carg.toString(),
        win, winSize,
        slot: cslot, stackTop: cstackTop, tlsBase: ctlsBase,
      });
      const handle = handles.length;
      handles.push(cslot);
      ex.temen_par_deliver_handle(v, handle);
      continue;
    }
    if (evc === JOIN) {
      const cslot = handles[Number(ex.temen_par_ev_a(v))];
      if (cslot === undefined) { ex.temen_par_deliver_join(v, 0n, 1); continue; } // bad handle → trap, never wait(0)
      Atomics.wait(i32(), cslot >> 2, 0); // block until the child sets its done flag
      const trapped = Atomics.load(i32(), cslot >> 2) === 2;
      ex.temen_par_deliver_join(v, i64()[(cslot + 8) >> 3], trapped ? 1 : 0);
      continue;
    }
    if (evc === INSTANTIATE) {
      // §14 confined executor child (THREADS.md 4c-domain §14-D2): the engine already validated the
      // carve + built everything authority-bearing; the operands are inert integers we shuttle into
      // a new Worker (whose window IS the carve), joined via the same completion-slot protocol.
      const am = ex.temen_par_ev_a(v); // (module << 32) | entry
      const csmod = Number(am >> 32n), centry = Number(BigInt.asUintN(32, am));
      const carve = Number(ex.temen_par_ev_b(v)), cslog = Number(ex.temen_par_ev_c(v));
      const cfuel = ex.temen_par_ev_d(v); // i64 → BigInt, shuttled verbatim
      const cslot = ex.temen_par_alloc(SLOT);
      const cstackTop = ex.temen_par_alloc(STACK) + STACK;
      const ctlsBase = tlsSize > 0 ? roundUp(ex.temen_par_alloc(tlsSize + tlsAlign), tlsAlign) : 0;
      self.postMessage({
        kind: 'spawn', role: 'confined', smod: csmod, entry: centry, slog: cslog,
        fuel: cfuel.toString(), win: win + carve, winSize: 1 << cslog,
        slot: cslot, stackTop: cstackTop, tlsBase: ctlsBase,
      });
      const handle = handles.length;
      handles.push(cslot);
      ex.temen_par_deliver_handle(v, handle);
      continue;
    }
    if (evc === INSTANTIATE_DETACHED) {
      // §5 detached child (#1286 slice 3b): the engine already resolved the Instantiator, took the
      // WindowMinter's quota and compiled the module; what is left is the window itself. Mint the child
      // a fresh shared Memory — one host header page (env cell, page-state, DETACHED_JIT.md §3.1) +
      // the declared window, growable in place to the detached ceiling — seed it from the event's
      // segment blob (data segments + args payload; `{off:u64, len:u32, bytes}` records after a u32
      // count) at `header + off`, and post it to a new Worker. A shared Memory posts by reference, so
      // the child Worker and this one see the same bytes; the engine here never addresses them again.
      const am = ex.temen_par_ev_a(v); // (module << 32) | entry
      const csmod = Number(am >> 32n), centry = Number(BigInt.asUintN(32, am));
      const cslog = Number(ex.temen_par_ev_b(v)), cfuel = ex.temen_par_ev_c(v);
      const hdr = ex.temen_detached_header_bytes(), PAGE = 65536;
      const cmem = new WebAssembly.Memory({
        initial: Math.ceil((hdr + 2 ** cslog) / PAGE),
        maximum: Math.ceil((hdr + Number(ex.temen_detached_max_bytes())) / PAGE),
        shared: true,
      });
      const cu8 = new Uint8Array(cmem.buffer);
      const sptr = Number(ex.temen_par_det_seed_ptr(v)), slen = Number(ex.temen_par_det_seed_len(v));
      const sdv = new DataView(memory.buffer, sptr, slen), su8 = new Uint8Array(memory.buffer, sptr, slen);
      let p = 4;
      for (let n = sdv.getUint32(0, true); n > 0; n--) {
        const off = Number(sdv.getBigUint64(p, true)), len = sdv.getUint32(p + 8, true);
        cu8.set(su8.subarray(p + 12, p + 12 + len), hdr + off);
        p += 12 + len;
      }
      const cslot = ex.temen_par_alloc(SLOT);
      const cstackTop = ex.temen_par_alloc(STACK) + STACK;
      const ctlsBase = tlsSize > 0 ? roundUp(ex.temen_par_alloc(tlsSize + tlsAlign), tlsAlign) : 0;
      // No `win`/`winSize`: the child has no window in the engine memory. No tier-up: the emitted tier
      // binds the engine memory (a separate-module child keeps the bitmap inert regardless).
      self.postMessage({
        kind: 'spawn', role: 'detached', childMem: cmem, smod: csmod, entry: centry, slog: cslog,
        fuel: cfuel.toString(), win: 0, winSize: 2 ** cslog, tierup: false,
        slot: cslot, stackTop: cstackTop, tlsBase: ctlsBase,
      });
      const handle = handles.length;
      handles.push(cslot);
      ex.temen_par_deliver_handle(v, handle);
      continue;
    }
    if (evc === WAIT) {
      const addr = Number(ex.temen_par_ev_a(v));
      const expected = Number(BigInt.asIntN(32, ex.temen_par_ev_b(v)));
      const timeoutNs = ex.temen_par_ev_d(v);
      const ms = timeoutNs <= 0n ? Infinity : Number(timeoutNs) / 1e6;
      const r = Atomics.wait(fi32(), (fbase + addr) >> 2, expected, ms); // 'ok' | 'not-equal' | 'timed-out'
      ex.temen_par_deliver_code(v, r === 'ok' ? 0 : r === 'not-equal' ? 1 : 2);
      continue;
    }
    if (evc === NOTIFY) {
      const addr = Number(ex.temen_par_ev_a(v)), count = Number(ex.temen_par_ev_b(v));
      ex.temen_par_deliver_code(v, Atomics.notify(fi32(), (fbase + addr) >> 2, count));
      continue;
    }
    if (evc === TIERUP) {
      // Run the emitted `f{func}(win, env, ...i64 args)` over the shared window instead of
      // interpreting. A trap throws (Temen fault → `env.trap` + `unreachable`, or a wasm trap) — we
      // surface it as a vCPU trap. Otherwise marshal the i64 result slots back to the engine.
      const func = Number(ex.temen_par_ev_a(v));
      const argvPtr = Number(ex.temen_par_tierup_argv_ptr(v)), n = Number(ex.temen_par_tierup_argv_len(v));
      const args = [];
      for (let i = 0; i < n; i++) args.push(i64()[(argvPtr >> 3) + i]); // i64 args → BigInt
      // #717 host sync: the event's committed-extent snapshot → the emitted `"mapped"` global, so
      // the emitted bounds check admits exactly what the interpreter would (idempotent over today's
      // fully-mapped par window; load-bearing once the window can `vm_map`-grow). On a #750 paged
      // run, operand b is the page-state table's COVERAGE (the engine computes it with the table).
      emitted.mapped.value = ex.temen_par_ev_b(v);
      // #750 paged runs: point the emitted `"pagestate"` global at the engine-built table — its
      // Rust-heap address is a linear-memory address (one shared memory, zero copies). Empty (and
      // the global absent) on unpaged runs.
      if (Number(ex.temen_par_tierup_pagestate_len(v)) > 0)
        emitted.pagestate.value = Number(ex.temen_par_tierup_pagestate_ptr(v));
      new DataView(memory.buffer).setBigInt64(envCell, 1n << 61n, true); // ample fuel; preempt = write < 0
      if (tierupCell) Atomics.add(i32(), tierupCell >> 2, 1); // count tier-ups (non-vacuity)
      try {
        const ret = emitted['f' + func](win, envCell, ...args);
        const rets = ret === undefined ? [] : Array.isArray(ret) ? ret : [ret];
        const rptr = Number(ex.temen_par_alloc(Math.max(1, rets.length) * 8));
        for (let i = 0; i < rets.length; i++) i64()[(rptr >> 3) + i] = BigInt(rets[i]);
        ex.temen_par_deliver_tierup(v, rptr, rets.length);
      } catch {
        ex.temen_par_deliver_tierup_trap(v);
      }
      continue;
    }
    if (evc === JIT_INVOKE) {
      // §22 guest-JIT real codegen: the guest `Jit.invoke`d a unit — run the emitted unit's
      // `f0(win, env, ...args)` over the shared window instead of the interpreter, then deliver its
      // result slots. Args marshal by declared type (i32 → JS Number, i64 → BigInt) so a unit need not
      // be all-i64; results go back as `BigInt(ret)` (the engine re-tags by result type). A trap
      // throws and surfaces as a vCPU trap (as an interp invoke would).
      const argvPtr = Number(ex.temen_par_jit_argv_ptr(v)), n = Number(ex.temen_par_jit_argv_len(v));
      const ptypes = new Uint8Array(memory.buffer, Number(ex.temen_par_jit_param_types_ptr(v)), n);
      const args = [];
      for (let i = 0; i < n; i++) args.push(jitArg(i64()[(argvPtr >> 3) + i], ptypes[i]));
      new DataView(memory.buffer).setBigInt64(jitEnvCell, 1n << 61n, true); // ample fuel
      if (tierupCell) Atomics.add(i32(), tierupCell >> 2, 1); // count emitted invokes (non-vacuity)
      // Model B2: mirror the shared dispatch table into this Worker's table, then run the *invoked*
      // unit (resolved by its code handle) — whose `call_indirect`s now reach installed units locally.
      // Otherwise the fixed-unit codegen path runs the run's single pre-instantiated `jitUnit`.
      let unit = jitUnit;
      if (jitB2) {
        jitSyncTable();
        unit = jitUnitFor(ex.temen_par_jit_code(v));
      } else if (!unit) {
        // Runtime-`Jit.compile` path without B2: resolve the invoked unit by its code handle (each
        // Worker instantiates + caches per handle; the emitted bytes live on the shared host).
        unit = jitUnitFor(ex.temen_par_jit_code(v));
      }
      if (!unit) { ex.temen_par_deliver_jit_invoke_trap(v); continue; }
      // #717 host sync: the event's committed-extent snapshot → the unit instance's `"mapped"`
      // global (same contract as TIERUP above; an invoke the scalar can't describe never surfaces
      // here — the engine services it on the interpreter instead).
      unit.mapped.value = ex.temen_par_ev_b(v);
      try {
        const ret = unit['f0'](win, jitEnvCell, ...args);
        const rets = ret === undefined ? [] : Array.isArray(ret) ? ret : [ret];
        const rn = Number(ex.temen_par_jit_result_types_len(v));
        const rtypes = new Uint8Array(memory.buffer, Number(ex.temen_par_jit_result_types_ptr(v)), rn);
        const rptr = Number(ex.temen_par_alloc(Math.max(1, rets.length) * 8));
        for (let i = 0; i < rets.length; i++) i64()[(rptr >> 3) + i] = jitRes(rets[i], rtypes[i]);
        ex.temen_par_deliver_jit_invoke(v, rptr, rets.length);
      } catch {
        ex.temen_par_deliver_jit_invoke_trap(v);
      }
      continue;
    }
  }
  } catch (err) {
    // Liveness backstop (see the note at the top): a trap escaped the setup/codegen path above.
    // Wake any joiner so the parent's `Atomics.wait` on our completion slot doesn't cascade-hang,
    // then report a structured failure carrying the Rust panic location the hook stashed.
    try {
      if (role !== 'root' && slot !== undefined) {
        const iv = new Int32Array(memory.buffer);
        Atomics.store(iv, slot >> 2, 2); // 2 = trapped → the parent's deliver_join sees a trap
        Atomics.notify(iv, slot >> 2);
      }
    } catch { /* memory unusable — nothing more we can do */ }
    let why = `vcpu ${role} setup/host trap: ${err && err.message ? err.message : err}`;
    try {
      const plen = ex && ex.temen_par_last_panic_len ? ex.temen_par_last_panic_len() : 0;
      if (plen > 0) {
        const p = Number(ex.temen_par_last_panic_ptr());
        why += ` | panic: ${new TextDecoder().decode(new Uint8Array(memory.buffer).slice(p, p + plen))}`;
      }
    } catch { /* accessor absent or memory unusable — the trap text alone still ships */ }
    self.postMessage({ kind: 'fail', why });
  }
};
