// Browser wasm-JIT tier — §22 **Model B2** domain wiring (BROWSER.md § "wasm-JIT tier"). An
// `install`ed unit becomes a funcref another instance's `call_indirect` reaches, through one shared
// `WebAssembly.Table` the host owns. This is the JS analog of the native differential host in
// `crates/svm-wasm-jit/tests/b2_install.rs`: the interpreter's `DomainTable` is host-populated, so
// here the JS host owns the table and writes each slot on `install`/`uninstall` — the emitted units
// (via `svm_wasmjit_compile_b2`) *import* the table and populate nothing.
//
// Scope: this wires **one instance-domain** (a single Worker / single `WebAssembly.Table`). wasm
// funcrefs are not transferable across Workers, so cross-Worker B2 needs each Worker to hold its own
// `WebAssembly.Table` mirroring the shared `DomainTable`'s slot→unit map (each Worker instantiating an
// installed unit locally and `table.set`ting its own funcref at the shared slot index) — deferred,
// tracked in BROWSER.md. The shared-table dispatch itself is proven native in `b2_install.rs`.

// The cdylib's linear memory: exported by the plain build, or passed in by the threads build.
function memoryOf(ex, memory) {
  return memory ?? ex.memory;
}

// Read the most-recently-emitted wasm bytes out of the cdylib's linear memory (a later `svm_alloc`
// could move the stash, so copy eagerly).
function takeEmitted(ex, mem) {
  const wptr = Number(ex.svm_wasmjit_ptr());
  const wlen = ex.svm_wasmjit_len();
  return new Uint8Array(mem.buffer).slice(wptr, wptr + wlen);
}

// Open a Model-B2 domain over the cdylib `ex`. `tableLog2` must match the `Jit` grant's table
// reservation (the emitted `call_indirect` masks `idx & (2^tableLog2 - 1)`, so the shared table must
// be exactly `2^tableLog2` slots). `shared` selects the emitted `env.memory` import's shared flag
// (`true` for the threads build), matching the memory the units link against.
//
// Returns `{ table, compile(moduleBytes), install(instance, slot), uninstall(slot) }`:
//   * `compile` emits a B2 unit and instantiates it against the cdylib memory + the shared table,
//     returning its `WebAssembly.Instance` (its `f{i}` are the domain's funcrefs);
//   * `install(instance, slot)` writes the unit's `f0` into a reserved slot (the host owning slots);
//   * `uninstall(slot)` nulls the slot back to trapping padding (reusable; a stale call traps).
// A unit's own `call_indirect` then dispatches — at native wasm speed — to whatever is installed.
export function openB2Domain(ex, { tableLog2, shared = true, memory = null } = {}) {
  if (typeof tableLog2 !== 'number') throw new Error('openB2Domain: tableLog2 required');
  const mem = memoryOf(ex, memory);
  const size = 1 << tableLog2;

  // The one shared funcref table (all null → every slot traps until the host writes it). This is the
  // wasm analog of the interpreter's reserved `DomainTable`; the host is its sole populator.
  const table = new WebAssembly.Table({ initial: size, maximum: size, element: 'anyfunc' });

  let lastTrap = 0;
  const imports = () => ({
    env: {
      memory: mem,
      trap: (code) => { lastTrap = code; },
      // A B2 unit reaches other units through the shared table, not the interpreter — but a unit may
      // still have genuine cross-tier interp leaves. Route them as the compute path does.
      call_interp: (func, argsPtr) => {
        if (ex.svm_wasmjit_call_interp(func, argsPtr) !== 0) throw new Error('cross-tier trap');
      },
      __indirect_function_table: table,
    },
  });

  async function compile(moduleBytes) {
    const mptr = ex.svm_alloc(moduleBytes.length);
    new Uint8Array(mem.buffer).set(moduleBytes, Number(mptr));
    const ok = ex.svm_wasmjit_compile_b2(mptr, moduleBytes.length, tableLog2, shared ? 1 : 0);
    ex.svm_dealloc(mptr, moduleBytes.length);
    if (ok !== 1) return null; // outside the emitter subset → caller keeps the interpreter tier
    const wasm = takeEmitted(ex, mem);
    const module = await WebAssembly.compile(wasm);
    return WebAssembly.instantiate(module, imports());
  }

  return {
    table,
    lastTrap: () => lastTrap,
    compile,
    // `install`: the host writes the unit's `f0` funcref into `slot` — the JS analog of
    // `DomainTable::install` / the `table.set` in `b2_install.rs`.
    install(instance, slot) {
      table.set(slot, instance.exports.f0);
    },
    // `uninstall`: null the slot back to trapping padding (DESIGN.md §22 — reusable; stale calls trap).
    uninstall(slot) {
      table.set(slot, null);
    },
  };
}
