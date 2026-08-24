// Single-vCPU §22 **runtime** `Jit.compile` on the wasm tier (BROWSER.md § "wasm-JIT tier" — the
// dynamic-loop slice). Unlike jitcodegen.mjs (a *fixed* unit host-compiled at setup), here the guest
// builds an IR blob AT RUNTIME, `cap.call 11 0` compiles it — minting a unit AND emitting its wasm in
// the shared host — then `cap.call 11 1` invokes it. With codegen on, the JS host instantiates that
// per-unit emitted wasm (keyed by the returned code handle) and runs `f0(win, env, …args)`; with
// codegen off, the interpreter services the invoke. Both must yield the same result (ground truth
// service(6,7) = 6*7+100 = 142) — the MISCOMPILE-grade differential the tier holds.
//
// Run: node browser/jitcodegen_runtime.mjs
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { engineImports } from './engine-imports.mjs';

const WASM = fileURLToPath(new URL('./target/wasm32-unknown-unknown/release/temen_browser.wasm', import.meta.url));
const PAR_DONE = 0, PAR_TRAP = 1, PAR_JIT_INVOKE = 8;
const STACK = 1 << 16;
const WIN = 1 << 16;
const BLOB_OFF = 0x2000; // where the guest stages its unit blob in its window (clear of its own use)

// The unit the guest compiles at runtime: `service(a, b) = a*b + 100`. Pure i32 compute (`6*7+100=142`).
const UNIT = `
memory 16
func (i32, i32) -> (i32) {
block 0 (va: i32, vb: i32) {
  vm = i32.mul va vb
  v100 = i32.const 100
  vr = i32.add vm v100
  return vr
  }
}
`;

// The guest `(jit) -> (i64)`: compile the blob staged at (BLOB_OFF, len), then invoke it with (6, 7).
const guestSrc = (blobLen) => `
memory 16
func (i32) -> (i64) {
block 0 (vjit: i32) {
  vptr = i64.const ${BLOB_OFF}
  vlen = i64.const ${blobLen}
  vcode = cap.call 11 0 (i64, i64) -> (i64) vjit (vptr, vlen)
  va = i32.const 6
  vb = i32.const 7
  vr = cap.call 11 1 (i64, i32, i32) -> (i32) vjit (vcode, va, vb)
  vr64 = i64.extend_i32_u vr
  return vr64
  }
}
`;

const _sdv = new DataView(new ArrayBuffer(8));
const jitArg = (slot, tc) => tc === 0 ? Number(BigInt.asIntN(32, slot))
  : tc === 1 ? slot
  : tc === 2 ? (_sdv.setInt32(0, Number(BigInt.asIntN(32, slot)), true), _sdv.getFloat32(0, true))
  : (_sdv.setBigInt64(0, slot, true), _sdv.getFloat64(0, true));
const jitRes = (ret, tc) => tc === 0 ? BigInt(ret)
  : tc === 1 ? ret
  : tc === 2 ? (_sdv.setFloat32(0, ret, true), BigInt(_sdv.getUint32(0, true)))
  : (_sdv.setFloat64(0, ret, true), _sdv.getBigInt64(0, true));

async function main() {
  const module = await WebAssembly.compile(readFileSync(WASM));
  const memory = new WebAssembly.Memory({ initial: 2048, maximum: 16384, shared: true });
  const { exports: ex } = await WebAssembly.instantiate(module, engineImports(memory));
  ex.__stack_pointer.value = Number(ex.temen_par_alloc(STACK)) + STACK;
  // Single-vCPU run: TLS init only if the toolchain exported it (newer nightlies drop __wasm_init_tls).
  if (ex.__wasm_init_tls && ex.__tls_size && ex.__tls_size.value > 0) {
    ex.__wasm_init_tls(Number(ex.temen_par_alloc(ex.__tls_size.value + (ex.__tls_align?.value ?? 16))));
  }
  const u8 = () => new Uint8Array(memory.buffer);

  // Parse Temen text → encoded module bytes (a stable copy in a fresh allocation).
  const encode = (src) => {
    const enc = new TextEncoder().encode(src);
    const sptr = Number(ex.temen_par_alloc(enc.length));
    u8().set(enc, sptr);
    if (ex.temen_parse(sptr, enc.length) !== 1) {
      const p = Number(ex.temen_parse_ptr()), l = ex.temen_parse_len();
      throw new Error('parse failed: ' + new TextDecoder().decode(u8().slice(p, p + l)));
    }
    return u8().slice(Number(ex.temen_parse_ptr()), Number(ex.temen_parse_ptr()) + ex.temen_parse_len());
  };

  const unitBytes = encode(UNIT);
  const guestBytes = encode(guestSrc(unitBytes.length));
  const gptr = Number(ex.temen_par_alloc(guestBytes.length));
  u8().set(guestBytes, gptr);

  const unitEnv = {
    env: { memory, trap: () => {}, call_interp: (f, a) => { if (ex.temen_wasmjit_call_interp(f, a) !== 0) throw new Error('cross-tier trap'); } },
  };

  const run = (codegen) => {
    if (ex.temen_par_powerbox_jit_runtime(gptr, guestBytes.length) !== 1) throw new Error('runtime powerbox failed');
    ex.temen_par_jit_set_codegen(codegen ? 1 : 0);
    const prog = ex.temen_par_compile_jit(gptr, guestBytes.length);
    const win = Number(ex.temen_par_alloc(WIN));
    u8().set(unitBytes, win + BLOB_OFF); // stage the unit blob where the guest's `compile` reads it
    const v = ex.temen_par_root(prog, win, WIN, 0);
    const units = new Map(); // code handle → { f, envCell } — one emitted instance per runtime-compiled unit
    let invokes = 0;
    for (;;) {
      const evc = ex.temen_par_run(v);
      if (evc === PAR_DONE) { const r = ex.temen_par_ev_a(v); ex.temen_par_free(v); return { value: r, invokes }; }
      if (evc === PAR_TRAP) { ex.temen_par_free(v); return { trap: true, invokes }; }
      if (evc === PAR_JIT_INVOKE) {
        invokes++;
        const code = Number(ex.temen_par_jit_code(v));
        if (!units.has(code)) {
          const wptr = Number(ex.temen_par_jit_code_wasm_ptr(v)), wlen = Number(ex.temen_par_jit_code_wasm_len(v));
          if (wlen === 0) throw new Error(`no emitted wasm for code ${code}`);
          const inst = new WebAssembly.Instance(new WebAssembly.Module(u8().slice(wptr, wptr + wlen)), unitEnv);
          units.set(code, { f: inst.exports, envCell: Number(ex.temen_par_alloc(ex.temen_wasmjit_env_bytes())) });
        }
        const { f, envCell } = units.get(code);
        const argvPtr = Number(ex.temen_par_jit_argv_ptr(v)), n = Number(ex.temen_par_jit_argv_len(v));
        const ptypes = new Uint8Array(memory.buffer, Number(ex.temen_par_jit_param_types_ptr(v)), n);
        const args = [];
        for (let i = 0; i < n; i++) args.push(jitArg(new BigInt64Array(memory.buffer)[(argvPtr >> 3) + i], ptypes[i]));
        // fuel now lives in the emitted `fuel` global (register-allocatable, see WASM.md);
        // re-arm the per-region budget on the exported global (the old env-cell write was here).
        f.fuel.value = 1n << 61n;
        try {
          const ret = f['f0'](win, envCell, ...args);
          const rets = ret === undefined ? [] : Array.isArray(ret) ? ret : [ret];
          const rn = Number(ex.temen_par_jit_result_types_len(v));
          const rtypes = new Uint8Array(memory.buffer, Number(ex.temen_par_jit_result_types_ptr(v)), rn);
          const rptr = Number(ex.temen_par_alloc(Math.max(1, rets.length) * 8));
          const o64 = new BigInt64Array(memory.buffer);
          for (let i = 0; i < rets.length; i++) o64[(rptr >> 3) + i] = jitRes(rets[i], rtypes[i]);
          ex.temen_par_deliver_jit_invoke(v, rptr, rets.length);
        } catch {
          ex.temen_par_deliver_jit_invoke_trap(v);
        }
        continue;
      }
      throw new Error('unexpected event ' + evc);
    }
  };

  const interp = run(false), codegen = run(true);
  console.log(`interp → ${interp.value} (${interp.invokes} JS invokes) · codegen → ${codegen.value} (${codegen.invokes} JS invokes)`);
  if (interp.value !== 142n) throw new Error(`FAIL: interp expected 142, got ${interp.value}`);
  if (interp.invokes !== 0) throw new Error(`FAIL: interp surfaced ${interp.invokes} JS invokes (should service on the interpreter)`);
  if (codegen.value !== 142n) throw new Error(`FAIL: codegen ${codegen.value} != 142`);
  if (codegen.invokes !== 1) throw new Error(`FAIL: codegen ran ${codegen.invokes} emitted invokes (expected 1)`);
  console.log('OK: §22 runtime Jit.compile → invoke on emitted wasm matches the interpreter (142)');
}

main().catch((e) => { console.error(e); process.exit(1); });
