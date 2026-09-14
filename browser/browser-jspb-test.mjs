// **A powerbox defined in JavaScript**, end to end and headless: the engine's `temen_host.js_cap_call`
// import, the `temen_jspb_*` exports (`browser/src/jspb.rs`) and the shipped page module
// `web/powerbox.js` — the same file the playground's "JS powerbox" card runs — driven over the plain
// wasm32 build. `browser/tests/jspb.rs` proves the Rust half natively; this proves the JS half: the
// wasm import really reaches a page function, the pointer/length marshalling is right on this ABI,
// the window accessors read and write the guest's memory, and an unbound import refuses the run.
//
// Usage: node browser-jspb-test.mjs [temen_browser.wasm]
//   (build: cargo build --release --lib --target wasm32-unknown-unknown)
import { readFileSync, existsSync } from 'node:fs';

import { engineImports } from './engine-imports.mjs';
import { definePowerbox } from './web/powerbox.js';

const wasmPath = process.argv[2] ?? 'target/wasm32-unknown-unknown/release/temen_browser.wasm';
if (!existsSync(wasmPath)) {
  console.log(`SKIP: no engine wasm at ${wasmPath} (build it first)`);
  process.exit(0);
}

// The guest the playground card ships: one capability that reads its window, one that writes it, one
// pure scalar. Kept in step with `web/play.js` and `tests/jspb.rs`.
const GUEST = `
; Every capability this guest calls is implemented in JavaScript, bound by name at instantiation.
memory 16
data 16384 "hello from the guest\\n"
data 16448 "the host is javascript\\n"
export 0 func "_start" 0
func () -> (i64) {
block 0 () {
  v0 = i32.const 0
  v1 = i64.const 16384
  v2 = i64.const 21
  v3 = call.sym "js.log" (i64, i64) -> (i64) v0 (v1, v2)
  v4 = i32.const 0
  v5 = i64.const 16448
  v6 = i64.const 23
  v7 = call.sym "js.upper" (i64, i64) -> (i64) v4 (v5, v6)
  v8 = i32.const 0
  v9 = call.sym "js.log" (i64, i64) -> (i64) v8 (v5, v6)
  v10 = i32.const 0
  v11 = call.sym "js.now" () -> (i64) v10 ()
  return v11
  }
}
`;

const mod = await WebAssembly.compile(readFileSync(wasmPath));
const { exports: ex } = await WebAssembly.instantiate(mod, engineImports());
const eng = { ex, memory: ex.memory };
const is64 = ex.temen_abi_is64() === 1;
const N = (x) => (is64 ? BigInt(x) : Number(x));
const dec = new TextDecoder();

// Compile the guest text inside the sandbox, exactly as the playground does.
function parse(src) {
  const b = new TextEncoder().encode(src);
  const p = ex.temen_alloc(N(b.length));
  new Uint8Array(eng.memory.buffer).set(b, Number(p));
  const ok = ex.temen_parse(p, N(b.length));
  ex.temen_dealloc(p, N(b.length));
  const out = new Uint8Array(eng.memory.buffer)
    .slice(Number(ex.temen_parse_ptr()), Number(ex.temen_parse_ptr()) + Number(ex.temen_parse_len()));
  if (ok !== 1) throw new Error(dec.decode(out));
  return out;
}

const guest = parse(GUEST);
const CLOCK = 1700000000000n;
const logged = [];

// The powerbox — the card's three handlers.
const caps = {
  'js.log': (args, mem) => {
    logged.push(mem.str(args[0], args[1]));
    return args[1];
  },
  'js.upper': (args, mem) => (mem.write(args[0], mem.str(args[0], args[1]).toUpperCase()) ? 0 : -14),
  'js.now': () => CLOCK,
};

const run = definePowerbox(eng, caps).run(guest);
const okStatus = run.status === 0;
const okValue = BigInt(run.value) === CLOCK;
const okRead = logged[0] === 'hello from the guest\n';
const okWrite = logged[1] === 'THE HOST IS JAVASCRIPT\n';

// Fail-closed: drop one capability and the run is refused *before* any guest op — the message names
// the missing import, and the remaining capability is never called.
logged.length = 0;
const { 'js.now': _dropped, ...partial } = caps;
const refused = definePowerbox(eng, partial).run(guest);
const okRefused = refused.status !== 0 && refused.error.includes('js.now') && logged.length === 0;

console.log(`status=${run.status} value=${run.value} (expect ${CLOCK})`);
console.log(`  logged: ${JSON.stringify(logged.length ? logged : [okRead, okWrite])}`);
console.log(`  unbound import refused: ${JSON.stringify(refused.error)}`);
const pass = okStatus && okValue && okRead && okWrite && okRefused;
console.log(`\n${pass ? 'PASS' : 'FAIL'}: run ${okStatus ? '✓' : '✗'}, host clock ${okValue ? '✓' : '✗'}, ` +
  `window read ${okRead ? '✓' : '✗'}, window write ${okWrite ? '✓' : '✗'}, fail-closed ${okRefused ? '✓' : '✗'}`);
process.exit(pass ? 0 : 1);
