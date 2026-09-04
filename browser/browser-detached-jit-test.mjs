// Real-browser (V8) differential for **a detached child on the wasm-JIT tier** (#1285, DETACHED_JIT.md
// §3): the guest runs its emitted `_start` in its OWN `WebAssembly.Memory` — not a carve of the engine's —
// reached by the cdylib only through `Region::Foreign` (#1284). It reads the spawn-time argv payload at its
// args base, `self.attest`s, `vm_map`-grows past its declared 64 KiB window (the child memory grows with
// it), stores into and loads back the grown page, and streams `[value, attest]`. The stdout and result must
// be byte-identical to the interpreter oracle (`temen_detached_oracle_run`: the same module over a fresh
// root-sized reservation — which is what a detached window is). Attest must read `1` (tier 1,
// window_exposed = false) on the emitted tier, and the stored value must be found in the CHILD memory.
import { startServer } from './serve.mjs';
import { fileURLToPath } from 'node:url';
import { dirname } from 'node:path';
import { existsSync } from 'node:fs';
const ROOT = dirname(fileURLToPath(import.meta.url));
async function loadChromium() {
  for (const s of ['playwright', '/opt/node22/lib/node_modules/playwright/index.js']) {
    try { const m = await import(s); return m.chromium ?? m.default?.chromium; } catch {}
  }
  throw new Error('playwright not found');
}
if (!existsSync(`${ROOT}/target/wasm32-unknown-unknown/release/temen_browser.wasm`)) {
  console.log('SKIP: threads wasm absent'); process.exit(0);
}
const chromium = await loadChromium();
const { server, port } = await startServer(ROOT);
const browser = await chromium.launch({ args: process.env.CI ? ['--no-sandbox'] : [] });
const page = await browser.newPage();
const errors = [];
const benign = (t) => /Failed to load resource|status of 404/i.test(t);
page.on('pageerror', (e) => errors.push(String(e)));
page.on('console', (m) => { if (m.type() === 'error' && !benign(m.text())) errors.push(m.text()); });
await page.goto(`http://127.0.0.1:${port}/web/play.html`);

// argv payload: one string. The args blob is `{argc u32, envc u32}` then the packed NUL-terminated strings,
// seeded at module_args_base() = 16384 + 128; the guest loads the first 8 bytes of the first string.
const ARGV = 'hello-detached';
const ARGS_BASE = 16384 + 128;
const GUEST = `memory 16
import 0 "write" (i64, i64) -> (i64)
import 1 "vm_map" (i64, i64, i32) -> (i64)
func () -> (i64) {
block 0 () {
  vab = i64.const ${ARGS_BASE + 8}
  vargs = i64.load vab
  vz = i32.const 0
  vat = call.cap 4294967295 4 () -> (i64) vz ()
  voff = i64.const 65536
  vlen = i64.const 16384
  vprot = i32.const 3
  vg = call.import 1 (voff, vlen, vprot)
  vp = i64.const 65600
  i64.store vp vargs
  vld = i64.load vp
  vsl = i64.const 34816
  i64.store vsl vld
  vsl2 = i64.const 34824
  i64.store vsl2 vat
  vn = i64.const 16
  vw = call.import 0 (vsl, vn)
  vsum = i64.add vld vat
  return vsum
  }
}
export 0 func "_start" 0
`;

const res = await page.evaluate(async ({ guestSrc, argv }) => {
  const { loadEngine } = await import('./par.js');
  const { driveDetachedRun, jitCacheStats } = await import('./wasmjit-module.js');
  const { registerForeign } = await import('./foreign-mem.js');
  const eng = await loadEngine();
  const ex = eng.ex, memory = eng.memory;
  const u8 = () => new Uint8Array(memory.buffer);
  const slice = (p, n) => u8().slice(p, p + n);
  const push = (bytes) => { const p = Number(ex.temen_alloc(bytes.length)); u8().set(bytes, p); return p; };
  const parse = (src) => {
    const b = new TextEncoder().encode(src);
    const p = push(b);
    const okp = ex.temen_parse(p, b.length);
    const outp = slice(ex.temen_parse_ptr(), ex.temen_parse_len());
    ex.temen_dealloc(p, b.length);
    if (okp !== 1) throw new Error('parse: ' + new TextDecoder().decode(outp));
    return outp;
  };
  const b64 = (a) => btoa(String.fromCharCode(...a));
  const readStdout = () => slice(Number(ex.temen_stdout_ptr()), ex.temen_stdout_len());
  const guest = parse(guestSrc);
  const args = new TextEncoder().encode(argv + '\0');

  // Interpreter oracle.
  const gp = push(guest), ap = push(args);
  const oracleValue = ex.temen_detached_oracle_run(gp, guest.length, ap, args.length, 0, 0);
  const oracleStatus = ex.temen_status();
  const oracleOut = b64(readStdout());

  // Detached emitted run: a fresh shared Memory for the child (1 page = the header; the cdylib grows it
  // to the declared window at open and on each vm_map bounce), registered with the header as base.
  const header = ex.temen_detached_header_bytes();
  const child = new WebAssembly.Memory({ initial: 1, maximum: 256, shared: true });
  const id = registerForeign(child, header);
  const pagesAfterRegister = child.buffer.byteLength / 65536;
  const compilesBefore = jitCacheStats.compiles;
  const openStatus = ex.temen_detached_jit_run_open(gp, guest.length, id, ap, args.length, 0, 0);
  const pagesAfterOpen = child.buffer.byteLength / 65536;
  let jitStatus = null, jitErr = null, jitOut = '', jitValue = 0n;
  if (openStatus === 0) {
    try {
      jitStatus = await driveDetachedRun(ex, memory, child, 'detached-test');
      jitOut = b64(readStdout());
      jitValue = ex.temen_run_value();
    } catch (e) { jitErr = String(e && e.message || e); }
  }
  ex.temen_dealloc(gp, guest.length); ex.temen_dealloc(ap, args.length);
  const pagesAfterRun = child.buffer.byteLength / 65536;
  // The grown-page store (guest 65600) must be in the CHILD memory at header + 65600, and the engine's
  // memory must not carry the guest's argv word at the guest address (it is not the engine's window).
  const cv = new DataView(child.buffer);
  const storedInChild = cv.getBigInt64(header + 65600, true);
  const argvWord = new DataView(new TextEncoder().encode(argv.slice(0, 8)).buffer).getBigInt64(0, true);
  return {
    oracleStatus, oracleValue: String(oracleValue), oracleOut,
    openStatus, jitStatus, jitErr, jitOut, jitValue: String(jitValue),
    pagesAfterRegister, pagesAfterOpen, pagesAfterRun,
    storedInChild: String(storedInChild), argvWord: String(argvWord),
    compiles: jitCacheStats.compiles - compilesBefore,
  };
}, { guestSrc: GUEST, argv: ARGV });

await browser.close();
await new Promise((r) => server.close(r));
console.log('RESULT', JSON.stringify(res, null, 2));
if (errors.length) console.log('ERRORS', errors.slice(0, 5));
const words = (b) => { const d = Buffer.from(b, 'base64'); return d.length >= 16 ? [d.readBigInt64LE(0), d.readBigInt64LE(8)] : null; };
const ow = words(res.oracleOut), jw = words(res.jitOut);
const ok = errors.length === 0
  && res.openStatus === 0 && !res.jitErr
  && res.oracleStatus === res.jitStatus
  && res.oracleOut === res.jitOut && res.oracleValue === res.jitValue
  && ow !== null && ow[0] === BigInt(res.argvWord) && ow[1] === 1n   // argv landed; attest = tier 1, unexposed
  && res.storedInChild === res.argvWord                               // the emitted store landed in the CHILD memory
  && res.pagesAfterRegister === 1 && res.pagesAfterOpen === 2 && res.pagesAfterRun >= 3 // header → declared → grown
  && res.compiles >= 1;                                                // non-vacuity: the child actually emitted
console.log(`  detached child on wasm-JIT: stdout≡=${res.oracleOut === res.jitOut} value ${res.oracleValue}/${res.jitValue} attest=${jw ? jw[1] : '?'} argv=${jw ? jw[0] === BigInt(res.argvWord) : '?'} childPages ${res.pagesAfterRegister}→${res.pagesAfterOpen}→${res.pagesAfterRun} storedInChild=${res.storedInChild === res.argvWord}${res.jitErr ? ` · ERR ${res.jitErr}` : ''}`);
console.log(ok ? 'PASS — a detached child ran on the emitted tier in its own WebAssembly.Memory, grew it on vm_map, and matched the interpreter oracle byte-for-byte' : 'FAIL');
process.exit(ok ? 0 : 1);
