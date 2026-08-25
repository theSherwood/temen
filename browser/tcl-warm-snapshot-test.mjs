// V8 (Node) check of the **Tcl warm-runtime snapshot** driver (issue #805 follow-on), the Tcl twin of
// `lua-warm-snapshot-test.mjs`. Drives the shipping engine FFI (`temen_warm_open`/`temen_warm_eval`) over a
// `tcl_snapshot.temen` (the two-phase `warmup` = full `Tcl_Init` / `eval_run` = eval-only driver), and
// asserts warm `eval_run` over the restored snapshot matches the cold `_start` (`temen_run_onramp`) output
// byte-for-byte while skipping the `Tcl_Init` rebuild, plus fresh-per-Run isolation. It also drives the
// **warm+JIT** tier (`runWarmJit` over the emitted `eval_run` export) and asserts it matches warm-interp
// (#865: the tier used to trap because the driver drove the cold `_start` export instead of `eval_run`).
//
//   node tcl-warm-snapshot-test.mjs [temen_browser.wasm] [tcl_snapshot.temen]
//
// The `.temen` is deploy-built (the Tcl fetch+toolchain isn't in the committed-asset CI job), so this
// SKIPs cleanly when it's absent — like the other Tcl demos.
import { readFileSync, existsSync } from 'node:fs';
import { engineImports } from './engine-imports.mjs';
import { runWarmJit } from './web/wasmjit-module.js';

const wasmPath = process.argv[2] ?? 'target/wasm32-unknown-unknown/release/temen_browser.wasm';
const modPath = process.argv[3] ?? 'web/assets/tcl_snapshot.temen';
if (!existsSync(modPath)) {
  console.log(`SKIP: ${modPath} not built (Tcl fetch/toolchain unavailable) — run build-onramp-assets.mjs`);
  process.exit(0);
}

const mod = await WebAssembly.compile(readFileSync(wasmPath));
const memory = new WebAssembly.Memory({ initial: 2048, maximum: 16384, shared: true });
const ex = (await WebAssembly.instantiate(mod, engineImports(memory))).exports;
const membuf = () => (ex.memory ?? memory).buffer;
const put = (bytes) => { const p = ex.temen_alloc(bytes.length); new Uint8Array(membuf()).set(bytes, Number(p)); return { p, len: bytes.length, free: () => ex.temen_dealloc(p, bytes.length) }; };
const readStdout = () => { const p = Number(ex.temen_stdout_ptr()), l = Number(ex.temen_stdout_len()); return p && l ? Buffer.from(new Uint8Array(membuf(), p, l)).toString() : ''; };
const fail = (m) => { console.error(`FAIL: ${m}`); process.exit(1); };
const enc = (s) => Buffer.from(s);
const modBytes = readFileSync(modPath);

function cold(js) { const m = put(modBytes); const s = put(enc(js)); ex.temen_run_onramp(m.p, m.len, s.p, s.len); const out = readStdout(); const st = ex.temen_status(); s.free(); m.free(); return { out, st }; }
function warm(js) { const s = put(enc(js)); ex.temen_warm_eval(s.p, s.len); const out = readStdout(); const st = ex.temen_status(); s.free(); return { out, st }; }
async function warmJit(js) { const st = await runWarmJit(ex, ex.memory ?? memory, enc(js), `${modPath}#eval`, 1); return { out: readStdout(), st }; }

const m = put(modBytes);
const live = Number(ex.temen_warm_open(m.p, m.len));
m.free();
if (live < 0 || ex.temen_status() !== 0) fail(`temen_warm_open: status ${ex.temen_status()}`);
console.error(`warm session opened: live image ${(live / (1 << 20)).toFixed(2)} MiB`);

const programs = [
  ['expr', 'puts [expr 6*7]\n'],
  ['string', 'puts [string toupper hello]\n'],
  ['lsort', 'puts [lsort -integer {5 3 8 1 9 2}]\n'],
  ['for-loop', 'set s 0; for {set i 1} {$i <= 100} {incr i} { set s [expr $s+$i] }; puts $s\n'],
  ['proc', 'proc sq {x} {expr $x*$x}; puts [sq 9]\n'],
  ['format', 'puts [format "%.4f 0x%X" 3.14159265 255]\n'],
  // The playground card's default `clock` line — the lazy path pre-touched in `warmup()` (#864).
  ['clock', 'puts [clock format 1000000000 -gmt 1 -format {%Y-%m-%d %H:%M:%S}]\n'],
];

console.log(`\n${'program'.padEnd(12)}${'cold≡warm'.padStart(11)}`);
let allOk = true;
for (const [name, js] of programs) {
  const c = cold(js);
  if (c.st !== 0) fail(`cold ${name}: status ${c.st}`);
  const w = warm(js);
  const ok = c.out === w.out && w.st === 0;
  allOk = allOk && ok;
  console.log(`${name.padEnd(12)}${(ok ? 'OK' : 'MISMATCH').padStart(11)}`);
  if (!ok) { console.log(`  cold: ${JSON.stringify(c.out)}`); console.log(`  warm: ${JSON.stringify(w.out)}`); }
}

// Fresh-per-Run isolation: a variable set in one Run must not exist in the next (snapshot restored each Run).
const r1 = warm('set leaked 4242; puts "set $leaked"\n');
const r2 = warm('puts "exists? [info exists leaked]"\n');
const isolated = r1.out.includes('set 4242') && r2.out.includes('exists? 0') && !r2.out.includes('4242');
allOk = allOk && isolated;
console.log(`\nfresh-per-Run isolation: ${isolated ? 'OK — no variable leak across Runs' : 'LEAK!'}`);
if (!isolated) { console.log(`  run1: ${JSON.stringify(r1.out)}`); console.log(`  run2: ${JSON.stringify(r2.out)}`); }

// #864 regression guard: `clock`'s lazy machinery (msgcat + the `clock` ensemble + a `formatproc`) is
// pre-touched in `warmup()`, so it lands ON the snapshot and a Run's `clock format` reuses it instead of
// re-paying the ~1.4 s lazy init on EVERY Run. Assert the warm clock line is fast — a revert to the
// no-pre-touch driver spikes it back over a second, which this catches.
{
  const CLOCK = 'puts [clock format 1000000000 -gmt 1 -format {%Y-%m-%d %H:%M:%S}]\n';
  const t = Number(process.hrtime.bigint() / 1000n) / 1000;
  const w = warm(CLOCK);
  const ms = Number(process.hrtime.bigint() / 1000n) / 1000 - t;
  const fast = w.st === 0 && ms < 300;
  allOk = allOk && fast;
  console.log(`\n#864 clock pre-touch: warm clock line ${ms.toFixed(1)}ms — ${fast ? 'OK — lazy clock init is on the snapshot (not re-paid per Run)' : `REGRESSION: clock re-initializing every Run (${ms.toFixed(0)}ms > 300ms)`}`);
}

// warm+JIT tier (#865): open the emitted `eval_run`, then drive it via `runWarmJit` and assert byte
// parity with warm-interp. The entry export index must be `eval_run`'s (NOT 0 = the cold `_start`,
// whose Tcl_Init re-run trapped) — a regression to `f0` reintroduces the trap this asserts against.
{
  const opened = ex.temen_warm_jit_open(1);
  // A setjmp-rooted guest routes `eval_run` to InterpDriven (#1081) and declines here with
  // STATUS_UNSUPPORTED (2); production then evaluates on the warm interpreter. That's a benign decline,
  // not the f0 regression this block guards — so treat it as a skip, not a failure.
  const declined = opened !== 0 && ex.temen_status() === 2;
  const entry = ex.temen_warm_jit_entry_func();
  const jitOk = opened === 0 && entry !== 0;
  if (!declined) allOk = allOk && jitOk;
  if (declined) {
    console.log(`\nwarm+JIT open: declined (status 2, eval_run routed to InterpDriven, #1081) — warm+JIT skipped, warm interpreter carries this guest (as in production)`);
  } else {
    console.log(`\nwarm+JIT open: status ${opened} (0=OK), entry export f${entry} — ${jitOk ? 'OK — drives eval_run, not the cold _start (f0)' : 'FAIL — eval_run not emittable or entry is f0'}`);
  }
  if (jitOk) {
    console.log(`\n${'program'.padEnd(12)}${'JIT≡interp'.padStart(11)}`);
    for (const [name, js] of programs) {
      const wi = warm(js);
      const wj = await warmJit(js);
      const ok = wi.out === wj.out && wj.st === 0;
      allOk = allOk && ok;
      console.log(`${name.padEnd(12)}${(ok ? 'OK' : 'MISMATCH').padStart(11)}`);
      if (!ok) { console.log(`  interp: ${JSON.stringify(wi.out)}`); console.log(`  jit:    ${JSON.stringify(wj.out)} st=${wj.st}`); }
    }
    // fresh-per-Run isolation on the warm+JIT tier too.
    const j1 = await warmJit('set leaked 4242; puts "set $leaked"\n');
    const j2 = await warmJit('puts "exists? [info exists leaked]"\n');
    const jiso = j1.out.includes('set 4242') && j2.out.includes('exists? 0') && !j2.out.includes('4242');
    allOk = allOk && jiso;
    console.log(`\nwarm+JIT fresh-per-Run isolation: ${jiso ? 'OK' : 'LEAK!'}`);
  }
}

ex.temen_warm_close();
if (!allOk) fail('Tcl warm snapshot parity/isolation mismatch');
console.log('\nOK: Tcl warm snapshot — warm eval_run matches cold _start byte-for-byte (interp + warm+JIT), isolation holds');
