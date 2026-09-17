// `web/engine-mem.js` — the engine's memory ceiling is the host's, bounded by the build's.
//
// The threads build imports its memory, so the host picks the `maximum`; `--max-memory` bakes a
// maximum into the module's memory *import declaration* and instantiation refuses any supplied memory
// exceeding it. `declaredMaxPages` reads that declaration out of the module bytes so the host can ask
// for what it wants and still run against a build that declares less (an older build, or CI before a
// raised `--max-memory` lands) — clamping instead of throwing `LinkError`.
import { existsSync, readFileSync } from 'node:fs';
import { declaredMaxPages, engineMemory, ENGINE_MAX_PAGES } from './web/engine-mem.js';

let failed = 0;
const check = (name, got, want) => {
  const ok = got === want;
  if (!ok) failed++;
  console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${name}: ${got}${ok ? '' : ` (want ${want})`}`);
};

// A hand-built module header: an imported table with no maximum (so the walk must step over a
// limits-without-max), then the imported shared memory whose declaration is the ceiling.
const leb = (n) => { const o = []; do { let b = n & 0x7f; n >>>= 7; if (n) b |= 0x80; o.push(b); } while (n); return o; };
const str = (s) => [s.length, ...[...s].map((c) => c.charCodeAt(0))];
const moduleDeclaring = (maxPages) => {
  const entries = [
    ...str('env'), ...str('tbl'), 0x01, 0x70, 0x00, ...leb(1),
    ...str('env'), ...str('memory'), 0x02,
    ...(maxPages === undefined ? [0x02, ...leb(2048)] : [0x03, ...leb(2048), ...leb(maxPages)]),
  ];
  const body = [...leb(2), ...entries];
  return new Uint8Array([0, 0x61, 0x73, 0x6d, 1, 0, 0, 0, 2, ...leb(body.length), ...body]);
};

console.log('engine-mem:');
check('a 1 GiB build declares its maximum', declaredMaxPages(moduleDeclaring(16384)), 16384);
check('a 4 GiB build declares its maximum', declaredMaxPages(moduleDeclaring(65536)), 65536);
check('no declared maximum reads as none', declaredMaxPages(moduleDeclaring(undefined)), undefined);
check('a module importing no memory reads as none',
  declaredMaxPages(new Uint8Array([0, 0x61, 0x73, 0x6d, 1, 0, 0, 0])), undefined);
check('the host is clamped to a build that declares less',
  engineMemory(moduleDeclaring(16384)).maxPages, 16384);
check('the host keeps its own policy under a build that declares more',
  engineMemory(moduleDeclaring(65536)).maxPages, ENGINE_MAX_PAGES);
check('an explicit request wins over the default',
  engineMemory(moduleDeclaring(65536), { maxPages: 4096 }).maxPages, 4096);
check('without bytes the host gets what it asked for', engineMemory(null).maxPages, ENGINE_MAX_PAGES);

// The real build, when one is present: whatever it declares, an engine memory must instantiate against
// it — the property every browser test would otherwise discover as a LinkError.
const WASM = 'target/wasm32-unknown-unknown/release/temen_browser.wasm';
if (existsSync(WASM)) {
  const bytes = readFileSync(WASM);
  const declared = declaredMaxPages(bytes);
  const { memory, maxPages } = engineMemory(bytes);
  console.log(`  build declares ${declared} pages; engine takes ${maxPages} (${(maxPages * 65536) / 2 ** 30} GiB)`);
  check('the engine never exceeds the build', maxPages <= declared, true);
  const mod = await WebAssembly.compile(bytes);
  const stub = new Proxy({}, { get: () => new Proxy({}, { get: (_t, k) => (k === 'memory' ? memory : () => 0) }) });
  await WebAssembly.instantiate(mod, stub); // throws LinkError if the clamp is wrong
  console.log('  ok   the real build instantiates over it');
} else {
  console.log('  SKIP: no threads build present');
}

console.log(failed ? 'FAIL' : 'PASS — the ceiling is the host\'s, bounded by the build\'s');
process.exit(failed ? 1 : 0);
