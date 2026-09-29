// nimony's own driver in a real browser (#958): the wasm export `temen_nim_open` opens a build of a nim
// program the way `scripts/ci/nim-selfhost-lane.sh`'s step 4 builds it natively — `nimony t --isMain
// prog.nim`, which forks and execs nifmake, `/bin/sh`, nifler2, nimsem, hexer and temen-link, and a
// compile-time evaluation's own build — and the cooperative driver (`driveCoopTierupRun`) runs it inside
// Chromium: the process tree on the interpreter tier, each leaf process that cannot park whole on the
// emitted tier (#1896), and temen-link natively. Reports the build's wall-clock and the engine's linear
// memory after it (a wasm memory only grows, so that is its peak). A measurement, not a gate; it fails
// only when the build does not link the module it should.
//
//   NIM_LANE_DIR=<dir> NIM_TREE=<tree> NIM_PROG=prog.nim [NIM_AT=<dir>] [NIM_LIB=<pack>] \
//     [NIM_EXPECT=<module>] [NIM_MAX_PAGES=65536] node bench_nimony.mjs
//
// `NIM_LANE_DIR` holds the lane's toolchain as `src/nimbuild.rs` reads it (`nimony.temen`, …, `sh.ir`,
// `libc.temeno`); `NIM_TREE` is a lane program tree, seeded and built at `NIM_AT` (by default its host
// path). `NIM_LIB` is a library pack (`nimbuild --pack`, from a build at the same `NIM_AT`), seeded
// after the tree so the build compiles only the program's own modules. The page receives them over
// routed URLs.
import { startServer } from './serve.mjs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { existsSync, readFileSync, readdirSync, realpathSync, statSync } from 'node:fs';
import { createHash } from 'node:crypto';

const ROOT = dirname(fileURLToPath(import.meta.url));
const { NIM_LANE_DIR: LANE, NIM_TREE, NIM_PROG, NIM_AT, NIM_LIB, NIM_EXPECT } = process.env;
const WASM = `${ROOT}/target/wasm32-unknown-unknown/release/temen_browser.wasm`;
if (!LANE || !NIM_TREE || !NIM_PROG || !existsSync(WASM)) {
  console.log('SKIP: set NIM_LANE_DIR, NIM_TREE and NIM_PROG, and build the threads wasm');
  process.exit(0);
}
const TOOLS = ['nimony', 'nifmake', 'nimsem', 'nifler2', 'hexer'];
const tree = realpathSync(NIM_TREE);
const dir = NIM_AT || tree;

// The tree at `dir`, without `bin/` (the toolchain) or `nimcache/` (what a build writes), in the
// order `nimbuild` seeds it; then the library pack; then the guest libc where temen-link looks for it.
const files = [];
const walk = (d, prefix) => {
  for (const name of readdirSync(d).sort()) {
    const p = join(d, name);
    if (statSync(p).isDirectory()) walk(p, `${prefix}${name}/`);
    else files.push({ path: `${prefix}${name}`, disk: p });
  }
};
for (const name of readdirSync(tree).sort()) {
  if (name === 'bin' || name === 'nimcache') continue;
  const p = join(tree, name);
  if (statSync(p).isDirectory()) walk(p, `${dir}/${name}/`);
  else files.push({ path: `${dir}/${name}`, disk: p });
}
if (NIM_LIB) {
  // A registry blob: u32 count, then per entry u32 name length, the name, u32 length, the bytes.
  const pack = readFileSync(NIM_LIB);
  let o = 4;
  for (let i = pack.readUInt32LE(0); i > 0; i--) {
    const n = pack.readUInt32LE(o);
    const path = pack.toString('utf8', o + 4, o + 4 + n);
    const len = pack.readUInt32LE(o + 4 + n);
    o += 8 + n;
    files.push({ path, body: pack.subarray(o, o + len) });
    o += len;
  }
}
files.push({ path: '/lib/temen/libc.temeno', disk: join(LANE, 'libc.temeno') });
const manifest = {
  dir,
  prog: NIM_PROG,
  tools: TOOLS.map((t) => ({ name: t, paths: [`${dir}/bin/${t}`, `/bin/${t}`] })),
  files: files.map((f) => f.path),
};

async function loadChromium() {
  for (const s of ['playwright', '/opt/node22/lib/node_modules/playwright/index.js']) {
    try { const m = await import(s); return m.chromium ?? m.default?.chromium; } catch {}
  }
  throw new Error('playwright not found');
}
const chromium = await loadChromium();
const { server, port } = await startServer(ROOT);
const browser = await chromium.launch({ args: process.env.CI ? ['--no-sandbox'] : [] });
const page = await browser.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(String(e)));
page.on('console', (m) => {
  if (m.type() === 'error') errors.push(m.text());
  else if (m.text().startsWith('nimbench: ')) console.log(m.text());
});
const bytes = (disk) => ({ status: 200, contentType: 'application/octet-stream', body: readFileSync(disk) });
const file = (f) => (f.body ? { status: 200, contentType: 'application/octet-stream', body: f.body } : bytes(f.disk));
await page.route((url) => url.pathname.startsWith('/web/nimbench/'), (route) => {
  const rest = new URL(route.request().url()).pathname.slice('/web/nimbench/'.length);
  if (rest === 'manifest.json') return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(manifest) });
  if (rest === 'sh.ir') return route.fulfill(bytes(join(LANE, 'sh.ir')));
  if (rest.startsWith('tool/')) return route.fulfill(bytes(join(LANE, `${rest.slice(5)}.temen`)));
  if (rest.startsWith('file/')) return route.fulfill(file(files[Number(rest.slice(5))]));
  return route.fulfill({ status: 404, body: '' });
});
await page.goto(`http://127.0.0.1:${port}/web/play.html`);

const maxPages = Number(process.env.NIM_MAX_PAGES || 0) || undefined;
const res = await page.evaluate(async ({ maxPages }) => {
  const par = await import('./par.js');
  const { driveCoopTierupRun } = await import('./wasmjit-module.js');
  const eng = await par.loadEngine(null, { maxPages });
  const { ex, memory } = eng;
  const fetchB = async (u) => new Uint8Array(await (await fetch(u)).arrayBuffer());
  const enc = new TextEncoder();
  // A registry blob: u32 count, then per entry u32 name length, the name, u32 length, the bytes.
  const blob = (entries) => {
    const named = entries.map(([n, b]) => [enc.encode(n), b]);
    const out = new Uint8Array(named.reduce((t, [n, b]) => t + 8 + n.length + b.length, 4));
    const dv = new DataView(out.buffer);
    let o = 0;
    dv.setUint32(o, named.length, true); o += 4;
    for (const [n, b] of named) {
      dv.setUint32(o, n.length, true); o += 4; out.set(n, o); o += n.length;
      dv.setUint32(o, b.length, true); o += 4; out.set(b, o); o += b.length;
    }
    return out;
  };
  const put = (u8) => {
    const p = ex.temen_alloc(u8.length);
    new Uint8Array(memory.buffer).set(u8, p);
    return [p, u8.length];
  };
  const m = await (await fetch('./nimbench/manifest.json')).json();
  const sh = await fetchB('./nimbench/sh.ir');
  const [sp, sl] = put(sh);
  if (ex.temen_parse(sp, sl) !== 1) throw new Error('sh.ir does not parse');
  const shMod = new Uint8Array(memory.buffer, ex.temen_parse_ptr(), ex.temen_parse_len()).slice();
  const tools = await Promise.all(m.tools.map(async (t) => [t, await fetchB(`./nimbench/tool/${t.name}`)]));
  const cmds = blob([['/bin/sh', shMod], ...tools.map(([t, b]) => [t.paths.join('\n'), b])]);
  const tree = blob(await Promise.all(m.files.map(async (p, i) => [p, await fetchB(`./nimbench/file/${i}`)])));
  const argv = enc.encode(['bin/nimony', 't', '--isMain', m.prog].map((a) => `${a}\0`).join(''));
  const driver = tools[0][1];
  const args = [put(driver), put(cmds), put(tree), put(argv), put(enc.encode(m.dir))].flat();
  const before = memory.buffer.byteLength;
  const t0 = performance.now();
  if (ex.temen_nim_open(...args) !== 0) throw new Error(`temen_nim_open: status ${ex.temen_status()}`);
  const status = await driveCoopTierupRun(ex, memory);
  const secs = (performance.now() - t0) / 1000;
  // Reported at once, so a failure reading the results back cannot lose the measurement.
  console.log(`nimbench: status ${status} in ${secs.toFixed(1)} s, linear memory ` +
    `${memory.buffer.byteLength / 2 ** 20} MiB (${before / 2 ** 20} MiB before the build)`);
  // A view of shared memory cannot be decoded in place.
  const text = (p, n) => new TextDecoder().decode(new Uint8Array(memory.buffer, p, n).slice());
  const stdout = text(ex.temen_stdout_ptr(), ex.temen_stdout_len());
  const stderr = text(ex.temen_stderr_ptr(), ex.temen_stderr_len());
  // The linked program, `nimcache/<main>.temen/<prog>.temen`, where `<main>` is nimony's stem for the
  // path it was given; read back from the build's memfs.
  const [qp, ql] = put(enc.encode(m.prog));
  const ml = ex.temen_nim_module_suffix(qp, ql); // onto the stdout slot, read above
  const main = text(ex.temen_stdout_ptr(), ml);
  const prog = m.prog.replace(/^.*\//, '').replace(/\.nim$/, '');
  const [pp, pl] = put(enc.encode(`${m.dir}/nimcache/${main}.temen/${prog}.temen`));
  const n = Number(ex.temen_nim_file(pp, pl));
  const built = n >= 0 ? new Uint8Array(memory.buffer, ex.temen_nim_file_ptr(), n).slice() : null;
  const digest = built && [...new Uint8Array(await crypto.subtle.digest('SHA-256', built))]
    .map((b) => b.toString(16).padStart(2, '0')).join('');
  return {
    status, exit: ex.temen_exit_code(), secs, before, after: memory.buffer.byteLength,
    maxPages: eng.maxPages, builtLen: built?.length ?? -1, digest, stdout, stderr,
  };
}, { maxPages });

await browser.close();
await new Promise((r) => server.close(r));
const mib = (b) => (b / 2 ** 20).toFixed(0);
console.log(`nimony t --isMain ${NIM_PROG} in Chromium: status ${res.status} exit ${res.exit} in ${res.secs.toFixed(1)} s; ` +
  `linear memory ${mib(res.before)} → ${mib(res.after)} MiB (ceiling ${mib(res.maxPages * 65536)} MiB)`);
let ok = res.builtLen >= 0 && errors.length === 0;
if (res.builtLen < 0) console.error(`no linked module\n--- stdout ---\n${res.stdout}--- stderr ---\n${res.stderr}`);
if (errors.length) console.error('page errors:', errors);
if (ok && NIM_EXPECT) {
  const want = createHash('sha256').update(readFileSync(NIM_EXPECT)).digest('hex');
  if (want === res.digest) console.log(`✓ it linked ${NIM_EXPECT}, byte for byte`);
  else { console.error(`the module built in Chromium (${res.builtLen} B) is not ${NIM_EXPECT}`); ok = false; }
}
process.exit(ok ? 0 : 1);
