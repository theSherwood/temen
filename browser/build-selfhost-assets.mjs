// Build the **chibicc self-host closure image** — the committed-at-CI fs asset the "chibicc compiles
// its own source" playground card mounts (SELFHOST_C.md §7 step 5). For each of chibicc's own cc1 TUs
// it discovers the TU's full `#include` closure with `chibicc -M` (chibicc.h pulls ~96 glibc headers),
// unions them across TUs (the closure is shared — seeded once), adds the TU sources + the self-host
// prelude, and encodes one `temen_fs` image → `web/assets/chibicc_selfhost.img`. The card then runs
// `chibicc.temen --emit-object <tu>` over this memfs (the `temen_selfhost_*_emit_object_fs` FFI), emitting
// that TU's linkable object — chibicc compiling its own parser/tokenizer/codegen, client-side.
//
// Like `build-onramp-assets.mjs`, the image is **regenerable / gitignored** (it embeds the runner's
// system headers — not committed), built in the `real-browser` CI job before the Playwright gate, which
// SKIPs if it's absent. Needs a native `frontend/chibicc/chibicc` (built here if missing) + its Linux
// header tree; the guest reads headers from the image, native from the real `/usr/include`, so the two
// stay byte-comparable on the same machine.
//
// Usage:  node build-selfhost-assets.mjs
import { execFileSync } from 'node:child_process';
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = join(HERE, '..');
const CHIBICC = join(REPO, 'frontend/chibicc/chibicc');
const PRELUDE = 'crates/temen-run/demos/chibicc_selfhost/selfhost_prelude.h';
const OUT = join(HERE, 'web/assets/chibicc_selfhost.img');

// The **tractable** cc1 TUs (SELFHOST_C.md — ≤ ~800 lines). The three giants (preprocess/parse/
// codegen_ir) share the same header closure and are added later; each still compiles in a few hundred
// ms on the wasm-JIT, so the split is about PR size, not feasibility.
const TUS = ['strings.c', 'hashmap.c', 'unicode.c', 'type.c', 'tokenize.c'];

function ensureChibicc() {
  if (existsSync(CHIBICC)) return;
  console.log('building native chibicc…');
  execFileSync('make', ['chibicc'], { cwd: join(REPO, 'frontend/chibicc'), stdio: 'inherit' });
}

// temen_fs::encode_image wire form (crates/temen-fs/src/lib.rs): the unified TEMEN wire header
// (WIRE.md — "TEMEN\0\0\0" + u16 kind=3 fs-image + u16 version=1 + u32 flags=0) + dirs(u32 n, each
// u32 len+bytes) + files(u32 n, each u32 len+path, u64 len+data). Little-endian.
function encodeImage(files, dirs) {
  const enc = new TextEncoder(); const parts = []; let n = 0;
  const u16 = (v) => { const b = new Uint8Array(2); new DataView(b.buffer).setUint16(0, v, true); parts.push(b); n += 2; };
  const u32 = (v) => { const b = new Uint8Array(4); new DataView(b.buffer).setUint32(0, v, true); parts.push(b); n += 4; };
  const u64 = (v) => { const b = new Uint8Array(8); new DataView(b.buffer).setBigUint64(0, BigInt(v), true); parts.push(b); n += 8; };
  const raw = (b) => { parts.push(b); n += b.length; };
  raw(enc.encode('TEMEN')); raw(new Uint8Array(3)); u16(3); u16(1); u32(0); // TEMEN header: kind=fs-image, v1
  u32(dirs.length); for (const d of dirs) { const b = enc.encode(d); u32(b.length); raw(b); }
  u32(files.length); for (const [p, data] of files) { const b = enc.encode(p); u32(b.length); raw(b); u64(data.length); raw(data); }
  const out = new Uint8Array(n); let o = 0; for (const p of parts) { out.set(p, o); o += p.length; } return out;
}

// The union closure of the TU set, keyed exactly as the guest resolves paths (mirrors
// `chibicc_tu_closure` in crates/temen/tests/c_link.rs): repo-relative for repo files, `usr/...` for
// system headers. Every ancestor dir is registered so the memfs mount has the directory nodes.
ensureChibicc();
const files = new Map(); const dirs = new Set();
const addDirs = (rel) => { let d = dirname(rel); while (d && d !== '.' && d !== '/') { dirs.add(d); d = dirname(d); } };
const add = (rel, real) => { if (!files.has(rel)) { files.set(rel, readFileSync(real)); addDirs(rel); } };
for (const tu of TUS) {
  const tuRel = `frontend/chibicc/${tu}`;
  const out = execFileSync(CHIBICC, ['-M', '-c', tuRel, '-Ifrontend/chibicc'], { cwd: REPO }).toString();
  for (const tok of out.slice(out.indexOf(':') + 1).split(/\s+/)) {
    if (!tok || tok === '\\') continue;
    const real = tok.startsWith('/') ? tok : join(REPO, tok.replace(/^\.\//, ''));
    const rel = real.startsWith(REPO + '/') ? real.slice(REPO.length + 1) : real.replace(/^\//, '');
    add(rel, real);
  }
}
add(PRELUDE, join(REPO, PRELUDE)); // force-included on every emit-object compile
const fileList = [...files.entries()];
const blob = encodeImage(fileList, [...dirs]);
writeFileSync(OUT, blob);
const kb = (blob.length / 1024).toFixed(0);
console.log(`  ✓ chibicc_selfhost.img (${kb} KB — ${fileList.length} files, ${dirs.size} dirs, TUs: ${TUS.join(' ')})`);
