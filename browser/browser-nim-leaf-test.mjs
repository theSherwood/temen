// #1896 — a process tree's **leaf processes** run whole on the emitted tier, checked through the real
// JS driver. `temen_nim_open` opens a POSIX build as a cooperative tier-up session: the tree forks,
// execs and waits on the interpreter, and an exec'd image that cannot park pauses the run as a TIERUP
// of its own program, which `driveCoopTierupRun` instantiates and runs whole. The native pin is
// `temen-posix/tests/leaf_tierup.rs` (the entry served by a bounce); this is the shipped path.
//
// The tree: the driver forks, the child execs `/bin/leaf`, and the driver exits with the child's
// status. `/bin/leaf` writes `leaf` to `out.txt` and exits 5. Asserts, for a leaf child: one tier-up,
// of a program other than 0, unpaged, and the tree ends with exit 5 and the file written. A leaf that
// makes a page read-only, then reads memory from emitted code, tiers up page-checked and ends the
// same way. With a pipe made before the fork, the child holds pipe ends it could block on, so it runs
// interpreted — no tier-up — and the tree ends the same way.
//
// Usage:  node browser-nim-leaf-test.mjs [module.wasm]   (build the threads cdylib first)

import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { driveCoopTierupRun } from './web/wasmjit-module.js';
import { engineImports } from './engine-imports.mjs';

const ROOT = dirname(fileURLToPath(import.meta.url));
const wasmPath = process.argv[2]
  ?? join(ROOT, 'target/wasm32-unknown-unknown/release/temen_browser.wasm');
if (!existsSync(wasmPath)) {
  console.error(`SKIP: ${wasmPath} missing`);
  process.exit(0);
}

const mod = await WebAssembly.compile(readFileSync(wasmPath));
const memory = new WebAssembly.Memory({ initial: 2048, maximum: 16384, shared: true });
const { exports: ex } = await WebAssembly.instantiate(mod, engineImports(memory));
const u8 = () => new Uint8Array(memory.buffer);
const enc = new TextEncoder();
const dec = new TextDecoder();

// A leaf image. The entry is compute the emitter takes; the helper it calls makes the `__px_*`
// calls, which the emitted entry bounces to the interpreter — the shape of a real nim module's
// `_start`. With \`ro\`, a second helper first makes a page of the window read-only (\`vm_protect\`)
// and returns where the status' source byte is, and the entry then reads it itself ('l' − 103 = 5),
// an access the emitter cannot prove in bounds: the page state is past one bound mid-run, which only
// a page-checked emit (#1896: the engine offers an image that can change its page state paged) runs
// without a false fault.
const leafImage = (ro) => `memory 17
import 0 "__px_open" (i64, i64, i64) -> (i64)
import 1 "__px_write" (i64, i64, i64) -> (i64)
import 2 "__px_close" (i64) -> (i64)
import 3 "__px_exit" (i64) -> ()
${ro ? 'import 4 "vm_protect" (i64, i64, i64) -> (i64)\nimport 5 "vm_page_size" () -> (i64)\n' : ''}data 40000 "out.txt"
data 40100 "leaf"
func (i64) -> (i64) {
block 0 (vcap: i64) {
${ro ? `  vp = call 2 ()
  vb = i32.load8_u vp
  vbl = i64.extend_i32_u vb
  vk = i64.const 103
  vfive = i64.sub vbl vk
` : '  vfive = i64.const 5\n'}  vr = call 1 (vfive)
  return vr
  }
}
func (i64) -> (i64) {
block 0 (vstatus: i64) {
  vpath = i64.const 40000
  vplen = i64.const 7
  vflags = i64.const 577
  vfd = call.import 0 (vpath, vplen, vflags)
  vbuf = i64.const 40100
  vn = i64.const 4
  vw = call.import 1 (vfd, vbuf, vn)
  vc = call.import 2 (vfd)
  call.import 3 (vstatus)
  unreachable
  }
}
${ro ? `func () -> (i64) {
block 0 () {
  vpg = call.import 5 ()
  vtwenty = i64.const 20
  vat = i64.mul vpg vtwenty
  vread = i64.const 1
  vr = call.import 4 (vat, vpg, vread)
  vl = i64.const 40100
  vsrc = i64.add vl vr
  return vsrc
  }
}
` : ''}`;

const driver = (pipe) => `memory 17
import 0 "__px_execve" (i64, i64, i64) -> (i64)
import 1 "__px_exit" (i64) -> ()
import 2 "__px_fork" () -> (i64)
import 3 "__px_wait4" (i64, i64, i64, i64) -> (i64)
import 4 "__px_pipe" (i64) -> (i64)
data 40000 "/bin/leaf\\x00"
func () -> () {
block 0 () {
${pipe ? '  vfds = i64.const 42000\n  vpp = call.import 4 (vfds)\n' : ''}  vpid = call.import 2 ()
  vz = i64.const 0
  vchild = i64.eq vpid vz
  br_if vchild 1() 2(vpid)
  }
block 1 () {
  vp = i64.const 40000
  vz = i64.const 0
  vr = call.import 0 (vp, vz, vz)
  vnine = i64.const 9
  call.import 1 (vnine)
  unreachable
  }
block 2 (xpid: i64) {
  vst = i64.const 41000
  vz = i64.const 0
  vw = call.import 3 (xpid, vst, vz, vz)
  vhi = i64.const 41001
  vsw = i32.load8_u vhi
  vs = i64.extend_i32_u vsw
  call.import 1 (vs)
  unreachable
  }
}
export 0 func "_start" 0
`;

const put = (bytes) => {
  const p = Number(ex.temen_alloc(bytes.length));
  u8().set(bytes, p);
  return [p, bytes.length];
};
const parse = (text) => {
  const [p, n] = put(enc.encode(text));
  if (ex.temen_parse(p, n) !== 1) throw new Error('the IR does not parse');
  return u8().slice(Number(ex.temen_parse_ptr()), Number(ex.temen_parse_ptr()) + ex.temen_parse_len());
};
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

// Build the tree and drive it, counting the tier-ups of programs other than 0 (each TIERUP service
// reads `temen_coop_module` once) and whether any event ran page-checked (`temen_coop_paged`).
const run = async (pipe, ro) => {
  const leafTierups = [];
  let paged = false;
  const watch = {
    temen_coop_module: (m) => { if (m !== 0) leafTierups.push(m); },
    temen_coop_paged: (p) => { paged ||= p !== 0; },
  };
  const counted = Object.fromEntries(
    Object.entries(Object.getOwnPropertyDescriptors(ex)).map(([k, d]) => {
      const v = d.value;
      const seen = watch[k];
      return [k, seen ? (...a) => { const r = v(...a); seen(r); return r; } : v];
    }));
  const args = [
    put(parse(driver(pipe))),
    put(blob([['/w/bin/leaf\n/bin/leaf', parse(leafImage(ro))]])),
    put(blob([])),
    put(enc.encode('bin/driver\0')),
    put(enc.encode('/w')),
  ].flat();
  if (ex.temen_nim_open(...args) !== 0) throw new Error(`temen_nim_open: status ${ex.temen_status()}`);
  await driveCoopTierupRun(counted, memory);
  const [pp, pl] = put(enc.encode('/w/out.txt'));
  const n = Number(ex.temen_nim_file(pp, pl));
  const wrote = n >= 0 ? dec.decode(u8().slice(Number(ex.temen_nim_file_ptr()), Number(ex.temen_nim_file_ptr()) + n)) : null;
  return { exit: ex.temen_exit_code(), wrote, leafTierups, paged };
};

const fail = (msg) => { console.error(`FAIL: ${msg}`); process.exitCode = 1; };
const ended = (name, r) => {
  if (r.exit !== 5 || r.wrote !== 'leaf') fail(`${name} tree: exit ${r.exit}, wrote ${r.wrote}`);
};
const leaf = await run(false, false);
ended('leaf', leaf);
if (leaf.leafTierups.length !== 1 || leaf.paged) {
  fail(`the leaf child tiers up once, unpaged: ${leaf.leafTierups}, paged ${leaf.paged}`);
}
const ro = await run(false, true);
ended('read-only page', ro);
if (ro.leafTierups.length !== 1 || !ro.paged) {
  fail(`the protecting leaf tiers up once, paged: ${ro.leafTierups}, paged ${ro.paged}`);
}
const piped = await run(true, false);
ended('piped', piped);
if (piped.leafTierups.length !== 0) fail(`a child holding a pipe runs interpreted: ${piped.leafTierups}`);
if (process.exitCode) process.exit(process.exitCode);
console.log(`ok — the leaf child ran on the emitted tier (program ${leaf.leafTierups[0]}); the one `
  + 'that protects a page ran page-checked; the piped one interpreted; every tree exits 5 with the '
  + 'file written');
