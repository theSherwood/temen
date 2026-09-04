// How `opcodes.corpus` was produced (kept for reproducibility; NOT run in CI). One line per program:
//   <program hex> <wst hex | -> <rst hex | -> <fnv32 of ram[0..0x10000] ++ dev[0..256]>
// Expectations come from uxn5's spec-compliant core, the `uxn.wasm` npm package (MIT): 100 random
// straight-line programs from each of three seeds (a program whose stores rewrote it into a loop is
// dropped — `uxn_test` is the uxn.c driver from the same fuzz), plus cf.tal / cf2.tal / primes.tal
// assembled with uxnasm. Run from a directory holding the three .rom files, `uxn_test`, and an
// `../uxnwasm/package` unpack of the npm tarball:  node gencorpus.mjs > opcodes.corpus
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { Uxn } from '../uxnwasm/package/dist/uxn.esm.js';
import { mux } from '../uxnwasm/package/dist/util.esm.js';
const CF = new Set([0x0c, 0x0d, 0x0e]);
const hex = (a) => Buffer.from(a).toString('hex');
async function expect(buf) {
  const u = new Uxn(); await u.init(mux(u, {})); u.load(buf); u.eval(0x100);
  let h = 2166136261 >>> 0;
  for (let i = 0; i < 0x10000; i++) h = Math.imul(h ^ u.ram[i], 16777619) >>> 0;
  for (let i = 0; i < 256; i++) h = Math.imul(h ^ u.dev[i], 16777619) >>> 0;
  const st = (s) => { const b = []; for (let i = 0; i < s.ptr(); i++) b.push(s.get(i)); return hex(b) || '-'; };
  return `${hex(buf)} ${st(u.wst)} ${st(u.rst)} ${h.toString(16).padStart(8, '0')}`;
}
const out = [];
for (const f of ['cf.rom', 'cf2.rom', 'primes_c.rom']) out.push(await expect(readFileSync(f)));
for (const [seed0, count] of [[7, 100], [12345, 100], [777, 100]]) {
  let seed = seed0;
  const rnd = () => { seed = (Math.imul(seed, 1103515245) + 12345) >>> 0; return seed >>> 8; };
  let kept = 0;
  while (kept < count) {
    const prog = [];
    for (let i = 0; i < 6; i++) { prog.push(0x80, rnd() & 0xff); prog.push(0xc0, rnd() & 0xff); }
    const len = 4 + rnd() % 40;
    for (let i = 0; i < len; i++) {
      const op = rnd() & 0xff, base = op & 0x1f;
      if (base === 0) { if (!(op & 0x80)) continue; prog.push(op, rnd() & 0xff, rnd() & 0xff); continue; }
      if (CF.has(base)) continue;
      prog.push(op);
    }
    prog.push(0x00);
    const buf = Buffer.from(prog);
    try { execFileSync('./uxn_test', { input: buf, timeout: 2000 }); } catch { continue; }
    out.push(await expect(buf));
    kept++;
  }
}
process.stdout.write(out.join('\n') + '\n');
