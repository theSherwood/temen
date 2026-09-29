// #1896 — the leaf-process cases `browser-nim-leaf-test.mjs` runs, in any host that has the engine:
// Node, and Chromium, whose JSPI suspends a leaf's emitted frames where a call parks. Each case is a
// process tree `temen_nim_open` opens as a cooperative tier-up session and `drive`
// (`driveCoopTierupRun`) runs: a driver forks, its child execs `/bin/leaf`, and the driver exits
// with the child's status. `runCases` reports how each tree ended and what it asked of the host.
// The native pins are `temen-posix/tests/leaf_tierup.rs`; this is the shipped JS path.

// A leaf image. The entry is compute the emitter takes; the helper it calls makes the `__px_*`
// calls, which the emitted entry bounces to the interpreter — the shape of a real nim module's
// `_start`. It writes `leaf` to `out.txt` and exits 5, but for its `kind`:
// - `ro`: a second helper first makes a page of the window read-only (`vm_protect`) and returns
//   where the status' source byte is, and the entry reads it itself ('l' − 103 = 5) — an access the
//   emitter cannot prove in bounds, after the page state went past one bound: only a page-checked
//   emit (#1896: the engine offers an image that can change its page state paged) runs it without a
//   false fault;
// - `null`: the entry first loads through a null pointer, which faults on both tiers — an exec'd
//   image starts behind the NULL guard (#1094), as the emitted code's baked guard does — so the
//   process crashes and its parent reaps 128.
const leafImage = (kind) => `memory 17
import 0 "__px_open" (i64, i64, i64) -> (i64)
import 1 "__px_write" (i64, i64, i64) -> (i64)
import 2 "__px_close" (i64) -> (i64)
import 3 "__px_exit" (i64) -> ()
${kind === 'ro' ? 'import 4 "vm_protect" (i64, i64, i64) -> (i64)\nimport 5 "vm_page_size" () -> (i64)\n' : ''}data 40000 "out.txt"
data 40100 "leaf"
func (i64) -> (i64) {
block 0 (vcap: i64) {
${{
  ro: `  vp = call 2 ()
  vb = i32.load8_u vp
  vbl = i64.extend_i32_u vb
  vk = i64.const 103
  vfive = i64.sub vbl vk
`,
  null: '  vnull = i64.const 8\n  vx = i64.load vnull\n  vfive = i64.const 5\n',
  plain: '  vfive = i64.const 5\n',
}[kind]}  vr = call 1 (vfive)
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
${kind === 'ro' ? `func () -> (i64) {
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

// The C shim's `read`/`write` over a core pipe end (`c_posix.rs`'s `PIPE_SHIM`): the personality op
// answers a core pipe fd with a redirect tag naming its handle, and the stream call on that handle
// moves the bytes, parking on an empty or full pipe. `read`/`write` are the personality imports'
// indices; the two helpers, read then write, follow a module's own functions.
const shim = (read, write) => [[read, 0], [write, 1]].map(([imp, op]) => `func (i64, i64, i64) -> (i64) {
block 0 (vfd: i64, vbuf: i64, vlen: i64) {
  vr = call.import ${imp} (vfd, vbuf, vlen)
  vlim = i64.const -1048576
  vtag = i64.le_s vr vlim
  br_if vtag 1(vr, vbuf, vlen) 2(vr)
  }
block 1 (xr: i64, xbuf: i64, xlen: i64) {
  vbase = i64.const 1048576
  vsum = i64.add xr vbase
  vz = i64.const 0
  vh64 = i64.sub vz vsum
  vh = i32.wrap_i64 vh64
  vn = call.cap 0 ${op} (i64, i64) -> (i64) vh (xbuf, xlen)
  return vn
  }
block 2 (xr: i64) {
  return xr
  }
}
`).join('');

// A driver that answers its child over two core pipes, `down` (fds 3, 4) and `up` (fds 5, 6): it reads
// the child's 4 bytes from `up`, writes `pong` down, and exits with the child's status.
const pingPong = () => `memory 17
import 0 "__px_execve" (i64, i64, i64) -> (i64)
import 1 "__px_exit" (i64) -> ()
import 2 "__px_fork" () -> (i64)
import 3 "__px_wait4" (i64, i64, i64, i64) -> (i64)
import 4 "__px_pipe_adopt" (i64, i64, i64) -> (i64)
import 5 "__px_read" (i64, i64, i64) -> (i64)
import 6 "__px_write" (i64, i64, i64) -> (i64)
data 40000 "/bin/leaf\\x00"
data 40100 "pong"
func () -> () {
block 0 () {
  vh0 = i32.const 0
  vhs = i64.const 42000
  vfds = i64.const 42100
  vpd = call.cap 4294967295 16 (i64) -> (i64) vh0 (vhs)
  vrh32 = i32.load vhs
  vrh = i64.extend_i32_u vrh32
  vwh32 = i32.load vhs offset=4
  vwh = i64.extend_i32_u vwh32
  vad = call.import 4 (vrh, vwh, vfds)
  vpu = call.cap 4294967295 16 (i64) -> (i64) vh0 (vhs)
  vrh32u = i32.load vhs
  vrhu = i64.extend_i32_u vrh32u
  vwh32u = i32.load vhs offset=4
  vwhu = i64.extend_i32_u vwh32u
  vau = call.import 4 (vrhu, vwhu, vfds)
  vpid = call.import 2 ()
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
  vupr = i64.const 5
  vbuf = i64.const 43000
  vfour = i64.const 4
  vn = call 1 (vupr, vbuf, vfour)
  vdownw = i64.const 4
  vpong = i64.const 40100
  vw = call 2 (vdownw, vpong, vfour)
  vst = i64.const 41000
  vz = i64.const 0
  vwt = call.import 3 (xpid, vst, vz, vz)
  vhi = i64.const 41001
  vsw = i32.load8_u vhi
  vs = i64.extend_i32_u vsw
  call.import 1 (vs)
  unreachable
  }
}
${shim(5, 6)}export 0 func "_start" 0
`;

// `/bin/leaf` for `pingPong`: writes `ping` up (fd 6), after `fills` writes of 16 KiB (four fill the
// pipe), then reads the reply from down (fd 3), writes it to `out.txt`, and returns the count it read
// plus 10. The entry only calls, so it emits; the helpers bounce to the interpreter, and a read of an
// empty pipe, or a write to a full one, parks in them.
const pingLeaf = (fills) => `memory 17
import 0 "__px_open" (i64, i64, i64) -> (i64)
import 1 "__px_write" (i64, i64, i64) -> (i64)
import 2 "__px_close" (i64) -> (i64)
import 3 "__px_exit" (i64) -> ()
import 4 "__px_read" (i64, i64, i64) -> (i64)
data 40000 "out.txt"
data 40100 "ping"
func (i64) -> (i64) {
block 0 (vcap: i64) {
  vr = call 1 ()
  vst = call 2 (vr)
  return vst
  }
}
func () -> (i64) {
block 0 () {
  vup = i64.const 6
  vzeros = i64.const 100000
  vchunk = i64.const 16384
${Array.from({ length: fills }, (_, i) => `  vf${i} = call 4 (vup, vzeros, vchunk)\n`).join('')}  vping = i64.const 40100
  vfour = i64.const 4
  vw = call 4 (vup, vping, vfour)
  vdown = i64.const 3
  vbuf = i64.const 40200
  vr = call 3 (vdown, vbuf, vfour)
  return vr
  }
}
func (i64) -> (i64) {
block 0 (vr: i64) {
  vpath = i64.const 40000
  vplen = i64.const 7
  vflags = i64.const 577
  vfd = call.import 0 (vpath, vplen, vflags)
  vbuf = i64.const 40200
  vw = call.import 1 (vfd, vbuf, vr)
  vc = call.import 2 (vfd)
  vten = i64.const 10
  vst = i64.add vr vten
  return vst
  }
}
${shim(4, 1)}`;

// Run every case with the engine `ex` over `memory`, `drive` its driver, `suspends` whether the host
// suspends a leaf's frames. Each result is how the tree ended (`exit`, what it `wrote` to
// `/w/out.txt`) and what the run asked of the host: the program of each leaf tier-up (each TIERUP
// service reads `temen_coop_module` once), how many parked calls it resumed, whether any event ran
// page-checked.
export async function runCases({ ex, memory, drive, suspends }) {
  const u8 = () => new Uint8Array(memory.buffer);
  const enc = new TextEncoder();
  const dec = new TextDecoder();
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
  const run = async (driverText, leafText) => {
    const leafTierups = [];
    let resumes = 0;
    let paged = false;
    const watch = {
      temen_coop_module: (m) => { if (m !== 0) leafTierups.push(m); },
      temen_coop_run: (ev) => { if (ev === 4 /* COOP_RUN_RESUME */) resumes++; },
      temen_coop_paged: (p) => { paged ||= p !== 0; },
    };
    const counted = Object.fromEntries(
      Object.entries(Object.getOwnPropertyDescriptors(ex)).map(([k, d]) => {
        const v = d.value;
        const seen = watch[k];
        return [k, seen ? (...a) => { const r = v(...a); seen(r); return r; } : v];
      }));
    const args = [
      put(parse(driverText)),
      put(blob([['/w/bin/leaf\n/bin/leaf', parse(leafText)]])),
      put(blob([])),
      put(enc.encode('bin/driver\0')),
      put(enc.encode('/w')),
    ].flat();
    args.push(suspends ? 1 : 0);
    if (ex.temen_nim_open(...args) !== 0) throw new Error(`temen_nim_open: status ${ex.temen_status()}`);
    await drive(counted, memory);
    const [pp, pl] = put(enc.encode('/w/out.txt'));
    const n = Number(ex.temen_nim_file(pp, pl));
    const at = Number(ex.temen_nim_file_ptr());
    const wrote = n >= 0 ? dec.decode(u8().slice(at, at + n)) : null;
    return { exit: ex.temen_exit_code(), wrote, leafTierups, resumes, paged };
  };
  return {
    leaf: await run(driver(false), leafImage('plain')),
    ro: await run(driver(false), leafImage('ro')),
    piped: await run(driver(true), leafImage('plain')),
    nul: await run(driver(false), leafImage('null')),
    nulPiped: await run(driver(true), leafImage('null')),
    parkRead: await run(pingPong(), pingLeaf(0)),
    parkWrite: await run(pingPong(), pingLeaf(4)),
  };
}
