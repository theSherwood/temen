// Runs on the **emitted tier** — the run-to-completion cards (`reactor.js` runs a reactor's frames).
// `runJitModule` (a stdin guest), `runJitCompiler` and `runJitSelfhost` (chibicc over
// a seeded memfs + argv) open the pump, which runs the root whole as a leaf (#1414 S4-3), and drive it
// with `driveCoopTierupRun`. Each returns the run status (0 = returned, 5 = exited) or throws when the
// root isn't emittable or the run traps (the caller falls back to the interpreter). The run's stdout,
// stderr and exit code land in the shared slots, read through the usual `temen_stdout_*` /
// `temen_exit_code` accessors exactly as after the interpreter's `temen_run_onramp`.

import {
  jitCacheStats, moduleCacheClear, driveCoopTierupRun, suspendsLeaves,
} from './coop-driver.js';

export { jitCacheStats, driveCoopTierupRun, suspendsLeaves };

export function jitCacheClear() {
  moduleCacheClear();
}

// A **registry blob** (the cdylib's `registry_blob`): all integers little-endian, a u32 entry count,
// then per entry a u32 name length, the name (UTF-8), a u32 length and the bytes. Its entries, each
// `[name, bytes]` with `bytes` a view into `blob`.
export function blobEntries(blob) {
  const dv = new DataView(blob.buffer, blob.byteOffset, blob.byteLength);
  const dec = new TextDecoder();
  const out = [];
  let o = 4;
  for (let n = dv.getUint32(0, true); n > 0; n--) {
    const nl = dv.getUint32(o, true);
    const name = dec.decode(blob.subarray(o + 4, o + 4 + nl));
    const bl = dv.getUint32(o + 4 + nl, true);
    o += 8 + nl;
    out.push([name, blob.subarray(o, o + bl)]);
    o += bl;
  }
  return out;
}

// The registry blob of `entries`, each `[name, bytes]` ([`blobEntries`]'s inverse).
export function registryBlob(entries) {
  const enc = new TextEncoder();
  const named = entries.map(([n, b]) => [enc.encode(n), b]);
  const out = new Uint8Array(named.reduce((t, [n, b]) => t + 8 + n.length + b.length, 4));
  const dv = new DataView(out.buffer);
  dv.setUint32(0, named.length, true);
  let o = 4;
  for (const [n, b] of named) {
    dv.setUint32(o, n.length, true);
    out.set(n, o + 4);
    dv.setUint32(o + 4 + n.length, b.length, true);
    out.set(b, o + 8 + n.length);
    o += 8 + n.length + b.length;
  }
  return out;
}

// #958 — nimony's toolchain as the nim card ships it (`web/assets/nimony.blob.gz`, written by
// temen-run's `nim_card_bundle`): a registry blob of `cwd`, the directory a build with it runs in (its library
// pack names it); `commands`, as `temen_nim_toolchain` takes them, nimony first; and `files`, what its
// first build seeds before its program — nimony's library, the library pack, the guest libc.
export function nimToolchain(bundle) {
  const e = new Map(blobEntries(bundle));
  return {
    cwd: new TextDecoder().decode(e.get('cwd')),
    commands: e.get('commands'),
    nimony: blobEntries(e.get('commands'))[0][1],
    files: blobEntries(e.get('files')),
  };
}

// The toolchain each engine holds (`temen_nim_toolchain`), by its exports: `{ toolchain, token }`.
const installed = new WeakMap();

// #958 — compile `source` as `prog.nim` with nimony's own driver and run what it builds:
// `nimony t -r --novalidate --isMain prog.nim`, one `temen_nim_open` session over `toolchain`
// ([`nimToolchain`]) that `driveCoopTierupRun` runs. An engine is given the toolchain once
// (`temen_nim_toolchain`): its first build with it ships the toolchain's files, a later one only
// `prog.nim` (#2366). No plugin validator ships with the toolchain, so
// `--novalidate` keeps nimony from warning about it at every plugin it builds. The process tree
// (nimony, nifmake, nimsem, the shell) runs on the interpreter, each leaf process (nifler2, shoggoth,
// the program itself) whole on the emitted tier
// (#1896), and temen-link natively. Each build continues in the directory the engine's last one left
// (#2099), so nimony redoes only what `source` changed. nimony prints nothing when a build succeeds,
// so what the session prints is the program's. When the build fails, nothing is linked and what it
// printed is nimony's diagnostics. Resolves `{ status, exit, stdout, stderr, built, leaves, resumes }`:
// `built` is whether this build linked the program, and so ran it.
export async function nimCompileRun(ex, memory, toolchain, source) {
  const enc = new TextEncoder();
  const text = (p, n) => new TextDecoder().decode(new Uint8Array(memory.buffer, p, n).slice());
  const held = [];
  const put = (bytes) => {
    const p = Number(ex.temen_alloc(bytes.length));
    new Uint8Array(memory.buffer).set(bytes, p);
    held.push([p, bytes.length]);
    return [p, bytes.length];
  };
  const { cwd } = toolchain;
  // `nimcache/t/<stem>/prog.temen`, `<stem>` nimony's for `prog.nim`, which lands on the stdout
  // slot (read it only after the call: the slot moves).
  const stemLen = ex.temen_nim_module_suffix(...put(enc.encode('prog.nim')));
  const program = enc.encode(`${cwd}/nimcache/t/${text(Number(ex.temen_stdout_ptr()), stemLen)}/prog.temen`);
  let inst = installed.get(ex);
  let seed = [];
  if (inst?.toolchain !== toolchain) {
    const token = Number(ex.temen_nim_toolchain(...put(toolchain.nimony), ...put(toolchain.commands)));
    if (token < 0) throw new Error(`temen_nim_toolchain: status ${ex.temen_status()}`);
    inst = { toolchain, token };
    installed.set(ex, inst);
    seed = toolchain.files;
  }
  // A build continues in the directory the last one left (#2099), so the program there may be the
  // last build's: remove it, and the program after this build is this build's.
  ex.temen_nim_remove(...put(program));
  const files = registryBlob([...seed, [`${cwd}/prog.nim`, enc.encode(source)]]);
  const argv = enc.encode(['bin/nimony', 't', '-r', '--novalidate', '--isMain', 'prog.nim'].map((a) => `${a}\0`).join(''));
  const args = [put(files), put(argv), put(enc.encode(cwd))];
  const opened = ex.temen_nim_open(inst.token, ...args.flat(), suspendsLeaves ? 1 : 0);
  // The session holds its own copies of what it was opened with.
  for (const [p, n] of held.splice(0)) ex.temen_dealloc(p, n);
  if (opened !== 0) {
    // The next build gives the engine the toolchain again.
    installed.delete(ex);
    throw new Error(`temen_nim_open: status ${ex.temen_status()}`);
  }
  const counts = {};
  const status = await driveCoopTierupRun(ex, memory, { counts });
  const stdout = text(Number(ex.temen_stdout_ptr()), ex.temen_stdout_len());
  const stderr = text(Number(ex.temen_stderr_ptr()), ex.temen_stderr_len());
  const exit = ex.temen_exit_code();
  const built = Number(ex.temen_nim_file(...put(program))) >= 0;
  for (const [p, n] of held.splice(0)) ex.temen_dealloc(p, n);
  return { status, exit, stdout, stderr, built, ...counts };
}

// `temen_coop_open`'s `mode`: offer the root program to run whole on emitted wasm, as a leaf.
const COOP_LEAF_ROOT = 1;

// Run an on-ramp module whose input is **stdin** (Lua, SQLite, hello, a compiled C program) on the
// emitted tier, on the pump (#1414 S4-3a). One open decides how: when the emitter takes the root, it
// runs whole on emitted wasm, as a leaf; when it doesn't (it can park, or it leaves the emitter's
// subset), the interpreter drives `_start` and eligible regions run emitted. Returns the run status
// (0 = returned, 5 = exited). Throws when nothing is emittable or the run traps; the caller then runs
// the guest on the interpreter.
export async function runJitModule(ex, memory, moduleBytes, stdinBytes, cacheKey, shared = 1) {
  const u8 = () => new Uint8Array(memory.buffer);
  const modP = Number(ex.temen_alloc(moduleBytes.length));
  u8().set(moduleBytes, modP);
  let stdinP = 0;
  const stdinLen = stdinBytes ? stdinBytes.length : 0;
  if (stdinLen) {
    stdinP = Number(ex.temen_alloc(stdinLen));
    u8().set(stdinBytes, stdinP);
  }
  // `shared`: 1 (default) instantiates the emitted module against the cdylib's **shared** memory
  // (cross-origin-isolated threads build); a plain single-threaded host (e.g. the JACL playground's
  // non-threads cdylib) passes 0 — a shared-mode emit LinkErrors against a non-shared memory. Same
  // knob as `runWarmCoop`.
  const opened = ex.temen_coop_open(modP, moduleBytes.length, stdinP, stdinLen, shared, 0, 0, COOP_LEAF_ROOT);
  ex.temen_dealloc(modP, moduleBytes.length);
  if (stdinP) ex.temen_dealloc(stdinP, stdinLen);
  if (opened !== 0) {
    throw new Error(`JIT module open failed: status ${ex.temen_status()} (2 = nothing emittable)`);
  }
  return driveCoopTierupRun(ex, memory, { cacheKey });
}

// Run the warm session's `eval_run` on the **warm-coop** tier (#816 item 4): the cooperative
// tier-up drive over the restored warm image. `eval_run` is offered whole as a leaf first, as a cold
// module's root is (#1414 S4-3c); when the emitter declines it (it can park, or a function leaves the
// emitter's subset), the interpreter owns the eval and eligible pure leaves run on emitted wasm. The
// engine outlines and compiles the module once per warm session, and the emits are kept (the compiled
// `WebAssembly.Module` across Runs under `cacheKey`). Each Run is prepare (restore the image,
// re-establish the captured page map, arm `eval_run` on the coop scheduler) + the standard
// `driveCoopTierupRun` event loop. Assumes `temen_warm_open` already succeeded. Returns the run status;
// throws if the eval has nothing for the emitted tier or the run traps (the caller falls back to
// `temen_warm_eval`, the interpreter warm path).
export async function runWarmCoop(ex, memory, stdinBytes, cacheKey, shared = 1) {
  const u8 = () => new Uint8Array(memory.buffer);
  if (ex.temen_warm_coop_open(shared, COOP_LEAF_ROOT) !== 0) {
    throw new Error(`warm-coop open failed: status ${ex.temen_status()}`);
  }
  let stdinP = 0;
  const stdinLen = stdinBytes ? stdinBytes.length : 0;
  if (stdinLen) {
    stdinP = Number(ex.temen_alloc(stdinLen));
    u8().set(stdinBytes, stdinP);
  }
  const prepared = ex.temen_warm_coop_prepare(stdinP, stdinLen);
  if (stdinP) ex.temen_dealloc(stdinP, stdinLen);
  if (prepared !== 0) throw new Error(`warm-coop prepare failed: status ${ex.temen_status()} (2 = nothing emittable)`);
  return driveCoopTierupRun(ex, memory, { cacheKey });
}

// **Pre-warm** the warm+JIT `eval_run` for `cacheKey` — emit + `WebAssembly.compile` + instantiate **and
// dry-run it once** (empty input, over the restored image), so the first real `runWarmCoop` is instant.
// Called during pre-warm (off the main thread), so all of that cost is hidden.
//
// Why the dry run matters (and why compile+instantiate alone did not): V8 compiles a wasm module's
// **function bodies lazily — on first call**, not during `WebAssembly.compile`. So caching the compiled
// Module + instance still left the *first* `f0()` call paying ~1.5 s of function compilation on the
// user's first Run (measured: run1 4.7 s vs run2 2.3 s even with the instance primed). Making one `f0`
// call here forces that compilation now. Empty input is language-agnostic (QuickJS/Lua) and still enters
// the interpreter, warming the hot functions; the next real Run does `temen_warm_coop_prepare` (restores the
// image + a fresh powerbox), so the dry run leaves no state and fresh-per-Run isolation holds.
//
// Best-effort: returns false if `eval_run` isn't wasm-drivable or the dry run traps (the card then just
// uses warm-interp), true once compiled + warmed.
export async function primeWarmJit(ex, memory, cacheKey, shared = 1) {
  try {
    // A full dry Run with no stdin: opens/emits `eval_run`, compiles + instantiates (cached under
    // `cacheKey`), and — the point — makes the first call so V8 compiles the bodies now.
    await runWarmCoop(ex, memory, null, cacheKey, shared);
    return true;
  } catch {
    return false; // eval_run not emittable / trapped → card stays warm-interp
  }
}

// Run the **chibicc compiler** on the emitted tier, on the pump (#1414 S4-3b): feed it the user's C
// `srcBytes` (seeded at `/in.c`) plus the built-in libc headers under `/include`. The cdylib assembles
// the memfs + argv (`temen_coop_open_fs`, sharing the bytecode card's `chibicc_card_image`) and runs
// chibicc's root whole as a leaf, so this driver just hands over the module + source. The emitted
// TEMEN-IR comes back on `temen_stdout_*` at the run's end.
// `flags` picks how to compile: bit 0 = `-g`, bit 1 = a linkable **program unit** against libc
// declarations only (#1392). chibicc's emitted `_start` is independent of both (the source and argv are
// fed through the memfs, not baked into the code), so `cacheKey` stays valid across them.
export async function runJitCompiler(ex, memory, moduleBytes, srcBytes, flags = 0, cacheKey) {
  const u8 = () => new Uint8Array(memory.buffer);
  const modP = Number(ex.temen_alloc(moduleBytes.length));
  const srcP = Number(ex.temen_alloc(srcBytes.length));
  u8().set(moduleBytes, modP);
  u8().set(srcBytes, srcP);
  // Empty header image (0, 0) — the cdylib seeds the built-in playground headers itself. `flags` is the
  // same word the bytecode `temen_run_onramp_fs` takes; `1` = the play threads build's shared memory.
  const opened = ex.temen_coop_open_fs(modP, moduleBytes.length, 0, 0, srcP, srcBytes.length, flags, 1);
  ex.temen_dealloc(modP, moduleBytes.length);
  ex.temen_dealloc(srcP, srcBytes.length);
  if (opened !== 0) {
    throw new Error(`JIT compiler open failed: status ${ex.temen_status()} (2 = the root is not emittable)`);
  }
  return driveCoopTierupRun(ex, memory, { cacheKey });
}

// Run the **self-host** compile on the wasm-JIT (SELFHOST_C.md §7 step 5): chibicc.temen compiles one of
// its own cc1 TUs (`tuBytes`, a memfs-relative path like `frontend/chibicc/hashmap.c`) to a linkable
// object, reading the TU + its glibc header closure from `imgBytes` (the committed closure image). Same
// shape as `runJitCompiler` but through `temen_coop_open_selfhost_fs` (raw image + `--emit-object` argv;
// the window grows for the giants). The emitted object text comes back on `temen_stdout_*` at the end.
export async function runJitSelfhost(ex, memory, moduleBytes, imgBytes, tuBytes, debugInfo = 0, cacheKey) {
  const u8 = () => new Uint8Array(memory.buffer);
  const modP = Number(ex.temen_alloc(moduleBytes.length));
  const imgP = Number(ex.temen_alloc(imgBytes.length));
  const tuP = Number(ex.temen_alloc(tuBytes.length));
  u8().set(moduleBytes, modP); u8().set(imgBytes, imgP); u8().set(tuBytes, tuP);
  const opened = ex.temen_coop_open_selfhost_fs(modP, moduleBytes.length, imgP, imgBytes.length, tuP, tuBytes.length, debugInfo, 1);
  ex.temen_dealloc(modP, moduleBytes.length);
  ex.temen_dealloc(imgP, imgBytes.length);
  ex.temen_dealloc(tuP, tuBytes.length);
  if (opened !== 0) {
    throw new Error(`JIT self-host open failed: status ${ex.temen_status()} (2 = the root is not emittable)`);
  }
  return driveCoopTierupRun(ex, memory, { cacheKey });
}
