// Build the playground's **on-ramp `.temen` assets** — real C/C++ guests (Lua, SQLite) compiled
// through `clang -O2 -emit-llvm` and translated by `temen-llvm-translate` into TEMEN-IR modules the
// browser engine runs via `temen_run_onramp` (see `web/play.js`).
//
// Every asset is translated with **`--host-page 65536`**: a wasm host has 64 KiB pages, so a
// read-only global must not share a host page with the writable data stack (it would fault under
// D40). The native default (16 KiB) is wrong for the browser — see the `temen-llvm` stack-page commit.
//
// Every clang-translated asset uses the guarded powerbox layout (#964/#1094): scratch/args one guard
// up so a NULL dereference traps on every tier. This covers a `main` program's synthesized `_start`
// and **entry-less reactor kernels** too (gradient, bounce, mandelzoom, gpu_shader — `tick` only, no
// `main`): temen-llvm bases their globals one guard up, so a NULL deref in a kernel traps like
// everywhere else. The guard is **unconditional** now (#1094 — the one canonical layout; the
// `--null-guard` flag is a redundant no-op and the `__null_guard` marker export is retired). Old
// assets a host still runs are seeded the same way, so mixed old/new assets coexist harmlessly.
//
// Usage:  node build-onramp-assets.mjs           (builds whatever the toolchain + caches allow)
// Needs `clang`/`llvm-dis` on PATH. SQLite/Lua sources are fetched-and-cached (skipped offline).
// Outputs to `web/assets/*.temen` (gitignored except the tiny committed `hello_c.temen`).

import { execFileSync } from 'node:child_process';
import { mkdirSync, existsSync, readFileSync, copyFileSync, rmSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = join(HERE, '..');
const ASSETS = join(HERE, 'web', 'assets');
const HOST_PAGE = '65536';
mkdirSync(ASSETS, { recursive: true });

// Build the translator once (release), reuse its path.
const TR = join(REPO, 'crates', 'temen-llvm', 'target', 'release', 'temen-llvm-translate');
function ensureTranslator() {
  if (existsSync(TR)) return;
  console.log('building temen-llvm-translate…');
  execFileSync('cargo', ['build', '--release', '--bin', 'temen-llvm-translate'], {
    cwd: join(REPO, 'crates', 'temen-llvm'), stdio: 'inherit',
  });
}

// clang a C source to bitcode, then translate to a 64 KiB-page `.temen`. Extra clang flags per guest.
function buildC(name, src, includes = [], defines = []) {
  const bc = join(ASSETS, `${name}.bc`);
  const temen = join(ASSETS, `${name}.temen`);
  const flags = ['-O2', '-emit-llvm', '-c', '-fno-vectorize', '-fno-slp-vectorize'];
  execFileSync('clang', [...flags, ...defines, ...includes.map((i) => `-I${i}`), src, '-o', bc], { stdio: 'inherit' });
  execFileSync(TR, [bc, '-o', temen, '--host-page', HOST_PAGE, '--null-guard'], { stdio: 'inherit' });
  const size = execFileSync('wc', ['-c', temen]).toString().trim().split(/\s+/)[0];
  console.log(`  ✓ ${name}.temen (${size} B)`);
}

// Translate an **already-committed** `.bc` fixture to a 64 KiB-page `.temen` (no clang step — the
// bitcode is a golden input in the tree, e.g. the Lua test fixtures).
function buildBc(name, bcPath) {
  const temen = join(ASSETS, `${name}.temen`);
  execFileSync(TR, [bcPath, '-o', temen, '--host-page', HOST_PAGE, '--null-guard'], { stdio: 'inherit' });
  const size = execFileSync('wc', ['-c', temen]).toString().trim().split(/\s+/)[0];
  console.log(`  ✓ ${name}.temen (${size} B)`);
}

ensureTranslator();

// 1) hello — the tiny always-present example (also committed so the playground works out of the box).
try {
  buildC('hello_c', join(REPO, 'crates', 'temen-run', 'demos', 'hello.c'));
} catch (e) {
  console.log(`  ✗ hello_c: ${e.message}`);
}

// 1b) gradient — the framebuffer demo: a C guest renders an RGBA image and presents one frame through
//     the `display` capability; the page blits it to a <canvas>. The output waist Doom will ride.
try {
  buildC('gradient', join(REPO, 'crates', 'temen-run', 'demos', 'display', 'gradient.c'));
} catch (e) {
  console.log(`  ✗ gradient: ${e.message}`);
}

// 1c) bounce — the interactive reactor demo: a C guest whose exported `tick()` the page calls once per
//     requestAnimationFrame, steering a bouncing box with the arrow keys (the `keyboard` cap in, the
//     `display` cap out). The per-frame run model + input waist Doom rides.
try {
  buildC('bounce', join(REPO, 'crates', 'temen-run', 'demos', 'display', 'bounce.c'));
} catch (e) {
  console.log(`  ✗ bounce: ${e.message}`);
}

// 1d) life — Conway's Game of Life over a malloc heap ABOVE the mapped window: the reactor must
//     persist the guest's whole memory (heap included) between frames or the glider freezes. The
//     heap-persistence proof Doom's zone allocator needs.
try {
  buildC('life', join(REPO, 'crates', 'temen-run', 'demos', 'display', 'life.c'));
} catch (e) {
  console.log(`  ✗ life: ${e.message}`);
}

// 1e) mandelzoom — an interactive Mandelbrot zoom: each reactor `tick()` computes a full
//     double-precision Mandelbrot for the current (auto-zooming, arrow-steerable) view on the CPU
//     in-guest and presents it through `display`. Pure f64 + an integer palette — no libm bundled.
try {
  buildC('mandelzoom', join(REPO, 'crates', 'temen-run', 'demos', 'display', 'mandelzoom.c'));
} catch (e) {
  console.log(`  ✗ mandelzoom: ${e.message}`);
}

// 1f) gpu_shader — the GPU demo: the guest ships a WGSL fragment shader through the `webgpu` capability
//     and the browser renders it (a Mandelbrot zoom) each frame on the real GPU via navigator.gpu.
try {
  buildC('gpu_shader', join(REPO, 'crates', 'temen-run', 'demos', 'display', 'gpu_shader.c'));
} catch (e) {
  console.log(`  ✗ gpu_shader: ${e.message}`);
}

// 2) SQLite (interactive) — the unmodified 3.50.2 amalgamation with a driver that reads a SQL script
//    from **stdin** and runs it against an in-memory database, printing each statement's result table.
//    The page pipes the editor's SQL in as stdin. Fetch-and-cache the amalgamation (same version +
//    cache dir the temen-llvm test harness uses); skip offline.
const CACHE = '/tmp/temen_sqlite_cache';
const AMALG = join(CACHE, 'sqlite-amalgamation-3500200');
function ensureAmalgamation() {
  if (existsSync(join(AMALG, 'sqlite3.c'))) return true;
  mkdirSync(CACHE, { recursive: true });
  const zip = join(CACHE, 'amalgamation.zip');
  try {
    execFileSync('curl', ['-sfL', '--max-time', '120', '-o', zip,
      'https://sqlite.org/2025/sqlite-amalgamation-3500200.zip'], { stdio: 'inherit' });
    execFileSync('unzip', ['-o', '-q', zip, '-d', CACHE], { stdio: 'inherit' });
    return existsSync(join(AMALG, 'sqlite3.c'));
  } catch {
    return false;
  }
}
if (ensureAmalgamation()) {
  try {
    buildC('sqlite_repl', join(REPO, 'crates', 'temen-run', 'demos', 'sqlite', 'sqlite_repl.c'), [AMALG]);
  } catch (e) {
    console.log(`  ✗ sqlite_repl: ${e.message}`);
  }
} else {
  console.log('  – sqlite_repl skipped (amalgamation fetch failed — offline?)');
}

// 2b) QuickJS (interactive) — Bellard's QuickJS 2024-01-13 with a driver (`qjs_repl.c`) that reads a
//     JS program from **stdin**, evaluates it (print/console.log + the completion value), and prints.
//     Multi-TU, mirroring the `demo_quickjs_eval_vs_native` test: the engine + a guest libm (openlibm,
//     for the address-taken Math functions) + the reused printf/strtod/libc shims, `llvm-link`ed into
//     one `.ll`, then translated. Fetched-and-cached (QuickJS from bellard.org, openlibm from GitHub —
//     see `ensureOpenlibm` for why that one needs two mirrors); when either fetch is unavailable this
//     rebuild is skipped and the **committed** `web/assets/qjs_repl.temen` is left in place, so the JS
//     playground works out of the box regardless (see `web/assets/.gitignore` whitelist).
const QJS_VER = '2024-01-13';
const QJS_CACHE = '/tmp/temen_quickjs_cache';
const QJS_DIR = join(QJS_CACHE, `quickjs-${QJS_VER}`);
const OL_VER = '0.8.5';
const OL_CACHE = '/tmp/temen_openlibm_cache';
const OL_DIR = join(OL_CACHE, `openlibm-${OL_VER}`);
// The openlibm double set QuickJS's `Math` object takes the address of (kept in sync with the
// temen-llvm test's OPENLIBM_SRCS + QUICKJS_OPENLIBM_EXTRA).
const OPENLIBM_SRCS = [
  'e_log', 'e_log10', 'e_log2', 'e_exp', 's_exp2', 'e_pow', 's_sin', 's_cos', 's_tan',
  'k_sin', 'k_cos', 'k_tan', 'e_rem_pio2', 'k_rem_pio2', 'e_asin', 'e_acos', 's_atan',
  'e_atan2', 'e_sinh', 'e_cosh', 's_tanh', 's_cbrt', 'e_fmod', 's_scalbn', 's_copysign',
  's_fabs', 'k_exp', 's_expm1', 's_asinh', 'e_acosh', 'e_atanh', 's_log1p', 'e_hypot',
  's_floor', 's_ceil', 's_trunc', 'e_sqrt',
];
function ensureQuickJS() {
  if (existsSync(join(QJS_DIR, 'quickjs.c'))) return true;
  mkdirSync(QJS_CACHE, { recursive: true });
  try {
    const tar = join(QJS_CACHE, `quickjs-${QJS_VER}.tar.xz`);
    execFileSync('curl', ['-sfL', '--max-time', '120', '-o', tar,
      `https://bellard.org/quickjs/quickjs-${QJS_VER}.tar.xz`], { stdio: 'inherit' });
    execFileSync('tar', ['xf', tar, '-C', QJS_CACHE], { stdio: 'inherit' });
    return existsSync(join(QJS_DIR, 'quickjs.c'));
  } catch { return false; }
}
// GitHub's **archive** endpoint (`/archive/refs/tags/*.tar.gz`) is gated on some networks — it 403s
// while `github.com` git and `raw.githubusercontent.com` stay reachable. `demos/doom/fetch.sh` already
// works around exactly this split for doomgeneric; openlibm never got the same treatment, so a gated
// archive silently skipped the whole QuickJS rebuild. Mirror order: archive (fast, what CI takes),
// then a shallow **tag clone** — tag-pinned to the same commit, and it needs no per-file list to stay
// in sync with whichever sources a consumer happens to compile.
function ensureOpenlibm() {
  if (existsSync(join(OL_DIR, 'src', 'e_log.c'))) return true;
  mkdirSync(OL_CACHE, { recursive: true });
  try {
    const tgz = join(OL_CACHE, 'openlibm.tar.gz');
    execFileSync('curl', ['-sfL', '--max-time', '120', '-o', tgz,
      `https://github.com/JuliaMath/openlibm/archive/refs/tags/v${OL_VER}.tar.gz`], { stdio: 'inherit' });
    execFileSync('tar', ['xf', tgz, '-C', OL_CACHE], { stdio: 'inherit' });
    if (existsSync(join(OL_DIR, 'src', 'e_log.c'))) return true;
  } catch (e) {
    // Say which mirror failed and why — a silent `catch` here is what hid the doom outage (I42).
    console.log(`    – openlibm archive unavailable: ${e.message}`);
  }
  try {
    rmSync(OL_DIR, { recursive: true, force: true });
    execFileSync('git', ['-c', 'advice.detachedHead=false', 'clone', '-q', '--depth', '1', '--branch', `v${OL_VER}`,
      'https://github.com/JuliaMath/openlibm', OL_DIR], { stdio: 'inherit' });
    return existsSync(join(OL_DIR, 'src', 'e_log.c'));
  } catch (e) {
    console.log(`    – openlibm shallow clone failed: ${e.message}`);
    return false;
  }
}
function buildQuickJS() {
  const demos = join(REPO, 'crates', 'temen-run', 'demos');
  const cflags = ['-O2', '-emit-llvm', '-S', '-c', '-fno-vectorize', '-fno-slp-vectorize',
    '-DNDEBUG', '-D_GNU_SOURCE', `-DCONFIG_VERSION="${QJS_VER}"`, '-DASSEMBLER=0'];
  const incs = [QJS_DIR, OL_DIR, join(OL_DIR, 'include'), join(OL_DIR, 'src'), join(OL_DIR, 'amd64')]
    .map((i) => `-I${i}`);
  const cc = (src, tag) => {
    const out = join(ASSETS, `qjs_${tag}.ll`);
    execFileSync('clang', [...cflags, ...incs, src, '-o', out], { stdio: 'inherit' });
    return out;
  };
  // Compile the shared engine + shims + openlibm once; each driver just re-links against them.
  const shared = [];
  for (const tu of ['quickjs', 'libregexp', 'libunicode', 'cutils', 'libbf']) shared.push(cc(join(QJS_DIR, `${tu}.c`), tu));
  shared.push(cc(join(demos, 'postgres', 'printf_shim.c'), 'printf_shim'));
  shared.push(cc(join(demos, 'strtod', 'strtod.c'), 'strtod'));
  shared.push(cc(join(demos, 'quickjs', 'libc_shim.c'), 'libc_shim'));
  for (const s of OPENLIBM_SRCS) shared.push(cc(join(OL_DIR, 'src', `${s}.c`), s));
  // Two drivers over the same engine: `qjs_repl` (single `main`, the shipping card) and `qjs_snapshot`
  // (the WASM_AOT.md warm-runtime-snapshot two-phase driver: `main` + `warmup` + `eval_run`).
  const drivers = [['qjs_repl.c', 'repl', 'qjs_repl.temen'], ['qjs_snapshot.c', 'snapshot', 'qjs_snapshot.temen']];
  for (const [driverSrc, tag, out] of drivers) {
    const driverLl = cc(join(demos, 'quickjs', driverSrc), tag);
    const linked = join(ASSETS, `${out.replace('.temen', '')}_linked.ll`);
    execFileSync('llvm-link', ['-S', driverLl, ...shared, '-o', linked], { stdio: 'inherit' });
    const temen = join(ASSETS, out);
    execFileSync(TR, [linked, '-o', temen, '--host-page', HOST_PAGE, '--null-guard'], { stdio: 'inherit' });
    const size = execFileSync('wc', ['-c', temen]).toString().trim().split(/\s+/)[0];
    console.log(`  ✓ ${out} (${size} B)`);
  }
}
if (ensureQuickJS() && ensureOpenlibm()) {
  try {
    buildQuickJS();
  } catch (e) {
    console.log(`  ✗ qjs_repl: ${e.message}`);
  }
} else {
  console.log('  – qjs_repl rebuild skipped (quickjs/openlibm fetch failed) — using committed qjs_repl.temen');
}

// 2c) Tcl (interactive) — the reference Tcl 8.6 interpreter, built by its demo script
//     (`demos/tcl/build_bitcode.sh`: configure → native oracle → 162-TU bitcode + openlibm → llvm-link).
//     The script links these variants; the playground translates two to a 64 KiB-page `.temen` with
//     `--stub-externs` (the `tcl_init` variant stays behind for the Rust translate test only):
//       • `tcl_repl.temen` — the minimal-embedding REPL (`tcl_repl.c`, no `Tcl_Init`, no filesystem).
//       • `tcl_snapshot.temen` — the two-phase warm-runtime-snapshot driver (`tcl_snapshot.c`:
//         `warmup` = full `Tcl_Init` over an in-guest `Tcl_Filesystem` VFS serving the embedded script
//         library so `clock`/`file`/`glob`/`auto_load`/`package` all work; `eval_run` = eval-only), so
//         the playground warms the Tcl runtime once on the snapshot worker and evals per Run (issue
//         #805 follow-on). Runs byte-identical to native (`demo_tcl_init_stdin`).
//     The playground warm Tcl card uses the snapshot asset. Fail-soft: skipped (example absent) if the
//     toolchain/fetch is unavailable, like SQLite/Doom/chibicc offline.
try {
  const tclScript = join(REPO, 'crates', 'temen-run', 'demos', 'tcl', 'build_bitcode.sh');
  execFileSync('bash', [tclScript], { stdio: 'inherit' });
  const cache = process.env.TEMEN_TCL_CACHE ?? '/tmp/temen_tcl_cache';
  for (const [linkedName, temenName] of [['tcl_linked.ll', 'tcl_repl.temen'], ['tcl_snapshot_linked.ll', 'tcl_snapshot.temen']]) {
    const linked = join(cache, linkedName);
    if (!existsSync(linked)) throw new Error(`build script produced no ${linkedName}`);
    const temen = join(ASSETS, temenName);
    // Tcl is guarded again (#986): its one NULL read was `TclSetupEnv` walking a NULL `environ`
    // (the extern was laid out as zeroed BSS) — `tcl_shim.c` now defines a real empty `environ`.
    execFileSync(TR, [linked, '-o', temen, '--host-page', HOST_PAGE, '--stub-externs', '--null-guard'], { stdio: 'inherit' });
    const size = execFileSync('wc', ['-c', temen]).toString().trim().split(/\s+/)[0];
    console.log(`  ✓ ${temenName} (${size} B)`);
  }
} catch (e) {
  console.log(`  – tcl skipped (${e.message} — offline, or no clang/llvm-link)`);
}

// 3) Lua (interactive) — the warm Lua card ships the committed prebuilt **`lua_snapshot.temen`** (the
//    two-phase `main`/`warmup`/`eval_run` driver, issue #805), so nothing is built here. It's a
//    generated binary asset like the vendored `doom1.wad`: regenerate it by hand from
//    `lua_snapshot_harness.c` + Lua 5.4.7 via the recipe in `crates/temen-llvm/tests/fixtures/lua/README.md`
//    ("Lua warm-runtime-snapshot fixture") and commit the resulting `.temen`. We deliberately do NOT
//    commit the ~76k-line intermediate `.ll` (unlike `lua_eval.ll`, no Rust test consumes it).

// 4) Doom (interactive reactor) — doomgeneric through the on-ramp, driven one `tick` per frame over
//    the persistent window; `_start` reads the shareware IWAD through the `fs` capability. Two assets:
//    the module (`demos/doom/{fetch,build}.sh` — id Software's Doom *source* is fetched-and-built,
//    needs the toolchain) and the shareware `doom1.wad`, which is now **vendored in-tree**
//    (`crates/temen-run/demos/doom/doom1.wad`) rather than fetched. Vendoring the WAD retires the
//    recurring dead-mirror outage (ISSUES.md I42/I43): a mirror going away can no longer drop the WAD.
//    The WAD is staged unconditionally (it's a committed file — always reachable, no network); only
//    the module stays fail-soft (skipped, so the playground omits the example, if the toolchain or the
//    source fetch is unavailable).
const DOOM = join(REPO, 'crates', 'temen-run', 'demos', 'doom');
const DCACHE = '/tmp/doomgeneric_cache';
const VENDORED_WAD = join(DOOM, 'doom1.wad');

// Build doom.temen via the demo scripts (fetch the sources, then compile+link+translate). Returns the
// built module path, or null if the fetch/build failed (offline, or no clang/llvm-link).
function ensureDoomModule() {
  const built = join(DCACHE, 'bc', 'doom.temen');
  if (existsSync(built)) return built;
  try {
    execFileSync('sh', [join(DOOM, 'fetch.sh')], { stdio: 'inherit' });
    execFileSync('sh', [join(DOOM, 'build.sh')], { stdio: 'inherit' });
    return existsSync(built) ? built : null;
  } catch (e) {
    console.log(`  ✗ doom build: ${e.message}`);
    return null;
  }
}

// The shareware IWAD, vendored in-tree — freely-redistributable id Software DOOM shareware v1.9
// (md5 f0cefca49926d00903cf57551d901abe, 4196020 bytes; provenance/license in demos/doom/README.md).
// No network, no mirrors. Verify the IWAD magic as a cheap checkout-integrity guard.
function vendoredWad() {
  const buf = readFileSync(VENDORED_WAD);
  if (buf.subarray(0, 4).toString('latin1') !== 'IWAD') {
    throw new Error(`vendored ${VENDORED_WAD} is not an IWAD (magic ${JSON.stringify(buf.subarray(0, 4).toString('latin1'))}) — corrupt checkout?`);
  }
  return VENDORED_WAD;
}

// Stage the WAD first, unconditionally — it's committed, so it's always reachable regardless of the
// toolchain. The module build is the only fail-soft half now.
copyFileSync(vendoredWad(), join(ASSETS, 'doom1.wad'));
const doomModule = ensureDoomModule();
const mb = (n) => (readFileSync(n).length / (1024 * 1024)).toFixed(2);
if (doomModule) {
  copyFileSync(doomModule, join(ASSETS, 'doom.temen'));
  console.log(`  ✓ doom.temen (${mb(doomModule)} MB) + doom1.wad (vendored, ${mb(join(ASSETS, 'doom1.wad'))} MB)`);
} else {
  console.log(`  – doom.temen skipped (module build failed — offline, or no toolchain?); doom1.wad (vendored) staged`);
}

// chibicc — the in-browser C compiler (SELFHOST_C.md §7 step 5). Multi-TU like QuickJS/Doom, so it's
// built by its demo script (per-TU bitcode → llvm-link → translate → verify) and the resulting
// `chibicc.temen` copied in. The playground compiles a C source with it, `temen_parse`s the emitted IR,
// and runs the result. Fail-soft: no clang/llvm ⇒ the demo is simply absent (the card shows a build
// hint), like Lua/Doom.
try {
  const chibiccScript = join(REPO, 'crates', 'temen-run', 'demos', 'chibicc_selfhost', 'build_chibicc_temen.sh');
  const chibiccCache = process.env.TEMEN_CHIBICC_CACHE ?? '/tmp/temen_chibicc_cache';
  execFileSync('bash', [chibiccScript], { stdio: 'inherit' });
  const built = join(chibiccCache, 'chibicc.temen');
  if (!existsSync(built)) throw new Error('build script produced no chibicc.temen');
  copyFileSync(built, join(ASSETS, 'chibicc.temen'));
  const kb = (readFileSync(built).length / 1024).toFixed(0);
  console.log(`  ✓ chibicc.temen (${kb} KB)`);
} catch (e) {
  console.log(`  – chibicc skipped (${e.message} — offline, or no clang/llvm-18?)`);
}

// temen-leng — the in-browser leng→TEMEN-IR self-host card (NIM.md §3e): the real `temen-leng` translator,
// compiled to a verified Temen module through the LLVM on-ramp, run over a real hexer Leng file. Unlike
// chibicc it needs the `-Z build-std`/`llvm-18` toolchain to rebuild, so — like `shell.temen` — its
// bytes are the committed in-tree asset (`crates/temen-run/demos/leng_selfhost/temen-leng.temen`, kept in
// sync with `temen-leng` by that demo's own code-coupling gate). Copy it in (offline-safe); rebuild with
// `bash crates/temen-run/demos/leng_selfhost/build_leng_temen.sh` when `temen-leng`/the encoder changes.
try {
  const lengAsset = join(REPO, 'crates', 'temen-run', 'demos', 'leng_selfhost', 'temen-leng.temen');
  if (!existsSync(lengAsset)) throw new Error('demos/leng_selfhost/temen-leng.temen missing (run build_leng_temen.sh)');
  copyFileSync(lengAsset, join(ASSETS, 'temen-leng.temen'));
  const kb = (readFileSync(lengAsset).length / 1024).toFixed(0);
  console.log(`  ✓ temen-leng.temen (${kb} KB)`);
} catch (e) {
  console.log(`  – temen-leng skipped (${e.message})`);
}

// Shell — the `temen-posix` shell (STAGE1.md, playground-shell). Unlike the clang/on-ramp guests above,
// the shell is compiled by the in-tree **chibicc** onto the POSIX personality and run on the tree-walk
// interpreter (it carries Instantiator call.caps the wasm-JIT/bytecode paths don't take). Its module
// bytes are the committed fixture `tests/fixtures/shell.temen`, produced from the canonical source
// (`crates/temen-run/demos/shell/*.c`) by the differential's generator:
//   cargo test -p temen --test c_shell -- --ignored --exact gen_browser_shell_fixture
// Copy it into web/assets/ (offline-safe, like the committed hello_c.temen); rebuild the fixture with
// the command above when the shell source changes.
try {
  const fixture = join(HERE, 'tests', 'fixtures', 'shell.temen');
  if (!existsSync(fixture)) throw new Error('tests/fixtures/shell.temen missing (run the generator)');
  copyFileSync(fixture, join(ASSETS, 'shell.temen'));
  const kb = (readFileSync(fixture).length / 1024).toFixed(0);
  console.log(`  ✓ shell.temen (${kb} KB)`);
  // The `__stage` ring-filter runner — granted alongside the shell so pipelines take the concurrent
  // ring path (op 11 + SharedRegion + futex). Committed next to shell.temen by the same generator.
  const runner = join(HERE, 'tests', 'fixtures', 'stage_runner.temen');
  if (!existsSync(runner)) throw new Error('tests/fixtures/stage_runner.temen missing (run the generator)');
  copyFileSync(runner, join(ASSETS, 'stage_runner.temen'));
  const rkb = (readFileSync(runner).length / 1024).toFixed(0);
  console.log(`  ✓ stage_runner.temen (${rkb} KB)`);
  // The external commands: `primes` (a generator) and `upper` (a stdin filter) — separate compiled-C
  // programs the shell exec's as op-13 children.
  for (const cmd of ['primes', 'upper']) {
    const p = join(HERE, 'tests', 'fixtures', `${cmd}.temen`);
    if (!existsSync(p)) throw new Error(`tests/fixtures/${cmd}.temen missing (run the generator)`);
    copyFileSync(p, join(ASSETS, `${cmd}.temen`));
    console.log(`  ✓ ${cmd}.temen (${(readFileSync(p).length / 1024).toFixed(0)} KB)`);
  }
} catch (e) {
  console.log(`  – shell skipped (${e.message})`);
}

console.log('done. Assets in web/assets/. Serve with `node serve.mjs` and open /web/play.html');
