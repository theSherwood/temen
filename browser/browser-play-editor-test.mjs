// Chromium smoke for the playground's per-demo card layout + CodeMirror editor (BROWSER.md §
// playground). Drives the real page: the sidebar lists every demo, each demo is a self-contained card
// (own editor + controls + output), SVM text highlights, a demo runs end-to-end, the editable-module
// stdin path reads its card's editor, parse errors pin the offending line, and Vim mode engages.
//
// Reuses the wasm32 module built by the CI real-browser job (and `serve.mjs` for COOP/COEP). Run:
//   node browser-play-editor-test.mjs
import { startServer } from './serve.mjs';
import { benignAssetMiss } from './play-test-errors.mjs';
import { existsSync, readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

// The Lua/SQLite `.svmb` guests are built by `build-onramp-assets.mjs`, which the CI real-browser job
// doesn't run (only committed assets are present there). So the editable-module stdin check only runs
// when the Lua asset is actually built — otherwise it's SKIPped, not failed.
const HERE = dirname(fileURLToPath(import.meta.url));
const luaBuilt = existsSync(join(HERE, 'web', 'assets', 'lua_snapshot.svmb'));
const chibiccBuilt = existsSync(join(HERE, 'web', 'assets', 'chibicc.svmb'));
// The self-host card needs the committed closure image (`build-selfhost-assets.mjs`); the byte-identity
// check additionally needs the native `chibicc` (built by that same script) as the reference oracle.
const selfhostBuilt = existsSync(join(HERE, 'web', 'assets', 'chibicc_selfhost.img'));
const nativeChibicc = join(HERE, '..', 'frontend', 'chibicc', 'chibicc');

const chromium = (await import('playwright')).chromium;
const { server, port } = await startServer(process.cwd());
const browser = await chromium.launch({ args: ['--no-sandbox'] });
let failed = false;
const ok = (m) => console.log(`  ok: ${m}`);
const fail = (m) => { failed = true; console.log(`  FAIL: ${m}`); };

// A demo card is addressed by its data-demo attribute (the exact EXAMPLES key).
const card = (name) => `[data-demo="${name}"]`;
const runCard = async (page, name, timeout = 20_000) => {
  await page.click(`${card(name)} .run`);
  await page.waitForFunction(
    (sel) => ['done', 'error', 'stopped'].includes(document.querySelector(sel).dataset.state),
    `${card(name)} .state`, { timeout },
  );
};

try {
  const page = await browser.newPage();
  page.on('pageerror', (e) => fail(`pageerror: ${e.message}`));
  page.on('console', (m) => { if (m.type() === 'error' && !benignAssetMiss(m)) fail(`console.error: ${m.text()}`); });
  await page.goto(`http://127.0.0.1:${port}/web/play.html`, { waitUntil: 'load' });
  await page.waitForFunction(
    () => document.getElementById('engine-state').dataset.state === 'ready',
    { timeout: 30_000 },
  );

  // The sidebar lists every demo, and every editable demo mounted a CodeMirror editor; the Vim keymap
  // (a vendored bundle script) actually loaded.
  const layout = await page.evaluate(() => ({
    demos: document.querySelectorAll('main#demos .demo').length,
    navLinks: document.querySelectorAll('#nav-list .nav-link').length,
    editors: document.querySelectorAll('.CodeMirror').length,
    vim: typeof window.CodeMirror?.keyMap?.vim,
  }));
  layout.demos > 0 && layout.navLinks === layout.demos && layout.editors > 0 && layout.vim === 'object'
    ? ok(`${layout.demos} demo cards, ${layout.navLinks} nav links, ${layout.editors} editors, vim keymap`)
    : fail(`layout: ${JSON.stringify(layout)}`);

  // The hello card is SVM text → the custom mode highlights keywords, opcodes, and types.
  const tok = await page.evaluate((sel) => ({
    kw: !!document.querySelector(`${sel} .cm-keyword`),
    bi: !!document.querySelector(`${sel} .cm-builtin`),
    ty: !!document.querySelector(`${sel} .cm-type`),
  }), card('hello'));
  tok.kw && tok.bi && tok.ty ? ok('SVM syntax highlighting active') : fail(`SVM tokens: ${JSON.stringify(tok)}`);

  // Running the hello card reads its editor and completes (its 14-byte greeting length).
  await runCard(page, 'hello');
  const hello = await page.evaluate((sel) => ({
    state: document.querySelector(`${sel} .state`).dataset.state,
    result: document.querySelector(`${sel} .result`).textContent.trim(),
  }), card('hello'));
  hello.state === 'done' && hello.result === '14'
    ? ok('SVM text ran via the editor → 14')
    : fail(`hello run: ${JSON.stringify(hello)}`);

  // The Lua card mounted a Lua-mode editor with the Lua source…
  const lua = await page.evaluate((sel) => {
    const cm = document.querySelector(`${sel} .CodeMirror`)?.CodeMirror;
    return { mode: cm?.getOption('mode'), hasPrint: (cm?.getValue() || '').includes('print(') };
  }, card('Lua (5.4.7 — write & run)'));
  lua.mode === 'lua' && lua.hasPrint ? ok('Lua card → lua mode') : fail(`Lua card: ${JSON.stringify(lua)}`);

  // …and running it feeds the card's editor contents to the guest as stdin (when the asset is built).
  if (luaBuilt) {
    await runCard(page, 'Lua (5.4.7 — write & run)', 30_000);
    const luaOut = await page.evaluate((sel) => document.querySelector(`${sel} .stdout`).textContent,
      card('Lua (5.4.7 — write & run)'));
    luaOut.includes('Hello from Lua') ? ok('editable-module stdin reads the card editor') : fail(`Lua stdout: ${luaOut.slice(0, 80)}`);
  } else {
    console.log('  SKIP: editable-module stdin (lua_snapshot.svmb not built — run build-onramp-assets.mjs)');
  }

  // The "run real Nim" card: runs the committed nim_hello.svmb — a real Nim program
  // (`write(stdout, "hello, svm\n")`) compiled through nimony → svm-leng → the nim→powerbox bridge to a
  // runnable module — and shows its **real stdout**, a Nim program printing on the SVM, client-side.
  // The asset is committed (always present), so no build guard.
  const nimCard = 'nim (Nim → SVM, runs)';
  await runCard(page, nimCard, 30_000);
  const nim = await page.evaluate((sel) => ({
    state: document.querySelector(`${sel} .state`).dataset.state,
    stdout: document.querySelector(`${sel} .stdout`).textContent,
  }), card(nimCard));
  nim.state === 'done' && nim.stdout.includes('hello, svm')
    ? ok('run-real-Nim card: nim_hello.svmb printed its greeting in-browser')
    : fail(`nim run: state=${nim.state} stdout=${nim.stdout.slice(0, 80)}`);

  // The svm-leng self-host card (NIM.md §3e): its editor is pre-filled with a real hexer Leng file, and
  // running it pipes that to the committed `svm-leng.svmb` (always present) on stdin — the translator
  // emits SVM IR text on stdout and exits 0. The IR carries `func`/`block` (svm-text), proving the real
  // leng→SVM-IR translator ran client-side over genuine nimony output.
  const lengCard = 'svm-leng: translate real nimony Leng → SVM IR (self-host)';
  const lengSrc = await page.evaluate((sel) => {
    const cm = document.querySelector(`${sel} .CodeMirror`)?.CodeMirror;
    return (cm?.getValue() || '');
  }, card(lengCard));
  lengSrc.includes('stmts') && lengSrc.includes('wasMoved')
    ? ok('svm-leng card → editor holds the real hexer Leng')
    : fail(`svm-leng editor: ${lengSrc.slice(0, 80)}`);
  await runCard(page, lengCard, 30_000);
  const leng = await page.evaluate((sel) => ({
    state: document.querySelector(`${sel} .state`).dataset.state,
    result: document.querySelector(`${sel} .result`).textContent.trim(),
    stdout: document.querySelector(`${sel} .stdout`).textContent,
  }), card(lengCard));
  leng.state === 'done' && leng.result === '0' && leng.stdout.includes('func') && leng.stdout.includes('block')
    ? ok('svm-leng self-host card: real hexer Leng → SVM IR in-browser')
    : fail(`svm-leng run: state=${leng.state} result=${leng.result} stdout=${leng.stdout.slice(0, 80)}`);

  // The nifler front-end card (NIM.md §3c/§3e slice 4, "compile Nim in the browser"): its editor holds
  // a small Nim program, and running it inflates the committed `nifler.svmb.gz` (the first real nimony
  // phase, always present) and parses that Nim to nimony's NIF — the front edge of the toolchain, run
  // client-side on the SVM. Assert the editor holds Nim and the run emits a parsed `.p.nif`.
  const niflerCard = 'nifler: parse real Nim → NIF (nimony front-end, in your browser)';
  const niflerSrc = await page.evaluate(
    (sel) => document.querySelector(`${sel} .CodeMirror`).CodeMirror.getValue(),
    card(niflerCard),
  );
  niflerSrc.includes('proc fib') && niflerSrc.includes('echo')
    ? ok('nifler card → editor holds a Nim program')
    : fail(`nifler editor: ${niflerSrc.slice(0, 80)}`);
  await runCard(page, niflerCard, 40_000);
  const nifler = await page.evaluate((sel) => ({
    state: document.querySelector(`${sel} .state`).dataset.state,
    result: document.querySelector(`${sel} .result`).textContent.trim(),
    stdout: document.querySelector(`${sel} .stdout`).textContent,
  }), card(niflerCard));
  nifler.state === 'done' && nifler.result.endsWith('B') &&
    nifler.stdout.includes('(.nif') && nifler.stdout.includes('(proc fib')
    ? ok('nifler front-end card: real Nim → parsed NIF in-browser (Nim compiled on the SVM)')
    : fail(`nifler run: state=${nifler.state} result=${nifler.result} stdout=${nifler.stdout.slice(0, 100)}`);

  // The whole-program nim compiler card (NIM.md §3c/§3e; #958) — the toolchain capstone: it inflates
  // the three committed phase guests (`nifler`/`nimsem`/`hexer` `.svmb.gz`) + the stdlib image
  // (`nim_stdlib.img.gz`, all committed → always present) and compiles the editor's whole Nim program
  // **client-side** — the page plays nifmake (stems, `import` crawl), runs nimsem (spawning nifler via
  // an `exec` cap) + hexer over the closure, links through the nim→powerbox bridge, and runs `_start`.
  // Assert the editor holds an I/O Nim program and the run prints the program's real stdout. This is the
  // heaviest card (four assets, multi-phase compile on the tree-walker), so a generous timeout.
  const nimcCard = 'nim: compile & run a whole Nim program → SVM (the full toolchain, in your browser)';
  const nimcSrc = await page.evaluate(
    (sel) => document.querySelector(`${sel} .CodeMirror`).CodeMirror.getValue(),
    card(nimcCard),
  );
  nimcSrc.includes('import std/syncio') && nimcSrc.includes('proc greet') &&
    nimcSrc.includes('write(stdout')
    ? ok('nim compiler card → editor holds a whole Nim program')
    : fail(`nim compiler editor: ${nimcSrc.slice(0, 80)}`);
  await runCard(page, nimcCard, 180_000);
  const nimc = await page.evaluate((sel) => ({
    state: document.querySelector(`${sel} .state`).dataset.state,
    result: document.querySelector(`${sel} .result`).textContent.trim(),
    stdout: document.querySelector(`${sel} .stdout`).textContent,
  }), card(nimcCard));
  nimc.state === 'done' && nimc.result.endsWith('B stdout') &&
    nimc.stdout.includes('hello, Nim') &&
    nimc.stdout.includes('hello, the SVM')
    ? ok('whole-program nim compiler card: the full toolchain compiled + ran a Nim program in-browser')
    : fail(`nimc run: state=${nimc.state} result=${nimc.result} stdout=${nimc.stdout.slice(0, 120)}`);

  // #1005: the compile runs on the snapshot worker's own engine, not the main thread. Confirm the nim
  // worker was spawned (the offload path was taken, not the main-thread fallback).
  const nimOffloaded = await page.evaluate(() =>
    !!(globalThis.__snapshotClient && globalThis.__snapshotClient._workers.has('__nimc__')));
  nimOffloaded
    ? ok('nim compiler card: the compile ran on a Web Worker (page not blocked)')
    : fail('nimc offload: no nim worker spawned — compile ran on the main thread');

  // The real regression guard for #1005: while a compile is in flight the **main thread stays
  // responsive**. Start a re-run without awaiting it, then round-trip a trivial `page.evaluate` through
  // the page's main-thread event loop — if the compile were synchronous on the main thread (the old bug)
  // this would block for the whole multi-minute compile; via the worker it returns immediately.
  await page.evaluate((sel) => document.querySelector(`${sel} .CodeMirror`).CodeMirror.setValue(
    'import std/syncio\nwrite(stdout, "edited: recompiled in the browser\\n")\n'), card(nimcCard));
  await page.click(`${card(nimcCard)} .run`);
  // Wait until the compile is genuinely in flight (state 'running'), then time a main-thread round-trip.
  await page.waitForFunction(
    (sel) => document.querySelector(sel).dataset.state === 'running',
    `${card(nimcCard)} .state`, { timeout: 30_000 });
  const probeStart = Date.now();
  await page.evaluate(() => 1 + 1); // resolves on the page's main-thread task queue
  const probeMs = Date.now() - probeStart;
  probeMs < 5_000
    ? ok(`nim compiler card: main thread responsive during compile (${probeMs}ms round-trip)`)
    : fail(`nimc responsiveness: main-thread round-trip took ${probeMs}ms (compile blocking the UI thread?)`);
  await page.waitForFunction(
    (sel) => ['done', 'error', 'stopped'].includes(document.querySelector(sel).dataset.state),
    `${card(nimcCard)} .state`, { timeout: 180_000 });
  const nimc2 = await page.evaluate((sel) => ({
    state: document.querySelector(`${sel} .state`).dataset.state,
    stdout: document.querySelector(`${sel} .stdout`).textContent,
  }), card(nimcCard));
  nimc2.state === 'done' && nimc2.stdout.includes('edited: recompiled in the browser')
    ? ok('nim compiler card: editing the Nim recompiles + reruns it (the edit prints)')
    : fail(`nimc edit re-run: state=${nimc2.state} stdout=${nimc2.stdout.slice(0, 120)}`);

  // The QuickJS card is wired to the warm-runtime snapshot (WASM_AOT.md): it defaults to the warm path
  // (svm_warm_open runs the QuickJS runtime init once, svm_warm_eval restores that image + evals per Run).
  // Its qjs_snapshot.svmb is committed (always present), so no build guard is needed.
  {
    const qjsName = 'JavaScript (QuickJS — write & run JS)';
    const msOf = (t) => { const m = /· (\d+)ms/.exec(t); return m ? Number(m[1]) : NaN; };
    const readQ = () => page.evaluate((sel) => ({
      state: document.querySelector(`${sel} .state`).dataset.state,
      msg: document.querySelector(`${sel} .state`).textContent,
      stdout: document.querySelector(`${sel} .stdout`).textContent,
    }), card(qjsName));
    const setQ = (src) => page.evaluate(({ sel, src }) =>
      document.querySelector(`${sel} .CodeMirror`).CodeMirror.setValue(src), { sel: card(qjsName), src });
    // 1) First Run of the sample program — warms the runtime once; tier reports warm-snapshot.
    await runCard(page, qjsName, 30_000);
    const q1 = await readQ();
    q1.state === 'done' && q1.msg.includes('warm-snapshot')
      && q1.stdout.includes('fib(0..10): 0 1 1 2 3 5 8 13 21 34 55') && q1.stdout.includes('sorted: 1,2,3,5,7,8,9')
      ? ok('QuickJS card runs on the warm-runtime snapshot → correct output')
      : fail(`qjs first run: ${JSON.stringify({ state: q1.state, msg: q1.msg, out: q1.stdout.slice(0, 80) })}`);
    // 2) Second Run — warm session reused: byte-identical output. The runtime rebuild is paid once, but
    // since the card pre-warms on the snapshot worker at page load (issue #804), the FIRST Run is already
    // warm too — so this asserts output stability across Runs, not a first-vs-second timing gap (that gap
    // moved into the worker's pre-warm; warm-snapshot-test measures it directly). `msOf` kept for logging.
    await runCard(page, qjsName, 30_000);
    const q2 = await readQ();
    q2.state === 'done' && q2.stdout === q1.stdout
      ? ok(`QuickJS warm reuse: byte-identical across Runs (${msOf(q1.msg)}ms → ${msOf(q2.msg)}ms)`)
      : fail(`qjs warm reuse: ${JSON.stringify({ q1: msOf(q1.msg), q2: msOf(q2.msg), same: q2.stdout === q1.stdout })}`);
    // 3) Fresh-per-Run isolation: a global defined in one Run must NOT survive into the next.
    await setQ('var leaked = 42; typeof leaked;\n');
    await runCard(page, qjsName, 30_000);
    const qd = (await readQ()).stdout;
    await setQ('typeof leaked;\n');
    await runCard(page, qjsName, 30_000);
    const qa = (await readQ()).stdout;
    qd.trim().endsWith('number') && qa.trim().endsWith('undefined')
      ? ok('QuickJS fresh-per-Run isolation: a global from one Run does not leak into the next')
      : fail(`qjs isolation: define=${JSON.stringify(qd.slice(-20))} after=${JSON.stringify(qa.slice(-20))}`);
  }

  // The C-compiler card mounted a C-mode editor.
  const cc = await page.evaluate((sel) => document.querySelector(`${sel} .CodeMirror`)?.CodeMirror?.getOption('mode'),
    card('C compiler (chibicc → SVM — compile & run)'));
  cc === 'text/x-csrc' ? ok('chibicc card → C mode') : fail(`chibicc mode: ${cc}`);

  // …and running it compiles the editor's C with chibicc.svmb IN THE BROWSER, svm_parse-es the emitted
  // IR, runs the result, and shows main()'s return value. A trivial program pins an exact expected
  // value; the stdout pane must carry the emitted SVM IR. (Skipped when the asset isn't built.)
  if (chibiccBuilt) {
    const ccName = 'C compiler (chibicc → SVM — compile & run)';
    await page.evaluate((sel) => document.querySelector(`${sel} .CodeMirror`).CodeMirror.setValue(
      'int main(void){ int x = 7 * 6; return x; }'), card(ccName));
    await runCard(page, ccName, 30_000);
    const cco = await page.evaluate((sel) => ({
      state: document.querySelector(`${sel} .state`).dataset.state,
      msg: document.querySelector(`${sel} .state`).textContent,
      result: document.querySelector(`${sel} .result`).textContent,
      ir: document.querySelector(`${sel} .stdout`).textContent,
    }), card(ccName));
    // The card's wasm-JIT toggle defaults on, so this exercises chibicc's compile on the emitted-wasm
    // tier (the `.state` message reports `(wasm-JIT)`, so a silent interpreter fallback would fail here).
    cco.state === 'done' && cco.result === '42' && cco.ir.includes('func') && cco.ir.includes('_start')
    && cco.msg.includes('wasm-JIT')
      ? ok('chibicc compiled C → SVM IR → ran it → 42 (in-browser, wasm-JIT)')
      : fail(`chibicc run: ${JSON.stringify({ state: cco.state, msg: cco.msg, result: cco.result, ir: cco.ir.slice(0, 60) })}`);

    // "Prove interp ≡ JIT": compile the same C on both tiers and assert the emitted IR is byte-identical.
    await page.click(`${card(ccName)} button.prove`);
    await page.waitForFunction(
      (sel) => ['done', 'error'].includes(document.querySelector(sel).dataset.state),
      `${card(ccName)} .state`, { timeout: 60_000 },
    );
    const ccp = await page.evaluate((sel) => ({
      state: document.querySelector(`${sel} .state`).dataset.state,
      msg: document.querySelector(`${sel} .state`).textContent,
    }), card(ccName));
    ccp.state === 'done' && ccp.msg.includes('byte-identical')
      ? ok('chibicc interpreter ≡ wasm-JIT — byte-identical emitted IR (in-browser)')
      : fail(`chibicc parity: ${JSON.stringify(ccp)}`);

    // #include + printf: the seeded <stdio.h> makes a text-emitting program actually print (its
    // output shows in the stdout pane, above the emitted IR) instead of trapping on an unresolved call.
    // Includes a %f/%g line — guest-C float formatting compiled into the program.
    await page.evaluate((sel) => document.querySelector(`${sel} .CodeMirror`).CodeMirror.setValue(
      '#include <stdio.h>\nint main(void){ for(int i=1;i<=3;i++) printf("i=%d\\n", i); printf("pi=%.2f e=%g\\n", 3.14159, 2.5); return 0; }'),
      card(ccName));
    await runCard(page, ccName, 30_000);
    const pf = await page.evaluate((sel) => ({
      state: document.querySelector(`${sel} .state`).dataset.state,
      out: document.querySelector(`${sel} .stdout`).textContent,
    }), card(ccName));
    pf.state === 'done' && pf.out.startsWith('i=1\ni=2\ni=3\npi=3.14 e=2.5\n') && pf.out.includes('SVM IR')
      ? ok('chibicc #include <stdio.h> + printf (incl. %f/%g floats) → real output in-browser')
      : fail(`chibicc printf: ${JSON.stringify({ state: pf.state, out: pf.out.slice(0, 90) })}`);

    // The expanded libc (SELFHOST_C.md §7 "larger libc"): <math.h> algebra + <assert.h> (which needs
    // the now-wired predefined __FILE__/__LINE__ macros) + <string.h>/<stdlib.h> additions, in-browser.
    await page.evaluate((sel) => document.querySelector(`${sel} .CodeMirror`).CodeMirror.setValue(
      '#include <stdio.h>\n#include <math.h>\n#include <string.h>\n#include <assert.h>\n' +
      'int main(void){ assert(sizeof(int)==4); char*d=strdup("libc");\n' +
      'printf("%s sqrt=%g pow=%g\\n", d, sqrt(169.0), pow(2.0,10.0)); return 0; }'),
      card(ccName));
    await runCard(page, ccName, 30_000);
    const lc = await page.evaluate((sel) => ({
      state: document.querySelector(`${sel} .state`).dataset.state,
      out: document.querySelector(`${sel} .stdout`).textContent,
    }), card(ccName));
    lc.state === 'done' && lc.out.startsWith('libc sqrt=13 pow=1024\n')
      ? ok('chibicc expanded libc (<math.h> + <assert.h> + strdup) → real output in-browser')
      : fail(`chibicc libc: ${JSON.stringify({ state: lc.state, out: lc.out.slice(0, 90) })}`);

    // Multi-file project (SELFHOST_C.md §7 stage-2 lever): `//// file: NAME` markers split the editor
    // into memfs files, and /in.c #includes a sibling header + .c (unity build) — all compiled in-browser.
    await page.evaluate((sel) => document.querySelector(`${sel} .CodeMirror`).CodeMirror.setValue(
      '#include <stdio.h>\n#include "mathx.h"\n#include "mathx.c"\n' +
      'int main(void){ printf("gcd=%d\\n", gcd(48, 36)); return 0; }\n' +
      '//// file: mathx.h\nint gcd(int, int);\n' +
      '//// file: mathx.c\nint gcd(int a, int b){ while(b){ int t=a%b; a=b; b=t; } return a; }\n'),
      card(ccName));
    await runCard(page, ccName, 30_000);
    const mf = await page.evaluate((sel) => ({
      state: document.querySelector(`${sel} .state`).dataset.state,
      out: document.querySelector(`${sel} .stdout`).textContent,
    }), card(ccName));
    mf.state === 'done' && mf.out.startsWith('gcd=12\n')
      ? ok('chibicc multi-file project (//// file: markers → #include sibling .h/.c) → ran in-browser')
      : fail(`chibicc multifile: ${JSON.stringify({ state: mf.state, out: mf.out.slice(0, 90) })}`);

    // open_memstream + buffered FILE* (SELFHOST_C.md §7 stage-2): the <stdio.h> upgrade that lets
    // chibicc's own format()-style code compile and run — a memory FILE built up with fprintf, read back.
    await page.evaluate((sel) => document.querySelector(`${sel} .CodeMirror`).CodeMirror.setValue(
      '#include <stdio.h>\nint main(void){ char*b; size_t n; FILE*m=open_memstream(&b,&n);\n' +
      'for(int i=0;i<3;i++) fprintf(m,"[%d]", i*i); fclose(m);\n' +
      'printf("%s len=%lu\\n", b, (unsigned long)n); return 0; }'),
      card(ccName));
    await runCard(page, ccName, 30_000);
    const ms = await page.evaluate((sel) => ({
      state: document.querySelector(`${sel} .state`).dataset.state,
      out: document.querySelector(`${sel} .stdout`).textContent,
    }), card(ccName));
    ms.state === 'done' && ms.out.startsWith('[0][1][4] len=9\n')
      ? ok('chibicc open_memstream + buffered FILE* (the self-host stdio) → ran in-browser')
      : fail(`chibicc memstream: ${JSON.stringify({ state: ms.state, out: ms.out.slice(0, 90) })}`);
  } else {
    console.log('  SKIP: chibicc compile-and-run (chibicc.svmb not built — run build-onramp-assets.mjs)');
  }

  // The self-host capstone card (SELFHOST_C.md §7 step 5): chibicc.svmb compiles chibicc's *own* cc1
  // TUs to linkable objects, in-browser, on the wasm-JIT. Pins: (1) it runs and emits a real object;
  // (2) the object is byte-identical to a native `chibicc --emit-object` (the fixpoint, over the real
  // glibc header closure seeded from the committed image); (3) the interpreter and JIT tiers agree.
  if (selfhostBuilt) {
    const shName = 'chibicc compiles its own source (self-host → SVM)';
    // Pick a substantial TU (tokenize.c, ~800 lines) via the card's translation-unit dropdown.
    const tuRel = 'frontend/chibicc/tokenize.c';
    await page.evaluate(([sel, tu]) => {
      const s = document.querySelector(`${sel} select`);
      s.value = tu;
      s.dispatchEvent(new Event('change'));
    }, [card(shName), tuRel]);
    await runCard(page, shName, 60_000);
    const sh = await page.evaluate((sel) => ({
      state: document.querySelector(`${sel} .state`).dataset.state,
      msg: document.querySelector(`${sel} .state`).textContent,
      result: document.querySelector(`${sel} .result`).textContent,
      pane: document.querySelector(`${sel} .stdout`).textContent,
    }), card(shName));
    // The pane is a one-line header (`──── chibicc compiled its own tokenize.c → … ────`) then the object.
    const guestObj = sh.pane.slice(sh.pane.indexOf('\n') + 1);
    const wellFormed = sh.state === 'done' && guestObj.includes('func') && guestObj.includes('export')
      && sh.msg.includes('wasm-JIT');
    wellFormed
      ? ok('self-host: chibicc compiled its own tokenize.c → linkable SVM-IR object in-browser (wasm-JIT)')
      : fail(`self-host run: ${JSON.stringify({ state: sh.state, msg: sh.msg, pane: sh.pane.slice(0, 80) })}`);

    // Byte-identity to native chibicc — the fixpoint, enforced. The native binary is the reference
    // oracle (built by build-selfhost-assets.mjs); same relative flags as `chibicc_selfhost_argv`
    // (no --data-page — the object is canonical). Skipped if the native binary isn't present.
    if (wellFormed && existsSync(nativeChibicc)) {
      const REPO = join(HERE, '..');
      const prelude = 'crates/svm-run/demos/chibicc_selfhost/selfhost_prelude.h';
      const refOut = join(REPO, 'target', 'selfhost_playtest_tokenize.svm');
      try {
        execFileSync(nativeChibicc, [
          '-cc1', '-include', prelude, '-Ifrontend/chibicc', '-Ifrontend/chibicc/include',
          '-I/usr/include/x86_64-linux-gnu', '-I/usr/include', '--emit-object',
          '-cc1-input', tuRel, '-cc1-output', refOut, tuRel,
        ], { cwd: REPO });
        const nativeObj = readFileSync(refOut, 'utf8');
        guestObj === nativeObj
          ? ok(`self-host: in-browser object byte-identical to native chibicc (${nativeObj.length} B)`)
          : fail(`self-host byte-identity: guest ${guestObj.length} B vs native ${nativeObj.length} B differ`);
      } catch (e) {
        console.log(`  SKIP: self-host byte-identity (native chibicc reference failed: ${e.message})`);
      }
    } else if (wellFormed) {
      console.log('  SKIP: self-host byte-identity (native frontend/chibicc/chibicc not built)');
    }

    // "Prove interp ≡ JIT": recompile the same TU on both engines and assert the objects are identical.
    await page.click(`${card(shName)} button.prove`);
    await page.waitForFunction(
      (sel) => ['done', 'error'].includes(document.querySelector(sel).dataset.state),
      `${card(shName)} .state`, { timeout: 90_000 },
    );
    const shp = await page.evaluate((sel) => ({
      state: document.querySelector(`${sel} .state`).dataset.state,
      msg: document.querySelector(`${sel} .state`).textContent,
    }), card(shName));
    shp.state === 'done' && shp.msg.includes('byte-identical')
      ? ok('self-host interpreter ≡ wasm-JIT — byte-identical object (in-browser)')
      : fail(`self-host parity: ${JSON.stringify(shp)}`);
  } else {
    console.log('  SKIP: chibicc self-host (chibicc_selfhost.img not built — run build-selfhost-assets.mjs)');
  }

  // The SQL card mounted a SQL-mode editor.
  const sqlMode = await page.evaluate((sel) => document.querySelector(`${sel} .CodeMirror`)?.CodeMirror?.getOption('mode'),
    card('SQLite (:memory: — write & run SQL)'));
  sqlMode === 'text/x-sql' ? ok('SQL card → sql mode') : fail(`SQL mode: ${sqlMode}`);

  // A parse error pins the offending line in that card's editor: a bad opcode on line 3 (unique token).
  await page.evaluate((sel) => document.querySelector(`${sel} .CodeMirror`).CodeMirror.setValue(
    'func () -> (i64) {\nblock 0 () {\n  v0 = i64.notanopcode 1\n  return v0\n  }\n}'), card('hello'));
  await runCard(page, 'hello');
  const mark = await page.evaluate((sel) => {
    const cm = document.querySelector(`${sel} .CodeMirror`).CodeMirror;
    const info = cm.lineInfo(2); // 0-based line 2 = the bad-opcode line
    return {
      gutter: !!(info.gutterMarkers && info.gutterMarkers['svm-error-gutter']),
      lineClass: (info.bgClass || '').includes('cm-error-line'),
      widget: !!document.querySelector(`${sel} .cm-error-widget`),
    };
  }, card('hello'));
  (mark.gutter && mark.lineClass && mark.widget)
    ? ok('parse error pinned to the right line (gutter + line + inline message)')
    : fail(`error decoration: ${JSON.stringify(mark)}`);
  // Editing clears the decoration.
  await page.evaluate((sel) => document.querySelector(`${sel} .CodeMirror`).CodeMirror.setValue(
    'func () -> (i64) {\nblock 0 () {\n  v0 = i64.const 1\n  return v0\n  }\n}'), card('hello'));
  const cleared = await page.evaluate((sel) => !document.querySelector(`${sel} .cm-error-widget`), card('hello'));
  cleared ? ok('error decoration clears on edit') : fail('error decoration not cleared on edit');

  // Phase 3/4: every JIT-emittable demo exposes the wasm-JIT toggle + a "Prove it" button — both the
  // interactive reactors (per-frame tick) and the run-to-completion modules (whole _start). Running the
  // parity check confirms the interpreter and wasm-JIT tiers are byte-identical.
  const jitCards = await page.evaluate(() =>
    [...document.querySelectorAll('.demo')].filter((d) => d.querySelector('.jit-label')).map((d) => d.dataset.demo));
  const hasReactorJit = jitCards.includes('bounce (interactive — arrow keys)')
    && jitCards.includes('life (Conway — heap persistence)');
  const hasModuleJit = jitCards.includes('hello (C → SVM)')
    && jitCards.includes('SQLite (:memory: — write & run SQL)');
  const hasChibiccJit = jitCards.includes('C compiler (chibicc → SVM — compile & run)');
  hasReactorJit && hasModuleJit && hasChibiccJit
    ? ok(`wasm-JIT toggle on ${jitCards.length} demos (reactors + hello/Lua/SQLite/chibicc modules)`)
    : fail(`jit cards: ${JSON.stringify(jitCards)}`);

  // The hello module card runs end-to-end via runModule (JIT toggle default-on): this exercises the
  // streamed module fetch (download-progress path) and the single-shot module JIT in CI, since
  // hello_c.svmb is committed. Runs before the module parity check so the asset streams fresh (uncached).
  await runCard(page, 'hello (C → SVM)');
  const helloMod = await page.evaluate((sel) => ({
    state: document.querySelector(`${sel} .state`).dataset.state,
    stdout: document.querySelector(`${sel} .stdout`).textContent,
  }), card('hello (C → SVM)'));
  helloMod.state === 'done' && helloMod.stdout.length > 0
    ? ok(`hello module ran end-to-end (${JSON.stringify(helloMod.stdout.trim().slice(0, 20))})`)
    : fail(`hello module run: ${JSON.stringify(helloMod)}`);

  // Prove interp ≡ JIT on the bounce reactor (committed asset, fast) — framebuffer byte-identical.
  await page.click(`${card('bounce (interactive — arrow keys)')} .prove`);
  await page.waitForFunction(
    (sel) => ['done', 'error'].includes(document.querySelector(sel).dataset.state),
    `${card('bounce (interactive — arrow keys)')} .state`, { timeout: 30_000 });
  const parity = await page.evaluate((sel) => document.querySelector(sel).textContent,
    `${card('bounce (interactive — arrow keys)')} .state`);
  parity.includes('interpreter ≡ wasm-JIT') && parity.includes('byte-identical')
    ? ok(`reactor parity proven in-page: ${parity}`)
    : fail(`parity: ${parity}`);

  // Prove interp ≡ JIT on the hello module (committed asset): the whole _start runs on both tiers and
  // the captured stdout is byte-identical (the module twin of the reactor's per-frame parity).
  await page.click(`${card('hello (C → SVM)')} .prove`);
  await page.waitForFunction(
    (sel) => ['done', 'error'].includes(document.querySelector(sel).dataset.state),
    `${card('hello (C → SVM)')} .state`, { timeout: 30_000 });
  const modParity = await page.evaluate((sel) => document.querySelector(sel).textContent,
    `${card('hello (C → SVM)')} .state`);
  modParity.includes('interpreter ≡ wasm-JIT') && modParity.includes('byte-identical stdout')
    ? ok(`module parity proven in-page: ${modParity}`)
    : fail(`module parity: ${modParity}`);

  // Touch dpad: every reactor card carries an on-screen dpad (4 arrows + fire/use/enter/esc) so the
  // interactive guests are playable without a physical keyboard. Structural check (CSS gates visibility
  // to touch/narrow screens); pressing a key while no reactor runs is a guarded no-op.
  const dpad = await page.evaluate((sel) => {
    const d = document.querySelector(`${sel} .dpad`);
    return { present: !!d, keys: d ? d.querySelectorAll('.dkey').length : 0 };
  }, card('bounce (interactive — arrow keys)'));
  dpad.present && dpad.keys === 8
    ? ok(`touch dpad on reactor cards (${dpad.keys} keys)`) : fail(`dpad: ${JSON.stringify(dpad)}`);

  // The Vim toggle engages the Vim keymap on the editors (registered + editor holds vim state).
  await page.check('#vim');
  const vim = await page.evaluate((sel) => {
    const cm = document.querySelector(`${sel} .CodeMirror`)?.CodeMirror;
    return { opt: cm?.getOption('keyMap'), state: !!cm?.state?.vim };
  }, card('hello'));
  vim.opt === 'vim' && vim.state ? ok('vim mode engaged') : fail(`vim: ${JSON.stringify(vim)}`);

  // Phase 4: an edit persists under the card slug and survives a reload; Reset restores the default and
  // clears storage; a Share permalink round-trips the editor contents through the URL hash.
  const sel = card('hello');
  const setCM = (s, v) => page.evaluate(([s, v]) => document.querySelector(`${s} .CodeMirror`).CodeMirror.setValue(v), [s, v]);
  const getCM = (s) => page.evaluate((s) => document.querySelector(`${s} .CodeMirror`).CodeMirror.getValue(), s);
  const waitReady = () => page.waitForFunction(
    () => document.getElementById('engine-state').dataset.state === 'ready', { timeout: 30_000 });

  await setCM(sel, 'PERSIST_SENTINEL');
  const saved = await page.evaluate(() => localStorage.getItem('svm-play:src:hello'));
  saved === 'PERSIST_SENTINEL' ? ok('edit persisted to localStorage') : fail(`persist: ${saved}`);
  await page.reload({ waitUntil: 'load' });
  await waitReady();
  (await getCM(sel)) === 'PERSIST_SENTINEL'
    ? ok('editor restored from localStorage after reload') : fail('editor not restored after reload');
  await page.click(`${sel} .reset`);
  const afterReset = await page.evaluate((s) => ({
    val: document.querySelector(`${s} .CodeMirror`).CodeMirror.getValue(),
    stored: localStorage.getItem('svm-play:src:hello'),
  }), sel);
  (afterReset.val !== 'PERSIST_SENTINEL' && afterReset.val.includes('cap.call') && afterReset.stored === null)
    ? ok('Reset restores the default source and clears storage')
    : fail(`reset: ${JSON.stringify({ v: afterReset.val.slice(0, 30), s: afterReset.stored })}`);

  // Share: the button emits a permalink; navigating to it (with storage cleared) restores the source
  // purely from the `#demo=…&src=…` hash.
  await setCM(sel, 'SHARED_ROUNDTRIP_42');
  await page.click(`${sel} .share`);
  await page.waitForFunction((s) => /#demo=/.test(document.querySelector(`${s} .log`).textContent), sel, { timeout: 5_000 });
  const shareURL = await page.evaluate((s) => {
    const m = document.querySelector(`${s} .log`).textContent.match(/https?:\/\/\S+#demo=\S+/);
    return m ? m[0] : null;
  }, sel);
  if (shareURL && shareURL.includes('demo=hello')) {
    await page.evaluate(() => localStorage.removeItem('svm-play:src:hello')); // prove the hash, not storage
    await page.goto(shareURL, { waitUntil: 'load' });
    await waitReady();
    (await getCM(sel)) === 'SHARED_ROUNDTRIP_42'
      ? ok('share permalink round-trips the editor via the URL hash') : fail('share permalink did not restore');
  } else {
    fail(`share URL not emitted: ${shareURL}`);
  }

  // The DAP debugger card: a breakpoint is pre-placed, Debug pauses on the bytecode engine at the
  // source line (highlighted), the Variables pane shows the loop locals, and Continue advances the loop.
  const dbgCard = card('Debugger (SVM — breakpoints, step, variables)');
  const dbg0 = await page.evaluate((sel) => ({
    hasDebugBtn: !!document.querySelector(`${sel} .debug`),
    bpDots: document.querySelectorAll(`${sel} .cm-bp-marker`).length,
  }), dbgCard);
  dbg0.hasDebugBtn && dbg0.bpDots === 1
    ? ok(`debug card: Debug button + ${dbg0.bpDots} pre-placed breakpoint`)
    : fail(`debug card: ${JSON.stringify(dbg0)}`);

  // Start debugging → it runs to the breakpoint and pauses.
  await page.click(`${dbgCard} .debug`);
  await page.waitForFunction((sel) => document.querySelector(`${sel} .state`).textContent.includes('paused'),
    dbgCard, { timeout: 20_000 });
  const paused = await page.evaluate((sel) => ({
    active: document.querySelector(`${sel} .dbg`).classList.contains('active'),
    stopLine: !!document.querySelector(`${sel} .cm-stop-line`),
    vars: document.querySelector(`${sel} .dbg-vars`).textContent,
    readonly: document.querySelector(`${sel} .CodeMirror`).CodeMirror.getOption('readOnly'),
  }), dbgCard);
  paused.active && paused.stopLine && /i\s*=\s*5/.test(paused.vars) && /acc\s*=\s*0/.test(paused.vars) && paused.readonly
    ? ok(`debugger paused at the breakpoint — i=5, acc=0, line highlighted`)
    : fail(`debug paused: ${JSON.stringify(paused)}`);

  // Continue once → next loop iteration: acc accumulates (5), i decrements (4).
  await page.click(`${dbgCard} .dbg-controls button[data-cmd="continue"]`);
  await page.waitForFunction((sel) => /acc\s*=\s*5/.test(document.querySelector(`${sel} .dbg-vars`).textContent),
    dbgCard, { timeout: 10_000 });
  const stepped = await page.evaluate((sel) => document.querySelector(`${sel} .dbg-vars`).textContent, dbgCard);
  /i\s*=\s*4/.test(stepped) && /acc\s*=\s*5/.test(stepped)
    ? ok('Continue advanced the loop — i=4, acc=5')
    : fail(`debug continue: ${stepped}`);

  // Reverse: run backward to the previous breakpoint hit (deterministic replay) — the locals rewind.
  await page.click(`${dbgCard} .dbg-controls button[data-cmd="reverseContinue"]`);
  await page.waitForFunction((sel) => /i\s*=\s*5/.test(document.querySelector(`${sel} .dbg-vars`).textContent),
    dbgCard, { timeout: 10_000 });
  const reversed = await page.evaluate((sel) => document.querySelector(`${sel} .dbg-vars`).textContent, dbgCard);
  /i\s*=\s*5/.test(reversed) && /acc\s*=\s*0/.test(reversed)
    ? ok('Reverse walked back to the previous breakpoint — i=5, acc=0')
    : fail(`reverse: ${reversed}`);

  // Stop ends the session: the panel hides and the editor is writable again.
  await page.click(`${dbgCard} .dbg-controls button[data-cmd="stop"]`);
  const ended = await page.evaluate((sel) => ({
    active: document.querySelector(`${sel} .dbg`).classList.contains('active'),
    stopLine: !!document.querySelector(`${sel} .cm-stop-line`),
    readonly: document.querySelector(`${sel} .CodeMirror`).CodeMirror.getOption('readOnly'),
  }), dbgCard);
  !ended.active && !ended.stopLine && !ended.readonly
    ? ok('Stop ended the debug session (panel hidden, editor writable)')
    : fail(`debug stop: ${JSON.stringify(ended)}`);

  // The chibicc source-level debug card: chibicc compiles the C **with -g** in the browser, and the DAP
  // debugger runs the emitted IR — stopping on a **C source line** with the **C locals named** (i, acc).
  // A whole C program debugged at source level, client-side. (Compute-only: no printf → no powerbox.)
  if (chibiccBuilt) {
    const ccDbg = card('C source-level debugging (chibicc → SVM — breakpoints on C lines)');
    const ccDbg0 = await page.evaluate((sel) => ({
      debugEnabled: !!document.querySelector(`${sel} .debug:not([disabled])`),
      gChecked: document.querySelector(`${sel} input[type=checkbox]`)?.checked === true, // the -g toggle (gOn)
      bpDots: document.querySelectorAll(`${sel} .cm-bp-marker`).length,
    }), ccDbg);
    ccDbg0.debugEnabled && ccDbg0.gChecked && ccDbg0.bpDots === 1
      ? ok('chibicc debug card: -g on, Debug enabled, breakpoint pre-placed')
      : fail(`chibicc debug card: ${JSON.stringify(ccDbg0)}`);

    // Debug → compile with -g (a beat) → run to the C-line breakpoint and pause with the C locals shown.
    await page.click(`${ccDbg} .debug`);
    await page.waitForFunction((sel) => document.querySelector(`${sel} .state`).textContent.includes('paused'),
      ccDbg, { timeout: 30_000 });
    const ccPaused = await page.evaluate((sel) => ({
      vars: document.querySelector(`${sel} .dbg-vars`).textContent,
      stopLine: !!document.querySelector(`${sel} .cm-stop-line`),
    }), ccDbg);
    /\bi\b/.test(ccPaused.vars) && /\bacc\b/.test(ccPaused.vars) && ccPaused.stopLine
      ? ok('chibicc: debugged C at source level — stopped on a C line, C locals i/acc named')
      : fail(`chibicc debug paused: ${JSON.stringify(ccPaused)}`);

    // Step Over (next) advances forward a source line and stays in the program (the paused frame is still
    // `main`) — exercising the forward-stepping button end-to-end. Non-disruptive: it lands on the printf
    // line, so the Continue below still prints "i=3, acc=3".
    await page.click(`${ccDbg} .dbg-controls button[data-cmd="next"]`);
    await page.waitForFunction((sel) => document.querySelector(`${sel} .state`).textContent.includes('paused'),
      ccDbg, { timeout: 15_000 });
    /\bmain\b/.test(await page.evaluate((sel) => document.querySelector(`${sel} .dbg-vars`).textContent, ccDbg))
      ? ok('chibicc: Step Over advanced a source line, staying in main')
      : fail('chibicc Step Over: frame left main');

    // Continue once: the loop body runs a `printf`, so the guest's output streams into the stdout pane
    // under the on-ramp I/O powerbox (deny-all would trap the `write`). It stops at the next iteration.
    await page.click(`${ccDbg} .dbg-controls button[data-cmd="continue"]`);
    await page.waitForFunction((sel) => /i=3, acc=3/.test(document.querySelector(`${sel} .stdout`).textContent),
      ccDbg, { timeout: 15_000 });
    ok('chibicc: printf output captured under the powerbox while debugging');

    // Reverse back to the earlier breakpoint: the run is rebuilt + replayed, so the captured output
    // **rewinds** — the first printf hasn't run yet, so the pane no longer shows "i=3".
    await page.click(`${ccDbg} .dbg-controls button[data-cmd="reverseContinue"]`);
    await page.waitForFunction((sel) => !/i=3/.test(document.querySelector(`${sel} .stdout`).textContent),
      ccDbg, { timeout: 15_000 });
    ok('chibicc: reverse debugging rewound the captured output');

    // Step Back is **depth-aware** — it rewinds within `main`, not down into the guest libc `printf`
    // internals. The Variables pane header shows the paused frame's function; before the fix a step-back
    // from a line that called `printf` descended into `__pf_flush`/stdio.h.
    await page.click(`${ccDbg} .dbg-controls button[data-cmd="stepBack"]`);
    await page.waitForFunction((sel) => /main/.test(document.querySelector(`${sel} .dbg-vars`).textContent),
      ccDbg, { timeout: 15_000 });
    ok('chibicc: Step Back stayed in main (not the libc printf internals)');

    await page.click(`${ccDbg} .dbg-controls button[data-cmd="stop"]`);
  }

  // The watchpoint card: a counter at a fixed window address, named `count` by its `debug` section, so
  // the Variables pane can arm a data breakpoint on it. Debug pauses at the pre-placed loop-body
  // breakpoint; clicking `count`'s ● toggle arms the watch; Continue then stops for the data breakpoint.
  const wpCard = card('Debugger (SVM — watchpoints / data breakpoints)');
  await page.click(`${wpCard} .debug`);
  await page.waitForFunction((sel) => document.querySelector(`${sel} .state`).textContent.includes('paused'),
    wpCard, { timeout: 20_000 });
  const wpPaused = await page.evaluate((sel) => ({
    vars: document.querySelector(`${sel} .dbg-vars`).textContent,
    // `count` is memory-located ⇒ its ● toggle is enabled (a watchable data breakpoint target).
    toggleEnabled: !!document.querySelector(`${sel} .dbg-vars button[data-watch="count"]:not([disabled])`),
  }), wpCard);
  wpPaused.toggleEnabled && /count\s*=\s*0/.test(wpPaused.vars)
    ? ok('watchpoint card paused — count=0, ● toggle armable')
    : fail(`watchpoint paused: ${JSON.stringify(wpPaused)}`);

  // Arm the data breakpoint on `count`, then Continue → the loop-body store trips it (reason "data
  // breakpoint"), and the ● shows armed (.on).
  await page.click(`${wpCard} .dbg-vars button[data-watch="count"]`);
  const armed = await page.evaluate((sel) =>
    !!document.querySelector(`${sel} .dbg-vars button[data-watch="count"].on`), wpCard);
  armed ? ok('clicking ● armed the data breakpoint on count') : fail('watch toggle did not arm (.on)');
  await page.click(`${wpCard} .dbg-controls button[data-cmd="continue"]`);
  await page.waitForFunction((sel) => /data breakpoint/.test(document.querySelector(`${sel} .state`).textContent),
    wpCard, { timeout: 10_000 });
  const tripped = await page.evaluate((sel) => document.querySelector(`${sel} .state`).textContent, wpCard);
  ok(`watchpoint tripped — ${tripped.replace(/\s+/g, ' ').trim()}`);
  await page.click(`${wpCard} .dbg-controls button[data-cmd="stop"]`);

  // The threads card: a thread.spawn guest on the multithreaded scheduled bytecode engine. Debug stops
  // in a worker; the Variables pane grows a thread selector (one chip per live vCPU); selecting another
  // thread focuses its stack without resuming; Continue catches the second worker; the guest finishes.
  const thCard = card('Debugger (SVM — threads)');
  await page.click(`${thCard} .debug`);
  await page.waitForFunction((sel) => /paused .*thread-/.test(document.querySelector(`${sel} .state`).textContent),
    thCard, { timeout: 20_000 });
  const thPaused = await page.evaluate((sel) => {
    const chips = [...document.querySelectorAll(`${sel} .dbg-threads .thr`)];
    return {
      count: chips.length,
      selected: chips.filter((b) => b.classList.contains('sel')).map((b) => b.dataset.thread),
      marked: chips.filter((b) => b.textContent.includes('●')).map((b) => b.dataset.thread),
      vars: document.querySelector(`${sel} .dbg-vars`).textContent,
    };
  }, thCard);
  thPaused.count >= 3 && thPaused.selected.length === 1 && thPaused.selected[0] === thPaused.marked[0]
    ? ok(`threads card paused in a worker — ${thPaused.count} thread chips, stopped one selected + marked ●`)
    : fail(`threads paused: ${JSON.stringify(thPaused)}`);

  // Focus a *different* thread (not the stopped one) → its chip becomes selected, without resuming.
  const otherThread = await page.evaluate((sel) => {
    const chips = [...document.querySelectorAll(`${sel} .dbg-threads .thr`)];
    const stopped = chips.find((b) => b.classList.contains('sel'))?.dataset.thread;
    return chips.map((b) => b.dataset.thread).find((t) => t !== stopped);
  }, thCard);
  await page.click(`${thCard} .dbg-threads .thr[data-thread="${otherThread}"]`);
  const switched = await page.evaluate((sel) =>
    document.querySelector(`${sel} .dbg-threads .thr.sel`)?.dataset.thread, thCard);
  switched === otherThread
    ? ok(`selecting another thread (${otherThread}) focuses its stack without resuming`)
    : fail(`thread switch: selected ${switched}, wanted ${otherThread}`);

  // Continue → the second worker hits the same breakpoint (a distinct thread), still paused.
  await page.click(`${thCard} .dbg-controls button[data-cmd="continue"]`);
  await page.waitForFunction((sel) => /paused .*thread-/.test(document.querySelector(`${sel} .state`).textContent),
    thCard, { timeout: 10_000 });
  const secondThread = await page.evaluate((sel) =>
    document.querySelector(`${sel} .dbg-threads .thr.sel`)?.dataset.thread, thCard);
  ok(`threads card: Continue caught the second worker (thread ${secondThread})`);

  // ◀◀ Reverse → deterministic replay walks *backward* to the previous worker breakpoint (an earlier
  // global turn) — the scheduled engine's reverse debugging, in the panel.
  await page.click(`${thCard} .dbg-controls button[data-cmd="reverseContinue"]`);
  await page.waitForFunction((sel) => /paused .*thread-/.test(document.querySelector(`${sel} .state`).textContent),
    thCard, { timeout: 10_000 });
  const reversedThread = await page.evaluate((sel) =>
    document.querySelector(`${sel} .dbg-threads .thr.sel`)?.dataset.thread, thCard);
  reversedThread && reversedThread !== secondThread
    ? ok(`threads card: Reverse walked back to the earlier worker (thread ${reversedThread})`)
    : fail(`threads reverse: landed on ${reversedThread}, expected the earlier worker (not ${secondThread})`);
  await page.click(`${thCard} .dbg-controls button[data-cmd="stop"]`);

  // The wait/notify card: a futex handoff. The worker parks on atomic.wait until the root's notify
  // wakes it; a breakpoint after the wait fires only once woken — proving wait/notify drive under the
  // debug scheduler. Then Continue finishes the handoff.
  const wnCard = card('Debugger (SVM — wait / notify)');
  await page.click(`${wnCard} .debug`);
  await page.waitForFunction((sel) => /paused .*thread-/.test(document.querySelector(`${sel} .state`).textContent),
    wnCard, { timeout: 20_000 });
  const wnStopped = await page.evaluate((sel) =>
    document.querySelector(`${sel} .dbg-threads .thr.sel`)?.dataset.thread, wnCard);
  wnStopped && wnStopped !== '1'
    ? ok(`wait/notify card: the worker woke and stopped after the wait (thread ${wnStopped})`)
    : fail(`wait/notify paused: selected ${wnStopped}, expected a worker (not the root, 1)`);
  await page.click(`${wnCard} .dbg-controls button[data-cmd="continue"]`);
  await page.waitForFunction((sel) => /finished/.test(document.querySelector(`${sel} .state`).textContent),
    wnCard, { timeout: 10_000 });
  ok('wait/notify card: the handoff finished after resuming');

  // The fibers card: a generator. A breakpoint inside the fiber fires only once cont.resume switches
  // the debugged continuation into it — the debugger follows into the fiber and highlights its line;
  // Continue runs the suspend/resume handoff to completion.
  const fbCard = card('Debugger (SVM — fibers / generators)');
  await page.click(`${fbCard} .debug`);
  await page.waitForFunction((sel) => document.querySelector(`${sel} .state`).textContent.includes('paused'),
    fbCard, { timeout: 20_000 });
  const fbPaused = await page.evaluate((sel) => ({
    stopLine: !!document.querySelector(`${sel} .cm-stop-line`),
    // The stop is inside the fiber body (line 19) — the frame header names the line.
    vars: document.querySelector(`${sel} .dbg-vars`).textContent,
  }), fbCard);
  fbPaused.stopLine && /line 19\b/.test(fbPaused.vars)
    ? ok('fibers card: cont.resume stepped into the fiber — stopped inside it (line 19)')
    : fail(`fibers paused: ${JSON.stringify(fbPaused)}`);
  await page.click(`${fbCard} .dbg-controls button[data-cmd="continue"]`);
  await page.waitForFunction((sel) => /finished/.test(document.querySelector(`${sel} .state`).textContent),
    fbCard, { timeout: 10_000 });
  ok('fibers card: the generator finished (36) after resuming');

  // The fibers+threads card: two workers each run a generator fiber. A breakpoint inside the fiber body
  // fires on a *worker* vCPU (a thread selector appears; the stopped chip is not the root), proving fibers
  // compose with threads under the scheduled debugger. Continue catches the other worker; the run finishes.
  const ftCard = card('Debugger (SVM — fibers + threads)');
  await page.click(`${ftCard} .debug`);
  await page.waitForFunction((sel) => /paused .*thread-/.test(document.querySelector(`${sel} .state`).textContent),
    ftCard, { timeout: 20_000 });
  const ftPaused = await page.evaluate((sel) => {
    const chips = [...document.querySelectorAll(`${sel} .dbg-threads .thr`)];
    const stopped = chips.find((b) => b.classList.contains('sel'))?.dataset.thread;
    return {
      count: chips.length,
      stopped,
      stopLine: !!document.querySelector(`${sel} .cm-stop-line`),
      vars: document.querySelector(`${sel} .dbg-vars`).textContent,
    };
  }, ftCard);
  // ≥3 live chips (root + two workers), the stop is inside the fiber (line 37), and the stopped vCPU is a
  // worker (thread id ≠ 1, the root).
  ftPaused.count >= 3 && ftPaused.stopLine && /line 37\b/.test(ftPaused.vars) && ftPaused.stopped !== '1'
    ? ok(`fibers+threads card: a worker (thread ${ftPaused.stopped}) stopped inside its fiber (line 37)`)
    : fail(`fibers+threads paused: ${JSON.stringify(ftPaused)}`);
  await page.click(`${ftCard} .dbg-controls button[data-cmd="continue"]`);
  await page.waitForFunction((sel) => /paused .*thread-/.test(document.querySelector(`${sel} .state`).textContent),
    ftCard, { timeout: 10_000 });
  const ftSecond = await page.evaluate((sel) =>
    document.querySelector(`${sel} .dbg-threads .thr.sel`)?.dataset.thread, ftCard);
  ftSecond !== ftPaused.stopped
    ? ok(`fibers+threads card: Continue caught the other worker's fiber (thread ${ftSecond})`)
    : fail(`fibers+threads second stop: same thread ${ftSecond}`);
  await page.click(`${ftCard} .dbg-controls button[data-cmd="continue"]`);
  await page.waitForFunction((sel) => /finished/.test(document.querySelector(`${sel} .state`).textContent),
    ftCard, { timeout: 10_000 });
  ok('fibers+threads card: the run finished (50) after both workers');

  // Theme picker: selecting "dark" forces <html data-theme="dark"> and persists; a reload keeps it.
  await page.selectOption('#theme', 'dark');
  const themed = await page.evaluate(() => ({
    attr: document.documentElement.dataset.theme,
    stored: localStorage.getItem('svm-play:theme'),
  }));
  themed.attr === 'dark' && themed.stored === 'dark'
    ? ok('theme picker forces + persists dark') : fail(`theme: ${JSON.stringify(themed)}`);
  await page.reload({ waitUntil: 'load' });
  await waitReady();
  const themeAfter = await page.evaluate(() => ({
    attr: document.documentElement.dataset.theme,
    sel: document.getElementById('theme').value,
  }));
  themeAfter.attr === 'dark' && themeAfter.sel === 'dark'
    ? ok('theme preference survives a reload (no flash — set in <head>)') : fail(`theme reload: ${JSON.stringify(themeAfter)}`);
} finally {
  await browser.close();
  server.close();
}

if (failed) {
  console.log('\nplay editor smoke FAILED');
  process.exit(1);
}
console.log('\nplay editor smoke passed');
