# MicroPython — Python on the LLVM on-ramp

**MicroPython 1.24.1** driven through the LLVM→TEMEN-IR on-ramp: the language core — the compiler
(lexer + parser + bytecode emitter), the VM, the object model, the GC, and `int`/`float`/`str`/
`list`/`dict`/`tuple`/comprehensions/closures/exceptions — compiled to bitcode, translated, verified,
and run in the sandbox, **byte-identical to the same sources built natively with `cc`**.

MicroPython joins the same genre as `../quickjs`, `../tcl`, and the Lua/SQLite ports: a self-contained
C interpreter for a scripting language, reached with **no new VM capabilities** — a frontend +
libc-waist + REPL-driver + playground-registration job. It is the first Python on Temen; the plan is
`PYTHON.md`, this is slice A (#1326).

## It runs

`build_bitcode.sh` translates ~729 functions, verifies them, and runs this program byte-for-byte
against the native oracle:

```python
print('hello, temen!')
print([x*x for x in range(6)])
d={'a':1,'b':2}; print(sum(d.values()))
def fib(n):
    return n if n<2 else fib(n-1)+fib(n-2)
print([fib(i) for i in range(10)])
print('float:', 3.0/2, round(2**0.5, 6))
try:
    1/0
except Exception as e:
    print('caught', repr(e))
```
```
hello, temen!
[0, 1, 4, 9, 16, 25]
3
[0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
float: 1.5 1.414214
caught ZeroDivisionError('divide by zero',)
```

Validated end-to-end by `demo_micropython_repl_stdin` (`crates/temen-llvm/tests/translate.rs`):
translate → verify → run under the powerbox with the program on stdin, asserting stdout byte-matches
the native `cc` driver. `#[ignore]`d for wall-clock (clones + builds MicroPython, runs a whole
interpreter); run with `--ignored`. Skips loudly when git/clang/make are unavailable.

## Files

- `micropython.c` — the browser-playground REPL driver. Reads a Python program from **stdin** (the
  `Stream` capability), runs it in one `ports/embed` interpreter (`mp_embed_init` / `mp_embed_exec_str`
  / `mp_embed_deinit`), prints via the HAL override. The minimal embedding — no filesystem, no ambient
  OS surface (the direct analog of `qjs_eval.c` / `tcl_repl.c`).
- `micropython_snapshot.c` — the two-phase warm-runtime-snapshot driver (`warmup` + `eval_run`), so the
  playground pays interpreter bring-up once and evaluates per Run over a restored snapshot (the
  QuickJS/Tcl `*_snapshot.c` contract; slice B/C).
- `mpconfigport.h` — the Temen embed config. Every switch is a supported MicroPython feature toggle, so
  **no MicroPython source is patched** — it is how the gap-walk below is closed.
- `py_shim.c` — the libc/HAL waist the on-ramp neither synthesizes nor resolves to a capability: the
  stdout HAL override (→ `Stream` write) and the string/mem helpers the on-ramp doesn't provide.
- `build_bitcode.sh` — the clone → embed-package → per-TU bitcode → openlibm → llvm-link → translate
  pipeline (fetched-not-vendored; skips cleanly offline). Also builds the native `cc` oracle.

MicroPython is **not vendored** (MIT; cloned + cached at build time via `git`, because GitHub's tarball
archive is proxy-gated in CI). The build uses MicroPython's own `ports/embed` package generator, so it
stays faithful to a real MicroPython.

## The gap-walk (all closed by config — no source patching)

Pushing the linked module through the fail-closed translator one gap at a time (the QuickJS method):

| # | Gap the translator surfaced | Cause | Fix (config only) |
|---|---|---|---|
| 1 | `printf: dynamic precision (.*)` in `mp_hal_stdout_tx_strn_cooked` | the stock HAL does `printf("%.*s", len, str)`; the on-ramp's inline printf lowering rejects dynamic precision | `py_shim.c` overrides the HAL to `write(1, str, len)` (Stream cap); drop `port/mphalport.c` from the link |
| 2 | `type half (Milestone 1+)` in `mp_binary_get_val` | native `_Float16` for the array/struct `'e'` typecode is outside the f64/f32 scope | `MICROPY_FLOAT_USE_NATIVE_FLT16 = 0` — the pure-integer software half-float codec (keeps real floats) |
| 3 | `inline asm (unrecognized template) … jmp nlr_push_tail` in `nlr_push` | x86-64 hand-written asm for non-local returns (and the GC register scan) | `MICROPY_NLR_SETJMP = 1` + `MICROPY_GCREGS_SETJMP = 1` — routes both through setjmp/longjmp, which Temen lowers to its core `SetJmp`/`LongJmp` ops |
| 4 | runtime `Unreachable` in the `mp_embed_exec_str` path | the compile/exec path reaches float-math libm symbols (`fmod`/`nan`/…) that were unresolved | `llvm-link` guest **openlibm** (the QuickJS "address-taken libm" mechanism), step `[4/6]` |

`setjmp`/`longjmp`/`_setjmp` themselves need no shim — the on-ramp lowers them to the `SetJmp`/`LongJmp`
core ops (`temen-llvm` `lower_setjmp_call`). `memcpy`/`memmove`/`memset` are on-ramp-synthesized; the
rest of the string waist (`strlen`/`strcmp`/`memcmp`/`memchr`/`strchr`/`bcmp`) is in `py_shim.c`.

## Reproduce

```sh
# needs: git, clang, llvm-link, cc, make, python3
(cd crates/temen-llvm && cargo build --release --example try_translate)   # once
bash crates/temen-run/demos/micropython/build_bitcode.sh                  # → $CACHE/mp_linked.ll + native oracle
cargo test -p temen-llvm --test translate demo_micropython -- --ignored --nocapture
```
Artifacts land in `$TEMEN_MICROPYTHON_CACHE` (default `/tmp/temen_micropython_cache`).

## Where this is going (PYTHON.md)

- **Slice B (#1327):** freeze the MicroPython stdlib in + the warm-snapshot driver (`micropython_snapshot.c`).
- **Slice C (#1329) — the capstone:** a MicroPython REPL card in the browser playground
  (`browser/build-onramp-assets.mjs` + a `play.js` card, `--host-page 65536`).
- **Config breadth:** this build is `MICROPY_CONFIG_ROM_LEVEL_MINIMUM` + compiler + float. Raising the
  ROM level (more builtins, `array`/`struct`, more of the stdlib) is straightforward follow-on, each new
  feature walked through the same fail-closed translator.
