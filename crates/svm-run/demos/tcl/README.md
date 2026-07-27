# Tcl — a scripting-language interpreter on the LLVM on-ramp

The reference **Tcl 8.6.14** interpreter (Tcl/BSD license) driven through the LLVM→SVM-IR
on-ramp: the whole language core — the bytecode compiler + execution engine, the `expr`
engine, `string`/`list`/`dict`, Henry Spencer's regex, `Tcl_Obj` value model, namespaces,
TclOO, and libtommath bignums — compiled to bitcode, translated, verified, and run in the
sandbox, byte-identical to the same sources built natively with `cc`.

Tcl joins the same genre as the `../quickjs`, `../lua` (see `LLVM.md`), and `../sqlite`
ports: a self-contained C interpreter for a scripting language, reached with **no new VM
capabilities** — a frontend + libc-waist + REPL-driver + playground-registration job. It is a
**big lift**, tracked as an in-progress target (like QuickJS was), not a landed capstone.

## Files

- `tcl_repl.c` — the browser-playground REPL driver. Reads a Tcl script from **stdin**
  (the `Stream` capability), evaluates the whole buffer in one interpreter, and prints the
  completion result (or error). Uses the **minimal embedding** — `Tcl_CreateInterp` +
  `Tcl_Eval`, deliberately **no `Tcl_Init`** — so the language core runs with **no filesystem
  and no ambient OS surface** (the direct analog of QuickJS's `qjs_eval.c`). `Tcl_CreateInterp`
  registers the whole language and nearly all built-in commands in C; only `init.tcl`-backed
  conveniences (`auto_load`, `unknown`, `parray`, `file`/`glob` niceties) need the script
  library, which the minimal REPL skips. **Validated against native `libtcl8.6.a`** — see below.
- `tcl_shim.c` — the Tcl-specific OS/libc waist the on-ramp neither synthesizes, resolves to
  an svm-posix capability, nor covers via a reused shim: the `clock` time surface (deterministic
  stubs), the tty/termios probes Tcl's channel layer runs at startup, locale, `strtol`/`strtoul`,
  case-insensitive compares, and the socket/process/extra-fs surface the minimal REPL never
  reaches (stubbed to clean Tcl errors, never escapes).
- `build_bitcode.sh` — the fetch → configure → native-oracle → per-TU-bitcode → `llvm-link` →
  translate pipeline the test automates (fetched-not-vendored, skips cleanly offline). Reuses the
  Makefile's own compile flags per TU, so the build stays faithful to a real Tcl.

The Tcl sources are **not vendored** (fetched + cached at build time from the SourceForge
mirror). The core object set is the 162 members of `libtcl8.6.a` (Tcl generic + unix + Spencer
regex + TclOO + libtommath).

## The libc/OS waist — the reuse map

`POSIX.md`'s split (authority → capabilities, pure compute → guest code) says exactly where each
of Tcl's 192 external symbols goes. **Nothing here is baked into the TCB.**

| Surface | Count | Source |
|---|---|---|
| **mem/string/alloc/qsort + `llvm.*` intrinsics** | ~50 | on-ramp-synthesized (slices N/O/X) |
| **posix caps** — `open`/`read`/`write`/`close`/`lseek`/`stat`/`opendir`/`readdir`/`getcwd`/`chdir`/`getenv`/`setenv`/`unlink`/`exit` | 18 | **svm-posix** ops 0–20, resolved at load (POSIX.md) |
| **printf/scanf/ctype** — the runtime-`va_list` `vsnprintf` family, `__isoc99_sscanf`, `__ctype_*` | ~15 | reused `../postgres/{printf,scanf}_shim.c` + ctype tables |
| **`strtod`** | 1 | reused `../strtod/strtod.c` |
| **libm transcendentals** — `sin`/`cos`/`pow`/`sqrt`/`atan2`/… | 20 | guest **openlibm** (the QuickJS slice CO mechanism; `OPENLIBM_DIR`) |
| **zlib** — `deflate`/`inflate`/`crc32`/… (the `zlib`/`binary`/zipfs commands) | 15 | trap-stubbed (uncalled on the eval path) or a guest zlib — see gap list |
| **Tcl OS waist** — `clock` time, tty/termios, locale, `strtol`/`strtoul`, sockets/exec/extra-fs | ~70 | **`tcl_shim.c`** (this dir): real where cheap, clean-error stubs where out of MVP scope |

## Progress — the gap-walk record

The build is **validated** through link; the on-ramp translate walks the fail-closed chokepoint
one gap at a time (the Postgres/QuickJS workflow). Status:

**DONE — native oracle + minimal embedding proven.** `tcl_repl.c` linked against the native
`libtcl8.6.a` runs the whole language core with **no `Tcl_Init` and no filesystem** —
`set`/`expr`/`proc`+recursion/`lsort`/`for`/`foreach`/`format`/`dict`/`regexp`/`string`/`**`
all correct (`puts` works too, via Tcl's stdout channel over the `write` cap). This is the
`svm-run` differential oracle.

**DONE — faithful whole-program bitcode.** `build_bitcode.sh` configures Tcl
(`--disable-shared --disable-threads --disable-load`), builds the native oracle, and compiles all
**162 core TUs** to LLVM IR with the Makefile's own flags, `llvm-link`-ing them + the driver +
`tcl_shim.c` + the reused printf/strtod shims into one ~19.6 MB textual module that the on-ramp
ingests.

**NEXT (translate gap #1) — constexpr `icmp` / `select`.** The first fail-closed stop is a
constant-expression comparison the on-ramp's LLVM constant parser doesn't yet evaluate:

```llvm
; tclInterp.c, DeleteScriptLimitCallback wrapper — a function-pointer-vs-sentinel compare
; LLVM left unfolded as a constexpr operand of an instruction-level `select`:
%39 = select i1 icmp eq (ptr inttoptr (i64 3 to ptr), ptr @DeleteScriptLimitCallback),
             ptr @WrapFree, ptr @DeleteScriptLimitCallback
```

`src/ll/parse.rs`'s `constant()` handles constexpr conversions/binops/`getelementptr`/aggregates
but not `icmp`/`fcmp`/`select`. Closing it means: add the `Constant::ICmp`/`Select` AST variants
+ parse arms, and lower them at the constant-operand site in `lib.rs` (emit the compare as real IR,
since a global-address operand is a relocation, not a compile-time literal). This is a general
on-ramp improvement, not Tcl-specific. Expect further gaps after it (the QuickJS port closed ~a
dozen); the `demo_tcl_repl_stdin` test is `#[ignore]`d until the chain clears.

**THEN — resolve-stage waist + byte-identical output.** Once translate clears, the load stage
reports the undefined-symbol set; grow `tcl_shim.c` / stage openlibm / trap-stub uncalled zlib
until it resolves, then diff stdout against the native oracle over a language-breadth script.

## Follow-ups (beyond the minimal REPL)

- **Full `Tcl_Init`** — seed the Tcl script `library/` + encodings into the svm-posix memfs (the
  same seeding `../sqlite`/chibicc use), point `TCL_LIBRARY` at it, and enable `auto_load`,
  `unknown`, real `file`/`glob`, and `clock` over a host time cap.
- **wasm-JIT tier** — prove `_start` is wasm-JIT-emittable (`browser-jit-module-test`) so the
  whole interpreter runs on emitted wasm in the playground (near-native), like Lua/SQLite/QuickJS.

## Running by hand

```sh
# fetch + configure + native oracle + per-TU bitcode + link + translate (first gap prints)
./build_bitcode.sh                       # → /tmp/svm_tcl_cache/tcl_linked.ll

# native oracle by hand (the minimal-embedding REPL against real libtcl):
cd /tmp/svm_tcl_cache/tcl8.6.14
cc -O2 -Igeneric -Iunix path/to/tcl_repl.c unix/libtcl8.6.a -lz -lm -o /tmp/tcl_repl_native
printf 'set n 10\nproc f {x} {expr {$x<2?$x:[f [expr $x-1]]+[f [expr $x-2]]}}\nputs [f $n]\n' \
  | /tmp/tcl_repl_native            # → 55

# translate → verify → run in the sandbox (once the gaps above close)
cd ../../crates/svm-llvm && cargo build --release --example try_translate
./target/release/examples/try_translate /tmp/svm_tcl_cache/tcl_linked.ll
```
