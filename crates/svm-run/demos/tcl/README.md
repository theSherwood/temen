# Tcl — a scripting-language interpreter on the LLVM on-ramp

The reference **Tcl 8.6.14** interpreter (Tcl/BSD license) driven through the LLVM→SVM-IR
on-ramp: the whole language core — the bytecode compiler + execution engine, the `expr`
engine, `string`/`list`/`dict`, Henry Spencer's regex, `Tcl_Obj` value model, namespaces,
TclOO, and libtommath bignums — compiled to bitcode, translated, verified, and run in the
sandbox, byte-identical to the same sources built natively with `cc`.

Tcl joins the same genre as the `../quickjs`, `../lua` (see `LLVM.md`), and `../sqlite`
ports: a self-contained C interpreter for a scripting language, reached with **no new VM
capabilities** — a frontend + libc-waist + REPL-driver + playground-registration job. **It runs,
byte-identical to native** (see "It runs" below); getting there closed three general on-ramp
translator gaps.

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

The build is **validated** through link, and Tcl now **translates (2669 funcs) and verifies**. The
walk (Postgres/QuickJS workflow) so far:

**DONE — native oracle + minimal embedding proven.** `tcl_repl.c` linked against the native
`libtcl8.6.a` runs the whole language core with **no `Tcl_Init` and no filesystem** —
`set`/`expr`/`proc`+recursion/`lsort`/`for`/`foreach`/`format`/`dict`/`regexp`/`string`/`**`
all correct (`puts` works too, via Tcl's stdout channel over the `write` cap). This is the
`svm-run` differential oracle.

**DONE — faithful whole-program bitcode.** `build_bitcode.sh` configures Tcl
(`--disable-shared --disable-threads --disable-load`), builds the native oracle, and compiles all
**162 core TUs** to LLVM IR with the Makefile's own flags, `llvm-link`-ing them + the driver +
`tcl_shim.c` + openlibm + the reused printf/strtod shims into one ~19.9 MB textual module.

**DONE — three on-ramp translator gaps closed (general fixes, not Tcl-specific):**
1. **Constexpr `icmp`** — `tclInterp.c`'s limit-callback wrapper emits
   `select i1 icmp eq (ptr inttoptr (i64 3 to ptr), ptr @DeleteScriptLimitCallback), …` (a
   function-address-vs-sentinel compare LLVM left unfolded). Added the `Constant::ICmp` AST variant
   + parser arm, lowered at the constant-operand site as a runtime `IntCmp` (a global address is a
   relocation, not a compile-time literal). Test `constexpr_icmp_operand` (interp ≡ JIT).
2. **Vector `ptrtoint`** — `FinalizeOONextFilter` (TclOO) packs a pointer pair:
   `load <2 x ptr>` → `ptrtoint <2 x ptr> to <2 x i64>` → `trunc → <2 x i32>`. Pointers are i64
   lanes, so the ptrtoint is a representational identity on the packed v128. Test
   `vector_ptrtoint_identity` (interp ≡ JIT).
3. **Vector `inttoptr`** — the inverse identity, added symmetrically.

**DONE — the libc/OS waist.** `tcl_shim.c` now provides qsort/bsearch (heapsort), the string/mem
ops the on-ramp doesn't synthesize *and* the address-taken ones (`&strcmp`/`&strlen` stored as
comparators / encoding `lengthProc`), the glibc ctype tables (reused from `../postgres`), strerror,
and the full time/tty/locale/socket/process/file surface — all as **defined** functions (benign
errors where out of MVP scope), so nothing faults on a signature or a missing body. Address-taken
libm resolves via linked openlibm. Genuinely-unreached leftovers (zlib, `__isoc99_sscanf`, `fts`)
are trap-stubbed via `SVM_STUB_EXTERNS` — never called on the eval path.

**DONE — first execution.** The last runtime blocker was `zlibVersion()`, which `TclZlibInit`
(called from `Tcl_CreateInterp`) reads for its package config — a trap-stub that faulted at startup.
With the zlib surface given benign bodies (and the file/tty/locale surface likewise), the module
translates, verifies, and **runs**.

## ★ It runs — byte-identical to native

The Tcl 8.6 core (2669 functions) **translates, verifies, and executes** a script piped in on stdin,
with stdout byte-identical to the native `cc` build (`demo_tcl_repl_stdin`, `#[ignore]`d only for
wall-clock — a whole interpreter on the tree-walker takes tens of seconds):

```
fib: 0 1 1 2 3 5 8 13 21 34
sorted: 1 2 3 5 7 8 9
pi ~ 3.1416, 255 = 0xFF, sqrt2 = 1.414214       ← libtommath expr + openlibm sqrt
dict: a 1 b 2
TCL ON SVM
```

Recursion, `lsort`, `format` (`%.4f`/`%X`), `dict`, `string`, `expr` (`**` + the transcendental
`sqrt` through linked openlibm) — all correct. Wired into the browser playground:
`build-onramp-assets.mjs` builds `tcl_repl.svmb` (`--stub-externs`, 64 KiB pages) and `web/play.js`
registers it as a "Tcl (8.6 — write & run)" example. Boot is milliseconds.

## Full `Tcl_Init` — the whole standard library (`tcl_init.c`)

**DONE.** `tcl_init.c` is a second driver that runs a *complete* Tcl: it registers an **in-guest
`Tcl_Filesystem` VFS** that serves the Tcl script library from **embedded byte arrays**
(`tcl_library.h`, generated by `gen_tcl_library.py`), points `tcl_library` at the VFS mount, and calls
`Tcl_Init`. That sources `init.tcl` and unlocks `auto_load`/`unknown`/`package`, the `clock`/`parray`/
`history` script commands, and real `file`/`glob` — all **byte-identical to native** (`demo_tcl_init_stdin`):

```
Tcl 8.6.14
clock:  2001-09-09 01:46:40         ← clock.tcl + auto-loaded msgcat, from the embedded VFS
lib:    auto.tcl clock.tcl history.tcl init.tcl package.tcl parray.tcl safe.tcl tm.tcl word.tcl
file:   /a/b/c  ext=.tcl  root=y
dict:   a 1 b 2
regexp: user@host user host
fib:    1 1 2 3 5 8 13 21
```

**Why the VFS, not svm-posix.** The VFS callbacks fill Tcl's *own* `Tcl_StatBuf` in C, so there is
**no libc `struct stat` ABI gap** (svm-posix's `stat` writes a minimal `{mode,size}`, which Tcl's
glibc-layout `struct stat` would misread); it needs **no filesystem capability** (the scripts live in
the guest binary); and it runs identically native, on svm, and in the browser. This is how embedded
Tcl applications ship their library. `build_bitcode.sh` links both variants (`tcl_linked.ll` +
`tcl_init_linked.ll`); the playground card uses the full-init `tcl_init.svmb`.

*Scope of the embedded library:* `init.tcl` + `tclIndex` + the auto-loaded core scripts + `msgcat`
(~270 KB). Timezone data (`tzdata/`, ~900 files) is excluded — UTC `clock` works without it;
local-zone conversion + `encoding` files are the remaining nicety.

## Follow-up

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
