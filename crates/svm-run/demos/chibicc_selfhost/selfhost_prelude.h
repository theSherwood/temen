/* Self-host prelude for the chibicc `--emit-object` / `--emit-ir` path (SELFHOST_C.md, task #20).
 *
 * chibicc's own parser trips over a handful of declarations in modern glibc's `<stdlib.h>`, leaving
 * those functions *implicitly declared* at their call sites — a hard error. The cause is glibc 2.38+'s
 * ISO C23 `strto*` rewrite: when the compiler does not define glibc's internal `__REDIRECT` macro
 * (chibicc doesn't), `<stdlib.h>` takes the branch that declares `__isoc23_strtoul` (et al.) behind
 * `__THROW __nonnull((1))` attribute forms and then `#define strtoul __isoc23_strtoul`. chibicc's
 * preprocessor/parser doesn't fully resolve that path, so `strtoul`/`strtol`/`atoi`/… never get a
 * visible prototype. (The LLVM on-ramp self-host build never hits this: clang defines `__REDIRECT` and
 * parses the redirected forms natively. It bites only when *chibicc itself* compiles chibicc.)
 *
 * The fix is deliberately minimal and self-contained: force-include this header (`-include`) ahead of
 * the system headers so each affected function has a plain, standard prototype visible at its call
 * site. These are the exact C89/C99 signatures; they redeclare compatibly with (the parts chibicc does
 * see of) `<stdlib.h>`, so no TU that already compiled is disturbed. Only functions chibicc actually
 * fails to pick up are listed — `strtod`/`malloc`/`free`/… parse fine from glibc and are left alone.
 *
 * This is a *compile-time* shim for chibicc's frontend, not a runtime libc: the definitions live in the
 * emit-object libc (increment 2b). Used by the `c_link.rs` self-host tests and, going forward, the
 * `build_chibicc_svmb.sh` emit-object build.
 */
#ifndef SVM_SELFHOST_PRELUDE_H
#define SVM_SELFHOST_PRELUDE_H

/* Shared allocator state for the multi-TU emit-object build (SELFHOST_C.md §7, 2c). The bundled
 * `<stdlib.h>` bump allocator keeps its free-pointer in file-scope state; in a whole-program build
 * that is a self-contained `static`, but here every cc1 TU `#include`s `<stdlib.h>` separately and
 * would mint its *own* `static` bump pointer at the same 256 MiB heap base — so the per-TU allocators
 * would hand out overlapping addresses and corrupt each other at run time. Defining `__SVM_LIBC_EXTERN`
 * (this prelude is force-included in every emit-object TU) makes that state a single **shared** symbol:
 * each TU sees an `extern`, and the libc unit (`emit_libc.c`, which sets `__SVM_LIBC_OWNER`) holds the
 * one definition — resolved cross-TU by `svm_ir::link` exactly like chibicc's shared `ty_int`. */
#define __SVM_LIBC_EXTERN 1

/* The ISO C23 `strto*` family glibc hides behind its `__isoc23_*` redirect (the core gap). */
unsigned long strtoul(const char *, char **, int);
long strtol(const char *, char **, int);
long long strtoll(const char *, char **, int);
unsigned long long strtoull(const char *, char **, int);

/* `ato*` — plain prototypes glibc decorates with `__wur`/`__nonnull`/`__extern_inline` forms that
 * chibicc skips, so the base declaration never lands. */
int atoi(const char *);
long atol(const char *);

/* `strtold`: chibicc parses `long double` float literals with it (tokenize.c). Under the self-host
 * build's `-mlong-double-64`, `long double == double`, so the emit-object libc forwards it to `strtod`. */
long double strtold(const char *, char **);

#endif /* SVM_SELFHOST_PRELUDE_H */
