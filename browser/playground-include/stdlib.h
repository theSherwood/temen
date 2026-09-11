#ifndef __STDLIB_H
#define __STDLIB_H

// <stdlib.h> for the playground. `malloc` is a **map-growing, thread-safe** bump allocator: it
// claims bytes with a lock-free atomic fetch-add on a bump pointer in the reserved window tail and
// commits fresh host pages on demand with `__vm_map` (the same allocator the frontend's own
// stdlib.h ships, so a threaded lesson's pthread stacks — 256 KiB each — and any large malloc grow
// past the modest initial window instead of hitting a fixed arena cap). `free` is a no-op (MVP:
// no reclamation). No authority beyond the granted Memory capability, all guest C.
#include <__pg_linkage.h>
#include <stdarg.h>

typedef unsigned long size_t;
#define NULL ((void *)0)
#define RAND_MAX 0x7fffffff

void exit(int code);

// The Memory-capability builtins (§3e/§4), lowered to `cap.call` on the granted Memory handle.
// `__vm_map` commits `[off, off+len)` (prot READ|WRITE = 3), returning 0 or a negative errno;
// `__vm_page_size` is the host MMU granularity `map` rounds to. The atomics make the bump pointer
// thread-safe (a single-threaded program pays only an uncontended atomic and never pulls in the
// thread runtime — only `thread.spawn`/`wait`/`notify` mark a module threaded).
long __vm_map(long off, long len, int prot);
long __vm_page_size(void);
long __vm_atomic_add(void *p, long v);                     // fetch-add (i64), returns old
long __vm_atomic_load(void *p);                            // load (i64)
void __vm_atomic_store(void *p, long v);                   // store (i64)
int __vm_atomic_cas32(void *p, int expected, int desired); // CAS (i32), returns old
void __vm_atomic_store32(void *p, int v);                  // store (i32)

#define __PG_HEAP_BASE 268435456L // 256 MiB: above the backed prefix, in the reserved tail
#define __PG_HDR 16L              // per-allocation header (holds the payload size; 16-byte aligned)

typedef struct { int quot, rem; } div_t;
typedef struct { long quot, rem; } ldiv_t;

// ---- prototypes (a program unit, #1392) --------------------------------------------------
// Same split as <stdio.h>: a translation unit compiled decls-only (`-include __pg_decls_only.h`)
// sees these prototypes and links against the prebuilt libc unit that carries the bodies once,
// instead of recompiling the allocator and the string/number conversions into every program. The
// bodies are compiled in by default, where `__PG_FN` makes them `static inline` so an unused one is
// dead-stripped — hence the guard around the prototypes, which would otherwise make them roots.
#ifdef __PG_LIBC_DECLS_ONLY
void *malloc(size_t n);
void free(void *p);
void *calloc(size_t nm, size_t sz);
void *realloc(void *old, size_t n);
void abort(void);
int abs(int x);
long labs(long x);
int atoi(const char *s);
long atol(const char *s);
long strtol(const char *s, char **end, int base);
unsigned long strtoul(const char *s, char **end, int base);
long long strtoll(const char *s, char **end, int base);
unsigned long long strtoull(const char *s, char **end, int base);
long long atoll(const char *s);
long long llabs(long long x);
double strtod(const char *s, char **end);
double atof(const char *s);
long double strtold(const char *s, char **end);
div_t div(int a, int b);
ldiv_t ldiv(long a, long b);
void *bsearch(const void *key, const void *base, size_t n, size_t sz,
              int (*cmp)(const void *, const void *));
char *getenv(const char *name);
int rand(void);
void srand(unsigned s);
void qsort(void *base, size_t n, size_t sz, int (*cmp)(const void *, const void *));
#endif /* __PG_LIBC_DECLS_ONLY */

// ---- bodies -----------------------------------------------------------------------------
// In their own file, not behind an `#ifdef` here: chibicc tokenizes a header in full before the
// preprocessor drops the skipped groups, so text left in place would still be *tokenized* by a
// decls-only compile — which is where nearly all of its remaining time goes (measured: the seeded libc
// bodies are ~90% of a decls-only program's compile). A separate file is never opened at all.
#ifndef __PG_LIBC_DECLS_ONLY
#include <__pg_stdlib_impl.h>
#endif

#endif
