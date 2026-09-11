#ifndef __STDIO_H
#define __STDIO_H

// A small, self-contained <stdio.h> for the browser playground's C compiler (SELFHOST_C.md §7).
// The program the user writes is compiled by chibicc-the-guest and run under the fixed powerbox,
// which provides ambient `write`/`read`/`exit` (§3e). Everything here is guest C compiled *into*
// the program — no libc is linked — so `printf` is just formatting over `write(1, …)`. Integer,
// char, string, and pointer conversions plus `%f`/`%e`/`%g` (guest-C float formatting, correctly
// rounded to the requested precision — not a shortest-round-trip bignum dtoa, so `%.17g` and a few
// exact-tie roundings can differ from glibc; see the float helpers below) are all supported.

#include <__pg_linkage.h>
#include <stdarg.h>
#include <stdlib.h> // malloc/realloc — the growable buffer behind open_memstream

// Ambient powerbox syscalls (the frontend lowers these to `cap.call` on the stashed handles).
int write(int fd, char *buf, long n);
int read(int fd, char *buf, long n);
void exit(int code);

// #1323 (c_interpret #16, file I/O): file access over the powerbox `fs` capability. `__vm_fs(op,…)`
// is the on-ramp fs seam recognized by the frontend (it lowers to `call.sym "vm_fs"`, with the op
// selected by arg0), so one call carries the whole open/read/write/seek/close protocol — the same
// `temen-fs` backend Postgres/chibicc use. The on-ramp mounts a private, in-memory read-write memfs
// (`grant_onramp_caps`). fd 0/1/2 stay on the ambient Stream (`write`/`read` above); *file* fds
// (>= 3, minted by FS_OPEN) reach the memfs. A plain compile-and-run card with no fs cap simply
// never calls these (a program that only writes stdout goes through `write`).
extern long __vm_fs(long op, long a, long b, long c, long d);
enum { __FS_OPEN = 0, __FS_READ = 1, __FS_WRITE = 2, __FS_SEEK = 3, __FS_CLOSE = 4 };
enum { __FS_O_READ = 1, __FS_O_WRITE = 2, __FS_O_APPEND = 4, __FS_O_TRUNC = 8, __FS_O_CREATE = 16 };
#ifndef SEEK_SET
#define SEEK_SET 0
#define SEEK_CUR 1
#define SEEK_END 2
#endif

typedef unsigned long size_t;
typedef long ssize_t;

// A real buffered FILE: either **fd-backed** (0/1/2 = the powerbox stream, or an `fopen` fd) or
// **memory-backed** (`open_memstream`, fd == -1 → bytes accumulate in a growable malloc buffer whose
// pointer/length are handed back through `memp`/`memlenp`). chibicc's `format()` builds every string
// through a memory stream — `open_memstream` → `vfprintf` → `fclose` — so a real FILE is what lets
// chibicc compile *its own source* in the sandbox (SELFHOST_C.md §7, stage-2).
typedef struct __pg_FILE {
  int fd;                       // >= 0: powerbox/fs fd; -1: memory stream
  char **memp;                  // open_memstream: where to publish the buffer pointer
  size_t *memlenp;              // open_memstream: where to publish the length
  char *mem;                    // the growable buffer (memory stream)
  size_t memcap, memlen;
} FILE;
// The three standard streams. Normally this translation unit's own definition; under
// `__PG_LIBC_DECLS_ONLY` (a program unit linking against a prebuilt libc unit, #1392) it is a
// declaration the linker resolves to that unit's data symbol, so both units share *these* FILEs
// rather than a private copy each. External linkage either way, so the libc unit can export it.
#ifdef __PG_LIBC_DECLS_ONLY
extern FILE __pg_std[3];
#else
__PG_DATA FILE __pg_std[3] = {
  {0, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0}, {2, 0, 0, 0, 0, 0},
};
#endif
#define stdin (&__pg_std[0])
#define stdout (&__pg_std[1])
#define stderr (&__pg_std[2])
#define EOF (-1)


// ---- output sink: either a FILE (batched through a local buffer) or a caller string (s[n]printf) ----
struct __pf_sink {
  char *out;   // non-null → writing into a string (snprintf); null → writing to `file`
  size_t cap;  // string capacity (incl. space for the NUL)
  size_t n;    // total chars the full output *would* be (printf return value)
  FILE *file;  // target FILE when out == 0
  char buf[128];
  int bn;
};










// `fopen`/`fread` over the powerbox fs: the compiled program reads a served file (chibicc reads its
// `/in.c`). `open` is the fs-cap syscall; in the plain compile-and-run card there is no fs cap, so a
// program that only writes stdout never calls these — they exist so file-reading C (chibicc's own
// sources) compiles. A memory stream (`open_memstream`, fd -1) needs no fs.
int open(const char *path, int flags, ...);
int close(int fd);



// ---- prototypes (a program unit, #1392) --------------------------------------------------
// Compiling the bodies below into a program costs ~3.2 s for this header alone, and they are
// identical in every program — so a translation unit can instead be compiled decls-only
// (`-include __pg_decls_only.h`) against these prototypes and linked against a libc unit that
// carries the bodies once. Bodies are compiled in by default, so every existing whole-program
// caller is unaffected; there they are `static inline` (`__PG_FN`) and an unused one is
// dead-stripped, so the prototypes would only make them roots — hence the guard.

#ifdef __PG_LIBC_DECLS_ONLY
long __pg_slen(const char *s);
void __pg_fwrite_raw(FILE *f, const char *p, size_t n);
void __pf_flush(struct __pf_sink *s);
void __pf_emit(struct __pf_sink *s, char c);
int __pf_utoa(unsigned long val, unsigned base, int upper, char *tmp);
double __pf_pow10(int e);
int __pf_fix(char *out, double x, int prec);
int __pf_sci(char *out, double x, int prec, int upper);
int __pf_gen(char *out, double x, int prec, int upper, int alt);
int __pf_float(char *out, double x, int prec, char conv, int alt);
int __pf_vprint(struct __pf_sink *s, const char *fmt, va_list ap);
int vfprintf(FILE *stream, const char *fmt, va_list ap);
int fprintf(FILE *stream, const char *fmt, ...);
int printf(const char *fmt, ...);
int vsnprintf(char *str, size_t size, const char *fmt, va_list ap);
int snprintf(char *str, size_t size, const char *fmt, ...);
int sprintf(char *str, const char *fmt, ...);
int fputc(int c, FILE *stream);
int putc(int c, FILE *stream);
int putchar(int c);
int fputs(const char *s, FILE *stream);
int puts(const char *s);
size_t fwrite(const void *ptr, size_t sz, size_t nm, FILE *stream);
int fflush(FILE *stream);
FILE *fopen(const char *path, const char *mode);
FILE *open_memstream(char **bufp, size_t *lenp);
size_t fread(void *ptr, size_t sz, size_t nm, FILE *stream);
int fseek(FILE *stream, long off, int whence);
long ftell(FILE *stream);
void rewind(FILE *stream);
int fclose(FILE *stream);
int getchar(void);
char *fgets(char *s, int size, FILE *stream);
#endif /* __PG_LIBC_DECLS_ONLY */

// ---- bodies -----------------------------------------------------------------------------
// In their own file, not behind an `#ifdef` here: chibicc tokenizes a header in full before the
// preprocessor drops the skipped groups, so text left in place would still be *tokenized* by a
// decls-only compile — which is where nearly all of its remaining time goes (measured: the seeded libc
// bodies are ~90% of a decls-only program's compile). A separate file is never opened at all.
#ifndef __PG_LIBC_DECLS_ONLY
#include <__pg_stdio_impl.h>
#endif

#endif
