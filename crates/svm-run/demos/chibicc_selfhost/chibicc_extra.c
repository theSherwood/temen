/* chibicc self-host libc — the remainder of Appendix A.5 not covered by the reused Postgres
 * shims (SELFHOST_C.md §5 B / Appendix B). `#include`d by chibicc_libc.c under -DSVM_GUEST,
 * AFTER os_shim.c (uses its open/read/write/close), so declarations resolve within the one TU.
 *
 * Two pieces the Postgres shims don't supply because Postgres doesn't need them:
 *   (1) the allocator: `malloc`/`realloc` are LEFT UNDEFINED so the LLVM on-ramp synthesizes its
 *       **`vm_map`-growable** heap for them (grows into the reserved tail on demand, §1a) — this is
 *       what lets chibicc compile a whole *multi-file* program (its own source) whose working set
 *       exceeds a fixed arena. Only `calloc` (a thin zeroing wrapper over the synth `malloc`) and
 *       `free` (a no-op — the synth heap never reclaims) stay in C. (Earlier bring-up used a fixed
 *       24 MiB static arena; the grow-via-`__vm_map` follow-up is now done — SELFHOST_C.md §8.)
 *   (2) open_memstream — strings.c format()/tokenize.c build strings through a memory FILE*; the
 *       stdio here is fd-backed OR memory-backed, so vfprintf (printf_shim) composes with both.
 * Plus the trivial string/ctype/time gaps.
 */
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <sys/stat.h>

/* os_shim.c bottom edge (fd-dispatch: 0/1/2 → powerbox Stream, ≥3 → fs cap). */
extern long write(int fd, const void *buf, unsigned long n);
extern long read(int fd, void *buf, unsigned long n);
extern int open(const char *path, int flags, ...);
extern int close(int fd);
extern long lseek(int fd, long off, int whence);

/* ---- allocator ---------------------------------------------------------------------------------
 * `malloc`/`realloc` are intentionally NOT defined here: the on-ramp synthesizes its vm_map-growable
 * heap for any undefined `malloc`/`realloc` in a module with a `main` (svm-llvm `synth_malloc`/
 * `synth_realloc`), so chibicc's heap grows into the reserved tail instead of hitting a fixed cap —
 * the difference between compiling one source file and compiling the whole compiler. `calloc` wraps
 * the synth `malloc` (fresh vm_map pages are already zero, but zero explicitly for clarity); `free`
 * is a no-op (the bump-style synth heap does not reclaim). Only these two stay in C. */
extern void *malloc(unsigned long n);
void free(void *p) { (void)p; }
void *calloc(unsigned long nm, unsigned long sz) {
  unsigned long n = nm * sz;
  unsigned char *p = (unsigned char *)malloc(n);
  if (p) for (unsigned long i = 0; i < n; i++) p[i] = 0;
  return p;
}

/* ---- trivial string/ctype gaps (Appendix A.5) ----------------------------------------------- */
char *strchr(const char *s, int c) {
  for (; *s; s++) if (*s == (char)c) return (char *)s;
  return c == 0 ? (char *)s : (char *)0;
}
void *memchr(const void *s, int c, unsigned long n) {
  const unsigned char *p = (const unsigned char *)s;
  for (unsigned long i = 0; i < n; i++) if (p[i] == (unsigned char)c) return (void *)(p + i);
  return (void *)0;
}
extern unsigned long strlen(const char *); /* mem_shim.c */
char *strndup(const char *s, unsigned long n) {
  unsigned long l = 0; while (l < n && s[l]) l++;
  char *d = (char *)malloc(l + 1);
  if (d) { for (unsigned long i = 0; i < l; i++) d[i] = s[i]; d[l] = 0; }
  return d;
}
static int lc(int c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }
int strncasecmp(const char *a, const char *b, unsigned long n) {
  for (unsigned long i = 0; i < n; i++) {
    int d = lc((unsigned char)a[i]) - lc((unsigned char)b[i]);
    if (d || !a[i]) return d;
  }
  return 0;
}

/* strtold: chibicc parses float literals with it; under -mlong-double-64 (F3) long double = double,
 * so it is strtod (the real correctly-rounded one, ../strtod/strtod.c, included by chibicc_libc.c). */
extern double strtod(const char *, char **);
long double strtold(const char *s, char **end) { return (long double)strtod(s, end); }

/* dirname: preprocess.c resolves a relative #include against the current file's dir (pure string;
 * mutates its argument, like glibc). */
char *dirname(char *path) {
  if (!path || !*path) return (char *)".";
  char *last = (char *)0;
  for (char *p = path; *p; p++) if (*p == '/') last = p;
  if (!last) return (char *)".";
  if (last == path) { path[1] = 0; return path; } /* "/x" → "/" */
  *last = 0;
  return path;
}

/* ---- time: only __DATE__/__TIME__/__TIMESTAMP__ — fixed epoch ⇒ reproducible builds (F4) ----- */
typedef long time_t;
time_t time(time_t *t) { if (t) *t = 0; return 0; }
struct tm { int tm_sec, tm_min, tm_hour, tm_mday, tm_mon, tm_year, tm_wday, tm_yday, tm_isdst; };
static struct tm g_tm; /* 1970-01-01 00:00:00, a Thursday */
struct tm *localtime(const time_t *t) { (void)t; g_tm.tm_year = 70; g_tm.tm_mday = 1; g_tm.tm_wday = 4; return &g_tm; }
char *ctime_r(const time_t *t, char *buf) {
  (void)t; const char *s = "Thu Jan  1 00:00:00 1970\n";
  char *d = buf;
  while ((*d++ = *s++)) {
  }
  return buf;
}

/* strerror: chibicc only prints these on I/O error paths; a terse message matches nothing the
 * differential inspects (error output goes to stderr, not the emitted IR). */
char *strerror(int e) { (void)e; return (char *)"error"; }

/* assert: glibc lowers `assert(x)` to `__assert_fail(expr, file, line, func)`. chibicc asserts on
 * its own invariants (hashmap.c, parse.c); a fired assert is a compiler bug, so report to stderr and
 * exit non-zero (the one Appendix-A.5 symbol not on-ramp-recognized nor supplied by a reused shim). */
extern void exit(int);
static void put2(const char *s) { if (s) write(2, s, strlen(s)); }
void __assert_fail(const char *expr, const char *file, unsigned int line, const char *func) {
  put2(file); put2(":"); put2(func); put2(": Assertion `"); put2(expr); put2("' failed\n");
  (void)line;
  exit(2);
}

/* ---- stdio: fd-backed OR memory-backed FILE (open_memstream), sharing one fwrite/fputc -------- */
typedef struct {
  int fd;              /* ≥0: fs/stream fd; -1: memory stream */
  int eof, err;
  /* memory-stream state (open_memstream): */
  char **memp; size_t *memlenp; char *mem; size_t memcap, memlen;
} CFILE;

static CFILE g_stdin = { 0, 0, 0, 0, 0, 0, 0, 0 };
static CFILE g_stdout = { 1, 0, 0, 0, 0, 0, 0, 0 };
static CFILE g_stderr = { 2, 0, 0, 0, 0, 0, 0, 0 };
FILE *stdin = (FILE *)&g_stdin;
FILE *stdout = (FILE *)&g_stdout;
FILE *stderr = (FILE *)&g_stderr;

static int mode_flags(const char *m) { /* O_* per os_shim's FS_O_* mapping via fcntl.h values */
  int r = 0;
  if (m[0] == 'r') r = 0;              /* O_RDONLY */
  else if (m[0] == 'w') r = 01 | 0100 | 01000; /* O_WRONLY|O_CREAT|O_TRUNC */
  else if (m[0] == 'a') r = 01 | 0100 | 02000; /* O_WRONLY|O_CREAT|O_APPEND */
  return r;
}
FILE *fopen(const char *path, const char *mode) {
  int fd = open(path, mode_flags(mode));
  if (fd < 0) return (FILE *)0;
  CFILE *f = (CFILE *)malloc(sizeof *f);
  if (!f) { close(fd); return (FILE *)0; }
  f->fd = fd; f->eof = f->err = 0; f->mem = 0; f->memp = 0; f->memlenp = 0; f->memcap = f->memlen = 0;
  return (FILE *)f;
}
FILE *open_memstream(char **bufp, size_t *lenp) {
  CFILE *f = (CFILE *)malloc(sizeof *f);
  if (!f) return (FILE *)0;
  f->fd = -1; f->eof = f->err = 0;
  f->memp = bufp; f->memlenp = lenp; f->memcap = 128; f->memlen = 0;
  f->mem = (char *)malloc(f->memcap);
  if (!f->mem) return (FILE *)0;
  f->mem[0] = 0; *bufp = f->mem; *lenp = 0;
  return (FILE *)f;
}
size_t fwrite(const void *ptr, size_t size, size_t nmemb, FILE *stream) {
  CFILE *f = (CFILE *)stream;
  size_t n = size * nmemb;
  if (f->fd == -1) {                    /* memory stream: append + keep NUL-terminated */
    if (f->memlen + n + 1 > f->memcap) {
      size_t cap = f->memcap; while (f->memlen + n + 1 > cap) cap *= 2;
      f->mem = (char *)realloc(f->mem, cap); f->memcap = cap;
    }
    const char *s = (const char *)ptr;
    for (size_t i = 0; i < n; i++) f->mem[f->memlen + i] = s[i];
    f->memlen += n; f->mem[f->memlen] = 0;
    *f->memp = f->mem; *f->memlenp = f->memlen;
    return nmemb;
  }
  long w = write(f->fd, ptr, n);
  if (w < 0) { f->err = 1; return 0; }
  return size ? (size_t)w / size : 0;
}
size_t fread(void *ptr, size_t size, size_t nmemb, FILE *stream) {
  CFILE *f = (CFILE *)stream;
  long r = read(f->fd, ptr, size * nmemb);
  if (r <= 0) { f->eof = 1; return 0; }
  return size ? (size_t)r / size : 0;
}
int fputc(int c, FILE *stream) { unsigned char b = (unsigned char)c; return fwrite(&b, 1, 1, stream) == 1 ? c : -1; }
int fputs(const char *s, FILE *stream) { return (int)fwrite(s, 1, strlen(s), stream); }
int fflush(FILE *stream) { (void)stream; return 0; }
int fclose(FILE *stream) {
  CFILE *f = (CFILE *)stream;
  if (f->fd >= 0) close(f->fd);
  /* memory stream: buffer belongs to the caller (returned via *memp) — do not free. */
  return 0;
}
int puts(const char *s) { fputs(s, stdout); return fputc('\n', stdout) == '\n' ? 0 : -1; }
