/* Minimal self-host libc for the chibicc `--emit-object` path — increment 1 (SELFHOST_C.md, task
 * #20). This is NOT the clang-tuned aggregator (`demos/chibicc_selfhost/chibicc_libc.c`, which relies
 * on the LLVM on-ramp synthesizing `malloc`/`memcpy`/`memset`/`__vm_fmt_*` and does not compile under
 * emit-object). It is a from-scratch libc *designed for chibicc's own frontend*: everything below is
 * plain C89-ish that emit-object lowers directly, and the only undefined symbol is `write` — a
 * powerbox capability the host binds at instantiation (a retained manifest import, exactly like the
 * cross-TU `write` in `c_link.rs::links_and_runs_under_powerbox_with_argv`).
 *
 * It proves the composition the real emit-object libc needs — a bump `malloc`, the `mem*`/`str*`
 * family, and a **varargs** `printf` that writes through the powerbox `write` cap — all flowing
 * through emit-object → link → run on interp==JIT. Later increments grow this toward the full surface
 * (`realloc`, `fopen`/file I/O, float formatting) that lets all ~9 cc1 TUs run end-to-end.
 *
 * No *system* `#include`: emit-object crafted units avoid system headers (chibicc's default include
 * path is Linux-hardcoded — see `c_link.rs::object_unit_real`), so this stays cross-platform like the
 * other crafted units. The one `#include <stdarg.h>` below is a *compiler-provided* header chibicc
 * supplies itself (not system libc), needed for the varargs `printf`. Non-`static` definitions export
 * as cross-TU symbols the consumer links against.
 */

typedef unsigned long size_t;

/* The one host capability: fd-dispatch write (0/1/2 → powerbox streams). Left undefined so the linker
 * retains it as a manifest import the powerbox binds — chibicc's real libc reaches the fs the same way. */
extern long write(int fd, const void *buf, size_t n);

/* ---- allocator: a static-arena bump `malloc` -------------------------------------------------------
 * The real emit-object libc will grow into the reserved tail via a `vm_map` primitive; increment 1 uses
 * a fixed arena (enough to prove malloc'd memory is live and cross-TU-callable). 16-byte aligned. */
static char g_arena[1 << 15];
static size_t g_off = 0;
void *malloc(size_t n) {
  n = (n + 15) & ~(size_t)15; /* round up to 16 */
  if (g_off + n > sizeof g_arena) return (void *)0;
  void *p = &g_arena[g_off];
  g_off += n;
  return p;
}
void *calloc(size_t nmemb, size_t sz) {
  size_t n = nmemb * sz;
  unsigned char *p = (unsigned char *)malloc(n);
  if (p) for (size_t i = 0; i < n; i++) p[i] = 0;
  return p;
}
void free(void *p) { (void)p; } /* bump arena never reclaims */

/* ---- mem and str family --------------------------------------------------------------------------- */
void *memcpy(void *d, const void *s, size_t n) {
  unsigned char *dp = (unsigned char *)d;
  const unsigned char *sp = (const unsigned char *)s;
  for (size_t i = 0; i < n; i++) dp[i] = sp[i];
  return d;
}
void *memset(void *d, int c, size_t n) {
  unsigned char *dp = (unsigned char *)d;
  for (size_t i = 0; i < n; i++) dp[i] = (unsigned char)c;
  return d;
}
size_t strlen(const char *s) {
  size_t n = 0;
  while (s[n]) n++;
  return n;
}
int strcmp(const char *a, const char *b) {
  while (*a && *a == *b) { a++; b++; }
  return (int)(unsigned char)*a - (int)(unsigned char)*b;
}
char *strcpy(char *d, const char *s) {
  char *r = d;
  while ((*d++ = *s++)) {}
  return r;
}

/* ---- varargs printf: %d %u %x %c %s %% (integer/string only — floats are a later increment) --------
 * Formats into a stack buffer and writes it through the powerbox `write` cap in one call. Proves
 * `va_start`/`va_arg(int)`/`va_arg(char*)`/`va_end` lower correctly under emit-object — the mechanism
 * chibicc's own `fprintf`/`vfprintf` stand on. */
#include <stdarg.h> /* stdarg is a compiler-provided header, not a system libc one — emit-object supplies it */

/* append the decimal/hex text of an unsigned magnitude; returns chars written */
static int put_uint(char *out, unsigned long v, int base, int upper) {
  const char *dig = upper ? "0123456789ABCDEF" : "0123456789abcdef";
  char tmp[24];
  int n = 0;
  do { tmp[n++] = dig[v % (unsigned long)base]; v /= (unsigned long)base; } while (v);
  for (int i = 0; i < n; i++) out[i] = tmp[n - 1 - i];
  return n;
}

int printf(const char *fmt, ...) {
  char buf[512];
  int len = 0;
  va_list ap;
  va_start(ap, fmt);
  for (const char *p = fmt; *p; p++) {
    if (*p != '%') { buf[len++] = *p; continue; }
    p++;
    switch (*p) {
    case 'd': {
      int v = va_arg(ap, int);
      unsigned long m;
      if (v < 0) { buf[len++] = '-'; m = (unsigned long)(-(long)v); } else m = (unsigned long)v;
      len += put_uint(buf + len, m, 10, 0);
      break;
    }
    case 'u': len += put_uint(buf + len, (unsigned long)va_arg(ap, unsigned int), 10, 0); break;
    case 'x': len += put_uint(buf + len, (unsigned long)va_arg(ap, unsigned int), 16, 0); break;
    case 'c': buf[len++] = (char)va_arg(ap, int); break;
    case 's': {
      const char *s = va_arg(ap, const char *);
      while (*s) buf[len++] = *s++;
      break;
    }
    case '%': buf[len++] = '%'; break;
    default: buf[len++] = '%'; buf[len++] = *p; break;
    }
  }
  va_end(ap);
  write(1, buf, (size_t)len);
  return len;
}
