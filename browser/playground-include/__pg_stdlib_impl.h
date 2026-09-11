#ifndef __PG_STDLIB_IMPL_H
#define __PG_STDLIB_IMPL_H

// The bodies of the seeded <stdlib.h> (#1392) — see that header for why they live in their own
// file, and `__pg_linkage.h` for what `__PG_FN` means in each of the three compile modes.

#include <stdlib.h>

static long __pg_brk = __PG_HEAP_BASE;       // next free byte (bump pointer)
static long __pg_committed = __PG_HEAP_BASE;  // first byte past committed
static long __pg_page = 0;                     // cached host page granularity
static int __pg_grow_lock = 0;                 // spinlock for heap *growth* only

static inline long __pg_pagesize(void) {
  if (__pg_page == 0) {
    long p = __vm_page_size();
    __pg_page = p > 0 ? p : 4096L;
  }
  return __pg_page;
}

// Lock-free fast path (atomic fetch-add claims a unique region); only page growth is serialized, so
// a page is mapped exactly once. `__pg_committed` is published *after* the map, so a caller seeing
// `committed >= end` knows its region is backed.
__PG_FN void *malloc(size_t n) {
  n = (n + 15UL) & ~15UL; // 16-byte align the payload
  long total = __PG_HDR + (long)n;
  long hdr = __vm_atomic_add(&__pg_brk, total);
  long payload = hdr + __PG_HDR;
  long end = hdr + total;
  if (end > __vm_atomic_load(&__pg_committed)) {
    while (__vm_atomic_cas32(&__pg_grow_lock, 0, 1) != 0) {
    }
    long cur = __vm_atomic_load(&__pg_committed);
    if (end > cur) {
      long pg = __pg_pagesize();
      long need = (end - cur + (pg - 1)) & ~(pg - 1);
      if (__vm_map(cur, need, 3) != 0) {
        __vm_atomic_store32(&__pg_grow_lock, 0);
        return NULL; // out of memory
      }
      __vm_atomic_store(&__pg_committed, cur + need);
    }
    __vm_atomic_store32(&__pg_grow_lock, 0);
  }
  *(size_t *)hdr = n;
  return (void *)payload;
}
__PG_FN void free(void *p) { (void)p; }
__PG_FN void *calloc(size_t nm, size_t sz) {
  // Fresh window pages are zero-filled by `map` and the bump allocator never reuses a byte, so the
  // payload is already zero.
  return malloc(nm * sz);
}
__PG_FN void *realloc(void *old, size_t n) {
  if (!old) return malloc(n);
  size_t oldn = *(size_t *)((char *)old - __PG_HDR);
  char *p = malloc(n);
  if (p) {
    size_t c = oldn < n ? oldn : n;
    for (size_t i = 0; i < c; i++) p[i] = ((char *)old)[i];
  }
  return p;
}

__PG_FN void abort(void) { exit(134); }

__PG_FN int abs(int x) { return x < 0 ? -x : x; }
__PG_FN long labs(long x) { return x < 0 ? -x : x; }

__PG_FN int atoi(const char *s) {
  int sign = 1, v = 0;
  while (*s == ' ' || *s == '\t' || *s == '\n') s++;
  if (*s == '-') { sign = -1; s++; } else if (*s == '+') s++;
  while (*s >= '0' && *s <= '9') v = v * 10 + (*s++ - '0');
  return sign * v;
}
__PG_FN long atol(const char *s) {
  long sign = 1, v = 0;
  while (*s == ' ' || *s == '\t' || *s == '\n') s++;
  if (*s == '-') { sign = -1; s++; } else if (*s == '+') s++;
  while (*s >= '0' && *s <= '9') v = v * 10 + (*s++ - '0');
  return sign * v;
}
__PG_FN long strtol(const char *s, char **end, int base) {
  long sign = 1, v = 0;
  while (*s == ' ' || *s == '\t' || *s == '\n') s++;
  if (*s == '-') { sign = -1; s++; } else if (*s == '+') s++;
  if ((base == 0 || base == 16) && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) { s += 2; base = 16; }
  if (base == 0) base = 10;
  for (;;) {
    int c = *s, d;
    if (c >= '0' && c <= '9') d = c - '0';
    else if (c >= 'a' && c <= 'z') d = c - 'a' + 10;
    else if (c >= 'A' && c <= 'Z') d = c - 'A' + 10;
    else break;
    if (d >= base) break;
    v = v * base + d;
    s++;
  }
  if (end) *end = (char *)s;
  return sign * v;
}

__PG_FN unsigned long strtoul(const char *s, char **end, int base) {
  unsigned long v = 0;
  while (*s == ' ' || *s == '\t' || *s == '\n') s++;
  if (*s == '+') s++;
  if ((base == 0 || base == 16) && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) { s += 2; base = 16; }
  if (base == 0) base = 10;
  for (;;) {
    int c = *s, d;
    if (c >= '0' && c <= '9') d = c - '0';
    else if (c >= 'a' && c <= 'z') d = c - 'a' + 10;
    else if (c >= 'A' && c <= 'Z') d = c - 'A' + 10;
    else break;
    if (d >= base) break;
    v = v * base + d;
    s++;
  }
  if (end) *end = (char *)s;
  return v;
}
__PG_FN long long strtoll(const char *s, char **end, int base) {
  long long sign = 1;
  while (*s == ' ' || *s == '\t' || *s == '\n') s++;
  if (*s == '-') { sign = -1; s++; } else if (*s == '+') s++;
  return sign * (long long)strtoul(s, end, base);
}
__PG_FN unsigned long long strtoull(const char *s, char **end, int base) {
  return (unsigned long long)strtoul(s, end, base);
}
__PG_FN long long atoll(const char *s) { return strtoll(s, (char **)0, 10); }
__PG_FN long long llabs(long long x) { return x < 0 ? -x : x; }

// `strtod` — decimal float parse (integer part, fraction, optional `e` exponent). Demo-accurate: it
// accumulates in `double` rather than doing bignum shortest-round-trip, so the last ULP of a long
// mantissa can differ from glibc (the same trade the `<stdio.h>` float formatter documents).
__PG_FN double strtod(const char *s, char **end) {
  while (*s == ' ' || *s == '\t' || *s == '\n') s++;
  double sign = 1;
  if (*s == '-') { sign = -1; s++; } else if (*s == '+') s++;
  double v = 0;
  while (*s >= '0' && *s <= '9') v = v * 10 + (*s++ - '0');
  if (*s == '.') {
    s++;
    double scale = 0.1;
    while (*s >= '0' && *s <= '9') { v += (*s++ - '0') * scale; scale *= 0.1; }
  }
  if (*s == 'e' || *s == 'E') {
    s++;
    int esign = 1, e = 0;
    if (*s == '-') { esign = -1; s++; } else if (*s == '+') s++;
    while (*s >= '0' && *s <= '9') e = e * 10 + (*s++ - '0');
    double p = 1, base = 10;
    for (int i = 0; i < e; i++) p *= base;
    if (esign < 0) v /= p; else v *= p;
  }
  if (end) *end = (char *)s;
  return sign * v;
}
__PG_FN double atof(const char *s) { return strtod(s, (char **)0); }
// `long double` is built as `double` (chibicc's -mlong-double-64), so strtold is strtod.
__PG_FN long double strtold(const char *s, char **end) { return strtod(s, end); }

__PG_FN div_t div(int a, int b) { div_t r; r.quot = a / b; r.rem = a % b; return r; }
__PG_FN ldiv_t ldiv(long a, long b) { ldiv_t r; r.quot = a / b; r.rem = a % b; return r; }

// `bsearch` over a sorted array (the qsort companion).
__PG_FN void *bsearch(const void *key, const void *base, size_t n, size_t sz,
                            int (*cmp)(const void *, const void *)) {
  size_t lo = 0, hi = n;
  const char *a = (const char *)base;
  while (lo < hi) {
    size_t mid = lo + (hi - lo) / 2;
    int c = cmp(key, a + mid * sz);
    if (c < 0) hi = mid;
    else if (c > 0) lo = mid + 1;
    else return (void *)(a + mid * sz);
  }
  return 0;
}

// No environment in the sandbox powerbox — a program that reads `getenv` gets NULL (unset), which is
// the portable "not present" path every getenv caller must already handle.
__PG_FN char *getenv(const char *name) { (void)name; return 0; }

// Deterministic LCG (no wall clock in the sandbox).
static unsigned long __pg_rng = 1;
__PG_FN int rand(void) { __pg_rng = __pg_rng * 6364136223846793005UL + 1442695040888963407UL; return (int)((__pg_rng >> 33) & 0x7fffffff); }
__PG_FN void srand(unsigned s) { __pg_rng = s; }

// Simple qsort (insertion sort — fine for demo-sized arrays; stable enough, no recursion depth).
__PG_FN void qsort(void *base, size_t n, size_t sz, int (*cmp)(const void *, const void *)) {
  char *a = base;
  char tmp[256];
  if (sz > sizeof(tmp)) return; // demo cap
  for (size_t i = 1; i < n; i++) {
    for (size_t j = i; j > 0 && cmp(a + j * sz, a + (j - 1) * sz) < 0; j--) {
      for (size_t k = 0; k < sz; k++) tmp[k] = a[j * sz + k];
      for (size_t k = 0; k < sz; k++) a[j * sz + k] = a[(j - 1) * sz + k];
      for (size_t k = 0; k < sz; k++) a[(j - 1) * sz + k] = tmp[k];
    }
  }
}

#endif
