#ifndef __STDLIB_H
#define __STDLIB_H

// <stdlib.h> for the playground. `malloc` is a bump allocator over a static arena (each block
// carries an 8-byte size header so `realloc` can copy); `free` is a no-op. That's plenty for a
// single-shot compile-and-run demo — no authority, all guest C.
#include <stdarg.h>

typedef unsigned long size_t;
#define NULL ((void *)0)
#define RAND_MAX 0x7fffffff

void exit(int code);

#define __PG_HEAP_BYTES (1 << 16) // 64 KiB arena (a demo allocator; keeps the guest window modest)
static char __pg_heap[__PG_HEAP_BYTES];
static size_t __pg_brk = 0;

static inline void *malloc(size_t n) {
  size_t need = ((n + 8) + 15) & ~(size_t)15; // 8-byte header + 16-byte align
  if (__pg_brk + need > __PG_HEAP_BYTES) return NULL;
  char *p = __pg_heap + __pg_brk;
  __pg_brk += need;
  *(size_t *)p = n;
  return p + 8;
}
static inline void free(void *p) { (void)p; }
static inline void *calloc(size_t nm, size_t sz) {
  size_t n = nm * sz;
  char *p = malloc(n);
  if (p) for (size_t i = 0; i < n; i++) p[i] = 0;
  return p;
}
static inline void *realloc(void *old, size_t n) {
  if (!old) return malloc(n);
  size_t oldn = *(size_t *)((char *)old - 8);
  char *p = malloc(n);
  if (p) {
    size_t c = oldn < n ? oldn : n;
    for (size_t i = 0; i < c; i++) p[i] = ((char *)old)[i];
  }
  return p;
}

static inline void abort(void) { exit(134); }

static inline int abs(int x) { return x < 0 ? -x : x; }
static inline long labs(long x) { return x < 0 ? -x : x; }

static inline int atoi(const char *s) {
  int sign = 1, v = 0;
  while (*s == ' ' || *s == '\t' || *s == '\n') s++;
  if (*s == '-') { sign = -1; s++; } else if (*s == '+') s++;
  while (*s >= '0' && *s <= '9') v = v * 10 + (*s++ - '0');
  return sign * v;
}
static inline long atol(const char *s) {
  long sign = 1, v = 0;
  while (*s == ' ' || *s == '\t' || *s == '\n') s++;
  if (*s == '-') { sign = -1; s++; } else if (*s == '+') s++;
  while (*s >= '0' && *s <= '9') v = v * 10 + (*s++ - '0');
  return sign * v;
}
static inline long strtol(const char *s, char **end, int base) {
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

// Deterministic LCG (no wall clock in the sandbox).
static unsigned long __pg_rng = 1;
static inline int rand(void) { __pg_rng = __pg_rng * 6364136223846793005UL + 1442695040888963407UL; return (int)((__pg_rng >> 33) & 0x7fffffff); }
static inline void srand(unsigned s) { __pg_rng = s; }

// Simple qsort (insertion sort — fine for demo-sized arrays; stable enough, no recursion depth).
static inline void qsort(void *base, size_t n, size_t sz, int (*cmp)(const void *, const void *)) {
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
