#ifndef __STRING_H
#define __STRING_H

// Pure-computation <string.h> for the playground (no authority) — all guest C, compiled in on use.
typedef unsigned long size_t;

static inline size_t strlen(const char *s) {
  size_t n = 0;
  while (s[n]) n++;
  return n;
}
static inline int strcmp(const char *a, const char *b) {
  while (*a && *a == *b) { a++; b++; }
  return (unsigned char)*a - (unsigned char)*b;
}
static inline int strncmp(const char *a, const char *b, size_t n) {
  for (size_t i = 0; i < n; i++) {
    if (a[i] != b[i]) return (unsigned char)a[i] - (unsigned char)b[i];
    if (!a[i]) break;
  }
  return 0;
}
static inline char *strcpy(char *d, const char *s) {
  char *r = d;
  while ((*d++ = *s++)) {}
  return r;
}
static inline char *strncpy(char *d, const char *s, size_t n) {
  size_t i = 0;
  for (; i < n && s[i]; i++) d[i] = s[i];
  for (; i < n; i++) d[i] = 0;
  return d;
}
static inline char *strcat(char *d, const char *s) {
  char *r = d;
  while (*d) d++;
  while ((*d++ = *s++)) {}
  return r;
}
static inline char *strchr(const char *s, int c) {
  for (; *s; s++) if (*s == (char)c) return (char *)s;
  return (char)c ? 0 : (char *)s;
}
static inline char *strrchr(const char *s, int c) {
  const char *last = 0;
  for (; *s; s++) if (*s == (char)c) last = s;
  return (char *)last;
}
static inline char *strstr(const char *hay, const char *needle) {
  if (!*needle) return (char *)hay;
  for (; *hay; hay++) {
    const char *h = hay, *n = needle;
    while (*h && *n && *h == *n) { h++; n++; }
    if (!*n) return (char *)hay;
  }
  return 0;
}
static inline void *memcpy(void *d, const void *s, size_t n) {
  char *dd = d;
  const char *ss = s;
  for (size_t i = 0; i < n; i++) dd[i] = ss[i];
  return d;
}
static inline void *memmove(void *d, const void *s, size_t n) {
  char *dd = d;
  const char *ss = s;
  if (dd < ss) for (size_t i = 0; i < n; i++) dd[i] = ss[i];
  else for (size_t i = n; i > 0; i--) dd[i - 1] = ss[i - 1];
  return d;
}
static inline void *memset(void *d, int c, size_t n) {
  char *dd = d;
  for (size_t i = 0; i < n; i++) dd[i] = (char)c;
  return d;
}
static inline int memcmp(const void *a, const void *b, size_t n) {
  const unsigned char *aa = a, *bb = b;
  for (size_t i = 0; i < n; i++) if (aa[i] != bb[i]) return aa[i] - bb[i];
  return 0;
}
static inline void *memchr(const void *s, int c, size_t n) {
  const unsigned char *p = s;
  for (size_t i = 0; i < n; i++) if (p[i] == (unsigned char)c) return (void *)(p + i);
  return 0;
}

#endif
