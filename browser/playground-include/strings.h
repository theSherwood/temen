#ifndef __STRINGS_H
#define __STRINGS_H
// <strings.h> — the BSD case-insensitive compares live in the playground <string.h>; re-export them.
#include <string.h>
static inline int bcmp(const void *a, const void *b, size_t n) { return memcmp(a, b, n); }
static inline void bzero(void *s, size_t n) { char *p = s; for (size_t i = 0; i < n; i++) p[i] = 0; }
#endif
