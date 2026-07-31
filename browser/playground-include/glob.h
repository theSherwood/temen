#ifndef __GLOB_H
#define __GLOB_H
// Stub: the sandbox driver never globs (cc1 --emit-ir only). Types exist so chibicc.h parses.
#include <stddef.h>
typedef struct { size_t gl_pathc; char **gl_pathv; size_t gl_offs; } glob_t;
#define GLOB_ERR 1
static inline int glob(const char *p, int f, int (*e)(const char *, int), glob_t *g) {
  (void)p; (void)f; (void)e; if (g) { g->gl_pathc = 0; g->gl_pathv = 0; } return 1; /* GLOB_NOMATCH */
}
static inline void globfree(glob_t *g) { (void)g; }
#endif
