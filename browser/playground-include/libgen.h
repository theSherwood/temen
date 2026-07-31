#ifndef __LIBGEN_H
#define __LIBGEN_H
#include <string.h>
// Minimal basename/dirname over a static buffer (driver-only in chibicc; here for parse + link).
static char __pg_bn[1024];
static inline char *basename(char *path) {
  if (!path || !*path) { __pg_bn[0] = '.'; __pg_bn[1] = 0; return __pg_bn; }
  char *s = strrchr(path, '/');
  return s ? s + 1 : path;
}
static inline char *dirname(char *path) {
  if (!path || !*path) { __pg_bn[0] = '.'; __pg_bn[1] = 0; return __pg_bn; }
  char *s = strrchr(path, '/');
  if (!s) { __pg_bn[0] = '.'; __pg_bn[1] = 0; return __pg_bn; }
  *s = 0; return path;
}
#endif
