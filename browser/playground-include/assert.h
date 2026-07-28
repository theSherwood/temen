#ifndef __PG_ASSERT_H
#define __PG_ASSERT_H

// <assert.h> — on a failed assertion, print `file:line: Assertion `expr' failed.` to stderr and
// `abort()` (exit 134), matching glibc's shape. `NDEBUG` compiles asserts out, as usual. Re-includable
// (the macro is redefined per `NDEBUG` at each include), so the header guard covers only the helper.
#include <stdio.h>
#include <stdlib.h>
static inline void __pg_assert_fail(const char *expr, const char *file, int line) {
  fprintf(stderr, "%s:%d: Assertion `%s' failed.\n", file, line, expr);
  abort();
}
#endif

#undef assert
#ifdef NDEBUG
#define assert(x) ((void)0)
#else
#define assert(x) ((x) ? (void)0 : __pg_assert_fail(#x, __FILE__, __LINE__))
#endif
