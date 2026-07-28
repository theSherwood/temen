#ifndef __TIME_H
#define __TIME_H
// <time.h> — no wall clock in the sandbox, so time() is a fixed 1970 epoch (matches chibicc_extra.c,
// the guest build's stubs). Enough for chibicc's __DATE__/__TIME__/__TIMESTAMP__ predefined macros.
#include <stddef.h>
typedef long time_t;
struct tm {
  int tm_sec, tm_min, tm_hour, tm_mday, tm_mon, tm_year, tm_wday, tm_yday, tm_isdst;
};
static struct tm __pg_tm;
static inline time_t time(time_t *t) { if (t) *t = 0; return 0; }
static inline struct tm *localtime(const time_t *t) {
  (void)t; __pg_tm.tm_year = 70; __pg_tm.tm_mday = 1; __pg_tm.tm_wday = 4; return &__pg_tm;
}
static inline struct tm *gmtime(const time_t *t) { return localtime(t); }
static inline char *ctime_r(const time_t *t, char *buf) {
  (void)t; const char *s = "Thu Jan  1 00:00:00 1970\n";
  int i = 0; while (s[i]) { buf[i] = s[i]; i++; } buf[i] = 0; return buf;
}
#endif
