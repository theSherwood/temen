/* nifler-on-SVM bottom-edge shim (NIM.md §3c/§3e, "nimony in the browser" slice 1).
 *
 * `nifler` — the first nimony phase (Nim source → NIF) — is a real Nim program compiled to C by the
 * stock Nim compiler's ARC backend and on-ramped to SVM (clang-18 → svm-llvm). Its whole-program
 * bitcode leaves ~115 undefined externals: the libc bottom edge Nim's runtime + `std/os`/`std/io`
 * assume. This one translation unit *defines* the reachable part of that edge over the sandbox's
 * `fs` capability + powerbox Stream, reusing the exact shims that already run Postgres on SVM:
 *
 *   - the POSIX fd/dir/stat syscalls  → ../postgres/os_shim.c   (open/read/write/stat/opendir/…)
 *   - the buffered FILE* surface       → ../postgres/stdio_shim.c (fopen/fread/fgets/fwrite/…)
 *   - the shared guest `errno` cell     → ../postgres/shim_errno.h (__errno_location)
 *
 * Composed into ONE TU (like the eventual whole-Postgres build) so `shim_errno.h`'s include guard
 * yields a single `__errno_location`, and `fdopen` below can reach stdio_shim's `static` ShimFile.
 * The supplement here is only the handful of extra symbols nifler's *parse* path reaches at init;
 * the math family (sin/cos/…) and the process/spawn fringe (posix_spawn/waitpid/system/glob/…) are
 * never called on the parse path and stay `--stub-externs` traps (a call would fault, not escape).
 *
 * No ambient authority: every file byte rides the embedder-granted `fs` cap; with no cap, no bytes.
 */

/* os_shim's `getcwd` returns "." (the cap root as Postgres saw it). Nim's `std/os` `absolutePath`
 * rejects a non-absolute cwd ("The specified root is not absolute"), so rename os_shim's version out
 * of the way and provide one below that returns "/" — the cap root spelled absolutely. */
#define getcwd os_shim_getcwd_unused
#include "../postgres/os_shim.c"
#undef getcwd
#include "../postgres/stdio_shim.c"

char *getcwd(char *buf, size_t size) {
  if (!buf || size < 2) {
    shim_errno = 34; /* ERANGE */
    return (char *)0;
  }
  buf[0] = '/';
  buf[1] = 0;
  return buf;
}

/* ---- supplement: the extra libc edge nifler reaches, none in the two shims above --------------- */

#include <time.h>

/* `fdopen` — wrap an existing fs-cap fd in a FILE. stdio_shim's `shim_new` (same TU, `static`) builds
 * exactly the ShimFile a plain `fopen` returns, so an fd already opened by `open` becomes a FILE. */
FILE *fdopen(int fd, const char *mode) { return (FILE *)shim_new(fd, mode); }

/* Environment — the sandbox exposes none. An empty `environ` + null `getenv` are what a guest with no
 * inherited env sees; `setenv`/`unsetenv` succeed as no-ops (nothing consults them downstream). */
static char *g_empty_environ[1] = {0};
char **environ = g_empty_environ;
char *getenv(const char *name) {
  (void)name;
  return (char *)0;
}
int setenv(const char *n, const char *v, int o) {
  (void)n;
  (void)v;
  (void)o;
  return 0;
}
int unsetenv(const char *n) {
  (void)n;
  return 0;
}

/* Time — deterministic zero clock (a compiler phase must be reproducible; nifler stamps no real time
 * into a parse output, and the browser has no wall clock to grant anyway). */
int clock_gettime(clockid_t clk, struct timespec *ts) {
  (void)clk;
  if (ts) {
    ts->tv_sec = 0;
    ts->tv_nsec = 0;
  }
  return 0;
}
int clock_nanosleep(clockid_t clk, int flags, const struct timespec *req, struct timespec *rem) {
  (void)clk;
  (void)flags;
  (void)req;
  (void)rem;
  return 0;
}

/* Terminal/exit/proc — a batch guest: no tty, nothing to run at exit, a fixed pid. */
int isatty(int fd) {
  (void)fd;
  return 0;
}
int atexit(void (*fn)(void)) {
  (void)fn;
  return 0;
}
int getpid(void) { return 1; }

/* Single-threaded guest — the mutex ops the Nim allocator references are no-ops (no contention). */
int pthread_mutex_init(void *m, const void *a) {
  (void)m;
  (void)a;
  return 0;
}
int pthread_mutex_lock(void *m) {
  (void)m;
  return 0;
}
int pthread_mutex_unlock(void *m) {
  (void)m;
  return 0;
}
