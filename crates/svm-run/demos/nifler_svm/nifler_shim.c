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

/* `system` — nimsem shells out to `nifler` to parse stdlib modules on demand (deps.nim). Route the
 * shell command to the `exec` capability: split it into argv, EXEC_RUN it (nifler.svmb runs as an
 * isolated child domain sharing our memfs, writing its `.p.nif` where we can read it), then return
 * its exit status POSIX-wait-encoded so Nim's WEXITSTATUS recovers the code. `__vm_cap_resolve` /
 * `__vm_host_call` are the same §7 host-cap surface os_shim uses for `fs`. */
enum { EXEC_RUN = 0, EXEC_STATUS = 3, EXEC_CLOSE = 4 };
int system(const char *cmd) {
  if (!cmd) return 1; /* "is a shell available?" probe → yes */
  static int g_exec = -2;
  if (g_exec == -2) g_exec = __vm_cap_resolve("exec", 4);
  if (g_exec < 0) return -1;
  /* split cmd into a NUL-separated argv buffer (simple whitespace split, strips ' and " quotes) */
  static char av[8192];
  size_t n = 0;
  int in_tok = 0;
  for (const char *p = cmd; *p && n < sizeof(av) - 1; p++) {
    char ch = *p;
    if (ch == ' ' || ch == '\t' || ch == '\n') {
      if (in_tok) { av[n++] = 0; in_tok = 0; }
    } else if (ch == '"' || ch == '\'') {
      /* drop quote chars; a token may still be open */
    } else {
      av[n++] = ch;
      in_tok = 1;
    }
  }
  if (in_tok && n < sizeof(av)) av[n++] = 0;
  long job = __vm_host_call(g_exec, EXEC_RUN, (long)av, (long)n, (long)av, 0);
  if (job < 0) return -1;
  long st = __vm_host_call(g_exec, EXEC_STATUS, job, 0, 0, 0);
  __vm_host_call(g_exec, EXEC_CLOSE, job, 0, 0, 0);
  return (int)((st & 0xff) << 8); /* WEXITSTATUS(ret) == child exit code */
}

/* `readlink` — nimsem calls getAppFilename() → readlink("/proc/self/exe") to derive its stdlib path. */
ssize_t readlink(const char *path, char *buf, size_t bufsiz) {
  (void)path;
  static const char exe[] = "/bin/nimsem";
  size_t n = sizeof(exe) - 1;
  if (n > bufsiz) n = bufsiz;
  memcpy(buf, exe, n);
  return (ssize_t)n;
}
/* `strerror` — nimony phases format errnos with it; not an on-ramp builtin. */
char *strerror(int e) {
  switch (e) {
    case 2: return (char *)"No such file or directory";
    case 13: return (char *)"Permission denied";
    case 21: return (char *)"Is a directory";
    case 22: return (char *)"Invalid argument";
    default: return (char *)"error";
  }
}

/* `mmap`/`munmap` — hexer/nimony read NIF files through `std/memfiles` (`nifreader.nim`'s
 * `vfsOpenMmap`), i.e. `mmap(nil, size, PROT_READ, MAP_SHARED, fd, offset)`. There is no host address
 * space to map into, but a read-only file map is observationally just "the file's bytes at a stable
 * pointer", so serve it as malloc + read-the-region over the `fs` cap (the parse never writes back
 * through the map). An anonymous map (`fd < 0` / `MAP_ANONYMOUS`) is zeroed memory. `munmap` frees.
 * nifler never calls these (it parses `.nim` via stdio); they're the shared nimony-phase edge. */
#include <sys/mman.h>
void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset) {
  (void)addr;
  (void)prot;
  if (length == 0) return MAP_FAILED;
  void *p = malloc(length);
  if (!p) return MAP_FAILED;
  if (fd < 0 || (flags & MAP_ANONYMOUS)) {
    memset(p, 0, length);
    return p;
  }
  /* file-backed: copy [offset, offset+length) out of the fd via the fs cap */
  if (lseek(fd, offset, 0 /* SEEK_SET */) < 0) {
    free(p);
    return MAP_FAILED;
  }
  size_t got = 0;
  while (got < length) {
    long n = read(fd, (char *)p + got, length - got);
    if (n <= 0) break; /* short file: leave the tail as the malloc'd bytes (memfiles sizes to fstat) */
    got += (size_t)n;
  }
  return p;
}
int munmap(void *addr, size_t length) {
  (void)length;
  free(addr);
  return 0;
}

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
