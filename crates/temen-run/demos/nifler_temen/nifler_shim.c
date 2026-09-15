/* nifler-on-Temen bottom-edge shim (NIM.md §3c/§3e, "nimony in the browser" slice 1).
 *
 * `nifler` — the first nimony phase (Nim source → NIF) — is a real Nim program compiled to C by the
 * stock Nim compiler's ARC backend and on-ramped to Temen (clang-18 → temen-llvm). Its whole-program
 * bitcode leaves ~115 undefined externals: the libc bottom edge Nim's runtime + `std/os`/`std/io`
 * assume. This one translation unit *defines* the reachable part of that edge over the sandbox's
 * `fs` capability + powerbox Stream, reusing the exact shims that already run Postgres on Temen:
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
 * shell command to the `exec` capability: split it into argv, EXEC_RUN it (nifler.temen runs as an
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

/* `strtod` — Nim's `parseBiggestFloat` (system.nim) pre-normalizes a float literal to a clean decimal
 * string `[-]d.dddE±exp` and then calls `strtod` for the decimal→binary step. That call sits off the
 * bare parse path, so it was a `--stub-externs` trap — compiling any module with a float literal (e.g.
 * `std/parseutils`, `std/strutils`) traps the guest with `Unreachable` while native nimony links libc's
 * strtod (#1382).
 *
 * This must be **correctly rounded**, not merely close. nifler bakes the parsed value straight into
 * the `.p.nif` it emits, so a last-place error is a compiler that silently disagrees with native
 * nimony about what a constant means. The first version here scaled by repeated multiplication in
 * chunks of 10^22, which rounds once per chunk: `1.5e100` came out `1.5000000000000001E+100` and
 * DBL_MAX came out one ulp low — caught by this demo's `floats.nim` oracle diff (#1364).
 *
 * Two paths:
 *
 *   Fast (Clinger): significand ≤ 2^53 and |exp10| ≤ 22, so both operands are exactly representable
 *   and a single multiply or divide is correctly rounded by IEEE-754 alone. This covers essentially
 *   every literal in real source.
 *
 *   Exact fallback: the classic big-decimal shift (Gay / Go's `strconv.decimal`). Hold the value as an
 *   arbitrary-precision **decimal** digit string and binary-shift it — halving or doubling a decimal
 *   string is exact — until the significand sits in [2^52, 2^53). Read off 53 bits, then round to
 *   nearest-even using the remaining digits to break the tie exactly. No wide multiplication, no
 *   generated power tables, and no accumulated rounding: the only rounding is the final one. */

/* Enough digits for the widest meaningful input: a subnormal needs ~1080 significant bits, and each
 * right shift adds at most one digit. `parseBiggestFloat` hands us at most ~325 digits + exponent;
 * left-shifting that into range stays well inside this. Excess digits set `trunc`, which only ever
 * affects an exact tie (and then only by breaking it away from even, which is the correct answer). */
#define NIM_DEC_CAP 1600

typedef struct {
  unsigned char d[NIM_DEC_CAP]; /* decimal digits, most significant first, no leading zero */
  int nd;                       /* number of digits held */
  int dp;                       /* decimal point: value = 0.d[0]d[1]... * 10^dp */
  int trunc;                    /* digits were dropped past NIM_DEC_CAP */
} nim_dec;

static void nim_dec_trim(nim_dec *a) {
  while (a->nd > 0 && a->d[a->nd - 1] == 0) a->nd--;
  if (a->nd == 0) a->dp = 0;
}

/* Right-shift by `k` (divide by 2^k), exactly. One long-division pass over the decimal string, with
 * a running remainder carried in `acc`.
 *
 * `k` is capped per pass: `acc` holds the remainder shifted in with the next digit, so it must stay
 * under 2^k * 10 — at k >= 64 the `acc >> k` below is outright undefined, and well before that it
 * overflows. 28 bits keeps `acc` under ~2.7e9 with room to spare, and the caller's chunking loop
 * makes an arbitrary shift exact. (This only ever ran with k = 1 until subnormal rounding needed a
 * larger one, which is how the UB went unnoticed.) */
#define NIM_SHR_MAX 28
static void nim_dec_shr1(nim_dec *a, int k);
static void nim_dec_shr(nim_dec *a, int k) {
  while (k > NIM_SHR_MAX) { nim_dec_shr1(a, NIM_SHR_MAX); k -= NIM_SHR_MAX; if (a->nd == 0) return; }
  if (k > 0) nim_dec_shr1(a, k);
}
static void nim_dec_shr1(nim_dec *a, int k) {
  int r = 0, w = 0, n = 0;
  /* `static`, not a local: the guest's stack is small and its size is fixed at translate time from
   * the worst-case frame, so a 1.6 KB automatic here inflates every guest that links the shim — the
   * `--child-entry` nifler, which runs in a carve, faulted on entry with it on the stack. The guest
   * is single-threaded and `strtod` is not reentrant in nim's use, so one shared scratch is sound. */
  static unsigned char out[NIM_DEC_CAP];
  unsigned long long acc = 0;
  int rd = 0;
  /* Emit leading digits produced before the first output digit by lowering `dp`. */
  while (acc >> k == 0) {
    if (rd >= a->nd) {
      if (acc == 0) { a->nd = 0; a->dp = 0; a->trunc = 0; return; }
      while (acc >> k == 0) { acc *= 10; n++; }
      break;
    }
    acc = acc * 10 + a->d[rd++];
    n++;
  }
  a->dp -= n - 1;
  for (;;) {
    unsigned long long q = acc >> k;
    r = (int)(acc - (q << k));
    if (w < NIM_DEC_CAP) out[w++] = (unsigned char)q;
    else if (q != 0) a->trunc = 1;
    if (rd >= a->nd) {
      if (r == 0) break;
      acc = (unsigned long long)r * 10;
    } else {
      acc = ((unsigned long long)r) * 10 + a->d[rd++];
    }
  }
  /* Any remainder left when the buffer filled is lost precision. */
  if (r != 0 && w >= NIM_DEC_CAP) a->trunc = 1;
  for (int i = 0; i < w; i++) a->d[i] = out[i];
  a->nd = w;
  nim_dec_trim(a);
}

/* Left-shift by `k` (multiply by 2^k), exactly — one pass, same 28-bit cap and reasoning as
 * [`nim_dec_shr`]: each digit times 2^k plus the carry must stay inside 64 bits. */
#define NIM_SHL_MAX 28
static void nim_dec_shl1(nim_dec *a, int k) {
  if (a->nd == 0) return;
  unsigned long long carry = 0;
  for (int i = a->nd - 1; i >= 0; i--) {
    unsigned long long v = ((unsigned long long)a->d[i] << k) + carry;
    a->d[i] = (unsigned char)(v % 10);
    carry = v / 10;
  }
  /* Emit the carry's digits at the front in one shift rather than one memmove per digit. */
  unsigned char head[24];
  int hn = 0;
  while (carry) { head[hn++] = (unsigned char)(carry % 10); carry /= 10; }
  if (hn) {
    int keep = a->nd;
    if (keep + hn > NIM_DEC_CAP) { keep = NIM_DEC_CAP - hn; a->trunc = 1; }
    for (int i = keep - 1; i >= 0; i--) a->d[i + hn] = a->d[i];
    for (int i = 0; i < hn; i++) a->d[i] = head[hn - 1 - i];
    a->nd = keep + hn;
    a->dp += hn;
  }
  nim_dec_trim(a);
}
static void nim_dec_shl(nim_dec *a, int k) {
  while (k > NIM_SHL_MAX) { nim_dec_shl1(a, NIM_SHL_MAX); k -= NIM_SHL_MAX; if (a->nd == 0) return; }
  if (k > 0) nim_dec_shl1(a, k);
}

/* Compare the digits after the binary point against exactly one half. */
static int nim_dec_cmp_half(const nim_dec *a, int from) {
  if (from >= a->nd) return a->trunc ? 1 : -1; /* nothing left: below half (or just above, if cut) */
  if (a->d[from] != 5) return a->d[from] > 5 ? 1 : -1;
  for (int i = from + 1; i < a->nd; i++)
    if (a->d[i] != 0) return 1;
  return a->trunc ? 1 : 0; /* exactly one half unless digits were dropped */
}

/* The exact fallback: shift the decimal into [2^52, 2^53), read 53 bits, round to nearest-even. */
static double nim_dec_to_double(nim_dec *a) {
  if (a->nd == 0) return 0.0;
  /* Bring the value into [1, 2) by binary shifts, tracking the binary exponent. `dp` is the power of
   * ten; the loops below are exact in both directions. */
  int exp2 = 0;
  for (;;) {
    if (a->dp > 310) return 1.0 / 0.0;  /* beyond DBL_MAX by any rounding */
    if (a->dp < -350) return 0.0;       /* below the smallest subnormal */
    /* Take the shift in chunks sized from `dp` rather than one bit at a time: reaching a subnormal
     * needs ~1075 single-bit passes over a ~1000-digit string, which is ~10^6 digit operations per
     * literal — fine natively, seconds in the interpreted guest that does the crawl. `3` deliberately
     * **under**-estimates log2(10) ≈ 3.32, so a chunk can never overshoot past the target range and
     * the remaining single-bit steps below finish the job. */
    if (a->dp > 1) { int k = (a->dp - 1) * 3; if (k > NIM_SHR_MAX) k = NIM_SHR_MAX; nim_dec_shr(a, k); exp2 += k; continue; }
    if (a->dp <= 0) { int k = (-a->dp) * 3; if (k < 1) k = 1; if (k > NIM_SHL_MAX) k = NIM_SHL_MAX; nim_dec_shl(a, k); exp2 -= k; continue; }
    /* dp == 1: within one binary step of the target. Finish exactly. */
    if (a->d[0] >= 2) { nim_dec_shr(a, 1); exp2++; continue; }
    break; /* dp == 1 and leading digit is 1 => value in [1, 2) */
  }
  /* **Subnormals get fewer than 53 bits**, so the rounding position must be chosen *before* rounding.
   * Normalizing to a full 53-bit significand and then scaling down rounds twice, and the second
   * rounding is not innocent: it was the only remaining disagreement with glibc across 300k random
   * cases (169 of them, every one a subnormal). Shift the extra amount first — exactly — so the
   * single rounding below happens at the representable position. */
  if (exp2 < -1022) {
    int extra = -1022 - exp2;
    if (extra > 1200) return 0.0; /* far below the smallest subnormal */
    nim_dec_shr(a, extra);
    exp2 += extra;
  }
  /* Scale so the integer part is the significand (53 bits when normal, fewer when subnormal). */
  nim_dec_shl(a, 52);
  exp2 -= 52;
  /* Read the integer part (digits before the decimal point, i.e. the first `dp` digits). */
  unsigned long long mant = 0;
  int ip = a->dp;
  if (ip > 19) return exp2 > 0 ? 1.0 / 0.0 : 0.0; /* cannot happen for in-range values */
  for (int i = 0; i < ip; i++) mant = mant * 10 + (i < a->nd ? a->d[i] : 0);
  /* Round to nearest, ties to even, using everything after the point. */
  int c = ip < 0 ? -1 : nim_dec_cmp_half(a, ip);
  if (c > 0 || (c == 0 && (mant & 1))) {
    mant++;
    if (mant >= (1ULL << 53)) { mant >>= 1; exp2++; } /* carried out of 53 bits */
  }
  /* Assemble by scaling — `ldexp` without libc. Subnormals and overflow fall out of the loop bounds. */
  double r = (double)mant;
  while (exp2 > 0) { int st = exp2 > 30 ? 30 : exp2; r *= (double)(1UL << st); exp2 -= st; }
  while (exp2 < 0) { int st = -exp2 > 30 ? 30 : -exp2; r /= (double)(1UL << st); exp2 += st; }
  return r;
}

static const double NIM_POW10[] = {
    1e0,  1e1,  1e2,  1e3,  1e4,  1e5,  1e6,  1e7,  1e8,  1e9,  1e10, 1e11,
    1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22};

double strtod(const char *s, char **endptr) {
  const char *p = s;
  while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == '\f' || *p == '\v') p++;
  int neg = 0;
  if (*p == '+' || *p == '-') { neg = (*p == '-'); p++; }
  unsigned long long mant = 0; /* significand accumulated exactly until it would overflow */
  int exp10 = 0;               /* net power-of-ten scale (fraction digits / dropped int digits / exponent) */
  int any = 0, over = 0;       /* `over`: digits were dropped, so `mant` alone is not the value */
  static nim_dec dec; /* see `nim_dec_shr1` — off the stack, for the same reason */
  dec.nd = 0; dec.dp = 0; dec.trunc = 0;
  int seen_point = 0;
  /* `dp` is the position of the decimal point within the stored digit string (value =
   * 0.d[0]d[1]... * 10^dp), so a **leading zero decrements it** rather than being stored — the
   * convention Gay/Go use. Counting stored digits instead lost the scale of `0.0001` entirely. */
  while (1) {
    if (*p >= '0' && *p <= '9') {
      any = 1;
      if (*p == '0' && dec.nd == 0) {
        dec.dp--; /* leading zero: carries no digit, only scale */
      } else if (dec.nd < NIM_DEC_CAP) {
        dec.d[dec.nd++] = (unsigned char)(*p - '0');
      } else if (*p != '0') {
        dec.trunc = 1; /* past the cap: only a non-zero digit loses information */
      }
      /* u64 fast-path accumulation (independent of the exact-decimal copy above) */
      if (mant < 1000000000000000000ULL) { mant = mant * 10 + (unsigned)(*p - '0'); if (seen_point) exp10--; }
      else { over = 1; if (!seen_point) exp10++; }
      p++;
    } else if (*p == '.' && !seen_point) {
      seen_point = 1;
      dec.dp = dec.nd; /* assignment, not accumulation: the stored digits before the point ARE the
                        * point's position, and it supersedes any leading-zero adjustment above
                        * (`075.5` -> dp = 2, `0.0001` -> dp reset to 0 then decremented to -3). */
      p++;
    } else {
      break;
    }
  }
  if (!any) { if (endptr) *endptr = (char *)s; return 0.0; } /* not a number */
  if (!seen_point) dec.dp = dec.nd; /* integer-only: the point sits past every stored digit */
  if (*p == 'e' || *p == 'E') {
    const char *pe = p + 1;
    int eneg = 0;
    if (*pe == '+' || *pe == '-') { eneg = (*pe == '-'); pe++; }
    if (*pe >= '0' && *pe <= '9') {
      int e = 0;
      while (*pe >= '0' && *pe <= '9') { if (e < 100000) e = e * 10 + (*pe - '0'); pe++; }
      exp10 += eneg ? -e : e;
      dec.dp += eneg ? -e : e;
      p = pe;
    }
  }
  if (endptr) *endptr = (char *)p;
  double result;
  if (mant == 0 && dec.nd == 0) {
    result = 0.0;
  } else if (!over && exp10 >= 0 && exp10 <= 22 && mant < (1ULL << 53)) {
    result = (double)mant * NIM_POW10[exp10]; /* exact operands ⇒ correctly rounded */
  } else if (!over && exp10 < 0 && exp10 >= -22 && mant < (1ULL << 53)) {
    result = (double)mant / NIM_POW10[-exp10]; /* exact operands ⇒ correctly rounded */
  } else {
    nim_dec_trim(&dec);
    result = nim_dec_to_double(&dec); /* exact; rounds exactly once, nearest-even */
  }
  return neg ? -result : result;
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
