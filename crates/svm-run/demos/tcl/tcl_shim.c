/* Tcl libc shim — the Tcl-specific OS/libc surface the on-ramp neither synthesizes, resolves to an
 * svm-posix capability, nor covers via a reused shim. Linked into the guest build (and the native
 * oracle, so the differential stays honest). Everything here is ordinary guest C; a guest definition
 * shadows the on-ramp's would-be trap stub (same discipline as `../quickjs/libc_shim.c`).
 *
 * The reuse map (see README) — NOT redefined here:
 *   - printf/stdio family  → `../postgres/printf_shim.c` + `../postgres/stdio_shim.c`
 *   - sscanf               → `../postgres/scanf_shim.c`   (`__isoc99_sscanf`)
 *   - ctype                → the Postgres ctype tables        (`__ctype_b_loc` / `__ctype_tolower_loc`)
 *   - strtod               → `../strtod/strtod.c`
 *   - libm transcendentals → guest openlibm (the QuickJS slice CO mechanism)
 *   - mem/string/qsort/malloc/llvm.* → on-ramp-synthesized
 *   - open/read/write/close/lseek/stat/opendir/readdir/getcwd/chdir/getenv/setenv/unlink/exit
 *                          → svm-posix capabilities (POSIX.md ops 0–20), resolved at load
 *
 * What remains — and lives here — is Tcl's own OS entanglement: the `clock` time surface, the tty /
 * termios probes Tcl's channel layer runs at startup, locale, and the socket / process / extra-fs
 * surface that the MINIMAL (no-`Tcl_Init`) REPL never reaches. Those last are stubbed to clean
 * errors, not escapes: Tcl's `socket`/`exec`/`file` commands would return a Tcl error, exactly as a
 * platform without those facilities does. Wiring the fuller surface (real `clock`, `file`/`glob` over
 * the fs cap) is the documented "Full Tcl_Init" follow-up.
 */
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>

extern void exit(int);

/* --- errno ------------------------------------------------------------------------------------ */
static int __tcl_errno;
int *__errno_location(void) { return &__tcl_errno; }

/* --- time: deterministic stubs (fixed epoch) --------------------------------------------------
 * Tcl's `clock` command and channel timestamps read the wall clock; a differential-vs-native demo
 * must be deterministic (the same choice `../quickjs/libc_shim.c` and SQLite's fixed-clock VFS make),
 * and the REPL driver's output does not depend on the time. A real `clock` would bind a host time
 * capability. struct layouts match the x86-64 Linux ABI. */
struct __shim_timeval {
    long tv_sec;
    long tv_usec;
};
struct __shim_timespec {
    long tv_sec;
    long tv_nsec;
};
struct __shim_tm {
    int tm_sec, tm_min, tm_hour, tm_mday, tm_mon, tm_year, tm_wday, tm_yday, tm_isdst;
    long tm_gmtoff;
    const char *tm_zone;
};

long time(long *t) {
    if (t)
        *t = 0;
    return 0;
}
int gettimeofday(struct __shim_timeval *tv, void *tz) {
    (void)tz;
    if (tv) {
        tv->tv_sec = 0;
        tv->tv_usec = 0;
    }
    return 0;
}
int clock_gettime(int clk, struct __shim_timespec *ts) {
    (void)clk;
    if (ts) {
        ts->tv_sec = 0;
        ts->tv_nsec = 0;
    }
    return 0;
}
static struct __shim_tm *fill_epoch(struct __shim_tm *out) {
    if (out) {
        out->tm_sec = out->tm_min = out->tm_hour = 0;
        out->tm_mday = 1; /* the epoch: 1970-01-01 00:00:00 UTC, Thursday */
        out->tm_mon = 0;
        out->tm_year = 70;
        out->tm_wday = 4;
        out->tm_yday = 0;
        out->tm_isdst = 0;
        out->tm_gmtoff = 0;
        out->tm_zone = 0;
    }
    return out;
}
struct __shim_tm *localtime_r(const long *t, struct __shim_tm *out) {
    (void)t;
    return fill_epoch(out);
}
struct __shim_tm *gmtime_r(const long *t, struct __shim_tm *out) {
    (void)t;
    return fill_epoch(out);
}
long mktime(struct __shim_tm *tm) {
    (void)tm;
    return 0;
}
void tzset(void) {}

/* --- strtol / strtoul ------------------------------------------------------------------------- */
static long parse_long(const char *s, char **end, int base, int is_signed) {
    const char *p = s;
    while (*p == ' ' || (*p >= '\t' && *p <= '\r'))
        p++;
    int neg = 0;
    if (*p == '+' || *p == '-')
        neg = (*p++ == '-');
    if ((base == 0 || base == 16) && p[0] == '0' && (p[1] == 'x' || p[1] == 'X')) {
        p += 2;
        base = 16;
    } else if (base == 0) {
        base = (p[0] == '0') ? 8 : 10;
    }
    unsigned long acc = 0;
    int any = 0;
    for (;; p++) {
        int c = (unsigned char)*p, d;
        if (c >= '0' && c <= '9')
            d = c - '0';
        else if (c >= 'a' && c <= 'z')
            d = c - 'a' + 10;
        else if (c >= 'A' && c <= 'Z')
            d = c - 'A' + 10;
        else
            break;
        if (d >= base)
            break;
        acc = acc * (unsigned long)base + (unsigned long)d;
        any = 1;
    }
    if (end)
        *end = (char *)(any ? p : s);
    (void)is_signed;
    long v = (long)acc;
    return neg ? -v : v;
}
long strtol(const char *s, char **end, int base) { return parse_long(s, end, base, 1); }
unsigned long strtoul(const char *s, char **end, int base) {
    return (unsigned long)parse_long(s, end, base, 0);
}
/* The C23-renamed aliases glibc >= 2.38 emits. */
long __isoc23_strtol(const char *s, char **end, int base) { return strtol(s, end, base); }
unsigned long __isoc23_strtoul(const char *s, char **end, int base) { return strtoul(s, end, base); }

/* --- address-taken string compares -----------------------------------------------------------
 * The on-ramp synthesizes strcmp/strncmp/memcmp at *call* sites, so they have no funcref address.
 * Tcl passes them **as function pointers** (comparators to `qsort`, `Tcl_LsearchObjCmd`, sort
 * routines), so `&strcmp` needs a real body. A guest definition supplies both the calls and the
 * address (it shadows the on-ramp's inline builtin). Same pattern as QuickJS's address-taken libm. */
int strcmp(const char *a, const char *b) {
    while (*a && *a == *b) {
        a++;
        b++;
    }
    return (int)(unsigned char)*a - (int)(unsigned char)*b;
}
int strncmp(const char *a, const char *b, size_t n) {
    for (; n; n--, a++, b++) {
        if (*a != *b)
            return (int)(unsigned char)*a - (int)(unsigned char)*b;
        if (!*a)
            break;
    }
    return 0;
}
int memcmp(const void *a, const void *b, size_t n) {
    const unsigned char *p = a, *q = b;
    for (; n; n--, p++, q++)
        if (*p != *q)
            return (int)*p - (int)*q;
    return 0;
}

/* --- address-taken / remaining string + mem ops ----------------------------------------------
 * The on-ramp synthesizes these inline at call sites (no funcref address). Tcl stores several as
 * struct function pointers — the encoding `lengthProc` is `&strlen`, filesystem tables hold
 * `&strcpy`/`&strlen`, etc. — so each needs a real guest body. Providing them real (shadowing the
 * inline synthesis) supplies both the calls and the addresses; the mem ops coexist with the
 * `llvm.memcpy`/`memset` intrinsics (a distinct mechanism). */
size_t strlen(const char *s) {
    const char *p = s;
    while (*p)
        p++;
    return (size_t)(p - s);
}
char *strcpy(char *dst, const char *src) {
    char *d = dst;
    while ((*d++ = *src++)) {
    }
    return dst;
}
char *strchr(const char *s, int c) {
    for (;; s++) {
        if (*s == (char)c)
            return (char *)s;
        if (!*s)
            return 0;
    }
}
char *strrchr(const char *s, int c) {
    const char *last = 0;
    for (;; s++) {
        if (*s == (char)c)
            last = s;
        if (!*s)
            return (char *)last;
    }
}
void *memchr(const void *s, int c, size_t n) {
    const unsigned char *p = s;
    for (; n; n--, p++)
        if (*p == (unsigned char)c)
            return (void *)p;
    return 0;
}
void *memcpy(void *dst, const void *src, size_t n) {
    unsigned char *d = dst;
    const unsigned char *s = src;
    while (n--)
        *d++ = *s++;
    return dst;
}
void *memmove(void *dst, const void *src, size_t n) {
    unsigned char *d = dst;
    const unsigned char *s = src;
    if (d < s)
        while (n--)
            *d++ = *s++;
    else
        while (n--)
            d[n] = s[n];
    return dst;
}
void *memset(void *dst, int c, size_t n) {
    unsigned char *d = dst;
    while (n--)
        *d++ = (unsigned char)c;
    return dst;
}
size_t strspn(const char *s, const char *set) {
    const char *p = s;
    for (; *p; p++) {
        const char *t = set;
        while (*t && *t != *p)
            t++;
        if (!*t)
            break;
    }
    return (size_t)(p - s);
}
size_t strcspn(const char *s, const char *set) {
    const char *p = s;
    for (; *p; p++)
        for (const char *t = set; *t; t++)
            if (*t == *p)
                return (size_t)(p - s);
    return (size_t)(p - s);
}

/* --- other string ops ------------------------------------------------------------------------- */
char *strncpy(char *dst, const char *src, size_t n) {
    size_t i = 0;
    for (; i < n && src[i]; i++)
        dst[i] = src[i];
    for (; i < n; i++)
        dst[i] = 0;
    return dst;
}
char *strcat(char *dst, const char *src) {
    char *d = dst;
    while (*d)
        d++;
    while ((*d++ = *src++)) {
    }
    return dst;
}
char *strncat(char *dst, const char *src, size_t n) {
    char *d = dst;
    while (*d)
        d++;
    while (n-- && *src)
        *d++ = *src++;
    *d = 0;
    return dst;
}
char *strstr(const char *hay, const char *needle) {
    if (!*needle)
        return (char *)hay;
    for (; *hay; hay++) {
        const char *h = hay, *n = needle;
        while (*h && *n && *h == *n) {
            h++;
            n++;
        }
        if (!*n)
            return (char *)hay;
    }
    return 0;
}
char *strpbrk(const char *s, const char *set) {
    for (; *s; s++)
        for (const char *t = set; *t; t++)
            if (*s == *t)
                return (char *)s;
    return 0;
}

/* --- case-insensitive string compares --------------------------------------------------------- */
static int lc(int c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }
int strcasecmp(const char *a, const char *b) {
    while (*a && lc((unsigned char)*a) == lc((unsigned char)*b)) {
        a++;
        b++;
    }
    return lc((unsigned char)*a) - lc((unsigned char)*b);
}
int strncasecmp(const char *a, const char *b, size_t n) {
    for (; n; n--, a++, b++) {
        int d = lc((unsigned char)*a) - lc((unsigned char)*b);
        if (d || !*a)
            return d;
    }
    return 0;
}

/* --- tty / termios: "not a terminal" -----------------------------------------------------------
 * Tcl's channel layer probes stdin/stdout/stderr with isatty()/tcgetattr() at startup to decide
 * buffering and echo. In the sandbox the standard channels are the non-tty Stream capability, so
 * report exactly that (isatty → 0), and make the termios calls harmless no-ops. */
int isatty(int fd) {
    (void)fd;
    return 0;
}
int tcgetattr(int fd, void *t) {
    (void)fd;
    (void)t;
    return -1; /* ENOTTY-ish: not a terminal */
}
int tcsetattr(int fd, int act, const void *t) {
    (void)fd;
    (void)act;
    (void)t;
    return -1;
}
unsigned int cfgetospeed(const void *t) {
    (void)t;
    return 0;
}
int cfsetospeed(void *t, unsigned int s) {
    (void)t;
    (void)s;
    return 0;
}
int cfsetispeed(void *t, unsigned int s) {
    (void)t;
    (void)s;
    return 0;
}
int ioctl(int fd, unsigned long req, ...) {
    (void)fd;
    (void)req;
    return -1;
}

/* --- locale: C / UTF-8 only ------------------------------------------------------------------- */
char *setlocale(int cat, const char *loc) {
    (void)cat;
    (void)loc;
    static char c[2] = "C";
    return c;
}
char *nl_langinfo(int item) {
    (void)item;
    static char empty[1] = "";
    return empty; /* Tcl falls back to its built-in iso8859-1/utf-8 handling */
}

/* --- identity / system: single anonymous user ------------------------------------------------- */
int getpid(void) { return 1; }
unsigned int getuid(void) { return 0; }
unsigned int geteuid(void) { return 0; }
unsigned int getgid(void) { return 0; }
unsigned int getegid(void) { return 0; }
void *getpwuid(unsigned int uid) {
    (void)uid;
    return 0;
}
void *getpwnam(const char *n) {
    (void)n;
    return 0;
}
void *getgrgid(unsigned int gid) {
    (void)gid;
    return 0;
}
void *getgrnam(const char *n) {
    (void)n;
    return 0;
}
int uname(void *buf) {
    (void)buf;
    return -1;
}

/* --- process / exec: unavailable in the minimal REPL -------------------------------------------
 * `exec`, `open |pipe`, and background processes want fork/exec/pipe/wait. Stub to failure so Tcl's
 * `exec` raises a normal Tcl error ("couldn't ...") rather than escaping. Wiring these to the §14
 * Instantiator / Pipe capabilities is future work (Stage-1 posix_spawn, see STAGE1.md/POSIX.md). */
int fork(void) { return -1; }
int vfork(void) { return -1; }
int execvp(const char *f, char *const argv[]) {
    (void)f;
    (void)argv;
    return -1;
}
int waitpid(int pid, int *st, int opt) {
    (void)pid;
    (void)st;
    (void)opt;
    return -1;
}
int pipe(int fds[2]) {
    (void)fds;
    return -1;
}
int dup2(int a, int b) {
    (void)a;
    (void)b;
    return -1;
}
void (*signal(int sig, void (*h)(int)))(int) {
    (void)sig;
    (void)h;
    return 0;
}

/* --- sockets: unavailable in the minimal REPL -------------------------------------------------
 * The `socket` command wants BSD sockets; stub to failure so it raises a Tcl error. Networking as a
 * host capability is future work. */
int socket(int d, int t, int p) {
    (void)d;
    (void)t;
    (void)p;
    return -1;
}
int connect(int s, const void *a, unsigned l) {
    (void)s;
    (void)a;
    (void)l;
    return -1;
}
int bind(int s, const void *a, unsigned l) {
    (void)s;
    (void)a;
    (void)l;
    return -1;
}
int listen(int s, int b) {
    (void)s;
    (void)b;
    return -1;
}
int accept(int s, void *a, unsigned *l) {
    (void)s;
    (void)a;
    (void)l;
    return -1;
}
long send(int s, const void *b, size_t n, int f) {
    (void)s;
    (void)b;
    (void)n;
    (void)f;
    return -1;
}
long recv(int s, void *b, size_t n, int f) {
    (void)s;
    (void)b;
    (void)n;
    (void)f;
    return -1;
}
int setsockopt(int s, int l, int o, const void *v, unsigned n) {
    (void)s;
    (void)l;
    (void)o;
    (void)v;
    (void)n;
    return -1;
}
int getsockopt(int s, int l, int o, void *v, unsigned *n) {
    (void)s;
    (void)l;
    (void)o;
    (void)v;
    (void)n;
    return -1;
}
int getsockname(int s, void *a, unsigned *l) {
    (void)s;
    (void)a;
    (void)l;
    return -1;
}
int getpeername(int s, void *a, unsigned *l) {
    (void)s;
    (void)a;
    (void)l;
    return -1;
}
int shutdown(int s, int how) {
    (void)s;
    (void)how;
    return -1;
}
int getaddrinfo(const char *n, const char *s, const void *h, void **r) {
    (void)n;
    (void)s;
    (void)h;
    (void)r;
    return -1;
}
void freeaddrinfo(void *r) { (void)r; }
const char *gai_strerror(int e) {
    (void)e;
    return "name resolution unavailable";
}
int getnameinfo(const void *a, unsigned l, char *h, unsigned hl, char *s, unsigned sl, int f) {
    (void)a;
    (void)l;
    (void)h;
    (void)hl;
    (void)s;
    (void)sl;
    (void)f;
    return -1;
}
void *gethostbyname(const char *n) {
    (void)n;
    return 0;
}
void *gethostbyaddr(const void *a, unsigned l, int t) {
    (void)a;
    (void)l;
    (void)t;
    return 0;
}
void *getservbyname(const char *n, const char *p) {
    (void)n;
    (void)p;
    return 0;
}
char *inet_ntoa(unsigned int in) {
    (void)in;
    static char buf[1] = "";
    return buf;
}

/* --- extra filesystem surface (beyond the svm-posix caps) --------------------------------------
 * The `file` command's metadata mutators and `glob`'s directory walk. Unreached by the minimal REPL
 * (no `Tcl_Init`, no `file`/`glob`); stub to failure so any use is a clean Tcl error, not an escape.
 * The read path (open/read/stat/opendir/readdir/getcwd) is the real svm-posix cap. */
int access(const char *p, int m) {
    (void)p;
    (void)m;
    return -1;
}
int lstat(const char *p, void *b) {
    (void)p;
    (void)b;
    return -1;
}
int chmod(const char *p, unsigned m) {
    (void)p;
    (void)m;
    return -1;
}
int chown(const char *p, unsigned u, unsigned g) {
    (void)p;
    (void)u;
    (void)g;
    return -1;
}
int mkdir(const char *p, unsigned m) {
    (void)p;
    (void)m;
    return -1;
}
int rmdir(const char *p) {
    (void)p;
    return -1;
}
int rename(const char *a, const char *b) {
    (void)a;
    (void)b;
    return -1;
}
long readlink(const char *p, char *b, size_t n) {
    (void)p;
    (void)b;
    (void)n;
    return -1;
}
char *realpath(const char *p, char *out) {
    (void)p;
    (void)out;
    return 0;
}
unsigned umask(unsigned m) {
    (void)m;
    return 0;
}
int ftruncate(int fd, long len) {
    (void)fd;
    (void)len;
    return -1;
}
int link(const char *a, const char *b) {
    (void)a;
    (void)b;
    return -1;
}
int symlink(const char *a, const char *b) {
    (void)a;
    (void)b;
    return -1;
}
int fcntl(int fd, int cmd, ...) {
    (void)fd;
    (void)cmd;
    return 0;
}

/* --- qsort / bsearch -------------------------------------------------------------------------
 * Pure-computation libc the on-ramp does not synthesize. Tcl calls `qsort` in the regex NFA
 * compaction (`regc_nfa.c`) and the ensemble/TclOO command tables. Implemented as an in-place
 * **heapsort** — O(n log n) worst case, no recursion-depth or pivot-degeneracy concerns, and no
 * scratch allocation — over arbitrary element sizes with a byte-wise swap. The comparator is called
 * through a function pointer (→ `call_indirect`). Linked into the guest *and* the native oracle (like
 * the other shims), so the sort order is identical on both sides of the differential regardless of
 * how each libc's own `qsort` breaks ties. */
static void __bswap(unsigned char *a, unsigned char *b, size_t n) {
    while (n--) {
        unsigned char t = *a;
        *a++ = *b;
        *b++ = t;
    }
}
void qsort(void *base, size_t n, size_t size, int (*cmp)(const void *, const void *)) {
    unsigned char *a = (unsigned char *)base;
    if (n < 2 || size == 0)
        return;
    /* Build a max-heap, then repeatedly extract the max to the end (sift-down heapsort). */
    for (size_t start = n / 2; start-- > 0;) {
        size_t root = start;
        for (;;) {
            size_t child = 2 * root + 1;
            if (child >= n)
                break;
            if (child + 1 < n && cmp(a + child * size, a + (child + 1) * size) < 0)
                child++;
            if (cmp(a + root * size, a + child * size) >= 0)
                break;
            __bswap(a + root * size, a + child * size, size);
            root = child;
        }
    }
    for (size_t end = n; end-- > 1;) {
        __bswap(a, a + end * size, size);
        size_t root = 0;
        for (;;) {
            size_t child = 2 * root + 1;
            if (child >= end)
                break;
            if (child + 1 < end && cmp(a + child * size, a + (child + 1) * size) < 0)
                child++;
            if (cmp(a + root * size, a + child * size) >= 0)
                break;
            __bswap(a + root * size, a + child * size, size);
            root = child;
        }
    }
}
void *bsearch(const void *key, const void *base, size_t n, size_t size,
              int (*cmp)(const void *, const void *)) {
    const unsigned char *a = (const unsigned char *)base;
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        int c = cmp(key, a + mid * size);
        if (c < 0)
            hi = mid;
        else if (c > 0)
            lo = mid + 1;
        else
            return (void *)(a + mid * size);
    }
    return 0;
}

/* --- strerror --------------------------------------------------------------------------------
 * Tcl's `Tcl_ErrnoMsg` maps errno → message for its error results. A compact table of the common
 * values (the ones the fs/exec/socket stubs and the posix caps produce); anything else gets a
 * generic string. Linked into guest + oracle, so the messages agree on both sides of the
 * differential. */
char *strerror(int e) {
    static char unknown[] = "Unknown error";
    switch (e) {
    case 0:
        return (char *)"Success";
    case 1:
        return (char *)"Operation not permitted";
    case 2:
        return (char *)"No such file or directory";
    case 9:
        return (char *)"Bad file descriptor";
    case 12:
        return (char *)"Cannot allocate memory";
    case 13:
        return (char *)"Permission denied";
    case 17:
        return (char *)"File exists";
    case 20:
        return (char *)"Not a directory";
    case 21:
        return (char *)"Is a directory";
    case 22:
        return (char *)"Invalid argument";
    case 28:
        return (char *)"No space left on device";
    case 38:
        return (char *)"Function not implemented";
    default:
        return unknown;
    }
}

/* --- ctype tables (glibc `__ctype_*_loc` ABI) ------------------------------------------------
 * Tcl's tokenizer/`expr`/`string` use the `<ctype.h>` `isalpha`/`isdigit`/`isspace`/… macros,
 * which glibc expands to `(*__ctype_b_loc())[c] & _ISbit`. The tables + accessors below are the
 * glibc ABI (bits/ctype-info.h `_ISbit` values), reused verbatim from `../postgres/libc_shim.c`
 * (differentially validated there over every byte 0..255). A guest def shadows the on-ramp's
 * trap stub; both guest and oracle link it, so classification agrees on both sides. */
static const unsigned short shim_ctype_b[384] = {
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0002, 0x0002, 0x0002, 0x0002,
  0x0002, 0x0002, 0x0002, 0x0002, 0x0002, 0x2003, 0x2002, 0x2002, 0x2002, 0x2002, 0x0002, 0x0002,
  0x0002, 0x0002, 0x0002, 0x0002, 0x0002, 0x0002, 0x0002, 0x0002, 0x0002, 0x0002, 0x0002, 0x0002,
  0x0002, 0x0002, 0x0002, 0x0002, 0x6001, 0xc004, 0xc004, 0xc004, 0xc004, 0xc004, 0xc004, 0xc004,
  0xc004, 0xc004, 0xc004, 0xc004, 0xc004, 0xc004, 0xc004, 0xc004, 0xd808, 0xd808, 0xd808, 0xd808,
  0xd808, 0xd808, 0xd808, 0xd808, 0xd808, 0xd808, 0xc004, 0xc004, 0xc004, 0xc004, 0xc004, 0xc004,
  0xc004, 0xd508, 0xd508, 0xd508, 0xd508, 0xd508, 0xd508, 0xc508, 0xc508, 0xc508, 0xc508, 0xc508,
  0xc508, 0xc508, 0xc508, 0xc508, 0xc508, 0xc508, 0xc508, 0xc508, 0xc508, 0xc508, 0xc508, 0xc508,
  0xc508, 0xc508, 0xc508, 0xc004, 0xc004, 0xc004, 0xc004, 0xc004, 0xc004, 0xd608, 0xd608, 0xd608,
  0xd608, 0xd608, 0xd608, 0xc608, 0xc608, 0xc608, 0xc608, 0xc608, 0xc608, 0xc608, 0xc608, 0xc608,
  0xc608, 0xc608, 0xc608, 0xc608, 0xc608, 0xc608, 0xc608, 0xc608, 0xc608, 0xc608, 0xc608, 0xc004,
  0xc004, 0xc004, 0xc004, 0x0002, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
  0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000
};
static const int shim_ctype_tolower[384] = {
  -128, -127, -126, -125, -124, -123, -122, -121, -120, -119, -118, -117,
  -116, -115, -114, -113, -112, -111, -110, -109, -108, -107, -106, -105,
  -104, -103, -102, -101, -100, -99, -98, -97, -96, -95, -94, -93,
  -92, -91, -90, -89, -88, -87, -86, -85, -84, -83, -82, -81,
  -80, -79, -78, -77, -76, -75, -74, -73, -72, -71, -70, -69,
  -68, -67, -66, -65, -64, -63, -62, -61, -60, -59, -58, -57,
  -56, -55, -54, -53, -52, -51, -50, -49, -48, -47, -46, -45,
  -44, -43, -42, -41, -40, -39, -38, -37, -36, -35, -34, -33,
  -32, -31, -30, -29, -28, -27, -26, -25, -24, -23, -22, -21,
  -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9,
  -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3,
  4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
  16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
  28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
  40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
  52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
  64, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107,
  108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
  120, 121, 122, 91, 92, 93, 94, 95, 96, 97, 98, 99,
  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
  112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
  124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135,
  136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147,
  148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
  160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171,
  172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183,
  184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195,
  196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
  208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219,
  220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231,
  232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243,
  244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255
};
static const int shim_ctype_toupper[384] = {
  -128, -127, -126, -125, -124, -123, -122, -121, -120, -119, -118, -117,
  -116, -115, -114, -113, -112, -111, -110, -109, -108, -107, -106, -105,
  -104, -103, -102, -101, -100, -99, -98, -97, -96, -95, -94, -93,
  -92, -91, -90, -89, -88, -87, -86, -85, -84, -83, -82, -81,
  -80, -79, -78, -77, -76, -75, -74, -73, -72, -71, -70, -69,
  -68, -67, -66, -65, -64, -63, -62, -61, -60, -59, -58, -57,
  -56, -55, -54, -53, -52, -51, -50, -49, -48, -47, -46, -45,
  -44, -43, -42, -41, -40, -39, -38, -37, -36, -35, -34, -33,
  -32, -31, -30, -29, -28, -27, -26, -25, -24, -23, -22, -21,
  -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9,
  -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3,
  4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
  16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
  28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
  40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
  52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
  64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
  76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87,
  88, 89, 90, 91, 92, 93, 94, 95, 96, 65, 66, 67,
  68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
  80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 123,
  124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135,
  136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147,
  148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
  160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171,
  172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183,
  184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195,
  196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
  208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219,
  220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231,
  232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243,
  244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255
};

/* Each pointer is initialized *at load time* to element 128 (code point 0) of its table — a
 * compile-time-constant address, so no initializer function runs in the guest. */
static const unsigned short *shim_ctype_b_ptr = shim_ctype_b + 128;
static const int *shim_ctype_tolower_ptr = shim_ctype_tolower + 128;
static const int *shim_ctype_toupper_ptr = shim_ctype_toupper + 128;

const unsigned short **__ctype_b_loc(void) { return &shim_ctype_b_ptr; }
const int **__ctype_tolower_loc(void) { return &shim_ctype_tolower_ptr; }
const int **__ctype_toupper_loc(void) { return &shim_ctype_toupper_ptr; }

/* --- POSIX file/dir surface (benign, not reached by the minimal REPL) ------------------------
 * These are compiled into Tcl's channel/filesystem layer but unreached without `Tcl_Init` /
 * `file`/`glob`/`open`. Real file I/O would bind the svm-posix caps at load; here they return clean
 * errors so any use is a Tcl error, and — crucially — a **defined** function rather than a trap
 * stub, so Tcl's startup channel probing never faults. `read`/`write` on the standard channels are
 * the recognized `Stream` capability (not stubbed). */
int open(const char *p, int flags, ...) {
    (void)p;
    (void)flags;
    return -1;
}
int close(int fd) {
    (void)fd;
    return 0;
}
long lseek(int fd, long off, int whence) {
    (void)fd;
    (void)off;
    (void)whence;
    return -1;
}
int fstat(int fd, void *buf) {
    (void)fd;
    (void)buf;
    return -1; /* not stat-able; Tcl's channel layer tolerates a failed fstat on a stream */
}
int stat(const char *p, void *buf) {
    (void)p;
    (void)buf;
    return -1;
}
char *getcwd(char *buf, size_t n) {
    if (buf && n) {
        buf[0] = '/';
        if (n > 1)
            buf[1] = 0;
    }
    return buf;
}
int chdir(const char *p) {
    (void)p;
    return -1;
}
void *opendir(const char *p) {
    (void)p;
    return 0;
}
void *readdir(void *d) {
    (void)d;
    return 0;
}
int closedir(void *d) {
    (void)d;
    return 0;
}
int unlink(const char *p) {
    (void)p;
    return -1;
}
void *fdopen(int fd, const char *m) {
    (void)fd;
    (void)m;
    return 0;
}
int mkstemp(char *t) {
    (void)t;
    return -1;
}
int mkstemps(char *t, int s) {
    (void)t;
    (void)s;
    return -1;
}
int mkfifo(const char *p, unsigned m) {
    (void)p;
    (void)m;
    return -1;
}
int mknod(const char *p, unsigned m, unsigned long d) { /* dev_t is 64-bit */
    (void)p;
    (void)m;
    (void)d;
    return -1;
}
int utime(const char *p, const void *t) {
    (void)p;
    (void)t;
    return -1;
}
int select(int n, void *r, void *w, void *e, void *t) {
    (void)n;
    (void)r;
    (void)w;
    (void)e;
    (void)t;
    return 0;
}
void *fts_open(char *const *p, int o, void *c) {
    (void)p;
    (void)o;
    (void)c;
    return 0;
}
void *fts_read(void *f) {
    (void)f;
    return 0;
}
int fts_close(void *f) {
    (void)f;
    return 0;
}

/* --- misc ------------------------------------------------------------------------------------- */
void _exit(int code) {
    exit(code);
    for (;;) {
    }
}
void abort(void) {
    exit(134); /* 128 + SIGABRT */
    for (;;) {
    }
}
