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
