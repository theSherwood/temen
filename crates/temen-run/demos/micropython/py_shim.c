/* MicroPython libc/HAL waist for the LLVM on-ramp — the small surface the on-ramp neither synthesizes
 * nor resolves to a capability, mirroring `demos/quickjs/libc_shim.c` and `demos/tcl/tcl_shim.c`. All
 * guest code (outside the escape-TCB): the verifier re-checks the emitted IR.
 */
#include <stddef.h>

extern long write(int fd, const void *buf, long n);

/* HAL override (replaces the embed port's `port/mphalport.c`, which is dropped from the link).
 *
 * The stock HAL does `printf("%.*s", (int)len, str)` — a *dynamic*-precision constant format the
 * on-ramp's inline printf lowering rejects (`Unsupported: printf dynamic precision (.*)`). Routing
 * cooked stdout straight to the Stream write capability sidesteps printf entirely and is what
 * `print(...)` output flows through. */
void mp_hal_stdout_tx_strn_cooked(const char *str, size_t len) {
    write(1, str, (long)len);
}

/* String/mem helpers MicroPython uses that the on-ramp does not synthesize (it provides
 * memcpy/memmove/memset; these are the rest). Reused-elsewhere shims (postgres printf/scanf, strtod)
 * cover the runtime-format stdio waist if a wider config references it. */
size_t strlen(const char *s) {
    const char *p = s;
    while (*p)
        p++;
    return (size_t)(p - s);
}

int strcmp(const char *a, const char *b) {
    while (*a && *a == *b) {
        a++;
        b++;
    }
    return (int)(unsigned char)*a - (int)(unsigned char)*b;
}

int strncmp(const char *a, const char *b, size_t n) {
    for (; n && *a && *a == *b; n--, a++, b++) {
    }
    return n ? (int)(unsigned char)*a - (int)(unsigned char)*b : 0;
}

int memcmp(const void *a, const void *b, size_t n) {
    const unsigned char *x = a, *y = b;
    for (; n; n--, x++, y++)
        if (*x != *y)
            return (int)*x - (int)*y;
    return 0;
}

int bcmp(const void *a, const void *b, size_t n) {
    return memcmp(a, b, n);
}

void *memchr(const void *s, int c, size_t n) {
    const unsigned char *p = s;
    for (; n; n--, p++)
        if (*p == (unsigned char)c)
            return (void *)p;
    return NULL;
}

char *strchr(const char *s, int c) {
    for (;; s++) {
        if (*s == (char)c)
            return (char *)s;
        if (!*s)
            return NULL;
    }
}
