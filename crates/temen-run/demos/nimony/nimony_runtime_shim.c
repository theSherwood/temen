// Minimal bottom-edge shim for on-ramping nimony's libc-free C output (NIM.md §2 step 3).
// nimony allocates via mmap and masks returned addresses to 4096 to find chunk headers, so
// mmap must return absolutely page-aligned pointers. Bump allocator over a static window arena.
// The alignment runs on a RUNTIME pointer (g_next), not a folded ptrtoint constant, which the
// on-ramp rejects as a constant expression.
#include <stddef.h>
extern void exit(int);

#define ARENA_BYTES (8u << 20)
_Alignas(4096) static unsigned char g_arena[ARENA_BYTES + 4096];
static unsigned char *g_next = 0;

void *mmap(void *addr, unsigned long length, int prot, int flags, int fd, long offset) {
    (void)addr; (void)prot; (void)flags; (void)fd; (void)offset;
    if (g_next == 0) g_next = g_arena;
    unsigned long a = ((unsigned long)g_next + 4095ul) & ~4095ul;   // runtime ptrtoint, not const.
    unsigned char *p = (unsigned char *)a;
    if (p + length > g_arena + ARENA_BYTES) return (void *)-1;      // MAP_FAILED (GEP compare)
    g_next = p + length;
    return p;
}

int getpid(void) { return 1; }
int kill(int pid, int sig) { (void)pid; (void)sig; return 0; }
void *dlopen(const char *p, int f) { (void)p; (void)f; return NULL; }
void *dlsym(void *h, const char *s) { (void)h; (void)s; return NULL; }
void _exit(int code) { exit(code); }
