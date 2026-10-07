/* The on-ramp's C heap: Doug Lea's dlmalloc 2.8.6 (`malloc.c`, vendored verbatim, MIT-0), configured
 * for a Temen guest window. `gen.sh` compiles this file to `dlmalloc.ll`, which the on-ramp merges into
 * any guest that calls the `malloc` family without defining it (LLVM.md slice S, #1603).
 *
 * The heap grows through `__temen_sbrk`, which the on-ramp synthesizes: it commits pages above the
 * window's mapped boundary through the `Memory` capability and returns the old break, or `MFAIL`
 * (`(void *)-1`) when the window cannot grow. dlmalloc calls it only while holding its global lock,
 * so it needs no lock of its own. */
#include <stddef.h>

void *__temen_sbrk(ptrdiff_t increment);

/* `dlmalloc`, `dlfree`, … — never the libc names, so clang cannot fold this code's own calls into
 * builtins; the on-ramp aliases the libc names a guest uses onto these. */
#define USE_DL_PREFIX 1
/* One contiguous heap grown by `__temen_sbrk`; there is no `mmap` to fall back on, and the window is
 * never given back. */
#define HAVE_MMAP 0
#define HAVE_MREMAP 0
#define MORECORE __temen_sbrk
#define MORECORE_CONTIGUOUS 1
#define MORECORE_CANNOT_TRIM 1
#define DEFAULT_GRANULARITY ((size_t)64U * (size_t)1024U)
#define malloc_getpagesize ((size_t)4096U)
/* vCPUs share one heap (#1097): a spin lock on atomics, with no `sched_yield` to call. */
#define USE_LOCKS 1
#define USE_SPIN_LOCKS 1
#define LACKS_SCHED_H 1
/* Deterministic: no `time()` seed for the chunk magic, and no stdio. */
#define LACKS_TIME_H 1
#define NO_MALLOC_STATS 1
#define NO_MALLINFO 1

#include "malloc.c"
