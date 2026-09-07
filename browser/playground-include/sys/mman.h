#ifndef __SYS_MMAN_H
#define __SYS_MMAN_H

// <sys/mman.h> for the playground — **anonymous** memory mapping over the granted Memory capability.
// `mmap` with `MAP_ANONYMOUS` hands back page-aligned, zero-filled memory from the same map-growing
// allocator `malloc` uses (`__vm_map` under the hood — see <stdlib.h>); `munmap`/`mprotect` are
// no-ops (MVP: no reclamation, like `free`). File-backed mappings (a real `fd`) are **not** supported
// in the sandbox powerbox — they return `MAP_FAILED`. No authority beyond the Memory capability, all
// guest C.
#include <stdlib.h> // size_t, malloc, __pg_pagesize (the map-growing page allocator)

#define PROT_NONE 0x0
#define PROT_READ 0x1
#define PROT_WRITE 0x2
#define PROT_EXEC 0x4

#define MAP_SHARED 0x01
#define MAP_PRIVATE 0x02
#define MAP_FIXED 0x10
#define MAP_ANONYMOUS 0x20
#define MAP_ANON 0x20

#define MAP_FAILED ((void *)-1L)

// `mmap(addr, len, prot, flags, fd, offset)` — anonymous only. Returns page-aligned memory (whole
// pages, as `mmap` promises) or `MAP_FAILED`. `addr`/`prot`/`offset` are accepted and ignored (the
// mapping is always readable+writable within the confined window); a file `fd` (>= 0) or a
// non-anonymous mapping is refused. Fresh window pages are zero-filled, matching `MAP_ANONYMOUS`.
static inline void *mmap(void *addr, size_t len, int prot, int flags, int fd, long offset) {
  (void)addr;
  (void)prot;
  (void)offset;
  if (fd != -1 || !(flags & MAP_ANONYMOUS) || len == 0) return MAP_FAILED;
  long pg = __pg_pagesize();
  // Over-allocate by a page so the payload can be rounded up to a page boundary.
  char *raw = (char *)malloc(len + (size_t)pg);
  if (!raw) return MAP_FAILED;
  long aligned = ((long)raw + pg - 1) & ~(pg - 1);
  return (void *)aligned;
}

// No reclamation in the MVP allocator (like `free`), so `munmap`/`mprotect` succeed as no-ops.
static inline int munmap(void *addr, size_t len) {
  (void)addr;
  (void)len;
  return 0;
}
static inline int mprotect(void *addr, size_t len, int prot) {
  (void)addr;
  (void)len;
  (void)prot;
  return 0;
}

#endif
