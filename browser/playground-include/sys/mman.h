#ifndef __SYS_MMAN_H
#define __SYS_MMAN_H

// <sys/mman.h> for the playground: memory mapping over the granted Memory capability, and over the
// `vm_fs` memfs for a file.
//
// - `MAP_ANONYMOUS` hands back page-aligned, zero-filled memory from the heap (`calloc`; see
//   <stdlib.h>).
// - A file mapping gets the same fresh pages with the file's bytes copied in. `MAP_PRIVATE` stops
//   there. `MAP_SHARED` registers the pages with the memfs (its `FS_MMAP`), which copies them back to
//   the file on `msync`, on `munmap`, and when the file's descriptor is closed.
//
// Unmapping reclaims nothing, and `mprotect` is a no-op: the mapping is always readable and writable
// within the confined window. No authority beyond the Memory and `vm_fs` capabilities; all
// guest C.
#include <stdlib.h> // size_t, calloc, __vm_page_size

#define PROT_NONE 0x0
#define PROT_READ 0x1
#define PROT_WRITE 0x2
#define PROT_EXEC 0x4

#define MAP_SHARED 0x01
#define MAP_PRIVATE 0x02
#define MAP_FIXED 0x10
#define MAP_ANONYMOUS 0x20
#define MAP_ANON 0x20

#define MS_ASYNC 1
#define MS_INVALIDATE 2
#define MS_SYNC 4

#define MAP_FAILED ((void *)-1L)

// The memfs seam and the ops this header uses (see <unistd.h> for the seam itself).
extern long __vm_fs(long op, long a, long b, long c, long d);
enum { __FS_MM_READ = 1, __FS_MM_SEEK = 3, __FS_MM_MMAP = 9, __FS_MM_MSYNC = 10, __FS_MM_MUNMAP = 11 };

// `len` bytes of fresh, zero-filled, page-aligned memory: whole pages, as `mmap` promises. The page size
// comes from the Memory capability. The heap reuses freed memory, so it is `calloc` that guarantees the
// zeros.
static inline char *__mmap_pages(size_t len) {
  long pg = __vm_page_size();
  if (pg <= 0) pg = 4096;
  // Over-allocate by a page so the payload can be rounded up to a page boundary.
  char *raw = (char *)calloc(1, len + (size_t)pg);
  if (!raw) return 0;
  return (char *)(((long)raw + pg - 1) & ~(pg - 1));
}

// `mmap(addr, len, prot, flags, fd, offset)`. `addr` and `prot` are accepted and ignored. Returns the
// mapping, or `MAP_FAILED` (a zero length, a bad descriptor, or no memory left).
static inline void *mmap(void *addr, size_t len, int prot, int flags, int fd, long offset) {
  (void)addr;
  (void)prot;
  if (len == 0) return MAP_FAILED;
  if (flags & MAP_ANONYMOUS) {
    if (fd != -1) return MAP_FAILED;
    char *p = __mmap_pages(len);
    return p ? (void *)p : MAP_FAILED;
  }
  if (fd < 3 || offset < 0) return MAP_FAILED;
  char *p = __mmap_pages(len);
  if (!p) return MAP_FAILED;
  if (flags & MAP_SHARED) {
    // The memfs copies the file in and remembers the pages, to copy them back later.
    return __vm_fs(__FS_MM_MMAP, fd, offset, (long)len, (long)p) < 0 ? MAP_FAILED : (void *)p;
  }
  // MAP_PRIVATE: a copy of the file's bytes, read at `offset` without moving the descriptor. Past
  // the end of the file the pages stay zero, as they do for a real mapping.
  long pos = __vm_fs(__FS_MM_SEEK, fd, 1 /* SEEK_CUR */, 0, 0);
  if (pos < 0 || __vm_fs(__FS_MM_SEEK, fd, 0 /* SEEK_SET */, offset, 0) < 0) return MAP_FAILED;
  long got = __vm_fs(__FS_MM_READ, fd, (long)p, (long)len, 0);
  __vm_fs(__FS_MM_SEEK, fd, 0 /* SEEK_SET */, pos, 0);
  return got < 0 ? MAP_FAILED : (void *)p;
}

// Copy a shared file mapping back to its file and forget it. Anonymous and private mappings have
// nothing to copy back (the memfs doesn't know them), and their memory is never reclaimed, so
// unmapping them succeeds as a no-op.
static inline int munmap(void *addr, size_t len) {
  (void)len;
  __vm_fs(__FS_MM_MUNMAP, (long)addr, 0, 0, 0);
  return 0;
}
// Copy `[addr, addr + len)` of a shared file mapping back to its file.
static inline int msync(void *addr, size_t len, int flags) {
  (void)flags;
  long r = __vm_fs(__FS_MM_MSYNC, (long)addr, (long)len, 0, 0);
  return r < 0 ? -1 : 0;
}
static inline int mprotect(void *addr, size_t len, int prot) {
  (void)addr;
  (void)len;
  (void)prot;
  return 0;
}

#endif
