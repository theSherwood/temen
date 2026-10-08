/* The playground's C heap (#2172): dlmalloc, configured exactly as the LLVM on-ramp's heap
 * (`temen_dlmalloc.c`), built by clang and translated as a link unit (`temen-llvm-translate
 * --link-unit`) into `web/assets/pg_heap.temeno`. Every C card program links it beside the chibicc
 * libc unit, which only declares `malloc`/`free`/`calloc`/`realloc`. Compiled by chibicc, dlmalloc's
 * code was several times the size and a third of the speed, and the bump heap it replaces never gave
 * memory back. Built by `scripts/rebuild-assets.sh` (`pg_heap`). */
#include <stddef.h>

long __vm_map(long off, long len, int prot);
long __vm_page_size(void);
long __vm_write_stderr(long buf, long len);
void exit(int code);

/* The heap's two words in the powerbox scratch page above the NULL guard (`temen_ir::POWERBOX_HEAP_BRK`
 * and `POWERBOX_HEAP_TOP`): the break, the end of what dlmalloc has been given, and the first byte past
 * the committed pages. The program's `_start` seeds both to the window's mapped boundary, and the heap
 * grows from there by committing host pages with `__vm_map`, so the committed window stays one range
 * from 0: the shape the wasm-JIT tier's bounds check follows. */
#define PG_HEAP_BRK ((volatile long *)(16384 + 32))
#define PG_HEAP_TOP ((volatile long *)(16384 + 40))
static long pg_base; /* where the heap starts: the first break, once dlmalloc has asked */

/* dlmalloc's `MORECORE`: move the break up by `inc`, committing the pages it crosses, and return the
 * old break, or `(void *)-1` when the window cannot grow (so `malloc` returns NULL). dlmalloc calls it
 * only under its global lock, and never with a negative `inc` (`MORECORE_CANNOT_TRIM`). */
static void *__temen_sbrk(ptrdiff_t inc) {
  long old = *PG_HEAP_BRK, end = old + inc, top = *PG_HEAP_TOP;
  if (!old)
    return (void *)-1; /* no `_start` seeded the heap */
  if (!pg_base)
    pg_base = old;
  if (end > top) {
    long page = __vm_page_size();
    if (page <= 0)
      page = 4096;
    long need = (end - top + page - 1) & ~(page - 1);
    if (__vm_map(top, need, 3) != 0)
      return (void *)-1;
    *PG_HEAP_TOP = top + need;
  }
  *PG_HEAP_BRK = end;
  return (void *)old;
}

/* A misused `free` or `realloc` aborts with glibc's words, which the playground's lessons teach: a
 * chunk inside the heap that is not in use was already freed; anything else is not a pointer
 * `malloc` returned. `exit(134)` is what `abort` does (as SIGABRT reads). */
static void pg_bad_free(int in_heap) {
  const char *msg = in_heap ? "free(): double free detected\n" : "free(): invalid pointer\n";
  long n = 0;
  while (msg[n])
    n++;
  __vm_write_stderr((long)msg, n);
  exit(134);
}
#define USAGE_ERROR_ACTION(m, p) pg_bad_free(ok_address(m, p))

/* dlmalloc's own entry points stay internal to the unit: only the four names below are exported.
 * (`malloc.c` declares `dlmalloc_usable_size` without `DLMALLOC_EXPORT`; declaring it `static` first
 * keeps it internal too.) */
#define DLMALLOC_EXPORT static
static size_t dlmalloc_usable_size(void *mem);
#include "temen_dlmalloc.c"

/* Only a 16-byte-aligned address below the break can be a block `malloc` returned. dlmalloc's own check
 * has no lower bound until the heap is first used, so a stack pointer freed before any `malloc` would
 * otherwise be taken for a chunk. */
static void pg_check_block(void *p) {
  long a = (long)p, brk = *PG_HEAP_BRK;
  if (p && (a < (pg_base ? pg_base : brk) || a >= brk || (a & 15)))
    pg_bad_free(0);
}

void *malloc(size_t n) { return dlmalloc(n); }

void free(void *p) {
  pg_check_block(p);
  dlfree(p);
}

void *calloc(size_t n, size_t size) { return dlcalloc(n, size); }

void *realloc(void *p, size_t n) {
  pg_check_block(p);
  return dlrealloc(p, n);
}
