// Minimal <stdlib.h> for the Temen sandbox target (the whole-program guest libc, §3d).
//
// The headline piece is a real **`malloc`/`free`/`calloc`/`realloc`** built on the Memory
// capability (§3e/§4): the heap lives at a fixed high base in the *reserved tail* of the window
// and **grows on demand** by committing pages with `__vm_map` — the §1a sparse-address-space win,
// available to any program that just `#include <stdlib.h>` (no special prelude). `free` is a
// no-op (a bump allocator — the §3d MVP; no reclamation), with a per-allocation size header so
// `realloc` can copy. A native `cc` build of the same source uses the platform libc instead;
// this header shadows the system one only for the sandbox frontend (chibicc searches its bundled
// include dir first), so demos stay byte-identical to native.
//
// Deliberately small: the allocator, `exit`/`abort`, and the `EXIT_*`/`NULL`/`size_t` boilerplate
// real programs pull from <stdlib.h>. Anything else a program calls is a clean "undefined
// function" error (there is no libc to link — the whole program is the translation unit).
#ifndef __TEMEN_STDLIB_H
#define __TEMEN_STDLIB_H

#include <stddef.h> // size_t, NULL

#define EXIT_SUCCESS 0
#define EXIT_FAILURE 1

// `exit` is a powerbox builtin (§3e), intercepted by name; declaring it here is enough.
void exit(int code);

// The Memory-capability builtins (§3e/§4), lowered to `cap.call` on the granted Memory handle.
// `__vm_map` commits `[off, off+len)` with `prot` (READ|WRITE = 3), returning 0 or a negative errno.
// `__vm_page_size` returns the host MMU page granularity the window is managed in (the unit `map`
// rounds to), so the guest can align to the *real* page instead of assuming a fixed size.
long __vm_map(long off, long len, int prot);
long __vm_page_size(void);

// Atomics for the **thread-safe** bump allocator (below). These lower to plain atomic ops, so a
// single-threaded program pays only an uncontended atomic and never pulls in the thread runtime
// (only `thread.spawn`/`wait`/`notify` mark a module threaded — not atomics).
long __vm_atomic_add(void *p, long v);   // fetch-add (i64), returns old
long __vm_atomic_load(void *p);          // load (i64)
void __vm_atomic_store(void *p, long v); // store (i64)
int __vm_atomic_cas32(void *p, int expected, int desired); // CAS (i32), returns old
void __vm_atomic_store32(void *p, int v);                  // store (i32)

static void abort(void) {
  exit(134); // 128 + SIGABRT, the conventional code
}

// --- the map-growing heap -------------------------------------------------------------------
#define __TEMEN_HEAP_BASE 268435456L // 256 MiB: above the (<= 64 MiB) backed prefix, in the tail
#define __TEMEN_HDR 16L // per-allocation header (holds the payload size; keeps 16-byte alignment)

// Allocator-state linkage. A whole-program build (the on-ramp, and every playground program) is one
// translation unit, so the bump-pointer state is a self-contained `static`. The **emit-object
// multi-TU self-host** build is different: each TU `#include`s this header and would otherwise mint
// its *own* `static __temen_brk` at 256 MiB, so the per-TU allocators would hand out **overlapping**
// addresses and corrupt each other at run time. There the state must be a **single shared instance**:
// the emit-object build force-includes `selfhost_prelude.h` (which sets `__TEMEN_LIBC_EXTERN`), the libc
// unit (`emit_libc.c`) additionally sets `__TEMEN_LIBC_OWNER` to hold the one definition, and every other
// TU sees an `extern` and links to it cross-TU (the same data-symbol path as chibicc's `ty_int`). The
// allocator *functions* below stay `static` per unit — only this *state* is shared, so all the per-TU
// `malloc`s cooperate on one bump pointer.
#if defined(__TEMEN_LIBC_OWNER)
#define __TEMEN_HEAP_STATE            // exported definition (non-`static` ⇒ an emit-object data symbol)
#define __TEMEN_HEAP_INIT(v) = (v)
#elif defined(__TEMEN_LIBC_EXTERN)
#define __TEMEN_HEAP_STATE extern     // imported: declaration only, no storage, no initializer
#define __TEMEN_HEAP_INIT(v)
#else
#define __TEMEN_HEAP_STATE static     // whole-program: self-contained, as before
#define __TEMEN_HEAP_INIT(v) = (v)
#endif
__TEMEN_HEAP_STATE long __temen_brk __TEMEN_HEAP_INIT(__TEMEN_HEAP_BASE);       // next free byte (bump pointer)
__TEMEN_HEAP_STATE long __temen_committed __TEMEN_HEAP_INIT(__TEMEN_HEAP_BASE); // first byte past committed
__TEMEN_HEAP_STATE long __temen_page __TEMEN_HEAP_INIT(0);                    // cached host page granularity
__TEMEN_HEAP_STATE int __temen_grow_lock __TEMEN_HEAP_INIT(0);               // spinlock for heap *growth* only

// Heap-growth granularity = the **host page**, queried once and cached. The runtime's `map` commits
// and zero-fills the whole host page(s) covering a request (host-page default: 4 KiB on x86-64,
// 16 KiB on Apple Silicon, …); growing in that unit means each growth covers fresh page(s) on any
// host (a smaller step could re-`map` — and so re-zero — a page already holding live allocations).
// `__TEMEN_HEAP_BASE` (256 MiB) is a multiple of every realistic page, so growth stays page-aligned.
static long __temen_pagesize(void) {
  if (__temen_page == 0) {
    long p = __vm_page_size();
    __temen_page = p > 0 ? p : 4096L; // defensive floor if the query is unavailable
  }
  return __temen_page;
}

// **Thread-safe `malloc`.** The fast path is **lock-free**: an atomic fetch-add on the bump pointer
// claims a unique `[hdr, end)` region, so concurrent callers never overlap. Only **growth** —
// committing new pages when a claim runs past the committed boundary — is serialized (a brief
// spinlock around one `__vm_map`), so a page is mapped exactly once (re-mapping would re-zero live
// allocations). `__temen_committed` is published *after* the pages are mapped, so any caller that
// observes `committed >= end` sees its region already backed. A single-threaded caller pays only the
// uncontended atomics; the spinlock never spins. (On OOM the claimed range is leaked — but `free` is a
// no-op anyway, §3d MVP, so this only forfeits some address space; it never corrupts live data.)
static void *malloc(size_t n) {
  n = (n + 15UL) & ~15UL; // 16-byte align the payload
  long total = __TEMEN_HDR + (long)n;
  long hdr = __vm_atomic_add(&__temen_brk, total); // atomically claim [hdr, hdr+total)
  long payload = hdr + __TEMEN_HDR;
  long end = hdr + total;
  if (end > __vm_atomic_load(&__temen_committed)) {
    while (__vm_atomic_cas32(&__temen_grow_lock, 0, 1) != 0) {
    } // acquire the growth lock
    long cur = __vm_atomic_load(&__temen_committed);
    if (end > cur) { // still short after the lock — another grower may have caught up
      long pg = __temen_pagesize();
      long need = (end - cur + (pg - 1)) & ~(pg - 1); // whole host pages covering the shortfall
      if (__vm_map(cur, need, 3) != 0) {
        __vm_atomic_store32(&__temen_grow_lock, 0); // release
        return NULL;                              // out of memory
      }
      __vm_atomic_store(&__temen_committed, cur + need); // publish growth *after* it is mapped
    }
    __vm_atomic_store32(&__temen_grow_lock, 0); // release
  }
  *(size_t *)hdr = n; // remember the size for realloc (disjoint per allocation)
  return (void *)payload;
}

static void free(void *p) {
  (void)p; // bump allocator: no reclamation (MVP, §3d)
}

static void *calloc(size_t n, size_t sz) {
  // Fresh window pages are zero-filled by `map`, and the bump allocator never reuses a byte, so
  // the payload is already zero.
  return malloc(n * sz);
}

static void *realloc(void *p, size_t n) {
  if (!p)
    return malloc(n);
  size_t old = *(size_t *)((char *)p - __TEMEN_HDR);
  void *q = malloc(n);
  if (!q)
    return NULL;
  size_t c = old < n ? old : n;
  for (size_t i = 0; i < c; i++) // self-contained copy (no <string.h> dependency)
    ((char *)q)[i] = ((char *)p)[i];
  return q;
}

#endif // __TEMEN_STDLIB_H
