
int __vm_wait32(void *p, int expected, long timeout_ns);
int __vm_notify(void *p, int count);
int __vm_atomic_load32(void *p);
void __vm_atomic_store32(void *p, int v);

static long rcap = 0;      /* ring data capacity: map granule - 64 */
static int ring_bail = 0;  /* set when a wait timed out 6+ times — a lost wake, surfaced loudly */

static long rmin2(long a, long b) { return a < b ? a : b; }
static long ring_to(void) { long s = 5; return s * 1000000000; }   /* 5 s in ns */

/* Write n bytes into the ring at b. Returns n; 0 when the reader closed its end (stop producing —
   the SIGPIPE analogue); -1 after repeated timeouts (ring_bail set). */
static long ring_write(long b, char *src, long n) {
  long done_n = 0; int tos = 0;
  while (done_n < n) {
    if (__vm_atomic_load32((void *)(b + 12))) return 0;
    long h = (long)__vm_atomic_load32((void *)b);
    long t = (long)__vm_atomic_load32((void *)(b + 4));
    long freeb = rcap - (h - t);
    if (freeb <= 0) {
      int st = __vm_wait32((void *)(b + 4), (int)t, ring_to());
      if (st == 2) { tos++; if (tos > 6) { ring_bail = 1; return -1; } }
      continue;
    }
    long k = rmin2(n - done_n, freeb);
    char *d = (char *)(b + 64);
    for (long i = 0; i < k; i++) d[(h + i) % rcap] = src[done_n + i];
    __vm_atomic_store32((void *)b, (int)(h + k));
    __vm_notify((void *)b, 1);
    done_n += k;
  }
  return n;
}

/* Read up to cap bytes from the ring at b. Returns the byte count (>0); 0 at EOF (writer done and
   the ring drained); -1 after repeated timeouts (ring_bail set). */
static long ring_read(long b, char *dst, long cap) {
  int tos = 0;
  for (;;) {
    long h = (long)__vm_atomic_load32((void *)b);
    long t = (long)__vm_atomic_load32((void *)(b + 4));
    long used = h - t;
    if (used > 0) {
      long k = rmin2(cap, used);
      char *d = (char *)(b + 64);
      for (long i = 0; i < k; i++) dst[i] = d[(t + i) % rcap];
      __vm_atomic_store32((void *)(b + 4), (int)(t + k));
      __vm_notify((void *)(b + 4), 1);
      return k;
    }
    if (__vm_atomic_load32((void *)(b + 8))) return 0;
    int st = __vm_wait32((void *)b, (int)h, ring_to());
    if (st == 2) { tos++; if (tos > 6) { ring_bail = 1; return -1; } }
  }
}

/* Writer finished: set done and wake the reader (it drains, then sees EOF). */
static void ring_done(long b) { __vm_atomic_store32((void *)(b + 8), 1); __vm_notify((void *)b, 1); }
/* Reader finished (possibly early — `head`): set rclosed and wake the writer so it stops. */
static void ring_close_read(long b) { __vm_atomic_store32((void *)(b + 12), 1); __vm_notify((void *)(b + 4), 1); }
