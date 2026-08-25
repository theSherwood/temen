
int __vm_cap_count(void);
int __vm_cap_at(int i, int *type_id_out);
long write(long fd, void *buf, long n);
long __vm_region_map(int r, long win_off, long region_off, long len, int prot);
long __vm_region_page_size(int r);

/* The stage window: `1 << TEMEN_STAGE_LOG2` bytes — equal to the carve the shell spawns each stage into
   (a §14 child's carve must equal its declared memory). Default 18 (256 KiB), the size chibicc lands
   at under the native differential's 16 KiB data page (with `window_pin_` below); the browser's 64 KiB
   page rounds the data sections up so the window is naturally 19 (512 KiB), and the fixture builds with
   `-D TEMEN_STAGE_LOG2=19` so the ring maps below sit at the right half-window offset. */
#ifndef TEMEN_STAGE_LOG2
#define TEMEN_STAGE_LOG2 18
#endif
#define TEMEN_STAGE_WIN (1L << TEMEN_STAGE_LOG2)

/* Pin the declared window up to `TEMEN_STAGE_WIN` (chibicc sizes it to the next power of two ≥ 2·data_end)
   under the 16 KiB page, where the runner's static region is otherwise too small. Harmless under the
   64 KiB page (the section alignment already reaches the next size); the harness asserts the result.
   #1059: the NULL guard shifts every datum up one 16 KiB guard, so the pad is one guard smaller than
   the pre-guard 50000 — otherwise `data_end` lands *at* the half-window ring-map offset
   (`TEMEN_STAGE_WIN/2`) and the data stack collides with the mapped ring. The reduced pad still forces
   window 18 (2·data_end > 128 KiB) while leaving the stack clear of the ring. */
static char window_pin_[33616];

static long slen(char *s) { long n = 0; while (s[n]) n = n + 1; return n; }
static int streq(char *a, char *b) { int i = 0; while (a[i] && a[i] == b[i]) i++; return a[i] == 0 && b[i] == 0; }
static long atoi_(char *s) {
  long v = 0; int i = 0;
  while (s[i] >= '0' && s[i] <= '9') { v = v * 10 + (s[i] - '0'); i++; }
  return v;
}
static int contains(char *hay, char *needle) {
  for (int i = 0; hay[i]; i++) {
    int j = 0; while (needle[j] && hay[i + j] == needle[j]) j++;
    if (needle[j] == 0) return 1;
  }
  return needle[0] == 0;
}
static int scmp(char *a, char *b) {
  int i = 0; while (a[i] && a[i] == b[i]) i++;
  return (int)(unsigned char)a[i] - (int)(unsigned char)b[i];
}
static void scpy(char *d, char *s) {
  int i = 0; while (s[i] && i < 255) { d[i] = s[i]; i++; } d[i] = 0;
}

/* Mapped ring bases: rin is always present; rout == 0 means "final stage — write the stdout
   Stream" (fd 1, the chibicc Stream builtin over the granted handle). */
static long rin = 0;
static long rout = 0;
static int out_dead = 0;   /* our reader closed (SIGPIPE-lite): swallow further output */

static void outb(char *b, long n) {
  if (rout) {
    if (out_dead) return;
    if (ring_write(rout, b, n) <= 0) out_dead = 1;
  } else {
    write(1, b, n);
  }
}
static void oputs(char *s) { outb(s, slen(s)); }
static void onum(long n) {
  static char b[24]; int i = 24;
  if (n == 0) { oputs("0"); return; }
  while (n > 0) { b[--i] = '0' + (int)(n % 10); n /= 10; }
  outb(b + i, 24 - i);
}

/* Buffered byte/line input over the input ring. */
static char rbuf[256];
static long rlen = 0, rpos = 0;
static int rin_eof = 0;
static int rgetc(void) {
  if (rpos >= rlen) {
    if (rin_eof) return -1;
    long r = ring_read(rin, rbuf, 256);
    if (r <= 0) { rin_eof = 1; return -1; }
    rlen = r; rpos = 0;
  }
  int c = rbuf[rpos]; rpos++;
  return c & 255;
}
static long rline(char *b, long lim) {
  long n = 0;
  for (;;) {
    int c = rgetc();
    if (c < 0) { if (n == 0) return -1; break; }
    if (c == '\n') break;
    if (n < lim - 1) b[n++] = (char)c;
  }
  b[n] = 0;
  return n;
}

int main(int argc, char **argv) {
  /* `regs`/`nregs` are deliberately LOCAL (frame storage) and filled with an indexed post-increment
     store — the exact shape ISSUES.md I35 was about. It regression-guards the I35 fix: the
     `--child-entry` argv relocation must place `main`'s frame just above the argv array, not round it
     up to the next 16 KiB page, or the frame collides with the ring this runner maps at 131072 and
     these locals read back garbage. Keep it local; do not "harden" it to `static`. */
  int regs[2];
  int nregs = 0;
  /* Discover the grants by reflection: regions in grant order (rin, then rout when present). */
  int n = __vm_cap_count();
  for (int i = 0; i < n; i++) {
    int t = 0;
    int h = __vm_cap_at(i, &t);
    if (t == 4 && nregs < 2) regs[nregs++] = h;
  }
  if (nregs > 2) nregs = 2;
  if (nregs == 0 || argc == 0) return 126;
  window_pin_[0] = 0;   /* keep the pad live (see its comment) */
  long g = __vm_region_page_size(regs[0]);
  if (g <= 64) return 125;
  rcap = g - 64;
  /* Map the rings into the **upper half** of the window (`TEMEN_STAGE_WIN/2`) — clear of the
     globals/args/frame region below and shallow enough that the data stack (near the top) never
     reaches down here. The half-window base scales with the window size, so it holds under both the
     18 (256 KiB, base 128 KiB) native and 19 (512 KiB, base 256 KiB) browser fixtures. */
  long mb = TEMEN_STAGE_WIN / 2;
  if (__vm_region_map(regs[0], mb, 0, g, 3) != 0) return 125;
  rin = mb;
  if (nregs > 1) {
    if (__vm_region_map(regs[1], mb + g, 0, g, 3) != 0) return 125;
    rout = mb + g;
  }

  char *cmd = argv[0];
  int st = 0;
  if (streq(cmd, "cat")) {
    static char buf[256]; long r;
    while ((r = ring_read(rin, buf, 256)) > 0) outb(buf, r);
  } else if (streq(cmd, "wc")) {
    static char buf[256];
    long r, lines = 0, words = 0, bytes = 0; int inword = 0;
    while ((r = ring_read(rin, buf, 256)) > 0) {
      for (long i = 0; i < r; i++) {
        char c = buf[i]; bytes++;
        if (c == '\n') lines++;
        if (c == ' ' || c == '\n' || c == '\t') inword = 0;
        else { if (!inword) words++; inword = 1; }
      }
    }
    onum(lines); oputs(" "); onum(words); oputs(" "); onum(bytes); oputs("\n");
  } else if (streq(cmd, "grep")) {
    int ai = 1, inv = 0, cnt = 0;
    while (ai < argc && argv[ai][0] == '-') {
      if (streq(argv[ai], "-v")) inv = 1;
      else if (streq(argv[ai], "-c")) cnt = 1;
      ai++;
    }
    long matches = 0;
    if (ai < argc) {
      char *pat = argv[ai];
      static char lb[256];
      while (rline(lb, 256) >= 0) {
        int m = contains(lb, pat);
        if (inv) m = !m;
        if (m) { matches++; if (!cnt) { oputs(lb); oputs("\n"); } }
      }
    }
    if (cnt) { onum(matches); oputs("\n"); }
    if (matches == 0) st = 1;
  } else if (streq(cmd, "head")) {
    long want = 10;
    if (argc > 2 && streq(argv[1], "-n")) want = atoi_(argv[2]);
    static char lb[256];
    for (long k = 0; k < want && rline(lb, 256) >= 0; k++) { oputs(lb); oputs("\n"); }
  } else if (streq(cmd, "tail")) {
    long want = 10;
    if (argc > 2 && streq(argv[1], "-n")) want = atoi_(argv[2]);
    if (want > 16) want = 16;
    static char lines[16][256];
    long count = 0;
    while (rline(lines[count % 16], 256) >= 0) count++;
    long start = count > want ? count - want : 0;
    for (long k = start; k < count; k++) { oputs(lines[k % 16]); oputs("\n"); }
  } else if (streq(cmd, "sort")) {
    static char buf[64][256]; int nl = 0;
    while (nl < 64 && rline(buf[nl], 256) >= 0) nl++;
    for (int i = 1; i < nl; i++) {
      static char key[256]; scpy(key, buf[i]);
      int j = i - 1;
      while (j >= 0 && scmp(buf[j], key) > 0) { scpy(buf[j + 1], buf[j]); j--; }
      scpy(buf[j + 1], key);
    }
    for (int i = 0; i < nl; i++) { oputs(buf[i]); oputs("\n"); }
  } else if (streq(cmd, "uniq")) {
    static char cur[256], prev[256]; int have = 0;
    while (rline(cur, 256) >= 0)
      if (!have || scmp(cur, prev) != 0) { oputs(cur); oputs("\n"); scpy(prev, cur); have = 1; }
  } else {
    st = 127;   /* the shell's classifier admits only the filters above */
  }

  /* Epilogue: finish our output ring (the next stage drains then sees EOF) and close our input ring
     even when we exited early (`head`) — the producer stops instead of parking to a timeout. */
  if (rout) ring_done(rout);
  ring_close_read(rin);
  if (ring_bail && st == 0) st = 99;   /* a lost wake surfaces loudly in $? */
  return st;
}
