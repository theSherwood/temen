/* util.c — shared runtime for the /bin coreutils (#801).
 *
 * Concatenated FIRST into every utility's translation unit by the embedder
 * that stages /bin (see crates/svm/tests/c_posix.rs `stage_coreutils`).
 * Freestanding: no includes; identical duplicate `__px_` declarations in the
 * tool files are legal C.
 *
 * fd I/O follows the #972 tag protocol: a personality op that lands on a core
 * pipe/terminal end returns PX_TAG_BASE - handle (<= -(1<<20)); the wrapper
 * re-issues the transfer on the core handle, where empty-with-writers PARKS
 * and writer-count 0 is true EOF. Real errnos stay > -4096 and pass through.
 */
long __vm_pipe(int *fds);
long __vm_read(int fd, void *buf, long len);
long __vm_write(int fd, void *buf, long len);
long __vm_close(int h);
long __px_pipe_adopt(int cap, long rh, long wh, long fdp);
long __px_read(int cap, long fd, long buf, long len);
long __px_write(int cap, long fd, long buf, long len);
long __px_close(int cap, long fd);
long __px_kill(int cap, long pid, long sig);

static long u_h_(long r) { return r <= -1048576 ? -(r + 1048576) : -1; }
long read(long fd, void *buf, long n) {
  for (;;) {
    long r = __px_read(0, fd, (long)buf, n);
    if (r == -85) continue; /* -ERESTART: SIGTTIN stopped us before the read (rung-3 tail) —
                               re-issue; the stop benches at the re-issued dispatch and a later
                               SIGCONT re-runs the op under the then-current pgid. */
    long h = u_h_(r);
    if (h < 0) return r;
    return __vm_read((int)h, buf, n);
  }
}
long write(long fd, void *buf, long n) {
  long r = __px_write(0, fd, (long)buf, n);
  long h = u_h_(r);
  if (h < 0) return r;
  r = __vm_write((int)h, buf, n);
  if (r == -32) __px_kill(0, 0, 13); /* -EPIPE: raise SIGPIPE per disposition */
  return r;
}
long close(long fd) {
  long r = __px_close(0, fd);
  long h = u_h_(r);
  if (h < 0) return r;
  __vm_close((int)h);
  return 0;
}
long pipe(int *fds) {
  int h[2];
  long r = __vm_pipe(h);
  if (r != 0) return r;
  return __px_pipe_adopt(0, h[0], h[1], (long)fds);
}

long u_strlen(char *s) {
  long n = 0;
  while (s[n]) n = n + 1;
  return n;
}
int u_streq(char *a, char *b) {
  long i = 0;
  while (a[i] && a[i] == b[i]) i = i + 1;
  return a[i] == b[i];
}
int u_strcmp(char *a, char *b) {
  long i = 0;
  while (a[i] && a[i] == b[i]) i = i + 1;
  return (int)a[i] - (int)b[i];
}
long u_atoi(char *s) {
  long v = 0, i = 0, neg = 0;
  if (s[0] == '-') { neg = 1; i = 1; }
  while (s[i] >= '0' && s[i] <= '9') { v = v * 10 + (s[i] - '0'); i = i + 1; }
  return neg ? -v : v;
}
long u_puts(long fd, char *s) { return write(fd, s, u_strlen(s)); }
long u_putn(long fd, long v) {
  char b[24];
  long i = 24, neg = 0;
  if (v < 0) { neg = 1; v = -v; }
  if (v == 0) { i = i - 1; b[i] = '0'; }
  while (v > 0) { i = i - 1; b[i] = '0' + (v % 10); v = v / 10; }
  if (neg) { i = i - 1; b[i] = '-'; }
  return write(fd, b + i, 24 - i);
}
/* One line from fd, byte-at-a-time (a read of 1 parks like any other — the
 * simplicity is the point; these are witnesses, not hot paths). Returns the
 * length excluding the '\n', or -1 on EOF with nothing read; a final
 * unterminated line is returned as-is. Overlong lines are truncated. */
long u_rdline(long fd, char *out, long cap) {
  long n = 0;
  for (;;) {
    char c;
    long r = read(fd, &c, 1);
    if (r <= 0) {
      if (n == 0) return -1;
      break;
    }
    if (c == '\n') break;
    if (n < cap - 1) { out[n] = c; n = n + 1; }
  }
  out[n] = 0;
  return n;
}
