/* rev(1) — reverse the characters of each stdin line. A line's trailing newline is preserved (and a
 * final line without one stays unterminated), so `abc\n` → `cba\n` and `abc` → `cba`. Streams
 * line-at-a-time — a pipeline-polite citizen. */
long read(long fd, void *buf, long n);
long write(long fd, void *buf, long n);

/* Read one line into out[0..cap); set *had_nl when it ended on '\n'. Returns the length, or -1 at
 * EOF with no bytes read (so a trailing newline-less line is still delivered once). */
static long rd(char *out, long cap, int *had_nl) {
  long n = 0;
  *had_nl = 0;
  for (;;) {
    char c;
    long r = read(0, &c, 1);
    if (r <= 0) {
      if (n == 0) return -1;
      break;
    }
    if (c == '\n') { *had_nl = 1; break; }
    if (n < cap - 1) { out[n] = c; n = n + 1; }
  }
  return n;
}

static char line[8192];

int main(void) {
  long n;
  int had_nl;
  while ((n = rd(line, 8192, &had_nl)) >= 0) {
    long i;
    for (i = n - 1; i >= 0; i = i - 1) write(1, line + i, 1);
    if (had_nl) write(1, "\n", 1);
  }
  return 0;
}
