/* tac(1) — concatenate stdin in reverse line order. `\n` is a trailing separator: each line keeps
 * its own newline, and a final line without one stays unterminated, so `a\nb\nc` reverses to
 * `cb\na\n` (matching GNU). Buffers the input (bounded) and emits recorded segments back-to-front. */
long read(long fd, void *buf, long n);
long write(long fd, void *buf, long n);

static char buf[32768];
static long starts[2048]; /* segment start offsets */

int main(void) {
  long total = 0;
  for (;;) {
    long r = read(0, buf + total, 32768 - total);
    if (r <= 0) break;
    total = total + r;
    if (total >= 32768) break;
  }
  if (total == 0) return 0;
  long nseg = 0;
  starts[0] = 0;
  nseg = 1;
  long p;
  for (p = 0; p < total; p = p + 1) {
    if (buf[p] == '\n' && p + 1 < total) {
      if (nseg < 2048) { starts[nseg] = p + 1; nseg = nseg + 1; }
    }
  }
  long i;
  for (i = nseg - 1; i >= 0; i = i - 1) {
    long end = (i + 1 < nseg) ? starts[i + 1] : total;
    write(1, buf + starts[i], end - starts[i]);
  }
  return 0;
}
