/* tail(1) — last N lines of stdin (default 10; `-n N` or the `-N` shorthand). Buffers the input
 * (bounded) and walks the recorded line-starts so the last N newline-terminated segments are kept —
 * a final line without a trailing newline is preserved. head's mirror image. */
long read(long fd, void *buf, long n);
long write(long fd, void *buf, long n);
int u_streq(char *a, char *b);
long u_atoi(char *s);

static char buf[32768];
static long starts[1026]; /* ring of the last (n+1) line-start offsets */

int main(int argc, char **argv) {
  long n = 10;
  if (argc >= 3 && u_streq(argv[1], "-n")) {
    n = u_atoi(argv[2]);
  } else if (argc >= 2 && argv[1][0] == '-' && argv[1][1] >= '0' && argv[1][1] <= '9') {
    n = u_atoi(argv[1] + 1); /* POSIX `-N` shorthand */
  }
  if (n < 0) n = 0;
  if (n > 1024) n = 1024;
  long total = 0;
  for (;;) {
    long r = read(0, buf + total, 32768 - total);
    if (r <= 0) break;
    total = total + r;
    if (total >= 32768) break; /* bounded: demo tool, not a general file cat */
  }
  if (n == 0 || total == 0) return 0;
  long cap = n + 1;
  long head = 0;
  long p;
  starts[0] = 0; /* line 0 begins at offset 0 */
  head = 1;
  for (p = 0; p < total; p = p + 1) {
    if (buf[p] == '\n' && p + 1 < total) {
      starts[head % cap] = p + 1;
      head = head + 1;
    }
  }
  long begin;
  if (head <= n) begin = 0;
  else begin = starts[(head - n) % cap];
  write(1, buf + begin, total - begin);
  return 0;
}
