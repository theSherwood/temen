/* uniq(1) — collapse ADJACENT duplicate stdin lines; `-c` prefixes each
 * surviving line with its run count and a space (no column padding). */
long write(long fd, void *buf, long n);
int u_streq(char *a, char *b);
long u_puts(long fd, char *s);
long u_putn(long fd, long v);
long u_rdline(long fd, char *out, long cap);

static char uq_a_[4096], uq_b_[4096];
static long uq_emit_(char *s, long n, int cflag) {
  if (cflag) {
    if (u_putn(1, n) < 0) return 1;
    if (write(1, " ", 1) != 1) return 1;
  }
  if (u_puts(1, s) < 0) return 1;
  if (write(1, "\n", 1) != 1) return 1;
  return 0;
}
int main(int argc, char **argv) {
  int cflag = argc > 1 && u_streq(argv[1], "-c");
  char *prev = 0, *cur = uq_a_;
  long run = 0;
  for (;;) {
    long n = u_rdline(0, cur, 4096);
    if (n < 0) break;
    if (prev && u_streq(prev, cur)) { run = run + 1; continue; }
    if (prev && uq_emit_(prev, run, cflag)) return 1;
    /* the just-read line becomes prev; reuse the old prev buffer for the next read */
    char *t = prev ? prev : uq_b_;
    prev = cur;
    cur = t;
    run = 1;
  }
  if (prev && uq_emit_(prev, run, cflag)) return 1;
  return 0;
}
