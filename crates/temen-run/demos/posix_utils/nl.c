/* nl(1) — number lines of stdin, GNU defaults: only non-empty lines are numbered (`-b t`), the
 * count printed right-justified in a 6-wide field followed by a TAB, then the line. A non-numbered
 * (empty) line emits the 6-space field and no separator. Streams line-at-a-time via u_rdline. */
long write(long fd, void *buf, long n);
long u_rdline(long fd, char *out, long cap);

static char line[8192];

int main(void) {
  long counter = 1;
  long n;
  while ((n = u_rdline(0, line, 8192)) >= 0) {
    if (n == 0) {
      write(1, "       \n", 8); /* blank line: the 6-wide field + separator column, all spaces (GNU) */
      continue;
    }
    /* decimal of counter, digits collected least-significant first */
    char tmp[24];
    long t = 0;
    long v = counter;
    if (v == 0) { tmp[0] = '0'; t = 1; }
    while (v > 0) { tmp[t] = '0' + (v % 10); t = t + 1; v = v / 10; }
    long pad = 6 - t;
    long j;
    for (j = 0; j < pad; j = j + 1) write(1, " ", 1);
    for (j = t - 1; j >= 0; j = j - 1) write(1, tmp + j, 1);
    write(1, "\t", 1);
    write(1, line, n);
    write(1, "\n", 1);
    counter = counter + 1;
  }
  return 0;
}
