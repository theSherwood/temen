/* fold(1) — wrap each stdin line to a fixed width (default 80; `-w N` or the `-N` shorthand),
 * breaking at exactly N columns (byte count, GNU default). Each wrapped chunk and the line's own
 * final chunk are newline-terminated. Streams line-at-a-time via u_rdline. */
long write(long fd, void *buf, long n);
long u_rdline(long fd, char *out, long cap);
long u_atoi(char *s);
int u_streq(char *a, char *b);

static char line[8192];

int main(int argc, char **argv) {
  long w = 80;
  if (argc >= 3 && u_streq(argv[1], "-w")) {
    w = u_atoi(argv[2]);
  } else if (argc >= 2 && argv[1][0] == '-' && argv[1][1] == 'w' && argv[1][2]) {
    w = u_atoi(argv[1] + 2); /* -wN glued form */
  } else if (argc >= 2 && argv[1][0] == '-' && argv[1][1] >= '0' && argv[1][1] <= '9') {
    w = u_atoi(argv[1] + 1); /* -N shorthand */
  }
  if (w < 1) w = 1;
  long n;
  while ((n = u_rdline(0, line, 8192)) >= 0) {
    if (n == 0) { write(1, "\n", 1); continue; }
    long p = 0;
    while (p < n) {
      long chunk = (n - p < w) ? (n - p) : w;
      write(1, line + p, chunk);
      write(1, "\n", 1);
      p = p + chunk;
    }
  }
  return 0;
}
