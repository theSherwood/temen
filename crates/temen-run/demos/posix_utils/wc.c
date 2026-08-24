/* wc(1) — count stdin. `-l` lines, `-w` words, `-c` bytes; no flag prints all
 * three as "L W C". Each count on its own `write` so pipes see one burst. */
long read(long fd, void *buf, long n);
long write(long fd, void *buf, long n);
int u_streq(char *a, char *b);
long u_putn(long fd, long v);

static char wc_buf_[4096];
int main(int argc, char **argv) {
  long lines = 0, words = 0, bytes = 0, inword = 0;
  for (;;) {
    long n = read(0, wc_buf_, 4096);
    if (n < 0) return 1;
    if (n == 0) break;
    bytes = bytes + n;
    long i;
    for (i = 0; i < n; i = i + 1) {
      char c = wc_buf_[i];
      if (c == '\n') lines = lines + 1;
      if (c == ' ' || c == '\t' || c == '\n') inword = 0;
      else if (!inword) { inword = 1; words = words + 1; }
    }
  }
  if (argc > 1 && u_streq(argv[1], "-l")) { u_putn(1, lines); }
  else if (argc > 1 && u_streq(argv[1], "-w")) { u_putn(1, words); }
  else if (argc > 1 && u_streq(argv[1], "-c")) { u_putn(1, bytes); }
  else {
    u_putn(1, lines); write(1, " ", 1);
    u_putn(1, words); write(1, " ", 1);
    u_putn(1, bytes);
  }
  write(1, "\n", 1);
  return 0;
}
