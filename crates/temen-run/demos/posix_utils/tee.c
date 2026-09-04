/* tee(1) — copy stdin to stdout and to each file argument. `-a` appends instead of truncating.
 * Opens each target through the personality's memfs (`__px_open` with O_WRONLY|O_CREAT|O_TRUNC, or
 * |O_APPEND for -a) and fans every read block out to fd 1 and all the file fds. */
long __px_open(int cap, long path, long len, long flags);
long read(long fd, void *buf, long n);
long write(long fd, void *buf, long n);
long close(long fd);
long u_strlen(char *s);
int u_streq(char *a, char *b);

static char buf[8192];
static long fds[64];

int main(int argc, char **argv) {
  int append = 0;
  int i = 1;
  while (i < argc && argv[i][0] == '-' && argv[i][1]) {
    if (u_streq(argv[i], "-a")) append = 1;
    else break;
    i = i + 1;
  }
  /* O_WRONLY|O_CREAT is 1|0100; add O_TRUNC (01000) or O_APPEND (02000) */
  long flags = append ? (1 | 0100 | 02000) : (1 | 0100 | 01000);
  long nf = 0;
  for (; i < argc && nf < 64; i = i + 1) {
    long fd = __px_open(0, (long)argv[i], u_strlen(argv[i]), flags);
    if (fd < 0) return 1;
    fds[nf] = fd;
    nf = nf + 1;
  }
  for (;;) {
    long n = read(0, buf, 8192);
    if (n < 0) return 1;
    if (n == 0) break;
    if (write(1, buf, n) != n) return 1;
    long k;
    for (k = 0; k < nf; k = k + 1) if (write(fds[k], buf, n) != n) return 1;
  }
  long k;
  for (k = 0; k < nf; k = k + 1) close(fds[k]);
  return 0;
}
