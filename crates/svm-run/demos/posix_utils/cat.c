/* cat(1) — stream stdin (no args) or each file argument to stdout. */
long __px_open(int cap, long path, long len, long flags);
long read(long fd, void *buf, long n);
long write(long fd, void *buf, long n);
long close(long fd);
long u_strlen(char *s);

static char cat_buf_[4096];
static long cat_fd_(long fd) {
  for (;;) {
    long n = read(fd, cat_buf_, 4096);
    if (n < 0) return 1;
    if (n == 0) return 0;
    if (write(1, cat_buf_, n) != n) return 1;
  }
}
int main(int argc, char **argv) {
  if (argc < 2) return (int)cat_fd_(0);
  int i;
  for (i = 1; i < argc; i = i + 1) {
    long fd = __px_open(0, (long)argv[i], u_strlen(argv[i]), 0);
    if (fd < 0) return 1;
    long bad = cat_fd_(fd);
    close(fd);
    if (bad) return 1;
  }
  return 0;
}
