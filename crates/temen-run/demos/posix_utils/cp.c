/* cp(1) — copy SRC to DST (regular files). Opens SRC read-only and DST with
 * O_WRONLY|O_CREAT|O_TRUNC, then streams the bytes through the memfs. Two operands only; no
 * recursive or directory-target modes. */
long __px_open(int cap, long path, long len, long flags);
long read(long fd, void *buf, long n);
long write(long fd, void *buf, long n);
long close(long fd);
long u_strlen(char *s);

static char buf[8192];

int main(int argc, char **argv) {
  if (argc < 3) return 1; /* usage: cp SRC DST */
  long src = __px_open(0, (long)argv[1], u_strlen(argv[1]), 0);
  if (src < 0) return 1;
  long dst = __px_open(0, (long)argv[2], u_strlen(argv[2]), 1 | 0100 | 01000); /* WRONLY|CREAT|TRUNC */
  if (dst < 0) { close(src); return 1; }
  for (;;) {
    long n = read(src, buf, 8192);
    if (n < 0) { close(src); close(dst); return 1; }
    if (n == 0) break;
    if (write(dst, buf, n) != n) { close(src); close(dst); return 1; }
  }
  close(src);
  close(dst);
  return 0;
}
