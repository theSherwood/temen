/* pwd(1) — the current working directory, newline-terminated. */
long __px_getcwd(int cap, long buf, long size);
long write(long fd, void *buf, long n);
long u_puts(long fd, char *s);

static char pwd_buf_[512];
int main(void) {
  if (__px_getcwd(0, (long)pwd_buf_, 512) <= 0) return 1;
  if (u_puts(1, pwd_buf_) < 0) return 1;
  if (write(1, "\n", 1) != 1) return 1;
  return 0;
}
