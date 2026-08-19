/* grep(1) — POSIX ERE over stdin or a file: `grep PATTERN [FILE]`. Prints
 * matching lines; exits 0 on any match, 1 on none, 2 on a bad pattern or
 * unopenable file. Concatenated AFTER posix_libc/regex.c in its translation
 * unit — regex_t/regcomp/regexec come from there (cflags 1|8 = ERE|NOSUB). */
long __px_open(int cap, long path, long len, long flags);
long write(long fd, void *buf, long n);
long u_strlen(char *s);
long u_puts(long fd, char *s);
long u_rdline(long fd, char *out, long cap);

static char grep_line_[4096];
static regex_t grep_rx_;
int main(int argc, char **argv) {
  if (argc < 2) return 2;
  if (regcomp(&grep_rx_, argv[1], 1 | 8) != 0) return 2;
  long fd = 0;
  if (argc > 2) {
    fd = __px_open(0, (long)argv[2], u_strlen(argv[2]), 0);
    if (fd < 0) return 2;
  }
  long hit = 0;
  for (;;) {
    long n = u_rdline(fd, grep_line_, 4096);
    if (n < 0) break;
    if (regexec(&grep_rx_, grep_line_, 0, 0, 0) == 0) {
      hit = 1;
      if (u_puts(1, grep_line_) < 0) return 2;
      if (write(1, "\n", 1) != 1) return 2;
    }
  }
  return hit ? 0 : 1;
}
