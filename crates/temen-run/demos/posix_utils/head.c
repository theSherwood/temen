/* head(1) — first N lines of stdin (default 10; `-n N` or the `-N` shorthand). Reads
 * byte-at-a-time so it never consumes past its last line — the POSIX-polite pipeline citizen. */
long read(long fd, void *buf, long n);
long write(long fd, void *buf, long n);
int u_streq(char *a, char *b);
long u_atoi(char *s);

int main(int argc, char **argv) {
  long left = 10;
  if (argc >= 3 && u_streq(argv[1], "-n")) {
    left = u_atoi(argv[2]);
  } else if (argc >= 2 && argv[1][0] == '-' && argv[1][1] >= '0' && argv[1][1] <= '9') {
    left = u_atoi(argv[1] + 1); /* POSIX `-N` shorthand for `-n N` (bash's `declare -f f | head -1`) */
  }
  while (left > 0) {
    char c;
    long r = read(0, &c, 1);
    if (r <= 0) break;
    if (write(1, &c, 1) != 1) return 1;
    if (c == '\n') left = left - 1;
  }
  return 0;
}
