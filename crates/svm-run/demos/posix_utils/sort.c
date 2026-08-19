/* sort(1) — read every stdin line, insertion-sort by byte order, print.
 * Fixed arena (64 KiB text / 2048 lines) — plenty for shell-sized inputs;
 * overflow is exit 2, never silent truncation. */
long write(long fd, void *buf, long n);
int u_strcmp(char *a, char *b);
long u_puts(long fd, char *s);
long u_rdline(long fd, char *out, long cap);

static char sort_arena_[65536];
static char *sort_lines_[2048];
int main(void) {
  long used = 0, cnt = 0;
  for (;;) {
    if (cnt >= 2048 || used + 4096 > 65536) return 2;
    long n = u_rdline(0, sort_arena_ + used, 4096);
    if (n < 0) break;
    sort_lines_[cnt] = sort_arena_ + used;
    cnt = cnt + 1;
    used = used + n + 1;
  }
  long i, j;
  for (i = 1; i < cnt; i = i + 1) {
    char *k = sort_lines_[i];
    for (j = i; j > 0 && u_strcmp(sort_lines_[j - 1], k) > 0; j = j - 1)
      sort_lines_[j] = sort_lines_[j - 1];
    sort_lines_[j] = k;
  }
  for (i = 0; i < cnt; i = i + 1) {
    if (u_puts(1, sort_lines_[i]) < 0) return 1;
    if (write(1, "\n", 1) != 1) return 1;
  }
  return 0;
}
