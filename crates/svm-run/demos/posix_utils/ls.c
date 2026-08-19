/* ls(1) — the entries of DIR (default "."), sorted, one per line. Fixed
 * arena (16 KiB / 512 entries), overflow exits 2. */
long __px_opendir(int cap, long path, long len);
long __px_readdir(int cap, long dir, long buf, long capn);
long __px_closedir(int cap, long dir);
long write(long fd, void *buf, long n);
long u_strlen(char *s);
int u_strcmp(char *a, char *b);
long u_puts(long fd, char *s);

static char ls_names_[16384];
static char *ls_ents_[512];
int main(int argc, char **argv) {
  char *dir = argc > 1 ? argv[1] : ".";
  long d = __px_opendir(0, (long)dir, u_strlen(dir));
  if (d < 0) return 1;
  long used = 0, cnt = 0;
  for (;;) {
    if (cnt >= 512 || used + 256 > 16384) { __px_closedir(0, d); return 2; }
    long r = __px_readdir(0, d, (long)(ls_names_ + used), 256);
    if (r <= 0) break;
    ls_ents_[cnt] = ls_names_ + used;
    cnt = cnt + 1;
    used = used + u_strlen(ls_names_ + used) + 1;
  }
  __px_closedir(0, d);
  long i, j;
  for (i = 1; i < cnt; i = i + 1) {
    char *k = ls_ents_[i];
    for (j = i; j > 0 && u_strcmp(ls_ents_[j - 1], k) > 0; j = j - 1)
      ls_ents_[j] = ls_ents_[j - 1];
    ls_ents_[j] = k;
  }
  for (i = 0; i < cnt; i = i + 1) {
    if (u_puts(1, ls_ents_[i]) < 0) return 1;
    if (write(1, "\n", 1) != 1) return 1;
  }
  return 0;
}
