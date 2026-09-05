/* cp(1) — copy SRC to DST. Default: a regular-file copy (SRC read-only → DST O_WRONLY|O_CREAT|O_TRUNC,
 * streamed through the memfs). `-r`/`-R`: recursive — if SRC is a directory, `mkdir` DST and copy every
 * entry into it (depth-first). Two operands; no copy-into-existing-directory (GNU) nuance. */
long __px_open(int cap, long path, long len, long flags);
long __px_opendir(int cap, long path, long len);
long __px_readdir(int cap, long dir, long buf, long cap_n);
long __px_closedir(int cap, long dir);
long __px_mkdir(int cap, long path, long len, long mode);
long read(long fd, void *buf, long n);
long write(long fd, void *buf, long n);
long close(long fd);
long u_strlen(char *s);

static char buf[8192];

static int copy_file(char *src, char *dst) {
  long s = __px_open(0, (long)src, u_strlen(src), 0);
  if (s < 0) return 1;
  long d = __px_open(0, (long)dst, u_strlen(dst), 1 | 0100 | 01000); /* WRONLY|CREAT|TRUNC */
  if (d < 0) { close(s); return 1; }
  int rc = 0;
  for (;;) {
    long n = read(s, buf, 8192);
    if (n < 0) { rc = 1; break; }
    if (n == 0) break;
    if (write(d, buf, n) != n) { rc = 1; break; }
  }
  close(s);
  close(d);
  return rc;
}

/* Recursive copy of `src` → `dst`. If `src` is a directory, `mkdir` `dst` and recurse each child;
 * else a plain file copy. Children are buffered before recursion (readdir over a stable dir). */
static int cp_path(char *src, char *dst) {
  long dh = __px_opendir(0, (long)src, u_strlen(src));
  if (dh < 0) return copy_file(src, dst);
  __px_mkdir(0, (long)dst, u_strlen(dst), 0755); /* ignore EEXIST — the target dir may exist */
  char names[2048];
  char *ents[128];
  long used = 0;
  long cnt = 0;
  for (;;) {
    if (cnt >= 128 || used + 256 > 2048) { __px_closedir(0, dh); return 1; }
    long r = __px_readdir(0, dh, (long)(names + used), 256);
    if (r <= 0) break;
    ents[cnt] = names + used;
    cnt = cnt + 1;
    used = used + u_strlen(names + used) + 1;
  }
  __px_closedir(0, dh);
  int rc = 0;
  long sl = u_strlen(src);
  long dl = u_strlen(dst);
  long i;
  char sc[1024];
  char dc[1024];
  for (i = 0; i < cnt; i = i + 1) {
    long el = u_strlen(ents[i]);
    if (sl + 1 + el + 1 > 1024 || dl + 1 + el + 1 > 1024) { rc = 1; continue; }
    long k;
    for (k = 0; k < sl; k = k + 1) sc[k] = src[k];
    sc[sl] = '/';
    for (k = 0; k < el; k = k + 1) sc[sl + 1 + k] = ents[i][k];
    sc[sl + 1 + el] = 0;
    for (k = 0; k < dl; k = k + 1) dc[k] = dst[k];
    dc[dl] = '/';
    for (k = 0; k < el; k = k + 1) dc[dl + 1 + k] = ents[i][k];
    dc[dl + 1 + el] = 0;
    if (cp_path(sc, dc)) rc = 1;
  }
  return rc;
}

int main(int argc, char **argv) {
  int recur = 0, i = 1;
  while (i < argc && argv[i][0] == '-' && argv[i][1]) {
    char *a = argv[i];
    int ok = 1;
    long j;
    for (j = 1; a[j]; j = j + 1) {
      if (a[j] == 'r' || a[j] == 'R') recur = 1;
      else { ok = 0; break; }
    }
    if (!ok) break;
    i = i + 1;
  }
  if (argc - i < 2) return 1; /* usage: cp [-r] SRC DST */
  if (recur) return cp_path(argv[i], argv[i + 1]);
  return copy_file(argv[i], argv[i + 1]);
}
