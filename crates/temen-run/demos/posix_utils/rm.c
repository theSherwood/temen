/* rm(1) — remove file operands (`__px_unlink`). Flags (combinable, e.g. `-rf`): `-f` suppresses the
 * error/nonzero-exit for a missing file; `-r`/`-R` removes directories recursively (depth-first:
 * collect a dir's children, remove each, then `__px_rmdir` the now-empty dir). Without `-r`, a
 * directory operand is an error (that is rmdir's job). */
long __px_unlink(int cap, long path, long len);
long __px_opendir(int cap, long path, long len);
long __px_readdir(int cap, long dir, long buf, long cap_n);
long __px_closedir(int cap, long dir);
long __px_rmdir(int cap, long path, long len);
long u_strlen(char *s);

/* Depth-first recursive remove of `path` (a file or a directory). Returns 0 on success. Children are
 * buffered *before* any removal so we never mutate a directory mid-`readdir`. */
static int rm_path(char *path) {
  long d = __px_opendir(0, (long)path, u_strlen(path));
  if (d < 0) {
    /* not a directory (or absent): a plain file removal */
    return __px_unlink(0, (long)path, u_strlen(path)) < 0 ? 1 : 0;
  }
  char names[2048];
  char *ents[128];
  long used = 0;
  long cnt = 0;
  for (;;) {
    if (cnt >= 128 || used + 256 > 2048) { __px_closedir(0, d); return 1; } /* dir too big for the buffer */
    long r = __px_readdir(0, d, (long)(names + used), 256);
    if (r <= 0) break;
    ents[cnt] = names + used;
    cnt = cnt + 1;
    used = used + u_strlen(names + used) + 1;
  }
  __px_closedir(0, d);
  int rc = 0;
  long pl = u_strlen(path);
  long i;
  char child[1024];
  for (i = 0; i < cnt; i = i + 1) {
    long el = u_strlen(ents[i]);
    if (pl + 1 + el + 1 > 1024) { rc = 1; continue; }
    long k;
    for (k = 0; k < pl; k = k + 1) child[k] = path[k];
    child[pl] = '/';
    for (k = 0; k < el; k = k + 1) child[pl + 1 + k] = ents[i][k];
    child[pl + 1 + el] = 0;
    if (rm_path(child)) rc = 1;
  }
  /* Remove the now-empty directory. In the memfs a directory that exists only because it is the
   * prefix of some file (no explicit `mkdir` marker) simply vanishes when its last child is removed,
   * so `rmdir` can report "already gone" — which is success for `rm -r`. Only a directory that is
   * STILL openable after the rmdir attempt is a real failure. */
  if (__px_rmdir(0, (long)path, pl) < 0) {
    long chk = __px_opendir(0, (long)path, pl);
    if (chk >= 0) { __px_closedir(0, chk); rc = 1; }
  }
  return rc;
}

int main(int argc, char **argv) {
  int force = 0, recur = 0, i = 1;
  while (i < argc && argv[i][0] == '-' && argv[i][1]) {
    char *a = argv[i];
    int ok = 1;
    long j;
    for (j = 1; a[j]; j = j + 1) {
      if (a[j] == 'f') force = 1;
      else if (a[j] == 'r' || a[j] == 'R') recur = 1;
      else { ok = 0; break; }
    }
    if (!ok) break;
    i = i + 1;
  }
  int rc = 0;
  for (; i < argc; i = i + 1) {
    if (recur) {
      if (rm_path(argv[i]) && !force) rc = 1;
    } else {
      if (__px_unlink(0, (long)argv[i], u_strlen(argv[i])) < 0 && !force) rc = 1;
    }
  }
  return rc;
}
