/* touch(1) — ensure each path exists as an empty file (creating it if absent), leaving an existing
 * file's contents untouched. Opens O_WRONLY|O_CREAT *without* O_TRUNC and closes immediately. (The
 * memfs carries no mtimes, so touch only creates — it cannot bump a timestamp.) */
long __px_open(int cap, long path, long len, long flags);
long close(long fd);
long u_strlen(char *s);

int main(int argc, char **argv) {
  int i;
  int rc = 0;
  for (i = 1; i < argc; i = i + 1) {
    long fd = __px_open(0, (long)argv[i], u_strlen(argv[i]), 1 | 0100); /* O_WRONLY|O_CREAT */
    if (fd < 0) { rc = 1; continue; }
    close(fd);
  }
  return rc;
}
