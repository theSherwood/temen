/* rm(1) — remove each file operand (`__px_unlink`). `-f` suppresses the error (and the nonzero exit)
 * for a missing file. Directories are not removed here — that is rmdir's job — so this stays a small,
 * obvious unlink loop. */
long __px_unlink(int cap, long path, long len);
long u_strlen(char *s);
int u_streq(char *a, char *b);

int main(int argc, char **argv) {
  int force = 0;
  int i = 1;
  while (i < argc && argv[i][0] == '-' && argv[i][1]) {
    if (u_streq(argv[i], "-f")) force = 1;
    else break;
    i = i + 1;
  }
  int rc = 0;
  for (; i < argc; i = i + 1) {
    long r = __px_unlink(0, (long)argv[i], u_strlen(argv[i]));
    if (r < 0 && !force) rc = 1;
  }
  return rc;
}
